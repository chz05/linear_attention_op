# Qwen3.5-0.8B linear attention on Wormhole n150: design, optimizations, metrics

Decode for the Gated DeltaNet (GDN) layers of Qwen3.5-0.8B, projections included, on one n150
(8x8 = 64 Tensix cores, ~1.36 MB of L1 per core). The goal: every weight and every state stays in the
cores' local SRAM (L1), so a decode step never touches DRAM. A short version of this document is
[REPORT.md](REPORT.md).

- Model: 24 layers = 18 GDN + 6 full attention (layers 3, 7, 11, 15, 19, 23). This work covers the GDN
  mixer: `in_proj -> conv1d + SiLU -> l2norm(q, k) -> gated delta rule -> RMSNorm(o) * silu(z) -> out_proj`.
- GDN geometry: hidden 1024, 16 k-heads, 16 v-heads, dk = dv = 128, conv kernel 4.
- One decode step = one token for each of the B sequences (B <= 16). Row b of every tile is sequence b;
  BP = B rounded up to a power of 2.

## 1. How the weights are split

**Core c owns head h = c // 4 and column quarter s = c % 4** (32 of the head's 128 v columns, i.e. v
channels `32c .. 32c+31`). Splitting the heads splits the weights with them, and each slice lives in its
core's L1 for the life of the layer.

```
head h -> cores 4h .. 4h+3                       grid half 0 = rows 0-3 = cores  0-31 = heads 0-7
                                                 grid half 1 = rows 4-7 = cores 32-63 = heads 8-15
                 core 4h       core 4h+1     core 4h+2     core 4h+3
in_proj q cols   q[0:32]       q[32:64]      q[64:96]      q[96:128]     -+ projected split; l2-norm partial sums
in_proj k cols   k[0:32]       k[32:64]      k[64:96]      k[96:128]     -+ exchanged, then the row-scaled
                                                                            quarters swapped in the group
in_proj v cols   v[0:32]       v[32:64]      v[64:96]      v[96:128]     stays split
in_proj z cols   z[0:32]       z[32:64]      z[64:96]      z[96:128]     stays split (output gate)
a, b             a, b of all 16 heads (one tile column) on every core
state S          S[:, 0:32]    S[:, 32:64]   S[:, 64:96]   S[:, 96:128]  split by v columns
out_proj         core c: W_o^T[v channels of its half, output columns 32 (c % 32) .. +31], norm_w folded in
```

Per core, per layer:

| What | Shape | bf16 | bfp8 | bfp4 |
|---|---|---|---|---|
| in_proj slice `[q/4 \| k/4 \| v/4 \| z/4 \| a,b]` | [1024, 160] | 320 KB | 170 KB | 90 KB |
| out_proj: its half's 1024 v channels x its 32 output columns, RMSNorm weight folded in | [1024, 32] | 64 KB | 34 KB | 18 KB |
| conv tap diagonals, bf16 | 12 tiles | 24 KB | 24 KB | 24 KB |
| recurrent state per sequence, fp32 | [128, 32] | 16 KB | 16 KB | 16 KB |
| conv state (last 3 inputs of q/4, k/4, v/4), fp32, all sequences | 9 BP x 32 tiles | 1.1 KB x BP | | |

- **v columns are free to split.** In the delta rule, `k^T S`, `q^T S` and the update of column j only
  touch column j of `S`, so each core keeps `S[:, its 32 columns]` and computes its own 32 outputs.
  Splitting dk (rows) would need a reduction on every token.
- **q and k are needed whole** (the `k^T S` sum runs over all 128 k dims, and l2norm needs the full
  vector), but their *weights* are still split by quarter. Each core convolves its quarter, sends the
  l2-norm partial sums of the quarter (`|q|^2`, `|k|^2`, `q.k`, one tile) to the head's 4 cores, computes the
  gates, row-scales its own quarter (`d*q`, `-d*k`, `beta*k`) and swaps the scaled quarters. Only the BP batch
  rows travel. Full q/k weights on every core would cost ~204 KB more per core per layer in bfp8 plus 2.2x the
  in_proj and 3x the conv work.
- **out_proj split by output columns and grid half.** y (the gated RMSNorm output, `[B, 2048]`, core c holding
  channels `32c ..`) is gathered within each half of the grid; core c then computes output columns
  `32 (c % 32) ..` over its half's 1024 channels, and cores c and c ^ 32 add their two partials. Each core
  holds the same amount of out_proj weight as with a row split, but no 64-way reduction is needed. The RMSNorm
  weight is folded into the rows on the host (`y_c * w_c * W_o[:, c] = y_c * (w_c W_o[:, c])`).

## 2. One decode step, per core (`kernels/`)

```
x (B rows) -- core 0 multicasts it to all 64 cores in up to 8 pieces (lands in c_prep, read in place)
  |
  v  in_proj (resident weights, x as BP x 32 tiles)       -> P = [q/4 k/4 v/4 z/4 ab]
  v  causal conv1d + SiLU (conv state read in place)        -> q/4 k/4 v/4, and q/4*q/4, k/4*k/4, q/4*k/4
  v  l2-norm partial sums of the quarter -> head's 4 cores  (one tile, 2 blocks per core)
  v  head sums + a, b -> gates rq, rk, d, beta, w           (one SFPU pass over the batch rows)
  v  d*q, -d*k, beta*k of the quarter -> head's 4 cores     (2 blocks per tile)
  v  (beta K)^T; gated delta rule (recurrent state in L1, updated in place)
  v  RMSNorm over the head (4 partial sums of squares swapped in the group) * silu(z)
  v  y -> the half's coordinator, which multicasts the half's y to the half
  v  out_proj: output tile c % 32 over the half's 1024 v channels
  v  + the partial of core c ^ 32; this core's face -> the output
out (B rows)
```

Three RISC-V kernels per core: the **reader** hands in_proj the input pieces as they land, rolls the conv
state, builds the per-token recurrence inputs and tells the y gather coordinator when `c_prep` is free; the
**compute** kernel runs the math; the **writer** moves all data between cores (input multicast, l2-sum
exchange, q/k swap, norm exchange, y gather, pair exchange, output). Each transfer is followed by a flush and
a semaphore increment on the same route and static VC, so the increment lands after the data without a
write barrier.

**Activation layout.** The layer input and output use a face-major layout that matches how tiles store
data: value (row r, column `32k + 16h + e`) at word `((2k + h) * BP + r) * 16 + e`. Each 32-column tile's B
rows of one face are one contiguous `BP x 64 B` block, so moving a tile's batch costs 2 transfers whatever
the batch, and the two blocks of tile k are adjacent: the input is already a sequence of `BP x 32` tiles.

**Only the batch rows.** Every activation has its B sequences in rows 0..B-1 (B <= 16). The matrix unit
reads `BP x 32` tiles where an operand is a batch-row activation (the in_proj input, the conv state, the
gathered y), which cuts
its MVMULs and source data; the vector unit (SFPU) touches only the rows that matter. Measured on the card:
in a face, SFPU row `i` (`dst_reg[i]`) holds tile rows `4 (i/2) .. 4 (i/2) + 3` at the even (`i` even) or odd
columns, and a tile is 32 SFPU rows, so the per-row scalars (column 0 of rows 0..B-1) are `ceil(B/4)` SFPU
rows instead of 32 per tile, and the SiLUs need `2 ceil(B/4)` per face of the top two faces.

## 3. How the matrix multiplications are done

Everything is tile matmuls on the matrix unit (FPU) with fp32 accumulation in DST, at HiFi4 fidelity except
in_proj (HiFi3, accuracy-neutral, see section 5). Weights go to srcA in their storage format (bf16 / bfp8 /
bfp4); activations are fp32 in srcB.

**in_proj, `[B, 1024] x [1024, 160]`.** The core's weight slice is an L1-sharded tensor bound directly to
a circular buffer, so the matrix unit reads the weights in place: no copy, no DRAM. The input is read in place
too: `c_x` is a `BP x 32`-tile view of the memory where core 0's multicast lands. The 5 output tiles
accumulate in DST across all 32 K tiles, one `matmul_block(ct_dim=5)` per K tile.

**conv.** `sum_s x_shift_s @ diag(w_tap_s)`, all 12 matmuls in one DST pass. The three previous inputs come
straight from the conv state, which is stored as `BP x 32` tiles (tile `3 (s-1) + ct` = channel tile ct, s
steps back); after the conv has read it, the reader shifts it with 2 block copies and writes this token's
inputs in (6 face blocks).

**out_proj, `[B, 1024] x [1024, 32]` per core.** 32 K-tile matmuls into one DST tile; srcB is the gathered
y of the core's half (32 `BP x 32` tiles, slot k = core `32 half + k`). The **y gather**: every core writes
its y block (one `BP x 32` tile) into the half's coordinator (core 0 or 32), which multicasts the dense
`4 KB x BP` buffer to the half once all 32 slots are in and every core of the half has reported its
`c_prep` free. A single multicast source avoids the contention of 32 cores multicasting to the same
rectangle (tried: 95 us at batch 1). The **pair sum**: cores c and c ^ 32 hold partials of the same output
tile; each sends the other one face (one block) and adds the face it writes (`add_tiles`).

**Everything else is also expressed as tile matmuls (plus fp32 SFPU math in DST):**

| Operation | How |
|---|---|
| `\|q\|^2`, `\|k\|^2`, `q.k` | the quarter's products on the SFPU in the conv's DST pass; row sums by matmuls with ones in columns 0, 1, 16 (one tile, so the gate pass finds them in the same SFPU lanes); the head's 4 partials (rows `s BP + b`) added by a selector matmul |
| this head's a, b out of the a/b tile | `ab @ selector` |
| gates and norms | one SFPU pass over column 0 of the batch rows: rsqrt, softplus, exp, sigmoid, products |
| row scaling `d*q`, `-d*k`, `beta*k` | the core's own quarter only, `mul_tiles_bcast_cols`, before the swap |
| recurrence pass | `I @ v + sum_i X_i @ S_i` with `X_i` = [`-d*k` at row b; `d*q` at row b + 16]: u in row b, `(d q)^T S` in row b + 16 |
| readout `o = q^T S_new` | `M @ pass`, `M = e_b (e_{b+16} + w e_b)^T` (w = `beta_b qn_b.kn_b`), packed with L1 accumulation into one tile that collects every sequence's row |
| state update `S += (d-1) S + (beta k) u^T` | `diag(d-1) @ S + (beta K)^T @ pass`, packed with **L1 accumulation** onto the fp32 state; row b + 16 of the pass meets column b + 16 of `(beta K)^T`, which is zero |
| RMSNorm | `o * silu(z)` on the SFPU (batch rows), sum of squares `@ ones_col`, rstd in one SFPU pass, `* rstd` (broadcast); weight folded into out_proj |

`(beta K)^T` is computed **once per step** (`I @ KB^T` with the matmul transpose flag): its column b is
`beta_b k_b`, and since u only has row b filled for sequence b, `(beta K)^T @ u` is exactly that
sequence's outer product. The state update only sends the *increment* through the FPU (which reads fp32
as tf32 and truncates) and lets the packer add it to the fp32 state in L1, which keeps the state at full
fp32 and scales the truncation error by `1 - d`.

## 4. Optimizations

### Round 1: projections (bfp8, batch 1, device profiler)

| Step | in_proj | out_proj | layer |
|---|---|---|---|
| first version: tile-by-tile matmuls, every core reads the full x tiles | 45 us | 10 us | 126 us |
| `matmul_block` (x tile x a row of 5 / 8 weight tiles per call) | 42 us | 7 us | 125 us |
| **read only the B real rows of x, not the 32-row tile padding** | **14 us** | 7 us | **93 us** |
| core 0 fetches x once and multicasts it; tiles rebuilt by local DMA | 13 us | 5 us | 90 us |
| reduce-scatter: wait for "flushed" per tile, one write barrier at the end | 13 us | 5 us | 87 us |

- **in_proj was waiting for its input, not computing**: 64 cores pulled the same 128 KB of padded x
  tiles from the same L1 banks; at batch 1 only 4 KB of it is real.
- **RISC-V loads/stores are slow** (several cycles per L1 access): the same copy as local NoC DMA is ~2x
  faster.
- **Math fidelity hardly mattered** for speed then (HiFi4 -> LoFi saved ~1 us at up to 18% error), so the
  kernel stayed at HiFi4; round 3 runs in_proj at HiFi3, which is accuracy-neutral (section 5).

### Round 2: the rest of the layer, for every batch size

Layer time per launch (bfp8, device profiler, median of 100 launches):

| Step | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| after round 1 | 85.7 us | 136.9 us | | |
| conv inputs built early by DMA, conv under one init; q/k sums in DST; RMSNorm weight folded into out_proj, `o*silu(z)` computed during the norm exchange | 77.3 | 117.9 | 174.9 | |
| recurrence: sequence b stays in row b (per-token inputs are 64 B DMAs), `(beta K)^T` once per step | 71.0 | 97.6 | 135.2 | |
| face-major activations; reduce-scatter in 2 transfers per owner; x tiles in 2 DMAs | **71.0** | **90.7** | **117.1** | **171.2** |

What changed and why it scales:

| Phase | Before | After | Why |
|---|---|---|---|
| conv | 14.9 us (B=1), 27.0 (B=4) | 9.9, 11.5 | its shifted inputs depend only on the conv state, so the reader builds them (by DMA) while compute is on in_proj; one FPU init for all 12 tap matmuls |
| q/k norm sums | 11.3 | 10.4 | products summed in DST by SFPU adds: 3 tiles go back through a buffer instead of 12 |
| RMSNorm | 10.7 | 7.3 | one matmul fewer (weight folded), `o*silu(z)` overlaps the exchange |
| recurrence | 9.1 us per sequence | 3.2 per sequence | per-token reader work went from ~400 RISC-V word copies plus a strided column build to a few 64 B DMAs |
| reduce-scatter (B=8) | 18.8 + 24.7 us | 12.1 + 17.4 | 2 writes per owner instead of 2 per row |

Breakdown after round 2 (math-thread zones, which round 3 found to hide waits):

| Part | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| in_proj | 12.8 us (18%) | 13.9 | 15.8 | 19.4 |
| conv1d + SiLU | 9.9 (14%) | 11.5 | 13.2 | 16.1 |
| q/k l2norm sums | 10.4 (15%) | 10.4 | 10.4 | 10.5 |
| gating scalars | 11.0 (15%) | 11.0 | 11.0 | 10.9 |
| recurrence inputs | 4.1 (6%) | 4.1 | 4.1 | 4.1 |
| linear attention state update + readout | 3.2 (5%) | 12.9 | 25.9 | 51.8 |
| gated RMSNorm | 7.3 (10%) | 7.3 | 7.3 | 7.4 |
| out_proj partials | 4.7 (7%) | 7.8 | 12.1 | 21.5 |
| reduce-scatter sum (owners) | 1.7 (2%) | 10.2 | 17.4 | 32.8 |
| **total** | **71.0** | **90.7** | **117.1** | **171.2** |

### Round 3: batch rows only, fewer exchanges, out_proj without a reduce-scatter

First, finer profiler zones for every phase on all five RISC-V cores and a script that turns the device log
into per-core timelines ([`profile_layer.py`](profile_layer.py)). Two things it showed: the old breakdown
(math-thread zones) hid waits, e.g. the q/k swap held the unpacker for 12.5 us at batch 1 while the math
thread showed 0.02 us; and compute zones record when the TRISCs *issue* instructions, while the hardware can
finish up to ~1 us later (the packer publishes a tile with a Tensix instruction once the pack is done), so
phases are timed at the CB handoffs on the data-movement cores.

Layer time per launch (bfp8, device profiler, median of 100 launches; measured with the round-3 profiler
zones, which add ~1 us at batch 1):

| Step | B=1 | B=16 |
|---|---|---|
| after round 2 | 72.1 us | 172.4 us |
| SFPU work on the batch rows only: every per-row scalar in one fused pass over column 0 (gating 11.1 -> 1.6 us), SiLUs on rows 0..B-1 (conv SiLU 7.0 -> 1.5 us); q/k row sums and the gates in one DST pass | 52.5 | 157.1 |
| q/k swap: only the BP batch rows travel (2 blocks per tile instead of 4 KB) | 49.2 | 155.1 |
| out_proj reduce-scatter in a staggered owner order (later replaced) | 48.1 | 148.5 |
| row scaling in 4 DST passes; w and d - 1 read by the reader directly | 47.8 | 148.6 |
| in_proj: 16 x 32 input tiles, HiFi3 | 45.6 | 146.1 |
| conv inputs off the critical path; input multicast in up to 8 pieces | 44.0 | 135.4 |
| conv state as blocks of the batch rows (18 DMAs per step instead of 36 per sequence) | 44.0 | 133.7 |
| recurrence: one state pass for u and `(dq)^T S`, readout + update in one DST pass (14 matmuls per sequence instead of 18) | 43.7 | 129.3 |
| **out_proj: y gathered per half, column split, pair sum (no reduce-scatter)** | **40.3** | **106.4** |
| RMSNorm: outputs accumulated in L1 (no per-token copy), one small block per norm exchange | 37.9 | 104.8 |
| in_proj reads the input in place as `BP x 32` tiles | 35.7 | 104.4 |
| conv reads its state in place (`BP x 32` tiles), state rolled with 8 DMAs | 35.2 | 103.2 |
| recurrence: no per-sequence sync (the sequences use disjoint state tiles) | 34.7 | 97.7 |
| l2-norm partial sums before the swap; each core scales its own quarter (3 multiplies instead of 12) | 32.3 | 97.1 |
| exchanges: flush + same-VC semaphore increment instead of a write barrier | 31.8 | 96.9 |
| **final** (header cleanup) | **32.0** | **96.7** |

Batch 4 and 8, measured at the start and the end only: 90.7 -> 42.5 us and 117.1 -> 60.6 us.

Lessons from this round:

- **Concurrent multicasts to one rectangle contend.** 32 cores each multicasting their y block to the same
  32 cores took 95 us at batch 1 (multicast path reservation); one coordinator per half collecting the
  blocks and multicasting once takes 2.5 us.
- **A data write and a semaphore increment use different command buffers**, so the increment may only be
  sent after the data has left (`noc_async_writes_flushed`); a full write barrier is not needed, since both
  travel the same route on the same static VC. The data and flag multicasts of the input need the same flush:
  without it, receivers read input pieces that had not landed.
- **CB read pointers are per RISC.** A CB consumed by two kernels must not depend on the other kernel's
  pops (y now goes to the writer through its own CB), and a buffer one kernel fills and another reads needs a
  handshake even when the timing makes the race unlikely (the conv state roll once ran before its inputs
  existed on core 0).
- **What did not help:** `matmul_block` for the state update (two extra inits per sequence cost more than
  the operand reuse saved: +1.2 us per sequence at batch 1); HiFi2 in_proj (1.1% error).
- Device profiler zones are keyed by a 16-bit hash of name, file and line; when two collide the run aborts
  ("Source location hashes are colliding"), and moving one zone by a line fixes it.

### Current breakdown (bfp8, device profiler, us per phase, slowest core, timed at the CB handoffs)

| Phase | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| input multicast (core 0 reads it, multicasts it) | 3.6 (11%) | 6.0 | 8.8 | 10.6 |
| in_proj, after the last input piece | 5.7 (18%) | 3.7 | 1.5 | 1.6 |
| conv + SiLU + l2 products | 3.5 (11%) | 3.5 | 4.2 | 6.1 |
| l2-norm partial sums, exchange | 1.7 (5%) | 1.7 | 2.0 | 2.7 |
| gates, own quarter row-scaled | 2.8 (9%) | 2.8 | 3.2 | 4.1 |
| q/k swap | 1.2 (4%) | 1.5 | 2.5 | 4.4 |
| **recurrence** (+ sum of squares) | **5.5 (17%)** | 13.4 | 24.1 | 45.2 |
| RMSNorm (norm exchange, rstd, gate) | 2.0 (6%) | 2.0 | 2.2 | 2.9 |
| y gather | 2.5 (8%) | 4.2 | 8.0 | 13.9 |
| out_proj | 2.3 (7%) | 2.5 | 2.9 | 3.8 |
| pair sum, output write | 1.2 (4%) | 1.3 | 1.3 | 1.5 |
| **total** | **32.0** | **42.5** | **60.6** | **96.7** |

The recurrence costs ~2.8 us per sequence at batch 16 (each sequence has its own state); the input multicast and the
y gather grow with the bytes moved (4 KB per sequence each); the rest hardly depends on the batch.

## 5. Metrics

### Correctness (vs Hugging Face `Qwen3_5GatedDeltaNet`)

Kernel error, against HF with the same (quantised) weights: **0.36-0.39%, PCC 0.999998+**, for batch 1,
3, 4, 8, 16 (batch 3 checks the non-power-of-2 padding), and with 2 / 3 / 6 layers resident at once,
8 decode steps from zero state, in both weight modes (all 11 tests of `test_gdn_l1.py`). Over 32 decode steps:
0.38% at batch 1 and 0.40% at batch 8, the same at the last step as overall (no drift). in_proj at HiFi3
instead of HiFi4: 0.40% vs 0.39% (32 steps); HiFi2 would be 1.1%. Total error against HF with the model's
bf16 weights (measured before round 2): bf16 0.5%, bfp8 1.26% (PCC 0.99994), bfp4 17.6% (too lossy).

### How many layers fit in L1 (measured on the card)

| Batch | bf16 | bfp8 | bfp4 |
|---|---|---|---|
| 1 | 2 | **4** (was 3) | **7** (was 6) |
| 2 | 2 | **4** (was 3) | **6** (was 5) |
| 4 | **2** (was 1) | 3 | **5** (was 4) |
| 8 | 1 | 2 | 3 |
| 16 | 1 | 1 | **2** (was 1) |

The kernel's working buffers (shared by all layers) take ~260 KB per core at batch 1 (was ~380 KB) and
~386 KB at batch 16 (was ~508 KB); the rest holds weights and states.

### Weights in L1 vs streamed from DRAM (500 traced decode steps, `bench_decode.py`)

Same kernel, same per-core slices; `weights_in_dram=True` keeps the weights in DRAM and streams them in
every step. States are in L1 in both. Traced, so host dispatch drops out. Layers run one after another,
all resident at once.

| Config | Weights in L1 | Weights in DRAM | Speed-up |
|---|---|---|---|
| bfp8, 3 layers, batch 1 | **33 us/layer** (99 us/step; was 74) | 97 us/layer (was 134) | 2.9x |
| bf16, 2 layers, batch 1 | **35 us/layer** (70 us/step; was 76) | 164 us/layer (was 200) | 4.7x |
| bfp8, 3 layers, batch 4 | **44 us/layer** (133 us/step; was 94) | 106 us/layer (was 147) | 2.4x |
| bfp8, 2 layers, batch 8 | **64 us/layer** (128 us/step; was 122) | 122 us/layer (was 167) | 1.9x |
| bfp8, 4 layers, batch 1 / 2 | **33 / 36 us/layer** (130 / 143 us/step) | | |

### Energy per token (`bench_energy.py`)

Board power from the card's hwmon sensor, sampled every 5 ms over ~5 s of traced decode. Idle with the
device open: 19.8 W. GDN layers only:

| Config | Weights | Tokens/s | Board power | Energy / token / layer | Tokens/J (was) |
|---|---|---|---|---|---|
| bfp8, 3 layers, batch 1 | L1 | 9,416 | 44.0 W | 1.56 mJ | **214** (95) |
| | DRAM | 3,397 | 38.6 W | 3.79 mJ | 88 (58) |
| bf16, 2 layers, batch 1 | L1 | 13,227 | 46.5 W | 1.76 mJ | **284** (130) |
| | DRAM | 3,055 | 36.4 W | 5.96 mJ | 84 (62) |
| bfp8, 3 layers, batch 4 | L1 | 28,429 | 50.5 W | 0.59 mJ | **563** (267) |
| | DRAM | 12,486 | 42.8 W | 1.14 mJ | 292 (189) |
| bfp8, 2 layers, batch 8 | L1 | 59,000 | 53.6 W | 0.45 mJ | **1,100** (581) |
| | DRAM | 31,913 | 48.0 W | 0.75 mJ | 665 (448) |

L1-resident weights use 40-70% less energy per token than streamed ones (23-52% before round 3), and the
optimized kernel gets 1.9-2.3x more tokens per joule than before at the same configurations. About 20 W of
the board power is idle power, so batching is the biggest energy lever: batch 8 costs 0.45 mJ per token per
layer, 3.5x less than batch 1.

## 6. Ideas and next steps

**Input already in place.** At batch 1 the first 3.6 us of every layer go to core 0 reading the input from
an interleaved L1 tensor and multicasting it (10.6 us at batch 16). If the previous layer wrote its output
straight to core 0 (a tensor sharded there), the read would go; if it gathered and multicast its output
itself, the next layer would start with its input in place.

**in_proj's a/b column.** Every core computes all 16 heads' a and b (32 of its 160 tile matmuls) but uses
two values. Splitting that column's K over the head's 4 cores and adding the partials with the l2-norm sums
would save ~0.8 us and 26 KB of L1 per layer.

**Large batch.** The recurrence (~2.8 us per sequence: 14 tile matmuls on separate states) and the y gather
(a 64 KB multicast at batch 16, ~14 us) dominate. A two-level gather (along rows, then columns, with no two
multicasts sharing a rectangle) or 16-row out_proj input tiles would cut the latter.

**Fuse across layers.** Each layer is a launch, and its output goes through an L1 tensor that the next
layer reads back; one launch could run all resident layers, with the output gathered and multicast straight
into the next layer's input.

**Rotate 3-layer groups through L1.** The model is 3 GDN layers then 1 full-attention layer, repeated 6
times. One group of 3 GDN layers fits with all 18 layers' states resident (~20 MB at batch 1). While the
attention layer runs elsewhere, the next group's weights are loaded (weights are read-only, so the old
group is simply overwritten). This hides the weight load behind the attention time; it does not remove
the DRAM traffic.

**More chips.** All 18 GDN mixers fit in about 2 chips' L1 in bfp4 or 3-4 in bfp8, so no GDN weight would
ever touch DRAM; the same head split extends across chips.

**A fairer DRAM baseline.** The comparison streamer is simple (one reader, a barrier per chunk); a tuned
one would narrow the gap.

**Complete the decoder layer.** Input RMSNorm, residual adds and the MLP (3 x 1024 x 3584, as large as the
GDN mixer) around the mixer.

## Files

| File | |
|---|---|
| `gdn_l1.py` | `Qwen35_08B_GDNStackL1`: weight split, L1 tensors, activation layout, decode / traced decode |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the fused per-core layer kernel (profiler zones per phase) |
| `test_gdn_l1.py` | decode tests vs HF, batch 1-16, L1 and DRAM weight modes |
| `bench_decode.py` | 500-step decode, L1 vs DRAM, plain dispatch and traced |
| `bench_energy.py` | power sampling and energy per token |
| `profile_layer.py` | device time of one layer launch per phase and per core, from the profiler zones |
| `REPORT.md` | short report: the layer, the split, the optimizations with file pointers, results, energy |
