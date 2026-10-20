# Qwen3.5-0.8B linear attention on Wormhole n150: design, optimizations, metrics

Decode for the Gated DeltaNet (GDN) layers of Qwen3.5-0.8B, projections included, on one n150
(8x8 = 64 Tensix cores, ~1.36 MB of L1 per core). The goal: every weight and every state stays in the
cores' local SRAM (L1), so a decode step never touches DRAM.

- Model: 24 layers = 18 GDN + 6 full attention (layers 3, 7, 11, 15, 19, 23). This work covers the GDN
  mixer: `in_proj -> conv1d + SiLU -> l2norm(q, k) -> gated delta rule -> RMSNorm(o) * silu(z) -> out_proj`.
- GDN geometry: hidden 1024, 16 k-heads, 16 v-heads, dk = dv = 128, conv kernel 4.
- One decode step = one token for each of the B sequences (B <= 16). Row b of every tile is sequence b.

## 1. How the weights are split

**Core c owns head h = c // 4 and column quarter s = c % 4** (32 of the head's 128 v columns). Splitting
the heads splits the weights with them, and each slice lives in its core's L1 for the life of the layer.

```
head h -> cores 4h .. 4h+3

                 core 4h       core 4h+1     core 4h+2     core 4h+3
in_proj q cols   q[0:32]       q[32:64]      q[64:96]      q[96:128]     -+ projected split,
in_proj k cols   k[0:32]       k[32:64]      k[64:96]      k[96:128]     -+ then swapped in the group
in_proj v cols   v[0:32]       v[32:64]      v[64:96]      v[96:128]     stays split
in_proj z cols   z[0:32]       z[32:64]      z[64:96]      z[96:128]     stays split (output gate)
a, b             a, b of all 16 heads (one tile column) on every core
state S          S[:, 0:32]    S[:, 32:64]   S[:, 64:96]   S[:, 96:128]  split by v columns
out_proj rows    norm_w * W_o^T[ch 0:32, :]  ...                         split by rows (its 32 channels)
```

Per core, per layer:

| What | Shape | bf16 | bfp8 | bfp4 |
|---|---|---|---|---|
| in_proj slice `[q/4 \| k/4 \| v/4 \| z/4 \| a,b]` | [1024, 160] | 320 KB | 170 KB | 90 KB |
| out_proj rows of its 32 channels, RMSNorm weight folded in | [32, 1024] | 64 KB | 34 KB | 18 KB |
| conv tap diagonals, bf16 | 12 tiles | 24 KB | 24 KB | 24 KB |
| recurrent state per sequence, fp32 | [128, 32] | 16 KB | 16 KB | 16 KB |
| conv state per sequence (last 3 inputs), fp32 | 1 tile | 4 KB | 4 KB | 4 KB |

- **v columns are free to split.** In the delta rule, `k^T S`, `q^T S` and the update of column j only
  touch column j of `S`, so each core keeps `S[:, its 32 columns]` and computes its own 32 outputs.
  Splitting dk (rows) would need a reduction on every token.
- **q and k are needed whole** (the `k^T S` sum runs over all 128 k dims, and l2norm needs the full
  vector), but their *weights* are still split by quarter and the 4 cores swap quarters after the conv
  (2 tiles). Measured: the swap takes 4.4 us on the writer but compute waits 0.02 us for it, because it
  overlaps the v conv. Full q/k weights on every core would cost ~209 KB more per core per layer in bfp8
  (3 -> 2 resident layers) plus 2.2x the in_proj and 3x the conv work.
- **out_proj split by rows (o is never concatenated).** Core c multiplies its 32 channels of y by its 32
  rows of `W_o^T`, giving a partial `[B, 1024]`; a reduce-scatter sums the 64 partials (core j < 32 owns
  output columns `32j .. 32j+31`). The RMSNorm weight is folded into these rows on the host
  (`y_c * w_c * W_o[:, c] = y_c * (w_c W_o[:, c])`).

## 2. One decode step, per core (`kernels/`)

```
x (B rows) -- core 0 fetches once, multicasts to all 64 cores
  |
  v  in_proj (in-kernel, resident weights)          -> P = [q/4 k/4 v/4 z/4 ab]
  v  causal conv1d + SiLU (conv state in L1)         -> q/4 k/4 v/4
  v  q/k quarter swap inside the head's 4 cores      -> full q, k of the head
  v  l2norm, g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)
  v  gated delta rule (recurrent state in L1, updated in place)
  v  RMSNorm over the head (4 partial sums of squares swapped in the group) * silu(z)
  v  out_proj partial (resident rows)                -> 32 partial output tiles
  v  reduce-scatter to the 32 owner cores, summed there
out (B rows)
```

Three RISC-V kernels per core: the **reader** prepares inputs (x tiles, conv shifts, per-token rows), the
**compute** kernel runs the math, the **writer** does all cross-core traffic (x multicast, q/k swap, norm
exchange, reduce-scatter).

**Activation layout.** The layer input and output use a face-major layout that matches how tiles store
data: value (row r, column `32k + 16h + e`) at word `((2k + h) * BP + r) * 16 + e`, with
`BP` = batch rounded up to a power of 2. Each 32-column tile's B rows of one face are then one contiguous
`BP x 64 B` block, so moving a tile's batch costs 2 transfers whatever the batch.

## 3. How the matrix multiplications are done

Everything is tile matmuls on the matrix unit (FPU) with fp32 accumulation in DST and HiFi4 fidelity.
Weights go to srcA in their storage format (bf16 / bfp8 / bfp4); activations are fp32 in srcB.

**in_proj, `[B, 1024] x [1024, 160]`.** The core's weight slice is an L1-sharded tensor bound directly to
a circular buffer, so the matrix unit reads the weights in place: no copy, no DRAM. The 5 output tiles
accumulate in DST across all 32 K tiles; x arrives in 4-tile K chunks, each x tile built from 2 local
DMAs. Each call is a `matmul_block(ct_dim=5)`: one x tile against the row of 5 weight tiles.

**out_proj, `[B, 32] x [32, 1024]`.** `matmul_block(ct_dim=8)`: the y tile against 8 weight tiles per
call, 4 calls for the 32 output tiles.

**Reduce-scatter.** Owner j keeps 2 * BP receive tiles; sender c gets the `BP`-row slot
`(c % (32/BP)) * BP` of tile `c / (32/BP)`. Every core sends the BP rows of its partial tile j as two
contiguous blocks (left / right face): 2 writes per owner, independent of the batch, and never the 32-row
tile padding. The owner sums its 2 * BP tiles with a selector matmul (`row r <- every row j with
j % BP == r`), accumulated in DST.

**Everything else is also expressed as tile matmuls (plus fp32 SFPU math in DST):**

| Operation | How |
|---|---|
| causal conv (4 taps, per channel) | `sum_s x_shift_s @ diag(w_tap_s)`, all 12 matmuls under one init |
| `\|q\|^2`, `\|k\|^2`, `q.k` | FPU products into DST, summed over the 4 head tiles by SFPU adds, then `@ ones_col` |
| this head's a, b out of the a/b tile | `ab @ selector` |
| recurrence `u = v - (d k)^T S` | `I @ v + sum_i (-d k)_i @ S_i`, sequence b in row b |
| readout `o = q^T S_new` | `sum_i (d q)_i @ S_i + W_b @ u`, W_b = `beta_b q_b.k_b` at [b, b] |
| state update `S += (d-1) S + (beta k) u^T` | `diag(d-1) @ S + (beta K)^T @ u`, packed with **L1 accumulation** |
| RMSNorm | `o * silu(z)` in fp32 SFPU, then `* rstd` (broadcast); weight folded into out_proj |

`(beta K)^T` is computed **once per step** (`I @ KB^T` with the matmul transpose flag): its column b is
`beta_b k_b`, and since `u` only has row b filled for sequence b, `(beta K)^T @ u` is exactly that
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
- **Math fidelity hardly matters** for speed (HiFi4 -> LoFi saves ~1 us at up to 18% error), so the
  kernel stays at HiFi4.

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

### Current breakdown (bfp8, device profiler)

| Part | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| in_proj | 12.8 us (18%) | 13.9 | 15.8 | 19.4 |
| conv1d + SiLU | 9.9 (14%) | 11.5 | 13.2 | 16.1 |
| q/k l2norm sums | 10.4 (15%) | 10.4 | 10.4 | 10.5 |
| gating scalars | 11.0 (15%) | 11.0 | 11.0 | 10.9 |
| recurrence inputs | 4.1 (6%) | 4.1 | 4.1 | 4.1 |
| **linear attention state update + readout** | **3.2 (5%)** | 12.9 | 25.9 | 51.8 |
| gated RMSNorm | 7.3 (10%) | 7.3 | 7.3 | 7.4 |
| out_proj partials | 4.7 (7%) | 7.8 | 12.1 | 21.5 |
| reduce-scatter sum (owners) | 1.7 (2%) | 10.2 | 17.4 | 32.8 |
| **total** | **71.0** | **90.7** | **117.1** | **171.2** |

Per-tile work (q/k sums, gating, RMSNorm) does not grow with the batch; the recurrence grows by ~3.2 us
per sequence (each sequence has its own state), and out_proj + reduce-scatter grow with the bytes moved.
Phases on the critical path include waiting on other cores (the swap, the norm exchange, the 64 senders).

## 5. Metrics

### Correctness (vs Hugging Face `Qwen3_5GatedDeltaNet`)

Kernel error, against HF with the same (quantised) weights: **0.38-0.41%, PCC 0.999999**, for batch 1,
3, 4, 8, 16 (batch 3 checks the non-power-of-2 padding), and with 2 / 3 / 6 layers resident at once,
8 decode steps from zero state, in both weight modes. Total error against HF with the model's bf16
weights (measured before round 2): bf16 0.5%, bfp8 1.26% (PCC 0.99994), bfp4 17.6% (too lossy).

### How many layers fit in L1 (measured on the card)

| Batch | bf16 | bfp8 | bfp4 |
|---|---|---|---|
| 1 | 2 | **3** | 6 |
| 2 | 2 | 3 | 5 |
| 4 | 1 | 3 | 4 |
| 8 | 1 | 2 | 3 |
| 16 | 1 | 1 | 1 |

~370 KB per core goes to the kernel's working buffers, shared by all layers; the rest holds weights and
states.

### Weights in L1 vs streamed from DRAM (500 traced decode steps, `bench_decode.py`)

Same kernel, same per-core slices; `weights_in_dram=True` keeps the weights in DRAM and streams them in
every step. States are in L1 in both. Traced, so host dispatch drops out.

| Config | Weights in L1 | Weights in DRAM | Speed-up |
|---|---|---|---|
| bfp8, 3 layers, batch 1 | **74 us/layer** (223 us/step) | 134 us/layer | 1.8x |
| bf16, 2 layers, batch 1 | **76 us/layer** (152 us/step) | 200 us/layer | 2.6x |
| bfp8, 3 layers, batch 4 | **94 us/layer** (281 us/step) | 147 us/layer | 1.56x |
| bfp8, 2 layers, batch 8 | **122 us/layer** (243 us/step) | 167 us/layer | 1.37x |

### Energy per token (`bench_energy.py`)

Board power from the card's hwmon sensor, sampled every 5 ms over ~5 s of traced decode. Idle with the
device open: 19.9 W. GDN layers only:

| Config | Weights | Tokens/s | Board power | Energy / token / layer | Tokens/J |
|---|---|---|---|---|---|
| bfp8, 3 layers, batch 1 | L1 | 4,345 | 45.7 W | 3.5 mJ | **95** |
| | DRAM | 2,492 | 43.2 W | 5.8 mJ | 58 |
| bf16, 2 layers, batch 1 | L1 | 6,255 | 48.1 W | 3.8 mJ | **130** |
| | DRAM | 2,497 | 40.1 W | 8.0 mJ | 62 |
| bfp8, 3 layers, batch 4 | L1 | 14,045 | 52.5 W | 1.25 mJ | **267** |
| | DRAM | 9,053 | 48.0 W | 1.77 mJ | 189 |
| bfp8, 2 layers, batch 8 | L1 | 32,072 | 55.2 W | 0.86 mJ | **581** |
| | DRAM | 23,997 | 53.5 W | 1.12 mJ | 448 |

L1-resident weights use 23-52% less energy per token. About 20 W of the board power is idle power, so
batching is the biggest energy lever: batch 8 gives ~4x the energy efficiency per layer of batch 1.

## 6. Ideas and next steps

**Per-tile phases at small batch.** At batch 1, in_proj (12.8 us), the gating scalars (11.0), the q/k sums
(10.4) and conv (9.9) are ~62% of the layer. The gating phase runs ~18 SFPU ops on whole tiles where only
column 0 matters; restricting them to the first column (where the API allows) or fusing them into fewer
passes should halve it. in_proj still waits on the x multicast and tile build for part of its time.

**Large batch.** The recurrence is ~3.2 us per sequence (separate state per sequence; the matrix unit uses
one row of each tile), and the reduce-scatter still grows with the bytes moved (32.8 us at batch 16).
Posted writes, or a two-level reduction inside each head group first, would cut the latter.

**Fit more layers.** The ~370 KB of working buffers is the fixed cost. Single-buffering more of them,
packing the conv state (only ~1.2 KB of each 4 KB tile is real) and sharing buffers between phases would
leave room for a 4th bfp8 layer at batch 1.

**Fuse across layers.** Each layer is a launch, and its output goes through an L1 tensor that the next
layer reads back. The 32 owner cores could multicast their output blocks straight into every core's input
buffer for the next layer, and one launch could run all resident layers.

**Rotate 3-layer groups through L1.** The model is 3 GDN layers then 1 full-attention layer, repeated 6
times. One group of 3 GDN layers fits with all 18 layers' states resident (~20 MB at batch 1). While the
attention layer runs elsewhere, the next group's weights are loaded (weights are read-only, so the old
group is simply overwritten). This hides the weight load behind the attention time; it does not remove
the DRAM traffic.

**More chips.** All 18 GDN mixers fit in about 2 chips' L1 in bfp4 or 3-4 in bfp8, so no GDN weight would
ever touch DRAM; the same head split extends across chips.

**A fairer DRAM baseline.** The comparison streamer is simple (one reader, a barrier per chunk,
~60-85 GB/s); a tuned one would narrow the gap.

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
