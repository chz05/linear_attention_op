# Qwen3.5-0.8B linear attention on Wormhole n150: design, optimizations, metrics

Decode for the Gated DeltaNet (GDN) layers of Qwen3.5-0.8B, projections included, on one n150
(8x8 = 64 Tensix cores, ~1.36 MB of L1 per core). The goal: every weight and every state stays in the
cores' local SRAM (L1), so a decode step never touches DRAM.

- Model: 24 layers = 18 GDN + 6 full attention (layers 3, 7, 11, 15, 19, 23). This work covers the GDN
  mixer: `in_proj -> conv1d + SiLU -> l2norm(q, k) -> gated delta rule -> RMSNorm(o) * silu(z) -> out_proj`.
- GDN geometry: hidden 1024, 16 k-heads, 16 v-heads, dk = dv = 128, conv kernel 4.
- One decode step = one token for each of the B sequences. Row b of every tile is sequence b (B <= 32).

## 1. How the weights are split

**Core c owns head h = c // 4 and column quarter s = c % 4** (32 of the head's 128 v columns). Splitting
the heads splits the weights with them, and each slice lives in its core's L1 for the life of the layer.

```
head h -> cores 4h .. 4h+3

                 core 4h       core 4h+1     core 4h+2     core 4h+3
in_proj q cols   q[0:32]       q[32:64]      q[64:96]      q[96:128]     -+ projected split,
in_proj k cols   k[0:32]       k[32:64]      k[64:96]      k[96:128]     -+ then all-gathered in the group
in_proj v cols   v[0:32]       v[32:64]      v[64:96]      v[96:128]     stays split
in_proj z cols   z[0:32]       z[32:64]      z[64:96]      z[96:128]     stays split (output gate)
a, b             a, b of all 16 heads (one tile column) on every core
state S          S[:, 0:32]    S[:, 32:64]   S[:, 64:96]   S[:, 96:128]  split by v columns
out_proj rows    W_o^T[ch 0:32, :]  ...                                  split by rows (its 32 channels)
```

Per core, per layer:

| What | Shape | bf16 | bfp8 | bfp4 |
|---|---|---|---|---|
| in_proj slice `[q/4 \| k/4 \| v/4 \| z/4 \| a,b]` | [1024, 160] | 320 KB | 170 KB | 90 KB |
| out_proj rows of its 32 channels (`W_o^T`) | [32, 1024] | 64 KB | 34 KB | 18 KB |
| conv tap diagonals (12) + norm-weight diagonal (1), bf16 | 13 tiles | 26 KB | 26 KB | 26 KB |
| recurrent state per sequence, fp32 | [128, 32] | 16 KB | 16 KB | 16 KB |
| conv state per sequence (last 3 inputs), fp32 | 1 tile | 4 KB | 4 KB | 4 KB |

Why this split:

- **v columns are free to split.** In the delta rule, `k^T S`, `q^T S` and the update of column j only
  touch column j of `S`, so each core keeps `S[:, its 32 columns]` and computes its own 32 outputs.
  Splitting dk (rows) would need a reduction on every token.
- **q and k are needed whole** (the `k^T S` sum runs over all 128 k dims, and l2norm needs the full
  vector). Their *weights* are still split by quarter, then the 4 cores swap their quarters after the
  conv (2 tiles over the NoC). Storing full q/k weights on every core instead would cost ~200 KB more per
  core per layer in bfp8 and one resident layer.
- **out_proj split by rows (no concatenation of o).** Core c multiplies its 32 channels of y by its 32
  rows of `W_o^T`, giving a partial `[B, 1024]`. A reduce-scatter sums the 64 partials: core j < 32 owns
  output columns `32j .. 32j+31`. Per-core weight bytes are the same as a column split, but y never
  leaves its core and only partial sums move.

## 2. One decode step, per core (`kernels/`)

```
x (B rows) -- core 0 fetches once, multicasts to all 64 cores
  |
  v  in_proj (in-kernel, resident weights)         -> P = [q/4 k/4 v/4 z/4 ab]
  v  causal conv1d + SiLU (conv state in L1)        -> q/4 k/4 v/4
  v  q/k quarter swap inside the head's 4 cores     -> full q, k of the head
  v  l2norm, g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)
  v  gated delta rule (recurrent state in L1, updated in place)
  v  RMSNorm over the head (4 partial sums of squares swapped in the group) * w * silu(z)
  v  out_proj partial (resident rows)               -> 32 partial output tiles
  v  reduce-scatter: real rows only, to the 32 owner cores, summed there
out (B rows, row-major)
```

Three RISC-V kernels per core: the **reader** prepares inputs (x tiles, conv shifts, per-token selector
tiles), the **compute** kernel runs the math, the **writer** does all cross-core traffic (x multicast,
q/k swap, norm exchange, reduce-scatter).

## 3. How the matrix multiplications are done

Everything is tile matmuls on the matrix unit (FPU) with fp32 accumulation in DST and HiFi4 fidelity.
Weights go to srcA in their storage format (bf16 / bfp8 / bfp4); activations are fp32 in srcB.

**in_proj, `[B, 1024] x [1024, 160]`.** The core's weight slice is an L1-sharded tensor bound directly to
a circular buffer, so the matrix unit reads the weights in place: no copy, no DRAM. The 5 output tiles
accumulate in DST across all 32 K tiles; x arrives in 4-tile K chunks. Each call is a
`matmul_block(ct_dim=5)`: one x tile against the row of 5 weight tiles, so the x tile is unpacked once
instead of five times.

**out_proj, `[B, 32] x [32, 1024]`.** `matmul_block(ct_dim=8)`: the y tile against 8 weight tiles per
call, 4 calls for the 32 output tiles, all 8 fp32 DST tiles in use.

**Reduce-scatter.** For output tile j, every core sends only the B real rows of its partial (2 x 64 B per
row, never the 32-row tile padding) into owner j's buffer, one row slot per sender. The owner sums the
64 senders with two matmuls against a ones-row tile (`ones_row @ R` sums the rows of R).

**Everything else is also expressed as tile matmuls**, so the FPU rarely has to switch modes:

| Operation | As a matmul |
|---|---|
| causal conv (4 taps, per channel) | `sum_s x_shift_s @ diag(w_tap_s)` |
| row sums (`\|q\|^2`, `\|k\|^2`, `q.k`, RMSNorm sum of squares) | `(a * b) @ ones_col` |
| pick this head's a, b out of the a/b tile | `ab @ selector` |
| norm weight | `y @ diag(w)` |
| recurrence `u = v - (d k)^T S` | `I @ v + sum_i (-d k)_i @ S_i` |
| readout `o = q^T S_new` | `sum_i (d q)_i @ S_i + (beta q.k) u` |
| state update `S += (d-1) S + (beta k) u^T` | `diag(d-1) @ S + kb_col @ u`, packed with **L1 accumulation** |

The state update only sends the *increment* through the FPU (which reads fp32 as tf32 and truncates)
and lets the packer add it to the fp32 state in L1. That keeps the state at full fp32 and scales the
truncation error by `1 - d`.

## 4. Optimizations

Device time per layer, bfp8, batch 1, from the device profiler:

| Step | in_proj | out_proj | layer |
|---|---|---|---|
| first version: tile-by-tile matmuls, every core reads the full x tiles | 45 us | 10 us | 126 us |
| `matmul_block` (x tile x a row of 5 / 8 weight tiles per call) | 42 us | 7 us | 125 us |
| **read only the B real rows of x, not the 32-row tile padding** | **14 us** | 7 us | **93 us** |
| core 0 fetches x once and multicasts it; tiles rebuilt by local DMA | 13 us | 5 us | 90 us |
| reduce-scatter: wait for "flushed" per tile, one write barrier at the end | 13 us | 5 us | **87 us** |

What the profiling showed:

- **in_proj was waiting for its input, not computing.** All 64 cores pulled the same 128 KB of padded x
  tiles from the same L1 banks (37 us); at batch 1 only 4 KB of it is real. Fetching only the real rows
  fixed most of it; a single fetch plus NoC multicast fixed the rest.
- **Rebuilding tiles with RISC-V loads/stores is slow** (several cycles per L1 access: 15.7 us). The same
  copy as local NoC DMA took 8.6 us.
- **Math fidelity hardly matters for speed here**, because the matmuls are not the bottleneck:

  | Fidelity (with matmul_block) | Layer time | Error vs HF |
  |---|---|---|
  | HiFi4 | 124.7 us | 0.52% |
  | HiFi3 | 123.2 us | 0.57% |
  | HiFi2 | 122.3 us | 3.8% |
  | LoFi | 121.3 us | 18% |

  so the kernel stays at HiFi4.
- **Tried and reverted:** 8-tile x chunks saved ~4 us more but cost ~36 KB of L1 per core and a resident
  layer in some configurations. The x rows are instead staged in a buffer that compute only uses later in
  the step, so the multicast costs no extra L1.
- **Host dispatch** (~200 us per launch) is larger than the kernel itself; trace capture
  (`ttnn.begin_trace_capture` / `execute_trace`) removes it and is what the benchmarks below use.

## 5. Metrics

### Correctness (vs Hugging Face `Qwen3_5GatedDeltaNet`)

| Weights | Kernel error (HF with the same quantised weights) | Total error (HF with bf16 weights) |
|---|---|---|
| bf16 | 0.49-0.52%, PCC 0.999999 | 0.50% |
| bfp8 | 0.49-0.52% | 1.26%, PCC 0.99994 |
| bfp4 | 0.50-0.52% | 17.6%, PCC 0.985 (too lossy) |

Tested with 8 decode steps from zero state at batch 1 / 4 / 8, and with 2 / 3 / 6 layers resident at
once, each layer against its HF counterpart.

### How many layers fit in L1 (measured on the card)

| Batch | bf16 | bfp8 | bfp4 |
|---|---|---|---|
| 1 | 2 | **3** | 6 |
| 2 | 2 | 3 | 5 |
| 4 | 1 | 3 | 4 |
| 8 | 1 | 2 | 3 |
| 16 | 1 | 1 | 1 |
| 32 | 0 | 0 | 0 |

~370 KB per core goes to the kernel's working buffers, shared by all layers; the rest holds weights and
states.

### Weights in L1 vs streamed from DRAM (500 traced decode steps)

Same kernel, same per-core slices; `weights_in_dram=True` keeps the weights in DRAM and streams them in
every step. States are in L1 in both.

| Config | Weights in L1 | Weights in DRAM | Speed-up |
|---|---|---|---|
| bfp8, 3 layers, batch 1 | 91 us/layer (272 us/step) | 150 us/layer | 1.65x |
| bf16, 2 layers, batch 1 | 92 us/layer (184 us/step) | 215 us/layer | 2.3x |
| bfp8, 3 layers, batch 4 | 143 us/layer (429 us/step) | 188 us/layer | 1.3x |

### Energy per token

Board power from the card's hwmon sensor, sampled every 5 ms over ~5 s of traced decode. Idle with the
device open: 19.6 W. GDN layers only:

| Config | Weights | Tokens/s | Board power | Energy / token / layer | Tokens/J |
|---|---|---|---|---|---|
| bfp8, 3 layers, batch 1 | L1 | 3,611 | 42.5 W | 3.9 mJ | 85 |
| | DRAM | 2,224 | 41.4 W | 6.2 mJ | 54 |
| bf16, 2 layers, batch 1 | L1 | 5,193 | 43.9 W | 4.2 mJ | 118 |
| | DRAM | 2,327 | 38.6 W | 8.3 mJ | 60 |
| bfp8, 3 layers, batch 4 | L1 | 9,336 | 43.1 W | 1.5 mJ | 216 |
| | DRAM | 7,074 | 43.3 W | 2.0 mJ | 163 |

Power is about the same either way, so energy follows time: L1-resident weights use 25-49% less energy
per token. About half the board power is idle power, so batching is the biggest energy lever.

## 6. Ideas and next steps

**Make the rest of the layer faster.** The projections are now ~18 of the ~87 us. Measured before the
projection work, the other phases were: conv 14 us, q/k norms 13 us, gating scalars 11 us, RMSNorm 10 us,
recurrence ~9 us per sequence. Most of these are many small FPU/SFPU ops on single tiles, each with its
own init; fusing them and running the per-row scalar math on fewer tiles should cut them substantially.

**Fit more layers.** The ~370 KB of working buffers is the fixed cost. Single-buffering more of them,
packing the conv state (only ~1.2 KB of each 4 KB tile is real), and sharing buffers between phases would
leave room for a 4th bfp8 layer at batch 1.

**Fuse across layers.** Today each layer is a launch, and its output goes through an L1 tensor that the
next layer reads back. The 32 owner cores could instead multicast their output rows straight into every
core's input buffer for the next layer, and one launch could run all resident layers back to back.

**Rotate 3-layer groups through L1.** The model is 3 GDN layers then 1 full-attention layer, repeated 6
times. One group of 3 GDN layers (bfp8, ~0.25 MB per core per layer) fits with all 18 layers' states
resident (~20 MB at batch 1). While the attention layer runs elsewhere (another accelerator), the next
group's weights are loaded into L1. Weights are read-only, so the old group is simply overwritten; only
states would ever need writing back, and at batch 1 they all stay resident. This hides the weight load
behind the attention time; it does not remove the DRAM traffic, so it wins when the attention window is
at least as long as the load (~0.17 ms per bfp8 group at ~200 GB/s, less with a dedicated copy kernel).

**Use more chips to hold everything.** All 18 GDN mixers fit in about 2 chips' L1 in bfp4 (~107 MB plus
~18 MB of states), or 3-4 chips in bfp8. Then no GDN weight ever touches DRAM, and the same head split
extends across chips.

**Batch for energy.** Idle power is half the board power; going from batch 1 to batch 4 already gives 2.5x
the tokens per joule. The decode cost per step is mostly per 32-row tile, so batches up to the L1 limit
are nearly free except the recurrence (~9 us per sequence).

**Make the comparison fair.** The DRAM streamer used for comparison is simple (one reader, a barrier per
chunk, ~60-85 GB/s). A tuned streamer would narrow the gap; with weights in L1 the layer is now
compute-bound, so the L1 advantage grows as the compute gets faster.

**Complete the decoder layer.** Input RMSNorm, the residual adds and the MLP (3 x 1024 x 3584) around the
mixer; the MLP is as large as the GDN mixer, so it needs the same weight-split treatment (or bfp4).

## Files

| File | |
|---|---|
| `gdn_l1.py` | `Qwen35_08B_GDNStackL1`: weight split, L1 tensors, program descriptors, decode / traced decode |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the fused per-core layer kernel |
| `test_gdn_l1.py` | decode tests vs HF (L1 and DRAM weight modes) |
| `bench_decode.py` | 500-step decode, L1 vs DRAM, plain dispatch and traced |
| `bench_energy.py` | power sampling and energy per token |
