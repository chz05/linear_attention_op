# Qwen3.5-0.8B Gated DeltaNet on Wormhole n150, all weights in L1

The linear-attention (GDN) layers of Qwen3.5-0.8B, projections included, with **every weight and
every state resident in L1**. Nothing touches DRAM during a forward pass: weights, conv taps, norm
weights and both states live in L1-sharded tensors, and the activations between layers go through a
small L1-interleaved tensor. **Decode only**: one call is one token for each sequence in the batch.

Model (from `Qwen/Qwen3.5-0.8B` config.json): 24 layers, 18 GDN + 6 full attention (layers 3, 7, 11,
15, 19, 23); hidden 1024, 16 k-heads, 16 v-heads, dk = dv = 128, conv kernel 4. This code covers the
GDN mixer only (no input RMSNorm, residual or MLP).

See [OPTIMIZATION.md](OPTIMIZATION.md) for the weight split, how the matmuls are done, the optimizations,
all metrics and next ideas.

## Usage

```python
from gdn_l1 import Qwen35_08B_GDNStackL1

stack = Qwen35_08B_GDNStackL1(device, hf_gdn_layers, batch=B, weight_dtype=ttnn.bfloat8_b)
y = stack(x_t)               # one decode step: [B, 1024] -> [B, 1024], layers back to back
y = stack(x_t, layers=[i])   # one layer; each layer keeps its own L1 conv + recurrent state
stack.reset_state()          # new sequences
```

```
cd wormhole_n150/qwen3.5_0.8B && python -m pytest -s test_gdn_l1.py
```

## Design

Core c (8x8 grid, row-major) owns head h = c // 4 and column quarter s = c % 4 (32 of the head's 128
v columns). Splitting the heads splits the weights; each slice stays in its core's L1.

| Per core, per layer | Shape | bf16 | bfp8 | bfp4 |
|---|---|---|---|---|
| in_proj slice: [q/4 \| k/4 \| v/4 \| z/4 \| a,b of all 16 heads] | [1024, 160] | 320 KB | 170 KB | 90 KB |
| out_proj rows of its 32 channels (W_o^T) | [32, 1024] | 64 KB | 34 KB | 18 KB |
| conv tap diagonals + norm weight (bf16) | 13 tiles | 26 KB | 26 KB | 26 KB |
| recurrent state, per sequence (fp32) | [128, 32] | 16 KB | 16 KB | 16 KB |
| conv state, per sequence (fp32) | 1 tile | 4 KB | 4 KB | 4 KB |

One fused kernel per layer (`kernels/`):

1. **in_proj** in the kernel, from the resident slice.
2. **conv1d + SiLU** on the core's q/k/v quarters (conv state in L1).
3. **q/k swap**: the head's 4 cores exchange their quarters (NoC, 2 tiles), so each has the full head.
4. **l2norm, gating** (g, beta) and the **gated delta rule** (recurrent state in L1).
5. **Gated RMSNorm**: the head's 4 partial sums of squares are exchanged within the group.
6. **out_proj split by rows**: each core computes a partial `[B, 1024]` from its 32 channels, followed by
   a reduce-scatter. Core j < 32 owns output columns 32j..32j+31, every core sends it the B real rows of
   that partial (2 x 64 B per row, never the tile padding), and core j sums the 64 contributions.

Row b of every tile is sequence b (batch up to 32). The conv inputs always come from the conv state
(the last 3 inputs of each sequence), and each row updates its own recurrent state slot.

## Results (n150, tt-metal v0.77)

**How many layers fit in L1**, measured on the card (matches `l1_bytes_per_core`); usable L1 is
1361 KB per core, of which ~370 KB is the kernel's working buffers:

| Batch | bf16 | bfp8 | bfp4 |
|---|---|---|---|
| 1 | 2 | **3** | 6 |
| 2 | 2 | 3 | 5 |
| 4 | 1 | 3 | 4 |
| 8 | 1 | 2 | 3 |
| 16 | 1 | 1 | 1 |
| 32 | 0 | 0 | 0 |

**Accuracy** vs HF `Qwen3_5GatedDeltaNet`:

| Weights | vs HF with the same quantised weights (kernel error) | vs HF with bf16 weights (total) |
|---|---|---|
| bf16 | 0.49-0.52%, PCC 0.999999 | 0.50% |
| bfp8 | 0.49-0.52% | 1.26%, PCC 0.99994 |
| bfp4 | 0.50-0.52% | 17.6%, PCC 0.985 (too lossy) |

Tested: 8 decode steps from zero state at batch 1 / 4 / 8, and 2 / 3 / 6 layers resident at once
(3 layers also at batch 4), each layer against its HF counterpart.

**Decode time per layer** (bfp8, batch 1, device profiler): **~87 us**, all on chip. Projections:
in_proj ~13 us (overlapped with the input fetch), out_proj ~5 us; the rest is conv, norms, gating,
recurrence (~9 us per sequence) and the RMSNorm.

### Projection optimizations

| Step | in_proj | out_proj | layer |
|---|---|---|---|
| first version (tile-by-tile matmuls, every core reads the full x tiles) | 45 us | 10 us | 126 us |
| `matmul_block`: one x tile against a row of 5 / 8 weight tiles per call | 42 us | 7 us | 125 us |
| read only the real rows of x (B rows, not the 32-row tile padding) | 14 us | 7 us | 93 us |
| core 0 fetches x once and multicasts it; tiles rebuilt by local DMA | 13 us | 5 us | 90 us |
| reduce-scatter: wait for "flushed" per tile, one write barrier at the end | 13 us | 5 us | 87 us |

The math fidelity barely matters here (HiFi4 -> LoFi saves ~1 us, at up to 18% error), so the kernel
stays at HiFi4. in_proj was waiting on its input, not on the matrix unit: 64 cores pulling the same
128 KB of padded x tiles from the same L1 banks. Larger x chunks would save ~4 us more but cost a
resident layer in some configurations, so they were not kept.

## L1-resident vs DRAM weights (500 decode steps)

`weights_in_dram=True` runs the same kernel with the same per-core weight slices, but the weights live
in DRAM and the reader streams them in every step (in_proj in chunks overlapped with the matmul,
conv/norm constants and out_proj during earlier phases). States stay in L1 in both. `bench_decode.py`,
traced (the step is recorded once with `ttnn.begin_trace_capture` and replayed, so host dispatch drops
out; device time from the profiler agrees within ~5%):

| Config | Weights in L1 | Weights in DRAM | Speed-up |
|---|---|---|---|
| bfp8, 3 layers, batch 1 | 91 us/layer (272 us/step) | 150 us/layer | 1.65x |
| bf16, 2 layers, batch 1 | 92 us/layer (184 us/step) | 215 us/layer | 2.3x |
| bfp8, 3 layers, batch 4 | 143 us/layer (429 us/step) | 188 us/layer | 1.3x |

Without trace, every launch costs ~200 us of host dispatch, more than the kernel itself, and the two
modes look almost the same. The DRAM streamer is simple (one reader, a barrier per chunk), so a better
one would narrow the gap; with weights in L1 the layer is compute-bound.

## Energy per token (`bench_energy.py`)

Board power from the card's hwmon sensor (`/sys/class/hwmon/hwmon*/power1_input` of the `wormhole`
device at the PCI address tt-smi reports; whole-watt resolution, board level), sampled every 5 ms over
~5 s of traced decode. Idle with the device open: 19.6 W. One token per sequence per step, GDN layers
only:

| Config | Weights | Step | Tokens/s | Board power | Energy / token | / token / layer | Tokens/J |
|---|---|---|---|---|---|---|---|
| bfp8, 3 layers, batch 1 | L1 | 277 us | 3,611 | 42.5 W | 11.8 mJ | 3.9 mJ | 85 |
| | DRAM | 450 us | 2,224 | 41.4 W | 18.6 mJ | 6.2 mJ | 54 |
| bf16, 2 layers, batch 1 | L1 | 193 us | 5,193 | 43.9 W | 8.5 mJ | 4.2 mJ | 118 |
| | DRAM | 430 us | 2,327 | 38.6 W | 16.6 mJ | 8.3 mJ | 60 |
| bfp8, 3 layers, batch 4 | L1 | 428 us | 9,336 | 43.1 W | 4.6 mJ | 1.5 mJ | 216 |
| | DRAM | 565 us | 7,074 | 43.3 W | 6.1 mJ | 2.0 mJ | 163 |

Power is about the same either way, so energy per token follows step time: L1-resident weights use
25-49% less energy per token. Roughly half the board power is idle power (19.6 W), so batching helps
the most: batch 4 is 2.5x more tokens per joule than batch 1. tt-smi's one-off power readings taken
mid-run vary (22-46 W) and are not used; the table uses the averaged hwmon samples.

## Not done yet

- The rest of the layer (conv 14 us, q/k norms 13 us, gating scalars 11 us, RMSNorm 10 us) is now the
  bulk of the time. The working buffers (~370 KB) could shrink to fit more layers.
- Rotating 3-layer groups through L1 (load the next group while full attention runs elsewhere).
- RMSNorm / residual / MLP around the mixer.
