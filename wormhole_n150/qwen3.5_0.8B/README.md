# Qwen3.5-0.8B Gated DeltaNet on Wormhole n150, all weights in L1

The linear-attention (GDN) layers of Qwen3.5-0.8B, projections included, with **every weight and
every state resident in L1**. Nothing touches DRAM during a decode step: weights, conv taps and both
states live in L1-sharded tensors, and the activations between layers go through a small L1 tensor.
**Decode only**: one call is one token for each sequence in the batch (batch 1-16).

Model (from `Qwen/Qwen3.5-0.8B` config.json): 24 layers, 18 GDN + 6 full attention (layers 3, 7, 11,
15, 19, 23); hidden 1024, 16 k-heads, 16 v-heads, dk = dv = 128, conv kernel 4. This code covers the
GDN mixer only (no input RMSNorm, residual or MLP).

**See [REPORT.md](REPORT.md)** for a short report (the layer, the split over the cores, the optimizations
with pointers into the code, results and energy) and **[OPTIMIZATION.md](OPTIMIZATION.md)** for the
details: how the matmuls are done, all three optimization rounds, per-phase breakdowns, all metrics and
next ideas.

## Usage

```python
from gdn_l1 import Qwen35_08B_GDNStackL1

stack = Qwen35_08B_GDNStackL1(device, hf_gdn_layers, batch=B, weight_dtype=ttnn.bfloat8_b)
y = stack(x_t)               # one decode step: [B, 1024] -> [B, 1024], layers back to back
y = stack(x_t, layers=[i])   # one layer; each layer keeps its own L1 conv + recurrent state
stack.reset_state()          # new sequences

# shrinking batch: sequences finish at any step, in any order, down to batch 0
stack = Qwen35_08B_GDNStackL1(device, hf_gdn_layers, batch=16, weight_dtype=ttnn.bfloat8_b)
y = stack(x_t)               # x_t: [stack.batch, 1024], row b = sequence stack.seq_ids[b]
stack.retire([2, 7, 9])      # those rows finished: the others move up (on the device), batch 16 -> 13
```

After `retire`, later steps run with the program of the smaller batch (one per power of 2), so the step gets
faster as the batch shrinks; at batch 0 a call takes and returns an empty tensor. A captured trace has to be
captured again after `retire` (see `bench_shrink.py`).

For traced decode, `allocate_activations()` + `forward_into()` run a whole step into persistent buffers
that `ttnn.begin_trace_capture` can record (see `bench_decode.py`).

```
cd wormhole_n150/qwen3.5_0.8B
python -m pytest -s test_gdn_l1.py     # vs Hugging Face, batch 1-16, shrinking batch, L1 and DRAM weights
python bench_decode.py                 # 500 decode steps, weights in L1 vs DRAM
python bench_energy.py                 # tokens/s, board power, energy per token
python bench_shrink.py                 # 16 sequences finishing at random steps: batch 16 -> 0, traced
python profile_layer.py                # device time per phase and per core (sets TT_METAL_DEVICE_PROFILER)
```

## Design in one paragraph

Core c (8x8 grid) owns head `c // 4` and a quarter of its v columns. Its L1 holds its in_proj slice
(`[q/4 | k/4 | v/4 | z/4 | a,b]`, 1024 x 160), its conv taps, its out_proj slice (the 1024 v channels of
its half of the grid for output columns `32 (c % 32) ..`, RMSNorm weight folded in) and, per sequence, its
slice of the recurrent and conv state. One fused kernel per layer runs in_proj (reading the multicast input
in place as batch-row tiles), the conv (reading its state in place), an exchange of the l2-norm partial sums
inside the head's 4 cores, the gates, a swap of the row-scaled q/k quarters, the recurrence (state updated
in place) and the gated RMSNorm, then out_proj: y is gathered within each half of the grid by one
coordinator, every core computes one output tile over its half's channels, and cores c and c ^ 32 add their
two partials. Vector-unit (SFPU) work only touches the batch rows, and only the batch rows travel between
cores.

## Results (n150, tt-metal v0.77)

| | Batch 1 | Batch 4 | Batch 8 | Batch 16 |
|---|---|---|---|---|
| Layer time on chip (bfp8, profiler) | **32.0 us** (was 71.0) | 42.5 us (90.7) | 60.6 us (117.1) | 96.7 us (171.2) |
| bfp8 layers that fit in L1 | 4 (was 3) | 3 | 2 | 1 |

- **Shrinking batch** (bfp8, profiler): 97.1 us at batch 16, 89.1 at 13, 81.3 at 10, 60.1 at 8, 52.1 at 5,
  42.4 at 4, 39.8 at 3, 35.2 at 2, 32.1 at 1. Retiring sequences takes 2-13 us on the device (moving their
  states) and ~0.1 ms on the host.
- **Accuracy**: 0.36-0.39% vs Hugging Face with the same weights (PCC 0.999998+), also while the batch shrinks
  16 -> 0; bfp8 weights add ~1% vs the model's bf16 weights, bfp4 is too lossy (17.6%).
- **Traced decode** (500 steps): 33 us/layer for 3 bfp8 layers at batch 1 (99 us/step, was 74 us/layer),
  64 us/layer for 2 layers at batch 8 (was 122); 4 resident bfp8 layers: 130 us/step at batch 1.
- **L1 vs DRAM weights**: 2.9x faster with bfp8 at batch 1 (33 vs 97 us/layer), 4.7x with bf16, 1.9x at
  batch 8.
- **Energy**: 40-70% less energy per token with weights in L1; 214 tokens/J for 3 bfp8 layers at batch 1
  (was 95), 1,100 tokens/J for 2 layers at batch 8 (was 581).

Details and the full tables are in [REPORT.md](REPORT.md) and [OPTIMIZATION.md](OPTIMIZATION.md).
