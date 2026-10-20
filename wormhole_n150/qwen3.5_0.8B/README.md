# Qwen3.5-0.8B Gated DeltaNet on Wormhole n150, all weights in L1

The linear-attention (GDN) layers of Qwen3.5-0.8B, projections included, with **every weight and
every state resident in L1**. Nothing touches DRAM during a decode step: weights, conv taps and both
states live in L1-sharded tensors, and the activations between layers go through a small L1 tensor.
**Decode only**: one call is one token for each sequence in the batch (batch 1-16).

Model (from `Qwen/Qwen3.5-0.8B` config.json): 24 layers, 18 GDN + 6 full attention (layers 3, 7, 11,
15, 19, 23); hidden 1024, 16 k-heads, 16 v-heads, dk = dv = 128, conv kernel 4. This code covers the
GDN mixer only (no input RMSNorm, residual or MLP).

**See [OPTIMIZATION.md](OPTIMIZATION.md)** for the weight split, how the matmuls are done, both
optimization rounds, the per-batch breakdown, all metrics and next ideas.

## Usage

```python
from gdn_l1 import Qwen35_08B_GDNStackL1

stack = Qwen35_08B_GDNStackL1(device, hf_gdn_layers, batch=B, weight_dtype=ttnn.bfloat8_b)
y = stack(x_t)               # one decode step: [B, 1024] -> [B, 1024], layers back to back
y = stack(x_t, layers=[i])   # one layer; each layer keeps its own L1 conv + recurrent state
stack.reset_state()          # new sequences
```

For traced decode, `allocate_activations()` + `forward_into()` run a whole step into persistent buffers
that `ttnn.begin_trace_capture` can record (see `bench_decode.py`).

```
cd wormhole_n150/qwen3.5_0.8B
python -m pytest -s test_gdn_l1.py     # vs Hugging Face, batch 1-16, L1 and DRAM weight modes
python bench_decode.py                 # 500 decode steps, weights in L1 vs DRAM
python bench_energy.py                 # tokens/s, board power, energy per token
```

## Design in one paragraph

Core c (8x8 grid) owns head `c // 4` and a quarter of its v columns. Its L1 holds its in_proj slice
(`[q/4 | k/4 | v/4 | z/4 | a,b]`, 1024 x 160), its 32 rows of out_proj (with the RMSNorm weight folded in),
its conv taps and, per sequence, its slice of the recurrent and conv state. One fused kernel per layer
runs in_proj, conv, a q/k quarter swap inside the head's 4 cores, the norms and gating, the recurrence
(state updated in place) and the gated RMSNorm, then a partial out_proj whose 64 partials are
reduce-scattered to 32 owner cores. Activations use a face-major layout so a tile's batch rows always move
in 2 transfers.

## Results (n150, tt-metal v0.77)

| | Batch 1 | Batch 4 | Batch 8 | Batch 16 |
|---|---|---|---|---|
| Layer time on chip (bfp8, profiler) | **71 us** | 91 us | 117 us | 171 us |
| bfp8 layers that fit in L1 | 3 | 3 | 2 | 1 |

- **Accuracy**: 0.38-0.41% vs Hugging Face with the same weights (PCC 0.999999); bfp8 weights add ~1%
  vs the model's bf16 weights, bfp4 is too lossy (17.6%).
- **L1 vs DRAM weights** (500 traced steps): 1.8x faster with bfp8 at batch 1 (74 vs 134 us/layer), 2.6x
  with bf16, 1.37x at batch 8.
- **Energy**: 23-52% less energy per token with weights in L1; 95 tokens/J for 3 bfp8 layers at batch 1,
  581 tokens/J for 2 layers at batch 8.

Details and the full tables are in [OPTIMIZATION.md](OPTIMIZATION.md).
