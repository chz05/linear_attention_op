# Qwen3.5-9B Gated DeltaNet on Wormhole n150

The linear-attention (Gated DeltaNet, GDN) layer of Qwen3.5-9B on a single n150 (8x8 Tensix grid),
**including the projections**: one call runs the full layer, from hidden states in to hidden states out,
and matches Hugging Face `Qwen3_5GatedDeltaNet`.

```
x [B*T, 4096]
  -> in_proj_qkv, in_proj_z, in_proj_a, in_proj_b       (projections)
  -> causal conv1d (4 taps) + SiLU on q, k, v
  -> l2norm(q), l2norm(k);  g = -exp(A_log) * softplus(a + dt_bias),  beta = sigmoid(b)
  -> gated delta rule                                     (recurrent state in L1)
  -> RMSNorm over each head * norm.weight * silu(z)
  -> out_proj                                             (projection)
  -> [B*T, 4096]
```

Qwen3.5-9B GDN geometry: hidden 4096, 16 k-heads, 32 v-heads, dk = dv = 128. The model has 32 layers,
24 of them GDN (every 4th layer is full attention, not covered here). This code runs one GDN layer;
the full model is not wired up.

## Usage

```python
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from gdn_layer import Qwen35GatedDeltaNetTT

layer = Qwen35GatedDeltaNetTT(device, hf_layer, batch=B)  # weights split by head, states allocated in L1
y = layer(x)            # x: [B, T, 4096] -> [B, T, 4096]   (prefill: T tokens)
y = layer(x_next)       # [B, 1, 4096] decode step, continues from the L1 conv + recurrent state
layer.reset_state()     # start new sequences
S = layer.get_recurrent_state()   # [B, 32, 128, 128] copy to host, for checks only
```

`hf_layer` is a `Qwen3_5GatedDeltaNet` (random or loaded weights). `weight_dtype=ttnn.bfloat8_b` stores
the projection weights in bfp8 (half the DRAM traffic, ~1.2% error instead of ~0.6%).

```
cd wormhole_n150/qwen3.5_9B && python -m pytest -s test_gdn_layer.py test_gated_delta_l1.py
```

## Weights and shapes

| HF weight | Shape | bf16 |
|---|---|---|
| `in_proj_qkv` | [8192, 4096]: q 2048 (16x128), k 2048, v 4096 (32x128) | 67.1 MB |
| `in_proj_z` | [4096, 4096] | 33.6 MB |
| `in_proj_a`, `in_proj_b` | [32, 4096] each | 0.5 MB |
| `conv1d.weight` | [8192, 1, 4] | |
| `A_log`, `dt_bias` | [32] | |
| `norm.weight` | [128] | |
| `out_proj` | [4096, 4096] | 33.6 MB |

On device:

| Tensor | Shape | Where |
|---|---|---|
| `w_in` (qkv + z, re-ordered by core) | [4096, 12288] bf16; core c's 192 columns = [q/4 \| k/4 \| v/2 \| z/2] | DRAM |
| `w_ab` | [4096, 64] (a \| b) | DRAM |
| `w_out` (out_proj transposed) | [4096, 4096]; core c reads columns 64c..64c+63 (all rows) | DRAM |
| per-core constants (conv taps, norm weight, selectors) | 20 fp32 tiles per core | L1 |
| P = in_proj output | [Mp, 12288] fp32, [Mp, 192] per core | L1 (width-sharded) |
| y = norm output | [Mp, 4096] fp32, [Mp, 64] per core | L1 (width-sharded) |
| recurrent state | [B, 32, 128, 128] fp32; [B*128, 64] per core (B x 32 KB) | L1, permanent |
| conv state | last 3 inputs of the core's 128 channels (B x 16 KB per core) | L1, permanent |

## Design: split the heads, split the weights with them

Core `c` (row-major over the 8x8 grid) owns v-head `h = c // 2` and column half `s = c % 2`.

```
x ──► in_proj (ttnn matmul, column-parallel) ──► core c's L1: [q/4 | k/4 | v/2 | z/2]
        w_in columns re-ordered so core c's output columns are exactly its heads' columns
      │
      ▼  fused kernel, per core, local L1 (kernels/layer_*.cpp)
   conv1d + SiLU (conv state in L1)
   q/k quarters exchanged inside the 4-core group of the k-head  ◄── NoC, 2 tiles / block
   l2norm, g, beta
   gated delta rule (recurrent state in L1)
   RMSNorm over the head * w * silu(z)  ◄── partial sum of squares swapped with the partner core
      │
      ▼
   out_proj (ttnn matmul): every core multicasts its y slice, so each core has the concatenated y;
   core c then computes output columns 64c..64c+63 with all 4096 rows of w_out[:, those columns]
```

- **in_proj, column-parallel.** No weight column is read twice: the 4 cores that share a k-head (2 v-heads
  x 2 halves) each project one quarter of its q and k, and swap quarters after the conv. That swap is the
  only cross-core traffic before the recurrence. The matmul writes each core's columns straight into
  that core's L1, where the fused kernel reads them.
- **Fused kernel, all local.** Conv, norms, gating, recurrence and the gated RMSNorm run on the core that
  owns the head, from its own L1. The RMSNorm needs the whole head (128 values), so the two halves swap
  one partial sum of squares.
- **out_proj: concatenate y, split w_out by output columns.** The 1D-multicast matmul multicasts each
  core's `y` slice to all cores (an all-gather), then core c computes its 64 output columns from the full
  `y` and `w_out[:, 64c:64c+64]`. The sum over heads happens inside each core's own matmul, so no
  partial sums are reduced. The alternative, keeping `y` local and splitting `w_out` by rows, gives each
  core a partial `[M, 4096]` that needs a reduce-scatter across the 64 cores. Same weight bytes, but 126
  point-to-point tiles per core instead of one 2-tile multicast; it becomes attractive when each core keeps
  its own `w_out` rows resident in L1.
- **a/b** (one scalar per head per token) come from a small separate projection.
- **Rows are tokens**, sequence-major: `B * T` rows per launch, up to 128. Longer inputs are split into
  several launches; the conv and recurrent state carry across them.

### The recurrence

```
S = exp(g) * S          # per-head decay
u = v - k^T S           # delta-rule error
S = S + (beta * k) u^T  # state update
o = q^T S               # readout
```

- **State stays in L1.** It is an L1-sharded tensor that the kernel updates in place, so it never goes
  to DRAM between calls.
- **V-column split needs no reduce.** `k^T S`, `q^T S` and the update of column j only touch column j, so
  each core computes its own output columns. A reduce would only be needed for a split along dk rows,
  and then on every token.
- **All-matmul compute.** Each token is turned into small row / column / diagonal tiles so every step is
  a tile matmul: `u = I v - (d k)^T S`, `o = (d q)^T S + (beta q.k) u`, `S += (d-1) S + (beta k) u^T`.
- **Precision.** The matrix unit reads fp32 as tf32 and truncates toward zero. Only the state *increment*
  goes through it; the packer adds it to the fp32 state in L1 (L1 accumulation). That scales the
  truncation error by `1 - d`: at T = 1024 with d ~ 0.99 the error is 0.2-0.5% (4.7% when the whole state
  went through the matrix unit every token).

`gated_delta_l1.py` (`GatedDeltaRuleL1`) is this recurrence on its own (inputs q/k/v/g/beta, no
projections), with its own tests in `test_gated_delta_l1.py`.

## Results (n150, tt-metal v0.77)

**Accuracy** vs HF fp32 with the same bf16 weights: output within 0.55-0.63%, PCC 0.999998, for decode,
40-token prefill, batch 4, 20-token prefill followed by 12 decode steps, and a 300-token chunked prefill.
HF itself run in bf16 is 0.65-0.8% off fp32, so this is bf16-level. Most of the error is a consistent
~0.6% shrink from tf32 truncation at each elementwise step. The projections must use HiFi4: HiFi2 drops
low mantissa bits of the bf16 inputs and made the shrink 1.5%.

**Device time per decode step** (one layer, one token, device profiler):

| | bf16 weights | bfp8 weights |
|---|---|---|
| in_proj (100 MB / 50 MB of weights) | 517 us | 268 us |
| a/b projection | 42 us | 42 us |
| fused GDN kernel | 74 us | 74 us |
| out_proj (34 MB / 17 MB) | 161 us | 105 us |
| total | ~795 us | ~490 us |

The projections run at ~195 GB/s, close to the DRAM limit, so streaming the weights is most of the
cost; the fused kernel is under 10%. End-to-end wall time is higher (host dispatch).

**Prefill** uses the same token-by-token recurrence, about 16 us per token in the fused kernel, plus the
projections once per 128-token launch.

## Not done yet

- Trace capture to remove host dispatch cost; folding the a/b projection into in_proj.
- DRAM prefetching of the projection weights across op boundaries (`ttnn.dram_prefetcher`).
- Chunked (matmul-based) prefill instead of token-by-token.
- Only the Qwen3.5-9B geometry (16 / 32 heads, 128 dims, full 8x8 grid) is supported.

## Debugging

`layer.debug_defines = [("DEBUG_Y_IS_O", "1")]` returns the pre-norm `o`; `("DEBUG_STAGE", "1")` / `"2"`
returns the conv output `v` / the per-token `d - 1` and `w`. Set `layer.debug_return_y = True` to skip
out_proj and get the fused kernel's output directly.

## Files

| File | |
|---|---|
| `gdn_layer.py` | `Qwen35GatedDeltaNetTT`: full layer with projections, head-split weights, L1 states |
| `kernels/layer_reader.cpp` | conv shifts + conv state, a/b fetch, per-token recurrence tiles |
| `kernels/layer_compute.cpp` | conv, l2norm, gating, recurrence, gated RMSNorm |
| `kernels/layer_writer.cpp` | k-head group q/k exchange, output rows, partner sum-of-squares exchange |
| `test_gdn_layer.py` | full layer vs HF `Qwen3_5GatedDeltaNet` |
| `gated_delta_l1.py` | `GatedDeltaRuleL1`: recurrence only (no projections) |
| `kernels/gdn_*.cpp` | kernels of the recurrence-only op |
| `reference.py` | PyTorch recurrence reference (same math as HF `torch_recurrent_gated_delta_rule`) |
| `test_gated_delta_l1.py` | recurrence-only op vs the reference |
