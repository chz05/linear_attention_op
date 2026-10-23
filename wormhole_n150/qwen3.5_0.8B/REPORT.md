# Qwen3.5-0.8B linear attention on a Wormhole n150: optimization report

Decode for the Gated DeltaNet (linear attention) layers of Qwen3.5-0.8B with every weight and every state resident in
the L1 SRAM of one n150 card (8 × 8 = 64 Tensix cores, ~1.36 MB of L1 each), so a decode step never touches DRAM.
Code: [`gdn_l1.py`](gdn_l1.py) (host side) and [`kernels/`](kernels) (one fused reader / compute / writer kernel per
layer). Details and the full history: [OPTIMIZATION.md](OPTIMIZATION.md).

## 1. The linear-attention layer

Qwen3.5-0.8B has 24 decoder layers: 18 Gated DeltaNet (GDN) layers and 6 full-attention layers (3, 7, 11, 15, 19, 23).
A GDN layer has hidden size 1024, 16 key heads and 16 value heads of dimension 128, and a causal convolution with 4
taps. This work covers the GDN mixer in decode: one call is one token for each of B sequences (B = 1..16).

```
x [B, 1024]
 ├─ in_proj     P = x · W_in                           q, k, v, z: 2048 columns each; a, b: 16 each (one per head)
 ├─ conv1d      q, k, v = SiLU(causal_conv(P_qkv))     4 taps per channel; conv state = the last 3 inputs
 ├─ l2 norm     q = q / |q| · 128^-0.5,  k = k / |k|   per head
 ├─ gates       d = exp(-exp(A_log) · softplus(a + dt_bias)),  β = sigmoid(b)       per head and sequence
 ├─ recurrence  S = d·S;  u = v − Sᵀk;  S = S + β·k·uᵀ;  o = Sᵀq                    S: 128 × 128 per head and sequence
 ├─ norm        y = RMSNorm(o) · w · SiLU(z)           per head
 └─ out_proj    out = y · W_out                        [B, 2048] × [2048, 1024]
```

## 2. How the layer is split over the 64 Tensix cores

Core c (row-major over the 8 × 8 grid) owns head h = c / 4 and column quarter s = c % 4, i.e. value channels
32c .. 32c + 31. Every weight slice and state lives in that core's L1 for the life of the layer.

```
grid row 0   cores  0- 7   heads  0,  1   ┐
   ...                                    ├─ half 0: value channels    0..1023
grid row 3   cores 24-31   heads  6,  7   ┘
grid row 4   cores 32-39   heads  8,  9   ┐
   ...                                    ├─ half 1: value channels 1024..2047
grid row 7   cores 56-63   heads 14, 15   ┘
```

| Piece | Split |
|---|---|
| in_proj | core c holds the columns [q/4 \| k/4 \| v/4 \| z/4] of its head quarter plus a, b of all heads: 1024 × 160 |
| conv | the 12 tap diagonals of its q/4, k/4, v/4, and their conv state |
| q, k | each core computes a quarter of its head's q and k; the head's 4 cores exchange the l2-norm partial sums, then the row-scaled quarters d·q, −d·k, β·k, so every core has the head's full vectors |
| v, state | split by value columns: core c keeps S[:, its 32 columns] (128 × 32 fp32 per sequence) and computes its 32 outputs |
| RMSNorm | the head's 4 partial sums of squares are exchanged within its 4 cores |
| out_proj | core c computes output columns 32 (c % 32) .. +31 over the 1024 value channels of its half (a 1024 × 32 slice of W_outᵀ with the RMSNorm weight folded in); y is gathered within each half, and cores c and c ^ 32 add their two partials of the same columns |

## 3. Optimizations

The baseline is the previous version of this kernel (round 2 in [OPTIMIZATION.md](OPTIMIZATION.md)).

1. **Per-phase profiling.** Profiler zones now cover every phase on all five RISC-V cores of each Tensix, and
   `profile_layer.py` turns the device log into a per-core timeline; it showed that the compute zones record when
   instructions are issued, so the real timeline is read from the CB handoffs on the data-movement cores.
   *Files:* [`profile_layer.py`](profile_layer.py), [`kernels/`](kernels).
2. **SFPU work on the batch rows only.** All per-row scalar math (the l2-norm rsqrt, softplus / exp gates, sigmoid,
   RMSNorm rstd) is one fused SFPU pass over column 0 of the batch rows instead of ~18 passes over all 4 faces, and
   the SiLUs touch only rows 0..B-1 (gating 11.1 → 1.6 µs, conv SiLU 7.0 → 1.5 µs).
   *Files:* [`kernels/compute.cpp`](kernels/compute.cpp) (`gating_rows`, `silu_rows`, `silu_mul_rows`, `rstd_rows`).
3. **l2-norm partial sums before the swap.** Each core forms q·q, k·k, q·k of its own quarter inside the conv pass and
   sends one tile of partial sums to its head's 4 cores, which add them with a selector matmul.
   *Files:* [`kernels/compute.cpp`](kernels/compute.cpp) (C1–C3), [`kernels/writer.cpp`](kernels/writer.cpp) (step 1).
4. **Scale, then swap only the batch rows.** Each core row-scales its own quarter (3 multiplies instead of 12 for the
   whole head) and sends only the BP batch rows of each tile (2 small blocks instead of a full 4 KB tile).
   *Files:* [`kernels/compute.cpp`](kernels/compute.cpp) (C4), [`kernels/writer.cpp`](kernels/writer.cpp) (step 2).
5. **Batch-sized input tiles, read in place.** in_proj reads the multicast input directly as BP × 32 tiles (fewer
   matrix-unit passes and less data than 32-row tiles, no copy), at HiFi3 fidelity, which tested accuracy-neutral
   (in_proj 12.8 → 7.9 µs on the math thread).
   *Files:* [`kernels/compute.cpp`](kernels/compute.cpp) (C0), [`kernels/reader.cpp`](kernels/reader.cpp) (step 1),
   [`gdn_l1.py`](gdn_l1.py) (`_cbs`, `in_proj_fidelity`).
6. **Pipelined input multicast.** Core 0 multicasts the input in up to 8 pieces, each followed by a semaphore, so
   in_proj starts on the first piece (batch 16: in_proj finishes at 14 µs instead of 17).
   *Files:* [`kernels/writer.cpp`](kernels/writer.cpp) (step 0), [`kernels/reader.cpp`](kernels/reader.cpp) (step 1).
7. **Conv state read in place.** The conv state is stored as BP × 32 tiles that the conv matmuls use directly, and it
   is rolled with 8 block copies after the conv instead of building and rolling it row by row (36 copies per sequence).
   *Files:* [`kernels/compute.cpp`](kernels/compute.cpp) (C1), [`kernels/reader.cpp`](kernels/reader.cpp) (step 2),
   [`gdn_l1.py`](gdn_l1.py) (`conv_state_tiles`).
8. **Fused recurrence.** One pass over the state yields both u and (d·q)ᵀS, and the readout plus all four state-tile
   updates share a second DST pass (14 tile matmuls per sequence instead of 18, no per-sequence sync), for 2.8 µs per
   sequence instead of 3.2.
   *Files:* [`kernels/compute.cpp`](kernels/compute.cpp) (C5), [`kernels/reader.cpp`](kernels/reader.cpp) (step 3).
9. **RMSNorm with fewer round trips.** The recurrence accumulates every sequence's output into one tile in L1 (no
   per-token copy by the writer), and the head's sums of squares travel as one small block (with item 2:
   7.3 → 3.7 µs).
   *Files:* [`kernels/compute.cpp`](kernels/compute.cpp) (C5–C6), [`kernels/writer.cpp`](kernels/writer.cpp) (step 3).
10. **out_proj as all-gather + column split.** Each grid half gathers its y at one coordinator that multicasts it
    once, each core computes one output tile over its half's channels, and core pairs add one face each, instead of a
    64-way reduce-scatter of 2,048 partial tiles (after-RMSNorm tail: 11 → 6 µs at batch 1, 49 → 19 µs at batch 16).
    *Files:* [`kernels/compute.cpp`](kernels/compute.cpp) (C7–C8), [`kernels/writer.cpp`](kernels/writer.cpp)
    (steps 4–6), [`gdn_l1.py`](gdn_l1.py) (`_Layer`, `_kernels`).
11. **Cheaper exchanges.** Every transfer between cores is followed by a flush and a semaphore increment on the same
    static VC, instead of waiting for all acknowledgements with a write barrier.
    *Files:* [`kernels/writer.cpp`](kernels/writer.cpp).
12. **Smaller working buffers.** Buffers made unnecessary by the changes above were dropped, and the input and the
    gathered y share one buffer, so the kernel's working memory went from 380 to 260 KB per core at batch 1 and
    4 bfp8 layers fit in L1 at batch 1–2 instead of 3.
    *Files:* [`gdn_l1.py`](gdn_l1.py) (`WORK_CBS`, `_cbs`).

## 4. Results

**Layer time on the chip** (device profiler, bfp8 weights, median of 100 launches):

| | Batch 1 | Batch 4 | Batch 8 | Batch 16 |
|---|---|---|---|---|
| before | 71.0 µs | 90.7 µs | 117.1 µs | 171.2 µs |
| **now** | **32.0 µs** | **42.5 µs** | **60.6 µs** | **96.7 µs** |
| speed-up | 2.2× | 2.1× | 1.9× | 1.8× |

**Where the time goes now** (µs per phase, slowest core, taken at the CB handoffs):

| Phase | Batch 1 | Batch 16 |
|---|---|---|
| input multicast | 3.6 | 10.6 |
| in_proj (after the last input piece) | 5.7 | 1.6 |
| conv + SiLU | 3.5 | 6.1 |
| l2-norm sums, gates, scaled q/k swap | 5.7 | 11.1 |
| recurrence + sum of squares | 5.5 | 45.2 |
| RMSNorm (exchange, rstd, gate) | 2.0 | 2.9 |
| y gather | 2.5 | 13.9 |
| out_proj, pair sum, write | 3.5 | 5.3 |
| **total** | **32.0** | **96.7** |

**Decode, traced** (`bench_decode.py`, 500 steps; layers run one after another, all resident in L1 at once):

| Config | Weights in L1 | Weights streamed from DRAM | L1 vs DRAM |
|---|---|---|---|
| bfp8, 3 layers, batch 1 | 74 → **33 µs/layer** (99 µs/step) | 134 → 97 µs/layer | 1.8× → 2.9× |
| bf16, 2 layers, batch 1 | 76 → **35** (70) | 200 → 164 | 2.6× → 4.7× |
| bfp8, 3 layers, batch 4 | 94 → **44** (133) | 147 → 106 | 1.6× → 2.4× |
| bfp8, 2 layers, batch 8 | 122 → **64** (128) | 167 → 122 | 1.4× → 1.9× |
| bfp8, **4 layers**, batch 1 / 2 | did not fit → **33 / 36** (130 / 143) | | |

- **Accuracy** against Hugging Face with the same weights: 0.36–0.39% error, PCC ≥ 0.999998, for batch 1–16
  (was 0.38–0.41%); all 15 tests in `test_gdn_l1.py` pass.
- **L1 capacity** (bfp8): 4 layers at batch 1 and 2 (was 3), 3 at batch 4, 2 at batch 8, 1 at batch 16.

## 5. Energy

Board power from the n150's hwmon sensor during ~5 s of traced decode (`bench_energy.py`); idle power with the device
open is 19.8 W. Energy per token per GDN layer:

| Config | Weights | Tokens/s | Board power | Energy / token / layer | Tokens/J |
|---|---|---|---|---|---|
| bfp8, 3 layers, batch 1 | L1 | 4,345 → **9,416** | 45.7 → 44.0 W | 3.5 → **1.56 mJ** | 95 → **214** |
| | DRAM | 2,492 → 3,397 | 43.2 → 38.6 W | 5.8 → 3.79 mJ | 58 → 88 |
| bf16, 2 layers, batch 1 | L1 | 6,255 → **13,227** | 48.1 → 46.5 W | 3.8 → **1.76 mJ** | 130 → **284** |
| | DRAM | 2,497 → 3,055 | 40.1 → 36.4 W | 8.0 → 5.96 mJ | 62 → 84 |
| bfp8, 3 layers, batch 4 | L1 | 14,045 → **28,429** | 52.5 → 50.5 W | 1.25 → **0.59 mJ** | 267 → **563** |
| | DRAM | 9,053 → 12,486 | 48.0 → 42.8 W | 1.77 → 1.14 mJ | 189 → 292 |
| bfp8, 2 layers, batch 8 | L1 | 32,072 → **59,000** | 55.2 → 53.6 W | 0.86 → **0.45 mJ** | 581 → **1,100** |
| | DRAM | 23,997 → 31,913 | 53.5 → 48.0 W | 1.12 → 0.75 mJ | 448 → 665 |

**Conclusion.** At the same configurations the optimized kernel gets 1.9–2.3× more tokens per joule (bfp8, batch 1:
95 → 214 tokens/J), because each step is about 2× shorter while board power stays at 44–54 W. Keeping the weights in
L1 now uses 40–70% less energy per token than streaming them from DRAM (23–52% before). About 20 W of the board power
is idle power, so batching is still the biggest energy lever: at batch 8 a layer costs 0.45 mJ per token, 3.5× less
than at batch 1.

## 6. Shrinking batch

Sequences finish at random steps, so the batch can drop from 16 to any size down to 0. The kernels take the batch as
a runtime argument (one program per padded batch: 1, 2, 4, 8, 16), and `retire(rows)` moves the remaining sequences'
states up to rows 0..B'−1 on the device in 2–13 µs (~0.1 ms on the host), so every step runs at the speed of the
current batch. *Files:* [`kernels/compact.cpp`](kernels/compact.cpp), [`gdn_l1.py`](gdn_l1.py) (`retire`,
`_compact`), [`kernels/compute.cpp`](kernels/compute.cpp) and [`kernels/reader.cpp`](kernels/reader.cpp) (batch as a
runtime argument), [`test_gdn_l1.py`](test_gdn_l1.py) (`test_decode_shrinking_batch*`), [`bench_shrink.py`](bench_shrink.py).

| Batch | 16 | 13 | 10 | 8 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| µs per layer (bfp8, profiler) | 97.1 | 89.1 | 81.3 | 60.1 | 52.1 | 42.4 | 39.8 | 35.2 | 32.1 |

Every step of a shrinking batch (16 → 10 → 8 → 5 → 3 → 2 → 1 → 0, 16 → 15 → 0, and 16 random sequence lengths) matches Hugging Face
within 0.44%.
