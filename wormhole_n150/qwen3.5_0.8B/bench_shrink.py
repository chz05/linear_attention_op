# SPDX-License-Identifier: Apache-2.0
"""Decode with a shrinking batch: 16 sequences that each finish after a random number of tokens, until none is left.

    python bench_shrink.py                               # 1 bfp8 layer, lengths random in 1..200
    python bench_shrink.py --layers 2 --dtype bfp4 --max-len 500 --seed 3

Between two finishes the batch is constant, so each stretch is one traced step replayed: its first step runs
untraced (it compiles the program of a new padded batch the first time), the rest replay a trace captured for the
current batch. When sequences finish, retire() moves the others' states up on the device and the next stretch
runs with the smaller batch. Prints one line per stretch: batch, steps, traced time per step, retire() time.
"""
import argparse
import time

import torch
import ttnn

from gdn_l1 import Qwen35_08B_GDNStackL1, padded_batch
from test_gdn_l1 import make_hf_layers

DTYPES = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "bfp4": ttnn.bfloat4_b}


def run(dev, n_layers, dtype, lengths):
    stack = Qwen35_08B_GDNStackL1(dev, make_hf_layers(n_layers, seed=7), batch=len(lengths), weight_dtype=DTYPES[dtype])
    x0 = ttnn.from_torch(torch.randn(16, 1024), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    outs = stack.allocate_activations()
    print(f"{n_layers} {dtype} layer(s), sequence lengths {sorted(lengths)}")
    print(f"{'batch':>5s} {'bp':>3s} {'steps':>6s} {'us/step':>8s} {'us/layer':>9s} {'retire us':>10s}")
    t, total_steps, t_all = 0, 0, time.perf_counter()
    while stack.batch > 0:
        end = min(lengths[s] for s in stack.seq_ids)  # the next step at which a sequence finishes
        steps = end - t
        stack.forward_into(x0, outs)  # first step of the stretch, untraced
        dt = float("nan")
        if steps > 1:
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            stack.forward_into(x0, outs)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            t0 = time.perf_counter()
            for _ in range(steps - 1):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / (steps - 1)
            ttnn.release_trace(dev, tid)
        ttnn.synchronize_device(dev)
        b, bp = stack.batch, stack.bp
        t0 = time.perf_counter()
        stack.retire([r for r, s in enumerate(stack.seq_ids) if lengths[s] == end])
        ttnn.synchronize_device(dev)
        rt = (time.perf_counter() - t0) * 1e6
        print(f"{b:5d} {bp:3d} {steps:6d} {dt * 1e6:8.1f} {dt / n_layers * 1e6:9.1f} {rt:10.0f}")
        t, total_steps = end, total_steps + steps
    print(f"batch 0 after {total_steps} steps, {time.perf_counter() - t_all:.2f} s wall-clock "
          "(including compiles of new padded batches)")
    for tns in outs + [x0]:
        ttnn.deallocate(tns)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dtype", default="bfp8", choices=DTYPES)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lengths = torch.randint(1, args.max_len + 1, (16,), generator=torch.Generator().manual_seed(args.seed)).tolist()
    dev = ttnn.open_device(device_id=0, trace_region_size=4 << 20)
    try:
        run(dev, args.layers, args.dtype, lengths)
    finally:
        ttnn.close_device(dev)
