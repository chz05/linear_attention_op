# SPDX-License-Identifier: Apache-2.0
"""500 decode steps with the weights resident in L1 vs streamed from DRAM (same kernel otherwise).

    python bench_decode.py                     # wall-clock, all configs (plain dispatch and traced)
    TT_METAL_DEVICE_PROFILER=1 python bench_decode.py bfp8 3 1 dram   # one config, then parse
        ~/tt-metal/generated/profiler/.logs/profile_log_device.csv for per-launch device time
"""
import sys
import time

import torch
import ttnn

from gdn_l1 import Qwen35_08B_GDNStackL1
from test_gdn_l1 import make_hf_layers

STEPS = 500
DTYPES = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
CONFIGS = [("bfp8", 3, 1), ("bf16", 2, 1), ("bfp8", 3, 4)]


def run(dev, name, n_layers, batch, dram, profile=False):
    stack = Qwen35_08B_GDNStackL1(
        dev, make_hf_layers(n_layers, seed=7), batch=batch, weight_dtype=DTYPES[name], weights_in_dram=dram
    )
    x0 = ttnn.from_torch(
        torch.randn(32, 1024), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    for _ in range(10):  # warm-up: compile + program cache
        ttnn.deallocate(stack.forward_device(x0, keep_input=True))
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for step in range(STEPS):
        ttnn.deallocate(stack.forward_device(x0, keep_input=True))
        if profile and step % 50 == 49:  # keep the device profiler buffers from overflowing
            ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / STEPS
    where = "DRAM" if dram else "L1"
    print(f"{name} layers={n_layers} batch={batch} weights={where} dispatch: {dt * 1e6:.0f} us/step, {dt / n_layers * 1e6:.0f} us/layer")
    if profile:
        return
    # Traced: the whole step is recorded once and replayed, so host dispatch drops out.
    outs = stack.allocate_activations()
    stack.forward_into(x0, outs)  # compile + cache outside the capture
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    stack.forward_into(x0, outs)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    for _ in range(10):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / STEPS
    print(f"{name} layers={n_layers} batch={batch} weights={where} traced:   {dt * 1e6:.0f} us/step, {dt / n_layers * 1e6:.0f} us/layer")
    ttnn.release_trace(dev, tid)
    for t in outs + [x0]:
        ttnn.deallocate(t)


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, trace_region_size=4 << 20)
    try:
        if len(sys.argv) == 5:
            run(dev, sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "dram", profile=True)
            ttnn.ReadDeviceProfiler(dev)
        else:
            for name, n_layers, batch in CONFIGS:
                for dram in (False, True):
                    run(dev, name, n_layers, batch, dram)
    finally:
        ttnn.close_device(dev)
