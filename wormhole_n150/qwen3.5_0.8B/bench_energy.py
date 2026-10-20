# SPDX-License-Identifier: Apache-2.0
"""Energy per decoded token: traced decode for a few seconds per config while sampling board power.

Power comes from the card's hwmon sensor (sysfs `power1_input`, whole watts, board level), found by the
PCI address that tt-smi reports for /dev/tenstorrent/0. Energy per token = average power x time per step
/ tokens per step (= batch: one token per sequence). Only the GDN layers run here, so the numbers are per
token per GDN layer as well as per step.

    python bench_energy.py
"""
import glob
import json
import os
import statistics
import subprocess
import threading
import time

import torch
import ttnn

from gdn_l1 import Qwen35_08B_GDNStackL1
from test_gdn_l1 import make_hf_layers

SECONDS = 5.0
CONFIGS = [("bfp8", ttnn.bfloat8_b, 3, 1), ("bf16", ttnn.bfloat16, 2, 1), ("bfp8", ttnn.bfloat8_b, 3, 4)]


def card_power_file():
    info = json.loads(subprocess.run(["tt-smi", "-s"], capture_output=True, text=True).stdout)
    bus = info["device_info"][0]["board_info"]["bus_id"]
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if open(f"{h}/name").read().strip() == "wormhole" and os.path.realpath(f"{h}/device").endswith(bus):
            return f"{h}/power1_input"
    raise RuntimeError(f"no hwmon sensor for {bus}")


def tt_smi_telemetry():
    t = json.loads(subprocess.run(["tt-smi", "-s"], capture_output=True, text=True).stdout)["device_info"][0]["telemetry"]
    return float(t["power"]), int(t["aiclk"])


class PowerSampler:
    def __init__(self, path, period=0.005):
        self.path, self.period, self.samples, self._stop = path, period, [], threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(int(open(self.path).read()) / 1e6)
            time.sleep(self.period)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def mean(self):
        return statistics.mean(self.samples)


def measure(dev, power_file, name, dtype, n_layers, batch, dram):
    stack = Qwen35_08B_GDNStackL1(
        dev, make_hf_layers(n_layers, seed=7), batch=batch, weight_dtype=dtype, weights_in_dram=dram
    )
    x0 = ttnn.from_torch(
        torch.randn(32, 1024), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    outs = stack.allocate_activations()
    stack.forward_into(x0, outs)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    stack.forward_into(x0, outs)
    ttnn.end_trace_capture(dev, tid, cq_id=0)

    # step time, then a run of ~SECONDS with power sampling (trace replays queued in batches of 100)
    for _ in range(50):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(500):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    step = (time.perf_counter() - t0) / 500
    n_steps = int(SECONDS / step)
    with PowerSampler(power_file) as ps:
        time.sleep(0.3)  # let the reading settle under load
        ps.samples.clear()
        t0 = time.perf_counter()
        for i in range(n_steps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            if i == n_steps // 2:
                smi_power, smi_clk = tt_smi_telemetry()
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
    step = wall / n_steps
    ttnn.release_trace(dev, tid)
    for t in outs + [x0]:
        ttnn.deallocate(t)
    for L in stack.layers:
        for t in (L.w_in, L.w_out, L.lconst, L.rec_state, L.conv_state):
            ttnn.deallocate(t)
    ttnn.deallocate(stack.gconst)
    return dict(step_us=step * 1e6, power_w=ps.mean, smi_power_w=smi_power, aiclk=smi_clk, tokens=batch, layers=n_layers)


if __name__ == "__main__":
    power_file = card_power_file()
    dev = ttnn.open_device(device_id=0, trace_region_size=4 << 20)
    try:
        with PowerSampler(power_file) as ps:
            time.sleep(3)
        idle = ps.mean
        print(f"idle (device open): {idle:.1f} W  [{power_file}]")
        for name, dtype, n_layers, batch in CONFIGS:
            for dram in (False, True):
                r = measure(dev, power_file, name, dtype, n_layers, batch, dram)
                tok_s = r["tokens"] / (r["step_us"] * 1e-6)
                uj_tok = r["power_w"] * r["step_us"] / r["tokens"]  # W * us = uJ
                dyn_uj_tok = (r["power_w"] - idle) * r["step_us"] / r["tokens"]
                print(
                    f"{name} layers={n_layers} batch={batch} weights={'DRAM' if dram else 'L1  '}: "
                    f"step {r['step_us']:.0f} us, {tok_s:,.0f} tok/s, board {r['power_w']:.1f} W "
                    f"(tt-smi {r['smi_power_w']:.0f} W, aiclk {r['aiclk']} MHz), "
                    f"{uj_tok:.0f} uJ/token = {uj_tok / n_layers:.0f} uJ/token/layer, "
                    f"{1e6 / uj_tok:,.0f} tokens/J; above idle {dyn_uj_tok:.0f} uJ/token",
                    flush=True,
                )
    finally:
        ttnn.close_device(dev)
