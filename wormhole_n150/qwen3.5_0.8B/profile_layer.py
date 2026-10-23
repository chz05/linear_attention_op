# SPDX-License-Identifier: Apache-2.0
"""Device time of one GDN layer launch, per phase, from the profiler zones in kernels/.

    python profile_layer.py                          # bfp8, batch 1 4 8 16, 100 launches each
    python profile_layer.py -b 1 -n 200 -d bf16      # one config
    python profile_layer.py -b 8 --risc TRISC_2      # the pack thread's view of the compute zones

Sets TT_METAL_DEVICE_PROFILER=1 itself (zones compile to nothing otherwise). Times are per launch, relative to
the first kernel start on any of the 64 cores, then the median over launches:
  layer   first kernel start .. last kernel end over all cores
  zone    dur = mean over cores of the time spent in it (summed if it repeats), max = the slowest core,
          start / end = when the first core enters it / the last core leaves it
"""
import argparse
import collections
import csv
import glob
import os
import statistics
import sys

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

import torch  # noqa: E402
import ttnn  # noqa: E402

from gdn_l1 import Qwen35_08B_GDNStackL1  # noqa: E402
from test_gdn_l1 import make_hf_layers  # noqa: E402

DTYPES = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "bfp4": ttnn.bfloat4_b}
LOG = os.path.join(os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal")),
                   "generated", "profiler", ".logs", "profile_log_device.csv")
READ_EVERY = 20  # launches between profiler reads (the device buffers hold a few hundred markers per RISC)


def run(dev, batch, dtype, launches, warmup=5):
    stack = Qwen35_08B_GDNStackL1(dev, make_hf_layers(1, seed=7), batch=batch, weight_dtype=DTYPES[dtype])
    x0 = ttnn.from_torch(torch.randn(16, 1024), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    for i in range(warmup + launches):
        ttnn.deallocate(stack.forward_device(x0, keep_input=True))
        if i % READ_EVERY == READ_EVERY - 1:
            ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    ttnn.deallocate(x0)


def parse(path, skip=0):
    """-> list of launches; each {(risc, zone): {core: [(start, end), ...]}} in ns from the launch start.
    The log has physical NoC coordinates; the kernel runs on the whole 8x8 grid, so a core's logical
    (x, y) is the rank of its physical x and y (c = 8y + x, as in gdn_l1)."""
    by_run = collections.defaultdict(list)
    with open(path) as f:
        next(f)
        for r in csv.DictReader(f, skipinitialspace=True):
            by_run[int(r["run host ID"])].append(r)
    rows = [r for v in by_run.values() for r in v]
    xs = sorted({int(r["core_x"]) for r in rows})
    ys = sorted({int(r["core_y"]) for r in rows})
    assert len(xs) == len(ys) == 8, (xs, ys)
    core_of = {(x, y): 8 * ys.index(y) + xs.index(x) for x in xs for y in ys}
    launches = []
    for run_id in sorted(by_run)[skip:]:
        open_zones, zones = {}, collections.defaultdict(lambda: collections.defaultdict(list))
        for r in sorted(by_run[run_id], key=lambda r: int(r["time[cycles since reset]"])):
            core = core_of.get((int(r["core_x"]), int(r["core_y"])))
            if core is None:
                continue
            key = (r["RISC processor type"], r["zone name"], core)
            t = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                open_zones.setdefault(key, []).append(t)
            elif r["type"] == "ZONE_END" and open_zones.get(key):
                zones[key[:2]][core].append((open_zones[key].pop(), t))
        kernel = [se for (risc, z), cores in zones.items() if z.endswith("-KERNEL") for v in cores.values() for se in v]
        t0 = min(s for s, _ in kernel)
        launches.append({k: {c: [(s - t0, e - t0) for s, e in v] for c, v in cores.items()} for k, cores in zones.items()})
    return launches


def report(launches, risc_filter):
    med = statistics.median
    layer = [max(e for (r, z), cores in L.items() if z.endswith("-KERNEL") for v in cores.values() for _, e in v)
             for L in launches]
    print(f"  layer {med(layer) / 1e3:6.1f} us  (min {min(layer) / 1e3:.1f}, max {max(layer) / 1e3:.1f}, "
          f"{len(launches)} launches)")
    keys = sorted({k for L in launches for k in L if not k[1].endswith("-FW")},
                  key=lambda k: (k[0], med(min(s for v in L[k].values() for s, _ in v) for L in launches if k in L)))
    print(f"  {'risc':8s} {'zone':22s} {'dur':>6s} {'max':>6s} {'start':>6s} {'end':>6s}  cores")
    for risc, zone in keys:
        if risc_filter and risc not in risc_filter and not zone.endswith("-KERNEL"):
            continue
        Ls = [L[(risc, zone)] for L in launches if (risc, zone) in L]
        dur = [statistics.mean(sum(e - s for s, e in v) for v in cores.values()) for cores in Ls]
        dmax = [max(sum(e - s for s, e in v) for v in cores.values()) for cores in Ls]
        start = [min(s for v in cores.values() for s, _ in v) for cores in Ls]
        end = [max(e for v in cores.values() for _, e in v) for cores in Ls]
        n = len(Ls[0])
        print(f"  {risc:8s} {zone:22s} {med(dur) / 1e3:6.2f} {med(dmax) / 1e3:6.2f} {med(start) / 1e3:6.2f} "
              f"{med(end) / 1e3:6.2f}  {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("-d", "--dtype", default="bfp8", choices=DTYPES)
    ap.add_argument("-n", "--launches", type=int, default=100)
    ap.add_argument("--risc", nargs="*", default=["BRISC", "NCRISC", "TRISC_1"],
                    help="RISCs whose zones are listed (BRISC = writer, NCRISC = reader, TRISC_0/1/2 = "
                         "unpack/math/pack)")
    args = ap.parse_args()
    for b in args.batch:
        # stale zone-location logs from earlier kernel versions can collide with the new zones' hashes
        for f in glob.glob(os.path.join(os.path.dirname(LOG), "*")):
            os.remove(f)
        dev = ttnn.open_device(device_id=0)  # one session per config: the CSV is written when the device closes
        try:
            run(dev, b, args.dtype, args.launches)
        finally:
            ttnn.close_device(dev)
        launches = parse(LOG)[-args.launches:]
        print(f"batch {b}, {args.dtype}:")
        report(launches, args.risc)
        sys.stdout.flush()
