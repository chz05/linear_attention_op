# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet (Qwen3.5 / Qwen3.6 linear-attention layers) recurrent op for a Wormhole n150.

The recurrent state S[h] (dk x dv, fp32) of every head is kept in L1 for the lifetime of the
op object: it is an L1 height-sharded tensor and the kernel updates it in place, so decode
never moves the state to or from DRAM.

Work split: a work unit is one head, or a block of V columns of one head. Heads are spread
across cores first; if there are fewer heads than cores, each head's V columns are split
across several cores. Columns of S are independent in the delta rule (every column only
needs its own column of S plus the k/q vectors), so a V-column split needs no reduction:
each core writes its own output columns and the final result is just their concatenation.

Recurrence (per head, per token), matching fla / HF `torch_recurrent_gated_delta_rule`:
    S = exp(g) * S
    u = v - k^T S
    S = S + (beta * k) u^T
    o = q^T S
with q, k L2-normalised and q scaled by dk ** -0.5.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn

TILE = 32
TILE_BYTES = TILE * TILE * 4  # fp32
KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

C_QD, C_NKD, C_KB, C_V, C_D, C_W, C_ID, C_U, C_STATE, C_SYNC = range(10)
C_OUT, C_STAGE, C_OSTAGE = 16, 20, 21

# Keep well clear of the ~1.4 MB of usable L1 per Tensix core.
L1_BUDGET_BYTES = 1_200_000


@dataclass(frozen=True)
class SplitPlan:
    num_heads: int
    nsplit: int  # V-column splits per head
    units_per_core: int
    num_cores: int
    kt: int  # dk / 32
    nct: int  # v column tiles per unit

    @property
    def num_units(self):
        return self.num_heads * self.nsplit

    @property
    def state_bytes_per_core(self):
        return self.units_per_core * self.kt * self.nct * TILE_BYTES

    @property
    def l1_bytes_per_core(self):
        kt, nct, u = self.kt, self.nct, self.units_per_core
        tiles = 2 * (3 * kt + nct + 2) + 1 + nct + 2 * nct  # per-token CBs, identity, u, out
        tiles += u * (3 * kt + nct + 1) + u * nct  # reader / writer staging
        return tiles * TILE_BYTES + self.state_bytes_per_core


def plan_split(num_heads, head_k_dim, head_v_dim, max_cores):
    """Use as many cores as possible, splitting heads first and V columns only if needed."""
    assert head_k_dim % TILE == 0 and head_v_dim % TILE == 0
    kt, vt = head_k_dim // TILE, head_v_dim // TILE
    best = None
    for nsplit in [s for s in range(1, vt + 1) if vt % s == 0]:
        units = num_heads * nsplit
        u = math.ceil(units / max_cores)
        while units % u:
            u += 1
        plan = SplitPlan(num_heads, nsplit, u, units // u, kt, vt // nsplit)
        if best is None or plan.num_cores > best.num_cores:
            best = plan
    if best.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"{num_heads} heads need {best.l1_bytes_per_core} B of L1 per core "
            f"(budget {L1_BUDGET_BYTES}); the state does not fit in L1"
        )
    return best


def _pad_t(x, tp):
    return torch.nn.functional.pad(x, (0, 0, 0, tp - x.shape[1]))


def pack_inputs(q, k, v, g, beta, use_qk_l2norm=True, scale=None):
    """Turn fla-style inputs into the per-head, token-packed fp32 tensors the kernel reads.

    q, k: [B, T, Hk, dk]   v: [B, T, Hv, dv]   g (log decay), beta: [B, T, Hv]
    Returns (qd, nkd, kb, v, sc), each [B*Hv, Tp, *] with Tp = T rounded up to 32:
        qd = exp(g) * q, nkd = -exp(g) * k, kb = beta * k,
        sc[..., 0] = exp(g) - 1, sc[..., 1] = beta * (q . k)
    """
    B, T, Hk, dk = q.shape
    Hv = v.shape[2]
    assert Hv % Hk == 0
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if use_qk_l2norm:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * (dk**-0.5 if scale is None else scale)
    if Hv != Hk:
        q = q.repeat_interleave(Hv // Hk, dim=2)
        k = k.repeat_interleave(Hv // Hk, dim=2)

    def heads_first(x):  # [B, T, H, D] -> [B*H, T, D]
        return x.permute(0, 2, 1, 3).reshape(B * Hv, T, x.shape[-1])

    q, k, v = heads_first(q), heads_first(k), heads_first(v)
    d_minus_1 = heads_first(torch.expm1(g)[..., None])
    d = d_minus_1 + 1
    beta = heads_first(beta[..., None])

    sc = torch.zeros(B * Hv, T, TILE)
    sc[..., 0:1] = d_minus_1
    sc[..., 1:2] = beta * (q * k).sum(-1, keepdim=True)
    tp = math.ceil(T / TILE) * TILE
    return tuple(_pad_t(x.contiguous(), tp) for x in (d * q, -d * k, beta * k, v, sc))


def unpack_output(out, batch, seq_len):
    """[B*Hv, Tp, dv] -> [B, T, Hv, dv]"""
    nh, _, dv = out.shape
    return out[:, :seq_len].reshape(batch, nh // batch, seq_len, dv).permute(0, 2, 1, 3)


class GatedDeltaRuleL1:
    """Recurrent gated delta rule with the state resident in L1 across calls."""

    def __init__(self, device, num_heads, head_k_dim=128, head_v_dim=128, max_cores=None, initial_state=None):
        self.device = device
        self.dk, self.dv = head_k_dim, head_v_dim
        grid = device.compute_with_storage_grid_size()
        self.plan = plan_split(num_heads, head_k_dim, head_v_dim, max_cores or grid.x * grid.y)
        self.cores = ttnn.num_cores_to_corerangeset(self.plan.num_cores, grid, row_wise=True)
        self.core_list = ttnn.corerange_to_cores(self.cores, row_wise=True)
        self.kernels = {name: (KERNEL_DIR / f"gdn_{name}.cpp").read_text() for name in ("reader", "compute", "writer")}
        self.state = None
        self.reset_state(initial_state)

    # ---- state <-> torch ------------------------------------------------------------------
    def _state_to_shards(self, s):
        p = self.plan
        ncols = p.nct * TILE
        return s.float().reshape(p.num_heads, self.dk, p.nsplit, ncols).permute(0, 2, 1, 3).reshape(-1, ncols)

    def _shards_to_state(self, x):
        p = self.plan
        ncols = p.nct * TILE
        return x.reshape(p.num_heads, p.nsplit, self.dk, ncols).permute(0, 2, 1, 3).reshape(p.num_heads, self.dk, self.dv)

    def reset_state(self, initial_state=None):
        p = self.plan
        if initial_state is None:
            initial_state = torch.zeros(p.num_heads, self.dk, self.dv)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, (p.units_per_core * self.dk, p.nct * TILE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        if self.state is not None:
            ttnn.deallocate(self.state)
        self.state = ttnn.from_torch(
            self._state_to_shards(initial_state),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=mem,
        )

    def get_state(self):
        """Copy the L1 state back to the host as [num_heads, dk, dv] (for checks only)."""
        return self._shards_to_state(ttnn.to_torch(self.state))

    # ---- op ---------------------------------------------------------------------------------
    def _program(self, qd, nkd, kb, v, sc, out, seq_len):
        p = self.plan
        cores = self.cores

        def cb(index, tiles):
            return ttnn.CBDescriptor(
                total_size=tiles * TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.float32, TILE_BYTES)],
            )

        sync = ttnn.CBDescriptor(
            total_size=64, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(C_SYNC, ttnn.float32, 32)]
        )
        cbs = [
            cb(C_QD, 2 * p.kt),
            cb(C_NKD, 2 * p.kt),
            cb(C_KB, 2 * p.kt),
            cb(C_V, 2 * p.nct),
            cb(C_D, 2),
            cb(C_W, 2),
            cb(C_ID, 1),
            cb(C_U, p.nct),
            cb(C_OUT, 2 * p.nct),
            cb(C_STAGE, p.units_per_core * (3 * p.kt + p.nct + 1)),
            cb(C_OSTAGE, p.units_per_core * p.nct),
            sync,
            ttnn.cb_descriptor_from_sharded_tensor(C_STATE, self.state),
        ]

        reader_ct = [p.kt, p.nct, p.nsplit, p.units_per_core]
        for t in (qd, nkd, kb, v, sc):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = [p.nct, p.nsplit, p.units_per_core]
        writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addrs = [t.buffer_address() for t in (qd, nkd, kb, v, sc)]
        for i, core in enumerate(self.core_list):
            unit_start = i * p.units_per_core
            reader_rt[core.x][core.y] = addrs + [unit_start, seq_len]
            writer_rt[core.x][core.y] = [out.buffer_address(), unit_start, seq_len]
            compute_rt[core.x][core.y] = [seq_len]

        def kernel(name, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=self.kernels[name],
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )

        compute_cfg = ttnn.ComputeConfigDescriptor()
        compute_cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        compute_cfg.fp32_dest_acc_en = True
        return ttnn.ProgramDescriptor(
            kernels=[
                kernel("reader", reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
                kernel("writer", writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
                kernel("compute", [p.kt, p.nct, p.units_per_core], compute_rt, compute_cfg),
            ],
            semaphores=[],
            cbs=cbs,
        )

    def forward_packed(self, qd, nkd, kb, v, sc, seq_len):
        """Run on device tensors already in the packed layout (see `pack_inputs`).

        Returns the packed output [num_heads, Tp, dv] (fp32, DRAM); the L1 state is updated in place.
        """
        nh, tp, _ = v.shape
        assert nh == self.plan.num_heads and tp % TILE == 0 and 0 < seq_len <= tp
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([nh, tp, self.dv]), ttnn.float32, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        program = self._program(qd, nkd, kb, v, sc, out, seq_len)
        return ttnn.generic_op([qd, nkd, kb, v, sc, self.state, out], program)

    def __call__(self, q, k, v, g, beta, use_qk_l2norm=True, scale=None):
        """Torch-in / torch-out convenience wrapper. Shapes as in `pack_inputs`; returns [B, T, Hv, dv]."""
        B, T = q.shape[:2]
        packed = pack_inputs(q, k, v, g, beta, use_qk_l2norm, scale)
        dev = [
            ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device) for x in packed
        ]
        out = self.forward_packed(*dev, seq_len=T)
        for x in dev:
            ttnn.deallocate(x)
        result = ttnn.to_torch(out)
        ttnn.deallocate(out)
        return unpack_output(result, B, T)
