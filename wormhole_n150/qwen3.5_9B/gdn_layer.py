# SPDX-License-Identifier: Apache-2.0
"""Full Qwen3.5 / Qwen3.6 Gated DeltaNet layer on a Wormhole n150, head-parallel across the 8x8 grid.

    x -> in_proj (q,k,v,z | a,b) -> causal conv1d + SiLU -> l2norm(q,k) -> gated delta rule
      -> gated RMSNorm(o, z) -> out_proj

Core c (row-major over the 8x8 grid) owns v-head h = c // 2 and column half s = c % 2.
Splitting the heads splits the weights with them:

  in_proj   column-parallel. Core c's output columns are exactly what it needs: a quarter of its
            k-head's q and k (the 4 cores sharing a k-head take one quarter each, so no weight is
            read twice), its half of v and its half of z. The matmul writes them straight into the
            core's L1 (width-sharded output), where the fused kernel picks them up.
  middle    one fused kernel per core, all on local L1: conv (with its conv state), the q/k quarter
            exchange inside the 4-core group, l2norm, gating, the recurrence (state in L1), and the
            gated RMSNorm (partial sum of squares exchanged with the partner core).
  out_proj  the matmul multicasts every core's slice of y (an all-gather, so y is concatenated), then
            core c computes output columns 64c..64c+63 from the full y and W_o[:, those columns]. The
            sum over heads happens inside each core's matmul; no partial sums are reduced.

a/b (one scalar per v-head per token) come from a separate small projection.
"""

import math
import struct
from pathlib import Path

import torch
import ttnn

TILE = 32
TILE_BYTES_F32 = 4096
KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

N_CORES, GROUP, NSPLIT = 64, 4, 2
KT, NCT, P_TILES, CONV_TILES, N_CONST = 4, 2, 6, 4, 20
K_NW, K_SEL, K_ONES = 16, 18, 19

(C_QD, C_NKD, C_KB, C_V, C_D, C_W, C_ID, C_U, C_STATE, C_SYNC, C_P, C_CONST, C_XS, C_AB, C_QKLOC, C_QKFULL,
 C_OUT, C_CV, C_TMP, C_SCAL, C_SCAL2, C_PREP, C_OBLK, C_SS, C_SSP, C_RSTD, C_TMP2, C_TMP3, C_Y, C_CONV,
 C_SSK) = range(31)


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def core_heads(c):
    """core -> (v-head h, column half s, k-head kh, position g in the k-head group)"""
    h, s = divmod(c, NSPLIT)
    return h, s, h // 2, (h % 2) * NSPLIT + s


class Qwen35GatedDeltaNetTT:
    def __init__(self, device, hf_layer, batch=1, max_rows=128, weight_dtype=ttnn.bfloat16):
        L = hf_layer
        self.device, self.batch = device, batch
        self.hidden = L.hidden_size
        self.nk, self.nv, self.dk, self.dv = L.num_k_heads, L.num_v_heads, L.head_k_dim, L.head_v_dim
        grid = device.compute_with_storage_grid_size()
        assert (grid.x, grid.y) == (8, 8), "needs the full 8x8 n150 grid"
        assert (self.nk, self.nv, self.dk, self.dv) == (16, 32, 128, 128), "Qwen3.5/3.6 GDN geometry"
        assert L.conv_kernel_size == 4
        assert max_rows % TILE == 0 and batch <= max_rows
        self.max_rows = max_rows
        self.max_blocks = max_rows // TILE
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
        self.core_list = ttnn.corerange_to_cores(self.cores, row_wise=True)
        self.noc_xy = [device.worker_core_from_logical_core(c) for c in self.core_list]
        self._kernel_cache = {}
        self._semaphores = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.cores, 0) for i in range(2)]
        self.debug_defines = []  # e.g. [("DEBUG_Y_IS_O", "1")] to return the pre-norm o instead of y
        self.debug_return_y = False
        self.kernels = {k: (KERNEL_DIR / f"layer_{k}.cpp").read_text() for k in ("reader", "compute", "writer")}

        with torch.no_grad():
            self._prepare_weights(L, weight_dtype)
        self.reset_state()

    # ---- weights ------------------------------------------------------------------------------
    def _core_channels(self, c):
        """Channels of mixed_qkv (q | k | v) and of z that core c projects."""
        h, s, kh, g = core_heads(c)
        kd, q4 = self.nk * self.dk, self.dk // GROUP
        half = self.dv // NSPLIT
        q = torch.arange(kh * self.dk + g * q4, kh * self.dk + (g + 1) * q4)
        v = torch.arange(h * self.dv + s * half, h * self.dv + (s + 1) * half)
        return q, kd + q, 2 * kd + v, v

    def _prepare_weights(self, L, weight_dtype):
        dev = self.device
        w_qkv, w_z = L.in_proj_qkv.weight.float(), L.in_proj_z.weight.float()
        conv_w = L.conv1d.weight.float()[:, 0, :]  # [conv_dim, 4]
        rows, consts = [], []
        for c in range(N_CORES):
            h, s, _, _ = core_heads(c)
            q, k, v, z = self._core_channels(c)
            rows.append(torch.cat([w_qkv[q], w_qkv[k], w_qkv[v], w_z[z]]))
            ch = torch.cat([q, k, v])  # this core's 128 conv channels, same order as its P tiles 0..3
            tiles = []
            for shift in range(4):  # out[t] = sum_shift w[:, 3 - shift] * x[t - shift]
                for ct in range(CONV_TILES):
                    tiles.append(torch.diag(conv_w[ch[ct * TILE:(ct + 1) * TILE], 3 - shift]))
            nw = L.norm.weight.float()[s * NCT * TILE:(s + 1) * NCT * TILE]
            for j in range(NCT):
                t = torch.zeros(TILE, TILE)
                t[0] = nw[j * TILE:(j + 1) * TILE]
                tiles.append(t)
            sel = torch.zeros(TILE, TILE)
            sel[h, 0] = 1.0
            ones = torch.zeros(TILE, TILE)
            ones[:, 0] = 1.0
            tiles += [sel, ones]
            consts.append(torch.cat(tiles))
        self.w_in = ttnn.from_torch(torch.cat(rows).T.contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        w_ab = torch.cat([L.in_proj_a.weight.float(), L.in_proj_b.weight.float()])
        self.w_ab = ttnn.from_torch(w_ab.T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        self.w_out = ttnn.from_torch(
            L.out_proj.weight.float().T.contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=dev
        )
        self.consts = ttnn.from_torch(
            torch.cat(consts), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=self._sharded((N_CONST * TILE, TILE)),
        )
        self.neg_A = (-L.A_log.float().exp()).tolist()
        self.dt_bias = L.dt_bias.float().tolist()
        self.norm_eps = L.norm.variance_epsilon

    # ---- L1-resident state ----------------------------------------------------------------------
    def _sharded(self, shard_shape):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    def reset_state(self):
        B = self.batch
        for name in ("rec_state", "conv_state"):
            if getattr(self, name, None) is not None:
                ttnn.deallocate(getattr(self, name))
        self.rec_state = ttnn.from_torch(
            torch.zeros(N_CORES * B * self.dk, NCT * TILE),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device,
            memory_config=self._sharded((B * self.dk, NCT * TILE)),
        )
        self.conv_state = ttnn.from_torch(
            torch.zeros(N_CORES * B * TILE, CONV_TILES * TILE),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device,
            memory_config=self._sharded((B * TILE, CONV_TILES * TILE)),
        )

    def get_recurrent_state(self):
        """[B, Hv, dk, dv] copy of the L1 state (for checks)."""
        B = self.batch
        x = ttnn.to_torch(self.rec_state).reshape(self.nv, NSPLIT, B, self.dk, NCT * TILE)
        return x.permute(2, 0, 3, 1, 4).reshape(B, self.nv, self.dk, self.dv)

    # ---- program --------------------------------------------------------------------------------
    def _program(self, ab, p, y, M, T):
        # Kernel descriptors (64 cores x 3 kernels of runtime args) are the slow part to build and
        # only depend on the call shape and buffer addresses, so they are cached. The CB list holds
        # references to this call's P / y tensors and is rebuilt every time.
        key = (M, T, ab.buffer_address(), p.buffer_address(), y.buffer_address(), tuple(self.debug_defines))
        kernels = self._kernel_cache.get(key)
        if kernels is None:
            kernels = self._kernel_cache[key] = self._kernels(ab, M, T)
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=self._semaphores, cbs=self._cbs(p, y))

    def _cbs(self, p, y):
        cores = self.cores

        def cb(index, tiles, dtype=ttnn.float32, page=TILE_BYTES_F32):
            return ttnn.CBDescriptor(
                total_size=tiles * page, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page)],
            )

        mb = self.max_blocks
        cbs = [
            cb(C_QD, 2 * KT), cb(C_NKD, 2 * KT), cb(C_KB, 2 * KT), cb(C_V, 2 * NCT), cb(C_D, 2), cb(C_W, 2),
            cb(C_ID, 1), cb(C_U, NCT), cb(C_SYNC, 2, page=32),
            cb(C_XS, 3 * CONV_TILES), cb(C_AB, 2), cb(C_QKLOC, 2), cb(C_QKFULL, mb * 2 * KT),
            cb(C_OUT, 2 * NCT), cb(C_CV, 2 * NCT), cb(C_TMP, 3 * KT), cb(C_SCAL, 5), cb(C_SCAL2, 5),
            cb(C_PREP, 3 * KT + 2), cb(C_OBLK, 2 * NCT), cb(C_SS, 2), cb(C_SSP, mb), cb(C_RSTD, 2),
            cb(C_TMP2, NCT), cb(C_TMP3, NCT), cb(C_SSK, 2),
            ttnn.cb_descriptor_from_sharded_tensor(C_STATE, self.rec_state),
            ttnn.cb_descriptor_from_sharded_tensor(C_P, p),
            ttnn.cb_descriptor_from_sharded_tensor(C_Y, y),
            ttnn.cb_descriptor_from_sharded_tensor(C_CONV, self.conv_state),
            ttnn.cb_descriptor_from_sharded_tensor(C_CONST, self.consts),
        ]
        return cbs

    def _kernels(self, ab, M, T):
        cores, B = self.cores, self.batch
        reader_ct = [B]
        reader_ct += ttnn.TensorAccessorArgs(ab).get_compile_time_args()
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        scale2 = float(self.dk)  # (dk ** -0.5) ** -2
        for c, core in enumerate(self.core_list):
            h, s, kh, g = core_heads(c)
            reader_rt[core.x][core.y] = [ab.buffer_address(), M, T]
            partner = self.noc_xy[c ^ 1]
            group = [self.noc_xy[kh * GROUP + i] for i in range(GROUP)]
            writer_rt[core.x][core.y] = [M, g, partner.x, partner.y] + [v for xy in group for v in (xy.x, xy.y)]
            compute_rt[core.x][core.y] = [
                M, T, f32_bits(self.dt_bias[h]), f32_bits(self.neg_A[h]),
                f32_bits(scale2), f32_bits(1e-6 * scale2), f32_bits(1e-6),
                f32_bits(1.0 / self.dv), f32_bits(self.norm_eps),
            ]

        def kernel(name, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=self.kernels[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=config,
                defines=list(self.debug_defines),
            )

        compute_cfg = ttnn.ComputeConfigDescriptor()
        compute_cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        compute_cfg.fp32_dest_acc_en = True
        compute_cfg.dst_full_sync_en = True
        return [
            kernel("reader", reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
            kernel("writer", [0, 1], writer_rt, ttnn.WriterConfigDescriptor()),
            kernel("compute", [B], compute_rt, compute_cfg),
        ]

    # ---- forward --------------------------------------------------------------------------------
    def _matmul_cfg(self, mp, per_core_n, in0_block_w):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8), in0_block_w=in0_block_w, out_subblock_h=1,
            out_subblock_w=min(per_core_n, 2), per_core_M=mp // TILE, per_core_N=per_core_n,
            fuse_batch=True, fused_activation=None, mcast_in0=True,
        )

    def _width_sharded(self, mp, width, dtype=None):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, (mp, width), ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _forward_rows(self, x_rows, T):
        """x_rows: [M, hidden] torch, sequence-major rows (M = batch * T)."""
        M = x_rows.shape[0]
        mp = math.ceil(M / TILE) * TILE
        x = ttnn.from_torch(
            torch.nn.functional.pad(x_rows.float(), (0, 0, 0, mp - M)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        mm_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        p = ttnn.linear(
            x, self.w_in, program_config=self._matmul_cfg(mp, P_TILES, 4),
            memory_config=self._width_sharded(mp, P_TILES * TILE), dtype=ttnn.float32, compute_kernel_config=mm_cfg,
        )
        ab = ttnn.linear(x, self.w_ab, dtype=ttnn.float32, compute_kernel_config=mm_cfg)
        ttnn.deallocate(x)
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([mp, self.nv * self.dv]), ttnn.float32, ttnn.TILE_LAYOUT, self.device,
            self._width_sharded(mp, NCT * TILE),
        )
        ttnn.generic_op([ab, self.consts, p, self.rec_state, self.conv_state, y], self._program(ab, p, y, M, T))
        ttnn.deallocate(p)
        ttnn.deallocate(ab)
        if self.debug_return_y:
            result = ttnn.to_torch(y)[:M]
            ttnn.deallocate(y)
            return result
        out = ttnn.linear(
            y, self.w_out, program_config=self._matmul_cfg(mp, self.hidden // TILE // N_CORES, NCT),
            memory_config=self._width_sharded(mp, self.hidden // N_CORES), dtype=ttnn.float32,
            compute_kernel_config=mm_cfg,
        )
        ttnn.deallocate(y)
        result = ttnn.to_torch(out)[:M]
        ttnn.deallocate(out)
        return result

    def __call__(self, hidden_states):
        """hidden_states: [batch, T, hidden] (torch). Continues from the L1 state. Returns [batch, T, hidden]."""
        B, T, H = hidden_states.shape
        assert B == self.batch and H == self.hidden
        chunk = self.max_rows // B
        outs = []
        for t0 in range(0, T, chunk):
            xc = hidden_states[:, t0:t0 + chunk]
            tc = xc.shape[1]
            outs.append(self._forward_rows(xc.reshape(B * tc, H), tc).reshape(B, tc, -1))
        return torch.cat(outs, dim=1)
