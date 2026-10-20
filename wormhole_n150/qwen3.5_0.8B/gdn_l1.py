# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-0.8B Gated DeltaNet layers on a Wormhole n150 with every weight resident in L1.

Per layer: x -> in_proj (q,k,v,z,a,b) -> conv1d + SiLU -> l2norm -> gated delta rule
             -> RMSNorm(o) * w * silu(z) -> out_proj

Core c (row-major over the 8x8 grid) owns head h = c // 4 and column quarter s = c % 4. Splitting the
heads splits the weights with them, and every slice stays in that core's L1 for the life of the layer:

  in_proj   core c holds W[:, its 160 columns] = [q/4 | k/4 | v/4 | z/4 | a,b of all 16 heads]
            (1024 x 160). The head's 4 cores each project a quarter of q and k and swap quarters.
  out_proj  split by rows: core c holds W_o^T[its 32 channels, :] (32 x 1024) and computes a partial
            [batch, 1024]. The partials are reduce-scattered: core j (j < 32) owns output columns
            32j..32j+31 and sums the 64 cores' rows for them.
  states    recurrent (16 KB per sequence) and conv (4 KB) per core, in L1.

Decode only: one call = one token for each of the `batch` sequences (row b = sequence b). Layers run
back to back, one launch each; activations go through a small L1-interleaved tensor, so nothing
touches DRAM during a decode step.
"""

import math
import struct
from pathlib import Path

import torch
import ttnn

TILE = 32
F32_TILE = 4096
KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

N_CORES, GROUP, N_OUT = 64, 4, 32
KT, KX, P_TILES, CONV_TILES, X_CHUNK = 4, 32, 5, 3, 4
N_LCONST, N_GCONST = 13, 5
MAX_BATCH = 32

(C_QD, C_NKD, C_KB, C_V, C_D, C_W, C_RECV, C_U, C_STATE, C_SYNC, C_X, C_WIN, C_WOUT, C_LCONST, C_GCONST, C_P,
 C_XS, C_QKLOC, C_QKFULL, C_CV, C_TMP, C_SCAL, C_SCAL2, C_PREP, C_OUT, C_OBLK, C_SS, C_SSP, _FREE, C_TMP2,
 C_TMP3, C_CONV) = range(32)

# CB sizes in fp32 tiles. Added separately: C_RECV (2 tiles per sequence) and C_PREP (14 tiles, or one 4 KB input
# row per sequence if larger: it also receives the multicast input rows at the start of a launch).
WORK_CBS = {
    C_QD: KT, C_NKD: KT, C_KB: KT, C_V: 1, C_D: 1, C_W: 1, C_U: 1, C_X: 2 * X_CHUNK, C_P: P_TILES,
    C_XS: 3 * CONV_TILES, C_QKLOC: 2, C_QKFULL: 2 * KT, C_CV: 1, C_TMP: KT, C_SCAL: 5, C_SCAL2: 5,
    C_OUT: 2, C_OBLK: 1, C_SS: 2, C_SSP: GROUP, C_TMP2: 2, C_TMP3: 1,
}


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def core_head(c):
    return divmod(c, GROUP)  # (head, quarter)


def tile_bytes(dtype):
    return {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.float32: 4096}[dtype]


def l1_bytes_per_core(batch, num_layers, weight_dtype):
    """Static L1 estimate per core (working CBs + global constants + per-layer weights and states)."""
    work = (sum(WORK_CBS.values()) + 2 * batch + max(3 * KT + 2, batch)) * F32_TILE + 64
    per_layer = (KX * P_TILES + N_OUT) * tile_bytes(weight_dtype) + N_LCONST * 2048 + batch * (KT + 1) * F32_TILE
    return work + N_GCONST * F32_TILE + num_layers * per_layer


class _Layer:
    """One GDN layer's weights, constants and states (weights in L1, or DRAM for the comparison mode)."""

    def __init__(self, stack, L, weight_dtype):
        dev, B = stack.device, stack.batch
        nh, dk = stack.nh, stack.dk
        kd = nh * dk
        w_qkv, w_z = L.in_proj_qkv.weight.float(), L.in_proj_z.weight.float()
        w_ab = torch.cat([L.in_proj_a.weight.float(), L.in_proj_b.weight.float()])  # [32, 1024]
        conv_w = L.conv1d.weight.float()[:, 0, :]
        w_o = L.out_proj.weight.float()  # [1024, 2048]
        win, wout, lconst = [], [], []
        for c in range(N_CORES):
            h, s = core_head(c)
            ch = torch.arange(h * dk + s * TILE, h * dk + (s + 1) * TILE)
            q, k, v = ch, kd + ch, 2 * kd + ch
            win.append(torch.cat([w_qkv[q], w_qkv[k], w_qkv[v], w_z[ch], w_ab]))  # [160, 1024]
            wout.append(w_o[:, ch].T)  # [32, 1024]
            conv_ch = torch.cat([q, k, v])
            tiles = [torch.diag(conv_w[conv_ch[ct * TILE:(ct + 1) * TILE], 3 - sh]) for sh in range(4) for ct in range(3)]
            tiles.append(torch.diag(L.norm.weight.float()[s * TILE:(s + 1) * TILE]))
            lconst.append(torch.cat(tiles))
        dram = stack.weights_in_dram
        self.w_in = ttnn.from_torch(
            torch.cat(win).T.contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if dram else stack.width_sharded((KX * TILE, P_TILES * TILE)),
        )
        self.w_out = ttnn.from_torch(
            torch.cat(wout), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if dram else stack.height_sharded((TILE, N_OUT * TILE)),
        )
        self.lconst = ttnn.from_torch(
            torch.cat(lconst), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if dram else stack.height_sharded((N_LCONST * TILE, TILE)),
        )
        self.neg_A = (-L.A_log.float().exp()).tolist()
        self.dt_bias = L.dt_bias.float().tolist()
        self.norm_eps = L.norm.variance_epsilon
        self.rec_state = self.conv_state = None
        self.reset_state(stack)

    def reset_state(self, stack):
        for t in (self.rec_state, self.conv_state):
            if t is not None:
                ttnn.deallocate(t)
        B, dev = stack.batch, stack.device
        self.rec_state = ttnn.from_torch(
            torch.zeros(N_CORES * B * stack.dk, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=stack.height_sharded((B * stack.dk, TILE)),
        )
        self.conv_state = ttnn.from_torch(
            torch.zeros(N_CORES * B * TILE, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=stack.height_sharded((B * TILE, TILE)),
        )


class Qwen35_08B_GDNStackL1:
    """A stack of Qwen3.5-0.8B GDN layers, run back to back, with all weights and states in L1.

    weights_in_dram=True is the comparison mode: the same kernel and per-core weight slices, but the
    weights live in DRAM and are streamed into L1 every step (states stay in L1)."""

    def __init__(self, device, hf_layers, batch=1, weight_dtype=ttnn.bfloat8_b, weights_in_dram=False,
                 math_fidelity=ttnn.MathFidelity.HiFi4, mm_block=True):
        self.device, self.batch = device, batch
        self.weight_dtype, self.weights_in_dram = weight_dtype, weights_in_dram
        self.math_fidelity = math_fidelity
        self.defines = ([("WEIGHTS_IN_DRAM", "1")] if weights_in_dram else []) + ([("MM_BLOCK", "1")] if mm_block else [])
        L0 = hf_layers[0]
        self.hidden, self.nh, self.dk = L0.hidden_size, L0.num_v_heads, L0.head_k_dim
        assert (self.hidden, L0.num_k_heads, self.nh, self.dk, L0.head_v_dim) == (1024, 16, 16, 128, 128), \
            "Qwen3.5-0.8B GDN geometry"
        assert batch <= MAX_BATCH
        grid = device.compute_with_storage_grid_size()
        assert (grid.x, grid.y) == (8, 8)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
        self.core_list = ttnn.corerange_to_cores(self.cores, row_wise=True)
        self.noc_xy = [device.worker_core_from_logical_core(c) for c in self.core_list]
        self.kernels = {k: (KERNEL_DIR / f"{k}.cpp").read_text() for k in ("reader", "compute", "writer")}
        self._kernel_cache = {}
        self._semaphores = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.cores, 0) for i in range(4)]

        g = []
        for c in range(N_CORES):
            h, _ = core_head(c)
            ident = torch.eye(TILE)
            ones_col, ones_row = torch.zeros(TILE, TILE), torch.zeros(TILE, TILE)
            ones_col[:, 0] = 1.0
            ones_row[0, :] = 1.0
            sel_a, sel_b = torch.zeros(TILE, TILE), torch.zeros(TILE, TILE)
            sel_a[h, 0] = 1.0
            sel_b[self.nh + h, 0] = 1.0
            g.append(torch.cat([ident, ones_col, ones_row, sel_a, sel_b]))
        self.gconst = ttnn.from_torch(
            torch.cat(g), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=self.height_sharded((N_GCONST * TILE, TILE)),
        )
        with torch.no_grad():
            self.layers = [_Layer(self, L, weight_dtype) for L in hf_layers]

    # ---- memory configs -------------------------------------------------------------------------
    def _sharded(self, layout, shape):
        return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(self.cores, shape, ttnn.ShardOrientation.ROW_MAJOR))

    def height_sharded(self, shape):
        return self._sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, shape)

    def width_sharded(self, shape):
        return self._sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, shape)

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state(self)

    def get_recurrent_state(self, i):
        """[B, 16, 128, 128] copy of layer i's L1 state (for checks)."""
        B = self.batch
        x = ttnn.to_torch(self.layers[i].rec_state).reshape(self.nh, GROUP, B, self.dk, TILE)
        return x.permute(2, 0, 3, 1, 4).reshape(B, self.nh, self.dk, self.dk)

    # ---- program --------------------------------------------------------------------------------
    def _cbs(self, layer):
        def cb(index, tiles, page=F32_TILE, dtype=ttnn.float32):
            return ttnn.CBDescriptor(
                total_size=tiles * page, core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page)],
            )

        cbs = [cb(i, n) for i, n in WORK_CBS.items()]
        cbs += [cb(C_RECV, 2 * self.batch), cb(C_PREP, max(3 * KT + 2, self.batch)), cb(C_SYNC, 2, page=32)]
        cbs += [
            ttnn.cb_descriptor_from_sharded_tensor(C_STATE, layer.rec_state),
            ttnn.cb_descriptor_from_sharded_tensor(C_CONV, layer.conv_state),
            ttnn.cb_descriptor_from_sharded_tensor(C_GCONST, self.gconst),
        ]
        if self.weights_in_dram:  # streaming buffers instead of resident tensors
            wt = tile_bytes(self.weight_dtype)
            cbs += [
                cb(C_WIN, 2 * X_CHUNK * P_TILES, page=wt, dtype=self.weight_dtype),
                cb(C_WOUT, N_OUT, page=wt, dtype=self.weight_dtype),
                cb(C_LCONST, N_LCONST, page=2048, dtype=ttnn.bfloat16),
            ]
        else:
            cbs += [
                ttnn.cb_descriptor_from_sharded_tensor(C_WIN, layer.w_in),
                ttnn.cb_descriptor_from_sharded_tensor(C_WOUT, layer.w_out),
                ttnn.cb_descriptor_from_sharded_tensor(C_LCONST, layer.lconst),
            ]
        return cbs

    def _kernels(self, li, x, out):
        layer, B = self.layers[li], self.batch
        reader_ct = [B, 3]
        if self.weights_in_dram:
            reader_ct += [tile_bytes(self.weight_dtype)]
            for t in (layer.w_in, layer.w_out, layer.lconst):
                reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [0, 1, 2, 3] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        writer_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        mcast = [self.noc_xy[0].x, self.noc_xy[0].y, self.noc_xy[-1].x, self.noc_xy[-1].y]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        owners = [v for xy in self.noc_xy[:N_OUT] for v in (xy.x, xy.y)]
        scale2 = float(self.dk)
        for c, core in enumerate(self.core_list):
            h, _ = core_head(c)
            group = [v for xy in self.noc_xy[h * GROUP:(h + 1) * GROUP] for v in (xy.x, xy.y)]
            reader_rt[core.x][core.y] = [0]
            if self.weights_in_dram:
                reader_rt[core.x][core.y] = [0, layer.w_in.buffer_address(),
                                             layer.w_out.buffer_address(), layer.lconst.buffer_address(), c]
            writer_rt[core.x][core.y] = [out.buffer_address(), B, c] + group + owners + [x.buffer_address()] + mcast
            compute_rt[core.x][core.y] = [
                f32_bits(layer.dt_bias[h]), f32_bits(layer.neg_A[h]),
                f32_bits(scale2), f32_bits(1e-6 * scale2), f32_bits(1e-6),
                f32_bits(1.0 / self.dk), f32_bits(layer.norm_eps), int(c < N_OUT),
            ]

        def kernel(name, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=self.kernels[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores, compile_time_args=ct, runtime_args=rt, config=config,
                defines=self.defines,
            )

        compute_cfg = ttnn.ComputeConfigDescriptor()
        compute_cfg.math_fidelity = self.math_fidelity
        compute_cfg.fp32_dest_acc_en = True
        compute_cfg.dst_full_sync_en = True
        return [
            kernel("reader", reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
            kernel("writer", writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
            kernel("compute", [B], compute_rt, compute_cfg),
        ]

    def _program(self, li, x, out):
        key = (li, x.buffer_address(), out.buffer_address())
        kernels = self._kernel_cache.get(key)
        if kernels is None:
            kernels = self._kernel_cache[key] = self._kernels(li, x, out)
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=self._semaphores, cbs=self._cbs(self.layers[li]))

    # ---- forward (decode) ------------------------------------------------------------------------
    def forward_device(self, x, layers=None, keep_input=False):
        """x: [32, 1024] fp32 row-major L1 tensor, row b = this step's token of sequence b. Runs the layers in order."""
        for n, li in enumerate(range(len(self.layers)) if layers is None else layers):
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([MAX_BATCH, self.hidden]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self.device,
                ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.generic_op([x, self.layers[li].w_in, out], self._program(li, x, out))
            if n > 0 or not keep_input:
                ttnn.deallocate(x)
            x = out
        return x

    def allocate_activations(self):
        """One persistent row-major L1 output buffer per layer, for forward_into / trace capture."""
        return [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([MAX_BATCH, self.hidden]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self.device,
                ttnn.L1_MEMORY_CONFIG,
            )
            for _ in self.layers
        ]

    def forward_into(self, x, outs):
        """Like forward_device, but writes layer i into the preallocated outs[i] and frees nothing, so a
        whole decode step can be captured once with ttnn.begin_trace_capture and replayed."""
        for li, out in enumerate(outs):
            ttnn.generic_op([x, self.layers[li].w_in, out], self._program(li, x, out))
            x = out
        return x

    def __call__(self, hidden_states, layers=None):
        """One decode step. hidden_states: [batch, 1024] (or [batch, 1, 1024]) torch -> same shape.

        Continues from the L1 states and applies the selected layers one after another (GDN mixers
        only: no norms, residuals or MLP)."""
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        B, H = x.shape
        assert B == self.batch and H == self.hidden, "one token per sequence per call"
        rows = torch.zeros(MAX_BATCH, H)
        rows[:B] = x.float()
        xd = ttnn.from_torch(rows, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                             memory_config=ttnn.L1_MEMORY_CONFIG)
        y = self.forward_device(xd, layers)
        out = ttnn.to_torch(y)[:B].reshape(shape)
        ttnn.deallocate(y)
        return out
