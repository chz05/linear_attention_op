// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5-0.8B Gated DeltaNet layer, all weights in L1 -- compute. Decode only: one launch = one token
// for each of the B sequences (row b = sequence b, B <= 32):
//   C0  in_proj from this core's resident weight slice -> P = [q/4 | k/4 | v/4 | z/4 | a,b of all heads]
//   C1  causal conv1d + SiLU on q/4, k/4, v/4                          (x_s @ diag(w_tap))
//   C2  |q|^2, |k|^2, q.k over the full head (quarters gathered from the head's 4 cores)
//   C3  per-row scalars on the SFPU: 1/|q| (with dk^-0.5), 1/|k|, d = exp(g), beta, ...
//   C4  row-scaled vectors d*q, -d*k, beta*k for the recurrence
//   C5  recurrence, token by token, state in L1
//   C6  gated RMSNorm over the head (sum of squares combined across the head's 4 cores)
//   C7  partial out_proj: y (32 channels) @ this core's 32 rows of W_o^T -> 32 output tiles
//   C8  (owners) sum the 64 cores' partial rows for this core's output tile
//
// Weights, conv taps and norm weight are bf16 / bfp8 / bfp4 tiles in srcA; everything else is fp32.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "tools/profiler/kernel_profiler.hpp"  // zones compile to nothing unless profiling

constexpr uint32_t c_qd = 0, c_nkd = 1, c_kb = 2, c_v = 3, c_d = 4, c_w = 5, c_recv = 6, c_u = 7, c_state = 8,
                   c_sync = 9, c_x = 10, c_win = 11, c_wout = 12, c_lconst = 13, c_gconst = 14, c_p = 15, c_xs = 16,
                   c_qkloc = 17, c_qkfull = 18, c_cv = 19, c_tmp = 20, c_scal = 21, c_scal2 = 22, c_prep = 23,
                   c_out = 24, c_oblk = 25, c_ss = 26, c_ssp = 27, c_tmp2 = 29, c_tmp3 = 30;

constexpr uint32_t KT = 4, KX = 32, X_CHUNK = 4, P_TILES = 5, CONV_TILES = 3, N_OUT = 32, GROUP = 4;
// layer constants (bf16): conv diag [shift s][channel tile ct] -> s*3+ct, norm-weight diag -> 12
constexpr uint32_t L_NORM = 12, N_LCONST = 13;
// global constants (fp32): identity, ones column, ones row, a selector, b selector
constexpr uint32_t G_ID = 0, G_ONES_COL = 1, G_ONES_ROW = 2, G_SEL_A = 3, G_SEL_B = 4;

constexpr uint32_t F_ONE = 0x3f800000;
constexpr uint32_t F_TWENTY = 0x41a00000;

// matmul in0 @ in1: in0 -> srcB, in1 -> srcA
inline void mm_init(uint32_t in0, uint32_t in1) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1);
}

inline void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb);
    tile_regs_release();
}

inline void copy_init(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_tile_init(cb);
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    const uint32_t dt_bias = get_arg_val<uint32_t>(0);
    const uint32_t neg_A = get_arg_val<uint32_t>(1);
    const uint32_t qk_scale2 = get_arg_val<uint32_t>(2);
    const uint32_t qk_eps2 = get_arg_val<uint32_t>(3);
    const uint32_t qk_eps = get_arg_val<uint32_t>(4);
    const uint32_t inv_dv = get_arg_val<uint32_t>(5);
    const uint32_t norm_eps = get_arg_val<uint32_t>(6);
    const uint32_t is_owner = get_arg_val<uint32_t>(7);

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_x, c_win, c_p);

    // ---- C0: in_proj ------------------------------------------------------------------------------
    {
    DeviceZoneScopedN("IN_PROJ");
    cb_reserve_back(c_p, P_TILES);
#ifdef MM_BLOCK
    reconfig_data_format<SrcOrder::Reverse>(c_x, c_win);
    matmul_block_init(c_x, c_win, 0, P_TILES, 1, X_CHUNK);  // one x tile x a row of 5 weight tiles per call
#else
    mm_init(c_x, c_win);
#endif
    tile_regs_acquire();
    for (uint32_t k = 0; k < KX; k += X_CHUNK) {
        cb_wait_front(c_x, X_CHUNK);
#ifdef WEIGHTS_IN_DRAM
        cb_wait_front(c_win, X_CHUNK * P_TILES);  // streamed chunk
        const uint32_t w0 = 0;
#else
        const uint32_t w0 = k * P_TILES;  // resident slice
#endif
        for (uint32_t kk = 0; kk < X_CHUNK; ++kk) {
#ifdef MM_BLOCK
            matmul_block(c_x, c_win, kk, w0 + kk * P_TILES, 0, 0, P_TILES, 1, X_CHUNK);
#else
            for (uint32_t j = 0; j < P_TILES; ++j) {
                matmul_tiles(c_x, c_win, kk, w0 + kk * P_TILES + j, j);
            }
#endif
        }
        cb_pop_front(c_x, X_CHUNK);
#ifdef WEIGHTS_IN_DRAM
        cb_pop_front(c_win, X_CHUNK * P_TILES);
#endif
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < P_TILES; ++j) {
        pack_tile(j, c_p);
    }
    tile_regs_release();
    cb_push_back(c_p, P_TILES);
    cb_wait_front(c_p, P_TILES);
    }

    // ---- C1: conv + SiLU -------------------------------------------------------------------------
#ifdef WEIGHTS_IN_DRAM
    cb_wait_front(c_lconst, N_LCONST);
#endif
    cb_wait_front(c_xs, 3 * CONV_TILES);
    cb_reserve_back(c_qkloc, 2);
    cb_reserve_back(c_cv, 1);
    for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
        mm_init(c_p, c_lconst);
        tile_regs_acquire();
        matmul_tiles(c_p, c_lconst, ct, ct, 0);
        for (uint32_t s = 1; s <= 3; ++s) {
            matmul_tiles(c_xs, c_lconst, (s - 1) * CONV_TILES + ct, s * CONV_TILES + ct, 0);
        }
        silu_tile_init();
        silu_tile(0);
        pack_one(ct < 2 ? c_qkloc : c_cv);
    }
    cb_push_back(c_qkloc, 2);
    cb_push_back(c_cv, 1);
    cb_pop_front(c_xs, 3 * CONV_TILES);

    // ---- C2: row sums over the full head ---------------------------------------------------------
    cb_wait_front(c_qkfull, 2 * KT);  // [q0..q3][k0..k3]
    cb_reserve_back(c_scal, 5);
    for (uint32_t g = 0; g < 3; ++g) {  // |q|^2, |k|^2, q.k  (column 0)
        const uint32_t a0 = g == 1 ? KT : 0, b0 = g == 0 ? 0 : KT;
        cb_reserve_back(c_tmp, KT);
        reconfig_data_format(c_qkfull, c_qkfull);
        mul_init(c_qkfull, c_qkfull);
        for (uint32_t i = 0; i < KT; ++i) {
            tile_regs_acquire();
            mul_tiles(c_qkfull, c_qkfull, a0 + i, b0 + i, 0);
            pack_one(c_tmp);
        }
        cb_push_back(c_tmp, KT);
        cb_wait_front(c_tmp, KT);
        mm_init(c_tmp, c_gconst);
        tile_regs_acquire();
        for (uint32_t i = 0; i < KT; ++i) {
            matmul_tiles(c_tmp, c_gconst, i, G_ONES_COL, 0);
        }
        pack_one(c_scal);
        cb_pop_front(c_tmp, KT);
    }
    mm_init(c_p, c_gconst);  // a, b of this core's head -> column 0
    tile_regs_acquire();
    matmul_tiles(c_p, c_gconst, 4, G_SEL_A, 0);
    pack_one(c_scal);
    tile_regs_acquire();
    matmul_tiles(c_p, c_gconst, 4, G_SEL_B, 0);
    pack_one(c_scal);
    cb_push_back(c_scal, 5);

    // ---- C3: per-row scalars (fp32 SFPU) -----------------------------------------------------------
    cb_wait_front(c_scal, 5);
    cb_reserve_back(c_scal2, 5);
    copy_init(c_scal);
    tile_regs_acquire();
    for (uint32_t i = 0; i < 5; ++i) {
        copy_tile(c_scal, i, i);  // 0:|q|^2 1:|k|^2 2:q.k 3:a 4:b
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, qk_scale2);
    add_unary_tile(0, qk_eps2);
    add_unary_tile(1, qk_eps);
    add_unary_tile(3, dt_bias);
    rsqrt_tile_init();
    rsqrt_tile(0);  // dk^-0.5 / |q|
    rsqrt_tile(1);  // 1 / |k|
    softplus_tile_init();
    softplus_tile(3, F_ONE, F_ONE, F_TWENTY);
    binop_with_scalar_tile_init();
    mul_unary_tile(3, neg_A);  // g
    exp_tile_init();
    exp_tile(3);  // d = exp(g)
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(3, 5);
    binop_with_scalar_tile_init();
    sub_unary_tile(5, F_ONE);  // d - 1
    sigmoid_tile_init();
    sigmoid_tile(4);  // beta
    mul_binary_tile_init();
    mul_binary_tile(3, 0, 6);  // d * rq
    mul_binary_tile(3, 1, 7);  // d * rk
    mul_binary_tile(0, 2, 2);  // rq * q.k
    mul_binary_tile(4, 1, 1);  // beta * rk
    mul_binary_tile(2, 1, 2);  // beta * rq * rk * q.k
    negative_tile_init();
    negative_tile(7);  // -d * rk
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(6, c_scal2);  // 0: d*rq
    pack_tile(7, c_scal2);  // 1: -d*rk
    pack_tile(1, c_scal2);  // 2: beta*rk
    pack_tile(2, c_scal2);  // 3: w = beta * qn.kn
    pack_tile(5, c_scal2);  // 4: d - 1
    tile_regs_release();
    cb_push_back(c_scal2, 5);
    cb_pop_front(c_scal, 5);

    // ---- C4: row-scaled recurrence inputs ------------------------------------------------------------
    cb_wait_front(c_scal2, 5);
    cb_reserve_back(c_prep, 3 * KT + 2);
    reconfig_data_format(c_qkfull, c_scal2);
    mul_bcast_cols_init(c_qkfull, c_scal2);
    for (uint32_t g = 0; g < 3; ++g) {  // d*q, -d*k, beta*k
        const uint32_t src = g == 0 ? 0 : KT;
        for (uint32_t i = 0; i < KT; ++i) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(c_qkfull, c_scal2, src + i, g, 0);
            pack_one(c_prep);
        }
    }
    copy_init(c_scal2);
    tile_regs_acquire();
    copy_tile(c_scal2, 4, 0);
    pack_one(c_prep);  // d - 1
    tile_regs_acquire();
    copy_tile(c_scal2, 3, 0);
    pack_one(c_prep);  // w
    cb_push_back(c_prep, 3 * KT + 2);
    cb_pop_front(c_scal2, 5);
    cb_pop_front(c_qkfull, 2 * KT);

    // ---- C5: recurrence (row b = sequence b, its own state slot) -----------------------------------------
    mm_init(c_nkd, c_state);
    for (uint32_t r = 0; r < B; ++r) {
        if (r > 0) {
            cb_wait_front(c_sync, 1);
            cb_pop_front(c_sync, 1);
        }
        const uint32_t s0 = r * KT;

        cb_wait_front(c_nkd, KT);
        cb_wait_front(c_v, 1);
        cb_reserve_back(c_u, 1);
        tile_regs_acquire();
        matmul_tiles(c_gconst, c_v, G_ID, 0, 0);
        for (uint32_t i = 0; i < KT; ++i) {
            matmul_tiles(c_nkd, c_state, i, s0 + i, 0);
        }
        pack_one(c_u);
        cb_push_back(c_u, 1);
        cb_pop_front(c_nkd, KT);
        cb_pop_front(c_v, 1);

        cb_wait_front(c_u, 1);
        cb_wait_front(c_qd, KT);
        cb_wait_front(c_w, 1);
        cb_reserve_back(c_out, 1);
        tile_regs_acquire();
        for (uint32_t i = 0; i < KT; ++i) {
            matmul_tiles(c_qd, c_state, i, s0 + i, 0);
        }
        matmul_tiles(c_w, c_u, 0, 0, 0);
        pack_one(c_out);
        cb_push_back(c_out, 1);
        cb_pop_front(c_qd, KT);
        cb_pop_front(c_w, 1);

        cb_wait_front(c_d, 1);
        cb_wait_front(c_kb, KT);
        pack_reconfig_l1_acc(1);
        for (uint32_t i = 0; i < KT; ++i) {
            tile_regs_acquire();
            matmul_tiles(c_d, c_state, 0, s0 + i, 0);
            matmul_tiles(c_kb, c_u, i, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, c_state, s0 + i);
            tile_regs_release();
        }
        pack_reconfig_l1_acc(0);
        cb_pop_front(c_d, 1);
        cb_pop_front(c_kb, KT);
        cb_pop_front(c_u, 1);
        cb_reserve_back(c_sync, 1);
        cb_push_back(c_sync, 1);
    }
    cb_wait_front(c_sync, 1);
    cb_pop_front(c_sync, 1);

    // ---- C6: gated RMSNorm --------------------------------------------------------------------------
    cb_wait_front(c_oblk, 1);
    cb_reserve_back(c_tmp2, 1);
    reconfig_data_format(c_oblk, c_oblk);
    mul_init(c_oblk, c_oblk);
    tile_regs_acquire();
    mul_tiles(c_oblk, c_oblk, 0, 0, 0);
    pack_one(c_tmp2);
    cb_push_back(c_tmp2, 1);
    cb_wait_front(c_tmp2, 1);
    cb_reserve_back(c_ss, 1);
    mm_init(c_tmp2, c_gconst);
    tile_regs_acquire();
    matmul_tiles(c_tmp2, c_gconst, 0, G_ONES_COL, 0);
    pack_one(c_ss);  // to the head's 4 cores
    cb_push_back(c_ss, 1);
    cb_pop_front(c_tmp2, 1);

    cb_wait_front(c_ssp, GROUP);
    cb_reserve_back(c_tmp3, 1);  // rstd
    copy_init(c_ssp);
    tile_regs_acquire();
    for (uint32_t i = 0; i < GROUP; ++i) {
        copy_tile(c_ssp, i, i);
    }
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(2, 3, 2);
    add_binary_tile(0, 2, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_dv);
    add_unary_tile(0, norm_eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    pack_one(c_tmp3);
    cb_push_back(c_tmp3, 1);
    cb_pop_front(c_ssp, GROUP);

    cb_wait_front(c_tmp3, 1);
    cb_reserve_back(c_tmp2, 1);
    reconfig_data_format(c_oblk, c_tmp3);
    mul_bcast_cols_init(c_oblk, c_tmp3);
    tile_regs_acquire();
    mul_tiles_bcast_cols(c_oblk, c_tmp3, 0, 0, 0);
    pack_one(c_tmp2);  // o * rstd
    cb_push_back(c_tmp2, 1);
    cb_pop_front(c_tmp3, 1);
    cb_pop_front(c_oblk, 1);

    cb_wait_front(c_tmp2, 1);
    cb_reserve_back(c_tmp3, 1);
    mm_init(c_tmp2, c_lconst);
    tile_regs_acquire();
    matmul_tiles(c_tmp2, c_lconst, 0, L_NORM, 0);
    pack_one(c_tmp3);  // * norm weight
    cb_push_back(c_tmp3, 1);
    cb_pop_front(c_tmp2, 1);

    cb_reserve_back(c_tmp2, 1);
    copy_init(c_p);
    tile_regs_acquire();
    copy_tile(c_p, 3, 0);
    silu_tile_init();
    silu_tile(0);
    pack_one(c_tmp2);  // silu(z)
    cb_push_back(c_tmp2, 1);
    cb_pop_front(c_p, P_TILES);

    cb_wait_front(c_tmp2, 1);
    cb_wait_front(c_tmp3, 1);
    cb_reserve_back(c_tmp2, 1);  // capacity 2: y goes in the second slot
    reconfig_data_format(c_tmp3, c_tmp2);
    mul_init(c_tmp3, c_tmp2);
    tile_regs_acquire();
    mul_tiles(c_tmp3, c_tmp2, 0, 0, 0);
    pack_one(c_tmp2);
    cb_push_back(c_tmp2, 1);
    cb_pop_front(c_tmp2, 1);  // silu(z)
    cb_pop_front(c_tmp3, 1);

    // ---- C7: partial out_proj -----------------------------------------------------------------------
    {
    DeviceZoneScopedN("OUT_PROJ");
    cb_wait_front(c_tmp2, 1);  // y [B x 32 channels]
#ifdef WEIGHTS_IN_DRAM
    cb_wait_front(c_wout, N_OUT);
#endif
#ifdef MM_BLOCK
    constexpr uint32_t OB = 8;  // output tiles per call (all 8 fp32 DST tiles)
    reconfig_data_format<SrcOrder::Reverse>(c_tmp2, c_wout);
    matmul_block_init(c_tmp2, c_wout, 0, OB, 1, 1);
    for (uint32_t j = 0; j < N_OUT; j += OB) {
        tile_regs_acquire();
        matmul_block(c_tmp2, c_wout, 0, j, 0, 0, OB, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < OB; ++i) {
            cb_reserve_back(c_out, 1);
            pack_tile(i, c_out);
            cb_push_back(c_out, 1);
        }
        tile_regs_release();
    }
#else
    mm_init(c_tmp2, c_wout);
    for (uint32_t j = 0; j < N_OUT; ++j) {
        cb_reserve_back(c_out, 1);
        tile_regs_acquire();
        matmul_tiles(c_tmp2, c_wout, 0, j, 0);
        pack_one(c_out);
        cb_push_back(c_out, 1);
    }
#endif
    cb_pop_front(c_tmp2, 1);
#ifdef WEIGHTS_IN_DRAM
    cb_pop_front(c_wout, N_OUT);
    cb_pop_front(c_lconst, N_LCONST);
#endif
    }

    // ---- C8: owners reduce the 64 partials of their output tile ------------------------------------
    if (is_owner) {
        cb_wait_front(c_recv, 2 * B);  // row r: tiles 2r (cores 0..31), 2r+1 (cores 32..63), one core per row
        mm_init(c_gconst, c_recv);
        for (uint32_t r = 0; r < B; ++r) {
            cb_reserve_back(c_ss, 1);
            tile_regs_acquire();
            matmul_tiles(c_gconst, c_recv, G_ONES_ROW, 2 * r, 0);
            matmul_tiles(c_gconst, c_recv, G_ONES_ROW, 2 * r + 1, 0);
            pack_one(c_ss);  // row 0 = sum over the 64 cores
            cb_push_back(c_ss, 1);
        }
        cb_pop_front(c_recv, 2 * B);
    }
}
