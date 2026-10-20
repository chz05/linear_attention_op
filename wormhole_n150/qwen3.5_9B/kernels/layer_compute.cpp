// SPDX-License-Identifier: Apache-2.0
//
// Fused Gated DeltaNet layer -- compute. Per 32-row block (rows = tokens, sequence-major):
//   C1  causal conv1d + SiLU on this core's q/4, k/4, v/2 channels      (x_s @ diag(w_tap))
//   C2  |q|^2, |k|^2, q.k per row, on the full k-head gathered from the 4-core group
//   C3  per-row scalars on the SFPU: 1/|q| (with dk^-0.5), 1/|k|, d = exp(g), beta, ...
//   C4  row-scaled vectors d*q, -d*k, beta*k for the recurrence
//   C5  recurrence, token by token, state in L1 (see gdn_compute.cpp)
//   C6  gated RMSNorm over the head (partial sum of squares swapped with the partner core)
//       -> y = rmsnorm(o) * w * silu(z), written to this core's slice of Y for out_proj.

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

constexpr uint32_t c_qd = 0, c_nkd = 1, c_kb = 2, c_v = 3, c_d = 4, c_w = 5, c_id = 6, c_u = 7, c_state = 8,
                   c_sync = 9;
constexpr uint32_t c_p = 10, c_const = 11, c_xs = 12, c_ab = 13, c_qkloc = 14, c_qkfull = 15, c_out = 16, c_cv = 17,
                   c_tmp = 18, c_scal = 19, c_scal2 = 20, c_prep = 21, c_oblk = 22, c_ss = 23, c_ssp = 24,
                   c_rstd = 25, c_tmp2 = 26, c_tmp3 = 27, c_y = 28, c_ssk = 30;

constexpr uint32_t KT = 4, NCT = 2, P_TILES = 6, CONV_TILES = 4;
// constant tiles: [conv diag: shift s, channel tile ct -> s*4+ct][norm w rows: 16,17][head selector: 18][ones col: 19]
constexpr uint32_t K_NW = 16, K_SEL = 18, K_ONES = 19;

#ifndef DEBUG_STAGE
#define DEBUG_STAGE 0
#endif

constexpr uint32_t F_ONE = 0x3f800000;
constexpr uint32_t F_TWENTY = 0x41a00000;

inline void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb);
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    const uint32_t M = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t dt_bias = get_arg_val<uint32_t>(2);   // fp32 bits
    const uint32_t neg_A = get_arg_val<uint32_t>(3);     // -exp(A_log), fp32 bits
    const uint32_t qk_scale2 = get_arg_val<uint32_t>(4); // 1 / scale^2 = dk
    const uint32_t qk_eps2 = get_arg_val<uint32_t>(5);   // eps / scale^2
    const uint32_t qk_eps = get_arg_val<uint32_t>(6);
    const uint32_t inv_dv = get_arg_val<uint32_t>(7);    // 1 / head_v_dim
    const uint32_t norm_eps = get_arg_val<uint32_t>(8);
    const uint32_t num_blocks = (M + 31) / 32;

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_xs, c_const, c_qkloc);
    cb_wait_front(c_id, 1);
    cb_wait_front(c_const, K_ONES + 1);

    for (uint32_t rb = 0; rb < num_blocks; ++rb) {
        const uint32_t R = (M - rb * 32) < 32 ? (M - rb * 32) : 32;

        // ---- C1: conv + SiLU -------------------------------------------------------------
        cb_wait_front(c_xs, 3 * CONV_TILES);
        cb_reserve_back(c_qkloc, 2);
        cb_reserve_back(c_cv, NCT);
        for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
            matmul_init(c_xs, c_const);
            tile_regs_acquire();
            matmul_tiles(c_p, c_const, rb * P_TILES + ct, ct, 0);
            for (uint32_t s = 1; s <= 3; ++s) {
                matmul_tiles(c_xs, c_const, (s - 1) * CONV_TILES + ct, s * CONV_TILES + ct, 0);
            }
            silu_tile_init();
            silu_tile(0);
#if DEBUG_STAGE == 1
            if (ct >= 2) {
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, c_cv);
                pack_reconfig_data_format(c_y);
                pack_tile<true>(0, c_y, rb * NCT + ct - 2);
                pack_reconfig_data_format(c_cv);
                tile_regs_release();
                continue;
            }
#endif
            pack_one(ct < 2 ? c_qkloc : c_cv);
        }
        cb_push_back(c_qkloc, 2);
        cb_push_back(c_cv, NCT);
        cb_pop_front(c_xs, 3 * CONV_TILES);

        // ---- C2: row sums over the full k-head -------------------------------------------
        cb_wait_front(c_qkfull, 2 * KT);  // [q0..q3][k0..k3]
        cb_reserve_back(c_tmp, 3 * KT);
        mul_init(c_qkfull, c_qkfull);
        for (uint32_t i = 0; i < KT; ++i) {
            tile_regs_acquire();
            mul_tiles(c_qkfull, c_qkfull, i, i, 0);
            pack_one(c_tmp);
        }
        for (uint32_t i = 0; i < KT; ++i) {
            tile_regs_acquire();
            mul_tiles(c_qkfull, c_qkfull, KT + i, KT + i, 0);
            pack_one(c_tmp);
        }
        for (uint32_t i = 0; i < KT; ++i) {
            tile_regs_acquire();
            mul_tiles(c_qkfull, c_qkfull, i, KT + i, 0);
            pack_one(c_tmp);
        }
        cb_push_back(c_tmp, 3 * KT);
        cb_wait_front(c_tmp, 3 * KT);
        cb_reserve_back(c_scal, 5);
        matmul_init(c_tmp, c_const);
        for (uint32_t g = 0; g < 3; ++g) {  // |q|^2, |k|^2, q.k  (column 0)
            tile_regs_acquire();
            for (uint32_t i = 0; i < KT; ++i) {
                matmul_tiles(c_tmp, c_const, g * KT + i, K_ONES, 0);
            }
            pack_one(c_scal);
        }
        cb_pop_front(c_tmp, 3 * KT);

        // a, b of this core's head -> column 0
        cb_wait_front(c_ab, 2);
        matmul_init(c_ab, c_const);
        for (uint32_t i = 0; i < 2; ++i) {
            tile_regs_acquire();
            matmul_tiles(c_ab, c_const, i, K_SEL, 0);
            pack_one(c_scal);
        }
        cb_pop_front(c_ab, 2);
        cb_push_back(c_scal, 5);

        // ---- C3: per-row scalars (fp32 SFPU) ---------------------------------------------
        cb_wait_front(c_scal, 5);
        cb_reserve_back(c_scal2, 5);
        copy_tile_init(c_scal);
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
#if DEBUG_STAGE == 2
        pack_reconfig_data_format(c_y);
        pack_tile<true>(5, c_y, rb * NCT);
        pack_tile<true>(2, c_y, rb * NCT + 1);
        pack_reconfig_data_format(c_scal2);
#endif
        tile_regs_release();
        cb_push_back(c_scal2, 5);
        cb_pop_front(c_scal, 5);

        // ---- C4: row-scaled recurrence inputs --------------------------------------------
        cb_wait_front(c_scal2, 5);
        cb_reserve_back(c_prep, 3 * KT + 2);
        mul_bcast_cols_init(c_qkfull, c_scal2);
        for (uint32_t g = 0; g < 3; ++g) {  // d*q, -d*k, beta*k
            const uint32_t src = g == 0 ? 0 : KT;
            for (uint32_t i = 0; i < KT; ++i) {
                tile_regs_acquire();
                mul_tiles_bcast_cols(c_qkfull, c_scal2, src + i, g, 0);
                pack_one(c_prep);
            }
        }
        copy_tile_init(c_scal2);
        tile_regs_acquire();
        copy_tile(c_scal2, 4, 0);
        pack_one(c_prep);  // d - 1
        tile_regs_acquire();
        copy_tile(c_scal2, 3, 0);
        pack_one(c_prep);  // w
        cb_push_back(c_prep, 3 * KT + 2);
        cb_pop_front(c_scal2, 5);
        cb_pop_front(c_qkfull, 2 * KT);

        // ---- C5: recurrence (one row = one token of one sequence) -------------------------
        matmul_init(c_nkd, c_state);
        for (uint32_t r = 0; r < R; ++r) {
            if (rb + r > 0) {
                cb_wait_front(c_sync, 1);
                cb_pop_front(c_sync, 1);
            }
            const uint32_t s0 = ((rb * 32 + r) / T) * KT * NCT;

            cb_wait_front(c_nkd, KT);
            cb_wait_front(c_v, NCT);
            cb_reserve_back(c_u, NCT);
            for (uint32_t j = 0; j < NCT; ++j) {
                tile_regs_acquire();
                matmul_tiles(c_id, c_v, 0, j, 0);
                for (uint32_t i = 0; i < KT; ++i) {
                    matmul_tiles(c_nkd, c_state, i, s0 + i * NCT + j, 0);
                }
                pack_one(c_u);
            }
            cb_push_back(c_u, NCT);
            cb_pop_front(c_nkd, KT);
            cb_pop_front(c_v, NCT);

            cb_wait_front(c_u, NCT);
            cb_wait_front(c_qd, KT);
            cb_wait_front(c_w, 1);
            cb_reserve_back(c_out, NCT);
            for (uint32_t j = 0; j < NCT; ++j) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < KT; ++i) {
                    matmul_tiles(c_qd, c_state, i, s0 + i * NCT + j, 0);
                }
                matmul_tiles(c_w, c_u, 0, j, 0);
                pack_one(c_out);
            }
            cb_push_back(c_out, NCT);
            cb_pop_front(c_qd, KT);
            cb_pop_front(c_w, 1);

            cb_wait_front(c_d, 1);
            cb_wait_front(c_kb, KT);
            pack_reconfig_l1_acc(1);
            for (uint32_t i = 0; i < KT; ++i) {
                for (uint32_t j = 0; j < NCT; ++j) {
                    const uint32_t s = s0 + i * NCT + j;
                    tile_regs_acquire();
                    matmul_tiles(c_d, c_state, 0, s, 0);
                    matmul_tiles(c_kb, c_u, i, j, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, c_state, s);
                    tile_regs_release();
                }
            }
            pack_reconfig_l1_acc(0);
            cb_pop_front(c_d, 1);
            cb_pop_front(c_kb, KT);
            cb_pop_front(c_u, NCT);
            cb_reserve_back(c_sync, 1);
            cb_push_back(c_sync, 1);
        }

        // ---- C6: gated RMSNorm --------------------------------------------------------------
        // Every CB is always pushed in the same chunk size, so FIFO slots never straddle the wrap.
        cb_wait_front(c_oblk, NCT);
        cb_reserve_back(c_tmp2, NCT);
        mul_init(c_oblk, c_oblk);
        for (uint32_t j = 0; j < NCT; ++j) {
            tile_regs_acquire();
            mul_tiles(c_oblk, c_oblk, j, j, 0);
            pack_one(c_tmp2);
        }
        cb_push_back(c_tmp2, NCT);
        cb_wait_front(c_tmp2, NCT);
        cb_reserve_back(c_ss, 1);
        cb_reserve_back(c_ssk, 1);
        matmul_init(c_tmp2, c_const);
        tile_regs_acquire();
        for (uint32_t j = 0; j < NCT; ++j) {
            matmul_tiles(c_tmp2, c_const, j, K_ONES, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, c_ss);   // to the partner core
        pack_tile(0, c_ssk);  // kept here
        tile_regs_release();
        cb_push_back(c_ss, 1);
        cb_push_back(c_ssk, 1);
        cb_pop_front(c_tmp2, NCT);

        cb_wait_front(c_ssk, 1);
        cb_wait_front(c_ssp, 1);
        cb_reserve_back(c_rstd, 1);
        tile_regs_acquire();
        copy_tile_init(c_ssk);
        copy_tile(c_ssk, 0, 0);
        copy_tile_init(c_ssp);
        copy_tile(c_ssp, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_dv);
        add_unary_tile(0, norm_eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        pack_one(c_rstd);
        cb_push_back(c_rstd, 1);
        cb_pop_front(c_ssk, 1);
        cb_pop_front(c_ssp, 1);

        cb_wait_front(c_rstd, 1);
        cb_reserve_back(c_tmp2, NCT);
        mul_bcast_cols_init(c_oblk, c_rstd);
        for (uint32_t j = 0; j < NCT; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(c_oblk, c_rstd, j, 0, 0);
            pack_one(c_tmp2);
        }
        cb_push_back(c_tmp2, NCT);
        cb_pop_front(c_rstd, 1);
#ifndef DEBUG_Y_IS_O
        cb_pop_front(c_oblk, NCT);
#endif

        cb_wait_front(c_tmp2, NCT);
        cb_reserve_back(c_tmp3, NCT);
        mul_bcast_rows_init(c_tmp2, c_const);
        for (uint32_t j = 0; j < NCT; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(c_tmp2, c_const, j, K_NW + j, 0);
            pack_one(c_tmp3);
        }
        cb_push_back(c_tmp3, NCT);
        cb_pop_front(c_tmp2, NCT);

        cb_reserve_back(c_tmp2, NCT);
        for (uint32_t j = 0; j < NCT; ++j) {
            copy_tile_init(c_p);
            tile_regs_acquire();
            copy_tile(c_p, rb * P_TILES + 4 + j, 0);
            silu_tile_init();
            silu_tile(0);
            pack_one(c_tmp2);
        }
        cb_push_back(c_tmp2, NCT);

        cb_wait_front(c_tmp2, NCT);
        cb_wait_front(c_tmp3, NCT);
        mul_init(c_tmp3, c_tmp2);
        pack_reconfig_data_format(c_y);
        for (uint32_t j = 0; j < NCT; ++j) {
            tile_regs_acquire();
#ifdef DEBUG_Y_IS_O
            copy_tile_init(c_oblk);
            copy_tile(c_oblk, j, 0);
#else
            mul_tiles(c_tmp3, c_tmp2, j, j, 0);
#endif
            tile_regs_commit();
            tile_regs_wait();
#if DEBUG_STAGE == 0
            pack_tile<true>(0, c_y, rb * NCT + j);
#endif
            tile_regs_release();
        }
        pack_reconfig_data_format(c_tmp2);
        cb_pop_front(c_tmp2, NCT);
        cb_pop_front(c_tmp3, NCT);
#ifdef DEBUG_Y_IS_O
        cb_pop_front(c_oblk, NCT);
#endif
    }
    if (M > 0) {
        cb_wait_front(c_sync, 1);
        cb_pop_front(c_sync, 1);
    }
}
