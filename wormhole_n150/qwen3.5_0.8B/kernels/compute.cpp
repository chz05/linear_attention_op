// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5-0.8B Gated DeltaNet layer, all weights in L1 -- compute. Decode only: one launch = one token
// for each of the B sequences (row b = sequence b, B <= 16; B is a runtime argument, so a shrinking batch reuses
// the program of its BP; rows B..BP-1 are zero padding):
//   C0  in_proj from this core's resident weight slice -> P = [q/4 | k/4 | v/4 | z/4 | a,b of all heads]
//       (the input is read in place as BP x 32 tiles)
//   C1  causal conv1d + SiLU on q/4, k/4, v/4 (x_s @ diag(w_tap), x_s read in place from the conv state),
//       and this quarter's q*q, k*k, q*k
//   C2  this quarter's |q|^2, |k|^2, q.k as row sums in one tile (the writer sends it to the head's 4 cores)
//   C3  the head's sums, this head's a and b, then every per-row scalar of the step on the SFPU: 1/|q| (with
//       dk^-0.5), 1/|k|, d = exp(g), beta, ...
//   C4  this quarter's d*q, -d*k, beta*k (swapped within the head by the writer), then (beta*k)^T of the head
//   C5  recurrence, token by token, state in L1
//   C6  gated RMSNorm over the head (sum of squares combined across the head's 4 cores)
//   C7  out_proj: output column tile c % 32 over the v channels of this core's half of the grid (c / 32)
//   C8  + the other half's partial of the same tile
//
// Weights and conv taps are bf16 / bfp8 / bfp4 tiles in srcA; everything else is fp32. SFPU work only touches
// the batch rows (see below).

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

constexpr uint32_t c_qd = 0, c_kb = 2, c_v = 3, c_d = 4, c_w = 5, c_pp = 6, c_u = 7, c_state = 8, c_part = 9,
                   c_x = 10, c_win = 11, c_wout = 12, c_lconst = 13, c_gconst = 14, c_p = 15, c_pfull = 16,
                   c_qkloc = 17, c_qkfull = 18, c_cv = 19, c_tmp = 20, c_yg = 21, c_scal2 = 22, c_prep = 23,
                   c_out = 24, c_oblk = 25, c_ss = 26, c_ssp = 27, c_sc = 28, c_tmp2 = 29, c_tmp3 = 30, c_conv = 31;

constexpr uint32_t KT = 4, KX = 32, X_CHUNK = 4, P_TILES = 5, CONV_TILES = 3, KY = 32, GROUP = 4, N_HALF = 32;
// layer constants (bf16): conv diag [shift s][channel tile ct] -> s*3+ct. The RMSNorm weight is folded into
// this core's out_proj rows on the host.
constexpr uint32_t N_LCONST = 12;
// global constants (fp32): identity; ones in column 0, 1, 16 (row sums into that column); a selector, b selector;
// group-sum selector (row r <- every row j with j % BP == r)
constexpr uint32_t G_ID = 0, G_ONES_COL = 1, G_ONES_C1 = 2, G_ONES_C16 = 3, G_SEL_A = 4, G_SEL_B = 5, G_SEL_GRP = 6;

// matmul in0 @ in1: in0 -> srcB, in1 -> srcA
inline void mm_init(uint32_t in0, uint32_t in1) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1);
}

// matmul_block_init / matmul_block at a given math fidelity (the kernel-wide one is MATH_FIDELITY)
#ifndef IN_PROJ_FIDELITY
#define IN_PROJ_FIDELITY MATH_FIDELITY
#endif
template <MathFidelity fidelity>
ALWI void matmul_block_init_at(uint32_t in0, uint32_t in1, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<fidelity, MM_THROTTLE>(in0, in1, 0, ct_dim, rt_dim)));
}
template <MathFidelity fidelity>
ALWI void matmul_block_at(
    uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst, uint32_t ct_dim, uint32_t rt_dim,
    uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul<fidelity, MM_THROTTLE>(idst, ct_dim, rt_dim)));
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

// ---- SFPU work on the batch rows only ---------------------------------------------------------------------
// In a face, SFPU row i (dst_reg[i]) holds DST rows 4(i/2)..4(i/2)+3 at the even (i even) or odd (i odd)
// columns; an fp32 tile is 32 SFPU rows (faces 0..3, 8 each). With G = ceil(BP/4) groups of 4 rows, rows 0..BP-1
// of a tile are SFPU rows 0..2G-1 of faces 0 and 1, and their column 0 is SFPU rows 0, 2, .., 2G-2 of face 0
// (the zero padding rows B..BP-1 are computed too and stay finite).
// The SFPU functions below run once on the tile at the DST base (VectorMode::None) and touch only those rows.
#ifdef TRISC_MATH
namespace ckernel::sfpu {

constexpr int SFPU_TILE = 32, SFPU_FACE = 8;

// advance the DST base by N SFPU rows (one increment moves at most 3)
template <int N>
sfpi_inline void skip_rows() {
    if constexpr (N > 3) {
        sfpi::dst_reg += 3;
        skip_rows<N - 3>();
    } else if constexpr (N > 0) {
        sfpi::dst_reg += N;
    }
}

// 1/sqrt(x) for x > 0: _calculate_sqrt_body_<false, true> with immediate constants (the reciprocal used by
// sigmoid owns the programmable ones)
sfpi_inline sfpi::vFloat rsqrt_pos(const sfpi::vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - i);
    sfpi::vFloat xy = x * y;
    sfpi::vFloat c = -y * xy;
    y = y * (2.2825186f + c * (2.2533049f + c));
    xy = x * y;
    sfpi::vFloat one_minus_xyy = 1.0f - y * xy;
    return one_minus_xyy * sfpi::addexp(y, -1) + y;
}

// softplus(t) = max(t, 0) + ln(1 + exp(-|t|)), t >= 20 passed through (the fp32 path of calculate_softplus_body)
sfpi_inline sfpi::vFloat softplus_f32(const sfpi::vFloat t) {
    sfpi::vFloat y = t;
    v_if(t < 20.0f) {
        sfpi::vFloat a = sfpi::setsgn(t, 0);
        sfpi::vFloat r = PolynomialEvaluator::eval(
            a, SOFTPLUS_POLY_C0, SOFTPLUS_POLY_C1, SOFTPLUS_POLY_C2, SOFTPLUS_POLY_C3, SOFTPLUS_POLY_C4,
            SOFTPLUS_POLY_C5, SOFTPLUS_POLY_C6, SOFTPLUS_POLY_C7, SOFTPLUS_POLY_C8);
        v_if(a > SOFTPLUS_POLY_BOUNDARY) {
            sfpi::vFloat e = _sfpu_exp_fp32_accurate_(sfpi::setsgn(a, 1));
            r = e * (1.0f + e * (-0.5f + e * 0.333333343f));
        }
        v_endif;
        y = sfpi::max(t, 0.0f) + r;
    }
    v_endif;
    return y;
}

// x * sigmoid(x) on rows 0..B-1 (silu_tile_init() programs the reciprocal constants)
template <int G>
inline void silu_rows() {
    for (int f = 0; f < 2; ++f) {
#pragma GCC unroll 0
        for (int i = 0; i < 2 * G; ++i) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = x * _sfpu_sigmoid_<true>(x);
            sfpi::dst_reg++;
        }
        skip_rows<SFPU_FACE - 2 * G>();
    }
}

// silu(tile 0) * tile 1 -> tile 0, rows 0..B-1
template <int G>
inline void silu_mul_rows() {
    for (int f = 0; f < 2; ++f) {
#pragma GCC unroll 0
        for (int i = 0; i < 2 * G; ++i) {
            sfpi::vFloat z = sfpi::dst_reg[0];
            sfpi::vFloat o = sfpi::dst_reg[SFPU_TILE];
            sfpi::dst_reg[0] = z * _sfpu_sigmoid_<true>(z) * o;
            sfpi::dst_reg++;
        }
        skip_rows<SFPU_FACE - 2 * G>();
    }
}

// q*q, k*k, q*k (tiles 0, 1) -> tiles 3, 4, 5, rows 0..B-1
template <int G>
inline void prod3_rows() {
#pragma GCC unroll 2
    for (int f = 0; f < 2; ++f) {
#pragma GCC unroll 8
        for (int i = 0; i < 2 * G; ++i) {
            const int o = f * SFPU_FACE + i;
            sfpi::vFloat q = sfpi::dst_reg[o], k = sfpi::dst_reg[SFPU_TILE + o];
            sfpi::dst_reg[3 * SFPU_TILE + o] = q * q;
            sfpi::dst_reg[4 * SFPU_TILE + o] = k * k;
            sfpi::dst_reg[5 * SFPU_TILE + o] = q * k;
        }
    }
}

// RMSNorm rstd = 1 / sqrt((tiles 0 + 1 + 2 + 3) / dv + eps) -> tile 0, column 0 of rows 0..B-1
template <int G>
inline void rstd_rows(const uint32_t inv_dv, const uint32_t norm_eps) {
    const float inv = Converter::as_float(inv_dv), eps = Converter::as_float(norm_eps);
#pragma GCC unroll 4
    for (int g = 0; g < G; ++g) {
        const int o = 2 * g;
        sfpi::vFloat t0 = sfpi::dst_reg[o], t1 = sfpi::dst_reg[SFPU_TILE + o];
        sfpi::vFloat t2 = sfpi::dst_reg[2 * SFPU_TILE + o], t3 = sfpi::dst_reg[3 * SFPU_TILE + o];
        sfpi::dst_reg[o] = rsqrt_pos(((t0 + t1) + (t2 + t3)) * inv + eps);
    }
}

// The step's per-row scalars, for rows 0..B-1:
//   in   tile 0: |q|^2 in column 0, |k|^2 in column 1, q.k in column 16 (SFPU rows 2g, 2g + 1, 8 + 2g: same lanes);
//        tile 3: a, tile 4: b (column 0)
//   out  column 0 of tile 0: d*rq  1: -d*rk  2: beta*rk  3: w = beta*rq*rk*q.k  4: d - 1
// rq = dk^-0.5 / |q| and rk = 1 / |k| (l2norm, eps inside), d = exp(-exp(A_log) * softplus(a + dt_bias)),
// beta = sigmoid(b).
template <int G>
inline void gating_rows(
    const uint32_t dt_bias, const uint32_t neg_A, const uint32_t qk_scale2, const uint32_t qk_eps2,
    const uint32_t qk_eps) {
    sfpu_reciprocal_init<false>();  // for sigmoid
    const float dt = Converter::as_float(dt_bias), nA = Converter::as_float(neg_A);
    const float s2 = Converter::as_float(qk_scale2), e2 = Converter::as_float(qk_eps2), e = Converter::as_float(qk_eps);
#pragma GCC unroll 0
    for (int g = 0; g < G; ++g) {
        sfpi::vFloat d = _sfpu_exp_fp32_accurate_(nA * softplus_f32(sfpi::vFloat(sfpi::dst_reg[3 * SFPU_TILE]) + dt));
        sfpi::dst_reg[4 * SFPU_TILE] = _sfpu_sigmoid_<true>(sfpi::dst_reg[4 * SFPU_TILE]);
        sfpi::dst_reg[3 * SFPU_TILE] = d;
        sfpi::vFloat rq = rsqrt_pos(sfpi::vFloat(sfpi::dst_reg[0]) * s2 + e2);
        sfpi::vFloat rk = rsqrt_pos(sfpi::vFloat(sfpi::dst_reg[1]) + e);
        sfpi::vFloat beta = sfpi::dst_reg[4 * SFPU_TILE];
        sfpi::vFloat qk = sfpi::dst_reg[SFPU_FACE];
        sfpi::dst_reg[0] = d * rq;
        sfpi::dst_reg[SFPU_TILE] = -(d * rk);
        sfpi::dst_reg[2 * SFPU_TILE] = beta * rk;
        sfpi::dst_reg[3 * SFPU_TILE] = beta * rq * qk * rk;
        sfpi::dst_reg[4 * SFPU_TILE] = d - 1.0f;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
#endif

// run one of the batch-row SFPU functions above on DST tile idst
#define SFPU_ROWS(fn, idst, ...) \
    MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::fn, idst, VectorMode::None, ##__VA_ARGS__)))
#define SFPU_ROWS_INIT() MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused>()))

void kernel_main() {
    constexpr uint32_t BP = get_compile_time_arg_val(0);
    constexpr int G = (BP + 3) / 4;  // groups of 4 batch rows (SFPU work)
    constexpr uint32_t PFULL = BP * GROUP / 32 > 1 ? BP * GROUP / 32 : 1;  // tiles of the head's 4 partial sums
    const uint32_t dt_bias = get_arg_val<uint32_t>(0);
    const uint32_t neg_A = get_arg_val<uint32_t>(1);
    const uint32_t qk_scale2 = get_arg_val<uint32_t>(2);
    const uint32_t qk_eps2 = get_arg_val<uint32_t>(3);
    const uint32_t qk_eps = get_arg_val<uint32_t>(4);
    const uint32_t inv_dv = get_arg_val<uint32_t>(5);
    const uint32_t norm_eps = get_arg_val<uint32_t>(6);
    const uint32_t core = get_arg_val<uint32_t>(7);
    const uint32_t B = get_arg_val<uint32_t>(8);  // sequences in the batch, 1..BP

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_x, c_win, c_p);

    // ---- C0: in_proj ------------------------------------------------------------------------------
    // x tiles are BP x 32 (the batch rows only): a fraction of a full tile's srcB data and MVMULs
    {
    DeviceZoneScopedN("IN_PROJ");
    cb_reserve_back(c_p, P_TILES);
#ifdef MM_BLOCK
    reconfig_data_format<SrcOrder::Reverse>(c_x, c_win);
    matmul_block_init_at<IN_PROJ_FIDELITY>(c_x, c_win, P_TILES, 1, X_CHUNK);  // one x tile x a row of 5 weight tiles
#else
    mm_init(c_x, c_win);
#endif
    tile_regs_acquire();
    for (uint32_t k = 0; k < KX; k += X_CHUNK) {
        {
        DeviceZoneScopedN("IN_X_WAIT");
        cb_wait_front(c_x, X_CHUNK);
        }
#ifdef WEIGHTS_IN_DRAM
        cb_wait_front(c_win, X_CHUNK * P_TILES);  // streamed chunk
        const uint32_t w0 = 0;
#else
        const uint32_t w0 = k * P_TILES;  // resident slice
#endif
        for (uint32_t kk = 0; kk < X_CHUNK; ++kk) {
#ifdef MM_BLOCK
            matmul_block_at<IN_PROJ_FIDELITY>(c_x, c_win, kk, w0 + kk * P_TILES, 0, P_TILES, 1, X_CHUNK);
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
    { DeviceZoneScopedN("CONV");
#ifdef WEIGHTS_IN_DRAM
    cb_wait_front(c_lconst, N_LCONST);
#endif
    cb_reserve_back(c_qkloc, 2);
    cb_reserve_back(c_cv, 1);
    cb_reserve_back(c_tmp, 3);
    // all 12 tap matmuls in one DST pass (DST ct = channel tile ct): the inputs 1..3 steps back are read straight
    // from the conv state (BP x 32 tiles, the same shape as in_proj's x, so only the formats change), then this
    // token's P (full tiles); then SiLU on the three.
    tile_regs_acquire();
    {
    DeviceZoneScopedN("CONV_MM");
    reconfig_data_format<SrcOrder::Reverse>(c_conv, c_lconst);
    matmul_init(c_conv, c_lconst);
    for (uint32_t s = 1; s <= 3; ++s) {
        for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
            matmul_tiles(c_conv, c_lconst, (s - 1) * CONV_TILES + ct, s * CONV_TILES + ct, ct);
        }
    }
    reconfig_full_operand<SrcOrder::Reverse>(c_p, c_lconst);
    matmul_init(c_p, c_lconst);
    for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
        matmul_tiles(c_p, c_lconst, ct, ct, ct);
    }
    }
    {
    DeviceZoneScopedN("CONV_SILU");
    silu_tile_init();
    for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
        SFPU_ROWS(silu_rows<G>, ct);
    }
    }
    SFPU_ROWS_INIT();
    SFPU_ROWS(prod3_rows<G>, 0);  // this quarter's q*q, k*k, q*k for the l2 norms
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, c_qkloc);
    pack_tile(1, c_qkloc);
    pack_tile(2, c_cv);
    for (uint32_t i = 3; i < 6; ++i) {
        pack_tile(i, c_tmp);
    }
    tile_regs_release();
    cb_push_back(c_qkloc, 2);
    cb_push_back(c_cv, 1);
    cb_push_back(c_tmp, 3);

    }
    // ---- C2: this quarter's |q|^2, |k|^2, q.k, as row sums into columns 0, 1, 16 of one tile (to the head's 4 cores) --
    {
    DeviceZoneScopedN("QK_PARTIAL");
    cb_wait_front(c_tmp, 3);
    cb_reserve_back(c_part, 1);
    mm_init(c_tmp, c_gconst);
    tile_regs_acquire();
    matmul_tiles(c_tmp, c_gconst, 0, G_ONES_COL, 0);
    matmul_tiles(c_tmp, c_gconst, 1, G_ONES_C1, 0);
    matmul_tiles(c_tmp, c_gconst, 2, G_ONES_C16, 0);
    pack_one(c_part);
    cb_push_back(c_part, 1);
    cb_pop_front(c_tmp, 3);
    }

    // ---- C3: the head's sums (the 4 cores' partials, in rows s * BP + b, added by a selector matmul), this head's a
    // and b out of the a/b tile, then every per-row scalar of the step on the SFPU (column 0 of the batch rows) ------
    {
    DeviceZoneScopedN("GATING");
    {
    DeviceZoneScopedN("QK_PARTIAL_WAIT");
    cb_wait_front(c_pfull, PFULL);
    }
    cb_reserve_back(c_scal2, 5);
    mm_init(c_gconst, c_pfull);
    tile_regs_acquire();
    for (uint32_t t = 0; t < PFULL; ++t) {
        matmul_tiles(c_gconst, c_pfull, G_SEL_GRP, t, 0);
    }
    matmul_tiles(c_p, c_gconst, 4, G_SEL_A, 3);  // same formats: no re-init
    matmul_tiles(c_p, c_gconst, 4, G_SEL_B, 4);
    SFPU_ROWS_INIT();
    SFPU_ROWS(gating_rows<G>, 0, dt_bias, neg_A, qk_scale2, qk_eps2, qk_eps);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < 5; ++i) {
        pack_tile(i, c_scal2);  // 0: d*rq  1: -d*rk  2: beta*rk  3: w = beta * qn.kn  4: d - 1
    }
    tile_regs_release();
    cb_push_back(c_scal2, 5);
    cb_pop_front(c_pfull, PFULL);
    }

    // ---- C4: this quarter's recurrence inputs d*q, -d*k, beta*k (row-scaled), swapped within the head by the writer;
    // then (beta*k)^T once per step from the head's 4 quarters: column b = beta_b k_b. With u holding only row b,
    // (beta*k)^T @ u is exactly sequence b's outer product. The reader takes w and d - 1 straight from c_scal2.
    {
    DeviceZoneScopedN("PREP");
    cb_wait_front(c_qkloc, 2);
    cb_wait_front(c_scal2, 5);
    cb_reserve_back(c_sc, 3);
    reconfig_data_format(c_qkloc, c_scal2);
    mul_bcast_cols_init(c_qkloc, c_scal2);
    tile_regs_acquire();
    mul_tiles_bcast_cols(c_qkloc, c_scal2, 0, 0, 0);  // d*q
    mul_tiles_bcast_cols(c_qkloc, c_scal2, 1, 1, 1);  // -d*k
    mul_tiles_bcast_cols(c_qkloc, c_scal2, 1, 2, 2);  // beta*k
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < 3; ++i) {
        pack_tile(i, c_sc);
    }
    tile_regs_release();
    cb_push_back(c_sc, 3);
    cb_pop_front(c_qkloc, 2);
    }
    {
    DeviceZoneScopedN("PREP_KB");
    {
    DeviceZoneScopedN("QK_SWAP_WAIT");
    cb_wait_front(c_qkfull, 3 * KT);  // [d*q 0..3][-d*k 0..3][beta*k 0..3]; read only, the reader pops it
    }
    cb_reserve_back(c_kb, KT);
    reconfig_data_format<SrcOrder::Reverse>(c_gconst, c_qkfull);
    matmul_init(c_gconst, c_qkfull, 1);  // transpose in1: I @ KB^T
    tile_regs_acquire();
    for (uint32_t i = 0; i < KT; ++i) {
        matmul_tiles(c_gconst, c_qkfull, G_ID, 2 * KT + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < KT; ++i) {
        pack_tile(i, c_kb);
    }
    tile_regs_release();
    cb_push_back(c_kb, KT);
    }

    // ---- C5: recurrence (row b = sequence b, its own state slot) -------------------------------------------
    // Per token r, two DST passes:
    //   pass   I @ v + sum_i X_i @ S_i, where the reader put -d*k at row r and d*q at row r + 16 of X_i, so one read of
    //          the state gives u = v - (d k)^T S (row r) and (d q)^T S (row r + 16)
    //   then   o = M @ pass with M = e_r (e_{r+16} + w e_r)^T: (d q)^T S + w u at row r, all other rows zero, packed
    //          with L1 accumulation into c_oblk, which so collects every sequence's output in its row; and the state
    //          increment diag(d - 1) @ S_i + (beta K)^T_i @ pass, packed onto the fp32 state in L1. Row r + 16 of the
    //          pass meets column r + 16 of (beta K)^T, which is zero (batch <= 16).
    { DeviceZoneScopedN("RECURRENCE");
    mm_init(c_qd, c_state);
    cb_wait_front(c_kb, KT);
    cb_reserve_back(c_oblk, 1);
    for (uint32_t r = 0; r < B; ++r) {
        const uint32_t s0 = r * KT;

        cb_wait_front(c_qd, KT);
        cb_wait_front(c_v, 1);
        cb_reserve_back(c_u, 1);
        tile_regs_acquire();
        matmul_tiles(c_gconst, c_v, G_ID, 0, 0);
        for (uint32_t i = 0; i < KT; ++i) {
            matmul_tiles(c_qd, c_state, i, s0 + i, 0);
        }
        pack_one(c_u);
        cb_push_back(c_u, 1);
        cb_pop_front(c_qd, KT);
        cb_pop_front(c_v, 1);

        cb_wait_front(c_u, 1);
        cb_wait_front(c_w, 1);
        cb_wait_front(c_d, 1);
        const bool last = r == B - 1;  // the last output goes first: the RMSNorm waits on it, not on the state
        if (last) {
            tile_regs_acquire();
            matmul_tiles(c_w, c_u, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(r > 0);
            pack_tile<true>(0, c_oblk, 0);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
            cb_push_back(c_oblk, 1);
        }
        tile_regs_acquire();
        if (!last) {
            matmul_tiles(c_w, c_u, 0, 0, 0);
        }
        for (uint32_t i = 0; i < KT; ++i) {
            matmul_tiles(c_d, c_state, 0, s0 + i, 1 + i);
            matmul_tiles(c_kb, c_u, i, 0, 1 + i);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (!last) {
            pack_reconfig_l1_acc(r > 0);  // the first sequence's output initialises c_oblk
            pack_tile<true>(0, c_oblk, 0);
        }
        pack_reconfig_l1_acc(1);
        for (uint32_t i = 0; i < KT; ++i) {
            pack_tile<true>(1 + i, c_state, s0 + i);
        }
        pack_reconfig_l1_acc(0);
        tile_regs_release();
        cb_pop_front(c_w, 1);
        cb_pop_front(c_d, 1);
        cb_pop_front(c_u, 1);
    }
    cb_pop_front(c_kb, KT);

    }
    // ---- C6: gated RMSNorm --------------------------------------------------------------------------
    { DeviceZoneScopedN("RMSNORM");
    // y = o * rstd * silu(z); the norm weight lives in the out_proj rows. o * silu(z) is computed while
    // the head's 4 partial sums of squares are being exchanged; rstd is applied last.
    {
    DeviceZoneScopedN("RMS_OBLK_WAIT");
    cb_wait_front(c_oblk, 1);
    }
    {
    // sum of squares of this core's 32 channels (row sums), to the head's 4 cores
    DeviceZoneScopedN("RMS_SQ");
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
    }

    {
    DeviceZoneScopedN("RMS_ZSILU");
    cb_reserve_back(c_tmp2, 1);
    tile_regs_acquire();
    copy_init(c_p);
    copy_tile(c_p, 3, 0);  // z
    copy_init(c_oblk);
    copy_tile(c_oblk, 0, 1);  // o
    silu_tile_init();
    SFPU_ROWS(silu_mul_rows<G>, 0);  // o * silu(z), fp32
    pack_one(c_tmp2);
    cb_push_back(c_tmp2, 1);
    cb_pop_front(c_p, P_TILES);
    cb_pop_front(c_oblk, 1);
    }

    {
    DeviceZoneScopedN("RMS_SS_WAIT");
    cb_wait_front(c_ssp, GROUP);
    }
    {
    DeviceZoneScopedN("RMS_RSTD_APPLY");
    cb_reserve_back(c_tmp3, 1);  // rstd
    copy_init(c_ssp);
    tile_regs_acquire();
    for (uint32_t i = 0; i < GROUP; ++i) {
        copy_tile(c_ssp, i, i);
    }
    SFPU_ROWS_INIT();
    SFPU_ROWS(rstd_rows<G>, 0, inv_dv, norm_eps);
    pack_one(c_tmp3);
    cb_push_back(c_tmp3, 1);
    cb_pop_front(c_ssp, GROUP);

    cb_wait_front(c_tmp2, 1);
    cb_wait_front(c_tmp3, 1);
    cb_reserve_back(c_ss, 1);  // y -> the writer (its second use of c_ss), for the all-gather
    reconfig_data_format(c_tmp2, c_tmp3);
    mul_bcast_cols_init(c_tmp2, c_tmp3);
    tile_regs_acquire();
    mul_tiles_bcast_cols(c_tmp2, c_tmp3, 0, 0, 0);
    pack_one(c_ss);
    cb_push_back(c_ss, 1);
    cb_pop_front(c_tmp2, 1);  // o * silu(z)
    cb_pop_front(c_tmp3, 1);
    }

    }
    // ---- C7: out_proj -----------------------------------------------------------------------------------
    // Core c computes output column tile c % 32 over the 1024 v channels of its grid half (cores 32h .. 32h + 31 with
    // h = c / 32): the half's y blocks are all-gathered into c_yg (16 x 32 tiles, slot k = core 32h + k, gathered by
    // the writer), and this core's W_o^T rows are those channels for its 32 output columns.
    {
    DeviceZoneScopedN("OUT_PROJ");
    cb_wait_front(c_yg, KY);
#ifdef WEIGHTS_IN_DRAM
    cb_wait_front(c_wout, KY);
#endif
    cb_reserve_back(c_out, 1);
    reconfig_full_operand<SrcOrder::Reverse>(c_yg, c_wout);
    matmul_init(c_yg, c_wout);
    tile_regs_acquire();
    for (uint32_t k = 0; k < KY; ++k) {
        matmul_tiles(c_yg, c_wout, k, k, 0);
    }
    pack_one(c_out);  // the writer sends face 1 - h to the other half's core of this tile, which sends its face h
    cb_push_back(c_out, 1);
    cb_pop_front(c_yg, KY);
#ifdef WEIGHTS_IN_DRAM
    cb_pop_front(c_wout, KY);
    cb_pop_front(c_lconst, N_LCONST);
#endif
    }

    // ---- C8: this half's + the other half's partial of the tile (face h is the one this core writes) ------
    {
    DeviceZoneScopedN("PAIR_SUM");
    cb_wait_front(c_out, 1);
    cb_wait_front(c_pp, 1);
    cb_reserve_back(c_tmp3, 1);
    reconfig_full_operand<SrcOrder::Regular>(c_out, c_pp);
    add_tiles_init(c_out, c_pp);
    tile_regs_acquire();
    add_tiles(c_out, c_pp, 0, 0, 0);
    pack_one(c_tmp3);
    cb_push_back(c_tmp3, 1);
    cb_pop_front(c_out, 1);
    cb_pop_front(c_pp, 1);
    }
}
