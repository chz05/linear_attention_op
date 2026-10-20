// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet recurrent step -- reader (data movement).
//
// For every 32-token block it pulls the token-packed inputs of each work unit from DRAM
// into a local staging area, then, token by token, builds the small "selector" tiles the
// compute kernel consumes:
//   c_qd / c_nkd : row 0 = token vector (row form), rest zero
//   c_kb         : col 0 = token vector (column form), rest zero
//   c_v          : row 0 = v slice of this unit
//   c_d          : (d - 1) on the diagonal, d = exp(g)
//   c_w          : [0,0] = beta * (q . k)
// Only raw 32-bit copies are done here, the RISC-V never does float math.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t c_qd = 0;
constexpr uint32_t c_nkd = 1;
constexpr uint32_t c_kb = 2;
constexpr uint32_t c_v = 3;
constexpr uint32_t c_d = 4;
constexpr uint32_t c_w = 5;
constexpr uint32_t c_id = 6;
constexpr uint32_t c_stage = 20;

constexpr uint32_t TILE_WORDS = 1024;  // fp32 32x32
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

// element (r, c) of a 32x32 tile in face layout
FORCE_INLINE uint32_t tidx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * FACE_WORDS + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE void zero_cb(uint32_t cb, uint32_t num_tiles) {
    l1_ptr p = reinterpret_cast<l1_ptr>(get_write_ptr(cb));
    for (uint32_t i = 0; i < num_tiles * TILE_WORDS; ++i) {
        p[i] = 0;
    }
}

// row t of every src tile -> row 0 of the matching dst tile
FORCE_INLINE void copy_rows(l1_ptr dst, l1_ptr src, uint32_t num_tiles, uint32_t t) {
    const uint32_t off = (t >> 4) * 2 * FACE_WORDS + (t & 15) * 16;
    for (uint32_t n = 0; n < num_tiles; ++n) {
        l1_ptr d = dst + n * TILE_WORDS;
        l1_ptr s = src + n * TILE_WORDS + off;
        for (uint32_t e = 0; e < 16; ++e) {
            d[e] = s[e];
            d[FACE_WORDS + e] = s[FACE_WORDS + e];
        }
    }
}

// row t of every src tile -> col 0 of the matching dst tile (a transpose of one row)
FORCE_INLINE void copy_row_to_col(l1_ptr dst, l1_ptr src, uint32_t num_tiles, uint32_t t) {
    for (uint32_t n = 0; n < num_tiles; ++n) {
        l1_ptr d = dst + n * TILE_WORDS;
        l1_ptr s = src + n * TILE_WORDS;
        for (uint32_t i = 0; i < 32; ++i) {
            d[tidx(i, 0)] = s[tidx(t, i)];
        }
    }
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);      // dk / 32
    constexpr uint32_t NCT = get_compile_time_arg_val(1);     // v column tiles per unit
    constexpr uint32_t NSPLIT = get_compile_time_arg_val(2);  // column splits per head
    constexpr uint32_t U = get_compile_time_arg_val(3);       // work units on this core
    constexpr uint32_t VT = NCT * NSPLIT;                     // dv / 32

    constexpr auto qd_args = TensorAccessorArgs<4>();
    constexpr auto nkd_args = TensorAccessorArgs<qd_args.next_compile_time_args_offset()>();
    constexpr auto kb_args = TensorAccessorArgs<nkd_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<kb_args.next_compile_time_args_offset()>();
    constexpr auto sc_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t qd_addr = get_arg_val<uint32_t>(0);
    const uint32_t nkd_addr = get_arg_val<uint32_t>(1);
    const uint32_t kb_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t sc_addr = get_arg_val<uint32_t>(4);
    const uint32_t unit_start = get_arg_val<uint32_t>(5);
    const uint32_t T = get_arg_val<uint32_t>(6);
    const uint32_t Tb = (T + 31) / 32;

    const auto qd_acc = TensorAccessor(qd_args, qd_addr);
    const auto nkd_acc = TensorAccessor(nkd_args, nkd_addr);
    const auto kb_acc = TensorAccessor(kb_args, kb_addr);
    const auto v_acc = TensorAccessor(v_args, v_addr);
    const auto sc_acc = TensorAccessor(sc_args, sc_addr);

    // Selector tiles are only ever written in row 0 / col 0 / diagonal, so zero them once.
    zero_cb(c_qd, 2 * KT);
    zero_cb(c_nkd, 2 * KT);
    zero_cb(c_kb, 2 * KT);
    zero_cb(c_v, 2 * NCT);
    zero_cb(c_d, 2);
    zero_cb(c_w, 2);

    // Identity tile, used to load v into DST through the matmul unit.
    cb_reserve_back(c_id, 1);
    zero_cb(c_id, 1);
    {
        l1_ptr p = reinterpret_cast<l1_ptr>(get_write_ptr(c_id));
        for (uint32_t i = 0; i < 32; ++i) {
            p[tidx(i, i)] = 0x3f800000;  // 1.0f
        }
    }
    cb_push_back(c_id, 1);

    // Staging layout per unit: [QD x KT][NKD x KT][KB x KT][V x NCT][SC x 1]
    constexpr uint32_t STAGE_TILES = 3 * KT + NCT + 1;
    const uint32_t stage_base = get_write_ptr(c_stage);

    for (uint32_t b = 0; b < Tb; ++b) {
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t unit = unit_start + u;
            const uint32_t h = unit / NSPLIT;
            const uint32_t c = unit % NSPLIT;
            uint32_t l1 = stage_base + u * STAGE_TILES * TILE_BYTES;
            const uint32_t k_row = (h * Tb + b) * KT;
            for (uint32_t i = 0; i < KT; ++i) {
                noc_async_read_page(k_row + i, qd_acc, l1 + i * TILE_BYTES);
                noc_async_read_page(k_row + i, nkd_acc, l1 + (KT + i) * TILE_BYTES);
                noc_async_read_page(k_row + i, kb_acc, l1 + (2 * KT + i) * TILE_BYTES);
            }
            const uint32_t v_row = (h * Tb + b) * VT + c * NCT;
            for (uint32_t j = 0; j < NCT; ++j) {
                noc_async_read_page(v_row + j, v_acc, l1 + (3 * KT + j) * TILE_BYTES);
            }
            noc_async_read_page(h * Tb + b, sc_acc, l1 + (3 * KT + NCT) * TILE_BYTES);
        }
        noc_async_read_barrier();

        const uint32_t tcount = (T - b * 32) < 32 ? (T - b * 32) : 32;
        for (uint32_t t = 0; t < tcount; ++t) {
            for (uint32_t u = 0; u < U; ++u) {
                l1_ptr stage = reinterpret_cast<l1_ptr>(stage_base + u * STAGE_TILES * TILE_BYTES);
                l1_ptr s_qd = stage;
                l1_ptr s_nkd = stage + KT * TILE_WORDS;
                l1_ptr s_kb = stage + 2 * KT * TILE_WORDS;
                l1_ptr s_v = stage + 3 * KT * TILE_WORDS;
                l1_ptr s_sc = stage + (3 * KT + NCT) * TILE_WORDS;

                // Push order matches the order compute consumes them.
                cb_reserve_back(c_nkd, KT);
                copy_rows(reinterpret_cast<l1_ptr>(get_write_ptr(c_nkd)), s_nkd, KT, t);
                cb_push_back(c_nkd, KT);

                cb_reserve_back(c_v, NCT);
                copy_rows(reinterpret_cast<l1_ptr>(get_write_ptr(c_v)), s_v, NCT, t);
                cb_push_back(c_v, NCT);

                cb_reserve_back(c_qd, KT);
                copy_rows(reinterpret_cast<l1_ptr>(get_write_ptr(c_qd)), s_qd, KT, t);
                cb_push_back(c_qd, KT);

                cb_reserve_back(c_w, 1);
                reinterpret_cast<l1_ptr>(get_write_ptr(c_w))[0] = s_sc[tidx(t, 1)];
                cb_push_back(c_w, 1);

                cb_reserve_back(c_d, 1);
                {
                    l1_ptr p = reinterpret_cast<l1_ptr>(get_write_ptr(c_d));
                    const uint32_t d = s_sc[tidx(t, 0)];
                    for (uint32_t i = 0; i < 32; ++i) {
                        p[tidx(i, i)] = d;
                    }
                }
                cb_push_back(c_d, 1);

                cb_reserve_back(c_kb, KT);
                copy_row_to_col(reinterpret_cast<l1_ptr>(get_write_ptr(c_kb)), s_kb, KT, t);
                cb_push_back(c_kb, KT);
            }
        }
    }
}
