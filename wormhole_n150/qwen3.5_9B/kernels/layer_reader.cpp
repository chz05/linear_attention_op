// SPDX-License-Identifier: Apache-2.0
//
// Fused Gated DeltaNet layer -- reader (data movement).
//
// Core c owns v-head h, column half s. Everything it touches is local L1 except the tiny a/b
// projection tile:
//   P      : this core's slice of the input projection [q/4 | k/4 | v/2 | z/2] (L1 shard)
//   conv   : this core's causal-conv history, last 3 pre-conv inputs per sequence (L1 shard)
// Per 32-row block it
//   1. builds the 3 time-shifted copies of P's q/k/v tiles for the causal conv (rows from the
//      same sequence, or from the conv history for the first tokens),
//   2. fetches the a/b tile,
//   3. turns compute's prepared block (d*q, -d*k, beta*k, v, d-1, w) into the per-token
//      row / column / diagonal tiles of the recurrence.
// At the end it rolls the conv history forward. Only raw 32-bit copies happen here.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t c_qd = 0, c_nkd = 1, c_kb = 2, c_v = 3, c_d = 4, c_w = 5, c_id = 6;
constexpr uint32_t c_p = 10, c_const = 11, c_xs = 12, c_ab = 13, c_cv = 17, c_prep = 21, c_conv = 29;

constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t KT = 4, NCT = 2, P_TILES = 6, CONV_TILES = 4, N_CONST = 20;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

FORCE_INLINE uint32_t tidx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * FACE_WORDS + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE l1_ptr tile_ptr(uint32_t base, uint32_t tile) { return reinterpret_cast<l1_ptr>(base + tile * TILE_BYTES); }

// DMA from the hardware zero region instead of storing word by word from the RISC-V.
FORCE_INLINE void zero_tiles(uint32_t addr, uint32_t num_tiles) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < num_tiles * TILE_BYTES; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, addr + off, MEM_ZEROS_SIZE);
    }
}

// row sr of src -> row dr of dst
FORCE_INLINE void copy_row(l1_ptr dst, uint32_t dr, l1_ptr src, uint32_t sr) {
    l1_ptr d = dst + (dr >> 4) * 2 * FACE_WORDS + (dr & 15) * 16;
    l1_ptr s = src + (sr >> 4) * 2 * FACE_WORDS + (sr & 15) * 16;
    for (uint32_t e = 0; e < 16; ++e) {
        d[e] = s[e];
        d[FACE_WORDS + e] = s[FACE_WORDS + e];
    }
}

// row sr of src -> column 0 of dst
FORCE_INLINE void copy_row_to_col(l1_ptr dst, l1_ptr src, uint32_t sr) {
    for (uint32_t i = 0; i < 32; ++i) {
        dst[tidx(i, 0)] = src[tidx(sr, i)];
    }
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);  // sequences (state slots)
    constexpr auto ab_args = TensorAccessorArgs<1>();

    const uint32_t ab_addr = get_arg_val<uint32_t>(0);
    const uint32_t M = get_arg_val<uint32_t>(1);  // rows = B * T, sequence-major
    const uint32_t T = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = (M + 31) / 32;

    const auto ab_acc = TensorAccessor(ab_args, ab_addr);

    const uint32_t p_base = get_write_ptr(c_p);
    const uint32_t conv_base = get_write_ptr(c_conv);

    zero_tiles(get_write_ptr(c_qd), 2 * KT);
    zero_tiles(get_write_ptr(c_nkd), 2 * KT);
    zero_tiles(get_write_ptr(c_kb), 2 * KT);
    zero_tiles(get_write_ptr(c_v), 2 * NCT);
    zero_tiles(get_write_ptr(c_d), 2);
    zero_tiles(get_write_ptr(c_w), 2);
    zero_tiles(get_write_ptr(c_xs), 3 * CONV_TILES);
    noc_async_read_barrier();

    cb_reserve_back(c_id, 1);
    zero_tiles(get_write_ptr(c_id), 1);
    noc_async_read_barrier();
    {
        l1_ptr p = reinterpret_cast<l1_ptr>(get_write_ptr(c_id));
        for (uint32_t i = 0; i < 32; ++i) {
            p[tidx(i, i)] = 0x3f800000;
        }
    }
    cb_push_back(c_id, 1);

    // Per-core constants (conv tap diagonals, norm weight rows, head selector, ones column) live in
    // an L1-sharded tensor for the life of the layer; just mark them available to compute.
    cb_reserve_back(c_const, N_CONST);
    cb_push_back(c_const, N_CONST);

    for (uint32_t rb = 0; rb < num_blocks; ++rb) {
        const uint32_t R = (M - rb * 32) < 32 ? (M - rb * 32) : 32;

        // 1. time-shifted conv inputs: xs[s-1][ct] row r = x[t - s] of the same sequence
        cb_reserve_back(c_xs, 3 * CONV_TILES);
        {
            const uint32_t xs = get_write_ptr(c_xs);
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t gr = rb * 32 + r;
                const uint32_t b = gr / T;
                const uint32_t t = gr - b * T;
                for (uint32_t s = 1; s <= 3; ++s) {
                    for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
                        l1_ptr dst = tile_ptr(xs, (s - 1) * CONV_TILES + ct);
                        if (t >= s) {
                            const uint32_t sr = gr - s;
                            copy_row(dst, r, tile_ptr(p_base, (sr >> 5) * P_TILES + ct), sr & 31);
                        } else {
                            copy_row(dst, r, tile_ptr(conv_base, b * CONV_TILES + ct), 3 + t - s);
                        }
                    }
                }
            }
        }
        cb_push_back(c_xs, 3 * CONV_TILES);

        // 2. a/b projection tile pair for this block
        cb_reserve_back(c_ab, 2);
        {
            const uint32_t a = get_write_ptr(c_ab);
            noc_async_read_page(rb * 2, ab_acc, a);
            noc_async_read_page(rb * 2 + 1, ab_acc, a + TILE_BYTES);
            noc_async_read_barrier();
        }
        cb_push_back(c_ab, 2);

        // 3. per-token recurrence inputs
        cb_wait_front(c_prep, 3 * KT + 2);
        cb_wait_front(c_cv, NCT);
        const uint32_t prep = get_read_ptr(c_prep);
        const uint32_t cv = get_read_ptr(c_cv);
        for (uint32_t r = 0; r < R; ++r) {
            cb_reserve_back(c_nkd, KT);
            for (uint32_t i = 0; i < KT; ++i) {
                copy_row(tile_ptr(get_write_ptr(c_nkd), i), 0, tile_ptr(prep, KT + i), r);
            }
            cb_push_back(c_nkd, KT);

            cb_reserve_back(c_v, NCT);
            for (uint32_t j = 0; j < NCT; ++j) {
                copy_row(tile_ptr(get_write_ptr(c_v), j), 0, tile_ptr(cv, j), r);
            }
            cb_push_back(c_v, NCT);

            cb_reserve_back(c_qd, KT);
            for (uint32_t i = 0; i < KT; ++i) {
                copy_row(tile_ptr(get_write_ptr(c_qd), i), 0, tile_ptr(prep, i), r);
            }
            cb_push_back(c_qd, KT);

            cb_reserve_back(c_w, 1);
            tile_ptr(get_write_ptr(c_w), 0)[0] = tile_ptr(prep, 3 * KT + 1)[tidx(r, 0)];
            cb_push_back(c_w, 1);

            cb_reserve_back(c_d, 1);
            {
                l1_ptr p = tile_ptr(get_write_ptr(c_d), 0);
                const uint32_t dm1 = tile_ptr(prep, 3 * KT)[tidx(r, 0)];
                for (uint32_t i = 0; i < 32; ++i) {
                    p[tidx(i, i)] = dm1;
                }
            }
            cb_push_back(c_d, 1);

            cb_reserve_back(c_kb, KT);
            for (uint32_t i = 0; i < KT; ++i) {
                copy_row_to_col(tile_ptr(get_write_ptr(c_kb), i), tile_ptr(prep, 2 * KT + i), r);
            }
            cb_push_back(c_kb, KT);
        }
        cb_pop_front(c_prep, 3 * KT + 2);
        cb_pop_front(c_cv, NCT);
    }

    // Roll the conv history: new row k holds the input at time T - 3 + k.
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t k = 0; k < 3; ++k) {
            for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
                l1_ptr dst = tile_ptr(conv_base, b * CONV_TILES + ct);
                if (T + k >= 3) {
                    const uint32_t sr = b * T + T + k - 3;
                    copy_row(dst, k, tile_ptr(p_base, (sr >> 5) * P_TILES + ct), sr & 31);
                } else {
                    copy_row(dst, k, dst, T + k);  // older history shifts up; T + k > k, not yet overwritten
                }
            }
        }
    }
}
