// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5-0.8B Gated DeltaNet layer, all weights in L1 -- reader (data movement).
//
// Decode only: one launch = one token for each of the B sequences, row b = sequence b.
// Core c owns head h = c / 4 and column quarter s = c % 4. Per launch it
//   1. turns the layer input rows (multicast to every core by core 0's writer) into tiles for the
//      in-kernel in_proj,
//   2. builds the time-shifted conv inputs from the conv state (the last 3 inputs of each sequence),
//      then rolls the conv state forward with this token's projection,
//   3. turns compute's prepared block (d*q, -d*k, beta*k, v, d-1, w) into per-token recurrence tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"  // zones compile to nothing unless profiling

constexpr uint32_t c_qd = 0, c_nkd = 1, c_kb = 2, c_v = 3, c_d = 4, c_w = 5;
constexpr uint32_t c_x = 10, c_win = 11, c_wout = 12, c_lconst = 13, c_p = 15, c_xs = 16, c_cv = 19, c_prep = 23,
                   c_conv = 31;

constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t KT = 4, KX = 32, X_CHUNK = 4, CONV_TILES = 3, P_TILES = 5, N_OUT = 32, N_LCONST = 13,
                   N_CORES = 64;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

FORCE_INLINE uint32_t tidx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * FACE_WORDS + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE l1_ptr tile_ptr(uint32_t base, uint32_t tile) { return reinterpret_cast<l1_ptr>(base + tile * TILE_BYTES); }

FORCE_INLINE void zero_tiles(uint32_t addr, uint32_t num_tiles) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < num_tiles * TILE_BYTES; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, addr + off, MEM_ZEROS_SIZE);
    }
}

FORCE_INLINE void copy_row(l1_ptr dst, uint32_t dr, l1_ptr src, uint32_t sr) {
    l1_ptr d = dst + (dr >> 4) * 2 * FACE_WORDS + (dr & 15) * 16;
    l1_ptr s = src + (sr >> 4) * 2 * FACE_WORDS + (sr & 15) * 16;
    for (uint32_t e = 0; e < 16; ++e) {
        d[e] = s[e];
        d[FACE_WORDS + e] = s[FACE_WORDS + e];
    }
}

FORCE_INLINE void copy_row_to_col(l1_ptr dst, l1_ptr src, uint32_t sr) {
    for (uint32_t i = 0; i < 32; ++i) {
        dst[tidx(i, 0)] = src[tidx(sr, i)];
    }
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    const uint32_t sem_x = get_semaphore(get_compile_time_arg_val(1));
    const uint32_t conv_base = get_write_ptr(c_conv);
#ifdef WEIGHTS_IN_DRAM
    // Comparison mode: the same per-core weight slices, but stored in DRAM and streamed every step.
    constexpr uint32_t W_TILE_BYTES = get_compile_time_arg_val(2);
    constexpr auto win_args = TensorAccessorArgs<3>();
    constexpr auto wout_args = TensorAccessorArgs<win_args.next_compile_time_args_offset()>();
    constexpr auto lc_args = TensorAccessorArgs<wout_args.next_compile_time_args_offset()>();
    const auto win_acc = TensorAccessor(win_args, get_arg_val<uint32_t>(1));
    const auto wout_acc = TensorAccessor(wout_args, get_arg_val<uint32_t>(2));
    const auto lc_acc = TensorAccessor(lc_args, get_arg_val<uint32_t>(3));
    const uint32_t core = get_arg_val<uint32_t>(4);
#endif

    // 1. layer input: B rows of 1024 fp32 (one per sequence) multicast by core 0 into the c_prep buffer
    //    (free until compute's C4, long after x is consumed); only those rows of each x tile are written,
    //    the rest of the zeroed tile stays 0.
    zero_tiles(get_write_ptr(c_x), 2 * X_CHUNK);
    noc_async_read_barrier();
    noc_semaphore_wait(reinterpret_cast<l1_ptr>(sem_x), 1);
    const uint32_t xrow = get_write_ptr(c_prep);
    {
    DeviceZoneScopedN("X_STREAM");
    for (uint32_t k = 0; k < KX; k += X_CHUNK) {
        cb_reserve_back(c_x, X_CHUNK);
        const uint32_t a = get_write_ptr(c_x);
        // local DMA (64 B per face row) rather than RISC-V word copies, which are several cycles per access
        for (uint32_t i = 0; i < X_CHUNK; ++i) {
            for (uint32_t r = 0; r < B; ++r) {
                const uint32_t src = xrow + (r * KX + k + i) * 128;
                const uint32_t dst = a + i * TILE_BYTES + ((r >> 4) * 2 * FACE_WORDS + (r & 15) * 16) * 4;
                noc_async_read(get_noc_addr(src), dst, 64);
                noc_async_read(get_noc_addr(src + 64), dst + FACE_WORDS * 4, 64);
            }
        }
#ifdef WEIGHTS_IN_DRAM
        // in_proj rows k..k+3 of this core's 5 column tiles (w_in is [1024, 64 * 160], core-major columns)
        cb_reserve_back(c_win, X_CHUNK * P_TILES);
        const uint32_t w = get_write_ptr(c_win);
        for (uint32_t kk = 0; kk < X_CHUNK; ++kk) {
            for (uint32_t j = 0; j < P_TILES; ++j) {
                noc_async_read_page(
                    (k + kk) * N_CORES * P_TILES + core * P_TILES + j, win_acc, w + (kk * P_TILES + j) * W_TILE_BYTES);
            }
        }
#endif
        noc_async_read_barrier();
        cb_push_back(c_x, X_CHUNK);
#ifdef WEIGHTS_IN_DRAM
        cb_push_back(c_win, X_CHUNK * P_TILES);
#endif
    }
    }

    // buffers that are only written in row 0 / col 0 / diagonal later on
    {
    DeviceZoneScopedN("ZERO");
    zero_tiles(get_write_ptr(c_qd), KT);
    zero_tiles(get_write_ptr(c_nkd), KT);
    zero_tiles(get_write_ptr(c_kb), KT);
    zero_tiles(get_write_ptr(c_v), 1);
    zero_tiles(get_write_ptr(c_d), 1);
    zero_tiles(get_write_ptr(c_w), 1);
    zero_tiles(get_write_ptr(c_xs), 3 * CONV_TILES);
    noc_async_read_barrier();
    }
#ifdef WEIGHTS_IN_DRAM
    cb_reserve_back(c_lconst, N_LCONST);
    for (uint32_t i = 0; i < N_LCONST; ++i) {
        noc_async_read_page(core * N_LCONST + i, lc_acc, get_write_ptr(c_lconst) + i * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(c_lconst, N_LCONST);
#endif

    // 2. conv inputs: xs[s-1][ct] row b = the input s steps back of sequence b, from the conv state
    //    (one tile per sequence, row ct * 3 + k = input k - 3 steps back; P tiles 0..2 = q, k, v quarters)
    cb_wait_front(c_p, 5);  // read only; compute pops it after the norm
    const uint32_t p_base = get_read_ptr(c_p);
    cb_reserve_back(c_xs, 3 * CONV_TILES);
    {
        const uint32_t xs = get_write_ptr(c_xs);
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 1; s <= 3; ++s) {
                for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
                    copy_row(tile_ptr(xs, (s - 1) * CONV_TILES + ct), b, tile_ptr(conv_base, b), ct * 3 + 3 - s);
                }
            }
        }
    }
    cb_push_back(c_xs, 3 * CONV_TILES);

    // roll the conv state: drop the oldest input, append this token's
    for (uint32_t b = 0; b < B; ++b) {
        l1_ptr st = tile_ptr(conv_base, b);
        for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
            copy_row(st, ct * 3, st, ct * 3 + 1);
            copy_row(st, ct * 3 + 1, st, ct * 3 + 2);
            copy_row(st, ct * 3 + 2, tile_ptr(p_base, ct), b);
        }
    }

#ifdef WEIGHTS_IN_DRAM
    // out_proj rows of this core's 32 channels, fetched while compute works on the norms
    cb_reserve_back(c_wout, N_OUT);
    for (uint32_t j = 0; j < N_OUT; ++j) {
        noc_async_read_page(core * N_OUT + j, wout_acc, get_write_ptr(c_wout) + j * W_TILE_BYTES);
    }
    noc_async_read_barrier();
    cb_push_back(c_wout, N_OUT);
#endif

    // 3. per-token recurrence inputs
    cb_wait_front(c_prep, 3 * KT + 2);
    cb_wait_front(c_cv, 1);
    const uint32_t prep = get_read_ptr(c_prep);
    l1_ptr cv = tile_ptr(get_read_ptr(c_cv), 0);
    for (uint32_t r = 0; r < B; ++r) {
        cb_reserve_back(c_nkd, KT);
        for (uint32_t i = 0; i < KT; ++i) {
            copy_row(tile_ptr(get_write_ptr(c_nkd), i), 0, tile_ptr(prep, KT + i), r);
        }
        cb_push_back(c_nkd, KT);

        cb_reserve_back(c_v, 1);
        copy_row(tile_ptr(get_write_ptr(c_v), 0), 0, cv, r);
        cb_push_back(c_v, 1);

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
    cb_pop_front(c_cv, 1);
}
