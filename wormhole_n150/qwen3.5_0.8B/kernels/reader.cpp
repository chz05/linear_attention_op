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
//   3. turns compute's prepared block (d*q, -d*k, v, d-1, w) into per-token recurrence tiles, each sequence
//      in its own row.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"  // zones compile to nothing unless profiling

constexpr uint32_t c_qd = 0, c_nkd = 1, c_kb = 2, c_v = 3, c_d = 4, c_w = 5;
constexpr uint32_t c_x = 10, c_win = 11, c_wout = 12, c_lconst = 13, c_p = 15, c_xs = 16, c_cv = 19, c_prep = 23,
                   c_conv = 31;

constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t KT = 4, KX = 32, X_CHUNK = 4, CONV_TILES = 3, P_TILES = 5, N_OUT = 32, N_LCONST = 12,
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

// row sr of the tile at src -> row dr of the tile at dst, by local DMA (two 64 B face rows)
FORCE_INLINE void dma_row(uint32_t dst, uint32_t dr, uint32_t src, uint32_t sr) {
    const uint32_t d = dst + ((dr >> 4) * 2 * FACE_WORDS + (dr & 15) * 16) * 4;
    const uint32_t s = src + ((sr >> 4) * 2 * FACE_WORDS + (sr & 15) * 16) * 4;
    noc_async_read(get_noc_addr(s), d, 64);
    noc_async_read(get_noc_addr(s + FACE_WORDS * 4), d + FACE_WORDS * 4, 64);
}

FORCE_INLINE void dma_zero_row(uint64_t zeros, uint32_t dst, uint32_t dr) {
    const uint32_t d = dst + ((dr >> 4) * 2 * FACE_WORDS + (dr & 15) * 16) * 4;
    noc_async_read(zeros, d, 64);
    noc_async_read(zeros, d + FACE_WORDS * 4, 64);
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    const uint32_t sem_x = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t BP = get_compile_time_arg_val(2);  // batch rounded up to a power of 2 (<= 16)
    constexpr uint32_t BLOCK = BP * 64;                     // BP rows of one face
    const uint32_t conv_base = get_write_ptr(c_conv);
#ifdef WEIGHTS_IN_DRAM
    // Comparison mode: the same per-core weight slices, but stored in DRAM and streamed every step.
    constexpr uint32_t W_TILE_BYTES = get_compile_time_arg_val(3);
    constexpr auto win_args = TensorAccessorArgs<4>();
    constexpr auto wout_args = TensorAccessorArgs<win_args.next_compile_time_args_offset()>();
    constexpr auto lc_args = TensorAccessorArgs<wout_args.next_compile_time_args_offset()>();
    const auto win_acc = TensorAccessor(win_args, get_arg_val<uint32_t>(1));
    const auto wout_acc = TensorAccessor(wout_args, get_arg_val<uint32_t>(2));
    const auto lc_acc = TensorAccessor(lc_args, get_arg_val<uint32_t>(3));
    const uint32_t core = get_arg_val<uint32_t>(4);
#endif

    // 1. layer input: BP rows of 1024 fp32 (one per sequence) multicast by core 0 into the c_prep buffer
    //    (free until compute's C4, long after x is consumed), in the face-major layout, so each x tile is
    //    two DMAs (rows 0..BP-1 of its left and right face); the rest of the zeroed tile stays 0.
    zero_tiles(get_write_ptr(c_x), 2 * X_CHUNK);
    noc_async_read_barrier();
    noc_semaphore_wait(reinterpret_cast<l1_ptr>(sem_x), 1);
    const uint32_t xrow = get_write_ptr(c_prep);
    {
    DeviceZoneScopedN("X_STREAM");
    for (uint32_t k = 0; k < KX; k += X_CHUNK) {
        cb_reserve_back(c_x, X_CHUNK);
        const uint32_t a = get_write_ptr(c_x);
        for (uint32_t i = 0; i < X_CHUNK; ++i) {
            const uint32_t src = xrow + (k + i) * 2 * BLOCK;
            noc_async_read(get_noc_addr(src), a + i * TILE_BYTES, BLOCK);
            noc_async_read(get_noc_addr(src + BLOCK), a + i * TILE_BYTES + FACE_WORDS * 4, BLOCK);
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

    // 2. conv inputs. In decode every shifted input comes from the conv state (one tile per sequence,
    //    row ct * 3 + k = the input 3 - k steps back), so they are built right away, by DMA, while
    //    compute is still on in_proj: xs[s-1][ct] row b = input s steps back of sequence b.
    cb_reserve_back(c_xs, 3 * CONV_TILES);
    const uint32_t xs = get_write_ptr(c_xs);
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t s = 1; s <= 3; ++s) {
            for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
                dma_row(xs + ((s - 1) * CONV_TILES + ct) * TILE_BYTES, b, conv_base + b * TILE_BYTES, ct * 3 + 3 - s);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(c_xs, 3 * CONV_TILES);

    // roll the conv state: inputs 2 and 1 steps back are already in xs, the newest is this token's P
    cb_wait_front(c_p, 5);  // read only; compute pops it after the norm
    const uint32_t p_base = get_read_ptr(c_p);
    for (uint32_t b = 0; b < B; ++b) {
        const uint32_t st = conv_base + b * TILE_BYTES;
        for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
            dma_row(st, ct * 3, xs + (CONV_TILES + ct) * TILE_BYTES, b);
            dma_row(st, ct * 3 + 1, xs + ct * TILE_BYTES, b);
            dma_row(st, ct * 3 + 2, p_base + ct * TILE_BYTES, b);
        }
    }
    noc_async_read_barrier();

#ifdef WEIGHTS_IN_DRAM
    // out_proj rows of this core's 32 channels, fetched while compute works on the norms
    cb_reserve_back(c_wout, N_OUT);
    for (uint32_t j = 0; j < N_OUT; ++j) {
        noc_async_read_page(core * N_OUT + j, wout_acc, get_write_ptr(c_wout) + j * W_TILE_BYTES);
    }
    noc_async_read_barrier();
    cb_push_back(c_wout, N_OUT);
#endif

    // 3. per-token recurrence inputs. Sequence r keeps its own row r (no moving to row 0), so each input is
    //    two 64 B DMAs per tile plus clearing the previous sequence's row in the reused slot. beta*k is
    //    transposed once per step by compute, so its per-token column form is not built here.
    cb_wait_front(c_prep, 3 * KT + 2);
    cb_wait_front(c_cv, 1);
    const uint32_t prep = get_read_ptr(c_prep);
    const uint32_t cv = get_read_ptr(c_cv);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t r = 0; r < B; ++r) {
        cb_reserve_back(c_nkd, KT);
        cb_reserve_back(c_v, 1);
        cb_reserve_back(c_qd, KT);
        const uint32_t nkd = get_write_ptr(c_nkd), v = get_write_ptr(c_v), qd = get_write_ptr(c_qd);
        for (uint32_t i = 0; i < KT; ++i) {
            if (r > 0) {
                dma_zero_row(zeros, nkd + i * TILE_BYTES, r - 1);
                dma_zero_row(zeros, qd + i * TILE_BYTES, r - 1);
            }
            dma_row(nkd + i * TILE_BYTES, r, prep + (KT + i) * TILE_BYTES, r);
            dma_row(qd + i * TILE_BYTES, r, prep + i * TILE_BYTES, r);
        }
        if (r > 0) {
            dma_zero_row(zeros, v, r - 1);
        }
        dma_row(v, r, cv, r);
        noc_async_read_barrier();
        cb_push_back(c_nkd, KT);
        cb_push_back(c_v, 1);
        cb_push_back(c_qd, KT);

        cb_reserve_back(c_w, 1);  // w_r at [r, r]: W @ u picks row r of u
        {
            l1_ptr p = tile_ptr(get_write_ptr(c_w), 0);
            if (r > 0) {
                p[tidx(r - 1, r - 1)] = 0;
            }
            p[tidx(r, r)] = tile_ptr(prep, 3 * KT + 1)[tidx(r, 0)];
        }
        cb_push_back(c_w, 1);

        cb_reserve_back(c_d, 1);  // (d_r - 1) I
        {
            l1_ptr p = tile_ptr(get_write_ptr(c_d), 0);
            const uint32_t dm1 = tile_ptr(prep, 3 * KT)[tidx(r, 0)];
            for (uint32_t i = 0; i < 32; ++i) {
                p[tidx(i, i)] = dm1;
            }
        }
        cb_push_back(c_d, 1);
    }
    cb_pop_front(c_prep, 3 * KT + 2);
    cb_pop_front(c_cv, 1);
}
