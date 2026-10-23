// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5-0.8B Gated DeltaNet layer, all weights in L1 -- reader (data movement).
//
// Decode only: one launch = one token for each of the B sequences, row b = sequence b.
// Core c owns head h = c / 4 and column quarter s = c % 4. Per launch it
//   1. hands in_proj the layer input piece by piece as core 0's multicast lands (in the memory c_x shares
//      with c_prep, read in place),
//   2. rolls the conv state forward once compute's conv has read it, then tells this half's y gather
//      coordinator that c_prep is free,
//   3. turns the head's d*q and -d*k (gathered by the writer), v, w and d - 1 into per-token recurrence tiles,
//      each sequence in its own rows.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"  // zones compile to nothing unless profiling

constexpr uint32_t c_qd = 0, c_kb = 2, c_v = 3, c_d = 4, c_w = 5;
constexpr uint32_t c_x = 10, c_win = 11, c_wout = 12, c_lconst = 13, c_p = 15, c_qkfull = 18, c_cv = 19, c_yg = 21,
                   c_scal2 = 22, c_conv = 31;

constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t KT = 4, KX = 32, X_CHUNK = 4, CONV_TILES = 3, P_TILES = 5, KY = 32, N_LCONST = 12,
                   N_CORES = 64, N_HALF = 32;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

FORCE_INLINE uint32_t tidx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * FACE_WORDS + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE l1_ptr tile_ptr(uint32_t base, uint32_t tile) { return reinterpret_cast<l1_ptr>(base + tile * TILE_BYTES); }

FORCE_INLINE void zero_tiles(uint32_t addr, uint32_t num_tiles, uint32_t tile_bytes = TILE_BYTES) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < num_tiles * tile_bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, addr + off, MEM_ZEROS_SIZE);
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
    constexpr uint32_t BP = get_compile_time_arg_val(0);  // batch rounded up to a power of 2 (<= 16)
    const uint32_t sem_x = get_semaphore(get_compile_time_arg_val(1));
    const uint32_t sem_ready = get_semaphore(get_compile_time_arg_val(2));
    constexpr uint32_t BLOCK = BP * 64;                     // BP rows of one face
    const uint32_t conv_base = get_write_ptr(c_conv);
    // args 0: core; 1: batch (1..BP); 2..5: this half of the grid (first core, last core; the first is its y gather
    // coordinator)
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t B = get_arg_val<uint32_t>(1);
#ifdef WEIGHTS_IN_DRAM
    // Comparison mode: the same per-core weight slices, but stored in DRAM and streamed every step.
    constexpr uint32_t W_TILE_BYTES = get_compile_time_arg_val(3);
    constexpr auto win_args = TensorAccessorArgs<4>();
    constexpr auto wout_args = TensorAccessorArgs<win_args.next_compile_time_args_offset()>();
    constexpr auto lc_args = TensorAccessorArgs<wout_args.next_compile_time_args_offset()>();
    const auto win_acc = TensorAccessor(win_args, get_arg_val<uint32_t>(6));
    const auto wout_acc = TensorAccessor(wout_args, get_arg_val<uint32_t>(7));
    const auto lc_acc = TensorAccessor(lc_args, get_arg_val<uint32_t>(8));
#endif

    // 1. layer input: BP rows of 1024 fp32 (one per sequence), multicast by core 0 into the memory c_x shares with
    //    c_prep, in the face-major layout: input tile k (columns 32k .. 32k + 31, BP rows) is the two adjacent BP-row
    //    face blocks at 2k, i.e. already a BP x 32 tile, so in_proj reads it in place. Core 0 multicasts it in X_MCASTS
    //    pieces; sem_x = pieces landed.
    constexpr uint32_t X_MCASTS = BP < 8 ? BP : 8, X_MCAST_BYTES = TILE_BYTES * (BP / X_MCASTS);
    {
    DeviceZoneScopedN("X_STREAM");
    for (uint32_t k = 0; k < KX; k += X_CHUNK) {
        {
        DeviceZoneScopedN("X_MCAST_WAIT");
        const uint32_t end = (k + X_CHUNK) * 2 * BLOCK;  // bytes of input this chunk needs
        noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_x), (end + X_MCAST_BYTES - 1) / X_MCAST_BYTES);
        }
        cb_reserve_back(c_x, X_CHUNK);
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
        noc_async_read_barrier();
        cb_push_back(c_win, X_CHUNK * P_TILES);
#endif
        cb_push_back(c_x, X_CHUNK);
    }
    }
#ifdef WEIGHTS_IN_DRAM
    cb_reserve_back(c_lconst, N_LCONST);
    for (uint32_t i = 0; i < N_LCONST; ++i) {
        noc_async_read_page(core * N_LCONST + i, lc_acc, get_write_ptr(c_lconst) + i * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(c_lconst, N_LCONST);
#endif

    // 2. roll the conv state (BP x 32 tiles, tile 3 (s - 1) + ct = input s steps back), once compute's conv has read
    //    it (c_cv is pushed after its matmuls): 3 steps back <- 2, 2 <- 1, 1 <- this token's P. Each step moves 3 whole
    //    tiles, so a barrier between steps keeps the in-place shift ordered.
    {
    DeviceZoneScopedN("CONV_ROLL");
    {
    DeviceZoneScopedN("P_SEEN");
    cb_wait_front(c_p, 5);   // read only; compute pops it after the norm
    }
    {
    DeviceZoneScopedN("CV_SEEN");
    cb_wait_front(c_cv, 1);  // read only (popped after the recurrence inputs)
    }
    constexpr uint32_t STATE_TILE = 2 * BLOCK, SHIFT = CONV_TILES * STATE_TILE;
    noc_async_read(get_noc_addr(conv_base + SHIFT), conv_base + 2 * SHIFT, SHIFT);
    noc_async_read_barrier();
    noc_async_read(get_noc_addr(conv_base), conv_base + SHIFT, SHIFT);
    noc_async_read_barrier();
    const uint32_t p_base = get_read_ptr(c_p);
    for (uint32_t ct = 0; ct < CONV_TILES; ++ct) {
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_read(get_noc_addr(p_base + ct * TILE_BYTES + h * FACE_WORDS * 4), conv_base + ct * STATE_TILE + h * BLOCK, BLOCK);
        }
    }
    noc_async_read_barrier();
    }
    // in_proj is done (c_p exists), so the input in c_prep's memory is no longer read: tell this half's y gather
    // coordinator, which multicasts the gathered y there (c_yg) once the whole half is ready
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), sem_ready), 1);

    // buffers that are only written in row r / col 0 / diagonal later on
    {
    DeviceZoneScopedN("ZERO");
    zero_tiles(get_write_ptr(c_qd), KT);
    zero_tiles(get_write_ptr(c_v), 1);
    zero_tiles(get_write_ptr(c_d), 1);
    zero_tiles(get_write_ptr(c_w), 1);
    noc_async_read_barrier();
    }

#ifdef WEIGHTS_IN_DRAM
    // out_proj rows of this core's 32 channels, fetched while compute works on the norms
    cb_reserve_back(c_wout, KY);
    for (uint32_t k = 0; k < KY; ++k) {
        noc_async_read_page(core * KY + k, wout_acc, get_write_ptr(c_wout) + k * W_TILE_BYTES);
    }
    noc_async_read_barrier();
    cb_push_back(c_wout, KY);
#endif

    // 3. per-token recurrence inputs. Sequence r keeps its own rows: X_i (c_qd) gets -d*k at row r and d*q at row
    //    r + 16 (two 64 B DMAs each, plus clearing the previous sequence's rows), v gets row r, M (c_w) gets 1 at
    //    [r, r + 16] and w_r at [r, r], and (d_r - 1) I. beta*k is transposed once per step by compute.
    {
    DeviceZoneScopedN("PREP_WAIT");
    cb_wait_front(c_qkfull, 3 * KT);  // [d*q 0..3][-d*k 0..3][beta*k 0..3] of the whole head
    cb_wait_front(c_cv, 1);
    cb_wait_front(c_scal2, 5);  // 3: w  4: d - 1 (read here, after compute's last use of it)
    }
    DeviceZoneScopedN("REC_INPUTS");
    const uint32_t prep = get_read_ptr(c_qkfull);
    const uint32_t cv = get_read_ptr(c_cv);
    const uint32_t scal = get_read_ptr(c_scal2);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t r = 0; r < B; ++r) {
        cb_reserve_back(c_qd, KT);
        cb_reserve_back(c_v, 1);
        const uint32_t qd = get_write_ptr(c_qd), v = get_write_ptr(c_v);
        for (uint32_t i = 0; i < KT; ++i) {
            if (r > 0) {
                dma_zero_row(zeros, qd + i * TILE_BYTES, r - 1);
                dma_zero_row(zeros, qd + i * TILE_BYTES, r + 15);
            }
            dma_row(qd + i * TILE_BYTES, r, prep + (KT + i) * TILE_BYTES, r);
            dma_row(qd + i * TILE_BYTES, r + 16, prep + i * TILE_BYTES, r);
        }
        if (r > 0) {
            dma_zero_row(zeros, v, r - 1);
        }
        dma_row(v, r, cv, r);
        noc_async_read_barrier();
        cb_push_back(c_qd, KT);
        cb_push_back(c_v, 1);

        cb_reserve_back(c_w, 1);
        {
            l1_ptr p = tile_ptr(get_write_ptr(c_w), 0);
            if (r > 0) {
                p[tidx(r - 1, r + 15)] = 0;
                p[tidx(r - 1, r - 1)] = 0;
            }
            p[tidx(r, r + 16)] = 0x3f800000;  // 1.0f
            p[tidx(r, r)] = tile_ptr(scal, 3)[tidx(r, 0)];
        }
        cb_push_back(c_w, 1);

        cb_reserve_back(c_d, 1);  // (d_r - 1) I
        {
            l1_ptr p = tile_ptr(get_write_ptr(c_d), 0);
            const uint32_t dm1 = tile_ptr(scal, 4)[tidx(r, 0)];
            for (uint32_t i = 0; i < 32; ++i) {
                p[tidx(i, i)] = dm1;
            }
        }
        cb_push_back(c_d, 1);
    }
    cb_pop_front(c_qkfull, 3 * KT);
    cb_pop_front(c_cv, 1);
    cb_pop_front(c_scal2, 5);

    noc_async_atomic_barrier();
}
