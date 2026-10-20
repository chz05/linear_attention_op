// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5-0.8B Gated DeltaNet layer, all weights in L1 -- writer (the only kernel that talks to other cores).
//
//   0. (core 0) fetches the B input rows once and multicasts them to all 64 cores, instead of every core
//      pulling the same data from the same L1 banks.
//   1. q/k gather: the head's 4 cores each convolved a quarter of q and k; every core writes its quarter
//      into all 4 cores' c_qkfull, so each has the head's full q and k.
//   2. collects the per-token output rows into a tile.
//   3. RMSNorm: the 4 partial sums of squares of the head are exchanged within the group.
//   4. out_proj reduce-scatter: output tile j (32 columns of the 1024) is owned by core j. Every core
//      sends the BP real rows of its partial tile j (BP = batch rounded up to a power of 2) to core j as
//      two contiguous BP x 64 B blocks (left / right face). Core j packs 32 / BP senders per receive tile
//      (sender c -> tile c / (32/BP), rows (c % (32/BP)) * BP ...), so no tile padding travels and the
//      number of transfers does not grow with the batch.
//   5. (owners) write their 32 output columns, again as two BP x 64 B blocks.
//
// Activations (layer input x and output) use a face-major layout matching the tiles:
// value (row r, column 32k + 16h + e) at word ((2k + h) * BP + r) * 16 + e.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"  // zones compile to nothing unless profiling

constexpr uint32_t c_recv = 6, c_qkloc = 17, c_qkfull = 18, c_prep = 23, c_out = 24, c_oblk = 25, c_ss = 26,
                   c_ssp = 27;

constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t FACE_ROW_BYTES = 64;
constexpr uint32_t KT = 4, GROUP = 4, N_OUT = 32, N_CORES = 64;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

FORCE_INLINE uint32_t row_off_bytes(uint32_t r) { return ((r >> 4) * 2 * FACE_WORDS + (r & 15) * 16) * 4; }

FORCE_INLINE void copy_row0_to(l1_ptr dst, uint32_t dr, l1_ptr src) {
    l1_ptr d = dst + (dr >> 4) * 2 * FACE_WORDS + (dr & 15) * 16;
    for (uint32_t e = 0; e < 16; ++e) {
        d[e] = src[e];
        d[FACE_WORDS + e] = src[FACE_WORDS + e];
    }
}

void kernel_main() {
    constexpr uint32_t BP = get_compile_time_arg_val(4);
    constexpr uint32_t SENDERS_PER_TILE = 32 / BP;
    constexpr uint32_t BLOCK = BP * FACE_ROW_BYTES;  // BP rows of one face
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr auto x_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const uint32_t sem_qk = get_semaphore(get_compile_time_arg_val(0));
    const uint32_t sem_ss = get_semaphore(get_compile_time_arg_val(1));
    const uint32_t sem_red = get_semaphore(get_compile_time_arg_val(2));
    const uint32_t sem_x = get_semaphore(get_compile_time_arg_val(3));

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t M = get_arg_val<uint32_t>(1);  // batch: one row per sequence
    const uint32_t core = get_arg_val<uint32_t>(2);
    const uint32_t s = core % GROUP;
    // args 3..10: (x, y) of the head's 4 cores; args 11..74: (x, y) of the 32 output owners;
    // arg 75: x address; args 76..79: multicast rectangle of the 64 cores
    const auto out_acc = TensorAccessor(out_args, out_addr);
    const auto x_acc = TensorAccessor(x_args, get_arg_val<uint32_t>(75));

    // 0. input rows -> every core's c_prep buffer (unused until compute's C4)
    if (core == 0) {
        const uint32_t buf = get_write_ptr(c_prep);
        for (uint32_t r = 0; r < BP; ++r) {  // the face-major input occupies the first BP pages
            noc_async_read_page(r, x_acc, buf + r * TILE_BYTES);
        }
        noc_async_read_barrier();
        uint32_t x0 = get_arg_val<uint32_t>(76), y0 = get_arg_val<uint32_t>(77);
        uint32_t x1 = get_arg_val<uint32_t>(78), y1 = get_arg_val<uint32_t>(79);
        if (noc_index == 1) {
            const uint32_t tx = x0, ty = y0;
            x0 = x1, y0 = y1, x1 = tx, y1 = ty;
        }
        noc_async_write_multicast_loopback_src(buf, get_noc_multicast_addr(x0, y0, x1, y1, buf), BP * TILE_BYTES, N_CORES);
        noc_async_write_barrier();
        *reinterpret_cast<l1_ptr>(sem_x) = 1;
        noc_semaphore_set_multicast_loopback_src(sem_x, get_noc_multicast_addr(x0, y0, x1, y1, sem_x), N_CORES);
        noc_async_write_barrier();
    }

    const uint32_t qkfull_base = get_write_ptr(c_qkfull);
    const uint32_t ssp_base = get_write_ptr(c_ssp);
    const uint32_t recv_base = get_write_ptr(c_recv);

    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t a = get_write_ptr(c_oblk);
        for (uint32_t off = 0; off < TILE_BYTES; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, a + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }

    // 1. q/k quarter gather within the head
    cb_wait_front(c_qkloc, 2);
    {
        DeviceZoneScopedN("QK_SWAP_SEND");
        const uint32_t src = get_read_ptr(c_qkloc);
        for (uint32_t p = 0; p < GROUP; ++p) {
            const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
            noc_async_write(src, get_noc_addr(x, y, qkfull_base + s * TILE_BYTES), TILE_BYTES);
            noc_async_write(src + TILE_BYTES, get_noc_addr(x, y, qkfull_base + (KT + s) * TILE_BYTES), TILE_BYTES);
        }
        noc_async_write_barrier();
        for (uint32_t p = 0; p < GROUP; ++p) {
            const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
            noc_semaphore_inc(get_noc_addr(x, y, sem_qk), 1);
        }
    }
    cb_pop_front(c_qkloc, 2);
    noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_qk), GROUP);
    cb_reserve_back(c_qkfull, 2 * KT);
    cb_push_back(c_qkfull, 2 * KT);

    // 2. per-token outputs -> tile
    cb_reserve_back(c_oblk, 1);
    {
        l1_ptr dst = reinterpret_cast<l1_ptr>(get_write_ptr(c_oblk));
        for (uint32_t r = 0; r < M; ++r) {  // sequence r's output is in row r
            cb_wait_front(c_out, 1);
            l1_ptr src = reinterpret_cast<l1_ptr>(get_read_ptr(c_out)) + (r >> 4) * 2 * FACE_WORDS + (r & 15) * 16;
            copy_row0_to(dst, r, src);
            cb_pop_front(c_out, 1);
        }
    }
    cb_push_back(c_oblk, 1);

    // 3. RMSNorm partial sums within the head
    cb_wait_front(c_ss, 1);
    for (uint32_t p = 0; p < GROUP; ++p) {
        const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
        noc_async_write(get_read_ptr(c_ss), get_noc_addr(x, y, ssp_base + s * TILE_BYTES), TILE_BYTES);
    }
    noc_async_write_barrier();
    for (uint32_t p = 0; p < GROUP; ++p) {
        const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
        noc_semaphore_inc(get_noc_addr(x, y, sem_ss), 1);
    }
    cb_pop_front(c_ss, 1);
    noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_ss), GROUP);
    cb_reserve_back(c_ssp, GROUP);
    cb_push_back(c_ssp, GROUP);

    // 4. out_proj reduce-scatter: BP rows of partial tile j -> owner j, two blocks per owner
    {
        const uint32_t slot_row = row_off_bytes((core % SENDERS_PER_TILE) * BP);
        const uint32_t slot = recv_base + (core / SENDERS_PER_TILE) * TILE_BYTES + slot_row;
        for (uint32_t j = 0; j < N_OUT; ++j) {
            const uint32_t x = get_arg_val<uint32_t>(11 + 2 * j), y = get_arg_val<uint32_t>(12 + 2 * j);
            cb_wait_front(c_out, 1);
            const uint32_t src = get_read_ptr(c_out);
            noc_async_write(src, get_noc_addr(x, y, slot), BLOCK);
            noc_async_write(src + FACE_WORDS * 4, get_noc_addr(x, y, slot + FACE_WORDS * 4), BLOCK);
            noc_async_writes_flushed();  // data has left the slot; compute may reuse it after the pop
            cb_pop_front(c_out, 1);
        }
        noc_async_write_barrier();  // all blocks acknowledged before the owners are told
        for (uint32_t j = 0; j < N_OUT; ++j) {
            const uint32_t x = get_arg_val<uint32_t>(11 + 2 * j), y = get_arg_val<uint32_t>(12 + 2 * j);
            noc_semaphore_inc(get_noc_addr(x, y, sem_red), 1);
        }
    }

    // 5. owners: one tile with the BP reduced rows -> output columns 32*core .. 32*core + 31
    if (core < N_OUT) {
        noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_red), N_CORES);
        cb_reserve_back(c_recv, 2 * BP);
        cb_push_back(c_recv, 2 * BP);
        cb_wait_front(c_ss, 1);
        const uint32_t src = get_read_ptr(c_ss);
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t off = (2 * core + h) * BLOCK;
            noc_async_write(src + h * FACE_WORDS * 4, out_acc.get_noc_addr(off / TILE_BYTES, off % TILE_BYTES), BLOCK);
        }
        noc_async_write_barrier();
        cb_pop_front(c_ss, 1);
    }
    noc_async_atomic_barrier();
}
