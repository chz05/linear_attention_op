// SPDX-License-Identifier: Apache-2.0
//
// Fused Gated DeltaNet layer -- writer (data movement, the only kernel that talks to other cores).
//
// Per 32-row block:
//   1. k-head gather: the 4 cores sharing a k-head each convolved a quarter of q and k; every core
//      writes its quarter into all 4 group members' c_qkfull slot, so each ends up with the full
//      q/k of the head (the only cross-core traffic before the recurrence).
//   2. collects the per-token output rows from compute into a 32-row block.
//   3. swaps the partial RMSNorm sum of squares with the partner core (other half of the head).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t c_qkloc = 14, c_qkfull = 15, c_out = 16, c_oblk = 22, c_ss = 23, c_ssp = 24;

constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t KT = 4, NCT = 2, GROUP = 4;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

void kernel_main() {
    const uint32_t M = get_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(1);  // my position in the k-head group
    const uint32_t partner_x = get_arg_val<uint32_t>(2);
    const uint32_t partner_y = get_arg_val<uint32_t>(3);
    // args 4..11: (x, y) of the 4 group cores
    const uint32_t num_blocks = (M + 31) / 32;

    const uint32_t sem_qk = get_semaphore(get_compile_time_arg_val(0));
    const uint32_t sem_ss = get_semaphore(get_compile_time_arg_val(1));
    const uint32_t qkfull_base = get_write_ptr(c_qkfull);
    const uint32_t ssp_base = get_write_ptr(c_ssp);

    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t a = get_write_ptr(c_oblk);
        for (uint32_t off = 0; off < 2 * NCT * TILE_BYTES; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, a + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }

    for (uint32_t rb = 0; rb < num_blocks; ++rb) {
        const uint32_t R = (M - rb * 32) < 32 ? (M - rb * 32) : 32;

        // 1. k-head gather (slots are per block, so no slot is ever reused within a launch)
        cb_wait_front(c_qkloc, 2);
        {
            const uint32_t src = get_read_ptr(c_qkloc);
            const uint32_t slot = qkfull_base + rb * 2 * KT * TILE_BYTES;
            for (uint32_t p = 0; p < GROUP; ++p) {
                const uint32_t x = get_arg_val<uint32_t>(4 + 2 * p), y = get_arg_val<uint32_t>(5 + 2 * p);
                noc_async_write(src, get_noc_addr(x, y, slot + g * TILE_BYTES), TILE_BYTES);
                noc_async_write(src + TILE_BYTES, get_noc_addr(x, y, slot + (KT + g) * TILE_BYTES), TILE_BYTES);
            }
            noc_async_write_barrier();
            for (uint32_t p = 0; p < GROUP; ++p) {
                const uint32_t x = get_arg_val<uint32_t>(4 + 2 * p), y = get_arg_val<uint32_t>(5 + 2 * p);
                noc_semaphore_inc(get_noc_addr(x, y, sem_qk), 1);
            }
        }
        cb_pop_front(c_qkloc, 2);
        noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_qk), GROUP * (rb + 1));
        cb_reserve_back(c_qkfull, 2 * KT);
        cb_push_back(c_qkfull, 2 * KT);

        // 2. per-token outputs -> 32-row block
        cb_reserve_back(c_oblk, NCT);
        {
            l1_ptr dst = reinterpret_cast<l1_ptr>(get_write_ptr(c_oblk));
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t off = (r >> 4) * 2 * FACE_WORDS + (r & 15) * 16;
                cb_wait_front(c_out, NCT);
                l1_ptr src = reinterpret_cast<l1_ptr>(get_read_ptr(c_out));
                for (uint32_t j = 0; j < NCT; ++j) {
                    l1_ptr s = src + j * TILE_WORDS;
                    l1_ptr d = dst + j * TILE_WORDS + off;
                    for (uint32_t e = 0; e < 16; ++e) {
                        d[e] = s[e];
                        d[FACE_WORDS + e] = s[FACE_WORDS + e];
                    }
                }
                cb_pop_front(c_out, NCT);
            }
        }
        cb_push_back(c_oblk, NCT);

        // 3. RMSNorm partial sums: mine -> partner's slot for this block
        cb_wait_front(c_ss, 1);
        noc_async_write(get_read_ptr(c_ss), get_noc_addr(partner_x, partner_y, ssp_base + rb * TILE_BYTES), TILE_BYTES);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(partner_x, partner_y, sem_ss), 1);
        cb_pop_front(c_ss, 1);
        noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_ss), rb + 1);
        cb_reserve_back(c_ssp, 1);
        cb_push_back(c_ssp, 1);
    }
    noc_async_atomic_barrier();
}
