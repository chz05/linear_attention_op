// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet recurrent step -- compute.
//
// The state S (dk x ncols slice of one head, fp32) lives in this core's L1 for the whole
// lifetime of the model: the c_state CB is backed by an L1 height-sharded tensor and is
// updated in place, so it never travels to DRAM.
//
// Per token, with d = exp(g), kb = beta * k (column form), and everything else in row form:
//   u   = v - (d k)^T S                      (= I @ v + sum_i (-d k)_i @ S_i)
//   o   = (d q)^T S + beta (q . k) u         (= q^T S_new, computed from the old S)
//   S  += (d - 1) S + kb u^T                 (= D @ S + kb @ u, D = (d - 1) I)
// Every step is a tile matmul, so the FPU never has to be reconfigured mid-loop.
//
// The matmul unit reads S at reduced (tf32) precision, truncating toward zero. If S itself
// went through it every token (S = d S + ...), that bias would compound over the decay
// horizon (~5% error after 1k tokens at d ~ 0.99). Instead only the *increment* goes through
// the FPU and the packer adds it onto the fp32 state in L1 (L1 accumulation), so the
// truncation error is scaled by (1 - d) and the stored state stays full fp32.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

constexpr uint32_t c_qd = 0;
constexpr uint32_t c_nkd = 1;
constexpr uint32_t c_kb = 2;
constexpr uint32_t c_v = 3;
constexpr uint32_t c_d = 4;
constexpr uint32_t c_w = 5;
constexpr uint32_t c_id = 6;
constexpr uint32_t c_u = 7;
constexpr uint32_t c_state = 8;
constexpr uint32_t c_sync = 9;
constexpr uint32_t c_out = 16;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NCT = get_compile_time_arg_val(1);
    constexpr uint32_t U = get_compile_time_arg_val(2);
    const uint32_t T = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_nkd, c_state, c_u);
    matmul_init(c_nkd, c_state);

    cb_wait_front(c_id, 1);

    for (uint32_t t = 0; t < T; ++t) {
        // The previous token's in-place state writes (packer) must land before we read S again.
        if (t > 0) {
            cb_wait_front(c_sync, 1);
            cb_pop_front(c_sync, 1);
        }
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t s0 = u * KT * NCT;

            // u = v - (d k)^T S
            cb_wait_front(c_nkd, KT);
            cb_wait_front(c_v, NCT);
            cb_reserve_back(c_u, NCT);
            for (uint32_t j = 0; j < NCT; ++j) {
                tile_regs_acquire();
                matmul_tiles(c_id, c_v, 0, j, 0);
                for (uint32_t i = 0; i < KT; ++i) {
                    matmul_tiles(c_nkd, c_state, i, s0 + i * NCT + j, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, c_u);
                tile_regs_release();
            }
            cb_push_back(c_u, NCT);
            cb_pop_front(c_nkd, KT);
            cb_pop_front(c_v, NCT);

            // o = (d q)^T S + beta (q . k) u
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
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, c_out);
                tile_regs_release();
            }
            cb_push_back(c_out, NCT);
            cb_pop_front(c_qd, KT);
            cb_pop_front(c_w, 1);

            // S += (d - 1) S + kb u^T, accumulated in place by the packer
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
        }
        cb_reserve_back(c_sync, 1);
        cb_push_back(c_sync, 1);
    }
    // Leave the sync CB empty for the next launch.
    if (T > 0) {
        cb_wait_front(c_sync, 1);
        cb_pop_front(c_sync, 1);
    }
}
