// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet recurrent step -- writer (data movement).
//
// Compute emits one output tile per (token, unit, column tile) with the result in row 0.
// The writer packs 32 tokens into the rows of a staging tile and writes each finished
// block to DRAM, so the output has the same token-packed layout as the inputs.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t c_out = 16;
constexpr uint32_t c_ostage = 21;

constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

void kernel_main() {
    constexpr uint32_t NCT = get_compile_time_arg_val(0);
    constexpr uint32_t NSPLIT = get_compile_time_arg_val(1);
    constexpr uint32_t U = get_compile_time_arg_val(2);
    constexpr uint32_t VT = NCT * NSPLIT;
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t T = get_arg_val<uint32_t>(2);
    const uint32_t Tb = (T + 31) / 32;

    const auto out_acc = TensorAccessor(out_args, out_addr);
    const uint32_t stage_base = get_write_ptr(c_ostage);
    {
        l1_ptr p = reinterpret_cast<l1_ptr>(stage_base);
        for (uint32_t i = 0; i < U * NCT * TILE_WORDS; ++i) {
            p[i] = 0;
        }
    }

    for (uint32_t b = 0; b < Tb; ++b) {
        const uint32_t tcount = (T - b * 32) < 32 ? (T - b * 32) : 32;
        for (uint32_t t = 0; t < tcount; ++t) {
            const uint32_t off = (t >> 4) * 2 * FACE_WORDS + (t & 15) * 16;
            for (uint32_t u = 0; u < U; ++u) {
                cb_wait_front(c_out, NCT);
                l1_ptr src = reinterpret_cast<l1_ptr>(get_read_ptr(c_out));
                l1_ptr dst = reinterpret_cast<l1_ptr>(stage_base + u * NCT * TILE_BYTES);
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
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t unit = unit_start + u;
            const uint32_t h = unit / NSPLIT;
            const uint32_t c = unit % NSPLIT;
            const uint32_t row = (h * Tb + b) * VT + c * NCT;
            for (uint32_t j = 0; j < NCT; ++j) {
                noc_async_write_page(row + j, out_acc, stage_base + (u * NCT + j) * TILE_BYTES);
            }
        }
        noc_async_write_barrier();
    }
}
