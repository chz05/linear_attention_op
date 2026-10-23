// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5-0.8B Gated DeltaNet layers -- batch compaction (data movement), run when sequences leave the batch.
// The B' remaining sequences move up, in order, to rows 0..B'-1 (new row i was row src[i], src[i] >= i), so the
// next step runs with the program of BP' = B' rounded up to a power of 2. On every core, for every layer:
//   recurrent state  slot src[i] -> slot i (16 KB per slot), one slot at a time: a later copy can overwrite the
//                    source of an earlier one
//   conv state       its 18 BP-row face blocks (9 BP x 32 tiles) become BP'-row blocks with row src[i] -> row i and
//                    rows B'..BP'-1 zero; assembled in a scratch buffer, then copied back in one piece
// and once: the group-sum selector of the global constants for BP' (row r <- every row j with j % BP' == r).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t c_scratch = 0;

constexpr uint32_t TILE_BYTES = 4096;
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t ROW_BYTES = 64;                  // one face row: 16 fp32
constexpr uint32_t SLOT_BYTES = 4 * TILE_BYTES;     // one sequence's recurrent state on one core (128 x 32 fp32)
constexpr uint32_t CONV_BLOCKS = 2 * 9;             // conv state: 9 tiles x 2 faces
constexpr uint32_t G_SEL_GRP = 6, MAX_BATCH = 16;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

FORCE_INLINE uint32_t tidx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * FACE_WORDS + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    // Common runtime args (the same on every core) 0: BP (old); 1: BP' (new); 2: B'; 3: global constants; 4: layers;
    // 5..20: src; then per layer its recurrent state and conv state (the same L1 address on every core).
    const uint32_t bp_old = get_common_arg_val<uint32_t>(0), bp = get_common_arg_val<uint32_t>(1);
    const uint32_t batch = get_common_arg_val<uint32_t>(2), gconst = get_common_arg_val<uint32_t>(3);
    const uint32_t n_layers = get_common_arg_val<uint32_t>(4);
    uint32_t src[MAX_BATCH];
    for (uint32_t i = 0; i < batch; ++i) {
        src[i] = get_common_arg_val<uint32_t>(5 + i);
    }
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    const uint32_t scratch = get_write_ptr(c_scratch);

    for (uint32_t l = 0; l < n_layers; ++l) {
        const uint32_t rec = get_common_arg_val<uint32_t>(5 + MAX_BATCH + 2 * l);
        const uint32_t conv = get_common_arg_val<uint32_t>(6 + MAX_BATCH + 2 * l);
        for (uint32_t i = 0; i < batch; ++i) {
            if (src[i] != i) {
                noc_async_read(get_noc_addr(rec + src[i] * SLOT_BYTES), rec + i * SLOT_BYTES, SLOT_BYTES);
                noc_async_read_barrier();
            }
        }
        for (uint32_t blk = 0; blk < CONV_BLOCKS; ++blk) {
            for (uint32_t i = 0; i < bp; ++i) {
                const uint32_t dst = scratch + (blk * bp + i) * ROW_BYTES;
                if (i < batch) {
                    noc_async_read(get_noc_addr(conv + (blk * bp_old + src[i]) * ROW_BYTES), dst, ROW_BYTES);
                } else {
                    noc_async_read(zeros, dst, ROW_BYTES);
                }
            }
        }
        noc_async_read_barrier();
        noc_async_read(get_noc_addr(scratch), conv, CONV_BLOCKS * bp * ROW_BYTES);
        noc_async_read_barrier();
    }

    const uint32_t sel = gconst + G_SEL_GRP * TILE_BYTES;
    for (uint32_t off = 0; off < TILE_BYTES; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, sel + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    l1_ptr p = reinterpret_cast<l1_ptr>(sel);
    for (uint32_t j = 0; j < 32; ++j) {
        p[tidx(j % bp, j)] = 0x3f800000;  // 1.0f
    }
}
