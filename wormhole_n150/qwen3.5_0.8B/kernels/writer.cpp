// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5-0.8B Gated DeltaNet layer, all weights in L1 -- writer (the only kernel that sends to other cores).
//
//   0. (core 0) fetches the B input rows and multicasts them to all 64 cores in pieces, so in_proj starts on
//      the first piece instead of every core pulling the same data from the same L1 banks.
//   1. l2 norms: this core's quarter's |q|^2, |k|^2, q.k (one tile) -> the head's 4 cores.
//   2. q/k swap: this core's row-scaled quarter (d*q, -d*k, beta*k) -> the head's 4 cores, so each has the
//      head's full vectors. Only the BP batch rows travel (2 blocks per tile).
//   3. RMSNorm: the 4 partial sums of squares of the head are exchanged within the group.
//   4. y all-gather within each half of the grid: every core's y block -> the half's coordinator, which
//      multicasts the gathered y to the half (one multicast source, so no contention).
//   5. out_proj: cores c and c ^ 32 compute the same output tile over the two halves of the v channels; each
//      sends the other the face the other writes.
//   6. this core's face of its output tile -> the output.
// Each data transfer is followed by a flush and a semaphore increment on the same route and static VC, so the
// increment lands after the data without waiting for acknowledgements.
//
// Activations (layer input x and output) use a face-major layout matching the tiles:
// value (row r, column 32k + 16h + e) at word ((2k + h) * BP + r) * 16 + e.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"  // zones compile to nothing unless profiling

constexpr uint32_t c_yasm = 1, c_pp = 6, c_part = 9, c_pfull = 16, c_qkfull = 18, c_yg = 21, c_prep = 23, c_out = 24,
                   c_ss = 26, c_ssp = 27, c_sc = 28, c_tmp3 = 30;

constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = TILE_WORDS * 4;
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t FACE_ROW_BYTES = 64;
constexpr uint32_t FACE_BYTES = FACE_WORDS * 4;
constexpr uint32_t KT = 4, GROUP = 4, N_HALF = 32, N_CORES = 64;

using l1_ptr = volatile tt_l1_ptr uint32_t*;

// zero n bytes at addr (local DMA from the zero region, in MEM_ZEROS_SIZE pieces)
FORCE_INLINE void zero_bytes(uint64_t zeros, uint32_t addr, uint32_t n) {
    for (uint32_t off = 0; off < n; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, addr + off, n - off < MEM_ZEROS_SIZE ? n - off : MEM_ZEROS_SIZE);
    }
}

// zero the tile at addr except rows 0..BP-1 of its top two faces (the part that gets written)
template <uint32_t BLOCK>
FORCE_INLINE void zero_outside_rows(uint64_t zeros, uint32_t tile) {
    if constexpr (BLOCK < FACE_BYTES) {
        zero_bytes(zeros, tile + BLOCK, FACE_BYTES - BLOCK);
        zero_bytes(zeros, tile + FACE_BYTES + BLOCK, FACE_BYTES - BLOCK);
    }
    zero_bytes(zeros, tile + 2 * FACE_BYTES, 2 * FACE_BYTES);
}

void kernel_main() {
    constexpr uint32_t BP = get_compile_time_arg_val(4);
    constexpr uint32_t BLOCK = BP * FACE_ROW_BYTES;  // BP rows of one face
    constexpr uint32_t X_MCASTS = BP < 8 ? BP : 8, X_MCAST_PAGES = BP / X_MCASTS;  // the input's BP 4 KB pages
    constexpr auto out_args = TensorAccessorArgs<9>();
    constexpr auto x_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const uint32_t sem_qk = get_semaphore(get_compile_time_arg_val(0));
    const uint32_t sem_ss = get_semaphore(get_compile_time_arg_val(1));
    const uint32_t sem_pp = get_semaphore(get_compile_time_arg_val(2));
    const uint32_t sem_x = get_semaphore(get_compile_time_arg_val(3));
    const uint32_t sem_ready = get_semaphore(get_compile_time_arg_val(5));
    const uint32_t sem_yarrive = get_semaphore(get_compile_time_arg_val(6));
    const uint32_t sem_ydone = get_semaphore(get_compile_time_arg_val(7));
    const uint32_t sem_pt = get_semaphore(get_compile_time_arg_val(8));

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t M = get_arg_val<uint32_t>(1);  // batch: one row per sequence
    const uint32_t core = get_arg_val<uint32_t>(2);
    const uint32_t s = core % GROUP;
    const uint32_t half = core / N_HALF, j = core % N_HALF;  // this core writes face `half` of output tile j
    // args 3..10: (x, y) of the head's 4 cores; 11..14: this half of the grid (first core, last core);
    // 15, 16: the other half's core of output tile j; 17: x address; 18..21: multicast rectangle of the 64 cores
    const auto out_acc = TensorAccessor(out_args, out_addr);
    const auto x_acc = TensorAccessor(x_args, get_arg_val<uint32_t>(17));

    // 0. input rows -> every core's c_prep buffer (unused until compute's C4), in X_MCASTS pieces so that in_proj
    //    starts on the first: piece m is multicast as soon as its pages are read (transaction id m + 1), then sem_x
    //    is multicast as m + 1 once the piece has been sent (same NoC and VC, so it lands after the data).
    if (core == 0) {
        DeviceZoneScopedN("X_MCAST");
        const uint32_t buf = get_write_ptr(c_prep);
        for (uint32_t r = 0; r < BP; ++r) {  // the face-major input occupies the first BP pages
            noc_async_read_set_trid(1 + r / X_MCAST_PAGES);
            noc_async_read_page(r, x_acc, buf + r * TILE_BYTES);
        }
        noc_async_read_set_trid(0);
        uint32_t x0 = get_arg_val<uint32_t>(18), y0 = get_arg_val<uint32_t>(19);
        uint32_t x1 = get_arg_val<uint32_t>(20), y1 = get_arg_val<uint32_t>(21);
        if (noc_index == 1) {
            const uint32_t tx = x0, ty = y0;
            x0 = x1, y0 = y1, x1 = tx, y1 = ty;
        }
        const uint32_t counts = get_write_ptr(c_tmp3);  // the values sem_x is set to (unused until compute's RMSNorm)
        for (uint32_t m = 0; m < X_MCASTS; ++m) {
            const uint32_t piece = buf + m * X_MCAST_PAGES * TILE_BYTES;
            reinterpret_cast<l1_ptr>(counts)[4 * m] = m + 1;
            noc_async_read_barrier_with_trid(1 + m);
            noc_async_write_multicast_loopback_src(
                piece, get_noc_multicast_addr(x0, y0, x1, y1, piece), X_MCAST_PAGES * TILE_BYTES, N_CORES);
            noc_async_writes_flushed();  // the data and the flag use different command buffers: send the data first
            noc_semaphore_set_multicast_loopback_src(counts + 16 * m, get_noc_multicast_addr(x0, y0, x1, y1, sem_x), N_CORES);
        }
        noc_async_read_barrier();
        noc_async_write_barrier();
    }

    const uint32_t qkfull_base = get_write_ptr(c_qkfull);
    const uint32_t ssp_base = get_write_ptr(c_ssp);

    // Receive buffers of the two exchanges below: c_qkfull only ever gets rows 0..BP-1 of each tile's top faces, c_pfull
    // rows 0..4 BP - 1 (the head's 4 senders, BP rows each). Everything else is zeroed here, before compute is given
    // the tiles; no other core writes these bytes, so this cannot race with the senders.
    constexpr uint32_t PFULL = BP * GROUP / 32 > 1 ? BP * GROUP / 32 : 1;
    const uint32_t pfull_base = get_write_ptr(c_pfull);
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t t = 0; t < 3 * KT; ++t) {
            zero_outside_rows<BLOCK>(zeros, qkfull_base + t * TILE_BYTES);
        }
        if constexpr (GROUP * BP < 16) {  // the senders fill rows 0..4 BP - 1 of the top faces
            zero_outside_rows<GROUP * BLOCK>(zeros, pfull_base);
        } else if constexpr (GROUP * BP == 16) {
            zero_bytes(zeros, pfull_base + 2 * FACE_BYTES, 2 * FACE_BYTES);
        }
        noc_async_read_barrier();
    }

    // 1. the l2-norm partial sums of this core's quarter (one tile: |q|^2, |k|^2, q.k in columns 0, 1, 16) -> rows
    //    s * BP .. of the head's 4 cores' c_pfull (two blocks each), summed there by compute
    {
        DeviceZoneScopedN("QK_PARTIAL_CBWAIT");
        cb_wait_front(c_part, 1);
    }
    {
        DeviceZoneScopedN("QK_PARTIAL_SEND");
        const uint32_t src = get_read_ptr(c_part);
        const uint32_t row = s * BP, dst = pfull_base + (row / 32) * TILE_BYTES + (row % 32 / 16) * 2 * FACE_BYTES +
                                          (row % 16) * FACE_ROW_BYTES;
        for (uint32_t p = 0; p < GROUP; ++p) {
            const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_write(src + h * FACE_BYTES, get_noc_addr(x, y, dst + h * FACE_BYTES), BLOCK);
            }
        }
        noc_async_writes_flushed();  // sent: the increment below takes the same route and static VC, so it lands after
        for (uint32_t p = 0; p < GROUP; ++p) {
            const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
            noc_semaphore_inc(get_noc_addr(x, y, sem_pt), 1);
        }
        cb_pop_front(c_part, 1);
    }
    {
        DeviceZoneScopedN("QK_PARTIAL_RECV");
        noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_pt), GROUP);
    }
    cb_reserve_back(c_pfull, PFULL);
    cb_push_back(c_pfull, PFULL);

    // 2. this quarter's d*q, -d*k, beta*k -> tiles s, KT + s, 2 KT + s of the head's 4 cores' c_qkfull: rows 0..BP-1
    //    of each tile's two top faces, 2 blocks per tile
    {
        DeviceZoneScopedN("QK_SC_CBWAIT");
        cb_wait_front(c_sc, 3);
    }
    {
        DeviceZoneScopedN("QK_SWAP_SEND");
        const uint32_t src = get_read_ptr(c_sc);
        for (uint32_t p = 0; p < GROUP; ++p) {
            const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
            for (uint32_t t = 0; t < 3; ++t) {
                const uint32_t dst = qkfull_base + (t * KT + s) * TILE_BYTES;
                for (uint32_t h = 0; h < 2; ++h) {
                    noc_async_write(src + t * TILE_BYTES + h * FACE_BYTES, get_noc_addr(x, y, dst + h * FACE_BYTES), BLOCK);
                }
            }
        }
        noc_async_writes_flushed();  // sent: the increment below takes the same route and static VC, so it lands after
        for (uint32_t p = 0; p < GROUP; ++p) {
            const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
            noc_semaphore_inc(get_noc_addr(x, y, sem_qk), 1);
        }
        cb_pop_front(c_sc, 3);
    }
    {
    DeviceZoneScopedN("QK_SWAP_RECV");
    noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_qk), GROUP);
    }
    cb_reserve_back(c_qkfull, 3 * KT);
    cb_push_back(c_qkfull, 3 * KT);

    // 3. RMSNorm partial sums within the head
    {
    DeviceZoneScopedN("SS_WAIT");
    cb_wait_front(c_ss, 1);
    }
    {
    DeviceZoneScopedN("SS_SEND");
    for (uint32_t p = 0; p < GROUP; ++p) {
        const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
        noc_async_write(get_read_ptr(c_ss), get_noc_addr(x, y, ssp_base + s * TILE_BYTES), BLOCK);  // column 0's rows
    }
    noc_async_writes_flushed();  // sent: the increment below takes the same route and static VC, so it lands after
    for (uint32_t p = 0; p < GROUP; ++p) {
        const uint32_t x = get_arg_val<uint32_t>(3 + 2 * p), y = get_arg_val<uint32_t>(4 + 2 * p);
        noc_semaphore_inc(get_noc_addr(x, y, sem_ss), 1);
    }
    cb_pop_front(c_ss, 1);
    }
    {
    DeviceZoneScopedN("SS_RECV");
    noc_semaphore_wait_min(reinterpret_cast<l1_ptr>(sem_ss), GROUP);
    }
    cb_reserve_back(c_ssp, GROUP);
    cb_push_back(c_ssp, GROUP);

    // 4. y all-gather within this half of the grid (cores 32 half .. 32 half + 31; the first is the coordinator).
    //    This core's y, BP rows x 32 channels, goes to slot j of the coordinator's c_yasm as one BP x 32 tile (two
    //    blocks). Once all 32 slots are in and every core of the half has said its c_prep is free (the reader), the
    //    coordinator multicasts the dense 4 KB x BP buffer into the half's c_yg (which shares c_prep's memory): one
    //    multicast source, so no contention between concurrent multicasts.
    {
        DeviceZoneScopedN("Y_GATHER");
        uint32_t x0 = get_arg_val<uint32_t>(11), y0 = get_arg_val<uint32_t>(12);
        uint32_t x1 = get_arg_val<uint32_t>(13), y1 = get_arg_val<uint32_t>(14);
        const uint32_t yasm = get_write_ptr(c_yasm);
        {
        DeviceZoneScopedN("Y_SEEN");
        cb_wait_front(c_ss, 1);  // y
        }
        const uint32_t src = get_read_ptr(c_ss), slot = yasm + j * 2 * BLOCK;
        noc_async_write(src, get_noc_addr(x0, y0, slot), BLOCK);
        noc_async_write(src + FACE_BYTES, get_noc_addr(x0, y0, slot + BLOCK), BLOCK);
        noc_async_writes_flushed();  // sent: the increment below takes the same route and static VC, so it lands after
        noc_semaphore_inc(get_noc_addr(x0, y0, sem_yarrive), 1);
        cb_pop_front(c_ss, 1);
        if (j == 0) {
            DeviceZoneScopedN("Y_MCAST");
            if (noc_index == 1) {
                const uint32_t tx = x0, ty = y0;
                x0 = x1, y0 = y1, x1 = tx, y1 = ty;
            }
            noc_semaphore_wait(reinterpret_cast<l1_ptr>(sem_yarrive), N_HALF);
            noc_semaphore_wait(reinterpret_cast<l1_ptr>(sem_ready), N_HALF);
            const uint32_t yg = get_write_ptr(c_yg);
            noc_async_write_multicast_loopback_src(yasm, get_noc_multicast_addr(x0, y0, x1, y1, yg), N_HALF * 2 * BLOCK, N_HALF);
            noc_async_writes_flushed();  // the data and the flag use different command buffers: send the data first
            const uint32_t one = yasm + N_HALF * 2 * BLOCK;  // the flag's source word, past the slots
            *reinterpret_cast<l1_ptr>(one) = 1;
            noc_semaphore_set_multicast_loopback_src(one, get_noc_multicast_addr(x0, y0, x1, y1, sem_ydone), N_HALF);
        }
        {
        DeviceZoneScopedN("Y_RECV");
        noc_semaphore_wait(reinterpret_cast<l1_ptr>(sem_ydone), 1);
        }
        cb_reserve_back(c_yg, N_HALF);
        cb_push_back(c_yg, N_HALF);
    }

    // 5. the two halves' partials of output tile j: face 1 - half of this core's partial -> the other half's core
    //    (one block into its c_pp), which sends back its face `half`
    {
        DeviceZoneScopedN("PAIR_EXCHANGE");
        const uint32_t px = get_arg_val<uint32_t>(15), py = get_arg_val<uint32_t>(16);
        const uint32_t face = (1 - half) * FACE_BYTES;
        {
        DeviceZoneScopedN("PARTIAL_SEEN");
        cb_wait_front(c_out, 1);  // read only; compute pops it after the sum
        }
        noc_async_write(get_read_ptr(c_out) + face, get_noc_addr(px, py, get_write_ptr(c_pp) + face), BLOCK);
        noc_async_writes_flushed();  // sent: the increment below takes the same route and static VC, so it lands after
        noc_semaphore_inc(get_noc_addr(px, py, sem_pp), 1);
        noc_semaphore_wait(reinterpret_cast<l1_ptr>(sem_pp), 1);
        cb_reserve_back(c_pp, 1);
        cb_push_back(c_pp, 1);
    }

    // 6. face `half` of output tile j -> the output (one block)
    {
        DeviceZoneScopedN("OUT_WRITE");
        {
        DeviceZoneScopedN("FINAL_SEEN");
        cb_wait_front(c_tmp3, 1);
        }
        const uint32_t off = (2 * j + half) * BLOCK;
        noc_async_write(get_read_ptr(c_tmp3) + half * FACE_BYTES, out_acc.get_noc_addr(off / TILE_BYTES, off % TILE_BYTES), BLOCK);
        noc_async_write_barrier();
        cb_pop_front(c_tmp3, 1);
    }
    noc_async_atomic_barrier();
}
