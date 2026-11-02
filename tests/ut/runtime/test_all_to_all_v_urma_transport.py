# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute the builtin transport helpers with a release-mode URMA test double."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def indexed_sqe_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Execute the actual SIMT SQE builder with scalar host stores."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("indexed SQE probe requires a C++17 host compiler")
    root = Path(__file__).resolve().parents[3]
    template = (
        root / "python/pypto/runtime/builtins/collectives/indexed_all_to_all_v/templates/kernel.cpp.in"
    ).read_text()
    begin = template.index("static constexpr uint32_t kSgesPerSqe")
    end = template.index("// Self traffic", begin)
    folder = tmp_path_factory.mktemp("indexed_sqe")
    source = folder / "probe.cpp"
    source.write_text(
        r"""
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __launch_bounds__(...)
struct { uint32_t x; } threadIdx;
enum class ST_L2CacheType { L2_CACHE_HINT_NORMAL_FV };
enum class L1CacheType { NON_CACHEABLE };
template <ST_L2CacheType, L1CacheType>
void __stg(uint64_t *address, uint64_t value) { *address = value; }
"""
        + template[begin:end]
        + r"""
int main(int argc, char **argv) {
    if (argc != 6) return 1;
    const uint32_t head = std::strtoul(argv[1], nullptr, 0);
    const uint32_t depth = std::strtoul(argv[2], nullptr, 0);
    const uint32_t first = std::strtoul(argv[3], nullptr, 0);
    const uint32_t rows = std::strtoul(argv[4], nullptr, 0);
    const uint32_t row_bytes = std::strtoul(argv[5], nullptr, 0);
    const uint32_t num_sqes = (rows - first + 11) / 12;
    std::vector<int32_t> indices(rows);
    for (uint32_t i = 0; i < rows; ++i) indices[i] = (i * 7) % 19;
    std::vector<uint64_t> sq(depth * 8, 0xdeadbeefdeadbeefULL);
    for (uint32_t issued = 0; issued < num_sqes; issued += kSqesPerDoorbell) {
        const uint32_t n = std::min(kSqesPerDoorbell, num_sqes - issued);
        for (threadIdx.x = 0; threadIdx.x < 256; ++threadIdx.x)
            BuildIndexedSqes(sq.data(), indices.data(), 0x100000000ULL, 0x200000000ULL,
                row_bytes, 0x12345678, head + issued * kBbsPerSlot, depth,
                first + issued * kSgesPerSqe, rows, n,
                (3ULL << 40) | (1ULL << 28) | (2ULL << 29) | (0x5aULL << 32),
                0xabcdefULL | (0x54321ULL << 32), 0x1122334455667788ULL,
                0x8877665544332211ULL, 0x123456789abcdef0ULL, 19);
    }
    for (uint64_t word : sq) std::cout << word << "\n";
}
"""
    )
    binary = folder / "probe"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    return binary


@pytest.mark.parametrize(
    "head,depth,first,rows,row_bytes",
    [
        (0, 128, 0, 1, 256),
        (0, 128, 0, 12, 4096),
        (124, 128, 0, 25, 7168),
        (252, 128, 12, 37, 17),
        (112, 128, 0, 109, 8192),
        (0, 128, 0, 0, 256),
        (0, 512, 0, 769, 256),
        (508, 512, 0, 1536, 17),
    ],
)
def test_indexed_simt_sqe_wire_layout(indexed_sqe_probe, head, depth, first, rows, row_bytes):
    """Decode generated SQEs independently, including owner phases and NOPs."""
    result = subprocess.run(
        [str(indexed_sqe_probe), *map(str, (head, depth, first, rows, row_bytes))],
        check=True,
        capture_output=True,
        text=True,
    )
    ring = [int(word) for word in result.stdout.splitlines()]
    assert len(ring) == depth * 8
    num_sqes = (rows - first + 11) // 12
    touched = set()
    gathered = []
    for q in range(num_sqes):
        slot_head = head + 4 * q
        start = first + 12 * q
        count = min(12, rows - start)
        words = [ring[((slot_head + bb) % depth) * 8 + lane] for bb in range(4) for lane in range(8)]
        touched.update(((slot_head + bb) % depth) * 8 + lane for bb in range(4) for lane in range(8))
        # WRITE header: producer index, owner phase, opcode, ordering/fence/CQE.
        assert words[0] & 0xFFFF == slot_head % depth
        assert (words[0] >> 31) & 1 == int((slot_head // depth) % 2 == 0)
        assert (words[0] >> 40) & 0xFF == 3
        assert (words[0] >> 16) & 0x3F == (0x2D if q == num_sqes - 1 else 1)
        assert (words[0] >> 28) & 7 == 5
        assert (words[0] >> 32) & 0xFF == 0x5A
        assert words[1] & 0xFFFFFF == 0xABCDEF
        assert (words[1] >> 24) & 0xFF == count
        assert words[1] >> 32 == 0x54321
        assert words[2:5] == [0x1122334455667788, 0x8877665544332211, 0x123456789ABCDEF0]
        assert words[5] == 0x200000000 + start * row_bytes
        for j in range(count):
            length_and_key, source = words[6 + 2 * j : 8 + 2 * j]
            assert length_and_key & 0xFFFFFFFF == row_bytes
            assert length_and_key >> 32 == 0x12345678
            token = ((start + j) * 7) % 19
            assert source == 0x100000000 + token * row_bytes
            gathered.append(token)
        used_bbs = (48 + 16 * count + 63) // 64
        for lane in range(6 + count * 2, 32):
            bb, offset = divmod(lane, 8)
            if bb >= used_bbs and offset == 0:
                nop = words[lane]
                assert (nop >> 40) & 0xFF == 0x11
                assert nop & 0xFFFF == (slot_head + bb) % depth
                assert (nop >> 31) & 1 == int(((slot_head + bb) // depth) % 2 == 0)
                assert nop & (0x3F << 16) == 0
            else:
                assert words[lane] == 0
    assert gathered == [(i * 7) % 19 for i in range(first, rows)]
    assert all(word == 0xDEADBEEFDEADBEEF for i, word in enumerate(ring) if i not in touched)


@pytest.fixture(scope="module")
def indexed_credit_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the template's credit exchange against concurrent coherent atomics."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("indexed credit probe requires a C++17 host compiler")
    root = Path(__file__).resolve().parents[3]
    template = (
        root / "python/pypto/runtime/builtins/collectives/indexed_all_to_all_v/templates/kernel.cpp.in"
    ).read_text()
    begin = template.index("static constexpr uint32_t kBarrierThreads")
    end = template.index("AICORE inline void RunCreditRound", begin)
    folder = tmp_path_factory.mktemp("indexed_credits")
    source = folder / "probe.cpp"
    source.write_text(
        r"""
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __launch_bounds__(...)
struct ThreadIndex { uint32_t x; };
thread_local ThreadIndex threadIdx;
struct CommContext { uint64_t windowsIn[4]; };
std::atomic<int> overlaps{0};
void __nop() { std::this_thread::yield(); }
int32_t atomicAdd(int32_t *address, int32_t increment) {
    const int32_t old = __atomic_fetch_add(address, increment, __ATOMIC_SEQ_CST);
    if (increment == 1 && old >= 2) ++overlaps;
    if (increment == 0) std::this_thread::yield();
    return old;
}
"""
        + template[begin:end]
        + r"""
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    const int peers = std::atoi(argv[1]), blocks = std::atoi(argv[2]);
    const int stride = blocks + 2, rounds = 12;
    std::vector<std::vector<int32_t>> signal(peers, std::vector<int32_t>(peers * stride, 0));
    CommContext ctx{};
    for (int rank = 0; rank < peers; ++rank) {
        ctx.windowsIn[rank] = reinterpret_cast<uint64_t>(signal[rank].data());
        for (int source = 0; source < peers; ++source)
            for (int pad = blocks; pad < stride; ++pad) signal[rank][source * stride + pad] = 73;
    }
    std::atomic<int> producer[4][7], payload[4][7];
    for (int rank = 0; rank < 4; ++rank)
        for (int block = 0; block < blocks; ++block) {
            producer[rank][block] = 0; payload[rank][block] = 0;
        }
    std::atomic<int> errors{0};
    auto exchange = [&](int rank, int block, int threshold, int drop) {
        // All active peer lanes execute concurrently, as in one SIMT invocation.
        std::vector<std::thread> lanes;
        for (int lane = 0; lane < peers; ++lane)
            lanes.emplace_back([&, lane] {
                threadIdx.x = lane;
                ExchangeCredits(drop ? nullptr : &ctx, signal[rank].data(), rank,
                    peers, stride, block, threshold, drop);
            });
        for (auto &lane : lanes) lane.join();
    };
    std::vector<std::thread> ranks;
    for (int rank = 0; rank < peers; ++rank)
        for (int block = 0; block < blocks; ++block)
            ranks.emplace_back([&, rank, block] {
                for (int round = 1; round <= rounds; ++round) {
                    if ((rank + block + round) % 3 == 0)
                        std::this_thread::sleep_for(std::chrono::microseconds(70));
                    producer[rank][block].store(round);
                    exchange(rank, block, 1, 0);
                    for (int peer = 0; peer < peers; ++peer)
                        if (producer[peer][block].load() < round) ++errors;
                    if ((rank + round) % 2 == 0)
                        std::this_thread::sleep_for(std::chrono::microseconds(90));
                    payload[rank][block].store(round);
                    exchange(rank, block, 2, 0);
                    for (int peer = 0; peer < peers; ++peer)
                        if (payload[peer][block].load() < round) ++errors;
                    // Force peers' next-round credit to arrive before this drop.
                    if (rank == 0 && round % 3 == 0 && round != rounds)
                        std::this_thread::sleep_for(std::chrono::milliseconds(3));
                    exchange(rank, block, 0, 1);
                }
            });
    for (auto &rank : ranks) rank.join();
    for (int rank = 0; rank < peers; ++rank)
        for (int source = 0; source < peers; ++source)
            for (int lane = 0; lane < stride; ++lane) {
                const int32_t expected = lane < blocks ? 0 : 73;
                if (signal[rank][source * stride + lane] != expected) ++errors;
            }
    return errors.load() ? 2 : overlaps.load() == 0 ? 3 : 0;
}
"""
    )
    binary = folder / "probe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", str(source), "-o", str(binary)], check=True)
    return binary


@pytest.mark.parametrize("peers,blocks", [(2, 1), (2, 2), (2, 5), (4, 1), (4, 2), (4, 7)])
def test_indexed_atomic_credits_survive_skewed_reuse(indexed_credit_probe, peers, blocks):
    result = subprocess.run([str(indexed_credit_probe), str(peers), str(blocks)], check=False, timeout=20)
    assert result.returncode == 0, "credit barrier violated ordering, lost a credit, or missed forced overlap"


@pytest.fixture(scope="module")
def indexed_submission_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the actual submission loop against an instrumented SQ/CQ."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("indexed submission probe requires a C++17 host compiler")
    root = Path(__file__).resolve().parents[3]
    template = (
        root / "python/pypto/runtime/builtins/collectives/indexed_all_to_all_v/templates/kernel.cpp.in"
    ).read_text()
    begin = template.index("static constexpr uint32_t kSgesPerSqe")
    end = template.index("}  // namespace", begin)
    folder = tmp_path_factory.mktemp("indexed_submission")
    source = folder / "probe.cpp"
    source.write_text(
        r"""
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __launch_bounds__(...)
#define AICORE
#define PTO_ASSERT(condition, message) if (!(condition)) throw std::runtime_error(message)
struct { uint32_t x; } threadIdx;
enum class ST_L2CacheType { L2_CACHE_HINT_NORMAL_FV };
enum class L1CacheType { NON_CACHEABLE };
template <ST_L2CacheType, L1CacheType>
void __stg(uint64_t *address, uint64_t value) { *address = value; }
enum { PIPE_S, PIPE_V, PIPE_ALL, PIPE_MTE2, PIPE_MTE3, EVENT_ID0, DSB_DDR };
using event_t = int;
int ready[8]{}, loaded[8]{}, pending[8]{}, outstanding = 0, peak = 0;
void pipe_barrier(int) {
    if (outstanding) throw std::runtime_error("global fence inside MTE pipeline");
}
void set_flag(int src, int dst, int slot) {
    if (src == PIPE_MTE3 && dst == PIPE_MTE2) {
        if (ready[slot]++) throw std::runtime_error("duplicate free event");
    }
    if (src == PIPE_MTE2 && dst == PIPE_MTE3) {
        if (loaded[slot]++) throw std::runtime_error("duplicate load event");
    }
}
void wait_flag(int src, int dst, int slot) {
    if (src == PIPE_MTE3 && dst == PIPE_MTE2) {
        if (ready[slot]-- != 1) throw std::runtime_error("missing free event");
        if (pending[slot]) { pending[slot] = 0; --outstanding; }
    }
    if (src == PIPE_MTE2 && dst == PIPE_MTE3 && loaded[slot]-- != 1)
        throw std::runtime_error("store before load completion");
}
void dsb(int) {}
void trap() { throw std::runtime_error("trap"); }
uint32_t ld_dev(uint32_t *p, int) { return *p; }
namespace cce {
struct dim3 { uint32_t x, y, z; };
template <auto F, typename... Args>
void async_invoke(dim3 threads, Args... args) {
    for (threadIdx.x = 0; threadIdx.x < threads.x; ++threadIdx.x) F(args...);
}
}
struct CommContext { uint64_t windowsIn[2]; uint64_t workSpace; uint32_t rankId; };
#define TASSIGN(tile, offset) (tile).address = (offset)
namespace pto {
constexpr int DYNAMIC = -1;
enum class Layout { ND };
enum class TileType { Vec };
enum class BLayout { RowMajor };
enum class TStoreL2Hint { NotAllocClean };
template <int... Dims> struct Shape {
    int64_t size;
    Shape(int64_t, int64_t, int64_t, int64_t, int64_t n) : size(n) {}
};
template <int... Dims> struct Stride { Stride(int64_t, int64_t, int64_t, int64_t, int64_t) {} };
template <typename T, typename S, typename R, Layout> struct GlobalTensor {
    T* ptr;
    int64_t size;
    GlobalTensor(T* p, S s, R) : ptr(p), size(s.size) {}
};
template <TileType, typename T, int Rows, int Cols, BLayout, int, int> struct Tile {
    uint64_t address = 0;
    int ColMaskInternal;
    Tile(int, int cols) : ColMaskInternal(cols) {}
};
namespace comm { std::vector<int64_t> self_copies; }
uint8_t ub[65536];
template <typename T, typename G> void TLOAD(T& tile, G src) {
    if (src.size != tile.ColMaskInternal || src.size > 8192 || tile.address + src.size > sizeof(ub))
        throw std::runtime_error("MTE load extent");
    if (pending[tile.address / 8192]) throw std::runtime_error("overwrite live stage");
    comm::self_copies.push_back(src.size);
    std::memcpy(ub + tile.address, src.ptr, src.size);
}
template <TStoreL2Hint Hint, typename G, typename T> void TSTORE(G dst, T& tile) {
    if (dst.size != tile.ColMaskInternal) throw std::runtime_error("MTE store extent");
    const int slot = tile.address / 8192;
    if (pending[slot]) throw std::runtime_error("stage still in use");
    pending[slot] = 1;
    peak = std::max(peak, ++outstanding);
    std::memcpy(dst.ptr, ub + tile.address, dst.size);
}
}

namespace pto::comm {
enum class DmaEngine { URMA };
struct AsyncSession { uint64_t contextGm; };
namespace urma {
struct UrmaInfo {} info;
struct Wq {
    uint32_t wqeShiftSize = 6, depth, submittedWqeCount = 0;
    uint64_t headAddr, bufAddr;
} wq;
struct Cq { uint32_t depth = 8; } cq;
struct Mem {
    uint64_t len = 1ULL << 40, addr = 0x300000000ULL, eidAddr;
    uint32_t tokenValueValid = 1, rmtJettyType = 0, targetHint = 0;
    uint32_t tpn = 1, tid = 1, rmtTokenValue = 1;
} mem;
uint32_t producer, reclaimed, completed = 0, polls = 0;
std::vector<std::pair<uint32_t, uint32_t>> doorbells;
namespace detail {
Wq *GetWqContextAt(AsyncSession, int, int) { return &wq; }
Cq *GetCqContextAt(AsyncSession, int, int) { return &cq; }
Mem *GetRemoteMemInfo(UrmaInfo *, int, int) { return &mem; }
bool UrmaRingHasRoom(Wq *, Cq *, uint32_t head, uint32_t completions,
                     uint32_t bbs, uint32_t cqes) {
    return head - reclaimed + bbs <= wq.depth && completions - completed + cqes <= cq.depth;
}
void CommitPostedWqes(Wq *, uint32_t head, uint32_t completions) {
    producer = head;
    wq.submittedWqeCount = completions;
    doorbells.emplace_back(head, completions);
}
bool UrmaPollCqAt(AsyncSession, int, int, uint32_t head, uint32_t completions, bool, bool &done) {
    ++polls;
    uint32_t requests = 0;
    auto *sq = reinterpret_cast<uint64_t *>(wq.bufAddr);
    for (uint32_t bb = reclaimed; bb != head; bb += 4) {
        const auto flags = (sq[(bb % wq.depth) * 8] >> 16) & 0x3f;
        if (flags & 0x20) {
            ++requests;
            if (bb + 4 != head || flags != 0x2d) throw std::runtime_error("intermediate CQE");
        }
    }
    if (requests != 1 || completions != completed + 1) throw std::runtime_error("CQE count");
    reclaimed = head;
    completed = completions;
    done = true;
    return true;
}
}
}
template <DmaEngine>
bool BuildAsyncSession(uint8_t *, int, AsyncSession &s) {
    s.contextGm = reinterpret_cast<uint64_t>(&urma::info);
    return true;
}
}
"""
        + template[begin:end]
        + r"""
int main(int argc, char **argv) {
    using namespace pto::comm::urma;
    if ((argc == 4 || argc == 5) && std::string(argv[1]) == "self") {
        const uint32_t width = std::strtoul(argv[2], nullptr, 0);
        const uint32_t invalid = std::strtoul(argv[3], nullptr, 0);
        const uint32_t workers = argc == 5 ? std::strtoul(argv[4], nullptr, 0) : 1;
        std::vector<uint8_t> input(7 * width), target(10 * width, 123);
        for (uint32_t i = 0; i < input.size(); ++i) input[i] = (i * 13) % 251;
        int32_t ids[] = {3, 4, 4, 0, 1, 2, 6};
        if (invalid == 1) ids[6] = 7;
        if (invalid == 2) ids[6] = -1;
        if (invalid == 3) { ids[0] = 6; ids[1] = 7; }
        try {
            for (uint32_t worker = 0; worker < workers; ++worker)
                CopyIndexedSelf(input.data(), target.data(), ids, 7, width, 7, worker, workers);
        } catch (const std::runtime_error& e) {
            std::cout << e.what() << "\n";
            return 0;
        }
        for (uint32_t r = 0; r < 7; ++r)
            for (uint32_t col = 0; col < width; ++col)
                if (target[r * width + col] != input[ids[r] * width + col]) return 4;
        for (uint32_t i = 7 * width; i < target.size(); ++i) if (target[i] != 123) return 5;
        if (outstanding != 0) return 6;
        for (int slot = 0; slot < 8; ++slot) if (ready[slot] || loaded[slot]) return 7;
        if (workers == 1 && peak != std::min<int>(8, pto::comm::self_copies.size())) return 8;
        for (auto size : pto::comm::self_copies) std::cout << size << " ";
        std::cout << "\n";
        return 0;
    }
    if (argc != 3) return 1;
    const uint32_t rows = std::strtoul(argv[1], nullptr, 0);
    wq.depth = std::strtoul(argv[2], nullptr, 0);
    producer = reclaimed = wq.depth - 4;
    wq.headAddr = reinterpret_cast<uint64_t>(&producer);
    std::vector<uint64_t> ring(wq.depth * 8, 0xdeadbeefdeadbeefULL);
    wq.bufAddr = reinterpret_cast<uint64_t>(ring.data());
    uint64_t eid[2] = {1, 2};
    mem.eidAddr = reinterpret_cast<uint64_t>(eid);
    std::vector<int32_t> indices(rows, 0);
    CommContext ctx{{0x200000000ULL, 0}, reinterpret_cast<uint64_t>(&info), 0};
    try {
        for (int round = 0; round < 3; ++round) {
            const int before = polls;
            SubmitIndexed(&ctx, 1, reinterpret_cast<uint8_t *>(0x100000000ULL),
                          reinterpret_cast<uint8_t *>(0x200000000ULL), indices.data(), rows, 256, 1);
            if (polls != before) throw std::runtime_error("submit waited for CQ");
            uint8_t local_input[17], local_target[17]{};
            for (int i = 0; i < 17; ++i) local_input[i] = i + round;
            int32_t local_index = 0;
            CopyIndexedSelf(local_input, local_target, &local_index, 1, 17, 1);
            if (std::memcmp(local_input, local_target, 17) != 0 || polls != before)
                throw std::runtime_error("local copy failed between submit and wait");
            if (rows) WaitIndexed(&ctx, 1);
        }
    } catch (const std::runtime_error &e) {
        if (!doorbells.empty() || polls != 0) return 2;
        for (auto word : ring) if (word != 0xdeadbeefdeadbeefULL) return 3;
        std::cout << e.what() << "\n";
        return 0;
    }
    std::cout << polls << " " << doorbells.size() << " " << completed << "\n";
    for (auto [head, completions] : doorbells) std::cout << head << " " << completions << "\n";
}
"""
    )
    binary = folder / "probe"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    return binary


@pytest.mark.parametrize("rows,depth", [(0, 512), (1, 512), (769, 512), (1536, 512), (1537, 512)])
def test_indexed_segmented_doorbells_single_completion(indexed_submission_probe, rows, depth):
    """Assert capacity before posting, ring each segment, and drain only the final CQE."""
    result = subprocess.run(
        [str(indexed_submission_probe), str(rows), str(depth)],
        check=True,
        capture_output=True,
        text=True,
    )
    sqes = (rows + 11) // 12
    if sqes * 4 > depth:
        assert result.stdout.strip() == "indexed_all_to_all_v: peer transfer exceeds SQ capacity"
        return
    lines = [list(map(int, line.split())) for line in result.stdout.splitlines()]
    segments = (sqes + 63) // 64
    assert lines[0] == [3 if rows else 0, 3 * segments, 3 if rows else 0]
    head = depth - 4
    expected = []
    for round_index in range(3):
        for first in range(0, sqes, 64):
            n = min(64, sqes - first)
            head += n * 4
            expected.append([head, round_index + int(first + n == sqes)])
    assert lines[1:] == expected


@pytest.mark.parametrize("row_bytes", [1, 17, 64, 4096, 65536])
def test_indexed_self_mte_coalesces_runs_and_preserves_tail(indexed_submission_probe, row_bytes):
    result = subprocess.run(
        [str(indexed_submission_probe), "self", str(row_bytes), "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = []
    for size in (2 * row_bytes, row_bytes, 3 * row_bytes, row_bytes):
        expected.extend(min(8192, size - offset) for offset in range(0, size, 8192))
    assert list(map(int, result.stdout.split())) == expected


@pytest.mark.parametrize("invalid", [1, 2, 3])
def test_indexed_self_mte_rejects_invalid_index(indexed_submission_probe, invalid):
    result = subprocess.run(
        [str(indexed_submission_probe), "self", "17", str(invalid)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "indexed_all_to_all_v: self index outside [0,Nmax)"


@pytest.mark.parametrize("row_bytes", [17, 4096, 8192, 65536])
@pytest.mark.parametrize("workers", [2, 4, 13])
def test_indexed_self_mte_pipeline_partitions_whole_rows(indexed_submission_probe, row_bytes, workers):
    result = subprocess.run(
        [str(indexed_submission_probe), "self", str(row_bytes), "0", str(workers)],
        check=True,
        capture_output=True,
        text=True,
    )
    sizes = list(map(int, result.stdout.split()))
    assert sum(sizes) == 7 * row_bytes
    assert all(0 < size <= 8192 for size in sizes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
