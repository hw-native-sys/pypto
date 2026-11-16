/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

// CPU contract test for the generated AIV wrapper, not a hardware CCU simulator.
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#define __gm__
#define __aicore__
struct Tensor {
  struct {
    uint64_t addr;
  } buffer;
  uint64_t start_offset;
  uint64_t shapes[2];
};
struct CommContext {
  uint32_t rankId;
  uint32_t rankNum;
  uint64_t windowsIn[TEST_RANKS];
};
struct AsyncCtx {
  int64_t* original;
};
inline AsyncCtx get_async_ctx(int64_t* args) { return {args}; }
constexpr int SIMPLER_ERROR_ASYNC_COMPLETION_INVALID = 1;
namespace pto2::detail {
inline void defer_error(AsyncCtx&, int) { assert(false); }
}  // namespace pto2::detail
namespace simpler::ccu {
struct CcuCallArg {
  enum class Kind : uint32_t { Scalar, Window };
  Kind kind;
  uint32_t rank;
  uint64_t value;
  uint64_t size;
};
}  // namespace simpler::ccu
using Arg = simpler::ccu::CcuCallArg;
std::array<Arg, 8 + 2 * TEST_RANKS> captured;
int64_t* original_args;
inline bool submit_ccu_mission(AsyncCtx& ctx, const CommContext* comm, uint32_t id,
                               const Arg (&args)[8 + 2 * TEST_RANKS]) {
  assert(ctx.original == original_args);
  assert(comm == reinterpret_cast<const CommContext*>(original_args[4]));
  assert(id == 7);
  for (std::size_t i = 0; i < captured.size(); ++i) captured[i] = args[i];
  return true;
}

#include "generated_kernel.inc"

void CheckLargeRow(uint64_t bytes) {
  // AIV only packs metadata, so a small backing allocation is sufficient to
  // test large logical shapes without reserving multi-GiB windows on the CPU.
  std::array<uint16_t, 512> window{};
  const auto base = reinterpret_cast<uint64_t>(window.data());
  const uint64_t count = bytes / 2;
  Tensor input{{base + 64}, 3, {1, count}};
  Tensor target{{base + 512}, 5, {TEST_RANKS, count}};
  CommContext comm{};
  comm.rankId = TEST_RANKS - 1;
  comm.rankNum = TEST_RANKS;
  for (auto& address : comm.windowsIn) address = base;
  int64_t args[]{reinterpret_cast<int64_t>(&input), reinterpret_cast<int64_t>(&target), 0, 0,
                 reinterpret_cast<int64_t>(&comm)};
  original_args = args;
  kernel_entry(args);
  assert(captured[1].value == bytes && captured[2].value == bytes * (TEST_RANKS - 1));
  assert(captured[3].value == bytes / pypto_ccu::kSegmentBytes);
  assert(captured[5].value <= 8191);
  assert(captured[4].value == captured[5].value * 16 * 32768);
  const auto config = captured[6].value;
  const uint64_t repeat = (config >> 55) & 127;
  const uint64_t index = (config >> 48) & 127;
  const uint64_t total = (config >> 41) & 127;
  const uint64_t residual = captured[7].value;
  uint64_t remainder = 0;
  if (config != 0) {
    assert(total == 1 || total == 2);
    assert(index == total - 1 && repeat + total <= 16);
    assert(residual > 0 && residual <= 32768);
    remainder = total == 1 ? (repeat + 1) * residual : residual + (repeat + 1) * 32768;
  }
  assert(captured[4].value + remainder == bytes % pypto_ccu::kSegmentBytes);
}

int main() {
  for (uint32_t rank = 0; rank < TEST_RANKS; ++rank) {
    for (uint64_t count : {7U, 64U, 16384U, 16391U, 262144U, 262151U, 278528U}) {
      if (count > 64 && rank != 0 && rank + 1 != TEST_RANKS) continue;
      std::vector<uint16_t> window(512 + (TEST_RANKS + 1) * count);
      const auto base = reinterpret_cast<uint64_t>(window.data());
      // Nonzero buffer displacement and start_offset catch byte/element confusion.
      Tensor input{{base + 64}, 3, {1, count}};
      Tensor target{{base + 512 + count * 2}, 5, {TEST_RANKS, count}};
      auto* src = reinterpret_cast<uint16_t*>(input.buffer.addr) + input.start_offset;
      auto* dst = reinterpret_cast<uint16_t*>(target.buffer.addr) + target.start_offset;
      for (uint64_t i = 0; i < count; ++i) src[i] = static_cast<uint16_t>(0x7f80 + i);
      for (uint64_t i = 0; i < TEST_RANKS * count; ++i) dst[i] = 0xdead;
      CommContext comm{};
      comm.rankId = rank;
      comm.rankNum = TEST_RANKS;
      for (auto& address : comm.windowsIn) address = base;
      int64_t args[]{reinterpret_cast<int64_t>(&input), reinterpret_cast<int64_t>(&target), 0, 0,
                     reinterpret_cast<int64_t>(&comm)};
      original_args = args;
      kernel_entry(args);
      for (uint64_t i = 0; i < TEST_RANKS * count; ++i) assert(dst[i] == 0xdead);
      assert(captured[0].kind == Arg::Kind::Window && captured[0].rank == rank);
      assert(captured[0].value == 70 && captured[0].size == 2 * count);
      assert(captured[1].kind == Arg::Kind::Scalar && captured[1].value == 2 * count);
      assert(captured[2].kind == Arg::Kind::Scalar && captured[2].value == rank * 2 * count);
      const auto go = pypto_ccu::CalcGoSize(count * 2);
      assert(captured[3].kind == Arg::Kind::Scalar && captured[3].value == 0);
      assert(captured[4].value == go.addr_offset && captured[5].value == go.iterations);
      assert(captured[6].value == go.parallel && captured[7].value == go.residual);
      for (uint32_t peer = 0; peer < TEST_RANKS; ++peer) {
        const auto& window_arg = captured[8 + 2 * peer];
        const auto& rank_arg = captured[9 + 2 * peer];
        assert(window_arg.kind == Arg::Kind::Window && window_arg.rank == peer);
        assert(window_arg.value == 522 + count * 2 && window_arg.size == TEST_RANKS * 2 * count);
        assert(rank_arg.kind == Arg::Kind::Scalar && rank_arg.value == peer);
      }
    }
  }
  for (uint64_t bytes : {pypto_ccu::kSegmentBytes - 2, pypto_ccu::kSegmentBytes, pypto_ccu::kSegmentBytes + 2,
                         2 * pypto_ccu::kSegmentBytes + 32770}) {
    CheckLargeRow(bytes);
  }
  // Exact-limit GoSize must move the overflowing 8192nd stripe to remainder.
  constexpr auto maximum = pypto_ccu::CalcGoSize(pypto_ccu::kSegmentBytes);
  static_assert(maximum.iterations == 8191 && maximum.residual == 32768);
  static_assert(maximum.parallel == ((uint64_t{14} << 55) | (uint64_t{1} << 48) | (uint64_t{2} << 41)));
}
