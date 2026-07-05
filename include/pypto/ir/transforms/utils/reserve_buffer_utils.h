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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_RESERVE_BUFFER_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_RESERVE_BUFFER_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto {
namespace ir {

/// Shared `system.reserve_buffer` resolution — the SINGLE source of truth for both AllocateMemoryAddr
/// (which needs each reserve op's resolved base AND the per-space reserved end for address emission) and
/// MemoryReuse (which needs the per-space reserved end as `SpaceFootprint`'s `reserved_start`). Because
/// `reserved_start` is a direct input to the shared `SpaceFootprint` walk, resolving it here — rather than
/// re-deriving it in each pass — keeps the reserved base parity-by-construction, closing the last drift gap.
using ReserveBufferBaseMap = std::unordered_map<const Call*, int64_t>;
using ReservedEndBySpace = std::unordered_map<MemorySpace, uint64_t>;

/// `system.reserve_buffer` lives in the function's cross-core space: Mat for AIC, Vec for AIV/InCore.
inline MemorySpace GetReserveBufferMemorySpace(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "reserve_buffer resolution requires a valid function";
  switch (func->func_type_) {
    case FunctionType::AIC:
      return MemorySpace::Mat;
    case FunctionType::AIV:
    case FunctionType::InCore:
      return MemorySpace::Vec;
    default:
      INTERNAL_UNREACHABLE_SPAN(func->span_)
          << "cannot resolve reserve_buffer memory space for function '" << func->name_ << "' with type "
          << FunctionTypeToString(func->func_type_);
  }
  return MemorySpace::DDR;
}

struct ReserveBufferInfo {
  const Call* call = nullptr;
  int64_t size = 0;
  int64_t base = -1;
};

class ReserveBufferCollector : public IRVisitor {
 public:
  [[nodiscard]] const std::vector<ReserveBufferInfo>& GetReserveBuffers() const { return reserve_buffers_; }

  void VisitExpr_(const CallPtr& op) override {
    if (IsOp(op, "system.reserve_buffer")) {
      const int size = op->GetKwarg<int>("size", -1);
      const int base = op->GetKwarg<int>("base", -1);
      INTERNAL_CHECK_SPAN(size > 0, op->span_) << "reserve_buffer requires size > 0, got " << size;
      reserve_buffers_.push_back(
          ReserveBufferInfo{op.get(), static_cast<int64_t>(size), static_cast<int64_t>(base)});
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<ReserveBufferInfo> reserve_buffers_;
};

struct ReserveBufferResolution {
  ReserveBufferBaseMap resolved_bases;
  ReservedEndBySpace reserved_end_by_space;
};

/// Resolve reserve_buffer bases (auto-placed ones bump sequentially) and the per-space reserved end.
/// Validates that resolved ranges do not overlap.
inline ReserveBufferResolution ResolveReserveBufferBases(const FunctionPtr& func,
                                                         const MemoryAllocatorPolicy& policy) {
  ReserveBufferResolution resolution;
  if (!func || !func->body_) return resolution;

  ReserveBufferCollector collector;
  collector.VisitStmt(func->body_);
  if (collector.GetReserveBuffers().empty()) return resolution;

  const MemorySpace reserve_space = GetReserveBufferMemorySpace(func);

  std::unordered_map<MemorySpace, uint64_t> next_base_by_space;
  std::unordered_map<MemorySpace, std::map<uint64_t, uint64_t>> reserved_ranges_by_space;
  for (const auto& reserve : collector.GetReserveBuffers()) {
    uint64_t resolved_base = 0;
    auto& next_base = next_base_by_space[reserve_space];
    if (reserve.base >= 0) {
      resolved_base = static_cast<uint64_t>(reserve.base);
    } else {
      resolved_base = next_base;
    }

    INTERNAL_CHECK_SPAN(resolved_base <= static_cast<uint64_t>(std::numeric_limits<int>::max()), func->span_)
        << "resolved reserve_buffer base out of int range in function '" << func->name_
        << "': " << resolved_base;
    resolution.resolved_bases[reserve.call] = static_cast<int64_t>(resolved_base);

    const uint64_t buffer_end =
        policy.AlignAddress(resolved_base + static_cast<uint64_t>(reserve.size), reserve_space);
    auto& reserved_ranges = reserved_ranges_by_space[reserve_space];
    auto next_it = reserved_ranges.lower_bound(resolved_base);
    auto overlaps = [&](const std::pair<const uint64_t, uint64_t>& range) {
      return resolved_base < range.second && range.first < buffer_end;
    };
    INTERNAL_CHECK_SPAN(next_it == reserved_ranges.end() || !overlaps(*next_it), func->span_)
        << "overlapping reserve_buffer ranges in function '" << func->name_ << "': [" << resolved_base << ", "
        << buffer_end << ") overlaps with [" << next_it->first << ", " << next_it->second << ")";
    if (next_it != reserved_ranges.begin()) {
      auto prev_it = std::prev(next_it);
      INTERNAL_CHECK_SPAN(!overlaps(*prev_it), func->span_)
          << "overlapping reserve_buffer ranges in function '" << func->name_ << "': [" << resolved_base
          << ", " << buffer_end << ") overlaps with [" << prev_it->first << ", " << prev_it->second << ")";
    }
    reserved_ranges.emplace(resolved_base, buffer_end);

    next_base = std::max(next_base, buffer_end);

    // `reserved_end` is a max-END, not a free list: a reserve buffer placed high (e.g. base = 40 KB in a
    // 64 KB space) makes reused buffers start above its end, even though [0, 40 KB) is free. This matches
    // AllocateMemoryAddr's own placement (parity), so it is not a regression — but the capacity gate now
    // makes it observable as a merge decision under pressure. A packed free-list would be a separate change
    // to BOTH passes.
    auto& reserved_end = resolution.reserved_end_by_space[reserve_space];
    reserved_end = std::max(reserved_end, buffer_end);
  }

  return resolution;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_RESERVE_BUFFER_UTILS_H_
