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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reporter/report.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

bool IsReportableFunctionType(FunctionType func_type) {
  return func_type == FunctionType::InCore || func_type == FunctionType::AIC ||
         func_type == FunctionType::AIV;
}

// Visitor to collect MemRef objects and compute per-space high-water marks.
// Follows the same pattern as AllocatedMemoryAddrVerifier in allocate_memory_addr_pass.cpp.
class MemoryUsageCollector : public IRVisitor {
 public:
  MemoryUsageCollector() = default;

  // Increment a monotonic statement counter so each MemRef reference gets a
  // statement-order position — the standard live-range proxy.
  void VisitStmt(const StmtPtr& op) override {
    ++stmt_index_;
    IRVisitor::VisitStmt(op);
  }

  void VisitVarLike_(const VarPtr& op) override { CollectFromType(op->GetType()); }

  struct SpaceStats {
    uint64_t high_water = 0;
    uint32_t count = 0;
  };

  // Per-base (allocation) aggregate. View MemRefs sharing a base are folded in:
  // offset = min member byte_offset (base address), size = max member size
  // (slot), live range = union of member references.
  struct BufferInfo {
    std::string name;
    MemorySpace space = MemorySpace::DDR;
    bool allocated = false;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t live_start = 0;
    uint32_t live_end = 0;
  };

  [[nodiscard]] const std::unordered_map<MemorySpace, SpaceStats>& GetStats() const { return stats_; }
  [[nodiscard]] const std::map<const Var*, BufferInfo>& GetBuffers() const { return buffers_; }

 private:
  uint32_t stmt_index_ = 0;
  std::set<const MemRef*> seen_;
  std::unordered_map<MemorySpace, SpaceStats> stats_;
  std::map<const Var*, BufferInfo> buffers_;  // keyed by base_ Ptr identity

  void CollectFromType(const TypePtr& type) {
    auto tile_type = std::dynamic_pointer_cast<const TileType>(type);
    if (!tile_type || !tile_type->memref_.has_value()) return;

    auto memory_space = tile_type->GetMemorySpace();
    if (!memory_space.has_value() || *memory_space == MemorySpace::DDR) return;

    const auto& memref = tile_type->memref_.value();
    const MemorySpace space = *memory_space;

    auto const_offset = std::dynamic_pointer_cast<const ConstInt>(memref->byte_offset_);
    const bool member_allocated = const_offset && const_offset->value_ >= 0;
    const uint64_t member_offset = member_allocated ? static_cast<uint64_t>(const_offset->value_) : 0;
    const uint64_t member_size = memref->size_;

    // Per-base aggregation runs on EVERY occurrence so live ranges extend across
    // all references, not just the first.
    const Var* base_key = memref->base_ ? memref->base_.get() : memref.get();
    auto [it, inserted] = buffers_.try_emplace(base_key);
    BufferInfo& buf = it->second;
    if (inserted) {
      buf.space = space;
      buf.name = memref->base_ ? memref->base_->name_hint_ : memref->name_hint_;
      buf.allocated = member_allocated;
      buf.offset = member_offset;
      buf.size = member_size;
      buf.live_start = stmt_index_;
      buf.live_end = stmt_index_;
    } else {
      buf.live_start = std::min(buf.live_start, stmt_index_);
      buf.live_end = std::max(buf.live_end, stmt_index_);
      // Slot size = largest view of this base. The root MemRef (offset 0) is
      // sized to the full alloc, so this matches the allocator's slot sizing
      // (see AllocateMemoryAddresses); sub-tile-only bases under-report.
      buf.size = std::max(buf.size, member_size);
      if (member_allocated) {
        // Track the root (smallest) address as the base address.
        if (!buf.allocated || member_offset < buf.offset) buf.offset = member_offset;
        buf.allocated = true;
      }
    }

    // Per-space high-water + count, deduplicated by MemRef pointer.
    if (!seen_.insert(memref.get()).second) return;
    auto& s = stats_[space];
    s.count++;
    if (member_allocated) {
      uint64_t end = member_offset + member_size;
      if (end > s.high_water) s.high_water = end;
    } else {
      // Address not yet allocated — use size as a lower bound
      if (member_size > s.high_water) s.high_water = member_size;
    }
  }
};

// Concrete generator (analogous to AllocatedMemoryAddrPropertyVerifierImpl)
class MemoryReportGeneratorImpl : public ReportGenerator {
 public:
  [[nodiscard]] std::string GetName() const override { return "MemoryReportGenerator"; }

  std::vector<ReportPtr> Generate(const Pass& pass, const ProgramPtr& program) override {
    std::vector<ReportPtr> reports;
    if (!program) return reports;

    const backend::Backend* be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;

    static constexpr MemorySpace kSpaceOrder[] = {MemorySpace::Vec, MemorySpace::Mat, MemorySpace::Left,
                                                  MemorySpace::Right, MemorySpace::Acc};

    std::vector<MemoryReport::FunctionMemoryUsage> functions;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (!IsReportableFunctionType(func->func_type_)) continue;

      MemoryUsageCollector collector;
      collector.VisitStmt(func->body_);

      const auto& stats = collector.GetStats();
      if (stats.empty()) continue;

      std::vector<MemoryReport::MemorySpaceUsage> entries;
      for (auto space : kSpaceOrder) {
        auto it = stats.find(space);
        if (it == stats.end()) continue;

        uint64_t limit = be ? be->GetMemSize(space) : 0;
        entries.push_back({space, it->second.high_water, limit, it->second.count});
      }

      if (entries.empty()) continue;

      // Assemble per-base buffer detail, ordered by (space order, address).
      auto space_rank = [&](MemorySpace s) -> int {
        for (int i = 0; i < static_cast<int>(std::size(kSpaceOrder)); ++i) {
          if (kSpaceOrder[i] == s) return i;
        }
        return static_cast<int>(std::size(kSpaceOrder));
      };
      std::vector<MemoryReport::BufferDetail> buffers;
      buffers.reserve(collector.GetBuffers().size());
      for (const auto& [base, info] : collector.GetBuffers()) {
        buffers.push_back(
            {info.name, info.space, info.allocated, info.offset, info.size, info.live_start, info.live_end});
      }
      std::sort(buffers.begin(), buffers.end(),
                [&](const MemoryReport::BufferDetail& a, const MemoryReport::BufferDetail& b) {
                  int ra = space_rank(a.space);
                  int rb = space_rank(b.space);
                  if (ra != rb) return ra < rb;
                  if (a.offset != b.offset) return a.offset < b.offset;
                  // Tie-break by name so the order does not depend on the
                  // pointer-keyed map iteration (deterministic reports).
                  return a.name < b.name;
                });

      functions.push_back({func->name_, std::move(entries), std::move(buffers)});
    }

    if (!functions.empty()) {
      std::string backend_name = be ? be->GetTypeName() : "N/A";
      reports.push_back(
          std::make_unique<MemoryReport>(pass.GetName(), std::move(backend_name), std::move(functions)));
    }

    return reports;
  }
};

}  // namespace

ReportGeneratorPtr CreateMemoryReportGenerator() { return std::make_shared<MemoryReportGeneratorImpl>(); }

}  // namespace ir
}  // namespace pypto
