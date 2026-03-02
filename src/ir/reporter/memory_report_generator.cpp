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

#include <cstdint>
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
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Visitor to collect MemRef objects and compute per-space high-water marks.
// Follows the same pattern as AllocatedMemoryAddrVerifier in allocate_memory_addr_pass.cpp.
class MemoryUsageCollector : public IRVisitor {
 public:
  MemoryUsageCollector() = default;

  void VisitExpr_(const VarPtr& op) override { CollectFromType(op->GetType()); }

  void VisitExpr_(const IterArgPtr& op) override {
    CollectFromType(op->GetType());
    IRVisitor::VisitExpr_(op);
  }

  struct SpaceStats {
    uint64_t high_water = 0;
    uint32_t count = 0;
  };

  [[nodiscard]] const std::unordered_map<MemorySpace, SpaceStats>& GetStats() const { return stats_; }

 private:
  std::set<const MemRef*> seen_;
  std::unordered_map<MemorySpace, SpaceStats> stats_;

  void CollectFromType(const TypePtr& type) {
    auto tile_type = std::dynamic_pointer_cast<const TileType>(type);
    if (!tile_type || !tile_type->memref_.has_value()) return;

    const auto& memref = tile_type->memref_.value();
    if (memref->memory_space_ == MemorySpace::DDR) return;
    if (!seen_.insert(memref.get()).second) return;

    auto& s = stats_[memref->memory_space_];
    s.count++;

    auto const_addr = std::dynamic_pointer_cast<const ConstInt>(memref->addr_);
    if (const_addr && const_addr->value_ >= 0) {
      uint64_t end = static_cast<uint64_t>(const_addr->value_) + memref->size_;
      if (end > s.high_water) s.high_water = end;
    } else {
      // Address not yet allocated — use size as a lower bound
      if (memref->size_ > s.high_water) s.high_water = memref->size_;
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
      if (func->func_type_ != FunctionType::InCore) continue;

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

      if (!entries.empty()) {
        functions.push_back({func->name_, std::move(entries)});
      }
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
