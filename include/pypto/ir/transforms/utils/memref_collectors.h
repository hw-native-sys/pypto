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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_MEMREF_COLLECTORS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_MEMREF_COLLECTORS_H_

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace memref_collectors {

// ============================================================================
// Visitor-based collector
// ============================================================================

/// Collects unique (MemRefPtr, MemorySpace) pairs from TileType variables.
///
/// Requires that each TileType carries both memref_ and memory_space_.
/// Deduplicates by canonical (MemRef*, MemorySpace) — each MemRef appears at
/// most once; conflicting memory spaces for the same MemRef trigger a CHECK.
///
/// Set skip_ddr=true to exclude DDR MemRefs during collection.
class MemRefWithSpaceCollector : public IRVisitor {
 public:
  explicit MemRefWithSpaceCollector(bool skip_ddr = false) : skip_ddr_(skip_ddr) {}

  std::vector<std::pair<MemRefPtr, MemorySpace>> memrefs;

  void VisitVarLike_(const VarPtr& op) override {
    if (auto tile_type = GetTileTypeWithMemRef(op->GetType())) {
      auto memory_space = tile_type->GetMemorySpace();
      CHECK(memory_space.has_value())
          << "TileType with MemRef must have memory_space before MemRef collection";
      CHECK(tile_type->memref_.has_value()) << "TileType must carry MemRef before MemRef collection";
      const MemorySpace canonical_space = memory_space.value();
      if (skip_ddr_ && canonical_space == MemorySpace::DDR) return;

      const auto& memref = tile_type->memref_.value();
      if (TryRegisterUniqueMemRef(memref, canonical_space, seen_ptrs_)) {
        memrefs.emplace_back(memref, canonical_space);
      }
    }
  }

 private:
  bool skip_ddr_;
  std::map<const MemRef*, MemorySpace> seen_ptrs_;
};

// ============================================================================
// Free-function collectors
// ============================================================================

/// Collect all unique (MemRef, MemorySpace) pairs from TileType variables
/// in a statement subtree.
inline std::vector<std::pair<MemRefPtr, MemorySpace>> CollectMemRefsWithSpace(const StmtPtr& stmt) {
  MemRefWithSpaceCollector collector;
  collector.VisitStmt(stmt);
  return std::move(collector.memrefs);
}

/// Collect non-DDR (MemRef, MemorySpace) pairs from TileType variables
/// in a statement subtree.
inline std::vector<std::pair<MemRefPtr, MemorySpace>> CollectNonDDRMemRefsWithSpace(const StmtPtr& stmt) {
  MemRefWithSpaceCollector collector(/*skip_ddr=*/true);
  collector.VisitStmt(stmt);
  return std::move(collector.memrefs);
}

/// Collect unique MemRefPtrs from any ShapedType variable (TensorType or
/// TileType) in an expression subtree.
///
/// Unlike MemRefWithSpaceCollector, this works on both TensorType and TileType
/// and does not require memory_space_ to be set.
inline std::set<MemRefPtr> CollectShapedTypeMemRefs(const ExprPtr& expr) {
  class Collector : public IRVisitor {
   public:
    std::set<MemRefPtr> memrefs;
    void VisitVarLike_(const VarPtr& var) override {
      if (auto shaped_type = As<ShapedType>(var->GetType())) {
        if (shaped_type->memref_.has_value()) memrefs.insert(*shaped_type->memref_);
      }
    }
  };
  Collector collector;
  collector.VisitExpr(expr);
  return std::move(collector.memrefs);
}

/// Collect raw pointers of base_ Var objects referenced by MemRefs in
/// TileType and TensorType variables.
/// Used to identify unused tile.alloc/tensor.alloc Ptr variables.
inline std::set<const Var*> CollectUsedBasePtrs(const StmtPtr& stmt) {
  class Collector : public IRVisitor {
   public:
    std::set<const Var*> used_bases;
    void VisitVarLike_(const VarPtr& op) override {
      if (auto tile_type = GetTileTypeWithMemRef(op->GetType())) {
        used_bases.insert(GetDefinedMemRef(tile_type)->base_.get());
      } else if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(op->GetType())) {
        if (tensor_type->memref_.has_value()) {
          used_bases.insert(tensor_type->memref_.value()->base_.get());
        }
      }
    }
  };
  Collector collector;
  collector.VisitStmt(stmt);
  return std::move(collector.used_bases);
}

}  // namespace memref_collectors
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_MEMREF_COLLECTORS_H_
