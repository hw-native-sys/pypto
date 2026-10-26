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

#ifndef SRC_IR_TRANSFORMS_MEMORY_REUSE_STORAGE_H_
#define SRC_IR_TRANSFORMS_MEMORY_REUSE_STORAGE_H_

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace memory_reuse {

/// True when two MemRefs denote the same codegen-visible byte window.
/// Allocation identity alone is insufficient: one allocation can contain
/// multiple disjoint tile windows and pipeline slots.
bool SamePhysicalWindow(const MemRefPtr& lhs, const MemRefPtr& rhs);

// Equal unresolved annotations cannot prove that two tiles occupy the same
// hardware memory space, even when their MemRef windows match.
bool SameKnownTileMemorySpace(const TileTypePtr& lhs, const TileTypePtr& rhs);

bool SameTransferDescriptor(const TileTypePtr& source, const TileTypePtr& target);
bool SameStorageDescriptor(const TileTypePtr& source, const TileTypePtr& target);

// Path-compressed equivalence classes for physical windows and parallel copies.
size_t FindSet(std::vector<size_t>& parent, size_t value);
void UnionSets(std::vector<size_t>& parent, size_t lhs, size_t rhs);

// Number the same lexical statements with else before then. Together with the
// ordinary order this distinguishes mutually exclusive branch observations.
class ReverseBranchOrderVisitor : public IRVisitor {
 public:
  std::map<const Stmt*, size_t> stmt_order;
  std::map<const Stmt*, size_t> stmt_end_order;

  void Run(const StmtPtr& body) { VisitStmt(body); }

 protected:
  // IterArg initializers are expressions, not additional lexical statements.
  void VisitExpr(const ExprPtr&) override {}

  void VisitStmt(const StmtPtr& stmt) override {
    if (!stmt) return;
    stmt_order.emplace(stmt.get(), next_order_++);
    IRVisitor::VisitStmt(stmt);
    stmt_end_order.emplace(stmt.get(), next_order_ - 1);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    if (op->else_body_.has_value()) VisitStmt(*op->else_body_);
    VisitStmt(op->then_body_);
    for (const auto& return_var : op->return_vars_) VisitExpr(return_var);
  }

 private:
  size_t next_order_ = 0;
};

// One collection and a monotonically increasing suffix avoid repeated name
// scans when fanout transfers share a source or reconciliation runs after reuse.
class StorageNameSupply : public IRVisitor {
 public:
  void Collect(const StmtPtr& body, const std::vector<VarPtr>& parameters) {
    for (const auto& parameter : parameters) VisitVarLike_(parameter);
    VisitStmt(body);
  }
  std::string Fresh(const std::string& prefix) {
    std::string result;
    do {
      result = prefix + std::to_string(counter_++);
    } while (!names_.insert(result).second);
    return result;
  }

 protected:
  void VisitVarLike_(const VarPtr& value) override {
    names_.insert(value->name_hint_);
    if (auto memref = GetTypeMemRef(value->GetType()); memref && *memref && (*memref)->base_) {
      names_.insert((*memref)->base_->name_hint_);
    }
  }
  void VisitExpr_(const IterArgPtr& value) override { VisitVarLike_(value); }
  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }

 private:
  std::set<std::string> names_;
  size_t counter_ = 0;
};

// Bind composed dynamic slice coordinates at their lexical definitions so
// boundary transfers and assemble snapshots can use one dominating root view.
class BufferWindowGeometry {
 public:
  BufferWindowGeometry(const BufferViewAnalysis& views, StorageNameSupply& names)
      : views_(views), names_(names) {}

  std::vector<ExprPtr> RootOffsets(const BufferViewAnalysis::View& view, const Span& span) const;
  StmtPtr Index(const AssignStmtPtr& assign, const CallPtr& call);

 private:
  const BufferViewAnalysis& views_;
  StorageNameSupply& names_;
  std::unordered_map<const Call*, std::vector<ExprPtr>> root_offsets_;
};

// Cache each carry at its binding, including unchanged carries. The general
// mutator removes these mappings on loop exit; repeated body uses then remain
// constant-time instead of expanding nested initializer chains.
class StorageBoundaryMutator : public IRMutator {
 protected:
  ExprPtr VisitExpr_(const IterArgPtr& argument) override {
    const auto* context = PassContext::Current();
    if (context == nullptr || !context->GetEnableBufferIR()) return IRMutator::VisitExpr_(argument);
    auto found = var_remap_.find(argument.get());
    if (found != var_remap_.end()) return found->second;
    auto result = IRMutator::VisitExpr_(argument);
    var_remap_[argument.get()] = result;
    return result;
  }
};

// Emit one explicit storage transfer with the caller's expression remapping.
std::pair<VarPtr, StmtPtr> CreateStorageMove(const VarPtr& source, const MemRefPtr& target_memref,
                                             std::optional<MemorySpace> target_memory,
                                             const std::function<ExprPtr(const ExprPtr&)>& visit_expr,
                                             const std::string& name);
bool IsA5Target();
bool IsA5Prelu(const CallPtr& call);
const Var* TileMemRefBase(const VarPtr& value);

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_MEMORY_REUSE_STORAGE_H_
