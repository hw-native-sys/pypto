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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_INDEXED_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_INDEXED_ANALYSIS_H_

#include <cmath>
#include <cstddef>
#include <unordered_map>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto::ir {

/// The native scatter clears its destination; logical destination preservation
/// agrees with that only for a zero base. Accept a fresh full(0) with exactly
/// one use, so no alias, earlier mutation or later old-value read is hidden.
/// This O(N) index is rebuilt for selection and again after storage planning.
class BufferIndexedAnalysis : public IRVisitor {
 public:
  explicit BufferIndexedAnalysis(const StmtPtr& body) : region_(body.get()) { VisitStmt(body); }

  bool HasExclusiveZeroBase(const CallPtr& call) const {
    if (call->args_.empty()) return false;
    const auto base = AsVarLike(call->args_[0]);
    const auto use = uses_.find(base.get());
    const auto zero = zeros_.find(base.get());
    const auto region = call_regions_.find(call.get());
    return base && zero != zeros_.end() && use != uses_.end() && use->second == 1 &&
           region != call_regions_.end() && zero->second == region->second;
  }

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "tile.scatter")) call_regions_.emplace(call.get(), region_);
    IRVisitor::VisitExpr_(call);
  }
  void VisitVarLike_(const VarPtr& var) override { ++uses_[var.get()]; }
  void VisitExpr_(const IterArgPtr& var) override { VisitVarLike_(var); }
  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (const auto call = As<Call>(assign->value_);
        call && IsOp(call, "tile.full") && call->args_.size() == 2) {
      const auto integer = As<ConstInt>(call->args_[1]);
      const auto floating = As<ConstFloat>(call->args_[1]);
      if ((integer && integer->value_ == 0) ||
          (floating && floating->value_ == 0 && !std::signbit(floating->value_))) {
        zeros_.emplace(assign->var_.get(), region_);
      }
    }
    // Definitions are not uses. Call and Submit operand/dependency traversal
    // remains the base visitor's responsibility.
    VisitExpr(assign->value_);
  }
  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& arg : loop->iter_args_) VisitExpr(arg->initValue_);
    // A zero initialized outside a loop is no longer zero on its backedge.
    const auto* outer = region_;
    region_ = loop.get();
    IRVisitor::VisitStmt_(loop);
    region_ = outer;
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& arg : loop->iter_args_) VisitExpr(arg->initValue_);
    // A zero initialized outside a loop is no longer zero on its backedge.
    const auto* outer = region_;
    region_ = loop.get();
    IRVisitor::VisitStmt_(loop);
    region_ = outer;
  }

 private:
  std::unordered_map<const Var*, size_t> uses_;
  std::unordered_map<const Var*, const Stmt*> zeros_;
  std::unordered_map<const Call*, const Stmt*> call_regions_;
  const Stmt* region_;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_INDEXED_ANALYSIS_H_
