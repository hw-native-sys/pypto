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

#include "src/codegen/pto/pto_scalar_floor_div_mod_plan.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto::codegen {
namespace {

using ir::arith::detail::ConstIntFitsDtype;

bool IsPositiveLiteral(const ir::ExprPtr& expr, DataType dtype) {
  const auto literal = ir::As<ir::ConstInt>(expr);
  // A stored positive value can become negative when emitted at a narrower dtype.
  return literal && literal->value_ > 0 && ConstIntFitsDtype(literal->value_, dtype);
}

class ScalarFloorDivModPlanner : public ir::IRVisitor {
 public:
  std::unordered_set<const ir::Expr*> Prepare(const ir::StmtPtr& body,
                                              const std::vector<ir::VarPtr>& params) {
    for (const auto& param : params) VisitExpr(param);
    if (body) VisitStmt(body);

    std::unordered_set<const ir::Expr*> truncating;
    for (const auto& [expr, uses] : truncation_) {
      if (uses.all_exact) truncating.insert(expr);
    }
    return truncating;
  }

 protected:
  void VisitStmt_(const ir::ForStmtPtr& loop) override {
    INTERNAL_CHECK_SPAN(loop->loop_var_ && loop->body_, loop->span_)
        << "Internal error: ForStmt requires a loop variable and body";
    // Bounds and initial carry values are evaluated outside the loop body.
    VisitExpr(loop->start_);
    VisitExpr(loop->stop_);
    VisitExpr(loop->step_);
    for (const auto& arg : loop->iter_args_) VisitExpr(arg);

    const auto start = ir::As<ir::ConstInt>(loop->start_);
    const auto* var = loop->loop_var_.get();
    // Device scf.for requires a positive step. The literal must fit its own
    // emitted width before the separate cast to the loop's index type.
    const bool inserted = start && ConstIntFitsDtype(start->value_, start->dtype()) &&
                          loop_lower_bounds_.try_emplace(var, start->value_).second;
    VisitStmt(loop->body_);
    if (inserted) loop_lower_bounds_.erase(var);

    for (const auto& result : loop->return_vars_) VisitExpr(result);
  }

  void VisitVarLike_(const ir::VarPtr& var) override {
    // Entry view operands and tile allocation extents/addresses also go
    // through the scalar emitter, even when they occur only in type metadata.
    ir::var_collectors::VisitTypeExprFields(*this, var->GetType());
    if (const auto tile = ir::As<ir::TileType>(var->GetType())) {
      const ir::MemRefPtr memref = tile->memref_.value_or(nullptr);
      if (memref && memref->byte_offset_) VisitExpr(memref->byte_offset_);
    }
  }

  void VisitExpr_(const ir::FloorDivPtr& op) override { VisitFloorOp(op); }
  void VisitExpr_(const ir::FloorModPtr& op) override { VisitFloorOp(op); }

 private:
  void VisitFloorOp(const ir::BinaryExprPtr& op) {
    VisitBinaryExpr_(op);

    const auto dtype = ir::GetScalarDtype(op);
    if (dtype == DataType::INDEX || dtype.IsSignedInt()) {
      const bool exact = IsPositiveLiteral(op->right_, dtype) && IsNonNegative(op->left_, dtype);
      // Nodes can be shared. Select truncation only when the proof holds
      // in every scope where the node is used.
      auto& uses = truncation_[op.get()];
      uses.latest_exact = exact;
      uses.all_exact &= exact;
    }
  }

  [[nodiscard]] bool IsNonNegative(const ir::ExprPtr& expr, DataType dtype) const {
    if (const auto literal = ir::As<ir::ConstInt>(expr)) {
      return literal->value_ >= 0 && ConstIntFitsDtype(literal->value_, dtype);
    }
    // Check every intermediate operation at its own width. Narrowing can
    // change the sign even when the original induction variable is positive.
    if (ir::GetScalarDtype(expr) != dtype) return false;
    if (const auto var = ir::AsVarLike(expr)) {
      return HasLowerBound(var, 0);
    }
    // For 0 <= offset <= loop start, iv - offset is bounded by [0, iv].
    if (const auto sub = ir::As<ir::Sub>(expr)) {
      const auto var = ir::AsVarLike(sub->left_);
      const auto offset = ir::As<ir::ConstInt>(sub->right_);
      return var && ir::GetScalarDtype(var) == dtype && offset && offset->value_ >= 0 &&
             ConstIntFitsDtype(offset->value_, dtype) && HasLowerBound(var, offset->value_);
    }
    // A proven floor quotient or remainder is itself non-negative. The
    // operand was just visited in this scope, so an indexed lookup suffices
    // for a nested floor operation. Sums and products can overflow.
    const auto proof = truncation_.find(expr.get());
    return proof != truncation_.end() && proof->second.latest_exact;
  }

  [[nodiscard]] bool HasLowerBound(const ir::VarPtr& var, int64_t minimum) const {
    const auto bound = loop_lower_bounds_.find(var.get());
    return bound != loop_lower_bounds_.end() && bound->second >= minimum;
  }

  struct Uses {
    bool latest_exact = false;
    bool all_exact = true;
  };

  std::unordered_map<const ir::Var*, int64_t> loop_lower_bounds_;
  std::unordered_map<const ir::Expr*, Uses> truncation_;
};

}  // namespace

std::unordered_set<const ir::Expr*> PrepareScalarFloorDivMod(const ir::StmtPtr& body,
                                                             const std::vector<ir::VarPtr>& params) {
  return ScalarFloorDivModPlanner().Prepare(body, params);
}

}  // namespace pypto::codegen
