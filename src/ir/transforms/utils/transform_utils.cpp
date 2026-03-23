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

#include "pypto/ir/transforms/utils/transform_utils.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"

namespace pypto::ir::transform_utils {

namespace {

// ---------------------------------------------------------------------------
// SubstituteExpr helpers
// ---------------------------------------------------------------------------

/// Reconstruct a BinaryExpr with new operands, dispatching on ObjectKind.
ExprPtr ReconstructBinaryExpr(ObjectKind kind, const ExprPtr& left, const ExprPtr& right, DataType dtype,
                              const Span& span) {
  // clang-format off
  switch (kind) {
    case ObjectKind::Add:           return std::make_shared<Add>(left, right, dtype, span);
    case ObjectKind::Sub:           return std::make_shared<Sub>(left, right, dtype, span);
    case ObjectKind::Mul:           return std::make_shared<Mul>(left, right, dtype, span);
    case ObjectKind::FloorDiv:      return std::make_shared<FloorDiv>(left, right, dtype, span);
    case ObjectKind::FloorMod:      return std::make_shared<FloorMod>(left, right, dtype, span);
    case ObjectKind::FloatDiv:      return std::make_shared<FloatDiv>(left, right, dtype, span);
    case ObjectKind::Min:           return std::make_shared<Min>(left, right, dtype, span);
    case ObjectKind::Max:           return std::make_shared<Max>(left, right, dtype, span);
    case ObjectKind::Pow:           return std::make_shared<Pow>(left, right, dtype, span);
    case ObjectKind::Eq:            return std::make_shared<Eq>(left, right, dtype, span);
    case ObjectKind::Ne:            return std::make_shared<Ne>(left, right, dtype, span);
    case ObjectKind::Lt:            return std::make_shared<Lt>(left, right, dtype, span);
    case ObjectKind::Le:            return std::make_shared<Le>(left, right, dtype, span);
    case ObjectKind::Gt:            return std::make_shared<Gt>(left, right, dtype, span);
    case ObjectKind::Ge:            return std::make_shared<Ge>(left, right, dtype, span);
    case ObjectKind::And:           return std::make_shared<And>(left, right, dtype, span);
    case ObjectKind::Or:            return std::make_shared<Or>(left, right, dtype, span);
    case ObjectKind::Xor:           return std::make_shared<Xor>(left, right, dtype, span);
    case ObjectKind::BitAnd:        return std::make_shared<BitAnd>(left, right, dtype, span);
    case ObjectKind::BitOr:         return std::make_shared<BitOr>(left, right, dtype, span);
    case ObjectKind::BitXor:        return std::make_shared<BitXor>(left, right, dtype, span);
    case ObjectKind::BitShiftLeft:  return std::make_shared<BitShiftLeft>(left, right, dtype, span);
    case ObjectKind::BitShiftRight: return std::make_shared<BitShiftRight>(left, right, dtype, span);
    default:
      throw pypto::InternalError("ReconstructBinaryExpr: unsupported ObjectKind");
  }
  // clang-format on
}

/// Reconstruct a UnaryExpr with a new operand, dispatching on ObjectKind.
ExprPtr ReconstructUnaryExpr(ObjectKind kind, const ExprPtr& operand, DataType dtype, const Span& span) {
  switch (kind) {
    case ObjectKind::Abs:
      return std::make_shared<Abs>(operand, dtype, span);
    case ObjectKind::Neg:
      return std::make_shared<Neg>(operand, dtype, span);
    case ObjectKind::Not:
      return std::make_shared<Not>(operand, dtype, span);
    case ObjectKind::BitNot:
      return std::make_shared<BitNot>(operand, dtype, span);
    case ObjectKind::Cast:
      return std::make_shared<Cast>(operand, dtype, span);
    default:
      throw pypto::InternalError("ReconstructUnaryExpr: unsupported ObjectKind");
  }
}

// ---------------------------------------------------------------------------
// SubstituteStmt helper (IRMutator-based)
// ---------------------------------------------------------------------------

/// Mutator that substitutes Var/IterArg references by pointer identity.
///
/// Overrides both VisitExpr_(VarPtr) and VisitExpr_(IterArgPtr) to ensure
/// all variable references are substituted, including IterArgs used as
/// expression operands. For IterArg, initValue_ is also visited recursively.
class SubstituteVarsMutator : public IRMutator {
 public:
  explicit SubstituteVarsMutator(const std::unordered_map<const Var*, VarPtr>& var_map) : var_map_(var_map) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) {
      return it->second;
    }
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) {
      return it->second;
    }
    return IRMutator::VisitExpr_(op);
  }

 private:
  const std::unordered_map<const Var*, VarPtr>& var_map_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Public API — SubstituteExpr / SubstituteStmt
// ---------------------------------------------------------------------------

ExprPtr SubstituteExpr(const ExprPtr& expr, const std::unordered_map<const Var*, VarPtr>& var_map) {
  // Check IterArg first (inherits Var but has different ObjectKind)
  if (auto iter_arg = As<IterArg>(expr)) {
    auto it = var_map.find(iter_arg.get());
    if (it != var_map.end()) {
      return it->second;
    }
    return expr;
  }
  if (auto var = As<Var>(expr)) {
    auto it = var_map.find(var.get());
    if (it != var_map.end()) {
      return it->second;
    }
    return expr;
  }
  if (auto call = As<Call>(expr)) {
    std::vector<ExprPtr> new_args;
    new_args.reserve(call->args_.size());
    bool changed = false;
    for (const auto& arg : call->args_) {
      auto new_arg = SubstituteExpr(arg, var_map);
      new_args.push_back(new_arg);
      if (new_arg != arg) {
        changed = true;
      }
    }
    if (!changed) {
      return expr;
    }
    return std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->GetType(), call->span_);
  }
  if (auto make_tuple = As<MakeTuple>(expr)) {
    std::vector<ExprPtr> new_elements;
    new_elements.reserve(make_tuple->elements_.size());
    bool changed = false;
    for (const auto& elem : make_tuple->elements_) {
      auto new_elem = SubstituteExpr(elem, var_map);
      new_elements.push_back(new_elem);
      if (new_elem != elem) {
        changed = true;
      }
    }
    if (!changed) {
      return expr;
    }
    return std::make_shared<MakeTuple>(new_elements, make_tuple->span_);
  }
  if (auto tgi = As<TupleGetItemExpr>(expr)) {
    auto new_tuple = SubstituteExpr(tgi->tuple_, var_map);
    if (new_tuple == tgi->tuple_) {
      return expr;
    }
    return std::make_shared<TupleGetItemExpr>(new_tuple, tgi->index_, tgi->span_);
  }
  if (auto bin = As<BinaryExpr>(expr)) {
    auto new_left = SubstituteExpr(bin->left_, var_map);
    auto new_right = SubstituteExpr(bin->right_, var_map);
    if (new_left == bin->left_ && new_right == bin->right_) {
      return expr;
    }
    auto dtype = GetScalarDtype(expr);
    return ReconstructBinaryExpr(bin->GetKind(), new_left, new_right, dtype, bin->span_);
  }
  if (auto un = As<UnaryExpr>(expr)) {
    auto new_operand = SubstituteExpr(un->operand_, var_map);
    if (new_operand == un->operand_) {
      return expr;
    }
    auto dtype = GetScalarDtype(expr);
    return ReconstructUnaryExpr(un->GetKind(), new_operand, dtype, un->span_);
  }
  // For leaf expression types (ConstInt, ConstFloat, etc.), return as-is
  return expr;
}

StmtPtr SubstituteStmt(const StmtPtr& body, const std::unordered_map<const Var*, VarPtr>& var_map) {
  SubstituteVarsMutator mutator(var_map);
  return mutator.VisitStmt(body);
}

// ---------------------------------------------------------------------------
// CollectDefVars
// ---------------------------------------------------------------------------

void CollectDefVars(const StmtPtr& stmt, std::vector<VarPtr>& result) {
  if (!stmt) return;

  auto kind = stmt->GetKind();
  switch (kind) {
    case ObjectKind::AssignStmt: {
      auto assign = std::static_pointer_cast<const AssignStmt>(stmt);
      result.push_back(assign->var_);
      break;
    }
    case ObjectKind::SeqStmts: {
      auto seq = std::static_pointer_cast<const SeqStmts>(stmt);
      for (const auto& s : seq->stmts_) {
        CollectDefVars(s, result);
      }
      break;
    }
    case ObjectKind::ForStmt: {
      auto for_stmt = std::static_pointer_cast<const ForStmt>(stmt);
      CollectDefVars(for_stmt->body_, result);
      break;
    }
    case ObjectKind::WhileStmt: {
      auto while_stmt = std::static_pointer_cast<const WhileStmt>(stmt);
      CollectDefVars(while_stmt->body_, result);
      break;
    }
    case ObjectKind::IfStmt: {
      auto if_stmt = std::static_pointer_cast<const IfStmt>(stmt);
      CollectDefVars(if_stmt->then_body_, result);
      if (if_stmt->else_body_.has_value()) {
        CollectDefVars(*if_stmt->else_body_, result);
      }
      break;
    }
    case ObjectKind::ScopeStmt: {
      auto scope = std::static_pointer_cast<const ScopeStmt>(stmt);
      CollectDefVars(scope->body_, result);
      break;
    }
    default:
      // YieldStmt, ReturnStmt, EvalStmt, BreakStmt, ContinueStmt — no DEFs
      break;
  }
}

}  // namespace pypto::ir::transform_utils
