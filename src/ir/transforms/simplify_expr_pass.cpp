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

/*
 * The arithmetic simplification module takes reference from:
 * - Apache TVM (https://github.com/apache/tvm), Apache License 2.0
 * - MLC-Python (https://github.com/mlc-ai/mlc-python), Apache License 2.0
 */

#include <memory>
#include <optional>
#include <vector>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/ir_mutator_with_analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {

namespace {

/// Mutator that simplifies all scalar expressions in the IR.
///
/// Overrides statement visitors to apply Analyzer::Simplify to each
/// expression field. The base class IRMutatorWithAnalyzer automatically
/// binds ForStmt loop variables to their ranges, enabling range-aware
/// simplifications (e.g., i // 8 == 0 when i in [0, 8)).
///
/// Note: We override statement visitors rather than VisitExpr because
/// IRMutator's child visiting uses qualified calls (ExprFunctor::VisitExpr)
/// that bypass VisitExpr overrides. Analyzer::Simplify handles deep
/// recursive simplification of the entire expression tree.
class SimplifyExprMutator : public arith::IRMutatorWithAnalyzer {
 public:
  explicit SimplifyExprMutator(arith::Analyzer* analyzer) : IRMutatorWithAnalyzer(analyzer) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = analyzer_->Simplify(op->value_);
    if (new_value.get() == op->value_.get()) return op;
    return std::make_shared<AssignStmt>(op->var_, new_value, op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Simplify loop bounds.
    auto new_start = analyzer_->Simplify(op->start_);
    auto new_stop = analyzer_->Simplify(op->stop_);
    auto new_step = analyzer_->Simplify(op->step_);

    // Bind loop variable to simplified range [start, stop).
    auto start_ci = As<ConstInt>(new_start);
    auto stop_ci = As<ConstInt>(new_stop);
    if (start_ci && stop_ci && stop_ci->value_ > start_ci->value_) {
      analyzer_->Bind(op->loop_var_, start_ci->value_, stop_ci->value_);
    }

    // Visit iter_args (their init values may contain simplifiable expressions).
    std::vector<IterArgPtr> new_iter_args;
    bool iter_args_changed = false;
    new_iter_args.reserve(op->iter_args_.size());
    for (const auto& iter_arg : op->iter_args_) {
      auto new_init = analyzer_->Simplify(iter_arg->initValue_);
      if (new_init.get() != iter_arg->initValue_.get()) {
        auto new_ia =
            std::make_shared<IterArg>(iter_arg->name_hint_, iter_arg->GetType(), new_init, iter_arg->span_);
        new_iter_args.push_back(new_ia);
        var_remap_[iter_arg.get()] = new_ia;
        iter_args_changed = true;
      } else {
        new_iter_args.push_back(iter_arg);
      }
    }

    // Visit body with bindings active.
    auto new_body = VisitStmt(op->body_);

    // Clean up iter_arg remappings.
    for (const auto& old_ia : op->iter_args_) {
      var_remap_.erase(old_ia.get());
    }

    // Visit chunk_size if present.
    std::optional<ExprPtr> new_chunk_size = op->chunk_size_;
    bool chunk_size_changed = false;
    if (op->chunk_size_.has_value()) {
      auto new_cs = analyzer_->Simplify(*op->chunk_size_);
      if (new_cs.get() != (*op->chunk_size_).get()) {
        new_chunk_size = new_cs;
        chunk_size_changed = true;
      }
    }

    bool changed = (new_start.get() != op->start_.get()) || (new_stop.get() != op->stop_.get()) ||
                   (new_step.get() != op->step_.get()) || iter_args_changed ||
                   (new_body.get() != op->body_.get()) || chunk_size_changed;
    if (!changed) return op;

    return std::make_shared<ForStmt>(op->loop_var_, new_start, new_stop, new_step, new_iter_args, new_body,
                                     op->return_vars_, op->span_, op->kind_, new_chunk_size,
                                     op->chunk_policy_, op->loop_origin_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_condition = analyzer_->Simplify(op->condition_);

    // Enter constraint scope for then branch (condition is known true).
    StmtPtr new_then;
    {
      auto ctx = analyzer_->GetConstraintContext(new_condition);
      new_then = VisitStmt(op->then_body_);
    }

    // Enter constraint scope for else branch (condition is known false → Not(condition)).
    std::optional<StmtPtr> new_else;
    if (op->else_body_.has_value()) {
      auto ctx = analyzer_->GetConstraintContext(MakeNot(new_condition, new_condition->span_));
      new_else = VisitStmt(*op->else_body_);
    }

    bool changed = (new_condition.get() != op->condition_.get()) ||
                   (new_then.get() != op->then_body_.get()) ||
                   (new_else.has_value() != op->else_body_.has_value()) ||
                   (new_else.has_value() && new_else->get() != op->else_body_->get());
    if (!changed) return op;
    return std::make_shared<IfStmt>(new_condition, new_then, new_else, op->return_vars_, op->span_);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto new_condition = analyzer_->Simplify(op->condition_);

    // Simplify iter_args init values.
    std::vector<IterArgPtr> new_iter_args;
    bool iter_args_changed = false;
    new_iter_args.reserve(op->iter_args_.size());
    for (const auto& iter_arg : op->iter_args_) {
      auto new_init = analyzer_->Simplify(iter_arg->initValue_);
      if (new_init.get() != iter_arg->initValue_.get()) {
        auto new_ia =
            std::make_shared<IterArg>(iter_arg->name_hint_, iter_arg->GetType(), new_init, iter_arg->span_);
        new_iter_args.push_back(new_ia);
        var_remap_[iter_arg.get()] = new_ia;
        iter_args_changed = true;
      } else {
        new_iter_args.push_back(iter_arg);
      }
    }

    auto new_body = VisitStmt(op->body_);

    // Clean up iter_arg remappings.
    for (const auto& old_ia : op->iter_args_) {
      var_remap_.erase(old_ia.get());
    }

    bool changed = (new_condition.get() != op->condition_.get()) || iter_args_changed ||
                   (new_body.get() != op->body_.get());
    if (!changed) return op;
    return std::make_shared<WhileStmt>(new_condition, new_iter_args, new_body, op->return_vars_, op->span_);
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    bool changed = false;
    new_values.reserve(op->value_.size());
    for (const auto& val : op->value_) {
      auto new_val = analyzer_->Simplify(val);
      new_values.push_back(new_val);
      if (new_val.get() != val.get()) changed = true;
    }
    if (!changed) return op;
    return std::make_shared<ReturnStmt>(new_values, op->span_);
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    bool changed = false;
    new_values.reserve(op->value_.size());
    for (const auto& val : op->value_) {
      auto new_val = analyzer_->Simplify(val);
      new_values.push_back(new_val);
      if (new_val.get() != val.get()) changed = true;
    }
    if (!changed) return op;
    return std::make_shared<YieldStmt>(new_values, op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto new_expr = analyzer_->Simplify(op->expr_);
    if (new_expr.get() == op->expr_.get()) return op;
    return std::make_shared<EvalStmt>(new_expr, op->span_);
  }
};

FunctionPtr TransformSimplifyExpr(const FunctionPtr& func) {
  auto analyzer = std::make_shared<arith::Analyzer>();
  SimplifyExprMutator mutator(analyzer.get());
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_);
}

}  // namespace

namespace pass {

Pass SimplifyExpr() {
  return CreateFunctionPass(TransformSimplifyExpr, "SimplifyExpr", kSimplifyExprProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
