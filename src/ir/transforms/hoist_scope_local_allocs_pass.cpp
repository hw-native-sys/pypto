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

#include <any>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

/// True when @p stmt is ``X = tensor.create(...)``.
CallPtr AsTensorCreateAssign(const StmtPtr& stmt) {
  auto assign = As<AssignStmt>(stmt);
  if (!assign) return nullptr;
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_ || !IsOp(call, "tensor.create")) return nullptr;
  return call;
}

/// Collects every Var *defined* anywhere within a statement subtree.
///
/// A ``tensor.create`` sitting directly in a manual-scope body can only be
/// hoisted to the enclosing scope when its result shape is valid there — i.e.
/// no dimension references a value produced inside the body. This visitor
/// gathers that body-local def set. Collecting the whole subtree (rather than
/// only the top-level predecessors the codegen tracked) is a safe superset:
/// under SSA a top-level create's shape can only reference values that dominate
/// it, so a def nested in a for/if — or one appearing later — never matches.
class BodyDefCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> defs;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_) defs.insert(op->var_.get());
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->loop_var_) defs.insert(op->loop_var_.get());
    for (const auto& ia : op->iter_args_) {
      if (ia) defs.insert(ia.get());
    }
    for (const auto& rv : op->return_vars_) {
      if (rv) defs.insert(rv.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) {
      if (rv) defs.insert(rv.get());
    }
    IRVisitor::VisitStmt_(op);
  }
};

/// True when @p expr references any Var in @p vars. Shape dimensions are simple
/// integer expressions (Var / BinaryExpr / UnaryExpr / Cast), mirroring the
/// former codegen ``ExprRefsAnyOf`` helper this pass replaces.
bool ExprRefsAnyOf(const ExprPtr& expr, const std::unordered_set<const Var*>& vars) {
  if (!expr) return false;
  if (auto var = AsVarLike(expr)) return vars.count(var.get()) > 0;
  if (auto bin = As<BinaryExpr>(expr)) {
    return ExprRefsAnyOf(bin->left_, vars) || ExprRefsAnyOf(bin->right_, vars);
  }
  if (auto un = As<UnaryExpr>(expr)) return ExprRefsAnyOf(un->operand_, vars);
  if (auto cast_expr = As<Cast>(expr)) return ExprRefsAnyOf(cast_expr->operand_, vars);
  return false;
}

/// True when no dimension of the create's result shape references a body-local
/// Var — i.e. the buffer is enclosing-scope-valid and may be hoisted.
bool ShapeIsEnclosingScopeValid(const CallPtr& create, const std::unordered_set<const Var*>& body_defs) {
  auto result_type = AsTensorTypeLike(create->GetType());
  if (!result_type) return false;
  for (const auto& dim : result_type->shape_) {
    if (ExprRefsAnyOf(dim, body_defs)) return false;
  }
  return true;
}

/// Returns a copy of @p create carrying the ``hoistable_alloc`` attr. Idempotent
/// on the attr key so a re-run does not accumulate duplicates.
CallPtr WithHoistableAttr(const CallPtr& create) {
  auto copy = MutableCopy(create);
  for (auto& [k, v] : copy->attrs_) {
    if (k == kAttrHoistableAlloc) {
      v = std::any(true);
      return copy;
    }
  }
  copy->attrs_.emplace_back(kAttrHoistableAlloc, std::any(true));
  return copy;
}

/// Given the body of a manual scope, stamp every direct top-level
/// ``tensor.create`` whose shape is enclosing-scope-valid. Only the immediate
/// children of the body are considered — a create nested in a for/if within the
/// scope is not a direct-body statement and stays in place.
StmtPtr StampDirectBodyCreates(const StmtPtr& body, const std::unordered_set<const Var*>& body_defs,
                               bool* changed) {
  auto stamp_one = [&](const StmtPtr& stmt) -> StmtPtr {
    auto create = AsTensorCreateAssign(stmt);
    if (!create || !ShapeIsEnclosingScopeValid(create, body_defs)) return stmt;
    if (create->HasAttr(kAttrHoistableAlloc)) return stmt;  // already stamped
    auto assign = As<AssignStmt>(stmt);
    *changed = true;
    return std::make_shared<const AssignStmt>(assign->var_, WithHoistableAttr(create), assign->span_,
                                              assign->leading_comments_);
  };

  if (auto seq = As<SeqStmts>(body)) {
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(seq->stmts_.size());
    for (const auto& s : seq->stmts_) new_stmts.push_back(stamp_one(s));
    return std::make_shared<const SeqStmts>(std::move(new_stmts), seq->span_, seq->leading_comments_);
  }
  return stamp_one(body);
}

/// Walks an Orchestration function body, stamping the hoistable-alloc set of
/// every ``pl.manual_scope`` (``RuntimeScopeStmt`` with ``manual_ == true``).
/// Nested manual scopes are handled by recursing first: each scope stamps only
/// its own direct-body creates, computed against its own body-local def set.
class HoistAllocStamper : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    if (!op->manual_) return base;

    auto scope = As<RuntimeScopeStmt>(base);
    INTERNAL_CHECK_SPAN(scope, op->span_)
        << "Internal error: RuntimeScopeStmt mutation must yield a RuntimeScopeStmt";

    BodyDefCollector collector;
    collector.VisitStmt(scope->body_);

    bool changed = false;
    StmtPtr new_body = StampDirectBodyCreates(scope->body_, collector.defs, &changed);
    if (!changed) return base;

    auto copy = MutableCopy(scope);
    copy->body_ = std::move(new_body);
    return copy;
  }
};

}  // namespace

Pass HoistScopeLocalAllocs() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    auto new_functions = program->functions_;
    for (auto& [gvar, func] : new_functions) {
      if (!func || !func->body_) continue;
      // Only Orchestration functions carry ``pl.manual_scope`` regions whose
      // allocations the orchestration codegen hoists (issue #1697).
      if (func->func_type_ != FunctionType::Orchestration) continue;

      HoistAllocStamper stamper;
      auto new_body = stamper.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }
    if (new_functions == program->functions_) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "HoistScopeLocalAllocs", kHoistScopeLocalAllocsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
