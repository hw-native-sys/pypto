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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

/// True when @p stmt is already a compiler-inserted AUTO RuntimeScopeStmt.
/// Used to keep the pass idempotent and to avoid double-wrapping a body that
/// the DSL already expressed as an AUTO scope.
bool IsAutoScope(const StmtPtr& stmt) {
  if (!stmt) return false;
  if (auto scope = As<RuntimeScopeStmt>(stmt)) {
    return !scope->manual_;
  }
  // A user-written `with pl.auto_scope():` body may arrive as a single-statement
  // SeqStmts wrapper (before NormalizeStmtStructure collapses it). Peek through
  // so the wrap stays idempotent.
  if (auto seq = As<SeqStmts>(stmt); seq && seq->stmts_.size() == 1) {
    return IsAutoScope(seq->stmts_[0]);
  }
  return false;
}

/// Wrap @p body in an AUTO RuntimeScopeStmt (manual_ = false). Mirrors the
/// ``PTO2_SCOPE()`` block the orchestration codegen used to emit implicitly.
StmtPtr WrapAuto(const StmtPtr& body) {
  return std::make_shared<RuntimeScopeStmt>(/*manual=*/false, /*name_hint=*/"", body, body->span_);
}

/// Inserts AUTO RuntimeScopeStmt nodes around ForStmt and IfStmt bodies,
/// replicating the orchestration codegen's former implicit ``PTO2_SCOPE()``
/// wrapping. Insertion is suppressed inside a manual RuntimeScopeStmt — the
/// runtime forbids AUTO scope nested in MANUAL scope, exactly as codegen's
/// ``in_manual_scope_depth_`` counter enforced.
class InsertAutoScopeMutator : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    // A manual scope suppresses AUTO insertion within its body; track depth so
    // nested for/if bodies are left bare. AUTO scopes do not suppress nesting.
    if (op->manual_) ++manual_depth_;
    auto out = IRMutator::VisitStmt_(op);
    if (op->manual_) --manual_depth_;
    return out;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    if (manual_depth_ > 0) return base;
    auto for_stmt = As<ForStmt>(base);
    if (!for_stmt || !for_stmt->body_ || IsAutoScope(for_stmt->body_)) return base;
    auto copy = MutableCopy(for_stmt);
    copy->body_ = WrapAuto(for_stmt->body_);
    return copy;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    if (manual_depth_ > 0) return base;
    auto if_stmt = As<IfStmt>(base);
    if (!if_stmt) return base;

    bool changed = false;
    StmtPtr then_body = if_stmt->then_body_;
    if (then_body && !IsAutoScope(then_body)) {
      then_body = WrapAuto(then_body);
      changed = true;
    }
    std::optional<StmtPtr> else_body = if_stmt->else_body_;
    if (else_body.has_value() && *else_body && !IsAutoScope(*else_body)) {
      else_body = WrapAuto(*else_body);
      changed = true;
    }
    if (!changed) return base;

    auto copy = MutableCopy(if_stmt);
    copy->then_body_ = std::move(then_body);
    copy->else_body_ = std::move(else_body);
    return copy;
  }

 private:
  int manual_depth_ = 0;
};

}  // namespace

Pass MaterializeRuntimeScopes() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    // Only Orchestration functions are wrapped in PTO2_SCOPE blocks by codegen;
    // InCore/AIC/AIV/Group/Spmd bodies are never scope-wrapped.
    if (func->func_type_ != FunctionType::Orchestration) return func;

    // ``@pl.function(auto_scope=False)`` opts out of automatic AUTO-scope
    // insertion: the user places every scope by hand (``with pl.scope()`` /
    // ``pl.manual_scope()``), which the parser already materialised into the
    // IR. The simpler runtime's implicit top-level scope covers correctness, so
    // emitting zero compiler scopes is valid. Leave such functions untouched.
    if (!func->GetAttr<bool>("auto_scope", true)) return func;

    InsertAutoScopeMutator mutator;
    auto inner = mutator.VisitStmt(func->body_);

    // Always wrap the whole function body in an AUTO scope, matching the
    // always-on outermost ``PTO2_SCOPE()`` codegen emitted at function entry.
    StmtPtr new_body = IsAutoScope(inner) ? inner : WrapAuto(inner);

    // Mark the function ``auto_scope=False`` now that scopes are materialized.
    // This makes the pass idempotent (a second run early-returns) and lets the
    // output round-trip: the inserted ``with pl.scope()`` blocks parse back only
    // under ``auto_scope=False`` (the parser rejects hand-placed AUTO scopes in
    // the default auto_scope=True mode, where the compiler owns placement).
    std::vector<std::pair<std::string, std::any>> new_attrs;
    new_attrs.reserve(func->attrs_.size() + 1);
    for (const auto& kv : func->attrs_) {
      if (kv.first != "auto_scope") new_attrs.push_back(kv);
    }
    new_attrs.emplace_back("auto_scope", std::any(false));

    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, std::move(new_attrs));
  };
  return CreateFunctionPass(pass_func, "MaterializeRuntimeScopes", kMaterializeRuntimeScopesProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
