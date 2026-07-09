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

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

using ::pypto::codegen::BufferRootCollector;

/// Path of enclosing regions (loop / branch / scope bodies) for a program point.
/// Each entry identifies one region body; a definition is visible at a use iff
/// the definition's path is a prefix of the use's path.
///
/// Regions are exactly the constructs orchestration codegen renders as a C++
/// block: For/While bodies, each If branch, and every ScopeStmt body (notably
/// the ``PTO2_SCOPE(MANUAL)`` of a ``pl.manual_scope``). A name declared inside
/// such a block dies at its closing brace, so folding a copy onto a source that
/// lives deeper than one of its readers would reintroduce the out-of-scope C++
/// compile failure that the codegen band-aid papers over (issues #1697/#1713).
using ScopePath = std::vector<const void*>;

bool IsPrefix(const ScopePath& prefix, const ScopePath& path) {
  if (prefix.size() > path.size()) return false;
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (prefix[i] != path[i]) return false;
  }
  return true;
}

/// Walks a function body tracking the enclosing-region path, recording where
/// every Var is defined and where every Var is used.
///
/// Both are needed for the visibility guard: a copy ``X = Y`` may only be
/// folded when ``Y``'s defining region encloses every region in which ``X`` is
/// read.
class ScopeMapper : public IRVisitor {
 public:
  /// Definition site (region path) of each AssignStmt LHS. Function params are
  /// seeded with the empty path by Initialize().
  std::unordered_map<const Var*, ScopePath> def_scope;
  /// Every read of a Var, with the region path at the read.
  std::unordered_map<const Var*, std::vector<ScopePath>> use_scopes;

  void Initialize(const std::vector<VarPtr>& params) {
    for (const auto& p : params) def_scope[p.get()] = ScopePath{};
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_) def_scope[op->var_.get()] = path_;
    // Only the RHS contains reads; visiting the LHS would record a phantom use.
    if (op->value_) VisitExpr(op->value_);
  }

  void VisitVarLike_(const VarPtr& op) override { use_scopes[op.get()].push_back(path_); }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    for (const auto& ia : op->iter_args_) {
      if (ia->initValue_) VisitExpr(ia->initValue_);
      def_scope[ia.get()] = path_;
    }
    // return_vars are bound in the enclosing region, after the loop.
    for (const auto& rv : op->return_vars_) def_scope[rv.get()] = path_;
    EnterRegion(op.get(), op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (op->condition_) VisitExpr(op->condition_);
    for (const auto& ia : op->iter_args_) {
      if (ia->initValue_) VisitExpr(ia->initValue_);
      def_scope[ia.get()] = path_;
    }
    for (const auto& rv : op->return_vars_) def_scope[rv.get()] = path_;
    EnterRegion(op.get(), op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (op->condition_) VisitExpr(op->condition_);
    for (const auto& rv : op->return_vars_) def_scope[rv.get()] = path_;
    // Each branch is its own C++ block: give them distinct region identities so
    // a then-branch definition is never treated as visible in the else branch.
    EnterRegion(&op->then_body_, op->then_body_);
    if (op->else_body_.has_value()) EnterRegion(&op->else_body_, *op->else_body_);
  }

  // Every ScopeStmt subclass renders as its own C++ block. They are enumerated
  // explicitly because IRVisitor dispatches on the exact ObjectKind and offers
  // no ScopeStmt base hook; a new subclass must be added here.
  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitScopeLike(op); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitScopeLike(op); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitScopeLike(op); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitScopeLike(op); }
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override { VisitScopeLike(op); }
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override { VisitScopeLike(op); }
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override { VisitScopeLike(op); }

 private:
  template <typename ScopePtrT>
  void VisitScopeLike(const ScopePtrT& op) {
    VisitScopeAttrs(op);
    EnterRegion(op.get(), op->body_);
  }

  void EnterRegion(const void* id, const StmtPtr& body) {
    if (!body) return;
    path_.push_back(id);
    VisitStmt(body);
    path_.pop_back();
  }

  ScopePath path_;
};

/// Collect every Var that is a mutable loop/branch carry lvalue: the
/// ``iter_args`` / ``return_vars`` of For/While and the ``return_vars`` (phi
/// results) of If. Codegen manages these as reassigned C++ locals across
/// iterations/phases, so collapsing a copy onto (or off) one would break the
/// snapshot semantics the pipeline relies on.
class CarryVarCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> carry_vars;

 protected:
  void VisitStmt_(const ForStmtPtr& op) override {
    for (const auto& ia : op->iter_args_) carry_vars.insert(ia.get());
    for (const auto& rv : op->return_vars_) carry_vars.insert(rv.get());
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& ia : op->iter_args_) carry_vars.insert(ia.get());
    for (const auto& rv : op->return_vars_) carry_vars.insert(rv.get());
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) carry_vars.insert(rv.get());
    IRVisitor::VisitStmt_(op);
  }
};

/// Gather candidate pure-copy AssignStmts (``X = Y``, value is a Var/IterArg)
/// that alias the same physical buffer and touch no carry lvalue.
class CopyCollector : public IRVisitor {
 public:
  CopyCollector(const std::unordered_map<const Var*, const Var*>& buffer_roots,
                const std::unordered_set<const Var*>& carry_vars)
      : buffer_roots_(buffer_roots), carry_vars_(carry_vars) {}

  /// LHS Var* -> RHS value expr (a Var/IterArg node).
  std::unordered_map<const Var*, ExprPtr> copy_src;

 protected:
  void VisitStmt_(const AssignStmtPtr& assign) override {
    IRVisitor::VisitStmt_(assign);
    const Var* x = assign->var_ ? assign->var_.get() : nullptr;
    if (!x) return;
    auto y = AsVarLike(assign->value_);
    if (!y) return;  // not a pure Var/IterArg copy

    // Neither side may be a mutable loop/branch carry lvalue.
    if (carry_vars_.count(x) != 0 || carry_vars_.count(y.get()) != 0) return;

    // X and Y must denote the same physical buffer. For a pure copy this holds
    // by construction (BufferRootCollector propagates root(X) = root(Y)); check
    // explicitly so a Var with no known root is never folded.
    const Var* rx = Root(x);
    const Var* ry = Root(y.get());
    if (!rx || !ry || rx != ry) return;

    copy_src[x] = assign->value_;
  }

 private:
  [[nodiscard]] const Var* Root(const Var* v) const {
    auto it = buffer_roots_.find(v);
    return it != buffer_roots_.end() ? it->second : nullptr;
  }

  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  const std::unordered_set<const Var*>& carry_vars_;
};

/// Rewrites uses of each folded copy's LHS to its ultimate source and drops the
/// copy AssignStmt (an empty SeqStmts, which SeqStmts::Flatten inlines away).
class RedundantCopyRewriter : public IRMutator {
 public:
  RedundantCopyRewriter(std::unordered_map<const Var*, ExprPtr> final_target,
                        std::unordered_set<const Var*> drop)
      : final_target_(std::move(final_target)), drop_(std::move(drop)) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = final_target_.find(op.get());
    if (it != final_target_.end()) return it->second;
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_ && drop_.count(op->var_.get()) != 0) {
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }
    return IRMutator::VisitStmt_(op);
  }

 private:
  std::unordered_map<const Var*, ExprPtr> final_target_;
  std::unordered_set<const Var*> drop_;
};

/// Follow ``copy_src`` chains (X=Y, Z=X) so each folded LHS maps directly to its
/// ultimate non-folded source. Bounded by the map size, so O(N).
std::unordered_map<const Var*, ExprPtr> ResolveChains(
    const std::unordered_map<const Var*, ExprPtr>& copy_src) {
  std::unordered_map<const Var*, ExprPtr> final_target;
  final_target.reserve(copy_src.size());
  for (const auto& [x, src] : copy_src) {
    ExprPtr cur = src;
    size_t steps = 0;
    while (steps++ <= copy_src.size()) {
      auto v = AsVarLike(cur);
      if (!v) break;
      auto it = copy_src.find(v.get());
      if (it == copy_src.end()) break;
      cur = it->second;
    }
    final_target[x] = cur;
  }
  return final_target;
}

/// Drop candidates whose ultimate source is not visible at every read of the
/// folded name. Candidates sharing an ultimate source are dropped as a group:
/// they all rewrite to that one source, so if it is invisible at any reader the
/// whole chain must stay.
void DropInvisibleTargets(const ScopeMapper& scopes,
                          std::unordered_map<const Var*, ExprPtr>* final_target) {
  std::unordered_set<const Var*> bad_targets;
  for (const auto& [x, target] : *final_target) {
    auto tv = AsVarLike(target);
    if (!tv) {
      bad_targets.insert(nullptr);
      continue;
    }
    auto def_it = scopes.def_scope.find(tv.get());
    if (def_it == scopes.def_scope.end()) {
      bad_targets.insert(tv.get());
      continue;
    }
    auto use_it = scopes.use_scopes.find(x);
    if (use_it == scopes.use_scopes.end()) continue;
    for (const auto& use_path : use_it->second) {
      if (!IsPrefix(def_it->second, use_path)) {
        bad_targets.insert(tv.get());
        break;
      }
    }
  }
  if (bad_targets.empty()) return;
  for (auto it = final_target->begin(); it != final_target->end();) {
    auto tv = AsVarLike(it->second);
    const Var* key = tv ? tv.get() : nullptr;
    it = bad_targets.count(key) != 0 ? final_target->erase(it) : std::next(it);
  }
}

FunctionPtr RunOnFunction(const ProgramPtr& program, const FunctionPtr& func) {
  if (!func || !func->body_) return func;
  // Orchestration functions are the sole home of these lineage-redundant
  // Var-RHS copies (post-DeriveCallDirections SSA rebinds). InCore/Group/Spmd
  // bodies never carry them, and distributed host_orch has its own codegen.
  if (func->func_type_ != FunctionType::Orchestration) return func;

  BufferRootCollector br(program);
  br.Initialize(func->params_);
  br.VisitStmt(func->body_);

  CarryVarCollector carries;
  carries.VisitStmt(func->body_);

  CopyCollector copies(br.buffer_roots, carries.carry_vars);
  copies.VisitStmt(func->body_);
  if (copies.copy_src.empty()) return func;

  ScopeMapper scopes;
  scopes.Initialize(func->params_);
  scopes.VisitStmt(func->body_);

  auto final_target = ResolveChains(copies.copy_src);
  DropInvisibleTargets(scopes, &final_target);
  if (final_target.empty()) return func;

  std::unordered_set<const Var*> drop;
  drop.reserve(final_target.size());
  for (const auto& [x, _] : final_target) drop.insert(x);

  RedundantCopyRewriter rewriter(std::move(final_target), std::move(drop));
  auto new_body = rewriter.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                    func->return_types_, new_body, func->span_, func->func_type_,
                                    func->level_, func->role_, func->attrs_);
}

}  // namespace

Pass EliminateRedundantVarCopy() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;
    auto new_functions = program->functions_;
    bool changed = false;
    for (auto& [gvar, func] : new_functions) {
      auto new_func = RunOnFunction(program, func);
      if (new_func.get() != func.get()) {
        func = new_func;
        changed = true;
      }
    }
    if (!changed) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "EliminateRedundantVarCopy", kEliminateRedundantVarCopyProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
