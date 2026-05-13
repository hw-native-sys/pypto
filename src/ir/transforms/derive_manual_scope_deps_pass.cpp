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
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::codegen::IsBuiltinOp;

// ---------------------------------------------------------------------------
// First pass: resolve manual_dep_edges from user_manual_dep_edges (legacy
// behavior of DeriveManualScopeDeps, minus the auto-dataflow inference which
// was disabled to give callers full control via explicit deps=[]).
// ---------------------------------------------------------------------------
class ManualDepResolveMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    if (op->manual_) {
      ++manual_depth_;
      auto rewritten = IRMutator::VisitStmt_(op);
      --manual_depth_;
      return rewritten;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(base);
    if (!assign || manual_depth_ == 0) return assign;
    auto call = As<Call>(assign->value_);
    if (!call || IsBuiltinOp(call->op_->name_)) return assign;
    auto new_call = ResolveManualDepsForCall(call);
    if (new_call.get() == call.get()) return assign;
    return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto eval = As<EvalStmt>(base);
    if (!eval || manual_depth_ == 0) return eval;
    auto call = As<Call>(eval->expr_);
    if (!call || IsBuiltinOp(call->op_->name_)) return eval;
    auto new_call = ResolveManualDepsForCall(call);
    if (new_call.get() == call.get()) return eval;
    return std::make_shared<EvalStmt>(new_call, eval->span_);
  }

 private:
  // Resolve a Call's ``manual_dep_edges`` from its user-supplied
  // ``user_manual_dep_edges`` attr. We do NOT auto-derive edges from
  // dataflow (tensor args referring to prior-call producers): auto-derivation
  // proved unreliable for patterns where multiple parallel kernels share an
  // ``Out`` parameter (it would over-serialise them). Manual scope now
  // requires explicit ``deps=[...]`` from the user; the pass only translates
  // user-supplied edges and synthesises TaskId companions downstream.
  CallPtr ResolveManualDepsForCall(const CallPtr& call) {
    std::vector<VarPtr> deps;
    std::unordered_set<const Var*> seen;
    for (const auto& [k, v] : call->attrs_) {
      if (k != kAttrUserManualDepEdges) continue;
      const auto* user_deps = std::any_cast<std::vector<VarPtr>>(&v);
      INTERNAL_CHECK_SPAN(user_deps, call->span_)
          << "Internal error: " << kAttrUserManualDepEdges << " attr must hold std::vector<VarPtr>";
      for (const auto& var : *user_deps) {
        if (var && seen.insert(var.get()).second) deps.push_back(var);
      }
      break;
    }
    if (deps.empty()) return call;
    auto new_attrs = WithManualDepEdgesAttr(call->attrs_, std::move(deps));
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

  int manual_depth_ = 0;
};

// ---------------------------------------------------------------------------
// Second pass: closure analysis to find every Var that needs a TaskId
// companion. Starts from each Call's resolved manual_dep_edges and propagates
// upward through:
//   * ForStmt iter_arg ↔ return_var ↔ init_value
//   * Yield value ↔ destination return_var/iter_arg
//   * Trivial Var aliases (AssignStmt with Var RHS, TupleGetItemExpr)
// ---------------------------------------------------------------------------
class TaskRelevantVarCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> needs_tid_;   // Vars that need TaskId companion
  std::unordered_set<const Var*> kernel_lhs_;  // Vars produced by user kernel Calls
  std::unordered_set<const Var*> has_def_;     // Vars that have any AssignStmt def in the body
  std::unordered_map<const Var*, VarPtr>
      import_vars_;  // Vars in needs_tid_ with NO def (e.g. function params)

  // Maps for closure analysis.
  std::unordered_map<const Var*, VarPtr> alias_;           // var -> aliased var (Var-to-Var copy / tuple_get)
  std::unordered_map<const Var*, VarPtr> iter_arg_init_;   // for-iter_arg -> init_value Var
  std::unordered_map<const Var*, VarPtr> rv_to_iter_arg_;  // for-return_var -> for-iter_arg
  // For each YieldStmt, the destination return_vars (from enclosing scope).
  // We capture pairs (yield_value_var -> destination_return_var) in `yield_pairs_`.
  std::vector<std::pair<VarPtr, VarPtr>> yield_pairs_;

  void Run(const StmtPtr& body) {
    VisitStmt(body);
    // Propagate `needs_tid_` upward to closure.
    bool changed = true;
    while (changed) {
      changed = false;
      // Through Var aliases.
      for (auto& [v, alias] : alias_) {
        if (needs_tid_.count(v) && alias && needs_tid_.insert(alias.get()).second) changed = true;
        if (alias && needs_tid_.count(alias.get()) && needs_tid_.insert(v).second) changed = true;
      }
      // Through iter_arg <-> init_value.
      for (auto& [ia, init] : iter_arg_init_) {
        if (needs_tid_.count(ia) && init && needs_tid_.insert(init.get()).second) changed = true;
      }
      // Through iter_arg <-> return_var.
      for (auto& [rv, ia] : rv_to_iter_arg_) {
        if (needs_tid_.count(rv) && ia && needs_tid_.insert(ia.get()).second) changed = true;
        if (ia && needs_tid_.count(ia.get()) && needs_tid_.insert(rv).second) changed = true;
      }
      // Through yield: dest <-> source (bidirectional). The dest→src direction
      // covers ``deps=[<iter_arg>]`` (deps source is the loop carry; the kernel
      // LHS it gets yielded from must also need a tid). The src→dest direction
      // covers ``deps=[<kernel_lhs>]`` (deps source is the kernel LHS within
      // the loop body; the carry destination it gets yielded to must also need
      // a tid so the ForStmt mutator allocates the matching iter_arg companion).
      for (auto& [src, dest] : yield_pairs_) {
        if (dest && needs_tid_.count(dest.get()) && src && needs_tid_.insert(src.get()).second)
          changed = true;
        if (src && needs_tid_.count(src.get()) && dest && needs_tid_.insert(dest.get()).second)
          changed = true;
      }
    }

    // Identify "import" Vars: in needs_tid_ but with no AssignStmt def
    // anywhere in the body AND not an IterArg of an enclosing ForStmt
    // (IterArgs are defined by their ForStmt header, not by AssignStmt).
    // These are typically function parameters used as implicit iter_arg
    // init values; they need a synthesized
    // ``<var>__tid = system.task_invalid()`` AssignStmt at body entry so
    // the TaskId companion has an SSA def.
    for (const auto& [ia, init] : iter_arg_init_) {
      if (!init) continue;
      if (!needs_tid_.count(init.get())) continue;
      if (has_def_.count(init.get())) continue;
      if (init->GetKind() == ObjectKind::IterArg) continue;
      import_vars_[init.get()] = init;
    }
  }

 protected:
  void VisitExpr_(const CallPtr& call) override {
    for (const auto& [k, v] : call->attrs_) {
      if (k != kAttrManualDepEdges) continue;
      const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
      if (edges) {
        for (const auto& e : *edges) {
          if (e) needs_tid_.insert(e.get());
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (assign->var_ && assign->value_) {
      has_def_.insert(assign->var_.get());
      if (auto rhs_var = AsVarLike(assign->value_)) {
        alias_[assign->var_.get()] = rhs_var;
      }
      if (auto get_item = As<TupleGetItemExpr>(assign->value_)) {
        if (auto tup_var = AsVarLike(get_item->tuple_)) {
          alias_[assign->var_.get()] = tup_var;
        }
      }
      if (auto call = As<Call>(assign->value_)) {
        if (!IsBuiltinOp(call->op_->name_)) {
          kernel_lhs_.insert(assign->var_.get());
        }
      }
    }
    IRVisitor::VisitStmt_(assign);
  }

  void VisitStmt_(const ForStmtPtr& for_stmt) override {
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      const auto& ia = for_stmt->iter_args_[i];
      if (!ia) continue;
      if (auto init_var = AsVarLike(ia->initValue_)) {
        iter_arg_init_[ia.get()] = init_var;
      }
      if (i < for_stmt->return_vars_.size() && for_stmt->return_vars_[i]) {
        rv_to_iter_arg_[for_stmt->return_vars_[i].get()] = ia;
      }
    }
    // YieldStmts inside body destinations are for_stmt->return_vars_.
    scope_dest_stack_.push_back(for_stmt->return_vars_);
    IRVisitor::VisitStmt_(for_stmt);
    scope_dest_stack_.pop_back();
  }

  void VisitStmt_(const IfStmtPtr& if_stmt) override {
    scope_dest_stack_.push_back(if_stmt->return_vars_);
    IRVisitor::VisitStmt_(if_stmt);
    scope_dest_stack_.pop_back();
  }

  void VisitStmt_(const YieldStmtPtr& y) override {
    if (!scope_dest_stack_.empty()) {
      const auto& dests = scope_dest_stack_.back();
      for (size_t i = 0; i < y->value_.size() && i < dests.size(); ++i) {
        auto src = AsVarLike(y->value_[i]);
        if (src && dests[i]) {
          yield_pairs_.emplace_back(src, dests[i]);
        }
      }
    }
    IRVisitor::VisitStmt_(y);
  }

 private:
  std::vector<std::vector<VarPtr>> scope_dest_stack_;
};

// ---------------------------------------------------------------------------
// Third pass: rewrite IR to install parallel TaskId infrastructure.
// ---------------------------------------------------------------------------
class TaskIdLoweringMutator : public IRMutator {
 public:
  TaskIdLoweringMutator(const std::unordered_set<const Var*>& needs_tid,
                        const std::unordered_set<const Var*>& kernel_lhs,
                        std::unordered_map<const Var*, VarPtr>* tid_map)
      : needs_tid_(needs_tid), kernel_lhs_(kernel_lhs), tid_map_(tid_map) {}

  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    if (op->manual_) {
      ++manual_depth_;
      auto rewritten = IRMutator::VisitStmt_(op);
      --manual_depth_;
      return rewritten;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto seq = As<SeqStmts>(base);
    if (!seq || manual_depth_ == 0) return seq;

    // Inject TaskId companion definitions after each AssignStmt whose LHS
    // is in needs_tid_:
    //   * kernel Call: ``<lhs>__tid = task_id_of(<lhs>)``
    //   * tensor.create: ``<lhs>__tid = task_invalid()``
    //   * Var alias ``b = a``: ``b__tid = a__tid`` (forward the producer's
    //     companion to the alias; without this, a downstream ``deps=[b]``
    //     gets rewritten to ``b__tid`` which would never be defined).
    //   * Tuple extract ``b = tuple[i]``: ``b__tid = tuple_var__tid`` (the
    //     tuple-producing kernel has a single task id; all unpacked elements
    //     share it).
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(seq->stmts_.size() * 2);
    bool changed = false;
    for (const auto& stmt : seq->stmts_) {
      new_stmts.push_back(stmt);
      auto assign = As<AssignStmt>(stmt);
      if (!assign || !assign->var_) continue;
      if (!needs_tid_.count(assign->var_.get())) continue;
      VarPtr tid_var = LookupOrAllocateTid(assign->var_);
      if (!tid_var) continue;

      if (auto call = As<Call>(assign->value_)) {
        if (kernel_lhs_.count(assign->var_.get())) {
          // user kernel Call: synthesize task_id_of(lhs)
          auto op_var = std::make_shared<GlobalVar>("system.task_id_of");
          std::vector<ExprPtr> args = {assign->var_};
          std::vector<std::pair<std::string, std::any>> kwargs;
          std::vector<std::pair<std::string, std::any>> attrs;
          auto tid_call =
              std::make_shared<Call>(op_var, std::move(args), std::move(kwargs), std::move(attrs),
                                     std::make_shared<ScalarType>(DataType::TASK_ID), assign->span_);
          new_stmts.push_back(std::make_shared<AssignStmt>(tid_var, tid_call, assign->span_));
          changed = true;
        } else if (IsBuiltinOp(call->op_->name_) && call->op_->name_ == "tensor.create") {
          // Placeholder tensor.create: synthesize task_invalid()
          auto op_var = std::make_shared<GlobalVar>("system.task_invalid");
          std::vector<ExprPtr> args;
          std::vector<std::pair<std::string, std::any>> kwargs;
          std::vector<std::pair<std::string, std::any>> attrs;
          auto tid_call =
              std::make_shared<Call>(op_var, std::move(args), std::move(kwargs), std::move(attrs),
                                     std::make_shared<ScalarType>(DataType::TASK_ID), assign->span_);
          new_stmts.push_back(std::make_shared<AssignStmt>(tid_var, tid_call, assign->span_));
          changed = true;
        }
      } else if (auto rhs_var = AsVarLike(assign->value_)) {
        // Plain Var alias: forward the source's tid companion.
        if (auto rhs_tid = LookupOrAllocateTid(rhs_var)) {
          new_stmts.push_back(std::make_shared<AssignStmt>(tid_var, rhs_tid, assign->span_));
          changed = true;
        }
      } else if (auto get_item = As<TupleGetItemExpr>(assign->value_)) {
        // Tuple extract: forward the tuple-producing call's tid companion.
        if (auto tuple_var = AsVarLike(get_item->tuple_)) {
          if (auto tuple_tid = LookupOrAllocateTid(tuple_var)) {
            new_stmts.push_back(std::make_shared<AssignStmt>(tid_var, tuple_tid, assign->span_));
            changed = true;
          }
        }
      }
    }
    if (!changed) return seq;
    return std::make_shared<SeqStmts>(std::move(new_stmts), seq->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(base);
    if (!assign || manual_depth_ == 0) return assign;

    auto call = As<Call>(assign->value_);
    if (!call || IsBuiltinOp(call->op_->name_)) return assign;

    // Two reasons to rewrite this user kernel Call:
    //   * Its LHS is in needs_tid_ → attach kAttrTaskIdVar so subsequent
    //     deps=[lhs] can be resolved via the LHS's TaskId companion.
    //   * Its manual_dep_edges contain Vars that have TaskId companions
    //     → replace each Tensor Var with its TaskId Var so codegen sees a
    //     TaskId-typed dep edge.
    // Either case alone justifies rewriting; do both unconditionally and
    // let RewriteCallTaskIds skip if there's nothing to do.
    VarPtr tid_var;
    if (assign->var_ && needs_tid_.count(assign->var_.get())) {
      tid_var = LookupOrAllocateTid(assign->var_);
    }
    auto rewritten_call = RewriteCallTaskIds(call, tid_var);
    if (rewritten_call.get() == call.get()) return assign;
    return std::make_shared<AssignStmt>(assign->var_, rewritten_call, assign->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto eval = As<EvalStmt>(base);
    if (!eval || manual_depth_ == 0) return eval;

    auto call = As<Call>(eval->expr_);
    if (!call || IsBuiltinOp(call->op_->name_)) return eval;
    auto rewritten_call = RewriteCallTaskIds(call, /*lhs_tid=*/nullptr);
    if (rewritten_call.get() == call.get()) return eval;
    return std::make_shared<EvalStmt>(rewritten_call, eval->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto for_stmt = As<ForStmt>(base);
    if (!for_stmt || manual_depth_ == 0) return for_stmt;

    // Add parallel TaskId iter_args / return_vars for any iter_arg that needs
    // a companion.
    auto new_iter_args = for_stmt->iter_args_;
    auto new_return_vars = for_stmt->return_vars_;
    bool changed = false;
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      const auto& ia = for_stmt->iter_args_[i];
      if (!ia || !needs_tid_.count(ia.get())) continue;
      // PreallocateTaskIdVars already created the IterArg companion; just
      // reuse it. Creating a new shared_ptr here would orphan the one that
      // RewriteCallTaskIds put into manual_dep_edges, breaking pointer
      // identity that the codegen relies on to find the iter_arg's emit name.
      VarPtr ia_tid = LookupOrAllocateTid(ia);
      auto ia_tid_node = As<IterArg>(ia_tid);
      if (!ia_tid_node) continue;
      new_iter_args.push_back(ia_tid_node);

      // Mirror return_var.
      VarPtr rv_tid;
      if (i < for_stmt->return_vars_.size() && for_stmt->return_vars_[i]) {
        rv_tid = LookupOrAllocateTid(for_stmt->return_vars_[i]);
      }
      if (!rv_tid) continue;
      new_return_vars.push_back(rv_tid);
      changed = true;
    }

    if (!changed) return for_stmt;
    return std::make_shared<ForStmt>(for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_,
                                     std::move(new_iter_args), for_stmt->body_, std::move(new_return_vars),
                                     for_stmt->span_, for_stmt->kind_, for_stmt->chunk_config_,
                                     for_stmt->attrs_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto if_stmt = As<IfStmt>(base);
    if (!if_stmt || manual_depth_ == 0) return if_stmt;

    auto new_return_vars = if_stmt->return_vars_;
    bool changed = false;
    for (const auto& rv : if_stmt->return_vars_) {
      if (!rv || !needs_tid_.count(rv.get())) continue;
      VarPtr rv_tid = LookupOrAllocateTid(rv);
      if (!rv_tid) continue;
      new_return_vars.push_back(rv_tid);
      changed = true;
    }
    if (!changed) return if_stmt;
    return std::make_shared<IfStmt>(if_stmt->condition_, if_stmt->then_body_, if_stmt->else_body_,
                                    std::move(new_return_vars), if_stmt->span_);
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto y = As<YieldStmt>(base);
    if (!y || manual_depth_ == 0) return y;

    // Append parallel TaskId yield values for any source whose destination is
    // task-relevant. We need to know the destinations — but YieldStmt itself
    // doesn't carry them. Instead, we conservatively mirror: for each yield
    // value V, if V (or its alias) has a tid companion in tid_map_, append.
    auto new_values = y->value_;
    bool changed = false;
    for (size_t i = 0; i < y->value_.size(); ++i) {
      auto src = AsVarLike(y->value_[i]);
      if (!src) continue;
      if (!needs_tid_.count(src.get())) continue;
      VarPtr tid = LookupOrAllocateTid(src);
      if (!tid) continue;
      new_values.push_back(tid);
      changed = true;
    }
    if (!changed) return y;
    return std::make_shared<YieldStmt>(std::move(new_values), y->span_);
  }

 private:
  CallPtr RewriteCallTaskIds(const CallPtr& call, const VarPtr& lhs_tid) {
    bool changed = false;
    auto new_attrs = call->attrs_;
    // Replace manual_dep_edges Tensor Vars with TaskId Vars.
    for (auto& [k, v] : new_attrs) {
      if (k != kAttrManualDepEdges) continue;
      auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
      if (!edges) continue;
      std::vector<VarPtr> new_edges;
      new_edges.reserve(edges->size());
      bool any_replaced = false;
      for (const auto& e : *edges) {
        if (!e) {
          new_edges.push_back(e);
          continue;
        }
        VarPtr tid = LookupOrAllocateTid(e);
        if (tid && tid.get() != e.get()) {
          new_edges.push_back(tid);
          any_replaced = true;
        } else {
          new_edges.push_back(e);
        }
      }
      if (any_replaced) {
        v = std::move(new_edges);
        changed = true;
      }
      break;
    }
    // Add task_id_var attr.
    if (lhs_tid) {
      bool found = false;
      for (auto& [k, v] : new_attrs) {
        if (k == kAttrTaskIdVar) {
          v = lhs_tid;
          found = true;
          break;
        }
      }
      if (!found) {
        new_attrs.emplace_back(kAttrTaskIdVar, lhs_tid);
      }
      changed = true;
    }
    if (!changed) return call;
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

  VarPtr LookupOrAllocateTid(const VarPtr& v) {
    if (!v) return nullptr;
    auto it = tid_map_->find(v.get());
    if (it != tid_map_->end()) return it->second;
    auto tid = std::make_shared<Var>(v->name_hint_ + "__tid", std::make_shared<ScalarType>(DataType::TASK_ID),
                                     v->span_);
    (*tid_map_)[v.get()] = tid;
    return tid;
  }

  const std::unordered_set<const Var*>& needs_tid_;
  const std::unordered_set<const Var*>& kernel_lhs_;
  std::unordered_map<const Var*, VarPtr>* tid_map_;
  int manual_depth_ = 0;
};

// Pre-allocate TaskId companions for every Var in the closure. IterArg-typed
// Vars get an IterArg companion; ordinary Vars get a plain Var companion.
// The IterArg's init_value is wired to the init_var's TaskId companion.
void PreallocateTaskIdVars(const std::unordered_set<const Var*>& needs_tid,
                           const std::unordered_map<const Var*, VarPtr>& iter_arg_init,
                           std::unordered_map<const Var*, VarPtr>* tid_map) {
  // First pass: allocate Var/IterArg shells without binding init.
  // Pass A: allocate plain Var companions for non-IterArg vars.
  std::vector<const Var*> iter_arg_vs;
  for (auto* v : needs_tid) {
    if (!v) continue;
    auto tid_type = std::make_shared<ScalarType>(DataType::TASK_ID);
    if (v->GetKind() == ObjectKind::IterArg) {
      iter_arg_vs.push_back(v);
    } else {
      (*tid_map)[v] = std::make_shared<Var>(v->name_hint_ + "__tid", tid_type, v->span_);
    }
  }
  // Pass B: allocate IterArg companions with init bound to the outer Var's
  // tid companion. For nested loops the inner IterArg's init points at an
  // outer IterArg that itself needs a companion (allocated in this Pass B);
  // iterate to fixed-point so the inner allocation can see the outer one
  // regardless of which order ``iter_arg_vs`` happened to enumerate them.
  std::unordered_set<const Var*> pending(iter_arg_vs.begin(), iter_arg_vs.end());
  bool progressed = true;
  while (progressed && !pending.empty()) {
    progressed = false;
    for (auto it = pending.begin(); it != pending.end();) {
      const Var* v = *it;
      auto init_it = iter_arg_init.find(v);
      VarPtr init_tid;
      if (init_it != iter_arg_init.end() && init_it->second) {
        auto init_in_map = tid_map->find(init_it->second.get());
        if (init_in_map != tid_map->end()) {
          init_tid = init_in_map->second;
        } else if (pending.count(init_it->second.get())) {
          // Outer IterArg still pending — try again in the next sweep.
          ++it;
          continue;
        }
      }
      auto tid_type = std::make_shared<ScalarType>(DataType::TASK_ID);
      (*tid_map)[v] = std::make_shared<IterArg>(v->name_hint_ + "__tid", tid_type, init_tid, v->span_);
      it = pending.erase(it);
      progressed = true;
    }
  }
  // Any leftovers genuinely have no init source (cycle or missing entry);
  // allocate with null init so downstream IRBuilder validation fires a clear
  // error instead of leaving the entry unallocated.
  for (const Var* v : pending) {
    auto tid_type = std::make_shared<ScalarType>(DataType::TASK_ID);
    (*tid_map)[v] = std::make_shared<IterArg>(v->name_hint_ + "__tid", tid_type, nullptr, v->span_);
  }
}

// Run the full lowering on a function body.
StmtPtr LowerOneFunction(const StmtPtr& body) {
  if (!body) return body;

  // Stage 1: resolve user_manual_dep_edges → manual_dep_edges (legacy).
  ManualDepResolveMutator resolver;
  StmtPtr resolved = resolver.VisitStmt(body);

  // Stage 2: closure to find Vars needing TaskId companion.
  TaskRelevantVarCollector collector;
  collector.Run(resolved);

  if (collector.needs_tid_.empty()) return resolved;

  // Stage 3: pre-allocate TaskId companions.
  std::unordered_map<const Var*, VarPtr> tid_map;
  PreallocateTaskIdVars(collector.needs_tid_, collector.iter_arg_init_, &tid_map);

  // Stage 3b: for ``import`` Vars (in needs_tid_ but with no AssignStmt def
  // — e.g. function parameters used as implicit iter_arg init), prepend
  // ``<var>__tid = system.task_invalid()`` AssignStmts at body entry so the
  // TaskId companion has an SSA def the downstream codegen can reference.
  std::vector<StmtPtr> import_inits;
  for (const auto& [raw_v, var] : collector.import_vars_) {
    auto tid_it = tid_map.find(raw_v);
    if (tid_it == tid_map.end()) continue;
    auto op_var = std::make_shared<GlobalVar>("system.task_invalid");
    auto tid_call = std::make_shared<Call>(op_var, std::vector<ExprPtr>{},
                                           std::vector<std::pair<std::string, std::any>>{},
                                           std::vector<std::pair<std::string, std::any>>{},
                                           std::make_shared<ScalarType>(DataType::TASK_ID), var->span_);
    import_inits.push_back(std::make_shared<AssignStmt>(tid_it->second, tid_call, var->span_));
  }
  if (!import_inits.empty()) {
    auto existing_seq = As<SeqStmts>(resolved);
    std::vector<StmtPtr> new_stmts = std::move(import_inits);
    if (existing_seq) {
      for (const auto& s : existing_seq->stmts_) new_stmts.push_back(s);
      resolved = std::make_shared<SeqStmts>(std::move(new_stmts), existing_seq->span_);
    } else {
      new_stmts.push_back(resolved);
      resolved = std::make_shared<SeqStmts>(std::move(new_stmts), resolved->span_);
    }
  }

  // Stage 4: TaskId infrastructure synthesis (uses pre-built tid_map).
  TaskIdLoweringMutator lowerer(collector.needs_tid_, collector.kernel_lhs_, &tid_map);
  return lowerer.VisitStmt(resolved);
}

}  // namespace

namespace pass {

Pass DeriveManualScopeDeps() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;
    auto new_functions = program->functions_;
    bool any_changed = false;
    for (auto& [gvar, func] : new_functions) {
      if (!func || !func->body_) continue;
      auto new_body = LowerOneFunction(func->body_);
      if (new_body.get() == func->body_.get()) continue;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
      any_changed = true;
    }
    if (!any_changed) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "DeriveManualScopeDeps", kDeriveManualScopeDepsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
