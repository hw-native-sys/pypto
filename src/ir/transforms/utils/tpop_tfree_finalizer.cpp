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

#include "pypto/ir/transforms/utils/tpop_tfree_finalizer.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/core_side_ops.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;
const auto& GetLastYieldStmt = transform_utils::GetLastYieldStmt;

StmtPtr FinalizeNestedTpopTfrees(const StmtPtr& stmt, core_affinity::CoreSide side,
                                 const std::unordered_map<const Var*, VarPtr>& tpop_var_remap,
                                 bool reject_escaping_borrows);

}  // namespace

namespace tpop_tfree {

std::string GetTfreeOpName(core_affinity::CoreSide side) { return core_side_ops::TFreeOp(side); }

CallPtr CreateTfree(core_affinity::CoreSide side, const ExprPtr& tile, const Span& span,
                    std::optional<int> pipe_id) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  if (pipe_id.has_value()) {
    kwargs.emplace_back("id", std::any(pipe_id.value()));
  }
  return OpRegistry::GetInstance().Create(GetTfreeOpName(side), {tile}, kwargs, span);
}

bool IsTpopAssignStmt(const StmtPtr& stmt, VarPtr* result_var, CallPtr* tpop_call) {
  auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
  if (!assign) return false;
  auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
  if (!op_predicates::IsTPop(call)) return false;
  if (result_var) *result_var = assign->var_;
  if (tpop_call) *tpop_call = call;
  return true;
}

bool IsExpectedTpopOp(const std::string& op_name, FunctionType func_type) {
  // An AIC body expects to see tpop_from_aiv (receiving from its vector peer);
  // an AIV body expects tpop_from_aic (receiving from its cube peer).
  if (func_type == FunctionType::AIC) return op_name == core_side_ops::TPopOp(core_affinity::CoreSide::AIC);
  if (func_type == FunctionType::AIV) return op_name == core_side_ops::TPopOp(core_affinity::CoreSide::AIV);
  return false;
}

bool IsExpectedTpopAssignStmt(const StmtPtr& stmt, FunctionType func_type, VarPtr* result_var) {
  auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
  if (!assign) return false;
  auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
  auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
  if (!op || !IsExpectedTpopOp(op->name_, func_type)) return false;
  if (result_var) *result_var = assign->var_;
  return true;
}

bool IsTfreeStmt(const StmtPtr& stmt, VarPtr* tile_var, std::string* op_name) {
  auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
  if (!eval) return false;
  auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  if (!op_predicates::IsTFree(call)) return false;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (op_name) *op_name = op->name_;
  if (tile_var) *tile_var = !call->args_.empty() ? AsVarLike(call->args_[0]) : nullptr;
  return true;
}

std::unordered_set<const Var*> CollectStmtVarRefs(const StmtPtr& stmt) {
  var_collectors::VarDefUseCollector collector;
  collector.VisitStmt(stmt);
  return collector.GetAllVarRefs();
}

std::unordered_set<const Var*> CollectCallArgVarRefs(const StmtPtr& stmt) {
  CallPtr call;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  if (!call) return CollectStmtVarRefs(stmt);

  std::unordered_set<const Var*> refs_set;
  for (const auto& arg : call->args_) {
    var_collectors::VarDefUseCollector collector;
    collector.VisitExpr(arg);
    refs_set.insert(collector.var_uses.begin(), collector.var_uses.end());
  }
  for (const auto& [key, value] : call->kwargs_) {
    (void)key;
    if (value.type() == typeid(ExprPtr)) {
      var_collectors::VarDefUseCollector collector;
      collector.VisitExpr(std::any_cast<ExprPtr>(value));
      refs_set.insert(collector.var_uses.begin(), collector.var_uses.end());
    } else if (value.type() == typeid(VarPtr)) {
      var_collectors::VarDefUseCollector collector;
      collector.VisitExpr(std::any_cast<VarPtr>(value));
      refs_set.insert(collector.var_uses.begin(), collector.var_uses.end());
    } else if (value.type() == typeid(IterArgPtr)) {
      var_collectors::VarDefUseCollector collector;
      collector.VisitExpr(std::any_cast<IterArgPtr>(value));
      refs_set.insert(collector.var_uses.begin(), collector.var_uses.end());
    }
  }
  return refs_set;
}

const Var* CanonicalizeTpopRef(const Var* var, const std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  if (!var) return nullptr;
  auto it = tpop_var_remap.find(var);
  return (it != tpop_var_remap.end() && it->second) ? it->second.get() : var;
}

namespace {

// Check the finalized stream once, without rescanning each nested subtree for
// every outer receive. Accepted nested bodies release their local receives
// locally, so one structural visit also covers zero/multiple loop iterations.
// A deeper receive/release must never consume an enclosing body's FIFO prefix.
class NestedFifoValidator : public IRVisitor {
 protected:
  void VisitStmt_(const ForStmtPtr& op) override { VisitRegion(op); }
  void VisitStmt_(const WhileStmtPtr& op) override { VisitRegion(op); }
  void VisitStmt_(const IfStmtPtr& op) override { VisitRegion(op); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!op_predicates::IsTPop(call)) return;
    auto& pending = outstanding_[call->GetKwarg<int>("id", 0)];
    CHECK_SPAN(pending.empty() || pending.front() == depth_, op->span_)
        << "FIFO receive in a nested body cannot reuse a pipe with an outstanding outer receive; "
           "finish the outer receive's consumers and release it before entering the nested body, "
           "or use a different pipe";
    pending.push_back(depth_);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!op_predicates::IsTFree(call)) return;
    auto& pending = outstanding_[call->GetKwarg<int>("id", 0)];
    if (pending.empty()) return;
    CHECK_SPAN(pending.front() == depth_, op->span_)
        << "FIFO release in a nested body cannot release an outstanding outer receive; "
           "release the received slot in its defining body";
    pending.pop_front();
  }

 private:
  template <typename RegionPtr>
  void VisitRegion(const RegionPtr& op) {
    ++depth_;
    IRVisitor::VisitStmt_(op);
    --depth_;
  }

  size_t depth_ = 0;
  std::map<int, std::deque<size_t>> outstanding_;
};

std::vector<StmtPtr> FinalizeTpopTfreesImpl(const std::vector<StmtPtr>& stmts, core_affinity::CoreSide side,
                                            const std::unordered_map<const Var*, VarPtr>& tpop_var_remap,
                                            bool reject_escaping_borrows) {
  std::vector<StmtPtr> normalized_inputs;
  normalized_inputs.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    normalized_inputs.push_back(
        FinalizeNestedTpopTfrees(stmt, side, tpop_var_remap, reject_escaping_borrows));
  }

  std::map<const Var*, TpopLifetime> chains;
  std::vector<const Var*> tpop_order;
  const size_t no_tfree = std::numeric_limits<size_t>::max();
  const std::string expected_tfree = GetTfreeOpName(side);
  using TpopVarRemap = std::unordered_map<const Var*, VarPtr>;
  std::unordered_map<const Var*, VarPtr> local_tpop_var_remap = tpop_var_remap;

  auto canonicalize_var = [&](const VarPtr& var, const TpopVarRemap& remap) -> VarPtr {
    if (!var) {
      return nullptr;
    }
    auto it = remap.find(var.get());
    if (it != remap.end() && it->second) {
      return it->second;
    }
    return var;
  };

  auto try_resolve_canonical_tpop = [&](const ExprPtr& expr, const TpopVarRemap& remap,
                                        VarPtr* canonical_var) -> bool {
    // A zero-copy view or registry-declared physical writeback over a popped tile
    // aliases that buffer, so the view's own uses must extend the tpop's
    // lifetime. Peel through a *chain* of such views (e.g.
    // slice(transpose_view(tpop))) down to the underlying tile — a single-level
    // unwrap would leave a stacked-view return var unresolved and let the tpop
    // be freed before later uses of the returned view.
    ExprPtr alias_src = expr;
    VarPtr src_var;
    while (alias_src) {
      src_var = AsVarLike(alias_src);
      if (src_var) {
        break;
      }
      alias_src = op_predicates::GetBufferAliasSource(alias_src);
    }
    if (!src_var) {
      return false;
    }
    auto canonical = canonicalize_var(src_var, remap);
    if (!canonical || chains.find(canonical.get()) == chains.end()) {
      return false;
    }
    if (canonical_var) {
      *canonical_var = canonical;
    }
    return true;
  };

  std::function<void(const VarPtr&, const ExprPtr&, TpopVarRemap&)> propagate_alias_into;
  std::function<void(const StmtPtr&, TpopVarRemap&)> propagate_stmt_aliases_into;
  std::function<TpopVarRemap(const std::vector<StmtPtr>&, const TpopVarRemap&)> build_body_alias_map;

  propagate_alias_into = [&](const VarPtr& alias_var, const ExprPtr& src_expr, TpopVarRemap& remap) {
    VarPtr canonical_var;
    if (!alias_var || !try_resolve_canonical_tpop(src_expr, remap, &canonical_var)) {
      return;
    }
    remap[alias_var.get()] = canonical_var;
  };

  auto propagate_if_return_aliases_into = [&](const IfStmtPtr& if_stmt, TpopVarRemap& remap) {
    auto then_map = build_body_alias_map(FlattenBody(if_stmt->then_body_), remap);
    auto then_yield = GetLastYieldStmt(if_stmt->then_body_);
    std::optional<TpopVarRemap> else_map;
    YieldStmtPtr else_yield;
    if (if_stmt->else_body_.has_value()) {
      else_map = build_body_alias_map(FlattenBody(if_stmt->else_body_.value()), remap);
      else_yield = GetLastYieldStmt(if_stmt->else_body_.value());
    }

    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      VarPtr canonical_var;
      bool has_canonical = false;
      bool consistent = true;

      auto merge_yield_value = [&](const YieldStmtPtr& yield_stmt, const TpopVarRemap& branch_map) {
        if (!yield_stmt || i >= yield_stmt->value_.size()) {
          consistent = false;
          return;
        }
        VarPtr branch_canonical;
        if (!try_resolve_canonical_tpop(yield_stmt->value_[i], branch_map, &branch_canonical)) {
          consistent = false;
          return;
        }
        if (!has_canonical) {
          canonical_var = branch_canonical;
          has_canonical = true;
          return;
        }
        if (canonical_var.get() != branch_canonical.get()) {
          consistent = false;
        }
      };

      merge_yield_value(then_yield, then_map);
      if (else_map.has_value()) {
        merge_yield_value(else_yield, else_map.value());
      } else {
        consistent = false;
      }
      CHECK_SPAN(!reject_escaping_borrows || !has_canonical || consistent, if_stmt->span_)
          << "FIFO-backed Tile control-flow result must preserve the same received slot on every path; "
             "materialize independent Tile storage before merging different buffers";
      if (has_canonical && consistent) {
        remap[if_stmt->return_vars_[i].get()] = canonical_var;
      }
    }
  };

  auto propagate_loop_return_aliases_into = [&](const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
                                                const std::vector<VarPtr>& return_vars, TpopVarRemap& remap) {
    TpopVarRemap body_map = remap;
    for (const auto& iter_arg : iter_args) {
      propagate_alias_into(iter_arg, iter_arg ? iter_arg->initValue_ : nullptr, body_map);
    }
    body_map = build_body_alias_map(FlattenBody(body), body_map);
    auto yield_stmt = GetLastYieldStmt(body);
    if (!yield_stmt) {
      return;
    }
    if (!reject_escaping_borrows) {
      for (size_t i = 0; i < iter_args.size() && i < yield_stmt->value_.size(); ++i) {
        propagate_alias_into(iter_args[i], yield_stmt->value_[i], body_map);
      }
    }
    for (size_t i = 0; i < return_vars.size() && i < yield_stmt->value_.size(); ++i) {
      // The yielded alias may be defined inside RegScope. Resolve it using the
      // body's map before exporting its canonical receive to the enclosing body.
      VarPtr canonical_var;
      const bool yielded_borrow = try_resolve_canonical_tpop(yield_stmt->value_[i], body_map, &canonical_var);
      if (reject_escaping_borrows && i < iter_args.size()) {
        VarPtr initial_var;
        const bool initial_borrow = try_resolve_canonical_tpop(iter_args[i]->initValue_, remap, &initial_var);
        // A loop can run zero times. Do not assign only the yielded provenance
        // to its result, or an initial borrowed slot could be freed too early.
        CHECK_SPAN((!initial_borrow && !yielded_borrow) ||
                       (initial_borrow && yielded_borrow && initial_var.get() == canonical_var.get()),
                   yield_stmt->span_)
            << "FIFO-backed Tile loop carry must preserve the same received slot, including zero iterations; "
               "materialize independent Tile storage before carrying a different buffer";
      }
      if (yielded_borrow) {
        remap[return_vars[i].get()] = canonical_var;
      }
    }
  };

  propagate_stmt_aliases_into = [&](const StmtPtr& stmt, TpopVarRemap& remap) {
    if (auto ret = As<ReturnStmt>(stmt); ret && reject_escaping_borrows) {
      // This walk has the enclosing body's receive/alias map. The earlier
      // bottom-up finalization only knows local receives, so it cannot reject
      // an outer borrow returned from a nested branch or loop on its own.
      for (const auto& value : ret->value_) {
        CHECK_SPAN(!try_resolve_canonical_tpop(value, remap, nullptr), ret->span_)
            << "FIFO-backed Tile cannot escape the body where it is received through yield or return; "
               "materialize independent Tile storage before exporting the value";
      }
      return;
    }
    if (auto scope = As<RegScopeStmt>(stmt)) {
      // RegScope is an execute-once region whose Tile versions remain visible
      // afterward. Export aliases without flattening the scope or placing any
      // free inside it; last-use accounting retains the enclosing statement.
      for (const auto& child : FlattenBody(scope->body_)) propagate_stmt_aliases_into(child, remap);
      return;
    }
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      propagate_alias_into(assign->var_, assign->value_, remap);
      return;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      propagate_if_return_aliases_into(if_stmt, remap);
      return;
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      propagate_loop_return_aliases_into(for_stmt->iter_args_, for_stmt->body_, for_stmt->return_vars_,
                                         remap);
      return;
    }
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      propagate_loop_return_aliases_into(while_stmt->iter_args_, while_stmt->body_, while_stmt->return_vars_,
                                         remap);
    }
  };

  build_body_alias_map = [&](const std::vector<StmtPtr>& stmts, const TpopVarRemap& seed) -> TpopVarRemap {
    TpopVarRemap body_map = seed;
    for (const auto& stmt : stmts) {
      propagate_stmt_aliases_into(stmt, body_map);
    }
    return body_map;
  };

  for (size_t i = 0; i < normalized_inputs.size(); ++i) {
    VarPtr tpop_var;
    CallPtr tpop_call;
    if (IsTpopAssignStmt(normalized_inputs[i], &tpop_var, &tpop_call)) {
      const Var* canonical_var = CanonicalizeTpopRef(tpop_var.get(), local_tpop_var_remap);
      VarPtr chain_var = tpop_var;
      if (auto remap_it = local_tpop_var_remap.find(tpop_var.get());
          remap_it != local_tpop_var_remap.end() && remap_it->second) {
        chain_var = remap_it->second;
      }
      std::optional<int> pipe_id;
      if (tpop_call && tpop_call->HasKwarg("id")) {
        pipe_id = tpop_call->GetKwarg<int>("id", 0);
      }
      auto [it, inserted] = chains.emplace(canonical_var, TpopLifetime{i, no_tfree, chain_var, i, pipe_id});
      if (inserted) {
        tpop_order.push_back(canonical_var);
      } else {
        it->second.tpop_idx = i;
        it->second.tpop_var = chain_var;
        it->second.last_use_idx = (it->second.last_use_idx < i) ? i : it->second.last_use_idx;
        it->second.pipe_id = pipe_id;
      }
      continue;
    }

    VarPtr tfree_var;
    std::string tfree_op_name;
    if (IsTfreeStmt(normalized_inputs[i], &tfree_var, &tfree_op_name)) {
      const Var* canonical_var = CanonicalizeTpopRef(tfree_var.get(), local_tpop_var_remap);
      auto chain_it = chains.find(canonical_var);
      if (chain_it == chains.end() || tfree_op_name != expected_tfree) {
        continue;
      }
      if (tfree_var && chain_it->second.tpop_var && tfree_var.get() != chain_it->second.tpop_var.get()) {
        auto eval = std::dynamic_pointer_cast<const EvalStmt>(normalized_inputs[i]);
        auto call = eval ? std::dynamic_pointer_cast<const Call>(eval->expr_) : nullptr;
        INTERNAL_CHECK_SPAN(call && !call->args_.empty(), normalized_inputs[i]->span_)
            << "Internal error: expected tfree EvalStmt with a tile operand";
        auto new_call = MutableCopy(call);
        new_call->args_[0] = chain_it->second.tpop_var;
        auto new_eval = MutableCopy(eval);
        new_eval->expr_ = new_call;
        normalized_inputs[i] = new_eval;
      }
      chain_it->second.tfree_idx = i;
      continue;
    }

    const auto& stmt = normalized_inputs[i];
    const auto yield = As<YieldStmt>(stmt);
    const auto ret = As<ReturnStmt>(stmt);
    if (reject_escaping_borrows && (yield || ret)) {
      for (const auto& value : yield ? yield->value_ : ret->value_) {
        VarPtr escaped;
        CHECK_SPAN(!try_resolve_canonical_tpop(value, local_tpop_var_remap, &escaped), stmt->span_)
            << "FIFO-backed Tile cannot escape the body where it is received through yield or return; "
               "materialize independent Tile storage before exporting the value";
      }
    }

    const auto refs = CollectCallArgVarRefs(stmt);
    for (const auto* ref : var_collectors::GetSortedVarRefs(refs)) {
      const Var* canonical_ref = CanonicalizeTpopRef(ref, local_tpop_var_remap);
      auto chain_it = chains.find(canonical_ref);
      if (chain_it == chains.end()) {
        continue;
      }
      chain_it->second.last_use_idx = i;
    }
    propagate_stmt_aliases_into(normalized_inputs[i], local_tpop_var_remap);
  }

  if (tpop_order.empty()) {
    return normalized_inputs;
  }

  std::vector<bool> remove_existing_tfree(normalized_inputs.size(), false);
  std::unordered_map<size_t, std::vector<StmtPtr>> deferred_tfrees;
  // A hardware tfree releases the oldest slot of its pipe, not the tile
  // operand retained in IR. Later pops must therefore wait for the entire
  // preceding FIFO prefix, even when their own last uses occur earlier.
  std::map<int, size_t> preceding_release;
  for (const auto* var : tpop_order) {
    auto& chain = chains.find(var)->second;
    size_t release_idx = chain.last_use_idx;
    if (chain.tfree_idx != no_tfree) {
      release_idx = std::max(release_idx, chain.tfree_idx);
      remove_existing_tfree[chain.tfree_idx] = true;
    }
    auto& prefix_release = preceding_release[chain.pipe_id.value_or(0)];
    release_idx = std::max(release_idx, prefix_release);
    prefix_release = release_idx;
    // Retiming an authored free must preserve its explicit kwargs. Alias
    // operands were canonicalized above; only synthesized frees inherit the
    // originating pop's pipe id here.
    auto free_stmt =
        chain.tfree_idx != no_tfree
            ? normalized_inputs[chain.tfree_idx]
            : std::make_shared<EvalStmt>(
                  CreateTfree(side, chain.tpop_var, normalized_inputs[chain.tpop_idx]->span_, chain.pipe_id),
                  normalized_inputs[chain.tpop_idx]->span_);
    deferred_tfrees[release_idx].push_back(free_stmt);
  }

  std::vector<StmtPtr> result;
  result.reserve(normalized_inputs.size() + deferred_tfrees.size());
  for (size_t i = 0; i < normalized_inputs.size(); ++i) {
    auto deferred_it = deferred_tfrees.find(i);
    const bool has_deferred = deferred_it != deferred_tfrees.end();
    auto result_stmt = normalized_inputs[i];
    const auto yield = As<YieldStmt>(result_stmt);
    const bool is_terminator = yield != nullptr;
    if (has_deferred && is_terminator) {
      // A branch/loop-local receive cannot outlive its release. Materialize
      // Tile-only yield values before the frees, so later phi lowering reads
      // independent storage. Copy the yielded view, not just its FIFO root.
      // RegScope functions already reject these escapes above.
      auto copied_yield = MutableCopy(yield);
      for (auto& value : copied_yield->value_) {
        VarPtr received;
        if (!try_resolve_canonical_tpop(value, local_tpop_var_remap, &received)) continue;
        auto tile = As<TileType>(received->GetType());
        INTERNAL_CHECK_SPAN(tile && tile->memory_space_.has_value(), yield->span_)
            << "Internal error: received Tile must have a memory space before FIFO yield materialization";
        auto copy = OpRegistry::GetInstance().Create(
            "tile.move", {value}, {{"target_memory", std::any(*tile->memory_space_)}}, yield->span_);
        auto source = AsVarLike(value);
        auto name =
            auto_name::AddQualifier(source ? source->name_hint_ : received->name_hint_, "fifo_result");
        auto copied = std::make_shared<Var>(name, copy->GetType(), yield->span_);
        result.push_back(std::make_shared<AssignStmt>(copied, copy, yield->span_));
        value = copied;
      }
      result_stmt = copied_yield;
      // Keep YieldStmt last: downstream control-flow passes require it to be
      // the terminator even when the last FIFO consumer was a yielded value.
      result.insert(result.end(), deferred_it->second.begin(), deferred_it->second.end());
    }
    if (!remove_existing_tfree[i]) {
      result.push_back(result_stmt);
    }
    if (has_deferred && !is_terminator) {
      result.insert(result.end(), deferred_it->second.begin(), deferred_it->second.end());
    }
  }
  return result;
}

}  // namespace

std::vector<StmtPtr> FinalizeTpopTfrees(const std::vector<StmtPtr>& stmts, core_affinity::CoreSide side,
                                        const std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  // Establish the conservative register/FIFO contract once for the whole body,
  // including borrows that escape a branch before a later register scope.
  // Preserve the existing Tile-only control-flow lowering contract.
  class RegScopeFinder : public IRVisitor {
   public:
    bool found = false;
    void VisitStmt_(const RegScopeStmtPtr&) override { found = true; }
  } finder;
  for (const auto& stmt : stmts) finder.VisitStmt(stmt);
  auto finalized = FinalizeTpopTfreesImpl(stmts, side, tpop_var_remap, finder.found);
  if (finder.found) {
    NestedFifoValidator validator;
    for (const auto& stmt : finalized) validator.VisitStmt(stmt);
  }
  return finalized;
}

}  // namespace tpop_tfree

namespace {

StmtPtr FinalizeNestedTpopTfrees(const StmtPtr& stmt, core_affinity::CoreSide side,
                                 const std::unordered_map<const Var*, VarPtr>& tpop_var_remap,
                                 bool reject_escaping_borrows) {
  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    auto new_body = tpop_tfree::FinalizeTpopTfreesImpl(FlattenBody(for_stmt->body_), side, tpop_var_remap,
                                                       reject_escaping_borrows);
    return loop_repair::RebuildForStmt(for_stmt, loop_repair::MakeBody(new_body, for_stmt->span_));
  }
  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto new_then = tpop_tfree::FinalizeTpopTfreesImpl(FlattenBody(if_stmt->then_body_), side, tpop_var_remap,
                                                       reject_escaping_borrows);
    std::optional<std::vector<StmtPtr>> new_else;
    if (const auto& else_body = if_stmt->else_body_) {
      new_else = tpop_tfree::FinalizeTpopTfreesImpl(FlattenBody(*else_body), side, tpop_var_remap,
                                                    reject_escaping_borrows);
    }
    return loop_repair::RebuildIfStmt(if_stmt, new_then, new_else);
  }
  if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    auto new_body = tpop_tfree::FinalizeTpopTfreesImpl(FlattenBody(while_stmt->body_), side, tpop_var_remap,
                                                       reject_escaping_borrows);
    return loop_repair::RebuildWhileStmt(while_stmt, loop_repair::MakeBody(new_body, while_stmt->span_));
  }
  return stmt;
}

}  // namespace
}  // namespace ir
}  // namespace pypto
