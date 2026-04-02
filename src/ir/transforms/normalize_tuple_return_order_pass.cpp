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

#include <cstddef>
#include <functional>
#include <limits>
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
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr size_t kInvalidOutParamIndex = std::numeric_limits<size_t>::max();

std::vector<size_t> BuildReturnToParamMapping(
    const FunctionPtr& callee, const std::unordered_map<std::string, FunctionPtr>* func_by_name,
    std::unordered_set<const Function*>* visiting);

std::vector<size_t> CollectOutIndices(const FunctionPtr& func) {
  std::vector<size_t> out_indices;
  for (size_t i = 0; i < func->param_directions_.size(); ++i) {
    if (func->param_directions_[i] == ParamDirection::Out ||
        func->param_directions_[i] == ParamDirection::InOut) {
      out_indices.push_back(i);
    }
  }
  return out_indices;
}

/**
 * Resolve Out/InOut parameter index for a Var: direct param pointer match, then prior var_to_out_param
 * entries (SSA aliases / tile.store targets).
 */
size_t ResolveParamForVar(const Var* v, const std::unordered_map<const Var*, size_t>& var_to_out_param,
                          const std::function<size_t(const Var*)>& find_param_index) {
  if (!v) {
    return kInvalidOutParamIndex;
  }
  size_t pi = find_param_index(v);
  if (pi != kInvalidOutParamIndex) {
    return pi;
  }
  auto it = var_to_out_param.find(v);
  if (it != var_to_out_param.end()) {
    return it->second;
  }
  return kInvalidOutParamIndex;
}

/**
 * Collect tile.store result → Out param and ForStmt/WhileStmt yield → Out param mappings.
 * Control flow (IfStmt branches, nested SeqStmts, ScopeStmt) must be traversed: top-level-only analysis
 * misses stores under IfStmt (e.g. paged-attention kernel_online_update).
 *
 * Does not use the default IRVisitor statement traversal: AssignStmt must not descend into arbitrary
 * expressions; ForStmt/WhileStmt/IfStmt must only recurse into bodies (not bounds/conditions).
 */
class ReturnVarToOutParamCollector : public IRVisitor {
 public:
  ReturnVarToOutParamCollector(std::function<size_t(const Var*)> find_param_index,
                               std::unordered_map<const Var*, size_t>& var_to_out_param)
      : find_param_index_(std::move(find_param_index)), var_to_out_param_(var_to_out_param) {}

  void Collect(const StmtPtr& stmt) {
    if (stmt) {
      VisitStmt(stmt);
    }
  }

 protected:
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& s : op->stmts_) {
      VisitStmt(s);
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op->var_) {
      return;
    }
    auto call = As<Call>(op->value_);
    if (call && call->op_ && call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
      auto out_param = As<Var>(call->args_[2]);
      if (out_param) {
        const size_t pi = ResolveVarToOutParamIndex(out_param.get());
        if (pi != kInvalidOutParamIndex) {
          var_to_out_param_[op->var_.get()] = pi;
        }
      }
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    for (size_t ri = 0; ri < op->return_vars_.size() && ri < op->iter_args_.size(); ++ri) {
      const auto& iter_arg = op->iter_args_[ri];
      if (!iter_arg || !iter_arg->initValue_ || !op->return_vars_[ri]) continue;
      auto init_var = As<Var>(iter_arg->initValue_);
      if (init_var) {
        const size_t pi = ResolveVarToOutParamIndex(init_var.get());
        if (pi != kInvalidOutParamIndex) {
          var_to_out_param_[op->return_vars_[ri].get()] = pi;
        }
      }
    }
    if (op->body_) {
      VisitStmt(op->body_);
    }
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (size_t ri = 0; ri < op->return_vars_.size() && ri < op->iter_args_.size(); ++ri) {
      const auto& iter_arg = op->iter_args_[ri];
      if (!iter_arg || !iter_arg->initValue_ || !op->return_vars_[ri]) continue;
      auto init_var = As<Var>(iter_arg->initValue_);
      if (init_var) {
        const size_t pi = ResolveVarToOutParamIndex(init_var.get());
        if (pi != kInvalidOutParamIndex) {
          var_to_out_param_[op->return_vars_[ri].get()] = pi;
        }
      }
    }
    if (op->body_) {
      VisitStmt(op->body_);
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (op->then_body_) {
      VisitStmt(op->then_body_);
    }
    if (op->else_body_.has_value() && op->else_body_.value()) {
      VisitStmt(op->else_body_.value());
    }
  }

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (op->body_) {
      VisitStmt(op->body_);
    }
  }

  void VisitStmt_(const YieldStmtPtr& op) override {}
  void VisitStmt_(const ReturnStmtPtr& op) override {}
  void VisitStmt_(const EvalStmtPtr& op) override {}
  void VisitStmt_(const BreakStmtPtr& op) override {}
  void VisitStmt_(const ContinueStmtPtr& op) override {}
  void VisitStmt_(const StmtPtr& op) override {}

 private:
  size_t ResolveVarToOutParamIndex(const Var* v) {
    return ResolveParamForVar(v, var_to_out_param_, find_param_index_);
  }

  std::function<size_t(const Var*)> find_param_index_;
  std::unordered_map<const Var*, size_t>& var_to_out_param_;
};

/**
 * Record vars assigned directly from a Call (e.g. tuple temp from ``_tuple_tmp = kernel(...)``).
 */
class VarDefinedByCallCollector : public IRVisitor {
 public:
  explicit VarDefinedByCallCollector(std::unordered_map<const Var*, std::shared_ptr<const Call>>& var_to_call)
      : var_to_call_(var_to_call) {}

  void Collect(const StmtPtr& stmt) {
    if (stmt) {
      VisitStmt(stmt);
    }
  }

 protected:
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& s : op->stmts_) {
      VisitStmt(s);
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op->var_ || !op->value_) {
      return;
    }
    auto lhs = As<Var>(op->var_);
    auto call = As<Call>(op->value_);
    if (lhs && call) {
      var_to_call_[lhs.get()] = call;
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->body_) {
      VisitStmt(op->body_);
    }
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (op->body_) {
      VisitStmt(op->body_);
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (op->then_body_) {
      VisitStmt(op->then_body_);
    }
    if (op->else_body_.has_value() && op->else_body_.value()) {
      VisitStmt(op->else_body_.value());
    }
  }

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (op->body_) {
      VisitStmt(op->body_);
    }
  }

  void VisitStmt_(const YieldStmtPtr& op) override {}
  void VisitStmt_(const ReturnStmtPtr& op) override {}
  void VisitStmt_(const EvalStmtPtr& op) override {}
  void VisitStmt_(const BreakStmtPtr& op) override {}
  void VisitStmt_(const ContinueStmtPtr& op) override {}
  void VisitStmt_(const StmtPtr& op) override {}

 private:
  std::unordered_map<const Var*, std::shared_ptr<const Call>>& var_to_call_;
};

/**
 * Map ``lhs = tuple_tmp[i]`` to caller Out/InOut param index using callee's return-slot → param mapping.
 */
void MapTupleUnpackFromCallAssignments(
    const StmtPtr& stmt, const std::unordered_map<const Var*, std::shared_ptr<const Call>>& var_to_call,
    const std::unordered_map<std::string, FunctionPtr>* func_by_name,
    std::unordered_set<const Function*>* visiting, const std::function<size_t(const Var*)>& find_param_index,
    std::unordered_map<const Var*, size_t>& var_to_out_param) {
  if (!stmt || !func_by_name) {
    return;
  }
  if (auto seq = As<SeqStmts>(stmt)) {
    for (const auto& s : seq->stmts_) {
      MapTupleUnpackFromCallAssignments(s, var_to_call, func_by_name, visiting, find_param_index,
                                        var_to_out_param);
    }
    return;
  }
  if (auto assign = As<AssignStmt>(stmt)) {
    if (!assign->var_) {
      return;
    }
    auto tgi = As<TupleGetItemExpr>(assign->value_);
    if (tgi) {
      auto tuple_var = As<Var>(tgi->tuple_);
      if (tuple_var && tgi->index_ >= 0) {
        auto it = var_to_call.find(tuple_var.get());
        if (it != var_to_call.end() && it->second) {
          const CallPtr& call = it->second;
          auto fit = func_by_name->find(call->op_->name_);
          if (fit != func_by_name->end() && fit->second) {
            const auto callee_mapping = BuildReturnToParamMapping(fit->second, func_by_name, visiting);
            const size_t slot = static_cast<size_t>(tgi->index_);
            if (slot < callee_mapping.size() && callee_mapping[slot] != kInvalidOutParamIndex) {
              const size_t callee_pi = callee_mapping[slot];
              if (callee_pi < call->args_.size()) {
                if (auto arg_var = As<Var>(call->args_[callee_pi])) {
                  const size_t caller_pi =
                      ResolveParamForVar(arg_var.get(), var_to_out_param, find_param_index);
                  if (caller_pi != kInvalidOutParamIndex) {
                    var_to_out_param[assign->var_.get()] = caller_pi;
                  }
                }
              }
            }
          }
        }
      }
    }
    return;
  }
  if (auto for_stmt = As<ForStmt>(stmt)) {
    MapTupleUnpackFromCallAssignments(for_stmt->body_, var_to_call, func_by_name, visiting, find_param_index,
                                      var_to_out_param);
    return;
  }
  if (auto while_stmt = As<WhileStmt>(stmt)) {
    MapTupleUnpackFromCallAssignments(while_stmt->body_, var_to_call, func_by_name, visiting,
                                      find_param_index, var_to_out_param);
    return;
  }
  if (auto if_stmt = As<IfStmt>(stmt)) {
    MapTupleUnpackFromCallAssignments(if_stmt->then_body_, var_to_call, func_by_name, visiting,
                                      find_param_index, var_to_out_param);
    if (if_stmt->else_body_.has_value()) {
      MapTupleUnpackFromCallAssignments(if_stmt->else_body_.value(), var_to_call, func_by_name, visiting,
                                        find_param_index, var_to_out_param);
    }
    return;
  }
  if (auto scope = As<ScopeStmt>(stmt)) {
    MapTupleUnpackFromCallAssignments(scope->body_, var_to_call, func_by_name, visiting, find_param_index,
                                      var_to_out_param);
  }
}

/**
 * Find the yield payload in the "exit" position of a branch (last stmt in a seq, or nested).
 * SSA-style if/else uses YieldStmt paired with IfStmt::return_vars_ for phi merges.
 */
std::optional<std::vector<ExprPtr>> FindLastYieldValues(const StmtPtr& stmt) {
  if (!stmt) {
    return std::nullopt;
  }
  if (auto seq = As<SeqStmts>(stmt)) {
    if (seq->stmts_.empty()) {
      return std::nullopt;
    }
    return FindLastYieldValues(seq->stmts_.back());
  }
  if (auto y = As<YieldStmt>(stmt)) {
    return y->value_;
  }
  if (auto if_stmt = As<IfStmt>(stmt)) {
    auto t = FindLastYieldValues(if_stmt->then_body_);
    if (t.has_value()) {
      return t;
    }
    if (if_stmt->else_body_.has_value()) {
      return FindLastYieldValues(if_stmt->else_body_.value());
    }
    return std::nullopt;
  }
  if (auto for_stmt = As<ForStmt>(stmt)) {
    return FindLastYieldValues(for_stmt->body_);
  }
  if (auto while_stmt = As<WhileStmt>(stmt)) {
    return FindLastYieldValues(while_stmt->body_);
  }
  if (auto scope = As<ScopeStmt>(stmt)) {
    return FindLastYieldValues(scope->body_);
  }
  return std::nullopt;
}

/**
 * Propagate Out-param indices to IfStmt phi (return_vars_) from branch YieldStmt payloads.
 * Must run after tile.store / loop yield mappings are collected (inner if processed before outer).
 */
void MapPhiVarsFromIfStmtYields(const StmtPtr& stmt, std::unordered_map<const Var*, size_t>& var_to_out_param,
                                const std::function<size_t(const Var*)>& find_param_index) {
  if (!stmt) {
    return;
  }
  if (auto seq = As<SeqStmts>(stmt)) {
    for (const auto& s : seq->stmts_) {
      MapPhiVarsFromIfStmtYields(s, var_to_out_param, find_param_index);
    }
    return;
  }
  if (auto if_stmt = As<IfStmt>(stmt)) {
    MapPhiVarsFromIfStmtYields(if_stmt->then_body_, var_to_out_param, find_param_index);
    if (if_stmt->else_body_.has_value()) {
      MapPhiVarsFromIfStmtYields(if_stmt->else_body_.value(), var_to_out_param, find_param_index);
    }
    if (!if_stmt->return_vars_.empty()) {
      auto yield_then = FindLastYieldValues(if_stmt->then_body_);
      auto yield_else =
          if_stmt->else_body_.has_value() ? FindLastYieldValues(if_stmt->else_body_.value()) : std::nullopt;
      for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
        if (!if_stmt->return_vars_[i]) continue;
        const Var* phi = if_stmt->return_vars_[i].get();
        std::optional<size_t> pi_then;
        std::optional<size_t> pi_else;
        if (yield_then.has_value() && i < yield_then->size()) {
          if (auto v = As<Var>(yield_then->at(i))) {
            const size_t pi = ResolveParamForVar(v.get(), var_to_out_param, find_param_index);
            if (pi != kInvalidOutParamIndex) pi_then = pi;
          }
        }
        if (yield_else.has_value() && i < yield_else->size()) {
          if (auto v = As<Var>(yield_else->at(i))) {
            const size_t pi = ResolveParamForVar(v.get(), var_to_out_param, find_param_index);
            if (pi != kInvalidOutParamIndex) pi_else = pi;
          }
        }
        if (pi_then.has_value() && pi_else.has_value() && *pi_then != *pi_else) {
          INTERNAL_CHECK(false) << "NormalizeTupleReturnOrder: conflicting IfStmt phi merge for slot " << i;
        }
        std::optional<size_t> chosen = pi_then;
        if (!chosen.has_value()) {
          chosen = pi_else;
        }
        const size_t pi = chosen.value_or(kInvalidOutParamIndex);
        if (pi != kInvalidOutParamIndex) {
          var_to_out_param.insert_or_assign(phi, pi);
        }
      }
    }
    return;
  }
  if (auto for_stmt = As<ForStmt>(stmt)) {
    MapPhiVarsFromIfStmtYields(for_stmt->body_, var_to_out_param, find_param_index);
    return;
  }
  if (auto while_stmt = As<WhileStmt>(stmt)) {
    MapPhiVarsFromIfStmtYields(while_stmt->body_, var_to_out_param, find_param_index);
    return;
  }
  if (auto scope = As<ScopeStmt>(stmt)) {
    MapPhiVarsFromIfStmtYields(scope->body_, var_to_out_param, find_param_index);
    return;
  }
  // AssignStmt, YieldStmt, ReturnStmt, EvalStmt: no IfStmt phi merge
}

/**
 * Same analysis as former BuildReturnToParamMapping in orchestration_codegen.cpp:
 * map each return-tuple slot to the Out/InOut parameter index it corresponds to.
 *
 * ``func_by_name`` (optional) resolves callees for tuple-unpack assignments
 * ``a, b = kernel(...)`` (TupleGetItem from call result) — required for orchestration functions.
 */
std::vector<size_t> BuildReturnToParamMapping(
    const FunctionPtr& callee, const std::unordered_map<std::string, FunctionPtr>* func_by_name,
    std::unordered_set<const Function*>* visiting) {
  std::vector<size_t> mapping;
  if (!callee || !callee->body_) return mapping;

  auto seq = As<SeqStmts>(callee->body_);
  if (!seq || seq->stmts_.empty()) return mapping;
  auto return_stmt = As<ReturnStmt>(seq->stmts_.back());
  if (!return_stmt) return mapping;

  std::unique_ptr<void, std::function<void(void*)>> visit_guard(nullptr, [&](void* p) {
    if (p != nullptr && visiting != nullptr) {
      visiting->erase(callee.get());
    }
  });
  if (visiting != nullptr) {
    if (!visiting->insert(callee.get()).second) {
      return {};
    }
    visit_guard.reset(reinterpret_cast<void*>(0x1));
  }

  auto find_param_index = [&](const Var* v) -> size_t {
    for (size_t pi = 0; pi < callee->params_.size(); ++pi) {
      if (callee->params_[pi].get() == v) return pi;
    }
    return kInvalidOutParamIndex;
  };

  std::unordered_map<const Var*, size_t> var_to_out_param;
  std::unordered_map<const Var*, std::shared_ptr<const Call>> var_to_call;
  VarDefinedByCallCollector call_collector(var_to_call);
  for (size_t si = 0; si + 1 < seq->stmts_.size(); ++si) {
    call_collector.Collect(seq->stmts_[si]);
  }

  ReturnVarToOutParamCollector store_collector(find_param_index, var_to_out_param);
  for (size_t si = 0; si + 1 < seq->stmts_.size(); ++si) {
    store_collector.Collect(seq->stmts_[si]);
  }
  for (size_t si = 0; si + 1 < seq->stmts_.size(); ++si) {
    MapPhiVarsFromIfStmtYields(seq->stmts_[si], var_to_out_param, find_param_index);
  }
  for (size_t si = 0; si + 1 < seq->stmts_.size(); ++si) {
    MapTupleUnpackFromCallAssignments(seq->stmts_[si], var_to_call, func_by_name, visiting, find_param_index,
                                      var_to_out_param);
  }

  for (const auto& ret_expr : return_stmt->value_) {
    auto var = As<Var>(ret_expr);
    if (!var) {
      mapping.push_back(kInvalidOutParamIndex);
      continue;
    }
    auto it = var_to_out_param.find(var.get());
    if (it != var_to_out_param.end()) {
      mapping.push_back(it->second);
      continue;
    }
    mapping.push_back(find_param_index(var.get()));
  }
  return mapping;
}

std::vector<TypePtr> PermuteReturnTypes(const FunctionPtr& func, const std::vector<size_t>& perm) {
  const auto& rt = func->return_types_;
  const size_t n = perm.size();
  if (rt.size() == n) {
    std::vector<TypePtr> new_rt(n);
    for (size_t i = 0; i < n; ++i) {
      new_rt[i] = rt[perm[i]];
    }
    return new_rt;
  }
  if (rt.size() == 1) {
    auto tuple_t = As<TupleType>(rt[0]);
    if (tuple_t && tuple_t->types_.size() == n) {
      std::vector<TypePtr> new_inner(n);
      for (size_t i = 0; i < n; ++i) {
        new_inner[i] = tuple_t->types_[perm[i]];
      }
      return {std::make_shared<TupleType>(std::move(new_inner))};
    }
  }
  INTERNAL_CHECK(false) << "NormalizeTupleReturnOrder: cannot permute return_types_ for function '"
                        << func->name_ << "' (return_types_.size()=" << rt.size() << ", arity=" << n << ")";
  return {};
}

FunctionPtr TransformNormalizeTupleReturnOrder(
    const FunctionPtr& func, const std::unordered_map<std::string, FunctionPtr>* func_by_name) {
  const auto out_indices = CollectOutIndices(func);
  if (out_indices.size() <= 1) {
    return func;
  }

  if (!func->body_) {
    return func;
  }
  auto seq = As<SeqStmts>(func->body_);
  if (!seq || seq->stmts_.empty()) {
    return func;
  }
  auto return_stmt = As<ReturnStmt>(seq->stmts_.back());
  if (!return_stmt) {
    return func;
  }

  const auto& vals = return_stmt->value_;
  if (vals.size() <= 1) {
    return func;
  }

  INTERNAL_CHECK(vals.size() == out_indices.size())
      << "NormalizeTupleReturnOrder: return arity " << vals.size() << " != Out/InOut count "
      << out_indices.size() << " in function '" << func->name_ << "'";

  std::unordered_set<const Function*> visit;
  auto mapping = BuildReturnToParamMapping(func, func_by_name, func_by_name ? &visit : nullptr);
  INTERNAL_CHECK(mapping.size() == vals.size())
      << "NormalizeTupleReturnOrder: mapping size mismatch for function '" << func->name_ << "'";

  for (size_t i = 0; i < mapping.size(); ++i) {
    INTERNAL_CHECK(mapping[i] != kInvalidOutParamIndex)
        << "NormalizeTupleReturnOrder: could not resolve return slot " << i << " for function '"
        << func->name_ << "'";
  }

  const size_t n = out_indices.size();
  std::vector<size_t> perm;
  perm.reserve(n);
  for (size_t j = 0; j < n; ++j) {
    const size_t target_param = out_indices[j];
    size_t found = kInvalidOutParamIndex;
    for (size_t i = 0; i < mapping.size(); ++i) {
      if (mapping[i] == target_param) {
        INTERNAL_CHECK(found == kInvalidOutParamIndex)
            << "NormalizeTupleReturnOrder: duplicate mapping to param " << target_param << " in function '"
            << func->name_ << "'";
        found = i;
      }
    }
    INTERNAL_CHECK(found != kInvalidOutParamIndex)
        << "NormalizeTupleReturnOrder: missing return value for Out param index " << target_param
        << " in function '" << func->name_ << "'";
    perm.push_back(found);
  }

  bool identity = true;
  for (size_t i = 0; i < n; ++i) {
    if (perm[i] != i) {
      identity = false;
      break;
    }
  }
  if (identity) {
    return func;
  }

  std::vector<ExprPtr> new_values;
  new_values.reserve(n);
  for (size_t j = 0; j < n; ++j) {
    new_values.push_back(vals[perm[j]]);
  }

  auto new_return_stmt = std::make_shared<ReturnStmt>(std::move(new_values), return_stmt->span_);
  std::vector<StmtPtr> new_stmts = seq->stmts_;
  new_stmts.back() = new_return_stmt;
  auto new_body = std::make_shared<SeqStmts>(std::move(new_stmts), seq->span_);

  auto new_return_types = PermuteReturnTypes(func, perm);

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                    std::move(new_return_types), new_body, func->span_, func->func_type_,
                                    func->level_, func->role_, func->attrs_);
}

ProgramPtr NormalizeTupleReturnOrderProgram(const ProgramPtr& program) {
  INTERNAL_CHECK(program) << "NormalizeTupleReturnOrder: null program";
  std::unordered_map<std::string, FunctionPtr> by_name;
  for (const auto& [gv, f] : program->functions_) {
    by_name[f->name_] = f;
  }
  std::vector<FunctionPtr> transformed;
  transformed.reserve(program->functions_.size());
  for (const auto& [gv, f] : program->functions_) {
    transformed.push_back(TransformNormalizeTupleReturnOrder(f, &by_name));
  }
  return std::make_shared<Program>(transformed, program->name_, program->span_);
}

}  // namespace

namespace pass {

Pass NormalizeTupleReturnOrder() {
  return CreateProgramPass(NormalizeTupleReturnOrderProgram, "NormalizeTupleReturnOrder",
                           kNormalizeTupleReturnOrderProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
