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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// ═══════════════════════════════════════════════════════════════════════════
// Collectors — Pre-analysis visitors for loop variable classification
// ═══════════════════════════════════════════════════════════════════════════

class AssignmentCollector : public IRVisitor {
 public:
  std::set<std::string> assigned;
  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    assigned.insert(op->var_->name_hint_);
    VisitExpr(op->value_);
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    // Don't record loop_var — it's scoped to the loop body, not an outer assignment
    VisitStmt(op->body_);
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    VisitExpr(op->condition_);
    VisitStmt(op->body_);
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(*op->else_body_);
  }
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& s : op->stmts_) VisitStmt(s);
  }
};

class TypeCollector : public IRVisitor {
 public:
  std::unordered_map<std::string, TypePtr> types;
  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override { types[op->var_->name_hint_] = op->var_->GetType(); }
  void VisitStmt_(const ForStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const WhileStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const IfStmtPtr& op) override {
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(*op->else_body_);
  }
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& s : op->stmts_) VisitStmt(s);
  }
};

class UseCollector : public IRVisitor {
 public:
  std::set<std::string> used;
  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }
  void CollectExpr(const ExprPtr& expr) {
    if (expr) VisitExpr(expr);
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) used.insert(op->name_hint_);
    IRVisitor::VisitVarLike_(op);
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// Live-in analysis — computes variables needed from the outer scope
//
// Order-aware: a variable defined before use within a compound statement
// is NOT counted as live-in. This prevents false escaping-var promotion
// for loop-local temporaries (issue #592) while correctly detecting
// variables used before reassignment (CodeRabbit review concern).
// ═══════════════════════════════════════════════════════════════════════════

// Forward declaration for mutual recursion
static std::set<std::string> ComputeSeqLiveIn(const std::vector<StmtPtr>& stmts);

static std::set<std::string> ComputeStmtLiveIn(const StmtPtr& stmt) {
  if (!stmt) return {};

  if (auto op = As<AssignStmt>(stmt)) {
    UseCollector uc;
    uc.CollectExpr(op->value_);
    return uc.used;
  }
  if (auto op = As<EvalStmt>(stmt)) {
    UseCollector uc;
    uc.CollectExpr(op->expr_);
    return uc.used;
  }
  if (auto op = As<ReturnStmt>(stmt)) {
    UseCollector uc;
    for (const auto& v : op->value_) uc.CollectExpr(v);
    return uc.used;
  }
  if (auto op = As<YieldStmt>(stmt)) {
    UseCollector uc;
    for (const auto& v : op->value_) uc.CollectExpr(v);
    return uc.used;
  }
  if (auto op = As<SeqStmts>(stmt)) {
    return ComputeSeqLiveIn(op->stmts_);
  }
  if (auto op = As<ForStmt>(stmt)) {
    UseCollector uc;
    uc.CollectExpr(op->start_);
    uc.CollectExpr(op->stop_);
    uc.CollectExpr(op->step_);
    for (const auto& ia : op->iter_args_) uc.CollectExpr(ia->initValue_);
    if (op->chunk_size_.has_value()) uc.CollectExpr(*op->chunk_size_);
    auto body_li = ComputeStmtLiveIn(op->body_);
    body_li.erase(op->loop_var_->name_hint_);
    for (const auto& ia : op->iter_args_) body_li.erase(ia->name_hint_);
    uc.used.insert(body_li.begin(), body_li.end());
    return uc.used;
  }
  if (auto op = As<WhileStmt>(stmt)) {
    UseCollector uc;
    uc.CollectExpr(op->condition_);
    for (const auto& ia : op->iter_args_) uc.CollectExpr(ia->initValue_);
    auto body_li = ComputeStmtLiveIn(op->body_);
    for (const auto& ia : op->iter_args_) body_li.erase(ia->name_hint_);
    uc.used.insert(body_li.begin(), body_li.end());
    return uc.used;
  }
  if (auto op = As<IfStmt>(stmt)) {
    UseCollector uc;
    uc.CollectExpr(op->condition_);
    auto then_li = ComputeStmtLiveIn(op->then_body_);
    uc.used.insert(then_li.begin(), then_li.end());
    if (op->else_body_.has_value()) {
      auto else_li = ComputeStmtLiveIn(*op->else_body_);
      uc.used.insert(else_li.begin(), else_li.end());
    }
    return uc.used;
  }
  if (auto op = As<ScopeStmt>(stmt)) {
    return ComputeStmtLiveIn(op->body_);
  }
  if (auto op = As<OpStmts>(stmt)) {
    return ComputeSeqLiveIn(op->stmts_);
  }
  return {};
}

static std::set<std::string> ComputeSeqLiveIn(const std::vector<StmtPtr>& stmts) {
  std::set<std::string> defined;
  std::set<std::string> live_in;
  for (const auto& s : stmts) {
    auto stmt_li = ComputeStmtLiveIn(s);
    for (const auto& n : stmt_li) {
      if (!defined.count(n)) live_in.insert(n);
    }
    AssignmentCollector ac;
    ac.Collect(s);
    defined.insert(ac.assigned.begin(), ac.assigned.end());
  }
  return live_in;
}

// ═══════════════════════════════════════════════════════════════════════════
// SSA Converter — Transforms non-SSA IR to SSA form
//
// Algorithm:
//   1. Version each variable on every assignment (x → x_0, x_1, …)
//   2. Insert IterArg/YieldStmt/return_var for loop-carried values
//   3. Insert return_vars + YieldStmt phi nodes for IfStmt merges
//   4. Promote escaping variables (defined inside loops, used after)
// ═══════════════════════════════════════════════════════════════════════════

class SSAConverter {
 public:
  FunctionPtr ConvertFunction(const FunctionPtr& func) {
    INTERNAL_CHECK(func) << "ConvertToSSA cannot run on null function";
    orig_params_ = func->params_;
    orig_param_directions_ = func->param_directions_;

    // Create versioned parameters
    std::vector<VarPtr> new_params;
    std::vector<ParamDirection> new_dirs;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      new_params.push_back(
          AllocVersion(func->params_[i]->name_hint_, func->params_[i]->GetType(), func->params_[i]->span_));
      new_dirs.push_back(func->param_directions_[i]);
    }

    StmtPtr new_body = func->body_ ? ConvertStmt(func->body_) : nullptr;

    return std::make_shared<Function>(func->name_, new_params, new_dirs, func->return_types_, new_body,
                                      func->span_, func->func_type_, func->level_, func->role_);
  }

 private:
  // ── Expression substitution via lightweight IRMutator ──────────────

  class ExprSubstituter : public IRMutator {
   public:
    explicit ExprSubstituter(const std::unordered_map<std::string, VarPtr>& versions) : versions_(versions) {}

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = versions_.find(op->name_hint_);
      return it != versions_.end() ? it->second : op;
    }
    ExprPtr VisitExpr_(const IterArgPtr& op) override {
      auto it = versions_.find(op->name_hint_);
      return it != versions_.end() ? it->second : op;
    }

   private:
    const std::unordered_map<std::string, VarPtr>& versions_;
  };

  ExprPtr SubstExpr(const ExprPtr& e) { return e ? ExprSubstituter(cur_).VisitExpr(e) : e; }

  TypePtr SubstType(const TypePtr& type) {
    if (!type) return type;
    if (auto t = As<TensorType>(type)) {
      std::vector<ExprPtr> shape;
      bool changed = false;
      for (const auto& d : t->shape_) {
        auto nd = SubstExpr(d);
        if (nd != d) changed = true;
        shape.push_back(nd);
      }
      if (changed) {
        return std::make_shared<TensorType>(std::move(shape), t->dtype_, t->memref_, t->tensor_view_);
      }
      return type;
    }
    if (auto t = As<TileType>(type)) {
      if (!t->tile_view_.has_value()) return type;
      const auto& tv = t->tile_view_.value();
      if (tv.valid_shape.empty()) return type;
      std::vector<ExprPtr> vs;
      bool changed = false;
      for (const auto& v : tv.valid_shape) {
        auto nv = SubstExpr(v);
        if (nv != v) changed = true;
        vs.push_back(nv);
      }
      if (!changed) return type;
      TileView ntv = tv;
      ntv.valid_shape = std::move(vs);
      return std::make_shared<TileType>(t->shape_, t->dtype_, t->memref_, std::make_optional(std::move(ntv)),
                                        t->memory_space_);
    }
    return type;
  }

  // ── Version management ─────────────────────────────────────────────

  int NextVersion(const std::string& name) { return ver_[name]++; }

  VarPtr AllocVersion(const std::string& name, const TypePtr& type, const Span& span) {
    int v = NextVersion(name);
    auto var = std::make_shared<Var>(name + "_" + std::to_string(v), SubstType(type), span);
    cur_[name] = var;
    return var;
  }

  static std::string StripIterSuffix(const std::string& name) {
    auto pos = name.rfind("_iter_");
    if (pos == std::string::npos) return name;
    size_t after = pos + 6;
    if (after >= name.size()) return name;
    for (size_t i = after; i < name.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
        return name;
      }
    }
    return name.substr(0, pos);
  }

  void RegisterIterArgs(const std::vector<IterArgPtr>& ias) {
    for (const auto& ia : ias) {
      std::string base = StripIterSuffix(ia->name_hint_);
      cur_[base] = ia;
      if (ia->name_hint_ != base) cur_[ia->name_hint_] = ia;
    }
  }

  void RegisterExistingReturnVars(const std::vector<IterArgPtr>& ias, const std::vector<VarPtr>& rvs) {
    for (size_t i = 0; i < ias.size() && i < rvs.size(); ++i) {
      cur_[StripIterSuffix(ias[i]->name_hint_)] = rvs[i];
    }
  }

  // ── Statement dispatch ─────────────────────────────────────────────

  StmtPtr ConvertStmt(const StmtPtr& s) {
    if (!s) return s;
    auto kind = s->GetKind();
    if (kind == ObjectKind::AssignStmt) return ConvertAssign(As<AssignStmt>(s));
    if (kind == ObjectKind::SeqStmts) return ConvertSeq(As<SeqStmts>(s));
    if (kind == ObjectKind::ForStmt) return ConvertFor(As<ForStmt>(s));
    if (kind == ObjectKind::WhileStmt) return ConvertWhile(As<WhileStmt>(s));
    if (kind == ObjectKind::IfStmt) return ConvertIf(As<IfStmt>(s));
    if (kind == ObjectKind::ReturnStmt) return ConvertReturn(As<ReturnStmt>(s));
    if (kind == ObjectKind::YieldStmt) return ConvertYield(As<YieldStmt>(s));
    if (kind == ObjectKind::EvalStmt) return ConvertEval(As<EvalStmt>(s));
    if (kind == ObjectKind::ScopeStmt) return ConvertScope(As<ScopeStmt>(s));
    if (kind == ObjectKind::OpStmts) return ConvertOps(As<OpStmts>(s));
    return s;
  }

  // ── AssignStmt ─────────────────────────────────────────────────────

  StmtPtr ConvertAssign(const AssignStmtPtr& op) {
    auto val = SubstExpr(op->value_);
    auto var = AllocVersion(op->var_->name_hint_, op->var_->GetType(), op->var_->span_);
    return std::make_shared<AssignStmt>(var, val, op->span_);
  }

  // ── SeqStmts — computes future uses per-statement for escaping detection

  StmtPtr ConvertSeq(const SeqStmtsPtr& op) {
    size_t n = op->stmts_.size();

    // Precompute suffix_needs[i] = variables needed from the outer scope by stmts[i..N-1].
    // Uses order-aware live-in analysis: a variable defined before use within a compound
    // statement is NOT counted as needed. Single backward pass, O(N * stmt_size).
    std::vector<std::set<std::string>> suffix_needs(n + 1);
    for (size_t j = n; j > 0; --j) {
      auto live_in = ComputeStmtLiveIn(op->stmts_[j - 1]);
      AssignmentCollector ac;
      ac.Collect(op->stmts_[j - 1]);
      suffix_needs[j - 1] = live_in;
      for (const auto& name : suffix_needs[j]) {
        if (!ac.assigned.count(name)) {
          suffix_needs[j - 1].insert(name);
        }
      }
    }

    // Forward pass: convert each statement with correct future_needs_
    std::vector<StmtPtr> out;
    for (size_t i = 0; i < n; ++i) {
      future_needs_ = (i + 1 < n) ? suffix_needs[i + 1] : std::set<std::string>{};
      out.push_back(ConvertStmt(op->stmts_[i]));
    }
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

  // ── ForStmt ────────────────────────────────────────────────────────

  StmtPtr ConvertFor(const ForStmtPtr& op) {
    auto saved_future_needs = future_needs_;

    // Substitute range in outer scope
    auto new_start = SubstExpr(op->start_);
    auto new_stop = SubstExpr(op->stop_);
    auto new_step = SubstExpr(op->step_);
    auto before = cur_;

    // Process existing iter_args (substitute init values in outer scope)
    std::vector<IterArgPtr> ias;
    for (const auto& ia : op->iter_args_) {
      ias.push_back(
          std::make_shared<IterArg>(ia->name_hint_, ia->GetType(), SubstExpr(ia->initValue_), ia->span_));
    }

    // Pre-analysis: classify assigned variables
    AssignmentCollector ac;
    ac.Collect(op->body_);
    std::string lv_name = op->loop_var_->name_hint_;

    // Loop-carried: assigned in body AND existed before AND not loop_var/existing iter_arg
    std::vector<std::string> carried;
    for (const auto& n : ac.assigned) {
      if (n == lv_name) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (ia->name_hint_ == n) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      if (before.count(n)) carried.push_back(n);
    }

    // Pre-detect escaping vars: assigned in body AND NOT existed before AND needed
    // by future code (order-aware: used before redefined in the future sequence).
    // Must be detected BEFORE body conversion so the IfStmt handler can see them
    // in current_version_ (needed for single-branch phi creation, issue #600).
    TypeCollector tc;
    tc.Collect(op->body_);
    std::vector<std::string> escaping;
    for (const auto& n : ac.assigned) {
      if (n == lv_name) continue;
      if (before.count(n)) continue;
      if (!saved_future_needs.count(n)) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (ia->name_hint_ == n) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      escaping.push_back(n);
    }
    std::sort(escaping.begin(), escaping.end());

    // Create iter_args + return_vars for carried variables
    std::vector<VarPtr> carried_rvs;
    for (const auto& n : carried) {
      auto init = before.at(n);
      int iv = NextVersion(n);
      ias.push_back(
          std::make_shared<IterArg>(n + "_iter_" + std::to_string(iv), init->GetType(), init, op->span_));
      int rv = NextVersion(n);
      carried_rvs.push_back(std::make_shared<Var>(n + "_" + std::to_string(rv), init->GetType(), op->span_));
    }

    // Create iter_args + return_vars for escaping variables (pre-registered)
    std::vector<VarPtr> esc_rvs;
    for (const auto& n : escaping) {
      auto type_it = tc.types.find(n);
      if (type_it == tc.types.end()) continue;
      auto type = type_it->second;
      auto init = FindInitValue(type, before);
      if (!init) {
        // Last resort: create a placeholder using any variable with matching type
        // This covers zero-trip loop cases
        init = std::make_shared<Var>(n, type, op->span_);
      }
      int iv = NextVersion(n);
      ias.push_back(std::make_shared<IterArg>(n + "_iter_" + std::to_string(iv), type, init, op->span_));
      int rv = NextVersion(n);
      auto rv_var = std::make_shared<Var>(n + "_" + std::to_string(rv), type, op->span_);
      esc_rvs.push_back(rv_var);
    }

    // Version loop variable and register iter_args (including escaping)
    int lvv = NextVersion(lv_name);
    auto new_lv = std::make_shared<Var>(lv_name + "_" + std::to_string(lvv), op->loop_var_->GetType(),
                                        op->loop_var_->span_);
    cur_[lv_name] = new_lv;
    RegisterIterArgs(ias);

    // Convert body — IfStmt handler now sees escaping vars in cur_ via iter_args
    auto new_body = ConvertStmt(op->body_);
    auto after = cur_;

    // Restore outer scope, register return_vars
    cur_ = before;
    for (size_t i = 0; i < carried.size(); ++i) cur_[carried[i]] = carried_rvs[i];
    for (size_t i = 0; i < escaping.size() && i < esc_rvs.size(); ++i) cur_[escaping[i]] = esc_rvs[i];
    RegisterExistingReturnVars(op->iter_args_, op->return_vars_);

    // Build return_vars in iter_arg order: existing + carried + escaping
    std::vector<VarPtr> all_rvs;
    for (const auto& rv : op->return_vars_) all_rvs.push_back(rv);
    for (const auto& rv : carried_rvs) all_rvs.push_back(rv);
    for (const auto& rv : esc_rvs) all_rvs.push_back(rv);

    // Build yields in matching order
    std::vector<ExprPtr> yields;
    if (auto y = ExtractYield(new_body)) yields = y->value_;
    for (const auto& n : carried) yields.push_back(after.at(n));
    for (const auto& n : escaping) {
      auto it = after.find(n);
      if (it != after.end()) {
        yields.push_back(it->second);
      }
    }

    StmtPtr body = new_body;
    if (!yields.empty()) body = ReplaceOrAppendYield(new_body, yields, op->span_);

    return std::make_shared<ForStmt>(new_lv, new_start, new_stop, new_step, ias, body, all_rvs, op->span_,
                                     op->kind_, op->chunk_size_, op->chunk_policy_, op->loop_origin_);
  }

  // ── WhileStmt ──────────────────────────────────────────────────────

  StmtPtr ConvertWhile(const WhileStmtPtr& op) {
    auto saved_future_needs = future_needs_;
    auto before = cur_;

    // Process existing iter_args
    std::vector<IterArgPtr> ias;
    for (const auto& ia : op->iter_args_) {
      ias.push_back(
          std::make_shared<IterArg>(ia->name_hint_, ia->GetType(), SubstExpr(ia->initValue_), ia->span_));
    }

    // Pre-analysis
    AssignmentCollector ac;
    ac.Collect(op->body_);

    // Loop-carried classification
    std::vector<std::string> carried;
    for (const auto& n : ac.assigned) {
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (ia->name_hint_ == n) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      if (before.count(n)) carried.push_back(n);
    }

    // Pre-detect escaping vars (same logic as ForStmt — see issue #600 comment there)
    TypeCollector tc;
    tc.Collect(op->body_);
    std::vector<std::string> escaping;
    for (const auto& n : ac.assigned) {
      if (before.count(n)) continue;
      if (!saved_future_needs.count(n)) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (ia->name_hint_ == n) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      escaping.push_back(n);
    }
    std::sort(escaping.begin(), escaping.end());

    // Create iter_args + return_vars for carried
    std::vector<VarPtr> carried_rvs;
    for (const auto& n : carried) {
      auto init = before.at(n);
      int iv = NextVersion(n);
      ias.push_back(
          std::make_shared<IterArg>(n + "_iter_" + std::to_string(iv), init->GetType(), init, op->span_));
      int rv = NextVersion(n);
      carried_rvs.push_back(std::make_shared<Var>(n + "_" + std::to_string(rv), init->GetType(), op->span_));
    }

    // Create iter_args + return_vars for escaping variables (pre-registered)
    std::vector<VarPtr> esc_rvs;
    for (const auto& n : escaping) {
      auto type_it = tc.types.find(n);
      if (type_it == tc.types.end()) continue;
      auto type = type_it->second;
      auto init = FindInitValue(type, before);
      if (!init) init = std::make_shared<Var>(n, type, op->span_);
      int iv = NextVersion(n);
      ias.push_back(std::make_shared<IterArg>(n + "_iter_" + std::to_string(iv), type, init, op->span_));
      int rv = NextVersion(n);
      esc_rvs.push_back(std::make_shared<Var>(n + "_" + std::to_string(rv), type, op->span_));
    }

    // Register iter_args (including escaping), substitute condition, convert body
    RegisterIterArgs(ias);
    auto new_cond = SubstExpr(op->condition_);
    auto new_body = ConvertStmt(op->body_);
    auto after = cur_;

    // Restore outer scope
    cur_ = before;
    for (size_t i = 0; i < carried.size(); ++i) cur_[carried[i]] = carried_rvs[i];
    for (size_t i = 0; i < escaping.size() && i < esc_rvs.size(); ++i) cur_[escaping[i]] = esc_rvs[i];
    RegisterExistingReturnVars(op->iter_args_, op->return_vars_);

    // Build return_vars: existing + carried + escaping
    std::vector<VarPtr> all_rvs;
    for (const auto& rv : op->return_vars_) all_rvs.push_back(rv);
    for (const auto& rv : carried_rvs) all_rvs.push_back(rv);
    for (const auto& rv : esc_rvs) all_rvs.push_back(rv);

    // Build yields
    std::vector<ExprPtr> yields;
    if (auto y = ExtractYield(new_body)) yields = y->value_;
    for (const auto& n : carried) yields.push_back(after.at(n));
    for (const auto& n : escaping) {
      auto it = after.find(n);
      if (it != after.end()) yields.push_back(it->second);
    }

    StmtPtr body = new_body;
    if (!yields.empty()) body = ReplaceOrAppendYield(new_body, yields, op->span_);

    return std::make_shared<WhileStmt>(new_cond, ias, body, all_rvs, op->span_);
  }

  // ── IfStmt — phi node synthesis ────────────────────────────────────

  StmtPtr ConvertIf(const IfStmtPtr& op) {
    auto cond = SubstExpr(op->condition_);
    auto before = cur_;

    // Convert then branch
    auto new_then = ConvertStmt(op->then_body_);
    auto then_ver = cur_;

    // Restore and convert else branch
    cur_ = before;
    std::optional<StmtPtr> new_else;
    if (op->else_body_.has_value()) {
      new_else = ConvertStmt(*op->else_body_);
    }
    auto else_ver = op->else_body_.has_value() ? cur_ : before;

    // Find variables that diverged between branches
    std::vector<std::string> phis;
    std::set<std::string> seen;
    for (const auto& [n, v] : then_ver) {
      seen.insert(n);
      auto bi = before.find(n);
      if (bi != before.end()) {
        bool then_changed = (bi->second != v);
        auto ei = else_ver.find(n);
        bool else_changed = (ei != else_ver.end() && bi->second != ei->second);
        if (then_changed || else_changed) phis.push_back(n);
      } else if (else_ver.count(n)) {
        // New variable defined in BOTH branches needs a phi
        phis.push_back(n);
      }
    }
    for (const auto& [n, v] : else_ver) {
      if (seen.count(n)) continue;
      auto bi = before.find(n);
      if (bi != before.end() && bi->second != v) phis.push_back(n);
    }
    std::sort(phis.begin(), phis.end());

    // No divergence — return simple IfStmt
    if (phis.empty() && op->return_vars_.empty()) {
      cur_ = before;
      return std::make_shared<IfStmt>(cond, new_then, new_else, std::vector<VarPtr>{}, op->span_);
    }

    // No new phis but existing return_vars (explicit SSA) — version return_vars, keep branch yields
    if (phis.empty()) {
      cur_ = before;
      std::vector<VarPtr> return_vars;
      for (const auto& rv : op->return_vars_) {
        int v = NextVersion(rv->name_hint_);
        auto nrv = std::make_shared<Var>(rv->name_hint_ + "_" + std::to_string(v), rv->GetType(), rv->span_);
        return_vars.push_back(nrv);
        cur_[rv->name_hint_] = nrv;
      }
      return std::make_shared<IfStmt>(cond, new_then, new_else, return_vars, op->span_);
    }

    // Create phi outputs
    cur_ = before;
    std::vector<VarPtr> return_vars;
    std::vector<ExprPtr> then_yields, else_yields;

    for (const auto& n : phis) {
      VarPtr tv = then_ver.count(n) ? then_ver.at(n) : before.at(n);
      VarPtr ev = else_ver.count(n) ? else_ver.at(n) : before.at(n);
      int pv = NextVersion(n);
      auto phi = std::make_shared<Var>(n + "_" + std::to_string(pv), tv->GetType(), op->span_);
      return_vars.push_back(phi);
      then_yields.push_back(tv);
      else_yields.push_back(ev);
      cur_[n] = phi;
    }

    // Preserve any existing return_vars not already handled as phis
    for (const auto& rv : op->return_vars_) {
      bool handled = false;
      for (const auto& p : phis) {
        if (p == rv->name_hint_) {
          handled = true;
          break;
        }
      }
      if (!handled) {
        int v = NextVersion(rv->name_hint_);
        auto nrv = std::make_shared<Var>(rv->name_hint_ + "_" + std::to_string(v), rv->GetType(), rv->span_);
        return_vars.push_back(nrv);
        cur_[rv->name_hint_] = nrv;
      }
    }

    // Append yields to branches
    auto then_with_yield = ReplaceOrAppendYield(new_then, then_yields, op->span_);
    StmtPtr else_with_yield;
    if (new_else.has_value()) {
      else_with_yield = ReplaceOrAppendYield(*new_else, else_yields, op->span_);
    } else {
      // No else branch: yield pre-if values directly (not wrapped in SeqStmts)
      else_with_yield = std::make_shared<YieldStmt>(else_yields, op->span_);
    }

    return std::make_shared<IfStmt>(cond, then_with_yield, std::make_optional(else_with_yield), return_vars,
                                    op->span_);
  }

  // ── Simple statements ──────────────────────────────────────────────

  StmtPtr ConvertReturn(const ReturnStmtPtr& op) {
    std::vector<ExprPtr> vals;
    for (const auto& v : op->value_) vals.push_back(SubstExpr(v));
    return std::make_shared<ReturnStmt>(vals, op->span_);
  }

  StmtPtr ConvertYield(const YieldStmtPtr& op) {
    std::vector<ExprPtr> vals;
    for (const auto& v : op->value_) vals.push_back(SubstExpr(v));
    return std::make_shared<YieldStmt>(vals, op->span_);
  }

  StmtPtr ConvertEval(const EvalStmtPtr& op) {
    auto e = SubstExpr(op->expr_);
    return e != op->expr_ ? std::make_shared<EvalStmt>(e, op->span_) : op;
  }

  StmtPtr ConvertScope(const ScopeStmtPtr& op) {
    auto body = ConvertStmt(op->body_);
    return body != op->body_
               ? std::make_shared<ScopeStmt>(op->scope_kind_, body, op->span_, op->level_, op->role_)
               : op;
  }

  StmtPtr ConvertOps(const OpStmtsPtr& op) {
    std::vector<StmtPtr> out;
    bool changed = false;
    for (const auto& s : op->stmts_) {
      auto ns = ConvertStmt(s);
      if (ns != s) changed = true;
      out.push_back(ns);
    }
    return changed ? OpStmts::Flatten(std::move(out), op->span_) : op;
  }

  // ── Helpers ────────────────────────────────────────────────────────

  VarPtr FindInitValue(const TypePtr& type, const std::unordered_map<std::string, VarPtr>& pre) {
    // Prefer Out/InOut parameter with matching type
    for (size_t i = 0; i < orig_params_.size(); ++i) {
      if (orig_param_directions_[i] == ParamDirection::Out ||
          orig_param_directions_[i] == ParamDirection::InOut) {
        auto it = pre.find(orig_params_[i]->name_hint_);
        if (it != pre.end() && it->second->GetType() == type) return it->second;
      }
    }
    // Fall back to any pre-loop variable with matching type (deterministic ordering)
    std::vector<std::pair<std::string, VarPtr>> candidates;
    for (const auto& [n, v] : pre) {
      if (v->GetType() == type) {
        candidates.emplace_back(n, v);
      }
    }
    if (!candidates.empty()) {
      std::sort(candidates.begin(), candidates.end());
      return candidates.front().second;
    }
    return nullptr;
  }

  static YieldStmtPtr ExtractYield(const StmtPtr& s) {
    if (auto y = As<YieldStmt>(s)) {
      return y;
    }
    if (auto seq = As<SeqStmts>(s)) {
      if (!seq->stmts_.empty()) {
        return As<YieldStmt>(seq->stmts_.back());
      }
    }
    return nullptr;
  }

  static StmtPtr ReplaceOrAppendYield(const StmtPtr& s, const std::vector<ExprPtr>& vals, const Span& span) {
    auto yield = std::make_shared<YieldStmt>(vals, span);
    if (auto seq = As<SeqStmts>(s)) {
      std::vector<StmtPtr> stmts = seq->stmts_;
      bool has_trailing_yield = !stmts.empty() && As<YieldStmt>(stmts.back());
      if (has_trailing_yield) {
        stmts.pop_back();
      }
      stmts.push_back(yield);
      return SeqStmts::Flatten(std::move(stmts), seq->span_);
    }
    if (As<YieldStmt>(s)) {
      return yield;
    }
    return SeqStmts::Flatten({s, yield}, span);
  }

  // ── State ──────────────────────────────────────────────────────────

  std::unordered_map<std::string, VarPtr> cur_;  // name → latest version
  std::unordered_map<std::string, int> ver_;     // name → next version number
  std::set<std::string> future_needs_;           // vars needed (used-before-defined) in subsequent stmts
  std::vector<VarPtr> orig_params_;              // original function params
  std::vector<ParamDirection> orig_param_directions_;
};

FunctionPtr TransformConvertToSSA(const FunctionPtr& func) {
  SSAConverter converter;
  return converter.ConvertFunction(func);
}

}  // namespace

namespace pass {
Pass ConvertToSSA() {
  return CreateFunctionPass(TransformConvertToSSA, "ConvertToSSA", kConvertToSSAProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
