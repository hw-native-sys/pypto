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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

struct AutoNameSeed {
  std::string base_name;
  std::string qualifier;
};

static AutoNameSeed GetAutoNameSeed(const std::string& name) {
  auto parsed = auto_name::Parse(name);
  if (parsed.has_auto_suffix && (parsed.role.has_value() || parsed.version.has_value())) {
    return AutoNameSeed{parsed.base_name, parsed.qualifier};
  }
  return AutoNameSeed{name, ""};
}

static std::string GetAutoNameKey(const std::string& name) {
  auto seed = GetAutoNameSeed(name);
  return auto_name::BuildName(seed.base_name, seed.qualifier);
}

static std::string BuildAutoNamedVersion(const std::string& name, const std::string& role, int version) {
  auto seed = GetAutoNameSeed(name);
  return auto_name::BuildName(seed.base_name, seed.qualifier, role, version);
}

// ═══════════════════════════════════════════════════════════════════════════
// CanonicalVarTracker — confines name-based identity to a single class
//
// Pre-SSA IR creates new Var objects per assignment (let x = e1; let x = e2
// -> two distinct Vars). This tracker maps each name to a canonical Var*
// representative, so all SSA logic can use pointer identity instead of
// string comparisons.
// ═══════════════════════════════════════════════════════════════════════════

class CanonicalVarTracker {
 public:
  /// Get the canonical Var* for a given Var. First occurrence of a name
  /// becomes the canonical; subsequent occurrences return the same pointer.
  const Var* GetCanonical(const Var* var) {
    auto [it, inserted] = name_to_canonical_.try_emplace(var->name_hint_, var);
    return it->second;
  }

  /// Explicitly register a Var as canonical for its name (overrides previous).
  void Register(const Var* var) { name_to_canonical_[var->name_hint_] = var; }

  /// Look up the canonical Var* for a base name (used by RegisterIterArgs).
  [[nodiscard]] const Var* FindByName(const std::string& name) const {
    auto it = name_to_canonical_.find(name);
    return (it != name_to_canonical_.end()) ? it->second : nullptr;
  }

  using State = std::unordered_map<std::string, const Var*>;
  [[nodiscard]] State Save() const { return name_to_canonical_; }
  void Restore(State saved) { name_to_canonical_ = std::move(saved); }

 private:
  std::unordered_map<std::string, const Var*> name_to_canonical_;
};

// ═══════════════════════════════════════════════════════════════════════════
// Collectors — Pre-analysis visitors for loop variable classification
// ═══════════════════════════════════════════════════════════════════════════

class AssignmentCollector : public IRVisitor {
 public:
  explicit AssignmentCollector(CanonicalVarTracker& tracker) : tracker_(tracker) {}
  std::unordered_set<const Var*> assigned;  // canonical pointers
  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    assigned.insert(tracker_.GetCanonical(op->var_.get()));
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

 private:
  CanonicalVarTracker& tracker_;
};

class TypeCollector : public IRVisitor {
 public:
  explicit TypeCollector(CanonicalVarTracker& tracker) : tracker_(tracker) {}
  std::unordered_map<const Var*, TypePtr> types;  // canonical key
  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    types[tracker_.GetCanonical(op->var_.get())] = op->var_->GetType();
  }
  void VisitStmt_(const ForStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const WhileStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const IfStmtPtr& op) override {
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(*op->else_body_);
  }
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& s : op->stmts_) VisitStmt(s);
  }

 private:
  CanonicalVarTracker& tracker_;
};

class UseCollector : public IRVisitor {
 public:
  explicit UseCollector(CanonicalVarTracker& tracker) : tracker_(tracker) {}
  std::unordered_set<const Var*> used;  // canonical pointers
  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }
  void CollectExpr(const ExprPtr& expr) {
    if (expr) VisitExpr(expr);
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) used.insert(tracker_.GetCanonical(op.get()));
    IRVisitor::VisitVarLike_(op);
  }

 private:
  CanonicalVarTracker& tracker_;
};

// ═══════════════════════════════════════════════════════════════════════════
// Live-in analysis — computes variables needed from the outer scope
//
// Order-aware: a variable defined before use within a compound statement
// is NOT counted as live-in. This prevents false escaping-var promotion
// for loop-local temporaries (issue #592) while correctly detecting
// variables used before reassignment (CodeRabbit review concern).
//
// Returns canonical Var* pointers via CanonicalVarTracker.
// ═══════════════════════════════════════════════════════════════════════════

// Forward declaration for mutual recursion
static std::unordered_set<const Var*> ComputeSeqLiveIn(const std::vector<StmtPtr>& stmts,
                                                       CanonicalVarTracker& tracker);

static std::unordered_set<const Var*> ComputeStmtLiveIn(const StmtPtr& stmt, CanonicalVarTracker& tracker) {
  if (!stmt) return {};

  if (auto op = As<AssignStmt>(stmt)) {
    UseCollector uc(tracker);
    uc.CollectExpr(op->value_);
    return uc.used;
  }
  if (auto op = As<EvalStmt>(stmt)) {
    UseCollector uc(tracker);
    uc.CollectExpr(op->expr_);
    return uc.used;
  }
  if (auto op = As<ReturnStmt>(stmt)) {
    UseCollector uc(tracker);
    for (const auto& v : op->value_) uc.CollectExpr(v);
    return uc.used;
  }
  if (auto op = As<YieldStmt>(stmt)) {
    UseCollector uc(tracker);
    for (const auto& v : op->value_) uc.CollectExpr(v);
    return uc.used;
  }
  if (auto op = As<SeqStmts>(stmt)) {
    return ComputeSeqLiveIn(op->stmts_, tracker);
  }
  if (auto op = As<ForStmt>(stmt)) {
    UseCollector uc(tracker);
    uc.CollectExpr(op->start_);
    uc.CollectExpr(op->stop_);
    uc.CollectExpr(op->step_);
    for (const auto& ia : op->iter_args_) uc.CollectExpr(ia->initValue_);
    if (op->chunk_size_.has_value()) uc.CollectExpr(*op->chunk_size_);
    auto body_li = ComputeStmtLiveIn(op->body_, tracker);
    body_li.erase(tracker.GetCanonical(op->loop_var_.get()));
    for (const auto& ia : op->iter_args_) body_li.erase(tracker.GetCanonical(ia.get()));
    uc.used.insert(body_li.begin(), body_li.end());
    return uc.used;
  }
  if (auto op = As<WhileStmt>(stmt)) {
    UseCollector uc(tracker);
    uc.CollectExpr(op->condition_);
    for (const auto& ia : op->iter_args_) uc.CollectExpr(ia->initValue_);
    auto body_li = ComputeStmtLiveIn(op->body_, tracker);
    for (const auto& ia : op->iter_args_) body_li.erase(tracker.GetCanonical(ia.get()));
    uc.used.insert(body_li.begin(), body_li.end());
    return uc.used;
  }
  if (auto op = As<IfStmt>(stmt)) {
    UseCollector uc(tracker);
    uc.CollectExpr(op->condition_);
    auto then_li = ComputeStmtLiveIn(op->then_body_, tracker);
    uc.used.insert(then_li.begin(), then_li.end());
    if (op->else_body_.has_value()) {
      auto else_li = ComputeStmtLiveIn(*op->else_body_, tracker);
      uc.used.insert(else_li.begin(), else_li.end());
    }
    return uc.used;
  }
  if (auto op = As<ScopeStmt>(stmt)) {
    return ComputeStmtLiveIn(op->body_, tracker);
  }
  if (auto op = As<OpStmts>(stmt)) {
    return ComputeSeqLiveIn(op->stmts_, tracker);
  }
  return {};
}

static std::unordered_set<const Var*> ComputeSeqLiveIn(const std::vector<StmtPtr>& stmts,
                                                       CanonicalVarTracker& tracker) {
  std::unordered_set<const Var*> defined;
  std::unordered_set<const Var*> live_in;
  for (const auto& s : stmts) {
    auto stmt_li = ComputeStmtLiveIn(s, tracker);
    for (const auto& canonical : stmt_li) {  // NOLINT: set insertion is order-independent
      if (!defined.count(canonical)) live_in.insert(canonical);
    }
    AssignmentCollector ac(tracker);
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
      auto canonical = tracker_.GetCanonical(func->params_[i].get());
      new_params.push_back(AllocVersion(canonical, func->params_[i]->GetType(), func->params_[i]->span_));
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
    ExprSubstituter(const std::unordered_map<const Var*, VarPtr>& versions, CanonicalVarTracker& tracker)
        : versions_(versions), tracker_(tracker) {}

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = versions_.find(tracker_.GetCanonical(op.get()));
      return it != versions_.end() ? it->second : op;
    }
    ExprPtr VisitExpr_(const IterArgPtr& op) override {
      auto it = versions_.find(tracker_.GetCanonical(op.get()));
      return it != versions_.end() ? it->second : op;
    }

   private:
    const std::unordered_map<const Var*, VarPtr>& versions_;
    CanonicalVarTracker& tracker_;
  };

  ExprPtr SubstExpr(const ExprPtr& e) { return e ? ExprSubstituter(cur_, tracker_).VisitExpr(e) : e; }

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

  int NextVersion(const Var* canonical) { return ver_[canonical]++; }

  VarPtr AllocVersion(const Var* canonical, const TypePtr& type, const Span& span) {
    int v = NextVersion(canonical);
    auto var =
        std::make_shared<Var>(BuildAutoNamedVersion(canonical->name_hint_, "ssa", v), SubstType(type), span);
    cur_[canonical] = var;
    return var;
  }

  void RegisterIterArgs(const std::vector<IterArgPtr>& ias) {
    for (const auto& ia : ias) {
      std::string key = GetAutoNameKey(ia->name_hint_);
      auto base_canonical = tracker_.FindByName(key);
      if (base_canonical) {
        cur_[base_canonical] = ia;
      }
      // Also register under the iter_arg's own canonical
      auto ia_canonical = tracker_.GetCanonical(ia.get());
      if (ia_canonical != base_canonical) {
        cur_[ia_canonical] = ia;
      }
    }
  }

  void RegisterExistingReturnVars(const std::vector<IterArgPtr>& ias, const std::vector<VarPtr>& rvs) {
    for (size_t i = 0; i < ias.size() && i < rvs.size(); ++i) {
      std::string key = GetAutoNameKey(ias[i]->name_hint_);
      auto base_canonical = tracker_.FindByName(key);
      if (base_canonical) {
        cur_[base_canonical] = rvs[i];
      }
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
    auto canonical = tracker_.GetCanonical(op->var_.get());
    auto var = AllocVersion(canonical, op->var_->GetType(), op->var_->span_);
    return std::make_shared<AssignStmt>(var, val, op->span_);
  }

  // ── SeqStmts — computes future uses per-statement for escaping detection

  StmtPtr ConvertSeq(const SeqStmtsPtr& op) {
    size_t n = op->stmts_.size();

    // Precompute suffix_needs[i] = variables needed from the outer scope by stmts[i..N-1].
    // Uses order-aware live-in analysis: a variable defined before use within a compound
    // statement is NOT counted as needed. Single backward pass, O(N * stmt_size).
    std::vector<std::unordered_set<const Var*>> suffix_needs(n + 1);
    for (size_t j = n; j > 0; --j) {
      auto live_in = ComputeStmtLiveIn(op->stmts_[j - 1], tracker_);
      AssignmentCollector ac(tracker_);
      ac.Collect(op->stmts_[j - 1]);
      suffix_needs[j - 1] = live_in;
      for (const auto& canonical : suffix_needs[j]) {
        if (!ac.assigned.count(canonical)) {
          suffix_needs[j - 1].insert(canonical);
        }
      }
    }

    // Forward pass: convert each statement with correct future_needs_
    std::vector<StmtPtr> out;
    for (size_t i = 0; i < n; ++i) {
      future_needs_ = (i + 1 < n) ? suffix_needs[i + 1] : std::unordered_set<const Var*>{};
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
    AssignmentCollector ac(tracker_);
    ac.Collect(op->body_);
    auto lv_canonical = tracker_.GetCanonical(op->loop_var_.get());

    // Loop-carried: assigned in body AND existed before AND not loop_var/existing iter_arg
    std::vector<const Var*> carried;
    for (const auto& canonical : ac.assigned) {
      if (canonical == lv_canonical) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (tracker_.GetCanonical(ia.get()) == canonical) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      if (before.count(canonical)) carried.push_back(canonical);
    }
    std::sort(carried.begin(), carried.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // Pre-detect escaping vars: assigned in body AND NOT existed before AND needed
    // by future code (order-aware: used before redefined in the future sequence).
    // Must be detected BEFORE body conversion so the IfStmt handler can see them
    // in current_version_ (needed for single-branch phi creation, issue #600).
    TypeCollector tc(tracker_);
    tc.Collect(op->body_);
    std::vector<const Var*> escaping;
    for (const auto& canonical : ac.assigned) {
      if (canonical == lv_canonical) continue;
      if (before.count(canonical)) continue;
      if (!saved_future_needs.count(canonical)) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (tracker_.GetCanonical(ia.get()) == canonical) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      escaping.push_back(canonical);
    }
    std::sort(escaping.begin(), escaping.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // Create iter_args + return_vars for carried variables
    std::vector<VarPtr> carried_rvs;
    for (const auto& canonical : carried) {
      auto init = before.at(canonical);
      int iv = NextVersion(canonical);
      ias.push_back(std::make_shared<IterArg>(BuildAutoNamedVersion(canonical->name_hint_, "iter", iv),
                                              init->GetType(), init, op->span_));
      int rv = NextVersion(canonical);
      carried_rvs.push_back(std::make_shared<Var>(BuildAutoNamedVersion(canonical->name_hint_, "rv", rv),
                                                  init->GetType(), op->span_));
    }

    // Create iter_args + return_vars for escaping variables (pre-registered)
    std::vector<VarPtr> esc_rvs;
    for (const auto& canonical : escaping) {
      auto type_it = tc.types.find(canonical);
      if (type_it == tc.types.end()) continue;
      auto type = type_it->second;
      auto init = FindInitValue(type, before);
      if (!init) {
        // Last resort: create a placeholder using any variable with matching type
        // This covers zero-trip loop cases
        init = std::make_shared<Var>(canonical->name_hint_, type, op->span_);
      }
      int iv = NextVersion(canonical);
      ias.push_back(std::make_shared<IterArg>(BuildAutoNamedVersion(canonical->name_hint_, "iter", iv), type,
                                              init, op->span_));
      int rv = NextVersion(canonical);
      auto rv_var =
          std::make_shared<Var>(BuildAutoNamedVersion(canonical->name_hint_, "rv", rv), type, op->span_);
      esc_rvs.push_back(rv_var);
    }

    // Version loop variable and register iter_args (including escaping)
    int lvv = NextVersion(lv_canonical);
    auto new_lv = std::make_shared<Var>(BuildAutoNamedVersion(lv_canonical->name_hint_, "idx", lvv),
                                        op->loop_var_->GetType(), op->loop_var_->span_);
    cur_[lv_canonical] = new_lv;
    RegisterIterArgs(ias);
    for (size_t i = 0; i < carried.size(); ++i) cur_[carried[i]] = ias[op->iter_args_.size() + i];
    for (size_t i = 0; i < escaping.size(); ++i) {
      cur_[escaping[i]] = ias[op->iter_args_.size() + carried.size() + i];
    }

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
    for (const auto& canonical : carried) yields.push_back(after.at(canonical));
    for (const auto& canonical : escaping) {
      auto it = after.find(canonical);
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
    AssignmentCollector ac(tracker_);
    ac.Collect(op->body_);

    // Loop-carried classification
    std::vector<const Var*> carried;
    for (const auto& canonical : ac.assigned) {
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (tracker_.GetCanonical(ia.get()) == canonical) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      if (before.count(canonical)) carried.push_back(canonical);
    }
    std::sort(carried.begin(), carried.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // Pre-detect escaping vars (same logic as ForStmt — see issue #600 comment there)
    TypeCollector tc(tracker_);
    tc.Collect(op->body_);
    std::vector<const Var*> escaping;
    for (const auto& canonical : ac.assigned) {
      if (before.count(canonical)) continue;
      if (!saved_future_needs.count(canonical)) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (tracker_.GetCanonical(ia.get()) == canonical) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      escaping.push_back(canonical);
    }
    std::sort(escaping.begin(), escaping.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // Create iter_args + return_vars for carried
    std::vector<VarPtr> carried_rvs;
    for (const auto& canonical : carried) {
      auto init = before.at(canonical);
      int iv = NextVersion(canonical);
      ias.push_back(std::make_shared<IterArg>(BuildAutoNamedVersion(canonical->name_hint_, "iter", iv),
                                              init->GetType(), init, op->span_));
      int rv = NextVersion(canonical);
      carried_rvs.push_back(std::make_shared<Var>(BuildAutoNamedVersion(canonical->name_hint_, "rv", rv),
                                                  init->GetType(), op->span_));
    }

    // Create iter_args + return_vars for escaping variables (pre-registered)
    std::vector<VarPtr> esc_rvs;
    for (const auto& canonical : escaping) {
      auto type_it = tc.types.find(canonical);
      if (type_it == tc.types.end()) continue;
      auto type = type_it->second;
      auto init = FindInitValue(type, before);
      if (!init) init = std::make_shared<Var>(canonical->name_hint_, type, op->span_);
      int iv = NextVersion(canonical);
      ias.push_back(std::make_shared<IterArg>(BuildAutoNamedVersion(canonical->name_hint_, "iter", iv), type,
                                              init, op->span_));
      int rv = NextVersion(canonical);
      esc_rvs.push_back(
          std::make_shared<Var>(BuildAutoNamedVersion(canonical->name_hint_, "rv", rv), type, op->span_));
    }

    // Register iter_args (including escaping), substitute condition, convert body
    RegisterIterArgs(ias);
    for (size_t i = 0; i < carried.size(); ++i) cur_[carried[i]] = ias[op->iter_args_.size() + i];
    for (size_t i = 0; i < escaping.size(); ++i) {
      cur_[escaping[i]] = ias[op->iter_args_.size() + carried.size() + i];
    }
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
    for (const auto& canonical : carried) yields.push_back(after.at(canonical));
    for (const auto& canonical : escaping) {
      auto it = after.find(canonical);
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
    std::vector<const Var*> phis;
    std::unordered_set<const Var*> seen;
    // NOLINT next two loops: phis is sorted afterward, so iteration order is irrelevant
    for (const auto& [canonical, v] :
         then_ver) {  // NOLINT(bugprone-nondeterministic-pointer-iteration-order)
      seen.insert(canonical);
      auto bi = before.find(canonical);
      if (bi != before.end()) {
        bool then_changed = (bi->second != v);
        auto ei = else_ver.find(canonical);
        bool else_changed = (ei != else_ver.end() && bi->second != ei->second);
        if (then_changed || else_changed) phis.push_back(canonical);
      } else if (else_ver.count(canonical)) {
        // New variable defined in BOTH branches needs a phi
        phis.push_back(canonical);
      }
    }
    for (const auto& [canonical, v] :
         else_ver) {  // NOLINT(bugprone-nondeterministic-pointer-iteration-order)
      if (seen.count(canonical)) continue;
      auto bi = before.find(canonical);
      if (bi != before.end() && bi->second != v) phis.push_back(canonical);
    }
    std::sort(phis.begin(), phis.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

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
        auto rv_canonical = tracker_.GetCanonical(rv.get());
        int v = NextVersion(rv_canonical);
        auto nrv = std::make_shared<Var>(BuildAutoNamedVersion(rv_canonical->name_hint_, "rv", v),
                                         rv->GetType(), rv->span_);
        return_vars.push_back(nrv);
        cur_[rv_canonical] = nrv;
      }
      return std::make_shared<IfStmt>(cond, new_then, new_else, return_vars, op->span_);
    }

    // Create phi outputs
    cur_ = before;
    std::vector<VarPtr> return_vars;
    std::vector<ExprPtr> then_yields, else_yields;

    for (const auto& canonical : phis) {
      VarPtr tv = then_ver.count(canonical) ? then_ver.at(canonical) : before.at(canonical);
      VarPtr ev = else_ver.count(canonical) ? else_ver.at(canonical) : before.at(canonical);
      int pv = NextVersion(canonical);
      auto phi = std::make_shared<Var>(BuildAutoNamedVersion(canonical->name_hint_, "phi", pv), tv->GetType(),
                                       op->span_);
      return_vars.push_back(phi);
      then_yields.push_back(tv);
      else_yields.push_back(ev);
      cur_[canonical] = phi;
    }

    // Preserve any existing return_vars not already handled as phis
    for (const auto& rv : op->return_vars_) {
      auto rv_canonical = tracker_.GetCanonical(rv.get());
      bool handled = false;
      for (const auto& p : phis) {
        if (p == rv_canonical) {
          handled = true;
          break;
        }
      }
      if (!handled) {
        int v = NextVersion(rv_canonical);
        auto nrv = std::make_shared<Var>(BuildAutoNamedVersion(rv_canonical->name_hint_, "rv", v),
                                         rv->GetType(), rv->span_);
        return_vars.push_back(nrv);
        cur_[rv_canonical] = nrv;
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

  VarPtr FindInitValue(const TypePtr& type, const std::unordered_map<const Var*, VarPtr>& pre) {
    // Prefer Out/InOut parameter with matching type
    for (size_t i = 0; i < orig_params_.size(); ++i) {
      if (orig_param_directions_[i] == ParamDirection::Out ||
          orig_param_directions_[i] == ParamDirection::InOut) {
        auto canonical = tracker_.GetCanonical(orig_params_[i].get());
        auto it = pre.find(canonical);
        if (it != pre.end() && it->second->GetType() == type) return it->second;
      }
    }
    // Fall back to any pre-loop variable with matching type (deterministic ordering by UniqueId)
    std::vector<std::pair<uint64_t, VarPtr>> candidates;
    for (const auto& [canonical, v] : pre) {
      if (v->GetType() == type) {
        candidates.emplace_back(canonical->UniqueId(), v);
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

  CanonicalVarTracker tracker_;                  // confines name-based identity to one class
  std::unordered_map<const Var*, VarPtr> cur_;   // canonical → latest version
  std::unordered_map<const Var*, int> ver_;      // canonical → next version number
  std::unordered_set<const Var*> future_needs_;  // canonical vars needed in subsequent stmts
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
