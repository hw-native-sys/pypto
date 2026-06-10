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
#include <any>
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
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
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

static std::string BuildAutoNamedVersion(const std::string& name, const std::string& role, int version) {
  auto seed = GetAutoNameSeed(name);
  return auto_name::BuildName(seed.base_name, seed.qualifier, role, version);
}

// ═══════════════════════════════════════════════════════════════════════════
// Collectors — Pre-analysis visitors for loop variable classification
//
// All collectors use raw Var pointer identity.  After the parser/frontend
// reuses Var pointers for same-variable reassignment (#647), every
// occurrence of the same source-level variable already carries the same
// Var*, so no name-based canonicalization is needed.
// ═══════════════════════════════════════════════════════════════════════════

class TypeCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, TypePtr> types;
  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override { types[op->var_.get()] = op->var_->GetType(); }
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
  std::unordered_set<const Var*> used;
  void Collect(const StmtPtr& stmt) {
    if (stmt) VisitStmt(stmt);
  }
  void CollectExpr(const ExprPtr& expr) {
    if (expr) VisitExpr(expr);
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) used.insert(op.get());
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
//
// Returns raw Var* pointers (pointer identity = variable identity).
// ═══════════════════════════════════════════════════════════════════════════

// Forward declaration for mutual recursion
static std::unordered_set<const Var*> ComputeSeqLiveIn(const std::vector<StmtPtr>& stmts);

static std::unordered_set<const Var*> ComputeStmtLiveIn(const StmtPtr& stmt) {
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
    if (op->chunk_config_.has_value()) uc.CollectExpr(op->chunk_config_->size);
    auto body_li = ComputeStmtLiveIn(op->body_);
    body_li.erase(op->loop_var_.get());
    for (const auto& ia : op->iter_args_) body_li.erase(ia.get());
    uc.used.insert(body_li.begin(), body_li.end());
    return uc.used;
  }
  if (auto op = As<WhileStmt>(stmt)) {
    UseCollector uc;
    uc.CollectExpr(op->condition_);
    for (const auto& ia : op->iter_args_) uc.CollectExpr(ia->initValue_);
    auto body_li = ComputeStmtLiveIn(op->body_);
    for (const auto& ia : op->iter_args_) body_li.erase(ia.get());
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
    auto body_li = ComputeStmtLiveIn(op->body_);
    // SpmdScopeStmt::core_num_ is a direct ExprPtr field evaluated in the
    // enclosing scope — its uses are live-in there, same as
    // ForStmt::start_/stop_/step_ above.
    if (auto spmd = As<SpmdScopeStmt>(stmt)) {
      UseCollector uc;
      uc.CollectExpr(spmd->core_num_);
      body_li.insert(uc.used.begin(), uc.used.end());
    }
    return body_li;
  }
  return {};
}

static std::unordered_set<const Var*> ComputeSeqLiveIn(const std::vector<StmtPtr>& stmts) {
  std::unordered_set<const Var*> defined;
  std::unordered_set<const Var*> live_in;
  for (const auto& s : stmts) {
    auto stmt_li = ComputeStmtLiveIn(s);
    for (const auto& v : stmt_li) {  // NOLINT: set insertion is order-independent
      if (!defined.count(v)) live_in.insert(v);
    }
    var_collectors::VarDefUseCollector stmt_collector;
    stmt_collector.VisitStmt(s);
    defined.insert(stmt_collector.var_assign_defs.begin(), stmt_collector.var_assign_defs.end());
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
//
// Variable identity uses raw Var pointers.  The parser/frontend guarantees
// that reassignments of the same source-level variable share the same Var
// pointer (#647), so no name-based canonicalization is needed.
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
          AllocVersion(func->params_[i], func->params_[i]->GetType(), func->params_[i]->span_));
      new_dirs.push_back(func->param_directions_[i]);
    }

    StmtPtr new_body = func->body_ ? ConvertStmt(func->body_) : nullptr;

    auto result = MutableCopy(func);
    result->params_ = std::move(new_params);
    result->param_directions_ = std::move(new_dirs);
    result->body_ = std::move(new_body);
    return result;
  }

  /// Snapshot of all Var renames performed by ConvertFunction (original Var
  /// pointer → final SSA Var pointer). Exposed so a program-level driver can
  /// propagate renames into sibling functions' Var-bearing attrs (e.g. an
  /// outlined Spmd function's ``core_num`` references a Var defined in the
  /// dispatching Orchestration function — when the latter is re-versioned,
  /// the cross-function attr must follow). Only entries where the SSA Var
  /// differs from the original are useful.
  const std::unordered_map<const Var*, VarPtr>& cur() const { return cur_; }

 private:
  // ── Expression substitution via lightweight IRMutator ──────────────

  class ExprSubstituter : public IRMutator {
   public:
    explicit ExprSubstituter(SSAConverter& converter) : converter_(converter), versions_(converter.cur_) {}

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = versions_.find(op.get());
      return it != versions_.end() ? it->second : op;
    }
    ExprPtr VisitExpr_(const IterArgPtr& op) override {
      auto it = versions_.find(op.get());
      return it != versions_.end() ? it->second : op;
    }
    ExprPtr VisitExpr_(const CallPtr& op) override {
      // Substitute arguments first (base class recurses into nested Calls)
      auto result = IRMutator::VisitExpr_(op);
      // Then substitute within the Call's return type (e.g., TensorView.valid_shape)
      auto call = std::static_pointer_cast<const Call>(result);
      auto new_type = converter_.SubstType(call->GetType());
      auto new_attrs = converter_.SubstCallAttrs(call->attrs_);
      bool type_changed = new_type.get() != call->GetType().get();
      bool attrs_changed = new_attrs.has_value();
      if (type_changed || attrs_changed) {
        std::vector<std::pair<std::string, std::any>> attrs_to_use;
        if (attrs_changed) {
          attrs_to_use = std::move(*new_attrs);
        } else {
          attrs_to_use = call->attrs_;
        }
        return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(attrs_to_use),
                                            type_changed ? new_type : call->GetType(), call->span_);
      }
      return result;
    }

    ExprPtr VisitExpr_(const SubmitPtr& op) override {
      // The base IRMutator already SSA-renames args_ and deps_ (both
      // first-class fields on Submit), so we only need to also rewrite the
      // return type and surface any kAttrArgDirOverrideVars Var renames
      // that the base did. Call's manual_dep_edges branch in SubstCallAttrs
      // does not apply on Submit — deps live in deps_ now.
      auto result = IRMutator::VisitExpr_(op);
      auto submit = std::static_pointer_cast<const Submit>(result);
      auto new_type = converter_.SubstType(submit->GetType());
      if (new_type.get() != submit->GetType().get()) {
        return std::make_shared<const Submit>(submit->op_, submit->args_, submit->deps_, submit->kwargs_,
                                              submit->attrs_, new_type, submit->span_, submit->core_num_,
                                              submit->sync_start_);
      }
      return result;
    }

   private:
    SSAConverter& converter_;
    const std::unordered_map<const Var*, VarPtr>& versions_;
  };

  ExprPtr SubstExpr(const ExprPtr& e) { return e ? ExprSubstituter(*this).VisitExpr(e) : e; }

  /// Substitute all expressions in a vector, returning the new vector and whether anything changed.
  std::pair<std::vector<ExprPtr>, bool> SubstExprVec(const std::vector<ExprPtr>& vec) {
    std::vector<ExprPtr> out;
    bool changed = false;
    out.reserve(vec.size());
    for (const auto& e : vec) {
      auto ne = SubstExpr(e);
      if (ne != e) changed = true;
      out.push_back(ne);
    }
    return {std::move(out), changed};
  }

  /// Substitute Var references stored in Call attrs. Currently covers:
  ///   * ``kAttrManualDepEdges`` — ``std::vector<VarPtr>`` (dep edges)
  ///   * ``kAttrDumpVars`` — ``std::vector<VarPtr>`` (selective dump
  ///     targets from ``pl.dump_tag`` / ``dumps=``)
  ///   * ``kAttrDevice`` — ``ExprPtr`` (host-orch dispatch device selector,
  ///     typically a loop induction Var that SSA must version)
  ///
  /// ``kAttrArgDirOverrideVars`` is scope-only and handled by the separate
  /// ``SubstScopeAttrs`` path below.
  /// Returns a rebuilt attrs vector when any Var was rewritten, otherwise
  /// std::nullopt so the caller can keep the existing attrs vector verbatim.
  std::optional<std::vector<std::pair<std::string, std::any>>> SubstCallAttrs(
      const std::vector<std::pair<std::string, std::any>>& attrs) {
    bool changed = false;
    std::vector<std::pair<std::string, std::any>> out;
    out.reserve(attrs.size());
    for (const auto& [k, v] : attrs) {
      if (k == kAttrManualDepEdges || k == kAttrDumpVars) {
        const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
        if (edges) {
          std::vector<VarPtr> new_edges;
          new_edges.reserve(edges->size());
          bool any = false;
          for (const auto& e : *edges) {
            if (!e) {
              new_edges.push_back(e);
              continue;
            }
            auto it = cur_.find(e.get());
            if (it != cur_.end() && it->second.get() != e.get()) {
              new_edges.push_back(it->second);
              any = true;
            } else {
              new_edges.push_back(e);
            }
          }
          if (any) {
            changed = true;
            out.emplace_back(k, std::any(std::move(new_edges)));
            continue;
          }
        }
      } else if (k == kAttrDevice) {
        const auto* dev = std::any_cast<ExprPtr>(&v);
        if (dev && *dev) {
          auto new_dev = SubstExpr(*dev);
          if (new_dev.get() != dev->get()) {
            changed = true;
            out.emplace_back(k, std::any(std::move(new_dev)));
            continue;
          }
        }
      }
      out.emplace_back(k, v);
    }
    if (!changed) return std::nullopt;
    return out;
  }

  TypePtr SubstType(const TypePtr& type) {
    if (!type) return type;
    if (auto t = AsTensorTypeLike(type)) {
      auto [shape, changed] = SubstExprVec(t->shape_);
      std::optional<TensorView> new_tv = t->tensor_view_;
      if (t->tensor_view_.has_value()) {
        const auto& tv = t->tensor_view_.value();
        auto [vs, vs_changed] = SubstExprVec(tv.valid_shape);
        auto [st, st_changed] = SubstExprVec(tv.stride);
        if (vs_changed || st_changed) {
          changed = true;
          new_tv = TensorView(std::move(st), tv.layout, std::move(vs), tv.pad);
        }
      }
      if (!changed) return type;
      if (auto dt = As<DistributedTensorType>(type)) {
        return std::make_shared<DistributedTensorType>(std::move(shape), t->dtype_, t->memref_,
                                                       std::move(new_tv), dt->window_buffer_);
      }
      return std::make_shared<TensorType>(std::move(shape), t->dtype_, t->memref_, std::move(new_tv));
    }
    if (auto t = As<TileType>(type)) {
      if (!t->tile_view_.has_value()) return type;
      const auto& tv = t->tile_view_.value();
      auto [vs, changed] = SubstExprVec(tv.valid_shape);
      if (!changed) return type;
      TileView ntv = tv;
      ntv.valid_shape = std::move(vs);
      return std::make_shared<TileType>(t->shape_, t->dtype_, t->memref_, std::make_optional(std::move(ntv)),
                                        t->memory_space_);
    }
    return type;
  }

  // ── Version management ─────────────────────────────────────────────

  int NextVersion(const Var* key) { return ver_[key]++; }

  /// True if ``name`` is already in versioned auto-name form (e.g. ``x__ssa_v0``,
  /// ``t__tmp_v1``, ``r__idx_v0``). Such names are produced by ConvertToSSA's
  /// own LHS renaming and escape-promotion paths (``ssa`` / ``iter`` / ``rv`` /
  /// ``phi`` / ``idx`` roles) as well as by upstream CSE / normalization passes
  /// (``tmp`` role). On a second ConvertToSSA run we keep these at their
  /// original pointer identity instead of re-minting, to avoid:
  ///
  ///   1. Stale cross-function references — an outlined Spmd function's
  ///      ``core_num`` ExprPtr or a ``DistributedTensorType``'s
  ///      ``window_buffer_`` field that the per-function ``cur_`` cannot reach.
  ///   2. ``name_hint_`` collisions in downstream codegen that uses the raw
  ///      name as a Python identifier (host_orch ``tensors[...]`` keys,
  ///      ``CommBufferSpec(name=...)``) — re-versioning ``t__tmp_v1`` and
  ///      ``t__tmp_v2`` would both produce ``name_hint_ = "t__ssa_v0"`` with
  ///      distinct Var pointers, collapsing them into one Python slot.
  ///
  /// First-run user IR (vars without auto-name suffixes) is unaffected.
  static bool IsAutoVersionedName(const std::string& name) {
    auto parsed = auto_name::Parse(name);
    return parsed.has_auto_suffix && parsed.version.has_value();
  }

  /// Allocate a fresh SSA version for ``original``. When ``original`` is
  /// already SSA-named, preserve its pointer identity (idempotence) and
  /// avoid minting a new Var with a colliding ``name_hint_``.
  ///
  /// Identity-preservation is critical for two downstream paths:
  ///   1. Cross-function references — e.g. an outlined Spmd function's
  ///      ``core_num`` attr ExprPtr or a ``DistributedTensorType``'s
  ///      ``window_buffer_`` field — that hold the Var pointer directly
  ///      and cannot be reached by per-function ``cur_`` substitution.
  ///   2. Codegen paths that use ``name_hint_`` as a Python identifier
  ///      (host_orch ``tensors[...]`` keys, ``CommBufferSpec(name=...)``)
  ///      where two Vars with the same ``name_hint_`` collapse to the same
  ///      slot.
  VarPtr AllocVersion(const VarPtr& original, const TypePtr& type, const Span& span) {
    auto key = original.get();
    // Idempotence path: an auto-versioned Var (``foo__ssa_v0`` / ``foo__tmp_v1``
    // / ``foo__idx_v0`` etc.) that has not been seen yet as a write site in
    // this function is treated as already-SSA — keep its pointer identity.
    // If we've already recorded a write through this Var pointer (e.g. an
    // earlier AssignStmt LHS in straight-line code, or a control-flow pass
    // emitted a multi-assigned auto-named temp), fall through to normal
    // re-versioning so SSA's "exactly one def per Var" invariant holds.
    bool already_seen = cur_.find(key) != cur_.end();
    if (!already_seen && IsAutoVersionedName(original->name_hint_)) {
      cur_[key] = original;
      return original;
    }
    int v = NextVersion(key);
    auto var =
        std::make_shared<Var>(BuildAutoNamedVersion(original->name_hint_, "ssa", v), SubstType(type), span);
    cur_[key] = var;
    return var;
  }

  /// Register pre-existing iter_args from the original loop into cur_.
  /// Each iter_arg needs to update cur_ for its source variable so that
  /// uses inside the loop body get substituted correctly.
  ///
  /// For pre-existing iter_args (from a prior SSA pass / roundtrip), the
  /// original Var pointer is not directly available.  We store a mapping
  /// from iter_arg pointer → source Var pointer when we first process the
  /// loop's existing iter_args (see ConvertFor/ConvertWhile), so we can
  /// look them up here.
  void RegisterIterArgs(const std::vector<IterArgPtr>& ias,
                        const std::unordered_map<const Var*, const Var*>& ia_to_source) {
    for (const auto& ia : ias) {
      auto it = ia_to_source.find(ia.get());
      if (it != ia_to_source.end()) {
        cur_[it->second] = ia;
      }
      // Also register under the iter_arg's own pointer for self-reference
      cur_[ia.get()] = ia;
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
    if (kind == ObjectKind::InCoreScopeStmt || kind == ObjectKind::AutoInCoreScopeStmt ||
        kind == ObjectKind::ClusterScopeStmt || kind == ObjectKind::HierarchyScopeStmt ||
        kind == ObjectKind::SpmdScopeStmt || kind == ObjectKind::RuntimeScopeStmt ||
        kind == ObjectKind::CommDomainScopeStmt) {
      return ConvertScope(As<ScopeStmt>(s));
    }
    return s;
  }

  // ── AssignStmt ─────────────────────────────────────────────────────

  StmtPtr ConvertAssign(const AssignStmtPtr& op) {
    auto val = SubstExpr(op->value_);
    auto var = AllocVersion(op->var_, op->var_->GetType(), op->var_->span_);
    auto result = MutableCopy(op);
    result->var_ = var;
    result->value_ = val;
    return result;
  }

  // ── SeqStmts — computes future uses per-statement for escaping detection

  StmtPtr ConvertSeq(const SeqStmtsPtr& op) {
    AssertNoMidBodyYield(op);
    size_t n = op->stmts_.size();

    // Precompute suffix_needs[i] = variables needed from the outer scope by stmts[i..N-1].
    // Uses order-aware live-in analysis: a variable defined before use within a compound
    // statement is NOT counted as needed. Single backward pass, O(N * stmt_size).
    std::vector<std::unordered_set<const Var*>> suffix_needs(n + 1);
    for (size_t j = n; j > 0; --j) {
      auto live_in = ComputeStmtLiveIn(op->stmts_[j - 1]);
      var_collectors::VarDefUseCollector stmt_collector;
      stmt_collector.VisitStmt(op->stmts_[j - 1]);
      suffix_needs[j - 1] = live_in;
      for (const auto& v : suffix_needs[j]) {
        if (!stmt_collector.var_assign_defs.count(v)) {
          suffix_needs[j - 1].insert(v);
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
    // and build ia_to_source mapping for RegisterIterArgs.
    std::vector<IterArgPtr> ias;
    std::unordered_map<const Var*, const Var*> ia_to_source;
    for (const auto& ia : op->iter_args_) {
      auto new_ia = std::make_shared<IterArg>(ia->name_hint_, SubstType(ia->GetType()),
                                              SubstExpr(ia->initValue_), ia->span_);
      ias.push_back(new_ia);
      // The original iter_arg IS the source variable for cur_ mapping
      ia_to_source[new_ia.get()] = ia.get();
    }

    // Pre-analysis: classify assigned variables
    var_collectors::VarDefUseCollector body_collector;
    body_collector.VisitStmt(op->body_);
    const auto& assigned = body_collector.var_assign_defs;
    auto lv_key = op->loop_var_.get();

    // Loop-carried: assigned in body AND existed before AND not loop_var/existing iter_arg
    std::vector<const Var*> carried;
    for (const auto& assigned_var : assigned) {  // NOLINT: result is sorted below
      if (assigned_var == lv_key) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (ia.get() == assigned_var) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      if (before.count(assigned_var)) carried.push_back(assigned_var);
    }
    std::sort(carried.begin(), carried.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // Pre-detect escaping vars: assigned in body AND NOT existed before AND needed
    // by future code (order-aware: used before redefined in the future sequence).
    // Must be detected BEFORE body conversion so the IfStmt handler can see them
    // in current_version_ (needed for single-branch phi creation, issue #600).
    TypeCollector tc;
    tc.Collect(op->body_);
    std::vector<const Var*> escaping;
    for (const auto& assigned_var : assigned) {  // NOLINT: result is sorted below
      if (assigned_var == lv_key) continue;
      if (before.count(assigned_var)) continue;
      if (!saved_future_needs.count(assigned_var)) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (ia.get() == assigned_var) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      escaping.push_back(assigned_var);
    }
    std::sort(escaping.begin(), escaping.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // Create iter_args + return_vars for carried variables
    std::vector<VarPtr> carried_rvs;
    for (const auto& key : carried) {
      auto init = before.at(key);
      auto init_type = SubstType(init->GetType());
      int iv = NextVersion(key);
      auto new_ia = std::make_shared<IterArg>(BuildAutoNamedVersion(key->name_hint_, "iter", iv), init_type,
                                              init, op->span_);
      ias.push_back(new_ia);
      ia_to_source[new_ia.get()] = key;
      int rv = NextVersion(key);
      carried_rvs.push_back(
          std::make_shared<Var>(BuildAutoNamedVersion(key->name_hint_, "rv", rv), init_type, op->span_));
    }

    // Create iter_args + return_vars for escaping variables (pre-registered)
    std::vector<VarPtr> esc_rvs;
    for (const auto& key : escaping) {
      auto type_it = tc.types.find(key);
      if (type_it == tc.types.end()) continue;
      auto type = SubstType(type_it->second);
      // Pass the unsubstituted ``type_it->second`` so the pointer comparison
      // in ``FindInitValue`` stays on original TypePtr identities — see the
      // FindInitValue docstring.
      auto init = FindInitValue(type_it->second, before);
      if (!init) {
        // Last resort: create a placeholder using any variable with matching type
        // This covers zero-trip loop cases
        init = std::make_shared<Var>(key->name_hint_, type, op->span_);
      }
      int iv = NextVersion(key);
      auto new_ia = std::make_shared<IterArg>(BuildAutoNamedVersion(key->name_hint_, "iter", iv), type, init,
                                              op->span_);
      ias.push_back(new_ia);
      ia_to_source[new_ia.get()] = key;
      int rv = NextVersion(key);
      auto rv_var = std::make_shared<Var>(BuildAutoNamedVersion(key->name_hint_, "rv", rv), type, op->span_);
      esc_rvs.push_back(rv_var);
    }

    // Version loop variable and register iter_args (including escaping)
    int lvv = NextVersion(lv_key);
    auto new_lv = std::make_shared<Var>(BuildAutoNamedVersion(lv_key->name_hint_, "idx", lvv),
                                        SubstType(op->loop_var_->GetType()), op->loop_var_->span_);
    cur_[lv_key] = new_lv;
    RegisterIterArgs(ias, ia_to_source);
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

    // Build return_vars in iter_arg order: existing + carried + escaping.
    // Existing return_vars are freshened when their type embeds renamed Vars
    // (idempotence: re-running ConvertToSSA on already-SSA IR with dynamic
    // shape/valid_shape Var refs must keep type-embedded refs consistent).
    std::vector<VarPtr> all_rvs;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      auto new_type = SubstType(rv->GetType());
      VarPtr eff_rv = (new_type.get() == rv->GetType().get())
                          ? rv
                          : std::make_shared<Var>(rv->name_hint_, new_type, rv->span_);
      if (eff_rv.get() != rv.get()) cur_[rv.get()] = eff_rv;
      if (i < ias.size()) {
        auto it = ia_to_source.find(ias[i].get());
        if (it != ia_to_source.end()) cur_[it->second] = eff_rv;
      }
      all_rvs.push_back(eff_rv);
    }
    for (const auto& rv : carried_rvs) all_rvs.push_back(rv);
    for (const auto& rv : esc_rvs) all_rvs.push_back(rv);

    // Build yields in matching order
    std::vector<ExprPtr> yields;
    if (auto y = ExtractYield(new_body)) yields = y->value_;
    for (const auto& key : carried) yields.push_back(after.at(key));
    for (const auto& key : escaping) {
      auto it = after.find(key);
      if (it != after.end()) {
        yields.push_back(it->second);
      }
    }

    StmtPtr body = new_body;
    if (!yields.empty()) body = ReplaceOrAppendYield(new_body, yields, op->span_);

    auto result = MutableCopy(op);
    result->loop_var_ = std::move(new_lv);
    result->start_ = std::move(new_start);
    result->stop_ = std::move(new_stop);
    result->step_ = std::move(new_step);
    result->iter_args_ = std::move(ias);
    result->body_ = std::move(body);
    result->return_vars_ = std::move(all_rvs);
    return result;
  }

  // ── WhileStmt ──────────────────────────────────────────────────────

  StmtPtr ConvertWhile(const WhileStmtPtr& op) {
    auto saved_future_needs = future_needs_;
    auto before = cur_;

    // Process existing iter_args
    std::vector<IterArgPtr> ias;
    std::unordered_map<const Var*, const Var*> ia_to_source;
    for (const auto& ia : op->iter_args_) {
      auto new_ia = std::make_shared<IterArg>(ia->name_hint_, SubstType(ia->GetType()),
                                              SubstExpr(ia->initValue_), ia->span_);
      ias.push_back(new_ia);
      ia_to_source[new_ia.get()] = ia.get();
    }

    // Pre-analysis
    var_collectors::VarDefUseCollector body_collector;
    body_collector.VisitStmt(op->body_);
    const auto& assigned = body_collector.var_assign_defs;

    // Loop-carried classification
    std::vector<const Var*> carried;
    for (const auto& assigned_var : assigned) {  // NOLINT: result is sorted below
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (ia.get() == assigned_var) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      if (before.count(assigned_var)) carried.push_back(assigned_var);
    }
    std::sort(carried.begin(), carried.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // Pre-detect escaping vars (same logic as ForStmt — see issue #600 comment there)
    TypeCollector tc;
    tc.Collect(op->body_);
    std::vector<const Var*> escaping;
    for (const auto& assigned_var : assigned) {  // NOLINT: result is sorted below
      if (before.count(assigned_var)) continue;
      if (!saved_future_needs.count(assigned_var)) continue;
      bool is_existing_ia = false;
      for (const auto& ia : op->iter_args_) {
        if (ia.get() == assigned_var) {
          is_existing_ia = true;
          break;
        }
      }
      if (is_existing_ia) continue;
      escaping.push_back(assigned_var);
    }
    std::sort(escaping.begin(), escaping.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // Create iter_args + return_vars for carried
    std::vector<VarPtr> carried_rvs;
    for (const auto& key : carried) {
      auto init = before.at(key);
      auto init_type = SubstType(init->GetType());
      int iv = NextVersion(key);
      auto new_ia = std::make_shared<IterArg>(BuildAutoNamedVersion(key->name_hint_, "iter", iv), init_type,
                                              init, op->span_);
      ias.push_back(new_ia);
      ia_to_source[new_ia.get()] = key;
      int rv = NextVersion(key);
      carried_rvs.push_back(
          std::make_shared<Var>(BuildAutoNamedVersion(key->name_hint_, "rv", rv), init_type, op->span_));
    }

    // Create iter_args + return_vars for escaping variables (pre-registered)
    std::vector<VarPtr> esc_rvs;
    for (const auto& key : escaping) {
      auto type_it = tc.types.find(key);
      if (type_it == tc.types.end()) continue;
      auto type = SubstType(type_it->second);
      // Pass the unsubstituted ``type_it->second`` so pointer comparison in
      // ``FindInitValue`` stays on original TypePtr identities — see the
      // FindInitValue docstring.
      auto init = FindInitValue(type_it->second, before);
      if (!init) init = std::make_shared<Var>(key->name_hint_, type, op->span_);
      int iv = NextVersion(key);
      auto new_ia = std::make_shared<IterArg>(BuildAutoNamedVersion(key->name_hint_, "iter", iv), type, init,
                                              op->span_);
      ias.push_back(new_ia);
      ia_to_source[new_ia.get()] = key;
      int rv = NextVersion(key);
      esc_rvs.push_back(
          std::make_shared<Var>(BuildAutoNamedVersion(key->name_hint_, "rv", rv), type, op->span_));
    }

    // Register iter_args (including escaping), substitute condition, convert body
    RegisterIterArgs(ias, ia_to_source);
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

    // Build return_vars: existing + carried + escaping. Freshen existing rvs
    // when their type embeds renamed Vars (idempotence — see ConvertFor for
    // the same pattern).
    std::vector<VarPtr> all_rvs;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      auto new_type = SubstType(rv->GetType());
      VarPtr eff_rv = (new_type.get() == rv->GetType().get())
                          ? rv
                          : std::make_shared<Var>(rv->name_hint_, new_type, rv->span_);
      if (eff_rv.get() != rv.get()) cur_[rv.get()] = eff_rv;
      if (i < ias.size()) {
        auto it = ia_to_source.find(ias[i].get());
        if (it != ia_to_source.end()) cur_[it->second] = eff_rv;
      }
      all_rvs.push_back(eff_rv);
    }
    for (const auto& rv : carried_rvs) all_rvs.push_back(rv);
    for (const auto& rv : esc_rvs) all_rvs.push_back(rv);

    // Build yields
    std::vector<ExprPtr> yields;
    if (auto y = ExtractYield(new_body)) yields = y->value_;
    for (const auto& key : carried) yields.push_back(after.at(key));
    for (const auto& key : escaping) {
      auto it = after.find(key);
      if (it != after.end()) yields.push_back(it->second);
    }

    StmtPtr body = new_body;
    if (!yields.empty()) body = ReplaceOrAppendYield(new_body, yields, op->span_);

    auto result = MutableCopy(op);
    result->condition_ = std::move(new_cond);
    result->iter_args_ = std::move(ias);
    result->body_ = std::move(body);
    result->return_vars_ = std::move(all_rvs);
    return result;
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
    for (const auto& [key, v] : then_ver) {  // NOLINT(bugprone-nondeterministic-pointer-iteration-order)
      seen.insert(key);
      auto bi = before.find(key);
      if (bi != before.end()) {
        bool then_changed = (bi->second != v);
        auto ei = else_ver.find(key);
        bool else_changed = (ei != else_ver.end() && bi->second != ei->second);
        if (then_changed || else_changed) phis.push_back(key);
      } else if (else_ver.count(key)) {
        // New variable defined in BOTH branches needs a phi
        phis.push_back(key);
      }
    }
    for (const auto& [key, v] : else_ver) {  // NOLINT(bugprone-nondeterministic-pointer-iteration-order)
      if (seen.count(key)) continue;
      auto bi = before.find(key);
      if (bi != before.end() && bi->second != v) phis.push_back(key);
    }
    std::sort(phis.begin(), phis.end(),
              [](const Var* a, const Var* b) { return a->name_hint_ < b->name_hint_; });

    // No divergence — return simple IfStmt
    if (phis.empty() && op->return_vars_.empty()) {
      cur_ = before;
      auto result = MutableCopy(op);
      result->condition_ = std::move(cond);
      result->then_body_ = std::move(new_then);
      result->else_body_ = std::move(new_else);
      result->return_vars_ = {};
      return result;
    }

    // No new phis but existing return_vars (explicit SSA) — version return_vars, keep branch yields
    if (phis.empty()) {
      cur_ = before;
      std::vector<VarPtr> return_vars;
      for (const auto& rv : op->return_vars_) {
        auto rv_key = rv.get();
        int v = NextVersion(rv_key);
        auto nrv = std::make_shared<Var>(BuildAutoNamedVersion(rv_key->name_hint_, "rv", v),
                                         SubstType(rv->GetType()), rv->span_);
        return_vars.push_back(nrv);
        cur_[rv_key] = nrv;
      }
      auto result = MutableCopy(op);
      result->condition_ = std::move(cond);
      result->then_body_ = std::move(new_then);
      result->else_body_ = std::move(new_else);
      result->return_vars_ = std::move(return_vars);
      return result;
    }

    // Create phi outputs
    cur_ = before;
    std::vector<VarPtr> return_vars;
    std::vector<ExprPtr> then_yields, else_yields;

    for (const auto& key : phis) {
      VarPtr tv = then_ver.count(key) ? then_ver.at(key) : before.at(key);
      VarPtr ev = else_ver.count(key) ? else_ver.at(key) : before.at(key);
      int pv = NextVersion(key);
      auto phi = std::make_shared<Var>(BuildAutoNamedVersion(key->name_hint_, "phi", pv),
                                       SubstType(tv->GetType()), op->span_);
      return_vars.push_back(phi);
      then_yields.push_back(tv);
      else_yields.push_back(ev);
      cur_[key] = phi;
    }

    // Preserve any existing return_vars not already handled as phis
    for (const auto& rv : op->return_vars_) {
      auto rv_key = rv.get();
      bool handled = false;
      for (const auto& p : phis) {
        if (p == rv_key) {
          handled = true;
          break;
        }
      }
      if (!handled) {
        int v = NextVersion(rv_key);
        auto nrv = std::make_shared<Var>(BuildAutoNamedVersion(rv_key->name_hint_, "rv", v),
                                         SubstType(rv->GetType()), rv->span_);
        return_vars.push_back(nrv);
        cur_[rv_key] = nrv;
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

    auto result = MutableCopy(op);
    result->condition_ = std::move(cond);
    result->then_body_ = std::move(then_with_yield);
    result->else_body_ = std::make_optional(std::move(else_with_yield));
    result->return_vars_ = std::move(return_vars);
    return result;
  }

  // ── Simple statements ──────────────────────────────────────────────

  StmtPtr ConvertReturn(const ReturnStmtPtr& op) {
    std::vector<ExprPtr> vals;
    for (const auto& v : op->value_) vals.push_back(SubstExpr(v));
    auto result = MutableCopy(op);
    result->value_ = std::move(vals);
    return result;
  }

  StmtPtr ConvertYield(const YieldStmtPtr& op) {
    std::vector<ExprPtr> vals;
    for (const auto& v : op->value_) vals.push_back(SubstExpr(v));
    auto result = MutableCopy(op);
    result->value_ = std::move(vals);
    return result;
  }

  StmtPtr ConvertEval(const EvalStmtPtr& op) {
    auto e = SubstExpr(op->expr_);
    if (e == op->expr_) return op;
    auto result = MutableCopy(op);
    result->expr_ = e;
    return result;
  }

  /// Substitute Var-typed entries in a ScopeStmt's ``attrs_``
  /// (``manual_dep_edges`` / ``task_id_var`` / ``arg_direction_overrides_vars`` /
  /// ``dump_vars``). Returns the rebuilt attrs and a flag indicating whether any
  /// entry was rewritten — mirrors the per-Call ``SubstCallAttrs`` so SSA
  /// renaming propagates into scope-level attrs the same way it does for Call
  /// attrs. ``dump_vars`` rides here as the carrier from ``pl.dump_tag`` /
  /// ``dumps=`` to the outliner: substituting BEFORE the body (see
  /// ``ConvertScope``) resolves each tagged tensor to the SSA version visible at
  /// scope entry — exactly the value the synthesised dispatch receives as an arg.
  std::pair<std::vector<std::pair<std::string, std::any>>, bool> SubstScopeAttrs(
      const std::vector<std::pair<std::string, std::any>>& attrs) {
    bool changed = false;
    std::vector<std::pair<std::string, std::any>> out;
    out.reserve(attrs.size());
    for (const auto& [k, v] : attrs) {
      if (k == kAttrManualDepEdges || k == kAttrArgDirOverrideVars || k == kAttrDumpVars) {
        const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
        if (edges) {
          std::vector<VarPtr> new_edges;
          new_edges.reserve(edges->size());
          bool any = false;
          for (const auto& e : *edges) {
            if (!e) {
              new_edges.push_back(e);
              continue;
            }
            auto it = cur_.find(e.get());
            if (it != cur_.end() && it->second.get() != e.get()) {
              new_edges.push_back(it->second);
              any = true;
            } else {
              new_edges.push_back(e);
            }
          }
          if (any) {
            changed = true;
            out.emplace_back(k, std::any(std::move(new_edges)));
            continue;
          }
        }
      } else if (k == kAttrTaskIdVar) {
        const auto* var = std::any_cast<VarPtr>(&v);
        if (var && *var) {
          auto it = cur_.find(var->get());
          if (it != cur_.end() && it->second.get() != var->get()) {
            changed = true;
            out.emplace_back(k, std::any(it->second));
            continue;
          }
        }
      }
      out.emplace_back(k, v);
    }
    return {std::move(out), changed};
  }

  StmtPtr ConvertScope(const ScopeStmtPtr& op) {
    // Substitute attrs (manual_dep_edges / task_id_var /
    // arg_direction_overrides_vars) BEFORE converting the body. Body
    // conversion advances ``cur_`` past any writes performed inside the
    // scope, so substituting after would resolve attr Var references to
    // post-body yield-result versions rather than to the SSA versions
    // visible at scope entry. The latter is what the user wrote — e.g.
    // ``with pl.at(..., no_dep_args=[k_cache])`` where ``k_cache`` names
    // an outer loop iter_arg, the body then reassigns ``k_cache`` via
    // ``pl.assemble``. The attr must point at the iter_arg, not the rebuilt
    // ``k_cache__rv_*`` from the yielded body.
    auto subst = SubstScopeAttrs(op->attrs_);
    // SpmdScopeStmt::core_num_ is evaluated in the OUTER scope — substitute
    // it BEFORE converting the body so it picks up the pre-body ``cur_``
    // versions (mirrors ConvertFor's handling of start_/stop_/step_).
    // Other ScopeStmt subclasses have no Var-bearing direct fields.
    ExprPtr new_core_num;
    bool core_num_changed = false;
    if (auto spmd = As<SpmdScopeStmt>(op)) {
      new_core_num = SubstExpr(spmd->core_num_);
      core_num_changed = (new_core_num.get() != spmd->core_num_.get());
    }

    // Block escaping-var promotion across non-Runtime scope boundaries (#1351).
    //
    // ``HierarchyScopeStmt`` / ``InCoreScopeStmt`` / ``AutoInCoreScopeStmt`` /
    // ``ClusterScopeStmt`` / ``SpmdScopeStmt`` separate the loops *inside*
    // their body from the use-site of any variable defined further down the
    // *outer* sequence. The inner loops cannot manufacture a working init
    // value for such a use (FindInitValue typically falls back to an
    // unversioned ``foo__FREE_VAR`` placeholder, which the SSA verifier
    // rejects) and would carry a value that is never threaded back to the
    // outer use site anyway. Trim ``future_needs_`` to variables already
    // visible in ``cur_`` at scope entry so the inner-loop escaping
    // detection (see ConvertFor::escaping above) ignores scope-local vars.
    //
    // ``cur_`` is intentionally NOT restored after the body — variables
    // first-defined inside the body and referenced after the scope must
    // still substitute to their in-body SSA version (relied on by passes
    // like InterchangeChunkLoops that emit ``out = pl.assemble(...)`` inside
    // ``pl.at`` and return ``out`` outside). Whether such a leak is
    // user-legal is enforced by other property verifiers, not by SSA.
    //
    // ``RuntimeScopeStmt`` is a thin ``pl.scope()`` codegen wrapper, not a
    // boundary — its body shares SSA state with the enclosing function and
    // stays fully transparent.
    const bool is_outline_boundary = !As<RuntimeScopeStmt>(op);
    std::unordered_set<const Var*> saved_future_needs;
    if (is_outline_boundary) {
      saved_future_needs = future_needs_;
      std::unordered_set<const Var*> restricted;
      restricted.reserve(future_needs_.size());
      for (const auto* v : future_needs_) {
        if (cur_.count(v)) restricted.insert(v);
      }
      future_needs_ = std::move(restricted);
    }

    auto body = ConvertStmt(op->body_);

    if (is_outline_boundary) {
      future_needs_ = std::move(saved_future_needs);
    }
    auto& new_attrs = subst.first;
    const bool attrs_changed = subst.second;
    if (body == op->body_ && !attrs_changed && !core_num_changed) return op;
    // ScopeStmt is abstract; dispatch on the concrete derived class so MutableCopy
    // can construct the right subclass. Structured bindings are intentionally
    // avoided above — capturing them in this lambda is non-portable C++17
    // (clang-tidy rejects it as clang-diagnostic-error).
    auto rewrite = [&](auto&& concrete) -> StmtPtr {
      auto result = MutableCopy(concrete);
      result->body_ = body;
      if (attrs_changed) result->attrs_ = std::move(new_attrs);
      return result;
    };
    if (auto in_core = As<InCoreScopeStmt>(op)) return rewrite(in_core);
    if (auto auto_in_core = As<AutoInCoreScopeStmt>(op)) return rewrite(auto_in_core);
    if (auto cluster = As<ClusterScopeStmt>(op)) return rewrite(cluster);
    if (auto hier = As<HierarchyScopeStmt>(op)) return rewrite(hier);
    if (auto spmd = As<SpmdScopeStmt>(op)) {
      auto result = MutableCopy(spmd);
      result->body_ = body;
      if (attrs_changed) result->attrs_ = std::move(new_attrs);
      if (core_num_changed) result->core_num_ = std::move(new_core_num);
      return result;
    }
    if (auto runtime_scope = As<RuntimeScopeStmt>(op)) return rewrite(runtime_scope);
    if (auto comm_domain = As<CommDomainScopeStmt>(op)) return rewrite(comm_domain);
    INTERNAL_UNREACHABLE_SPAN(op->span_) << "Unknown ScopeStmt subclass: " << op->TypeName();
    return op;
  }

  // ── Helpers ────────────────────────────────────────────────────────

  /// Find a pre-loop Var whose original type matches ``orig_type``. ``pre``
  /// keys are the original (pre-substitution) Var pointers and values are
  /// their current SSA versions. Caller passes the **un-substituted**
  /// ``type_it->second`` from ``tc.types`` (not the ``SubstType``-applied
  /// version) so this comparison stays on the original TypePtr identities —
  /// substituting both sides via ``SubstType`` would mint fresh
  /// shared_ptrs that compare unequal even when structurally identical,
  /// causing escape-promotion to fall back to a ``foo__FREE_VAR`` placeholder
  /// the SSA verifier rejects.
  VarPtr FindInitValue(const TypePtr& orig_type, const std::unordered_map<const Var*, VarPtr>& pre) {
    // Prefer Out/InOut parameter with matching original type.
    for (size_t i = 0; i < orig_params_.size(); ++i) {
      if (orig_param_directions_[i] == ParamDirection::Out ||
          orig_param_directions_[i] == ParamDirection::InOut) {
        auto key = orig_params_[i].get();
        if (orig_params_[i]->GetType() != orig_type) continue;
        auto it = pre.find(key);
        if (it != pre.end()) return it->second;
      }
    }
    // Fall back to any pre-loop variable with matching original type.
    std::vector<std::pair<uint64_t, VarPtr>> candidates;
    for (const auto& [key, v] : pre) {
      if (key->GetType() == orig_type) {
        candidates.emplace_back(key->UniqueId(), v);
      }
    }
    if (!candidates.empty()) {
      std::sort(candidates.begin(), candidates.end());
      return candidates.front().second;
    }
    return nullptr;
  }

  // Invariant: a YieldStmt may appear only as the trailing statement of its
  // scope. ConvertSeq and the loop/if helpers below only inspect / pop the
  // last stmt, so a mid-body YieldStmt would silently survive
  // ReplaceOrAppendYield and produce a SeqStmts with two yields. Assert at the
  // source rather than letting the structural verifier blame the resulting
  // shape.
  static void AssertNoMidBodyYield(const StmtPtr& s) {
    auto seq = As<SeqStmts>(s);
    if (!seq) return;
    for (size_t i = 0; i + 1 < seq->stmts_.size(); ++i) {
      INTERNAL_CHECK_SPAN(!As<YieldStmt>(seq->stmts_[i]), seq->stmts_[i]->span_)
          << "ConvertToSSA: body has a YieldStmt at position " << i << " of " << seq->stmts_.size()
          << "; YieldStmt must be the trailing statement of its scope. "
          << "A producing pass emitted malformed IR.";
    }
  }

  static YieldStmtPtr ExtractYield(const StmtPtr& s) {
    AssertNoMidBodyYield(s);
    if (auto y = As<YieldStmt>(s)) {
      return y;
    }
    // RuntimeScopeStmt is transparent to SSA: a for/if body wrapped in a
    // ``with pl.scope()`` keeps its trailing carry-yield inside the scope, so
    // look through it (and through any trailing nested scope) to find it.
    if (auto scope = As<RuntimeScopeStmt>(s)) {
      return ExtractYield(scope->body_);
    }
    if (auto seq = As<SeqStmts>(s)) {
      if (!seq->stmts_.empty()) {
        return ExtractYield(seq->stmts_.back());
      }
    }
    return nullptr;
  }

  static StmtPtr ReplaceOrAppendYield(const StmtPtr& s, const std::vector<ExprPtr>& vals, const Span& span) {
    AssertNoMidBodyYield(s);
    auto yield = std::make_shared<YieldStmt>(vals, span);
    // Transparent through a RuntimeScopeStmt: replace the carry-yield *inside*
    // the scope body and keep the scope wrapper (codegen still needs it).
    if (auto scope = As<RuntimeScopeStmt>(s)) {
      auto copy = MutableCopy(scope);
      copy->body_ = ReplaceOrAppendYield(scope->body_, vals, span);
      return copy;
    }
    if (auto seq = As<SeqStmts>(s)) {
      std::vector<StmtPtr> stmts = seq->stmts_;
      // The trailing statement carries the yield when it is a YieldStmt or a
      // scope ending in one — replace it in place; otherwise append.
      bool has_trailing_yield = !stmts.empty() && ExtractYield(stmts.back()) != nullptr;
      if (has_trailing_yield) {
        stmts.back() = ReplaceOrAppendYield(stmts.back(), vals, span);
      } else {
        stmts.push_back(yield);
      }
      return SeqStmts::Flatten(std::move(stmts), seq->span_);
    }
    if (As<YieldStmt>(s)) {
      return yield;
    }
    return SeqStmts::Flatten({s, yield}, span);
  }

  // ── State ──────────────────────────────────────────────────────────

  std::unordered_map<const Var*, VarPtr> cur_;   // var pointer → latest version
  std::unordered_map<const Var*, int> ver_;      // var pointer → next version number
  std::unordered_set<const Var*> future_needs_;  // vars needed in subsequent stmts
  std::vector<VarPtr> orig_params_;              // original function params
  std::vector<ParamDirection> orig_param_directions_;
};

/// Should this function be skipped by ConvertToSSA?
///
/// HOST-level (Linqu >= 3) SubWorker functions carry their pure-Python body
/// as an opaque InlineStmt. Their parameters are not used by IR statements,
/// and SubWorker code generation references the user's original parameter
/// names verbatim from the captured source — so SSA renaming would only
/// desync the params from the body.
bool ShouldSkipFunction(const FunctionPtr& func) {
  return func && func->role_.has_value() && *func->role_ == Role::SubWorker && func->level_.has_value() &&
         LevelToLinquLevel(*func->level_) >= 3;
}

/// Result of converting one function to SSA: the new function plus the rename
/// map (original Var pointer → SSA Var) captured from the converter's ``cur_``.
struct PerFunctionSSAResult {
  FunctionPtr func;
  std::unordered_map<const Var*, VarPtr> renames;
};

PerFunctionSSAResult TransformConvertToSSAFunction(const FunctionPtr& func) {
  if (ShouldSkipFunction(func)) {
    return {func, {}};
  }
  SSAConverter converter;
  auto new_func = converter.ConvertFunction(func);
  return {new_func, converter.cur()};
}

/// Substitute Var references inside an Expr using a flat rename map. Used to
/// propagate SSA renames from the function in which a Var was defined into
/// sibling functions' Var-bearing attrs (cross-function references that the
/// per-function converter cannot see).
class ExprVarRenamer : public IRMutator {
 public:
  explicit ExprVarRenamer(const std::unordered_map<const Var*, VarPtr>& renames) : renames_(renames) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = renames_.find(op.get());
    return it != renames_.end() ? it->second : op;
  }
  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = renames_.find(op.get());
    return it != renames_.end() ? it->second : op;
  }

 private:
  const std::unordered_map<const Var*, VarPtr>& renames_;
};

/// Walk a function's attrs and substitute Var refs per the program-wide rename
/// map. Returns ``true`` if any attr was rewritten.
///
/// Known Var-bearing function attrs:
///   - ``core_num`` (ExprPtr): SpmdScopeStmt's dispatch core count, lifted to
///     the outlined Spmd function by OutlineIncoreScopes/OutlineClusterScopes.
///     References Vars defined in the *dispatching* Orchestration function.
///
/// Extend this enumeration if a future outline pass introduces another Var-
/// bearing function-level attr.
bool SubstFunctionAttrs(std::vector<std::pair<std::string, std::any>>& attrs,
                        const std::unordered_map<const Var*, VarPtr>& renames) {
  if (renames.empty()) return false;
  bool changed = false;
  for (auto& [k, v] : attrs) {
    if (k == "core_num") {
      const auto* expr_ptr = std::any_cast<ExprPtr>(&v);
      if (expr_ptr && *expr_ptr) {
        ExprVarRenamer renamer(renames);
        auto new_expr = renamer.VisitExpr(*expr_ptr);
        if (new_expr.get() != expr_ptr->get()) {
          v = std::any(new_expr);
          changed = true;
        }
      }
    }
  }
  return changed;
}

/// Program-level ConvertToSSA driver.
///
/// Two phases:
///   1. Per-function SSA conversion. Accumulate the (original Var → SSA Var)
///      mapping into a single ``program_renames`` table.
///   2. Walk every function's attrs and substitute cross-function Var refs
///      using ``program_renames``. This is what keeps an outlined Spmd
///      function's ``core_num`` attr in sync after the dispatching
///      Orchestration function is re-versioned by Phase 1 (the per-function
///      converter has no visibility into sibling functions' attrs).
ProgramPtr TransformConvertToSSAProgram(const ProgramPtr& program) {
  if (!program) return program;
  auto new_functions = program->functions_;
  std::unordered_map<const Var*, VarPtr> program_renames;
  bool changed = false;

  // Phase 1: per-function SSA conversion.
  for (auto& [gvar, func] : new_functions) {
    if (!func) continue;
    auto result = TransformConvertToSSAFunction(func);
    if (result.func.get() != func.get()) {
      func = result.func;
      changed = true;
    }
    for (const auto& [orig, current] : result.renames) {
      if (orig != current.get()) {
        program_renames[orig] = current;
      }
    }
  }

  // Phase 2: cross-function attrs Var substitution.
  if (!program_renames.empty()) {
    for (auto& [gvar, func] : new_functions) {
      if (!func || func->attrs_.empty()) continue;
      auto attrs_copy = func->attrs_;
      if (SubstFunctionAttrs(attrs_copy, program_renames)) {
        auto mutated = MutableCopy(func);
        mutated->attrs_ = std::move(attrs_copy);
        func = mutated;
        changed = true;
      }
    }
  }

  if (!changed) return program;
  return std::make_shared<const Program>(std::move(new_functions), program->name_, program->span_);
}

}  // namespace

namespace pass {
Pass ConvertToSSA() {
  return CreateProgramPass(TransformConvertToSSAProgram, "ConvertToSSA", kConvertToSSAProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
