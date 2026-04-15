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

#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/verifier/property_verifier_registry.h"

namespace pypto {
namespace ir {
namespace stmt_dep {

// ---------------------------------------------------------------------------
// BuildStmtDependencyGraph
// ---------------------------------------------------------------------------

StmtDependencyGraph BuildStmtDependencyGraph(const StmtPtr& region, const ProgramPtr& program) {
  if (program) CheckInOutUseDiscipline(region, program);

  StmtDependencyGraph graph;
  if (!region) return graph;

  // Non-SeqStmts regions are single-node graphs with no edges.
  auto seq = As<SeqStmts>(region);
  if (!seq) {
    graph.stmts.push_back(region);
    graph.predecessors[region.get()] = {};
    return graph;
  }

  graph.stmts = seq->stmts_;
  // Ensure every stmt has an entry, even if it has no predecessors.
  for (const auto& stmt : graph.stmts) {
    graph.predecessors[stmt.get()] = {};
  }

  // Last stmt in the region that defined each Var (tracked by raw pointer).
  std::unordered_map<const Var*, const Stmt*> last_def;

  for (const auto& stmt : graph.stmts) {
    var_collectors::VarDefUseCollector collector;
    collector.VisitStmt(stmt);

    const Stmt* raw_stmt = stmt.get();

    // Uses → predecessor edges from the last intra-region def of the read var.
    for (const Var* v : collector.var_uses) {
      auto it = last_def.find(v);
      if (it != last_def.end() && it->second != raw_stmt) {
        graph.predecessors[raw_stmt].insert(it->second);
      }
    }

    // Defs → update last_def. A stmt that both defines and uses the same var
    // shadows any prior definition for subsequent stmts; the guard above
    // prevents self-loops.
    for (const Var* v : collector.var_defs) {
      last_def[v] = raw_stmt;
    }
  }

  return graph;
}

// ---------------------------------------------------------------------------
// CheckInOutUseDiscipline
// ---------------------------------------------------------------------------

namespace {

/// Visitor that walks a region in CFG order and flags post-call reads of
/// InOut/Out-passed variables.
///
/// The visitor distinguishes read contexts from definition contexts by
/// overriding each stmt visitor: we only descend into RHS / condition / body
/// fields, never into LHS / loop_var_ / return_vars_ / iter_args themselves.
/// This prevents definition sites from being falsely reported as reads.
///
/// Loop back-edges are modelled conservatively: before entering a loop body,
/// the visitor pre-populates `dead_` with every var the body would kill. This
/// captures back-edge reachability: a read in iteration N+1 of a var that
/// iteration N's call marked InOut/Out is flagged on the first (and only)
/// pass over the body.
class InOutUseDisciplineChecker : public IRVisitor {
 public:
  explicit InOutUseDisciplineChecker(ProgramPtr program) : program_(std::move(program)) {}

  std::vector<Diagnostic> TakeDiagnostics() { return std::move(diagnostics_); }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    // Only read contexts reach this hook — the overridden stmt visitors skip
    // definition fields before recursion lands here.
    const Var* raw = op.get();
    if (dead_.count(raw) != 0) {
      auto origin_it = dead_origin_.find(raw);
      std::string origin_str =
          origin_it != dead_origin_.end() ? origin_it->second.to_string() : std::string("<unknown>");
      std::string msg = "variable '" + op->name_hint_ + "' was passed as InOut/Out at " + origin_str +
                        "; read the post-call return value instead of the pre-call variable";
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "InOutUseDiscipline", 0, std::move(msg),
                                op->span_);
    }
    // Delegate to base so TensorType::shape_ expressions on the var are still visited.
    IRVisitor::VisitVarLike_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    // Visit args first — reads in the args happen logically before the call's
    // effect, so self-reads like `f(T, inout=T)` remain allowed.
    for (const auto& arg : op->args_) {
      VisitExpr(arg);
    }

    // Resolve callee. Built-in ops (tile.*, tensor.*, system.*) won't resolve;
    // they do not contribute to the dead set. Their memory mutations are
    // handled as Mode B in RFC #1026, which is out of scope here.
    if (!program_) return;
    FunctionPtr callee = program_->GetFunction(op->op_->name_);
    if (!callee) return;

    const size_t n = std::min(callee->param_directions_.size(), op->args_.size());
    for (size_t i = 0; i < n; ++i) {
      ParamDirection dir = callee->param_directions_[i];
      if (dir != ParamDirection::InOut && dir != ParamDirection::Out) continue;
      VarPtr var = AsVarLike(op->args_[i]);
      if (!var) continue;
      const Var* raw = var.get();
      dead_.insert(raw);
      // Only record the first origin span per var — subsequent InOut/Out
      // passes of the same var don't change the "dead" status.
      dead_origin_.emplace(raw, op->span_);
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // LHS (`op->var_`) is a definition — skip it. Only the RHS is a read.
    VisitExpr(op->value_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    // Then- and else-branches are mutually exclusive at runtime, so a
    // post-call mark added in one branch must not bleed into the other.
    // `return_vars_` are definitions — skip them.
    VisitExpr(op->condition_);

    auto snapshot = dead_;
    VisitStmt(op->then_body_);
    auto dead_after_then = dead_;

    dead_ = std::move(snapshot);
    if (op->else_body_.has_value()) {
      VisitStmt(*op->else_body_);
    }

    // Merge: a var is dead after the if iff it was dead in either branch.
    // Iteration order doesn't affect the result (insert into unordered_set is
    // commutative and idempotent), so the range-insert form is deterministic.
    dead_.insert(dead_after_then.begin(), dead_after_then.end());
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    // Header reads: bounds and iter_args' initial values.
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    if (op->chunk_config_.has_value() && op->chunk_config_->size) {
      VisitExpr(op->chunk_config_->size);
    }
    for (const auto& ia : op->iter_args_) {
      if (ia->initValue_) VisitExpr(ia->initValue_);
    }
    // `loop_var_`, `iter_args_` themselves, and `return_vars_` are definitions
    // at the loop header — skip them.

    // Back-edge modelling: any kill anywhere in the body is reachable from
    // iteration N's tail to iteration N+1's start, so the var must be dead
    // at body entry for the subsequent iteration's reads to be flagged.
    // Pre-populate `dead_` with the body's kills, then walk the body.
    CollectKillsInto(op->body_, dead_, dead_origin_);
    VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    VisitExpr(op->condition_);
    for (const auto& ia : op->iter_args_) {
      if (ia->initValue_) VisitExpr(ia->initValue_);
    }
    // `iter_args_` and `return_vars_` are definitions — skip them.
    CollectKillsInto(op->body_, dead_, dead_origin_);
    VisitStmt(op->body_);
  }

 private:
  /// Pre-scan a subtree and merge every var it would mark InOut/Out-dead into
  /// `dead` (with origin span recorded in `origin`). Uses a fresh sub-checker
  /// so the main walk's state and diagnostics are untouched.
  void CollectKillsInto(const StmtPtr& stmt, std::unordered_set<const Var*>& dead,
                        std::unordered_map<const Var*, Span>& origin) const {
    if (!stmt) return;
    InOutUseDisciplineChecker sub(program_);
    sub.VisitStmt(stmt);
    dead.insert(sub.dead_.begin(), sub.dead_.end());
    for (const auto& [v, span] : sub.dead_origin_) {
      origin.emplace(v, span);
    }
    // Sub-diagnostics are intentionally discarded: this is a precondition
    // walk, not the real check. The main walk over the same body will
    // surface any violations.
  }

  ProgramPtr program_;
  std::unordered_set<const Var*> dead_;
  std::unordered_map<const Var*, Span> dead_origin_;
  std::vector<Diagnostic> diagnostics_;
};

}  // namespace

void CheckInOutUseDiscipline(const StmtPtr& region, const ProgramPtr& program) {
  if (!region) return;
  InOutUseDisciplineChecker checker(program);
  checker.VisitStmt(region);
  auto diagnostics = checker.TakeDiagnostics();
  if (diagnostics.empty()) return;

  std::string report = PropertyVerifierRegistry::GenerateReport(diagnostics);
  throw VerificationError(report, std::move(diagnostics));
}

}  // namespace stmt_dep
}  // namespace ir
}  // namespace pypto
