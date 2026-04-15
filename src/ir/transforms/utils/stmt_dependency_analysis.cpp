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

namespace pypto {
namespace ir {
namespace stmt_dep {

// ---------------------------------------------------------------------------
// BuildStmtDependencyGraph
// ---------------------------------------------------------------------------

StmtDependencyGraph BuildStmtDependencyGraph(const StmtPtr& region) {
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
class InOutUseDisciplineChecker : public IRVisitor {
 public:
  explicit InOutUseDisciplineChecker(ProgramPtr program) : program_(std::move(program)) {}

  std::vector<Diagnostic> TakeDiagnostics() { return std::move(diagnostics_); }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
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
    // Delegate to base so IterArg::initValue_ is still visited.
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

  void VisitStmt_(const IfStmtPtr& op) override {
    // Then- and else-branches are mutually exclusive at runtime, so a
    // post-call mark added in one branch must not bleed into the other.
    // Snapshot `dead_` before each branch, then take the union afterwards —
    // reads *after* the if-stmt still see the effect of either branch, but
    // reads *within* the sibling branch don't.
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

 private:
  ProgramPtr program_;
  std::unordered_set<const Var*> dead_;
  std::unordered_map<const Var*, Span> dead_origin_;
  std::vector<Diagnostic> diagnostics_;
};

}  // namespace

std::vector<Diagnostic> CheckInOutUseDiscipline(const StmtPtr& region, const ProgramPtr& program) {
  if (!region) return {};
  InOutUseDisciplineChecker checker(program);
  checker.VisitStmt(region);
  return checker.TakeDiagnostics();
}

}  // namespace stmt_dep
}  // namespace ir
}  // namespace pypto
