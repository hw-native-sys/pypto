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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// Helpers
// ============================================================================

static ExprPtr MakeConstBool(bool value, const Span& span) {
  return std::make_shared<ConstBool>(value, span);
}

static ExprPtr MakeAndExpr(const ExprPtr& left, const ExprPtr& right, const Span& span) {
  return std::make_shared<And>(left, right, DataType::BOOL, span);
}

static StmtPtr MakeSeq(std::vector<StmtPtr> stmts, const Span& span) {
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return std::make_shared<SeqStmts>(std::move(stmts), span);
}

static std::vector<StmtPtr> FlattenToVec(const StmtPtr& stmt) {
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    return seq->stmts_;
  }
  return {stmt};
}

static std::string GetAutoNamingBase(const std::string& name) {
  auto parsed = auto_name::Parse(name);
  if (parsed.has_auto_suffix && (parsed.role.has_value() || parsed.version.has_value())) {
    return parsed.base_name;
  }
  return auto_name::GetCompatibleBaseName(name);
}

static std::string GetExprAutoNamingBase(const ExprPtr& expr) {
  if (auto var = std::dynamic_pointer_cast<const Var>(expr)) {
    return GetAutoNamingBase(var->name_hint_);
  }
  if (auto iter_arg = std::dynamic_pointer_cast<const IterArg>(expr)) {
    return GetAutoNamingBase(iter_arg->name_hint_);
  }
  return "phi";
}

/// Simple mutator that substitutes one Var (or IterArg) for another by unique_id.
class VarSubstituter : public IRMutator {
 public:
  VarSubstituter(uint64_t old_id, ExprPtr replacement)
      : old_id_(old_id), replacement_(std::move(replacement)) {}

  ExprPtr VisitExpr_(const VarPtr& op) override { return MaybeReplace(op); }
  ExprPtr VisitExpr_(const IterArgPtr& op) override { return MaybeReplace(op); }

 private:
  uint64_t old_id_;
  ExprPtr replacement_;

  template <typename T>
  ExprPtr MaybeReplace(const T& op) {
    return op->UniqueId() == old_id_ ? replacement_ : static_cast<ExprPtr>(op);
  }
};

// ============================================================================
// Scanner: detect break/continue without entering nested loops
// ============================================================================

struct ScanResult {
  bool has_break = false;
  bool has_continue = false;
};

class BreakContinueScanner : public IRVisitor {
 public:
  ScanResult Scan(const StmtPtr& stmt) {
    result_ = {};
    VisitStmt(stmt);
    return result_;
  }

 protected:
  void VisitStmt_(const BreakStmtPtr& /*op*/) override { result_.has_break = true; }
  void VisitStmt_(const ContinueStmtPtr& /*op*/) override { result_.has_continue = true; }
  void VisitStmt_(const ForStmtPtr& /*op*/) override {}
  void VisitStmt_(const WhileStmtPtr& /*op*/) override {}

 private:
  ScanResult result_;
};

// ============================================================================
// Backward Resolution: compute yield values at escape point
// ============================================================================

/// Given the original yield values and the statements that precede the escape
/// point, compute what values to yield. For each yield value, if the variable
/// is defined before the escape point, use it; otherwise use the iter_arg.
static std::vector<ExprPtr> ResolveYieldAtEscape(const std::vector<ExprPtr>& original_yield_values,
                                                 const std::vector<StmtPtr>& pre_escape_stmts,
                                                 const std::vector<IterArgPtr>& iter_args) {
  std::unordered_set<uint64_t> available_vars;
  for (const auto& stmt : pre_escape_stmts) {
    auto flat = FlattenToVec(stmt);
    for (const auto& s : flat) {
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
        available_vars.insert(assign->var_->UniqueId());
      }
    }
  }

  for (const auto& ia : iter_args) {
    available_vars.insert(ia->UniqueId());
  }

  std::vector<ExprPtr> resolved;
  resolved.reserve(original_yield_values.size());

  for (size_t j = 0; j < original_yield_values.size(); ++j) {
    const auto& val = original_yield_values[j];
    auto var = std::dynamic_pointer_cast<const Var>(val);
    if (!var) {
      resolved.push_back(val);
      continue;
    }

    if (available_vars.count(var->UniqueId())) {
      resolved.push_back(val);
      continue;
    }

    if (j < iter_args.size()) {
      resolved.push_back(iter_args[j]);
    } else {
      resolved.push_back(val);
    }
  }

  return resolved;
}

// ============================================================================
// Body processing with phi-node approach
// ============================================================================

struct BodyResult {
  std::vector<StmtPtr> stmts;
  std::vector<ExprPtr> yield_values;
};

struct SplitAt {
  std::vector<StmtPtr> pre;
  std::vector<StmtPtr> post;

  SplitAt(const std::vector<StmtPtr>& stmts, size_t i)
      : pre(stmts.begin(), stmts.begin() + static_cast<ptrdiff_t>(i)),
        post(stmts.begin() + static_cast<ptrdiff_t>(i) + 1, stmts.end()) {}
};

static std::shared_ptr<const YieldStmt> FindTrailingYield(const StmtPtr& body) {
  if (auto yield = std::dynamic_pointer_cast<const YieldStmt>(body)) {
    return yield;
  }
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(body)) {
    if (!seq->stmts_.empty()) {
      return FindTrailingYield(seq->stmts_.back());
    }
  }
  return nullptr;
}

static std::vector<StmtPtr> RemoveTrailingYieldToVec(const StmtPtr& body) {
  auto flat = FlattenToVec(body);
  if (!flat.empty()) {
    if (std::dynamic_pointer_cast<const YieldStmt>(flat.back())) {
      flat.pop_back();
    }
  }
  return flat;
}

struct DecomposedBody {
  std::vector<StmtPtr> stmts;
  std::vector<ExprPtr> yield_values;
  bool had_yield;
};

static DecomposedBody DecomposeBody(const StmtPtr& body) {
  auto trailing = FindTrailingYield(body);
  auto stmts = RemoveTrailingYieldToVec(body);
  auto values = trailing ? trailing->value_ : std::vector<ExprPtr>{};
  return {std::move(stmts), std::move(values), trailing != nullptr};
}

static StmtPtr BuildFinalBody(BodyResult result, bool had_yield, const Span& span) {
  if (!result.yield_values.empty() || had_yield) {
    result.stmts.push_back(std::make_shared<YieldStmt>(std::move(result.yield_values), span));
  }
  return MakeSeq(std::move(result.stmts), span);
}

/// Check if a branch ends directly with the given target kind (break or continue).
static bool BranchEndsWith(const StmtPtr& branch, ObjectKind target) {
  auto flat = FlattenToVec(branch);
  return !flat.empty() && flat.back()->GetKind() == target;
}

static std::vector<VarPtr> CreatePhiVars(const std::vector<ExprPtr>& values, int& counter, const Span& span) {
  std::vector<VarPtr> phis;
  phis.reserve(values.size());
  for (const auto& val : values) {
    auto type = val->GetType();
    auto name = auto_name::BuildName(GetExprAutoNamingBase(val), "", "phi", counter++);
    phis.push_back(std::make_shared<Var>(name, type, span));
  }
  return phis;
}

static std::vector<ExprPtr> VarsToExprs(const std::vector<VarPtr>& vars) {
  std::vector<ExprPtr> exprs;
  exprs.reserve(vars.size());
  for (const auto& v : vars) {
    exprs.push_back(v);
  }
  return exprs;
}

/// Collect the statements from the "normal" (non-escape) branch of an IfStmt,
/// followed by the statements after the IfStmt.
static std::vector<StmtPtr> CollectNormalPath(const std::shared_ptr<const IfStmt>& if_stmt,
                                              bool escape_in_then, const std::vector<StmtPtr>& post) {
  std::vector<StmtPtr> normal_stmts;
  if (escape_in_then) {
    if (if_stmt->else_body_.has_value()) {
      auto flat = FlattenToVec(*if_stmt->else_body_);
      normal_stmts.insert(normal_stmts.end(), flat.begin(), flat.end());
    }
  } else {
    auto flat = FlattenToVec(if_stmt->then_body_);
    normal_stmts.insert(normal_stmts.end(), flat.begin(), flat.end());
  }
  normal_stmts.insert(normal_stmts.end(), post.begin(), post.end());
  return normal_stmts;
}

/// Build an IfStmt that routes the escape branch to yield escape_values,
/// and the normal branch to yield normal_result values.
/// escape_prefix_stmts are prepended to the escape branch (e.g., AssignStmt for break flag).
static BodyResult BuildEscapeIfStmt(const std::shared_ptr<const IfStmt>& if_stmt, bool escape_in_then,
                                    const std::vector<StmtPtr>& pre,
                                    const std::vector<StmtPtr>& escape_prefix_stmts,
                                    const std::vector<ExprPtr>& escape_values, BodyResult normal_result,
                                    int& name_counter, const Span& span) {
  auto phi_vars = CreatePhiVars(escape_values, name_counter, span);
  auto phi_exprs = VarsToExprs(phi_vars);

  // Build escape branch: prefix stmts + yield
  std::vector<StmtPtr> escape_parts(escape_prefix_stmts.begin(), escape_prefix_stmts.end());
  escape_parts.push_back(std::make_shared<YieldStmt>(escape_values, if_stmt->span_));
  auto escape_body = MakeSeq(std::move(escape_parts), if_stmt->span_);

  // Build normal branch: stmts + yield
  std::vector<StmtPtr> normal_parts(normal_result.stmts.begin(), normal_result.stmts.end());
  normal_parts.push_back(std::make_shared<YieldStmt>(std::move(normal_result.yield_values), if_stmt->span_));
  auto normal_body = MakeSeq(std::move(normal_parts), if_stmt->span_);

  StmtPtr then_body = escape_in_then ? escape_body : normal_body;
  StmtPtr else_body = escape_in_then ? normal_body : escape_body;

  auto new_if = std::make_shared<IfStmt>(if_stmt->condition_, then_body, std::make_optional(else_body),
                                         phi_vars, if_stmt->span_);

  std::vector<StmtPtr> result(pre.begin(), pre.end());
  result.push_back(new_if);
  return BodyResult{std::move(result), std::move(phi_exprs)};
}

// ============================================================================
// ProcessBodyForContinue
// ============================================================================

static BodyResult ProcessBodyForContinue(const std::vector<StmtPtr>& stmts,
                                         const std::vector<IterArgPtr>& iter_args,
                                         const std::vector<ExprPtr>& original_yield_values, int& name_counter,
                                         const Span& span) {
  BreakContinueScanner scanner;

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Case 1: bare ContinueStmt
    if (std::dynamic_pointer_cast<const ContinueStmt>(stmt)) {
      std::vector<StmtPtr> pre(stmts.begin(), stmts.begin() + static_cast<ptrdiff_t>(i));
      auto continue_values = ResolveYieldAtEscape(original_yield_values, pre, iter_args);
      return BodyResult{std::move(pre), std::move(continue_values)};
    }

    // Case 2: IfStmt containing continue
    auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt);
    if (!if_stmt) continue;

    auto then_scan = scanner.Scan(if_stmt->then_body_);
    bool else_has_continue = false;
    if (if_stmt->else_body_.has_value()) {
      else_has_continue = scanner.Scan(*if_stmt->else_body_).has_continue;
    }

    if (!then_scan.has_continue && !else_has_continue) continue;

    bool then_ends = BranchEndsWith(if_stmt->then_body_, ObjectKind::ContinueStmt);
    bool else_ends =
        if_stmt->else_body_.has_value() && BranchEndsWith(*if_stmt->else_body_, ObjectKind::ContinueStmt);

    SplitAt split(stmts, i);

    // CASE A: continue at end of a branch → use CollectNormalPath
    if (then_ends || else_ends) {
      bool escape_in_then = then_ends;

      auto continue_values = ResolveYieldAtEscape(original_yield_values, split.pre, iter_args);
      auto normal_stmts = CollectNormalPath(if_stmt, escape_in_then, split.post);
      // Resolve yield values considering the full normal-path scope (split.pre +
      // normal_stmts). Variables defined in split.post (now part of normal_stmts)
      // are kept; variables not available at this scope fall back to iter_args.
      // This avoids the old bug where premature resolution at split.pre discarded
      // vars from split.post, producing dead code.
      std::vector<StmtPtr> normal_scope(split.pre.begin(), split.pre.end());
      normal_scope.insert(normal_scope.end(), normal_stmts.begin(), normal_stmts.end());
      auto normal_yield_values = ResolveYieldAtEscape(original_yield_values, normal_scope, iter_args);
      auto normal_result =
          ProcessBodyForContinue(normal_stmts, iter_args, normal_yield_values, name_counter, span);

      if (original_yield_values.empty()) {
        auto empty_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, if_stmt->span_);
        auto filled_body = MakeSeq(std::move(normal_result.stmts), if_stmt->span_);
        StmtPtr then_body = escape_in_then ? static_cast<StmtPtr>(empty_body) : filled_body;
        StmtPtr else_body = escape_in_then ? filled_body : static_cast<StmtPtr>(empty_body);
        auto new_if = std::make_shared<IfStmt>(if_stmt->condition_, then_body, std::make_optional(else_body),
                                               std::vector<VarPtr>{}, if_stmt->span_);
        split.pre.push_back(new_if);
        return BodyResult{std::move(split.pre), {}};
      }

      return BuildEscapeIfStmt(if_stmt, escape_in_then, split.pre, {}, continue_values,
                               std::move(normal_result), name_counter, span);
    }

    // CASE B: continue nested inside a branch (not at end)
    {
      auto process_branch = [&](const StmtPtr& branch, bool has_target) -> BodyResult {
        auto decomposed = DecomposeBody(branch);
        auto& vals = decomposed.had_yield ? decomposed.yield_values : original_yield_values;
        if (!has_target) {
          // No continue in this branch: yield values must only reference vars
          // available at the escape point (split.pre), not vars defined in
          // split.post (which come after the IfStmt).
          auto resolved = ResolveYieldAtEscape(vals, split.pre, iter_args);
          return {std::move(decomposed.stmts), std::move(resolved)};
        }
        return ProcessBodyForContinue(decomposed.stmts, iter_args, vals, name_counter, span);
      };

      auto then_result = process_branch(if_stmt->then_body_, then_scan.has_continue);

      BodyResult else_result;
      if (if_stmt->else_body_.has_value()) {
        else_result = process_branch(*if_stmt->else_body_, else_has_continue);
      } else {
        auto resolved = ResolveYieldAtEscape(original_yield_values, split.pre, iter_args);
        else_result = {{}, std::move(resolved)};
      }

      // Rebuild IfStmt with phi nodes
      if (!then_result.yield_values.empty() || !else_result.yield_values.empty()) {
        auto& ref_values =
            then_result.yield_values.empty() ? else_result.yield_values : then_result.yield_values;
        auto phi_vars = CreatePhiVars(ref_values, name_counter, span);
        auto phi_exprs = VarsToExprs(phi_vars);

        then_result.stmts.push_back(std::make_shared<YieldStmt>(std::move(then_result.yield_values), span));
        auto new_then = MakeSeq(std::move(then_result.stmts), span);

        else_result.stmts.push_back(std::make_shared<YieldStmt>(std::move(else_result.yield_values), span));
        auto new_else = MakeSeq(std::move(else_result.stmts), span);

        auto new_if = std::make_shared<IfStmt>(if_stmt->condition_, new_then, std::make_optional(new_else),
                                               phi_vars, if_stmt->span_);
        split.pre.push_back(new_if);

        if (!split.post.empty()) {
          auto post_result = ProcessBodyForContinue(split.post, iter_args, phi_exprs, name_counter, span);
          split.pre.insert(split.pre.end(), post_result.stmts.begin(), post_result.stmts.end());
          return BodyResult{std::move(split.pre), std::move(post_result.yield_values)};
        }
        return BodyResult{std::move(split.pre), std::move(phi_exprs)};
      }

      // No yield values — just restructure
      auto new_then = MakeSeq(std::move(then_result.stmts), span);
      std::optional<StmtPtr> new_else;
      if (if_stmt->else_body_.has_value()) {
        new_else = MakeSeq(std::move(else_result.stmts), span);
      }
      auto new_if = std::make_shared<IfStmt>(if_stmt->condition_, new_then, new_else, std::vector<VarPtr>{},
                                             if_stmt->span_);
      split.pre.push_back(new_if);
      if (!split.post.empty()) {
        auto post_result =
            ProcessBodyForContinue(split.post, iter_args, original_yield_values, name_counter, span);
        split.pre.insert(split.pre.end(), post_result.stmts.begin(), post_result.stmts.end());
        return BodyResult{std::move(split.pre), std::move(post_result.yield_values)};
      }
      return BodyResult{std::move(split.pre), original_yield_values};
    }
  }

  return BodyResult{std::vector<StmtPtr>(stmts.begin(), stmts.end()), original_yield_values};
}

// ============================================================================
// ProcessBodyForBreak
// ============================================================================

static BodyResult ProcessBodyForBreak(const std::vector<StmtPtr>& stmts, const VarPtr& break_var,
                                      const std::vector<IterArgPtr>& iter_args,
                                      const std::vector<ExprPtr>& original_yield_values, int& name_counter,
                                      const Span& span) {
  BreakContinueScanner scanner;

  // Helper: build the break escape values (current iter_arg values at break point)
  auto build_break_values = [&](const std::vector<StmtPtr>& pre_stmts) -> std::vector<ExprPtr> {
    auto resolved = ResolveYieldAtEscape(original_yield_values, pre_stmts, iter_args);
    // For break: non-Var expressions (e.g. i+1) are "next-iteration" computations
    // that should not execute at break. Fall back to current iter_arg value.
    for (size_t j = 0; j < resolved.size() && j < iter_args.size(); ++j) {
      if (!std::dynamic_pointer_cast<const Var>(resolved[j])) {
        resolved[j] = iter_args[j];
      }
    }
    return resolved;
  };

  // Helper: build break escape prefix stmts (AssignStmt setting break flag)
  auto break_prefix = [&]() -> std::vector<StmtPtr> {
    return {std::make_shared<AssignStmt>(break_var, MakeConstBool(true, span), span)};
  };

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Case 1: bare BreakStmt
    if (std::dynamic_pointer_cast<const BreakStmt>(stmt)) {
      std::vector<StmtPtr> pre(stmts.begin(), stmts.begin() + static_cast<ptrdiff_t>(i));
      pre.push_back(std::make_shared<AssignStmt>(break_var, MakeConstBool(true, span), span));
      auto break_values = build_break_values(pre);
      return BodyResult{std::move(pre), std::move(break_values)};
    }

    // Case 2: IfStmt containing break
    auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt);
    if (!if_stmt) continue;

    auto then_scan = scanner.Scan(if_stmt->then_body_);
    bool else_has_break = false;
    if (if_stmt->else_body_.has_value()) {
      else_has_break = scanner.Scan(*if_stmt->else_body_).has_break;
    }

    if (!then_scan.has_break && !else_has_break) continue;

    bool then_ends = BranchEndsWith(if_stmt->then_body_, ObjectKind::BreakStmt);
    bool else_ends =
        if_stmt->else_body_.has_value() && BranchEndsWith(*if_stmt->else_body_, ObjectKind::BreakStmt);

    // CASE A: break at end of a branch → use CollectNormalPath
    if (then_ends || else_ends) {
      bool escape_in_then = then_ends;
      SplitAt split(stmts, i);

      auto break_values = build_break_values(split.pre);
      auto normal_stmts = CollectNormalPath(if_stmt, escape_in_then, split.post);
      // Resolve yield values considering the full normal-path scope so that
      // variables defined in split.post (now part of normal_stmts) are kept.
      std::vector<StmtPtr> normal_scope(split.pre.begin(), split.pre.end());
      normal_scope.insert(normal_scope.end(), normal_stmts.begin(), normal_stmts.end());
      auto normal_yield_values = ResolveYieldAtEscape(original_yield_values, normal_scope, iter_args);
      auto normal_result =
          ProcessBodyForBreak(normal_stmts, break_var, iter_args, normal_yield_values, name_counter, span);

      return BuildEscapeIfStmt(if_stmt, escape_in_then, split.pre, break_prefix(), break_values,
                               std::move(normal_result), name_counter, span);
    }

    // CASE B: break nested inside a branch (not at end)
    {
      SplitAt split(stmts, i);

      auto process_branch = [&](const StmtPtr& branch, bool has_target) -> BodyResult {
        auto decomposed = DecomposeBody(branch);
        auto& vals = decomposed.had_yield ? decomposed.yield_values : original_yield_values;
        if (!has_target) {
          auto resolved = ResolveYieldAtEscape(vals, split.pre, iter_args);
          return {std::move(decomposed.stmts), std::move(resolved)};
        }
        return ProcessBodyForBreak(decomposed.stmts, break_var, iter_args, vals, name_counter, span);
      };

      auto then_result = process_branch(if_stmt->then_body_, then_scan.has_break);

      BodyResult else_result;
      if (if_stmt->else_body_.has_value()) {
        else_result = process_branch(*if_stmt->else_body_, else_has_break);
      } else {
        auto resolved = ResolveYieldAtEscape(original_yield_values, split.pre, iter_args);
        else_result = {{}, std::move(resolved)};
      }

      // Rebuild IfStmt with phi nodes
      auto& ref_values =
          then_result.yield_values.empty() ? else_result.yield_values : then_result.yield_values;
      auto phi_vars = CreatePhiVars(ref_values, name_counter, span);
      auto phi_exprs = VarsToExprs(phi_vars);

      if (!then_result.yield_values.empty()) {
        then_result.stmts.push_back(std::make_shared<YieldStmt>(std::move(then_result.yield_values), span));
      }
      auto new_then = MakeSeq(std::move(then_result.stmts), span);

      if (!else_result.yield_values.empty()) {
        else_result.stmts.push_back(std::make_shared<YieldStmt>(std::move(else_result.yield_values), span));
      }
      auto new_else = MakeSeq(std::move(else_result.stmts), span);

      auto new_if = std::make_shared<IfStmt>(if_stmt->condition_, new_then, std::make_optional(new_else),
                                             phi_vars, if_stmt->span_);

      split.pre.push_back(new_if);

      // Guard remaining stmts with if (!__break)
      if (!split.post.empty()) {
        auto post_result =
            ProcessBodyForBreak(split.post, break_var, iter_args, phi_exprs, name_counter, span);
        post_result.stmts.push_back(std::make_shared<YieldStmt>(std::move(post_result.yield_values), span));
        auto guarded_body = MakeSeq(std::move(post_result.stmts), span);

        auto guard_else =
            std::make_shared<YieldStmt>(std::vector<ExprPtr>(phi_exprs.begin(), phi_exprs.end()), span);

        auto guard_phi_vars = CreatePhiVars(phi_exprs, name_counter, span);
        auto guard_phi_exprs = VarsToExprs(guard_phi_vars);

        auto guard_if =
            std::make_shared<IfStmt>(MakeNot(break_var, span), guarded_body,
                                     std::make_optional<StmtPtr>(guard_else), guard_phi_vars, span);
        split.pre.push_back(guard_if);
        return BodyResult{std::move(split.pre), std::move(guard_phi_exprs)};
      }

      return BodyResult{std::move(split.pre), std::move(phi_exprs)};
    }
  }

  return BodyResult{std::vector<StmtPtr>(stmts.begin(), stmts.end()), original_yield_values};
}

// ============================================================================
// Main Mutator
// ============================================================================

class CtrlFlowTransformMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto new_body = VisitStmt(op->body_);

    // Only process sequential for loops
    if (op->kind_ != ForKind::Sequential) {
      return RebuildForIfChanged(op, new_body);
    }

    BreakContinueScanner scanner;
    auto scan = scanner.Scan(new_body);

    if (!scan.has_break && !scan.has_continue) {
      return RebuildForIfChanged(op, new_body);
    }

    if (scan.has_break) {
      return LowerForWithBreak(op, new_body, scan.has_continue);
    }

    return LowerForWithContinue(op, new_body);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto new_body = VisitStmt(op->body_);

    BreakContinueScanner scanner;
    auto scan = scanner.Scan(new_body);

    if (!scan.has_break && !scan.has_continue) {
      if (new_body.get() != op->body_.get()) {
        auto result = MutableCopy(op);
        result->body_ = new_body;
        return result;
      }
      return op;
    }

    if (scan.has_break) {
      return LowerWhileWithBreak(op, new_body, scan.has_continue);
    }

    return LowerWhileWithContinue(op, new_body);
  }

 private:
  int name_counter_ = 0;

  std::string FreshInternalName(const std::string& base_name, const std::string& role) {
    return auto_name::BuildName(base_name, "", role, name_counter_++);
  }

  static StmtPtr RebuildForIfChanged(const ForStmtPtr& op, const StmtPtr& new_body) {
    if (new_body.get() == op->body_.get()) {
      return op;
    }
    auto result = MutableCopy(op);
    result->body_ = new_body;
    return result;
  }

  BodyResult ProcessBodyForBreakAndContinue(const DecomposedBody& decomposed,
                                            const std::vector<IterArgPtr>& iter_args, const VarPtr& break_var,
                                            bool also_has_continue, const Span& span) {
    if (also_has_continue) {
      auto continue_result =
          ProcessBodyForContinue(decomposed.stmts, iter_args, decomposed.yield_values, name_counter_, span);
      return ProcessBodyForBreak(continue_result.stmts, break_var, iter_args, continue_result.yield_values,
                                 name_counter_, span);
    }
    return ProcessBodyForBreak(decomposed.stmts, break_var, iter_args, decomposed.yield_values, name_counter_,
                               span);
  }

  // --------------------------------------------------------------------------
  // ForStmt with only continue
  // --------------------------------------------------------------------------
  StmtPtr LowerForWithContinue(const ForStmtPtr& op, const StmtPtr& body) {
    auto decomposed = DecomposeBody(body);
    auto result = ProcessBodyForContinue(decomposed.stmts, op->iter_args_, decomposed.yield_values,
                                         name_counter_, op->span_);
    auto final_body = BuildFinalBody(std::move(result), decomposed.had_yield, op->span_);

    auto new_for = MutableCopy(op);
    new_for->body_ = final_body;
    return new_for;
  }

  // --------------------------------------------------------------------------
  // ForStmt with break (and possibly continue)
  // --------------------------------------------------------------------------
  StmtPtr LowerForWithBreak(const ForStmtPtr& op, const StmtPtr& body, bool also_has_continue) {
    Span span = op->span_;

    auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);
    auto break_var = std::make_shared<Var>(FreshInternalName("break", "tmp"), bool_type, span);

    auto loop_var_type = op->loop_var_->GetType();
    auto loop_var = std::make_shared<Var>(
        FreshInternalName(GetAutoNamingBase(op->loop_var_->name_hint_), "idx"), loop_var_type, span);

    VarSubstituter sub(op->loop_var_->UniqueId(), loop_var);
    auto substituted_body = sub.VisitStmt(body);

    // Choose comparison based on step sign: Lt for positive step, Gt for negative step.
    // Non-const steps (rare in practice) default to Lt (positive direction).
    auto const_step = std::dynamic_pointer_cast<const ConstInt>(op->step_);
    ExprPtr loop_bound_cond = (const_step && const_step->value_ < 0) ? MakeGt(loop_var, op->stop_, span)
                                                                     : MakeLt(loop_var, op->stop_, span);
    ExprPtr while_cond = MakeAndExpr(loop_bound_cond, MakeNot(break_var, span), span);

    auto decomposed = DecomposeBody(substituted_body);
    auto processed =
        ProcessBodyForBreakAndContinue(decomposed, op->iter_args_, break_var, also_has_continue, span);

    // The YieldStmt must terminate the loop body (SSA-form invariant: any
    // For/While/If with non-empty return_vars_ ends with a YieldStmt). The
    // counter-advance must happen BEFORE the yield. To keep the yield reading
    // pre-increment values regardless of whether yielded expressions reference
    // the loop counter (e.g. `pl.yield_(acc + i)` where `i` is the loop var),
    // we materialize any non-Var yielded expression into a temp BEFORE the
    // increment, then yield the temp. Already-bound Var/IterArg references
    // need no materialization — their value is fixed at the binding site.
    std::vector<ExprPtr> trailing_yield_values;
    if (!processed.yield_values.empty()) {
      trailing_yield_values.reserve(processed.yield_values.size());
      for (auto& yield_value : processed.yield_values) {
        // AsVarLike (not As<Var>) so IterArg — a Var subclass — also
        // short-circuits: its value is fixed at the binding site.
        if (AsVarLike(yield_value)) {
          trailing_yield_values.push_back(yield_value);
          continue;
        }
        auto yield_tmp =
            std::make_shared<Var>(FreshInternalName("yield", "tmp"), yield_value->GetType(), span);
        processed.stmts.push_back(std::make_shared<AssignStmt>(yield_tmp, yield_value, span));
        trailing_yield_values.push_back(yield_tmp);
      }
    }

    auto iter_adv = std::make_shared<AssignStmt>(loop_var, MakeAdd(loop_var, op->step_, span), span);
    auto guarded_adv = std::make_shared<IfStmt>(MakeNot(break_var, span), iter_adv, std::nullopt,
                                                std::vector<VarPtr>{}, span);
    processed.stmts.push_back(guarded_adv);

    if (!trailing_yield_values.empty() || decomposed.had_yield) {
      processed.stmts.push_back(std::make_shared<YieldStmt>(std::move(trailing_yield_values), span));
    }

    auto final_body = MakeSeq(std::move(processed.stmts), span);

    std::vector<StmtPtr> init_stmts;
    init_stmts.push_back(std::make_shared<AssignStmt>(loop_var, op->start_, span));
    init_stmts.push_back(std::make_shared<AssignStmt>(break_var, MakeConstBool(false, span), span));
    init_stmts.push_back(
        std::make_shared<WhileStmt>(while_cond, op->iter_args_, final_body, op->return_vars_, span));

    return std::make_shared<SeqStmts>(init_stmts, span);
  }

  // --------------------------------------------------------------------------
  // WhileStmt with only continue
  // --------------------------------------------------------------------------
  StmtPtr LowerWhileWithContinue(const WhileStmtPtr& op, const StmtPtr& body) {
    auto decomposed = DecomposeBody(body);
    auto result = ProcessBodyForContinue(decomposed.stmts, op->iter_args_, decomposed.yield_values,
                                         name_counter_, op->span_);
    auto final_body = BuildFinalBody(std::move(result), decomposed.had_yield, op->span_);

    auto new_while = MutableCopy(op);
    new_while->body_ = final_body;
    return new_while;
  }

  // --------------------------------------------------------------------------
  // WhileStmt with break (and possibly continue)
  // --------------------------------------------------------------------------
  StmtPtr LowerWhileWithBreak(const WhileStmtPtr& op, const StmtPtr& body, bool also_has_continue) {
    Span span = op->span_;

    auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);
    auto break_var = std::make_shared<Var>(FreshInternalName("break", "tmp"), bool_type, span);

    ExprPtr new_condition = MakeAndExpr(op->condition_, MakeNot(break_var, span), span);

    auto decomposed = DecomposeBody(body);
    auto processed =
        ProcessBodyForBreakAndContinue(decomposed, op->iter_args_, break_var, also_has_continue, span);
    auto final_body = BuildFinalBody(std::move(processed), decomposed.had_yield, span);

    std::vector<StmtPtr> init_stmts;
    init_stmts.push_back(std::make_shared<AssignStmt>(break_var, MakeConstBool(false, span), span));
    init_stmts.push_back(
        std::make_shared<WhileStmt>(new_condition, op->iter_args_, final_body, op->return_vars_, span));
    return std::make_shared<SeqStmts>(init_stmts, span);
  }
};

// ============================================================================
// Pass entry point
// ============================================================================

FunctionPtr TransformCtrlFlow(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "CtrlFlowTransform cannot run on null function";

  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

  CtrlFlowTransformMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);

  if (new_body.get() == func->body_.get()) {
    return func;
  }

  auto result = MutableCopy(func);
  result->body_ = new_body;
  return result;
}

}  // namespace

namespace pass {
Pass CtrlFlowTransform() {
  return CreateFunctionPass(TransformCtrlFlow, "CtrlFlowTransform", kCtrlFlowTransformProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
