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

#include "pypto/ir/transforms/utils/transform_utils.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"

namespace pypto::ir::transform_utils {

namespace {

/// Seeds IRMutator::var_remap_ from the user-supplied substitution map; the
/// base mutator's type-aware visitors do the rest.
template <typename ValueT>
class SubstituteMutator : public IRMutator {
 public:
  explicit SubstituteMutator(const std::unordered_map<const Var*, ValueT>& var_map) {
    for (const auto& [k, v] : var_map) {
      var_remap_[k] = v;
    }
  }
};

class CallLikeFinder : public IRVisitor {
 public:
  bool found = false;

 private:
  // Finding the outer call is sufficient; its arguments cannot change the answer.
  void VisitExpr_(const CallPtr&) override { found = true; }
  void VisitExpr_(const SubmitPtr&) override { found = true; }
};

}  // namespace

bool ExprContainsCallLike(const ExprPtr& expr) {
  if (!expr) return false;
  CallLikeFinder finder;
  finder.VisitExpr(expr);
  return finder.found;
}

StmtPtr RewriteLastYieldStmt(const StmtPtr& body,
                             const std::function<StmtPtr(const YieldStmtPtr&)>& rewrite) {
  if (auto yield = As<YieldStmt>(body)) return rewrite(yield);
  if (auto seq = As<SeqStmts>(body)) {
    if (seq->stmts_.empty()) return body;
    auto tail = RewriteLastYieldStmt(seq->stmts_.back(), rewrite);
    if (tail == seq->stmts_.back()) return body;
    auto stmts = seq->stmts_;
    if (tail) {
      stmts.back() = std::move(tail);
    } else {
      stmts.pop_back();
    }
    if (stmts.empty()) return nullptr;
    if (stmts.size() == 1) return stmts.front();
    // Keep nested sequences nested: flattening the growing suffix at every
    // ancestor would make a trailing sequence chain quadratic. Normalized
    // input stays normalized; the dedicated normalization pass handles other IR.
    auto result = MutableCopy(seq);
    result->stmts_ = std::move(stmts);
    return result;
  }
  auto rewrite_scope = [&](const auto& scope) -> StmtPtr {
    auto inner = RewriteLastYieldStmt(scope->body_, rewrite);
    if (inner == scope->body_) return scope;
    auto result = MutableCopy(scope);
    // A scope has scheduling semantics even when its only yield disappears.
    result->body_ = inner ? inner : std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, scope->body_->span_);
    return result;
  };
  if (auto scope = As<RuntimeScopeStmt>(body)) return rewrite_scope(scope);
  if (auto scope = As<SplitAivScopeStmt>(body)) return rewrite_scope(scope);
  return body;
}

// Var* → VarPtr
ExprPtr Substitute(const ExprPtr& expr, const std::unordered_map<const Var*, VarPtr>& var_map) {
  SubstituteMutator<VarPtr> mutator(var_map);
  return mutator.VisitExpr(expr);
}
StmtPtr Substitute(const StmtPtr& body, const std::unordered_map<const Var*, VarPtr>& var_map) {
  SubstituteMutator<VarPtr> mutator(var_map);
  return mutator.VisitStmt(body);
}

// Var* → ExprPtr
ExprPtr Substitute(const ExprPtr& expr, const std::unordered_map<const Var*, ExprPtr>& var_map) {
  SubstituteMutator<ExprPtr> mutator(var_map);
  return mutator.VisitExpr(expr);
}
StmtPtr Substitute(const StmtPtr& body, const std::unordered_map<const Var*, ExprPtr>& var_map) {
  SubstituteMutator<ExprPtr> mutator(var_map);
  return mutator.VisitStmt(body);
}

// ---------------------------------------------------------------------------
// CollectDefVars
// ---------------------------------------------------------------------------

void CollectDefVars(const StmtPtr& stmt, std::vector<VarPtr>& result) {
  if (!stmt) return;

  auto kind = stmt->GetKind();
  switch (kind) {
    case ObjectKind::AssignStmt: {
      auto assign = std::static_pointer_cast<const AssignStmt>(stmt);
      result.push_back(assign->var_);
      break;
    }
    case ObjectKind::SeqStmts: {
      auto seq = std::static_pointer_cast<const SeqStmts>(stmt);
      for (const auto& s : seq->stmts_) {
        CollectDefVars(s, result);
      }
      break;
    }
    case ObjectKind::ForStmt: {
      auto for_stmt = std::static_pointer_cast<const ForStmt>(stmt);
      CollectDefVars(for_stmt->body_, result);
      break;
    }
    case ObjectKind::WhileStmt: {
      auto while_stmt = std::static_pointer_cast<const WhileStmt>(stmt);
      CollectDefVars(while_stmt->body_, result);
      break;
    }
    case ObjectKind::IfStmt: {
      auto if_stmt = std::static_pointer_cast<const IfStmt>(stmt);
      CollectDefVars(if_stmt->then_body_, result);
      if (if_stmt->else_body_.has_value()) {
        CollectDefVars(*if_stmt->else_body_, result);
      }
      break;
    }
    case ObjectKind::InCoreScopeStmt:
    case ObjectKind::ClusterScopeStmt:
    case ObjectKind::HierarchyScopeStmt:
    case ObjectKind::SpmdScopeStmt:
    case ObjectKind::SplitAivScopeStmt:
    case ObjectKind::RegScopeStmt:
    case ObjectKind::GraphScopeStmt: {
      auto scope = std::static_pointer_cast<const ScopeStmt>(stmt);
      CollectDefVars(scope->body_, result);
      break;
    }
    default:
      // YieldStmt, ReturnStmt, EvalStmt, BreakStmt, ContinueStmt — no DEFs
      break;
  }
}

}  // namespace pypto::ir::transform_utils
