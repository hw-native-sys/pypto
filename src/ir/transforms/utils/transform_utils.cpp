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

#include <memory>
#include <unordered_map>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"

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

}  // namespace

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
    case ObjectKind::AutoInCoreScopeStmt:
    case ObjectKind::ClusterScopeStmt:
    case ObjectKind::HierarchyScopeStmt:
    case ObjectKind::SpmdScopeStmt: {
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
