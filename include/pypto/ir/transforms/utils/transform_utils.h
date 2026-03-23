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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_TRANSFORM_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_TRANSFORM_UTILS_H_

#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir::transform_utils {

/// Substitute variables in an expression using a pointer-identity map.
///
/// Recursively traverses Call, MakeTuple, BinaryExpr, UnaryExpr, and
/// TupleGetItemExpr to replace Var/IterArg references whose raw pointer
/// appears in @p var_map.
///
/// @param expr    Expression to transform.
/// @param var_map Pointer-based substitution map (original Var* -> replacement VarPtr).
/// @return Transformed expression.
ExprPtr SubstituteExpr(const ExprPtr& expr, const std::unordered_map<const Var*, VarPtr>& var_map);

/// Substitute variable references in a statement subtree by pointer identity.
///
/// Walks the IR subtree via IRMutator and replaces each Var whose raw pointer
/// appears in @p var_map with the mapped VarPtr.  IterArg nodes are handled by
/// the base IRMutator (preserving their type for ForStmt/WhileStmt slots).
///
/// @param body    Statement subtree to transform.
/// @param var_map Pointer-based substitution map (original Var* -> replacement VarPtr).
/// @return Transformed statement subtree.
StmtPtr SubstituteStmt(const StmtPtr& body, const std::unordered_map<const Var*, VarPtr>& var_map);

/// Find the first YieldStmt inside a statement body (searches through SeqStmts).
inline YieldStmtPtr FindYieldStmt(const StmtPtr& body) {
  if (auto yield = As<YieldStmt>(body)) return yield;
  if (auto seq = As<SeqStmts>(body)) {
    for (const auto& child : seq->stmts_) {
      auto result = FindYieldStmt(child);
      if (result) return result;
    }
  }
  return nullptr;
}

/// Find the trailing YieldStmt in a statement body (checks only the last element).
///
/// Unlike FindYieldStmt which searches for the first yield anywhere in the tree,
/// this function only looks at the back of SeqStmts containers, finding
/// the yield that acts as the loop-exit value producer.
inline YieldStmtPtr GetLastYieldStmt(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) {
    if (seq->stmts_.empty()) return nullptr;
    return GetLastYieldStmt(seq->stmts_.back());
  }
  return As<YieldStmt>(body);
}

/// Unwrap a StmtPtr into a flat vector of statements.
///
/// If the statement is a SeqStmts, returns its children;
/// otherwise returns a single-element vector.
inline std::vector<StmtPtr> FlattenToStmts(const StmtPtr& stmt) {
  if (auto seq = As<SeqStmts>(stmt)) {
    return seq->stmts_;
  }
  return {stmt};
}

/// Collect all AssignStmt var_ (DEF sites) from a statement tree.
///
/// When the body is visited multiple times (inner + remainder), the same
/// VarPtr would appear as a DEF in both, violating SSA. This function
/// collects all such DEF vars so we can create fresh copies before the
/// second visit.
void CollectDefVars(const StmtPtr& stmt, std::vector<VarPtr>& result);

/// Convenience overload: collect DEF vars and return them as a new vector.
inline std::vector<VarPtr> CollectDefVars(const StmtPtr& stmt) {
  std::vector<VarPtr> result;
  CollectDefVars(stmt, result);
  return result;
}

}  // namespace pypto::ir::transform_utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_TRANSFORM_UTILS_H_
