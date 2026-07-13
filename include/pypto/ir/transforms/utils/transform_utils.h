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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/attrs.h"

namespace pypto::ir::transform_utils {

/// Substitute variable references in an expression by pointer identity.
///
/// Recursively traverses Call, MakeTuple, BinaryExpr, UnaryExpr, and
/// TupleGetItemExpr to replace Var/IterArg references whose raw pointer
/// appears in @p var_map.
ExprPtr Substitute(const ExprPtr& expr, const std::unordered_map<const Var*, VarPtr>& var_map);
ExprPtr Substitute(const ExprPtr& expr, const std::unordered_map<const Var*, ExprPtr>& var_map);

/// Substitute variable references in a statement subtree by pointer identity.
///
/// Walks the IR subtree via IRMutator and replaces each Var whose raw pointer
/// appears in @p var_map with the mapped replacement.
StmtPtr Substitute(const StmtPtr& body, const std::unordered_map<const Var*, VarPtr>& var_map);
StmtPtr Substitute(const StmtPtr& body, const std::unordered_map<const Var*, ExprPtr>& var_map);

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

/// Extract the Call value of a leaf statement, or nullptr if none.
///
/// Covers the two forms a Call appears in: AssignStmt.value and EvalStmt.expr.
/// Replaces the ~10x-repeated cast ladder
///   CallPtr call;
///   if (auto a = dynamic_pointer_cast<const AssignStmt>(stmt)) call = ...->value_;
///   else if (auto e = dynamic_pointer_cast<const EvalStmt>(stmt)) call = ...->expr_;
inline CallPtr GetCallFromStmt(const StmtPtr& stmt) {
  if (auto assign = As<AssignStmt>(stmt)) return As<Call>(assign->value_);
  if (auto eval = As<EvalStmt>(stmt)) return As<Call>(eval->expr_);
  return nullptr;
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

// ============================================================================
// Op classification
// ============================================================================

/// Returns true if op_name is a compute tensor op (not a host-side memory/transfer/metadata op).
///
/// Host-side ops are memory allocation/transfer (create, read, write, slice, assemble, dim)
/// and metadata-only transforms (reshape, transpose at tensor level).
inline bool IsComputeTensorOp(const OpPtr& op) {
  if (!op || op->name_.compare(0, 7, "tensor.") != 0) return false;
  return !(IsOp(op, "tensor.create") || IsOp(op, "tensor.read") || IsOp(op, "tensor.write") ||
           IsOp(op, "tensor.slice") || IsOp(op, "tensor.assemble") || IsOp(op, "tensor.dim") ||
           IsOp(op, "tensor.reshape") || IsOp(op, "tensor.transpose") || IsOp(op, "tensor.view"));
}

// ============================================================================
// Call-like views and constant evaluation
// ============================================================================

/// Returns a Call-shaped view of @p expr when it is a Call or a Submit, else
/// null. ``Submit`` (a task launch) is the canonical IR form after
/// ``DeriveCallDirections``; analyses that do not care about task-launch
/// semantics funnel it through ``SubmitToCallView`` so the Call-based logic
/// applies unchanged. Maps keyed on node identity must use the binding Var,
/// never this transient view.
inline CallPtr AsCallOrSubmitView(const ExprPtr& expr) {
  if (auto call = As<Call>(expr)) return call;
  if (auto submit = As<Submit>(expr)) return SubmitToCallView(submit);
  return nullptr;
}

/// Constant-evaluate @p expr if it is a ``ConstInt``; ``nullopt`` otherwise.
inline std::optional<int64_t> EvalConstInt(const ExprPtr& expr) {
  if (auto ci = As<ConstInt>(expr)) return ci->value_;
  return std::nullopt;
}

/// Return the const trip count of @p for_stmt when start/stop/step are all
/// ``ConstInt`` and step is positive; 0 otherwise.
inline int64_t EvalConstTripCount(const ForStmtPtr& for_stmt) {
  auto start = EvalConstInt(for_stmt->start_);
  auto stop = EvalConstInt(for_stmt->stop_);
  auto step = EvalConstInt(for_stmt->step_);
  if (!start || !stop || !step || *step <= 0) return 0;
  int64_t trip = (*stop - *start + *step - 1) / *step;
  return trip > 0 ? trip : 0;
}

/// Peek through a leading compiler-inserted ``RuntimeScopeStmt`` so structural
/// analyses reach the original statements.
///
/// ``MaterializeRuntimeScopes`` wraps the orchestration function body and each
/// ForStmt / IfStmt branch body in an AUTO ``RuntimeScopeStmt`` so codegen emits
/// ``PTO2_SCOPE()`` 1:1 from the IR. ``GetLastYieldStmt`` / ``FlattenToStmts``
/// do not descend through a scope node, so callers unwrap first. User
/// ``pl.manual_scope`` scopes stay opaque — they were never auto-wrapped —
/// except for compiler-synthesised manual scopes, which carry
/// ``kAttrCompilerAutoManualScopeCandidate``.
///
/// A user-written ``with pl.auto_scope():`` body may arrive as a single-statement
/// ``SeqStmts`` wrapper (before ``NormalizeStmtStructure`` collapses it); peek
/// through it (and any nested AUTO scopes) too.
inline StmtPtr UnwrapAutoScope(const StmtPtr& stmt) {
  if (auto scope = As<RuntimeScopeStmt>(stmt);
      scope && (!scope->manual_ || scope->GetAttr<bool>(kAttrCompilerAutoManualScopeCandidate, false))) {
    return UnwrapAutoScope(scope->body_);
  }
  if (auto seq = As<SeqStmts>(stmt); seq && seq->stmts_.size() == 1) {
    return UnwrapAutoScope(seq->stmts_[0]);
  }
  return stmt;
}

// ============================================================================
// iter_arg carry classification (attrs stamped by ``ClassifyIterArgCarry``)
// ============================================================================

/// True when iter_arg @p idx needs a materialised mutable carry variable.
/// False (the default when the attr is absent) means the iter_arg is a trivial
/// alias of its init value.
inline bool IterArgIsRebind(const ForStmtPtr& for_stmt, size_t idx) {
  return for_stmt->GetAttr<bool>(IterArgRebindAttrKey(idx), false);
}

/// TaskId manual-scope array-carry extent for iter_arg @p idx; 0 means the
/// scalar / tensor / ArrayType carry path.
inline int64_t IterArgArraySize(const ForStmtPtr& for_stmt, size_t idx) {
  return static_cast<int64_t>(for_stmt->GetAttr<int>(IterArgArraySizeAttrKey(idx), 0));
}

}  // namespace pypto::ir::transform_utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_TRANSFORM_UTILS_H_
