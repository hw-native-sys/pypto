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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_DEAD_CODE_ELIMINATION_H_
#define PYPTO_IR_TRANSFORMS_UTILS_DEAD_CODE_ELIMINATION_H_

#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace dce {

/// Extract the Op name from an AssignStmt or EvalStmt containing a Call.
/// Returns empty string if the statement doesn't match this pattern.
std::string GetStmtOpName(const StmtPtr& stmt);

/// Check if a statement is a side-effect op that must be preserved.
/// The predicate is customizable; this is a default implementation for
/// cross-core and tile ops.
bool IsSideEffectOp(const StmtPtr& stmt);

/// Collect all AssignStmts recursively from nested statements.
void CollectAllAssignStmts(const std::vector<StmtPtr>& stmts,
                           std::vector<std::shared_ptr<const AssignStmt>>& assigns);

/// Eliminate dead code from a statement list.
/// Dead statements are those whose defined variable is not transitively
/// used by any return, yield, or side-effect statement.
std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts);

/// Conservative scalar-only DCE.
///
/// Removes every `AssignStmt` that satisfies ALL of:
///   - LHS Var has `ScalarType`
///   - RHS expression contains no `Call` anywhere (Call may have side effects)
///   - LHS Var is not transitively used by any preserved statement
///
/// Preserves every other statement kind: AssignStmts with non-scalar LHS,
/// Call-containing AssignStmts, EvalStmt, ReturnStmt, YieldStmt, and the
/// control-flow nodes themselves (ForStmt/IfStmt/WhileStmt/ScopeStmt). The
/// bodies of those control-flow nodes are filtered recursively, so nested
/// scalar assignments remain eligible for removal.
///
/// Like `EliminateDeadCode`, iterates to a fixed point so chains of scalar
/// bindings (`a = 5; b = a + 1; c = b + 1` with `c` unused) collapse fully.
std::vector<StmtPtr> EliminateDeadScalarAssignments(const std::vector<StmtPtr>& stmts);

/// Drop the yielded slots nobody reads, and the matching slot from every
/// trailing `YieldStmt` that feeds them. Two node kinds carry such slots:
///
///   - `IfStmt`: slot `i` is dead when `return_vars_[i]` has no use in the
///     surrounding statement list.
///   - `ForStmt` / `WhileStmt`: a loop-carried slot has *two* consumers —
///     `iter_args_[i]` inside the body and `return_vars_[i]` after the loop —
///     so slot `i` is dead only when neither is used. A body that assigns the
///     carried name before reading it (the same Python local reused across two
///     scopes) leaves exactly this shape: SSA seeds the loop with the earlier
///     value, the body overwrites it on every trip, and nothing reads either
///     end. Dropping the slot removes the spurious live-out — which for a
///     device scope is what would otherwise force a Scalar into the outlined
///     kernel's return set (see `PlanScalarOutputHoist`).
///
/// Recurses into nested control-flow and scope bodies, iterating to a fixed
/// point so cascading deaths (outer slot drop → inner slot becomes dead) are
/// fully collapsed.
///
/// A slot whose `IterArg::initValue_` or yielded value contains a `Call` /
/// `Submit` is kept whatever its liveness: dropping the slot deletes that
/// expression, and before `FlattenCallExpr` a yielded value can still BE a
/// call — a task launch or any other effectful op. This mirrors
/// `EliminateDeadScalarAssignments`, which never drops a call-backed
/// assignment, for the same reason: the IR carries no purity annotations.
///
/// A `Scalar[TASK_ID]` / `Array[TASK_ID]` carry is exempt. It is a scheduling
/// channel, not data: `AutoDeriveTaskDependencies` and `ExpandManualPhaseFence`
/// read the *shape* of such a carry — a task id produced in a loop and carried
/// out is how the compiler learns to fan every iteration's handle in to a later
/// consumer — so no ordinary Var use marks it live, and dropping it silently
/// deletes the dependency edges those passes would have derived.
///
/// The helper is type-agnostic: Scalar, Tile, and Tensor slots are all
/// candidates when no direct Var* use exists. Side effects in the bodies are
/// preserved by `EliminateDeadScalarAssignments` (Call/Submit RHS
/// conservatively kept) when this helper is composed with it; only the yield
/// slot and the header entries are removed.
///
/// Liveness is purely identity-based: a slot is dead iff none of its Vars is
/// collected by `VarDefUseCollector` over the entire statement list. That
/// collector already handles ScopeStmt attrs (`manual_dep_edges` /
/// `task_id_var` / `arg_direction_overrides_vars`), `Submit::deps_`, and every
/// `YieldStmt::value_` slot — so all known channels through which a Var can be
/// referenced are covered.
std::vector<StmtPtr> EliminateDeadYieldSlots(const std::vector<StmtPtr>& stmts);

}  // namespace dce
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_DEAD_CODE_ELIMINATION_H_
