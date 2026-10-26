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

#ifndef SRC_IR_TRANSFORMS_TENSOR_TO_TILE_TYPE_PROPAGATION_H_
#define SRC_IR_TRANSFORMS_TENSOR_TO_TILE_TYPE_PROPAGATION_H_

#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"

namespace pypto {
namespace ir {
namespace tensor_to_tile {

// ============================================================================
// TypePropagatingMutator: base class that extends IRMutator with type
// propagation through control flow (IterArg types, ForStmt/WhileStmt
// return_vars, IfStmt return_vars from yield types).
//
// Subclasses override VisitStmt_(AssignStmtPtr) for domain-specific logic
// (op conversion, call-site updates, etc.) and call HandlePassThroughAssign
// for non-converted assignments.
// ============================================================================

class TypePropagatingMutator : public IRMutator {
 protected:
  /// Override IterArg to propagate type from initValue_ when it changes.
  /// The base IRMutator preserves the original type; we want the new type
  /// so that downstream references see the correct (e.g. TileType) type.
  ExprPtr VisitExpr_(const IterArgPtr& op) override;

  /// Override ForStmt to update return_vars types to match iter_arg types.
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;

  /// Override WhileStmt to update return_vars types to match iter_arg types.
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override;

  /// Override IfStmt to (a) isolate var_remap_ per branch and
  /// (b) update return_vars types from yield types.
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;

  /// Handle a non-converted assignment: propagate type change if value type changed.
  StmtPtr HandlePassThroughAssign(const AssignStmtPtr& op, const ExprPtr& new_value);

  /// Keep a Var shared_ptr alive for the lifetime of this mutator.
  ///
  /// ``var_remap_`` is keyed by raw ``const Expr*`` pointers (inherited from
  /// IRMutator). Converters create temporary Vars (e.g. ``paged_gather`` builds
  /// per-row scalars like ``pg_idx`` in its loop body); when a converter's
  /// ``AssignStmt`` is replaced during conversion, the old Var is freed and the
  /// allocator can hand its address to a *later* Var. A stale ``var_remap_``
  /// entry keyed on the freed address would then mis-resolve the new Var,
  /// silently rewriting an unrelated value (observed: a matmul result resolving
  /// to a freed ``pg_idx`` scalar). Retaining every mapped-from Var prevents the
  /// address reuse that triggers the collision.
  void RetainVar(const ExprPtr& v);

 private:
  /// Shared logic for ForStmt/WhileStmt: update return_vars types to match iter_arg types.
  template <typename ReconstructFn>
  StmtPtr UpdateLoopReturnVars(const std::vector<IterArgPtr>& new_iter_args,
                               const std::vector<VarPtr>& new_return_vars,
                               const std::vector<VarPtr>& orig_return_vars, ReconstructFn reconstruct,
                               const StmtPtr& original);

  /// Vars kept alive for the pass lifetime — see RetainVar.
  std::vector<ExprPtr> retained_vars_;
};

}  // namespace tensor_to_tile
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_TENSOR_TO_TILE_TYPE_PROPAGATION_H_
