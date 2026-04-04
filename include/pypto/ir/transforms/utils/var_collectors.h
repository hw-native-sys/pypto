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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_VAR_COLLECTORS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_VAR_COLLECTORS_H_

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace var_collectors {

// ============================================================================
// Visitor-based collector
// ============================================================================

/// Collects variable definitions and uses from an IR subtree in a single pass.
///
/// Fields:
///   var_defs         — variables defined by statements (unordered).
///   var_uses         — variables referenced in expressions (unordered).
///   var_defs_ordered — same as var_defs but in DFS pre-order.
///   var_assign_defs  — subset of var_defs: only AssignStmt::var_.
///
/// Definition order within each statement follows CollectVarDefsInOrder
/// convention: loop_var → return_vars → iter_args → recurse body.
class VarDefUseCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> var_defs;
  std::unordered_set<const Var*> var_uses;
  std::vector<const Var*> var_defs_ordered;
  std::unordered_set<const Var*> var_assign_defs;

  /// Return var_defs ∪ var_uses — all variables referenced in the subtree.
  std::unordered_set<const Var*> GetAllVarRefs() const {
    auto result = var_defs;
    result.insert(var_uses.begin(), var_uses.end());
    return result;
  }

 protected:
  void VisitExpr_(const VarPtr& op) override { var_uses.insert(op.get()); }
  void VisitExpr_(const IterArgPtr& op) override { var_uses.insert(op.get()); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    var_defs.insert(op->var_.get());
    var_defs_ordered.push_back(op->var_.get());
    var_assign_defs.insert(op->var_.get());
    VisitExpr(op->value_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    // Defs: loop_var → return_vars → iter_args (matches DFS pre-order convention)
    var_defs.insert(op->loop_var_.get());
    var_defs_ordered.push_back(op->loop_var_.get());
    for (const auto& rv : op->return_vars_) {
      var_defs.insert(rv.get());
      var_defs_ordered.push_back(rv.get());
    }
    for (const auto& ia : op->iter_args_) {
      var_defs.insert(ia.get());
      var_defs_ordered.push_back(ia.get());
      if (ia->initValue_) VisitExpr(ia->initValue_);
    }
    // Uses: loop bounds and body
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    if (op->chunk_size_.has_value() && *op->chunk_size_) {
      VisitExpr(*op->chunk_size_);
    }
    VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    // Defs: return_vars → iter_args
    for (const auto& rv : op->return_vars_) {
      var_defs.insert(rv.get());
      var_defs_ordered.push_back(rv.get());
    }
    for (const auto& ia : op->iter_args_) {
      var_defs.insert(ia.get());
      var_defs_ordered.push_back(ia.get());
      if (ia->initValue_) VisitExpr(ia->initValue_);
    }
    VisitExpr(op->condition_);
    VisitStmt(op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& rv : op->return_vars_) {
      var_defs.insert(rv.get());
      var_defs_ordered.push_back(rv.get());
    }
    VisitExpr(op->condition_);
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(*op->else_body_);
  }
};

// ============================================================================
// Free-function collectors
// ============================================================================

/// Collect variables defined by a single statement (non-recursive).
///
/// Returns the set of variables that become "visible" after executing the
/// statement: AssignStmt::var_ and control-flow return_vars_.
/// Does NOT include ForStmt::loop_var_ or iter_args_ (internal to loop body).
/// Does NOT recurse into nested statements.
inline std::unordered_set<const Var*> CollectStmtDefinedVars(const StmtPtr& stmt) {
  std::unordered_set<const Var*> defs;
  if (auto assign = As<AssignStmt>(stmt)) {
    defs.insert(assign->var_.get());
  } else if (auto for_stmt = As<ForStmt>(stmt)) {
    for (const auto& ret : for_stmt->return_vars_) {
      defs.insert(ret.get());
    }
  } else if (auto if_stmt = As<IfStmt>(stmt)) {
    for (const auto& ret : if_stmt->return_vars_) {
      defs.insert(ret.get());
    }
  } else if (auto while_stmt = As<WhileStmt>(stmt)) {
    for (const auto& ret : while_stmt->return_vars_) {
      defs.insert(ret.get());
    }
  }
  return defs;
}

// ============================================================================
// Type expression visitors
// ============================================================================

/// Visit all expression fields embedded in a type using the given visitor.
///
/// Covers: TensorType::shape_, tensor_view_.{valid_shape, stride};
///         TileType::shape_, tile_view_.{valid_shape, stride, start_offset};
///         TupleType elements (recursively).
inline void VisitTypeExprFields(IRVisitor& visitor, const TypePtr& type) {
  if (!type) return;

  auto visit_exprs = [&visitor](const std::vector<ExprPtr>& exprs) {
    for (const auto& e : exprs) {
      if (e) visitor.VisitExpr(e);
    }
  };

  if (auto tensor_type = As<TensorType>(type)) {
    visit_exprs(tensor_type->shape_);
    if (tensor_type->tensor_view_.has_value()) {
      const auto& tv = tensor_type->tensor_view_.value();
      visit_exprs(tv.valid_shape);
      visit_exprs(tv.stride);
    }
  } else if (auto tile_type = As<TileType>(type)) {
    visit_exprs(tile_type->shape_);
    if (tile_type->tile_view_.has_value()) {
      const auto& tv = tile_type->tile_view_.value();
      visit_exprs(tv.valid_shape);
      visit_exprs(tv.stride);
      if (tv.start_offset) visitor.VisitExpr(tv.start_offset);
    }
  } else if (auto tuple_type = As<TupleType>(type)) {
    for (const auto& elem : tuple_type->types_) {
      VisitTypeExprFields(visitor, elem);
    }
  }
}

/// Collect all Var pointers from a type's expression fields (shape, view, etc.).
/// Operates on types (not IR statements), so kept as a free function.
inline std::unordered_set<const Var*> CollectTypeVars(const TypePtr& type) {
  VarDefUseCollector collector;
  VisitTypeExprFields(collector, type);
  return collector.var_uses;
}

// ============================================================================
// Sorting utilities
// ============================================================================

/// Sort a set of Var pointers deterministically by name_hint_ then UniqueId().
///
/// Useful for iteration-order-sensitive algorithms that process var sets
/// built from unordered containers (e.g. CollectStmtDefinedVars results).
inline std::vector<const Var*> GetSortedVarRefs(const std::unordered_set<const Var*>& refs) {
  std::vector<const Var*> sorted_refs(refs.begin(), refs.end());
  std::sort(sorted_refs.begin(), sorted_refs.end(), [](const Var* lhs, const Var* rhs) {
    if (lhs == rhs) return false;
    if (lhs->name_hint_ != rhs->name_hint_) return lhs->name_hint_ < rhs->name_hint_;
    return lhs->UniqueId() < rhs->UniqueId();
  });
  return sorted_refs;
}

}  // namespace var_collectors
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_VAR_COLLECTORS_H_
