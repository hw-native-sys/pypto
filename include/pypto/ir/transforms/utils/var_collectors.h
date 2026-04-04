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
// Visitor-based collectors
// ============================================================================

/// Collect all variable references in an IR subtree (by pointer identity).
///
/// Traverses the full subtree and records every Var/IterArg pointer encountered
/// in expressions, including definition-site variables (AssignStmt::var_, etc.).
/// For IterArgs defined by ForStmt/WhileStmt, their initValue_ expressions are
/// visited explicitly so that outer-scope references are captured.
class VarRefCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> var_refs;

 protected:
  void VisitExpr_(const VarPtr& op) override { var_refs.insert(op.get()); }

  void VisitExpr_(const IterArgPtr& op) override {
    var_refs.insert(op.get());
    // Do not traverse initValue_: when an IterArg appears as an expression
    // reference (defined by an outer loop), its initValue_ belongs to that
    // outer loop's initialization, not to the scope being analyzed.
    // InitValues of IterArgs defined by inner ForStmt/WhileStmt are visited
    // explicitly in VisitStmt_ overrides below.
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    // Explicitly visit initValue_ for IterArgs defined by THIS ForStmt.
    // These are genuine references to variables from the enclosing scope
    // that must be captured as inputs when outlining.
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg->initValue_) {
        VisitExpr(iter_arg->initValue_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg->initValue_) {
        VisitExpr(iter_arg->initValue_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }
};

/// Collect all variable definitions in an IR subtree (by pointer identity).
///
/// Records every variable that is *defined* by a statement node:
/// AssignStmt::var_, ForStmt::loop_var_/iter_args_/return_vars_,
/// WhileStmt::iter_args_/return_vars_, IfStmt::return_vars_.
/// Traverses the full subtree recursively.
class VarDefCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> var_defs;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    var_defs.insert(op->var_.get());
    // Don't visit the RHS - we only care about definitions
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    var_defs.insert(op->loop_var_.get());
    for (const auto& iter_arg : op->iter_args_) {
      var_defs.insert(iter_arg.get());
    }
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& iter_arg : op->iter_args_) {
      var_defs.insert(iter_arg.get());
    }
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& return_var : op->return_vars_) {
      var_defs.insert(return_var.get());
    }
    IRVisitor::VisitStmt_(op);
  }
};

/// Collect variable use-sites only (skips definition-site LHS variables).
///
/// Unlike VarRefCollector which captures every Var pointer including definitions,
/// this collector only records variables that appear as RHS uses — i.e., variables
/// that are read, not defined. AssignStmt LHS is explicitly skipped.
class VarUseCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> var_uses;

 protected:
  void VisitExpr_(const VarPtr& op) override { var_uses.insert(op.get()); }
  void VisitExpr_(const IterArgPtr& op) override { var_uses.insert(op.get()); }
  void VisitStmt_(const AssignStmtPtr& op) override {
    // Only visit RHS value, not LHS var — definitions are not uses.
    VisitExpr(op->value_);
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

/// Collect all variable definition sites in DFS pre-order.
///
/// Similar scope to VarDefCollector but preserves traversal order:
/// AssignStmt::var_, ForStmt::loop_var_/return_vars_/iter_args_,
/// IfStmt/WhileStmt::return_vars_. Recurses into all control-flow bodies
/// and ScopeStmt.
inline void CollectVarDefsInOrder(const StmtPtr& stmt, std::vector<const Var*>& out) {
  if (!stmt) return;
  if (auto assign = As<AssignStmt>(stmt)) {
    out.push_back(assign->var_.get());
  } else if (auto for_stmt = As<ForStmt>(stmt)) {
    out.push_back(for_stmt->loop_var_.get());
    for (auto& rv : for_stmt->return_vars_) out.push_back(rv.get());
    for (auto& ia : for_stmt->iter_args_) out.push_back(ia.get());
    CollectVarDefsInOrder(for_stmt->body_, out);
  } else if (auto if_stmt = As<IfStmt>(stmt)) {
    for (auto& rv : if_stmt->return_vars_) out.push_back(rv.get());
    CollectVarDefsInOrder(if_stmt->then_body_, out);
    if (if_stmt->else_body_.has_value()) CollectVarDefsInOrder(*if_stmt->else_body_, out);
  } else if (auto while_stmt = As<WhileStmt>(stmt)) {
    for (auto& rv : while_stmt->return_vars_) out.push_back(rv.get());
    CollectVarDefsInOrder(while_stmt->body_, out);
  } else if (auto seq = As<SeqStmts>(stmt)) {
    for (auto& s : seq->stmts_) CollectVarDefsInOrder(s, out);
  } else if (auto scope = As<ScopeStmt>(stmt)) {
    CollectVarDefsInOrder(scope->body_, out);
  }
}

/// Collect only AssignStmt variable definitions recursively.
///
/// Unlike VarDefCollector which also collects loop variables, iter_args, and
/// return_vars, this function collects only variables on the LHS of
/// AssignStmt nodes. Recurses into all control-flow bodies.
inline void CollectAssignDefs(const StmtPtr& stmt, std::unordered_set<const Var*>& result) {
  if (!stmt) return;
  if (auto assign = As<AssignStmt>(stmt)) {
    result.insert(assign->var_.get());
  } else if (auto for_stmt = As<ForStmt>(stmt)) {
    // Don't record loop_var — it's scoped to the loop body, not an outer assignment
    CollectAssignDefs(for_stmt->body_, result);
  } else if (auto while_stmt = As<WhileStmt>(stmt)) {
    CollectAssignDefs(while_stmt->body_, result);
  } else if (auto if_stmt = As<IfStmt>(stmt)) {
    CollectAssignDefs(if_stmt->then_body_, result);
    if (if_stmt->else_body_.has_value()) CollectAssignDefs(*if_stmt->else_body_, result);
  } else if (auto seq = As<SeqStmts>(stmt)) {
    for (auto& s : seq->stmts_) CollectAssignDefs(s, result);
  } else if (auto scope = As<ScopeStmt>(stmt)) {
    CollectAssignDefs(scope->body_, result);
  }
}

/// Convenience overload returning a new set.
inline std::unordered_set<const Var*> CollectAssignDefs(const StmtPtr& stmt) {
  std::unordered_set<const Var*> result;
  CollectAssignDefs(stmt, result);
  return result;
}

// ============================================================================
// Type expression visitors
// ============================================================================

/// Visit all expression fields embedded in a type using the given visitor.
///
/// Covers: TensorType::shape_, tensor_view_.{valid_shape, stride};
///         TileType::shape_, tile_view_.{valid_shape, stride, start_offset};
///         TupleType elements (recursively).
///
/// This captures dynamic shape variables (ScalarExpr/Var nodes) that appear
/// inside type annotations — e.g., Tensor[[N, M], FP32] where N, M are
/// dynamic dimension vars.
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
///
/// Walks expression trees in shape, valid_shape, stride, and start_offset
/// to find all referenced Var nodes. These represent dynamic dimension
/// variables that are implicitly defined by appearing in function parameter
/// or return type annotations.
inline std::unordered_set<const Var*> CollectTypeVars(const TypePtr& type) {
  VarRefCollector collector;
  VisitTypeExprFields(collector, type);
  return collector.var_refs;
}

// ============================================================================
// Sorting utilities
// ============================================================================

/// Sort a set of Var pointers deterministically by name_hint_ then UniqueId().
///
/// Useful for iteration-order-sensitive algorithms that process var sets
/// built from unordered containers.
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
