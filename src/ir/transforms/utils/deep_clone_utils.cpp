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

#include "pypto/ir/transforms/utils/deep_clone_utils.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/structural_comparison.h"

namespace pypto {
namespace ir {

namespace {

/// Mutator that deep-copies an IR subtree, creating fresh Var/IterArg/MemRef
/// objects at every definition site (DefField). Uses GetFieldDescriptors
/// reflection to identify which Var fields are definition sites.
class DeepCloneMutator : public IRMutator {
 public:
  explicit DeepCloneMutator(const std::unordered_map<const Var*, ExprPtr>& var_map, bool clone_def_vars)
      : expr_map_(var_map), clone_def_vars_(clone_def_vars) {}

  /// Get the accumulated definition-site Var mapping (excludes non-Var substitutions).
  [[nodiscard]] std::unordered_map<const Var*, VarPtr> GetVarMap() const {
    std::unordered_map<const Var*, VarPtr> result;
    for (const auto& [key, val] : expr_map_) {
      auto var = std::dynamic_pointer_cast<const Var>(val);
      if (var) {
        result[key] = var;
      }
    }
    return result;
  }

 protected:
  // Override VisitStmt_ for each statement type with DefField vars.
  // Pre-register fresh copies BEFORE calling base visitor, which handles
  // traversal and reconstruction. This ensures VisitExpr_(VarPtr) finds
  // the fresh copies during its map lookup.

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (clone_def_vars_) PreRegisterDefFields(*op);

    INTERNAL_CHECK(op->var_) << "AssignStmt has null var";
    INTERNAL_CHECK(op->value_) << "AssignStmt has null value";

    auto new_var_expr = IRMutator::VisitExpr(op->var_);
    auto new_value = IRMutator::VisitExpr(op->value_);
    INTERNAL_CHECK(new_var_expr) << "AssignStmt var mutated to null";
    INTERNAL_CHECK(new_value) << "AssignStmt value mutated to null";

    auto new_var = As<Var>(new_var_expr);
    if (!new_var) {
      auto memref = As<MemRef>(new_var_expr);
      if (memref) {
        new_var = std::static_pointer_cast<const Var>(memref);
      }
    }
    INTERNAL_CHECK(new_var) << "AssignStmt var is not a Var after mutation";

    bool type_matches = false;
    if (new_var->GetType().get() == new_value->GetType().get()) {
      type_matches = true;
    } else {
      try {
        assert_structural_equal(new_var->GetType(), new_value->GetType(), true);
        type_matches = true;
      } catch (const ValueError&) {
        type_matches = false;
      }
    }

    if (!type_matches) {
      auto corrected_var = std::make_shared<Var>(new_var->name_hint_, new_value->GetType(), new_var->span_);
      expr_map_[op->var_.get()] = corrected_var;
      new_var = corrected_var;
    }

    if (new_var.get() != op->var_.get() || new_value.get() != op->value_.get()) {
      return std::make_shared<const AssignStmt>(std::move(new_var), std::move(new_value), op->span_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (clone_def_vars_) PreRegisterDefFields(*op);
    auto mutated = IRMutator::VisitStmt_(op);
    auto for_stmt = As<ForStmt>(mutated);
    if (!for_stmt) return mutated;

    bool corrected = false;
    auto corrected_return_vars = for_stmt->return_vars_;
    const size_t count = std::min(corrected_return_vars.size(), for_stmt->iter_args_.size());
    for (size_t i = 0; i < count; ++i) {
      if (!TypesMatch(corrected_return_vars[i]->GetType(), for_stmt->iter_args_[i]->GetType())) {
        auto corrected_var =
            std::make_shared<Var>(corrected_return_vars[i]->name_hint_, for_stmt->iter_args_[i]->GetType(),
                                  corrected_return_vars[i]->span_);
        expr_map_[op->return_vars_[i].get()] = corrected_var;
        corrected_return_vars[i] = corrected_var;
        corrected = true;
      }
    }

    if (!corrected) return mutated;
    return std::make_shared<const ForStmt>(
        for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_, for_stmt->iter_args_,
        for_stmt->body_, corrected_return_vars, for_stmt->span_, for_stmt->kind_, for_stmt->chunk_size_,
        for_stmt->chunk_policy_, for_stmt->loop_origin_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    if (clone_def_vars_) PreRegisterDefFields(*op);
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    if (clone_def_vars_) PreRegisterDefFields(*op);
    auto mutated = IRMutator::VisitStmt_(op);
    auto while_stmt = As<WhileStmt>(mutated);
    if (!while_stmt) return mutated;

    bool corrected = false;
    auto corrected_return_vars = while_stmt->return_vars_;
    const size_t count = std::min(corrected_return_vars.size(), while_stmt->iter_args_.size());
    for (size_t i = 0; i < count; ++i) {
      if (!TypesMatch(corrected_return_vars[i]->GetType(), while_stmt->iter_args_[i]->GetType())) {
        auto corrected_var =
            std::make_shared<Var>(corrected_return_vars[i]->name_hint_, while_stmt->iter_args_[i]->GetType(),
                                  corrected_return_vars[i]->span_);
        expr_map_[op->return_vars_[i].get()] = corrected_var;
        corrected_return_vars[i] = corrected_var;
        corrected = true;
      }
    }

    if (!corrected) return mutated;
    return std::make_shared<const WhileStmt>(while_stmt->condition_, while_stmt->iter_args_,
                                             while_stmt->body_, corrected_return_vars, while_stmt->span_);
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = expr_map_.find(op.get());
    if (it != expr_map_.end()) {
      return it->second;
    }
    // External variable not in map — return as-is
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = expr_map_.find(op.get());
    if (it != expr_map_.end()) {
      return it->second;
    }
    // Create fresh IterArg with cloned initValue_
    INTERNAL_CHECK(op->initValue_) << "IterArg has null initValue";
    auto new_init = IRMutator::VisitExpr(op->initValue_);
    auto iter_type = op->GetType();
    if (!TypesMatch(iter_type, new_init->GetType())) {
      iter_type = new_init->GetType();
    }
    auto fresh = std::make_shared<IterArg>(op->name_hint_, iter_type, std::move(new_init), op->span_);
    expr_map_[op.get()] = fresh;
    return fresh;
  }

  ExprPtr VisitExpr_(const MemRefPtr& op) override {
    auto it = expr_map_.find(op.get());
    if (it != expr_map_.end()) {
      return it->second;
    }
    // Create fresh MemRef with cloned addr_
    auto new_addr = op->addr_ ? IRMutator::VisitExpr(op->addr_) : op->addr_;
    auto fresh = std::make_shared<MemRef>(op->name_hint_, std::move(new_addr), op->size_, op->id_, op->span_);
    expr_map_[op.get()] = fresh;
    return fresh;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    std::vector<ExprPtr> new_args;
    bool changed = false;
    new_args.reserve(op->args_.size());

    for (const auto& arg : op->args_) {
      INTERNAL_CHECK(arg) << "Call has null argument";
      auto new_arg = IRMutator::VisitExpr(arg);
      INTERNAL_CHECK(new_arg) << "Call argument mutated to null";
      new_args.push_back(new_arg);
      if (new_arg.get() != arg.get()) {
        changed = true;
      }
    }

    if (!changed) {
      return op;
    }

    if (auto opnode = std::dynamic_pointer_cast<const Op>(op->op_)) {
      const auto& op_name = opnode->name_;
      if (op_name.rfind("tile.", 0) == 0 && op_name != "tile.tpush_to_aic" &&
          op_name != "tile.tpush_to_aiv" && op_name != "tile.tpop_from_aic" &&
          op_name != "tile.tpop_from_aiv") {
        return OpRegistry::GetInstance().Create(op_name, new_args, op->kwargs_, op->span_);
      }
    }

    return std::make_shared<const Call>(op->op_, std::move(new_args), op->kwargs_, op->GetType(), op->span_);
  }

 private:
  static bool TypesMatch(const TypePtr& lhs, const TypePtr& rhs) {
    if (lhs.get() == rhs.get()) return true;
    try {
      assert_structural_equal(lhs, rhs, true);
      return true;
    } catch (const ValueError&) {
      return false;
    }
  }

  /// Create a fresh Var with same name and type, register in expr_map_.
  void CloneVar(const VarPtr& op) {
    if (expr_map_.count(op.get())) return;  // Already mapped (e.g. pre-seeded)
    // Check if the actual runtime type is MemRef — don't create a plain Var for MemRef
    if (op->GetKind() == ObjectKind::MemRef) {
      // MemRef will be handled by VisitExpr_(MemRefPtr) during traversal
      return;
    }
    auto fresh = std::make_shared<Var>(op->name_hint_, op->GetType(), op->span_);
    expr_map_[op.get()] = fresh;
  }

  /// Use GetFieldDescriptors to find DefField VarPtr/vector<VarPtr> entries
  /// and pre-register fresh copies in expr_map_.
  template <typename StmtType>
  void PreRegisterDefFields(const StmtType& stmt) {
    constexpr auto descriptors = StmtType::GetFieldDescriptors();
    std::apply([this, &stmt](const auto&... desc) { (PreRegisterOneField(desc, stmt), ...); }, descriptors);
  }

  template <typename Desc, typename StmtType>
  void PreRegisterOneField(const Desc& desc, const StmtType& stmt) {
    using KindTag = typename Desc::kind_tag;
    using FieldType = typename Desc::field_type;

    if constexpr (!std::is_same_v<KindTag, reflection::DefFieldTag>) {
      return;  // Only process DefField entries
    } else if constexpr (std::is_same_v<FieldType, VarPtr>) {
      const auto& var = desc.Get(stmt);
      if (var) CloneVar(var);
    } else if constexpr (std::is_same_v<FieldType, std::vector<VarPtr>>) {
      for (const auto& var : desc.Get(stmt)) {
        if (var) CloneVar(var);
      }
    }
    // IterArgPtr and vector<IterArgPtr> DefFields are handled by VisitExpr_(IterArgPtr)
  }

  std::unordered_map<const Var*, ExprPtr> expr_map_;
  bool clone_def_vars_;
};

}  // namespace

DeepCloneResult DeepClone(const StmtPtr& body, const std::unordered_map<const Var*, ExprPtr>& var_map,
                          bool clone_def_vars) {
  DeepCloneMutator mutator(var_map, clone_def_vars);
  auto cloned = mutator.VisitStmt(body);
  return {cloned, mutator.GetVarMap()};
}

}  // namespace ir
}  // namespace pypto
