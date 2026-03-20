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

#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"

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
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (clone_def_vars_) PreRegisterDefFields(*op);
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    if (clone_def_vars_) PreRegisterDefFields(*op);
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    if (clone_def_vars_) PreRegisterDefFields(*op);
    return IRMutator::VisitStmt_(op);
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
    auto fresh = std::make_shared<IterArg>(op->name_hint_, op->GetType(), std::move(new_init), op->span_);
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

 private:
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
