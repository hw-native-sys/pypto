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
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Mutator that deep-copies an IR subtree, creating fresh Var/IterArg/MemRef
/// objects at every definition site (DefField). Uses GetFieldDescriptors
/// reflection to identify which Var fields are definition sites.
class DeepCloneMutator : public IRMutator {
 public:
  explicit DeepCloneMutator(const std::unordered_map<const Var*, ExprPtr>& var_map, bool clone_def_vars)
      : expr_map_(var_map), clone_def_vars_(clone_def_vars) {
    seed_keys_.reserve(var_map.size());
    for (const auto& [key, _val] : var_map) {
      seed_keys_.insert(key);
    }
  }

  /// Get the definition-site Var mapping created by the clone traversal —
  /// excludes caller-seeded entries (whether they pointed at Vars or not) so
  /// the result matches the "definition-site clones only" contract.
  [[nodiscard]] std::unordered_map<const Var*, VarPtr> GetVarMap() const {
    std::unordered_map<const Var*, VarPtr> result;
    for (const auto& [key, val] : expr_map_) {
      if (seed_keys_.count(key)) continue;
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
    // Create fresh IterArg with cloned initValue_ and a remapped type — the
    // type may embed expressions (shape dims, TileView/TensorView fields,
    // MemRef byte_offset) that reference Vars in expr_map_.
    INTERNAL_CHECK_SPAN(op->initValue_, op->span_) << "IterArg has null initValue";
    auto new_init = IRMutator::VisitExpr(op->initValue_);
    auto new_type = RemapType(op->GetType());
    auto fresh =
        std::make_shared<IterArg>(op->name_hint_, std::move(new_type), std::move(new_init), op->span_);
    expr_map_[op.get()] = fresh;
    return fresh;
  }

  ExprPtr VisitExpr_(const MemRefPtr& op) override {
    auto it = expr_map_.find(op.get());
    if (it != expr_map_.end()) {
      return it->second;
    }
    // Remap base_ through the mutator so a substituted Ptr var carries through
    // into the cloned MemRef (preserves allocation identity across a scope
    // substitution). If the remapped expr is not a Var, fall back to the
    // original base_ — the alternative is silently corrupting MemRef invariants.
    VarPtr new_base = op->base_;
    if (op->base_) {
      auto remapped_base = IRMutator::VisitExpr(op->base_);
      if (auto as_var = std::dynamic_pointer_cast<const Var>(remapped_base)) {
        new_base = as_var;
      }
    }
    auto new_offset = op->byte_offset_ ? IRMutator::VisitExpr(op->byte_offset_) : op->byte_offset_;
    auto fresh = std::make_shared<MemRef>(op->name_hint_, std::move(new_base), std::move(new_offset),
                                          op->size_, op->span_);
    expr_map_[op.get()] = fresh;
    return fresh;
  }

 private:
  /// Create a fresh Var with a remapped type, register in expr_map_.
  /// The type is rewritten so shape dims, TileView/TensorView fields, and any
  /// embedded MemRef's byte_offset are substituted via expr_map_ — otherwise
  /// the fresh Var's type would still reference the caller's old Var pointers.
  void CloneVar(const VarPtr& op) {
    if (expr_map_.count(op.get())) return;  // Already mapped (e.g. pre-seeded)
    // Check if the actual runtime type is MemRef — don't create a plain Var for MemRef
    if (op->GetKind() == ObjectKind::MemRef) {
      // MemRef will be handled by VisitExpr_(MemRefPtr) during traversal
      return;
    }
    auto new_type = RemapType(op->GetType());
    auto fresh = std::make_shared<Var>(op->name_hint_, std::move(new_type), op->span_);
    expr_map_[op.get()] = fresh;
  }

  /// Remap expressions inside a TypePtr (shape, TileView/TensorView fields, MemRef).
  /// Recurses into TupleType element types so nested shaped types pick up
  /// substitutions too. Returns the original pointer unchanged if nothing
  /// inside references a remapped var.
  TypePtr RemapType(const TypePtr& type) {
    if (!type) return type;
    // TupleType element types may themselves embed remappable expressions.
    if (auto tuple_type = std::dynamic_pointer_cast<const TupleType>(type)) {
      std::vector<TypePtr> new_types;
      new_types.reserve(tuple_type->types_.size());
      bool changed = false;
      for (const auto& elem : tuple_type->types_) {
        auto new_elem = RemapType(elem);
        if (new_elem.get() != elem.get()) {
          changed = true;
        }
        new_types.push_back(std::move(new_elem));
      }
      if (!changed) return type;
      return std::make_shared<TupleType>(std::move(new_types));
    }
    // Remap the embedded MemRef (if any) so its byte_offset_ expression picks up
    // substitutions. Dispatches through VisitExpr_(const MemRefPtr&), which creates
    // a fresh MemRef with the mutated offset and caches it in expr_map_.
    auto original_memref_opt = GetTypeMemRef(type);
    std::optional<MemRefPtr> new_memref_opt = original_memref_opt;
    if (original_memref_opt.has_value()) {
      auto remapped = IRMutator::VisitExpr(*original_memref_opt);
      if (auto as_memref = std::dynamic_pointer_cast<const MemRef>(remapped)) {
        new_memref_opt = as_memref;
      }
    }
    return CloneTypeWithMemRefAndRemapExprs(type, new_memref_opt,
                                            [this](const ExprPtr& e) { return IRMutator::VisitExpr(e); });
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
  /// Keys present in the caller-supplied seed map; filtered out of GetVarMap()
  /// so only entries created by the clone traversal are returned.
  std::unordered_set<const Var*> seed_keys_;
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
