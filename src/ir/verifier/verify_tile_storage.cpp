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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir {
namespace {

bool SameWindow(const TileTypePtr& lhs, const TileTypePtr& rhs, bool physical = false) {
  if (!lhs || !rhs || lhs->GetMemorySpace() != rhs->GetMemorySpace()) return false;
  const auto a = GetDefinedMemRef(lhs);
  const auto b = GetDefinedMemRef(rhs);
  return (physical || MemRef::SameAllocation(a, b)) && a->size_ == b->size_ &&
         structural_equal(a->byte_offset_, b->byte_offset_) &&
         (physical ||
          (a->slot_count_ == b->slot_count_ && a->slot_index_.has_value() == b->slot_index_.has_value() &&
           (!a->slot_index_.has_value() || structural_equal(*a->slot_index_, *b->slot_index_))));
}

// A fixed walk checks storage closure; sorted windows detect conflicts in
// O(N log N) total work. This is not a data-initialization/lifetime proof.
class TileStorageVisitor : public IRVisitor {
 public:
  TileStorageVisitor(std::vector<Diagnostic>& diagnostics, bool physical)
      : diagnostics_(diagnostics), physical_(physical) {}

  void VisitFunction(const FunctionPtr& function) override {
    function_name_ = function->name_;
    IRVisitor::VisitFunction(function);
  }

 protected:
  void VisitVarLike_(const VarPtr& var) override {
    if (checked_.insert(var.get()).second) CheckTile(var);
    IRVisitor::VisitVarLike_(var);
  }

  // An IterArg occurrence reads the current value; its initializer is visited
  // exactly once at the loop boundary, not recursively at every body use.
  void VisitExpr_(const IterArgPtr& var) override { VisitVarLike_(var); }

  void VisitStmt_(const ForStmtPtr& loop) override {
    CheckLoop(loop->iter_args_, loop->return_vars_, loop->body_, loop->span_);
    VisitExpr(loop->loop_var_);
    VisitExpr(loop->start_);
    VisitExpr(loop->stop_);
    VisitExpr(loop->step_);
    VisitStmt(loop->body_);
    for (const auto& [key, value] : loop->attrs_) {
      (void)key;
      ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
    }
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    CheckLoop(loop->iter_args_, loop->return_vars_, loop->body_, loop->span_);
    VisitExpr(loop->condition_);
    VisitStmt(loop->body_);
  }

  void VisitStmt_(const IfStmtPtr& branch) override {
    std::vector<TileTypePtr> targets;
    for (const auto& result : branch->return_vars_) {
      if (auto tile = CheckTile(result)) targets.push_back(tile);
    }
    CheckDestinations(targets, branch->span_, /*allow_identical=*/true);
    for (const auto& body : {std::optional<StmtPtr>(branch->then_body_), branch->else_body_}) {
      const auto yield = body ? transform_utils::GetLastYieldStmt(*body) : nullptr;
      for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
        auto target = CheckTile(branch->return_vars_[i]);
        if (!target) continue;
        auto source = yield && i < yield->value_.size() ? CheckTile(yield->value_[i]) : nullptr;
        if (!SameWindow(source, target)) {
          Error("Every tile branch arm must yield its declared storage", branch->span_);
        }
      }
    }
    VisitExpr(branch->condition_);
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
  }

  void VisitStmt_(const AssignStmtPtr& statement) override {
    auto call = As<Call>(statement->value_);
    if (physical_ && IsOp(call, "tile.move") && !call->args_.empty()) {
      auto source = CheckTile(call->args_[0]);
      auto target = CheckTile(statement->var_);
      // An exact self-copy is a no-op that final lowering can eliminate.
      if (source && target && !SameWindow(source, target, /*physical=*/true)) {
        CheckDestinations({source, target}, statement->span_, /*allow_identical=*/false);
      }
    }
    IRVisitor::VisitStmt_(statement);
  }

 private:
  void Error(const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error,
                              physical_ ? "TileStorageAllocated" : "TileStorageLegalized", 1,
                              message + " in function '" + function_name_ + "'", span);
  }

  TileTypePtr CheckTile(const ExprPtr& value) {
    if (!value) return nullptr;
    if (As<TupleType>(value->GetType()) && ContainsTile(value->GetType())) {
      Error("Tile storage in tuple values must be flattened before storage legalization", value->span_);
      return nullptr;
    }
    const auto tile = As<TileType>(value->GetType());
    if (!tile) return nullptr;
    if (!tile->memref_.has_value() || !*tile->memref_ || !(*tile->memref_)->base_ ||
        !tile->GetMemorySpace().has_value() || !(*tile->memref_)->byte_offset_ ||
        (*tile->memref_)->size_ == 0) {
      Error("On-chip tile storage must have a MemRef and memory space", value->span_);
      return nullptr;
    }
    if (physical_) {
      if (auto offset = As<ConstInt>((*tile->memref_)->byte_offset_); offset && offset->value_ < 0) {
        Error("Allocated tile storage cannot have a negative effective address", value->span_);
        return nullptr;
      }
    }
    return tile;
  }

  bool ContainsTile(const TypePtr& type) {
    auto known = contains_tile_.find(type.get());
    if (known != contains_tile_.end()) return known->second;
    bool result = IsA<TileType>(type);
    if (auto tuple = As<TupleType>(type)) {
      for (const auto& element : tuple->types_) result = ContainsTile(element) || result;
    }
    contains_tile_[type.get()] = result;
    return result;
  }

  void CheckLoop(const std::vector<IterArgPtr>& arguments, const std::vector<VarPtr>& results,
                 const StmtPtr& body, const Span& span) {
    auto yield = transform_utils::GetLastYieldStmt(body);
    for (const auto& result : results) CheckTile(result);
    std::vector<TileTypePtr> targets;
    for (size_t i = 0; i < arguments.size(); ++i) {
      VisitExpr(arguments[i]->initValue_);
      auto target = CheckTile(arguments[i]);
      if (!target) continue;
      targets.push_back(target);
      auto initial = CheckTile(arguments[i]->initValue_);
      auto yielded = yield && i < yield->value_.size() ? CheckTile(yield->value_[i]) : nullptr;
      auto result = i < results.size() ? CheckTile(results[i]) : nullptr;
      if (!SameWindow(initial, target) || !SameWindow(yielded, target) || !SameWindow(result, target)) {
        Error("Tile loop initializer, iter_arg, yield and result must share canonical storage", span);
      }
    }
    CheckDestinations(targets, span, /*allow_identical=*/false);
  }

  void CheckDestinations(const std::vector<TileTypePtr>& tiles, const Span& span, bool allow_identical) {
    using Key = std::pair<MemorySpace, const Var*>;
    struct KeyLess {
      bool operator()(const Key& a, const Key& b) const {
        if (a.first != b.first) return a.first < b.first;
        return std::less<const Var*>{}(a.second, b.second);
      }
    };
    std::map<Key, std::vector<TileTypePtr>, KeyLess> groups;
    for (const auto& tile : tiles) {
      auto memref = GetDefinedMemRef(tile);
      groups[{*tile->GetMemorySpace(), physical_ ? nullptr : memref->base_.get()}].push_back(tile);
    }
    for (auto& [key, group] : groups) {
      (void)key;
      if (group.size() < 2) continue;
      std::vector<std::pair<int64_t, TileTypePtr>> windows;
      for (const auto& tile : group) {
        auto offset = As<ConstInt>(GetDefinedMemRef(tile)->byte_offset_);
        if (!offset) {
          Error("Simultaneous tile storage windows have unprovable symbolic overlap", span);
          return;
        }
        windows.emplace_back(offset->value_, tile);
      }
      std::sort(windows.begin(), windows.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      auto previous = windows.front();
      __int128 end = static_cast<__int128>(previous.first) + GetDefinedMemRef(previous.second)->size_;
      for (size_t i = 1; i < windows.size(); ++i) {
        const auto& current = windows[i];
        if (current.first < end && !(allow_identical && SameWindow(previous.second, current.second))) {
          Error("Simultaneous tile storage windows overlap", span);
          return;
        }
        const __int128 current_end =
            static_cast<__int128>(current.first) + GetDefinedMemRef(current.second)->size_;
        if (current_end > end) {
          previous = current;
          end = current_end;
        }
      }
    }
  }

  std::vector<Diagnostic>& diagnostics_;
  bool physical_;
  std::string function_name_;
  std::unordered_set<const Var*> checked_;
  std::unordered_map<const Type*, bool> contains_tile_;
};

class TileStoragePropertyVerifier : public PropertyVerifier {
 public:
  explicit TileStoragePropertyVerifier(bool physical) : physical_(physical) {}
  [[nodiscard]] std::string GetName() const override {
    return physical_ ? "TileStorageAllocated" : "TileStorageLegalized";
  }
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    for (const auto& [name, function] : program->functions_) {
      (void)name;
      if (IsInCoreType(function->func_type_)) {
        TileStorageVisitor(diagnostics, physical_).VisitFunction(function);
      }
    }
  }

 private:
  bool physical_;
};
}  // namespace

PropertyVerifierPtr CreateTileStorageLegalizedPropertyVerifier() {
  return std::make_shared<TileStoragePropertyVerifier>(false);
}

PropertyVerifierPtr CreateTileStorageAllocatedPropertyVerifier() {
  return std::make_shared<TileStoragePropertyVerifier>(true);
}

}  // namespace pypto::ir
