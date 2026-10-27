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
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/auto_tile_matmul_l0/internal.h"

namespace pypto::ir::auto_tile_matmul_l0 {
namespace {

// A base is an allocation identity, not a byte address. Unresolved declared
// MemRefs have size zero, so MemRef::MayAlias cannot prove slot disjointness.
// A null base represents ambiguous control-flow provenance.
struct Storage {
  const Var* base;
  std::optional<int64_t> slot;
  std::optional<MemorySpace> space;
};

Storage UnknownStorage(const VarPtr& var) {
  const auto tile = As<TileType>(var->GetType());
  return Storage{nullptr, std::nullopt, tile ? tile->GetMemorySpace() : std::nullopt};
}

struct Writes {
  std::vector<size_t> all;
  std::vector<size_t> wildcard;
  std::unordered_map<int64_t, std::vector<size_t>> slots;
};

bool InInterval(const std::vector<size_t>& points, size_t start, size_t end) {
  const auto next = std::upper_bound(points.begin(), points.end(), start);
  return next != points.end() && *next < end;
}

// One lexical traversal records storage lineage and ordered writes. Each
// deferred operand then needs at most two O(log N) interval queries. Neither
// nested regions nor the interval from matmul to store is rescanned.
class SnapshotIndex : public IRVisitor {
 public:
  std::unordered_map<const Var*, std::string> Analyze(const FunctionPtr& func) {
    for (const auto& param : func->params_) {
      // Tile parameters have no fresh-allocation/noalias proof at this
      // boundary; ordinary tensor parameters do not occupy on-chip storage.
      if (As<TileType>(param->GetType())) storage_[param.get()] = UnknownStorage(param);
    }
    VisitStmt(func->body_);
    std::unordered_map<const Var*, std::string> hazards;
    for (const auto& query : queries_) {
      const auto store = stores_.find(query.result);
      if (store == stores_.end() || store->second <= query.point) continue;
      for (const auto& [operand, storage] : query.inputs) {
        if (MayWrite(storage, query.point, store->second)) {
          hazards.emplace(query.result, operand->name_hint_);
          break;
        }
      }
    }
    return hazards;
  }

 protected:
  // Types and storage annotations describe values, not executed effects.
  // Do not rewalk shared metadata every time a Var is read.
  void VisitVarLike_(const VarPtr& /*op*/) override {}
  void VisitExpr_(const IterArgPtr& /*op*/) override {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    VisitExpr(op->value_);
    const size_t execution_point = point_;
    const size_t point = ++point_;
    auto call = As<Call>(op->value_);
    const auto* entry = call ? LookupOpEntry(call->op_) : nullptr;
    // Follow the same alias edges as InitMemRef. A declared result binding
    // takes precedence; an ordinary fresh result owns its own SSA storage.
    if (auto alias = AsVarLike(op->value_)) {
      storage_[op->var_.get()] = Resolve(alias).value_or(UnknownStorage(op->var_));
    } else if (!entry) {
      storage_[op->var_.get()] = UnknownStorage(op->var_);
    } else {
      auto reuse = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
      if (!reuse && op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
        for (size_t i = 0; i < call->args_.size(); ++i) {
          if (As<TileType>(call->args_[i]->GetType())) {
            reuse = i;
            break;
          }
        }
      }
      if (reuse) {
        storage_[op->var_.get()] = Resolve(AsVarLike(call->args_[*reuse])).value_or(UnknownStorage(op->var_));
      }
    }
    if (!call) return;
    if (!entry || entry->GetExecutionMemoryAccessEvidence() != ExecutionMemoryAccessEvidence::NoAccess) {
      if (auto result = Resolve(op->var_)) RecordWrite(*result, point);
    }
    if (IsOp(call, "tile.store") && !call->args_.empty()) {
      if (auto source = AsVarLike(call->args_[0])) stores_.emplace(source.get(), execution_point);
    }
    if (IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_bias")) {
      Query query{op->var_.get(), point, {}};
      for (const auto& argument : call->args_) {
        if (auto operand = AsVarLike(argument)) {
          if (auto storage = Resolve(operand)) query.inputs.emplace_back(operand, *storage);
        }
      }
      queries_.push_back(std::move(query));
    }
  }

  void VisitExpr_(const CallPtr& op) override {
    for (const auto& arg : op->args_) VisitExpr(arg);
    for (const auto& [key, value] : op->attrs_) {
      ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
    }
    RecordCall(op, ++point_);
  }
  void VisitExpr_(const SubmitPtr& /*op*/) override { opaque_.push_back(++point_); }

  void VisitStmt_(const ForStmtPtr& op) override {
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    for (const auto& [key, value] : op->attrs_) {
      ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
    }
    for (const auto& arg : op->iter_args_) {
      VisitExpr(arg->initValue_);
      // A yielded allocation may differ from the first initializer in later
      // iterations. Do not assume initializer lineage for all body reads.
      storage_[arg.get()] = UnknownStorage(arg);
    }
    VisitStmt(op->body_);
    for (const auto& result : op->return_vars_) storage_[result.get()] = UnknownStorage(result);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& arg : op->iter_args_) {
      VisitExpr(arg->initValue_);
      storage_[arg.get()] = UnknownStorage(arg);
    }
    VisitExpr(op->condition_);
    VisitStmt(op->body_);
    for (const auto& result : op->return_vars_) storage_[result.get()] = UnknownStorage(result);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    VisitStmt(op->then_body_);
    if (op->else_body_) VisitStmt(*op->else_body_);
    for (const auto& result : op->return_vars_) storage_[result.get()] = UnknownStorage(result);
  }

 private:
  struct Query {
    const Var* result;
    size_t point;
    std::vector<std::pair<VarPtr, Storage>> inputs;
  };

  std::optional<Storage> Resolve(const VarPtr& var) const {
    auto tile = var ? As<TileType>(var->GetType()) : nullptr;
    if (!tile) return std::nullopt;
    const auto found = storage_.find(var.get());
    if (found != storage_.end()) return found->second;
    if (tile->memref_) {
      const auto& memref = *tile->memref_;
      const auto slot = memref->slot_index_ ? As<ConstInt>(*memref->slot_index_) : nullptr;
      return Storage{memref->base_.get(), slot ? std::optional<int64_t>(slot->value_) : std::nullopt,
                     tile->GetMemorySpace()};
    }
    return Storage{var.get(), std::nullopt, tile->GetMemorySpace()};
  }

  void RecordWrite(const Storage& storage, size_t point) {
    all_writes_.push_back(point);
    space_writes_[storage.space].push_back(point);
    if (!storage.base) {
      ambiguous_writes_.push_back(point);
      ambiguous_space_writes_[storage.space].push_back(point);
      return;
    }
    auto& writes = writes_[storage.base];
    writes.all.push_back(point);
    if (storage.slot) {
      writes.slots[*storage.slot].push_back(point);
    } else {
      writes.wildcard.push_back(point);
    }
  }

  void RecordCall(const CallPtr& call, size_t point) {
    const auto* entry = LookupOpEntry(call->op_);
    if (!entry) {
      opaque_.push_back(point);
      return;
    }
    if (entry->GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::NoAccess) return;
    // Unknown can hide reserved on-chip effects (for example tfree/next
    // tpop reusing a cross-core ring). The one route-specific exception is
    // tile.store: it reads its Tile source and writes only its GM Tensor
    // destination, including atomic stores. Keep its explicit arg effects.
    if (entry->GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::Unknown &&
        !IsOp(call, "tile.store")) {
      opaque_.push_back(point);
    }
    for (size_t i = 0; i < call->args_.size(); ++i) {
      if (ArgEffectWrites(entry->GetArgEffect(i, call->kwargs_))) {
        if (auto storage = Resolve(AsVarLike(call->args_[i]))) RecordWrite(*storage, point);
      }
    }
  }

  bool MayWrite(const Storage& input, size_t start, size_t end) const {
    if (InInterval(opaque_, start, end)) return true;
    // Unknown provenance does not erase a known address space. Acc/Vec
    // writes cannot overwrite a Mat input. Truly opaque effects still fence
    // every space; unknown-base writes fence their own space and unknown space.
    const auto in_space = [&](const auto& index) {
      const auto unknown = index.find(std::nullopt);
      if (unknown != index.end() && InInterval(unknown->second, start, end)) return true;
      const auto found = index.find(input.space);
      return found != index.end() && InInterval(found->second, start, end);
    };
    if (!input.base) {
      return input.space ? in_space(space_writes_) : InInterval(all_writes_, start, end);
    }
    if (input.space ? in_space(ambiguous_space_writes_) : InInterval(ambiguous_writes_, start, end)) {
      return true;
    }
    const auto found = writes_.find(input.base);
    if (found == writes_.end()) return false;
    const auto& writes = found->second;
    if (!input.slot) return InInterval(writes.all, start, end);
    if (InInterval(writes.wildcard, start, end)) return true;
    const auto slot = writes.slots.find(*input.slot);
    return slot != writes.slots.end() && InInterval(slot->second, start, end);
  }

  size_t point_ = 0;
  std::unordered_map<const Var*, Storage> storage_;
  std::unordered_map<const Var*, Writes> writes_;
  std::vector<size_t> all_writes_;
  std::map<std::optional<MemorySpace>, std::vector<size_t>> space_writes_;
  std::vector<size_t> ambiguous_writes_;
  std::map<std::optional<MemorySpace>, std::vector<size_t>> ambiguous_space_writes_;
  std::vector<size_t> opaque_;
  std::unordered_map<const Var*, size_t> stores_;
  std::vector<Query> queries_;
};
}  // namespace

std::unordered_map<const Var*, std::string> AnalyzeDelayedInputHazards(const FunctionPtr& func) {
  return SnapshotIndex().Analyze(func);
}
}  // namespace pypto::ir::auto_tile_matmul_l0
