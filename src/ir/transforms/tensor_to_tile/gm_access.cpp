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
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/tensor_to_tile/internal.h"

namespace pypto {
namespace ir {
namespace tensor_to_tile {

using tile_conversion_utils::MakeShapeTuple;
using transform_utils::FlattenToStmts;

bool ExprUsesVar(const ExprPtr& expr, const Var* target) {
  if (!expr || !target) return false;
  var_collectors::VarDefUseCollector collector;
  collector.VisitExpr(expr);
  return collector.var_uses.count(target) > 0;
}

bool StmtUsesVar(const StmtPtr& stmt, const Var* target) {
  if (!stmt || !target) return false;
  var_collectors::VarDefUseCollector collector;
  collector.VisitStmt(stmt);
  return collector.var_uses.count(target) > 0;
}

namespace {

struct FullTensorScalarUpdateCandidate {
  size_t assemble_index;
  VarPtr target;
  VarPtr staging;
  AssignStmtPtr full_stmt;
  AssignStmtPtr assemble_stmt;
  bool safe = true;
  size_t scalar_write_count = 0;
};

bool IsZeroOffsetTuple(const ExprPtr& expr, size_t rank) {
  auto offsets = As<MakeTuple>(expr);
  if (!offsets || offsets->elements_.size() != rank) return false;
  return std::all_of(offsets->elements_.begin(), offsets->elements_.end(), [](const ExprPtr& offset) {
    auto value = As<ConstInt>(offset);
    return value && value->value_ == 0;
  });
}

/** @brief Analyze all full-init candidates in one O(N) traversal. */
class StagedScalarUseAnalyzer : public IRVisitor {
 public:
  explicit StagedScalarUseAnalyzer(std::vector<FullTensorScalarUpdateCandidate>* candidates)
      : candidates_(candidates) {
    for (size_t i = 0; i < candidates_->size(); ++i) {
      const auto& candidate = (*candidates_)[i];
      candidate_by_target_[candidate.target.get()] = i;
      candidate_by_staging_[candidate.staging.get()] = i;
      definition_stmts_.insert(candidate.full_stmt.get());
      definition_stmts_.insert(candidate.assemble_stmt.get());
    }
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    auto target_it = candidate_by_target_.find(op.get());
    if (target_it != candidate_by_target_.end()) {
      (*candidates_)[target_it->second].safe = false;
    }
    auto staging_it = candidate_by_staging_.find(op.get());
    if (staging_it != candidate_by_staging_.end()) {
      (*candidates_)[staging_it->second].safe = false;
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (definition_stmts_.count(op.get()) > 0) return;
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    if (IsOp(op, "tensor.write") && op->args_.size() == 3) {
      auto target = AsVarLike(op->args_[0]);
      auto it = target ? candidate_by_target_.find(target.get()) : candidate_by_target_.end();
      if (it != candidate_by_target_.end()) {
        ++(*candidates_)[it->second].scalar_write_count;
        VisitExpr(op->args_[1]);
        VisitExpr(op->args_[2]);
        return;
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    for (const auto& value : op->value_) {
      auto var = AsVarLike(value);
      if (var && candidate_by_target_.count(var.get()) > 0) continue;
      VisitExpr(value);
    }
  }

 private:
  std::vector<FullTensorScalarUpdateCandidate>* candidates_;
  std::unordered_map<const Var*, size_t> candidate_by_target_;
  std::unordered_map<const Var*, size_t> candidate_by_staging_;
  std::unordered_set<const Stmt*> definition_stmts_;
};

class StagedScalarWriteRewriter : public IRMutator {
 public:
  explicit StagedScalarWriteRewriter(std::unordered_map<const Var*, VarPtr> staging_by_target)
      : staging_by_target_(std::move(staging_by_target)) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = As<Call>(IRMutator::VisitExpr_(op));
    if (!visited || !IsOp(visited, "tensor.write") || visited->args_.empty()) return visited;

    auto target = AsVarLike(visited->args_[0]);
    auto it = target ? staging_by_target_.find(target.get()) : staging_by_target_.end();
    if (it == staging_by_target_.end()) return visited;

    auto rewritten = MutableCopy(visited);
    rewritten->args_[0] = it->second;
    return rewritten;
  }

 private:
  std::unordered_map<const Var*, VarPtr> staging_by_target_;
};

/**
 * @brief Stage full-tensor initialization plus scalar overrides in one local tensor.
 *
 * Rewrites this safe subset:
 *
 *   staging = tensor.full(full_shape, value)
 *   updated = tensor.assemble(gm, staging, zeros)
 *   ... tensor.write(updated, dynamic_indices, value) ...
 *
 * into local writes followed by one final tensor.assemble. This preserves the
 * source semantics while ensuring all GM traffic uses MTE3.
 */
StmtPtr CanonicalizeFullTensorScalarUpdates(const StmtPtr& body, const std::vector<VarPtr>& params) {
  auto stmts = FlattenToStmts(body);
  if (stmts.size() < 3 || !As<ReturnStmt>(stmts.back())) return body;

  std::unordered_set<const Var*> param_set;
  for (const auto& param : params) {
    param_set.insert(param.get());
  }

  std::vector<FullTensorScalarUpdateCandidate> candidates;
  std::unordered_set<const Var*> used_targets;
  std::unordered_set<const Var*> used_staging;
  for (size_t i = 1; i + 1 < stmts.size(); ++i) {
    auto staging_assign = As<AssignStmt>(stmts[i - 1]);
    auto staging_call = staging_assign ? As<Call>(staging_assign->value_) : nullptr;
    auto assemble_stmt = As<AssignStmt>(stmts[i]);
    auto assemble = assemble_stmt ? As<Call>(assemble_stmt->value_) : nullptr;
    if (!staging_assign || !staging_call || !IsOp(staging_call, "tensor.full") || !assemble ||
        !IsOp(assemble, "tensor.assemble") || assemble->args_.size() != 3) {
      continue;
    }

    auto destination = AsVarLike(assemble->args_[0]);
    auto target = assemble_stmt->var_;
    auto staging = AsVarLike(assemble->args_[1]);
    auto target_type = target ? As<TensorType>(target->GetType()) : nullptr;
    auto destination_type = destination ? As<TensorType>(destination->GetType()) : nullptr;
    auto staging_type = staging ? As<TensorType>(staging->GetType()) : nullptr;
    if (!destination || !target || !staging || staging.get() != staging_assign->var_.get() ||
        param_set.count(destination.get()) == 0 || target.get() == staging.get() || !target_type ||
        !destination_type || !staging_type || target_type->dtype_ != destination_type->dtype_ ||
        target_type->dtype_ != staging_type->dtype_ || target_type->tensor_view_.has_value() ||
        destination_type->tensor_view_.has_value() || staging_type->tensor_view_.has_value() ||
        !AreExprVectorsEqual(target_type->shape_, destination_type->shape_) ||
        !AreExprVectorsEqual(target_type->shape_, staging_type->shape_) ||
        !IsZeroOffsetTuple(assemble->args_[2], target_type->shape_.size()) ||
        used_targets.count(target.get()) > 0 || used_staging.count(staging.get()) > 0) {
      continue;
    }

    candidates.push_back({i, target, staging, staging_assign, assemble_stmt});
    used_targets.insert(target.get());
    used_staging.insert(staging.get());
  }
  if (candidates.empty()) return body;

  StagedScalarUseAnalyzer analyzer(&candidates);
  analyzer.VisitStmt(body);

  std::vector<FullTensorScalarUpdateCandidate> accepted;
  for (const auto& candidate : candidates) {
    if (candidate.safe && candidate.scalar_write_count > 0) accepted.push_back(candidate);
  }
  if (accepted.empty()) return body;

  std::unordered_map<const Var*, VarPtr> staging_by_target;
  std::unordered_set<size_t> removed_assemble_indices;
  for (const auto& candidate : accepted) {
    staging_by_target[candidate.target.get()] = candidate.staging;
    removed_assemble_indices.insert(candidate.assemble_index);
  }
  StagedScalarWriteRewriter rewriter(std::move(staging_by_target));

  std::vector<StmtPtr> rewritten;
  rewritten.reserve(stmts.size());
  for (size_t i = 0; i + 1 < stmts.size(); ++i) {
    if (removed_assemble_indices.count(i) == 0) {
      rewritten.push_back(rewriter.VisitStmt(stmts[i]));
    }
  }
  for (const auto& candidate : accepted) {
    rewritten.push_back(candidate.assemble_stmt);
  }
  rewritten.push_back(rewriter.VisitStmt(stmts.back()));
  return SeqStmts::Flatten(std::move(rewritten), body->span_);
}

/**
 * @brief Rewrite a constant contiguous scalar-fill loop into one tensor.full + tensor.assemble.
 *
 * A scalar loop that writes every element of one contiguous tensor region with
 * the same constant does not need the D-cache path. Canonicalizing it before
 * tensor-to-tile conversion makes the normal tensor.full/tensor.assemble
 * lowering emit tile.full/tile.store instead. This is both faster and avoids a
 * false mixed-store rejection when another control-flow path stores the same GM
 * tensor through MTE3.
 */
class ConstantScalarFillLoopCanonicalizer : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto visited = As<ForStmt>(IRMutator::VisitStmt_(op));
    if (!visited) return op;

    auto rewrite = TryRewrite(visited);
    return rewrite ? rewrite : visited;
  }

 private:
  static bool IsVar(const ExprPtr& expr, const Var* expected) {
    auto var = AsVarLike(expr);
    return var && var.get() == expected;
  }

  static ExprPtr MatchUnitStrideIndex(const ExprPtr& index, const Var* loop_var, const Span& span) {
    if (IsVar(index, loop_var)) {
      return std::make_shared<ConstInt>(0, DataType::INDEX, span);
    }

    auto add = As<Add>(index);
    if (!add) return nullptr;
    if (IsVar(add->left_, loop_var) && !ExprUsesVar(add->right_, loop_var)) {
      return add->right_;
    }
    if (IsVar(add->right_, loop_var) && !ExprUsesVar(add->left_, loop_var)) {
      return add->left_;
    }
    return nullptr;
  }

  static StmtPtr TryRewrite(const ForStmtPtr& loop) {
    auto start = As<ConstInt>(loop->start_);
    auto stop = As<ConstInt>(loop->stop_);
    auto step = As<ConstInt>(loop->step_);
    if (!start || start->value_ != 0 || !stop || stop->value_ <= 0 || !step || step->value_ != 1 ||
        loop->kind_ != ForKind::Sequential || !loop->iter_args_.empty() || !loop->return_vars_.empty() ||
        !loop->attrs_.empty()) {
      return nullptr;
    }

    auto body_stmts = FlattenToStmts(loop->body_);
    if (body_stmts.empty()) return nullptr;

    std::vector<StmtPtr> rewritten;
    rewritten.reserve(body_stmts.size() * 3);
    std::vector<ExprPtr> reference_indices;
    std::vector<ExprPtr> reference_shape;
    std::optional<DataType> reference_dtype;
    bool reference_has_tensor_view = false;
    for (size_t stmt_index = 0; stmt_index < body_stmts.size(); ++stmt_index) {
      auto eval = As<EvalStmt>(body_stmts[stmt_index]);
      auto write = eval ? As<Call>(eval->expr_) : nullptr;
      if (!write || !IsOp(write, "tensor.write") || write->args_.size() != 3 ||
          (!As<ConstInt>(write->args_[2]) && !As<ConstFloat>(write->args_[2]))) {
        return nullptr;
      }

      auto target_type = As<TensorType>(write->args_[0]->GetType());
      auto indices = As<MakeTuple>(write->args_[1]);
      if (!target_type || !indices || indices->elements_.size() != target_type->shape_.size()) {
        return nullptr;
      }

      // Hoisting multiple scalar writes changes the iteration order from
      // (element, statement) to (statement, element). Keep that rewrite only
      // when every statement has the same contiguous access pattern and type;
      // then even aliased targets observe the same per-element statement order.
      if (stmt_index == 0) {
        reference_indices = indices->elements_;
        reference_shape = target_type->shape_;
        reference_dtype = target_type->dtype_;
        reference_has_tensor_view = target_type->tensor_view_.has_value();
      } else if (reference_has_tensor_view || target_type->tensor_view_.has_value() ||
                 target_type->dtype_ != *reference_dtype ||
                 !AreExprVectorsEqual(target_type->shape_, reference_shape) ||
                 !AreExprVectorsEqual(indices->elements_, reference_indices)) {
        return nullptr;
      }

      std::optional<size_t> varying_axis;
      std::vector<ExprPtr> offsets;
      offsets.reserve(indices->elements_.size());
      for (size_t i = 0; i < indices->elements_.size(); ++i) {
        const auto& index = indices->elements_[i];
        if (!ExprUsesVar(index, loop->loop_var_.get())) {
          offsets.push_back(index);
          continue;
        }
        if (varying_axis.has_value()) return nullptr;
        auto base = MatchUnitStrideIndex(index, loop->loop_var_.get(), write->span_);
        if (!base) return nullptr;
        varying_axis = i;
        offsets.push_back(base);
      }
      if (!varying_axis.has_value()) return nullptr;

      int64_t fill_bits = 0;
      const int64_t element_bits = static_cast<int64_t>(target_type->dtype_.GetBit());
      if (__builtin_mul_overflow(stop->value_, element_bits, &fill_bits) || fill_bits % 256 != 0) {
        return nullptr;
      }

      // A rectangular [1, ..., trip_count, ..., 1] tensor is contiguous only
      // when every physical dimension after the varying axis is a singleton.
      // Restrict the rewrite to that case rather than changing strided-write
      // semantics for a general tensor.
      for (size_t i = *varying_axis + 1; i < target_type->shape_.size(); ++i) {
        auto extent = As<ConstInt>(target_type->shape_[i]);
        auto offset = As<ConstInt>(offsets[i]);
        if (!extent || extent->value_ != 1 || !offset || offset->value_ != 0) {
          return nullptr;
        }
      }

      std::vector<ExprPtr> logical_shape;
      logical_shape.reserve(target_type->shape_.size());
      for (size_t i = 0; i < target_type->shape_.size(); ++i) {
        const int64_t extent = i == *varying_axis ? stop->value_ : 1;
        logical_shape.push_back(std::make_shared<ConstInt>(extent, DataType::INDEX, write->span_));
      }
      std::vector<ExprPtr> physical_shape = {
          std::make_shared<ConstInt>(1, DataType::INDEX, write->span_),
          std::make_shared<ConstInt>(stop->value_, DataType::INDEX, write->span_),
      };

      auto& op_registry = OpRegistry::GetInstance();
      std::vector<std::pair<std::string, std::any>> full_kwargs = {{"dtype", target_type->dtype_}};
      auto full =
          op_registry.Create("tensor.full", {MakeShapeTuple(physical_shape, write->span_), write->args_[2]},
                             full_kwargs, write->span_);

      auto target = AsVarLike(write->args_[0]);
      const std::string base_name =
          auto_name::GetBaseName(target ? target->name_hint_ : loop->loop_var_->name_hint_);
      auto full_var = std::make_shared<Var>(auto_name::BuildName(base_name, "", "scalar_fill_storage"),
                                            full->GetType(), write->span_);
      auto comments = eval->leading_comments_;
      if (stmt_index == 0) {
        comments.insert(comments.begin(), loop->leading_comments_.begin(), loop->leading_comments_.end());
      }
      rewritten.push_back(std::make_shared<AssignStmt>(full_var, full, write->span_, std::move(comments)));

      auto reshape = op_registry.Create(
          "tensor.reshape", {full_var, MakeShapeTuple(logical_shape, write->span_)}, write->span_);
      auto reshape_var = std::make_shared<Var>(auto_name::BuildName(base_name, "", "scalar_fill"),
                                               reshape->GetType(), write->span_);
      rewritten.push_back(std::make_shared<AssignStmt>(reshape_var, reshape, write->span_));

      auto assemble = op_registry.Create(
          "tensor.assemble",
          {write->args_[0], reshape_var, std::make_shared<MakeTuple>(offsets, write->span_)}, write->span_);
      rewritten.push_back(std::make_shared<EvalStmt>(assemble, write->span_));
    }
    return SeqStmts::Flatten(std::move(rewritten), loop->span_);
  }
};

// ============================================================================
// Param direction inference: analyze read/write patterns to upgrade In→Out/InOut
//
// Traces tensor alias chains through ForStmt/WhileStmt iter-args, IfStmt
// branches, tile.store, tensor.write, and tensor.assemble to determine
// which function parameters are written to.  Parameters that are written
// but not read become Out; those that are both read and written become InOut.
// ============================================================================

using ParamOrigins = std::vector<size_t>;
using AliasOriginMap = std::unordered_map<const Var*, ParamOrigins>;

struct YieldAliasInfo {
  bool has_yield = false;
  std::vector<ParamOrigins> origins;
};

void AddOrigin(ParamOrigins& origins, size_t index) {
  if (std::find(origins.begin(), origins.end(), index) == origins.end()) {
    origins.push_back(index);
  }
}

void MergeOrigins(ParamOrigins& dst, const ParamOrigins& src) {
  for (size_t index : src) {
    AddOrigin(dst, index);
  }
}

void MarkAccess(const ParamOrigins& origins, std::vector<bool>& flags) {
  for (size_t index : origins) {
    if (index < flags.size()) {
      flags[index] = true;
    }
  }
}

void RecordFirstStoreSpan(const ParamOrigins& origins, std::vector<std::optional<Span>>& spans,
                          const Span& span) {
  for (size_t index : origins) {
    if (index < spans.size() && !spans[index].has_value()) {
      spans[index].emplace(span);
    }
  }
}

ParamOrigins LookupOrigins(const Var* var, const AliasOriginMap& origin_map) {
  if (!var) return {};
  auto it = origin_map.find(var);
  if (it == origin_map.end()) return {};
  return it->second;
}

ParamOrigins CollectReferencedOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map);

/// The argument whose buffer this call's SSA result names, or null when the
/// result is a fresh value.
///
/// This is the destination-rebind idiom — `c2 = tile.store(t, off, c)`, where
/// reading `c2` reads `c` — and it is a *narrower* question than "which argument
/// does this operator write", which the registry now answers (`set_arg_effect`).
/// The two differ: `tile.mgather` clobbers a GM scratch operand yet returns a
/// fresh tile, so it writes an argument it does not alias.
///
/// An operator declaring `set_output_reuses_input(N)` states exactly this
/// relation, so that declaration is consulted first. The remaining entries are
/// the tensor-level and cross-rank operators whose result rebinds a destination
/// without reusing an on-chip MemRef; unifying them onto one declaration is
/// follow-up work, and the two sources are kept consistent by the write-effect
/// check below.
ExprPtr ResultAliasedDestination(const CallPtr& call) {
  if (auto index = ResultAliasedArgIndex(call)) return call->args_[*index];
  return nullptr;
}

void UpdateTensorAliasOrigin(const VarPtr& var, const ParamOrigins& origins, AliasOriginMap& origin_map) {
  if (AsTensorTypeLike(var->GetType()) && !origins.empty()) {
    origin_map[var.get()] = origins;
  } else {
    origin_map.erase(var.get());
  }
}

ParamOrigins GetAliasOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map) {
  if (!expr) return {};

  if (auto var = AsVarLike(expr)) {
    return LookupOrigins(var.get(), origin_map);
  }

  if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
    if (auto tuple = As<MakeTuple>(tuple_get->tuple_)) {
      if (tuple_get->index_ >= 0 && static_cast<size_t>(tuple_get->index_) < tuple->elements_.size()) {
        return GetAliasOrigins(tuple->elements_[static_cast<size_t>(tuple_get->index_)], origin_map);
      }
    }
    return {};
  }

  auto call = As<Call>(expr);
  if (!call) return {};

  if (auto write_target = ResultAliasedDestination(call)) {
    return GetAliasOrigins(write_target, origin_map);
  }
  if ((IsOp(call, "tensor.slice") || IsOp(call, "tensor.view")) && !call->args_.empty()) {
    return GetAliasOrigins(call->args_[0], origin_map);
  }
  return {};
}

ParamOrigins CollectReferencedOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map) {
  if (!expr) return {};

  if (auto var = AsVarLike(expr)) {
    return LookupOrigins(var.get(), origin_map);
  }

  if (auto tuple = As<MakeTuple>(expr)) {
    ParamOrigins origins;
    for (const auto& element : tuple->elements_) {
      MergeOrigins(origins, CollectReferencedOrigins(element, origin_map));
    }
    return origins;
  }

  if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
    if (auto tuple = As<MakeTuple>(tuple_get->tuple_)) {
      if (tuple_get->index_ >= 0 && static_cast<size_t>(tuple_get->index_) < tuple->elements_.size()) {
        return CollectReferencedOrigins(tuple->elements_[static_cast<size_t>(tuple_get->index_)], origin_map);
      }
    }
    return CollectReferencedOrigins(tuple_get->tuple_, origin_map);
  }

  auto call = As<Call>(expr);
  if (!call) return {};

  ParamOrigins origins;
  for (const auto& arg : call->args_) {
    MergeOrigins(origins, CollectReferencedOrigins(arg, origin_map));
  }
  return origins;
}

/// Mark the parameter origins each argument of @p call reads and writes.
///
/// Which argument a call writes is a property of the operator, declared once on
/// the registry (`set_arg_effect`) and read here — this pass used to carry its
/// own table of twenty operators, whose default arm counted every argument of an
/// operator it did not recognise as a read. That default is why a GM tensor
/// written only by `tile.mscatter` kept direction `In`.
///
/// Reads resolve through `CollectReferencedOrigins`, since an operand may merely
/// *mention* a buffer (an offsets tuple built from `tensor.dim`). Writes resolve
/// through `GetAliasOrigins`, since a destination operand names the buffer
/// itself. A pure `Write` argument is not marked as a read: a store that lands
/// on a sub-region never reads the untouched remainder.
void AnalyzeCallAccess(const CallPtr& call, const AliasOriginMap& origin_map, std::vector<bool>& has_read,
                       std::vector<bool>& has_write, std::vector<std::optional<Span>>& dma_store_spans,
                       std::vector<std::optional<Span>>& scalar_store_spans) {
  if (!call) return;

  // A call to a user function reaches here with a GlobalVar callee and no
  // registry entry. Its writes are propagated separately, from the callee's
  // declared param directions; every operand counts as a read here.
  const auto* entry = LookupOpEntry(call->op_);

  for (size_t i = 0; i < call->args_.size(); ++i) {
    const auto effect = entry ? entry->GetArgEffect(i, call->kwargs_) : ArgEffect::Read;

    if (ArgEffectReads(effect)) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    if (!ArgEffectWrites(effect)) continue;

    auto origins = GetAliasOrigins(call->args_[i], origin_map);
    MarkAccess(origins, has_write);
    // Only a GM store participates in the mixed-channel diagnostic below; an
    // operator that writes a tile, an array or a signal slot declares no
    // channel and is skipped.
    if (auto channel = entry->GetWriteChannel()) {
      RecordFirstStoreSpan(origins, *channel == WriteChannel::Scalar ? scalar_store_spans : dma_store_spans,
                           call->span_);
    }
  }
}

YieldAliasInfo MergeYieldInfos(const YieldAliasInfo& lhs, const YieldAliasInfo& rhs) {
  if (!lhs.has_yield) return rhs;
  if (!rhs.has_yield) return lhs;

  YieldAliasInfo merged;
  merged.has_yield = true;
  size_t count = std::max(lhs.origins.size(), rhs.origins.size());
  merged.origins.resize(count);
  for (size_t i = 0; i < lhs.origins.size(); ++i) {
    MergeOrigins(merged.origins[i], lhs.origins[i]);
  }
  for (size_t i = 0; i < rhs.origins.size(); ++i) {
    MergeOrigins(merged.origins[i], rhs.origins[i]);
  }
  return merged;
}

YieldAliasInfo AnalyzeStmtAliases(const StmtPtr& stmt, AliasOriginMap& origin_map,
                                  std::vector<bool>& has_read, std::vector<bool>& has_write,
                                  std::vector<std::optional<Span>>& dma_store_spans,
                                  std::vector<std::optional<Span>>& scalar_store_spans);

void BindLoopOrigins(const std::vector<IterArgPtr>& iter_args, const std::vector<ParamOrigins>& origins,
                     AliasOriginMap& origin_map) {
  for (size_t i = 0; i < iter_args.size(); ++i) {
    if (i < origins.size() && !origins[i].empty()) {
      origin_map[iter_args[i].get()] = origins[i];
    } else {
      origin_map.erase(iter_args[i].get());
    }
  }
}

std::vector<ParamOrigins> AnalyzeLoopCarriedOrigins(const StmtPtr& body,
                                                    const std::vector<IterArgPtr>& iter_args,
                                                    const AliasOriginMap& origin_map,
                                                    std::vector<bool>& has_read, std::vector<bool>& has_write,
                                                    std::vector<std::optional<Span>>& dma_store_spans,
                                                    std::vector<std::optional<Span>>& scalar_store_spans) {
  std::vector<ParamOrigins> carried_origins(iter_args.size());
  for (size_t i = 0; i < iter_args.size(); ++i) {
    carried_origins[i] = GetAliasOrigins(iter_args[i]->initValue_, origin_map);
  }

  auto body_map = origin_map;
  BindLoopOrigins(iter_args, carried_origins, body_map);
  auto yield_info =
      AnalyzeStmtAliases(body, body_map, has_read, has_write, dma_store_spans, scalar_store_spans);

  bool origins_widened = false;
  if (yield_info.has_yield) {
    for (size_t i = 0; i < carried_origins.size() && i < yield_info.origins.size(); ++i) {
      size_t previous_size = carried_origins[i].size();
      MergeOrigins(carried_origins[i], yield_info.origins[i]);
      origins_widened |= carried_origins[i].size() != previous_size;
    }
  }

  // Alias transfers only preserve or union parameter origins. One widened
  // rescan therefore covers stores in later iterations without an unbounded
  // fixed-point traversal of the loop body.
  if (origins_widened) {
    body_map = origin_map;
    BindLoopOrigins(iter_args, carried_origins, body_map);
    yield_info = AnalyzeStmtAliases(body, body_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    if (yield_info.has_yield) {
      for (size_t i = 0; i < carried_origins.size() && i < yield_info.origins.size(); ++i) {
        MergeOrigins(carried_origins[i], yield_info.origins[i]);
      }
    }
  }
  return carried_origins;
}

YieldAliasInfo AnalyzeStmtSequenceAliases(const std::vector<StmtPtr>& stmts, AliasOriginMap& origin_map,
                                          std::vector<bool>& has_read, std::vector<bool>& has_write,
                                          std::vector<std::optional<Span>>& dma_store_spans,
                                          std::vector<std::optional<Span>>& scalar_store_spans) {
  YieldAliasInfo last_yield;
  for (const auto& stmt : stmts) {
    auto yield_info =
        AnalyzeStmtAliases(stmt, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    if (yield_info.has_yield) {
      last_yield = yield_info;
    }
  }
  return last_yield;
}

YieldAliasInfo AnalyzeStmtAliases(const StmtPtr& stmt, AliasOriginMap& origin_map,
                                  std::vector<bool>& has_read, std::vector<bool>& has_write,
                                  std::vector<std::optional<Span>>& dma_store_spans,
                                  std::vector<std::optional<Span>>& scalar_store_spans) {
  if (!stmt) return {};

  if (auto assign = As<AssignStmt>(stmt)) {
    if (auto call = As<Call>(assign->value_)) {
      AnalyzeCallAccess(call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }

    if (AsTensorTypeLike(assign->var_->GetType())) {
      auto origins = GetAliasOrigins(assign->value_, origin_map);
      UpdateTensorAliasOrigin(assign->var_, origins, origin_map);
    }
    return {};
  }

  if (auto eval = As<EvalStmt>(stmt)) {
    if (auto call = As<Call>(eval->expr_)) {
      AnalyzeCallAccess(call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    return {};
  }

  if (auto seq = As<SeqStmts>(stmt)) {
    return AnalyzeStmtSequenceAliases(seq->stmts_, origin_map, has_read, has_write, dma_store_spans,
                                      scalar_store_spans);
  }

  if (auto scope = As<ScopeStmt>(stmt)) {
    return AnalyzeStmtAliases(scope->body_, origin_map, has_read, has_write, dma_store_spans,
                              scalar_store_spans);
  }

  if (auto if_stmt = As<IfStmt>(stmt)) {
    if (auto cond_call = As<Call>(if_stmt->condition_)) {
      AnalyzeCallAccess(cond_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    auto then_map = origin_map;
    auto then_yield = AnalyzeStmtAliases(if_stmt->then_body_, then_map, has_read, has_write, dma_store_spans,
                                         scalar_store_spans);
    YieldAliasInfo else_yield;
    if (auto else_body = if_stmt->else_body_.value_or(nullptr)) {
      auto else_map = origin_map;
      else_yield =
          AnalyzeStmtAliases(else_body, else_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    auto merged_yield = MergeYieldInfos(then_yield, else_yield);
    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins;
      if (merged_yield.has_yield && i < merged_yield.origins.size()) {
        origins = merged_yield.origins[i];
      }
      UpdateTensorAliasOrigin(if_stmt->return_vars_[i], origins, origin_map);
    }
    return merged_yield;
  }

  if (auto for_stmt = As<ForStmt>(stmt)) {
    if (auto start_call = As<Call>(for_stmt->start_)) {
      AnalyzeCallAccess(start_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    if (auto stop_call = As<Call>(for_stmt->stop_)) {
      AnalyzeCallAccess(stop_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    if (auto step_call = As<Call>(for_stmt->step_)) {
      AnalyzeCallAccess(step_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }

    auto carried_origins =
        AnalyzeLoopCarriedOrigins(for_stmt->body_, for_stmt->iter_args_, origin_map, has_read, has_write,
                                  dma_store_spans, scalar_store_spans);
    for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins = i < carried_origins.size() ? carried_origins[i] : ParamOrigins{};
      UpdateTensorAliasOrigin(for_stmt->return_vars_[i], origins, origin_map);
    }
    return {};
  }

  if (auto while_stmt = As<WhileStmt>(stmt)) {
    if (auto cond_call = As<Call>(while_stmt->condition_)) {
      AnalyzeCallAccess(cond_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }

    auto carried_origins =
        AnalyzeLoopCarriedOrigins(while_stmt->body_, while_stmt->iter_args_, origin_map, has_read, has_write,
                                  dma_store_spans, scalar_store_spans);
    for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins = i < carried_origins.size() ? carried_origins[i] : ParamOrigins{};
      UpdateTensorAliasOrigin(while_stmt->return_vars_[i], origins, origin_map);
    }
    return {};
  }

  if (auto yield = As<YieldStmt>(stmt)) {
    YieldAliasInfo info;
    info.has_yield = true;
    info.origins.reserve(yield->value_.size());
    for (const auto& value : yield->value_) {
      info.origins.push_back(GetAliasOrigins(value, origin_map));
    }
    return info;
  }

  return {};
}

struct ParamAccessInfo {
  explicit ParamAccessInfo(size_t count)
      : has_read(count, false), has_write(count, false), dma_store_spans(count), scalar_store_spans(count) {}

  [[nodiscard]] bool HasMixedStores() const {
    for (size_t i = 0; i < dma_store_spans.size(); ++i) {
      if (dma_store_spans[i].has_value() && scalar_store_spans[i].has_value()) return true;
    }
    return false;
  }

  std::vector<bool> has_read;
  std::vector<bool> has_write;
  std::vector<std::optional<Span>> dma_store_spans;
  std::vector<std::optional<Span>> scalar_store_spans;
};

ParamAccessInfo AnalyzeParamAccess(const std::vector<StmtPtr>& stmts, const std::vector<VarPtr>& params) {
  ParamAccessInfo access(params.size());
  AliasOriginMap origin_map;
  for (size_t i = 0; i < params.size(); ++i) {
    // Include distributed windows, which also name GM storage.
    if (AsTensorTypeLike(params[i]->GetType())) origin_map[params[i].get()] = ParamOrigins{i};
  }
  AnalyzeStmtSequenceAliases(stmts, origin_map, access.has_read, access.has_write, access.dma_store_spans,
                             access.scalar_store_spans);
  return access;
}

}  // namespace

/// Upgrade In params to Out/InOut based on tile.store/tensor.write usage analysis.
void UpgradeWrittenTensorParamDirections(const std::vector<StmtPtr>& stmts, const std::vector<VarPtr>& params,
                                         std::vector<ParamDirection>& param_directions) {
  const auto access = AnalyzeParamAccess(stmts, params);
  for (size_t i = 0; i < params.size() && i < param_directions.size(); ++i) {
    if (access.dma_store_spans[i].has_value() && access.scalar_store_spans[i].has_value()) {
      CHECK_SPAN(false, access.scalar_store_spans[i].value_or(Span::unknown()))
          << "GM tensor '" << params[i]->name_hint_
          << "' mixes MTE3 and scalar stores in one InCore function. tile.store/tensor.assemble "
             "uses the MTE3 path while tensor.write uses the scalar D-cache path; PyPTO cannot "
             "guarantee ordering or cache-line coherence between them. Rewrite the updates through "
             "one UB tile (use tile.write, then one tile.store), or use tensor.write for every GM "
             "element written to this tensor.";
    }
    if (param_directions[i] != ParamDirection::In || !access.has_write[i]) {
      continue;
    }
    param_directions[i] = access.has_read[i] ? ParamDirection::InOut : ParamDirection::Out;
  }
}

StmtPtr CanonicalizeScalarUpdates(const FunctionPtr& func) {
  auto staged_body = CanonicalizeFullTensorScalarUpdates(func->body_, func->params_);
  ConstantScalarFillLoopCanonicalizer scalar_fill_canonicalizer;
  auto canonical_body = scalar_fill_canonicalizer.VisitStmt(staged_body);
  // A bulk fill must not introduce a second GM store channel beside scalar
  // updates that cannot be promoted. Reuse the alias-aware access analysis and
  // retain the original scalar loops when the tentative rewrite would mix
  // channels. Existing explicit mixed stores still fail the check below.
  if (canonical_body != staged_body &&
      AnalyzeParamAccess(FlattenToStmts(canonical_body), func->params_).HasMixedStores()) {
    canonical_body = staged_body;
  }

  return canonical_body;
}

}  // namespace tensor_to_tile
}  // namespace ir
}  // namespace pypto
