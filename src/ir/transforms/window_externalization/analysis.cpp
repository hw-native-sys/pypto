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
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/window_externalization/internal.h"

namespace pypto {
namespace ir {
namespace window_externalization {
using transform_utils::FlattenToStmts;

namespace {

std::unordered_set<const Var*> CollectAllowedVars(const std::vector<VarPtr>& vars,
                                                  const Var* extra_allowed = nullptr) {
  std::unordered_set<const Var*> allowed;
  allowed.reserve(vars.size() + (extra_allowed ? 1 : 0));
  for (const auto& var : vars) {
    if (var) allowed.insert(var.get());
  }
  if (extra_allowed) allowed.insert(extra_allowed);
  return allowed;
}

bool ExprsReferenceOnlyVarsIn(const std::vector<ExprPtr>& exprs,
                              const std::unordered_set<const Var*>& allowed) {
  for (const auto& expr : exprs) {
    if (!ExprReferencesOnlyVarsIn(expr, allowed)) return false;
  }
  return true;
}

bool IsAllZeroOffsets(const std::vector<ExprPtr>& offsets) {
  for (const auto& offset : offsets) {
    auto ci = As<ConstInt>(offset);
    if (!ci || ci->value_ != 0) return false;
  }
  return true;
}

bool IsOutputDirection(ParamDirection direction, bool include_inout) {
  return direction == ParamDirection::Out || (include_inout && direction == ParamDirection::InOut);
}

std::vector<size_t> CollectOutParamIndices(const FunctionPtr& func) {
  std::vector<size_t> result;
  if (!func) return result;
  for (size_t i = 0; i < func->param_directions_.size() && i < func->params_.size(); ++i) {
    if (IsOutputDirection(func->param_directions_[i], /*include_inout=*/true)) {
      result.push_back(i);
    }
  }
  return result;
}

// ============================================================================
// Pattern 5: Static Out-window externalization
//
// Rewrites statically provable local-window writes into explicit
// slice -> windowed callee -> assemble structure at the orchestration callsite.
//
// Supported shapes:
// - FinalStore: single call writes one final local window of an Out param
// - AggregateWindowLoop: an outlined non-builtin callee writes a loop-carried
//   aggregate window into one or more Out params.
//
// Multi-Out policy is per-output and conservative: each Out param is rewritten
// only when its own read/write footprint can be proven as one or more dense
// pieces representable with the existing tensor.slice/tensor.assemble runtime
// views. Unproven Out params stay as baseline full-tensor args/results.
// ============================================================================

struct DenseRect {
  std::vector<ExprPtr> offsets;
  std::vector<ExprPtr> shape;
};

struct VarUseIndex {
  std::unordered_map<const Var*, size_t> counts;
  std::unordered_map<const Var*, std::vector<const AssignStmt*>> assign_users;
};

DenseRegionPiece MakeDensePiece(std::vector<ExprPtr> window_shape, std::vector<ExprPtr> callsite_offsets,
                                std::vector<ExprPtr> local_offsets) {
  return DenseRegionPiece{std::move(window_shape), std::move(callsite_offsets), std::move(local_offsets)};
}

bool DenseRectsAreDisjoint(const std::vector<ExprPtr>& lhs_offsets, const std::vector<ExprPtr>& lhs_shape,
                           const std::vector<ExprPtr>& rhs_offsets, const std::vector<ExprPtr>& rhs_shape) {
  if (lhs_offsets.size() != rhs_offsets.size() || lhs_shape.size() != rhs_shape.size() ||
      lhs_offsets.size() != lhs_shape.size()) {
    return false;
  }
  arith::Analyzer analyzer;
  for (size_t dim = 0; dim < lhs_offsets.size(); ++dim) {
    auto lhs_dim = As<ConstInt>(lhs_shape[dim]);
    auto rhs_dim = As<ConstInt>(rhs_shape[dim]);
    if (!lhs_dim || !rhs_dim) return false;
    auto diff = analyzer.Simplify(MakeSub(rhs_offsets[dim], lhs_offsets[dim], rhs_offsets[dim]->span_));
    auto diff_ci = As<ConstInt>(diff);
    if (!diff_ci) continue;
    // Tensor windows are half-open intervals, so touching boundaries are disjoint.
    if (diff_ci->value_ >= lhs_dim->value_ || -diff_ci->value_ >= rhs_dim->value_) {
      return true;
    }
  }
  return false;
}

struct OrderedLoopOffsets {
  ExprPtr min;
  ExprPtr max;
};

VarUseIndex BuildVarUseIndex(const StmtPtr& stmt) {
  class Collector : public IRVisitor {
   public:
    [[nodiscard]] VarUseIndex TakeIndex() { return std::move(index_); }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      const AssignStmt* saved_assign = current_assign_;
      current_assign_ = op.get();
      IRVisitor::VisitStmt_(op);
      current_assign_ = saved_assign;
    }

    void VisitExpr_(const VarPtr& op) override {
      Record(op.get());
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const IterArgPtr& op) override {
      Record(op.get());
      IRVisitor::VisitExpr_(op);
    }

   private:
    void Record(const Var* var) {
      ++index_.counts[var];
      if (current_assign_) index_.assign_users[var].push_back(current_assign_);
    }

    VarUseIndex index_;
    const AssignStmt* current_assign_ = nullptr;
  };

  Collector collector;
  collector.VisitStmt(stmt);
  return collector.TakeIndex();
}

uint64_t HashExprVector(const std::vector<ExprPtr>& exprs) {
  uint64_t hash = exprs.size();
  for (const auto& expr : exprs) {
    const uint64_t value = expr ? structural_hash(expr) : 0;
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  }
  return hash;
}

uint64_t HashAccessRegion(const std::vector<ExprPtr>& shape, const std::vector<ExprPtr>& offsets) {
  uint64_t hash = HashExprVector(shape);
  const uint64_t offset_hash = HashExprVector(offsets);
  hash ^= offset_hash + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

std::optional<int64_t> CheckedShapeVolume(const std::vector<ExprPtr>& shape) {
  int64_t volume = 1;
  for (const auto& dim : shape) {
    auto value = As<ConstInt>(dim);
    if (!value || value->value_ <= 0) return std::nullopt;
    auto next = CheckedMul(volume, value->value_);
    if (!next.has_value()) return std::nullopt;
    volume = *next;
  }
  return volume;
}

std::optional<std::pair<int64_t, int64_t>> DenseRectToLinearInterval(
    const DenseRect& rect, const std::vector<ExprPtr>& bounds_offsets,
    const std::vector<ExprPtr>& bounds_shape) {
  const size_t rank = bounds_shape.size();
  if (rank == 0 || rect.offsets.size() != rank || rect.shape.size() != rank ||
      bounds_offsets.size() != rank) {
    return std::nullopt;
  }

  std::vector<int64_t> strides(rank, 1);
  for (size_t dim = rank; dim-- > 1;) {
    auto bound = As<ConstInt>(bounds_shape[dim]);
    if (!bound || bound->value_ <= 0) return std::nullopt;
    auto stride = CheckedMul(strides[dim], bound->value_);
    if (!stride.has_value()) return std::nullopt;
    strides[dim - 1] = *stride;
  }

  arith::Analyzer analyzer;
  int64_t linear_start = 0;
  int64_t linear_last = 0;
  for (size_t dim = 0; dim < rank; ++dim) {
    auto bound = As<ConstInt>(bounds_shape[dim]);
    auto extent = As<ConstInt>(rect.shape[dim]);
    auto relative =
        analyzer.Simplify(MakeSub(rect.offsets[dim], bounds_offsets[dim], rect.offsets[dim]->span_));
    auto relative_value = As<ConstInt>(relative);
    if (!bound || !extent || !relative_value || bound->value_ <= 0 || extent->value_ <= 0 ||
        relative_value->value_ < 0) {
      return std::nullopt;
    }
    auto rect_end = CheckedAdd(relative_value->value_, extent->value_);
    if (!rect_end.has_value() || *rect_end > bound->value_) return std::nullopt;

    auto start_term = CheckedMul(relative_value->value_, strides[dim]);
    auto last_term = CheckedMul(*rect_end - 1, strides[dim]);
    if (!start_term.has_value() || !last_term.has_value()) return std::nullopt;
    auto next_start = CheckedAdd(linear_start, *start_term);
    auto next_last = CheckedAdd(linear_last, *last_term);
    if (!next_start.has_value() || !next_last.has_value()) return std::nullopt;
    linear_start = *next_start;
    linear_last = *next_last;
  }

  auto volume = CheckedShapeVolume(rect.shape);
  auto linear_end = volume.has_value() ? CheckedAdd(linear_start, *volume) : std::nullopt;
  auto contiguous_size = CheckedSub(linear_last, linear_start);
  if (!linear_end.has_value() || !contiguous_size.has_value()) return std::nullopt;
  contiguous_size = CheckedAdd(*contiguous_size, 1);
  if (!contiguous_size.has_value() || *contiguous_size != *volume) return std::nullopt;
  return std::make_pair(linear_start, *linear_end);
}

bool DenseRectsExactlyCoverBounds(const std::vector<DenseRect>& rects,
                                  const std::vector<ExprPtr>& bounds_offsets,
                                  const std::vector<ExprPtr>& bounds_shape) {
  auto bounds_volume = CheckedShapeVolume(bounds_shape);
  if (rects.empty() || !bounds_volume.has_value()) return false;

  std::vector<std::pair<int64_t, int64_t>> intervals;
  intervals.reserve(rects.size());
  for (const auto& rect : rects) {
    auto interval = DenseRectToLinearInterval(rect, bounds_offsets, bounds_shape);
    if (!interval.has_value()) return false;
    intervals.push_back(*interval);
  }
  std::sort(intervals.begin(), intervals.end());

  int64_t covered_end = 0;
  for (const auto& [start, end] : intervals) {
    if (start != covered_end || end <= start) return false;
    covered_end = end;
  }
  return covered_end == *bounds_volume;
}

std::optional<int64_t> GetConstantSpanValue(const ExprPtr& max_extent, const ExprPtr& min_offset,
                                            const Span& span) {
  arith::Analyzer analyzer;
  auto span_expr = analyzer.Simplify(MakeSub(max_extent, min_offset, span));
  if (auto span_ci = As<ConstInt>(span_expr)) return span_ci->value_;
  return ConstantDiffIfSameLinearBase(max_extent, min_offset);
}

std::optional<ExprPtr> SelectMinExpr(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
  if (!lhs) return rhs;
  if (!rhs) return lhs;
  if (AreExprsEqual(lhs, rhs)) return lhs;

  arith::Analyzer analyzer;
  auto diff = analyzer.Simplify(MakeSub(lhs, rhs, span));
  auto diff_ci = As<ConstInt>(diff);
  if (diff_ci) return diff_ci->value_ <= 0 ? lhs : rhs;
  auto linear_diff = ConstantDiffIfSameLinearBase(lhs, rhs);
  if (!linear_diff.has_value()) return std::nullopt;
  return *linear_diff <= 0 ? lhs : rhs;
}

std::optional<ExprPtr> SelectMaxExpr(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
  if (!lhs) return rhs;
  if (!rhs) return lhs;
  if (AreExprsEqual(lhs, rhs)) return lhs;

  arith::Analyzer analyzer;
  auto diff = analyzer.Simplify(MakeSub(lhs, rhs, span));
  auto diff_ci = As<ConstInt>(diff);
  if (diff_ci) return diff_ci->value_ >= 0 ? lhs : rhs;
  auto linear_diff = ConstantDiffIfSameLinearBase(lhs, rhs);
  if (!linear_diff.has_value()) return std::nullopt;
  return *linear_diff >= 0 ? lhs : rhs;
}

struct FinalStoreInfo {
  size_t return_index;
  std::vector<ExprPtr> window_shape;
  std::vector<ExprPtr> offsets;
};

struct AggregateWindowInfo {
  size_t return_index;
  std::vector<ExprPtr> window_shape;
  std::vector<ExprPtr> base_offsets;
  std::vector<ExprPtr> local_offsets;
  size_t iter_arg_index;
};

struct InputWindowUse {
  std::vector<ExprPtr> window_shape;
  std::vector<ExprPtr> offsets;
  size_t param_refs_in_stmt = 0;
};

struct InputParamUseSummary {
  size_t total_refs = 0;
  bool unsupported_ref = false;
  std::vector<InputWindowUse> uses;
};

bool CanMaterializeWindowParamType(const std::shared_ptr<const TensorType>& tensor_type,
                                   const std::vector<ExprPtr>& window_shape) {
  if (!tensor_type) return false;
  auto window_type = MakeWindowTensorType(tensor_type, tensor_type->shape_, window_shape);
  if (!window_type) return false;
  auto allowed_vars =
      var_collectors::CollectTypeVars(std::make_shared<TensorType>(window_shape, tensor_type->dtype_));
  auto window_tensor_type = As<TensorType>(window_type);
  if (!window_tensor_type) return false;
  if (!ExprsReferenceOnlyVarsIn(window_tensor_type->shape_, allowed_vars)) return false;
  if (window_tensor_type->tensor_view_.has_value()) {
    const auto& view = *window_tensor_type->tensor_view_;
    if (!ExprsReferenceOnlyVarsIn(view.stride, allowed_vars)) return false;
    if (!ExprsReferenceOnlyVarsIn(view.valid_shape, allowed_vars)) return false;
  }
  return true;
}

bool CanMaterializeOutputWindowParamType(const std::shared_ptr<const TensorType>& tensor_type,
                                         const std::vector<ExprPtr>& window_shape) {
  if (!tensor_type) return false;
  auto window_type = MakeWindowTensorType(tensor_type, tensor_type->shape_, window_shape);
  if (!window_type) return false;
  auto allowed_vars = var_collectors::CollectTypeVars(tensor_type);
  auto window_vars =
      var_collectors::CollectTypeVars(std::make_shared<TensorType>(window_shape, tensor_type->dtype_));
  allowed_vars.insert(window_vars.begin(), window_vars.end());
  auto window_tensor_type = As<TensorType>(window_type);
  if (!window_tensor_type) return false;
  if (!ExprsReferenceOnlyVarsIn(window_tensor_type->shape_, allowed_vars)) return false;
  if (window_tensor_type->tensor_view_.has_value()) {
    const auto& view = *window_tensor_type->tensor_view_;
    if (!ExprsReferenceOnlyVarsIn(view.stride, allowed_vars)) return false;
    if (!ExprsReferenceOnlyVarsIn(view.valid_shape, allowed_vars)) return false;
  }
  return true;
}

bool CanWindowOutputWithinDynamicParent(const std::shared_ptr<const TensorType>& tensor_type,
                                        const std::vector<ExprPtr>& window_shape,
                                        const std::vector<ExprPtr>& offsets) {
  if (!tensor_type || tensor_type->shape_.size() != window_shape.size() ||
      tensor_type->shape_.size() != offsets.size()) {
    return false;
  }

  for (size_t dim = 0; dim < tensor_type->shape_.size(); ++dim) {
    if (As<ConstInt>(tensor_type->shape_[dim])) continue;
    auto offset = As<ConstInt>(offsets[dim]);
    if (offset && As<ConstInt>(window_shape[dim]) &&
        !AreExprsEqual(window_shape[dim], tensor_type->shape_[dim])) {
      return false;
    }
  }
  return true;
}

std::optional<size_t> FindReturnIndexForOutParam(const FunctionPtr& func, size_t out_param_index) {
  if (!func || out_param_index >= func->params_.size()) return std::nullopt;
  auto body_stmts = FlattenToStmts(func->body_);
  ReturnStmtPtr ret_stmt;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      ret_stmt = ret;
      break;
    }
  }
  if (!ret_stmt) return std::nullopt;

  const auto* out_param = func->params_[out_param_index].get();
  for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
    auto ret_var = AsVarLike(ret_stmt->value_[ret_i]);
    if (!ret_var) continue;
    if (ret_var.get() == out_param) return ret_i;
  }
  return std::nullopt;
}

std::optional<OrderedLoopOffsets> GetOrderedLoopOffsets(const ExprPtr& expr, const ForStmtPtr& loop,
                                                        const ExprPtr& first_loop_value,
                                                        const ExprPtr& last_loop_value) {
  if (!expr || !loop || !first_loop_value || !last_loop_value) return std::nullopt;
  auto first_offset = SimplifyWithLoopValue(expr, loop->loop_var_, first_loop_value);
  auto last_offset = SimplifyWithLoopValue(expr, loop->loop_var_, last_loop_value);
  if (!first_offset.has_value() || !last_offset.has_value()) return std::nullopt;

  auto affine = ParseAffineInLoop(expr, loop->loop_var_.get());
  auto loop_step = transform_utils::EvalConstInt(loop->step_);
  if (!affine.has_value() || !loop_step.has_value()) return std::nullopt;
  // Only the *sign* of `coeff * step` decides the order, and the operand signs
  // decide the sign on their own -- so read it off them instead of evaluating a
  // product that can overflow int64 (`ParseAffineInLoop` can return INT64_MAX,
  // and a one-trip loop with step 2 still reaches here). Exactly equivalent to
  // `coeff * step >= 0` for every representable pair, so no path changes.
  const bool ascending = affine->coeff == 0 || *loop_step == 0 || (affine->coeff > 0) == (*loop_step > 0);
  if (ascending) {
    return OrderedLoopOffsets{*first_offset, *last_offset};
  }
  return OrderedLoopOffsets{*last_offset, *first_offset};
}

std::optional<ExprPtr> ExpandLoopLocalExpr(const ExprPtr& expr,
                                           const std::unordered_map<const Var*, ExprPtr>& scalar_defs) {
  if (!expr) return std::nullopt;
  return transform_utils::Substitute(expr, scalar_defs);
}

struct FixedTileLoadAccess {
  std::vector<ExprPtr> window_shape;
  MakeTuplePtr offsets;
};

std::optional<FixedTileLoadAccess> MatchFixedTileLoadAccess(const CallPtr& call, const Var* param) {
  if (!call || !param || !IsOp(call, "tile.load") || call->args_.size() < 3) return std::nullopt;

  auto parent = AsVarLike(call->args_[0]);
  auto offsets = As<MakeTuple>(call->args_[1]);
  auto tile_type = As<TileType>(call->GetType());
  auto read_shape = As<MakeTuple>(call->args_[2]);
  if (!parent || parent.get() != param || !offsets || !tile_type || !read_shape) return std::nullopt;

  if (call->args_.size() >= 4) {
    auto valid_shape = As<MakeTuple>(call->args_[3]);
    if (!valid_shape || !AreExprVectorsEqual(valid_shape->elements_, read_shape->elements_)) {
      return std::nullopt;
    }
  }

  std::vector<ExprPtr> window_shape;
  if (call->GetKwarg<bool>("transpose", false)) {
    if (read_shape->elements_.size() != 2) return std::nullopt;
    window_shape = read_shape->elements_;
  } else {
    window_shape = tile_type->shape_;
    if (!AreExprVectorsEqual(window_shape, read_shape->elements_)) return std::nullopt;
  }
  return FixedTileLoadAccess{std::move(window_shape), offsets};
}

std::optional<InputWindowUse> MatchDirectTensorWindowAccess(const AssignStmtPtr& assign, const Var* param) {
  if (!assign || !param) return std::nullopt;
  auto call = As<Call>(assign->value_);
  if (!call || call->args_.empty()) return std::nullopt;

  std::vector<ExprPtr> window_shape;
  MakeTuplePtr offsets;
  if (IsOp(call, "tile.load") && call->args_.size() >= 3) {
    auto access = MatchFixedTileLoadAccess(call, param);
    if (!access.has_value()) return std::nullopt;
    window_shape = access->window_shape;
    offsets = access->offsets;
  } else if (IsOp(call, "tensor.slice") && call->args_.size() >= 3) {
    auto parent = AsVarLike(call->args_[0]);
    offsets = As<MakeTuple>(call->args_[2]);
    auto tensor_type = As<TensorType>(call->GetType());
    if (!parent || parent.get() != param || !offsets || !tensor_type) return std::nullopt;
    window_shape = tensor_type->shape_;
  } else {
    return std::nullopt;
  }

  if (window_shape.size() != offsets->elements_.size()) return std::nullopt;
  size_t refs = CountVarRefsInStmt(assign, param);
  if (refs == 0) return std::nullopt;
  return InputWindowUse{std::move(window_shape), offsets->elements_, refs};
}

bool IsProvenSameRegionInOutAccess(const FunctionPtr& func, size_t out_param_index,
                                   const AssignStmtPtr& store_assign, const std::vector<ExprPtr>& store_shape,
                                   const std::vector<ExprPtr>& store_offsets,
                                   const ReturnStmtPtr& ret_stmt = nullptr) {
  if (!func || out_param_index >= func->params_.size() || out_param_index >= func->param_directions_.size() ||
      func->param_directions_[out_param_index] != ParamDirection::InOut) {
    return false;
  }
  const auto* param = func->params_[out_param_index].get();
  size_t total_refs = CountVarRefsInStmt(func->body_, param);
  size_t matched_refs = store_assign ? CountVarRefsInStmt(store_assign, param) : 0;
  if (total_refs == 0 || matched_refs == 0 || matched_refs > total_refs) return false;

  auto body_stmts = FlattenToStmts(func->body_);
  for (const auto& stmt : body_stmts) {
    auto assign = As<AssignStmt>(stmt);
    size_t refs = CountVarRefsInStmt(stmt, param);
    if (refs == 0) continue;
    if (assign && store_assign && assign.get() == store_assign.get()) continue;
    if (ret_stmt && stmt.get() == ret_stmt.get()) {
      matched_refs += refs;
      continue;
    }

    auto use = MatchDirectTensorWindowAccess(assign, param);
    if (!use.has_value()) return false;
    if (!AreExprVectorsEqual(use->window_shape, store_shape) ||
        !AreExprVectorsEqual(use->offsets, store_offsets)) {
      return false;
    }
    matched_refs += use->param_refs_in_stmt;
  }
  return matched_refs == total_refs;
}

bool IsProvenSideEffectStoreWithDirectReturn(const FunctionPtr& func, size_t out_param_index,
                                             const AssignStmtPtr& store_assign,
                                             const std::vector<ExprPtr>& store_shape,
                                             const std::vector<ExprPtr>& store_offsets,
                                             const ReturnStmtPtr& ret_stmt) {
  if (!func || !store_assign || !ret_stmt || out_param_index >= func->params_.size() ||
      out_param_index >= func->param_directions_.size()) {
    return false;
  }
  const auto direction = func->param_directions_[out_param_index];
  const auto* param = func->params_[out_param_index].get();
  size_t total_refs = CountVarRefsInStmt(func->body_, param);
  size_t matched_refs = CountVarRefsInStmt(store_assign, param) + CountVarRefsInStmt(ret_stmt, param);
  if (total_refs == 0 || matched_refs == 0 || matched_refs > total_refs) return false;

  auto body_stmts = FlattenToStmts(func->body_);
  for (const auto& stmt : body_stmts) {
    size_t refs = CountVarRefsInStmt(stmt, param);
    if (refs == 0 || stmt.get() == store_assign.get() || stmt.get() == ret_stmt.get()) continue;
    if (direction != ParamDirection::InOut) return false;
    auto use = MatchDirectTensorWindowAccess(As<AssignStmt>(stmt), param);
    if (!use.has_value()) return false;
    if (!AreExprVectorsEqual(use->window_shape, store_shape) ||
        !AreExprVectorsEqual(use->offsets, store_offsets)) {
      return false;
    }
    matched_refs += use->param_refs_in_stmt;
  }
  return matched_refs == total_refs;
}

std::optional<FinalStoreInfo> AnalyzeFinalStore(const FunctionPtr& func, size_t out_param_index) {
  if (!func || out_param_index >= func->params_.size()) return std::nullopt;

  auto body_stmts = FlattenToStmts(func->body_);
  std::unordered_map<const Var*, AssignStmtPtr> var_defs;
  for (const auto& stmt : body_stmts) {
    if (auto assign = As<AssignStmt>(stmt)) var_defs[assign->var_.get()] = assign;
  }

  ReturnStmtPtr ret_stmt;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      ret_stmt = ret;
      break;
    }
  }
  if (!ret_stmt) return std::nullopt;

  size_t total_out_refs = CountVarRefsInStmt(func->body_, func->params_[out_param_index].get());
  for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
    auto ret_var = AsVarLike(ret_stmt->value_[ret_i]);
    if (!ret_var) continue;
    auto def_it = var_defs.find(ret_var.get());
    if (def_it == var_defs.end()) continue;
    auto store_call = As<Call>(def_it->second->value_);
    if (!store_call || !IsOp(store_call, "tile.store") || store_call->args_.size() < 3) continue;

    auto out_target = AsVarLike(store_call->args_[2]);
    if (!out_target || out_target.get() != func->params_[out_param_index].get()) continue;
    auto offset_tuple = As<MakeTuple>(store_call->args_[1]);
    auto tile_type = As<TileType>(store_call->args_[0]->GetType());
    if (!offset_tuple || !tile_type) return std::nullopt;

    size_t matched_refs = CountVarRefsInStmt(def_it->second, func->params_[out_param_index].get());
    if (total_out_refs != matched_refs &&
        !IsProvenSameRegionInOutAccess(func, out_param_index, def_it->second, tile_type->shape_,
                                       offset_tuple->elements_)) {
      return std::nullopt;
    }

    return FinalStoreInfo{ret_i, tile_type->shape_, offset_tuple->elements_};
  }

  auto direct_return_index = FindReturnIndexForOutParam(func, out_param_index);
  if (!direct_return_index.has_value()) return std::nullopt;
  for (const auto& stmt : body_stmts) {
    auto assign = As<AssignStmt>(stmt);
    if (!assign) continue;
    auto store_call = As<Call>(assign->value_);
    if (!store_call || !IsOp(store_call, "tile.store") || store_call->args_.size() < 3) continue;
    auto out_target = AsVarLike(store_call->args_[2]);
    if (!out_target || out_target.get() != func->params_[out_param_index].get()) continue;
    auto offset_tuple = As<MakeTuple>(store_call->args_[1]);
    auto tile_type = As<TileType>(store_call->args_[0]->GetType());
    if (!offset_tuple || !tile_type) return std::nullopt;
    if (!IsProvenSideEffectStoreWithDirectReturn(func, out_param_index, assign, tile_type->shape_,
                                                 offset_tuple->elements_, ret_stmt)) {
      continue;
    }
    return FinalStoreInfo{*direct_return_index, tile_type->shape_, offset_tuple->elements_};
  }
  return std::nullopt;
}

bool HasOnlyFullShapeZeroOffsetReturnOutputs(const FunctionPtr& func,
                                             const std::vector<size_t>& out_indices) {
  if (!func) return false;
  for (const auto& out_index : out_indices) {
    auto out_tensor_type = As<TensorType>(func->params_[out_index]->GetType());
    if (!out_tensor_type) return false;
    auto info = AnalyzeFinalStore(func, out_index);
    if (!info.has_value()) return false;
    if (!AreExprVectorsEqual(info->window_shape, out_tensor_type->shape_) ||
        !IsAllZeroOffsets(info->offsets)) {
      return false;
    }
  }
  return true;
}

std::optional<InputWindowUse> MatchInputWindowUse(const AssignStmtPtr& assign, const Var* param,
                                                  size_t refs_in_stmt) {
  if (!assign || !param) return std::nullopt;
  auto call = As<Call>(assign->value_);
  if (!call || call->args_.empty()) return std::nullopt;

  std::vector<ExprPtr> window_shape;
  MakeTuplePtr offsets;
  if (IsOp(call, "tile.load") && call->args_.size() >= 3) {
    auto access = MatchFixedTileLoadAccess(call, param);
    if (!access.has_value()) return std::nullopt;
    window_shape = access->window_shape;
    offsets = access->offsets;
  } else if (IsOp(call, "tensor.slice") && call->args_.size() >= 3) {
    auto parent = AsVarLike(call->args_[0]);
    offsets = As<MakeTuple>(call->args_[2]);
    auto tensor_type = As<TensorType>(call->GetType());
    if (!parent || parent.get() != param || !offsets || !tensor_type) return std::nullopt;
    // The slice op is itself the complete access to the parent region. Any
    // later use must reference the slice value, so total_refs accounting below
    // rejects extra reads from the original full input.
    window_shape = tensor_type->shape_;
  } else {
    return std::nullopt;
  }

  if (window_shape.size() != offsets->elements_.size()) return std::nullopt;
  if (refs_in_stmt == 0) return std::nullopt;
  return InputWindowUse{std::move(window_shape), offsets->elements_, refs_in_stmt};
}

std::optional<InputWindowUse> MatchExpandedInputWindowUse(
    const AssignStmtPtr& assign, const Var* param, size_t refs_in_stmt,
    const std::unordered_map<const Var*, ExprPtr>& subst) {
  auto use = MatchInputWindowUse(assign, param, refs_in_stmt);
  if (!use.has_value()) return std::nullopt;

  arith::Analyzer analyzer;
  for (auto& dim : use->window_shape) {
    dim = analyzer.Simplify(transform_utils::Substitute(dim, subst));
  }
  for (auto& offset : use->offsets) {
    offset = analyzer.Simplify(transform_utils::Substitute(offset, subst));
  }
  return use;
}

struct ExtractedInputAccessSet {
  size_t total_refs = 0;
  bool unsupported_ref = false;
  std::vector<InputWindowUse> uses;
};

/// Per-statement count of `Var`/`IterArg` references to one parameter, for
/// every statement in a subtree, computed in a single traversal.
///
/// `ExtractInputAccessSet` asks "how many times does this subtree touch
/// `param`?" at every statement it walks. Answering each question with its own
/// `CountVarRefsInStmt` scan re-reads the whole remaining subtree once per
/// nesting level, so a chain of N nested loops costs O(N^2). Precomputing the
/// answers bottom-up costs one traversal and makes each lookup O(1).
class ParamRefIndex : public IRVisitor {
 public:
  explicit ParamRefIndex(const Var* target) : target_(target) {}

  /// Reference count of `stmt`'s whole subtree; 0 for a statement not indexed.
  [[nodiscard]] size_t Count(const StmtPtr& stmt) const {
    auto it = counts_.find(stmt.get());
    return it == counts_.end() ? 0 : it->second;
  }

  void VisitStmt(const StmtPtr& stmt) override {
    if (!stmt) return;
    const size_t before = hits_;
    IRVisitor::VisitStmt(stmt);
    counts_[stmt.get()] = hits_ - before;
  }

 protected:
  void VisitExpr_(const VarPtr& op) override {
    if (op.get() == target_) ++hits_;
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IterArgPtr& op) override {
    if (op.get() == target_) ++hits_;
    IRVisitor::VisitExpr_(op);
  }

 private:
  const Var* target_;
  size_t hits_ = 0;
  std::unordered_map<const Stmt*, size_t> counts_;
};

ExtractedInputAccessSet ExtractInputAccessSet(const StmtPtr& root, const Var* param,
                                              std::unordered_map<const Var*, ExprPtr> subst = {}) {
  ExtractedInputAccessSet result;
  if (!root || !param) return result;

  // Keep recursive input access extraction bounded; larger dynamic patterns
  // stay on the baseline path instead of expanding compile-time work.
  constexpr int64_t kMaxEnumeratedLoopTripCount = 256;
  constexpr size_t kMaxEnumeratedInputUses = 512;
  arith::Analyzer analyzer;

  // One traversal answers every "does this subtree read `param`?" question
  // below, including the repeats a trip-enumerated body would otherwise force.
  ParamRefIndex ref_index(param);
  ref_index.VisitStmt(root);

  auto simplify_with_subst = [&](const ExprPtr& expr) -> ExprPtr {
    return analyzer.Simplify(transform_utils::Substitute(expr, subst));
  };

  std::function<void(const StmtPtr&)> visit_stmt = [&](const StmtPtr& stmt) {
    if (!stmt || result.unsupported_ref) return;

    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) visit_stmt(child);
      return;
    }

    if (auto assign = As<AssignStmt>(stmt)) {
      size_t refs = ref_index.Count(assign);
      if (refs != 0) {
        auto use = MatchExpandedInputWindowUse(assign, param, refs, subst);
        if (!use.has_value()) {
          result.unsupported_ref = true;
          return;
        }
        result.total_refs += refs;
        result.uses.push_back(std::move(*use));
        if (result.uses.size() > kMaxEnumeratedInputUses) {
          result.unsupported_ref = true;
          return;
        }
      }
      if (As<ScalarType>(assign->var_->GetType())) {
        subst[assign->var_.get()] = simplify_with_subst(assign->value_);
      }
      return;
    }

    if (auto loop = As<ForStmt>(stmt)) {
      if (CountVarRefsInExpr(loop->start_, param) != 0 || CountVarRefsInExpr(loop->stop_, param) != 0 ||
          CountVarRefsInExpr(loop->step_, param) != 0) {
        result.unsupported_ref = true;
        return;
      }

      // A body that never mentions `param` records nothing however many trips
      // it is unrolled for, so enumerating it is pure cost -- and nested loops
      // multiply that cost by their trip counts. Skip it, exactly as the
      // unknown-trip-count branch below already does. Any scalar defs the body
      // would have contributed are loop-local and get restored after the trip
      // loop anyway, so this cannot change the extracted access set. The
      // lookup is O(1): `ref_index` answered it during its single pass.
      const bool body_reads_param = ref_index.Count(loop->body_) != 0;

      auto trip_count = GetKnownPositiveTripCount(loop);
      if (!trip_count.has_value() || *trip_count < 0 || *trip_count > kMaxEnumeratedLoopTripCount) {
        if (body_reads_param) result.unsupported_ref = true;
        return;
      }
      if (!body_reads_param) return;

      auto saved_subst = subst;
      for (int64_t trip = 0; trip < *trip_count; ++trip) {
        auto loop_value = GetLoopValueAtTrip(loop, trip);
        if (!loop_value.has_value()) {
          result.unsupported_ref = true;
          break;
        }
        subst = saved_subst;
        subst[loop->loop_var_.get()] = analyzer.Simplify(transform_utils::Substitute(*loop_value, subst));
        visit_stmt(loop->body_);
        if (result.unsupported_ref) break;
      }
      subst = std::move(saved_subst);
      return;
    }

    size_t refs = ref_index.Count(stmt);
    if (refs != 0) result.unsupported_ref = true;
  };

  visit_stmt(root);
  return result;
}

std::optional<InputRewriteInfo> BuildDenseInputWindowFromAccessSet(
    const FunctionPtr& func, size_t param_index, const ExtractedInputAccessSet& access_set) {
  if (!func || param_index >= func->params_.size()) return std::nullopt;
  auto tensor_type = As<TensorType>(func->params_[param_index]->GetType());
  if (!tensor_type || access_set.unsupported_ref || access_set.uses.empty() || access_set.total_refs == 0) {
    return std::nullopt;
  }

  std::vector<ExprPtr> base_offsets(tensor_type->shape_.size());
  std::vector<ExprPtr> max_extents(tensor_type->shape_.size());
  arith::Analyzer analyzer;
  bool expands_beyond_single_access = access_set.uses.size() > 1;
  for (const auto& use : access_set.uses) {
    if (use.offsets.size() != use.window_shape.size() || use.offsets.size() != tensor_type->shape_.size()) {
      return std::nullopt;
    }
    for (size_t dim = 0; dim < use.offsets.size(); ++dim) {
      auto min_expr = SelectMinExpr(base_offsets[dim], use.offsets[dim], func->span_);
      if (!min_expr.has_value()) return std::nullopt;
      base_offsets[dim] = *min_expr;

      auto extent = analyzer.Simplify(MakeAdd(use.offsets[dim], use.window_shape[dim], func->span_));
      auto max_expr = SelectMaxExpr(max_extents[dim], extent, func->span_);
      if (!max_expr.has_value()) return std::nullopt;
      max_extents[dim] = *max_expr;
    }
  }

  std::vector<ExprPtr> window_shape;
  std::vector<ExprPtr> local_zero_offsets;
  window_shape.reserve(tensor_type->shape_.size());
  local_zero_offsets.reserve(tensor_type->shape_.size());
  for (size_t dim = 0; dim < tensor_type->shape_.size(); ++dim) {
    if (!base_offsets[dim] || !max_extents[dim]) return std::nullopt;
    auto span = GetConstantSpanValue(max_extents[dim], base_offsets[dim], func->span_);
    if (!span.has_value()) return std::nullopt;
    int64_t span_value = *span;
    if (span_value <= 0) return std::nullopt;
    window_shape.push_back(std::make_shared<ConstInt>(span_value, DataType::INDEX, func->span_));
    local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
  }

  if (!expands_beyond_single_access && access_set.uses.size() == 1) {
    expands_beyond_single_access = !AreExprVectorsEqual(access_set.uses.front().window_shape, window_shape) ||
                                   !AreExprVectorsEqual(access_set.uses.front().offsets, base_offsets);
  }
  if (!expands_beyond_single_access) return std::nullopt;

  if (AreExprVectorsEqual(window_shape, tensor_type->shape_) && IsAllZeroOffsets(base_offsets)) {
    return std::nullopt;
  }
  if (!CanMaterializeWindowParamType(tensor_type, window_shape)) return std::nullopt;

  auto allowed_params = CollectAllowedVars(func->params_);
  if (!ExprsReferenceOnlyVarsIn(window_shape, allowed_params) ||
      !ExprsReferenceOnlyVarsIn(base_offsets, allowed_params)) {
    return std::nullopt;
  }

  auto piece = MakeDensePiece(window_shape, base_offsets, local_zero_offsets);
  return InputRewriteInfo{param_index,
                          tensor_type->shape_,
                          std::move(window_shape),
                          std::move(base_offsets),
                          std::move(local_zero_offsets),
                          MakeDenseRegion({std::move(piece)})};
}

std::unordered_map<const Var*, InputParamUseSummary> CollectInputParamUsesInStmt(
    const StmtPtr& root, const std::unordered_map<const Var*, size_t>& candidate_indices) {
  std::unordered_map<const Var*, InputParamUseSummary> summaries;
  if (!root || candidate_indices.empty()) return summaries;

  auto body_stmts = FlattenToStmts(root);
  class CandidateRefCollector : public IRVisitor {
   public:
    explicit CandidateRefCollector(const std::unordered_map<const Var*, size_t>& candidate_indices)
        : candidate_indices_(candidate_indices) {}

    [[nodiscard]] const std::unordered_map<const Var*, size_t>& refs() const { return refs_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (candidate_indices_.count(op.get())) ++refs_[op.get()];
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (candidate_indices_.count(op.get())) ++refs_[op.get()];
      IRVisitor::VisitExpr_(op);
    }

   private:
    const std::unordered_map<const Var*, size_t>& candidate_indices_;
    std::unordered_map<const Var*, size_t> refs_;
  };

  for (const auto& stmt : body_stmts) {
    CandidateRefCollector collector(candidate_indices);
    collector.VisitStmt(stmt);

    for (const auto& [param, refs_in_stmt] : collector.refs()) {
      auto& summary = summaries[param];
      summary.total_refs += refs_in_stmt;

      auto use = MatchInputWindowUse(As<AssignStmt>(stmt), param, refs_in_stmt);
      if (!use.has_value()) {
        summary.unsupported_ref = true;
        continue;
      }
      summary.uses.push_back(std::move(*use));
    }
  }

  return summaries;
}

std::unordered_map<const Var*, InputParamUseSummary> CollectInputParamUses(
    const FunctionPtr& func, const std::unordered_map<const Var*, size_t>& candidate_indices) {
  if (!func) return {};
  return CollectInputParamUsesInStmt(func->body_, candidate_indices);
}

std::vector<InputRewriteInfo> AnalyzeInputWindows(const FunctionPtr& func) {
  std::vector<InputRewriteInfo> inputs;
  if (!func) return inputs;
  if (func->return_types_.empty()) return inputs;

  auto allowed_params = CollectAllowedVars(func->params_);

  std::unordered_map<const Var*, size_t> candidate_indices;
  std::vector<std::pair<const Var*, size_t>> ordered_candidates;
  for (size_t param_index = 0; param_index < func->params_.size(); ++param_index) {
    if (param_index >= func->param_directions_.size()) continue;
    if (func->param_directions_[param_index] != ParamDirection::In) continue;
    if (!As<TensorType>(func->params_[param_index]->GetType())) continue;
    candidate_indices.emplace(func->params_[param_index].get(), param_index);
    ordered_candidates.emplace_back(func->params_[param_index].get(), param_index);
  }

  auto summaries = CollectInputParamUses(func, candidate_indices);
  for (const auto& [param_ptr, param_index] : ordered_candidates) {
    const auto& param = func->params_[param_index];
    auto summary_it = summaries.find(param_ptr);
    if (summary_it == summaries.end() || summary_it->second.total_refs == 0) continue;

    auto tensor_type = As<TensorType>(param->GetType());
    if (!tensor_type) continue;

    std::optional<InputRewriteInfo> input_info;
    std::optional<InputWindowUse> matched;
    size_t matched_refs = 0;
    bool unsupported_ref = summary_it->second.unsupported_ref;
    for (const auto& use : summary_it->second.uses) {
      if (!AreExprVectorsEqual(use.window_shape, matched ? matched->window_shape : use.window_shape) ||
          !AreExprVectorsEqual(use.offsets, matched ? matched->offsets : use.offsets)) {
        unsupported_ref = true;
        break;
      }
      matched = use;
      matched_refs += use.param_refs_in_stmt;
    }
    if (!unsupported_ref && matched.has_value() && matched_refs == summary_it->second.total_refs &&
        !(AreExprVectorsEqual(matched->window_shape, tensor_type->shape_) &&
          IsAllZeroOffsets(matched->offsets)) &&
        CanMaterializeWindowParamType(tensor_type, matched->window_shape) &&
        ExprsReferenceOnlyVarsIn(matched->window_shape, allowed_params) &&
        ExprsReferenceOnlyVarsIn(matched->offsets, allowed_params)) {
      std::vector<ExprPtr> local_zero_offsets;
      local_zero_offsets.reserve(matched->offsets.size());
      for (size_t i = 0; i < matched->offsets.size(); ++i) {
        local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
      }
      auto piece = MakeDensePiece(matched->window_shape, matched->offsets, local_zero_offsets);
      input_info = InputRewriteInfo{
          param_index,      tensor_type->shape_,           matched->window_shape,
          matched->offsets, std::move(local_zero_offsets), MakeDenseRegion({std::move(piece)})};
    }

    if (!input_info.has_value()) {
      auto access_set = ExtractInputAccessSet(func->body_, param_ptr);
      input_info = BuildDenseInputWindowFromAccessSet(func, param_index, access_set);
    }
    if (input_info.has_value()) inputs.push_back(std::move(*input_info));
  }

  return inputs;
}

std::optional<InputRewriteInfo> AnalyzeAggregateInputWindowInLoop(const FunctionPtr& func, size_t param_index,
                                                                  const ForStmtPtr& loop, size_t total_refs,
                                                                  const InputParamUseSummary& loop_summary) {
  if (!func || param_index >= func->params_.size() || !loop) return std::nullopt;
  auto tensor_type = As<TensorType>(func->params_[param_index]->GetType());
  if (!tensor_type) return std::nullopt;

  auto trip_count = GetKnownPositiveTripCount(loop);
  if (!trip_count.has_value() || *trip_count <= 0) return std::nullopt;
  auto first_loop_value = GetLoopValueAtTrip(loop, 0);
  auto last_loop_value = GetLoopValueAtTrip(loop, *trip_count - 1);
  if (!first_loop_value.has_value() || !last_loop_value.has_value()) return std::nullopt;

  if (total_refs == 0 || total_refs != loop_summary.total_refs || loop_summary.unsupported_ref) {
    return std::nullopt;
  }

  auto loop_body_stmts = FlattenToStmts(loop->body_);
  std::unordered_map<const Var*, ExprPtr> scalar_defs;
  for (const auto& stmt : loop_body_stmts) {
    if (auto assign = As<AssignStmt>(stmt)) {
      if (As<ScalarType>(assign->var_->GetType())) {
        scalar_defs[assign->var_.get()] = assign->value_;
      }
    }
  }

  const auto& uses = loop_summary.uses;
  size_t matched_refs = 0;
  for (const auto& use : uses) matched_refs += use.param_refs_in_stmt;
  if (uses.empty() || matched_refs != total_refs) return std::nullopt;

  auto allowed = CollectAllowedVars(func->params_, loop->loop_var_.get());

  std::optional<InputRewriteInfo> result;
  for (const auto& use : uses) {
    if (use.offsets.size() != use.window_shape.size() || use.offsets.size() != tensor_type->shape_.size()) {
      return std::nullopt;
    }

    std::vector<ExprPtr> base_offsets;
    std::vector<ExprPtr> local_offsets;
    std::vector<ExprPtr> window_shape;
    bool expands_across_loop = false;
    arith::Analyzer analyzer;
    for (size_t i = 0; i < use.offsets.size(); ++i) {
      auto expanded = ExpandLoopLocalExpr(use.offsets[i], scalar_defs);
      if (!expanded.has_value()) return std::nullopt;
      if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) return std::nullopt;

      auto ordered_offsets = GetOrderedLoopOffsets(*expanded, loop, *first_loop_value, *last_loop_value);
      if (!ordered_offsets.has_value()) return std::nullopt;

      auto max_extent = analyzer.Simplify(MakeAdd(ordered_offsets->max, use.window_shape[i], func->span_));
      auto span_value = GetConstantSpanValue(max_extent, ordered_offsets->min, func->span_);
      if (!span_value.has_value() || *span_value <= 0) return std::nullopt;

      if (!AreExprsEqual(ordered_offsets->min, ordered_offsets->max)) {
        expands_across_loop = true;
      }
      base_offsets.push_back(ordered_offsets->min);
      local_offsets.push_back(
          analyzer.Simplify(MakeSub(use.offsets[i], ordered_offsets->min, use.offsets[i]->span_)));
      window_shape.push_back(std::make_shared<ConstInt>(*span_value, DataType::INDEX, func->span_));
    }
    if (!expands_across_loop) return std::nullopt;

    auto current_window_shape = std::move(window_shape);
    auto current_base_offsets = std::move(base_offsets);
    auto current_local_offsets = std::move(local_offsets);
    auto current_piece = MakeDensePiece(current_window_shape, current_base_offsets, current_local_offsets);
    InputRewriteInfo current{param_index,
                             tensor_type->shape_,
                             std::move(current_window_shape),
                             std::move(current_base_offsets),
                             std::move(current_local_offsets),
                             MakeDenseRegion({std::move(current_piece)})};
    if (!CanMaterializeWindowParamType(tensor_type, current.window_shape)) return std::nullopt;

    if (!result.has_value()) {
      result = std::move(current);
      continue;
    }
    if (!AreExprVectorsEqual(result->window_shape, current.window_shape) ||
        !AreExprVectorsEqual(result->callsite_offsets, current.callsite_offsets) ||
        !AreExprVectorsEqual(result->local_read_offsets, current.local_read_offsets)) {
      return std::nullopt;
    }
  }

  if (!result.has_value()) return std::nullopt;
  auto allowed_params = CollectAllowedVars(func->params_);
  if (!ExprsReferenceOnlyVarsIn(result->window_shape, allowed_params) ||
      !ExprsReferenceOnlyVarsIn(result->callsite_offsets, allowed_params)) {
    return std::nullopt;
  }
  return result;
}

std::vector<InputRewriteInfo> AnalyzeAggregateInputWindows(
    const FunctionPtr& func, const std::vector<InputRewriteInfo>& existing_inputs, const ForStmtPtr& loop) {
  std::vector<InputRewriteInfo> inputs;
  if (!func || !loop) return inputs;

  std::unordered_set<size_t> existing_indices;
  for (const auto& input : existing_inputs) existing_indices.insert(input.in_param_index);

  std::unordered_map<const Var*, size_t> candidate_indices;
  std::vector<std::pair<const Var*, size_t>> ordered_candidates;
  for (size_t param_index = 0; param_index < func->params_.size(); ++param_index) {
    if (existing_indices.count(param_index)) continue;
    if (param_index >= func->param_directions_.size()) continue;
    if (func->param_directions_[param_index] != ParamDirection::In) continue;
    if (!As<TensorType>(func->params_[param_index]->GetType())) continue;
    candidate_indices.emplace(func->params_[param_index].get(), param_index);
    ordered_candidates.emplace_back(func->params_[param_index].get(), param_index);
  }
  if (candidate_indices.empty()) return inputs;

  auto total_summaries = CollectInputParamUsesInStmt(func->body_, candidate_indices);
  auto loop_summaries = CollectInputParamUsesInStmt(loop->body_, candidate_indices);
  for (const auto& [param_ptr, param_index] : ordered_candidates) {
    auto total_it = total_summaries.find(param_ptr);
    auto loop_it = loop_summaries.find(param_ptr);
    if (total_it == total_summaries.end() || loop_it == loop_summaries.end()) continue;

    auto matched = AnalyzeAggregateInputWindowInLoop(func, param_index, loop, total_it->second.total_refs,
                                                     loop_it->second);
    if (!matched.has_value()) {
      auto access_set = ExtractInputAccessSet(func->body_, param_ptr);
      matched = BuildDenseInputWindowFromAccessSet(func, param_index, access_set);
    }
    if (matched.has_value()) inputs.push_back(std::move(*matched));
  }
  return inputs;
}

/// True when an aggregate output covers its whole parent at offset zero, so
/// windowing it would not narrow the dependency at all.
bool CoversFullParent(const OutputRewriteInfo& info) {
  return AreExprVectorsEqual(info.window_shape, info.parent_shape) && IsAllZeroOffsets(info.callsite_offsets);
}

/// Analyze the aggregate output-window loop of `func`.
///
/// The result carries *every* provable aggregate output, full-parent ones
/// included. Those are not windowable on their own, so `Analyze` drops them
/// before using the analysis as a rewrite plan -- but the pure-input-window
/// verdict needs to see them, and running this traversal a second time to
/// recover them costs as much as the first.
std::optional<CalleeRewriteAnalysis> AnalyzeAggregateWindowLoop(
    const FunctionPtr& func, const std::vector<size_t>& out_indices,
    const std::vector<InputRewriteInfo>& existing_inputs) {
  if (!func || out_indices.empty()) return std::nullopt;

  auto body_stmts = FlattenToStmts(func->body_);
  if (body_stmts.empty()) return std::nullopt;

  ReturnStmtPtr ret_stmt = As<ReturnStmt>(body_stmts.back());
  if (!ret_stmt) return std::nullopt;

  struct AggregateLoopOutputMatch {
    size_t out_param_index;
    size_t return_index;
    size_t iter_arg_index;
  };

  ForStmtPtr loop;
  std::vector<AggregateLoopOutputMatch> loop_matches;
  for (const auto& stmt : body_stmts) {
    auto candidate = As<ForStmt>(stmt);
    if (!candidate || candidate->iter_args_.empty()) continue;
    std::vector<AggregateLoopOutputMatch> candidate_matches;
    std::unordered_set<size_t> matched_iter_arg_indices;

    for (const auto& out_param_index : out_indices) {
      std::optional<size_t> direct_return_index = FindReturnIndexForOutParam(func, out_param_index);
      VarPtr direct_returned;
      if (direct_return_index.has_value() && *direct_return_index < ret_stmt->value_.size()) {
        direct_returned = AsVarLike(ret_stmt->value_[*direct_return_index]);
      }

      for (size_t i = 0; i < candidate->iter_args_.size() && i < candidate->return_vars_.size(); ++i) {
        auto init_var = AsVarLike(candidate->iter_args_[i]->initValue_);
        if (!init_var || init_var.get() != func->params_[out_param_index].get()) continue;

        std::optional<size_t> return_index = direct_return_index;
        if (direct_returned && direct_returned.get() != candidate->return_vars_[i].get() &&
            direct_returned.get() != func->params_[out_param_index].get()) {
          return_index = std::nullopt;
        }
        for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
          if (return_index.has_value()) break;
          auto returned = AsVarLike(ret_stmt->value_[ret_i]);
          if (returned && returned.get() == candidate->return_vars_[i].get()) {
            return_index = ret_i;
            break;
          }
        }
        if (!return_index.has_value()) continue;

        if (!matched_iter_arg_indices.insert(i).second) return std::nullopt;
        candidate_matches.push_back(AggregateLoopOutputMatch{out_param_index, *return_index, i});
        break;
      }
    }

    if (candidate_matches.empty()) continue;
    if (candidate->iter_args_.size() != candidate->return_vars_.size()) return std::nullopt;

    if (loop) return std::nullopt;
    loop = candidate;
    loop_matches = std::move(candidate_matches);
  }
  if (!loop) return std::nullopt;

  auto stop = transform_utils::EvalConstInt(loop->stop_);
  auto step = transform_utils::EvalConstInt(loop->step_);
  if (!stop.has_value() || !step.has_value()) {
    auto known_trip_count = GetKnownPositiveTripCount(loop);
    if (!known_trip_count.has_value() || *known_trip_count <= 0) return std::nullopt;
  } else if (*step <= 0) {
    return std::nullopt;
  }
  auto trip_count = GetKnownPositiveTripCount(loop);
  if (!trip_count.has_value() || *trip_count <= 0) return std::nullopt;
  auto first_loop_value = GetLoopValueAtTrip(loop, 0);
  auto last_loop_value = GetLoopValueAtTrip(loop, *trip_count - 1);
  if (!first_loop_value.has_value() || !last_loop_value.has_value()) return std::nullopt;

  auto loop_body_stmts = FlattenToStmts(loop->body_);
  YieldStmtPtr yield_stmt;
  struct AggregateUpdate {
    AssignStmtPtr assign;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> offsets;
  };

  std::unordered_map<size_t, std::vector<AggregateUpdate>> updates_by_iter_arg_index;
  std::unordered_map<size_t, std::vector<AggregateUpdate>> reads_by_iter_arg_index;
  std::unordered_map<size_t, const Var*> update_tail_by_iter_arg_index;
  std::unordered_set<const Var*> carrier_vars;
  std::unordered_set<const AssignStmt*> recognized_carrier_accesses;
  for (const auto& match : loop_matches) {
    if (match.iter_arg_index >= loop->iter_args_.size()) return std::nullopt;
    update_tail_by_iter_arg_index[match.iter_arg_index] = loop->iter_args_[match.iter_arg_index].get();
    carrier_vars.insert(loop->iter_args_[match.iter_arg_index].get());
  }
  std::unordered_map<const Var*, ExprPtr> scalar_defs;
  constexpr int64_t kMaxNestedAccessTripCount = 32;

  auto substitute_local_scalars = [](const ExprPtr& expr,
                                     const std::unordered_map<const Var*, ExprPtr>& local_defs) -> ExprPtr {
    return transform_utils::Substitute(expr, local_defs);
  };

  std::function<bool(const StmtPtr&, std::unordered_map<size_t, const Var*>*,
                     std::unordered_map<const Var*, ExprPtr>*, YieldStmtPtr*)>
      collect_accesses;

  collect_accesses = [&](const StmtPtr& stmt, std::unordered_map<size_t, const Var*>* tails,
                         std::unordered_map<const Var*, ExprPtr>* local_scalar_defs,
                         YieldStmtPtr* seen_yield) -> bool {
    if (!stmt || !tails || !local_scalar_defs || !seen_yield) return false;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) {
        if (!collect_accesses(child, tails, local_scalar_defs, seen_yield)) return false;
      }
      return true;
    }

    if (auto assign = As<AssignStmt>(stmt)) {
      auto call = As<Call>(assign->value_);
      if (call) {
        const Var* updated_tail = nullptr;
        const Var* read_tail = nullptr;
        std::vector<ExprPtr> window_shape;
        std::vector<ExprPtr> offsets;
        if (IsOp(call, "tile.store") && call->args_.size() >= 3) {
          auto out_arg = AsVarLike(call->args_[2]);
          auto offset_tuple = As<MakeTuple>(call->args_[1]);
          auto tile_type = As<TileType>(call->args_[0]->GetType());
          if (out_arg && offset_tuple && tile_type) {
            updated_tail = out_arg.get();
            window_shape = tile_type->shape_;
            offsets = offset_tuple->elements_;
          }
        } else if (IsOp(call, "tensor.assemble") && call->args_.size() >= 3) {
          auto parent_arg = AsVarLike(call->args_[0]);
          auto offset_tuple = As<MakeTuple>(call->args_[2]);
          auto source_type = As<TensorType>(call->args_[1]->GetType());
          if (parent_arg && offset_tuple && source_type) {
            updated_tail = parent_arg.get();
            window_shape = source_type->shape_;
            offsets = offset_tuple->elements_;
          }
        } else if (IsOp(call, "tile.load") && call->args_.size() >= 3) {
          auto parent_arg = AsVarLike(call->args_[0]);
          auto offset_tuple = As<MakeTuple>(call->args_[1]);
          auto tile_type = As<TileType>(call->GetType());
          if (parent_arg && offset_tuple && tile_type) {
            read_tail = parent_arg.get();
            window_shape = tile_type->shape_;
            offsets = offset_tuple->elements_;
          }
        } else if (IsOp(call, "tensor.slice") && call->args_.size() >= 3) {
          auto parent_arg = AsVarLike(call->args_[0]);
          auto offset_tuple = As<MakeTuple>(call->args_[2]);
          auto source_type = As<TensorType>(call->GetType());
          if (parent_arg && offset_tuple && source_type) {
            read_tail = parent_arg.get();
            window_shape = source_type->shape_;
            offsets = offset_tuple->elements_;
          }
        }

        if (updated_tail) {
          for (auto& [iter_arg_index, tail] : *tails) {
            if (updated_tail != tail) continue;
            for (auto& offset : offsets) {
              offset = substitute_local_scalars(offset, *local_scalar_defs);
            }
            updates_by_iter_arg_index[iter_arg_index].push_back(
                AggregateUpdate{assign, std::move(window_shape), std::move(offsets)});
            tail = assign->var_.get();
            carrier_vars.insert(assign->var_.get());
            recognized_carrier_accesses.insert(assign.get());
            return true;
          }
        }
        if (read_tail) {
          for (auto& [iter_arg_index, tail] : *tails) {
            if (read_tail != tail) continue;
            for (auto& offset : offsets) {
              offset = substitute_local_scalars(offset, *local_scalar_defs);
            }
            reads_by_iter_arg_index[iter_arg_index].push_back(
                AggregateUpdate{assign, std::move(window_shape), std::move(offsets)});
            recognized_carrier_accesses.insert(assign.get());
            return true;
          }
        }
      }

      if (As<ScalarType>(assign->var_->GetType())) {
        (*local_scalar_defs)[assign->var_.get()] =
            substitute_local_scalars(assign->value_, *local_scalar_defs);
      }
      return true;
    }

    if (auto nested_loop = As<ForStmt>(stmt)) {
      std::unordered_map<size_t, size_t> nested_iter_by_outer_iter;
      for (auto& [outer_iter_index, tail] : *tails) {
        for (size_t nested_i = 0; nested_i < nested_loop->iter_args_.size(); ++nested_i) {
          auto init_var = AsVarLike(nested_loop->iter_args_[nested_i]->initValue_);
          if (init_var && init_var.get() == tail) {
            nested_iter_by_outer_iter.emplace(outer_iter_index, nested_i);
            break;
          }
        }
      }
      if (nested_iter_by_outer_iter.empty()) return true;

      auto nested_trip_count = GetKnownPositiveTripCount(nested_loop);
      if (!nested_trip_count.has_value() || *nested_trip_count < 0 ||
          *nested_trip_count > kMaxNestedAccessTripCount) {
        return false;
      }
      if (nested_loop->iter_args_.size() != nested_loop->return_vars_.size()) return false;

      for (int64_t trip = 0; trip < *nested_trip_count; ++trip) {
        auto loop_value = GetLoopValueAtTrip(nested_loop, trip);
        if (!loop_value.has_value()) return false;

        auto trip_scalar_defs = *local_scalar_defs;
        trip_scalar_defs[nested_loop->loop_var_.get()] =
            substitute_local_scalars(*loop_value, trip_scalar_defs);

        std::unordered_map<size_t, const Var*> trip_tails;
        for (const auto& [outer_iter_index, nested_i] : nested_iter_by_outer_iter) {
          trip_tails[outer_iter_index] = nested_loop->iter_args_[nested_i].get();
          carrier_vars.insert(nested_loop->iter_args_[nested_i].get());
        }

        YieldStmtPtr nested_yield;
        if (!collect_accesses(nested_loop->body_, &trip_tails, &trip_scalar_defs, &nested_yield)) {
          return false;
        }
        if (!nested_yield || nested_yield->value_.size() != nested_loop->return_vars_.size()) {
          return false;
        }
        for (const auto& [outer_iter_index, nested_i] : nested_iter_by_outer_iter) {
          auto yielded = AsVarLike(nested_yield->value_[nested_i]);
          auto tail_it = trip_tails.find(outer_iter_index);
          if (!yielded || tail_it == trip_tails.end() || yielded.get() != tail_it->second) {
            return false;
          }
        }
      }

      for (const auto& [outer_iter_index, nested_i] : nested_iter_by_outer_iter) {
        (*tails)[outer_iter_index] = nested_loop->return_vars_[nested_i].get();
        carrier_vars.insert(nested_loop->return_vars_[nested_i].get());
      }
      return true;
    }

    if (auto yield = As<YieldStmt>(stmt)) {
      if (*seen_yield) return false;
      *seen_yield = yield;
      return true;
    }

    return true;
  };

  if (!collect_accesses(loop->body_, &update_tail_by_iter_arg_index, &scalar_defs, &yield_stmt)) {
    return std::nullopt;
  }

  if (!yield_stmt) return std::nullopt;
  // Mirror the nested-loop guard above: indexing the yield by iter-arg index is
  // only safe once the outer yield is known to carry one value per carry.
  if (yield_stmt->value_.size() != loop->return_vars_.size()) return std::nullopt;

  const auto function_use_index = BuildVarUseIndex(func->body_);
  const auto loop_use_index = BuildVarUseIndex(loop);
  const auto loop_body_use_index = BuildVarUseIndex(loop->body_);
  const auto return_use_index = BuildVarUseIndex(ret_stmt);
  // Visit order does not escape: the loop bails out when *any* carrier has an
  // unrecognized user, which is order-independent.
  // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
  for (const auto* carrier_var : carrier_vars) {
    auto users_it = loop_body_use_index.assign_users.find(carrier_var);
    if (users_it == loop_body_use_index.assign_users.end()) continue;
    for (const auto* user : users_it->second) {
      if (recognized_carrier_accesses.count(user) == 0) return std::nullopt;
    }
  }

  auto allowed = CollectAllowedVars(func->params_, loop->loop_var_.get());

  CalleeRewriteAnalysis analysis;
  analysis.kind = RewriteKind::AggregateWindowLoop;

  for (const auto& match : loop_matches) {
    auto update_it = updates_by_iter_arg_index.find(match.iter_arg_index);
    if (update_it == updates_by_iter_arg_index.end()) continue;
    const auto& updates = update_it->second;
    if (updates.empty()) continue;
    const auto* tail = update_tail_by_iter_arg_index[match.iter_arg_index];

    auto yielded = AsVarLike(yield_stmt->value_[match.iter_arg_index]);
    if (!yielded || yielded.get() != tail) continue;

    if (!As<TensorType>(loop->iter_args_[match.iter_arg_index]->GetType()) ||
        !As<TensorType>(loop->return_vars_[match.iter_arg_index]->GetType())) {
      continue;
    }

    auto out_param = func->params_[match.out_param_index].get();
    auto total_refs_it = function_use_index.counts.find(out_param);
    const size_t total_out_refs =
        total_refs_it == function_use_index.counts.end() ? 0 : total_refs_it->second;
    auto loop_refs_it = loop_use_index.counts.find(out_param);
    auto return_refs_it = return_use_index.counts.find(out_param);
    const size_t carrier_out_refs =
        (loop_refs_it == loop_use_index.counts.end() ? 0 : loop_refs_it->second) +
        (return_refs_it == return_use_index.counts.end() ? 0 : return_refs_it->second);
    if (total_out_refs == 0 || total_out_refs != carrier_out_refs) {
      continue;
    }

    bool update_chain_is_linear = true;
    for (const auto& update : updates) {
      auto refs_it = loop_body_use_index.counts.find(update.assign->var_.get());
      const size_t result_refs = refs_it == loop_body_use_index.counts.end() ? 0 : refs_it->second;
      if (result_refs == 0 || result_refs > 2) {
        update_chain_is_linear = false;
        break;
      }
    }
    if (!update_chain_is_linear) continue;

    const auto read_it = reads_by_iter_arg_index.find(match.iter_arg_index);
    if (read_it != reads_by_iter_arg_index.end()) {
      std::unordered_map<uint64_t, std::vector<const AggregateUpdate*>> updates_by_region;
      updates_by_region.reserve(updates.size());
      for (const auto& update : updates) {
        updates_by_region[HashAccessRegion(update.window_shape, update.offsets)].push_back(&update);
      }
      bool reads_match_writes = true;
      for (const auto& read : read_it->second) {
        bool matched_write = false;
        auto candidates_it = updates_by_region.find(HashAccessRegion(read.window_shape, read.offsets));
        if (candidates_it == updates_by_region.end()) {
          reads_match_writes = false;
          break;
        }
        for (const auto* update : candidates_it->second) {
          if (AreExprVectorsEqual(read.window_shape, update->window_shape) &&
              AreExprVectorsEqual(read.offsets, update->offsets)) {
            matched_write = true;
            break;
          }
        }
        if (!matched_write) {
          reads_match_writes = false;
          break;
        }
      }
      if (!reads_match_writes) continue;
    }

    auto out_tensor_type = As<TensorType>(func->params_[match.out_param_index]->GetType());
    if (!out_tensor_type) continue;

    auto try_build_static_pieces = [&]() -> std::vector<DenseRegionPiece> {
      // Static pieces are for small, exactly tiled loop nests; larger loops
      // would bloat signatures and orchestration code, so they stay baseline.
      constexpr int64_t kMaxStaticPieces = 32;
      if (*trip_count <= 0 || *trip_count > kMaxStaticPieces) return {};

      std::vector<DenseRegionPiece> pieces;
      pieces.reserve(static_cast<size_t>(*trip_count));
      arith::Analyzer analyzer;
      for (int64_t trip = 0; trip < *trip_count; ++trip) {
        auto loop_value = GetLoopValueAtTrip(loop, trip);
        if (!loop_value.has_value()) return {};

        std::vector<ExprPtr> piece_offsets(out_tensor_type->shape_.size());
        std::vector<ExprPtr> piece_extents(out_tensor_type->shape_.size());
        std::vector<DenseRect> update_rects;
        update_rects.reserve(updates.size());
        for (const auto& update : updates) {
          if (update.offsets.size() != update.window_shape.size() ||
              update.offsets.size() != out_tensor_type->shape_.size()) {
            return {};
          }
          DenseRect update_rect;
          update_rect.shape = update.window_shape;
          update_rect.offsets.resize(update.offsets.size());

          for (size_t dim = 0; dim < update.offsets.size(); ++dim) {
            auto expanded = ExpandLoopLocalExpr(update.offsets[dim], scalar_defs);
            if (!expanded.has_value()) return {};
            if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) return {};
            auto offset_at_trip = SimplifyWithLoopValue(*expanded, loop->loop_var_, *loop_value);
            if (!offset_at_trip.has_value()) return {};
            update_rect.offsets[dim] = *offset_at_trip;
            auto min_expr = SelectMinExpr(piece_offsets[dim], *offset_at_trip, func->span_);
            if (!min_expr.has_value()) return {};
            piece_offsets[dim] = *min_expr;
            auto extent = analyzer.Simplify(MakeAdd(*offset_at_trip, update.window_shape[dim], func->span_));
            auto max_expr = SelectMaxExpr(piece_extents[dim], extent, func->span_);
            if (!max_expr.has_value()) return {};
            piece_extents[dim] = *max_expr;
          }
          update_rects.push_back(std::move(update_rect));
        }

        std::vector<ExprPtr> piece_shape;
        std::vector<ExprPtr> local_zero_offsets;
        piece_shape.reserve(out_tensor_type->shape_.size());
        local_zero_offsets.reserve(out_tensor_type->shape_.size());
        for (size_t dim = 0; dim < out_tensor_type->shape_.size(); ++dim) {
          if (!piece_offsets[dim] || !piece_extents[dim]) return {};
          auto span_value = GetConstantSpanValue(piece_extents[dim], piece_offsets[dim], func->span_);
          if (!span_value.has_value() || *span_value <= 0) return {};
          piece_shape.push_back(std::make_shared<ConstInt>(*span_value, DataType::INDEX, func->span_));
          local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
        }

        if (!DenseRectsExactlyCoverBounds(update_rects, piece_offsets, piece_shape)) return {};
        DenseRegionPiece piece =
            MakeDensePiece(std::move(piece_shape), std::move(piece_offsets), std::move(local_zero_offsets));
        for (const auto& existing : pieces) {
          if (!DenseRectsAreDisjoint(existing.callsite_offsets, existing.window_shape, piece.callsite_offsets,
                                     piece.window_shape)) {
            return {};
          }
        }
        pieces.push_back(std::move(piece));
      }
      return pieces;
    };

    std::vector<ExprPtr> base_offsets;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> max_extents;
    std::vector<ExprPtr> first_iter_base_offsets;
    std::vector<ExprPtr> first_iter_max_extents;
    std::vector<bool> dim_varies;
    base_offsets.resize(out_tensor_type->shape_.size());
    max_extents.resize(out_tensor_type->shape_.size());
    first_iter_base_offsets.resize(out_tensor_type->shape_.size());
    first_iter_max_extents.resize(out_tensor_type->shape_.size());
    dim_varies.resize(out_tensor_type->shape_.size(), false);
    arith::Analyzer analyzer;
    bool output_window_is_proven = true;
    std::vector<DenseRect> first_iter_update_rects;
    first_iter_update_rects.reserve(updates.size());
    for (const auto& update : updates) {
      if (update.offsets.size() != update.window_shape.size() ||
          update.offsets.size() != out_tensor_type->shape_.size()) {
        output_window_is_proven = false;
        break;
      }
      if (!CheckedShapeVolume(update.window_shape).has_value()) {
        output_window_is_proven = false;
        break;
      }
      DenseRect first_iter_update_rect;
      first_iter_update_rect.shape = update.window_shape;
      first_iter_update_rect.offsets.resize(update.offsets.size());
      for (size_t i = 0; i < update.offsets.size(); ++i) {
        auto expanded = ExpandLoopLocalExpr(update.offsets[i], scalar_defs);
        if (!expanded.has_value()) {
          output_window_is_proven = false;
          break;
        }
        if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) {
          output_window_is_proven = false;
          break;
        }

        auto ordered_offsets = GetOrderedLoopOffsets(*expanded, loop, *first_loop_value, *last_loop_value);
        if (!ordered_offsets.has_value()) {
          output_window_is_proven = false;
          break;
        }
        if (!AreExprsEqual(ordered_offsets->min, ordered_offsets->max)) dim_varies[i] = true;

        auto min_expr = SelectMinExpr(base_offsets[i], ordered_offsets->min, func->span_);
        if (!min_expr.has_value()) {
          output_window_is_proven = false;
          break;
        }
        base_offsets[i] = *min_expr;

        auto extent = analyzer.Simplify(MakeAdd(ordered_offsets->max, update.window_shape[i], func->span_));
        auto max_expr = SelectMaxExpr(max_extents[i], extent, func->span_);
        if (!max_expr.has_value()) {
          output_window_is_proven = false;
          break;
        }
        max_extents[i] = *max_expr;

        auto first_offset = SimplifyWithLoopValue(*expanded, loop->loop_var_, *first_loop_value);
        if (!first_offset.has_value()) {
          output_window_is_proven = false;
          break;
        }
        first_iter_update_rect.offsets[i] = *first_offset;
        auto first_min_expr = SelectMinExpr(first_iter_base_offsets[i], *first_offset, func->span_);
        if (!first_min_expr.has_value()) {
          output_window_is_proven = false;
          break;
        }
        first_iter_base_offsets[i] = *first_min_expr;
        auto first_extent = analyzer.Simplify(MakeAdd(*first_offset, update.window_shape[i], func->span_));
        auto first_max_expr = SelectMaxExpr(first_iter_max_extents[i], first_extent, func->span_);
        if (!first_max_expr.has_value()) {
          output_window_is_proven = false;
          break;
        }
        first_iter_max_extents[i] = *first_max_expr;
      }
      if (!output_window_is_proven) break;
      first_iter_update_rects.push_back(std::move(first_iter_update_rect));
    }
    if (!output_window_is_proven) {
      auto pieces = try_build_static_pieces();
      if (pieces.empty()) continue;
      analysis.outputs.push_back(OutputRewriteInfo{match.out_param_index,
                                                   match.return_index,
                                                   out_tensor_type->shape_,
                                                   pieces.front().window_shape,
                                                   pieces.front().callsite_offsets,
                                                   pieces.front().local_offsets,
                                                   MakeDenseRegion(std::move(pieces)),
                                                   {},
                                                   match.iter_arg_index});
      continue;
    }

    std::vector<ExprPtr> local_zero_offsets;
    std::vector<ExprPtr> first_iter_window_shape;
    local_zero_offsets.reserve(out_tensor_type->shape_.size());
    window_shape.reserve(out_tensor_type->shape_.size());
    first_iter_window_shape.reserve(out_tensor_type->shape_.size());
    size_t varying_dim_count = 0;
    bool allow_static_fallback = true;
    for (size_t i = 0; i < out_tensor_type->shape_.size(); ++i) {
      if (!base_offsets[i] || !max_extents[i]) {
        output_window_is_proven = false;
        break;
      }
      auto span_value = GetConstantSpanValue(max_extents[i], base_offsets[i], func->span_);
      if (!span_value.has_value() || *span_value <= 0) {
        output_window_is_proven = false;
        break;
      }

      if (dim_varies[i]) {
        ++varying_dim_count;
        if (!first_iter_base_offsets[i] || !first_iter_max_extents[i]) {
          output_window_is_proven = false;
          break;
        }
        auto first_iter_span_value =
            GetConstantSpanValue(first_iter_max_extents[i], first_iter_base_offsets[i], func->span_);
        if (!first_iter_span_value.has_value() || *first_iter_span_value <= 0) {
          output_window_is_proven = false;
          break;
        }
        auto expected_dense_span = CheckedMul(*first_iter_span_value, *trip_count);
        if (!expected_dense_span.has_value() || *span_value != *expected_dense_span) {
          output_window_is_proven = false;
          break;
        }
      }

      if (!first_iter_base_offsets[i] || !first_iter_max_extents[i]) {
        output_window_is_proven = false;
        break;
      }
      auto first_iter_span_value =
          GetConstantSpanValue(first_iter_max_extents[i], first_iter_base_offsets[i], func->span_);
      if (!first_iter_span_value.has_value() || *first_iter_span_value <= 0) {
        output_window_is_proven = false;
        break;
      }
      first_iter_window_shape.push_back(
          std::make_shared<ConstInt>(*first_iter_span_value, DataType::INDEX, func->span_));
      window_shape.push_back(std::make_shared<ConstInt>(*span_value, DataType::INDEX, func->span_));
      local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
    }
    if (varying_dim_count > 1) {
      output_window_is_proven = false;
      allow_static_fallback = false;
    } else if (!DenseRectsExactlyCoverBounds(first_iter_update_rects, first_iter_base_offsets,
                                             first_iter_window_shape)) {
      output_window_is_proven = false;
    }
    if (!output_window_is_proven) {
      if (!allow_static_fallback) continue;
      auto pieces = try_build_static_pieces();
      if (pieces.empty()) continue;
      analysis.outputs.push_back(OutputRewriteInfo{match.out_param_index,
                                                   match.return_index,
                                                   out_tensor_type->shape_,
                                                   pieces.front().window_shape,
                                                   pieces.front().callsite_offsets,
                                                   pieces.front().local_offsets,
                                                   MakeDenseRegion(std::move(pieces)),
                                                   {},
                                                   match.iter_arg_index});
      continue;
    }

    // Only the dense path was ever gated on this: both static-piece fallbacks
    // above `continue` before reaching here, so a full-parent piece from the
    // fallback stayed a real rewrite. Record which case this is instead of
    // re-deriving it from the shapes, which cannot tell the two apart.
    const bool covers_full_parent =
        AreExprVectorsEqual(window_shape, out_tensor_type->shape_) && IsAllZeroOffsets(base_offsets);

    auto output_window_shape = std::move(window_shape);
    auto output_base_offsets = std::move(base_offsets);
    auto output_local_offsets = std::move(local_zero_offsets);
    auto output_piece = MakeDensePiece(output_window_shape, output_base_offsets, output_local_offsets);
    analysis.outputs.push_back(OutputRewriteInfo{match.out_param_index,
                                                 match.return_index,
                                                 out_tensor_type->shape_,
                                                 std::move(output_window_shape),
                                                 std::move(output_base_offsets),
                                                 std::move(output_local_offsets),
                                                 MakeDenseRegion({std::move(output_piece)}),
                                                 {},
                                                 match.iter_arg_index,
                                                 covers_full_parent});
  }

  if (analysis.outputs.empty()) return std::nullopt;

  analysis.inputs = existing_inputs;
  auto aggregate_inputs = AnalyzeAggregateInputWindows(func, existing_inputs, loop);
  analysis.inputs.insert(analysis.inputs.end(), std::make_move_iterator(aggregate_inputs.begin()),
                         std::make_move_iterator(aggregate_inputs.end()));
  return analysis;
}

/// True when every `out_indices` entry has an aggregate output covering its
/// whole parent -- the callee writes its outputs wholesale, so only its inputs
/// are worth windowing.
bool AllAggregateOutputsCoverFullParent(const std::optional<CalleeRewriteAnalysis>& analysis,
                                        const std::vector<size_t>& out_indices) {
  if (!analysis.has_value() || analysis->outputs.size() != out_indices.size()) return false;

  for (const auto& out_index : out_indices) {
    auto it = std::find_if(
        analysis->outputs.begin(), analysis->outputs.end(),
        [out_index](const OutputRewriteInfo& info) { return info.out_param_index == out_index; });
    if (it == analysis->outputs.end() || !CoversFullParent(*it)) return false;
  }
  return true;
}

AnalysisMap Analyze(const ProgramPtr& program) {
  AnalysisMap analyses;
  for (const auto& [gvar, func] : program->functions_) {
    if (!func || op_predicates::IsBuiltinOp(func->name_) || !IsInCoreType(func->func_type_)) {
      continue;
    }

    if (!IsWindowizeEnabled(func)) continue;
    auto out_indices = CollectOutParamIndices(func);
    auto input_windows = AnalyzeInputWindows(func);
    if (out_indices.empty()) {
      if (!input_windows.empty()) {
        CalleeRewriteAnalysis analysis;
        analysis.kind = RewriteKind::FinalStore;
        analysis.inputs = std::move(input_windows);
        analyses.emplace(func->name_, std::move(analysis));
      }
      continue;
    }

    CalleeRewriteAnalysis analysis;
    for (const auto& out_index : out_indices) {
      auto info = AnalyzeFinalStore(func, out_index);
      if (!info.has_value()) {
        continue;
      }

      auto out_tensor_type = As<TensorType>(func->params_[out_index]->GetType());
      if (!out_tensor_type) {
        continue;
      }
      if (AreExprVectorsEqual(info->window_shape, out_tensor_type->shape_) &&
          IsAllZeroOffsets(info->offsets)) {
        continue;
      }

      auto allowed_params = CollectAllowedVars(func->params_);
      if (!ExprsReferenceOnlyVarsIn(info->window_shape, allowed_params) ||
          !ExprsReferenceOnlyVarsIn(info->offsets, allowed_params)) {
        continue;
      }

      std::vector<ExprPtr> local_zero_offsets;
      local_zero_offsets.reserve(info->offsets.size());
      for (size_t i = 0; i < info->offsets.size(); ++i) {
        local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
      }
      auto output_piece = MakeDensePiece(info->window_shape, info->offsets, local_zero_offsets);
      analysis.outputs.push_back(OutputRewriteInfo{out_index,
                                                   info->return_index,
                                                   out_tensor_type->shape_,
                                                   info->window_shape,
                                                   info->offsets,
                                                   local_zero_offsets,
                                                   MakeDenseRegion({std::move(output_piece)}),
                                                   {},
                                                   SIZE_MAX});
    }
    if (!analysis.outputs.empty()) {
      analysis.kind = RewriteKind::FinalStore;
      analysis.inputs = std::move(input_windows);
      analyses.emplace(func->name_, std::move(analysis));
      continue;
    }

    auto aggregate_analysis = AnalyzeAggregateWindowLoop(func, out_indices, input_windows);
    // Read the pure-input-window verdict off the superset before narrowing it to
    // the windowable outputs -- both answers come from this one traversal.
    const bool outputs_are_wholesale = AllAggregateOutputsCoverFullParent(aggregate_analysis, out_indices);
    if (aggregate_analysis.has_value()) {
      auto& outputs = aggregate_analysis->outputs;
      outputs.erase(std::remove_if(outputs.begin(), outputs.end(),
                                   [](const OutputRewriteInfo& output) {
                                     return output.dense_window_covers_full_parent;
                                   }),
                    outputs.end());
      if (!outputs.empty()) {
        analyses.emplace(func->name_, std::move(*aggregate_analysis));
        continue;
      }
    }

    if (!input_windows.empty() &&
        (HasOnlyFullShapeZeroOffsetReturnOutputs(func, out_indices) || outputs_are_wholesale)) {
      CalleeRewriteAnalysis input_only_analysis;
      input_only_analysis.kind = RewriteKind::FinalStore;
      input_only_analysis.inputs = std::move(input_windows);
      analyses.emplace(func->name_, std::move(input_only_analysis));
      continue;
    }
  }
  return analyses;
}

void ApplyWindowRewritePolicy(const ProgramPtr& program, AnalysisMap* analyses) {
  if (!analyses) return;
  auto function_lookup = BuildFunctionLookup(program);
  for (auto it = analyses->begin(); it != analyses->end();) {
    const auto& callee_name = it->first;
    auto& analysis = it->second;
    auto func_it = function_lookup.find(callee_name);
    auto func = func_it == function_lookup.end() ? nullptr : func_it->second;

    if (!IsWindowizeEnabled(func)) {
      it = analyses->erase(it);
      continue;
    }

    // Type/ABI safety filter (always applies).
    analysis.outputs.erase(
        std::remove_if(analysis.outputs.begin(), analysis.outputs.end(),
                       [&](const OutputRewriteInfo& output) {
                         if (!func || output.out_param_index >= func->params_.size()) return true;
                         auto tensor_type = As<TensorType>(func->params_[output.out_param_index]->GetType());
                         return !CanMaterializeOutputWindowParamType(tensor_type, output.window_shape) ||
                                !CanWindowOutputWithinDynamicParent(tensor_type, output.window_shape,
                                                                    output.callsite_offsets);
                       }),
        analysis.outputs.end());
    analysis.inputs.erase(
        std::remove_if(analysis.inputs.begin(), analysis.inputs.end(),
                       [&](const InputRewriteInfo& input) {
                         if (!func || input.in_param_index >= func->params_.size()) return true;
                         auto tensor_type = As<TensorType>(func->params_[input.in_param_index]->GetType());
                         return !CanMaterializeWindowParamType(tensor_type, input.window_shape);
                       }),
        analysis.inputs.end());

    if (analysis.outputs.empty() && analysis.inputs.empty()) {
      it = analyses->erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace

AnalysisMap AnalyzeProgram(const ProgramPtr& program) {
  auto analyses = Analyze(program);
  ApplyWindowRewritePolicy(program, &analyses);
  return analyses;
}

}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto
