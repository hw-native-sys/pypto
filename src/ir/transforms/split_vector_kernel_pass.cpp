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

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kDualAivDispatchAttr = "dual_aiv_dispatch";

int SplitDimension(SplitMode mode) { return (mode == SplitMode::UpDown) ? 0 : 1; }

bool RequiresNoSplitDualAivSync(const FunctionPtr& func) {
  return func != nullptr && func->func_type_ == FunctionType::AIV &&
         pypto::backend::BackendConfig::IsConfigured() &&
         pypto::ir::PassContext::Current()->GetBackendHandler()->RequiresNoSplitDualAivDispatch() &&
         func->HasAttr(kDualAivDispatchAttr) && func->GetAttr<bool>(kDualAivDispatchAttr, false);
}

bool IsCrossCoreSplitOp(const std::string& op_name) {
  return op_name == "tile.tpush_to_aiv" || op_name == "tile.tpush_to_aic" ||
         op_name == "tile.tpop_from_aiv" || op_name == "tile.tpop_from_aic";
}

std::optional<SplitMode> SplitModeFromInt(int split) {
  if (split == 0) return std::nullopt;
  if (split == 1) return SplitMode::UpDown;
  if (split == 2) return SplitMode::LeftRight;
  throw pypto::ValueError("SplitVectorKernel found invalid cross-core split attribute: " +
                          std::to_string(split));
}

class CrossCoreSplitCollector : public IRVisitor {
 public:
  [[nodiscard]] std::optional<SplitMode> GetInferredMode() const { return inferred_mode_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    ConsiderCall(As<Call>(op->value_));
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    ConsiderCall(As<Call>(op->expr_));
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::optional<SplitMode> inferred_mode_;

  void ConsiderCall(const CallPtr& call) {
    if (!call || !call->op_ || !IsCrossCoreSplitOp(call->op_->name_)) return;

    auto mode = SplitModeFromInt(call->GetKwarg<int>("split", 0));
    if (!mode.has_value()) return;

    if (!inferred_mode_.has_value()) {
      inferred_mode_ = mode;
      return;
    }

    if (inferred_mode_.value() != mode.value()) {
      throw pypto::ValueError("SplitVectorKernel found conflicting cross-core split modes in function body");
    }
  }
};

std::optional<SplitMode> ResolveSplitMode(const FunctionPtr& func) {
  CrossCoreSplitCollector collector;
  if (func->body_) {
    collector.VisitStmt(func->body_);
  }
  auto inferred_mode = collector.GetInferredMode();

  auto func_split_mode = func->GetSplitMode();
  if (func_split_mode.has_value() && func_split_mode.value() != SplitMode::None) {
    if (inferred_mode.has_value() && inferred_mode.value() != func_split_mode.value()) {
      throw pypto::ValueError("SplitVectorKernel found conflicting function split and cross-core op split");
    }
    return func_split_mode;
  }

  return inferred_mode;
}

std::vector<std::pair<std::string, std::any>> WithSplitAttr(const FunctionPtr& func, SplitMode mode) {
  auto attrs = func->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(), [](const auto& kv) { return kv.first == "split"; }),
              attrs.end());
  if (mode != SplitMode::None) {
    attrs.emplace_back("split", static_cast<int>(mode));
  }
  return attrs;
}

bool IsSingletonDim(const ExprPtr& dim_size) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size)) {
    return ci->value_ == 1;
  }
  return false;
}

bool IsReduceOnSplitAxis(const CallPtr& call, int split_dim) {
  if (!call->op_) return false;
  const auto& name = call->op_->name_;

  auto input_tile_type = [&]() -> std::shared_ptr<const TileType> {
    if (call->args_.empty()) return nullptr;
    return std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
  };

  if (name == "tile.row_sum" || name == "tile.row_max" || name == "tile.row_min") {
    auto tt = input_tile_type();
    int last_axis = tt ? static_cast<int>(tt->shape_.size()) - 1 : 1;
    return split_dim == last_axis;
  }
  if (name == "tile.sum" || name == "tile.max" || name == "tile.min") {
    int axis = call->GetKwarg<int>("axis", -1);
    auto tt = input_tile_type();
    if (axis < 0 && tt) {
      axis = static_cast<int>(tt->shape_.size()) + axis;
    }
    return axis == split_dim;
  }
  return false;
}

ExprPtr ComputeHalfDimSize(const ExprPtr& dim_size) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size)) {
    if ((ci->value_ % 2) != 0) {
      throw pypto::ValueError("SplitVectorKernel requires an even split dimension, got " +
                              std::to_string(ci->value_));
    }
    return std::make_shared<ConstInt>(ci->value_ / 2, ci->dtype(), ci->span_);
  }
  auto two = std::make_shared<ConstInt>(2, GetScalarDtype(dim_size), dim_size->span_);
  return MakeFloorDiv(dim_size, two, dim_size->span_);
}

ExprPtr MakeConstLike(const ExprPtr& ref, int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, GetScalarDtype(ref), span);
}

ExprPtr MakeIndexConst(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr MakeZeroTuple(size_t rank, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    elements.push_back(MakeIndexConst(0, span));
  }
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

ExprPtr MakeZeroTupleLike(const ExprPtr& tuple_expr, const Span& span) {
  if (auto tuple = std::dynamic_pointer_cast<const MakeTuple>(tuple_expr)) {
    return MakeZeroTuple(tuple->elements_.size(), span);
  }
  return tuple_expr;
}

TypePtr WithZeroValidShape(const TypePtr& type, const Span& span) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt) return type;

  TileView tile_view = tt->tile_view_.value_or(TileView{});
  tile_view.valid_shape.clear();
  tile_view.valid_shape.reserve(tt->shape_.size());
  for (size_t i = 0; i < tt->shape_.size(); ++i) {
    tile_view.valid_shape.push_back(MakeIndexConst(0, span));
  }

  return std::make_shared<TileType>(tt->shape_, tt->dtype_, tt->memref_, std::move(tile_view),
                                    tt->memory_space_);
}

ExprPtr LocalizeValidDimForSplit(const ExprPtr& valid_dim, const ExprPtr& original_dim,
                                 const ExprPtr& half_dim_size, const ExprPtr& subblock_idx) {
  if (!valid_dim) return valid_dim;
  if (!subblock_idx) {
    return half_dim_size;
  }
  if (AreExprsEqual(valid_dim, original_dim)) {
    return half_dim_size;
  }

  auto span = valid_dim->span_;
  auto zero = MakeConstLike(valid_dim, 0, span);
  auto subblock_offset = MakeMul(subblock_idx, half_dim_size, span);
  auto remaining = MakeSub(valid_dim, subblock_offset, span);
  return MakeMax(MakeMin(remaining, half_dim_size, span), zero, span);
}

CallPtr RebuildCallWithSplit(const CallPtr& call, int split_int) {
  std::vector<std::pair<std::string, std::any>> new_kwargs;
  bool has_split = false;
  for (const auto& [key, val] : call->kwargs_) {
    if (key == "split") {
      new_kwargs.emplace_back("split", std::any(split_int));
      has_split = true;
    } else {
      new_kwargs.emplace_back(key, val);
    }
  }
  if (!has_split) {
    new_kwargs.emplace_back("split", std::any(split_int));
  }
  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), call->GetType(), call->span_);
}

TypePtr HalveTileShape(const TypePtr& type, int dim, const ExprPtr& subblock_idx) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = ComputeHalfDimSize(tt->shape_[dim]);

  // Keep TileView.valid_shape consistent with halved physical shape, and for
  // partial valid regions localize the split dimension to the current subblock.
  std::optional<TileView> new_tile_view = tt->tile_view_;
  if (const auto& tile_view = tt->tile_view_; tile_view.has_value()) {
    TileView tv = tile_view.value();
    if (dim < static_cast<int>(tv.valid_shape.size())) {
      tv.valid_shape[dim] =
          LocalizeValidDimForSplit(tv.valid_shape[dim], tt->shape_[dim], new_shape[dim], subblock_idx);
    }
    new_tile_view = std::move(tv);
  }

  return std::make_shared<TileType>(new_shape, tt->dtype_, tt->memref_, new_tile_view, tt->memory_space_);
}

ExprPtr HalveTupleElement(const ExprPtr& tuple_expr, int dim) {
  auto tuple = std::dynamic_pointer_cast<const MakeTuple>(tuple_expr);
  if (!tuple || dim < 0 || dim >= static_cast<int>(tuple->elements_.size())) return tuple_expr;
  std::vector<ExprPtr> new_elements = tuple->elements_;
  new_elements[dim] = ComputeHalfDimSize(new_elements[dim]);
  return std::make_shared<MakeTuple>(std::move(new_elements), tuple_expr->span_);
}

ExprPtr LocalizeTupleElementForSplit(const ExprPtr& tuple_expr, int dim, const ExprPtr& original_dim,
                                     const ExprPtr& half_dim_size, const ExprPtr& subblock_idx) {
  auto tuple = std::dynamic_pointer_cast<const MakeTuple>(tuple_expr);
  if (!tuple || dim < 0 || dim >= static_cast<int>(tuple->elements_.size())) return tuple_expr;
  std::vector<ExprPtr> new_elements = tuple->elements_;
  new_elements[dim] =
      LocalizeValidDimForSplit(tuple->elements_[dim], original_dim, half_dim_size, subblock_idx);
  return std::make_shared<MakeTuple>(std::move(new_elements), tuple_expr->span_);
}

CallPtr RebuildTpopWithHalvedShape(const CallPtr& call, int split_int, int split_dim,
                                   const ExprPtr& subblock_idx) {
  auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx);

  std::vector<std::pair<std::string, std::any>> new_kwargs;
  bool has_split = false;
  for (const auto& [key, val] : call->kwargs_) {
    if (key == "split") {
      new_kwargs.emplace_back("split", std::any(split_int));
      has_split = true;
    } else {
      new_kwargs.emplace_back(key, val);
    }
  }
  if (!has_split) {
    new_kwargs.emplace_back("split", std::any(split_int));
  }

  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), new_result_type, call->span_);
}

struct TileInfo {
  ExprPtr half_dim_size;
};

ExprPtr AdjustOffsets(const ExprPtr& offsets_expr, int split_dim, const ExprPtr& half_size,
                      const ExprPtr& subblock_idx) {
  auto offsets = std::dynamic_pointer_cast<const MakeTuple>(offsets_expr);
  if (!offsets || split_dim < 0 || split_dim >= static_cast<int>(offsets->elements_.size())) {
    return offsets_expr;
  }

  std::vector<ExprPtr> new_elements = offsets->elements_;
  auto original_offset = offsets->elements_[split_dim];

  ExprPtr adjusted;
  if (auto subblock_const = std::dynamic_pointer_cast<const ConstInt>(subblock_idx)) {
    if (subblock_const->value_ == 0) {
      adjusted = original_offset;
    } else if (subblock_const->value_ == 1) {
      if (auto original_const = std::dynamic_pointer_cast<const ConstInt>(original_offset);
          original_const && original_const->value_ == 0) {
        adjusted = half_size;
      } else {
        adjusted = MakeAdd(original_offset, half_size, original_offset->span_);
      }
    }
  }

  if (!adjusted) {
    // offset = original + get_subblock_idx() * half_size
    auto adjustment = MakeMul(subblock_idx, half_size, original_offset->span_);
    adjusted = MakeAdd(original_offset, adjustment, original_offset->span_);
  }
  new_elements[split_dim] = adjusted;

  return std::make_shared<MakeTuple>(std::move(new_elements), offsets->span_);
}

TypePtr ApplyTrackedTileShape(const TypePtr& type, int dim, const ExprPtr& half_dim_size,
                              const ExprPtr& subblock_idx) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = half_dim_size;

  std::optional<TileView> new_tile_view = tt->tile_view_;
  if (const auto& tile_view = tt->tile_view_; tile_view.has_value()) {
    TileView tv = tile_view.value();
    if (dim < static_cast<int>(tv.valid_shape.size())) {
      tv.valid_shape[dim] =
          LocalizeValidDimForSplit(tv.valid_shape[dim], tt->shape_[dim], half_dim_size, subblock_idx);
    }
    new_tile_view = std::move(tv);
  }

  return std::make_shared<TileType>(new_shape, tt->dtype_, tt->memref_, new_tile_view, tt->memory_space_);
}

std::vector<StmtPtr> ProcessStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_int,
                                  int split_dim, std::unordered_map<const Var*, TileInfo>& tile_vars,
                                  bool is_aiv, const ExprPtr& subblock_idx,
                                  std::unordered_map<const Var*, VarPtr>& var_replacements);

StmtPtr ProcessStmt(const StmtPtr& stmt, SplitMode mode, int split_int, int split_dim,
                    std::unordered_map<const Var*, TileInfo>& tile_vars, bool is_aiv,
                    const ExprPtr& subblock_idx, std::unordered_map<const Var*, VarPtr>& var_replacements) {
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (!call || !call->op_) return stmt;

    const auto& op_name = call->op_->name_;

    if (op_name == "tile.tpush_to_aiv" || op_name == "tile.tpush_to_aic") {
      auto new_call = RebuildCallWithSplit(call, split_int);
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }

    // tpop_from_aic: AIV consumes from cube — halve the popped tile to match split vector lanes.
    // tpop_from_aiv: AIC consumes from vector — keep full tile shape; only sync split attribute
    // (vector-side split affects AIV compute, not the matmul operand tile delivered to cube).
    if (op_name == "tile.tpop_from_aiv") {
      auto new_call = RebuildCallWithSplit(call, split_int);
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }
    if (op_name == "tile.tpop_from_aic") {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      auto new_call = RebuildTpopWithHalvedShape(call, split_int, split_dim, subblock_idx);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      if (tt && split_dim < static_cast<int>(tt->shape_.size())) {
        TileInfo info{ComputeHalfDimSize(tt->shape_[split_dim])};
        tile_vars[assign->var_.get()] = info;
        tile_vars[new_var.get()] = info;
      }
      var_replacements[assign->var_.get()] = new_var;
      return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
    }

    // AIV only: tile.load — halve result shape, halve shape/valid_shape args, adjust offset.
    // Singleton split-dim tiles (e.g. broadcast [1, 128] under UP_DOWN) are preserved as-is.
    if (is_aiv && op_name == "tile.load" && call->args_.size() >= 4) {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      bool is_singleton =
          tt && split_dim < static_cast<int>(tt->shape_.size()) && IsSingletonDim(tt->shape_[split_dim]);

      if (is_singleton) {
        return stmt;
      }

      // Rank-deficient tiles (for example rank-1 loads under LEFT_RIGHT) do
      // not carry the split axis, so they must bypass split-specific rewrites.
      if (!tt || split_dim >= static_cast<int>(tt->shape_.size())) {
        return stmt;
      }
      ExprPtr half_dim_size = ComputeHalfDimSize(tt->shape_[split_dim]);

      auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[1] = AdjustOffsets(call->args_[1], split_dim, half_dim_size, subblock_idx);
      new_args[2] = HalveTupleElement(call->args_[2], split_dim);
      new_args[3] = LocalizeTupleElementForSplit(call->args_[3], split_dim, tt->shape_[split_dim],
                                                 half_dim_size, subblock_idx);

      auto new_call =
          std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type, call->span_);
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
      TileInfo info{half_dim_size};
      tile_vars[assign->var_.get()] = info;
      tile_vars[new_var.get()] = info;
      var_replacements[assign->var_.get()] = new_var;
      return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
    }

    // AIV only: tile.store — adjust offset using tracked tile info
    if (is_aiv && op_name == "tile.store" && call->args_.size() >= 3) {
      auto tile_var = std::dynamic_pointer_cast<const Var>(call->args_[0]);
      if (tile_var) {
        auto it = tile_vars.find(tile_var.get());
        if (it != tile_vars.end()) {
          auto new_offsets = AdjustOffsets(call->args_[1], split_dim, it->second.half_dim_size, subblock_idx);
          std::vector<ExprPtr> new_args = call->args_;
          new_args[1] = new_offsets;
          auto new_call = std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_,
                                                 call->GetType(), call->span_);
          return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
        }
      }
    }

    // AIV only: any other op producing TileType — halve result shape (and static shape args when present).
    // Reject reduce ops that reduce on the split axis (partial reduction is semantically incorrect).
    // Skip halving when the output split-dim is singleton (broadcast / degenerate tiles).
    if (is_aiv) {
      if (IsReduceOnSplitAxis(call, split_dim)) {
        throw pypto::ValueError("SplitVectorKernel: reduce op '" + op_name +
                                "' reduces on the split axis (dim " + std::to_string(split_dim) +
                                "); partial reduction in a split kernel is not supported");
      }

      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      if (tt && split_dim < static_cast<int>(tt->shape_.size())) {
        if (IsSingletonDim(tt->shape_[split_dim])) {
          return stmt;
        }
        auto half_dim_size = ComputeHalfDimSize(tt->shape_[split_dim]);
        auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx);
        std::vector<ExprPtr> new_args = call->args_;
        if ((op_name == "tile.full" || op_name == "tile.create") && call->args_.size() >= 1) {
          new_args[0] = HalveTupleElement(call->args_[0], split_dim);
        }
        auto new_call = std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type,
                                               call->span_);
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
        TileInfo info{half_dim_size};
        tile_vars[assign->var_.get()] = info;
        tile_vars[new_var.get()] = info;
        var_replacements[assign->var_.get()] = new_var;
        return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
      }
    }

    return stmt;
  }

  if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    if (!call || !call->op_) return stmt;

    const auto& op_name = call->op_->name_;

    if (op_name == "tile.tpush_to_aiv" || op_name == "tile.tpush_to_aic") {
      auto new_call = RebuildCallWithSplit(call, split_int);
      return std::make_shared<EvalStmt>(new_call, eval->span_);
    }

    if (is_aiv && op_name == "tile.store" && call->args_.size() >= 3) {
      auto tile_var = std::dynamic_pointer_cast<const Var>(call->args_[0]);
      if (tile_var) {
        auto it = tile_vars.find(tile_var.get());
        if (it != tile_vars.end()) {
          auto new_offsets = AdjustOffsets(call->args_[1], split_dim, it->second.half_dim_size, subblock_idx);
          std::vector<ExprPtr> new_args = call->args_;
          new_args[1] = new_offsets;
          auto new_call = std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_,
                                                 call->GetType(), call->span_);
          return std::make_shared<EvalStmt>(new_call, eval->span_);
        }
      }
    }

    return stmt;
  }

  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    // Eagerly substitute initValues while rebuilding iter_args. If this is
    // deferred to the final Substitute pass, it can create a second IterArg
    // instance whose pointer diverges from the one referenced by the rebuilt
    // loop body, breaking structural equality.
    std::vector<IterArgPtr> new_iter_args;
    new_iter_args.reserve(for_stmt->iter_args_.size());
    std::vector<VarPtr> new_return_vars = for_stmt->return_vars_;

    // Propagate tile_vars from init values to iter_args BEFORE processing body.
    // Iter_args carry the init_value into the loop; if the init is a tracked
    // halved tile, the iter_arg must also be tracked so that operations on it
    // inside the loop body are correctly recognized.
    for (const auto& ia : for_stmt->iter_args_) {
      auto new_init_value = ia->initValue_;
      if (new_init_value && !var_replacements.empty()) {
        new_init_value = transform_utils::Substitute(new_init_value, var_replacements);
      }
      TypePtr new_type = ia->GetType();
      bool has_tracked_tile = false;
      TileInfo tracked_info;
      if (ia->initValue_) {
        if (auto init_var = AsVarLike(ia->initValue_)) {
          auto it = tile_vars.find(init_var.get());
          if (it != tile_vars.end()) {
            has_tracked_tile = true;
            tracked_info = it->second;
            tile_vars[ia.get()] = it->second;
            new_type =
                ApplyTrackedTileShape(ia->GetType(), split_dim, it->second.half_dim_size, subblock_idx);
          }
        }
      }

      if (new_type != ia->GetType() || new_init_value != ia->initValue_) {
        auto new_iter_arg = std::make_shared<IterArg>(ia->name_hint_, new_type, new_init_value, ia->span_);
        new_iter_args.push_back(new_iter_arg);
        var_replacements[ia.get()] = new_iter_arg;
        if (has_tracked_tile) {
          tile_vars[new_iter_arg.get()] = tracked_info;
        }
      } else {
        new_iter_args.push_back(ia);
      }
    }

    auto flat = std::vector<StmtPtr>();
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(for_stmt->body_)) {
      flat = seq->stmts_;
    } else {
      flat.push_back(for_stmt->body_);
    }
    auto new_body_stmts =
        ProcessStmts(flat, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx, var_replacements);
    StmtPtr new_body = (new_body_stmts.size() == 1)
                           ? new_body_stmts[0]
                           : std::make_shared<SeqStmts>(new_body_stmts, for_stmt->span_);

    // Propagate tile_vars tracking from iter_args to return_vars.
    // ForStmt return_vars are the loop-exit versions of the corresponding
    // iter_args.  If an iter_arg carries a halved tile, the return_var must
    // inherit the tile info so that downstream tile.store gets the correct
    // subblock offset adjustment.
    INTERNAL_CHECK_SPAN(for_stmt->iter_args_.size() == for_stmt->return_vars_.size(), for_stmt->span_)
        << "Internal error: ForStmt iter_args and return_vars sizes must match, got "
        << for_stmt->iter_args_.size() << " vs " << for_stmt->return_vars_.size();
    for (size_t i = 0; i < new_iter_args.size() && i < new_return_vars.size(); ++i) {
      auto it = tile_vars.find(new_iter_args[i].get());
      if (it != tile_vars.end()) {
        tile_vars[new_return_vars[i].get()] = it->second;
        auto new_type = ApplyTrackedTileShape(new_return_vars[i]->GetType(), split_dim,
                                              it->second.half_dim_size, subblock_idx);
        if (new_type != new_return_vars[i]->GetType()) {
          auto new_return_var =
              std::make_shared<Var>(new_return_vars[i]->name_hint_, new_type, new_return_vars[i]->span_);
          new_return_vars[i] = new_return_var;
          tile_vars[new_return_var.get()] = it->second;
          var_replacements[for_stmt->return_vars_[i].get()] = new_return_var;
        }
      }
    }

    return loop_repair::RebuildForStmt(for_stmt, new_iter_args, new_body, new_return_vars);
  }

  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto then_flat = std::vector<StmtPtr>();
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(if_stmt->then_body_)) {
      then_flat = seq->stmts_;
    } else {
      then_flat.push_back(if_stmt->then_body_);
    }
    auto new_then = ProcessStmts(then_flat, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx,
                                 var_replacements);
    StmtPtr new_then_body =
        (new_then.size() == 1) ? new_then[0] : std::make_shared<SeqStmts>(new_then, if_stmt->span_);

    std::optional<StmtPtr> new_else;
    if (const auto& else_body_opt = if_stmt->else_body_; else_body_opt.has_value()) {
      const StmtPtr& else_body = else_body_opt.value();
      auto else_flat = std::vector<StmtPtr>();
      if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(else_body)) {
        else_flat = seq->stmts_;
      } else {
        else_flat.push_back(else_body);
      }
      auto new_else_stmts = ProcessStmts(else_flat, mode, split_int, split_dim, tile_vars, is_aiv,
                                         subblock_idx, var_replacements);
      new_else = (new_else_stmts.size() == 1) ? new_else_stmts[0]
                                              : std::make_shared<SeqStmts>(new_else_stmts, if_stmt->span_);
    }
    auto new_if = MutableCopy(if_stmt);
    new_if->then_body_ = new_then_body;
    new_if->else_body_ = new_else;
    return new_if;
  }

  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    auto new_stmts = ProcessStmts(seq->stmts_, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx,
                                  var_replacements);
    return std::make_shared<SeqStmts>(new_stmts, seq->span_);
  }

  return stmt;
}

std::vector<StmtPtr> ProcessStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_int,
                                  int split_dim, std::unordered_map<const Var*, TileInfo>& tile_vars,
                                  bool is_aiv, const ExprPtr& subblock_idx,
                                  std::unordered_map<const Var*, VarPtr>& var_replacements) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    result.push_back(
        ProcessStmt(stmt, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx, var_replacements));
  }
  return result;
}

struct SubblockInjectionResult {
  ExprPtr subblock_idx_expr;
  std::vector<StmtPtr> body_stmts;
  std::unordered_set<std::string> used_names;
};

std::string ReserveFreshName(std::unordered_set<std::string>& used_names, const std::string& base_name) {
  std::string name = base_name;
  if (used_names.count(name) != 0) {
    name = auto_name::GenerateFreshNameLike(base_name, used_names);
  }
  used_names.insert(name);
  return name;
}

SubblockInjectionResult InjectSubblockIdx(const FunctionPtr& func, bool is_aiv) {
  std::vector<StmtPtr> body_stmts;
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(func->body_)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(func->body_);
  }

  std::unordered_set<std::string> used_names;
  for (const auto& p : func->params_) {
    used_names.insert(p->name_hint_);
  }
  std::vector<VarPtr> def_vars;
  transform_utils::CollectDefVars(func->body_, def_vars);
  for (const auto& v : def_vars) {
    used_names.insert(v->name_hint_);
  }

  if (!is_aiv) {
    return {nullptr, std::move(body_stmts), std::move(used_names)};
  }

  auto idx_type = std::make_shared<ScalarType>(DataType::INDEX);
  std::string subblock_var_name = ReserveFreshName(used_names, "subblock_idx");

  auto& op_reg = OpRegistry::GetInstance();
  auto subblock_op = op_reg.GetOp("tile.get_subblock_idx");
  auto subblock_call =
      std::make_shared<Call>(subblock_op, std::vector<ExprPtr>{},
                             std::vector<std::pair<std::string, std::any>>{}, idx_type, func->span_);
  auto subblock_idx_var = std::make_shared<Var>(subblock_var_name, idx_type, func->span_);
  auto assign_stmt = std::make_shared<AssignStmt>(subblock_idx_var, subblock_call, func->span_);
  body_stmts.insert(body_stmts.begin(), assign_stmt);
  return {subblock_idx_var, std::move(body_stmts), std::move(used_names)};
}

using ExprReplacementMap = std::unordered_map<const Var*, ExprPtr>;

ExprPtr SubstituteExprIfNeeded(const ExprPtr& expr, const ExprReplacementMap& replacements) {
  if (replacements.empty() || !expr) return expr;
  return transform_utils::Substitute(expr, replacements);
}

StmtPtr SubstituteStmtIfNeeded(const StmtPtr& stmt, const ExprReplacementMap& replacements) {
  if (replacements.empty() || !stmt) return stmt;
  return transform_utils::Substitute(stmt, replacements);
}

CallPtr RebuildLane1CallWithZeroValidShape(const CallPtr& call, const ExprReplacementMap& replacements) {
  std::vector<ExprPtr> new_args;
  new_args.reserve(call->args_.size());
  bool args_changed = false;
  for (const auto& arg : call->args_) {
    auto new_arg = SubstituteExprIfNeeded(arg, replacements);
    args_changed = args_changed || (new_arg != arg);
    new_args.push_back(new_arg);
  }

  const std::string& op_name = call->op_->name_;
  if (op_name == "tile.load" && new_args.size() >= 3) {
    auto new_type = WithZeroValidShape(call->GetType(), call->span_);
    auto tile_type = std::dynamic_pointer_cast<const TileType>(new_type);
    if (!tile_type) return call;

    std::vector<std::pair<std::string, std::any>> kwargs;
    kwargs.emplace_back("dtype", tile_type->dtype_);
    if (const auto& memory_space = tile_type->memory_space_) {
      kwargs.emplace_back("target_memory", *memory_space);
    }

    auto create_op = OpRegistry::GetInstance().GetOp("tile.create");
    return std::make_shared<Call>(create_op, std::vector<ExprPtr>{new_args[2]}, std::move(kwargs), new_type,
                                  call->span_);
  } else if (op_name == "tile.slice" && new_args.size() >= 3) {
    auto zero_valid_shape = MakeZeroTupleLike(new_args[1], call->span_);
    if (new_args.size() >= 4) {
      new_args[3] = zero_valid_shape;
    } else {
      new_args.push_back(zero_valid_shape);
    }
    args_changed = true;
  } else if (op_name == "tile.set_validshape" && new_args.size() == 3) {
    new_args[1] = MakeIndexConst(0, call->span_);
    new_args[2] = MakeIndexConst(0, call->span_);
    args_changed = true;
  }

  auto new_type = WithZeroValidShape(call->GetType(), call->span_);
  bool type_changed = new_type != call->GetType();
  if (!args_changed && !type_changed) return call;
  return std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_type, call->span_);
}

bool IsNoSplitSharedPipeSetupCall(const CallPtr& call) {
  if (!call || !call->op_) return false;
  const std::string& op_name = call->op_->name_;
  return op_name == "system.reserve_buffer" || op_name == "system.import_peer_buffer" ||
         op_name == "system.aiv_initialize_pipe" || op_name == "system.aic_initialize_pipe";
}

bool IsNoSplitSharedPipeSetupStmt(const StmtPtr& stmt) {
  CallPtr call;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  return IsNoSplitSharedPipeSetupCall(call);
}

struct NoSplitSharedPrefix {
  std::vector<StmtPtr> shared_setup_stmts;
  std::vector<StmtPtr> branch_stmts;
};

NoSplitSharedPrefix SplitNoSplitSharedPipeSetupPrefix(const std::vector<StmtPtr>& stmts) {
  NoSplitSharedPrefix result;
  size_t prefix_len = 0;
  while (prefix_len < stmts.size() && IsNoSplitSharedPipeSetupStmt(stmts[prefix_len])) {
    result.shared_setup_stmts.push_back(stmts[prefix_len]);
    ++prefix_len;
  }
  result.branch_stmts.insert(result.branch_stmts.end(),
                             stmts.begin() + static_cast<std::ptrdiff_t>(prefix_len), stmts.end());
  return result;
}

// Lane 1 still needs to replay cross-core handshakes so Ascend910B GM-backed
// pipes stay balanced. Tile-producing replay work is forced to static
// valid_shape=0, letting PTO ops run as empty tiles while preserving the tpush /
// tpop / tfree synchronization protocol. The only visible side effects we
// intentionally suppress are tile.store writes and their SSA results.
std::vector<StmtPtr> BuildNoSplitLane1ReplayStmts(const std::vector<StmtPtr>& stmts,
                                                  ExprReplacementMap& replacements) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());

  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
      if (!call || !call->op_) {
        result.push_back(SubstituteStmtIfNeeded(stmt, replacements));
        continue;
      }

      const std::string& op_name = call->op_->name_;
      if (op_name == "tile.store" && call->args_.size() >= 3) {
        auto passthrough = SubstituteExprIfNeeded(call->args_[2], replacements);
        replacements[assign->var_.get()] = passthrough;
        result.push_back(std::make_shared<AssignStmt>(assign->var_, passthrough, assign->span_));
        continue;
      }

      auto new_value = RebuildLane1CallWithZeroValidShape(call, replacements);
      if (std::dynamic_pointer_cast<const TileType>(new_value->GetType())) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        replacements[assign->var_.get()] = new_var;
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
      } else {
        result.push_back(std::make_shared<AssignStmt>(assign->var_, new_value, assign->span_));
      }
      continue;
    }

    if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
      if (!call || !call->op_) {
        result.push_back(SubstituteStmtIfNeeded(stmt, replacements));
        continue;
      }

      const std::string& op_name = call->op_->name_;
      if (op_name == "tile.store") {
        continue;
      }

      auto new_expr = RebuildLane1CallWithZeroValidShape(call, replacements);
      result.push_back(std::make_shared<EvalStmt>(new_expr, eval->span_));
      continue;
    }

    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto loop_replacements = replacements;
      auto new_body_stmts =
          BuildNoSplitLane1ReplayStmts(transform_utils::FlattenToStmts(for_stmt->body_), loop_replacements);
      auto new_for = MutableCopy(for_stmt);
      new_for->body_ = loop_repair::MakeBody(new_body_stmts, for_stmt->span_);
      result.push_back(new_for);
      continue;
    }

    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto loop_replacements = replacements;
      auto new_body_stmts =
          BuildNoSplitLane1ReplayStmts(transform_utils::FlattenToStmts(while_stmt->body_), loop_replacements);
      auto new_while = MutableCopy(while_stmt);
      new_while->condition_ = SubstituteExprIfNeeded(while_stmt->condition_, replacements);
      new_while->body_ = loop_repair::MakeBody(new_body_stmts, while_stmt->span_);
      result.push_back(new_while);
      continue;
    }

    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto then_replacements = replacements;
      auto new_then_stmts = BuildNoSplitLane1ReplayStmts(transform_utils::FlattenToStmts(if_stmt->then_body_),
                                                         then_replacements);
      std::optional<StmtPtr> new_else;
      if (const auto& else_body = if_stmt->else_body_) {
        auto else_replacements = replacements;
        auto new_else_stmts =
            BuildNoSplitLane1ReplayStmts(transform_utils::FlattenToStmts(*else_body), else_replacements);
        new_else = loop_repair::MakeBody(new_else_stmts, if_stmt->span_);
      }
      auto new_if = MutableCopy(if_stmt);
      new_if->condition_ = SubstituteExprIfNeeded(if_stmt->condition_, replacements);
      new_if->then_body_ = loop_repair::MakeBody(new_then_stmts, if_stmt->span_);
      new_if->else_body_ = new_else;
      result.push_back(new_if);
      continue;
    }

    result.push_back(SubstituteStmtIfNeeded(stmt, replacements));
  }

  return result;
}

FunctionPtr ProcessFunction(const FunctionPtr& func, SplitMode mode) {
  if (mode == SplitMode::None) {
    return func;
  }
  int split_int = static_cast<int>(mode);
  int split_dim = SplitDimension(mode);
  bool is_aiv = (func->func_type_ == FunctionType::AIV);

  std::unordered_map<const Var*, TileInfo> tile_vars;
  std::unordered_map<const Var*, VarPtr> var_replacements;
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  for (const auto& param : func->params_) {
    auto new_param = std::make_shared<Var>(param->name_hint_, param->GetType(), param->span_);
    new_params.push_back(new_param);
    var_replacements[param.get()] = new_param;
  }

  auto injected = InjectSubblockIdx(func, is_aiv);

  auto new_stmts = ProcessStmts(injected.body_stmts, mode, split_int, split_dim, tile_vars, is_aiv,
                                injected.subblock_idx_expr, var_replacements);
  StmtPtr new_body =
      (new_stmts.size() == 1) ? new_stmts[0] : std::make_shared<SeqStmts>(new_stmts, func->span_);
  if (!var_replacements.empty()) {
    new_body = transform_utils::Substitute(new_body, var_replacements);
  }
  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  auto new_func = MutableCopy(func);
  new_func->params_ = new_params;
  new_func->body_ = cloned_body;
  new_func->attrs_ = WithSplitAttr(func, mode);
  return new_func;
}

FunctionPtr ProcessNoSplitDualAivFunction(const FunctionPtr& func) {
  // Plain INTERNAL_CHECK: RequiresNoSplitDualAivSync short-circuits on null
  // func, so func->span_ would dereference null in the failure path.
  INTERNAL_CHECK(RequiresNoSplitDualAivSync(func))
      << "Internal error: ProcessNoSplitDualAivFunction requires dual-dispatch AIV marker";

  std::unordered_map<const Var*, ExprPtr> param_replacements;
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  for (const auto& param : func->params_) {
    auto new_param = std::make_shared<Var>(param->name_hint_, param->GetType(), param->span_);
    new_params.push_back(new_param);
    param_replacements[param.get()] = new_param;
  }

  auto injected = InjectSubblockIdx(func, /*is_aiv=*/true);
  INTERNAL_CHECK(!injected.body_stmts.empty())
      << "Internal error: dual-dispatch no-split AIV body must contain injected subblock_idx";
  std::vector<StmtPtr> guarded_stmts(injected.body_stmts.begin() + 1, injected.body_stmts.end());
  auto hoisted_prefix = SplitNoSplitSharedPipeSetupPrefix(guarded_stmts);
  auto lane0_body = loop_repair::MakeBody(hoisted_prefix.branch_stmts, func->span_);

  ExprReplacementMap lane1_replacements = param_replacements;
  auto lane1_stmts = BuildNoSplitLane1ReplayStmts(hoisted_prefix.branch_stmts, lane1_replacements);
  auto lane1_body = loop_repair::MakeBody(lane1_stmts, func->span_);
  auto [lane1_cloned_body, lane1_clone_map_unused] = DeepClone(lane1_body);
  (void)lane1_clone_map_unused;
  lane1_body = lane1_cloned_body;

  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, func->span_);
  auto lane0_cond = MakeEq(injected.subblock_idx_expr, zero, func->span_);
  auto branch_stmt = std::make_shared<IfStmt>(lane0_cond, lane0_body, std::make_optional(lane1_body),
                                              std::vector<VarPtr>{}, func->span_);

  std::vector<StmtPtr> new_body_stmts{injected.body_stmts.front()};
  new_body_stmts.insert(new_body_stmts.end(), hoisted_prefix.shared_setup_stmts.begin(),
                        hoisted_prefix.shared_setup_stmts.end());
  new_body_stmts.push_back(branch_stmt);
  StmtPtr new_body = loop_repair::MakeBody(new_body_stmts, func->span_);
  if (!param_replacements.empty()) {
    new_body = transform_utils::Substitute(new_body, param_replacements);
  }
  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  auto new_func = MutableCopy(func);
  new_func->params_ = new_params;
  new_func->body_ = cloned_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass SplitVectorKernel() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    bool changed = false;

    for (const auto& [gvar, func] : program->functions_) {
      auto resolved_mode = ResolveSplitMode(func);
      bool should_split = resolved_mode.has_value() && resolved_mode.value() != SplitMode::None &&
                          (func->func_type_ == FunctionType::AIV || func->func_type_ == FunctionType::AIC);
      bool should_dual_dispatch_nosplit = RequiresNoSplitDualAivSync(func);
      // Only process AIC and AIV functions that have a non-None split mode
      if (should_split) {
        new_functions.push_back(ProcessFunction(func, resolved_mode.value()));
        changed = true;
      } else if (should_dual_dispatch_nosplit) {
        new_functions.push_back(ProcessNoSplitDualAivFunction(func));
        changed = true;
      } else {
        new_functions.push_back(func);
      }
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "SplitVectorKernel", kSplitVectorKernelProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
