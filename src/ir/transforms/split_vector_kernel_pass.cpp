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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

int SplitDimension(SplitMode mode) { return (mode == SplitMode::UpDown) ? 0 : 1; }

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

TypePtr HalveTileShape(const TypePtr& type, int dim) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = ComputeHalfDimSize(tt->shape_[dim]);

  // Keep TileView.valid_shape consistent with halved physical shape (was left at pre-split size).
  std::optional<TileView> new_tile_view = tt->tile_view_;
  if (const auto& tile_view = tt->tile_view_; tile_view.has_value()) {
    TileView tv = tile_view.value();
    if (dim < static_cast<int>(tv.valid_shape.size())) {
      tv.valid_shape[dim] = ComputeHalfDimSize(tv.valid_shape[dim]);
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

CallPtr RebuildTpopWithHalvedShape(const CallPtr& call, int split_int, int split_dim) {
  auto new_result_type = HalveTileShape(call->GetType(), split_dim);

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
      auto new_call = RebuildTpopWithHalvedShape(call, split_int, split_dim);
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

    // AIV only: tile.load — halve result shape, halve shape/valid_shape args, adjust offset
    if (is_aiv && op_name == "tile.load" && call->args_.size() >= 4) {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      ExprPtr half_dim_size;
      if (tt && split_dim < static_cast<int>(tt->shape_.size())) {
        half_dim_size = ComputeHalfDimSize(tt->shape_[split_dim]);
      }

      auto new_result_type = HalveTileShape(call->GetType(), split_dim);
      std::vector<ExprPtr> new_args = call->args_;
      if (half_dim_size) {
        new_args[1] = AdjustOffsets(call->args_[1], split_dim, half_dim_size, subblock_idx);
      }
      new_args[2] = HalveTupleElement(call->args_[2], split_dim);
      new_args[3] = HalveTupleElement(call->args_[3], split_dim);

      auto new_call =
          std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type, call->span_);
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
      if (half_dim_size) {
        TileInfo info{half_dim_size};
        tile_vars[assign->var_.get()] = info;
        tile_vars[new_var.get()] = info;
      }
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

    // AIV only: any other op producing TileType — halve result shape (and static shape args when present)
    if (is_aiv) {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      if (tt && split_dim < static_cast<int>(tt->shape_.size())) {
        auto half_dim_size = ComputeHalfDimSize(tt->shape_[split_dim]);
        auto new_result_type = HalveTileShape(call->GetType(), split_dim);
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
    return std::make_shared<ForStmt>(for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_,
                                     for_stmt->iter_args_, new_body, for_stmt->return_vars_, for_stmt->span_,
                                     for_stmt->kind_, for_stmt->chunk_size_, for_stmt->chunk_policy_,
                                     for_stmt->loop_origin_);
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
    return std::make_shared<IfStmt>(if_stmt->condition_, new_then_body, new_else, if_stmt->return_vars_,
                                    if_stmt->span_);
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

  // For AIV functions, emit tile.get_subblock_idx() at the top so both vector
  // lanes share the same kernel body and select their slice dynamically.
  ExprPtr subblock_idx_expr;
  std::vector<StmtPtr> body_stmts;
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(func->body_)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(func->body_);
  }

  if (is_aiv) {
    auto idx_type = std::make_shared<ScalarType>(DataType::INT64);
    std::unordered_set<std::string> used_subblock_names;
    for (const auto& p : func->params_) {
      used_subblock_names.insert(p->name_hint_);
    }
    std::vector<VarPtr> def_vars;
    transform_utils::CollectDefVars(func->body_, def_vars);
    for (const auto& v : def_vars) {
      used_subblock_names.insert(v->name_hint_);
    }
    std::string subblock_var_name = "subblock_idx";
    if (used_subblock_names.count(subblock_var_name) != 0) {
      subblock_var_name = auto_name::GenerateFreshNameLike("subblock_idx", used_subblock_names);
    }

    auto& op_reg = OpRegistry::GetInstance();
    auto subblock_op = op_reg.GetOp("tile.get_subblock_idx");
    auto subblock_call =
        std::make_shared<Call>(subblock_op, std::vector<ExprPtr>{},
                               std::vector<std::pair<std::string, std::any>>{}, idx_type, func->span_);
    auto subblock_idx_var = std::make_shared<Var>(subblock_var_name, idx_type, func->span_);
    auto assign_stmt = std::make_shared<AssignStmt>(subblock_idx_var, subblock_call, func->span_);
    body_stmts.insert(body_stmts.begin(), assign_stmt);
    subblock_idx_expr = subblock_idx_var;
  }

  auto new_stmts = ProcessStmts(body_stmts, mode, split_int, split_dim, tile_vars, is_aiv, subblock_idx_expr,
                                var_replacements);
  StmtPtr new_body =
      (new_stmts.size() == 1) ? new_stmts[0] : std::make_shared<SeqStmts>(new_stmts, func->span_);
  if (!var_replacements.empty()) {
    new_body = transform_utils::Substitute(new_body, var_replacements);
  }
  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  auto attrs = WithSplitAttr(func, mode);
  return std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                    cloned_body, func->span_, func->func_type_, func->level_, func->role_,
                                    attrs);
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
      // Only process AIC and AIV functions that have a non-None split mode
      if (should_split) {
        new_functions.push_back(ProcessFunction(func, resolved_mode.value()));
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
