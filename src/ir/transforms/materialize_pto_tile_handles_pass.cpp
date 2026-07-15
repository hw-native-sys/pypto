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

/**
 * @file materialize_pto_tile_handles_pass.cpp
 * @brief Step-3 bridge from logical Tile SSA values to explicit PTO buffer handles.
 *
 * The pass deliberately does not rewrite high-level tile operations yet. It
 * inserts one typed ``pto.alloc_tile`` per supported Tile result and records
 * the exact input/output handle plan on each high-level Call. This temporary,
 * verified bridge makes the following DPS rewrite mechanical: it consumes
 * handles instead of recovering destinations from MemRefs or result context.
 *
 * Complexity: O(N) expected time and O(N) space for N statements/operands.
 */

#include <any>
#include <cstddef>
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
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

enum class SupportedTileOp { None, Load, Unary, Binary, Store };

SupportedTileOp ClassifySupportedTileOp(const CallPtr& call) {
  if (IsOp(call, "tile.load")) return SupportedTileOp::Load;
  if (IsOp(call, "tile.sqrt")) return SupportedTileOp::Unary;
  if (IsOp(call, "tile.add") || IsOp(call, "tile.mul")) return SupportedTileOp::Binary;
  if (IsOp(call, "tile.store")) return SupportedTileOp::Store;
  return SupportedTileOp::None;
}

bool TypeContainsTile(const TypePtr& type) {
  if (IsA<TileType>(type)) return true;
  auto tuple = As<TupleType>(type);
  if (!tuple) return false;
  for (const auto& element : tuple->types_) {
    if (TypeContainsTile(element)) return true;
  }
  return false;
}

bool CallTouchesTile(const CallPtr& call) {
  if (!call) return false;
  if (TypeContainsTile(call->GetType())) return true;
  for (const auto& arg : call->args_) {
    if (arg && TypeContainsTile(arg->GetType())) return true;
  }
  return false;
}

std::vector<std::pair<std::string, std::any>> WithHandlePlanAttrs(
    const std::vector<std::pair<std::string, std::any>>& attrs, std::vector<VarPtr> input_handles,
    const std::optional<VarPtr>& output_handle) {
  std::vector<std::pair<std::string, std::any>> result;
  result.reserve(attrs.size() + 2);
  for (const auto& [key, value] : attrs) {
    if (key != kAttrPTOInputHandles && key != kAttrPTOOutputHandle) {
      result.emplace_back(key, value);
    }
  }
  result.emplace_back(kAttrPTOInputHandles, std::move(input_handles));
  if (output_handle.has_value()) {
    result.emplace_back(kAttrPTOOutputHandle, *output_handle);
  }
  return result;
}

class HandleMaterializer final {
 public:
  explicit HandleMaterializer(MemoryPlanner planner) : planner_(planner) {}

  FunctionPtr Run(const FunctionPtr& func) {
    if (!func || !IsInCoreType(func->func_type_)) return func;

    for (const auto& param : func->params_) {
      CHECK_SPAN(!TypeContainsTile(param->GetType()), param->span_)
          << "MaterializePTOTileHandles does not support Tile-typed function parameters; "
             "load tiles from Tensor parameters inside the function";
    }

    std::vector<StmtPtr> rewritten;
    const auto stmts = transform_utils::FlattenToStmts(func->body_);
    rewritten.reserve(stmts.size() * 2);

    for (const auto& stmt : stmts) {
      if (auto assign = As<AssignStmt>(stmt)) {
        RewriteAssign(assign, rewritten);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        RewriteEval(eval, rewritten);
      } else if (auto ret = As<ReturnStmt>(stmt)) {
        for (const auto& value : ret->value_) {
          CHECK_SPAN(!TypeContainsTile(value->GetType()), ret->span_)
              << "MaterializePTOTileHandles does not support returning Tile values";
        }
        rewritten.push_back(stmt);
      } else {
        CHECK_SPAN(false, stmt->span_)
            << "MaterializePTOTileHandles currently supports straight-line functions only; found "
            << stmt->TypeName();
      }
    }

    CHECK_SPAN(claimed_handles_.size() == allocated_handles_.size(), func->span_)
        << "MaterializePTOTileHandles found an unclaimed pto.alloc_tile; the input is partially "
           "materialized or malformed";

    auto new_body = SeqStmts::Flatten(std::move(rewritten), func->body_->span_);
    if (new_body.get() == func->body_.get()) return func;
    return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                            func->return_types_, std::move(new_body), func->span_,
                                            func->func_type_, func->level_, func->role_, func->attrs_,
                                            func->requires_runtime_binding_);
  }

 private:
  void RewriteAssign(const AssignStmtPtr& assign, std::vector<StmtPtr>& rewritten) {
    auto call = As<Call>(assign->value_);
    if (!call) {
      CHECK_SPAN(!TypeContainsTile(assign->var_->GetType()), assign->span_)
          << "MaterializePTOTileHandles does not support non-Call Tile definitions";
      rewritten.push_back(assign);
      return;
    }

    if (IsOp(call, "pto.alloc_tile")) {
      ValidateExistingAllocation(assign, call);
      rewritten.push_back(assign);
      return;
    }

    CHECK_SPAN(call->op_->name_.rfind("pto.", 0) != 0, call->span_)
        << "MaterializePTOTileHandles only accepts pre-existing pto.alloc_tile target ops";

    auto kind = ClassifySupportedTileOp(call);
    if (kind == SupportedTileOp::None) {
      CHECK_SPAN(!CallTouchesTile(call), call->span_)
          << "MaterializePTOTileHandles does not yet support Tile operation '" << call->op_->name_
          << "'; supported operations are tile.load, tile.sqrt, tile.add, tile.mul, and tile.store";
      rewritten.push_back(assign);
      return;
    }

    RewriteSupportedCall(assign->var_, call, assign->span_, assign->leading_comments_, rewritten,
                         /*is_eval=*/false);
  }

  void ValidateExistingAllocation(const AssignStmtPtr& assign, const CallPtr& call) {
    CHECK_SPAN(IsA<PTOTileBufType>(assign->var_->GetType()) && IsA<PTOTileBufType>(call->GetType()),
               assign->span_)
        << "Existing pto.alloc_tile definition and result must have PTOTileBufType";
    CHECK_SPAN(structural_equal(assign->var_->GetType(), call->GetType()), assign->span_)
        << "Existing pto.alloc_tile definition and result types must match";

    const size_t expected_arg_count = planner_ == MemoryPlanner::PyPTO ? 3 : 2;
    CHECK_SPAN(call->args_.size() == expected_arg_count, call->span_)
        << "Existing pto.alloc_tile has " << call->args_.size() << " metadata operands, but "
        << (planner_ == MemoryPlanner::PyPTO ? "PYPTO" : "PTOAS") << " planning requires "
        << expected_arg_count;
    for (const auto& arg : call->args_) {
      CHECK_SPAN(As<ConstInt>(arg), call->span_)
          << "MaterializePTOTileHandles Step 3 requires constant pto.alloc_tile metadata operands";
    }

    CHECK_SPAN(allocated_handles_.insert(assign->var_.get()).second, assign->span_)
        << "PTO handle is allocated more than once";
  }

  void RewriteEval(const EvalStmtPtr& eval, std::vector<StmtPtr>& rewritten) {
    auto call = As<Call>(eval->expr_);
    CHECK_SPAN(call, eval->span_) << "MaterializePTOTileHandles supports only Call expressions in EvalStmt";
    CHECK_SPAN(call->op_->name_.rfind("pto.", 0) != 0, call->span_)
        << "MaterializePTOTileHandles only accepts pre-existing pto.alloc_tile target ops";
    auto kind = ClassifySupportedTileOp(call);
    if (kind == SupportedTileOp::None) {
      CHECK_SPAN(!CallTouchesTile(call), call->span_)
          << "MaterializePTOTileHandles does not yet support Tile operation '" << call->op_->name_
          << "'; supported operations are tile.load, tile.sqrt, tile.add, tile.mul, and tile.store";
      rewritten.push_back(eval);
      return;
    }
    CHECK_SPAN(kind == SupportedTileOp::Store, call->span_)
        << "Only tile.store may appear as an EvalStmt in the Step-3 lowering slice";
    RewriteSupportedCall(std::nullopt, call, eval->span_, eval->leading_comments_, rewritten,
                         /*is_eval=*/true);
  }

  void RewriteSupportedCall(const std::optional<VarPtr>& result_var, const CallPtr& call, const Span& span,
                            const std::vector<std::string>& leading_comments, std::vector<StmtPtr>& rewritten,
                            bool is_eval) {
    const VarPtr assigned_result = result_var.value_or(nullptr);
    std::vector<VarPtr> input_handles;
    for (const auto& arg : call->args_) {
      if (!IsA<TileType>(arg->GetType())) continue;
      auto logical_var = AsVarLike(arg);
      CHECK_SPAN(logical_var, call->span_)
          << "MaterializePTOTileHandles requires flattened Tile operands, got " << arg->TypeName();
      auto it = logical_to_handle_.find(logical_var.get());
      CHECK_SPAN(it != logical_to_handle_.end(), call->span_)
          << "Tile operand '" << logical_var->name_hint_
          << "' has no dominating PTO handle; ensure its producer is in the same straight-line function";
      input_handles.push_back(it->second);
    }

    std::optional<VarPtr> output_handle;
    const bool has_tile_result = assigned_result && IsA<TileType>(assigned_result->GetType());
    if (has_tile_result) {
      CHECK_SPAN(!is_eval, span) << "Tile-producing operation must be assigned to a logical Tile variable";
      if (call->HasAttr(kAttrPTOOutputHandle)) {
        auto existing = call->GetAttr<VarPtr>(kAttrPTOOutputHandle);
        CHECK_SPAN(existing && allocated_handles_.count(existing.get()) != 0, call->span_)
            << "Existing pto_output_handle must name a dominating pto.alloc_tile result";
        output_handle = existing;
      } else {
        auto tile_type = As<TileType>(assigned_result->GetType());
        output_handle = BuildAllocation(assigned_result, tile_type, span, rewritten);
      }
      CHECK_SPAN(claimed_handles_.insert(output_handle->get()).second, call->span_)
          << "PTO handle is bound to more than one logical Tile value";
      logical_to_handle_[assigned_result.get()] = *output_handle;
    } else {
      CHECK_SPAN(!call->HasAttr(kAttrPTOOutputHandle), call->span_)
          << "Non-Tile operation must not carry pto_output_handle";
    }

    auto attrs = WithHandlePlanAttrs(call->attrs_, std::move(input_handles), output_handle);
    auto rewritten_call = std::make_shared<const Call>(call->op_, call->args_, call->kwargs_,
                                                       std::move(attrs), call->GetType(), call->span_);
    if (is_eval) {
      rewritten.push_back(std::make_shared<const EvalStmt>(rewritten_call, span, leading_comments));
    } else {
      CHECK_SPAN(assigned_result, span) << "Non-Eval supported operation must have an assigned result";
      rewritten.push_back(
          std::make_shared<const AssignStmt>(assigned_result, rewritten_call, span, leading_comments));
    }
  }

  VarPtr BuildAllocation(const VarPtr& logical_var, const TileTypePtr& tile_type, const Span& span,
                         std::vector<StmtPtr>& rewritten) {
    CHECK_SPAN(tile_type && tile_type->shape_.size() == 2, span)
        << "MaterializePTOTileHandles requires a static 2D Tile result";
    auto rows = As<ConstInt>(tile_type->shape_[0]);
    auto cols = As<ConstInt>(tile_type->shape_[1]);
    CHECK_SPAN(rows && cols, span)
        << "MaterializePTOTileHandles requires compile-time constant physical Tile dimensions";
    CHECK_SPAN(tile_type->memory_space_.has_value(), span)
        << "MaterializePTOTileHandles requires inferred Tile memory_space";
    const auto memory_space = tile_type->memory_space_.value_or(MemorySpace::DDR);
    CHECK_SPAN(memory_space != MemorySpace::DDR, span)
        << "MaterializePTOTileHandles cannot allocate a DDR Tile buffer";

    const TileView view = tile_view_semantics::GetEffectiveTileView(*tile_type);
    const auto& valid_shape = view.valid_shape.empty() ? tile_type->shape_ : view.valid_shape;
    CHECK_SPAN(valid_shape.size() == 2, span) << "MaterializePTOTileHandles requires a rank-2 valid shape";
    auto valid_rows = As<ConstInt>(valid_shape[0]);
    auto valid_cols = As<ConstInt>(valid_shape[1]);
    CHECK_SPAN(valid_rows && valid_cols, span)
        << "MaterializePTOTileHandles Step 3 does not support dynamic valid shape; defer it to Step 5";

    auto buffer_type = std::make_shared<const PTOTileBufType>(
        memory_space, tile_type->dtype_, rows->value_, cols->value_, view.blayout, view.slayout, view.fractal,
        view.pad, std::nullopt, std::nullopt);
    auto handle = std::make_shared<const Var>(
        logical_var->name_hint_ + "_pto_buf_" + std::to_string(next_handle_id_++), buffer_type, span);

    std::vector<ExprPtr> alloc_args;
    if (planner_ == MemoryPlanner::PyPTO) {
      const auto memref = tile_type->memref_.value_or(nullptr);
      CHECK_SPAN(memref, span) << "PYPTO planning requires a MemRef on every materialized Tile result";
      const auto& byte_offset = memref->byte_offset_;
      CHECK_SPAN(As<ConstInt>(byte_offset), span)
          << "PYPTO planning requires AllocateMemoryAddr to produce a constant Tile byte offset";
      alloc_args.push_back(byte_offset);
    }
    alloc_args.push_back(std::make_shared<const ConstInt>(valid_rows->value_, DataType::INDEX, span));
    alloc_args.push_back(std::make_shared<const ConstInt>(valid_cols->value_, DataType::INDEX, span));

    auto alloc_call = std::make_shared<const Call>(OpRegistry::GetInstance().GetOp("pto.alloc_tile"),
                                                   std::move(alloc_args), buffer_type, span);
    rewritten.push_back(std::make_shared<const AssignStmt>(handle, alloc_call, span));
    CHECK_SPAN(allocated_handles_.insert(handle.get()).second, span)
        << "Internal error: duplicate PTO handle allocation";
    return handle;
  }

  MemoryPlanner planner_;
  size_t next_handle_id_ = 0;
  std::unordered_map<const Var*, VarPtr> logical_to_handle_;
  std::unordered_set<const Var*> allocated_handles_;
  std::unordered_set<const Var*> claimed_handles_;
};

FunctionPtr TransformFunction(const FunctionPtr& func) {
  const auto* context = PassContext::Current();
  const auto planner = context ? context->GetMemoryPlanner() : MemoryPlanner::PyPTO;
  return HandleMaterializer(planner).Run(func);
}

}  // namespace

Pass MaterializePTOTileHandles() {
  return CreateFunctionPass(TransformFunction, "MaterializePTOTileHandles",
                            kMaterializePTOTileHandlesProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
