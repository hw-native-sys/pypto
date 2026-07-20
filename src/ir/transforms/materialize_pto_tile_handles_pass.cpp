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
#include "pypto/ir/pto_target_lowering.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
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

bool IsPTOIRPrinterSupportedParameter(const TypePtr& type) {
  return AsTensorTypeLike(type) || IsA<ScalarType>(type);
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
    if (func->GetAttr<bool>(kAttrPTOTargetLoweringDeferred, false)) return func;

    class UnsupportedTileOpFinder final : public IRVisitor {
     public:
      explicit UnsupportedTileOpFinder(MemoryPlanner planner) : planner_(planner) {}

      void VisitStmt_(const AssignStmtPtr& stmt) override {
        if (IsA<ArrayType>(stmt->var_->GetType())) found = true;
        CheckDirectCall(As<Call>(stmt->value_));
        if (!found) IRVisitor::VisitStmt_(stmt);
      }

      void VisitStmt_(const EvalStmtPtr& stmt) override {
        CheckDirectCall(As<Call>(stmt->expr_));
        if (!found) IRVisitor::VisitStmt_(stmt);
      }

      void VisitStmt_(const IfStmtPtr& stmt) override {
        for (const auto& result : stmt->return_vars_) {
          if (AsTensorTypeLike(result->GetType())) found = true;
        }
        if (!found) IRVisitor::VisitStmt_(stmt);
      }

      void VisitStmt_(const WhileStmtPtr&) override { found = true; }

      void VisitExpr_(const CallPtr& call) override {
        if (CallTouchesTile(call) && !IsPTOHandlePlanOp(call->op_->name_)) {
          found = true;
          return;
        }
        if (planner_ == MemoryPlanner::PtoAS && IsOp(call, "tile.reshape")) {
          found = true;
          return;
        }
        if (auto tile = As<TileType>(call->GetType()); tile && tile->shape_.size() != 2) {
          found = true;
          return;
        }
        IRVisitor::VisitExpr_(call);
      }

     private:
      static bool IsPrinterSupportedDirectCall(const CallPtr& call) {
        if (!call) return true;
        const auto& name = call->op_->name_;
        return IsPTOHandlePlanOp(name) || name == "tile.alloc" || name == "tile.get_block_idx" ||
               name == "tile.get_block_num" || name == "tile.get_subblock_idx" || name == "tensor.view" ||
               name == "tensor.read" || name == "tensor.write" || name.rfind("pto.", 0) == 0;
      }

      void CheckDirectCall(const CallPtr& call) {
        if (!IsPrinterSupportedDirectCall(call)) found = true;
      }

      MemoryPlanner planner_;

     public:
      bool found = false;
    } finder(planner_);
    for (const auto& param : func->params_) {
      if (!param) {
        finder.found = true;
        continue;
      }
      if (!IsPTOIRPrinterSupportedParameter(param->GetType())) finder.found = true;
      if (IsA<ArrayType>(param->GetType())) finder.found = true;
      if (auto tensor = AsTensorTypeLike(param->GetType())) {
        for (const auto& dim : tensor->shape_) {
          if (!As<ConstInt>(dim)) finder.found = true;
        }
      }
    }
    finder.VisitStmt(func->body_);
    if (finder.found) {
      auto attrs = func->attrs_;
      attrs.emplace_back(kAttrPTOTargetLoweringDeferred, true);
      return std::make_shared<const Function>(
          func->name_, func->params_, func->param_directions_, func->return_types_, func->body_, func->span_,
          func->func_type_, func->level_, func->role_, std::move(attrs), func->requires_runtime_binding_);
    }

    for (const auto& param : func->params_) {
      CHECK_SPAN(!TypeContainsTile(param->GetType()), param->span_)
          << "MaterializePTOTileHandles does not support Tile-typed function parameters; "
             "load tiles from Tensor parameters inside the function";
    }

    auto new_body = RewriteBody(func->body_);

    CHECK_SPAN(claimed_handles_.size() == allocated_handles_.size(), func->span_)
        << "MaterializePTOTileHandles found an unclaimed pto.alloc_tile; the input is partially "
           "materialized or malformed";

    if (new_body.get() == func->body_.get()) return func;
    auto attrs = func->attrs_;
    if (!control_flow_handles_.empty()) {
      attrs.emplace_back(kAttrPTOControlFlowHandles, control_flow_handles_);
    }
    return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                            func->return_types_, std::move(new_body), func->span_,
                                            func->func_type_, func->level_, func->role_, std::move(attrs),
                                            func->requires_runtime_binding_);
  }

 private:
  StmtPtr RewriteBody(const StmtPtr& body,
                      const std::vector<std::pair<VarPtr, VarPtr>>& tile_yield_targets = {}) {
    std::vector<StmtPtr> rewritten;
    for (const auto& stmt : transform_utils::FlattenToStmts(body)) {
      if (auto assign = As<AssignStmt>(stmt)) {
        RewriteAssign(assign, rewritten);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        RewriteEval(eval, rewritten);
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        RewriteFor(for_stmt, rewritten);
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        RewriteIf(if_stmt, rewritten);
      } else if (auto yield = As<YieldStmt>(stmt)) {
        RewriteYield(yield, tile_yield_targets, rewritten);
      } else if (auto ret = As<ReturnStmt>(stmt)) {
        for (const auto& value : ret->value_) {
          CHECK_SPAN(!TypeContainsTile(value->GetType()), ret->span_)
              << "MaterializePTOTileHandles does not support returning Tile values";
        }
        rewritten.push_back(stmt);
      } else {
        CHECK_SPAN(false, stmt->span_)
            << "MaterializePTOTileHandles does not yet support structured statement " << stmt->TypeName();
      }
    }
    return SeqStmts::Flatten(std::move(rewritten), body->span_);
  }

  void RewriteFor(const ForStmtPtr& loop, std::vector<StmtPtr>& rewritten) {
    CHECK_SPAN(loop->iter_args_.size() == loop->return_vars_.size(), loop->span_)
        << "ForStmt iter_args and return_vars must have the same size";
    std::vector<std::pair<VarPtr, VarPtr>> tile_targets(loop->iter_args_.size());
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& iter_arg = loop->iter_args_[i];
      if (!IsA<TileType>(iter_arg->GetType())) continue;
      auto init = AsVarLike(iter_arg->initValue_);
      CHECK_SPAN(init, loop->span_) << "Tile loop carry initial value must be a flattened Var";
      auto it = logical_to_handle_.find(init.get());
      CHECK_SPAN(it != logical_to_handle_.end(), loop->span_)
          << "Tile loop carry initial value has no dominating PTO handle";
      logical_to_handle_[iter_arg.get()] = it->second;
      logical_to_handle_[loop->return_vars_[i].get()] = it->second;
      control_flow_handles_.emplace_back(iter_arg, it->second);
      control_flow_handles_.emplace_back(loop->return_vars_[i], it->second);
      tile_targets[i] = {loop->return_vars_[i], it->second};
    }
    PreparePTOASYieldOutputs(loop->body_, tile_targets);
    auto body = RewriteBody(loop->body_, tile_targets);
    rewritten.push_back(std::make_shared<const ForStmt>(
        loop->loop_var_, loop->start_, loop->stop_, loop->step_, loop->iter_args_, std::move(body),
        loop->return_vars_, loop->span_, loop->kind_, loop->attrs_, loop->leading_comments_));
  }

  void RewriteIf(const IfStmtPtr& branch, std::vector<StmtPtr>& rewritten) {
    std::vector<std::pair<VarPtr, VarPtr>> tile_targets(branch->return_vars_.size());
    for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
      const auto& result = branch->return_vars_[i];
      auto tile_type = As<TileType>(result->GetType());
      if (!tile_type) continue;
      VarPtr handle;
      auto forced = forced_output_handles_.find(result.get());
      if (forced != forced_output_handles_.end()) {
        handle = forced->second;
        CHECK_SPAN(allocated_handles_.count(handle.get()) != 0, branch->span_)
            << "Forced PTO tile-phi output must name a dominating allocation";
        CHECK_SPAN(structural_equal(handle->GetType(), BuildBufferType(tile_type, branch->span_)),
                   branch->span_)
            << "Forced PTO tile-phi output type does not match the logical result";
      } else {
        handle = BuildAllocation(result, tile_type, branch->span_, rewritten);
        CHECK_SPAN(claimed_handles_.insert(handle.get()).second, branch->span_)
            << "PTO tile-phi handle is claimed more than once";
      }
      logical_to_handle_[result.get()] = handle;
      control_flow_handles_.emplace_back(result, handle);
      tile_targets[i] = {result, handle};
    }
    PreparePTOASYieldOutputs(branch->then_body_, tile_targets);
    auto then_body = RewriteBody(branch->then_body_, tile_targets);
    std::optional<StmtPtr> else_body;
    if (branch->else_body_) {
      PreparePTOASYieldOutputs(*branch->else_body_, tile_targets);
      else_body = RewriteBody(*branch->else_body_, tile_targets);
    }
    rewritten.push_back(std::make_shared<const IfStmt>(branch->condition_, std::move(then_body),
                                                       std::move(else_body), branch->return_vars_,
                                                       branch->span_, branch->leading_comments_));
  }

  static std::unordered_set<const Var*> CollectWritableDefinitions(const StmtPtr& body) {
    std::unordered_set<const Var*> writable;
    for (const auto& stmt : transform_utils::FlattenToStmts(body)) {
      if (auto assign = As<AssignStmt>(stmt)) {
        auto call = As<Call>(assign->value_);
        if (!call || IsOp(call, "tile.slice")) continue;
        if (auto lowering = FindPTOSimpleOpLowering(call->op_->name_);
            lowering && lowering->kind == PTOSimpleOpKind::AllocationOnly) {
          continue;
        }
        writable.insert(assign->var_.get());
        continue;
      }
      if (auto loop = As<ForStmt>(stmt)) {
        for (const auto& result : loop->return_vars_) {
          writable.insert(result.get());
        }
      }
      if (auto branch = As<IfStmt>(stmt)) {
        for (const auto& result : branch->return_vars_) {
          writable.insert(result.get());
        }
      }
    }
    return writable;
  }

  void PreparePTOASYieldOutputs(const StmtPtr& body,
                                const std::vector<std::pair<VarPtr, VarPtr>>& tile_targets) {
    if (planner_ != MemoryPlanner::PtoAS || tile_targets.empty()) return;
    auto yield = transform_utils::GetLastYieldStmt(body);
    if (!yield || yield->value_.size() != tile_targets.size()) return;
    const auto writable_definitions = CollectWritableDefinitions(body);
    for (size_t i = 0; i < tile_targets.size(); ++i) {
      const auto& destination = tile_targets[i].second;
      if (!destination) continue;
      auto source = AsVarLike(yield->value_[i]);
      auto source_type = source ? As<TileType>(source->GetType()) : nullptr;
      if (!source_type || writable_definitions.count(source.get()) == 0) continue;
      if (!structural_equal(destination->GetType(), BuildBufferType(source_type, yield->span_))) continue;
      forced_output_handles_[source.get()] = destination;
    }
  }

  static bool SharesPhysicalStorage(const ExprPtr& source, const VarPtr& destination) {
    auto source_type = source ? As<TileType>(source->GetType()) : nullptr;
    auto destination_type = destination ? As<TileType>(destination->GetType()) : nullptr;
    if (!source_type || !destination_type || !source_type->memref_ || !destination_type->memref_) {
      return false;
    }
    const auto& lhs = *source_type->memref_;
    const auto& rhs = *destination_type->memref_;
    return lhs->base_.get() == rhs->base_.get() && lhs->size_ == rhs->size_ &&
           AreExprsEqual(lhs->byte_offset_, rhs->byte_offset_);
  }

  void RewriteYield(const YieldStmtPtr& yield, const std::vector<std::pair<VarPtr, VarPtr>>& tile_targets,
                    std::vector<StmtPtr>& rewritten) {
    CHECK_SPAN(tile_targets.empty() || tile_targets.size() == yield->value_.size(), yield->span_)
        << "Structured Tile yield plan must match the yield arity";
    for (size_t i = 0; i < tile_targets.size(); ++i) {
      if (!tile_targets[i].second) continue;
      auto source = AsVarLike(yield->value_[i]);
      CHECK_SPAN(source && IsA<TileType>(source->GetType()), yield->span_)
          << "Tile control-flow result must yield a flattened Tile Var";
      auto source_it = logical_to_handle_.find(source.get());
      CHECK_SPAN(source_it != logical_to_handle_.end(), yield->span_)
          << "Yielded Tile value has no dominating PTO handle";
      const auto& destination = tile_targets[i].second;
      if (source_it->second.get() == destination.get() ||
          SharesPhysicalStorage(source, tile_targets[i].first)) {
        continue;
      }
      auto move = std::make_shared<const Call>(OpRegistry::GetInstance().GetOp("pto.tmov"),
                                               std::vector<ExprPtr>{source_it->second, destination},
                                               GetUnknownType(), yield->span_);
      rewritten.push_back(std::make_shared<const EvalStmt>(move, yield->span_));
    }
    rewritten.push_back(yield);
  }

  void RewriteAssign(const AssignStmtPtr& assign, std::vector<StmtPtr>& rewritten) {
    auto call = As<Call>(assign->value_);
    if (!call) {
      if (IsA<TileType>(assign->var_->GetType())) {
        auto source = AsVarLike(assign->value_);
        CHECK_SPAN(source && IsA<TileType>(source->GetType()), assign->span_)
            << "MaterializePTOTileHandles requires a Tile alias source Var";
        auto it = logical_to_handle_.find(source.get());
        CHECK_SPAN(it != logical_to_handle_.end(), assign->span_)
            << "Tile alias source has no dominating PTO handle";
        logical_to_handle_[assign->var_.get()] = it->second;
        control_flow_handles_.emplace_back(assign->var_, it->second);
        rewritten.push_back(assign);
        return;
      }
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

    if (IsOp(call, "pto.subview")) {
      ValidateExistingSubview(assign, call);
      rewritten.push_back(assign);
      return;
    }

    CHECK_SPAN(call->op_->name_.rfind("pto.", 0) != 0, call->span_)
        << "MaterializePTOTileHandles only accepts pre-existing PTO handle definitions";

    if (IsOp(call, "tile.slice")) {
      RewriteSlice(assign, call, rewritten);
      return;
    }

    if (!IsPTOHandlePlanOp(call->op_->name_)) {
      CHECK_SPAN(!CallTouchesTile(call), call->span_)
          << "MaterializePTOTileHandles does not yet support Tile operation '" << call->op_->name_
          << "'; supported operations are load/store and the registered basic unary, binary, and "
             "tile-scalar elementwise operations";
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
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto& arg = call->args_[i];
      CHECK_SPAN(IsA<ScalarType>(arg->GetType()), call->span_)
          << "MaterializePTOTileHandles requires scalar pto.alloc_tile metadata operands";
      if (planner_ == MemoryPlanner::PyPTO && i == 0) {
        CHECK_SPAN(As<ConstInt>(arg), call->span_)
            << "PYPTO planning requires a constant pto.alloc_tile address";
      }
    }

    CHECK_SPAN(allocated_handles_.insert(assign->var_.get()).second, assign->span_)
        << "PTO handle is allocated more than once";
  }

  void ValidateExistingSubview(const AssignStmtPtr& assign, const CallPtr& call) {
    CHECK_SPAN(IsA<PTOTileBufType>(assign->var_->GetType()) && IsA<PTOTileBufType>(call->GetType()),
               assign->span_)
        << "Existing pto.subview definition and result must have PTOTileBufType";
    CHECK_SPAN(structural_equal(assign->var_->GetType(), call->GetType()), assign->span_)
        << "Existing pto.subview definition and result types must match";
    CHECK_SPAN(call->args_.size() == 4, call->span_)
        << "Existing pto.subview must have source, shape, offset, and valid-shape operands";
    auto source = AsVarLike(call->args_[0]);
    CHECK_SPAN(source && allocated_handles_.count(source.get()) != 0, call->span_)
        << "pto.subview source must name a dominating PTO handle";
    for (size_t i = 1; i < call->args_.size(); ++i) {
      auto tuple = As<MakeTuple>(call->args_[i]);
      CHECK_SPAN(tuple && tuple->elements_.size() == 2, call->span_)
          << "pto.subview shape, offset, and valid-shape operands must be rank-2 tuples";
    }
    CHECK_SPAN(allocated_handles_.insert(assign->var_.get()).second, assign->span_)
        << "PTO handle is defined more than once";
  }

  void RewriteSlice(const AssignStmtPtr& assign, const CallPtr& call, std::vector<StmtPtr>& rewritten) {
    CHECK_SPAN(call->args_.size() >= 3, call->span_)
        << "tile.slice requires source, shape, and offset operands";
    auto logical_source = AsVarLike(call->args_[0]);
    CHECK_SPAN(logical_source, call->span_) << "tile.slice source must be a flattened Tile Var";
    auto source_it = logical_to_handle_.find(logical_source.get());
    CHECK_SPAN(source_it != logical_to_handle_.end(), call->span_)
        << "tile.slice source has no dominating PTO handle";

    VarPtr output_handle;
    if (call->HasAttr(kAttrPTOOutputHandle)) {
      output_handle = call->GetAttr<VarPtr>(kAttrPTOOutputHandle);
      CHECK_SPAN(output_handle && allocated_handles_.count(output_handle.get()) != 0, call->span_)
          << "Existing tile.slice pto_output_handle must name its dominating pto.subview";
    } else {
      auto tile_type = As<TileType>(assign->var_->GetType());
      auto buffer_type = BuildBufferType(tile_type, assign->span_, /*preserve_static_valid=*/true);
      output_handle =
          std::make_shared<const Var>(assign->var_->name_hint_ + "_pto_view", buffer_type, assign->span_);
      const TileView view = tile_view_semantics::GetEffectiveTileView(*tile_type);
      const auto& valid_shape = view.valid_shape.empty() ? tile_type->shape_ : view.valid_shape;
      auto valid_tuple = std::make_shared<const MakeTuple>(valid_shape, call->span_);
      auto target_call = std::make_shared<const Call>(
          OpRegistry::GetInstance().GetOp("pto.subview"),
          std::vector<ExprPtr>{source_it->second, call->args_[1], call->args_[2], valid_tuple},
          std::vector<std::pair<std::string, std::any>>{}, buffer_type, call->span_);
      rewritten.push_back(std::make_shared<const AssignStmt>(output_handle, target_call, assign->span_));
      CHECK_SPAN(allocated_handles_.insert(output_handle.get()).second, assign->span_)
          << "PTO subview handle is defined more than once";
    }

    CHECK_SPAN(claimed_handles_.insert(output_handle.get()).second, call->span_)
        << "PTO subview handle is bound to more than one logical Tile value";
    logical_to_handle_[assign->var_.get()] = output_handle;
    auto attrs = WithHandlePlanAttrs(call->attrs_, {source_it->second}, output_handle);
    auto rewritten_call = std::make_shared<const Call>(call->op_, call->args_, call->kwargs_,
                                                       std::move(attrs), call->GetType(), call->span_);
    rewritten.push_back(std::make_shared<const AssignStmt>(assign->var_, rewritten_call, assign->span_,
                                                           assign->leading_comments_));
  }

  void RewriteEval(const EvalStmtPtr& eval, std::vector<StmtPtr>& rewritten) {
    auto call = As<Call>(eval->expr_);
    CHECK_SPAN(call, eval->span_) << "MaterializePTOTileHandles supports only Call expressions in EvalStmt";
    CHECK_SPAN(call->op_->name_.rfind("pto.", 0) != 0, call->span_)
        << "MaterializePTOTileHandles only accepts pre-existing pto.alloc_tile target ops";
    if (!IsPTOHandlePlanOp(call->op_->name_)) {
      CHECK_SPAN(!CallTouchesTile(call), call->span_)
          << "MaterializePTOTileHandles does not yet support Tile operation '" << call->op_->name_
          << "'; supported operations are load/store and the registered basic unary, binary, and "
             "tile-scalar elementwise operations";
      rewritten.push_back(eval);
      return;
    }
    CHECK_SPAN(IsOp(call, "tile.store"), call->span_)
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
          << "' has no dominating PTO handle; ensure its producer is materialized in the same function";
      input_handles.push_back(it->second);
    }

    std::optional<VarPtr> output_handle;
    const bool has_tile_result = assigned_result && IsA<TileType>(assigned_result->GetType());
    if (has_tile_result) {
      CHECK_SPAN(!is_eval, span) << "Tile-producing operation must be assigned to a logical Tile variable";
      auto forced = forced_output_handles_.find(assigned_result.get());
      const bool uses_forced_output = forced != forced_output_handles_.end();
      if (call->HasAttr(kAttrPTOOutputHandle)) {
        auto existing = call->GetAttr<VarPtr>(kAttrPTOOutputHandle);
        CHECK_SPAN(existing && allocated_handles_.count(existing.get()) != 0, call->span_)
            << "Existing pto_output_handle must name a dominating pto.alloc_tile result";
        CHECK_SPAN(!uses_forced_output || existing.get() == forced->second.get(), call->span_)
            << "Existing pto_output_handle conflicts with the structured PTOAS output plan";
        output_handle = existing;
      } else if (uses_forced_output) {
        output_handle = forced->second;
        CHECK_SPAN(allocated_handles_.count(output_handle->get()) != 0, call->span_)
            << "Forced PTOAS output handle must name a dominating allocation";
      } else {
        auto tile_type = As<TileType>(assigned_result->GetType());
        output_handle = BuildAllocation(assigned_result, tile_type, span, rewritten);
      }
      if (!uses_forced_output) {
        CHECK_SPAN(claimed_handles_.insert(output_handle->get()).second, call->span_)
            << "PTO handle is bound to more than one logical Tile value";
      }
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

  std::shared_ptr<const PTOTileBufType> BuildBufferType(const TileTypePtr& tile_type, const Span& span,
                                                        bool preserve_static_valid = false) const {
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
    CHECK_SPAN(IsA<ScalarType>(valid_shape[0]->GetType()) && IsA<ScalarType>(valid_shape[1]->GetType()), span)
        << "MaterializePTOTileHandles requires scalar valid-shape operands";

    std::optional<int64_t> static_valid_rows;
    std::optional<int64_t> static_valid_cols;
    if (preserve_static_valid) {
      if (auto value = As<ConstInt>(valid_shape[0]); value && value->value_ > 0) {
        static_valid_rows = value->value_;
      }
      if (auto value = As<ConstInt>(valid_shape[1]); value && value->value_ > 0) {
        static_valid_cols = value->value_;
      }
    }
    return std::make_shared<const PTOTileBufType>(memory_space, tile_type->dtype_, rows->value_, cols->value_,
                                                  view.blayout, view.slayout, view.fractal, view.pad,
                                                  static_valid_rows, static_valid_cols);
  }

  VarPtr BuildAllocation(const VarPtr& logical_var, const TileTypePtr& tile_type, const Span& span,
                         std::vector<StmtPtr>& rewritten) {
    auto buffer_type = BuildBufferType(tile_type, span);
    const TileView view = tile_view_semantics::GetEffectiveTileView(*tile_type);
    const auto& valid_shape = view.valid_shape.empty() ? tile_type->shape_ : view.valid_shape;
    auto handle = std::make_shared<const Var>(
        logical_var->name_hint_ + "_pto_buf_" + std::to_string(next_handle_id_++), buffer_type, span);

    std::vector<ExprPtr> alloc_args;
    if (planner_ == MemoryPlanner::PyPTO) {
      const auto memref = tile_type->memref_.value_or(nullptr);
      CHECK_SPAN(memref, span) << "PYPTO planning requires a MemRef on every materialized Tile result";
      const auto& byte_offset = memref->byte_offset_;
      auto constant_offset = As<ConstInt>(byte_offset);
      CHECK_SPAN(constant_offset, span)
          << "PYPTO planning requires AllocateMemoryAddr to produce a constant Tile byte offset";
      // PTOAS defines pto.alloc_tile's physical address operand as i64. The
      // production AllocateMemoryAddr pass already emits that dtype, while
      // hand-built IR may use INDEX; canonicalize at the target-IR boundary so
      // explicit pto.alloc_tile is valid independently of how the MemRef arose.
      alloc_args.push_back(std::make_shared<const ConstInt>(constant_offset->value_, DataType::INT64, span));
    }
    alloc_args.push_back(valid_shape[0]);
    alloc_args.push_back(valid_shape[1]);

    auto alloc_call = std::make_shared<const Call>(OpRegistry::GetInstance().GetOp("pto.alloc_tile"),
                                                   std::move(alloc_args), buffer_type, span);
    rewritten.push_back(std::make_shared<const AssignStmt>(handle, alloc_call, span));
    INTERNAL_CHECK_SPAN(allocated_handles_.insert(handle.get()).second, span)
        << "Internal error: duplicate PTO handle allocation";
    return handle;
  }

  MemoryPlanner planner_;
  size_t next_handle_id_ = 0;
  std::unordered_map<const Var*, VarPtr> logical_to_handle_;
  std::unordered_set<const Var*> allocated_handles_;
  std::unordered_set<const Var*> claimed_handles_;
  std::unordered_map<const Var*, VarPtr> forced_output_handles_;
  std::vector<std::pair<VarPtr, VarPtr>> control_flow_handles_;
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
