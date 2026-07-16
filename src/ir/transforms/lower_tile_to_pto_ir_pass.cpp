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
 * @file lower_tile_to_pto_ir_pass.cpp
 * @brief Step-4 rewrite from logical Tile SSA calls to destination-passing PTO target IR.
 *
 * MaterializePTOTileHandles has already computed and verified the logical-value
 * to mutable-buffer mapping. This pass only consumes that plan: logical Tile
 * definitions disappear, target operations receive explicit input/output
 * handles, and load/store receive explicit offsets and valid extents.
 *
 * Complexity: O(N) expected time and O(N) space for N statements.
 */

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

const std::any* FindAttr(const CallPtr& call, const std::string& key) {
  for (const auto& [attr_key, value] : call->attrs_) {
    if (attr_key == key) return &value;
  }
  return nullptr;
}

VarPtr GetOutputHandle(const CallPtr& call) {
  const std::any* value = FindAttr(call, kAttrPTOOutputHandle);
  const auto* handle = value ? std::any_cast<VarPtr>(value) : nullptr;
  INTERNAL_CHECK_SPAN(handle && *handle, call->span_)
      << "LowerTileToPTOIR requires a verified pto_output_handle on '" << call->op_->name_ << "'";
  return *handle;
}

std::vector<VarPtr> GetInputHandles(const CallPtr& call) {
  const std::any* value = FindAttr(call, kAttrPTOInputHandles);
  const auto* handles = value ? std::any_cast<std::vector<VarPtr>>(value) : nullptr;
  INTERNAL_CHECK_SPAN(handles, call->span_)
      << "LowerTileToPTOIR requires verified pto_input_handles on '" << call->op_->name_ << "'";
  return *handles;
}

class TileToPTOLowerer final {
 public:
  FunctionPtr Run(const FunctionPtr& func) {
    if (!func || !IsInCoreType(func->func_type_)) return func;
    if (func->GetAttr<bool>(kAttrPTOTargetLoweringDeferred, false)) return func;

    DiscoverSpmdInterface(func);
    auto new_body = LowerBody(func->body_);
    auto params = func->params_;
    auto directions = func->param_directions_;
    std::vector<std::pair<std::string, std::any>> attrs;
    attrs.reserve(func->attrs_.size() + 2);
    for (const auto& attr : func->attrs_) {
      if (attr.first != kAttrPTOControlFlowHandles) attrs.push_back(attr);
    }
    if (uses_spmd_block_params_) {
      params.push_back(spmd_block_idx_param_);
      params.push_back(spmd_block_num_param_);
      directions.push_back(ParamDirection::In);
      directions.push_back(ParamDirection::In);
      attrs.emplace_back(kAttrPTOUsesSpmdBlockParams, true);
    }
    if (uses_subblock_param_) {
      params.push_back(spmd_subblock_idx_param_);
      directions.push_back(ParamDirection::In);
      attrs.emplace_back(kAttrPTOUsesSubblockParam, true);
    }
    return std::make_shared<const Function>(func->name_, std::move(params), std::move(directions),
                                            func->return_types_, std::move(new_body), func->span_,
                                            func->func_type_, func->level_, func->role_, std::move(attrs),
                                            func->requires_runtime_binding_);
  }

 private:
  StmtPtr LowerBody(const StmtPtr& body, const std::vector<bool>& keep_yield_slots = {}) {
    std::vector<StmtPtr> lowered;
    for (const auto& stmt : transform_utils::FlattenToStmts(body)) {
      if (auto assign = As<AssignStmt>(stmt)) {
        LowerAssign(assign, lowered);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        LowerEval(eval, lowered);
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        LowerFor(for_stmt, lowered);
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        LowerIf(if_stmt, lowered);
      } else if (auto yield = As<YieldStmt>(stmt)) {
        LowerYield(yield, keep_yield_slots, lowered);
      } else if (As<ReturnStmt>(stmt)) {
        lowered.push_back(stmt);
      } else {
        INTERNAL_CHECK_SPAN(false, stmt->span_)
            << "LowerTileToPTOIR received unsupported structured IR after handle materialization: "
            << stmt->TypeName();
      }
    }
    return SeqStmts::Flatten(std::move(lowered), body->span_);
  }

  void LowerFor(const ForStmtPtr& loop, std::vector<StmtPtr>& lowered) {
    INTERNAL_CHECK_SPAN(loop->iter_args_.size() == loop->return_vars_.size(), loop->span_)
        << "Verified ForStmt carry arity mismatch";
    std::vector<bool> keep(loop->iter_args_.size(), true);
    std::vector<IterArgPtr> iter_args;
    std::vector<VarPtr> return_vars;
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      keep[i] = !IsA<TileType>(loop->iter_args_[i]->GetType());
      if (!keep[i]) continue;
      iter_args.push_back(loop->iter_args_[i]);
      return_vars.push_back(loop->return_vars_[i]);
    }
    auto body = LowerBody(loop->body_, keep);
    lowered.push_back(std::make_shared<const ForStmt>(
        loop->loop_var_, loop->start_, loop->stop_, loop->step_, std::move(iter_args), std::move(body),
        std::move(return_vars), loop->span_, loop->kind_, loop->attrs_, loop->leading_comments_));
  }

  void LowerIf(const IfStmtPtr& branch, std::vector<StmtPtr>& lowered) {
    std::vector<bool> keep(branch->return_vars_.size(), true);
    std::vector<VarPtr> return_vars;
    for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
      keep[i] = !IsA<TileType>(branch->return_vars_[i]->GetType());
      if (keep[i]) return_vars.push_back(branch->return_vars_[i]);
    }
    auto then_body = LowerBody(branch->then_body_, keep);
    std::optional<StmtPtr> else_body;
    if (branch->else_body_) else_body = LowerBody(*branch->else_body_, keep);
    lowered.push_back(std::make_shared<const IfStmt>(branch->condition_, std::move(then_body),
                                                     std::move(else_body), std::move(return_vars),
                                                     branch->span_, branch->leading_comments_));
  }

  static void LowerYield(const YieldStmtPtr& yield, const std::vector<bool>& keep,
                         std::vector<StmtPtr>& lowered) {
    if (keep.empty()) {
      lowered.push_back(yield);
      return;
    }
    INTERNAL_CHECK_SPAN(keep.size() == yield->value_.size(), yield->span_)
        << "Verified structured yield arity mismatch";
    std::vector<ExprPtr> values;
    for (size_t i = 0; i < keep.size(); ++i) {
      if (keep[i]) values.push_back(yield->value_[i]);
    }
    lowered.push_back(
        std::make_shared<const YieldStmt>(std::move(values), yield->span_, yield->leading_comments_));
  }

  void DiscoverSpmdInterface(const FunctionPtr& func) {
    class Finder final : public IRVisitor {
     public:
      void VisitExpr_(const CallPtr& call) override {
        if (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num")) uses_block = true;
        if (IsOp(call, "tile.get_subblock_idx")) uses_subblock = true;
        IRVisitor::VisitExpr_(call);
      }

      bool uses_block = false;
      bool uses_subblock = false;
    } finder;
    finder.VisitStmt(func->body_);
    uses_spmd_block_params_ = finder.uses_block;
    uses_subblock_param_ = finder.uses_subblock;
    auto i32 = std::make_shared<const ScalarType>(DataType::INT32);
    if (uses_spmd_block_params_) {
      spmd_block_idx_param_ = std::make_shared<const Var>(kPTOSpmdBlockIdxParam, i32, func->span_);
      spmd_block_num_param_ = std::make_shared<const Var>(kPTOSpmdBlockNumParam, i32, func->span_);
    }
    if (uses_subblock_param_) {
      spmd_subblock_idx_param_ = std::make_shared<const Var>(kPTOSpmdSubblockIdxParam, i32, func->span_);
    }
  }

  void LowerAssign(const AssignStmtPtr& assign, std::vector<StmtPtr>& lowered) {
    auto call = As<Call>(assign->value_);
    if (!call) {
      if (IsA<TileType>(assign->var_->GetType())) {
        // Tile aliases have already been resolved to explicit handles by the
        // Step-3 control-flow plan.
        return;
      }
      INTERNAL_CHECK_SPAN(!IsA<TileType>(assign->var_->GetType()), assign->span_)
          << "Logical Tile assignment survived PTOHandlesMaterialized verification";
      lowered.push_back(assign);
      return;
    }

    if (IsOp(call, "pto.alloc_tile")) {
      RecordAllocation(assign, call);
      lowered.push_back(assign);
      return;
    }

    if (IsOp(call, "pto.subview")) {
      INTERNAL_CHECK_SPAN(call->args_.size() == 4, call->span_)
          << "Verified pto.subview must carry an explicit valid-shape tuple";
      auto valid = As<MakeTuple>(call->args_[3]);
      INTERNAL_CHECK_SPAN(valid && valid->elements_.size() == 2, call->span_)
          << "Verified pto.subview valid shape must be rank-2";
      auto [it, inserted] = allocation_valid_extents_.emplace(
          assign->var_.get(), std::make_pair(valid->elements_[0], valid->elements_[1]));
      INTERNAL_CHECK_SPAN(inserted, assign->span_) << "PTO handle is defined more than once";
      (void)it;
      lowered.push_back(assign);
      return;
    }

    if (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num") ||
        IsOp(call, "tile.get_subblock_idx")) {
      VarPtr source;
      if (IsOp(call, "tile.get_block_idx")) source = spmd_block_idx_param_;
      if (IsOp(call, "tile.get_block_num")) source = spmd_block_num_param_;
      if (IsOp(call, "tile.get_subblock_idx")) source = spmd_subblock_idx_param_;
      auto result_type = As<ScalarType>(assign->var_->GetType());
      INTERNAL_CHECK_SPAN(source && result_type, call->span_)
          << "SPMD identity call requires a materialized scalar target parameter";
      auto cast = std::make_shared<const Cast>(source, result_type->dtype_, call->span_);
      lowered.push_back(std::make_shared<const AssignStmt>(assign->var_, std::move(cast), assign->span_,
                                                           assign->leading_comments_));
      return;
    }

    if (IsOp(call, "tile.alloc")) {
      // The Ptr result exists only to root logical Tile MemRefs. Step 3 has
      // already copied the required address/extent data into pto.alloc_tile,
      // and Step 4 removes every logical TileType, so the region token has no
      // target-IR consumer and must not reach the mechanical printer.
      return;
    }

    if (const auto* lowering = FindPTOSimpleOpLowering(call->op_->name_);
        lowering && lowering->kind == PTOSimpleOpKind::AllocationOnly) {
      // The explicit pto.alloc_tile emitted by Step 3 is the complete target
      // representation of an uninitialized logical tile.create.
      return;
    }

    if (IsOp(call, "tile.slice")) {
      // Step 3 already emitted the explicit pto.subview SSA handle. The
      // logical slice definition now has no remaining target representation.
      return;
    }

    if (IsOp(call, "tile.load") || FindPTOSimpleOpLowering(call->op_->name_)) {
      lowered.push_back(MakeTargetEval(call, assign->span_, assign->leading_comments_));
      return;
    }

    if (IsOp(call, "tile.store")) {
      lowered.push_back(MakeTargetEval(call, assign->span_, assign->leading_comments_));
      INTERNAL_CHECK_SPAN(call->args_.size() >= 3, call->span_)
          << "Verified tile.store must carry its destination tensor";
      // tile.store is value-producing in logical SSA but side-effecting in PTO
      // target IR. Preserve only the tensor alias needed by later scalar/tensor
      // users; the target store itself is an EvalStmt.
      lowered.push_back(std::make_shared<const AssignStmt>(assign->var_, call->args_[2], assign->span_));
      return;
    }

    INTERNAL_CHECK_SPAN(!IsA<TileType>(assign->var_->GetType()), assign->span_)
        << "Unsupported logical Tile producer survived PTOHandlesMaterialized verification: '"
        << call->op_->name_ << "'";
    lowered.push_back(assign);
  }

  void LowerEval(const EvalStmtPtr& eval, std::vector<StmtPtr>& lowered) {
    auto call = As<Call>(eval->expr_);
    INTERNAL_CHECK_SPAN(call, eval->span_) << "Verified Step-4 EvalStmt must contain a direct Call";
    if (IsOp(call, "tile.store")) {
      lowered.push_back(MakeTargetEval(call, eval->span_, eval->leading_comments_));
      return;
    }
    if (call->op_->name_.rfind("pto.", 0) == 0) {
      lowered.push_back(eval);
      return;
    }
    INTERNAL_CHECK_SPAN(call->op_->name_.rfind("tile.", 0) != 0, call->span_)
        << "Unsupported logical Tile EvalStmt survived PTOHandlesMaterialized verification: '"
        << call->op_->name_ << "'";
    lowered.push_back(eval);
  }

  void RecordAllocation(const AssignStmtPtr& assign, const CallPtr& call) {
    INTERNAL_CHECK_SPAN(call->args_.size() == 2 || call->args_.size() == 3, call->span_)
        << "Verified pto.alloc_tile must carry valid-row and valid-col operands";
    auto [it, inserted] = allocation_valid_extents_.emplace(
        assign->var_.get(), std::make_pair(call->args_[call->args_.size() - 2], call->args_.back()));
    INTERNAL_CHECK_SPAN(inserted, assign->span_) << "PTO handle is allocated more than once";
    (void)it;
  }

  MakeTuplePtr ValidExtentsFor(const VarPtr& handle, const Span& span) const {
    auto it = allocation_valid_extents_.find(handle.get());
    INTERNAL_CHECK_SPAN(it != allocation_valid_extents_.end(), span)
        << "PTO handle '" << handle->name_hint_ << "' has no dominating pto.alloc_tile";
    return std::make_shared<const MakeTuple>(std::vector<ExprPtr>{it->second.first, it->second.second}, span);
  }

  EvalStmtPtr MakeTargetEval(const CallPtr& logical_call, const Span& stmt_span,
                             const std::vector<std::string>& leading_comments) const {
    auto& registry = OpRegistry::GetInstance();
    std::vector<ExprPtr> args;
    std::vector<std::pair<std::string, std::any>> target_kwargs;
    std::string target_name;

    if (IsOp(logical_call, "tile.load")) {
      INTERNAL_CHECK_SPAN(logical_call->args_.size() >= 3, logical_call->span_)
          << "Verified tile.load must carry tensor, offsets, and shapes operands";
      const VarPtr output = GetOutputHandle(logical_call);
      target_name = "pto.tload";
      // Keep the GM partition in the tensor's logical coordinate system.  The
      // tile buffer is always physical 2-D after FlattenTileNDTo2D, but an
      // N-D load still needs N-D offsets/extents for pto.partition_view.
      const ExprPtr partition_extents =
          logical_call->args_.size() >= 4 ? logical_call->args_[3] : logical_call->args_[2];
      args = {logical_call->args_[0], logical_call->args_[1], partition_extents, output};
    } else if (IsOp(logical_call, "tile.store")) {
      INTERNAL_CHECK_SPAN(logical_call->args_.size() >= 3, logical_call->span_)
          << "Verified tile.store must carry tile, offsets, and destination operands";
      const auto inputs = GetInputHandles(logical_call);
      INTERNAL_CHECK_SPAN(inputs.size() == 1, logical_call->span_)
          << "Verified tile.store must carry exactly one input PTO handle";
      target_name = "pto.tstore";
      // FlattenTileNDTo2D appends the original N-D store shape as arg #3.
      // For the ordinary 2-D path the buffer valid extents remain the transfer
      // extents.  This keeps tensor-rank metadata separate from physical tile
      // layout metadata at the target-IR boundary.
      const ExprPtr partition_extents = logical_call->args_.size() >= 4
                                            ? logical_call->args_[3]
                                            : ValidExtentsFor(inputs[0], logical_call->span_);
      args = {inputs[0], logical_call->args_[1], partition_extents, logical_call->args_[2]};
      for (const auto& [key, value] : logical_call->kwargs_) {
        if (key == "atomic") target_kwargs.emplace_back(key, value);
      }
    } else {
      const auto inputs = GetInputHandles(logical_call);
      const VarPtr output = GetOutputHandle(logical_call);
      const auto* lowering = FindPTOSimpleOpLowering(logical_call->op_->name_);
      INTERNAL_CHECK_SPAN(lowering, logical_call->span_)
          << "Unsupported logical Tile operation reached LowerTileToPTOIR: '" << logical_call->op_->name_
          << "'";
      target_name = lowering->target_name;
      if (lowering->kind == PTOSimpleOpKind::ScalarFill) {
        INTERNAL_CHECK_SPAN(logical_call->args_.size() == 2, logical_call->span_)
            << "Verified tile.full must carry shape and value operands";
        args.push_back(logical_call->args_[1]);
      } else {
        args.assign(inputs.begin(), inputs.end());
        for (const auto& arg : logical_call->args_) {
          if (!IsA<TileType>(arg->GetType())) args.push_back(arg);
        }
      }
      args.push_back(output);
    }

    auto target_call =
        std::make_shared<const Call>(registry.GetOp(target_name), std::move(args), std::move(target_kwargs),
                                     GetUnknownType(), logical_call->span_);
    return std::make_shared<const EvalStmt>(target_call, stmt_span, leading_comments);
  }

  std::unordered_map<const Var*, std::pair<ExprPtr, ExprPtr>> allocation_valid_extents_;
  bool uses_spmd_block_params_ = false;
  bool uses_subblock_param_ = false;
  VarPtr spmd_block_idx_param_;
  VarPtr spmd_block_num_param_;
  VarPtr spmd_subblock_idx_param_;
};

FunctionPtr TransformFunction(const FunctionPtr& func) { return TileToPTOLowerer().Run(func); }

}  // namespace

Pass LowerTileToPTOIR() {
  return CreateFunctionPass(TransformFunction, "LowerTileToPTOIR", kLowerTileToPTOIRProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
