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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
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
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_ir_selection.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

// Capability is decided before any Buffer-specific storage mutation. This is
// deliberately narrower than the explicit low-level lowering API. No exception
// is interpreted as an unsupported capability: malformed selected IR must fail.
class BufferKernelSupport : public IRVisitor {
 public:
  explicit BufferKernelSupport(const FunctionPtr& function) : return_types_(function->return_types_) {
    for (size_t i = 0; i < function->params_.size(); ++i) {
      const auto& param = function->params_[i];
      if (As<TensorType>(param->GetType())) {
        tensor_aliases_.emplace(param.get(), param);
        directions_.emplace(param.get(), function->param_directions_[i]);
      }
    }
  }

  bool supported = true;
  bool has_load = false;
  bool has_fill = false;
  bool has_store = false;

  static bool ConstantEquals(const ExprPtr& value, int64_t expected) {
    const auto constant = As<ConstInt>(value);
    return constant && constant->value_ == expected;
  }

  static bool SameShape(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                      [](const ExprPtr& a, const ExprPtr& b) { return structural_equal(a, b); });
  }

  static bool StaticShape(const std::vector<ExprPtr>& shape) {
    return shape.size() == 2 && As<ConstInt>(shape[0]) && As<ConstInt>(shape[1]) &&
           As<ConstInt>(shape[0])->value_ > 0 && As<ConstInt>(shape[1])->value_ > 1;
  }

  void CheckType(const TypePtr& type) {
    if (!checked_types_.insert(type.get()).second) return;
    if (auto tile = As<TileType>(type)) {
      const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
      if (!StaticShape(tile->shape_) || !backend::IsDenseVecBufferTile(tile, view)) {
        supported = false;
        return;
      }
      const auto cols = As<ConstInt>(tile->shape_[1])->value_;
      supported &= cols % (256 / tile->dtype_.GetBit()) == 0 &&
                   (!view.start_offset || ConstantEquals(view.start_offset, 0));
      supported &= view.stride.empty() || (view.stride.size() == 2 && ConstantEquals(view.stride[0], cols) &&
                                           ConstantEquals(view.stride[1], 1));
      if (!view.valid_shape.empty()) {
        supported &= view.valid_shape.size() == tile->shape_.size();
        for (size_t i = 0; supported && i < view.valid_shape.size(); ++i) {
          const auto valid = As<ConstInt>(view.valid_shape[i]);
          supported &= valid && valid->value_ > 0 && valid->value_ <= As<ConstInt>(tile->shape_[i])->value_;
        }
      }
      if (const auto memory = tile->memref_.value_or(nullptr)) {
        supported &= !memory->is_pinned_ && memory->slot_count_ == 1 && !memory->slot_index_;
      }
    } else if (auto tensor = As<TensorType>(type)) {
      supported &= StaticShape(tensor->shape_) && backend::IsDenseBufferTransferDtype(tensor->dtype_);
      if (supported && tensor->tensor_view_) {
        const auto& view = *tensor->tensor_view_;
        supported &= view.layout == TensorLayout::ND && view.pad == PadValue::null;
        supported &= view.valid_shape.empty() || SameShape(view.valid_shape, tensor->shape_);
        supported &=
            view.stride.empty() || (view.stride.size() == 2 &&
                                    ConstantEquals(view.stride[0], As<ConstInt>(tensor->shape_[1])->value_) &&
                                    ConstantEquals(view.stride[1], 1));
      }
    } else if (!As<ScalarType>(type) && !As<PtrType>(type) && !As<VoidType>(type) && !As<TupleType>(type)) {
      supported = false;
    }
  }

  void VisitExpr(const ExprPtr& expr) override {
    if (!supported || !expr) return;
    if (As<TupleGetItemExpr>(expr) || As<Submit>(expr) || As<Pow>(expr)) {
      supported = false;
      return;
    }
    CheckType(expr->GetType());
    IRVisitor::VisitExpr(expr);
  }

  void VisitStmt(const StmtPtr& stmt) override {
    if (!supported) return;
    if (!As<SeqStmts>(stmt) && !As<AssignStmt>(stmt) && !As<EvalStmt>(stmt) && !As<ReturnStmt>(stmt)) {
      supported = false;
      return;
    }
    IRVisitor::VisitStmt(stmt);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}

  void VisitBinaryExpr_(const BinaryExprPtr& expr) override {
    supported &= !GetScalarDtype(expr).IsUnsignedInt() && !GetScalarDtype(expr->left_).IsUnsignedInt() &&
                 !GetScalarDtype(expr->right_).IsUnsignedInt();
    IRVisitor::VisitBinaryExpr_(expr);
  }

  void VisitUnaryExpr_(const UnaryExprPtr& expr) override {
    const auto source = GetScalarDtype(expr->operand_);
    const auto target = GetScalarDtype(expr);
    if (As<Cast>(expr)) {
      supported &= (!source.IsUnsignedInt() && !target.IsUnsignedInt()) || source == DataType::INDEX ||
                   source == target ||
                   (!source.IsFloat() && target != DataType::INDEX && !target.IsFloat() &&
                    source.GetBit() == target.GetBit());
    } else {
      supported &= !source.IsUnsignedInt() && !target.IsUnsignedInt();
    }
    IRVisitor::VisitUnaryExpr_(expr);
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    IRVisitor::VisitStmt_(assign);
    if (!supported || !As<TensorType>(assign->var_->GetType())) return;
    ExprPtr source = assign->value_;
    if (auto call = As<Call>(source); call && IsOp(call, "tile.store") && call->args_.size() == 3) {
      source = call->args_[2];
    }
    auto var = AsVarLike(source);
    auto alias = tensor_aliases_.find(var.get());
    if (alias != tensor_aliases_.end()) tensor_aliases_.emplace(assign->var_.get(), alias->second);
  }

  void VisitStmt_(const ReturnStmtPtr& ret) override {
    // Buffer lowering erases store-result and plain GM aliases. Its codegen ABI
    // requires the resulting parameter type to match the declared result,
    // including TensorView metadata introduced by earlier frontend passes.
    if (ret->value_.size() != return_types_.size()) {
      supported = false;
      return;
    }
    for (size_t i = 0; i < ret->value_.size(); ++i) {
      auto value = AsVarLike(ret->value_[i]);
      auto alias = tensor_aliases_.find(value.get());
      if (alias == tensor_aliases_.end() || !structural_equal(alias->second->GetType(), return_types_[i])) {
        supported = false;
        return;
      }
    }
    IRVisitor::VisitStmt_(ret);
  }

  void VisitExpr_(const SubmitPtr&) override { supported = false; }
  void VisitExpr_(const CallPtr& call) override {
    if (As<GlobalVar>(call->op_)) {
      supported = false;
      return;
    }
    // Selection must reject metadata that the final lowering cannot carry.
    if (!StripAttr(call->attrs_, kPipelineMembershipAttr).empty()) {
      supported = false;
      return;
    }
    if (IsOp(call, "tile.load")) {
      has_load = true;
      CheckTransferParameter(call, 0, true);
      supported &= GetIntKwarg(call->kwargs_, "cache", 0) == 0;
    } else if (IsOp(call, "tile.store")) {
      has_store = true;
      CheckTransferParameter(call, 2, false);
      supported &= GetIntKwarg(call->kwargs_, "atomic", 0) == 0 &&
                   GetIntKwarg(call->kwargs_, "st_phase", 0) == 0 &&
                   !GetOptionalDoubleKwarg(call->kwargs_, "pre_quant") &&
                   !GetKwargOr<bool>(call->kwargs_, "pre_relu", false);
    } else if (!IsOp(call, "tile.alloc")) {
      const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call->op_->name_);
      supported &=
          recipe && recipe->default_enabled && backend::SupportsLogicalBufferElementwise(*recipe, call);
      if (IsOp(call, "tile.full")) has_fill = true;
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  void CheckTransferParameter(const CallPtr& call, size_t index, bool load) {
    if (index >= call->args_.size()) {
      supported = false;
      return;
    }
    const auto value = AsVarLike(call->args_[index]);
    const auto alias = tensor_aliases_.find(value.get());
    if (alias == tensor_aliases_.end()) {
      supported = false;
      return;
    }
    const auto direction = directions_.at(alias->second.get());
    supported &= load ? direction != ParamDirection::Out : direction != ParamDirection::In;
  }

  const std::vector<TypePtr>& return_types_;
  std::unordered_map<const Var*, ParamDirection> directions_;
  std::unordered_map<const Var*, VarPtr> tensor_aliases_;
  std::unordered_set<const Type*> checked_types_;
};

FunctionPtr SelectBufferFunction(const FunctionPtr& function, bool device_helper) {
  if (!IsInCoreType(function->func_type_) || function->ir_stage_ == FunctionIRStage::Buffer ||
      function->HasAttr(kAttrBufferIrSelected)) {
    return function;
  }
  BufferKernelSupport support(function);
  const auto* context = PassContext::Current();
  support.supported =
      !device_helper && (!context || context->GetEnableBufferIR()) && backend::BackendConfig::IsConfigured();
  if (support.supported) {
    const auto arch = backend::BackendConfig::GetBackend()->GetHandler()->GetPtoTargetArch();
    support.supported = arch == "a2a3";
  }
  for (const auto& param : function->params_) {
    // Device Tile parameters still require a separate helper ABI.
    if (!As<TensorType>(param->GetType()) && !As<ScalarType>(param->GetType())) support.supported = false;
    support.CheckType(param->GetType());
  }
  for (const auto& type : function->return_types_) {
    if (!As<TensorType>(type)) support.supported = false;
    support.CheckType(type);
  }
  support.VisitStmt(function->body_);
  auto selected = std::make_shared<Function>(*function);
  selected->attrs_.emplace_back(kAttrBufferIrSelected, support.supported &&
                                                           (support.has_load || support.has_fill) &&
                                                           support.has_store);
  return selected;
}

// A GM-shaped function called from device code is still a helper, not a
// separately dispatched kernel. Keep both caller and callee out of this rollout.
class DeviceHelperCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> callees;

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const CallPtr& call) override { Collect(call); }
  void VisitExpr_(const SubmitPtr& submit) override {
    Collect(submit);
    for (const auto& dep : submit->deps_) VisitExpr(dep);
  }

 private:
  template <typename CallLike>
  void Collect(const CallLike& call) {
    if (auto global = As<GlobalVar>(call->op_)) callees.insert(global->name_);
    for (const auto& arg : call->args_) VisitExpr(arg);
  }
};

ProgramPtr SelectBufferFunctions(const ProgramPtr& program) {
  DeviceHelperCollector helpers;
  for (const auto& [global, function] : program->functions_) {
    if (IsInCoreType(function->func_type_)) helpers.VisitStmt(function->body_);
  }
  std::vector<FunctionPtr> functions;
  functions.reserve(program->functions_.size());
  for (const auto& [global, function] : program->functions_) {
    functions.push_back(SelectBufferFunction(function, helpers.callees.count(function->name_) != 0));
  }
  return std::make_shared<Program>(functions, program->name_, program->span_);
}

}  // namespace

namespace pass {

Pass SelectBufferIRFunctions() {
  return CreateProgramPass(SelectBufferFunctions, "SelectBufferIRFunctions",
                           kSelectBufferIRFunctionsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
