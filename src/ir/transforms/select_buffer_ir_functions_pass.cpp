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
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_valid_analysis.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace {

// Capability is decided before any Buffer-specific storage mutation. This is
// deliberately narrower than the explicit low-level lowering API. No exception
// is interpreted as an unsupported capability: malformed selected IR must fail.
class BufferKernelSupport : public IRVisitor {
 public:
  explicit BufferKernelSupport(const FunctionPtr& function)
      : return_types_(function->return_types_), views_(function->body_) {
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
  bool has_tile_if = false;
  bool has_loop = false;
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
      if (!StaticShape(tile->shape_)) {
        supported = false;
        return;
      }
      const auto cols = As<ConstInt>(tile->shape_[1])->value_;
      const auto space = tile->GetMemorySpace();
      if (space && backend::IsMatrixBufferSpace(*space)) {
        const auto box = tile_view_semantics::GetBoxedTileAlignment(view, tile->dtype_);
        supported &= backend::IsMatrixBufferDtype(*space, tile->dtype_) && box &&
                     (view.fractal == (*space == MemorySpace::Acc ? 1024 : 512)) &&
                     (view.blayout == TileLayout::row_major || view.blayout == TileLayout::col_major) &&
                     view.pad == PadValue::null && view.compact == CompactMode::null && view.stride.empty();
        if (box) supported &= As<ConstInt>(tile->shape_[0])->value_ % box->rows == 0 && cols % box->cols == 0;
      } else {
        supported &= backend::IsDenseVecBufferTile(tile, view) && cols % (256 / tile->dtype_.GetBit()) == 0;
        supported &=
            view.stride.empty() || (view.stride.size() == 2 && ConstantEquals(view.stride[0], cols) &&
                                    ConstantEquals(view.stride[1], 1));
      }
      supported &= !view.start_offset || ConstantEquals(view.start_offset, 0);
      if (!view.valid_shape.empty()) {
        supported &= view.valid_shape.size() == tile->shape_.size();
        for (size_t i = 0; supported && i < view.valid_shape.size(); ++i) {
          const auto valid = As<ConstInt>(view.valid_shape[i]);
          const auto scalar = As<ScalarType>(view.valid_shape[i]->GetType());
          supported &= valid ? valid->value_ >= 0 && valid->value_ <= As<ConstInt>(tile->shape_[i])->value_
                             : space == MemorySpace::Vec && scalar &&
                                   (scalar->dtype_ == DataType::INDEX || scalar->dtype_.IsSignedInt());
          if (space != MemorySpace::Vec) supported &= valid && valid->value_ > 0;
        }
      }
      if (const auto memory = tile->memref_.value_or(nullptr)) {
        supported &= !memory->is_pinned_ && memory->slot_count_ == 1 && !memory->slot_index_;
      }
    } else if (auto tensor = As<TensorType>(type)) {
      supported &= StaticShape(tensor->shape_) &&
                   (backend::IsDenseBufferTransferDtype(tensor->dtype_) || tensor->dtype_ == DataType::INT8);
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
    if (const auto var = AsVarLike(expr)) {
      if (const auto tile = As<TileType>(var->GetType())) {
        if (const auto memory = tile->memref_.value_or(nullptr)) {
          const auto offset = As<ConstInt>(memory->byte_offset_);
          const auto view = views_.Get(var);
          // Only a slice with a known typed parent can carry a symbolic
          // member offset. Allocation origins themselves remain static.
          supported &= offset ? offset->value_ >= 0 && offset->value_ % 32 == 0
                              : view.window && view.root && view.row_bytes.has_value();
        }
      }
    }
    IRVisitor::VisitExpr(expr);
  }

  void VisitStmt(const StmtPtr& stmt) override {
    if (!supported) return;
    if (!As<SeqStmts>(stmt) && !As<AssignStmt>(stmt) && !As<EvalStmt>(stmt) && !As<ReturnStmt>(stmt) &&
        !As<IfStmt>(stmt) && !As<ForStmt>(stmt) && !As<WhileStmt>(stmt) && !As<YieldStmt>(stmt)) {
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
    if (!supported) return;
    if (As<ScalarType>(assign->var_->GetType())) {
      scalar_multiples_.emplace(assign->var_.get(), KnownAlignmentMultiple(assign->value_));
    }
    if (IsMatrixType(assign->var_->GetType())) {
      ExprPtr source = assign->value_;
      if (auto call = As<Call>(source);
          call && (IsOp(call, "tile.matmul_acc") || IsOp(call, "tile.transpose_view"))) {
        source = call->args_[0];
      }
      const auto alias = StableAlias(source);
      matrix_aliases_.emplace(assign->var_.get(), alias ? alias : assign->var_);
      return;
    }
    if (!As<TensorType>(assign->var_->GetType())) return;
    ExprPtr source = assign->value_;
    if (auto call = As<Call>(source); call && IsOp(call, "tile.store") && call->args_.size() == 3) {
      source = call->args_[2];
    }
    auto var = AsVarLike(source);
    auto alias = tensor_aliases_.find(var.get());
    if (alias != tensor_aliases_.end()) tensor_aliases_.emplace(assign->var_.get(), alias->second);
  }

  void VisitStmt_(const EvalStmtPtr& eval) override {
    if (auto call = As<Call>(eval->expr_); call && IsOp(call, "tile.set_validshape")) {
      // Metadata version tracking and lowering require an explicit SSA result.
      supported = false;
      return;
    }
    IRVisitor::VisitStmt_(eval);
  }

  void VisitStmt_(const ReturnStmtPtr& ret) override {
    // Buffer lowering erases store-result and plain GM aliases. Its codegen ABI
    // requires the resulting parameter type to match the declared result,
    // including TensorView metadata introduced by earlier frontend passes.
    if (region_ || ret->value_.size() != return_types_.size()) {
      supported = false;
      return;
    }
    for (size_t i = 0; i < ret->value_.size(); ++i) {
      auto value = AsVarLike(ret->value_[i]);
      auto alias = tensor_aliases_.find(value.get());
      if (alias == tensor_aliases_.end() ||
          !structural_equal(CloneTypeWithoutMemRef(alias->second->GetType()),
                            CloneTypeWithoutMemRef(return_types_[i]))) {
        supported = false;
        return;
      }
    }
    IRVisitor::VisitStmt_(ret);
  }

  void VisitStmt_(const IfStmtPtr& branch) override {
    for (const auto& result : branch->return_vars_) {
      has_tile_if |= As<TileType>(result->GetType()) != nullptr;
    }
    VisitExpr(branch->condition_);
    Region then_region(branch->return_vars_);
    CheckRegion(branch->then_body_, then_region);
    Region else_region(branch->return_vars_);
    if (branch->else_body_) CheckRegion(*branch->else_body_, else_region);
    for (size_t i = 0; supported && i < branch->return_vars_.size(); ++i) {
      const auto& result = branch->return_vars_[i];
      CheckRegionType(result->GetType());
      if (NeedsStableAlias(result->GetType())) {
        const auto& alias = then_region.aliases[i];
        if (!alias || alias != else_region.aliases[i]) {
          supported = false;
          return;
        }
        AliasMap(result->GetType()).emplace(result.get(), alias);
      }
    }
  }

  void VisitStmt_(const YieldStmtPtr& yield) override {
    if (!region_ || yield->value_.size() != region_->results.size()) {
      supported = false;
      return;
    }
    region_->yielded = true;
    for (size_t i = 0; supported && i < yield->value_.size(); ++i) {
      CheckRegionType(region_->results[i]->GetType());
      VisitExpr(yield->value_[i]);
      // A region result cannot export a branch-local window handle. Window
      // carries require scalar metadata and explicit reconstruction first.
      supported &= !views_.IsWindow(yield->value_[i]);
      if (NeedsStableAlias(region_->results[i]->GetType())) {
        region_->aliases[i] = StableAlias(yield->value_[i]);
        supported &= region_->aliases[i] != nullptr;
      }
    }
  }

  void VisitStmt_(const ForStmtPtr& loop) override {
    has_loop = true;
    supported &= GetScalarDtype(loop->loop_var_) == DataType::INDEX;
    for (const auto& bound : {loop->start_, loop->stop_, loop->step_}) {
      VisitExpr(bound);
      const auto dtype = GetScalarDtype(bound);
      supported &= dtype == DataType::INDEX || dtype.IsSignedInt();
    }
    const auto step = transform_utils::EvalConstInt(loop->step_);
    supported &= step && *step > 0;
    CheckLoop(loop);
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    has_loop = true;
    CheckLoop(loop);
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
      CheckTransferTypes(call, true);
      supported &= GetIntKwarg(call->kwargs_, "cache", 0) == 0;
    } else if (IsOp(call, "tile.store")) {
      has_store = true;
      CheckTransferParameter(call, 2, false);
      CheckTransferTypes(call, false);
      supported &= GetIntKwarg(call->kwargs_, "atomic", 0) == 0 &&
                   GetIntKwarg(call->kwargs_, "st_phase", 0) == 0 &&
                   !GetOptionalDoubleKwarg(call->kwargs_, "pre_quant") &&
                   !GetKwargOr<bool>(call->kwargs_, "pre_relu", false);
    } else if (IsOp(call, "tile.set_validshape")) {
      supported &= call->args_.size() == 3 && call->kwargs_.empty() && !views_.IsWindow(call->args_[0]) &&
                   As<TileType>(call->GetType())->GetMemorySpace() == MemorySpace::Vec;
    } else if (IsOp(call, "tile.slice") || IsOp(call, "tile.assemble")) {
      CheckWindowCall(call);
    } else if (IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view")) {
      const auto source = call->args_.empty() ? nullptr : As<TileType>(call->args_[0]->GetType());
      const auto destination = As<TileType>(call->GetType());
      const auto source_bytes = TileBytes(source);
      supported &= source && destination && source->GetMemorySpace() == MemorySpace::Vec &&
                   destination->GetMemorySpace() == MemorySpace::Vec && source_bytes &&
                   source_bytes == TileBytes(destination) && views_.HasDenseRowPitch(call->args_[0]) &&
                   (!views_.IsWindow(call->args_[0]) || views_.Get(call->args_[0]).offsets.has_value());
      if (IsOp(call, "tile.reshape")) {
        supported &= call->args_.size() == 2 && call->kwargs_.empty() && source && destination &&
                     source->dtype_ == destination->dtype_;
      } else {
        supported &= (call->args_.size() == 1 || call->args_.size() == 2) && call->kwargs_.size() == 1 &&
                     call->kwargs_[0].first == "dtype";
      }
    } else if (IsOp(call, "tile.move") || IsOp(call, "tile.extract") || IsOp(call, "tile.transpose_view") ||
               IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_acc")) {
      CheckMatrixCall(call);
    } else if (IsOp(call, "tile.create")) {
      supported &= As<TileType>(call->GetType()) != nullptr;
    } else if (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num") ||
               IsOp(call, "tile.get_subblock_idx")) {
      supported &= call->args_.empty() && call->kwargs_.empty();
    } else if (!IsOp(call, "tile.alloc")) {
      const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call->op_->name_);
      supported &=
          recipe && recipe->default_enabled && backend::SupportsLogicalBufferElementwise(*recipe, call);
      if (IsOp(call, "tile.full")) has_fill = true;
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  static constexpr int64_t kWindowAlignmentBytes = 32;

  // Cache scalar SSA facts and expression nodes once. Unknown values have no
  // guaranteed factor beyond one; arithmetic preserves only proven multiples.
  int64_t KnownAlignmentMultiple(const ExprPtr& expr) {
    const auto cached = expression_multiples_.find(expr.get());
    if (cached != expression_multiples_.end()) return cached->second;
    int64_t multiple = 1;
    if (const auto constant = As<ConstInt>(expr)) {
      multiple = std::gcd(constant->value_ % kWindowAlignmentBytes, kWindowAlignmentBytes);
    } else if (const auto var = AsVarLike(expr)) {
      const auto found = scalar_multiples_.find(var.get());
      if (found != scalar_multiples_.end()) multiple = found->second;
    } else if (const auto cast = As<Cast>(expr)) {
      const auto source = GetScalarDtype(cast->operand_);
      const auto target = GetScalarDtype(cast);
      const auto integer = [](DataType dtype) {
        return dtype == DataType::INDEX || dtype.IsSignedInt() || dtype.IsUnsignedInt();
      };
      if (integer(source) && integer(target) && source.GetBit() >= 5 && target.GetBit() >= 5) {
        multiple = KnownAlignmentMultiple(cast->operand_);
      }
    } else if (const auto binary = As<BinaryExpr>(expr);
               binary && (As<Add>(expr) || As<Sub>(expr) || As<Mul>(expr))) {
      const auto lhs = KnownAlignmentMultiple(binary->left_);
      const auto rhs = KnownAlignmentMultiple(binary->right_);
      multiple = As<Mul>(expr) ? std::gcd(lhs * rhs, kWindowAlignmentBytes) : std::gcd(lhs, rhs);
    }
    expression_multiples_.emplace(expr.get(), multiple);
    return multiple;
  }

  std::optional<int64_t> WindowOffsetResidue(const ExprPtr& offset, int64_t stride_bytes) {
    const auto stride = stride_bytes % kWindowAlignmentBytes;
    if (const auto constant = As<ConstInt>(offset)) {
      const auto residue =
          (constant->value_ % kWindowAlignmentBytes + kWindowAlignmentBytes) % kWindowAlignmentBytes;
      return residue * stride % kWindowAlignmentBytes;
    }
    if (KnownAlignmentMultiple(offset) * stride % kWindowAlignmentBytes == 0) return 0;
    return std::nullopt;
  }

  void CheckWindowCall(const CallPtr& call) {
    const bool slice = IsOp(call, "tile.slice");
    if (call->args_.size() < 3) {
      supported = false;
      return;
    }
    const auto parent = As<TileType>(call->args_[0]->GetType());
    const auto window = As<TileType>(slice ? call->GetType() : call->args_[1]->GetType());
    const auto offsets = As<MakeTuple>(call->args_[2]);
    if (!parent || !window || parent->GetMemorySpace() != MemorySpace::Vec ||
        window->GetMemorySpace() != MemorySpace::Vec || parent->dtype_ != window->dtype_ ||
        !StaticShape(parent->shape_) || !StaticShape(window->shape_) || !offsets ||
        offsets->elements_.size() != 2) {
      supported = false;
      return;
    }
    // Explicit-valid slices and assemble destinations bypass slice
    // canonicalization. Prove their byte address alignment independently,
    // using the inherited parent pitch rather than the window's own width.
    const auto row_bytes = views_.Get(call->args_[0]).row_bytes;
    const auto row_residue =
        row_bytes ? WindowOffsetResidue(offsets->elements_[0], *row_bytes) : std::nullopt;
    const auto col_residue =
        WindowOffsetResidue(offsets->elements_[1], static_cast<int64_t>(window->dtype_.GetBit() / 8));
    supported &= row_residue && col_residue && (*row_residue + *col_residue) % kWindowAlignmentBytes == 0;
    for (size_t i = 0; i < 2; ++i) {
      const auto offset = As<ConstInt>(offsets->elements_[i]);
      const auto scalar = As<ScalarType>(offsets->elements_[i]->GetType());
      const auto limit = As<ConstInt>(parent->shape_[i])->value_ - As<ConstInt>(window->shape_[i])->value_;
      // Dynamic bounds remain the slice/assemble caller's precondition. They
      // never establish a precise range for alias analysis.
      supported &= limit >= 0 && scalar &&
                   (scalar->dtype_ == DataType::INDEX || scalar->dtype_.IsSignedInt()) &&
                   (!offset || (offset->value_ >= 0 && offset->value_ <= limit));
    }
    if (slice) {
      supported &= call->kwargs_.empty() && call->args_.size() <= 5;
      const auto parent_view = views_.Get(call->args_[0]);
      supported &= !parent_view.window || parent_view.root != nullptr;
      if (call->args_.size() == 5) {
        const auto dropped = As<MakeTuple>(call->args_[4]);
        supported &= dropped && dropped->elements_.empty();
      }
    } else {
      supported &= call->args_.size() == 3 && !views_.IsWindow(call->args_[0]) &&
                   !GetOptionalDoubleKwarg(call->kwargs_, "pre_quant") &&
                   !GetKwargOr<bool>(call->kwargs_, "pre_relu", false);
      // Native Vec TMOV preserves row stride; copying a strided window into
      // a destination with a different pitch needs a separate transfer recipe.
      const auto source_view = views_.Get(call->args_[1]);
      supported &= source_view.row_bytes && source_view.row_bytes == views_.Get(call->args_[0]).row_bytes &&
                   (!source_view.window || (source_view.root && source_view.slice));
    }
    if (!views_.IsWindow(call->args_[0])) {
      const auto memory = parent->memref_.value_or(nullptr);
      if (memory) {
        const auto [entry, inserted] = window_parents_.emplace(memory->base_.get(), parent);
        // One symbolic allocation can have only one native typed parent.
        // Conflicting parent interpretations need additional legalization.
        supported &= inserted || (entry->second->dtype_ == parent->dtype_ &&
                                  SameShape(entry->second->shape_, parent->shape_));
      }
    }
  }

  static std::optional<uint64_t> TileBytes(const TileTypePtr& tile) {
    if (!tile || !StaticShape(tile->shape_)) return std::nullopt;
    const std::vector<int64_t> shape{As<ConstInt>(tile->shape_[0])->value_,
                                     As<ConstInt>(tile->shape_[1])->value_};
    return backend::PhysicalBufferBytes(std::make_shared<BufferType>(shape, tile->dtype_, MemorySpace::Vec));
  }

  static std::vector<ExprPtr> ValidShape(const TileTypePtr& tile) {
    const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
    return view.valid_shape.empty() ? tile->shape_ : view.valid_shape;
  }

  void CheckTransferTypes(const CallPtr& call, bool load) {
    if (call->args_.size() < 3) {
      supported = false;
      return;
    }
    const auto tile = As<TileType>(load ? call->GetType() : call->args_[0]->GetType());
    const auto tensor = As<TensorType>(call->args_[load ? 0 : 2]->GetType());
    if (!tile || !tensor) {
      supported = false;
      return;
    }
    const auto space = tile->GetMemorySpace();
    if (space == MemorySpace::Vec || (load && space == MemorySpace::Mat)) {
      supported &= tile->dtype_ == tensor->dtype_;
    } else {
      supported &=
          !load && space == MemorySpace::Acc && CubeWritebackSupportsDataType(tile->dtype_, tensor->dtype_);
    }
  }

  void CheckMatrixCall(const CallPtr& call) {
    const auto destination = As<TileType>(call->GetType());
    const auto source = call->args_.empty() ? nullptr : As<TileType>(call->args_[0]->GetType());
    if (!source || !destination) {
      supported = false;
      return;
    }
    if (IsOp(call, "tile.move") || IsOp(call, "tile.extract")) {
      supported &= source->GetMemorySpace() == MemorySpace::Mat &&
                   (destination->GetMemorySpace() == MemorySpace::Left ||
                    destination->GetMemorySpace() == MemorySpace::Right) &&
                   source->dtype_ == destination->dtype_;
      if (IsOp(call, "tile.move")) {
        supported &= SameShape(source->shape_, destination->shape_) &&
                     SameShape(ValidShape(source), ValidShape(destination));
      } else {
        supported &= call->args_.size() == 4;
      }
    } else if (IsOp(call, "tile.transpose_view")) {
      supported &=
          call->args_.size() == 1 && call->kwargs_.empty() && source->GetMemorySpace() == MemorySpace::Mat &&
          destination->GetMemorySpace() == MemorySpace::Mat && source->dtype_ == destination->dtype_ &&
          TileBytes(source) && TileBytes(source) == TileBytes(destination);
    } else {
      const bool accumulate = IsOp(call, "tile.matmul_acc");
      const size_t lhs_index = accumulate ? 1 : 0;
      if (call->args_.size() < lhs_index + 2) {
        supported = false;
        return;
      }
      const auto lhs = As<TileType>(call->args_[lhs_index]->GetType());
      const auto rhs = As<TileType>(call->args_[lhs_index + 1]->GetType());
      supported &= lhs && rhs && call->kwargs_.empty() && call->args_.size() <= (accumulate ? 4 : 2) &&
                   destination->GetMemorySpace() == MemorySpace::Acc;
      if (!supported) return;
      supported &= lhs->GetMemorySpace() == MemorySpace::Left &&
                   rhs->GetMemorySpace() == MemorySpace::Right && lhs->dtype_ == rhs->dtype_ &&
                   backend::IsCubeOperandDtype(lhs->dtype_) &&
                   destination->dtype_ == MatmulAccumulatorDataType(lhs->dtype_, rhs->dtype_);
      if (accumulate) supported &= source->GetMemorySpace() == MemorySpace::Acc;
    }
  }

  void CheckRegionType(const TypePtr& type) {
    supported &= As<TileType>(type) || As<TensorType>(type) || As<ScalarType>(type);
    CheckType(type);
  }

  struct Region {
    explicit Region(const std::vector<VarPtr>& results) : results(results), aliases(results.size()) {}
    const std::vector<VarPtr>& results;
    std::vector<VarPtr> aliases;
    bool yielded = false;
  };

  static bool IsMatrixType(const TypePtr& type) {
    const auto tile = As<TileType>(type);
    return tile && tile->GetMemorySpace() && backend::IsMatrixBufferSpace(*tile->GetMemorySpace());
  }

  static bool NeedsStableAlias(const TypePtr& type) { return As<TensorType>(type) || IsMatrixType(type); }

  std::unordered_map<const Var*, VarPtr>& AliasMap(const TypePtr& type) {
    return IsMatrixType(type) ? matrix_aliases_ : tensor_aliases_;
  }

  VarPtr StableAlias(const ExprPtr& value) {
    const auto var = AsVarLike(value);
    auto& aliases = AliasMap(value->GetType());
    const auto alias = aliases.find(var.get());
    return alias == aliases.end() ? nullptr : alias->second;
  }

  void CheckRegion(const StmtPtr& body, Region& region) {
    auto* outer = region_;
    region_ = &region;
    VisitStmt(body);
    supported &= region.results.empty() || region.yielded;
    region_ = outer;
  }

  template <typename LoopPtr>
  void CheckLoop(const LoopPtr& loop) {
    if (!supported || loop->iter_args_.size() != loop->return_vars_.size()) {
      supported = false;
      return;
    }
    // SSA identities are unique across regions. Retaining each normalized alias
    // once avoids copying an ever-growing map at nested branches and loops.
    for (size_t i = 0; supported && i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      CheckRegionType(argument->GetType());
      CheckRegionType(loop->return_vars_[i]->GetType());
      VisitExpr(argument->initValue_);
      supported &= !views_.IsWindow(argument->initValue_);
      if (NeedsStableAlias(argument->GetType())) {
        const auto alias = StableAlias(argument->initValue_);
        if (!alias) {
          supported = false;
          return;
        }
        AliasMap(argument->GetType()).emplace(argument.get(), alias);
      }
    }
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) VisitExpr(loop->condition_);
    Region region(loop->return_vars_);
    CheckRegion(loop->body_, region);
    for (size_t i = 0; supported && i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      if (NeedsStableAlias(argument->GetType())) {
        const auto initial = StableAlias(argument->initValue_);
        if (region.aliases[i] != initial) {
          supported = false;
          return;
        }
        AliasMap(argument->GetType()).emplace(loop->return_vars_[i].get(), initial);
      }
    }
  }

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

  Region* region_ = nullptr;
  const std::vector<TypePtr>& return_types_;
  BufferViewAnalysis views_;
  std::unordered_map<const Var*, int64_t> scalar_multiples_;
  std::unordered_map<const Expr*, int64_t> expression_multiples_;
  std::unordered_map<const Var*, TileTypePtr> window_parents_;
  std::unordered_map<const Var*, ParamDirection> directions_;
  std::unordered_map<const Var*, VarPtr> tensor_aliases_;
  std::unordered_map<const Var*, VarPtr> matrix_aliases_;
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
  // Buffer lowering erases Tile region results and requires fixed destinations.
  // Use native Tile SSA selection for branch-only kernels. Keep the existing
  // whole-kernel route for loops: their Buffer carry legalization is separate
  // from this change, and switching it to the legacy path can change semantics.
  if (support.has_tile_if && !support.has_loop) support.supported = false;
  if (support.supported) support.supported = BufferValidAnalysis(function->body_).supported();
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
