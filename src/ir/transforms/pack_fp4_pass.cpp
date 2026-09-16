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
 * @file pack_fp4_pass.cpp
 * @brief Convert frontend logical FP4 nibble types into packed FP4E2M1X2.
 *
 * Frontend IR counts FP4 as logical nibbles. PTOAS only addresses
 * !pto.f4E2M1x2 packed pairs. This pass rewrites dtypes, last-axis extents,
 * ND leading strides, and last-axis coordinates so later passes and codegen
 * work in packed units.
 *
 * Scope (static-only): last-axis covering sizes, offsets, and leading strides
 * must be ConstInt (positive even for sizes/strides; even for offsets). Dynamic
 * last-axis geometry is rejected — use pl.FP4E2M1X2 with physical shapes, or
 * the DSv4.1 UINT8 half-width cache ABI.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

enum class LastAxisKind {
  kCoveringSize,   // shape / valid_shape / slice size
  kOffset,         // offset / index on the packed axis
  kLeadingStride,  // ND leading stride (covering-size packing, stride-specific errors)
};

ExprPtr MakeIndexConst(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr HalveLastAxisExpr(const ExprPtr& value, LastAxisKind kind, const Span& span) {
  CHECK_SPAN(static_cast<bool>(value), span) << "PackFp4: last-axis expression must not be null";
  const bool covering = kind == LastAxisKind::kCoveringSize || kind == LastAxisKind::kLeadingStride;
  auto const_value = As<ConstInt>(value);
  CHECK_SPAN(const_value, span)
      << "PackFp4: last-axis " << (kind == LastAxisKind::kOffset ? "offset" : "extent/stride")
      << " must be a static ConstInt (pl.FP4 packing is static-only). Use pl.FP4E2M1X2 with a "
         "physical last dimension, or the UINT8 half-width cache ABI (e.g. DSv4.1)";
  if (covering) {
    if (kind == LastAxisKind::kLeadingStride) {
      CHECK_SPAN(const_value->value_ > 0 && const_value->value_ % 2 == 0, span)
          << "PackFp4: static leading stride must be a positive even logical element count, got "
          << const_value->value_;
    } else {
      CHECK_SPAN(const_value->value_ > 0 && const_value->value_ % 2 == 0, span)
          << "PackFp4: static last dimension must be a positive even logical extent, got "
          << const_value->value_;
    }
  } else {
    CHECK_SPAN(const_value->value_ >= 0, span)
        << "PackFp4: static last-axis coordinate must be non-negative, got " << const_value->value_;
    CHECK_SPAN(const_value->value_ % 2 == 0, span)
        << "PackFp4: last-axis offset must be even so it starts on a packed pair, got "
        << const_value->value_;
  }
  return MakeIndexConst(const_value->value_ / 2, span);
}

std::vector<ExprPtr> HalveLastAxisVector(std::vector<ExprPtr> dims, LastAxisKind kind, const Span& span) {
  if (dims.empty()) return dims;
  dims.back() = HalveLastAxisExpr(dims.back(), kind, span);
  return dims;
}

std::vector<ExprPtr> HalveLeadingStrides(const std::vector<ExprPtr>& strides, const Span& span) {
  if (strides.empty()) return strides;
  auto last = As<ConstInt>(strides.back());
  CHECK_SPAN(last && last->value_ == 1, span)
      << "PackFp4: packed FP4 requires a contiguous last axis (stride[-1] == 1); "
         "non-adjacent nibbles cannot form an FP4E2M1X2 pair";
  std::vector<ExprPtr> packed = strides;
  packed.back() = MakeIndexConst(1, span);
  // Leading strides count last-axis elements (e.g. ND row stride == last dim).
  for (size_t i = 0; i + 1 < packed.size(); ++i) {
    packed[i] = HalveLastAxisExpr(packed[i], LastAxisKind::kLeadingStride, span);
  }
  return packed;
}

bool TypeHasLogicalFp4(const TypePtr& type) {
  if (!type) return false;
  if (auto tuple_type = As<TupleType>(type)) {
    for (const auto& elem : tuple_type->types_) {
      if (TypeHasLogicalFp4(elem)) return true;
    }
    return false;
  }
  if (auto scalar = As<ScalarType>(type)) return scalar->dtype_.IsLogicalFp4();
  if (auto shaped = As<ShapedType>(type)) return shaped->dtype_.IsLogicalFp4();
  if (auto buffer = As<BufferType>(type)) return buffer->dtype_.IsLogicalFp4();
  if (auto multi = As<MultiBufferType>(type)) return TypeHasLogicalFp4(multi->element_type_);
  return false;
}

void RejectFp4CubeSpace(DataType dtype, std::optional<MemorySpace> space, const Span& span,
                        const char* where) {
  if (!dtype.IsFp4Family() || !space.has_value() || !IsCubeMemorySpace(*space)) return;
  CHECK_SPAN(false, span) << "PackFp4: " << where
                          << " cannot place FP4/FP4E2M1X2 on cube memory (Mat/Left/Right/Acc/Bias/scale); "
                             "keep the value on Vec and cast to FP8 before any cube path, got "
                          << MemorySpaceToString(*space);
}

void RejectFp4UnsupportedLayout(const std::optional<TensorView>& tensor_view,
                                const std::optional<TileView>& tile_view, const Span& span) {
  if (tensor_view.has_value()) {
    CHECK_SPAN(tensor_view->layout != TensorLayout::DN, span)
        << "PackFp4: FP4 DN layout is unsupported; packing only the last axis would pair "
           "non-adjacent nibbles";
    CHECK_SPAN(tensor_view->layout != TensorLayout::NZ, span) << "PackFp4: FP4 NZ layout is unsupported";
  }
  if (tile_view.has_value()) {
    CHECK_SPAN(tile_view->blayout != TileLayout::col_major, span)
        << "PackFp4: FP4 col_major tiles are unsupported; packing only the last axis would pair "
           "non-adjacent nibbles";
  }
}

/// Layout / cube / distributed guards for the whole FP4 family (logical + hand-written packed).
/// Must run even when PackType skips rewriting already-packed types.
void EnforceFp4FamilyGuards(const TypePtr& type, const Span& span) {
  if (!type) return;
  if (auto dist = As<DistributedTensorType>(type)) {
    CHECK_SPAN(!dist->dtype_.IsFp4Family(), span)
        << "PackFp4: DistributedTensor FP4 is not supported in this release (multi-device / remote "
           "FP4 is TODO; see docs/en/dev/fp4.md)";
    return;
  }
  if (auto tuple_type = As<TupleType>(type)) {
    for (const auto& elem : tuple_type->types_) EnforceFp4FamilyGuards(elem, span);
    return;
  }
  if (auto multi = As<MultiBufferType>(type)) {
    EnforceFp4FamilyGuards(multi->element_type_, span);
    return;
  }
  if (auto tensor = As<TensorType>(type); tensor && tensor->dtype_.IsFp4Family()) {
    RejectFp4UnsupportedLayout(tensor->tensor_view_, std::nullopt, span);
    return;
  }
  if (auto tile = As<TileType>(type); tile && tile->dtype_.IsFp4Family()) {
    RejectFp4CubeSpace(tile->dtype_, tile->GetMemorySpace(), span, "TileType");
    RejectFp4UnsupportedLayout(std::nullopt, tile->tile_view_, span);
    return;
  }
  if (auto buffer = As<BufferType>(type); buffer && buffer->dtype_.IsFp4Family()) {
    RejectFp4CubeSpace(buffer->dtype_, buffer->memory_space_, span, "BufferType");
    CHECK_SPAN(buffer->blayout_ != TileLayout::col_major, span)
        << "PackFp4: FP4 col_major buffers are unsupported; packing only the last axis would pair "
           "non-adjacent nibbles";
  }
}

std::optional<TensorView> PackTensorView(std::optional<TensorView> view, const Span& span) {
  if (!view.has_value()) return view;
  RejectFp4UnsupportedLayout(view, std::nullopt, span);
  view->valid_shape = HalveLastAxisVector(std::move(view->valid_shape), LastAxisKind::kCoveringSize, span);
  view->stride = HalveLeadingStrides(view->stride, span);
  return view;
}

std::optional<TileView> PackTileView(std::optional<TileView> view, const Span& span) {
  if (!view.has_value()) return view;
  RejectFp4UnsupportedLayout(std::nullopt, view, span);
  view->valid_shape = HalveLastAxisVector(std::move(view->valid_shape), LastAxisKind::kCoveringSize, span);
  view->stride = HalveLeadingStrides(view->stride, span);
  return view;
}

TypePtr PackType(const TypePtr& type, const Span& span) {
  if (!type) return type;

  // Family layout/cube/distributed rejects apply before the logical-only rewrite early-return.
  EnforceFp4FamilyGuards(type, span);

  if (As<DistributedTensorType>(type)) return type;

  if (!TypeHasLogicalFp4(type)) return type;

  if (auto tuple_type = As<TupleType>(type)) {
    std::vector<TypePtr> packed;
    packed.reserve(tuple_type->types_.size());
    bool changed = false;
    for (const auto& elem : tuple_type->types_) {
      auto new_elem = PackType(elem, span);
      if (new_elem.get() != elem.get()) changed = true;
      packed.push_back(std::move(new_elem));
    }
    if (!changed) return type;
    return std::make_shared<TupleType>(std::move(packed));
  }

  if (auto scalar = As<ScalarType>(type)) {
    // Scalar FP4 read/write would change nibble vs x2-carrier semantics; reject.
    CHECK_SPAN(!scalar->dtype_.IsLogicalFp4(), span)
        << "PackFp4: scalar FP4 (tensor/tile read/write) is not supported; use tile load/store or "
           "cast to a wider dtype first";
    return type;
  }

  if (auto tensor = As<TensorType>(type)) {
    CHECK_SPAN(!tensor->shape_.empty(), span) << "PackFp4: packed FP4 tensors must have rank >= 1";
    auto packed_shape = HalveLastAxisVector(tensor->shape_, LastAxisKind::kCoveringSize, span);
    auto packed_view = PackTensorView(tensor->tensor_view_, span);
    return std::make_shared<TensorType>(std::move(packed_shape), DataType::FP4E2M1X2, tensor->memref_,
                                        std::move(packed_view));
  }

  if (auto tile = As<TileType>(type)) {
    CHECK_SPAN(!tile->shape_.empty(), span) << "PackFp4: packed FP4 tiles must have rank >= 1";
    RejectFp4CubeSpace(tile->dtype_, tile->GetMemorySpace(), span, "TileType");
    auto packed_shape = HalveLastAxisVector(tile->shape_, LastAxisKind::kCoveringSize, span);
    auto packed_view = PackTileView(tile->tile_view_, span);
    return std::make_shared<TileType>(std::move(packed_shape), DataType::FP4E2M1X2, tile->memref_,
                                      std::move(packed_view), tile->memory_space_);
  }

  if (auto array_type = As<ArrayType>(type)) {
    CHECK_SPAN(false, span) << "PackFp4: ArrayType cannot carry logical FP4";
  }

  if (auto buffer = As<BufferType>(type)) {
    CHECK_SPAN(!buffer->shape_.empty(), span) << "PackFp4: packed FP4 buffers must have rank >= 1";
    RejectFp4CubeSpace(buffer->dtype_, buffer->memory_space_, span, "BufferType");
    CHECK_SPAN(buffer->blayout_ != TileLayout::col_major, span)
        << "PackFp4: FP4 col_major buffers are unsupported; packing only the last axis would pair "
           "non-adjacent nibbles";
    auto packed_shape = buffer->shape_;
    CHECK_SPAN(packed_shape.back() > 0 && packed_shape.back() % 2 == 0, span)
        << "PackFp4: BufferType last dimension must be a positive even logical extent, got "
        << packed_shape.back();
    packed_shape.back() /= 2;
    auto packed_valid = buffer->valid_shape_;
    if (!packed_valid.empty()) {
      if (packed_valid.back() == -1) {
        // Dynamic valid extent stays dynamic.
      } else {
        CHECK_SPAN(packed_valid.back() >= 0 && packed_valid.back() % 2 == 0, span)
            << "PackFp4: BufferType last valid dimension must be even or -1, got " << packed_valid.back();
        packed_valid.back() /= 2;
      }
    }
    return std::make_shared<BufferType>(std::move(packed_shape), DataType::FP4E2M1X2, buffer->memory_space_,
                                        std::move(packed_valid), buffer->blayout_, buffer->slayout_,
                                        buffer->fractal_, buffer->pad_, buffer->compact_);
  }

  if (auto multi = As<MultiBufferType>(type)) {
    auto packed_elem = std::dynamic_pointer_cast<const BufferType>(PackType(multi->element_type_, span));
    CHECK_SPAN(packed_elem, span) << "PackFp4: MultiBufferType element must remain a BufferType";
    return std::make_shared<MultiBufferType>(std::move(packed_elem), multi->slot_count_);
  }

  return type;
}

size_t TupleRank(const ExprPtr& tuple_expr) {
  if (auto make_tuple = As<MakeTuple>(tuple_expr)) return make_tuple->elements_.size();
  if (auto tuple_type = As<TupleType>(tuple_expr->GetType())) return tuple_type->types_.size();
  return 0;
}

ExprPtr RewriteLastAxisTuple(const ExprPtr& tuple_expr, LastAxisKind kind, const Span& span) {
  if (!tuple_expr) return tuple_expr;
  const size_t rank = TupleRank(tuple_expr);
  if (rank == 0) return tuple_expr;
  std::vector<ExprPtr> elements;
  elements.reserve(rank);
  if (auto make_tuple = As<MakeTuple>(tuple_expr)) {
    elements = make_tuple->elements_;
  } else {
    for (size_t i = 0; i < rank; ++i) {
      elements.push_back(std::make_shared<TupleGetItemExpr>(tuple_expr, static_cast<int>(i), span));
    }
  }
  elements.back() = HalveLastAxisExpr(elements.back(), kind, span);
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

bool CallNeedsLastAxisPack(const CallPtr& op) {
  if (TypeHasLogicalFp4(op->GetType())) return true;
  for (const auto& arg : op->args_) {
    if (arg && TypeHasLogicalFp4(arg->GetType())) return true;
  }
  for (const auto& [key, value] : op->kwargs_) {
    if (value.type() == typeid(DataType) && AnyCast<DataType>(value, key).IsLogicalFp4()) return true;
  }
  return false;
}

bool TypeHasFp4Family(const TypePtr& type) {
  if (!type) return false;
  if (auto tuple_type = As<TupleType>(type)) {
    for (const auto& elem : tuple_type->types_) {
      if (TypeHasFp4Family(elem)) return true;
    }
    return false;
  }
  if (auto scalar = As<ScalarType>(type)) return scalar->dtype_.IsFp4Family();
  if (auto shaped = As<ShapedType>(type)) return shaped->dtype_.IsFp4Family();
  if (auto buffer = As<BufferType>(type)) return buffer->dtype_.IsFp4Family();
  if (auto multi = As<MultiBufferType>(type)) return TypeHasFp4Family(multi->element_type_);
  return false;
}

void RejectFp4UnsupportedOp(const CallPtr& op, const char* reason) {
  CHECK_SPAN(false, op->span_) << "PackFp4: " << op->op_->name_ << " is not supported for FP4/FP4E2M1X2 ("
                               << reason << "; see docs/en/dev/fp4.md)";
}

std::pair<std::vector<std::pair<std::string, std::any>>, bool> RewriteDtypeKwargs(
    const std::vector<std::pair<std::string, std::any>>& kwargs) {
  std::vector<std::pair<std::string, std::any>> rewritten;
  rewritten.reserve(kwargs.size());
  bool changed = false;
  for (const auto& [key, value] : kwargs) {
    if (value.type() == typeid(DataType)) {
      DataType dtype = AnyCast<DataType>(value, key);
      if (dtype.IsLogicalFp4()) {
        rewritten.emplace_back(key, DataType::FP4E2M1X2);
        changed = true;
        continue;
      }
    }
    rewritten.emplace_back(key, value);
  }
  return {std::move(rewritten), changed};
}

std::optional<DataType> GetCallDtypeKwarg(const CallPtr& op) {
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "dtype" && value.type() == typeid(DataType)) {
      return AnyCast<DataType>(value, key);
    }
  }
  return std::nullopt;
}

/// Shape operands of reinterpret_view are in *destination* element units.
/// Only pack them when the destination is still logical FP4.
bool ReinterpretViewNeedsShapePack(const CallPtr& original) {
  if (auto dtype = GetCallDtypeKwarg(original)) return dtype->IsLogicalFp4();
  if (auto shaped = As<ShapedType>(original->GetType())) return shaped->dtype_.IsLogicalFp4();
  return false;
}

bool ArgIsRankedTuple(const ExprPtr& arg) {
  if (!arg) return false;
  if (auto make_tuple = As<MakeTuple>(arg)) return !make_tuple->elements_.empty();
  if (auto tuple_type = As<TupleType>(arg->GetType())) return !tuple_type->types_.empty();
  return false;
}

bool ArgIsIntegerScalarCoord(const ExprPtr& arg) {
  if (!arg || ArgIsRankedTuple(arg)) return false;
  if (As<ConstInt>(arg)) return true;
  if (auto scalar = As<ScalarType>(arg->GetType())) {
    return scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX;
  }
  return false;
}

void RejectUnrewrittenFp4CoordArgs(const CallPtr& op, const std::vector<ExprPtr>& args) {
  for (const auto& arg : args) {
    if (!ArgIsRankedTuple(arg) && !ArgIsIntegerScalarCoord(arg)) continue;
    CHECK_SPAN(false, op->span_)
        << "PackFp4: " << op->op_->name_
        << " last-axis coordinates are not yet rewritten; add a RewriteCallArgs branch "
           "or confirm the op has no packed-axis shape/offset operands (see docs/en/dev/fp4.md)";
  }
}

bool CallTouchesFp4Family(const CallPtr& op) {
  if (TypeHasFp4Family(op->GetType())) return true;
  for (const auto& arg : op->args_) {
    if (arg && TypeHasFp4Family(arg->GetType())) return true;
  }
  for (const auto& [key, value] : op->kwargs_) {
    if (value.type() == typeid(DataType) && AnyCast<DataType>(value, key).IsFp4Family()) return true;
  }
  return false;
}

std::vector<ExprPtr> RewriteCallArgs(const CallPtr& op, const CallPtr& original, bool rewrite_coords) {
  std::vector<ExprPtr> args = op->args_;
  const Span& span = op->span_;
  auto rewrite_tuple = [&](size_t index, LastAxisKind kind) {
    if (!rewrite_coords) return;
    if (index < args.size() && args[index]) {
      args[index] = RewriteLastAxisTuple(args[index], kind, span);
    }
  };
  auto rewrite_scalar = [&](size_t index, LastAxisKind kind) {
    if (!rewrite_coords) return;
    if (index < args.size() && args[index]) {
      args[index] = HalveLastAxisExpr(args[index], kind, span);
    }
  };

  if (IsOp(op->op_, "tensor.slice") || IsOp(op->op_, "tile.slice")) {
    rewrite_tuple(1, LastAxisKind::kCoveringSize);
    rewrite_tuple(2, LastAxisKind::kOffset);
    rewrite_tuple(3, LastAxisKind::kCoveringSize);
  } else if (IsOp(op->op_, "tile.load")) {
    rewrite_tuple(1, LastAxisKind::kOffset);
    rewrite_tuple(2, LastAxisKind::kCoveringSize);
    rewrite_tuple(3, LastAxisKind::kCoveringSize);
  } else if (IsOp(op->op_, "tile.store")) {
    rewrite_tuple(1, LastAxisKind::kOffset);
    rewrite_tuple(3, LastAxisKind::kCoveringSize);
  } else if (IsOp(op->op_, "tensor.assemble") || IsOp(op->op_, "tile.assemble")) {
    rewrite_tuple(2, LastAxisKind::kOffset);
  } else if (IsOp(op->op_, "tensor.view")) {
    rewrite_tuple(1, LastAxisKind::kCoveringSize);
    rewrite_tuple(2, LastAxisKind::kCoveringSize);  // optional valid_shape
  } else if (IsOp(op->op_, "tensor.fillpad_expand") || IsOp(op->op_, "tile.fillpad_expand")) {
    rewrite_tuple(1, LastAxisKind::kCoveringSize);
  } else if (IsOp(op->op_, "tile.reinterpret_view") || IsOp(op->op_, "tensor.reinterpret_view")) {
    if (rewrite_coords && ReinterpretViewNeedsShapePack(original)) {
      rewrite_tuple(1, LastAxisKind::kCoveringSize);
    }
  } else if (IsOp(op->op_, "tensor.create") || IsOp(op->op_, "tile.create") || IsOp(op->op_, "tensor.full") ||
             IsOp(op->op_, "tile.full")) {
    rewrite_tuple(0, LastAxisKind::kCoveringSize);
    if (rewrite_coords && args.size() >= 2 && args[1]) {
      if (auto const_int = As<ConstInt>(args[1]); const_int && const_int->dtype().IsLogicalFp4()) {
        args[1] = std::make_shared<ConstInt>(const_int->value_, DataType::FP4E2M1X2, const_int->span_);
      } else if (auto const_float = As<ConstFloat>(args[1]);
                 const_float && const_float->dtype().IsLogicalFp4()) {
        args[1] = std::make_shared<ConstFloat>(const_float->value_, DataType::FP4E2M1X2, const_float->span_);
      }
    }
  } else if (IsOp(op->op_, "tensor.set_validshape") || IsOp(op->op_, "tile.set_validshape")) {
    rewrite_scalar(2, LastAxisKind::kCoveringSize);
  } else if (IsOp(op->op_, "tile.extract")) {
    rewrite_scalar(2, LastAxisKind::kOffset);
    rewrite_tuple(3, LastAxisKind::kCoveringSize);
  } else if (IsOp(op->op_, "system.cacheinvalid")) {
    if (args.size() >= 3) {
      rewrite_tuple(1, LastAxisKind::kCoveringSize);
      rewrite_tuple(2, LastAxisKind::kOffset);
    }
  } else {
    RejectUnrewrittenFp4CoordArgs(op, args);
  }
  return args;
}

class PackFp4Mutator : public IRMutator {
 public:
  FunctionPtr VisitFunction(const FunctionPtr& func) override {
    ScopedTypeSpan scope(this, func ? func->span_ : Span::unknown());
    return IRMutator::VisitFunction(func);
  }

  TypePtr RemapTypeViaVisitor(const TypePtr& type) override {
    return PackType(IRMutator::RemapTypeViaVisitor(type), CurrentTypeSpan());
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    ScopedTypeSpan scope(this, op->span_);
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    ScopedTypeSpan scope(this, op->span_);
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    ScopedTypeSpan scope(this, op->span_);
    // Reject FP4-family ops that PackFp4 does not rewrite, even when dtype is already packed.
    if (TypeHasFp4Family(op->GetType()) ||
        std::any_of(op->args_.begin(), op->args_.end(),
                    [](const ExprPtr& a) { return a && TypeHasFp4Family(a->GetType()); })) {
      if (IsOp(op->op_, "tensor.transpose") || IsOp(op->op_, "tile.transpose")) {
        RejectFp4UnsupportedOp(op, "transpose is not supported for FP4 in this release");
      }
      if (IsOp(op->op_, "tensor.reshape") || IsOp(op->op_, "tile.reshape")) {
        RejectFp4UnsupportedOp(op, "reshape is not supported for FP4 in this release");
      }
      if (IsOp(op->op_, "tensor.read") || IsOp(op->op_, "tensor.write") || IsOp(op->op_, "tile.read") ||
          IsOp(op->op_, "tile.write")) {
        RejectFp4UnsupportedOp(op, "scalar read/write would change nibble vs packed-pair semantics");
      }
      if (IsOp(op->op_, "pld.tile.remote_load") || IsOp(op->op_, "pld.tile.remote_store") ||
          IsOp(op->op_, "pld.tensor.remote_store") || IsOp(op->op_, "pld.tile.put") ||
          IsOp(op->op_, "pld.tile.get") || IsOp(op->op_, "pld.tensor.put") ||
          IsOp(op->op_, "pld.tensor.get") || IsOp(op->op_, "pld.tensor.window")) {
        RejectFp4UnsupportedOp(op, "distributed / multi-device FP4 is TODO");
      }
    }
    const bool pack_coords = CallNeedsLastAxisPack(op);
    const bool touches_fp4_family = CallTouchesFp4Family(op);
    ExprPtr visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    INTERNAL_CHECK_SPAN(call, op->span_) << "PackFp4: Call mutation must remain a Call";
    auto [new_kwargs, kwargs_changed] = RewriteDtypeKwargs(call->kwargs_);
    std::vector<ExprPtr> new_args = call->args_;
    bool changed = kwargs_changed;
    if (pack_coords) {
      new_args = RewriteCallArgs(call, op, /*rewrite_coords=*/true);
      changed = true;
    } else if (touches_fp4_family) {
      // Already-packed (or family-only) calls: still walk the whitelist so unlisted
      // ops with shape/offset operands loud-fail instead of silently skipping.
      RewriteCallArgs(call, op, /*rewrite_coords=*/false);
    }
    if (!changed) return visited;
    return std::make_shared<Call>(call->op_, std::move(new_args), std::move(new_kwargs), call->attrs_,
                                  call->GetType(), call->span_);
  }

 private:
  class ScopedTypeSpan {
   public:
    ScopedTypeSpan(PackFp4Mutator* mutator, const Span& span) : mutator_(mutator) {
      mutator_->type_span_stack_.push_back(span);
    }
    ~ScopedTypeSpan() { mutator_->type_span_stack_.pop_back(); }

   private:
    PackFp4Mutator* mutator_;
  };

  Span CurrentTypeSpan() const {
    return type_span_stack_.empty() ? Span::unknown() : type_span_stack_.back();
  }

  std::vector<Span> type_span_stack_;
};

class RejectLeftoverLogicalFp4 : public IRVisitor {
 public:
  void VisitFunction(const FunctionPtr& func) override {
    for (const auto& param : func->params_) CheckType(param->GetType(), func->span_);
    for (const auto& rt : func->return_types_) CheckType(rt, func->span_);
    IRVisitor::VisitFunction(func);
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    CheckType(op->GetType(), op->span_);
    IRVisitor::VisitVarLike_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    CheckType(op->GetType(), op->span_);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MakeTuplePtr& op) override {
    CheckType(op->GetType(), op->span_);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const TupleGetItemExprPtr& op) override {
    CheckType(op->GetType(), op->span_);
    IRVisitor::VisitExpr_(op);
  }

 private:
  static void CheckType(const TypePtr& type, const Span& span) {
    CHECK_SPAN(!TypeHasLogicalFp4(type), span)
        << "PackFp4: logical DataType.FP4 must not remain after packing; expected FP4E2M1X2";
  }
};

FunctionPtr TransformPackFp4(const FunctionPtr& func) {
  if (!func) return func;
  PackFp4Mutator mutator;
  auto packed = mutator.VisitFunction(func);
  RejectLeftoverLogicalFp4 reject;
  reject.VisitFunction(packed);
  return packed;
}

}  // namespace

namespace pass {

Pass PackFp4() { return CreateFunctionPass(TransformPackFp4, "PackFp4", kPackFp4Properties); }

}  // namespace pass

}  // namespace ir
}  // namespace pypto
