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
 * @file transform.cpp
 * @brief Shape transformation tensor operations (reshape, transpose)
 *
 * This file implements shape transformation operations for tensors including
 * reshape and transpose operations.
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

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {
// ============================================================================
// Helper Functions (file-local)
// ============================================================================

/**
 * @brief Normalize axis index to handle negative indexing
 *
 * @param axis The axis index (can be negative)
 * @param ndim The number of dimensions
 * @return The normalized axis index
 */
int NormalizeAxis(int axis, size_t ndim) {
  if (axis < 0) {
    axis += static_cast<int>(ndim);
  }
  CHECK(axis >= 0 && axis < static_cast<int>(ndim))
      << "Axis " << axis << " is out of range for " << ndim << "D tensor";
  return axis;
}

// Stride / shape-product helpers live in
// include/pypto/ir/transforms/utils/tensor_view_semantics.h so passes,
// verifiers, and op type-inference all share one implementation.
using tensor_view_semantics::BuildRowMajorStrides;
using tensor_view_semantics::ComputeShapeProduct;
using tensor_view_semantics::MakeIndexMul;

}  // anonymous namespace

// ============================================================================
// Type Inference Functions
// ============================================================================

TypePtr DeduceTensorReshapeType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.reshape requires 2 arguments (input, shape) with optional 3rd (valid_shape)
  CHECK(args.size() == 2 || args.size() == 3)
      << "tensor.reshape requires 2 or 3 arguments (input, shape[, valid_shape]), but got " << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.reshape requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tensor.reshape requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  // Validate all shape elements are ScalarType with integer dtype
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.reshape shape tuple element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.reshape shape tuple element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  // Extract new shape dimensions
  // If args[1] is MakeTuple, extract elements directly to preserve constants
  // Otherwise use TupleGetItemExpr for runtime tuples
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_tuple_type->types_.size());

  if (auto make_tuple = As<MakeTuple>(args[1])) {
    // MakeTuple: extract elements directly to preserve ConstInt
    new_shape = make_tuple->elements_;
  } else {
    // Runtime tuple: use TupleGetItemExpr
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      new_shape.emplace_back(
          std::make_shared<TupleGetItemExpr>(args[1], static_cast<int>(i), args[1]->span_));
    }
  }

  // For static shapes, verify that the total number of elements matches
  int64_t old_product = ComputeShapeProduct(tensor_type->shape_);
  int64_t new_product = ComputeShapeProduct(new_shape);

  if (old_product > 0 && new_product > 0) {
    CHECK(old_product == new_product) << "tensor.reshape: cannot reshape tensor of size " << old_product
                                      << " into shape with size " << new_product;
  }

  // Return new TensorType with reshaped dimensions and same dtype
  // If valid_shape is provided as 3rd argument, store it in TensorView
  if (args.size() == 3) {
    auto valid_shape_tuple = As<MakeTuple>(args[2]);
    CHECK(valid_shape_tuple) << "tensor.reshape valid_shape (3rd argument) must be a MakeTuple";
    TensorView tensor_view({}, TensorLayout::ND, valid_shape_tuple->elements_);
    return std::make_shared<TensorType>(new_shape, tensor_type->dtype_, std::nullopt,
                                        std::make_optional(std::move(tensor_view)));
  }
  return std::make_shared<TensorType>(new_shape, tensor_type->dtype_);
}

TypePtr DeduceTensorTransposeType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.transpose requires 3 arguments (input, axis1, axis2) with optional 4th (valid_shape)
  CHECK(args.size() == 3 || args.size() == 4)
      << "tensor.transpose requires 3 or 4 arguments (input, axis1, axis2[, valid_shape]), but got "
      << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.transpose requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  const auto& input_shape = tensor_type->shape_;
  size_t ndim = input_shape.size();

  CHECK(ndim >= 2) << "tensor.transpose requires at least 2 dimensions, but got " << ndim;

  // Second argument is axis1 (ConstInt)
  auto axis1_const = As<ConstInt>(args[1]);
  CHECK(axis1_const) << "tensor.transpose requires second argument (axis1) to be a ConstInt";

  // Third argument is axis2 (ConstInt)
  auto axis2_const = As<ConstInt>(args[2]);
  CHECK(axis2_const) << "tensor.transpose requires third argument (axis2) to be a ConstInt";

  // Normalize axes (handle negative indexing)
  int axis1 = NormalizeAxis(static_cast<int>(axis1_const->value_), ndim);
  int axis2 = NormalizeAxis(static_cast<int>(axis2_const->value_), ndim);

  CHECK(axis1 != axis2) << "tensor.transpose: axis1 and axis2 must be different, but got axis1=" << axis1
                        << ", axis2=" << axis2;

  // Create new shape by swapping the specified dimensions
  std::vector<ExprPtr> new_shape = input_shape;
  std::swap(new_shape[axis1], new_shape[axis2]);

  // Encode the post-transpose layout via two complementary mechanisms:
  //
  //  1. Layout tag (ND/DN). Only the canonical trailing-two-dim swap can be
  //     described by toggling the tag — non-trailing transposes leave the tag
  //     unchanged because ND/DN only describes the trailing two dims. PTOAS
  //     reads this tag and EmitMakeTensorViews / EmitTileLoadPTO use it to
  //     drive the implicit "swap last two dims" path used by
  //     tile.load(transpose=True) sources (see LowerTransposeLoadParamLayout).
  //
  //  2. Explicit strides. tensor.transpose at orchestration level lowers to
  //     runtime Tensor::transpose, a metadata-only swap of shapes / offsets;
  //     the underlying GM data stays in the source's row-major layout. So the
  //     physical strides for the post-transpose view are the source's strides
  //     reordered at (axis1, axis2). Recording those strides on the result
  //     type lets EmitMakeTensorViews emit a make_tensor_view that matches
  //     the actual memory layout instead of fabricating column-major strides
  //     from the swapped shape (which would be wrong for a row-major source).
  //
  // Why both: the layout tag is needed for PTOAS contracts (it expects DN on
  // any trailing-transposed view at the kernel boundary), while the explicit
  // strides are needed because ND/DN alone cannot distinguish "source data is
  // column-major in the IR shape" (matmul's tile.load(transpose=True) path)
  // from "source data is row-major and we want a transposed view of it"
  // (this op's path). The codegen disambiguates by checking
  // tensor_view_->stride: if it's non-empty, skip the implicit DN swap.
  bool is_trailing_swap =
      (static_cast<size_t>(axis1) == ndim - 1 && static_cast<size_t>(axis2) == ndim - 2) ||
      (static_cast<size_t>(axis1) == ndim - 2 && static_cast<size_t>(axis2) == ndim - 1);

  TensorLayout in_layout = TensorLayout::ND;
  PadValue pad = PadValue::null;
  std::vector<ExprPtr> in_valid_shape;
  std::vector<ExprPtr> in_stride;
  if (tensor_type->tensor_view_.has_value()) {
    in_layout = tensor_type->tensor_view_->layout;
    pad = tensor_type->tensor_view_->pad;
    in_valid_shape = tensor_type->tensor_view_->valid_shape;
    in_stride = tensor_type->tensor_view_->stride;
  }
  TensorLayout out_layout = in_layout;
  if (is_trailing_swap) {
    out_layout = (in_layout == TensorLayout::ND) ? TensorLayout::DN : TensorLayout::ND;
  }

  // Resolve the post-transpose strides: prefer input's explicit strides;
  // otherwise derive row-major strides from the input shape (works for both
  // static and dynamic dims — ConstInt chains fold). Then swap at the same
  // axes as the shape swap to reflect the metadata-only transpose.
  std::vector<ExprPtr> result_stride =
      !in_stride.empty() ? std::move(in_stride) : BuildRowMajorStrides(input_shape);
  if (!result_stride.empty()) {
    std::swap(result_stride[axis1], result_stride[axis2]);
  }

  // Carry forward valid_shape. Two cases differ in coordinate system:
  //   - explicit 4th arg: already in the OUTPUT's coordinate system (user
  //     supplies it for the transposed tensor), so use as-is. We also CHECK
  //     that its rank matches the tensor rank to catch user errors early.
  //   - inherited from input's tensor_view_: in the INPUT's coordinate system,
  //     so swap at (axis1, axis2) to match the output shape.
  std::vector<ExprPtr> valid_shape;
  if (args.size() == 4) {
    auto valid_shape_tuple = As<MakeTuple>(args[3]);
    CHECK(valid_shape_tuple) << "tensor.transpose valid_shape (4th argument) must be a MakeTuple";
    valid_shape = valid_shape_tuple->elements_;
    CHECK(valid_shape.size() == ndim) << "tensor.transpose: valid_shape rank (" << valid_shape.size()
                                      << ") must match tensor rank (" << ndim << ")";
  } else if (!in_valid_shape.empty()) {
    valid_shape = std::move(in_valid_shape);
    std::swap(valid_shape[axis1], valid_shape[axis2]);
  }

  // Attach a TensorView whenever any non-default field needs to travel with
  // the result type (strides, non-default layout, valid_shape, or pad). The
  // identity transpose-of-transpose collapses back to a bare TensorType
  // because the strides round-trip to row-major and layout flips back to ND.
  auto strides_match_row_major = [&]() {
    if (result_stride.empty() || out_layout != TensorLayout::ND) return false;
    auto canonical = BuildRowMajorStrides(new_shape);
    if (canonical.size() != result_stride.size()) return false;
    // Equal when both are ConstInt with the same value, OR when the two
    // ExprPtrs point to the same underlying node (covers symbolic Var dims:
    // BuildRowMajorStrides reuses the input shape's ExprPtrs so that the
    // round-trip transpose-of-transpose lands on identical pointers for the
    // dynamic-shape case).
    return std::equal(canonical.begin(), canonical.end(), result_stride.begin(),
                      [](const ExprPtr& a, const ExprPtr& b) {
                        auto ca = As<ConstInt>(a);
                        auto cb = As<ConstInt>(b);
                        if (ca && cb) return ca->value_ == cb->value_;
                        return a == b;
                      });
  };
  bool record_stride = !result_stride.empty() && !strides_match_row_major();
  if (record_stride || out_layout != TensorLayout::ND || !valid_shape.empty() || pad != PadValue::null) {
    TensorView view(record_stride ? std::move(result_stride) : std::vector<ExprPtr>{}, out_layout,
                    std::move(valid_shape), pad);
    return std::make_shared<TensorType>(new_shape, tensor_type->dtype_, std::nullopt,
                                        std::make_optional(std::move(view)));
  }
  return std::make_shared<TensorType>(new_shape, tensor_type->dtype_);
}

// ============================================================================
// Registration Function for Tensor Transform Operations
// ============================================================================

namespace {
// Helper for reading typed kwargs from the deduce-type entry point.
// Mirrors the per-file copies in tile_ops/{memory,reduction,sort}.cpp and
// tensor_ops/reduction.cpp; consider extracting to a shared header in a
// follow-up cleanup.
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::optional<T>& default_value = std::nullopt) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  CHECK(default_value.has_value()) << "tensor op kwarg '" << key << "' is required but missing";
  return *default_value;
}
}  // namespace

TypePtr DeduceTensorAsLayoutType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.as_layout(src, layout=...) — pure layout-tag flip over the same
  // physical memory (RFC #1300 §3.3). Shape changes that come with the flip
  // are mechanical (per RFC §4.2 canonical pair: row-major [..,a,b] ND ≡
  // [..,b,a] DN-packed) and derived here; this op never reshapes.
  // ``tensor.reshape`` is the right tool for shape changes.
  CHECK(args.size() == 1) << "tensor.as_layout requires 1 arg (src) plus a 'layout' kwarg, but got "
                          << args.size() << " positional args";

  auto src_type = As<TensorType>(args[0]->GetType());
  CHECK(src_type) << "tensor.as_layout: src must be TensorType, got " << args[0]->GetType()->TypeName();

  auto new_layout = GetKwarg<TensorLayout>(kwargs, "layout");
  CHECK(new_layout != TensorLayout::NZ)
      << "tensor.as_layout: NZ layout is not allowed on TensorType (NZ is tile-only)";

  // The source must be packed canonical (or bare = implicit ND-packed).
  // Strided sub-views can't be reinterpreted via the §4.2 canonical pair
  // because the offset-map equivalence only holds for the packed forms.
  TensorLayout src_layout =
      src_type->tensor_view_.has_value() ? src_type->tensor_view_->layout : TensorLayout::ND;
  CHECK(src_layout != TensorLayout::NZ)
      << "tensor.as_layout: src has NZ layout (NZ is tile-only and not allowed on TensorType)";
  if (src_type->tensor_view_.has_value() && !src_type->tensor_view_->stride.empty()) {
    auto packed = tensor_view_semantics::BuildLogicalStridesFromLayout(src_type->shape_, src_layout);
    bool is_packed = packed.size() == src_type->tensor_view_->stride.size();
    for (size_t i = 0; is_packed && i < packed.size(); ++i) {
      auto pc = As<ConstInt>(packed[i]);
      auto sc = As<ConstInt>(src_type->tensor_view_->stride[i]);
      // Treat ExprPtr-identity as a match for symbolic dims; otherwise demand
      // ConstInt value equality.
      bool same = (packed[i].get() == src_type->tensor_view_->stride[i].get()) ||
                  (pc && sc && pc->value_ == sc->value_);
      is_packed = is_packed && same;
    }
    CHECK(is_packed) << "tensor.as_layout: src is a strided sub-view (stride does not match packed canonical "
                     << "for layout " << TensorLayoutToString(src_layout)
                     << "). Strided reinterprets are not "
                     << "supported; use tensor.slice or tensor.reshape on the packed parent first.";
  }

  // Derive the target shape:
  //   - same layout (or both effectively ND): identity, shape unchanged
  //   - cross ND ↔ DN: trailing-two-dim swap (the only canonical pair)
  std::vector<ExprPtr> new_shape = src_type->shape_;
  if (src_layout != new_layout) {
    CHECK(src_type->shape_.size() >= 2)
        << "tensor.as_layout: cross-layout reinterpret requires rank >= 2, got " << src_type->shape_.size();
    std::swap(new_shape[new_shape.size() - 2], new_shape[new_shape.size() - 1]);
  }

  auto new_view = tensor_view_semantics::CanonicalizeView(new_shape, new_layout);
  // Preserve view-extending metadata (``valid_shape`` / ``pad``) from the
  // source — both fields describe element-level semantics that are layout-
  // invariant under the §4.2 canonical pair, so dropping them would silently
  // make sliced or fill-padded tensors look like fully-valid views.
  if (src_type->tensor_view_.has_value()) {
    const auto& src_view = src_type->tensor_view_.value();
    if (!src_view.valid_shape.empty()) {
      std::vector<ExprPtr> new_valid_shape = src_view.valid_shape;
      if (src_layout != new_layout && new_valid_shape.size() >= 2) {
        std::iter_swap(new_valid_shape.end() - 2, new_valid_shape.end() - 1);
      }
      new_view.valid_shape = std::move(new_valid_shape);
    }
    new_view.pad = src_view.pad;
  }
  return std::make_shared<TensorType>(new_shape, src_type->dtype_, src_type->memref_,
                                      std::make_optional(std::move(new_view)));
}

REGISTER_OP("tensor.reshape")
    .set_op_category("TensorOp")
    .set_description("Reshape tensor to new shape")
    .add_argument("input", "Input tensor (TensorType)")
    .add_argument("shape", "New shape dimensions (TupleType of ScalarType(INT64))")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorReshapeType(args, kwargs);
    });

REGISTER_OP("tensor.transpose")
    .set_op_category("TensorOp")
    .set_description("Transpose tensor by swapping two axes")
    .add_argument("input", "Input tensor (TensorType)")
    .add_argument("axis1", "First axis to swap (ConstInt)")
    .add_argument("axis2", "Second axis to swap (ConstInt)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorTransposeType(args, kwargs);
    });

REGISTER_OP("tensor.as_layout")
    .set_op_category("TensorOp")
    .set_description(
        "Flip a TensorType's layout tag over the same physical memory (RFC #1300 §3.3). "
        "The trailing-two-dim shape swap that comes with a ND ↔ DN flip is mechanical "
        "and derived here; this op never reshapes (use tensor.reshape for shape changes). "
        "Pure metadata — emits no PTOAS instructions; downstream make_tensor_view "
        "consumes the new view directly. Internal-only; passes (e.g. "
        "LowerTransposeLoadParamLayout) inject this at orch ↔ InCore call sites.")
    .add_argument("input", "Input tensor (TensorType, packed canonical or bare)")
    // Inherit the input's MemRef: ``tensor.as_layout`` is a metadata-only
    // reinterpret of the same physical buffer, so its result must alias the
    // input's allocation. Without this, ``InitMemRef`` would mint a fresh
    // MemRef and allocate a separate buffer that the runtime alias
    // (``Tensor result = input;``) never writes to, leading to silent
    // memory corruption / wrong reads downstream.
    .set_output_memory_inherit_input()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorAsLayoutType(args, kwargs);
    });

TypePtr DeduceTensorConcatType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "tensor.concat requires 2 arguments (src0, src1), got " << args.size();

  auto t0 = As<TensorType>(args[0]->GetType());
  auto t1 = As<TensorType>(args[1]->GetType());
  CHECK(t0) << "tensor.concat: src0 must be TensorType, got " << args[0]->GetType()->TypeName();
  CHECK(t1) << "tensor.concat: src1 must be TensorType, got " << args[1]->GetType()->TypeName();
  CHECK(t0->dtype_ == t1->dtype_) << "tensor.concat: src0 and src1 must have same dtype, got "
                                  << t0->dtype_.ToString() << " and " << t1->dtype_.ToString();
  CHECK(t0->shape_.size() == 2 && t1->shape_.size() == 2) << "tensor.concat requires 2D tensors";

  auto r0 = As<ConstInt>(t0->shape_[0]);
  auto r1 = As<ConstInt>(t1->shape_[0]);
  if (r0 && r1) {
    CHECK(r0->value_ == r1->value_) << "tensor.concat: row count must match, got " << r0->value_ << " vs "
                                    << r1->value_;
  }

  std::vector<ExprPtr> out_shape = {t0->shape_[0]};
  auto c0 = As<ConstInt>(t0->shape_[1]);
  auto c1 = As<ConstInt>(t1->shape_[1]);
  if (c0 && c1) {
    out_shape.push_back(std::make_shared<ConstInt>(c0->value_ + c1->value_, c0->dtype(), args[0]->span_));
  } else {
    out_shape.push_back(std::make_shared<Add>(t0->shape_[1], t1->shape_[1], DataType::INDEX, args[0]->span_));
  }

  return std::make_shared<TensorType>(out_shape, t0->dtype_);
}

REGISTER_OP("tensor.concat")
    .set_op_category("TensorOp")
    .set_description("Concatenate two tensors along column dimension")
    .add_argument("src0", "First source tensor (TensorType)")
    .add_argument("src1", "Second source tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorConcatType(args, kwargs);
    });

TypePtr DeduceTensorSetValidShapeType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3)
      << "tensor.set_validshape requires exactly 3 arguments (tensor, valid_rows, valid_cols), but got "
      << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.set_validshape requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();
  CHECK(tensor_type->shape_.size() == 2)
      << "tensor.set_validshape requires a 2D tensor, but got rank " << tensor_type->shape_.size();

  auto check_scalar_index = [](const ExprPtr& arg, const char* name) {
    auto st = As<ScalarType>(arg->GetType());
    CHECK(st) << "tensor.set_validshape " << name << " must be ScalarType, but got "
              << arg->GetType()->TypeName();
    CHECK(st->dtype_.IsIndexLike()) << "tensor.set_validshape " << name
                                    << " must have dtype INT64, UINT64, or INDEX, but got "
                                    << st->dtype_.ToString();
  };
  check_scalar_index(args[1], "valid_rows");
  check_scalar_index(args[2], "valid_cols");

  auto check_const_bound = [&](const char* name, const ExprPtr& valid, const ExprPtr& bound) {
    if (auto c = As<ConstInt>(valid)) {
      CHECK(c->value_ >= 0) << "tensor.set_validshape " << name << " must be >= 0, got " << c->value_;
      if (auto b = As<ConstInt>(bound)) {
        CHECK(c->value_ <= b->value_) << "tensor.set_validshape " << name << " (" << c->value_
                                      << ") exceeds tensor bound " << b->value_;
      }
    }
  };
  check_const_bound("valid_rows", args[1], tensor_type->shape_[0]);
  check_const_bound("valid_cols", args[2], tensor_type->shape_[1]);

  TensorView tensor_view;
  if (tensor_type->tensor_view_.has_value()) {
    tensor_view = *tensor_type->tensor_view_;
  }
  tensor_view.valid_shape = {args[1], args[2]};

  return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, tensor_type->memref_,
                                      std::make_optional(std::move(tensor_view)));
}

// NOTE: Internal op for compiler-generated code only; should not be exposed to end users in future releases.
REGISTER_OP("tensor.set_validshape")
    .set_op_category("TensorOp")
    .set_description("Update valid-shape metadata of a tensor without data movement (internal)")
    .add_argument("tensor", "Input tensor (TensorType, 2D)")
    .add_argument("valid_rows", "Number of valid rows (ScalarType INDEX/INT64/UINT64)")
    .add_argument("valid_cols", "Number of valid columns (ScalarType INDEX/INT64/UINT64)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorSetValidShapeType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
