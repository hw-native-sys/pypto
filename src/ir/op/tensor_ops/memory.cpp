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
 * @file memory.cpp
 * @brief Memory tensor operations (create, slice, assemble)
 *
 * This file implements memory operations for tensors including allocation,
 * slice creation, and value assembly/updates.
 */

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
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorReadType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.read: Read a scalar value from a tensor at given indices
  // Args: (tensor, indices_tuple)
  // Returns: ScalarType with tensor's element dtype
  CHECK(args.size() == 2) << "tensor.read requires exactly 2 arguments (tensor, indices), but got "
                          << args.size();

  // First argument must be a tensor-shaped value. ``AsTensorTypeLike``
  // accepts both ``TensorType`` and its ``DistributedTensorType`` subclass
  // (which carries its own ``ObjectKind`` and so does NOT match a strict
  // ``As<TensorType>`` cast — see ``.claude/rules/ir-kind-traits.md``).
  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type) << "tensor.read requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (indices)
  auto indices_type = As<TupleType>(args[1]->GetType());
  CHECK(indices_type) << "tensor.read requires indices to be TupleType, but got "
                      << args[1]->GetType()->TypeName();

  // Validate indices count matches tensor rank
  CHECK(indices_type->types_.size() == tensor_type->shape_.size())
      << "tensor.read indices count (" << indices_type->types_.size() << ") must match tensor rank ("
      << tensor_type->shape_.size() << ")";

  // Validate all index elements are ScalarType with integer dtype
  for (size_t i = 0; i < indices_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(indices_type->types_[i]);
    CHECK(scalar_type) << "tensor.read index element " << i << " must be ScalarType, but got "
                       << indices_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.read index element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  return std::make_shared<ScalarType>(tensor_type->dtype_);
}

TypePtr DeduceTensorCreateType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.create: shape is a single TupleType argument
  // dtype comes from kwargs
  CHECK(args.size() == 1) << "tensor.create requires exactly 1 argument (shape tuple), but got "
                          << args.size();

  // Extract dtype from kwargs
  bool found_dtype = false;
  DataType dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "dtype") {
      dtype = AnyCast<DataType>(value, "kwarg key: dtype");
      found_dtype = true;
      break;
    }
  }
  CHECK(found_dtype) << "tensor.create requires 'dtype' kwarg";

  // First argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[0]->GetType());
  CHECK(shape_tuple_type) << "tensor.create requires shape to be TupleType, but got "
                          << args[0]->GetType()->TypeName();

  // Validate all shape elements are ScalarType with integer dtype
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.create shape tuple element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.create shape tuple element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  // Extract shape dimensions
  // If args[0] is MakeTuple, extract elements directly to preserve constants
  // Otherwise use TupleGetItemExpr for runtime tuples
  std::vector<ExprPtr> shape;
  shape.reserve(shape_tuple_type->types_.size());

  if (auto make_tuple = As<MakeTuple>(args[0])) {
    // MakeTuple: extract elements directly to preserve ConstInt
    shape = make_tuple->elements_;
  } else {
    // Runtime tuple: use TupleGetItemExpr
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      shape.emplace_back(std::make_shared<TupleGetItemExpr>(args[0], static_cast<int>(i), args[0]->span_));
    }
  }

  // Extract layout from kwargs (default: ND)
  TensorLayout layout = TensorLayout::ND;
  for (const auto& [key, value] : kwargs) {
    if (key == "layout") {
      layout = AnyCast<TensorLayout>(value, "kwarg key: layout");
      break;
    }
  }

  auto tensor_type = std::make_shared<TensorType>(shape, dtype);
  if (layout != TensorLayout::ND) {
    tensor_type->tensor_view_ = TensorView(std::vector<ExprPtr>{}, layout);
  }
  return tensor_type;
}

TypePtr DeduceTensorSliceType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.slice: (input, shape, offset[, valid_shape[, drop_dims]]).
  //   - valid_shape (4th arg): an empty MakeTuple means "no valid_shape".
  //   - drop_dims (5th arg): a MakeTuple of ConstInt listing axes to erase from
  //     the result type (numpy-style rank reduction); each listed axis must be a
  //     static unit dim of `shape`. An empty / absent operand drops nothing, so
  //     the 3- and 4-arg forms behave exactly as before.
  CHECK(args.size() >= 3 && args.size() <= 5)
      << "tensor.slice requires 3-5 arguments (input, shape, offset[, valid_shape[, drop_dims]]), but got "
      << args.size();

  // First argument must be tensor-shaped. ``AsTensorTypeLike`` accepts both
  // ``TensorType`` and ``DistributedTensorType`` — a slice of a window-buffer
  // tensor is still a view into the same comm-group allocation, so the result
  // type keeps the ``DistributedTensorType`` kind and propagates the original
  // ``window_buffer_`` reference (see end of this function).
  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type) << "tensor.slice requires first argument to be a TensorType or DistributedTensorType, "
                        "but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tensor.slice requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  // Validate all shape elements are ScalarType with integer dtype
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.slice shape tuple element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.slice shape tuple element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  // Third argument must be TupleType (offset)
  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tensor.slice requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  // Validate all offset elements are ScalarType with integer dtype
  for (size_t i = 0; i < offset_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(offset_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.slice offset tuple element " << i << " must be ScalarType, but got "
                       << offset_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.slice offset tuple element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  // Extract the full (pre-reduction) shape dimensions.
  // If args[1] is MakeTuple, extract elements directly to preserve constants
  // Otherwise use TupleGetItemExpr for runtime tuples
  std::vector<ExprPtr> full_shape;
  full_shape.reserve(shape_tuple_type->types_.size());

  if (auto make_tuple = As<MakeTuple>(args[1])) {
    // MakeTuple: extract elements directly to preserve ConstInt
    full_shape = make_tuple->elements_;
  } else {
    // Runtime tuple: use TupleGetItemExpr
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      full_shape.emplace_back(
          std::make_shared<TupleGetItemExpr>(args[1], static_cast<int>(i), args[1]->span_));
    }
  }

  // Optional drop_dims (5th arg): axes erased from the result type. Validated
  // against the full pre-reduction shape (each must be a static unit dim).
  const ExprPtr drop_dims_arg = args.size() == 5 ? args[4] : nullptr;
  const std::vector<int64_t> drop_dims = ParseSliceDropDims(drop_dims_arg, full_shape, "tensor.slice");
  const std::vector<ExprPtr> new_shape = ApplyDropDims(full_shape, drop_dims);

  // Read optional pad_value kwarg (default PadValue::null = no padding).
  PadValue pad_value = PadValue::null;
  for (const auto& [k, v] : kwargs) {
    if (k != "pad_value") continue;
    CHECK(v.type() == typeid(PadValue))
        << "tensor.slice pad_value must be a PadValue enum, got " << v.type().name();
    pad_value = std::any_cast<PadValue>(v);
    CHECK(pad_value == PadValue::null || pad_value == PadValue::zero || pad_value == PadValue::max ||
          pad_value == PadValue::min)
        << "tensor.slice pad_value has invalid enum value: " << static_cast<int>(pad_value);
    break;
  }

  // valid_shape (4th arg): an empty MakeTuple means "no valid_shape" — that form
  // exists so callers can pass drop_dims (5th arg) without a custom valid_shape.
  bool has_valid_shape = false;
  std::vector<ExprPtr> valid_shape;
  if (args.size() >= 4) {
    auto valid_shape_tuple = As<MakeTuple>(args[3]);
    CHECK(valid_shape_tuple) << "tensor.slice valid_shape (4th argument) must be a MakeTuple";
    if (!valid_shape_tuple->elements_.empty()) {
      has_valid_shape = true;
      if (drop_dims.empty()) {
        valid_shape = valid_shape_tuple->elements_;
      } else {
        // valid_shape is given over the full (pre-reduction) rank; drop the same axes.
        CHECK(valid_shape_tuple->elements_.size() == full_shape.size())
            << "tensor.slice valid_shape rank " << valid_shape_tuple->elements_.size()
            << " must match shape rank " << full_shape.size() << " when drop_dims is set";
        valid_shape = ApplyDropDims(valid_shape_tuple->elements_, drop_dims);
      }
    }
  }

  // View preserves dtype but has new shape (which can have different rank than input).
  // If valid_shape is provided or pad_value is set, build a TensorView. When the
  // source is a DistributedTensorType, the result keeps that kind and carries
  // the same window_buffer_ — a slice is still a view into the same comm-group
  // allocation.
  std::optional<TensorView> result_tv;
  if (has_valid_shape) {
    result_tv = TensorView({}, TensorLayout::ND, valid_shape, pad_value);
  } else if (pad_value != PadValue::null) {
    result_tv = TensorView(std::vector<ExprPtr>{}, TensorLayout::ND, std::vector<ExprPtr>{}, pad_value);
  }
  if (auto dt = As<DistributedTensorType>(args[0]->GetType())) {
    return std::make_shared<DistributedTensorType>(new_shape, tensor_type->dtype_, std::nullopt,
                                                   std::move(result_tv), dt->window_buffer_);
  }
  return std::make_shared<TensorType>(new_shape, tensor_type->dtype_, std::nullopt, std::move(result_tv));
}

TypePtr DeduceTensorFillpadType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.fillpad requires exactly 1 argument (tensor), but got " << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.fillpad requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  PadValue pad_value = PadValue::zero;
  for (const auto& kv : kwargs) {
    if (kv.first == "pad_value") {
      pad_value = std::any_cast<PadValue>(kv.second);
      CHECK(pad_value != PadValue::null) << "tensor.fillpad requires pad_value to be zero/max/min, not null";
    }
  }

  std::optional<TensorView> tensor_view = tensor_type->tensor_view_;
  if (tensor_view.has_value()) {
    tensor_view->valid_shape = tensor_type->shape_;
  }

  return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, tensor_type->memref_,
                                      std::move(tensor_view));
}

TypePtr DeduceTensorFillpadExpandType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.fillpad_expand(tensor, shape) — the destination may be larger than the
  // source in either dimension; the source's valid region is copied into the
  // top-left of the destination and the remainder is filled with pad_value.
  CHECK(args.size() == 2) << "tensor.fillpad_expand requires exactly 2 arguments (tensor, shape), but got "
                          << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.fillpad_expand requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  auto shape_tuple = As<MakeTuple>(args[1]);
  CHECK(shape_tuple) << "tensor.fillpad_expand shape must be a literal tuple of constants, but got "
                     << args[1]->GetType()->TypeName();
  const std::vector<ExprPtr>& new_shape = shape_tuple->elements_;
  CHECK(new_shape.size() == tensor_type->shape_.size())
      << "tensor.fillpad_expand shape rank (" << new_shape.size() << ") must match source rank ("
      << tensor_type->shape_.size() << ")";

  for (size_t i = 0; i < new_shape.size(); ++i) {
    auto dst_dim = As<ConstInt>(new_shape[i]);
    CHECK(dst_dim) << "tensor.fillpad_expand shape dimension " << i << " must be a constant integer";
    CHECK(dst_dim->value_ > 0) << "tensor.fillpad_expand shape dimension " << i << " must be positive, got "
                               << dst_dim->value_;
    if (auto src_dim = As<ConstInt>(tensor_type->shape_[i])) {
      CHECK(dst_dim->value_ >= src_dim->value_)
          << "tensor.fillpad_expand destination dimension " << i << " (" << dst_dim->value_
          << ") must be >= source dimension (" << src_dim->value_ << ")";
    }
  }

  PadValue pad_value = PadValue::zero;
  for (const auto& kv : kwargs) {
    if (kv.first == "pad_value") {
      pad_value = std::any_cast<PadValue>(kv.second);
      CHECK(pad_value != PadValue::null)
          << "tensor.fillpad_expand requires pad_value to be zero/max/min, not null";
    }
  }

  // After expand the entire destination is valid. Inherit the source view layout;
  // only the (larger) shape, valid_shape, and pad change.
  std::optional<TensorView> tensor_view = tensor_type->tensor_view_;
  if (tensor_view.has_value()) {
    tensor_view->valid_shape = new_shape;
    tensor_view->pad = pad_value;
  } else {
    TensorView view;
    view.valid_shape = new_shape;
    view.pad = pad_value;
    tensor_view = view;
  }

  // The destination is larger than the source, so it cannot share the source's
  // backing buffer — return a fresh tensor (no inherited MemRef).
  return std::make_shared<TensorType>(new_shape, tensor_type->dtype_, std::nullopt, std::move(tensor_view));
}

TypePtr DeduceTensorAssembleType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.assemble requires exactly 3 arguments: target, source, and offset tuple
  CHECK(args.size() == 3) << "tensor.assemble requires exactly 3 arguments (target, source, offset), but got "
                          << args.size();

  // First argument (target) must be tensor-shaped. ``AsTensorTypeLike`` accepts
  // both ``TensorType`` and ``DistributedTensorType``; the latter is preserved
  // (with its ``window_buffer_``) on the result so that assembling into a
  // window-buffer view keeps the comm-group binding for downstream passes.
  auto target_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(target_type)
      << "tensor.assemble requires first argument to be a TensorType or DistributedTensorType, but got "
      << args[0]->GetType()->TypeName();

  // Second argument (source) may also be a DistributedTensorType view.
  auto source_type = AsTensorTypeLike(args[1]->GetType());
  CHECK(source_type)
      << "tensor.assemble requires second argument to be a TensorType or DistributedTensorType, but got "
      << args[1]->GetType()->TypeName();

  // Third argument must be TupleType (offset)
  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tensor.assemble requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  // Validate all offset elements are ScalarType with integer dtype
  for (size_t i = 0; i < offset_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(offset_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.assemble offset tuple element " << i << " must be ScalarType, but got "
                       << offset_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.assemble offset tuple element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  // Optional atomic-add combine mode (split-K accumulation into a GM tensor).
  // Absent = AtomicType::kNone (plain overwrite).
  int atomic = static_cast<int>(AtomicType::kNone);
  for (const auto& [key, value] : kwargs) {
    if (key == "atomic") {
      atomic = AnyCast<int>(value, "kwarg key: atomic");
      break;
    }
  }
  CHECK(atomic == static_cast<int>(AtomicType::kNone) || atomic == static_cast<int>(AtomicType::kAdd))
      << "tensor.assemble atomic kwarg must be AtomicType.None_ or AtomicType.Add, but got int " << atomic;
  if (atomic == static_cast<int>(AtomicType::kAdd)) {
    const DataType& dt = target_type->dtype_;
    CHECK(dt == DataType::FP32 || dt == DataType::FP16 || dt == DataType::INT32 || dt == DataType::INT16 ||
          dt == DataType::INT8)
        << "tensor.assemble with atomic=AtomicType.Add requires an fp32/fp16/int32/int16/int8 target "
           "(hardware atomic-add dtypes), but got "
        << dt.ToString();
  }

  // Assemble returns a new tensor type with the same shape and dtype as target.
  // When the target is a DistributedTensorType, the result preserves that kind
  // along with its window_buffer_ — the assembled result is still a view into
  // the same comm-group allocation. A fresh shared_ptr avoids type aliasing.
  if (auto dt = As<DistributedTensorType>(args[0]->GetType())) {
    return std::make_shared<DistributedTensorType>(target_type->shape_, target_type->dtype_, std::nullopt,
                                                   std::nullopt, dt->window_buffer_);
  }
  return std::make_shared<TensorType>(target_type->shape_, target_type->dtype_);
}

TypePtr DeduceTensorFullType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "tensor.full requires exactly 2 arguments (shape, value), but got "
                          << args.size();

  // Extract dtype from kwargs
  bool found_dtype = false;
  DataType dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "dtype") {
      dtype = AnyCast<DataType>(value, "kwarg key: dtype");
      found_dtype = true;
      break;
    }
  }
  CHECK(found_dtype) << "tensor.full requires 'dtype' kwarg";

  // First argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[0]->GetType());
  CHECK(shape_tuple_type) << "tensor.full requires shape to be TupleType, but got "
                          << args[0]->GetType()->TypeName();

  // Validate all shape elements are ScalarType with integer dtype
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.full shape element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.full shape element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  // Second argument must be ConstInt or ConstFloat
  CHECK(As<ConstInt>(args[1]) || As<ConstFloat>(args[1]))
      << "tensor.full requires value to be ConstInt or ConstFloat, but got " << args[1]->TypeName();

  // Extract shape dimensions (same pattern as tensor.create)
  std::vector<ExprPtr> shape;
  shape.reserve(shape_tuple_type->types_.size());

  if (auto make_tuple = As<MakeTuple>(args[0])) {
    shape = make_tuple->elements_;
  } else {
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      shape.emplace_back(std::make_shared<TupleGetItemExpr>(args[0], static_cast<int>(i), args[0]->span_));
    }
  }

  return std::make_shared<TensorType>(shape, dtype);
}

// ============================================================================
// Registration Function for Tensor Memory Operations
// ============================================================================

REGISTER_OP("tensor.read")
    .set_op_category("TensorOp")
    .set_description("Read a scalar value from a tensor at given indices")
    .add_argument("tensor", "Input tensor (TensorType)")
    .add_argument("indices", "Index dimensions (TupleType of ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorReadType(args, kwargs);
    });

REGISTER_OP("tensor.create")
    .set_op_category("TensorOp")
    .set_description("Create a new tensor with specified shape and dtype")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .set_attr<DataType>("dtype")
    .set_attr<TensorLayout>("layout")
    .set_attr<bool>("manual_dep")
    .set_attr<double>("init_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCreateType(args, kwargs);
    });

REGISTER_OP("tensor.slice")
    .set_op_category("TensorOp")
    .set_description("Create a slice (view) of a tensor with new shape and offset")
    .add_argument("input", "Input tensor (TensorType)")
    .add_argument("shape", "New shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(INT64))")
    .add_argument("valid_shape", "Optional logical valid shape; an empty tuple means none")
    .add_argument("drop_dims", "Optional axes (MakeTuple of ConstInt) erased from the result type")
    .set_output_memory_inherit_input()
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorSliceType(args, kwargs);
    });

REGISTER_OP("tensor.assemble")
    .set_op_category("TensorOp")
    .set_description("Write/update tensor values at specified offset")
    .add_argument("target", "Target tensor (TensorType)")
    .add_argument("source", "Source tensor to write (TensorType)")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(INT64))")
    .set_attr<int>("atomic")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorAssembleType(args, kwargs);
    });

REGISTER_OP("tensor.fillpad")
    .set_op_category("TensorOp")
    .set_description("Fill invalid tensor view elements with a specified padding value")
    .add_argument("tensor", "Input tensor (TensorType)")
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorFillpadType(args, kwargs);
    });

REGISTER_OP("tensor.fillpad_expand")
    .set_op_category("TensorOp")
    .set_description("Copy a smaller source tensor into a larger destination tensor, padding the remainder")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("shape", "Destination shape (Tuple of ConstInt), each dim >= source dim")
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorFillpadExpandType(args, kwargs);
    });

REGISTER_OP("tensor.full")
    .set_op_category("TensorOp")
    .set_description("Create a tensor of specified shape filled with a constant value")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("value", "Filling value (ConstInt or ConstFloat)")
    .set_attr<DataType>("dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorFullType(args, kwargs);
    });

TypePtr DeduceTensorCiType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.ci signature: (start, shape) with attrs {dtype, descending}
  CHECK(args.size() == 2) << "tensor.ci requires exactly 2 arguments (start, shape), but got " << args.size();

  bool found_dtype = false;
  DataType dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "dtype") {
      dtype = AnyCast<DataType>(value, "kwarg key: dtype");
      found_dtype = true;
      break;
    }
  }
  CHECK(found_dtype) << "tensor.ci requires 'dtype' kwarg";
  CHECK(dtype == DataType::INT16 || dtype == DataType::INT32 || dtype == DataType::UINT16 ||
        dtype == DataType::UINT32)
      << "tensor.ci dtype must be one of {INT16, INT32, UINT16, UINT32}, but got " << dtype.ToString();

  // First arg: start scalar; dtype must match destination dtype.
  auto start_scalar_type = As<ScalarType>(args[0]->GetType());
  CHECK(start_scalar_type) << "tensor.ci requires first argument 'start' to be a scalar, but got "
                           << args[0]->GetType()->TypeName();
  CHECK(start_scalar_type->dtype_ == dtype)
      << "tensor.ci 'start' dtype (" << start_scalar_type->dtype_.ToString()
      << ") must match destination dtype (" << dtype.ToString() << ")";

  // Second arg: shape TupleType.
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tensor.ci requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.ci shape element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.ci shape element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  std::vector<ExprPtr> shape;
  shape.reserve(shape_tuple_type->types_.size());
  if (auto make_tuple = As<MakeTuple>(args[1])) {
    shape = make_tuple->elements_;
  } else {
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      shape.emplace_back(std::make_shared<TupleGetItemExpr>(args[1], static_cast<int>(i), args[1]->span_));
    }
  }
  CHECK(!shape.empty()) << "tensor.ci requires non-empty shape";

  // ISA constraint: innermost dim Cols != 1.
  if (auto last_const = As<ConstInt>(shape.back())) {
    CHECK(last_const->value_ != 1) << "tensor.ci requires the innermost dimension (Cols) to be != 1, got "
                                   << last_const->value_;
  }

  // ISA constraint: pto.tci only populates the first row. Reject multi-row compile-time
  // shapes so tensor.ci metadata stays consistent with the tile.ci lowering.
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    if (auto const_dim = As<ConstInt>(shape[i])) {
      CHECK(const_dim->value_ == 1)
          << "tensor.ci only populates the first row because pto.tci ignores valid rows; "
          << "leading dimensions must be 1, but got " << const_dim->value_ << " at index " << i;
    }
  }

  (void)kwargs;  // descending is optional bool kwarg, no validation needed beyond type.
  return std::make_shared<TensorType>(shape, dtype);
}

TypePtr DeduceTensorRandomType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.random signature: (key0, key1, counter0, counter1, counter2, counter3, shape)
  // with attrs {dtype, rounds}. Lowers to tile.random. Generates a tensor of
  // counter-based (Philox/ChaCha) pseudo-random values seeded by the key + 128-bit
  // counter scalars; there is no source tensor.
  CHECK(args.size() == 7) << "tensor.random requires exactly 7 arguments (key0, key1, counter0, "
                             "counter1, counter2, counter3, shape), but got "
                          << args.size();

  bool found_dtype = false;
  DataType dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "dtype") {
      dtype = AnyCast<DataType>(value, "kwarg key: dtype");
      found_dtype = true;
      break;
    }
  }
  CHECK(found_dtype) << "tensor.random requires 'dtype' kwarg";
  CHECK(dtype == DataType::INT32 || dtype == DataType::UINT32)
      << "tensor.random dtype must be one of {INT32, UINT32}, but got " << dtype.ToString();

  // The 6 seed arguments are 32-bit integer scalars (key[0..1], counter[0..3]).
  for (size_t i = 0; i < 6; ++i) {
    auto scalar_type = As<ScalarType>(args[i]->GetType());
    CHECK(scalar_type) << "tensor.random requires argument " << i << " (seed scalar) to be a scalar, but got "
                       << args[i]->GetType()->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT32)
        << "tensor.random requires seed argument " << i << " to have INT32 dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  // Shape: TupleType of integer scalars (mirrors tensor.ci).
  auto shape_tuple_type = As<TupleType>(args[6]->GetType());
  CHECK(shape_tuple_type) << "tensor.random requires shape to be TupleType, but got "
                          << args[6]->GetType()->TypeName();
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.random shape element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.random shape element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  std::vector<ExprPtr> shape;
  shape.reserve(shape_tuple_type->types_.size());
  if (auto make_tuple = As<MakeTuple>(args[6])) {
    shape = make_tuple->elements_;
  } else {
    for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
      shape.emplace_back(std::make_shared<TupleGetItemExpr>(args[6], static_cast<int>(i), args[6]->span_));
    }
  }
  CHECK(!shape.empty()) << "tensor.random requires non-empty shape";

  return std::make_shared<TensorType>(shape, dtype);
}

REGISTER_OP("tensor.ci")
    .set_op_category("TensorOp")
    .set_description("Generate a contiguous integer sequence into a tensor (lowers to tile.ci)")
    .add_argument("start", "Starting integer scalar (must match dst dtype)")
    .add_argument("shape", "Destination shape (TupleType of ScalarType integer)")
    .set_attr<DataType>("dtype")
    .set_attr<bool>("descending")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCiType(args, kwargs);
    });

REGISTER_OP("tensor.random")
    .set_op_category("TensorOp")
    .set_description("Generate counter-based pseudo-random values into a tensor (lowers to tile.random)")
    .add_argument("key0", "First key word (INT32 scalar)")
    .add_argument("key1", "Second key word (INT32 scalar)")
    .add_argument("counter0", "Counter word 0 (INT32 scalar)")
    .add_argument("counter1", "Counter word 1 (INT32 scalar)")
    .add_argument("counter2", "Counter word 2 (INT32 scalar)")
    .add_argument("counter3", "Counter word 3 (INT32 scalar)")
    .add_argument("shape", "Destination shape (TupleType of ScalarType integer)")
    .set_attr<DataType>("dtype")
    .set_attr<int>("rounds")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRandomType(args, kwargs);
    });

TypePtr DeduceTensorDimType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.dim: Extract a shape dimension from a tensor as a scalar
  // Args: (tensor, axis)
  // Returns: ScalarType(INT64)
  CHECK(args.size() == 2) << "tensor.dim requires exactly 2 arguments (tensor, axis), but got "
                          << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.dim requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  auto axis_const = As<ConstInt>(args[1]);
  CHECK(axis_const) << "tensor.dim requires axis to be a constant integer";

  int64_t axis = axis_const->value_;
  int64_t rank = static_cast<int64_t>(tensor_type->shape_.size());

  // Support negative indexing
  if (axis < 0) axis += rank;
  CHECK(axis >= 0 && axis < rank) << "tensor.dim axis " << axis_const->value_
                                  << " out of range for tensor of rank " << rank;

  return std::make_shared<ScalarType>(DataType(DataType::INDEX));
}

REGISTER_OP("tensor.dim")
    .set_op_category("TensorOp")
    .set_description("Extract a shape dimension from a tensor as a scalar value")
    .add_argument("tensor", "Input tensor (TensorType)")
    .add_argument("axis", "Dimension index (ConstInt, supports negative indexing)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorDimType(args, kwargs);
    });

TypePtr DeduceTensorWriteType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.write: Write a scalar value into a tensor at given indices
  // Args: (tensor, indices_tuple, value)
  // Returns: TensorType (the destination tensor, for chaining)
  CHECK(args.size() == 3) << "tensor.write requires exactly 3 arguments (tensor, indices, value), but got "
                          << args.size();

  // First argument must be a tensor-shaped value — see tensor.read above
  // for the rationale behind ``AsTensorTypeLike`` over a strict
  // ``As<TensorType>`` cast.
  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type) << "tensor.write requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  auto indices_type = As<TupleType>(args[1]->GetType());
  CHECK(indices_type) << "tensor.write requires indices to be TupleType, but got "
                      << args[1]->GetType()->TypeName();

  CHECK(indices_type->types_.size() == tensor_type->shape_.size())
      << "tensor.write indices count (" << indices_type->types_.size() << ") must match tensor rank ("
      << tensor_type->shape_.size() << ")";

  for (size_t i = 0; i < indices_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(indices_type->types_[i]);
    CHECK(scalar_type) << "tensor.write index element " << i << " must be ScalarType, but got "
                       << indices_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tensor.write index element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  auto value_type = As<ScalarType>(args[2]->GetType());
  CHECK(value_type) << "tensor.write requires third argument (value) to be a ScalarType, but got "
                    << args[2]->GetType()->TypeName();

  CHECK(value_type->dtype_ == tensor_type->dtype_)
      << "tensor.write requires value dtype to match tensor dtype, but got value dtype "
      << value_type->dtype_.ToString() << " and tensor dtype " << tensor_type->dtype_.ToString();

  // tensor.write returns the tensor (for chaining)
  return args[0]->GetType();
}

REGISTER_OP("tensor.write")
    .set_op_category("TensorOp")
    .set_description("Write a scalar value into a tensor at given indices")
    .add_argument("tensor", "Destination tensor (TensorType)")
    .add_argument("indices", "Index dimensions (TupleType of ScalarType)")
    .add_argument("value", "Value to write (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorWriteType(args, kwargs);
    });

REGISTER_OP("tensor.alloc")
    .set_op_category("TensorOp")
    .set_description("Declare DDR memory allocation, returning a Ptr")
    .add_argument("memory_space", "Memory space (DDR)")
    .add_argument("size", "Size in bytes (scalar)")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2) << "tensor.alloc expects 2 args (memory_space, size), got " << args.size();
      return GetPtrType();
    });

REGISTER_OP("tensor.get_block_idx")
    .set_op_category("TensorOp")
    .set_description("Get the current block index (tensor-scope alias of tile.get_block_idx)")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 0) << "tensor.get_block_idx requires no arguments, but got " << args.size();
      return std::make_shared<ScalarType>(DataType::INDEX);
    });

REGISTER_OP("tensor.get_subblock_idx")
    .set_op_category("TensorOp")
    .set_description(
        "Get the current sub-block (vector core) index (tensor-scope alias of tile.get_subblock_idx)")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 0) << "tensor.get_subblock_idx requires no arguments, but got " << args.size();
      return std::make_shared<ScalarType>(DataType::INDEX);
    });

REGISTER_OP("tensor.get_block_num")
    .set_op_category("TensorOp")
    .set_description(
        "Get the total number of blocks in the current SPMD task (tensor-scope alias of tile.get_block_num)")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 0) << "tensor.get_block_num requires no arguments, but got " << args.size();
      return std::make_shared<ScalarType>(DataType::INDEX);
    });

}  // namespace ir
}  // namespace pypto
