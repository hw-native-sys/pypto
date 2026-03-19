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

/**
 * @brief Compute the product of shape dimensions (for static shapes)
 *
 * @param shape The shape dimensions
 * @return The product if all dimensions are ConstInt, -1 otherwise
 */
int64_t ComputeShapeProduct(const std::vector<ExprPtr>& shape) {
  int64_t product = 1;
  for (const auto& dim : shape) {
    auto const_dim = As<ConstInt>(dim);
    if (!const_dim) {
      return -1;  // Dynamic shape, cannot compute product
    }
    product *= const_dim->value_;
  }
  return product;
}

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

  // Return new TensorType with transposed shape and same dtype
  // If valid_shape is provided as 4th argument, store it in TensorView
  if (args.size() == 4) {
    auto valid_shape_tuple = As<MakeTuple>(args[3]);
    CHECK(valid_shape_tuple) << "tensor.transpose valid_shape (4th argument) must be a MakeTuple";
    TensorView tensor_view({}, TensorLayout::ND, valid_shape_tuple->elements_);
    return std::make_shared<TensorType>(new_shape, tensor_type->dtype_, std::nullopt,
                                        std::make_optional(std::move(tensor_view)));
  }
  return std::make_shared<TensorType>(new_shape, tensor_type->dtype_);
}

// ============================================================================
// Registration Function for Tensor Transform Operations
// ============================================================================

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

}  // namespace ir
}  // namespace pypto
