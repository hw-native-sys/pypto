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
 * @brief Memory tensor operations (create, view, assemble)
 *
 * This file implements memory operations for tensors including allocation,
 * view creation, and value assembly/updates.
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

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

  // Validate all shape elements are ScalarType(INT64 or UINT64)
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.create shape tuple element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT64 || scalar_type->dtype_ == DataType::UINT64)
        << "tensor.create shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
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

  return std::make_shared<TensorType>(shape, dtype);
}

TypePtr DeduceTensorViewType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.view requires exactly 3 arguments: input tensor, shape tuple, and offset tuple
  CHECK(args.size() == 3) << "tensor.view requires exactly 3 arguments (input, shape, offset), but got "
                          << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.view requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (shape)
  auto shape_tuple_type = As<TupleType>(args[1]->GetType());
  CHECK(shape_tuple_type) << "tensor.view requires shape to be TupleType, but got "
                          << args[1]->GetType()->TypeName();

  // Validate all shape elements are ScalarType(INT64 or UINT64)
  for (size_t i = 0; i < shape_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(shape_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.view shape tuple element " << i << " must be ScalarType, but got "
                       << shape_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT64 || scalar_type->dtype_ == DataType::UINT64)
        << "tensor.view shape tuple element " << i << " must have dtype INT64 or UINT64, but got "
        << scalar_type->dtype_.ToString();
  }

  // Third argument must be TupleType (offset)
  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tensor.view requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  // Validate all offset elements are ScalarType(INT64 or UINT64)
  for (size_t i = 0; i < offset_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(offset_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.view offset tuple element " << i << " must be ScalarType, but got "
                       << offset_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT64 || scalar_type->dtype_ == DataType::UINT64)
        << "tensor.view offset tuple element " << i << " must have dtype INT64 or UINT64, but got "
        << scalar_type->dtype_.ToString();
  }

  // Extract shape dimensions
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

  // View preserves dtype but has new shape (which can have different rank than input)
  return std::make_shared<TensorType>(new_shape, tensor_type->dtype_);
}

TypePtr DeduceTensorAssembleType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.assemble requires exactly 3 arguments: target, source, and offset tuple
  CHECK(args.size() == 3) << "tensor.assemble requires exactly 3 arguments (target, source, offset), but got "
                          << args.size();

  // First argument (target) must be TensorType
  auto target_type = As<TensorType>(args[0]->GetType());
  CHECK(target_type) << "tensor.assemble requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument (source) must be TensorType
  auto source_type = As<TensorType>(args[1]->GetType());
  CHECK(source_type) << "tensor.assemble requires second argument to be a TensorType, but got "
                     << args[1]->GetType()->TypeName();

  // Third argument must be TupleType (offset)
  auto offset_tuple_type = As<TupleType>(args[2]->GetType());
  CHECK(offset_tuple_type) << "tensor.assemble requires offset to be TupleType, but got "
                           << args[2]->GetType()->TypeName();

  // Validate all offset elements are ScalarType(INT64 or UINT64)
  for (size_t i = 0; i < offset_tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(offset_tuple_type->types_[i]);
    CHECK(scalar_type) << "tensor.assemble offset tuple element " << i << " must be ScalarType, but got "
                       << offset_tuple_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT64 || scalar_type->dtype_ == DataType::UINT64)
        << "tensor.assemble offset tuple element " << i << " must have dtype INT64 or UINT64, but got "
        << scalar_type->dtype_.ToString();
  }

  // Assemble returns a new TensorType with the same shape and dtype as target
  // We need to create a new type object to avoid sharing type instances
  return std::make_shared<TensorType>(target_type->shape_, target_type->dtype_);
}

// ============================================================================
// Registration Function for Tensor Memory Operations
// ============================================================================

REGISTER_OP("tensor.create")
    .set_op_category("TensorOp")
    .set_description("Create a new tensor with specified shape and dtype")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(UINT64))")
    .set_attr<DataType>("dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCreateType(args, kwargs);
    });

REGISTER_OP("tensor.view")
    .set_op_category("TensorOp")
    .set_description("Create a view/slice of a tensor with new shape and offset")
    .add_argument("input", "Input tensor (TensorType)")
    .add_argument("shape", "New shape dimensions (TupleType of ScalarType(UINT64))")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(UINT64))")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorViewType(args, kwargs);
    });

REGISTER_OP("tensor.assemble")
    .set_op_category("TensorOp")
    .set_description("Write/update tensor values at specified offset")
    .add_argument("target", "Target tensor (TensorType)")
    .add_argument("source", "Source tensor to write (TensorType)")
    .add_argument("offset", "Offset dimensions (TupleType of ScalarType(UINT64))")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorAssembleType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
