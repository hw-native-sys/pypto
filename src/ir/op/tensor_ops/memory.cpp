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
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorCreateType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.create: all args are shape dimensions (Expr)
  // dtype comes from kwargs
  CHECK(args.size() >= 1) << "tensor.create requires at least 1 shape dimension argument";

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

  // All arguments are shape dimensions
  std::vector<ExprPtr> shape;
  shape.reserve(args.size());
  for (const auto& arg : args) {
    shape.emplace_back(arg);
  }

  return std::make_shared<TensorType>(shape, dtype);
}

TypePtr DeduceTensorViewType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.view requires at least 2 arguments: input tensor and shape_ndim
  // Followed by shape dimensions and offset dimensions
  CHECK(args.size() >= 2) << "tensor.view requires at least 2 arguments (input, shape_ndim), but got "
                          << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.view requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument is the number of shape dimensions (ConstInt)
  auto shape_ndim_const = As<ConstInt>(args[1]);
  CHECK(shape_ndim_const)
      << "tensor.view requires second argument to be a ConstInt indicating number of shape "
         "dimensions";

  size_t shape_ndim = static_cast<size_t>(shape_ndim_const->value_);
  CHECK(shape_ndim > 0) << "tensor.view requires at least 1 shape dimension";

  // Check we have enough arguments: input + shape_ndim + shape_dims + offset_dims
  CHECK(args.size() >= 2 + shape_ndim)
      << "tensor.view requires at least " << (2 + shape_ndim) << " arguments for shape_ndim=" << shape_ndim
      << ", but got " << args.size();

  // Extract new shape dimensions (args[2] to args[2 + shape_ndim - 1])
  std::vector<ExprPtr> new_shape;
  new_shape.reserve(shape_ndim);
  for (size_t i = 0; i < shape_ndim; ++i) {
    new_shape.emplace_back(args[2 + i]);
  }

  // The remaining arguments are offset dimensions (not used for type deduction)
  // View preserves dtype but has new shape (which can have different rank than input)
  return std::make_shared<TensorType>(new_shape, tensor_type->dtype_);
}

TypePtr DeduceTensorAssembleType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // tensor.assemble requires at least 2 arguments (target, source)
  // Followed by offset dimensions
  CHECK(args.size() >= 2) << "tensor.assemble requires at least 2 arguments, but got " << args.size();

  // First argument (target) must be TensorType
  auto target_type = As<TensorType>(args[0]->GetType());
  CHECK(target_type) << "tensor.assemble requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument (source) must be TensorType
  auto source_type = As<TensorType>(args[1]->GetType());
  CHECK(source_type) << "tensor.assemble requires second argument to be a TensorType, but got "
                     << args[1]->GetType()->TypeName();

  // Assemble returns the target tensor type (updated in-place semantically)
  return target_type;
}

// ============================================================================
// Registration Function for Tensor Memory Operations
// ============================================================================

REGISTER_OP("tensor.create")
    .set_op_category("TensorOp")
    .set_description("Create a new tensor with specified shape and dtype")
    .add_argument("shape_dims", "Shape dimensions (variable number of Expr)")
    .set_attr<DataType>("dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCreateType(args, kwargs);
    });

REGISTER_OP("tensor.view")
    .set_op_category("TensorOp")
    .set_description("Create a view/slice of a tensor with new shape and offset")
    .add_argument("input", "Input tensor (TensorType)")
    .add_argument("shape_dims", "New shape dimensions (variable number)")
    .add_argument("offset_dims", "Offset dimensions (variable number)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorViewType(args, kwargs);
    });

REGISTER_OP("tensor.assemble")
    .set_op_category("TensorOp")
    .set_description("Write/update tensor values at specified offset")
    .add_argument("target", "Target tensor (TensorType)")
    .add_argument("source", "Source tensor to write (TensorType)")
    .add_argument("offset_dims", "Offset dimensions (variable number)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorAssembleType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
