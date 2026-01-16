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
 * @file reduction.cpp
 * @brief Reduction tensor operations (row_max, row_sum)
 *
 * This file implements reduction operations for tensors that reduce along
 * specified axes.
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorReductionType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  // Reduction operations require at least 1 argument (input tensor)
  // Optional args: axis, keepDim, kind
  CHECK(args.size() >= 1) << "The operator " << op_name << " requires at least 1 argument, but got "
                          << args.size();

  // First argument must be TensorType
  auto tensor_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  const auto& input_shape = tensor_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());

  // Extract axis (default: -1, meaning last axis)
  int axis = -1;
  if (args.size() >= 2) {
    auto axis_const = std::dynamic_pointer_cast<const ConstInt>(args[1]);
    if (axis_const) {
      axis = axis_const->value_;
    }
  }

  // Normalize negative axis
  if (axis < 0) {
    axis = static_cast<int>(input_ndim) + axis;
  }
  CHECK(axis >= 0 && static_cast<int64_t>(axis) < input_ndim)
      << "The operator " << op_name << " axis " << axis << " is out of range for shape with " << input_ndim
      << " dimensions";

  // Extract keepDim flag (default: true)
  bool keep_dim = true;
  if (args.size() >= 3) {
    auto keep_dim_const = std::dynamic_pointer_cast<const ConstInt>(args[2]);
    if (keep_dim_const) {
      keep_dim = keep_dim_const->value_ != 0;
    }
  }

  // Build output shape
  std::vector<ExprPtr> output_shape;
  for (int64_t i = 0; i < input_ndim; ++i) {
    if (i == axis) {
      if (keep_dim) {
        // Keep dimension as 1
        output_shape.push_back(std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown()));
      }
      // Otherwise, skip this dimension (reduce it out)
    } else {
      output_shape.push_back(input_shape[i]);
    }
  }

  // If output shape is empty (all dimensions reduced and keepDim=false), return ScalarType
  if (output_shape.empty()) {
    return std::make_shared<ScalarType>(tensor_type->dtype_);
  }

  return std::make_shared<TensorType>(tensor_type->dtype_, output_shape);
}

// ============================================================================
// Registration Function for Tensor Reduction Operations
// ============================================================================

REGISTER_OP("tensor.row_max")
    .set_op_category("TensorOp")
    .set_description("Row-wise maximum reduction along specified axis")
    .add_argument("input", "Input tensor (TensorType)")
    .add_argument("axis", "Reduction axis (optional, ConstInt, default=-1)")
    .add_argument("keepDim", "Keep reduced dimension as 1 (optional, ConstInt bool, default=true)")
    .add_argument("kind", "Reduction kind string (optional)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTensorReductionType(args, "tensor.row_max");
    });

REGISTER_OP("tensor.row_sum")
    .set_op_category("TensorOp")
    .set_description("Row-wise sum reduction along specified axis")
    .add_argument("input", "Input tensor (TensorType)")
    .add_argument("axis", "Reduction axis (optional, ConstInt, default=-1)")
    .add_argument("keepDim", "Keep reduced dimension as 1 (optional, ConstInt bool, default=true)")
    .add_argument("kind", "Reduction kind string (optional)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTensorReductionType(args, "tensor.row_sum");
    });

}  // namespace ir
}  // namespace pypto
