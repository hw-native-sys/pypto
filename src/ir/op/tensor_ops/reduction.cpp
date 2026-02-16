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

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Helper to get kwargs value with default (uses vector to preserve order)
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const T& default_value = T{}) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  return default_value;
}

TypePtr DeduceTensorReductionType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  // Reduction operations require exactly 1 argument (input tensor)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  const auto& input_shape = tensor_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());

  // Extract axis from kwargs (default: -1, meaning last axis)
  int axis = GetKwarg<int>(kwargs, "axis", -1);

  // Normalize negative axis
  if (axis < 0) {
    axis = static_cast<int>(input_ndim) + axis;
  }
  CHECK(axis >= 0 && static_cast<int64_t>(axis) < input_ndim)
      << "The operator " << op_name << " axis " << axis << " is out of range for shape with " << input_ndim
      << " dimensions";

  // Extract keep_dim flag from kwargs (default: true)
  bool keep_dim = GetKwarg<bool>(kwargs, "keep_dim", true);

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

  // If output shape is empty (all dimensions reduced and keep_dim=false), return ScalarType
  if (output_shape.empty()) {
    return std::make_shared<ScalarType>(tensor_type->dtype_);
  }

  return std::make_shared<TensorType>(output_shape, tensor_type->dtype_);
}

// ============================================================================
// Registration Function for Tensor Reduction Operations
// ============================================================================

REGISTER_OP("tensor.row_max")
    .set_op_category("TensorOp")
    .set_description("Row-wise maximum reduction along specified axis")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorReductionType(args, kwargs, "tensor.row_max");
    });

REGISTER_OP("tensor.row_sum")
    .set_op_category("TensorOp")
    .set_description("Row-wise sum reduction along specified axis")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorReductionType(args, kwargs, "tensor.row_sum");
    });

}  // namespace ir
}  // namespace pypto
