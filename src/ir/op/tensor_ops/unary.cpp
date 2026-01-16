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
 * @file unary.cpp
 * @brief Unary tensor operations (exp, cast)
 *
 * This file implements unary operations for tensors that operate element-wise.
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

TypePtr DeduceTensorExpType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  auto tensor_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // exp should promote to float type if input is integer
  // Exponential always produces floating-point output (e.g., exp(1) = 2.718...)
  DataType out_dtype = tensor_type->dtype_;
  if (!out_dtype.IsFloat()) {
    // Promote to default float type (FP32)
    out_dtype = DataType::FP32;
  }

  return std::make_shared<TensorType>(out_dtype, tensor_type->shape_);
}

TypePtr DeduceTensorCastType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  // tensor.cast requires 2 arguments: (input, targetType)
  // Optional third argument: mode (string for rounding mode)
  CHECK(args.size() >= 2) << "The operator " << op_name << " requires at least 2 arguments, but got "
                          << args.size();

  auto tensor_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Extract target dtype from second argument
  auto target_dtype_const = std::dynamic_pointer_cast<const ConstInt>(args[1]);
  CHECK(target_dtype_const) << "The operator " << op_name
                            << " requires second argument to be a ConstInt representing target DataType";

  DataType target_dtype = static_cast<DataType>(target_dtype_const->value_);

  // Cast preserves shape but changes dtype
  return std::make_shared<TensorType>(target_dtype, tensor_type->shape_);
}

// ============================================================================
// Registration Function for Tensor Unary Operations
// ============================================================================

REGISTER_OP("tensor.exp")
    .set_op_category("TensorOp")
    .set_description("Element-wise exponential operation")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) { return DeduceTensorExpType(args, "tensor.exp"); });

REGISTER_OP("tensor.cast")
    .set_op_category("TensorOp")
    .set_description("Type casting operation")
    .add_argument("input", "Input tensor (TensorType)")
    .add_argument("targetType", "Target data type (ConstInt)")
    .add_argument("mode", "Rounding mode (optional string)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTensorCastType(args, "tensor.cast");
    });

}  // namespace ir
}  // namespace pypto
