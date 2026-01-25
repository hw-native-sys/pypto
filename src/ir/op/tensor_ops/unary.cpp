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

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
namespace pypto {
namespace ir {

TypePtr DeduceTensorExpType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.exp requires exactly 1 argument, but got " << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.exp requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // exp should promote to float type if input is integer
  // Exponential always produces floating-point output (e.g., exp(1) = 2.718...)
  DataType out_dtype = tensor_type->dtype_;
  if (!out_dtype.IsFloat()) {
    // Promote to default float type (FP32)
    out_dtype = DataType::FP32;
  }

  return std::make_shared<TensorType>(tensor_type->shape_, out_dtype);
}

TypePtr DeduceTensorCastType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.cast requires exactly 1 argument (input), but got " << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.cast requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Read target_type from kwargs
  bool found_target_type = false;
  DataType target_dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "target_type") {
      // Handle both DataType and int for backward compatibility
      if (value.type() == typeid(DataType)) {
        target_dtype = AnyCast<DataType>(value, "kwarg key: target_type");
      } else if (value.type() == typeid(int)) {
        target_dtype = static_cast<DataType>(AnyCast<int>(value, "kwarg key: target_type"));
      } else {
        throw TypeError("target_type must be a DataType or int, but got " + std::string(value.type().name()));
      }
      found_target_type = true;
      break;
    }
  }
  CHECK(found_target_type) << "tensor.cast requires 'target_type' kwarg";

  // mode kwarg is optional, not used in type deduction

  // Cast preserves shape but changes dtype
  return std::make_shared<TensorType>(tensor_type->shape_, target_dtype);
}

// ============================================================================
// Registration Function for Tensor Unary Operations
// ============================================================================

REGISTER_OP("tensor.exp")
    .set_op_category("TensorOp")
    .set_description("Element-wise exponential operation")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorExpType(args, kwargs);
    });

REGISTER_OP("tensor.cast")
    .set_op_category("TensorOp")
    .set_description("Type casting operation")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCastType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
