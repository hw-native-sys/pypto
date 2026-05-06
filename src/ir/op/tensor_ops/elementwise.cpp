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
 * @file elementwise.cpp
 * @brief Element-wise tensor operations (Add, Sub, Mul, Div)
 *
 * This file implements element-wise tensor operations that support
 * N-dimensional tensors with NumPy-style broadcasting.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Try TensorType first
  auto tensor_type1 = As<TensorType>(args[0]->GetType());
  auto tensor_type2 = As<TensorType>(args[1]->GetType());

  CHECK(tensor_type1) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                      << args[0]->GetType()->TypeName();
  CHECK(tensor_type2) << "The operator " << op_name
                      << " requires second argument to be a TensorType, but got "
                      << args[1]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, tensor_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  auto broadcast_result = BroadcastShapes(tensor_type1->shape_, tensor_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << FormatShape(tensor_type1->shape_) << " and "
                                  << FormatShape(tensor_type2->shape_);

  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorOpElementwiseScalarType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  auto tensor_type1 = As<TensorType>(args[0]->GetType());
  auto scalar_type2 = As<ScalarType>(args[1]->GetType());

  CHECK(tensor_type1) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                      << args[0]->GetType()->TypeName();
  CHECK(scalar_type2) << "The operator " << op_name
                      << " requires second argument to be a ScalarType, but got "
                      << args[1]->GetType()->TypeName();

  // TensorType + ScalarType - result is TensorType with same shape as first argument
  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, scalar_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  return std::make_shared<TensorType>(tensor_type1->shape_, *result_dtype);
}

// ============================================================================
// Registration Function for Tensor Element-wise Operations
// ============================================================================

REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.add");
    });

REGISTER_OP("tensor.adds")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.adds");
    });

REGISTER_OP("tensor.sub")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtraction of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.sub");
    });

REGISTER_OP("tensor.subs")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtraction of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.subs");
    });

REGISTER_OP("tensor.mul")
    .set_op_category("TensorOp")
    .set_description("Element-wise multiplication of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.mul");
    });

REGISTER_OP("tensor.muls")
    .set_op_category("TensorOp")
    .set_description("Element-wise multiplication of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.muls");
    });

REGISTER_OP("tensor.div")
    .set_op_category("TensorOp")
    .set_description("Element-wise division of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.div");
    });

REGISTER_OP("tensor.divs")
    .set_op_category("TensorOp")
    .set_description("Element-wise division of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.divs");
    });

REGISTER_OP("tensor.maximum")
    .set_op_category("TensorOp")
    .set_description("Element-wise maximum of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.maximum");
    });

REGISTER_OP("tensor.cmp")
    .set_op_category("TensorOp")
    .set_description("Element-wise comparison of tensor and tensor or scalar (returns 0/1 tensor)")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType) or scalar (ScalarType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2) << "The operator tensor.cmp requires exactly 2 arguments, but got "
                              << args.size();
      if (As<TensorType>(args[1]->GetType())) {
        return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.cmp");
      }
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.cmp");
    });

}  // namespace ir
}  // namespace pypto
