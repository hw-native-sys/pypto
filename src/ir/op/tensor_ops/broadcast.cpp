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
 * @file broadcast.cpp
 * @brief Broadcast tensor operations
 *
 * This file implements broadcast operations for tensors that perform
 * element-wise operations with row or column vector broadcasting,
 * single-argument row/col expansion, and scalar expansion.
 */

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorRowExpandType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument must be TensorType (the main tensor)
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TensorType (the row vector)
  auto row_type = As<TensorType>(args[1]->GetType());
  CHECK(row_type) << "The operator " << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();

  // Get shapes
  const auto& tensor_shape = tensor_type->shape_;
  const auto& row_shape = row_type->shape_;

  // Both must have at least 2 dimensions
  CHECK(tensor_shape.size() >= 2) << "The operator " << op_name
                                  << " requires first argument to have at least 2 dimensions, but got "
                                  << tensor_shape.size() << " dimensions";
  CHECK(row_shape.size() >= 2) << "The operator " << op_name
                               << " requires second argument to have at least 2 dimensions, but got "
                               << row_shape.size() << " dimensions";

  // Last dimension of row vector must be 1
  auto row_col_const = As<ConstInt>(row_shape[row_shape.size() - 1]);
  CHECK(row_col_const && row_col_const->value_ == 1)
      << "The operator " << op_name << " requires second argument's last dimension to be 1, but got "
      << row_shape[row_shape.size() - 1];

  // Second-to-last dimension (rows) must match
  auto tensor_rows_const = As<ConstInt>(tensor_shape[tensor_shape.size() - 2]);
  auto row_rows_const = As<ConstInt>(row_shape[row_shape.size() - 2]);

  if (tensor_rows_const && row_rows_const) {
    CHECK(tensor_rows_const->value_ == row_rows_const->value_)
        << "The operator " << op_name
        << " requires matching row dimensions, but got tensor rows=" << tensor_rows_const->value_
        << " and row_vec rows=" << row_rows_const->value_;
  }

  // Promote data types
  auto result_dtype = PromoteDataTypes(tensor_type->dtype_, row_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tensor_type->dtype_.ToString() << " and " << row_type->dtype_.ToString();

  // Output has the same shape as the main tensor
  return std::make_shared<TensorType>(tensor_shape, *result_dtype);
}

TypePtr DeduceTensorColExpandType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument is the target tensor (shape to match)
  auto target_type = As<TensorType>(args[0]->GetType());
  CHECK(target_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument is the column vector to expand (shape [1, N])
  auto col_type = As<TensorType>(args[1]->GetType());
  CHECK(col_type) << "The operator " << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();

  // Get shapes
  const auto& tensor_shape = target_type->shape_;
  const auto& col_shape = col_type->shape_;

  // Both must have at least 2 dimensions
  CHECK(tensor_shape.size() >= 2) << "The operator " << op_name
                                  << " requires first argument to have at least 2 dimensions, but got "
                                  << tensor_shape.size() << " dimensions";
  CHECK(col_shape.size() >= 2) << "The operator " << op_name
                               << " requires second argument to have at least 2 dimensions, but got "
                               << col_shape.size() << " dimensions";

  // Second-to-last dimension (row dimension) of column vector must be 1
  auto col_row_const = As<ConstInt>(col_shape[col_shape.size() - 2]);
  CHECK(col_row_const && col_row_const->value_ == 1)
      << "The operator " << op_name
      << " requires second argument's second-to-last dimension (row) to be 1, but got "
      << col_shape[col_shape.size() - 2];

  // Last dimension (columns) must match
  auto tensor_cols_const = As<ConstInt>(tensor_shape[tensor_shape.size() - 1]);
  auto col_cols_const = As<ConstInt>(col_shape[col_shape.size() - 1]);

  if (tensor_cols_const && col_cols_const) {
    CHECK(tensor_cols_const->value_ == col_cols_const->value_)
        << "The operator " << op_name
        << " requires matching column dimensions, but got tensor cols=" << tensor_cols_const->value_
        << " and col_vec cols=" << col_cols_const->value_;
  }

  // Promote data types
  auto result_dtype = PromoteDataTypes(target_type->dtype_, col_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << target_type->dtype_.ToString() << " and " << col_type->dtype_.ToString();

  // Output has the same shape as the target tensor
  return std::make_shared<TensorType>(tensor_shape, *result_dtype);
}

TypePtr DeduceTensorExpandScalarType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();

  return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_);
}

TypePtr DeduceTensorExpandCloneType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  CHECK(args.size() == 2) << op_name << " requires 2 arguments (input, target), but got " << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << op_name << "requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TensorType (target tensor)
  auto target_type = As<TensorType>(args[1]->GetType());
  CHECK(target_type) << op_name << "requires second argument to be a TensorType, but got "
                     << args[1]->GetType()->TypeName();

  // Extract new shape dimensions from target tensor
  std::vector<ExprPtr> new_shape = target_type->shape_;

  // Validate broadcast rules between input shape and new shape
  const auto& input_shape = tensor_type->shape_;
  CHECK(input_shape.size() == 3) << op_name << " requires input rank to be 3, but got " << input_shape.size();
  CHECK(input_shape.size() == new_shape.size()) << op_name << " requires input rank (" << input_shape.size()
                                                << ") to match target rank (" << new_shape.size() << ")";

  size_t broadcast_dims = 0;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (DimensionsEqual(input_shape[i], new_shape[i])) {
      continue;
    }

    auto input_const = GetConstantDimension(input_shape[i]);
    CHECK(input_const && *input_const == 1)
        << op_name << " only allows broadcasting from dimension 1, but input shape "
        << FormatShape(input_shape) << " cannot be expanded to " << FormatShape(new_shape) << " at dim " << i
        << " (got " << PythonPrint(input_shape[i]) << " -> " << PythonPrint(new_shape[i]) << ")";

    ++broadcast_dims;
  }

  CHECK(broadcast_dims <= 1) << op_name << " allows broadcasting in at most one dimension, but got "
                             << broadcast_dims << " (input shape " << FormatShape(input_shape)
                             << ", target shape " << FormatShape(new_shape) << ")";

  // Promote data types
  auto result_dtype = PromoteDataTypes(tensor_type->dtype_, target_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tensor_type->dtype_.ToString() << " and " << target_type->dtype_.ToString();

  return std::make_shared<TensorType>(new_shape, *result_dtype);
}

// ============================================================================
// Registration Function for Tensor Broadcast Operations
// ============================================================================

REGISTER_OP("tensor.row_expand_mul")
    .set_op_category("TensorOp")
    .set_description("Row-wise broadcast multiplication: tensor * row_vec (broadcasted)")
    .add_argument("tensor", "Input tensor (TensorType [M, N])")
    .add_argument("row_vec", "Row vector (TensorType [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandType(args, kwargs, "tensor.row_expand_mul");
    });

REGISTER_OP("tensor.row_expand_div")
    .set_op_category("TensorOp")
    .set_description("Row-wise broadcast division: tensor / row_vec (broadcasted)")
    .add_argument("tensor", "Input tensor (TensorType [M, N])")
    .add_argument("row_vec", "Row vector (TensorType [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandType(args, kwargs, "tensor.row_expand_div");
    });

REGISTER_OP("tensor.col_expand_mul")
    .set_op_category("TensorOp")
    .set_description("Column-wise broadcast multiplication: tensor * col_vec (broadcasted)")
    .add_argument("tensor", "Input tensor (TensorType [M, N])")
    .add_argument("col_vec", "Column vector (TensorType [1, N])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColExpandType(args, kwargs, "tensor.col_expand_mul");
    });

REGISTER_OP("tensor.row_expand")
    .set_op_category("TensorOp")
    .set_description("Row-wise expansion: expand row_vec [M,1] to target shape [M,N]")
    .add_argument("target", "Target tensor defining output shape (TensorType [M, N])")
    .add_argument("row_vec", "Row vector (TensorType [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandType(args, kwargs, "tensor.row_expand");
    });

REGISTER_OP("tensor.row_expand_add")
    .set_op_category("TensorOp")
    .set_description("Row-wise broadcast addition: tensor + row_vec (broadcasted)")
    .add_argument("tensor", "Input tensor (TensorType [M, N])")
    .add_argument("row_vec", "Row vector (TensorType [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandType(args, kwargs, "tensor.row_expand_add");
    });

REGISTER_OP("tensor.row_expand_sub")
    .set_op_category("TensorOp")
    .set_description("Row-wise broadcast subtraction: tensor - row_vec (broadcasted)")
    .add_argument("tensor", "Input tensor (TensorType [M, N])")
    .add_argument("row_vec", "Row vector (TensorType [M, 1])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandType(args, kwargs, "tensor.row_expand_sub");
    });

REGISTER_OP("tensor.col_expand")
    .set_op_category("TensorOp")
    .set_description("Column-wise expansion: expand col_vec [1,N] to target shape [M,N]")
    .add_argument("tensor", "Input tensor (TensorType [M, N])")
    .add_argument("col_vec", "Column vector (TensorType [1, N])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColExpandType(args, kwargs, "tensor.col_expand");
    });

REGISTER_OP("tensor.col_expand_sub")
    .set_op_category("TensorOp")
    .set_description("Column-wise broadcast subtraction: tensor - col_vec (broadcasted)")
    .add_argument("tensor", "Input tensor (TensorType [M, N])")
    .add_argument("col_vec", "Column vector (TensorType [1, N])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColExpandType(args, kwargs, "tensor.col_expand_sub");
    });

REGISTER_OP("tensor.col_expand_div")
    .set_op_category("TensorOp")
    .set_description("Column-wise broadcast division: tensor / col_vec (broadcasted)")
    .add_argument("tensor", "Input tensor (TensorType [M, N])")
    .add_argument("col_vec", "Column vector (TensorType [1, N])")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColExpandType(args, kwargs, "tensor.col_expand_div");
    });

REGISTER_OP("tensor.expands")
    .set_op_category("TensorOp")
    .set_description("Expand scalar to tensor shape")
    .add_argument("target", "Target tensor defining output shape (TensorType)")
    .add_argument("scalar", "Scalar to expand (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorExpandScalarType(args, kwargs, "tensor.expands");
    });

REGISTER_OP("tensor.expand_clone")
    .set_op_category("TensorOp")
    .set_description(
        "Expand tensor by cloning data (not broadcasting). All dimensions must be 1 or match target shape.")
    .add_argument("input", "Input tensor to expand (TensorType)")
    .add_argument("target", "Target tensor defining output shape (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorExpandCloneType(args, kwargs, "tensor.expand_clone");
    });

}  // namespace ir
}  // namespace pypto
