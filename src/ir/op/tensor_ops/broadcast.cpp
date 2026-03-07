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

TypePtr DeduceTensorRowExpandSameType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs,
                                      const std::string& op_name) {
  CHECK(args.size() == 2) << op_name << " requires exactly 2 arguments, but got " << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  auto row_type = As<TensorType>(args[1]->GetType());
  CHECK(tensor_type) << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();
  CHECK(row_type) << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tensor_type->dtype_, row_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";
  CHECK(tensor_type->shape_.size() >= 2) << op_name << " requires first argument to have at least 2 dims";
  CHECK(row_type->shape_.size() >= 2) << op_name << " requires second argument to have at least 2 dims";

  return std::make_shared<TensorType>(tensor_type->shape_, *result_dtype);
}

TypePtr DeduceTensorColExpandSameType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs,
                                      const std::string& op_name) {
  CHECK(args.size() == 2) << op_name << " requires exactly 2 arguments, but got " << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  auto col_type = As<TensorType>(args[1]->GetType());
  CHECK(tensor_type) << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();
  CHECK(col_type) << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tensor_type->dtype_, col_type->dtype_);
  CHECK(result_dtype) << op_name << " requires compatible data types";
  CHECK(tensor_type->shape_.size() >= 2) << op_name << " requires first argument to have at least 2 dims";
  CHECK(col_type->shape_.size() >= 2) << op_name << " requires second argument to have at least 2 dims";

  return std::make_shared<TensorType>(tensor_type->shape_, *result_dtype);
}

TypePtr DeduceTensorRowExpandType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "tensor.row_expand requires exactly 1 argument, but got " << args.size();
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "tensor.row_expand requires argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();
  CHECK(tensor_type->shape_.size() >= 2) << "tensor.row_expand requires at least 2 dims";
  return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_);
}

REGISTER_OP("tensor.row_expand")
    .set_op_category("TensorOp")
    .set_description("Broadcast first element of each source row across the destination row")
    .add_argument("input", "Input tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandType(args, kwargs);
    });

REGISTER_OP("tensor.row_expand_sub")
    .set_op_category("TensorOp")
    .set_description("Row-wise broadcast subtraction")
    .add_argument("tensor", "Input tensor (TensorType)")
    .add_argument("row_vec", "Row vector tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandSameType(args, kwargs, "tensor.row_expand_sub");
    });

REGISTER_OP("tensor.row_expand_div")
    .set_op_category("TensorOp")
    .set_description("Row-wise broadcast division")
    .add_argument("tensor", "Input tensor (TensorType)")
    .add_argument("row_vec", "Row vector tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandSameType(args, kwargs, "tensor.row_expand_div");
    });

REGISTER_OP("tensor.row_expand_mul")
    .set_op_category("TensorOp")
    .set_description("Row-wise broadcast multiplication")
    .add_argument("tensor", "Input tensor (TensorType)")
    .add_argument("row_vec", "Row vector tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandSameType(args, kwargs, "tensor.row_expand_mul");
    });

REGISTER_OP("tensor.row_expand_add")
    .set_op_category("TensorOp")
    .set_description("Row-wise broadcast addition")
    .add_argument("tensor", "Input tensor (TensorType)")
    .add_argument("row_vec", "Row vector tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorRowExpandSameType(args, kwargs, "tensor.row_expand_add");
    });

REGISTER_OP("tensor.col_expand")
    .set_op_category("TensorOp")
    .set_description("Expand column tensor to match target tensor shape")
    .add_argument("target", "Target tensor (TensorType)")
    .add_argument("col_vec", "Column tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColExpandSameType(args, kwargs, "tensor.col_expand");
    });

REGISTER_OP("tensor.col_expand_mul")
    .set_op_category("TensorOp")
    .set_description("Column-wise broadcast multiplication")
    .add_argument("target", "Target tensor (TensorType)")
    .add_argument("col_vec", "Column tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColExpandSameType(args, kwargs, "tensor.col_expand_mul");
    });

REGISTER_OP("tensor.col_expand_div")
    .set_op_category("TensorOp")
    .set_description("Column-wise broadcast division")
    .add_argument("target", "Target tensor (TensorType)")
    .add_argument("col_vec", "Column tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColExpandSameType(args, kwargs, "tensor.col_expand_div");
    });

REGISTER_OP("tensor.col_expand_sub")
    .set_op_category("TensorOp")
    .set_description("Column-wise broadcast subtraction")
    .add_argument("target", "Target tensor (TensorType)")
    .add_argument("col_vec", "Column tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColExpandSameType(args, kwargs, "tensor.col_expand_sub");
    });

}  // namespace ir
}  // namespace pypto
