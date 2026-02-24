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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// Helper function to calculate tensor size expression
static std::string CalculateTensorSizeExpr(const TensorTypePtr& tensor_type, CodegenBase& codegen) {
  std::ostringstream oss;

  // Calculate total number of elements by multiplying all dimensions
  bool first = true;
  for (const auto& dim : tensor_type->shape_) {
    if (first) {
      oss << codegen.GenerateExprString(dim);
      first = false;
    } else {
      oss << " * " << codegen.GenerateExprString(dim);
    }
  }

  // If shape is empty, it's a scalar (1 element)
  if (first) {
    oss << "1";
  }

  // Multiply by element size in bytes
  size_t element_bits = tensor_type->dtype_.GetBit();
  size_t element_bytes = (element_bits + 7) / 8;  // Round up to nearest byte
  oss << " * " << element_bytes;

  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_create, ("tensor.create")) {
  // tensor.create -> uint64_t var_shapes[N] = {...}; Tensor var = make_tensor(var_shapes, N, DataType::XX);
  auto result_type = As<TensorType>(op->GetType());
  CHECK(result_type) << "tensor.create must return TensorType";

  std::string result_var = codegen.GetCurrentResultTarget();
  size_t ndim = result_type->shape_.size();

  std::ostringstream oss;
  oss << "uint64_t " << result_var << "_shapes[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    oss << codegen.GenerateExprString(result_type->shape_[i]);
  }
  oss << "};\n";
  oss << "Tensor " << result_var << " = make_tensor(" << result_var << "_shapes, " << ndim << ", "
      << codegen.GetRuntimeDataTypeString(result_type->dtype_) << ");";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_read, ("tensor.read")) {
  // tensor.read(tensor, indices_tuple) -> scalar value
  CHECK(op->args_.size() == 2) << "tensor.read requires 2 arguments";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.read input must be a variable";

  auto input_type = As<TensorType>(op->args_[0]->GetType());
  CHECK(input_type) << "tensor.read input must be TensorType";

  auto result_type = As<ScalarType>(op->GetType());
  CHECK(result_type) << "tensor.read must return ScalarType";
  std::string cpp_type = result_type->dtype_.ToCTypeString();

  std::string result_var = codegen.GetCurrentResultTarget();
  std::string ptr_expr = codegen.GetTensorDataPtr(input_name);

  // Extract indices from MakeTuple
  auto indices_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(indices_tuple) << "tensor.read indices must be MakeTuple";

  // Compute linear index
  const auto& indices = indices_tuple->elements_;
  const auto& shape = input_type->shape_;

  // Build linear index expression
  std::ostringstream idx_oss;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) idx_oss << " + ";
    idx_oss << codegen.GenerateExprString(indices[i]);
    for (size_t j = i + 1; j < shape.size(); ++j) {
      idx_oss << " * " << codegen.GenerateExprString(shape[j]);
    }
  }
  std::string idx_expr = idx_oss.str();

  // Check if the index expression is a simple constant (all digits)
  bool is_simple = !idx_expr.empty() && std::all_of(idx_expr.begin(), idx_expr.end(), ::isdigit);

  std::ostringstream oss;
  if (is_simple) {
    // Inline constant index directly
    oss << cpp_type << " " << result_var << " = static_cast<" << cpp_type << "*>(" << ptr_expr << ")["
        << idx_expr << "];";
  } else {
    // Use intermediate variable for complex index expressions
    oss << "size_t idx_" << result_var << " = " << idx_expr << ";\n";
    oss << cpp_type << " " << result_var << " = static_cast<" << cpp_type << "*>(" << ptr_expr << ")[idx_"
        << result_var << "];";
  }

  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_view, ("tensor.view")) {
  // tensor.view(input, shape_tuple, offset_tuple) -> Tensor var = input.view({shapes}, {offsets});
  CHECK(op->args_.size() == 3) << "tensor.view requires 3 arguments (input, shape, offset)";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.view input must be a variable";

  std::string ext_input_name = codegen.GetExternalTensorName(input_name);
  std::string result_var = codegen.GetCurrentResultTarget();

  // Extract shape elements from MakeTuple
  auto shape_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(shape_tuple) << "tensor.view shape must be MakeTuple";

  // Extract offset elements from MakeTuple
  auto offset_tuple = As<MakeTuple>(op->args_[2]);
  CHECK(offset_tuple) << "tensor.view offset must be MakeTuple";

  std::ostringstream oss;
  oss << "Tensor " << result_var << " = " << ext_input_name << ".view({";
  for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << codegen.GenerateExprString(shape_tuple->elements_[i]);
  }
  oss << "}, {";
  for (size_t i = 0; i < offset_tuple->elements_.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << codegen.GenerateExprString(offset_tuple->elements_[i]);
  }
  oss << "});";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_dim, ("tensor.dim")) {
  // tensor.dim(tensor, axis) -> extract shape dimension as scalar
  // Validation already performed by DeduceTensorDimType during type deduction.
  INTERNAL_CHECK(op->args_.size() == 2) << "Internal error: tensor.dim expected 2 arguments";

  auto tensor_type = As<TensorType>(op->args_[0]->GetType());
  INTERNAL_CHECK(tensor_type) << "Internal error: tensor.dim input must be TensorType";

  auto axis_const = As<ConstInt>(op->args_[1]);
  INTERNAL_CHECK(axis_const) << "Internal error: tensor.dim axis must be ConstInt";

  int64_t axis = axis_const->value_;
  int64_t rank = static_cast<int64_t>(tensor_type->shape_.size());
  if (axis < 0) {
    axis += rank;
  }
  INTERNAL_CHECK(axis >= 0 && axis < rank) << "Internal error: tensor.dim axis out of range";

  std::string result_var = codegen.GetCurrentResultTarget();
  std::string dim_expr = codegen.GenerateExprString(tensor_type->shape_[axis]);

  std::ostringstream oss;
  oss << "int64_t " << result_var << " = " << dim_expr << ";";
  return oss.str();
}

}  // namespace codegen
}  // namespace pypto
