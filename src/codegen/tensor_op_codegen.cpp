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
#include <cstring>
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

static std::string EmitAsUint32(const ExprPtr& expr, CodegenBase& codegen) {
  std::string str = codegen.GenerateExprString(expr);
  if (As<ConstInt>(expr)) {
    return str;
  }
  return "static_cast<uint32_t>(" + str + ")";
}

REGISTER_ORCHESTRATION_OP(tensor_create, ("tensor.create")) {
  // tensor.create emits TensorCreateInfo for runtime memory allocation via alloc_tensors().
  // The batched alloc_tensors call and const Tensor& binding are emitted by
  // EmitBatchedAllocTensors at scope entry (SeqStmts).
  auto result_type = As<TensorType>(op->GetType());
  CHECK(result_type) << "tensor.create must return TensorType";

  std::string result_var = codegen.GetCurrentResultTarget();
  size_t ndim = result_type->shape_.size();
  const std::string create_scale_expr = codegen.GetTensorCreateScaleExpr(result_var);

  std::ostringstream oss;
  oss << "uint32_t " << result_var << "_ci_shapes[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    std::string dim_expr = EmitAsUint32(result_type->shape_[i], codegen);
    // Backend-side allocation expansion for injected GM pipe buffers in SPMD:
    // keep IR shape unchanged, scale host allocation at codegen.
    if (!create_scale_expr.empty() && ndim == 1 && i == 0) {
      dim_expr = "static_cast<uint32_t>((" + dim_expr + ") * (" + create_scale_expr + "))";
    }
    oss << dim_expr;
  }
  oss << "};\n";

  std::string dtype_str = codegen.GetRuntimeDataTypeString(result_type->dtype_);
  bool manual_dep = op->GetKwarg<bool>("manual_dep", false);
  oss << "TensorCreateInfo " << result_var << "_ci(" << result_var << "_ci_shapes, " << ndim << ", "
      << dtype_str;
  if (manual_dep) {
    oss << ", /*manual_dep=*/true";
  }
  oss << ");";
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
  // Default to "0" for rank-0 tensors (scalar tensor with empty shape/indices)
  std::string idx_expr = idx_oss.str().empty() ? "0" : idx_oss.str();

  // Check if the index expression is a simple constant (all digits)
  bool is_simple = std::all_of(idx_expr.begin(), idx_expr.end(), ::isdigit);

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

REGISTER_ORCHESTRATION_OP(tensor_write, ("tensor.write")) {
  // tensor.write(tensor, indices_tuple, value) -> write scalar value to tensor at indices
  CHECK(op->args_.size() == 3) << "tensor.write requires 3 arguments";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.write input must be a variable";

  auto input_type = As<TensorType>(op->args_[0]->GetType());
  CHECK(input_type) << "tensor.write input must be TensorType";

  std::string ptr_expr = codegen.GetTensorDataPtr(input_name);

  auto indices_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(indices_tuple) << "tensor.write indices must be MakeTuple";

  std::string value_expr = codegen.GenerateExprString(op->args_[2]);
  auto value_type = As<ScalarType>(op->args_[2]->GetType());
  CHECK(value_type) << "tensor.write value must be ScalarType";
  std::string cpp_type = value_type->dtype_.ToCTypeString();

  // Compute linear index
  const auto& indices = indices_tuple->elements_;
  const auto& shape = input_type->shape_;

  std::ostringstream idx_oss;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) idx_oss << " + ";
    idx_oss << codegen.GenerateExprString(indices[i]);
    for (size_t j = i + 1; j < shape.size(); ++j) {
      idx_oss << " * " << codegen.GenerateExprString(shape[j]);
    }
  }
  // Default to "0" for rank-0 tensors (scalar tensor with empty shape/indices)
  std::string idx_expr = idx_oss.str().empty() ? "0" : idx_oss.str();

  bool is_simple = std::all_of(idx_expr.begin(), idx_expr.end(), ::isdigit);

  std::ostringstream oss;
  if (is_simple) {
    oss << "static_cast<" << cpp_type << "*>(" << ptr_expr << ")[" << idx_expr << "] = " << value_expr << ";";
  } else {
    std::string result_var = codegen.GetCurrentResultTarget();
    oss << "size_t idx_" << result_var << " = " << idx_expr << ";\n";
    oss << "static_cast<" << cpp_type << "*>(" << ptr_expr << ")[idx_" << result_var << "] = " << value_expr
        << ";";
  }

  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_slice, ("tensor.slice")) {
  // tensor.slice(input, shape_tuple, offset_tuple[, valid_shape_tuple]) -> Generate array variables and call
  // .view()
  CHECK(op->args_.size() == 3 || op->args_.size() == 4)
      << "tensor.slice requires 3 or 4 arguments (input, shape, offset[, valid_shape])";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.slice input must be a variable";

  std::string ext_input_name = codegen.GetExternalTensorName(input_name);
  std::string result_var = codegen.GetCurrentResultTarget();

  // Extract shape elements from MakeTuple
  auto shape_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(shape_tuple) << "tensor.slice shape must be MakeTuple";

  // Extract offset elements from MakeTuple
  auto offset_tuple = As<MakeTuple>(op->args_[2]);
  CHECK(offset_tuple) << "tensor.slice offset must be MakeTuple";

  size_t ndim = shape_tuple->elements_.size();
  std::ostringstream oss;

  // Generate shape array
  oss << "uint32_t " << result_var << "_shapes[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    oss << EmitAsUint32(shape_tuple->elements_[i], codegen);
  }
  oss << "};\n";

  // Generate offset array
  oss << "uint32_t " << result_var << "_offsets[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    oss << EmitAsUint32(offset_tuple->elements_[i], codegen);
  }
  oss << "};\n";

  if (op->args_.size() == 4) {
    auto valid_shape_tuple = As<MakeTuple>(op->args_[3]);
    CHECK(valid_shape_tuple) << "tensor.slice valid_shape must be MakeTuple";
    CHECK(valid_shape_tuple->elements_.size() == ndim)
        << "tensor.slice valid_shape must have same rank as shape";
  }

  // Runtime tensor views use shape+offset; valid_shape only affects IR metadata.
  oss << "Tensor " << result_var << " = " << ext_input_name << ".view(" << result_var << "_shapes, "
      << result_var << "_offsets);";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_reshape, ("tensor.reshape")) {
  // tensor.reshape(input, shape_tuple[, valid_shape_tuple]) -> Generate shape array variable and call
  // .reshape() on the runtime Tensor (see runtime/.../tensor.h: Tensor::reshape).
  CHECK(op->args_.size() == 2 || op->args_.size() == 3)
      << "tensor.reshape requires 2 or 3 arguments (input, shape[, valid_shape])";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.reshape input must be a variable";

  std::string ext_input_name = codegen.GetExternalTensorName(input_name);
  std::string result_var = codegen.GetCurrentResultTarget();

  // Extract shape elements from MakeTuple
  auto shape_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(shape_tuple) << "tensor.reshape shape must be MakeTuple";
  size_t ndim = shape_tuple->elements_.size();

  std::ostringstream oss;

  // Generate shape array
  oss << "uint32_t " << result_var << "_shapes[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    oss << EmitAsUint32(shape_tuple->elements_[i], codegen);
  }
  oss << "};\n";

  if (op->args_.size() == 3) {
    auto valid_shape_tuple = As<MakeTuple>(op->args_[2]);
    CHECK(valid_shape_tuple) << "tensor.reshape valid_shape must be MakeTuple";
    CHECK(valid_shape_tuple->elements_.size() == ndim)
        << "tensor.reshape valid_shape must have same rank as shape";
  }

  // Runtime Tensor::reshape requires the source to be contiguous; valid_shape only affects IR metadata.
  oss << "Tensor " << result_var << " = " << ext_input_name << ".reshape(" << result_var << "_shapes, "
      << ndim << ");";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_transpose, ("tensor.transpose")) {
  // tensor.transpose(input, axis1, axis2) -> Tensor view with two axes swapped.
  // Lowered to runtime Tensor::transpose(x, y), a zero-copy metadata swap of
  // shapes / raw_shapes / offsets (see runtime tensor.h: Tensor::transpose).
  // The optional 4th `valid_shape` argument from the IR op is intentionally
  // ignored at the orchestration layer (it only affects IR metadata, mirroring
  // how tensor.reshape handles valid_shape here).
  CHECK(op->args_.size() == 3 || op->args_.size() == 4)
      << "tensor.transpose requires 3 or 4 arguments (input, axis1, axis2[, valid_shape])";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.transpose input must be a variable";

  auto input_type = As<TensorType>(op->args_[0]->GetType());
  CHECK(input_type) << "tensor.transpose input must be TensorType";

  auto axis1_const = As<ConstInt>(op->args_[1]);
  CHECK(axis1_const) << "tensor.transpose requires second argument (axis1) to be a ConstInt";
  auto axis2_const = As<ConstInt>(op->args_[2]);
  CHECK(axis2_const) << "tensor.transpose requires third argument (axis2) to be a ConstInt";

  // Normalize negative axes against the input rank (matches DeduceTensorTransposeType).
  int64_t ndim = static_cast<int64_t>(input_type->shape_.size());
  int64_t axis1 = axis1_const->value_;
  int64_t axis2 = axis2_const->value_;
  if (axis1 < 0) axis1 += ndim;
  if (axis2 < 0) axis2 += ndim;
  CHECK(axis1 >= 0 && axis1 < ndim) << "tensor.transpose axis1 out of range: " << axis1_const->value_
                                    << " for " << ndim << "D tensor";
  CHECK(axis2 >= 0 && axis2 < ndim) << "tensor.transpose axis2 out of range: " << axis2_const->value_
                                    << " for " << ndim << "D tensor";
  CHECK(axis1 != axis2) << "tensor.transpose axis1 and axis2 must be different, got " << axis1;

  // If the optional valid_shape operand is present, validate its structure even though it is
  // intentionally not emitted at the orchestration layer (mirrors tensor.reshape / tensor.slice).
  if (op->args_.size() == 4) {
    auto valid_shape_tuple = As<MakeTuple>(op->args_[3]);
    CHECK(valid_shape_tuple) << "tensor.transpose valid_shape must be MakeTuple";
    CHECK(static_cast<int64_t>(valid_shape_tuple->elements_.size()) == ndim)
        << "tensor.transpose valid_shape must have same rank as input shape";
  }

  std::string ext_input_name = codegen.GetExternalTensorName(input_name);
  std::string result_var = codegen.GetCurrentResultTarget();

  std::ostringstream oss;
  oss << "Tensor " << result_var << " = " << ext_input_name << ".transpose(" << axis1 << ", " << axis2
      << ");";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_dim, ("tensor.dim")) {
  // tensor.dim(tensor, axis) -> extract shape dimension as scalar
  // Validation already performed by DeduceTensorDimType during type deduction.
  INTERNAL_CHECK_SPAN(op->args_.size() == 2, op->span_) << "Internal error: tensor.dim expected 2 arguments";

  auto tensor_type = As<TensorType>(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tensor_type, op->span_) << "Internal error: tensor.dim input must be TensorType";

  auto axis_const = As<ConstInt>(op->args_[1]);
  INTERNAL_CHECK_SPAN(axis_const, op->span_) << "Internal error: tensor.dim axis must be ConstInt";

  int64_t axis = axis_const->value_;
  int64_t rank = static_cast<int64_t>(tensor_type->shape_.size());
  if (axis < 0) {
    axis += rank;
  }
  INTERNAL_CHECK_SPAN(axis >= 0 && axis < rank, op->span_) << "Internal error: tensor.dim axis out of range";

  std::string result_var = codegen.GetCurrentResultTarget();

  // For a compile-time constant dim, emit the literal directly.
  // For a dynamic dim (e.g. pl.dynamic("M")), GenerateExprString returns the
  // dynamic var name (e.g. "M"), which is not a valid C++ identifier in the
  // orchestration scope.  Read the runtime shape from the TaskArg instead.
  std::string dim_expr;
  if (As<ConstInt>(tensor_type->shape_[axis])) {
    dim_expr = codegen.GenerateExprString(tensor_type->shape_[axis]);
  } else {
    std::string tensor_name = codegen.TryGetVarName(op->args_[0]);
    CHECK(!tensor_name.empty()) << "tensor.dim: cannot resolve tensor name from first argument";
    dim_expr = codegen.GetTensorShapeDim(tensor_name, axis);
    CHECK(!dim_expr.empty()) << "tensor.dim: GetTensorShapeDim not supported for '" << tensor_name
                             << "' in this codegen context";
  }

  std::ostringstream oss;
  oss << "int64_t " << result_var << " = " << dim_expr << ";";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_scatter_update, ("tensor.scatter_update")) {
  // tensor.scatter_update(input, index, src):
  //   For each (i, j): input[index[i*s+j]] row = src[i*s+j] row
  // Works for both 2D ([rows, d]) and 4D ([blockNum, blockSize, 1, d]) inputs,
  // since their linear memory layouts are equivalent.
  CHECK(op->args_.size() == 3) << "tensor.scatter_update requires 3 arguments";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.scatter_update: input must be a variable";
  std::string index_name = codegen.TryGetVarName(op->args_[1]);
  CHECK(!index_name.empty()) << "tensor.scatter_update: index must be a variable";
  std::string src_name = codegen.TryGetVarName(op->args_[2]);
  CHECK(!src_name.empty()) << "tensor.scatter_update: src must be a variable";

  auto input_type = As<TensorType>(op->args_[0]->GetType());
  auto index_type = As<TensorType>(op->args_[1]->GetType());
  INTERNAL_CHECK_SPAN(input_type && index_type, op->span_)
      << "Internal error: invalid types for tensor.scatter_update";

  std::string input_ptr = codegen.GetTensorDataPtr(input_name);
  std::string index_ptr = codegen.GetTensorDataPtr(index_name);
  std::string src_ptr = codegen.GetTensorDataPtr(src_name);

  // b = index.shape[0], s = index.shape[1], d = last dimension of input
  std::string b_expr = codegen.GenerateExprString(index_type->shape_[0]);
  std::string s_expr = codegen.GenerateExprString(index_type->shape_[1]);
  std::string d_expr = codegen.GenerateExprString(input_type->shape_.back());

  size_t elem_bytes = (input_type->dtype_.GetBit() + 7) / 8;
  std::string result_var = codegen.GetCurrentResultTarget();

  // Use result_var as suffix to ensure unique local variable names
  std::string s_var = "s_" + result_var;
  std::string d_var = "d_" + result_var;
  std::string i_var = "i_" + result_var;
  std::string j_var = "j_" + result_var;
  std::string idx_var = "idx_" + result_var;
  std::string row_var = "row_" + result_var;

  std::string index_cpp_type = index_type->dtype_.ToCTypeString();

  std::ostringstream oss;
  oss << "size_t " << s_var << " = (size_t)(" << s_expr << ");\n";
  oss << "size_t " << d_var << " = (size_t)(" << d_expr << ");\n";
  oss << "for (size_t " << i_var << " = 0; " << i_var << " < (size_t)(" << b_expr << "); ++" << i_var
      << ") {\n";
  oss << "  for (size_t " << j_var << " = 0; " << j_var << " < " << s_var << "; ++" << j_var << ") {\n";
  oss << "    size_t " << idx_var << " = " << i_var << " * " << s_var << " + " << j_var << ";\n";
  oss << "    " << index_cpp_type << " " << row_var << " = static_cast<const " << index_cpp_type << "*>("
      << index_ptr << ")[" << idx_var << "];\n";
  oss << "    always_assert(" << row_var << " >= 0);\n";
  oss << "    memcpy(static_cast<char*>(" << input_ptr << ") + static_cast<size_t>(" << row_var << ") * "
      << d_var << " * " << elem_bytes << "ULL,\n";
  oss << "           static_cast<const char*>(" << src_ptr << ") + " << idx_var << " * " << d_var << " * "
      << elem_bytes << "ULL,\n";
  oss << "           " << d_var << " * " << elem_bytes << "ULL);\n";
  oss << "  }\n}\n";
  oss << "Tensor " << result_var << " = " << codegen.GetExternalTensorName(input_name) << ";";
  return oss.str();
}

}  // namespace codegen
}  // namespace pypto
