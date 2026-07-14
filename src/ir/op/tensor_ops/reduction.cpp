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
 * @brief Reduction tensor operations (row_max, row_sum, row_min, col_sum, col_max, col_min)
 *
 * This file implements reduction operations for tensors that reduce along
 * specified axes.
 */

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

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

namespace {

// Build the result type of a reduction over `axis`, given an already-validated input tensor.
//
// The reduction consumes the input's *valid* region on the reduced axis — the backend kernels bound
// their loops by the source's valid extent — so every cell it reads is real data and the reduced
// axis collapses to a fully valid output axis (extent 1 under keep_dim, dropped otherwise).
// Validity on the surviving axes carries over unchanged: reducing one axis does not change which
// rows/columns of the others hold real data.
TypePtr MakeReductionResultType(const std::shared_ptr<const TensorType>& tensor_type, int64_t axis,
                                bool keep_dim, const std::string& op_name, const Span& span,
                                std::optional<DataType> out_dtype) {
  // GetValidShape CHECKs that input_valid has the same rank as the physical shape, so indexing it
  // by physical axis below stays in bounds.
  const auto input_valid = GetValidShape(tensor_type);
  CheckReductionInputNonEmpty(input_valid, op_name, span);

  const auto& input_shape = tensor_type->shape_;
  const auto input_ndim = static_cast<int64_t>(input_shape.size());
  std::vector<ExprPtr> output_shape;
  std::vector<ExprPtr> output_valid;
  output_shape.reserve(input_shape.size());
  output_valid.reserve(input_shape.size());
  for (int64_t i = 0; i < input_ndim; ++i) {
    if (i == axis) {
      if (keep_dim) {
        output_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()));
        output_valid.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown()));
      }
      continue;  // keep_dim == false reduces the axis away entirely
    }
    output_shape.push_back(input_shape[i]);
    output_valid.push_back(input_valid[i]);
  }

  const DataType dtype = out_dtype.value_or(tensor_type->dtype_);
  // Every dimension reduced away (keep_dim=false): a scalar carries no view metadata, as there is
  // no region for a valid shape to describe.
  if (output_shape.empty()) {
    return std::make_shared<ScalarType>(dtype);
  }
  return MakeFreshTensorType(std::move(output_shape), dtype, std::move(output_valid));
}

}  // namespace

// `out_dtype` overrides the output element type — used by argmax/argmin index output (int32).
TypePtr DeduceTensorReductionType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name,
                                  std::optional<DataType> out_dtype = std::nullopt) {
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

  return MakeReductionResultType(tensor_type, axis, keep_dim, op_name, args[0]->span_, out_dtype);
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

REGISTER_OP("tensor.row_min")
    .set_op_category("TensorOp")
    .set_description("Row-wise minimum reduction along specified axis")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorReductionType(args, kwargs, "tensor.row_min");
    });

REGISTER_OP("tensor.row_prod")
    .set_op_category("TensorOp")
    .set_description("Row-wise product reduction along specified axis")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorReductionType(args, kwargs, "tensor.row_prod");
    });

// Type deduction for column reduction operations. Mirrors DeduceTensorReductionType but defaults
// the reduced axis to -2 (the column axis), matching tile.col_sum's [..., 1, N] output shape.
TypePtr DeduceTensorColReductionType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name,
                                     std::optional<DataType> out_dtype = std::nullopt) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  const auto& input_shape = tensor_type->shape_;
  int64_t input_ndim = static_cast<int64_t>(input_shape.size());
  CHECK(input_ndim >= 2) << "The operator " << op_name << " requires at least a 2D tensor, but got "
                         << input_ndim << " dimensions";

  // Column reduction reduces the second-to-last axis by default (the M dim of [..., M, N]).
  int axis = GetKwarg<int>(kwargs, "axis", -2);
  if (axis < 0) {
    axis = static_cast<int>(input_ndim) + axis;
  }
  CHECK(axis >= 0 && static_cast<int64_t>(axis) < input_ndim)
      << "The operator " << op_name << " axis " << axis << " is out of range for shape with " << input_ndim
      << " dimensions";

  bool keep_dim = GetKwarg<bool>(kwargs, "keep_dim", true);

  return MakeReductionResultType(tensor_type, axis, keep_dim, op_name, args[0]->span_, out_dtype);
}

REGISTER_OP("tensor.col_sum")
    .set_op_category("TensorOp")
    .set_description("Column-wise sum reduction (reduces along axis=-2 by default)")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColReductionType(args, kwargs, "tensor.col_sum");
    });

REGISTER_OP("tensor.col_max")
    .set_op_category("TensorOp")
    .set_description("Column-wise max reduction (reduces along axis=-2 by default)")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColReductionType(args, kwargs, "tensor.col_max");
    });

REGISTER_OP("tensor.col_min")
    .set_op_category("TensorOp")
    .set_description("Column-wise min reduction (reduces along axis=-2 by default)")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColReductionType(args, kwargs, "tensor.col_min");
    });

REGISTER_OP("tensor.col_prod")
    .set_op_category("TensorOp")
    .set_description("Column-wise product reduction (reduces along axis=-2 by default)")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColReductionType(args, kwargs, "tensor.col_prod");
    });

// ============================================================================
// Argmax / argmin reductions (index-typed int32 output)
// ============================================================================

REGISTER_OP("tensor.row_argmax")
    .set_op_category("TensorOp")
    .set_description("Row-wise argmax: index of the per-row maximum along specified axis")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorReductionType(args, kwargs, "tensor.row_argmax", DataType(DataType::INT32));
    });

REGISTER_OP("tensor.row_argmin")
    .set_op_category("TensorOp")
    .set_description("Row-wise argmin: index of the per-row minimum along specified axis")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorReductionType(args, kwargs, "tensor.row_argmin", DataType(DataType::INT32));
    });

REGISTER_OP("tensor.col_argmax")
    .set_op_category("TensorOp")
    .set_description("Column-wise argmax: index of the per-column maximum (reduces along axis=-2 by default)")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColReductionType(args, kwargs, "tensor.col_argmax", DataType(DataType::INT32));
    });

REGISTER_OP("tensor.col_argmin")
    .set_op_category("TensorOp")
    .set_description("Column-wise argmin: index of the per-column minimum (reduces along axis=-2 by default)")
    .add_argument("input", "Input tensor (TensorType)")
    .set_attr<int>("axis")
    .set_attr<bool>("keep_dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorColReductionType(args, kwargs, "tensor.col_argmin", DataType(DataType::INT32));
    });

}  // namespace ir
}  // namespace pypto
