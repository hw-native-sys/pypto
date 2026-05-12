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
 * @file gather.cpp
 * @brief Tensor-level gather operator.
 *
 * Supports rank >= 2 and any dim (including negative). Lowered to a sequence
 * of tile.transpose + tile.reshape + tile.gather by ConvertTensorToTileOps.
 */

#include <any>
#include <cstdint>
#include <memory>
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

namespace pypto {
namespace ir {

TypePtr DeduceTensorGatherType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (input, index), but got "
                          << args.size();

  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "The operator " << op_name << " requires input to be a TensorType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(input_type->dtype_ == DataType::FP16 || input_type->dtype_ == DataType::FP32 ||
        input_type->dtype_ == DataType::INT16 || input_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires input dtype to be FP16, FP32, INT16, or INT32, but got "
      << input_type->dtype_.ToString();

  auto index_type = As<TensorType>(args[1]->GetType());
  CHECK(index_type) << "The operator " << op_name << " requires index to be a TensorType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(index_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires index dtype to be INT32, but got "
      << index_type->dtype_.ToString();

  const int64_t rank = static_cast<int64_t>(input_type->shape_.size());
  CHECK(rank >= 2) << "The operator " << op_name << " requires rank >= 2, but got rank " << rank;
  CHECK(static_cast<int64_t>(index_type->shape_.size()) == rank)
      << "The operator " << op_name << " requires index rank (" << index_type->shape_.size()
      << ") to match input rank (" << rank << ")";

  int dim_val = -1;
  bool dim_seen = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "dim") {
      dim_val = AnyCast<int>(value, "kwarg key: dim");
      dim_seen = true;
      break;
    }
  }
  CHECK(dim_seen) << "The operator " << op_name << " requires a 'dim' keyword argument";

  // Normalize negative dim.
  int norm_dim = dim_val < 0 ? dim_val + static_cast<int>(rank) : dim_val;
  CHECK(norm_dim >= 0 && norm_dim < static_cast<int>(rank))
      << "The operator " << op_name << " requires dim in [" << -rank << ", " << rank - 1
      << "], but got dim=" << dim_val;

  // For non-gather axes: index.shape[i] <= input.shape[i] (static check when both are ConstInt).
  for (int64_t i = 0; i < rank; ++i) {
    if (i == static_cast<int64_t>(norm_dim)) continue;
    auto idx_const = As<ConstInt>(index_type->shape_[i]);
    auto inp_const = As<ConstInt>(input_type->shape_[i]);
    if (idx_const && inp_const) {
      CHECK(idx_const->value_ <= inp_const->value_)
          << "The operator " << op_name << " requires index.shape[" << i << "] (" << idx_const->value_
          << ") <= input.shape[" << i << "] (" << inp_const->value_ << ") on non-gather axes";
    }
  }

  return std::make_shared<TensorType>(index_type->shape_, input_type->dtype_);
}

REGISTER_OP("tensor.gather")
    .set_op_category("TensorOp")
    .set_description(
        "Gather elements of input along the specified dimension using the index tensor "
        "(tensor-level). Supports rank>=2 and any dim; lowered via tile.transpose + "
        "tile.reshape + tile.gather by ConvertTensorToTileOps.")
    .add_argument("input", "Input tensor (TensorType; FP16, FP32, INT16, or INT32)")
    .add_argument("index", "Index tensor (TensorType, INT32, same shape as output)")
    .set_attr<int>("dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorGatherType(args, kwargs, "tensor.gather");
    });

// ============================================================================
// tensor.gather_mask — mask-pattern element selection (tensor-level).
// 1:1 maps to tile.gather_mask in ConvertTensorToTileOps.
// ============================================================================

TypePtr DeduceTensorGatherMaskType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument (input), but got "
                          << args.size();

  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "The operator " << op_name << " requires input to be a TensorType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(input_type->dtype_ == DataType::FP16 || input_type->dtype_ == DataType::FP32 ||
        input_type->dtype_ == DataType::INT16 || input_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires input dtype to be FP16, FP32, INT16, or INT32, but got "
      << input_type->dtype_.ToString();

  CHECK(input_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D input, but got rank " << input_type->shape_.size();

  int pattern = -1;
  for (const auto& [key, value] : kwargs) {
    if (key == "mask_pattern") {
      pattern = AnyCast<int>(value, "kwarg key: mask_pattern");
      break;
    }
  }
  CHECK(pattern >= 1 && pattern <= 7)
      << "The operator " << op_name << " requires mask_pattern in range [1, 7], but got " << pattern;

  // Output last-dim shrink mirrors tile.gather_mask:
  //   P0101 (1), P1010 (2)               → divisor 2
  //   P0001 (3) .. P1000 (6)             → divisor 4
  //   P1111 (7)                           → no shrink
  const auto& src_shape = input_type->shape_;
  const ExprPtr& col_expr = src_shape[1];
  ExprPtr out_col_expr;
  if (pattern == 7) {
    out_col_expr = col_expr;
  } else {
    int64_t divisor = (pattern <= 2) ? 2 : 4;
    if (auto const_col = As<ConstInt>(col_expr)) {
      CHECK(const_col->value_ % divisor == 0)
          << "The operator " << op_name << " with mask_pattern=" << pattern
          << " requires src columns divisible by " << divisor << ", got " << const_col->value_;
      out_col_expr =
          std::make_shared<ConstInt>(const_col->value_ / divisor, DataType::INDEX, Span::unknown());
    } else {
      auto div_expr = std::make_shared<ConstInt>(divisor, DataType::INDEX, Span::unknown());
      out_col_expr = std::make_shared<FloorDiv>(col_expr, div_expr, DataType::INDEX, Span::unknown());
    }
  }
  std::vector<ExprPtr> out_shape = {src_shape[0], out_col_expr};

  // Optional output_dtype kwarg — same bit width as input dtype.
  bool has_output_dtype = false;
  DataType out_dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "output_dtype") {
      CHECK(value.type() == typeid(DataType) || value.type() == typeid(int))
          << "The operator " << op_name << " requires output_dtype to be DataType or int";
      if (value.type() == typeid(DataType)) {
        out_dtype = AnyCast<DataType>(value, "kwarg key: output_dtype");
      } else {
        out_dtype = static_cast<DataType>(AnyCast<int>(value, "kwarg key: output_dtype"));
      }
      has_output_dtype = true;
      break;
    }
  }
  if (!has_output_dtype) {
    out_dtype = input_type->dtype_;
  } else {
    CHECK(out_dtype.GetBit() == input_type->dtype_.GetBit())
        << "The operator " << op_name << " output_dtype must have the same bit width as input dtype ("
        << input_type->dtype_.ToString() << " = " << input_type->dtype_.GetBit() << " bits), but got "
        << out_dtype.ToString() << " = " << out_dtype.GetBit() << " bits";
  }

  return std::make_shared<TensorType>(out_shape, out_dtype);
}

REGISTER_OP("tensor.gather_mask")
    .set_op_category("TensorOp")
    .set_description(
        "Gather elements of input by mask pattern (tensor-level, maps to tile.gather_mask). "
        "Each row of the 2D input is compacted by selecting columns that the mask marks active.")
    .add_argument("input", "Input tensor (TensorType; FP16, FP32, INT16, or INT32)")
    .set_attr<int>("mask_pattern")
    .set_attr<DataType>("output_dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorGatherMaskType(args, kwargs, "tensor.gather_mask");
    });

// ============================================================================
// tensor.gather_compare — compare-form gather producing two outputs.
// 1:1 maps to tile.gather_compare in ConvertTensorToTileOps.
// ============================================================================
//
// Args : (input, kvalue)
//   input  — TensorType, 2D.
//   kvalue — ScalarType (UINT16 or UINT32; applied to every row of input).
//   ConvertTensorToTileOps materializes the workspace tmp, gathered-values dst,
//   and per-row count cdst tiles for the tile-level DPS form.
// Attrs: cmp_mode (int [0,5]), offset (int), out_cols (int — output column
//        count per row), count_dtype (DataType, optional — INT32 or UINT32,
//        defaults to INT32).
// Output: TupleType{ TensorType_dst, TensorType_cdst }
//   dst  shape  = [rows, out_cols], dtype = INT32 (gathered indices)
//   cdst shape  = [1, rows], dtype = count_dtype
//
// cdst is kept 2-D ([1, rows]) to mirror the tile-level convention (see
// tile_ops/gather.cpp) and to satisfy the PTO tile.store codegen requirement
// that tile valid_shape be 2D and physically contiguous.

static TypePtr DeduceTensorGatherCompareType(const std::vector<ExprPtr>& args,
                                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                                             const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (input, kvalue), but got "
                          << args.size();

  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "The operator " << op_name << " requires input to be a TensorType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(input_type->dtype_ == DataType::FP16 || input_type->dtype_ == DataType::FP32 ||
        input_type->dtype_ == DataType::INT16 || input_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires input dtype in {FP16, FP32, INT16, INT32}, but got "
      << input_type->dtype_.ToString();
  CHECK(input_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D input, but got rank " << input_type->shape_.size();

  auto kv_type = As<ScalarType>(args[1]->GetType());
  CHECK(kv_type) << "The operator " << op_name << " requires kvalue to be a ScalarType, but got "
                 << args[1]->GetType()->TypeName();
  CHECK(kv_type->dtype_ == input_type->dtype_)
      << "The operator " << op_name << " requires kvalue dtype equal to src dtype "
      << input_type->dtype_.ToString() << " , but got " << kv_type->dtype_.ToString();

  int cmp_mode = -1;
  bool cmp_mode_seen = false;
  int out_cols = -1;
  bool out_cols_seen = false;
  DataType count_dtype = DataType::INT32;
  for (const auto& [key, value] : kwargs) {
    if (key == "cmp_mode") {
      cmp_mode = AnyCast<int>(value, "kwarg key: cmp_mode");
      cmp_mode_seen = true;
    } else if (key == "out_cols") {
      out_cols = AnyCast<int>(value, "kwarg key: out_cols");
      out_cols_seen = true;
    } else if (key == "count_dtype") {
      if (value.type() == typeid(DataType)) {
        count_dtype = AnyCast<DataType>(value, "kwarg key: count_dtype");
      } else {
        count_dtype = static_cast<DataType>(AnyCast<int>(value, "kwarg key: count_dtype"));
      }
    }
  }
  CHECK(cmp_mode_seen) << "The operator " << op_name << " requires a 'cmp_mode' keyword argument";
  CHECK(cmp_mode >= 0 && cmp_mode <= 5)
      << "The operator " << op_name << " requires cmp_mode in [0, 5] (eq/ne/lt/le/gt/ge), but got "
      << cmp_mode;
  CHECK(out_cols_seen) << "The operator " << op_name << " requires an 'out_cols' keyword argument";
  CHECK(out_cols > 0) << "The operator " << op_name << " requires out_cols > 0, but got " << out_cols;
  CHECK(count_dtype == DataType::INT32 || count_dtype == DataType::UINT32)
      << "The operator " << op_name << " requires count_dtype to be INT32 or UINT32, but got "
      << count_dtype.ToString();

  const ExprPtr& rows_expr = input_type->shape_[0];
  auto out_cols_expr = std::make_shared<ConstInt>(out_cols, DataType::INDEX, Span::unknown());

  std::vector<ExprPtr> dst_shape = {rows_expr, out_cols_expr};
  auto dst_type = std::make_shared<TensorType>(dst_shape, DataType::INT32);
  // cdst is 2-D [1, rows] — see comment above the op for rationale.
  auto one_expr = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
  std::vector<ExprPtr> cdst_shape = {one_expr, rows_expr};
  auto cdst_type = std::make_shared<TensorType>(cdst_shape, count_dtype);

  std::vector<TypePtr> elements{dst_type, cdst_type};
  return std::make_shared<TupleType>(std::move(elements));
}

REGISTER_OP("tensor.gather_compare")
    .set_op_category("TensorOp")
    .set_description(
        "Compare-form gather (tensor-level, maps to tile.gather_compare). For each row of "
        "the 2D input, scan against the scalar kvalue threshold under the chosen cmp_mode "
        "and write the matching indices to dst and the per-row match count to cdst.")
    .add_argument("input", "Input tensor (TensorType, 2D; FP16/FP32/INT16/INT32)")
    .add_argument("kvalue", "Scalar threshold (ScalarType; dtype must match input; applied per row)")
    .set_attr<int>("cmp_mode")
    .set_attr<int>("offset")
    .set_attr<int>("out_cols")
    .set_attr<DataType>("count_dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorGatherCompareType(args, kwargs, "tensor.gather_compare");
    });

}  // namespace ir
}  // namespace pypto
