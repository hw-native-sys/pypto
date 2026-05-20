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
 * @file scatter.cpp
 * @brief Tensor-level scatter operators.
 *
 * Two forms mirror the gather family (no compare-form scatter):
 *
 * - tensor.scatter      — index form (rank-2 MVP). Returns the post-scatter
 *                         output tensor; lowered to tile.scatter by
 *                         ConvertTensorToTileOps.
 * - tensor.scatter_mask — mask-pattern form. Lowered 1:1 to tile.scatter_mask.
 *
 * Semantics (index form, rank-2):
 *   out = input
 *   for i in [0, src.shape[0]):
 *     out[indexes[i, 0], :] = src[i, :]
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
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

bool IsScatterElementDtype(const DataType& dt) {
  return dt == DataType::FP16 || dt == DataType::FP32 || dt == DataType::BF16 || dt == DataType::INT16 ||
         dt == DataType::INT32 || dt == DataType::INT8;
}

bool IsScatterIndexDtype(const DataType& dt) { return dt == DataType::INT16 || dt == DataType::INT32; }

void CheckTensorScatterDtypeSizing(const DataType& dst_dtype, const DataType& idx_dtype,
                                   const std::string& op_name) {
  const int dst_bytes = static_cast<int>(dst_dtype.GetBit()) / 8;
  const int idx_bytes = static_cast<int>(idx_dtype.GetBit()) / 8;
  const int required = (dst_bytes == 1) ? 2 : dst_bytes;
  CHECK(idx_bytes == required) << "The operator " << op_name << " with input dtype " << dst_dtype.ToString()
                               << " (" << dst_bytes << " bytes) requires index dtype of " << required
                               << " bytes, but got " << idx_dtype.ToString() << " (" << idx_bytes
                               << " bytes)";
}

}  // namespace

// ============================================================================
// tensor.scatter — index form
// ============================================================================

static TypePtr DeduceTensorScatterType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name
                          << " requires 3 arguments (input, index, src), but got " << args.size();

  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "The operator " << op_name << " requires input to be a TensorType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(IsScatterElementDtype(input_type->dtype_))
      << "The operator " << op_name << " requires input dtype in {I8, I16, I32, FP16, FP32, BF16}, but got "
      << input_type->dtype_.ToString();

  auto index_type = As<TensorType>(args[1]->GetType());
  CHECK(index_type) << "The operator " << op_name << " requires index to be a TensorType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(IsScatterIndexDtype(index_type->dtype_))
      << "The operator " << op_name << " requires index dtype in {INT16, INT32}, but got "
      << index_type->dtype_.ToString();

  auto src_type = As<TensorType>(args[2]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TensorType, but got "
                  << args[2]->GetType()->TypeName();
  CHECK(src_type->dtype_ == input_type->dtype_)
      << "The operator " << op_name << " requires src dtype (" << src_type->dtype_.ToString()
      << ") to match input dtype (" << input_type->dtype_.ToString() << ")";

  CheckTensorScatterDtypeSizing(input_type->dtype_, index_type->dtype_, op_name);

  const int64_t rank = static_cast<int64_t>(input_type->shape_.size());
  CHECK(rank == 2) << "The operator " << op_name << " currently supports rank-2 input only, got rank "
                   << rank;
  CHECK(static_cast<int64_t>(src_type->shape_.size()) == rank)
      << "The operator " << op_name << " requires src rank (" << src_type->shape_.size()
      << ") to match input rank (" << rank << ")";
  CHECK(index_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D index, but got rank " << index_type->shape_.size();

  // The `dim` kwarg controls along which axis the per-row indices address.
  // MVP only supports dim=0 (or dim=-2) — per-row row-scatter, which is what
  // pto.tscatter implements.
  int dim_val = 0;
  bool dim_seen = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "dim") {
      dim_val = AnyCast<int>(value, "kwarg key: dim");
      dim_seen = true;
      break;
    }
  }
  CHECK(dim_seen) << "The operator " << op_name << " requires a 'dim' keyword argument";
  const int norm_dim = dim_val < 0 ? dim_val + static_cast<int>(rank) : dim_val;
  CHECK(norm_dim == 0) << "The operator " << op_name
                       << " currently supports dim=0 (or dim=-2) only, got dim=" << dim_val;

  // Column count must match between src and input (whole rows scattered).
  auto src_cols = As<ConstInt>(src_type->shape_[1]);
  auto inp_cols = As<ConstInt>(input_type->shape_[1]);
  if (src_cols && inp_cols) {
    CHECK(src_cols->value_ == inp_cols->value_)
        << "The operator " << op_name << " requires src.shape[1] == input.shape[1], got src cols "
        << src_cols->value_ << " vs input cols " << inp_cols->value_;
  }

  // index.shape[0] must equal src.shape[0] (one row-index per src row).
  auto src_rows = As<ConstInt>(src_type->shape_[0]);
  auto idx_rows = As<ConstInt>(index_type->shape_[0]);
  if (src_rows && idx_rows) {
    CHECK(src_rows->value_ == idx_rows->value_)
        << "The operator " << op_name << " requires index.shape[0] == src.shape[0], got src rows "
        << src_rows->value_ << " vs index rows " << idx_rows->value_;
  }

  // Output shape/dtype mirror input (whole-tensor scatter, in-place semantics).
  return std::make_shared<TensorType>(input_type->shape_, input_type->dtype_);
}

REGISTER_OP("tensor.scatter")
    .set_op_category("TensorOp")
    .set_description(
        "Scatter rows from src into input at per-row indices along `dim` "
        "(tensor-level; MVP supports rank-2 with dim=0/-2). Lowered to "
        "tile.scatter by ConvertTensorToTileOps.")
    .add_argument("input", "Base tensor whose rows are propagated (TensorType, 2D)")
    .add_argument("index", "Per-row destination index tensor (TensorType, INT16 or INT32, 2D)")
    .add_argument("src", "Source tensor with rows to scatter (same dtype as input)")
    .set_attr<int>("dim")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorScatterType(args, kwargs, "tensor.scatter");
    });

// ============================================================================
// tensor.scatter_mask — mask-pattern form (lowered 1:1 to tile.scatter_mask).
// ============================================================================

static TypePtr DeduceTensorScatterMaskType(const std::vector<ExprPtr>& args,
                                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                                           const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (input, dst), but got "
                          << args.size();

  auto input_type = As<TensorType>(args[0]->GetType());
  CHECK(input_type) << "The operator " << op_name << " requires input to be a TensorType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(IsScatterElementDtype(input_type->dtype_))
      << "The operator " << op_name << " requires input dtype in {I8, I16, I32, FP16, FP32, BF16}, but got "
      << input_type->dtype_.ToString();

  auto dst_type = As<TensorType>(args[1]->GetType());
  CHECK(dst_type) << "The operator " << op_name << " requires dst to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(IsScatterElementDtype(dst_type->dtype_))
      << "The operator " << op_name << " requires dst dtype in {I8, I16, I32, FP16, FP32, BF16}, but got "
      << dst_type->dtype_.ToString();
  CHECK(input_type->dtype_ == dst_type->dtype_)
      << "The operator " << op_name << " requires input and dst to have the same dtype, got "
      << input_type->dtype_.ToString() << " vs " << dst_type->dtype_.ToString();

  CHECK(input_type->shape_.size() == 2 && dst_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D input/dst, but got input rank "
      << input_type->shape_.size() << " and dst rank " << dst_type->shape_.size();

  int pattern = -1;
  for (const auto& [key, value] : kwargs) {
    if (key == "mask_pattern") {
      pattern = AnyCast<int>(value, "kwarg key: mask_pattern");
      break;
    }
  }
  CHECK(pattern >= 1 && pattern <= 7)
      << "The operator " << op_name << " requires mask_pattern in [1, 7], but got " << pattern;

  // Column expansion: dst.cols == input.cols * stride (or equal for P1111).
  auto inp_cols_const = As<ConstInt>(input_type->shape_[1]);
  auto dst_cols_const = As<ConstInt>(dst_type->shape_[1]);
  if (inp_cols_const && dst_cols_const) {
    const int64_t stride = (pattern == 7) ? 1 : ((pattern <= 2) ? 2 : 4);
    CHECK(dst_cols_const->value_ == inp_cols_const->value_ * stride)
        << "The operator " << op_name << " with mask_pattern=" << pattern << " requires dst.shape[1] ("
        << dst_cols_const->value_ << ") == input.shape[1] (" << inp_cols_const->value_ << ") * " << stride;
  }

  return std::make_shared<TensorType>(dst_type->shape_, dst_type->dtype_);
}

REGISTER_OP("tensor.scatter_mask")
    .set_op_category("TensorOp")
    .set_description(
        "Scatter rows of input into mask-marked columns of dst (tensor-level, "
        "maps 1:1 to tile.scatter_mask). Each row of the 2D input is expanded "
        "into a dst row by writing values onto the columns selected by "
        "mask_pattern. Targeted at A3 / CPU-sim style backends.")
    .add_argument("input", "Source tensor with compact rows (TensorType, 2D)")
    .add_argument("dst", "Destination tensor (rewritten on positions selected by mask_pattern)")
    .set_attr<int>("mask_pattern")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorScatterMaskType(args, kwargs, "tensor.scatter_mask");
    });

}  // namespace ir
}  // namespace pypto
