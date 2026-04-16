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
 * @file sort.cpp
 * @brief Sorting tensor operations (sort32, mrgsort format1/format2)
 *
 * Tensor-level counterparts of the tile-level sort operators in
 * src/ir/op/tile_ops/sort.cpp. Converted to tile ops by ConvertTensorToTileOps
 * via a simple 1:1 name mapping registered in op_conversion_registry.cpp.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

// ============================================================================
// tensor.sort32 — sorts fixed 32-element blocks, output has last dim doubled.
// ============================================================================

TypePtr DeduceTensorSort32Type(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (src, idx), but got "
                          << args.size();

  auto src_type = As<TensorType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires src dtype to be FP16 or FP32, but got "
      << src_type->dtype_.ToString();

  const auto& input_shape = src_type->shape_;
  CHECK(!input_shape.empty()) << "The operator " << op_name << " requires non-empty input shape";
  if (auto const_last_dim = As<ConstInt>(input_shape.back())) {
    CHECK(const_last_dim->value_ > 0 && const_last_dim->value_ % 32 == 0)
        << "The operator " << op_name
        << " requires the last dimension to be a positive multiple of 32, but got " << const_last_dim->value_;
  }

  auto idx_type = As<TensorType>(args[1]->GetType());
  CHECK(idx_type) << "The operator " << op_name << " requires second argument to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(idx_type->dtype_ == DataType::UINT32)
      << "The operator " << op_name << " requires idx dtype to be UINT32, but got "
      << idx_type->dtype_.ToString();
  CHECK(idx_type->shape_.size() == input_shape.size())
      << "The operator " << op_name << " requires idx rank (" << idx_type->shape_.size()
      << ") to match src rank (" << input_shape.size() << ")";
  for (size_t i = 0; i < input_shape.size(); ++i) {
    CHECK(DimensionsEqual(input_shape[i], idx_type->shape_[i]))
        << "The operator " << op_name << " requires idx shape to match src shape at axis " << i;
  }

  std::vector<ExprPtr> output_shape(input_shape.begin(), input_shape.end() - 1);
  auto last_dim = input_shape.back();
  if (auto const_dim = As<ConstInt>(last_dim)) {
    int64_t doubled = const_dim->value_ * 2;
    output_shape.push_back(std::make_shared<ConstInt>(doubled, DataType::INDEX, Span::unknown()));
  } else {
    auto two = std::make_shared<ConstInt>(2, DataType::INDEX, Span::unknown());
    output_shape.push_back(std::make_shared<Mul>(last_dim, two, DataType::INDEX, Span::unknown()));
  }

  return std::make_shared<TensorType>(output_shape, src_type->dtype_);
}

REGISTER_OP("tensor.sort32")
    .set_op_category("TensorOp")
    .set_description("Sort fixed 32-element blocks (tensor-level, maps to tile.sort32)")
    .add_argument("src", "Input value tensor (TensorType, FP16 or FP32)")
    .add_argument("idx", "Input index tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorSort32Type(args, kwargs, "tensor.sort32");
    });

// ============================================================================
// tensor.mrgsort_format2 — 2-4 way merge sort (tensor-level).
//
// Unlike the tile-level op, the tensor-level mrgsort_format2 only takes source
// tiles: the `tmp` scratch and `executed` status tiles are synthesized as
// local Vec tile.create calls during ConvertTensorToTileOps. This keeps tiny
// scratch (e.g. 1×4 INT16 executed) off the GM boundary, where they would
// otherwise hit PTO tile alignment constraints (Cols * sizeof(dtype) % 32 == 0).
// ============================================================================

TypePtr DeduceTensorMrgSortType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  // Arg layout: (src0, ..., srcN-1)
  //   2-way: 2 args, 3-way: 3 args, 4-way: 4 args
  CHECK(args.size() >= 2 && args.size() <= 4)
      << "The operator " << op_name << " requires 2-4 arguments (2-4 srcs), but got " << args.size();

  auto src0_type = As<TensorType>(args[0]->GetType());
  CHECK(src0_type) << "The operator " << op_name << " requires argument 0 to be a TensorType, but got "
                   << args[0]->GetType()->TypeName();
  CHECK(src0_type->dtype_ == DataType::FP16 || src0_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires src dtype to be FP16 or FP32, but got "
      << src0_type->dtype_.ToString();
  CHECK(!src0_type->shape_.empty()) << "The operator " << op_name << " requires non-empty src shape";

  // All srcs must share dtype and rank.
  std::vector<std::shared_ptr<const TensorType>> src_types;
  src_types.push_back(src0_type);
  for (size_t i = 1; i < args.size(); ++i) {
    auto src_type = As<TensorType>(args[i]->GetType());
    CHECK(src_type) << "The operator " << op_name << " requires argument " << i
                    << " to be a TensorType, but got " << args[i]->GetType()->TypeName();
    CHECK(src_type->dtype_ == src0_type->dtype_)
        << "The operator " << op_name << " requires all src tensors to have matching dtype, but argument "
        << i << " has " << src_type->dtype_.ToString() << " (expected " << src0_type->dtype_.ToString()
        << ")";
    CHECK(src_type->shape_.size() == src0_type->shape_.size())
        << "The operator " << op_name << " requires all src tensors to have matching rank";
    src_types.push_back(src_type);
  }

  // Output shape: same as src0 except the last dim is the sum of all srcs' last dims
  // (matches the `tmp` shape required at tile-level: merged destination buffer).
  std::vector<ExprPtr> out_shape(src0_type->shape_.begin(), src0_type->shape_.end() - 1);
  ExprPtr last_dim;
  int64_t const_sum = 0;
  bool all_const = true;
  for (const auto& st : src_types) {
    auto c = As<ConstInt>(st->shape_.back());
    if (!c) {
      all_const = false;
      break;
    }
    const_sum += c->value_;
  }
  if (all_const) {
    last_dim = std::make_shared<ConstInt>(const_sum, DataType::INDEX, Span::unknown());
  } else {
    last_dim = src_types[0]->shape_.back();
    for (size_t i = 1; i < src_types.size(); ++i) {
      last_dim = std::make_shared<Add>(last_dim, src_types[i]->shape_.back(), DataType::INDEX,
                                       Span::unknown());
    }
  }
  out_shape.push_back(last_dim);
  return std::make_shared<TensorType>(out_shape, src0_type->dtype_);
}

REGISTER_OP("tensor.mrgsort_format2")
    .set_op_category("TensorOp")
    .set_description("Merge sort 2-4 sorted lists, format2 (tensor-level). "
                     "Args: (src0, src1[, src2[, src3]]). "
                     "The scratch tmp and executed tiles required by tile.mrgsort_format2 are "
                     "synthesized during conversion — users do not pass them at the tensor level.")
    .add_argument("src0", "First sorted input tensor (FP16 or FP32)")
    .add_argument("src1", "Second sorted input tensor")
    .add_argument("src2", "(3/4-way only) Third sorted input tensor")
    .add_argument("src3", "(4-way only) Fourth sorted input tensor")
    .set_attr<bool>("exhausted")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMrgSortType(args, kwargs, "tensor.mrgsort_format2");
    });

// ============================================================================
// tensor.mrgsort_format1 — single-list merge sort (tensor-level).
// ============================================================================

TypePtr DeduceTensorMrgSort1Type(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires 2 arguments (src, block_len), but got "
                          << args.size();

  auto src_type = As<TensorType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires argument 0 to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires src dtype to be FP16 or FP32, but got "
      << src_type->dtype_.ToString();

  auto block_len_type = As<ScalarType>(args[1]->GetType());
  CHECK(block_len_type) << "The operator " << op_name
                        << " requires argument 1 (block_len) to be a ScalarType, but got "
                        << args[1]->GetType()->TypeName();
  CHECK(block_len_type->dtype_.IsInt())
      << "The operator " << op_name << " requires block_len to be an integer type, but got "
      << block_len_type->dtype_.ToString();

  if (auto const_val = As<ConstInt>(args[1])) {
    CHECK(const_val->value_ > 0 && const_val->value_ % 64 == 0)
        << "The operator " << op_name << " requires block_len to be a positive multiple of 64, but got "
        << const_val->value_;
  }

  return std::make_shared<TensorType>(src_type->shape_, src_type->dtype_);
}

REGISTER_OP("tensor.mrgsort_format1")
    .set_op_category("TensorOp")
    .set_description("Single-list merge sort, format1 (tensor-level, maps to tile.mrgsort_format1)")
    .add_argument("src", "Input tensor containing pre-sorted runs (FP16 or FP32)")
    .add_argument("block_len", "Run length for merge sort (integer scalar, multiple of 64)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorMrgSort1Type(args, kwargs, "tensor.mrgsort_format1");
    });

}  // namespace ir
}  // namespace pypto
