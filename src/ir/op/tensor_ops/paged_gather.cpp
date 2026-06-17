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
 * @file paged_gather.cpp
 * @brief Tensor-level paged gather operator (gather directly into L1 / UB).
 *
 * Gathers scattered rows of a paged KV pool (`src`) selected by `indices`,
 * translated through a paged `block_table`, directly into an on-chip buffer
 * (L1 / MemorySpace::Mat by default, or UB / Vec). Lowered by
 * ConvertTensorToTileOps into a fully-scalar per-row loop on the Cube core:
 * each iteration scalar-reads the logical index + page table from GM, computes
 * the physical row in scalar registers, and issues a single GM->L1 DMA load
 * (tile.load with target_memory=Mat). The bulk KV data never touches UB, which
 * eliminates the GM round-trip that the gather_kv -> qk_pv pipeline pays today.
 *
 * Reference: gitcode.com/cann/pypto experimental::GatherInL1 (OpCoreType::AIC).
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
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Read an integer-valued kwarg (required unless a default is given).
int ReadIntAttr(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
                const std::string& op_name, bool required, int default_value) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<int>(v, "kwarg key: " + key);
    }
  }
  CHECK(!required) << "The operator " << op_name << " requires a '" << key << "' keyword argument";
  return default_value;
}

bool ReadBoolAttr(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
                  bool default_value) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<bool>(v, "kwarg key: " + key);
    }
  }
  return default_value;
}

}  // namespace

TypePtr DeduceTensorPagedGatherType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name
                          << " requires 3 arguments (src, indices, block_table), but got " << args.size();

  auto src_type = As<TensorType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::BF16 ||
        src_type->dtype_ == DataType::FP32 || src_type->dtype_ == DataType::INT8)
      << "The operator " << op_name << " requires src dtype in {FP16, BF16, FP32, INT8}, but got "
      << src_type->dtype_.ToString();
  CHECK(src_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 2D src, but got rank " << src_type->shape_.size();

  auto idx_type = As<TensorType>(args[1]->GetType());
  CHECK(idx_type) << "The operator " << op_name << " requires indices to be a TensorType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(idx_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires indices dtype to be INT32, but got "
      << idx_type->dtype_.ToString();
  CHECK(idx_type->shape_.size() == 1 || idx_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 1D or 2D indices, but got rank " << idx_type->shape_.size();
  // Lowering reads only row 0 of a rank-2 operand (indices[0, i]); enforce the
  // [1, n] contract so a leading dim > 1 cannot silently drop data.
  if (idx_type->shape_.size() == 2) {
    auto idx_rows = As<ConstInt>(idx_type->shape_[0]);
    CHECK(idx_rows && idx_rows->value_ == 1)
        << "The operator " << op_name << " requires 2D indices to have shape [1, n], but got first dim "
        << (idx_rows ? std::to_string(idx_rows->value_) : "dynamic");
  }

  auto bt_type = As<TensorType>(args[2]->GetType());
  CHECK(bt_type) << "The operator " << op_name << " requires block_table to be a TensorType, but got "
                 << args[2]->GetType()->TypeName();
  CHECK(bt_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires block_table dtype to be INT32, but got "
      << bt_type->dtype_.ToString();
  CHECK(bt_type->shape_.size() == 1 || bt_type->shape_.size() == 2)
      << "The operator " << op_name << " requires 1D or 2D block_table, but got rank "
      << bt_type->shape_.size();
  // Same [1, n] contract for rank-2 block_table (lowering reads block_table[0, b]).
  if (bt_type->shape_.size() == 2) {
    auto bt_rows = As<ConstInt>(bt_type->shape_[0]);
    CHECK(bt_rows && bt_rows->value_ == 1)
        << "The operator " << op_name << " requires 2D block_table to have shape [1, n], but got first dim "
        << (bt_rows ? std::to_string(bt_rows->value_) : "dynamic");
  }

  const int block_size = ReadIntAttr(kwargs, "block_size", op_name, /*required=*/true, 0);
  CHECK(block_size > 0) << "The operator " << op_name << " requires block_size > 0, but got " << block_size;
  const int size = ReadIntAttr(kwargs, "size", op_name, /*required=*/true, 0);
  CHECK(size > 0) << "The operator " << op_name << " requires size > 0, but got " << size;
  const int col_off = ReadIntAttr(kwargs, "col_off", op_name, /*required=*/false, 0);
  CHECK(col_off >= 0) << "The operator " << op_name << " requires col_off >= 0, but got " << col_off;
  const int max_indices = ReadIntAttr(kwargs, "max_indices", op_name, /*required=*/true, 0);
  CHECK(max_indices > 0) << "The operator " << op_name << " requires max_indices > 0, but got "
                         << max_indices;
  const bool is_trans = ReadBoolAttr(kwargs, "is_trans", false);
  // Surface the is_trans/space constraint as a user-facing error here rather than
  // an INTERNAL_CHECK deep in the lowering pass.
  MemorySpace space = MemorySpace::Mat;
  for (const auto& [k, v] : kwargs) {
    if (k == "space") space = AnyCast<MemorySpace>(v, "kwarg key: space");
  }
  CHECK(!is_trans || space == MemorySpace::Mat)
      << "The operator " << op_name << " requires space=MemorySpace::Mat (L1) when is_trans=true";

  // Static bound check: col_off + size must fit within src columns when const.
  if (auto src_cols = As<ConstInt>(src_type->shape_[1])) {
    CHECK(col_off + size <= src_cols->value_)
        << "The operator " << op_name << " requires col_off + size (" << col_off + size
        << ") <= src columns (" << src_cols->value_ << ")";
  }

  // Output is the static on-chip buffer: [max_indices, size] (or transposed).
  // `max_indices` is the compile-time row upper bound that sizes the L1/UB tile;
  // the runtime number of gathered rows (indices' last dim) only controls how
  // many rows the lowering loop fills. This mirrors CANN's static-L1-buffer model
  // and keeps the deduced shape independent of the (possibly dynamic) row count.
  auto rows_expr = std::make_shared<ConstInt>(max_indices, DataType::INDEX, Span::unknown());
  auto size_expr = std::make_shared<ConstInt>(size, DataType::INDEX, Span::unknown());

  std::vector<ExprPtr> out_shape;
  if (is_trans) {
    out_shape = {size_expr, rows_expr};
  } else {
    out_shape = {rows_expr, size_expr};
  }
  return std::make_shared<TensorType>(out_shape, src_type->dtype_);
}

REGISTER_OP("tensor.paged_gather")
    .set_op_category("TensorOp")
    .set_description(
        "Paged gather directly into an on-chip buffer (L1/Mat by default, or UB/Vec). Selects "
        "scattered rows of a 2D paged KV pool by logical indices translated through a paged "
        "block_table, and lowers (in ConvertTensorToTileOps) to a fully-scalar per-row GM->L1 "
        "load loop on the Cube core -- bulk KV never touches UB, eliminating the GM round-trip.")
    .add_argument("src", "Paged KV pool in GM (TensorType, 2D; FP16/BF16/FP32/INT8)")
    .add_argument("indices", "Logical row indices to gather (TensorType, INT32, 1D or [1, n])")
    .add_argument("block_table", "Page table: logical block -> physical block (TensorType, INT32)")
    .set_attr<int>("block_size")
    .set_attr<int>("size")
    .set_attr<int>("col_off")
    .set_attr<int>("max_indices")
    .set_attr<bool>("is_trans")
    .set_attr<bool>("is_b_matrix")
    .set_attr<MemorySpace>("space")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorPagedGatherType(args, kwargs, "tensor.paged_gather");
    });

}  // namespace ir
}  // namespace pypto
