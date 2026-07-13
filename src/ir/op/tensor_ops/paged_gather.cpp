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
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

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
  // paged_gather reads scattered src rows via runtime indices/block_table, so a
  // partially-valid operand makes the gathered buffer's valid region underivable —
  // reject rather than widen (North Star). Fully-valid operands pass silently, so the
  // current (bare) result is byte-identical.
  CheckTensorInputFullyValid(src_type, op_name, "src", args[0]->span_);
  CheckTensorInputFullyValid(idx_type, op_name, "indices", args[1]->span_);
  CheckTensorInputFullyValid(bt_type, op_name, "block_table", args[2]->span_);
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

// ============================================================================
// tensor.create_l1 + tensor.gather_row — kernel-driven paged gather into L1.
//
// These two ops let a tensor-level kernel build an on-chip (L1/Mat) accumulator
// row by row, with the kernel itself computing the physical source row per slot
// (block-table lookups, multi-source selection, invalid clamping) — everything
// `tensor.paged_gather` hardcodes. `tensor.create_l1` deduces a TensorType so the
// gathered result composes with the tensor-level `tensor.matmul` / softmax ops;
// the conversion lowers it to `tile.create(target_memory=Mat)`. `tensor.gather_row`
// is DPS (returns its accumulator type) and lowers to the per-row `tile.gather_row`.
// ============================================================================

TypePtr DeduceTensorCreateL1Type(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument (shape), but got "
                          << args.size();
  auto make_tuple = As<MakeTuple>(args[0]);
  CHECK(make_tuple) << "The operator " << op_name
                    << " requires shape to be a MakeTuple of static ConstInt dims, but got "
                    << args[0]->TypeName();
  std::vector<ExprPtr> shape;
  shape.reserve(make_tuple->elements_.size());
  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto c = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(c && c->value_ > 0) << "The operator " << op_name << " shape element " << i
                              << " must be a positive compile-time ConstInt";
    shape.push_back(make_tuple->elements_[i]);
  }
  CHECK(!shape.empty()) << "The operator " << op_name << " requires a non-empty shape";

  DataType dtype = DataType::FP32;
  bool has_dtype = false;
  for (const auto& [k, v] : kwargs) {
    if (k == "dtype") {
      dtype = AnyCast<DataType>(v, "kwarg key: dtype");
      has_dtype = true;
    }
  }
  CHECK(has_dtype) << "The operator " << op_name << " requires a 'dtype' keyword argument";
  return std::make_shared<TensorType>(shape, dtype);
}

TypePtr DeduceTensorGatherRowType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& /*kwargs*/,
                                  const std::string& op_name) {
  CHECK(args.size() == 5) << "The operator " << op_name
                          << " requires 5 arguments (acc, src, dst_offset, src_offset, shapes), but got "
                          << args.size();
  auto acc_type = As<TensorType>(args[0]->GetType());
  CHECK(acc_type) << "The operator " << op_name
                  << " requires acc to be a TensorType (from tensor.create_l1), but got "
                  << args[0]->GetType()->TypeName();
  auto src_type = As<TensorType>(args[1]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TensorType (GM), but got "
                  << args[1]->GetType()->TypeName();
  CHECK(acc_type->dtype_ == src_type->dtype_)
      << "The operator " << op_name << " requires acc and src to share dtype, but got "
      << acc_type->dtype_.ToString() << " and " << src_type->dtype_.ToString();
  // src_offset is runtime-computed (normally from a block-table lookup), so a
  // partial source does not provide enough information to prove that the row
  // window avoids padding. Fail closed instead of letting the DPS result hide
  // an indirect read outside src's valid prefix.
  CheckTensorInputFullyValid(src_type, op_name, "src", args[1]->span_);
  // DPS: result is the accumulator, written in place.
  return args[0]->GetType();
}

REGISTER_OP("tensor.create_l1")
    .set_op_category("TensorOp")
    .set_description(
        "Create an on-chip (L1/Mat) accumulator for kernel-driven paged gather_row. Deduces a "
        "TensorType so the gathered result composes with tensor-level matmul; lowers (in "
        "ConvertTensorToTileOps) to tile.create(target_memory=Mat).")
    .add_argument("shape", "Shape dimensions (TupleType of static ConstInt)")
    .set_attr<DataType>("dtype")
    .set_attr<bool>("transpose")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorCreateL1Type(args, kwargs, "tensor.create_l1");
    });

REGISTER_OP("tensor.gather_row")
    .set_op_category("TensorOp")
    .set_description(
        "Gather one GM row into a sub-region of an on-chip accumulator (DPS). Lowers (in "
        "ConvertTensorToTileOps) to the per-row tile.gather_row (pto.subview + pto.tload, no tmov). "
        "The caller computes the physical src_offset (block-table + bias) and the dst_offset slot.")
    .add_argument("acc", "On-chip accumulator (TensorType, from tensor.create_l1)")
    .add_argument("src", "Source tensor in GM (TensorType)")
    .add_argument("dst_offset", "[row, col] offset within acc (TupleType of ScalarType)")
    .add_argument("src_offset", "[row, col] offset within the GM src (TupleType of ScalarType)")
    .add_argument("shapes", "GM row window shape [r, c] (TupleType of ScalarType)")
    .set_attr<bool>("transpose")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorGatherRowType(args, kwargs, "tensor.gather_row");
    });

}  // namespace ir
}  // namespace pypto
