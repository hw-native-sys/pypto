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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_MEMREF_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_MEMREF_UTILS_H_

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// Re-export FindYieldStmt from transform_utils so existing consumers compile unchanged.
using transform_utils::FindYieldStmt;

inline std::optional<MemRefPtr> GetTypeMemRef(const TypePtr& type) {
  if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(type)) {
    return shaped_type->memref_;
  }
  return std::nullopt;
}

inline TypePtr CloneTypeWithMemRef(const TypePtr& type, const std::optional<MemRefPtr>& memref,
                                   std::optional<MemorySpace> tile_memory_space_override = std::nullopt) {
  if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(type)) {
    return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, memref,
                                        tensor_type->tensor_view_);
  }

  if (auto tile_type = std::dynamic_pointer_cast<const TileType>(type)) {
    auto memory_space =
        tile_memory_space_override.has_value() ? tile_memory_space_override : tile_type->memory_space_;
    return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, memref, tile_type->tile_view_,
                                      memory_space);
  }

  return type;
}

template <typename RemapExprFn>
inline std::vector<ExprPtr> RemapTypeExprVector(const std::vector<ExprPtr>& exprs,
                                                const RemapExprFn& remap_expr, bool& changed) {
  std::vector<ExprPtr> new_exprs;
  new_exprs.reserve(exprs.size());
  for (const auto& expr : exprs) {
    auto new_expr = remap_expr(expr);
    if (new_expr.get() != expr.get()) {
      changed = true;
    }
    new_exprs.push_back(std::move(new_expr));
  }
  return new_exprs;
}

template <typename RemapExprFn>
inline std::optional<TensorView> RemapTensorViewExprs(const std::optional<TensorView>& tensor_view,
                                                      const RemapExprFn& remap_expr, bool& changed) {
  if (!tensor_view.has_value()) {
    return tensor_view;
  }
  bool view_changed = false;
  auto new_stride = RemapTypeExprVector(tensor_view->stride, remap_expr, view_changed);
  auto new_valid_shape = RemapTypeExprVector(tensor_view->valid_shape, remap_expr, view_changed);
  if (!view_changed) {
    return tensor_view;
  }
  changed = true;
  return TensorView(std::move(new_stride), tensor_view->layout, std::move(new_valid_shape), tensor_view->pad);
}

template <typename RemapExprFn>
inline std::optional<TileView> RemapTileViewExprs(const std::optional<TileView>& tile_view,
                                                  const RemapExprFn& remap_expr, bool& changed) {
  if (!tile_view.has_value()) {
    return tile_view;
  }
  bool view_changed = false;
  auto new_valid_shape = RemapTypeExprVector(tile_view->valid_shape, remap_expr, view_changed);
  auto new_stride = RemapTypeExprVector(tile_view->stride, remap_expr, view_changed);
  ExprPtr new_start_offset = tile_view->start_offset;
  if (tile_view->start_offset) {
    new_start_offset = remap_expr(tile_view->start_offset);
    if (new_start_offset.get() != tile_view->start_offset.get()) {
      view_changed = true;
    }
  }
  if (!view_changed) {
    return tile_view;
  }
  changed = true;
  return TileView(std::move(new_valid_shape), std::move(new_stride), std::move(new_start_offset),
                  tile_view->blayout, tile_view->slayout, tile_view->fractal, tile_view->pad);
}

template <typename RemapExprFn>
inline TypePtr CloneTypeWithMemRefAndRemapExprs(
    const TypePtr& type, const std::optional<MemRefPtr>& memref, const RemapExprFn& remap_expr,
    std::optional<MemorySpace> tile_memory_space_override = std::nullopt) {
  const bool memref_changed = GetTypeMemRef(type) != memref;
  bool changed = memref_changed;

  if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(type)) {
    auto new_shape = RemapTypeExprVector(tensor_type->shape_, remap_expr, changed);
    auto new_tensor_view = RemapTensorViewExprs(tensor_type->tensor_view_, remap_expr, changed);
    if (!changed) {
      return type;
    }
    return std::make_shared<TensorType>(std::move(new_shape), tensor_type->dtype_, memref,
                                        std::move(new_tensor_view));
  }

  if (auto tile_type = std::dynamic_pointer_cast<const TileType>(type)) {
    auto memory_space =
        tile_memory_space_override.has_value() ? tile_memory_space_override : tile_type->memory_space_;
    auto new_shape = RemapTypeExprVector(tile_type->shape_, remap_expr, changed);
    auto new_tile_view = RemapTileViewExprs(tile_type->tile_view_, remap_expr, changed);
    if (!changed) {
      return type;
    }
    return std::make_shared<TileType>(std::move(new_shape), tile_type->dtype_, memref,
                                      std::move(new_tile_view), memory_space);
  }

  return type;
}

inline std::shared_ptr<const TileType> GetTileTypeWithMemRef(const TypePtr& type) {
  auto tile_type = std::dynamic_pointer_cast<const TileType>(type);
  if (!tile_type || !tile_type->memref_.has_value()) {
    return nullptr;
  }
  return tile_type;
}

inline MemRefPtr GetDefinedMemRef(const std::shared_ptr<const TileType>& tile_type) {
  CHECK(tile_type != nullptr) << "TileType must not be null";
  CHECK(tile_type->memref_.has_value()) << "TileType must carry MemRef";
  return *tile_type->memref_;
}

inline bool TryRegisterUniqueMemRef(const MemRefPtr& memref, MemorySpace memory_space,
                                    std::map<const MemRef*, MemorySpace>& seen_ptrs) {
  CHECK(memref != nullptr) << "MemRef must not be null";
  auto [it, inserted] = seen_ptrs.emplace(memref.get(), memory_space);
  CHECK(inserted || it->second == memory_space)
      << "Conflicting TileType.memory_space values found for the same MemRef";
  return inserted;
}

// ============================================================================
// Base Ptr name construction and parsing
// ============================================================================

/// Build a base Ptr variable name from memory space and counter: "mem_vec_7"
inline std::string BuildBasePtrName(MemorySpace space, uint64_t id) {
  std::string space_str = MemorySpaceToString(space);
  std::transform(space_str.begin(), space_str.end(), space_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return "mem_" + space_str + "_" + std::to_string(id);
}

/// Build a base Ptr variable name from counter only: "mem_7"
inline std::string BuildBasePtrName(uint64_t id) { return "mem_" + std::to_string(id); }

/// Extract the trailing numeric counter from a base Ptr name (e.g., "mem_vec_7" → 7).
/// Returns std::nullopt if the name has no trailing numeric suffix.
inline std::optional<uint64_t> ExtractNameCounter(const std::string& name) {
  auto pos = name.find_last_of('_');
  if (pos == std::string::npos || pos + 1 >= name.size()) return std::nullopt;
  const std::string suffix = name.substr(pos + 1);
  if (suffix.empty() ||
      !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
    return std::nullopt;
  }
  return std::stoull(suffix);
}

// ============================================================================
// Alloc statement creation
// ============================================================================

/// Create an alloc AssignStmt for a MemRef's base Ptr variable.
/// DDR → tensor.alloc, on-chip → tile.alloc.
/// Emits: base_ptr: Ptr = {tile,tensor}.alloc(memory_space, size)
inline StmtPtr CreateAllocStatement(const MemRefPtr& memref, MemorySpace memory_space) {
  std::string op_name = (memory_space == MemorySpace::DDR) ? "tensor.alloc" : "tile.alloc";
  auto alloc_op = std::make_shared<Op>(op_name);

  auto memspace_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(memory_space), DataType::INDEX, Span::unknown());
  auto size_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(memref->size_), DataType::INDEX, Span::unknown());

  std::vector<ExprPtr> alloc_args = {memspace_expr, size_expr};
  auto alloc_call = std::make_shared<Call>(alloc_op, alloc_args, GetPtrType(), Span::unknown());

  return std::make_shared<AssignStmt>(memref->base_, alloc_call, Span::unknown());
}

// ============================================================================
// Byte offset computation helpers
// ============================================================================

/// Create a ConstInt(0) expression for byte offset initialization.
inline ExprPtr MakeZeroByteOffset() {
  return std::make_shared<ConstInt>(0, DataType::INDEX, Span::unknown());
}

/// Create an addition expression: lhs + rhs.
/// Folds ConstInt + ConstInt into a single ConstInt.
inline ExprPtr AddByteOffsets(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto const_lhs = As<ConstInt>(lhs);
  auto const_rhs = As<ConstInt>(rhs);
  if (const_lhs && const_rhs) {
    return std::make_shared<ConstInt>(const_lhs->value_ + const_rhs->value_, DataType::INDEX,
                                      Span::unknown());
  }
  if (const_rhs && const_rhs->value_ == 0) return lhs;
  if (const_lhs && const_lhs->value_ == 0) return rhs;
  return std::make_shared<Add>(lhs, rhs, DataType::INDEX, Span::unknown());
}

/// Create a multiply expression: lhs * rhs.
/// Folds ConstInt * ConstInt into a single ConstInt.
inline ExprPtr MulByteOffsets(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto const_lhs = As<ConstInt>(lhs);
  auto const_rhs = As<ConstInt>(rhs);
  if (const_lhs && const_rhs) {
    return std::make_shared<ConstInt>(const_lhs->value_ * const_rhs->value_, DataType::INDEX,
                                      Span::unknown());
  }
  if (const_rhs && const_rhs->value_ == 1) return lhs;
  if (const_lhs && const_lhs->value_ == 1) return rhs;
  return std::make_shared<Mul>(lhs, rhs, DataType::INDEX, Span::unknown());
}

/// Compute byte offset for a slice operation.
/// byte_offset = (o0 * s1 * ... * sN + o1 * s2 * ... * sN + ... + oN) * elem_bytes
inline ExprPtr ComputeSliceByteOffset(const std::vector<ExprPtr>& offsets,
                                      const std::vector<ExprPtr>& parent_shape, uint64_t elem_bytes) {
  INTERNAL_CHECK(offsets.size() == parent_shape.size())
      << "Internal error: slice offset rank (" << offsets.size() << ") must match parent shape rank ("
      << parent_shape.size() << ")";

  ExprPtr result = MakeZeroByteOffset();

  for (size_t i = 0; i < offsets.size(); ++i) {
    ExprPtr stride = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
    for (size_t j = i + 1; j < parent_shape.size(); ++j) {
      stride = MulByteOffsets(stride, parent_shape[j]);
    }
    result = AddByteOffsets(result, MulByteOffsets(offsets[i], stride));
  }

  auto elem_size_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(elem_bytes), DataType::INDEX, Span::unknown());
  return MulByteOffsets(result, elem_size_expr);
}

/// Compute additional byte offset for a view operation.
/// Dispatches: slice ops → stride-based offset, others → zero offset.
inline ExprPtr ComputeViewByteOffset(const CallPtr& call, const TypePtr& parent_type) {
  const std::string& op_name = call->op_->name_;

  if (op_name == "tensor.slice" || op_name == "tile.slice") {
    auto shaped = std::dynamic_pointer_cast<const ShapedType>(parent_type);
    INTERNAL_CHECK_SPAN(shaped, call->span_) << "Internal error: slice parent must be ShapedType";

    // tensor.slice(input, shape, offset) → offset is args[2]
    // tile.slice(input, shape, offset[, valid_shape]) → offset is args[2]
    size_t offset_arg_idx = 2;
    INTERNAL_CHECK_SPAN(offset_arg_idx < call->args_.size(), call->span_)
        << "Internal error: " << op_name << " missing offset argument";

    // Extract individual offset elements from the MakeTuple expression
    std::vector<ExprPtr> offsets;
    if (auto make_tuple = As<MakeTuple>(call->args_[offset_arg_idx])) {
      offsets = make_tuple->elements_;
    } else {
      offsets.push_back(call->args_[offset_arg_idx]);
    }

    uint64_t elem_bytes = (shaped->dtype_.GetBit() + 7) / 8;
    return ComputeSliceByteOffset(offsets, shaped->shape_, elem_bytes);
  }

  // Non-slice view ops (reshape, transpose, extract):
  // No additional byte offset — same memory region, different interpretation
  return MakeZeroByteOffset();
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_MEMREF_UTILS_H_
