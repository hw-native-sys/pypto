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
#include <any>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
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
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/storage_size.h"
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
  // DistributedTensorType inherits TensorType: dispatch on it first so the
  // returned clone preserves the subclass identity (including the
  // window_buffer_ back-reference). Without this guard the more generic
  // ``TensorType`` branch below would silently downgrade DistributedTensor
  // params during memref/SSA rebuilds — by codegen time the var's type would
  // become plain TensorType and the cross-rank op codegen would lose the
  // ``DistributedTensorType`` kind discriminator (see N6 plan §codegen).
  if (auto dist_type = std::dynamic_pointer_cast<const DistributedTensorType>(type)) {
    return std::make_shared<DistributedTensorType>(dist_type->shape_, dist_type->dtype_, memref,
                                                   dist_type->tensor_view_, dist_type->window_buffer_);
  }

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
                  tile_view->blayout, tile_view->slayout, tile_view->fractal, tile_view->pad,
                  tile_view->compact);
}

/// Rewrite the SSA values a *pinned* MemRef's slot index names.
///
/// A declared allocation's slot index may be a runtime expression (`l0c[i & 1]`),
/// which makes it the only MemRef field that substitution has to follow: rename
/// `i` and the index must follow, or it dangles on a stale Var.
///
/// Restricted to pinned MemRefs on purpose. `byte_offset_` needs no remap — it is
/// `ConstInt(0)` until InitMemRef and a concrete address after — and confining
/// rebuilds to the pinned window keeps them strictly before every pass that keys
/// on MemRef *pointer* identity (`AllocateMemoryAddr` matches old→new by raw
/// pointer). Rebuilding one of those later would silently split a shared MemRef.
template <typename RemapExprFn>
inline std::optional<MemRefPtr> RemapPinnedMemRefExprs(const std::optional<MemRefPtr>& memref,
                                                       const RemapExprFn& remap_expr, bool& changed) {
  if (!memref.has_value() || !(*memref)->is_pinned_) return memref;
  const auto& slot_index = (*memref)->slot_index_;
  if (!slot_index.has_value() || !*slot_index) return memref;
  auto new_index = remap_expr(*slot_index);
  if (new_index == *slot_index) return memref;
  changed = true;
  const auto& old = *memref;
  return std::make_optional<MemRefPtr>(
      std::make_shared<MemRef>(old->name_hint_, old->base_, old->byte_offset_, old->size_, old->span_,
                               old->is_pinned_, old->slot_count_, std::make_optional(std::move(new_index))));
}

template <typename RemapExprFn>
inline TypePtr CloneTypeWithMemRefAndRemapExprs(
    const TypePtr& type, const std::optional<MemRefPtr>& memref_in, const RemapExprFn& remap_expr,
    std::optional<MemorySpace> tile_memory_space_override = std::nullopt) {
  const bool memref_changed = GetTypeMemRef(type) != memref_in;
  bool changed = memref_changed;
  const auto memref = RemapPinnedMemRefExprs(memref_in, remap_expr, changed);

  // DistributedTensorType clone path: matches the comment on
  // CloneTypeWithMemRef above. Distinct from the TensorType branch so the
  // window_buffer_ back-reference and the kind discriminator survive an
  // InitMemRef / SSA rebuild.
  if (auto dist_type = std::dynamic_pointer_cast<const DistributedTensorType>(type)) {
    auto new_shape = RemapTypeExprVector(dist_type->shape_, remap_expr, changed);
    auto new_tensor_view = RemapTensorViewExprs(dist_type->tensor_view_, remap_expr, changed);
    if (!changed) {
      return type;
    }
    return std::make_shared<DistributedTensorType>(std::move(new_shape), dist_type->dtype_, memref,
                                                   std::move(new_tensor_view), dist_type->window_buffer_);
  }

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

/// Do two MemRefs start at the same address in the same allocation?
///
/// Stronger than `MemRef::SameAllocation`, which compares only the base Ptr: two
/// slots of one `pl.MemRef(slots=N)` share a base at different offsets, so
/// "same allocation" does not mean "same storage". Use this wherever the
/// question is whether a value already sits where it has to end up — between two
/// slots of one allocation a reconciling copy is still required.
///
/// Size is deliberately not compared. A padded loop-carried accumulator carries
/// its buffer under one valid shape and yields it under another, so the two
/// MemRefs describe the same storage at differing extents; requiring equal sizes
/// would ask for a copy from a buffer onto itself, which for `Acc` has no legal
/// lowering at all.
///
/// A non-constant offset is only accepted when both sides carry the *same*
/// offset expression, so an unprovable pair reports "not the same address" and
/// the caller emits the copy rather than silently dropping it.
inline bool SameBaseAddress(const MemRefPtr& a, const MemRefPtr& b) {
  CHECK(a != nullptr && b != nullptr) << "MemRef must not be null";
  if (a->base_.get() != b->base_.get()) return false;
  if (a->byte_offset_.get() == b->byte_offset_.get()) return true;
  auto off_a = As<ConstInt>(a->byte_offset_);
  auto off_b = As<ConstInt>(b->byte_offset_);
  return off_a && off_b && off_a->value_ == off_b->value_;
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
/// `alloc_size` overrides the reserved bytes when the allocation is larger than
/// the MemRef that names it. That happens for a multi-slot declared allocation:
/// each slot MemRef is sized to its own slot (so its byte range stays inside the
/// slot), while the allocation has to cover every slot.
inline StmtPtr CreateAllocStatement(const MemRefPtr& memref, MemorySpace memory_space, bool pinned = false,
                                    std::optional<uint64_t> alloc_size = std::nullopt) {
  std::string op_name = (memory_space == MemorySpace::DDR) ? "tensor.alloc" : "tile.alloc";
  auto alloc_op = std::make_shared<Op>(op_name);

  auto memspace_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(memory_space), DataType::INDEX, Span::unknown());
  auto size_expr = std::make_shared<ConstInt>(static_cast<int64_t>(alloc_size.value_or(memref->size_)),
                                              DataType::INDEX, Span::unknown());

  std::vector<ExprPtr> alloc_args = {memspace_expr, size_expr};
  // Only emit the kwarg when set, so ordinary compiler allocations print and
  // compare exactly as before.
  std::vector<std::pair<std::string, std::any>> alloc_kwargs;
  if (pinned) alloc_kwargs.emplace_back("pinned", true);
  auto alloc_call =
      std::make_shared<Call>(alloc_op, alloc_args, std::move(alloc_kwargs), GetPtrType(), Span::unknown());

  return std::make_shared<AssignStmt>(memref->base_, alloc_call, Span::unknown());
}

/// Prepend alloc statements to a function body's top-level statement list.
///
/// Every consumer of the allocation list scans the body's top-level `SeqStmts`
/// (see `CollectPinnedAllocSizes`), so an allocation created after `InitMemRef`
/// has to land there too rather than beside its first use.
inline StmtPtr InsertAllocsIntoBody(const StmtPtr& body, const std::vector<StmtPtr>& alloc_stmts) {
  if (alloc_stmts.empty()) return body;

  std::vector<StmtPtr> new_seq_stmts;
  new_seq_stmts.insert(new_seq_stmts.end(), alloc_stmts.begin(), alloc_stmts.end());

  const Span& span = body ? body->span_ : alloc_stmts.front()->span_;
  if (body) {
    if (auto seq = As<SeqStmts>(body)) {
      new_seq_stmts.insert(new_seq_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
    } else {
      new_seq_stmts.push_back(body);
    }
  }

  return SeqStmts::Flatten(std::move(new_seq_stmts), span);
}

/// The base Ptr an alloc statement declares when it is a user-owned (pinned)
/// buffer, else null. Null for every compiler-created allocation.
inline VarPtr GetPinnedAllocBase(const StmtPtr& stmt) {
  auto assign = As<AssignStmt>(stmt);
  if (!assign) return nullptr;
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_) return nullptr;
  if (!IsOp(call, "tile.alloc") && !IsOp(call, "tensor.alloc")) return nullptr;
  return call->GetKwarg<bool>("pinned", false) ? assign->var_ : nullptr;
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
/// byte_offset = (o0 * s1 * ... * sN + o1 * s2 * ... * sN + ... + oN) * storage_bits / 8
///
/// MemRef carries a byte offset rather than a nibble offset. Packed 4-bit
/// slices therefore require a static, byte-aligned logical origin in v1.
inline ExprPtr ComputeSliceByteOffset(const std::vector<ExprPtr>& offsets,
                                      const std::vector<ExprPtr>& parent_shape, const DataType& dtype,
                                      const Span& span) {
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

  const uint64_t storage_bits = storage_size::GetStorageBitWidth(dtype);
  INTERNAL_CHECK_SPAN(storage_bits > 0, span)
      << "Internal error: slice dtype has no storage width: " << dtype.ToString();
  if (storage_bits % 8 == 0) {
    auto elem_size_expr =
        std::make_shared<ConstInt>(static_cast<int64_t>(storage_bits / 8), DataType::INDEX, Span::unknown());
    return MulByteOffsets(result, elem_size_expr);
  }

  auto logical_offset = As<ConstInt>(result);
  CHECK_SPAN(logical_offset, span)
      << "Packed 4-bit slice offsets must be compile-time constants because MemRef cannot represent "
         "a dynamic nibble offset";
  CHECK_SPAN(logical_offset->value_ >= 0, span)
      << "Packed 4-bit slice offsets must be non-negative, but got logical offset " << logical_offset->value_;
  const auto byte_offset =
      storage_size::StaticLogicalOffsetToByte(static_cast<uint64_t>(logical_offset->value_), dtype);
  CHECK_SPAN(byte_offset.has_value(), span)
      << "Packed 4-bit slice origins must be byte-aligned; logical linear offset " << logical_offset->value_
      << " selects the second nibble of a byte";
  CHECK_SPAN(*byte_offset <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), span)
      << "Packed 4-bit slice byte offset overflows int64";
  return std::make_shared<ConstInt>(static_cast<int64_t>(*byte_offset), DataType::INDEX, Span::unknown());
}

/// Compute additional byte offset for a view operation.
/// Dispatches: slice ops → stride-based offset, others → zero offset.
inline ExprPtr ComputeViewByteOffset(const CallPtr& call, const TypePtr& parent_type) {
  const std::string& op_name = call->op_->name_;

  if (IsOp(call, "tensor.slice") || IsOp(call, "tile.slice")) {
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

    return ComputeSliceByteOffset(offsets, shaped->shape_, shaped->dtype_, call->span_);
  }

  // Non-slice view ops (reshape, transpose, extract):
  // No additional byte offset — same memory region, different interpretation
  return MakeZeroByteOffset();
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_MEMREF_UTILS_H_
