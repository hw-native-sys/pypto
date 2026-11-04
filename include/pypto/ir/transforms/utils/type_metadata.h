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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_TYPE_METADATA_H_
#define PYPTO_IR_TRANSFORMS_UTILS_TYPE_METADATA_H_

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace type_metadata {

/// Storage fields have different validation and SSA-use policies. In particular,
/// Base is an allocation identity, whereas SlotIndex is an ordinary SSA use.
enum class StorageMetadataField { Base, ByteOffset, SlotIndex, Size };

/// Enumerate direct type children without recursion, caches or context changes.
///
/// on_type(TypePtr) receives tuple elements; on_geometry(ExprPtr) receives shape
/// and view expressions; on_storage(ExprPtr) receives MemRef/WindowBuffer
/// descriptors without treating their allocation identities as ordinary uses.
/// Callers retain control over shared-node memoization and lexical validation.
template <class OnType, class OnGeometry, class OnStorage>
inline void ForEachTypeMetadataChild(const TypePtr& type, OnType&& on_type, OnGeometry&& on_geometry,
                                     OnStorage&& on_storage) {
  if (!type) return;
  if (const auto tuple = As<TupleType>(type)) {
    for (const auto& element : tuple->types_) {
      if (element) on_type(element);
    }
    return;
  }

  auto visit_geometry = [&on_geometry](const auto& expressions) {
    for (const auto& expression : expressions) {
      if (expression) on_geometry(expression);
    }
  };

  // Report the descriptor first so the narrow SSA wrapper retains its previous
  // slot-before-geometry traversal order. A full checker may inspect all fields.
  if (const auto shaped = As<ShapedType>(type)) {
    if (shaped->memref_ && *shaped->memref_) on_storage(*shaped->memref_);
    visit_geometry(shaped->shape_);
  }
  if (const auto tensor = AsTensorTypeLike(type); tensor && tensor->tensor_view_) {
    visit_geometry(tensor->tensor_view_->valid_shape);
    visit_geometry(tensor->tensor_view_->stride);
  } else if (const auto tile = As<TileType>(type); tile && tile->tile_view_) {
    visit_geometry(tile->tile_view_->valid_shape);
    visit_geometry(tile->tile_view_->stride);
    if (tile->tile_view_->start_offset) on_geometry(tile->tile_view_->start_offset);
  }
  if (const auto tensor = As<DistributedTensorType>(type);
      tensor && tensor->window_buffer_ && *tensor->window_buffer_) {
    on_storage(*tensor->window_buffer_);
  }
}

/// Enumerate expression fields of a MemRef or WindowBuffer, ignoring null fields.
/// The descriptor itself is not visited. Field roles let clients exempt only
/// true allocation bases, or retain the narrow slot-only SSA traversal policy.
template <class OnExpr>
inline void ForEachStorageMetadataExpr(const ExprPtr& descriptor, OnExpr&& on_expr) {
  if (const auto memref = As<MemRef>(descriptor)) {
    if (memref->base_) on_expr(memref->base_, StorageMetadataField::Base);
    if (memref->byte_offset_) on_expr(memref->byte_offset_, StorageMetadataField::ByteOffset);
    if (memref->slot_index_ && *memref->slot_index_) {
      on_expr(*memref->slot_index_, StorageMetadataField::SlotIndex);
    }
  } else if (const auto window = As<WindowBuffer>(descriptor)) {
    if (window->base_) on_expr(window->base_, StorageMetadataField::Base);
    if (window->size_) on_expr(window->size_, StorageMetadataField::Size);
  }
}

}  // namespace type_metadata
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_TYPE_METADATA_H_
