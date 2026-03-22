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

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

/// Find the YieldStmt inside a statement body (searches through SeqStmts).
inline YieldStmtPtr FindYieldStmt(const StmtPtr& body) {
  if (auto yield = As<YieldStmt>(body)) return yield;
  if (auto seq = As<SeqStmts>(body)) {
    for (const auto& child : seq->stmts_) {
      auto result = FindYieldStmt(child);
      if (result) return result;
    }
  }
  return nullptr;
}

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
  return TensorView(std::move(new_stride), tensor_view->layout, std::move(new_valid_shape));
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

inline bool TryRegisterUniqueMemRef(const MemRefPtr& memref, std::set<const MemRef*>& seen_ptrs) {
  CHECK(memref != nullptr) << "MemRef must not be null";
  return seen_ptrs.insert(memref.get()).second;
}

inline bool TryRegisterUniqueMemRef(const MemRefPtr& memref, MemorySpace memory_space,
                                    std::map<const MemRef*, MemorySpace>& seen_ptrs) {
  CHECK(memref != nullptr) << "MemRef must not be null";
  auto [it, inserted] = seen_ptrs.emplace(memref.get(), memory_space);
  CHECK(inserted || it->second == memory_space)
      << "Conflicting TileType.memory_space values found for the same MemRef";
  return inserted;
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_MEMREF_UTILS_H_
