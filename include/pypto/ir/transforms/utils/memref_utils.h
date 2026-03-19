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

#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

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
