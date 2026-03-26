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

#include "pypto/ir/transforms/utils/core_affinity.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace core_affinity {

CoreAffinity CombineAffinity(CoreAffinity a, CoreAffinity b) {
  if (a == b) return a;
  if (a == CoreAffinity::SHARED) return b;
  if (b == CoreAffinity::SHARED) return a;
  return CoreAffinity::MIXED;
}

bool IsCubeOp(const std::string& name) {
  static const std::unordered_set<std::string> cube_ops = {
      "tile.matmul",   "tile.matmul_acc", "tile.matmul_bias", "tile.gemv",
      "tile.gemv_acc", "tile.gemv_bias",  "tile.batch_matmul"};
  return cube_ops.count(name) > 0;
}

bool IsCubeMemorySpace(MemorySpace ms) { return ms != MemorySpace::DDR && ms != MemorySpace::Vec; }

std::optional<MemorySpace> GetFirstTileArgMemory(const CallPtr& call) {
  for (const auto& arg : call->args_) {
    if (auto var = std::dynamic_pointer_cast<const Var>(arg)) {
      if (auto tile_type = std::dynamic_pointer_cast<const TileType>(var->GetType())) {
        return tile_type->memory_space_;
      }
    }
  }
  return std::nullopt;
}

CVDirection ClassifyMoveDirection(const CallPtr& call) {
  if (!call || !call->op_) return CVDirection::NONE;

  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op || op->name_ != "tile.move") return CVDirection::NONE;

  auto src_memory = GetFirstTileArgMemory(call);
  if (!src_memory.has_value()) return CVDirection::NONE;

  std::optional<MemorySpace> target_memory;
  for (const auto& [key, value] : call->kwargs_) {
    if (key == "target_memory") {
      target_memory = AnyCast<MemorySpace>(value, "target_memory");
      break;
    }
  }
  INTERNAL_CHECK(target_memory.has_value()) << "Internal error: tile.move missing target_memory kwarg";

  bool src_cube = IsCubeMemorySpace(src_memory.value());
  bool tgt_cube = IsCubeMemorySpace(target_memory.value());
  if (src_cube && !tgt_cube) return CVDirection::CUBE_TO_VECTOR;
  if (!src_cube && tgt_cube) return CVDirection::VECTOR_TO_CUBE;
  return CVDirection::NONE;
}

CoreAffinity ClassifyCallAffinity(const CallPtr& call) {
  if (!call || !call->op_) return CoreAffinity::SHARED;
  if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
    return CoreAffinity::SHARED;
  }
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return CoreAffinity::SHARED;
  const auto& name = op->name_;
  if (IsCubeOp(name)) return CoreAffinity::CUBE;
  if (name == "tile.move") {
    auto dir = ClassifyMoveDirection(call);
    if (dir != CVDirection::NONE) return CoreAffinity::BOUNDARY;
    auto ms = GetFirstTileArgMemory(call);
    if (ms.has_value() && IsCubeMemorySpace(ms.value())) return CoreAffinity::CUBE;
    return CoreAffinity::VECTOR;
  }
  static const std::unordered_set<std::string> tile_arg_classified_ops = {"tile.store", "tile.reshape"};
  if (tile_arg_classified_ops.count(name)) {
    auto ms = GetFirstTileArgMemory(call);
    if (ms.has_value() && IsCubeMemorySpace(ms.value())) return CoreAffinity::CUBE;
    return CoreAffinity::VECTOR;
  }
  if (name == "tile.load") {
    for (const auto& [key, value] : call->kwargs_) {
      if (key == "target_memory") {
        return IsCubeMemorySpace(AnyCast<MemorySpace>(value, "target_memory")) ? CoreAffinity::CUBE
                                                                               : CoreAffinity::VECTOR;
      }
    }
    return CoreAffinity::VECTOR;
  }
  static const std::unordered_set<std::string> vector_cross_core_ops = {
      "system.aiv_initialize_pipe", "system.tpush_to_aic", "system.tfree_to_aic", "tile.tpush_to_aic",
      "tile.tpop_from_aic"};
  if (vector_cross_core_ops.count(name)) return CoreAffinity::VECTOR;
  static const std::unordered_set<std::string> cube_cross_core_ops = {
      "system.aic_initialize_pipe", "system.tfree_to_aiv", "system.tpush_to_aiv", "tile.tpush_to_aiv",
      "tile.tpop_from_aiv"};
  if (cube_cross_core_ops.count(name)) return CoreAffinity::CUBE;
  if (name.substr(0, 5) == "tile.") return CoreAffinity::VECTOR;
  return CoreAffinity::SHARED;
}

}  // namespace core_affinity
}  // namespace ir
}  // namespace pypto
