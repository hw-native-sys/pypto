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

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace core_affinity {

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
  INTERNAL_CHECK_SPAN(target_memory.has_value(), call->span_)
      << "Internal error: tile.move missing target_memory kwarg";

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

  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op->name_)) return CoreAffinity::SHARED;
  const auto& entry = registry.GetEntry(op->name_);

  // 1. Explicit override — cross-core ops, SPMD shared ops, tile.create/alloc.
  if (auto fixed = entry.GetCoreAffinity()) return *fixed;

  // 2. tile.move is the one dynamically-classified op: MIXED when the move
  // crosses the C/V divide (the stmt logically runs on both sides — a tpush
  // on the producer plus a tpop on the consumer), otherwise inherits from
  // the src memory. The per-stmt "is this a boundary" decision lives in
  // boundary_moves; MIXED here is just the affinity label so CombineAffinity
  // rolls up correctly through compound stmts.
  if (op->name_ == "tile.move") {
    if (ClassifyMoveDirection(call) != CVDirection::NONE) return CoreAffinity::MIXED;
    auto ms = GetFirstTileArgMemory(call);
    return (ms && IsCubeMemorySpace(*ms)) ? CoreAffinity::CUBE : CoreAffinity::VECTOR;
  }

  // 3. Output memory — set_output_memory(...) / set_output_memory_from_kwarg(...).
  // Covers matmul family (Acc -> CUBE), vector elementwise (Vec -> VECTOR),
  // tile.load / tile.full / tile.create target_memory dispatch, and so on.
  const auto& spec = entry.GetMemorySpec();
  if (spec && spec->deduce_output_memory) {
    auto out = spec->deduce_output_memory(call->kwargs_);
    if (out.has_value()) {
      return IsCubeMemorySpace(*out) ? CoreAffinity::CUBE : CoreAffinity::VECTOR;
    }
    // nullopt means "inherit from first tile input" (set_output_memory_inherit_input)
    // — fall through to the first-tile-input branch below.
  }

  // 4. First tile input — covers view ops (slice/reshape/transpose/extract)
  // and non-tile-output ops that still run on the side of their input tile
  // (tile.store, tile.mscatter).
  auto in_ms = GetFirstTileArgMemory(call);
  if (in_ms.has_value()) {
    return IsCubeMemorySpace(*in_ms) ? CoreAffinity::CUBE : CoreAffinity::VECTOR;
  }

  return CoreAffinity::SHARED;
}

}  // namespace core_affinity
}  // namespace ir
}  // namespace pypto
