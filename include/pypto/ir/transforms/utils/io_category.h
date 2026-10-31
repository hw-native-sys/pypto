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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_IO_CATEGORY_H_
#define PYPTO_IR_TRANSFORMS_UTILS_IO_CATEGORY_H_

#include <memory>

#include "pypto/core/any_cast.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/// Registry-backed IO classification shared by pipeline scheduling passes.
/// Match canonical names so deserialized and synthesized Op objects agree.
struct IOCategoryOps {
  OpPtr tile_load;      ///< Read: tensor → tile data movement
  OpPtr tile_read;      ///< Read: extract scalar from a tile
  OpPtr tile_store;     ///< Write: tile → tensor data movement
  OpPtr tile_write;     ///< Write: put scalar into a tile
  OpPtr tile_extract;   ///< Sub-tile extract — load-like only when L1→L0 (see IsL1ToL0ExtractCall)
  OpPtr tile_assemble;  ///< Acc→Mat sub-tile drain (Mat-scratch path) — drain-like only under dbC

  static IOCategoryOps Build() {
    const auto& registry = OpRegistry::GetInstance();
    return {
        registry.GetOp("tile.load"),  registry.GetOp("tile.read"),    registry.GetOp("tile.store"),
        registry.GetOp("tile.write"), registry.GetOp("tile.extract"), registry.GetOp("tile.assemble"),
    };
  }

  [[nodiscard]] bool IsLoadLike(const OpPtr& op) const {
    return op && (op->name_ == tile_load->name_ || op->name_ == tile_read->name_);
  }
  [[nodiscard]] bool IsStoreLike(const OpPtr& op) const {
    return op && (op->name_ == tile_store->name_ || op->name_ == tile_write->name_);
  }
  [[nodiscard]] bool IsAssemble(const OpPtr& op) const { return op && op->name_ == tile_assemble->name_; }

  /// True when @p call is a `tile.extract` whose source lives in L1 (Mat) and
  /// whose destination lives in L0a/L0b (Left/Right) — i.e. the ISA TEXTRACT
  /// L1→L0 data-movement pattern emitted by AutoTileMatmulL0. Such extracts
  /// are load-like for scheduling purposes: clustering them ahead of the
  /// matmul/matmul_acc consumers in the iteration body lets the codegen
  /// ping-pong on Left/Right buffers (analogous to how tile.load clustering
  /// enables DDR→Mat ping-pong).
  ///
  /// Other tile.extract patterns — non-Mat source, non-{Left,Right} target,
  /// or unknown memory space — keep the default TileCompute tier so we don't
  /// disturb compute orderings the dependency graph already constrains.
  [[nodiscard]] bool IsL1ToL0ExtractCall(const Call& call) const {
    if (!call.op_ || call.op_->name_ != tile_extract->name_) return false;
    if (call.args_.empty()) return false;
    auto src_tile = std::dynamic_pointer_cast<const TileType>(call.args_[0]->GetType());
    if (!src_tile) return false;
    auto src_ms = src_tile->GetMemorySpace();
    if (!src_ms.has_value() || *src_ms != MemorySpace::Mat) return false;
    for (const auto& [k, v] : call.kwargs_) {
      if (k != "target_memory") continue;
      auto target = AnyCast<MemorySpace>(v, "kwarg key: target_memory");
      return target == MemorySpace::Left || target == MemorySpace::Right;
    }
    return false;
  }
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_IO_CATEGORY_H_
