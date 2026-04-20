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

#include "pypto/backend/910B/backend_910b_handler.h"

#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

const Ascend910BHandler& Ascend910BHandler::Instance() {
  static const Ascend910BHandler instance;
  return instance;
}

ir::TileView Ascend910BHandler::BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                           const ir::TileView& original_view) const {
  // Ascend910B (a2a3): cross-core transfer goes through GM. All GM -> Mat
  // transfers must be in NZ layout (hardware constraint), so Left / Right /
  // Mat destinations all use NZ at the transfer boundary. The final
  // Left/Right layout is later resolved by a Mat -> Left/Right move (MTE1).
  //
  // Vec destinations preserve the original Vec view: the GM-backed C2V pop
  // materialises through an ND GlobalTensor on the consumer side and PTO-ISA
  // only supports Vec loads for matching ND/DN/NZ layouts. Emitting an NZ Vec
  // bridge tile would make the generated kernel invalid.
  ir::TileView result = original_view;
  switch (dest_ms) {
    case ir::MemorySpace::Left:
    case ir::MemorySpace::Right:
    case ir::MemorySpace::Mat:
      result.blayout = ir::TileLayout::col_major;
      result.slayout = ir::TileLayout::row_major;
      return result;
    case ir::MemorySpace::Vec:
      return original_view;
    default:
      INTERNAL_UNREACHABLE << "cross-core move destination must be Vec, Mat, Left, or Right, got "
                           << static_cast<int>(dest_ms);
  }
}

}  // namespace backend
}  // namespace pypto
