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

#include "pypto/backend/950/backend_950_handler.h"

#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

const Ascend950Handler& Ascend950Handler::Instance() {
  static const Ascend950Handler instance;
  return instance;
}

ir::TileView Ascend950Handler::BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                          const ir::TileView& original_view) const {
  // Ascend950 (a5): hardware cross-core pipe carries data in fractal layout.
  //   Left -> NZ (col_major blayout, row_major slayout)
  //   Right -> NZ (vec -> Mat does not support ZN fractal, so use NZ as on
  //                Ascend910B; this also works for the final Right operand)
  //   Mat -> NZ
  //   Vec -> preserve original (already-final layout)
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
