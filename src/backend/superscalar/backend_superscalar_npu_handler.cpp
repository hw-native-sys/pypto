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

#include "pypto/backend/superscalar/backend_superscalar_npu_handler.h"

#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

const SuperscalarNPUHandler& SuperscalarNPUHandler::Instance() {
  static const SuperscalarNPUHandler instance;
  return instance;
}

ir::TileView SuperscalarNPUHandler::BuildCrossCoreTransferView(
    ir::MemorySpace dest_ms, [[maybe_unused]] const ir::TileView& original_view) const {
  // SuperscalarNPU has a single compute core (DDR + TREG); there are no
  // cross-core transfers, so this hook must never be reached.
  INTERNAL_UNREACHABLE << "SuperscalarNPU has no cross-core transfer; BuildCrossCoreTransferView called with "
                          "destination "
                       << static_cast<int>(dest_ms);
}

}  // namespace backend
}  // namespace pypto
