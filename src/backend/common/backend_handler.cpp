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

#include "pypto/backend/common/backend_handler.h"

#include <cstdint>
#include <limits>

#include "pypto/backend/common/backend.h"
#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"

namespace pypto {
namespace backend {

uint32_t BackendHandler::GetCapacityBytes32(ir::MemorySpace memory_space) const {
  // GetMemSize returns a per-core capacity, without multiplying by core/die
  // counts. Keep the legacy 32-bit accessors from silently truncating SoC data.
  const auto capacity = backend_.GetMemSize(memory_space);
  INTERNAL_CHECK(capacity <= std::numeric_limits<uint32_t>::max())
      << "Internal error: SoC capacity exceeds the BackendHandler uint32_t API for "
      << ir::MemorySpaceToString(memory_space);
  return static_cast<uint32_t>(capacity);
}

uint32_t BackendHandler::GetL0aCapacityBytes() const { return GetCapacityBytes32(ir::MemorySpace::Left); }

uint32_t BackendHandler::GetL0bCapacityBytes() const { return GetCapacityBytes32(ir::MemorySpace::Right); }

uint32_t BackendHandler::GetL0cCapacityBytes() const { return GetCapacityBytes32(ir::MemorySpace::Acc); }

uint32_t BackendHandler::GetBiasCapacityBytes() const { return GetCapacityBytes32(ir::MemorySpace::Bias); }

uint64_t BackendHandler::GetMatCapacityBytes() const { return backend_.GetMemSize(ir::MemorySpace::Mat); }

}  // namespace backend
}  // namespace pypto
