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

#include "pypto/backend/superscalar/backend_superscalar_npu.h"

#include <memory>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/soc.h"
#include "pypto/backend/superscalar/backend_superscalar_npu_handler.h"
#include "pypto/backend/superscalar/register_allocator_policy.h"
#include "pypto/ir/memory_allocator_policy.h"

namespace pypto {
namespace backend {

BackendSuperscalarNPU::BackendSuperscalarNPU() : Backend(CreateSuperscalarNPUSoC()) {
  // Codegen is not implemented for SuperscalarNPU; no operators are registered.
}

BackendSuperscalarNPU& BackendSuperscalarNPU::Instance() {
  static BackendSuperscalarNPU instance;
  return instance;
}

const BackendHandler* BackendSuperscalarNPU::GetHandler() const { return &SuperscalarNPUHandler::Instance(); }

ir::MemoryAllocatorPolicyPtr BackendSuperscalarNPU::CreateMemoryAllocatorPolicy() const {
  return std::make_unique<SuperscalarNPURegisterAllocatorPolicy>();
}

}  // namespace backend
}  // namespace pypto
