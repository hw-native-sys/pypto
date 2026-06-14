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

#ifndef PYPTO_BACKEND_SUPERSCALAR_BACKEND_SUPERSCALAR_NPU_H_
#define PYPTO_BACKEND_SUPERSCALAR_BACKEND_SUPERSCALAR_NPU_H_

#include <string>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/memory_allocator_policy.h"

namespace pypto {
namespace backend {

/**
 * @brief Backend implementation for the SuperscalarNPU architecture.
 *
 * Memory model: DDR (off-chip) plus a TREG register file of 256 fixed 4KB
 * blocks (1MB total), addressed by block index. There are no cube/vector cores.
 *
 * Code generation is not implemented yet — this backend exists to produce IR
 * (memory-space inference + register-renaming TREG allocation). It overrides
 * CreateMemoryAllocatorPolicy() to assign TREG block indices via
 * SuperscalarNPURegisterAllocatorPolicy.
 */
class BackendSuperscalarNPU : public Backend {
 public:
  /**
   * @brief Get the singleton instance.
   *
   * @return Reference to the singleton instance
   */
  static BackendSuperscalarNPU& Instance();

  /**
   * @brief Get backend type name.
   *
   * @return "SuperscalarNPU"
   */
  [[nodiscard]] std::string GetTypeName() const override { return "SuperscalarNPU"; }

  /**
   * @brief Get the BackendHandler singleton for this backend.
   */
  [[nodiscard]] const BackendHandler* GetHandler() const override;

  /**
   * @brief Create the register-renaming TREG allocation policy.
   */
  [[nodiscard]] ir::MemoryAllocatorPolicyPtr CreateMemoryAllocatorPolicy() const override;

 private:
  /**
   * @brief Private constructor (singleton pattern).
   */
  BackendSuperscalarNPU();
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_SUPERSCALAR_BACKEND_SUPERSCALAR_NPU_H_
