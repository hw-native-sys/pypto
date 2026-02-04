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

#ifndef PYPTO_BACKEND_BACKEND_910B_H_
#define PYPTO_BACKEND_BACKEND_910B_H_

#include <string>

#include "pypto/backend/backend.h"

namespace pypto {
namespace backend {

/**
 * @brief Backend implementation for 910B hardware
 *
 * Implements memory hierarchy and path finding for 910B architecture.
 * The SoC structure is fixed and created internally in the constructor.
 */
class Backend910B : public Backend {
 public:
  /**
   * @brief Construct 910B backend with standard configuration
   *
   * Creates the standard 910B SoC structure
   */
  Backend910B();

  /**
   * @brief Get backend type name
   *
   * @return "910B"
   */
  [[nodiscard]] std::string GetTypeName() const override { return "910B"; }
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_BACKEND_910B_H_
