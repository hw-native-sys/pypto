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

#ifndef PYPTO_BACKEND_910_BACKEND_910_H_
#define PYPTO_BACKEND_910_BACKEND_910_H_

#include <string>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace backend {

/**
 * @brief Backend implementation for 910 hardware with PTO code generation
 *
 * Provides PTO (MLIR) code generation for 910 architecture.
 * Uses shared 910 SoC configuration created by Create910SoC().
 * Operators are registered via REGISTER_BACKEND_OP macro in separate
 * compilation units.
 */
class Backend910 : public Backend {
 public:
  /**
   * @brief Get registration instance for static operator registration
   *
   * Returns a singleton instance used during static initialization
   * to register operators via REGISTER_BACKEND_OP macro.
   *
   * @return Reference to registration instance
   */
  static Backend910& Instance();

  /**
   * @brief Get backend type name
   *
   * @return "910"
   */
  [[nodiscard]] std::string GetTypeName() const override { return "910"; }

  /**
   * @brief Get the BackendHandler singleton for this backend.
   */
  [[nodiscard]] const BackendHandler* GetHandler() const override;

  /**
   * @brief Generate PTO MLIR code for program
   *
   * @param program IR program to generate code for
   * @return Generated MLIR code string
   */
  std::string GenerateCode(const ir::ProgramPtr& program);

 private:
  /**
   * @brief Private constructor (singleton pattern)
   *
   * Constructor is private to enforce singleton pattern.
   * Use Instance() to get the singleton instance.
   */
  Backend910();
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_910_BACKEND_910_H_
