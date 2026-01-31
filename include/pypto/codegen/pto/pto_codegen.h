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

#ifndef PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_
#define PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_

#include <memory>
#include <string>

namespace pypto {
namespace ir {
// Forward declarations
class Program;
using ProgramPtr = std::shared_ptr<const Program>;
}  // namespace ir

namespace codegen {

/**
 * @brief PTO MLIR code generator
 *
 * Generates PTO-ISA MLIR format code from PyPTO IR Program.
 * Automatically generates make_tensor_view, subview, and alloc_tile instructions.
 */
class PTOCodegen {
 public:
  PTOCodegen() = default;
  ~PTOCodegen() = default;

  /**
   * @brief Generate PTO-ISA MLIR format code from IR Program
   *
   * @param program Input PyPTO IR Program
   * @return MLIR code as string
   */
  std::string Generate(const ir::ProgramPtr& program);
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_
