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

#ifndef PYPTO_CODEGEN_PTO_PTO_IR_PRINTER_H_
#define PYPTO_CODEGEN_PTO_PTO_IR_PRINTER_H_

#include <string>

#include "pypto/backend/common/backend.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace codegen {

/**
 * @brief Mechanical PTO target-IR to PTO-ISA MLIR printer.
 *
 * This printer handles the supported Step-4 target IR, including structured
 * ``ForStmt``/``IfStmt`` regions. Allocation and destination decisions must
 * already be explicit in the IR; it never recovers a destination from an
 * enclosing assignment or a Tile MemRef.
 */
class PTOIRPrinter {
 public:
  explicit PTOIRPrinter(const backend::Backend* backend);

  std::string Generate(const ir::ProgramPtr& program, bool emit_tile_addr = true);

 private:
  const backend::Backend* backend_;
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_PTO_IR_PRINTER_H_
