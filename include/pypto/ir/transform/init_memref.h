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

#ifndef PYPTO_IR_TRANSFORM_INIT_MEMREF_H_
#define PYPTO_IR_TRANSFORM_INIT_MEMREF_H_

#include <string>

#include "pypto/ir/transform/base/pass.h"

namespace pypto {
namespace ir {

/**
 * @brief Initialize MemRef for all variables in a function
 *
 * This pass initializes the `memref` field for all `Var` nodes in the function.
 *
 * Rules:
 * 1. `memref.addr` is set to 0.
 * 2. `memref.size` is calculated based on static shape and dtype.
 *    If shape is dynamic (contains non-ConstInt), size defaults to 0.
 * 3. `memref.memory_space` defaults to `UB`.
 *    A variable's memory space is set to `DDR` if it is:
 *    - A function parameter (all input/output parameters are in main memory)
 *    - Source (1st arg) of `block.load`
 *    - Destination (6th arg) of `block.store`
 *
 * Special Handling:
 * - IterArg inherits MemRef from its initValue to maintain consistency
 *   (e.g., loop variables initialized with DDR parameters become DDR)
 * - Variables assigned from block.store inherit the MemRef of the 6th argument
 *   (the output tensor being stored to)
 */
class InitMemRefPass : public Pass {
 public:
  InitMemRefPass() = default;

  [[nodiscard]] std::string Name() const { return "InitMemRefPass"; }

  [[nodiscard]] FunctionPtr Run(const FunctionPtr& func) override;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_INIT_MEMREF_H_
