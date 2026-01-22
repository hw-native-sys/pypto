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

#ifndef PYPTO_IR_TRANSFORM_PASSES_INIT_MEMREF_H_
#define PYPTO_IR_TRANSFORM_PASSES_INIT_MEMREF_H_

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
 *    If a variable is used as:
 *    - Source (1st arg) of `block.load`
 *    - Destination (6th arg) of `block.store`
 *    Then its memory space is set to `DDR`.
 */
class InitMemRefPass : public Pass {
 public:
  InitMemRefPass() = default;

  [[nodiscard]] std::string Name() const { return "InitMemRefPass"; }

  [[nodiscard]] FunctionPtr Run(const FunctionPtr& func) override;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_PASSES_INIT_MEMREF_H_
