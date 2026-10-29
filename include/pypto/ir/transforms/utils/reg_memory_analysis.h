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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_REG_MEMORY_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_REG_MEMORY_ANALYSIS_H_

#include "pypto/ir/function.h"

namespace pypto::ir {
/// Check initialized byte intervals after MemRef alias resolution, only for
/// allocations whose Reg accesses and geometry are entirely static. Dynamic
/// allocations pass through without a proof; no Tile valid_shape is modified.
/// Visits the function in execution order and rejects uninitialized static reads.
/// This is a conservative verifier, not an SSA rewrite or runtime bounds check;
/// exhaustive branches retain their common facts; unknown loops add no facts.
void VerifyRegMemoryInitialization(const FunctionPtr& function);
}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_REG_MEMORY_ANALYSIS_H_
