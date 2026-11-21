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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_PER_KERNEL_MEMORY_PLANNER_H_
#define PYPTO_IR_TRANSFORMS_UTILS_PER_KERNEL_MEMORY_PLANNER_H_

#include "pypto/ir/function.h"
#include "pypto/ir/transforms/pass_context.h"

namespace pypto::ir {

inline constexpr auto kAttrPtoasPlanMemory = "ptoas_plan_memory";

/// A mixed kernel may delegate its local buffer layout to PTOAS even when
/// the rest of the program uses an in-tree memory planner.
inline MemoryPlanner GetFunctionMemoryPlanner(const FunctionPtr& function) {
  if (function->GetAttr<bool>(kAttrPtoasPlanMemory, false)) return MemoryPlanner::PtoAS;
  const auto* context = PassContext::Current();
  return context ? context->GetMemoryPlanner() : MemoryPlanner::PyPTO;
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_PER_KERNEL_MEMORY_PLANNER_H_
