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

#ifndef SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_INSPECTION_H_
#define SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_INSPECTION_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/type.h"

namespace pypto::ir::structural_comparison_detail {

// Private inspection of the exact comparison projection. Consumers apply their
// own eligibility policy; comparison itself continues to allow free captures.
struct Inspection {
  size_t weight = 1;
  std::vector<IRNodePtr> nodes;
  std::vector<TypePtr> types;
  std::vector<VarPtr> free_variables;
  // The flag identifies body callees handled by the standard IR mutator.
  std::vector<std::pair<OpPtr, bool>> operators;
};
Inspection Inspect(const FunctionPtr& function);

}  // namespace pypto::ir::structural_comparison_detail
#endif  // SRC_IR_TRANSFORMS_UTILS_STRUCTURAL_INSPECTION_H_
