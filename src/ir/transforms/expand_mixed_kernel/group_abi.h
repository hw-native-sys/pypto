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

#ifndef SRC_IR_TRANSFORMS_EXPAND_MIXED_KERNEL_GROUP_ABI_H_
#define SRC_IR_TRANSFORMS_EXPAND_MIXED_KERNEL_GROUP_ABI_H_

#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {
namespace expand_mixed_kernel {

struct NormalizedGroups {
  std::vector<FunctionPtr> functions;
};

// Normalize mixed Group members to the runtime's shared argument payload.
// The program must resolve the post-expansion functions supplied in functions.
NormalizedGroups NormalizeHandWrittenGroupAbis(const ProgramPtr& program,
                                               const std::vector<FunctionPtr>& functions);

}  // namespace expand_mixed_kernel
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_EXPAND_MIXED_KERNEL_GROUP_ABI_H_
