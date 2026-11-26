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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_INLINE_FUNCTIONS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_INLINE_FUNCTIONS_H_

#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir::transform_utils {
// Propagate dump tags from an eliminated boundary to the dispatches that
// receive the tagged arguments. Run before inlining or outlining those dispatches.
StmtPtr TransferDispatchDumpVars(const StmtPtr& body, std::vector<VarPtr> dump_vars);

// Complete ordinary Inline expansion in programs containing source register
// scopes before captures become a VF ABI. This exposes scalar producers and
// Out writes too. Programs without source register scopes remain unchanged.
ProgramPtr InlineRegScopeCalls(const ProgramPtr& program);
}  // namespace pypto::ir::transform_utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_INLINE_FUNCTIONS_H_
