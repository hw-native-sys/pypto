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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_UNROLL_LOOPS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_UNROLL_LOOPS_H_

#include "pypto/ir/function.h"

namespace pypto::ir::transform_utils {
/// Keep iteration-dependent storage on the declaring assignment when a mutable
/// name spans conditional arms. The builder also needs this invariant before verification.
StmtPtr PrepareUnrolledBranchBindings(const StmtPtr& body);
/// Expand static unroll loops before isolated helper SSA conversion.
FunctionPtr UnrollFunction(const FunctionPtr& function);
// Substitute enclosing compile-time loop indices before extracting register scopes.
FunctionPtr UnrollRegScopeAncestors(const FunctionPtr& func);
}  // namespace pypto::ir::transform_utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_UNROLL_LOOPS_H_
