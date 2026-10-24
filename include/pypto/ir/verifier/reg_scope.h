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
#ifndef PYPTO_IR_VERIFIER_REG_SCOPE_H_
#define PYPTO_IR_VERIFIER_REG_SCOPE_H_

#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/function.h"

namespace pypto::ir {
/// Derive region ownership from definitions; captures are never persisted.
/// final_placement is enabled only after ExpandMixedKernel (or at codegen).
void VerifyRegScopes(const FunctionPtr& function, std::vector<Diagnostic>& diagnostics,
                     bool final_placement = false);
/// Builder/parser boundary: report the first user-facing, source-located error.
void CheckRegScopes(const FunctionPtr& function);
}  // namespace pypto::ir
#endif  // PYPTO_IR_VERIFIER_REG_SCOPE_H_
