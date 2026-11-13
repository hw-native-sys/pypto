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

#ifndef PYPTO_IR_VERIFIER_SYNC_H_
#define PYPTO_IR_VERIFIER_SYNC_H_

#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/function.h"

namespace pypto::ir {

/**
 * @brief Check logical completion schemas, ownership, and lexical SSA uses.
 *
 * Appends diagnostics without changing the function. Tokens are local to one
 * specialized AIC/AIV invocation; temporary tuples and structured control-flow
 * carries are allowed, but function/container/VF escapes are rejected. This
 * verifies logical well-formedness, not physical flag allocation feasibility.
 */
void VerifyLogicalSync(const FunctionPtr& function, std::vector<Diagnostic>& diagnostics);

/** @brief Verify a function and report its first source-located user error. */
void CheckLogicalSync(const FunctionPtr& function);

}  // namespace pypto::ir

#endif  // PYPTO_IR_VERIFIER_SYNC_H_
