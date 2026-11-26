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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SSA_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SSA_UTILS_H_

#include "pypto/ir/function.h"

namespace pypto::ir {
/// Normalize one isolated function using the same converter as ConvertToSSA.
/// Scope outlining uses this to resolve pre-SSA rebindings in a new helper,
/// while its enclosing caller remains in the frontend phase.
FunctionPtr ConvertFunctionToSSA(const FunctionPtr& function);
}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SSA_UTILS_H_
