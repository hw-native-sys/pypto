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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SIMPLIFY_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SIMPLIFY_H_

#include "pypto/ir/function.h"

namespace pypto::ir::transform_utils {

/// Apply the Simplify pass's function transformation without a nested pass.
/// With constants_only, propagate proven constants and rebuild types while
/// preserving symbolic algebra, control flow and yielded slots. Memory
/// initialization needs that form before it consumes address metadata.
FunctionPtr SimplifyFunction(const FunctionPtr& func, bool constants_only = false);

}  // namespace pypto::ir::transform_utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SIMPLIFY_H_
