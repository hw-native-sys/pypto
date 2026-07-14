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

#ifndef PYPTO_SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_INTERNAL_H_
#define PYPTO_SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_INTERNAL_H_

#include "pypto/ir/function.h"

namespace pypto {
namespace ir {
namespace flatten_tile_nd_to_2d {

/// Read-only validation of the preconditions required by the rewrite phase.
void Analyze(const FunctionPtr& func);

/// Rewrite one analyzed InCore function from ND tile operations to 2D form.
[[nodiscard]] FunctionPtr Rewrite(const FunctionPtr& func);

}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_SRC_IR_TRANSFORMS_FLATTEN_TILE_ND_TO_2D_INTERNAL_H_
