/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_EMIT_H_
#define PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_EMIT_H_

#include <string>
#include <unordered_set>

#include "src/ir/transforms/auto_tile/vector_plan.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {

/** Emit exactly one vector kernel schedule for a marked function. */
[[nodiscard]] FunctionPtr EmitVectorSchedule(const VectorGraph& graph, const VectorSchedulePlan& plan,
                                             const std::unordered_set<std::string>& called_functions);

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_TILE_VECTOR_EMIT_H_
