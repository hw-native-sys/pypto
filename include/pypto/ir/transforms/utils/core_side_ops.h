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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_CORE_SIDE_OPS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_CORE_SIDE_OPS_H_

#include <string>

#include "pypto/ir/transforms/utils/core_affinity.h"

namespace pypto {
namespace ir {
namespace core_side_ops {

/// Op name for a tpush call issued by `self_side`.
/// AIC pushes to AIV, AIV pushes to AIC.
std::string TPushOp(core_affinity::CoreSide self_side);

/// Op name for a tpop call issued by `self_side` to drain data from the peer.
/// AIC pops from AIV, AIV pops from AIC.
std::string TPopOp(core_affinity::CoreSide self_side);

/// Op name for a tfree call issued by `self_side` after consuming a slot.
/// AIC frees the slot back to AIV, AIV frees the slot back to AIC.
std::string TFreeOp(core_affinity::CoreSide self_side);

/// Op name for the initialize_pipe call that runs on `self_side`.
std::string InitializePipeOp(core_affinity::CoreSide self_side);

/// CVDirection corresponding to a push issued by `self_side`.
/// AIC pushing == CUBE_TO_VECTOR, AIV pushing == VECTOR_TO_CUBE.
core_affinity::CVDirection PushDirection(core_affinity::CoreSide self_side);

}  // namespace core_side_ops
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_CORE_SIDE_OPS_H_
