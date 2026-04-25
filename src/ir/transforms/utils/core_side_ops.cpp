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

#include "pypto/ir/transforms/utils/core_side_ops.h"

#include <string>

#include "pypto/ir/transforms/utils/core_affinity.h"

namespace pypto {
namespace ir {
namespace core_side_ops {

using core_affinity::CoreSide;
using core_affinity::CVDirection;

std::string TPushOp(CoreSide self_side) {
  return (self_side == CoreSide::AIC) ? "tile.tpush_to_aiv" : "tile.tpush_to_aic";
}

std::string TPopOp(CoreSide self_side) {
  return (self_side == CoreSide::AIC) ? "tile.tpop_from_aiv" : "tile.tpop_from_aic";
}

std::string TFreeOp(CoreSide self_side) {
  return (self_side == CoreSide::AIC) ? "system.tfree_to_aiv" : "system.tfree_to_aic";
}

std::string InitializePipeOp(CoreSide self_side) {
  return (self_side == CoreSide::AIC) ? "system.aic_initialize_pipe" : "system.aiv_initialize_pipe";
}

CVDirection PushDirection(CoreSide self_side) {
  return (self_side == CoreSide::AIC) ? CVDirection::CUBE_TO_VECTOR : CVDirection::VECTOR_TO_CUBE;
}

}  // namespace core_side_ops
}  // namespace ir
}  // namespace pypto
