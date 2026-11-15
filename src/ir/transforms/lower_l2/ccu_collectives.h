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

#ifndef SRC_IR_TRANSFORMS_LOWER_L2_CCU_COLLECTIVES_H_
#define SRC_IR_TRANSFORMS_LOWER_L2_CCU_COLLECTIVES_H_

#include "pypto/ir/program.h"

namespace pypto::ir {
bool IsCcuAllGather(const CallPtr& call);
bool IsCcuCollective(const CallPtr& call);

// Part of LowerL2TensorCollectives, before direction/dependency derivation.
ProgramPtr LowerCcuCollectives(const ProgramPtr& program);
}  // namespace pypto::ir

#endif  // SRC_IR_TRANSFORMS_LOWER_L2_CCU_COLLECTIVES_H_
