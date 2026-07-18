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

#ifndef SRC_IR_TRANSFORMS_LOOP_INVARIANT_MAT_RESIDENCY_H_
#define SRC_IR_TRANSFORMS_LOOP_INVARIANT_MAT_RESIDENCY_H_

#include "pypto/ir/program.h"

namespace pypto {
namespace ir {
namespace loop_invariant_mat_residency {

/// Hoist eligible compiler-generated GM-to-Mat matmul operand prefixes after
/// tile memory spaces are explicit, then remove the private bridge provenance.
[[nodiscard]] ProgramPtr Apply(const ProgramPtr& program);

}  // namespace loop_invariant_mat_residency
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_LOOP_INVARIANT_MAT_RESIDENCY_H_
