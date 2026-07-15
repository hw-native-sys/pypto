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

/**
 * @file elementwise.cpp
 * @brief Internal destination-passing PTO elementwise operation schemas.
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceVoidTargetType(const std::vector<ExprPtr>&,
                             const std::vector<std::pair<std::string, std::any>>&) {
  return GetUnknownType();
}

PTOOpSpec BinaryDestinationPassingSpec() {
  return PTOOpSpec{{PTOOperandGroup{PTOOperandRole::Input, PTOMemoryEffect::Read,
                                    PTOOperandTypeConstraint::TileBuffer, 2, 2},
                    PTOOperandGroup{PTOOperandRole::Output, PTOMemoryEffect::Write,
                                    PTOOperandTypeConstraint::TileBuffer, 1, 1}}};
}

PTOOpSpec UnaryDestinationPassingSpec() {
  return PTOOpSpec{{PTOOperandGroup{PTOOperandRole::Input, PTOMemoryEffect::Read,
                                    PTOOperandTypeConstraint::TileBuffer, 1, 1},
                    PTOOperandGroup{PTOOperandRole::Output, PTOMemoryEffect::Write,
                                    PTOOperandTypeConstraint::TileBuffer, 1, 1}}};
}

}  // namespace

REGISTER_OP("pto.tadd")
    .set_op_category("PTOTargetOp")
    .set_description("Add two PTO tile buffers into an explicit destination buffer")
    .add_argument("lhs", "Left input PTO tile buffer")
    .add_argument("rhs", "Right input PTO tile buffer")
    .add_argument("output", "Destination PTO tile buffer")
    .set_internal_only()
    .set_pto_op_spec(BinaryDestinationPassingSpec())
    .f_deduce_type(DeduceVoidTargetType);

REGISTER_OP("pto.tmul")
    .set_op_category("PTOTargetOp")
    .set_description("Multiply two PTO tile buffers into an explicit destination buffer")
    .add_argument("lhs", "Left input PTO tile buffer")
    .add_argument("rhs", "Right input PTO tile buffer")
    .add_argument("output", "Destination PTO tile buffer")
    .set_internal_only()
    .set_pto_op_spec(BinaryDestinationPassingSpec())
    .f_deduce_type(DeduceVoidTargetType);

REGISTER_OP("pto.tsqrt")
    .set_op_category("PTOTargetOp")
    .set_description("Compute square root into an explicit PTO tile-buffer destination")
    .add_argument("input", "Input PTO tile buffer")
    .add_argument("output", "Destination PTO tile buffer")
    .set_internal_only()
    .set_pto_op_spec(UnaryDestinationPassingSpec())
    .f_deduce_type(DeduceVoidTargetType);

}  // namespace ir
}  // namespace pypto
