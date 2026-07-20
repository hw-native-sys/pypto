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

PTOOpSpec ScalarBinaryDestinationPassingSpec() {
  return PTOOpSpec{
      {PTOOperandGroup{PTOOperandRole::Input, PTOMemoryEffect::Read, PTOOperandTypeConstraint::TileBuffer, 1,
                       1},
       PTOOperandGroup{PTOOperandRole::Input, PTOMemoryEffect::None, PTOOperandTypeConstraint::Scalar, 1, 1},
       PTOOperandGroup{PTOOperandRole::Output, PTOMemoryEffect::Write, PTOOperandTypeConstraint::TileBuffer,
                       1, 1}}};
}

PTOOpSpec ScalarFillDestinationPassingSpec() {
  return PTOOpSpec{
      {PTOOperandGroup{PTOOperandRole::Input, PTOMemoryEffect::None, PTOOperandTypeConstraint::Scalar, 1, 1},
       PTOOperandGroup{PTOOperandRole::Output, PTOMemoryEffect::Write, PTOOperandTypeConstraint::TileBuffer,
                       1, 1}}};
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

REGISTER_OP("pto.tsub")
    .set_op_category("PTOTargetOp")
    .set_description("Subtract two PTO tile buffers into an explicit destination buffer")
    .add_argument("lhs", "Left input PTO tile buffer")
    .add_argument("rhs", "Right input PTO tile buffer")
    .add_argument("output", "Destination PTO tile buffer")
    .set_internal_only()
    .set_pto_op_spec(BinaryDestinationPassingSpec())
    .f_deduce_type(DeduceVoidTargetType);

REGISTER_OP("pto.tdiv")
    .set_op_category("PTOTargetOp")
    .set_description("Divide two PTO tile buffers into an explicit destination buffer")
    .add_argument("lhs", "Left input PTO tile buffer")
    .add_argument("rhs", "Right input PTO tile buffer")
    .add_argument("output", "Destination PTO tile buffer")
    .set_internal_only()
    .set_pto_op_spec(BinaryDestinationPassingSpec())
    .f_deduce_type(DeduceVoidTargetType);

REGISTER_OP("pto.tmatmul")
    .set_op_category("PTOTargetOp")
    .set_description("Multiply two PTO tile buffers into an explicit accumulator destination")
    .add_argument("lhs", "Left matrix PTO tile buffer")
    .add_argument("rhs", "Right matrix PTO tile buffer")
    .add_argument("output", "Destination accumulator PTO tile buffer")
    .set_internal_only()
    .set_pto_op_spec(BinaryDestinationPassingSpec())
    .f_deduce_type(DeduceVoidTargetType);

REGISTER_OP("pto.trowsum")
    .set_op_category("PTOTargetOp")
    .set_description("Reduce rows using an explicit temporary and destination PTO tile buffer")
    .add_argument("input", "Input PTO tile buffer")
    .add_argument("temporary", "Temporary PTO tile buffer")
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

#define REGISTER_PTO_UNARY(OP_NAME, DESCRIPTION)             \
  REGISTER_OP(OP_NAME)                                       \
      .set_op_category("PTOTargetOp")                        \
      .set_description(DESCRIPTION)                          \
      .add_argument("input", "Input PTO tile buffer")        \
      .add_argument("output", "Destination PTO tile buffer") \
      .set_internal_only()                                   \
      .set_pto_op_spec(UnaryDestinationPassingSpec())        \
      .f_deduce_type(DeduceVoidTargetType)

REGISTER_PTO_UNARY("pto.tabs", "Compute absolute value into an explicit PTO tile-buffer destination");
REGISTER_PTO_UNARY("pto.texp", "Compute exponential into an explicit PTO tile-buffer destination");
REGISTER_PTO_UNARY("pto.tlog", "Compute logarithm into an explicit PTO tile-buffer destination");
REGISTER_PTO_UNARY("pto.trecip", "Compute reciprocal into an explicit PTO tile-buffer destination");
REGISTER_PTO_UNARY("pto.tneg", "Compute negation into an explicit PTO tile-buffer destination");
REGISTER_PTO_UNARY("pto.tnot", "Compute bitwise not into an explicit PTO tile-buffer destination");
REGISTER_PTO_UNARY("pto.trelu", "Compute ReLU into an explicit PTO tile-buffer destination");
REGISTER_PTO_UNARY("pto.tmov", "Move a PTO tile buffer into an explicit destination buffer");
REGISTER_PTO_UNARY("pto.tfillpad", "Fill invalid elements of a PTO tile buffer");

#undef REGISTER_PTO_UNARY

#define REGISTER_PTO_SCALAR_BINARY(OP_NAME, DESCRIPTION)     \
  REGISTER_OP(OP_NAME)                                       \
      .set_op_category("PTOTargetOp")                        \
      .set_description(DESCRIPTION)                          \
      .add_argument("input", "Input PTO tile buffer")        \
      .add_argument("scalar", "Scalar input")                \
      .add_argument("output", "Destination PTO tile buffer") \
      .set_internal_only()                                   \
      .set_pto_op_spec(ScalarBinaryDestinationPassingSpec()) \
      .f_deduce_type(DeduceVoidTargetType)

REGISTER_PTO_SCALAR_BINARY("pto.tadds", "Add a scalar to a PTO tile buffer");
REGISTER_PTO_SCALAR_BINARY("pto.tsubs", "Subtract a scalar from a PTO tile buffer");
REGISTER_PTO_SCALAR_BINARY("pto.tmuls", "Multiply a PTO tile buffer by a scalar");
REGISTER_PTO_SCALAR_BINARY("pto.tdivs", "Divide a PTO tile buffer by a scalar");

#undef REGISTER_PTO_SCALAR_BINARY

REGISTER_OP("pto.texpands")
    .set_op_category("PTOTargetOp")
    .set_description("Fill an explicit PTO tile-buffer destination with a scalar")
    .add_argument("value", "Scalar fill value")
    .add_argument("output", "Destination PTO tile buffer")
    .set_internal_only()
    .set_pto_op_spec(ScalarFillDestinationPassingSpec())
    .f_deduce_type(DeduceVoidTargetType);

}  // namespace ir
}  // namespace pypto
