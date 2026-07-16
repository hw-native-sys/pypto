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
 * @file memory.cpp
 * @brief Internal PTO target-memory operation schemas.
 */

#include <any>
#include <cstddef>
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

PTOOperandGroup Group(PTOOperandRole role, PTOMemoryEffect effect, PTOOperandTypeConstraint type_constraint,
                      size_t min_count, size_t max_count) {
  return PTOOperandGroup{role, effect, type_constraint, min_count, max_count};
}

}  // namespace

REGISTER_OP("pto.alloc_tile")
    .set_op_category("PTOTargetOp")
    .set_description("Create a typed PTO tile-buffer handle")
    .add_argument("operands", "Optional address followed by valid-row and valid-col scalar operands")
    .set_internal_only()
    .set_pto_op_spec(PTOOpSpec{
        {Group(PTOOperandRole::Metadata, PTOMemoryEffect::None, PTOOperandTypeConstraint::Scalar, 0, 1),
         Group(PTOOperandRole::Metadata, PTOMemoryEffect::None, PTOOperandTypeConstraint::Scalar, 2, 2)},
        PTOResultKind::TileBuffer,
        PTOMemoryEffect::Allocate})
    .f_deduce_type(DeduceVoidTargetType);

REGISTER_OP("pto.subview")
    .set_op_category("PTOTargetOp")
    .set_description("Create a typed SSA tile-buffer view over an existing PTO handle")
    .add_argument("source", "Source PTO tile buffer")
    .add_argument("shape", "Static physical subview shape")
    .add_argument("offset", "Subview row and column offsets")
    .add_argument("valid_shape", "Explicit subview valid row and column extents")
    .set_internal_only()
    .set_pto_op_spec(PTOOpSpec{
        {Group(PTOOperandRole::Input, PTOMemoryEffect::None, PTOOperandTypeConstraint::TileBuffer, 1, 1),
         Group(PTOOperandRole::Metadata, PTOMemoryEffect::None, PTOOperandTypeConstraint::Any, 3, 3)},
        PTOResultKind::TileBuffer,
        PTOMemoryEffect::None})
    .f_deduce_type(DeduceVoidTargetType);

REGISTER_OP("pto.tload")
    .set_op_category("PTOTargetOp")
    .set_description("Load a tensor partition into an explicit PTO tile-buffer destination")
    .add_argument("source", "Source tensor")
    .add_argument("offsets", "Explicit source partition offsets")
    .add_argument("valid_extents", "Explicit source partition sizes")
    .add_argument("output", "Destination PTO tile buffer")
    .set_internal_only()
    .set_pto_op_spec(PTOOpSpec{
        {Group(PTOOperandRole::Input, PTOMemoryEffect::Read, PTOOperandTypeConstraint::Any, 1, 1),
         Group(PTOOperandRole::Metadata, PTOMemoryEffect::None, PTOOperandTypeConstraint::Any, 2, 2),
         Group(PTOOperandRole::Output, PTOMemoryEffect::Write, PTOOperandTypeConstraint::TileBuffer, 1, 1)}})
    .f_deduce_type(DeduceVoidTargetType);

REGISTER_OP("pto.tstore")
    .set_op_category("PTOTargetOp")
    .set_description("Store from a PTO tile buffer into an explicit tensor partition")
    .add_argument("input", "Source PTO tile buffer")
    .add_argument("offsets", "Explicit destination partition offsets")
    .add_argument("valid_extents", "Explicit destination partition sizes")
    .add_argument("destination", "Destination tensor")
    .set_attr<int>("atomic")
    .set_internal_only()
    .set_pto_op_spec(PTOOpSpec{
        {Group(PTOOperandRole::Input, PTOMemoryEffect::Read, PTOOperandTypeConstraint::TileBuffer, 1, 1),
         Group(PTOOperandRole::Metadata, PTOMemoryEffect::None, PTOOperandTypeConstraint::Any, 2, 2),
         Group(PTOOperandRole::Output, PTOMemoryEffect::Write, PTOOperandTypeConstraint::Any, 1, 1)}})
    .f_deduce_type(DeduceVoidTargetType);

}  // namespace ir
}  // namespace pypto
