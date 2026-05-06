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

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// Cross-Core Tile Transfer Operations (tpush / tpop)
// ============================================================================

// Push tile data to AIV (from AIC)
REGISTER_OP("tile.tpush_to_aiv")
    .set_description("Push tile data from AIC to AIV via cross-core pipe")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPush)
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Push tile data to AIC (from AIV)
REGISTER_OP("tile.tpush_to_aic")
    .set_description("Push tile data from AIV to AIC via cross-core pipe")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPush)
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Pop tile data from AIC (into AIV)
REGISTER_OP("tile.tpop_from_aic")
    .set_description("Pop tile data from AIC cross-core pipe into AIV")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPop)
    .no_argument()
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Pop tile data from AIV (into AIC)
REGISTER_OP("tile.tpop_from_aiv")
    .set_description("Pop tile data from AIV cross-core pipe into AIC")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::TPop)
    .no_argument()
    .set_attr<int>("split")
    .set_attr<int>("id")
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

}  // namespace ir
}  // namespace pypto
