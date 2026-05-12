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

// ============================================================================
// Cross-Rank Signal Operations (notify / wait)
// ============================================================================

// Notify a remote rank by writing or atomic-adding a value into its signal slot.
// The signal operand is a 1-element INT32 Tensor that views the destination
// rank's GM signal location (typically obtained via import_peer_buffer);
// codegen lowers this to `pto::comm::TNOTIFY` via PTOAS `pto.comm.tnotify`.
REGISTER_OP("tile.comm_notify")
    .set_description(
        "Send a flag notification to a remote rank: write or atomic-add an INT32 value "
        "into the destination signal slot")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("signal", "Destination signal tensor (1-element INT32, GM, remote-rank window)")
    .add_argument("value", "INT32 scalar value to write or atomic-add")
    .set_attr<std::string>("op")  // "atomic_add" | "set"
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Block until the local rank's signal slot satisfies `signal <cmp> cmp_value`.
// The signal operand is a 1-element INT32 Tensor in local-rank GM (the slot
// peers atomic-add or set into via tile.comm_notify); codegen lowers this to
// `pto::comm::TWAIT` via PTOAS `pto.comm.twait`.
REGISTER_OP("tile.comm_wait")
    .set_description("Block until a local INT32 signal slot satisfies the given comparison against cmp_value")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("signal", "Local signal tensor (1-element INT32, GM) to poll")
    .add_argument("cmp_value", "INT32 scalar comparison value")
    .set_attr<std::string>("cmp")  // "eq" | "ne" | "gt" | "ge" | "lt" | "le"
    .no_memory_spec()
    .f_deduce_type(DeduceUnknownType);

// Non-blocking poll of the local signal slot: returns a BOOL result equal to
// `signal <cmp> cmp_value`. Same operand shape as tile.comm_wait, but does not
// block; codegen lowers this to `pto::comm::TTEST` via PTOAS `pto.comm.ttest`,
// which produces an MLIR `i1`.
REGISTER_OP("tile.comm_test")
    .set_description(
        "Non-blocking poll: return BOOL = (local INT32 signal slot <cmp> cmp_value); does not block")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("signal", "Local signal tensor (1-element INT32, GM) to poll")
    .add_argument("cmp_value", "INT32 scalar comparison value")
    .set_attr<std::string>("cmp")  // "eq" | "ne" | "gt" | "ge" | "lt" | "le"
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>&, const std::vector<std::pair<std::string, std::any>>&) {
      return std::static_pointer_cast<const Type>(std::make_shared<ScalarType>(DataType::BOOL));
    });

}  // namespace ir
}  // namespace pypto
