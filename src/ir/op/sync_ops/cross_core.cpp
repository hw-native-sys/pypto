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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
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

// PTO emits `pto.reserve_buffer` / `pto.import_reserved_buffer` with `-> i32` result.
TypePtr DeduceI32ScalarType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)args;
  (void)kwargs;
  return std::make_shared<ScalarType>(DataType::INT32);
}

}  // namespace

// ============================================================================
// Registration Function for Cross-Core System Operations
// (tile.tpush/tpop are registered in tile_ops/cross_core.cpp)
// ============================================================================

// Release slot back to AIC producer (called by AIV consumer after tpop_from_aic)
REGISTER_OP("system.tfree_to_aic")
    .set_description("Release ring buffer slot back to AIC producer")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::TFree)
    .add_argument("tile", "Tile buffer obtained from tpop to release")
    .set_attr<int>("id")
    .f_deduce_type(DeduceUnknownType);

// Release slot back to AIV producer (called by AIC consumer after tpop_from_aiv)
REGISTER_OP("system.tfree_to_aiv")
    .set_description("Release ring buffer slot back to AIV producer")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::TFree)
    .add_argument("tile", "Tile buffer obtained from tpop to release")
    .set_attr<int>("id")
    .f_deduce_type(DeduceUnknownType);

// Initialize pipe on AIC side
REGISTER_OP("system.aic_initialize_pipe")
    .set_description("Initialize cross-core pipe on AIC side")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .set_cross_core_role(core_affinity::CrossCoreRole::InitializePipe)
    .add_argument("c2v_consumer_buf", "C2V consumer buffer base (i32 SSA)")
    .add_argument("v2c_consumer_buf", "V2C consumer buffer base (i32 SSA)")
    .set_attr<int>("dir_mask")
    .set_attr<int>("slot_size")
    .set_attr<int>("id")
    .f_deduce_type(DeduceUnknownType);

// Initialize pipe on AIV side
REGISTER_OP("system.aiv_initialize_pipe")
    .set_description("Initialize cross-core pipe on AIV side")
    .set_op_category("CrossCoreOp")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .set_cross_core_role(core_affinity::CrossCoreRole::InitializePipe)
    .add_argument("c2v_consumer_buf", "C2V consumer buffer base (i32 SSA)")
    .add_argument("v2c_consumer_buf", "V2C consumer buffer base (i32 SSA)")
    .set_attr<int>("dir_mask")
    .set_attr<int>("slot_size")
    .set_attr<int>("id")
    .f_deduce_type(DeduceUnknownType);

// Reserve a named buffer in a kernel
REGISTER_OP("system.reserve_buffer")
    .set_description("Reserve a named buffer for cross-core communication")
    .set_op_category("CrossCoreOp")
    .no_argument()
    .set_attr<std::string>("name")
    .set_attr<int>("size")
    .set_attr<int>("base")
    .f_deduce_type(DeduceI32ScalarType);

// Import a peer function's buffer
REGISTER_OP("system.import_peer_buffer")
    .set_description("Import a buffer from a peer function in the same group")
    .set_op_category("CrossCoreOp")
    .no_argument()
    .set_attr<std::string>("name")
    .set_attr<std::string>("peer_func")
    .f_deduce_type(DeduceI32ScalarType);

}  // namespace ir
}  // namespace pypto
