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

#include "pypto/core/logging.h"
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

TypePtr DeduceTilePassthroughType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "Expected 1 argument, got " << args.size();
  auto tile_type = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  CHECK(tile_type != nullptr) << "Expected TileType argument, got " << args[0]->GetType()->TypeName();
  return tile_type;
}

}  // namespace

// ============================================================================
// Registration Function for Cross-Core Operations
// ============================================================================

// Push tile data to AIV (from AIC)
REGISTER_OP("system.tpush_to_aiv")
    .set_description("Push tile data from AIC to AIV via cross-core pipe")
    .set_op_category("CrossCoreOp")
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("aiv_idx")
    .f_deduce_type(DeduceUnknownType);

// Push tile data to AIC (from AIV)
REGISTER_OP("system.tpush_to_aic")
    .set_description("Push tile data from AIV to AIC via cross-core pipe")
    .set_op_category("CrossCoreOp")
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("aiv_idx")
    .f_deduce_type(DeduceUnknownType);

// Pop tile data from AIC (into AIV)
REGISTER_OP("system.tpop_from_aic")
    .set_description("Pop tile data from AIC cross-core pipe into AIV")
    .set_op_category("CrossCoreOp")
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("aiv_idx")
    .f_deduce_type(DeduceTilePassthroughType);

// Pop tile data from AIV (into AIC)
REGISTER_OP("system.tpop_from_aiv")
    .set_description("Pop tile data from AIV cross-core pipe into AIC")
    .set_op_category("CrossCoreOp")
    .add_argument("tile", "Tile data to transfer")
    .set_attr<int>("aiv_idx")
    .f_deduce_type(DeduceTilePassthroughType);

// Initialize pipe on AIC side
REGISTER_OP("system.aic_initialize_pipe")
    .set_description("Initialize cross-core pipe on AIC side")
    .set_op_category("CrossCoreOp")
    .no_argument()
    .set_attr<int>("dir_mask")
    .set_attr<int>("slot_size")
    .f_deduce_type(DeduceUnknownType);

// Initialize pipe on AIV side
REGISTER_OP("system.aiv_initialize_pipe")
    .set_description("Initialize cross-core pipe on AIV side")
    .set_op_category("CrossCoreOp")
    .no_argument()
    .set_attr<int>("dir_mask")
    .set_attr<int>("slot_size")
    .f_deduce_type(DeduceUnknownType);

// Reserve a named buffer in a kernel
REGISTER_OP("system.reserve_buffer")
    .set_description("Reserve a named buffer for cross-core communication")
    .set_op_category("CrossCoreOp")
    .no_argument()
    .set_attr<std::string>("name")
    .set_attr<int>("size")
    .f_deduce_type(DeduceUnknownType);

// Import a peer function's buffer
REGISTER_OP("system.import_peer_buffer")
    .set_description("Import a buffer from a peer function in the same group")
    .set_op_category("CrossCoreOp")
    .no_argument()
    .set_attr<std::string>("name")
    .set_attr<std::string>("peer_func")
    .f_deduce_type(DeduceUnknownType);

}  // namespace ir
}  // namespace pypto
