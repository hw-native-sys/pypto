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
 * @file spmd.cpp
 * @brief SPMD runtime intrinsics (get_block_idx, get_block_num, get_subblock_idx)
 *
 * These operations query the SPMD launch context at runtime.
 * They return the block/sub-block identity of the current core,
 * allowing kernels to compute data-parallel partitioning offsets.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// ============================================================================
// Type deduction helpers
// ============================================================================

TypePtr DeduceTileGetBlockIdxType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_idx returns INT64 (matches PTO get_block_idx / i64 dialect result type)
  return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceTileGetBlockNumType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_num returns INT64 (matches PTO get_block_num / i64 dialect result type)
  return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceTileGetSubblockIdxType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_subblock_idx returns INT64 (matches PTO get_subblock_idx / i64 and signed index math)
  return std::make_shared<ScalarType>(DataType::INT64);
}

// ============================================================================
// Op registration
// ============================================================================

REGISTER_OP("tile.get_block_idx")
    .set_op_category("TileOp")
    .set_description("Get the current block index")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGetBlockIdxType(args, kwargs, "tile.get_block_idx");
    });

REGISTER_OP("tile.get_block_num")
    .set_op_category("TileOp")
    .set_description("Get the total number of blocks in the current SPMD launch")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGetBlockNumType(args, kwargs, "tile.get_block_num");
    });

REGISTER_OP("tile.get_subblock_idx")
    .set_op_category("TileOp")
    .set_description("Get the current sub-block (vector core) index")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGetSubblockIdxType(args, kwargs, "tile.get_subblock_idx");
    });

}  // namespace ir
}  // namespace pypto
