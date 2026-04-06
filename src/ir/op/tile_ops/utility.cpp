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
 * @file utility.cpp
 * @brief Utility tile operations (print)
 *
 * This file implements utility/debugging operations for tile-level programming.
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceTilePrintType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument (tile), but got "
                          << args.size();
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
  // Pass-through: returns the input tile type (print is a side-effect operation)
  return tile_type;
}

REGISTER_OP("tile.runtime_print")
    .set_op_category("TileOp")
    .set_description("Print tile contents for debugging (generates pto.tprint)")
    .add_argument("tile", "Input tile to print (TileType)")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTilePrintType(args, kwargs, "tile.runtime_print");
    });

}  // namespace ir
}  // namespace pypto
