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

#include "pypto/codegen/cce/isa_mapper.h"

#include <map>
#include <string>

#include "pypto/core/error.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto {

namespace codegen {

ISAMapper::ISAMapper() { InitializeMappings(); }

void ISAMapper::InitializeMappings() {
  // Memory operations
  mappings_["block.load"] = {"TLOAD"};
  mappings_["block.store"] = {"TSTORE"};
  mappings_["block.l0c_store"] = {"TSTORE"};
  mappings_["block.move"] = {"TMOV"};

  // Matrix operations
  mappings_["block.matmul"] = {"TMATMUL"};
  mappings_["block.matmul_acc"] = {"TMATMUL_ACC"};

  // Element-wise binary operations (Tile + Tile)
  mappings_["block.add"] = {"TADD"};
  mappings_["block.sub"] = {"TSUB"};
  mappings_["block.mul"] = {"TMUL"};
  mappings_["block.div"] = {"TDIV"};

  // Element-wise binary operations (Tile + Scalar)
  mappings_["block.adds"] = {"TADDS"};
  mappings_["block.subs"] = {"TSUBS"};
  mappings_["block.muls"] = {"TMULS"};
  mappings_["block.divs"] = {"TDIVS"};

  // Unary operations
  mappings_["block.sqrt"] = {"TSQRT"};

  // Synchronization operations
  mappings_["system.sync_src"] = {"set_flag"};
  mappings_["system.sync_dst"] = {"wait_flag"};
  mappings_["system.bar_v"] = {"pipe_barrier"};
  mappings_["system.bar_m"] = {"pipe_barrier"};
  mappings_["system.bar_all"] = {"pipe_barrier"};

  // Note: block.sum is handled specially in GetMapping based on 'axis' attribute
}

std::optional<ISAMapping> ISAMapper::GetMapping(const std::string& op_name,
                                                const std::map<std::string, ir::ExprPtr>& attrs) const {
  // Special handling for block.sum - needs axis attribute
  if (op_name == "block.sum") {
    auto axis_it = attrs.find("axis");
    if (axis_it != attrs.end()) {
      // Try to extract constant integer value for axis
      if (auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(axis_it->second)) {
        int64_t axis_value = const_int->value_;
        if (axis_value == 0) {
          // axis=0: sum across rows (collapse dim 0) → TCOLSUM
          return ISAMapping{"TCOLSUM"};
        }
        if (axis_value == 1) {
          // axis=1: sum across columns (collapse dim 1) → TROWSUM
          return ISAMapping{"TROWSUM"};
        }
        throw pypto::ValueError("Unsupported axis value for block.sum: " + std::to_string(axis_value) +
                                ". Only axis=0 or axis=1 are supported.");
      }
      throw pypto::ValueError("block.sum 'axis' attribute must be a constant integer");
    }
    throw pypto::ValueError("block.sum operation requires 'axis' attribute");
  }

  // Regular lookup for other operations
  auto it = mappings_.find(op_name);
  if (it != mappings_.end()) {
    return it->second;
  }

  return std::nullopt;
}

}  // namespace codegen

}  // namespace pypto
