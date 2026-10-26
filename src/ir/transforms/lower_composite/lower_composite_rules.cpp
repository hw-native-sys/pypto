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

#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {
namespace lower_composite {

// Explicit references keep every rule linked without cross-TU static registration.
// This table only selects recipes. The mutator owns function-context filtering,
// visited operands, temporary naming, and tuple alias tracking. Rules must emit
// ops outside this table so a second pass invocation remains a structural no-op.
CompositeLoweringFn LookupCompositeRule(const std::string& op_name) {
  static const std::unordered_map<std::string, CompositeLoweringFn> kRules = {
      {"tile.sin", &LowerSinRule},
      {"tile.cos", &LowerCosRule},
      // tile.tquant_mx → tile.tquant_mx_raw + tile.tmov_x2zz (value-returning SSA).
      // Scratch tiles are created with MemorySpace::Vec before InferTileMemorySpace.
      {"tile.tquant_mx", &LowerTileTQuantMxRule},
      // tile.select → tile.create(scratch) + tile.sel|tile.sels (mask is already packed).
      // Scratch tiles are created with MemorySpace::Vec before InferTileMemorySpace.
      {"tile.select", &LowerTileSelectRule},
      {"pld.tensor.allreduce", &LowerTensorAllReduceRule},
      {"pld.tensor.allgather", &LowerTensorAllGatherRule},
      {"pld.tensor.reduce_scatter", &LowerTensorReduceScatterRule},
      {"pld.tensor.barrier", &LowerTensorBarrierRule},
      {"pld.tensor.broadcast", &LowerTensorBroadcastRule},
      {"pld.tensor.all_to_all", &LowerTensorAllToAllRule},
      {"pld.tensor.all_to_all_v", &LowerTensorAllToAllVRule},
      {"pld.tensor.indexed_all_to_all_v",
       +[](const CallPtr& call, const std::vector<ExprPtr>&, LoweringBuilder&) -> ExprPtr {
         CHECK_SPAN(false, call->span_)
             << "indexed_all_to_all_v supports only L2 orchestration on A5 URMA; InCore is unsupported";
         return nullptr;
       }},
  };
  auto it = kRules.find(op_name);
  return it == kRules.end() ? nullptr : it->second;
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
