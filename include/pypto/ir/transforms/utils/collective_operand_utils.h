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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_COLLECTIVE_OPERAND_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_COLLECTIVE_OPERAND_UTILS_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"

namespace pypto::ir {

// Remote operands in kernel-context ABI order. Unlisted operands are local
// and impose no communication-engine requirement. The first remote operand
// defines the launch ranks; callers ensure the other domains are compatible.
struct CollectiveOperandRoles {
  std::vector<size_t> context_args;
  bool target_uses_urma = false;
};

[[nodiscard]] inline std::optional<CollectiveOperandRoles> GetCollectiveOperandRoles(const OpPtr& op) {
  if (!op) return std::nullopt;
  // Synthesized GlobalVars are not registered operators. Match their reserved
  // prefixes as well as the public and HOST builtin forms.
  if (IsOp(op, "pld.tensor.indexed_all_to_all_v") ||
      op->name_.rfind("__builtin_indexed_all_to_all_v__", 0) == 0) {
    return CollectiveOperandRoles{{1, 2, 3}, true};
  }
  if (IsOp(op, "pld.tensor.all_to_all_v") || IsOp(op, "builtin.tensor.all_to_all_v") ||
      op->name_.rfind("__builtin_all_to_all_v__", 0) == 0) {
    return CollectiveOperandRoles{{1, 2, 3}};
  }
  if (IsOp(op, "pld.tensor.allgather") || IsOp(op, "builtin.tensor.allgather") ||
      IsOp(op, "pld.tensor.all_to_all") || IsOp(op, "builtin.tensor.all_to_all")) {
    return CollectiveOperandRoles{{1, 2}};
  }
  if (IsOp(op, "pld.tensor.allreduce") || IsOp(op, "builtin.tensor.allreduce") ||
      IsOp(op, "builtin.tensor.allreduce_ring") || IsOp(op, "pld.tensor.broadcast") ||
      IsOp(op, "builtin.tensor.broadcast") || IsOp(op, "pld.tensor.reduce_scatter") ||
      IsOp(op, "builtin.tensor.reduce_scatter")) {
    return CollectiveOperandRoles{{0, 1}};
  }
  if (IsOp(op, "pld.tensor.barrier") || IsOp(op, "builtin.tensor.barrier")) {
    return CollectiveOperandRoles{{0}};
  }
  return std::nullopt;
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_COLLECTIVE_OPERAND_UTILS_H_
