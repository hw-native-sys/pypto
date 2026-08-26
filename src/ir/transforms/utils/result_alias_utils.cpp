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
#include "pypto/ir/transforms/utils/result_alias_utils.h"

#include <cstddef>
#include <optional>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"

namespace pypto {
namespace ir {

std::optional<size_t> ResultAliasedArgIndex(const CallPtr& call) {
  if (!call) return std::nullopt;

  const auto* entry = LookupOpEntry(call->op_);
  if (!entry) return std::nullopt;

  auto aliased = [&](size_t index) -> std::optional<size_t> {
    if (index >= call->args_.size()) return std::nullopt;
    INTERNAL_CHECK_SPAN(entry->HasDeclaredArgEffect(index), call->span_)
        << "Internal error: operator '" << call->op_->name_ << "' aliases its result to argument " << index
        << " without a declared effect on it";
    return index;
  };

  if (auto reused = entry->GetOutputReusesInputArg()) {
    return aliased(*reused);
  }

  // tensor.write(tensor, indices, value) and tensor.assemble(target, source,
  // offsets) both return the updated tensor.
  if (IsOp(call, "tensor.write") || IsOp(call, "tensor.assemble")) {
    return aliased(0);
  }
  // `tensor.expand_clone(input, target)` writes the expansion into `target`
  // and returns it.
  if (IsOp(call, "tensor.expand_clone")) {
    return aliased(1);
  }
  // Deliberately absent: the cross-rank transfers (`put` / `get` /
  // `remote_store`), `pld.system.notify` and `system.syncall`. Each writes a
  // window, but each deduces `UnknownType` — "side-effect-only, no SSA result
  // for downstream consumers". A result that does not exist cannot alias
  // anything, and their write targets already travel through `ArgEffect`,
  // `CallWriteTargets` and `AnalyzeCallAccess`. Listing them here would invite
  // a future consumer to read a destination alias out of a bare side effect.
  // Composite collectives return the window they reduced or gathered into.
  // allgather / all_to_all / all_to_all_v take a read-only local operand first,
  // so their result window is argument 1; the rest reduce argument 0 in place.
  if (IsOp(call, "pld.tensor.allgather") || IsOp(call, "pld.tensor.all_to_all") ||
      IsOp(call, "pld.tensor.all_to_all_v")) {
    return aliased(1);
  }
  if (IsOp(call, "pld.tensor.allreduce") || IsOp(call, "pld.tensor.reduce_scatter") ||
      IsOp(call, "pld.tensor.barrier") || IsOp(call, "pld.tensor.broadcast")) {
    return aliased(0);
  }
  // `LowerHostTensorCollectives` rewrites each `pld.tensor.*` collective into
  // its internal chip-dispatch twin, which lands bytes in the same window. The
  // twins are listed so the contract survives that rewrite; no pass that
  // consults this runs late enough to see one today, which is exactly why the
  // answer must be written down rather than re-derived.
  if (IsOp(call, "builtin.tensor.allgather") || IsOp(call, "builtin.tensor.all_to_all") ||
      IsOp(call, "builtin.tensor.all_to_all_v")) {
    return aliased(1);
  }
  if (IsOp(call, "builtin.tensor.allreduce") || IsOp(call, "builtin.tensor.allreduce_ring") ||
      IsOp(call, "builtin.tensor.reduce_scatter") || IsOp(call, "builtin.tensor.barrier") ||
      IsOp(call, "builtin.tensor.broadcast")) {
    return aliased(0);
  }
  return std::nullopt;
}

}  // namespace ir
}  // namespace pypto
