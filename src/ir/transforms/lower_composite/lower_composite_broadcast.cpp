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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_common.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {
namespace lower_composite {

// ============================================================================
// ``pld.tensor.broadcast`` lowering rule
//
// Broadcast root rank's data to every rank:
//   Phase 2:  barrier (AtomicAdd 1 -> wait Ge generation)
//   Phase 3:  tile.create(VEC stage) + pld.tile.get(target, peer=root, src=target, stage)
// Returns target (in-place rebind).  Single barrier — broadcast is read-only
// after staging, no WAR hazard.
// ============================================================================

ExprPtr LowerTensorBroadcastRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 2, span)
      << "pld.tensor.broadcast rule expects 2 args, got " << args.size();
  const auto& target = args[0];
  const auto& signal = args[1];
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.broadcast target must be DistributedTensorType (deducer-rejected otherwise)";
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  ValidateMeshSignalShape(signal_type, "pld.tensor.broadcast", span);

  auto root_value = GetRequiredKwarg<int>(call->kwargs_, "root", "pld.tensor.broadcast");

  // core_num > 1 is a HOST-builtin-only multicore knob (RFC #2521 plan 85/B6b,
  // mirroring allreduce's K2/#2160 restriction) — this composite path is
  // single-block by construction and has no block_idx to partition across.
  auto core_num = GetRequiredKwarg<int>(call->kwargs_, "core_num", "pld.tensor.broadcast");
  CHECK_SPAN(core_num == 1, span)
      << "pld.tensor.broadcast core_num > 1 is supported only in a HOST orchestrator; "
         "InCore composite calls must use core_num=1";

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  auto root_expr = std::make_shared<ConstInt>(root_value, DataType::INT32, span);

  // ---- Phase 2: barrier ----
  const int64_t generation = b.EmitBarrier(signal, comm, "", span);

  // ---- Phase 3: pld.tile.get(root's data → local target slot) ----
  // Emit tile.create + pld.tile.get directly (the tensor-level get has no
  // codegen and ConvertTensorToTileOps runs before this pass).
  //
  // Build a 2D VEC staging tile [rows, cols] where rows = prod(dims[:-1]),
  // cols = dims[-1], mirroring ConvertTensorToTileOps's lowering of
  // pld.tensor.get.
  // The stage is a bounded bounce buffer, not a copy of the transfer, so the
  // target shape no longer has to be static: a dynamic extent simply takes the
  // chunk bound. pld.tile.get slides the full extent through it.
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.broadcast");
  auto stage_shape_tuple =
      MakeCollectiveStageShape(target_type->shape_, chunk_geometry, span, "pld.tensor.broadcast");

  auto stage_tile =
      b.Bind("bcast_stage",
             reg.Create("tile.create", {stage_shape_tuple},
                        {{"dtype", target_type->dtype_}, {"target_memory", MemorySpace::Vec}}, span),
             span);

  b.Bind("get_ret", reg.Create("pld.tile.get", {target, root_expr, target, stage_tile}, {}, span), span);

  // Self-clearing epilogue: exactly one credit per peer this call.
  auto total_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // In-place rebind: return target so the LHS Var holds the post-broadcast view.
  return target;
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
