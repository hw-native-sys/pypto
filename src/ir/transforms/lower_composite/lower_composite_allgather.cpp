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
#include "pypto/ir/comm.h"
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
// ``pld.tensor.allgather`` lowering rule
//
// All-gather: each rank pushes its single chunk to every peer's window slot
// via pld.tile.put (TPUT-based).  After the barrier, the window itself holds
// the full [NR, SIZE] gathered result (window-as-result).  Fully N-rank
// general — NR is read from the target's compile-time shape at lowering time.
//
//   arg[0] = local_data  — Tensor [1, SIZE] (plain) or Tile [1, SIZE], this rank's chunk
//   arg[1] = target      — DistributedTensor [NR, SIZE], staging window (also the result)
//   arg[2] = signal      — DistributedTensor INT32, cross-rank barrier
//
// Phases:
//   0.  tile.create [stage_rows, stage_cols] VEC — bounded staging tile for
//       auto-chunking; rows * cols fits one 16-KiB chunk and each row is
//       32-byte aligned whenever the transfer is at least one alignment unit
//   1.  for peer in 0..NR-1:
//         pld.tile.put(target, peer, local_data, put_stage,
//                      [my_rank, 0], [0, 0], [1, SIZE])
//       — push this rank's chunk into every peer's window at row my_rank.
//       Self-store (peer == my_rank) uses HCCL identity mapping (same
//       trust model as pld.tile.get self-path).  pld.tile.put auto-chunks
//       when SIZE exceeds the staging-tile capacity.
//   2.  barrier (AtomicAdd 1 -> wait Ge generation)
//   return target  (DistributedTensor rebind) — window IS the gathered result
//
// Compared to the original pull-based allgather, this push-based variant drops
// the out Tensor parameter and the per-peer pld.tile.get gather loop.  Total
// HBM drops from (NR+1)×SIZE to NR×SIZE, at the cost of the window remaining
// occupied until the caller consumes the result.
// ============================================================================

ExprPtr LowerTensorAllGatherRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 3, span)
      << "pld.tensor.allgather rule expects 3 args (local_data, target, signal), got " << args.size();
  const auto& local_data = args[0];
  const auto& target = args[1];
  const auto& signal = args[2];

  // local_data is user-provided (the DSL allows Tensor | DistributedTensor) and
  // the deducer defers its validation to the lowering passes, so this is a
  // user-facing contract check -> CHECK_SPAN.  The InCore push path only
  // supports a plain Tensor source: pld.tile.put reads its `src` via
  // AsTensorTypeLike; a DistributedTensor local_data would fault downstream.
  CHECK_SPAN(As<TensorType>(local_data->GetType()), span)
      << "pld.tensor.allgather local_data must be a plain Tensor [1, SIZE] on the "
         "InCore path, got "
      << local_data->GetType()->TypeName();
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.allgather target must be DistributedTensorType (deducer-rejected otherwise)";
  INTERNAL_CHECK_SPAN(target_type->shape_.size() == 2, span)
      << "pld.tensor.allgather target must be 2D [NR, SIZE]";
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  ValidateMeshSignalShape(signal_type, "pld.tensor.allgather", span);

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  // Per-chunk shape: [1, SIZE] where SIZE = target.shape[1].
  auto size_expr = target_type->shape_[1];
  auto chunk_shape = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span), size_expr}, span);

  // Offsets [0, 0] for loading local_data.
  auto zero_row_offsets =
      std::make_shared<MakeTuple>(std::vector<ExprPtr>{std::make_shared<ConstInt>(0, DataType::INDEX, span),
                                                       std::make_shared<ConstInt>(0, DataType::INDEX, span)},
                                  span);

  // No explicit tile.load here: pld.tile.put reads from a Tensor source and
  // auto-chunks the transfer through the VEC staging tile.

  // ---- Phase 1: push — pld.tile.put this rank's chunk into every peer's window ----
  // Each peer receives this rank's chunk at target[my_rank, 0:SIZE].
  // Self-store (peer == my_rank) uses HCCL identity mapping — the same trust
  // model as the pld.tile.get self-path in the original pull-based allgather.
  // pld.tile.put auto-chunks when SIZE exceeds the staging-tile capacity, so the
  // stage is capped to one chunk rather than sized from SIZE — a [1, SIZE] stage
  // would reserve SIZE * dtype bytes of VEC and overflow UB for a large SIZE.
  // chunk_shape stays the *transfer* extent handed to pld.tile.put below.
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.allgather");
  auto stage_shape =
      MakeCollectiveStageShape({std::make_shared<ConstInt>(1, DataType::INDEX, span), size_expr},
                               chunk_geometry, span, "pld.tensor.allgather");
  auto put_stage =
      b.Bind("ag_stage",
             reg.Create("tile.create", {stage_shape},
                        {{"dtype", target_type->dtype_}, {"target_memory", MemorySpace::Vec}}, span),
             span);

  auto my_rank_offsets = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{comm.my_rank, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  b.EmitFor(
      "peer", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& peer) {
        // pld.tile.put(dst, peer, src, stage, dst_offsets, src_offsets, shape):
        // push local_data contents to every peer's window at row my_rank.
        // src is the original Tensor local_data — pld.tile.put handles
        // tile-load/chunking internally through the stage tile.
        body.Bind(
            "push",
            reg.Create("pld.tile.put",
                       {target, peer, local_data, put_stage, my_rank_offsets, zero_row_offsets, chunk_shape},
                       {{"atomic", static_cast<int>(AtomicType::kNone)}}, span),
            span);
      },
      span);

  // ---- Phase 2: barrier ----
  const int64_t generation = b.EmitBarrier(signal, comm, "", span);

  // Self-clearing epilogue: exactly one credit per peer this call.
  auto total_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // Return target — the window IS the gathered result (window-as-result).
  return target;
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
