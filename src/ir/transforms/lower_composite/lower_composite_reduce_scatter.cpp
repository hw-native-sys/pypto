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
// ``pld.tensor.reduce_scatter`` lowering rule
//
// Reduce-scatter: each rank holds NR chunks; rank r receives reduced chunk r.
// Target shape [NR, SIZE].  5-phase decomposition matching allreduce:
//   Phase 2:   ready barrier (AtomicAdd 1 -> wait Ge generation)
//   Phase 3:   acc = load(target, [my_rank, 0], [1, SIZE])
//              for peer != my_rank:
//                  recv = remote_load(target, peer, [my_rank, 0], [1, SIZE])
//                  acc = Reduce(op, acc, recv)
//   Phase 3.5: post-reduce barrier (AtomicAdd 1 -> wait Ge generation + 1)
//              — WAR prevention
//   Phase 4:   tile.store(acc, [my_rank, 0], target)
// Returns target (in-place rebind).  All four ReduceOps (kSum/kMax/kMin/kProd)
// supported — the same dispatch the allreduce rules use.
// ============================================================================

ExprPtr LowerTensorReduceScatterRule(const CallPtr& call, const std::vector<ExprPtr>& args,
                                     LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 2, span)
      << "pld.tensor.reduce_scatter rule expects 2 args, got " << args.size();
  const auto& target = args[0];
  const auto& signal = args[1];
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.reduce_scatter target must be DistributedTensorType (deducer-rejected otherwise)";
  INTERNAL_CHECK_SPAN(target_type->shape_.size() == 2, span)
      << "pld.tensor.reduce_scatter target must be 2D [NR, SIZE]";
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  ValidateMeshSignalShape(signal_type, "pld.tensor.reduce_scatter", span);

  auto op_value = GetRequiredKwarg<int>(call->kwargs_, "op", "pld.tensor.reduce_scatter");
  INTERNAL_CHECK_SPAN(
      op_value >= static_cast<int>(ReduceOp::kSum) && op_value <= static_cast<int>(ReduceOp::kProd), span)
      << "pld.tensor.reduce_scatter lowering received unknown ReduceOp " << op_value;

  // core_num > 1 is a HOST-builtin-only multicore knob (RFC #2521 plan
  // 85/B6b, mirroring allreduce's K2/#2160 restriction) — this composite
  // path is single-block by construction and has no block_idx to partition
  // across.
  auto core_num = GetRequiredKwarg<int>(call->kwargs_, "core_num", "pld.tensor.reduce_scatter");
  CHECK_SPAN(core_num == 1, span)
      << "pld.tensor.reduce_scatter core_num > 1 is supported only in a HOST orchestrator; "
         "InCore composite calls must use core_num=1";
  const auto reduce_op = static_cast<ReduceOp>(op_value);

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  // Per-chunk shape: [1, SIZE] where SIZE = target.shape[1].
  auto size_expr = target_type->shape_[1];
  auto chunk_shape = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span), size_expr}, span);

  // Helper: data offset [my_rank, 0] — each rank reads/writes its own row.
  auto my_data_offsets = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{comm.my_rank, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

  // ---- Phase 2: ready barrier ----
  b.EmitBarrier(signal, comm, "", span);

  // ---- Phase 3: accumulate peers' chunks at [my_rank, 0] ----
  auto acc_initial = b.Bind("acc_initial",
                            reg.Create("tile.load", {target, my_data_offsets, chunk_shape, chunk_shape},
                                       {{"target_memory", MemorySpace::Vec}}, span),
                            span);

  auto acc_final = b.EmitForReduce(
      "peer", zero_idx, comm.nranks_idx, one_idx, acc_initial,
      [&](LoweringBuilder& body, const VarPtr& peer, const VarPtr& acc) {
        return body.EmitIfExpr(
            body.NotEq(peer, comm.my_rank, span),
            [&](LoweringBuilder& then_body) {
              auto recv = then_body.Bind(
                  "recv",
                  OpRegistry::GetInstance().Create("pld.tile.remote_load",
                                                   {target, peer, my_data_offsets, chunk_shape}, {}, span),
                  span);
              return then_body.Bind("acc_next", then_body.Reduce(reduce_op, acc, recv, span), span);
            },
            [&](LoweringBuilder&) -> ExprPtr { return acc; }, span);
      },
      span);

  // ---- Phase 3.5: post-reduce barrier ----
  // Same WAR hazard as allreduce: fast rank could overwrite its row before
  // slow rank reads it.  See allreduce lowering for full rationale.
  const int64_t final_generation = b.EmitBarrier(signal, comm, "2", span);

  // ---- Phase 4: store reduced chunk back into target[my_rank, 0] ----
  b.Bind("store_ret", reg.Create("tile.store", {acc_final, my_data_offsets, target}, {}, span), span);

  // Self-clearing epilogue: 2 credits per peer this call (ready + post-reduce).
  auto total_i32 = std::make_shared<ConstInt>(final_generation, DataType::INT32, span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  return target;
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
