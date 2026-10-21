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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
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
  const auto reduce_op = static_cast<ReduceOp>(op_value);

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  // Per-chunk shape: [1, SIZE] where SIZE = target.shape[1].
  auto size_expr = target_type->shape_[1];

  // A static extent that fits one chunk keeps the original single-tile path, so
  // existing IR stays byte-identical. Anything larger, or symbolic, walks the own
  // row in UB-safe chunks instead: the whole-row load this rule used to emit holds
  // SIZE * dtype bytes of VEC at once, which no real payload satisfies (a single
  // FP32 token row is 20 KiB at D = 5120, and the row count is a runtime value).
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.reduce_scatter");
  const auto size_const = As<ConstInt>(size_expr);
  const bool fits_one_chunk =
      size_const && size_const->value_ > 0 && size_const->value_ <= chunk_geometry.chunk_elements;

  if (fits_one_chunk) {
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

  // ---- Chunked path ----
  //
  // Mirrors mesh allreduce's fully-valid path: one ready barrier, then one
  // read-complete barrier per chunk. The per-chunk barrier carries the same WAR
  // argument Phase 3.5 made for the whole row, scoped to one chunk — no rank
  // overwrites chunk k of its row until every peer has read chunk k.
  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);
  auto chunk_cols = SelectStaticChunkCols(chunk_geometry, size_expr, span);
  auto chunk_shape_tuple = tile_conversion_utils::MakeShapeTuple({one_idx, chunk_cols}, span);

  const int64_t ready_generation = b.EmitBarrier(signal, comm, "", span);

  b.EmitFor(
      "col", zero_idx, size_expr, chunk_cols,
      [&](LoweringBuilder& chunk_body, const VarPtr& col) {
        auto remaining = MakeSub(size_expr, col, span);
        auto valid_cols = MakeMin(chunk_cols, remaining, span);
        auto chunk_offsets = tile_conversion_utils::MakeShapeTuple({comm.my_rank, col}, span);
        auto chunk_valid_shape_tuple = tile_conversion_utils::MakeShapeTuple({one_idx, valid_cols}, span);

        // A2/A3 peer MTE transfers must end on a 32-byte boundary, so an FP16
        // ragged tail cannot be read at its logical element width. Widen the
        // physical read to the aligned element count and let it spill into the
        // window's tail padding, then narrow back to the logical tail before
        // reduction. FP32 needs no widening here because its element width
        // already divides the tile alignment. Same contract as the chunked
        // allreduce path; see `allow_physical_tail_padding` in comm.h.
        ExprPtr remote_valid_cols = valid_cols;
        std::vector<std::pair<std::string, std::any>> remote_load_kwargs;
        if (target_type->dtype_ == DataType::FP16) {
          remote_valid_cols =
              MakeMul(MakeFloorDiv(MakeAdd(valid_cols, chunk_geometry.alignment_minus_one_idx, span),
                                   chunk_geometry.alignment_elements_idx, span),
                      chunk_geometry.alignment_elements_idx, span);
          remote_load_kwargs.emplace_back("allow_physical_tail_padding", true);
        }
        auto remote_valid_shape_tuple =
            tile_conversion_utils::MakeShapeTuple({one_idx, remote_valid_cols}, span);

        auto acc_loaded = chunk_body.Bind(
            "acc_loaded",
            reg.Create("tile.load", {target, chunk_offsets, chunk_shape_tuple, chunk_valid_shape_tuple},
                       {{"target_memory", MemorySpace::Vec}}, span),
            span);
        // The ragged TLOAD carries a dynamic valid_shape. Zero its padding and
        // promote the accumulator to the fixed physical chunk type before it
        // becomes an if-result tile; otherwise allocation may hoist an alloc_tile
        // whose valid_col depends on the loop variable, breaking SSA dominance.
        auto acc_initial = chunk_body.Bind(
            "acc_initial",
            reg.Create("tile.fillpad_inplace", {acc_loaded}, {{"pad_value", PadValue::zero}}, span), span);

        auto acc_final = chunk_body.EmitForReduce(
            "peer", zero_idx, comm.nranks_idx, one_idx, acc_initial,
            [&](LoweringBuilder& peer_body, const VarPtr& peer, const VarPtr& acc) {
              return peer_body.EmitIfExpr(
                  peer_body.NotEq(peer, comm.my_rank, span),
                  [&](LoweringBuilder& then_body) {
                    auto recv_loaded = then_body.Bind(
                        "recv_loaded",
                        OpRegistry::GetInstance().Create(
                            "pld.tile.remote_load",
                            {target, peer, chunk_offsets, chunk_shape_tuple, remote_valid_shape_tuple},
                            remote_load_kwargs, span),
                        span);
                    // The widened FP16 read covers padding past the logical
                    // tail; drop it before the zero-fill so the reduced tile
                    // keeps the chunk's real extent.
                    ExprPtr recv_tail = recv_loaded;
                    if (target_type->dtype_ == DataType::FP16) {
                      recv_tail = then_body.Bind(
                          "recv_tail",
                          reg.Create("tile.set_validshape", {recv_loaded, one_idx, valid_cols}, {}, span),
                          span);
                    }
                    auto recv = then_body.Bind("recv",
                                               reg.Create("tile.fillpad_inplace", {recv_tail},
                                                          {{"pad_value", PadValue::zero}}, span),
                                               span);
                    return then_body.Bind("acc_next", then_body.Reduce(reduce_op, acc, recv, span), span);
                  },
                  [&](LoweringBuilder&) -> ExprPtr { return acc; }, span);
            },
            span);

        // The signal cell sits at ready_generation after the ready barrier, and
        // each completed chunk adds one, so chunk k waits for ready_generation + 1 + k.
        auto chunk_base = std::make_shared<ConstInt>(ready_generation + 1, DataType::INDEX, span);
        auto chunk_id = MakeFloorDiv(col, chunk_cols, span);
        auto expected_idx = MakeAdd(chunk_id, chunk_base, span);
        auto expected_i32 = chunk_body.Bind(
            "chunk_expected", std::make_shared<ir::Cast>(expected_idx, DataType::INT32, span), span);
        chunk_body.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32,
                                 "_chunk", span);
        chunk_body.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, expected_i32, "_chunk", span);

        // Accumulation uses the fixed physical chunk type; narrow the final alias
        // back to the real tail before store so the last chunk cannot write past
        // the logical extent.
        auto store_value = chunk_body.Bind(
            "store_value", reg.Create("tile.set_validshape", {acc_final, one_idx, valid_cols}, {}, span),
            span);
        chunk_body.Bind("store_ret", reg.Create("tile.store", {store_value, chunk_offsets, target}, {}, span),
                        span);
      },
      span);

  // Self-clearing epilogue: this call issued ready_generation (1) + chunk_count
  // credits per peer — one for the ready barrier, one per completed chunk.
  // chunk_count = ceil(SIZE / chunk_cols) is built as an IR expression because
  // SIZE may be a runtime scalar; pld.system.notify's value only needs
  // ScalarType, so a symbolic total is legal here.
  auto chunk_cols_minus_one = MakeSub(chunk_cols, one_idx, span);
  auto chunk_count_idx = MakeFloorDiv(MakeAdd(size_expr, chunk_cols_minus_one, span), chunk_cols, span);
  auto total_idx =
      MakeAdd(std::make_shared<ConstInt>(ready_generation, DataType::INDEX, span), chunk_count_idx, span);
  auto total_i32 = b.Bind("reduce_scatter_reset_total",
                          std::make_shared<ir::Cast>(total_idx, DataType::INT32, span), span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  return target;
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
