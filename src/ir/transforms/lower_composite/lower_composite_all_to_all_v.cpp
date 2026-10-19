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

#include <cstddef>
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
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_common.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {
namespace lower_composite {

// ============================================================================
// LowerTensorAllToAllVRule — pld.tensor.all_to_all_v (variable-size all-to-all)
//
// Variable-size all-to-all (MPI_Alltoallv pattern). Each rank pushes a
// runtime-sized block to every peer via one pld.tile.put per destination.
// ``send_counts[dest]`` is read from device data and clamped to
// ``[0, MAX_RECV]``; that clamped value becomes the dynamic transfer row count.
// The 5-arg API signature (input, target, signal,
// send_counts, recv_counts) extends the symmetric all_to_all's
// window-as-result pattern: the intrinsic returns target, and the caller
// reads back from the window with tile.load. Each rank atomically publishes
// the clamped count into peer dest's recv_counts[my_rank, 0]. Scalar Set is
// unsafe here: its DCCI write-back flushes the entire cache line and can
// overwrite another rank's adjacent count. All count and signal updates must
// remain atomic, including when their buffers share a cache line.
//
// 3-phase push-based decomposition:
//
//   Phase 1 (receiver-owned reset):
//     Invalidate cached GM reads, then reset each local count with
//     AtomicAdd(-old_count), then announce readiness to every peer. Before
//     publishing to a destination, wait for its readiness credit (Ge 1).
//
//   Phase 2 (push):
//     For each dest ∈ [0, NR):
//       rows = clamp(send_counts[dest], 0, MAX_RECV)   // runtime scalar read
//       if rows > 0:
//         wait(signal, [dest, 0], Ge 1)  // non-self destinations only
//         notify(recv_counts, dest, [my_rank, 0], rows, AtomicAdd)
//       // Single pld.tile.put per destination: contiguous [rows, SIZE] block
//       // at input[dest*MAX_RECV, :] → target[my_rank*MAX_RECV, :]. The
//       // transfer shape is the runtime [rows, SIZE] (PTOAS accepts dynamic
//       // partition-view dims on pto.comm.tput). A bounded
//       // [stage_rows, stage_cols] tile feeds the TPUT engine, which
//       // 2-D-slides the transfer through it.
//
//   Phase 3: completion barrier and self-clearing credits
//     AtomicAdd(+1) on every peer cell, then Wait(Ge 2)
//     EmitEpilogueReset(-2) — remove both rounds' credits for window reuse
//
// MAX_RECV = target.shape[0] / NR (both must be compile-time constants) is
// the per-peer *capacity*: it fixes the flat row-index arithmetic
// (dest*MAX_RECV+r) so a receiver can locate each sender's block without
// knowing that sender's count.  Counts are clamped to [0, MAX_RECV] so an
// out-of-range (or negative) count cannot push past peer dest's capacity
// slice or produce a negative extent.  The transfer moves exactly
// clamp(send_counts[dest], 0, MAX_RECV) rows per destination — rows beyond
// the runtime count never cross the wire, and the receiver uses
// recv_counts[src] (the same clamped value, published at push time) to know
// how many leading rows of source src's block are valid, the same
// MPI_Alltoallv semantics applied to the logical result.
// ============================================================================

ExprPtr LowerTensorAllToAllVRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 6, span)
      << "pld.tensor.all_to_all_v rule expects 6 args "
         "(input, target, signal, send_counts, recv_counts, core_num), got "
      << args.size();
  const auto& input = args[0];
  const auto& target = args[1];
  const auto& signal = args[2];
  const auto& send_counts = args[3];
  const auto& recv_counts = args[4];

  // The composite rail expands into point-to-point primitives inside one
  // kernel, so it is single-core by construction. A multi-AIV request belongs
  // on the managed CHIP/L2 rail, which submits a gang of AIV blocks instead.
  // core_num (args[5]) is carried as a Scalar[INDEX] argument. This rail's
  // contract is compile-time-only: a dynamic value can never satisfy it, so it
  // is rejected the same as any non-1 constant.
  auto core_num_const = As<ConstInt>(args[5]);
  CHECK_SPAN(core_num_const && core_num_const->value_ == 1, span)
      << "InCore pld.tensor.all_to_all_v requires a compile-time core_num=1, got "
      << (core_num_const ? std::to_string(core_num_const->value_) : std::string("a dynamic value"))
      << "; call it from a CHIP Orchestration function to use the managed multi-AIV path";

  // input may be a plain Tensor or a window (DistributedTensor) — pld.tile.put
  // accepts Tensor-like sources via AsTensorTypeLike.
  auto input_type = AsTensorTypeLike(input->GetType());
  INTERNAL_CHECK_SPAN(input_type, span)
      << "pld.tensor.all_to_all_v input must be Tensor or DistributedTensor, got "
      << input->GetType()->TypeName();
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.all_to_all_v target must be DistributedTensorType (deducer-rejected otherwise)";
  INTERNAL_CHECK_SPAN(target_type->shape_.size() == 2, span)
      << "pld.tensor.all_to_all_v target must be 2D [NR*MAX_RECV, SIZE]";
  auto counts_type = AsTensorTypeLike(send_counts->GetType());
  INTERNAL_CHECK_SPAN(counts_type, span)
      << "pld.tensor.all_to_all_v send_counts must be Tensor-like (deducer-rejected otherwise)";
  const size_t counts_rank = counts_type->shape_.size();
  INTERNAL_CHECK_SPAN(counts_rank == 1 || counts_rank == 2, span)
      << "pld.tensor.all_to_all_v send_counts must be 1D [NR] or 2D [NR, 1] (deducer-rejected otherwise)";
  auto recv_type = As<DistributedTensorType>(recv_counts->GetType());
  INTERNAL_CHECK_SPAN(recv_type, span)
      << "pld.tensor.all_to_all_v recv_counts must be DistributedTensorType (deducer-rejected otherwise)";
  INTERNAL_CHECK_SPAN(recv_type->shape_.size() == 2, span)
      << "pld.tensor.all_to_all_v recv_counts must be 2D [NR, 1] (deducer-rejected otherwise)";

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  // SIZE = target[1].
  auto size_expr = target_type->shape_[1];

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  // MAX_RECV = target[0] / NR.  NR is extracted from signal[0]
  // (deducer-enforced compile-time constant).  Signal is required to be 2D
  // [NR, 1] so MakeSignalOffsets(rank) → [rank, 0] matches notify/wait.
  // These three validate the caller's declared window/signal shapes, so a
  // violation is a user error, not a compiler invariant — report it as such.
  auto total_rows_c = As<ConstInt>(target_type->shape_[0]);
  CHECK_SPAN(total_rows_c, span)
      << "pld.tensor.all_to_all_v target dim 0 must be a compile-time constant (it is split as "
         "NR * MAX_RECV to give every sender a fixed-capacity slot)";
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  INTERNAL_CHECK_SPAN(signal_type, span) << "signal must be DistributedTensorType";
  ValidateMeshSignalShape(signal_type, "pld.tensor.all_to_all_v", span);
  auto nr_c = As<ConstInt>(signal_type->shape_[0]);
  CHECK_SPAN(nr_c, span) << "pld.tensor.all_to_all_v signal dim 0 (NR) must be a compile-time constant";
  int64_t max_recv_value = total_rows_c->value_ / nr_c->value_;
  // Divisibility is load-bearing, not incidental: the receiver locates sender
  // s's block at row s * MAX_RECV without knowing s's count, so every sender
  // needs an equal-capacity slot. This is deliberately not relaxed.
  CHECK_SPAN(max_recv_value * nr_c->value_ == total_rows_c->value_, span)
      << "pld.tensor.all_to_all_v target dim 0 (" << total_rows_c->value_ << ") must be divisible by NR ("
      << nr_c->value_
      << "): the receiver locates sender s's block at row s * MAX_RECV without knowing s's count, so "
         "every sender needs an equal-capacity slot. Round the window's row count up to a multiple of NR";
  auto max_recv_expr = std::make_shared<ConstInt>(max_recv_value, DataType::INDEX, span);

  // Reset only this receiver's own cells before allowing any rank to publish.
  // A scalar store (even to zero) can clobber a neighboring count or signal
  // during whole-line write-back. AtomicAdd avoids that race without padding
  // the public [NR, 1] layout. Per-destination readiness also prevents a fast
  // rank's next invocation from overwriting a slower receiver's previous result.
  b.EmitEval(reg.Create("system.cacheinvalid", {}, {}, span), span);
  b.EmitFor(
      "aav_reset_src", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& src) {
        auto offsets = tile_conversion_utils::MakeSignalOffsets(src, span);
        auto old_count =
            body.Bind("aav_old_count", reg.Create("tensor.read", {recv_counts, offsets}, {}, span), span);
        // Widen before negation so an arbitrary initial INT32_MIN is safe;
        // narrowing gives the modulo-2^32 delta for the hardware atomic add.
        auto reset_delta = body.Bind(
            "aav_reset_delta",
            MakeCast(MakeNeg(MakeCast(old_count, DataType::INT64, span), span), DataType::INT32, span), span);
        body.Bind("aav_count_reset",
                  reg.Create("pld.system.notify", {recv_counts, comm.my_rank, offsets, reset_delta},
                             {{"op", static_cast<int>(NotifyOp::kAtomicAdd)}}, span),
                  span);
      },
      span);
  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);
  // Announce to every peer before any blocking wait, avoiding cyclic waits.
  // Payloads to ready destinations need not wait for an unrelated slow rank.
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32, "_counts_reset",
                  span);

  // Per-destination staging tile, capped to one chunk — pto-isa slides the
  // static [MAX_RECV, SIZE] transfer through it, so the stage need not (and
  // should not) be sized from SIZE.
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.all_to_all_v");
  auto stage_shape =
      MakeCollectiveStageShape({max_recv_expr, size_expr}, chunk_geometry, span, "pld.tensor.all_to_all_v");

  // ---- Phase 2: push per-destination blocks to peer windows ----
  // One shared bounded [stage_rows, stage_cols] VEC tile is reused across all
  // destinations. A single pld.tile.put transfers [rows, SIZE], where rows is
  // the runtime count clamped to [0, MAX_RECV], and 2-D-slides that transfer
  // through the stage — so only the payload crosses the wire.
  // Flat row-index arithmetic:
  // source[dest*MAX_RECV, :] → target[my_rank*MAX_RECV, :].
  auto put_stage =
      b.Bind("aav_stage",
             reg.Create("tile.create", {stage_shape},
                        {{"dtype", target_type->dtype_}, {"target_memory", MemorySpace::Vec}}, span),
             span);

  // Offset of this rank's slot in peer recv_counts ([my_rank, 0]).
  auto my_recv_offsets = tile_conversion_utils::MakeSignalOffsets(comm.my_rank, span);

  b.EmitFor(
      "dest", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& dest_var) {
        auto dest_base = MakeMul(dest_var, max_recv_expr, span);
        auto my_base = MakeMul(comm.my_rank, max_recv_expr, span);

        // Per-destination row count, read from device data at runtime
        // (``tensor.read`` → ``pto.load``) and clamped to the
        // compile-time capacity: a count above MAX_RECV would otherwise push
        // into the next destination's slice of the peer window.
        std::vector<ExprPtr> count_indices{dest_var};
        if (counts_rank == 2) count_indices.push_back(zero_idx);
        auto count_value =
            body.Bind("aav_count",
                      reg.Create("tensor.read",
                                 {send_counts, std::make_shared<MakeTuple>(count_indices, span)}, {}, span),
                      span);
        // Clamped on BOTH sides: above by MAX_RECV (a larger count would push
        // into the next destination's slice of the peer window) and below by 0.
        // The lower clamp matters because ``rows`` now sizes the transfer: a
        // negative ``send_counts`` would otherwise yield a negative extent. The
        // HOST builtin kernel applies the identical two-sided clamp, so the two
        // rails stay bit-for-bit identical on the wire for every input,
        // including negative counts.
        auto rows =
            body.Bind("aav_rows",
                      MakeMax(MakeMin(MakeCast(count_value, DataType::INDEX, span), max_recv_expr, span),
                              zero_idx, span),
                      span);

        // Single pld.tile.put per destination transferring exactly the rows
        // being sent. The bounded [stage_rows, stage_cols] VEC tile feeds the
        // TPUT engine, which 2-D-slides the larger transfer through it.
        // 2D source offsets: input[dest * MAX_RECV, :]
        auto src_offsets = std::make_shared<MakeTuple>(
            std::vector<ExprPtr>{dest_base, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);
        // 2D target offsets: target[my_rank * MAX_RECV, :]
        auto dst_offsets = std::make_shared<MakeTuple>(
            std::vector<ExprPtr>{my_base, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);
        // Dynamic transfer shape: [rows, SIZE] — only the rows actually being
        // sent cross the interconnect, instead of the full MAX_RECV capacity.
        // PTOAS accepts dynamic partition-view dims on pto.comm.tput
        // (TPutOp::verify passes CommGlobalShapePolicy::AllowDynamicPartitionView),
        // and pld.tile.put needs no chunk_rows attr: it takes an explicit
        // bounded 2-D staging tile, and ValidateStageFitsTransfer skips dynamic
        // dims because the runtime extent bounds them.
        auto transfer_shape = std::make_shared<MakeTuple>(std::vector<ExprPtr>{rows, size_expr}, span);
        // Reset already establishes zero for empty edges; neither AtomicAdd(0)
        // nor a readiness wait is needed for them. The final completion wait
        // still covers every rank, including ranks with no outgoing payload.
        body.EmitIf(
            body.Gt(rows, zero_idx, span),
            [&](LoweringBuilder& then_body) {
              then_body.EmitIf(
                  then_body.NotEq(dest_var, comm.my_rank, span),
                  [&](LoweringBuilder& ready_body) {
                    auto offsets = tile_conversion_utils::MakeSignalOffsets(dest_var, span);
                    auto ready_wait = reg.Create("pld.system.wait", {signal, offsets, one_i32},
                                                 {{"cmp", static_cast<int>(WaitCmp::kGe)}}, span);
                    ready_body.Bind("aav_dest_ready", ready_wait, span);
                  },
                  /*else_fn=*/nullptr, span);
              auto count_i32 = then_body.Bind("aav_count_i32", MakeCast(rows, DataType::INT32, span), span);
              then_body.Bind(
                  "aav_count_notify",
                  reg.Create("pld.system.notify", {recv_counts, dest_var, my_recv_offsets, count_i32},
                             {{"op", static_cast<int>(NotifyOp::kAtomicAdd)}}, span),
                  span);
              then_body.Bind(
                  "aav_put",
                  reg.Create("pld.tile.put",
                             {target, dest_var, input, put_stage, dst_offsets, src_offsets, transfer_shape},
                             {{"atomic", static_cast<int>(AtomicType::kNone)}}, span),
                  span);
            },
            /*else_fn=*/nullptr, span);
      },
      span);

  // ---- Phase 3: self-clearing credit barrier ----
  auto total_i32 = std::make_shared<ConstInt>(2, DataType::INT32, span);
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32, "", span);
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, total_i32, "", span);

  // Self-clearing epilogue: remove both reset and completion credits.
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // Window-as-result: target[src*MAX_RECV+r, :] now holds the chunk from
  // rank src, offset r (full MAX_RECV capacity). The caller reads back from
  // the window with tile.load, using recv_counts[src] (clamped to MAX_RECV
  // at publish time) to identify valid rows and skip capacity holes.
  return target;
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
