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

#include <algorithm>
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
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_common.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {
namespace lower_composite {

// ============================================================================
// ``pld.tensor.allreduce`` lowering rule
//
// In-place all-reduce of a window-bound DistributedTensor across every rank
// of its comm group. Expands the single composite Call into a ready barrier
// followed by UB-sized reduction chunks, as exercised by
// ``test_l3_tensor_allreduce_intrinsic.py``:
//
//   Ready 1: for peer in 0..nranks:
//               if peer != my_rank:
//                 pld.system.notify(signal, peer, [my_rank, 0], 1, op=AtomicAdd)
//   Ready 2: for src  in 0..nranks:
//               if src != my_rank:
//                 pld.system.wait(signal, [src, 0], 1, cmp=Ge)
//   Chunks : for each UB-sized chunk:
//              acc = tile.load(target, offsets, shape, valid_shape)
//              remote_load and accumulate every peer's chunk
//              AtomicAdd and wait for the monotonic value chunk_id + 2
//              narrow the ragged tail and tile.store(acc, offsets, target)
//
// The loop bound ``nranks`` is read at runtime via
// ``pld.system.nranks(pld.system.get_comm_ctx(target))`` so the lowering does
// not depend on CommGroup materialisation (which runs later in the pipeline).
// ``ReduceOp`` dispatch selects tile.add / tile.maximum / tile.minimum /
// tile.mul for Sum / Max / Min / Prod respectively.
//
// The Call's source-level form is the in-place rebind idiom shared with
// ``pl.store``:
//
//     pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum)
//
// so the rule returns the (post-reduce) ``target`` ExprPtr and lets the
// mutator bind it to the AssignStmt's LHS Var.
// ============================================================================

namespace {
// Forward declaration — the ring rule is defined after the mesh rule but is
// called from the mode dispatch inside LowerTensorAllReduceRule. It is not
// in lower_composite_rules.h because nothing outside this TU dispatches to
// it directly; the mesh rule selects it from the ``mode`` kwarg.
ExprPtr LowerTensorRingAllReduceRule(const CallPtr& call, const std::vector<ExprPtr>& args,
                                     LoweringBuilder& b);
}  // namespace

ExprPtr LowerTensorAllReduceRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  // Host-orchestrator calls may omit the signal and get one synthesized before
  // host collective lowering. InCore/composite lowering keeps the old explicit
  // signal contract so users get a direct error instead of an internal assert.
  CHECK_SPAN(args.size() == 2, span)
      << "pld.tensor.allreduce requires an explicit signal outside host orchestrator functions. "
         "Use pld.tensor.allreduce(target, signal, op=...) for InCore/lowered composite paths.";
  const auto& target = args[0];
  const auto& signal = args[1];
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.allreduce target must be DistributedTensorType (deducer-rejected otherwise)";
  CheckAllReduceTargetIsPackedNd(target_type, span);

  auto op_value = GetRequiredKwarg<int>(call->kwargs_, "op", "pld.tensor.allreduce");
  INTERNAL_CHECK_SPAN(
      op_value >= static_cast<int>(ReduceOp::kSum) && op_value <= static_cast<int>(ReduceOp::kProd), span)
      << "pld.tensor.allreduce lowering received unknown ReduceOp " << op_value;
  const auto reduce_op = static_cast<ReduceOp>(op_value);

  auto core_num = GetRequiredKwarg<int>(call->kwargs_, "core_num", "pld.tensor.allreduce");
  CHECK_SPAN(core_num == 1, span)
      << "pld.tensor.allreduce core_num > 1 is supported only in a HOST orchestrator; "
         "use an enclosing pl.spmd(...) for multi-core InCore execution";

  // Mode dispatch: "ring" delegates to the chunked reduce-scatter + allgather
  // ring schedule; "mesh" (default) uses the direct-exchange lowering below.
  // `mode` is a public DSL kwarg, so an unknown value is a user error — reject
  // it explicitly instead of silently defaulting to mesh.
  auto mode = GetKwargOr<std::string>(call->kwargs_, "mode", std::string("mesh"));
  CHECK_SPAN(mode == "ring" || mode == "mesh", span)
      << R"(pld.tensor.allreduce mode must be "ring" or "mesh", got ")" << mode << "\"";
  if (mode == "ring") {
    return LowerTensorRingAllReduceRule(call, args, b);
  }

  auto signal_type = As<DistributedTensorType>(signal->GetType());
  ValidateMeshSignalShape(signal_type, "pld.tensor.allreduce", span);

  // ---- Pre-build expressions shared across phases ----
  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  // Loop bounds: INDEX (must agree across start/stop/step). Notify's `value`
  // and wait's `expected` are INT32 per the Python builder's int_dtype
  // override — keep separate constants for those distinct slots.
  //
  // Barrier protocol: the self-clearing credit barrier (see the file-header
  // comment). The ready barrier is generation 1; each chunk-complete barrier
  // is one more, so chunk ``k`` waits for ``1 + k``. Those per-chunk
  // generations are derived by hand here — the barrier lives inside the chunk
  // loop, so it cannot go through ``EmitBarrier`` — and the total credit count
  // this call issued is subtracted back out by ``EmitEpilogueReset`` below.
  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.allreduce");
  auto alignment_elements_idx = chunk_geometry.alignment_elements_idx;
  auto alignment_minus_one_idx = chunk_geometry.alignment_minus_one_idx;
  auto max_chunk_cols = chunk_geometry.max_chunk_cols;

  const auto* partial_valid_shape = GetPartialValidShape(target_type, span);
  // A fully-valid packed tensor is one logical 1D stream. Keep the view
  // physically 2D as [1, N] because tile load/store codegen is 2D. A partial
  // ND valid box may have row gaps after a full linear flatten, so preserve the
  // existing [rows, cols] rectangle for that case.
  auto flat_shape = partial_valid_shape != nullptr ? CollapseShapeTo2D(target_type->shape_, span)
                                                   : CollapseShapeToLinear2D(target_type->shape_, span);
  auto flat_valid_shape = flat_shape;
  ExprPtr chunk_cols = max_chunk_cols;
  std::vector<ExprPtr> rectangular_tile_shape;
  if (partial_valid_shape != nullptr) {
    const auto& valid_shape = *partial_valid_shape;
    CHECK_SPAN(tile_conversion_utils::IsRowMajorCollapseContiguous(valid_shape, target_type->shape_), span)
        << "pld.tensor.allreduce target valid_shape cannot be represented by a single 2D view";
    flat_valid_shape = CollapseShapeTo2D(valid_shape, span);
    bool valid_shape_is_static = true;
    for (const auto& dim : flat_valid_shape) {
      if (!As<ConstInt>(dim)) {
        valid_shape_is_static = false;
        break;
      }
    }
    // Prefer the compact valid rectangle when it is statically allocatable.
    // For symbolic validity, fall back to the source's fixed physical rectangle;
    // this accepts e.g. shape=[64, 64], valid_shape=[1, m] without asking the UB
    // allocator to size a tile from the runtime value of m.
    rectangular_tile_shape = valid_shape_is_static ? flat_valid_shape : flat_shape;
    auto rectangular_elements = tile_conversion_utils::MakeCanonicalIndexMul(
        rectangular_tile_shape[0], rectangular_tile_shape[1], span, "LowerCompositeOps");
    CHECK_SPAN(ProveValidExtentLessEqual(rectangular_elements, max_chunk_cols) == ProofResult::kTrue, span)
        << "pld.tensor.allreduce partial valid_shape must fit within one " << kAllReduceChunkBytes
        << "-byte mesh chunk using a statically bounded tile; chunking a partial rectangle with row gaps "
           "is not supported";
  } else {
    chunk_cols = SelectStaticChunkCols(chunk_geometry, flat_valid_shape[1], span);
  }
  auto flat_target = b.Bind(
      "target_2d", CreateAllReduceTargetView(target, flat_shape, flat_valid_shape, partial_valid_shape, span),
      span);

  // ---- Phase 2: ready barrier (AtomicAdd 1 → wait Ge ready_generation) ----
  const int64_t ready_generation = b.EmitBarrier(signal, comm, "", span);

  // A partial ND valid box is not a contiguous linear stream: the physical
  // rows can contain gaps after valid_cols. Keep the established single
  // rectangular load for this rarer metadata case. The arbitrary-length
  // chunk path below handles fully-valid packed tensors, which are safe to
  // reinterpret as one contiguous stream.
  if (partial_valid_shape != nullptr) {
    auto zero_offsets = tile_conversion_utils::MakeZeroOffsets(2, span);
    auto rectangular_shape_tuple = tile_conversion_utils::MakeShapeTuple(rectangular_tile_shape, span);
    auto flat_valid_shape_tuple = tile_conversion_utils::MakeShapeTuple(flat_valid_shape, span);
    auto acc_initial = b.Bind(
        "acc_initial",
        reg.Create("tile.load", {flat_target, zero_offsets, rectangular_shape_tuple, flat_valid_shape_tuple},
                   {{"target_memory", MemorySpace::Vec}}, span),
        span);
    auto acc_final = b.EmitForReduce(
        "peer", zero_idx, comm.nranks_idx, one_idx, acc_initial,
        [&](LoweringBuilder& body, const VarPtr& peer, const VarPtr& acc) {
          return body.EmitIfExpr(
              body.NotEq(peer, comm.my_rank, span),
              [&](LoweringBuilder& then_body) {
                auto recv = then_body.Bind("recv",
                                           reg.Create("pld.tile.remote_load",
                                                      {flat_target, peer, zero_offsets,
                                                       rectangular_shape_tuple, flat_valid_shape_tuple},
                                                      {}, span),
                                           span);
                return then_body.Bind("acc_next", then_body.Reduce(reduce_op, acc, recv, span), span);
              },
              [&](LoweringBuilder& /*else_body*/) -> ExprPtr { return acc; }, span);
        },
        span);
    // Post-reduce barrier — one further generation, so the rectangle path
    // issues exactly two credits per peer this call.
    const int64_t final_generation = b.EmitBarrier(signal, comm, "2", span);
    b.Bind("store_ret", reg.Create("tile.store", {acc_final, zero_offsets, flat_target}, {}, span), span);
    auto total_i32 = std::make_shared<ConstInt>(final_generation, DataType::INT32, span);
    b.EmitEpilogueReset(signal, comm, total_i32, span);
    return target;
  }

  auto chunk_shape_tuple = tile_conversion_utils::MakeShapeTuple({one_idx, chunk_cols}, span);

  // ---- Phases 3/3.5/4: reduce one UB-safe chunk at a time ----
  //
  // The physical tile uses the selected statically aligned chunk width. The
  // final chunk carries [1, min(chunk_cols, valid_cols - col)] as valid_shape,
  // so local and remote TLOADs never read past the tensor while allocation
  // remains static.
  // A post-reduce barrier is required for every chunk before that chunk is
  // written back; otherwise a fast rank can overwrite bytes a slow peer has
  // not remote-loaded yet.
  b.EmitFor(
      "col", zero_idx, flat_valid_shape[1], chunk_cols,
      [&](LoweringBuilder& chunk_body, const VarPtr& col) {
        auto remaining = MakeSub(flat_valid_shape[1], col, span);
        auto valid_cols = MakeMin(chunk_cols, remaining, span);
        auto chunk_valid_shape_tuple = tile_conversion_utils::MakeShapeTuple({one_idx, valid_cols}, span);
        auto chunk_offsets = tile_conversion_utils::MakeShapeTuple({zero_idx, col}, span);
        ExprPtr remote_valid_cols = valid_cols;
        std::vector<std::pair<std::string, std::any>> remote_load_kwargs;
        if (target_type->dtype_ == DataType::FP16) {
          remote_valid_cols = MakeMul(
              MakeFloorDiv(MakeAdd(valid_cols, alignment_minus_one_idx, span), alignment_elements_idx, span),
              alignment_elements_idx, span);
          remote_load_kwargs.emplace_back("allow_physical_tail_padding", true);
        }
        auto remote_valid_shape_tuple =
            tile_conversion_utils::MakeShapeTuple({one_idx, remote_valid_cols}, span);

        auto acc_loaded = chunk_body.Bind(
            "acc_loaded",
            reg.Create("tile.load", {flat_target, chunk_offsets, chunk_shape_tuple, chunk_valid_shape_tuple},
                       {{"target_memory", MemorySpace::Vec}}, span),
            span);
        // The ragged TLOAD carries a dynamic valid_shape. Fill its padding
        // with zero and promote the accumulator back to the fixed physical
        // chunk type before it becomes a loop-carried / if-result tile.
        // Otherwise memory allocation may hoist an alloc_tile whose
        // valid_col still depends on the chunk loop variable, violating SSA
        // dominance in the generated PTO.
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
                            {flat_target, peer, chunk_offsets, chunk_shape_tuple, remote_valid_shape_tuple},
                            remote_load_kwargs, span),
                        span);
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
                    // Bind the reduction result so codegen sees a named tile
                    // buffer to write into.
                    return then_body.Bind("acc_next", then_body.Reduce(reduce_op, acc, recv, span), span);
                  },
                  [&](LoweringBuilder& /*else_body*/) -> ExprPtr { return acc; }, span);
            },
            span);

        // The signal cell sits at ready_generation after the ready barrier.
        // Each completed chunk adds one, so chunk k waits for
        // ready_generation + 1 + k.
        auto chunk_base = std::make_shared<ConstInt>(ready_generation + 1, DataType::INDEX, span);
        auto chunk_id = MakeFloorDiv(col, chunk_cols, span);
        auto expected_idx = MakeAdd(chunk_id, chunk_base, span);
        auto expected_i32 = chunk_body.Bind(
            "chunk_expected", std::make_shared<ir::Cast>(expected_idx, DataType::INT32, span), span);
        chunk_body.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32,
                                 "_chunk", span);
        chunk_body.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, expected_i32, "_chunk", span);

        // Accumulation deliberately uses the fixed physical chunk type. Narrow
        // the final alias back to the real tail before store so the last chunk
        // cannot write beyond the logical tensor extent.
        auto store_value = chunk_body.Bind(
            "store_value", reg.Create("tile.set_validshape", {acc_final, one_idx, valid_cols}, {}, span),
            span);
        chunk_body.Bind("store_ret",
                        reg.Create("tile.store", {store_value, chunk_offsets, flat_target}, {}, span), span);
      },
      span);

  // Self-clearing epilogue: this call issued ready_generation (1) + chunk_count
  // credits per peer — one for the ready barrier, one per completed chunk.
  // chunk_count = ceil(valid_cols / chunk_cols); chunk_cols is always a
  // ConstInt, but valid_cols (the reduced extent) may be a runtime scalar, so
  // build the total as an IR expression rather than requiring it statically —
  // pld.system.notify's value only needs ScalarType, so a symbolic total is
  // legal here.
  auto chunk_cols_minus_one = MakeSub(chunk_cols, one_idx, span);
  auto chunk_count_idx =
      MakeFloorDiv(MakeAdd(flat_valid_shape[1], chunk_cols_minus_one, span), chunk_cols, span);
  auto total_idx =
      MakeAdd(std::make_shared<ConstInt>(ready_generation, DataType::INDEX, span), chunk_count_idx, span);
  auto total_i32 =
      b.Bind("allreduce_reset_total", std::make_shared<ir::Cast>(total_idx, DataType::INT32, span), span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // In-place semantics: the rebind LHS receives the (post-reduce) target view.
  return target;
}

// ============================================================================
// ``pld.tensor.allreduce`` ring lowering rule (mode="ring")
//
// NCCL-style reduce-scatter + allgather ring schedule with 2(P−1) rounds.
// Signal shape is [2*(NR−1), NR] — one row per ring round, one cell per rank.
// Each UB-sized subchunk advances its round row through a ready barrier and a
// read-complete barrier before store-back.
//
// The ring reinterprets any packed ND target as one [1, SIZE] linear stream.
// FP32 uses balanced floor(i*SIZE/NR) boundaries. FP16 aligns every non-empty
// segment start and remote span to 32 bytes, while valid_shape narrows each
// ragged logical tail. Both preserve arbitrary lengths, including SIZE < NR.
//
// Hand-rolled reference: tests/st/distributed/collectives/test_l3_allreduce_ring.py
// Runtime reference:     runtime/examples/workers/l3/allreduce_ring_distributed/
// ============================================================================

namespace {
ExprPtr LowerTensorRingAllReduceRule(const CallPtr& call, const std::vector<ExprPtr>& args,
                                     LoweringBuilder& b) {
  const Span& span = call->span_;
  CHECK_SPAN(args.size() == 2, span) << "pld.tensor.allreduce mode=ring requires an explicit signal. "
                                        "Use pld.tensor.allreduce(target, signal, mode=\"ring\")";
  const auto& target = args[0];
  const auto& signal = args[1];
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.allreduce target must be DistributedTensorType (deducer-rejected otherwise)";
  auto op_value = GetRequiredKwarg<int>(call->kwargs_, "op", "pld.tensor.allreduce");
  INTERNAL_CHECK_SPAN(
      op_value >= static_cast<int>(ReduceOp::kSum) && op_value <= static_cast<int>(ReduceOp::kProd), span)
      << "pld.tensor.allreduce mode=ring received unknown ReduceOp " << op_value;
  const auto reduce_op = static_cast<ReduceOp>(op_value);

  // Signal validation: the signal is user-supplied via its DSL type
  // annotation, so a wrong shape/dtype is a user error — use CHECK_SPAN.
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  CHECK_SPAN(signal_type, span) << "mode=ring signal must be a DistributedTensor";
  CHECK_SPAN(signal_type->shape_.size() == 2, span) << "mode=ring signal must be 2D [2*(NR-1), NR]";
  CHECK_SPAN(signal_type->dtype_ == DataType::INT32, span) << "mode=ring signal must be INT32";

  // Cross-check signal dimensions for self-consistency when they are
  // compile-time constants.  A signal built with mismatched shape[0] and
  // shape[1] — e.g. annotation [3*(NR-1), NR] instead of [2*(NR-1), NR]
  // — would silently produce wrong round counts or out-of-range barrier
  // row indexing at runtime.  Skip when either dimension is dynamic.
  auto sig_shape0_const = As<ConstInt>(signal_type->shape_[0]);
  auto sig_shape1_const = As<ConstInt>(signal_type->shape_[1]);
  if (sig_shape0_const && sig_shape1_const && sig_shape1_const->value_ > 0) {
    CHECK_SPAN(sig_shape0_const->value_ == 2 * (sig_shape1_const->value_ - 1), span)
        << "pld.tensor.allreduce mode=ring signal shape[0] (" << sig_shape0_const->value_
        << ") must equal 2*(NR-1) = " << 2 * (sig_shape1_const->value_ - 1)
        << " for NR = " << sig_shape1_const->value_;
  }

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  auto two_idx = std::make_shared<ConstInt>(2, DataType::INDEX, span);
  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);

  // Cast my_rank to INDEX for modulo arithmetic.
  auto my_rank_idx =
      b.Bind("my_rank_idx", std::make_shared<ir::Cast>(comm.my_rank, DataType::INDEX, span), span);

  // Ring communication is linear. Reinterpret any packed ND target as one
  // contiguous [1, N] stream, matching the fully-valid mesh path. A contiguous
  // partial prefix keeps the source's full physical extent and carries its
  // flattened logical extent as tensor.view valid_shape. This avoids emitting
  // tensor.slice after ConvertTensorToTileOps has already run.
  const auto* partial_valid_shape = GetPartialValidShape(target_type, span);
  auto flat_shape = CollapseShapeToLinear2D(target_type->shape_, span);
  auto flat_valid_shape = flat_shape;
  if (partial_valid_shape != nullptr) {
    CHECK_SPAN(IsRowMajorLinearPrefix(*partial_valid_shape, target_type->shape_), span)
        << "pld.tensor.allreduce mode=ring target valid_shape must be a contiguous row-major prefix";
    flat_valid_shape = CollapseShapeToLinear2D(*partial_valid_shape, span);
  }

  auto size_expr = flat_valid_shape[1];
  auto nr_expr = signal_type->shape_[1];
  auto size_const = As<ConstInt>(size_expr);
  auto nr_const = As<ConstInt>(nr_expr);

  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.allreduce mode=ring");
  const int64_t alignment_elements = chunk_geometry.alignment_elements;
  auto alignment_elements_idx = chunk_geometry.alignment_elements_idx;
  auto alignment_minus_one_idx = chunk_geometry.alignment_minus_one_idx;

  // FP32 keeps balanced floor(i * SIZE / NR) boundaries. FP16 rounds each
  // interior boundary up to a 32-byte position so every non-empty segment
  // starts at an MTE-safe address. Rounding can enlarge one segment by at most
  // alignment_elements - 1, which is reflected in the common loop bound.
  ExprPtr max_segment_cols;
  if (size_const && nr_const && nr_const->value_ > 0) {
    int64_t max_segment = (size_const->value_ + nr_const->value_ - 1) / nr_const->value_;
    if (target_type->dtype_ == DataType::FP16) {
      max_segment = std::min(size_const->value_, max_segment + alignment_elements - 1);
    }
    max_segment_cols = std::make_shared<ConstInt>(max_segment, DataType::INDEX, span);
  } else {
    max_segment_cols = MakeFloorDiv(MakeAdd(size_expr, MakeSub(nr_expr, one_idx, span), span), nr_expr, span);
    if (target_type->dtype_ == DataType::FP16) {
      max_segment_cols = MakeMin(size_expr, MakeAdd(max_segment_cols, alignment_minus_one_idx, span), span);
    }
  }

  ExprPtr chunk_cols = SelectStaticChunkCols(chunk_geometry, max_segment_cols, span);
  auto chunk_shape = tile_conversion_utils::MakeShapeTuple({one_idx, chunk_cols}, span);
  // Own a single explicit linear ND view for every subchunk. Besides making
  // the [1, 1] column-vector exception unambiguous, this keeps the remote-load,
  // local-load, and store aliases identical throughout the ring pipeline.
  auto ring_target = b.Bind(
      "target_2d", CreateAllReduceTargetView(target, flat_shape, flat_valid_shape, partial_valid_shape, span),
      span);
  // VEC staging tile for the TPUT push path (mirrors the allgather / all_to_all
  // host-builtin recipe).  pto-isa TPUT streams the transfer through this tile,
  // clamping the last chunk to the transfer extent, and the single-shot path
  // reads exactly the tile's valid mask — so each pld.tile.put below narrows it
  // to the subchunk's valid_cols via tile.set_validshape (the same mechanism as
  // the host builtin's ColMaskInternal).  The stage is shared across all ring
  // pushes; it is mutable per-use kernel state.
  auto put_stage =
      b.Bind("ring_put_stage",
             reg.Create("tile.create", {chunk_shape},
                        {{"dtype", target_type->dtype_}, {"target_memory", MemorySpace::Vec}}, span),
             span);
  // Value-producing IfExpr branches must agree on a fixed TileType. For an
  // inactive logical segment, read one in-bounds element and pad it to the
  // physical chunk shape. Using tile.create here would survive the default
  // pipeline as tensor.alloc, which has no kernel codegen.
  auto placeholder_offsets = tile_conversion_utils::MakeShapeTuple({zero_idx, zero_idx}, span);

  auto segment_boundary = [&](const ExprPtr& boundary_idx) {
    auto scaled_size = MakeMul(boundary_idx, size_expr, span);
    if (target_type->dtype_ != DataType::FP16) {
      return MakeFloorDiv(scaled_size, nr_expr, span);
    }
    auto aligned_denominator = MakeMul(nr_expr, alignment_elements_idx, span);
    auto aligned_boundary =
        MakeMul(MakeFloorDiv(MakeAdd(scaled_size, MakeSub(aligned_denominator, one_idx, span), span),
                             aligned_denominator, span),
                alignment_elements_idx, span);
    return MakeMin(aligned_boundary, size_expr, span);
  };
  auto segment_begin = [&](const ExprPtr& segment_idx) { return segment_boundary(segment_idx); };
  auto segment_end = [&](const ExprPtr& segment_idx) {
    return segment_boundary(MakeAdd(segment_idx, one_idx, span));
  };
  auto emit_barrier = [&](LoweringBuilder& body, const ExprPtr& round, const ExprPtr& expected,
                          const std::string& suffix) {
    body.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, round, NotifyOp::kAtomicAdd, one_i32, suffix,
                       span);
    body.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, round, expected, suffix, span);
  };

  // nr_minus_one = NR − 1 (loop bound, 0..NR-2 inclusive → P−1 steps)
  auto nr_minus_one = b.Bind("nr_minus_one", MakeSub(comm.nranks_idx, one_idx, span), span);

  // ------------------------------------------------------------------
  // Phase 1: Reduce-Scatter — P−1 ring steps (TPUT push + local reduce)
  // ------------------------------------------------------------------
  //
  // Push-model schedule (non-atomic TPUT + local reduce — preserves
  // Sum/Max/Min/Prod; the ReduceOp trade-off is an explicit design choice,
  // NOT a silent Sum-only regression):
  //
  //   * recv_idx = (my_rank − step − 1 + NR) % NR is the chunk this rank
  //     receives from the left neighbour; send_idx = (my_rank − step + NR) % NR
  //     is the chunk this rank pushes to the right neighbour.
  //   * Each rank first reads its OWN value of slot recv_idx into a register
  //     tile (the slot is stable — own value only — until the left neighbour's
  //     push lands).
  //   * Ready barrier (generation 2k+1): all ranks' own-value reads are done
  //     and no push has landed — this is what makes the own-read race-free.
  //   * Each rank then TPUTs (pld.tile.put, AtomicType::kNone) its current
  //     partial of slot send_idx into the RIGHT neighbour's slot of the same
  //     index.  The transfer extent is the exact (possibly ragged) valid_cols;
  //     the staging tile is narrowed to it via tile.set_validshape so the TPUT
  //     single-shot path reads exactly the transfer width (PTOAS >= v0.55
  //     accepts dynamic partition-view shapes — issue #1069).
  //   * Push-done barrier (generation 2k+2): all pushes have landed.
  //   * Each rank reads its own slot recv_idx again (now holding the left
  //     neighbour's partial), reduces it with the saved own value, and stores
  //     the result back.  The push is a plain copy into the receiver's own
  //     window — no remote read, so the only cross-rank visibility needed is
  //     the TPUT→notify ordering the credit-barrier protocol provides
  //     (pld.tile.put codegen emits pipe_barrier(PIPE_ALL) around the TPUT and
  //     the comm-fence pass orders it ahead of the notify).
  b.EmitFor(
      "rs_step", zero_idx, nr_minus_one, one_idx,
      [&](LoweringBuilder& body, const VarPtr& rs_step_var) {
        auto step = body.Bind("step", MakeAdd(rs_step_var, one_idx, span), span);

        // recv_idx = (my_rank − step − 1 + NR) % NR
        auto r1 = MakeSub(my_rank_idx, step, span);
        auto r2 = MakeSub(r1, one_idx, span);
        auto r3 = MakeAdd(r2, comm.nranks_idx, span);
        auto recv_idx = body.Bind("recv_idx", MakeFloorMod(r3, comm.nranks_idx, span), span);

        // send_idx = (my_rank − step + NR) % NR — one chunk ahead of recv_idx.
        auto s1 = MakeSub(my_rank_idx, step, span);
        auto s2 = MakeAdd(s1, comm.nranks_idx, span);
        auto send_idx = body.Bind("send_idx", MakeFloorMod(s2, comm.nranks_idx, span), span);

        // right = (my_rank + 1) % NR — the push destination.
        auto rr1 = MakeAdd(my_rank_idx, one_idx, span);
        auto right_peer = body.Bind("right", MakeFloorMod(rr1, comm.nranks_idx, span), span);

        auto recv_segment_offset = body.Bind("rs_recv_segment_begin", segment_begin(recv_idx), span);
        auto recv_segment_limit = body.Bind("rs_recv_segment_end", segment_end(recv_idx), span);
        auto recv_segment_cols =
            body.Bind("rs_recv_segment_cols", MakeSub(recv_segment_limit, recv_segment_offset, span), span);

        auto send_segment_offset = body.Bind("rs_send_segment_begin", segment_begin(send_idx), span);
        auto send_segment_limit = body.Bind("rs_send_segment_end", segment_end(send_idx), span);
        auto send_segment_cols =
            body.Bind("rs_send_segment_cols", MakeSub(send_segment_limit, send_segment_offset, span), span);

        body.EmitFor(
            "rs_col", zero_idx, max_segment_cols, chunk_cols,
            [&](LoweringBuilder& chunk_body, const VarPtr& subcol) {
              // Receive-side extent (own-value read, pushed-partial read,
              // reduce, store all operate on slot recv_idx).
              auto recv_active = MakeLt(subcol, recv_segment_cols, span);
              auto recv_remaining = MakeSub(recv_segment_cols, subcol, span);
              auto recv_valid_cols = MakeMin(chunk_cols, recv_remaining, span);
              // Keep the value-producing IfExpr branch metadata identical.
              // Inactive ranks use one safe element, while active ranks retain
              // the exact logical tail extent.
              auto load_valid_cols = MakeMax(one_idx, recv_valid_cols, span);
              auto load_valid_shape = tile_conversion_utils::MakeShapeTuple({one_idx, load_valid_cols}, span);
              auto recv_offsets = tile_conversion_utils::MakeShapeTuple(
                  {zero_idx, MakeAdd(recv_segment_offset, subcol, span)}, span);

              // Send-side extent (TPUT source = slot send_idx).
              auto send_active = MakeLt(subcol, send_segment_cols, span);
              auto send_remaining = MakeSub(send_segment_cols, subcol, span);
              auto send_valid_cols = MakeMin(chunk_cols, send_remaining, span);
              auto send_valid_shape = tile_conversion_utils::MakeShapeTuple({one_idx, send_valid_cols}, span);
              auto send_offsets = tile_conversion_utils::MakeShapeTuple(
                  {zero_idx, MakeAdd(send_segment_offset, subcol, span)}, span);

              auto chunk_id = MakeFloorDiv(subcol, chunk_cols, span);

              // ---- Own-value read: BEFORE the ready barrier, so no push has
              // landed in slot recv_idx yet (the slot is stable — own value).
              auto acc_own = chunk_body.EmitIfExpr(
                  recv_active,
                  [&](LoweringBuilder& then_body) {
                    auto acc_loaded = then_body.Bind(
                        "acc_rs_own_loaded",
                        reg.Create("tile.load", {ring_target, recv_offsets, chunk_shape, load_valid_shape},
                                   {{"target_memory", MemorySpace::Vec}}, span),
                        span);
                    return then_body.Bind("acc_rs_own",
                                          reg.Create("tile.fillpad_inplace", {acc_loaded},
                                                     {{"pad_value", PadValue::zero}}, span),
                                          span);
                  },
                  [&](LoweringBuilder& else_body) {
                    auto placeholder_loaded = else_body.Bind(
                        "acc_rs_own_placeholder_loaded",
                        reg.Create("tile.load",
                                   {ring_target, placeholder_offsets, chunk_shape, load_valid_shape},
                                   {{"target_memory", MemorySpace::Vec}}, span),
                        span);
                    return else_body.Bind("acc_rs_own_placeholder",
                                          reg.Create("tile.fillpad_inplace", {placeholder_loaded},
                                                     {{"pad_value", PadValue::zero}}, span),
                                          span);
                  },
                  span);

              // ---- Ready barrier: all ranks' own-value reads are done; no
              // push can land before this barrier completes.
              auto ready_epoch_idx = MakeAdd(MakeMul(chunk_id, two_idx, span), one_idx, span);
              auto ready_epoch = chunk_body.Bind(
                  "rs_ready_epoch", std::make_shared<ir::Cast>(ready_epoch_idx, DataType::INT32, span), span);
              emit_barrier(chunk_body, rs_step_var, ready_epoch, "_rs_ready");

              // ---- Push phase: TPUT slot send_idx into the right neighbour's
              // slot of the same index (plain copy; the receiver reduces
              // locally after the push-done barrier).  The staging tile is
              // narrowed to the transfer width so the TPUT single-shot path
              // reads exactly send_valid_cols.
              chunk_body.EmitIf(
                  send_active,
                  [&](LoweringBuilder& push_body) {
                    auto rs_stage_valid = push_body.Bind(
                        "rs_stage_valid",
                        reg.Create("tile.set_validshape", {put_stage, one_idx, send_valid_cols}, {}, span),
                        span);
                    push_body.Bind("push_rs",
                                   reg.Create("pld.tile.put",
                                              {ring_target, right_peer, ring_target, rs_stage_valid,
                                               send_offsets, send_offsets, send_valid_shape},
                                              {{"atomic", static_cast<int>(AtomicType::kNone)}}, span),
                                   span);
                  },
                  /*else_fn=*/nullptr, span);

              // ---- Push-done barrier: all pushes have landed — slot recv_idx
              // now holds the left neighbour's partial.
              auto read_epoch_idx = MakeAdd(ready_epoch_idx, one_idx, span);
              auto read_epoch = chunk_body.Bind(
                  "rs_read_epoch", std::make_shared<ir::Cast>(read_epoch_idx, DataType::INT32, span), span);
              emit_barrier(chunk_body, rs_step_var, read_epoch, "_rs_read");

              // ---- Local reduce + store: read the pushed partial (own slot
              // recv_idx, now = left's partial), combine with the saved own
              // value, store back.
              auto acc_full = chunk_body.EmitIfExpr(
                  recv_active,
                  [&](LoweringBuilder& then_body) {
                    auto recv_loaded = then_body.Bind(
                        "recv_rs_loaded",
                        reg.Create("tile.load", {ring_target, recv_offsets, chunk_shape, load_valid_shape},
                                   {{"target_memory", MemorySpace::Vec}}, span),
                        span);
                    auto recv = then_body.Bind("recv_rs",
                                               reg.Create("tile.fillpad_inplace", {recv_loaded},
                                                          {{"pad_value", PadValue::zero}}, span),
                                               span);
                    return then_body.Bind("acc_rs_next", then_body.Reduce(reduce_op, acc_own, recv, span),
                                          span);
                  },
                  [&](LoweringBuilder& else_body) {
                    auto placeholder_loaded = else_body.Bind(
                        "acc_rs_placeholder_loaded",
                        reg.Create("tile.load",
                                   {ring_target, placeholder_offsets, chunk_shape, load_valid_shape},
                                   {{"target_memory", MemorySpace::Vec}}, span),
                        span);
                    return else_body.Bind("acc_rs_placeholder",
                                          reg.Create("tile.fillpad_inplace", {placeholder_loaded},
                                                     {{"pad_value", PadValue::zero}}, span),
                                          span);
                  },
                  span);

              chunk_body.EmitIf(
                  recv_active,
                  [&](LoweringBuilder& store_body) {
                    // Encode the active-branch bounds in the store operands so
                    // valid-region inference can prove this write stays inside
                    // the flattened logical extent without relying on control
                    // flow predicates.
                    auto raw_store_col = MakeAdd(recv_segment_offset, subcol, span);
                    auto store_col = MakeSub(
                        size_expr, MakeMax(zero_idx, MakeSub(size_expr, raw_store_col, span), span), span);
                    auto raw_store_end = MakeAdd(store_col, recv_valid_cols, span);
                    auto store_end = MakeSub(
                        size_expr, MakeMax(zero_idx, MakeSub(size_expr, raw_store_end, span), span), span);
                    auto store_valid_cols = MakeSub(store_end, store_col, span);
                    auto store_offsets = tile_conversion_utils::MakeShapeTuple({zero_idx, store_col}, span);
                    auto narrowed = store_body.Bind(
                        "acc_rs_valid",
                        reg.Create("tile.set_validshape", {acc_full, one_idx, store_valid_cols}, {}, span),
                        span);
                    store_body.Bind(
                        "store_rs",
                        reg.Create("tile.store", {narrowed, store_offsets, ring_target}, {}, span), span);
                  },
                  /*else_fn=*/nullptr, span);
            },
            span);
      },
      span);

  // ------------------------------------------------------------------
  // Phase 2: AllGather — P−1 ring steps (TPUT push, non-atomic copy)
  // ------------------------------------------------------------------
  //
  // Each rank TPUTs its finalized chunk send_idx = (my_rank − step + 1 + NR)
  // % NR into the RIGHT neighbour's slot of the same index — a plain copy,
  // since the chunk value is already fully reduced.  The receiver does
  // nothing locally; the push performs the store.  The transfer extent is the
  // exact (possibly ragged) valid_cols with the staging tile narrowed to it
  // (PTOAS >= v0.55 dynamic partition-view shapes).  The ready barrier
  // (generation 2k+1) guarantees the slot being forwarded holds valid data —
  // not because the counters carry credit across rounds (each round uses its
  // own signal row, so row k starts at zero), but because every rank must pass
  // the previous round's push-done barrier before it enters round k.  The
  // push-done barrier (generation 2k+2) then guarantees the copy is visible
  // before any rank reads it at the next step.
  b.EmitFor(
      "ag_step", zero_idx, nr_minus_one, one_idx,
      [&](LoweringBuilder& body, const VarPtr& ag_step_var) {
        auto step = body.Bind("ag_step_val", MakeAdd(ag_step_var, one_idx, span), span);
        auto ag_round = body.Bind("ag_round", MakeAdd(ag_step_var, nr_minus_one, span), span);

        // send_idx = (my_rank − step + 1 + NR) % NR — the fully-reduced chunk
        // this rank forwards to the right neighbour.
        auto r1 = MakeSub(my_rank_idx, step, span);
        auto r2 = MakeAdd(r1, one_idx, span);
        auto r3 = MakeAdd(r2, comm.nranks_idx, span);
        auto send_idx = body.Bind("ag_send_idx", MakeFloorMod(r3, comm.nranks_idx, span), span);

        // right = (my_rank + 1) % NR — the push destination.
        auto rr1 = MakeAdd(my_rank_idx, one_idx, span);
        auto right_peer = body.Bind("ag_right", MakeFloorMod(rr1, comm.nranks_idx, span), span);

        auto segment_offset = body.Bind("ag_segment_begin", segment_begin(send_idx), span);
        auto segment_limit = body.Bind("ag_segment_end", segment_end(send_idx), span);
        auto segment_cols = body.Bind("ag_segment_cols", MakeSub(segment_limit, segment_offset, span), span);

        body.EmitFor(
            "ag_col", zero_idx, max_segment_cols, chunk_cols,
            [&](LoweringBuilder& chunk_body, const VarPtr& subcol) {
              auto active = MakeLt(subcol, segment_cols, span);
              auto remaining = MakeSub(segment_cols, subcol, span);
              auto valid_cols = MakeMin(chunk_cols, remaining, span);
              auto valid_shape = tile_conversion_utils::MakeShapeTuple({one_idx, valid_cols}, span);
              auto offsets = tile_conversion_utils::MakeShapeTuple(
                  {zero_idx, MakeAdd(segment_offset, subcol, span)}, span);

              auto chunk_id = MakeFloorDiv(subcol, chunk_cols, span);

              // ---- Ready barrier: every rank passes the previous round's
              // push-done barrier before entering this round, so the round-k-1
              // push into the slot being forwarded has landed and is visible —
              // the slot holds valid data.  (Counters are per-round: row k
              // starts at zero and carries no credit from row k-1.)
              auto ready_epoch_idx = MakeAdd(MakeMul(chunk_id, two_idx, span), one_idx, span);
              auto ready_epoch = chunk_body.Bind(
                  "ag_ready_epoch", std::make_shared<ir::Cast>(ready_epoch_idx, DataType::INT32, span), span);
              emit_barrier(chunk_body, ag_round, ready_epoch, "_ag_ready");

              // ---- Push phase: TPUT slot send_idx into the right neighbour's
              // slot of the same index (plain copy).  The staging tile is
              // narrowed to the transfer width so the TPUT single-shot path
              // reads exactly valid_cols.
              chunk_body.EmitIf(
                  active,
                  [&](LoweringBuilder& push_body) {
                    auto ag_stage_valid = push_body.Bind(
                        "ag_stage_valid",
                        reg.Create("tile.set_validshape", {put_stage, one_idx, valid_cols}, {}, span), span);
                    push_body.Bind("push_ag",
                                   reg.Create("pld.tile.put",
                                              {ring_target, right_peer, ring_target, ag_stage_valid, offsets,
                                               offsets, valid_shape},
                                              {{"atomic", static_cast<int>(AtomicType::kNone)}}, span),
                                   span);
                  },
                  /*else_fn=*/nullptr, span);

              // ---- Push-done barrier: all pushes have landed — every rank's
              // forwarded slot is visible before the next step reads it.
              auto read_epoch_idx = MakeAdd(ready_epoch_idx, one_idx, span);
              auto read_epoch = chunk_body.Bind(
                  "ag_read_epoch", std::make_shared<ir::Cast>(read_epoch_idx, DataType::INT32, span), span);
              emit_barrier(chunk_body, ag_round, read_epoch, "_ag_read");
            },
            span);
      },
      span);

  // Self-clearing epilogue: every row (round) of this call issued
  // 2 * chunk_count credits per peer — a ready + read-complete barrier for
  // every subchunk. chunk_count = ceil(max_segment_cols / chunk_cols) is
  // uniform across every row (every round's sub-chunk loop shares this same
  // bound), so one symbolic total resets every row of the [2*(NR-1), NR]
  // signal.
  auto chunk_cols_minus_one = MakeSub(chunk_cols, one_idx, span);
  auto chunk_count_idx =
      MakeFloorDiv(MakeAdd(max_segment_cols, chunk_cols_minus_one, span), chunk_cols, span);
  auto total_per_row_idx = MakeMul(two_idx, chunk_count_idx, span);
  auto total_per_row_i32 =
      b.Bind("ring_reset_total", std::make_shared<ir::Cast>(total_per_row_idx, DataType::INT32, span), span);
  b.EmitEpilogueReset(signal, comm, signal_type->shape_[0], total_per_row_i32, span);

  return target;
}

}  // namespace

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
