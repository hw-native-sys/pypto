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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// CommSetup — result struct for LoweringBuilder::EmitCommSetup()
// ============================================================================

/// Holds bound expressions from the comm-setup preamble (ctx, nranks, my_rank).
/// Returned by LoweringBuilder::EmitCommSetup() for use in subsequent phases.
struct CommSetup {
  ExprPtr ctx;         ///< Result of pld.system.get_comm_ctx
  ExprPtr nranks_i32;  ///< Result of pld.system.nranks (INT32)
  ExprPtr nranks_idx;  ///< nranks cast to INDEX (for loop bounds)
  ExprPtr my_rank;     ///< Result of pld.system.rank (INT32)
};

/// Safe negation: folds ConstInt(-value) directly when possible so the
/// PyPTO printer→parser roundtrip (which folds ``Neg(ConstInt)`` into
/// ``ConstInt(-value)``) produces structurally equal IR. Runtime
/// expressions are still wrapped with ``Neg``.
inline ExprPtr MakeNegation(const ExprPtr& value) {
  if (auto c = As<ConstInt>(value)) {
    return std::make_shared<ConstInt>(-c->value_, GetScalarDtype(value), value->span_);
  }
  return MakeNeg(value, value->span_);
}

// ============================================================================
// Self-clearing credit barrier — the shared, stateless barrier-signal protocol
// ============================================================================
//
// Every ``pld.tensor.*`` collective synchronises through one protocol:
//
//     Body:      barrier(1); barrier(2); ...; barrier(N)   # g counted within
//                                                           # this call only
//       barrier(g):
//         for peer != my_rank: notify(signal, peer, <my cell>, 1, op=AtomicAdd)
//         for src  != my_rank: wait  (signal, <src cell>, g,   cmp=Ge)
//
//     Epilogue:  for src != my_rank:
//                    notify(signal, my_rank, <src cell>, -N, op=AtomicAdd)
//
// ``AtomicAdd`` turns each cell into a credit counter: every notify is a
// producer's ``+1``, and the epilogue is the sole consumer's ``-N``. Because
// adds and subtracts are atomic and commutative, the signal is provably
// all-zero again once every rank has finished its epilogue for a call — the
// signal carries no state that outlives one call, so every call's ``g`` restarts
// at 1 and no cross-call bookkeeping is needed. A slow rank can inflate a fast
// rank's own next-call credit by at most 1 while it finishes the current call
// (bounded skew), so the counter never overflows and a fast rank can never
// observe a spurious pass.
//
// ``kGe`` (not ``kEq``) is load-bearing: a fast peer can advance a cell past the
// value the waiting rank is looking for before that rank ever polls it, so an
// equality wait would never unblock. For the same reason ``kSet`` must never be
// mixed with ``kAtomicAdd`` on the same cells — a set could clobber an already
// advanced counter.
//
// Because the protocol is call-local, the mesh (``[NR, 1]``, one cell per rank)
// and ring (``[2*(NR-1), NR]``, one row per round) signal shapes are the only
// remaining incompatibility between collectives sharing one signal buffer — see
// ``ValidateMeshSignalShape``. There is no compile-time generation ledger to
// poison, so collectives are legal inside ``for``/``while``/``if``, and a mesh
// allreduce's per-chunk credit total may be a runtime-computed scalar rather
// than a compile-time constant.
// ============================================================================

std::vector<ExprPtr> CollapseShapeTo2D(const std::vector<ExprPtr>& shape, const Span& span) {
  INTERNAL_CHECK_SPAN(!shape.empty(), span) << "Cannot flatten a rank-0 tensor shape";
  if (shape.size() == 1) {
    return {std::make_shared<ConstInt>(1, DataType::INDEX, span), shape[0]};
  }
  if (shape.size() == 2) return shape;

  ExprPtr rows = shape[0];
  for (size_t i = 1; i + 1 < shape.size(); ++i) {
    rows = tile_conversion_utils::MakeCanonicalIndexMul(rows, shape[i], span, "LowerCompositeOps");
  }
  return {rows, shape.back()};
}

std::vector<ExprPtr> CollapseShapeToLinear2D(const std::vector<ExprPtr>& shape, const Span& span) {
  INTERNAL_CHECK_SPAN(!shape.empty(), span) << "Cannot flatten a rank-0 tensor shape";
  ExprPtr elements = shape[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    elements = tile_conversion_utils::MakeCanonicalIndexMul(elements, shape[i], span, "LowerCompositeOps");
  }
  return {std::make_shared<ConstInt>(1, DataType::INDEX, span), elements};
}

// The current memory planner assigns distinct storage to the loop-carried
// accumulator and the branch/yield SSA aliases, so lowering can account for
// up to nine physical tile buffers before reuse. At the maximum chunk width,
// nine 16-KiB tiles stay below the smallest supported 184-KiB VEC UB budget
// with room for scalar metadata; statically smaller inputs shrink this width.
constexpr int64_t kAllReduceChunkBytes = 16LL * 1024;
constexpr int64_t kPTOTileAlignmentBytes = 32;

void CheckAllReduceTargetIsPackedNd(const DistributedTensorTypePtr& target_type, const Span& span) {
  if (!target_type->tensor_view_.has_value()) return;

  const auto& view = target_type->tensor_view_.value();
  CHECK_SPAN(view.layout == TensorLayout::ND, span)
      << "pld.tensor.allreduce target view only supports ND layout";
  if (view.stride.empty()) return;

  const auto packed_strides =
      tensor_view_semantics::BuildLogicalStridesFromLayout(target_type->shape_, TensorLayout::ND);
  CHECK_SPAN(view.stride.size() == packed_strides.size(), span)
      << "pld.tensor.allreduce target shape reinterpret requires a packed source";
  for (size_t i = 0; i < view.stride.size(); ++i) {
    CHECK_SPAN(AreExprsEqual(view.stride[i], packed_strides[i]), span)
        << "pld.tensor.allreduce target shape reinterpret requires a packed source";
  }
}

bool IsRowMajorLinearPrefix(const std::vector<ExprPtr>& valid, const std::vector<ExprPtr>& physical) {
  if (valid.size() != physical.size()) return false;
  bool past_boundary = false;
  for (size_t i = 0; i < valid.size(); ++i) {
    const bool is_full = AreExprsEqual(valid[i], physical[i]);
    if (past_boundary) {
      if (!is_full) return false;
      continue;
    }
    auto valid_const = As<ConstInt>(valid[i]);
    if (!(valid_const && valid_const->value_ == 1)) past_boundary = true;
  }
  return true;
}

const std::vector<ExprPtr>* GetPartialValidShape(const DistributedTensorTypePtr& target_type,
                                                 const Span& span) {
  if (!target_type->tensor_view_.has_value() || target_type->tensor_view_->valid_shape.empty()) {
    return nullptr;
  }

  const auto& valid_shape = target_type->tensor_view_->valid_shape;
  CHECK_SPAN(valid_shape.size() == target_type->shape_.size(), span)
      << "pld.tensor.allreduce target valid_shape rank must match target rank";
  // TensorType canonicalization removes an explicit valid_shape that exactly
  // equals shape. Any remaining valid_shape is therefore a genuine partial
  // region and must stay on the rectangular path below.
  return &valid_shape;
}

CallPtr CreateAllReduceTargetView(const ExprPtr& target, const std::vector<ExprPtr>& flat_shape,
                                  const std::vector<ExprPtr>& flat_valid_shape,
                                  const std::vector<ExprPtr>* partial_valid_shape, const Span& span) {
  // Allreduce owns this alias and reduces only flat_valid_shape. Public
  // tensor.view cannot infer a shape reinterpretation for partial validity.
  auto shape_tuple = tile_conversion_utils::MakeShapeTuple(flat_shape, span);
  std::vector<ExprPtr> view_args{target, shape_tuple};
  if (partial_valid_shape != nullptr) {
    view_args.push_back(tile_conversion_utils::MakeShapeTuple(flat_valid_shape, span));
  }
  return OpRegistry::GetInstance().Create("tensor.view", view_args, {}, span);
}

/// Validates that ``signal_type`` matches the mesh barrier convention (one
/// cell per rank: ``[NR, 1]``). Ring allreduce's signal is ``[2*(NR-1), NR]``
/// instead — one row per round, addressed ``[row, rank]``. Sharing one buffer
/// between the two conventions no longer trips a generation-table state error
/// (the self-clearing protocol is call-local, so there is no cross-call state
/// to mismatch); this shape check is the sole remaining guard against a mesh
/// op silently targeting the wrong cell of a ring-shaped signal. Skip the
/// second-dimension check when it is symbolic, matching the ring rule's own
/// existing shape checks.
void ValidateMeshSignalShape(const DistributedTensorTypePtr& signal_type, const std::string& op_name,
                             const Span& span) {
  CHECK_SPAN(signal_type, span) << op_name << " signal must be a DistributedTensor";
  CHECK_SPAN(signal_type->shape_.size() == 2, span)
      << op_name << " signal must be 2D [NR, 1], got rank " << signal_type->shape_.size();
  if (auto col_dim = As<ConstInt>(signal_type->shape_[1])) {
    CHECK_SPAN(col_dim->value_ == 1, span)
        << op_name << " signal shape[1] must be 1 (one cell per rank), got " << col_dim->value_
        << " — a signal shaped for mode=\"ring\" allreduce ([2*(NR-1), NR]) cannot be shared with "
           "this collective; give it its own [NR, 1] signal window";
  }
}

// ============================================================================
// LoweringBuilder
//
// Per-call scratchpad handed to a composite-lowering rule. A rule appends one
// ``AssignStmt`` per intermediate temp via ``Bind`` and returns the final
// result ``ExprPtr``; the mutator wraps that result in the original target
// ``Var`` (or a fresh result ``Var`` for ``ReturnStmt`` calls) before splicing
// the accumulated statements into the surrounding sequence.
//
// In addition to ``Bind`` and the primitive op builders, the builder exposes
// structured control-flow constructors — ``EmitFor`` / ``EmitForReduce`` /
// ``EmitIf`` / ``EmitIfExpr`` — that hand the body off to a nested builder
// callback. The nested builder shares this builder's temp counter so every
// emitted temp gets a unique name across the entire rule, regardless of
// nesting depth.
//
// The temp counter is borrowed from the mutator so unique temp names span
// distinct composite-op calls in the same function. Barrier generations are
// call-local (see the self-clearing credit-barrier protocol above), so each
// LoweringBuilder instance — one per top-level composite-op call — owns its
// own ``barrier_count_`` that always starts at 0.
// ============================================================================
class LoweringBuilder {
 public:
  /// @param base_name    Name hint to derive temp names from (typically the
  ///                     AssignStmt's LHS ``Var`` name).
  /// @param temp_counter Reference to a mutator-owned counter; bumped per Bind.
  LoweringBuilder(std::string base_name, std::size_t& temp_counter)
      : base_name_(std::move(base_name)), temp_counter_(temp_counter) {}

  LoweringBuilder(std::string base_name, std::size_t& temp_counter, bool nested)
      : base_name_(std::move(base_name)), temp_counter_(temp_counter), nested_(nested) {}

  /// Append an ``AssignStmt`` binding a fresh ``Var`` to ``expr`` and return
  /// the new ``Var`` so it can be used as input to subsequent ops. The
  /// ``qualifier`` is woven into the temp name for debuggability.
  ExprPtr Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span) {
    auto var = std::make_shared<Var>(MakeTempName(qualifier), expr->GetType(), span);
    stmts_.push_back(std::make_shared<AssignStmt>(var, expr, span));
    return var;
  }

  /// Append a side-effecting expression without manufacturing an unused SSA
  /// result. This is used by destination-passing ops whose output buffers are
  /// explicit operands.
  void EmitEval(const ExprPtr& expr, const Span& span) {
    stmts_.push_back(std::make_shared<EvalStmt>(expr, span));
  }

  // Primitive op builders -- type deduction is delegated to OpRegistry so the
  // result preserves the input TileType's shape/layout/dtype.
  ExprPtr Muls(const ExprPtr& x, float c, const Span& span) {
    auto tile_type = As<TileType>(x->GetType());
    INTERNAL_CHECK_SPAN(tile_type, span) << "tile.muls input must be TileType";
    auto scalar = std::make_shared<ConstFloat>(static_cast<double>(c), tile_type->dtype_, span);
    return OpRegistry::GetInstance().Create("tile.muls", {x, scalar}, {}, span);
  }
  ExprPtr Adds(const ExprPtr& x, float c, const Span& span) {
    auto tile_type = As<TileType>(x->GetType());
    INTERNAL_CHECK_SPAN(tile_type, span) << "tile.adds input must be TileType";
    auto scalar = std::make_shared<ConstFloat>(static_cast<double>(c), tile_type->dtype_, span);
    return OpRegistry::GetInstance().Create("tile.adds", {x, scalar}, {}, span);
  }
  ExprPtr Add(const ExprPtr& a, const ExprPtr& b, const Span& span) {
    return OpRegistry::GetInstance().Create("tile.add", {a, b}, {}, span);
  }
  ExprPtr Sub(const ExprPtr& a, const ExprPtr& b, const Span& span) {
    return OpRegistry::GetInstance().Create("tile.sub", {a, b}, {}, span);
  }
  ExprPtr Mul(const ExprPtr& a, const ExprPtr& b, const Span& span) {
    return OpRegistry::GetInstance().Create("tile.mul", {a, b}, {}, span);
  }
  ExprPtr Reduce(ReduceOp op, const ExprPtr& a, const ExprPtr& b, const Span& span) {
    const char* op_name;
    switch (op) {
      case ReduceOp::kSum:
        op_name = "tile.add";
        break;
      case ReduceOp::kMax:
        op_name = "tile.maximum";
        break;
      case ReduceOp::kMin:
        op_name = "tile.minimum";
        break;
      case ReduceOp::kProd:
        op_name = "tile.mul";
        break;
      default:
        INTERNAL_CHECK_SPAN(false, span)
            << "pld.tensor.allreduce lowering received unknown ReduceOp " << static_cast<int>(op);
    }
    return OpRegistry::GetInstance().Create(op_name, {a, b}, {}, span);
  }
  ExprPtr Cast(const ExprPtr& x, DataType to, int mode, const Span& span) {
    std::vector<std::pair<std::string, std::any>> kw = {{"target_type", to}, {"mode", mode}};
    return OpRegistry::GetInstance().Create("tile.cast", {x}, kw, span);
  }

  // ---- Scalar comparison helpers (yield BOOL-typed expressions, suitable as
  //      IfStmt conditions or loop guards). Delegated to the scalar_expr
  //      Make* helpers so operand promotion stays consistent with parser
  //      output.
  ExprPtr NotEq(const ExprPtr& left, const ExprPtr& right, const Span& span) {
    return MakeNe(left, right, span);
  }

  ExprPtr Gt(const ExprPtr& left, const ExprPtr& right, const Span& span) {
    return MakeGt(left, right, span);
  }

  // ---- Collective-op helpers (DRY extraction for barrier/broadcast/allgather/
  //      reduce_scatter/allreduce) ----

  /// Emit comm-setup preamble: get_comm_ctx, nranks, rank.
  /// Returns a CommSetup struct with the bound expressions for use in
  /// subsequent phases.
  CommSetup EmitCommSetup(const ExprPtr& comm_target, const Span& span) {
    auto& reg = OpRegistry::GetInstance();
    CommSetup s;
    s.ctx = Bind("ctx", reg.Create("pld.system.get_comm_ctx", {comm_target}, {}, span), span);
    s.nranks_i32 = Bind("nranks", reg.Create("pld.system.nranks", {s.ctx}, {}, span), span);
    s.nranks_idx = Bind("nranks_idx", std::make_shared<ir::Cast>(s.nranks_i32, DataType::INDEX, span), span);
    s.my_rank = Bind("my_rank", reg.Create("pld.system.rank", {s.ctx}, {}, span), span);
    return s;
  }

  /// Emit notify-all loop: for peer in 0..nranks: if peer != my_rank: notify(...)
  /// @param signal       The signal DistributedTensor
  /// @param nranks_idx   Loop bound (INDEX-typed)
  /// @param my_rank      This rank's ID (INT32)
  /// @param notify_op    NotifyOp::kSet or NotifyOp::kAtomicAdd
  /// @param value        Value to notify (e.g., one_i32)
  /// @param suffix       Suffix for loop variable names (e.g., "" or "2" for re-notify)
  /// @param span         Source span for error reporting
  void EmitNotifyAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                     NotifyOp notify_op, const ExprPtr& value, const std::string& suffix, const Span& span) {
    auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    auto my_offsets = tile_conversion_utils::MakeSignalOffsets(my_rank, span);

    EmitFor(
        "peer" + suffix, zero_idx, nranks_idx, one_idx,
        [&](LoweringBuilder& body, const VarPtr& peer) {
          body.EmitIf(
              body.NotEq(peer, my_rank, span),
              [&](LoweringBuilder& then_body) {
                auto call =
                    OpRegistry::GetInstance().Create("pld.system.notify", {signal, peer, my_offsets, value},
                                                     {{"op", static_cast<int>(notify_op)}}, span);
                then_body.Bind("notify" + suffix + "_ret", call, span);
              },
              /*else_fn=*/nullptr, span);
        },
        span);
  }

  /// Overload for 2D signal matrices (e.g. ring allreduce [2*(NR-1), NR]).
  /// @param row_offset   Row index expression for the 2D signal (e.g. ring step var)
  void EmitNotifyAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                     const ExprPtr& row_offset, NotifyOp notify_op, const ExprPtr& value,
                     const std::string& suffix, const Span& span) {
    auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    auto my_offsets = tile_conversion_utils::MakeSignalOffsets(my_rank, row_offset, span);

    EmitFor(
        "peer" + suffix, zero_idx, nranks_idx, one_idx,
        [&](LoweringBuilder& body, const VarPtr& peer) {
          body.EmitIf(
              body.NotEq(peer, my_rank, span),
              [&](LoweringBuilder& then_body) {
                auto call =
                    OpRegistry::GetInstance().Create("pld.system.notify", {signal, peer, my_offsets, value},
                                                     {{"op", static_cast<int>(notify_op)}}, span);
                then_body.Bind("notify" + suffix + "_ret", call, span);
              },
              /*else_fn=*/nullptr, span);
        },
        span);
  }

  /// Emit wait-all loop: for src in 0..nranks: if src != my_rank: wait(...)
  /// @param signal       The signal DistributedTensor
  /// @param nranks_idx   Loop bound (INDEX-typed)
  /// @param my_rank      This rank's ID (INT32)
  /// @param expected     Expected signal value — the barrier generation (INT32)
  /// @param suffix       Suffix for loop variable names (e.g., "" or "2" for re-wait)
  /// @param span         Source span for error reporting
  void EmitWaitAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                   const ExprPtr& expected, const std::string& suffix, const Span& span) {
    auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

    EmitFor(
        "src" + suffix, zero_idx, nranks_idx, one_idx,
        [&](LoweringBuilder& body, const VarPtr& src) {
          auto src_offsets = tile_conversion_utils::MakeSignalOffsets(src, span);
          body.EmitIf(
              body.NotEq(src, my_rank, span),
              [&](LoweringBuilder& then_body) {
                auto call =
                    OpRegistry::GetInstance().Create("pld.system.wait", {signal, src_offsets, expected},
                                                     {{"cmp", static_cast<int>(WaitCmp::kGe)}}, span);
                then_body.Bind("wait" + suffix + "_ret", call, span);
              },
              /*else_fn=*/nullptr, span);
        },
        span);
  }

  /// Overload for 2D signal matrices (e.g. ring allreduce [2*(NR-1), NR]).
  /// @param row_offset   Row index expression for the 2D signal (e.g. ring step var)
  void EmitWaitAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                   const ExprPtr& row_offset, const ExprPtr& expected, const std::string& suffix,
                   const Span& span) {
    auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

    EmitFor(
        "src" + suffix, zero_idx, nranks_idx, one_idx,
        [&](LoweringBuilder& body, const VarPtr& src) {
          auto src_offsets = tile_conversion_utils::MakeSignalOffsets(src, row_offset, span);
          body.EmitIf(
              body.NotEq(src, my_rank, span),
              [&](LoweringBuilder& then_body) {
                auto call =
                    OpRegistry::GetInstance().Create("pld.system.wait", {signal, src_offsets, expected},
                                                     {{"cmp", static_cast<int>(WaitCmp::kGe)}}, span);
                then_body.Bind("wait" + suffix + "_ret", call, span);
              },
              /*else_fn=*/nullptr, span);
        },
        span);
  }

  // ---- Self-clearing credit barrier protocol (see the file-header comment) ----

  /// Emit one complete cross-rank barrier on ``signal``: ``AtomicAdd(1)`` into
  /// every peer's cell, then wait for this call's generation on every peer's
  /// cell. Returns the generation waited for (1-based, scoped to *this call*
  /// only — every fresh ``LoweringBuilder`` starts counting at 0), so a rule
  /// that fans out further barriers (the mesh allreduce's per-chunk barriers)
  /// can continue the sequence from it, and so the rule can compute the total
  /// credit count its ``EmitEpilogueReset`` call must subtract.
  ///
  /// Call this only from a rule's straight-line code — one call consumes exactly
  /// one generation, so invoking it inside an ``EmitFor`` body would reserve a
  /// single generation for a barrier that executes many times. Loop-resident
  /// barriers must emit notify/wait by hand with a call-local expected value
  /// (see the ring / mesh-chunked rules below).
  int64_t EmitBarrier(const ExprPtr& signal, const CommSetup& comm, const std::string& suffix,
                      const Span& span) {
    INTERNAL_CHECK_SPAN(!nested_, span)
        << "Internal error: EmitBarrier must only be called from a top-level lowering rule, not from inside "
        << "EmitFor / EmitIf / EmitIfExpr bodies. Loop- or condition-resident barriers must "
        << "emit notify/wait by hand with a call-local expected value.";
    const int64_t generation = ++barrier_count_;
    auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto expected_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
    EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32, suffix, span);
    EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, expected_i32, suffix, span);
    return generation;
  }

  /// Self-clearing epilogue: subtract ``total`` from every non-self peer's
  /// contribution to *my own* cells, restoring the signal to all-zero once
  /// every rank has run its own epilogue. ``total`` is the number of
  /// ``AtomicAdd(+1)`` notifies this call issued per peer (the sum of every
  /// ``EmitBarrier`` / hand-rolled notify-wait pair the rule emitted) — it may
  /// be a runtime-computed expression, not just a ``ConstInt``:
  /// ``pld.system.notify``'s value only requires ``ScalarType``.
  ///
  /// This is a self-notify (``peer == my_rank``): the codegen path resolves
  /// ``peer == my_rank`` via the same identity mapping ``pld.tile.put`` /
  /// ``pld.tile.get`` already rely on for their self-rank case, so this lands
  /// on the exact same hardware atomic as an incoming remote add.
  ///
  /// Call this exactly once per rule invocation, from top-level code only,
  /// after every barrier the rule issues.
  void EmitEpilogueReset(const ExprPtr& signal, const CommSetup& comm, const ExprPtr& total,
                         const Span& span) {
    INTERNAL_CHECK_SPAN(!nested_, span)
        << "EmitEpilogueReset must only be called from a top-level lowering rule, exactly once "
           "per call, after every EmitBarrier / hand-rolled notify-wait pair the rule issues.";
    auto neg_total = MakeNegation(total);
    auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    EmitFor(
        "reset_src", zero_idx, comm.nranks_idx, one_idx,
        [&](LoweringBuilder& body, const VarPtr& src) {
          body.EmitIf(
              body.NotEq(src, comm.my_rank, span),
              [&](LoweringBuilder& then_body) {
                auto src_offsets = tile_conversion_utils::MakeSignalOffsets(src, span);
                auto call = OpRegistry::GetInstance().Create(
                    "pld.system.notify", {signal, comm.my_rank, src_offsets, neg_total},
                    {{"op", static_cast<int>(NotifyOp::kAtomicAdd)}}, span);
                then_body.Bind("epilogue_reset_ret", call, span);
              },
              /*else_fn=*/nullptr, span);
        },
        span);
  }

  /// 2D-signal overload (ring allreduce, ``[2*(NR-1), NR]``): subtract
  /// ``total_per_row`` from every non-self cell of every one of ``num_rows``
  /// rows. Ring credits every row independently (one per round / sub-chunk
  /// sequence), and every row's sub-chunk loop shares the same bound, so one
  /// symbolic ``total_per_row`` resets all rows uniformly.
  void EmitEpilogueReset(const ExprPtr& signal, const CommSetup& comm, const ExprPtr& num_rows,
                         const ExprPtr& total_per_row, const Span& span) {
    INTERNAL_CHECK_SPAN(!nested_, span)
        << "EmitEpilogueReset must only be called from a top-level lowering rule, exactly once per call.";
    auto neg_total = MakeNegation(total_per_row);
    auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    EmitFor(
        "reset_row", zero_idx, num_rows, one_idx,
        [&](LoweringBuilder& row_body, const VarPtr& row) {
          row_body.EmitFor(
              "reset_src", zero_idx, comm.nranks_idx, one_idx,
              [&](LoweringBuilder& body, const VarPtr& src) {
                body.EmitIf(
                    body.NotEq(src, comm.my_rank, span),
                    [&](LoweringBuilder& then_body) {
                      auto src_offsets = tile_conversion_utils::MakeSignalOffsets(src, row, span);
                      auto call = OpRegistry::GetInstance().Create(
                          "pld.system.notify", {signal, comm.my_rank, src_offsets, neg_total},
                          {{"op", static_cast<int>(NotifyOp::kAtomicAdd)}}, span);
                      then_body.Bind("epilogue_reset_ret", call, span);
                    },
                    /*else_fn=*/nullptr, span);
              },
              span);
        },
        span);
  }

  // ---- Structured control-flow constructors ----
  //
  // Each method takes a body callback that receives a freshly-constructed
  // nested ``LoweringBuilder`` scoped to the body region. The callback emits
  // its body via the nested builder; this builder then drains the nested
  // stmts, wraps them in a ``SeqStmts`` (when there is more than one), and
  // emits the resulting ``ForStmt`` / ``IfStmt`` against its own ``stmts_``.
  //
  // The nested builder shares this builder's ``temp_counter_`` reference so
  // emitted temp names stay unique across the entire rule regardless of
  // nesting depth.

  /// Emit a side-effect-only ``for`` loop:
  ///
  ///     for loop_var in range(start, stop, step):
  ///         <body_fn-produced stmts>
  ///
  /// ``body_fn`` receives a fresh body builder and the freshly-created loop
  /// variable. The callback's return value is discarded — use this overload
  /// for loops whose only purpose is side effects (e.g. issuing notify /
  /// wait sequences).
  void EmitFor(const std::string& loop_var_name, const ExprPtr& start, const ExprPtr& stop,
               const ExprPtr& step, const std::function<void(LoweringBuilder&, const VarPtr&)>& body_fn,
               const Span& span) {
    auto loop_var = std::make_shared<Var>(MakeTempName(loop_var_name), start->GetType(), span);
    LoweringBuilder body_builder(base_name_, temp_counter_, /*nested=*/true);
    body_fn(body_builder, loop_var);
    auto body_stmt = WrapBodyStmts(body_builder.TakeStmts(), span);
    stmts_.push_back(std::make_shared<ForStmt>(loop_var, start, stop, step, std::vector<IterArgPtr>{},
                                               body_stmt, std::vector<VarPtr>{}, span));
  }

  /// Emit a reducing ``for`` loop with one loop-carried accumulator. The
  /// body callback receives a nested builder, the loop variable, and the
  /// accumulator (typed via ``init_value``); it returns the next iteration's
  /// accumulator value. The method returns an expression holding the
  /// post-loop accumulator, ready to feed into subsequent ops.
  ExprPtr EmitForReduce(const std::string& loop_var_name, const ExprPtr& start, const ExprPtr& stop,
                        const ExprPtr& step, const ExprPtr& init_value,
                        const std::function<ExprPtr(LoweringBuilder&, const VarPtr&, const VarPtr&)>& body_fn,
                        const Span& span) {
    auto loop_var = std::make_shared<Var>(MakeTempName(loop_var_name), start->GetType(), span);
    auto iter_arg = std::make_shared<IterArg>(MakeTempName(loop_var_name + "_acc"), init_value->GetType(),
                                              init_value, span);
    LoweringBuilder body_builder(base_name_, temp_counter_, /*nested=*/true);
    ExprPtr yield_val = body_fn(body_builder, loop_var, iter_arg);
    INTERNAL_CHECK_SPAN(yield_val, span)
        << "EmitForReduce body_fn must return the next iteration's accumulator value";
    body_builder.stmts_.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{yield_val}, span));
    auto body_stmt = WrapBodyStmts(body_builder.TakeStmts(), span);
    auto return_var =
        std::make_shared<Var>(MakeTempName(loop_var_name + "_final"), init_value->GetType(), span);
    stmts_.push_back(std::make_shared<ForStmt>(loop_var, start, stop, step, std::vector<IterArgPtr>{iter_arg},
                                               body_stmt, std::vector<VarPtr>{return_var}, span));
    return return_var;
  }

  /// Emit a side-effect-only ``if`` statement:
  ///
  ///     if cond:
  ///         <then_fn stmts>
  ///     [else:
  ///         <else_fn stmts>]
  ///
  /// Pass ``nullptr`` for ``else_fn`` when there is no else branch.
  void EmitIf(const ExprPtr& cond, const std::function<void(LoweringBuilder&)>& then_fn,
              const std::function<void(LoweringBuilder&)>& else_fn, const Span& span) {
    LoweringBuilder then_builder(base_name_, temp_counter_, /*nested=*/true);
    then_fn(then_builder);
    auto then_body = WrapBodyStmts(then_builder.TakeStmts(), span);

    std::optional<StmtPtr> else_body = std::nullopt;
    if (else_fn) {
      LoweringBuilder else_builder(base_name_, temp_counter_, /*nested=*/true);
      else_fn(else_builder);
      else_body = WrapBodyStmts(else_builder.TakeStmts(), span);
    }
    stmts_.push_back(std::make_shared<IfStmt>(cond, then_body, else_body, std::vector<VarPtr>{}, span));
  }

  /// Emit a value-producing ``if`` statement. Both branches must yield a
  /// value (via their body_fn's ExprPtr return); the method returns an
  /// expression holding the chosen value, ready to feed into subsequent ops.
  ExprPtr EmitIfExpr(const ExprPtr& cond, const std::function<ExprPtr(LoweringBuilder&)>& then_fn,
                     const std::function<ExprPtr(LoweringBuilder&)>& else_fn, const Span& span) {
    INTERNAL_CHECK_SPAN(then_fn && else_fn, span)
        << "EmitIfExpr requires both then_fn and else_fn (the if must yield a value on every path)";
    LoweringBuilder then_builder(base_name_, temp_counter_, /*nested=*/true);
    ExprPtr then_val = then_fn(then_builder);
    INTERNAL_CHECK_SPAN(then_val, span) << "EmitIfExpr then_fn must return the yielded value";
    then_builder.stmts_.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{then_val}, span));
    auto then_body = WrapBodyStmts(then_builder.TakeStmts(), span);

    LoweringBuilder else_builder(base_name_, temp_counter_, /*nested=*/true);
    ExprPtr else_val = else_fn(else_builder);
    INTERNAL_CHECK_SPAN(else_val, span) << "EmitIfExpr else_fn must return the yielded value";
    else_builder.stmts_.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{else_val}, span));
    auto else_body = WrapBodyStmts(else_builder.TakeStmts(), span);

    auto return_var = std::make_shared<Var>(MakeTempName("if_res"), then_val->GetType(), span);
    stmts_.push_back(std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>(else_body),
                                              std::vector<VarPtr>{return_var}, span));
    return return_var;
  }

  /// Drain accumulated statements (called by the mutator after the rule
  /// returns).
  std::vector<StmtPtr> TakeStmts() { return std::move(stmts_); }

 private:
  std::string MakeTempName(const std::string& qualifier) {
    return auto_name::BuildName(auto_name::GetBaseName(base_name_), qualifier, "tmp",
                                static_cast<int>(temp_counter_++));
  }

  // Wrap a sequence of body stmts into a single StmtPtr: pass through a sole
  // stmt, wrap multiple into a SeqStmts, and synthesise an empty SeqStmts
  // when the body is empty (a no-op body is still a valid loop / if branch).
  static StmtPtr WrapBodyStmts(std::vector<StmtPtr> body_stmts, const Span& span) {
    if (body_stmts.empty()) return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    if (body_stmts.size() == 1) return body_stmts.front();
    return std::make_shared<SeqStmts>(std::move(body_stmts), span);
  }

  std::string base_name_;
  std::size_t& temp_counter_;
  bool nested_ = false;
  int64_t barrier_count_ = 0;  ///< Call-local generation counter; see EmitBarrier.
  std::vector<StmtPtr> stmts_;
};

// Signature for a composite-lowering rule.
//
// @param call     Original composite-op Call. Rules read ``call->kwargs_``,
//                 ``call->span_``, and ``call->op_->name_`` for diagnostics.
// @param args     Visited operand expressions (var-remap already applied).
//                 Prefer these over ``call->args_`` so the rule sees post-
//                 visitor expressions.
// @param builder  Scratchpad: rule appends intermediate temps via builder.Bind
//                 (and structured control-flow via EmitFor / EmitIf / ...) and
//                 returns the final result expression.
// @return Final result expression. The mutator binds this to the target ``Var``
//         and splices the builder's accumulated statements before it.
using CompositeLoweringFn = ExprPtr (*)(const CallPtr& call, const std::vector<ExprPtr>& args,
                                        LoweringBuilder& builder);

// ============================================================================
// FP32 ``tile.sin`` / ``tile.cos`` lowering rules
//
// Recipe (matches gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h):
//   1. Range-reduce ``x`` to ``t ∈ [-π/2, π/2]`` via Cody-Waite (4-part π
//      split for sin; same plus +π/2 head/tail interleaved for cos).
//   2. Compute ``sign = (-1)^k = floor(k/2)·4 - 2·k + 1`` without a branch.
//   3. Evaluate degree-9 odd Horner polynomial ``P(t²)`` approximating
//      ``sin(t)/t``.
//   4. ``out = sign · t · P(t²)``.
//
// The two rules share ``LowerSinCos`` (parameterised by ``is_cos``).
// ============================================================================

// FP32 constants for Cody-Waite range reduction + degree-9 odd Horner. Values
// are the verbatim CANN/PyPTO recipe used by the framework reference at
// gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h. They
// are single-precision FP32 literals.
constexpr float kPiInv = 0.31830988732818603515625f;       ///< 1/pi (head)
constexpr float kPiV2 = 3.140625f;                         ///< pi head
constexpr float kPiC1 = 0.0009670257568359375f;            ///< pi split-1
constexpr float kPiC2 = 6.2771141529083251953125e-7f;      ///< pi split-2
constexpr float kPiC3 = 1.21644916362129151821e-10f;       ///< pi split-3
constexpr float kPiC4 = -1.0290623200529979163e-13f;       ///< pi split-4
constexpr float kPiHalfHead = 1.57079637050628662109375f;  ///< pi/2 head (cos only)
constexpr float kPiHalfTail = -4.371139000189375e-8f;      ///< pi/2 tail (cos only)
constexpr float kHalf = 0.5f;
constexpr float kM4 = 4.0f;
constexpr float kNeg2 = -2.0f;
constexpr float kOne = 1.0f;
constexpr float kR0 = 2.604926501e-6f;
constexpr float kR1 = -1.980894471e-4f;
constexpr float kR2 = 8.333049340e-3f;
constexpr float kR3 = -1.666665792e-1f;

// Round modes for tile.cast (mirrors the registration in
// src/ir/op/tile_ops/unary.cpp): None=0, RINT=1, ROUND=2, FLOOR=3.
constexpr int kCastModeNone = 0;
constexpr int kCastModeRint = 1;
constexpr int kCastModeRound = 2;
constexpr int kCastModeFloor = 3;

// Shared validator: tile.sin / tile.cos accept exactly one FP32 TileType arg.
void ValidateTrigArgs(const std::vector<ExprPtr>& args, const Span& span, const char* op_name) {
  INTERNAL_CHECK_SPAN(args.size() == 1, span)
      << op_name << " requires exactly 1 argument, got " << args.size();
  auto in_tile_type = As<TileType>(args[0]->GetType());
  INTERNAL_CHECK_SPAN(in_tile_type, span)
      << op_name << " requires a TileType argument, got " << args[0]->GetType()->TypeName();
  INTERNAL_CHECK_SPAN(in_tile_type->dtype_ == DataType::FP32, span)
      << op_name << " is FP32-only, got dtype " << in_tile_type->dtype_.ToString();
}

// Decompose sin(x) or cos(x) into primitives. ``b`` accumulates the prelude
// statements; the returned ExprPtr is the final result (not yet bound).
ExprPtr LowerSinCos(const ExprPtr& x, bool is_cos, LoweringBuilder& b, const Span& span) {
  // ---- Step 1: range reduction --------------------------------------------
  // k_f = float(rint(x * PI_INV + 0.5))  for cos
  // k_f = float(round(x * PI_INV))        for sin
  auto pi_inv_x = b.Bind("pi_inv_x", b.Muls(x, kPiInv, span), span);
  ExprPtr k_i;
  if (is_cos) {
    auto k_pre = b.Bind("k_pre", b.Adds(pi_inv_x, kHalf, span), span);
    k_i = b.Bind("k_i", b.Cast(k_pre, DataType::INT32, kCastModeRint, span), span);
  } else {
    k_i = b.Bind("k_i", b.Cast(pi_inv_x, DataType::INT32, kCastModeRound, span), span);
  }
  auto k_f = b.Bind("k_f", b.Cast(k_i, DataType::FP32, kCastModeNone, span), span);

  // t = x - k_f * pi (4-part Cody-Waite). For cos, +pi/2 head/tail are
  // interleaved between PI_C1 and PI_C2, and after PI_C4 respectively.
  auto kpv2 = b.Bind("k_pi_v2", b.Muls(k_f, kPiV2, span), span);
  auto t = b.Bind("t0", b.Sub(x, kpv2, span), span);
  auto kpc1 = b.Bind("k_pi_c1", b.Muls(k_f, kPiC1, span), span);
  t = b.Bind("t1", b.Sub(t, kpc1, span), span);
  if (is_cos) {
    t = b.Bind("t1h", b.Adds(t, kPiHalfHead, span), span);
  }
  auto kpc2 = b.Bind("k_pi_c2", b.Muls(k_f, kPiC2, span), span);
  t = b.Bind("t2", b.Sub(t, kpc2, span), span);
  auto kpc3 = b.Bind("k_pi_c3", b.Muls(k_f, kPiC3, span), span);
  t = b.Bind("t3", b.Sub(t, kpc3, span), span);
  auto kpc4 = b.Bind("k_pi_c4", b.Muls(k_f, kPiC4, span), span);
  t = b.Bind("t4", b.Sub(t, kpc4, span), span);
  if (is_cos) {
    t = b.Bind("t4t", b.Adds(t, kPiHalfTail, span), span);
  }

  // ---- Step 2: sign = floor(k_f / 2) * 4 + k_f * (-2) + 1 ------------------
  auto half_k = b.Bind("half_k", b.Muls(k_f, kHalf, span), span);
  auto floor_hk_i = b.Bind("floor_hk_i", b.Cast(half_k, DataType::INT32, kCastModeFloor, span), span);
  auto floor_hk_f = b.Bind("floor_hk_f", b.Cast(floor_hk_i, DataType::FP32, kCastModeNone, span), span);
  auto floor_x4 = b.Bind("floor_x4", b.Muls(floor_hk_f, kM4, span), span);
  auto neg2_k = b.Bind("neg2_k", b.Muls(k_f, kNeg2, span), span);
  auto sign_pre = b.Bind("sign_pre", b.Add(floor_x4, neg2_k, span), span);
  auto sign = b.Bind("sign", b.Adds(sign_pre, kOne, span), span);

  // ---- Step 3: Horner P(t^2) = (((R0*t^2 + R1)*t^2 + R2)*t^2 + R3)*t^2 + 1
  auto t2 = b.Bind("t2sq", b.Mul(t, t, span), span);
  auto p = b.Bind("p_r0", b.Muls(t2, kR0, span), span);
  p = b.Bind("p_r1", b.Adds(p, kR1, span), span);
  p = b.Bind("p_t2_r1", b.Mul(p, t2, span), span);
  p = b.Bind("p_r2", b.Adds(p, kR2, span), span);
  p = b.Bind("p_t2_r2", b.Mul(p, t2, span), span);
  p = b.Bind("p_r3", b.Adds(p, kR3, span), span);
  p = b.Bind("p_t2_r3", b.Mul(p, t2, span), span);
  p = b.Bind("p_one", b.Adds(p, kOne, span), span);

  // ---- Step 4: out = sign * t * P(t^2) -------------------------------------
  auto t_p = b.Bind("t_p", b.Mul(t, p, span), span);
  return b.Mul(sign, t_p, span);
}

ExprPtr LowerSinRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& builder) {
  ValidateTrigArgs(args, call->span_, "tile.sin");
  return LowerSinCos(args[0], /*is_cos=*/false, builder, call->span_);
}

ExprPtr LowerCosRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& builder) {
  ValidateTrigArgs(args, call->span_, "tile.cos");
  return LowerSinCos(args[0], /*is_cos=*/true, builder, call->span_);
}

// ============================================================================
// ``tile.tquant_mx`` lowering — grouped TQUANT + exponent X-to-ZZ.
//
// Public group_axis=1 is the A-side [M,K] path. group_axis=0 is the B-side
// [N,K] path: transpose to [K,N] first, then PTOAS axis0. Axis0 X-to-ZZ follows
// pto-isa TMovDnTo2Zz (pin be5ccb76): DN [M̂,N] -> ZZ [N,M̂] row/row, then a
// zero-copy tile.transpose_view yields the public [M̂,N] col/col scale. Same-
// InCore mix with matmul_mx is not supported yet; stage through GM between AIV
// and AIC (follow-up).
ExprPtr LowerTileTQuantMxRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const auto& span = call->span_;
  auto& reg = OpRegistry::GetInstance();
  auto src = args[0];
  const int group_axis = call->GetKwarg<int>("group_axis");
  INTERNAL_CHECK_SPAN(group_axis == 0 || group_axis == 1, span)
      << "Internal error: tile.tquant_mx group_axis must be 0 or 1";
  const bool packed_b = group_axis == 0;

  if (packed_b) {
    auto axis0 = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto axis1 = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    src = b.Bind("tq_src_kn", reg.Create("tile.transpose", {src, axis0, axis1}, {}, span), span);
  }

  auto src_tile = As<TileType>(src->GetType());
  INTERNAL_CHECK_SPAN(src_tile && src_tile->shape_.size() == 2, span)
      << "Internal error: tile.tquant_mx lowering requires a 2D source tile";
  auto rows_const = As<ConstInt>(src_tile->shape_[0]);
  auto cols_const = As<ConstInt>(src_tile->shape_[1]);
  // PTOAS special requirement: TQUANT/X2ZZ need static physical+valid extents so
  // EmitStaticValidTileView can emit a concrete treshape result type (not v_*=?).
  INTERNAL_CHECK_SPAN(rows_const && cols_const, span)
      << "Internal error: tile.tquant_mx lowering requires static source shapes";
  const int64_t rows = rows_const->value_;
  const int64_t cols = cols_const->value_;
  // Positive dims / group-axis divisibility already enforced by DeduceTileTQuantMxType;
  // re-check before `/ 32` so a broken invariant cannot silently truncate.
  INTERNAL_CHECK_SPAN(rows > 0 && cols > 0, span)
      << "Internal error: tile.tquant_mx lowering requires positive source dimensions";
  INTERNAL_CHECK_SPAN((group_axis == 1 && cols % 32 == 0) || (group_axis == 0 && rows % 32 == 0), span)
      << "Internal error: tile.tquant_mx source is not divisible by its group axis";
  const int64_t group_rows = group_axis == 0 ? rows / 32 : rows;
  const int64_t group_cols = group_axis == 0 ? cols : cols / 32;
  INTERNAL_CHECK_SPAN(group_rows <= std::numeric_limits<int64_t>::max() / group_cols, span)
      << "Internal error: tile.tquant_mx scale-group count overflows int64";
  const int64_t groups = group_rows * group_cols;

  auto make_dim = [&](int64_t value) { return std::make_shared<ConstInt>(value, DataType::INDEX, span); };
  auto make_shape = [&](int64_t dim0, int64_t dim1) {
    return std::make_shared<MakeTuple>(std::vector<ExprPtr>{make_dim(dim0), make_dim(dim1)}, span);
  };
  auto bind_typed_create = [&](const std::string& name, int64_t physical_rows, int64_t physical_cols,
                               int64_t valid_rows, int64_t valid_cols, DataType dtype, TileLayout slayout,
                               int64_t fractal = 512, TileLayout blayout = TileLayout::row_major) {
    auto shape = make_shape(physical_rows, physical_cols);
    std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", dtype},
                                                                   {"target_memory", MemorySpace::Vec}};
    auto created = As<Call>(reg.Create("tile.create", {shape}, create_kwargs, span));
    INTERNAL_CHECK_SPAN(created, span) << "Internal error: tile.create did not produce a Call";
    TileView view;
    view.valid_shape = {make_dim(valid_rows), make_dim(valid_cols)};
    view.blayout = blayout;
    view.slayout = slayout;
    view.fractal = fractal;
    auto type =
        std::make_shared<TileType>(std::vector<ExprPtr>{make_dim(physical_rows), make_dim(physical_cols)},
                                   dtype, std::nullopt, view, MemorySpace::Vec);
    auto typed_create =
        std::make_shared<Call>(created->op_, created->args_, created->kwargs_, created->attrs_, type, span);
    return b.Bind(name, typed_create, span);
  };

  // Axis1 uses the legacy-flat exponent branch so FP32, FP16, and BF16 all
  // remain supported by the pinned PTO-ISA. Axis0 uses canonical [M/32,N].
  const int64_t aux_rows = group_axis == 0 ? group_rows : 1;
  const int64_t aux_cols = group_axis == 0 ? group_cols : groups;
  auto max_tile = bind_typed_create("tq_max", aux_rows, aux_cols, aux_rows, aux_cols, src_tile->dtype_,
                                    TileLayout::none_box);

  int64_t scaling_physical_cols = aux_cols;
  if (group_axis == 1) {
    // PTOAS grouped-TQUANT scratch sizing and FP32 unroll thresholds.
    constexpr int64_t kFp32ScaleAlignment = 64;
    constexpr int64_t kFp16ScaleAlignment = 128;
    constexpr int64_t kFp32UnrollMinElements = 1024;
    constexpr int64_t kFp32UnrollMultiple = 256;
    const int64_t align = src_tile->dtype_ == DataType::FP32 ? kFp32ScaleAlignment : kFp16ScaleAlignment;
    int64_t scale_elements = groups;
    INTERNAL_CHECK_SPAN(rows <= std::numeric_limits<int64_t>::max() / cols, span)
        << "Internal error: tile.tquant_mx source element count overflows int64";
    const int64_t source_elements = rows * cols;
    const bool unroll = src_tile->dtype_ == DataType::FP32 && source_elements > kFp32UnrollMinElements &&
                        source_elements % kFp32UnrollMultiple == 0;
    INTERNAL_CHECK_SPAN(!unroll || scale_elements <= std::numeric_limits<int64_t>::max() / 2, span)
        << "Internal error: tile.tquant_mx unrolled scale scratch size overflows int64";
    if (unroll) scale_elements *= 2;
    INTERNAL_CHECK_SPAN(scale_elements <= std::numeric_limits<int64_t>::max() - (align - 1), span)
        << "Internal error: tile.tquant_mx aligned scale scratch size overflows int64";
    scaling_physical_cols = (scale_elements + align - 1) / align * align;
  }
  auto scaling_tile = bind_typed_create("tq_scaling", aux_rows, scaling_physical_cols, aux_rows, aux_cols,
                                        src_tile->dtype_, TileLayout::none_box);

  auto public_types = As<TupleType>(call->GetType());
  INTERNAL_CHECK_SPAN(public_types && public_types->types_.size() == 2, span)
      << "Internal error: tile.tquant_mx must return exactly two tile types";
  auto public_dst_type = As<TileType>(public_types->types_[0]);
  INTERNAL_CHECK_SPAN(public_dst_type && public_dst_type->dtype_ == DataType::FP8E4M3FN, span)
      << "Internal error: tile.tquant_mx public destination must be FP8E4M3FN";

  // Value-returning TQUANT (gather_compare-style): Bind the TupleType result,
  // then project dst/exp so InitMemRef + ResolveTupleResultElements see real
  // TupleGetItem consumers. max/scaling remain write-only workspace inputs.
  DataType dtype = call->GetKwarg<DataType>("dtype", DataType::FP8E4M3FN);
  auto raw_tuple = b.Bind("tq_raw",
                          reg.Create("tile.tquant_mx_raw", {src, max_tile, scaling_tile},
                                     {{"dtype", dtype}, {"group_axis", group_axis}}, span),
                          span);
  auto raw_dst = b.Bind("tq_dst", std::make_shared<TupleGetItemExpr>(raw_tuple, 0, span), span);
  auto raw_exp = b.Bind("tq_exp", std::make_shared<TupleGetItemExpr>(raw_tuple, 1, span), span);

  // MXFP8 uses PTOAS's raw INT8 destination and exposes a zero-copy FP8 alias.
  ExprPtr dst_tile =
      b.Bind("tq_quant",
             reg.Create("tile.reinterpret_view", {raw_dst}, {{"dtype", DataType::FP8E4M3FN}}, span), span);

  // Axis1 X-to-ZZ tmp: 64 + ceil(rows/16)*cols bytes. Axis0 (TMovDnTo2Zz): ISA
  // still requires a Vec tmp operand; use one 32-byte Vec pad unit.
  constexpr int64_t kVecByteAlign = 32;
  int64_t tmp_bytes = kVecByteAlign;
  if (group_axis == 1) {
    INTERNAL_CHECK_SPAN(group_rows <= std::numeric_limits<int64_t>::max() - 15, span)
        << "Internal error: tile.tquant_mx padded exponent rows overflow int64";
    const int64_t row_blocks = (group_rows + 15) / 16;
    INTERNAL_CHECK_SPAN(row_blocks <= (std::numeric_limits<int64_t>::max() - 64) / group_cols, span)
        << "Internal error: tile.tquant_mx exponent temporary size overflows int64";
    tmp_bytes = 64 + row_blocks * group_cols;
  }
  INTERNAL_CHECK_SPAN(tmp_bytes <= std::numeric_limits<int64_t>::max() - (kVecByteAlign - 1), span)
      << "Internal error: tile.tquant_mx aligned exponent temporary size overflows int64";
  const int64_t tmp_physical_bytes = (tmp_bytes + kVecByteAlign - 1) / kVecByteAlign * kVecByteAlign;
  auto x2zz_tmp = bind_typed_create("tq_x2zz_tmp", 1, tmp_physical_bytes, 1, tmp_physical_bytes,
                                    DataType::UINT8, TileLayout::none_box);
  // Value-returning X-to-ZZ: InitMemRef allocates ZZ dst from the deduced type.
  // Axis1 needs dst_rows/dst_cols because TQUANT exp is legacy-flat [1,M*G].
  std::vector<std::pair<std::string, std::any>> x2zz_kwargs = {{"group_axis", group_axis}};
  if (group_axis == 1) {
    x2zz_kwargs.emplace_back("dst_rows", static_cast<int>(group_rows));
    x2zz_kwargs.emplace_back("dst_cols", static_cast<int>(group_cols));
  }
  auto zz_exp =
      b.Bind("tq_exp_zz", reg.Create("tile.tmov_x2zz", {raw_exp, x2zz_tmp}, x2zz_kwargs, span), span);
  ExprPtr exp_tile = b.Bind(
      "tq_scale", reg.Create("tile.reinterpret_view", {zz_exp}, {{"dtype", DataType::FP8E8M0}}, span), span);
  if (group_axis == 0) {
    // ZZ [N,M̂] row/row <-> public MX_B [M̂,N] col/col over the same bytes.
    exp_tile = b.Bind("tq_scale_nn", reg.Create("tile.transpose_view", {exp_tile}, {}, span), span);
  }
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{dst_tile, exp_tile}, span);
}

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

// Forward declaration — the ring rule is defined after the mesh rule but is
// called from the mode dispatch inside LowerTensorAllReduceRule.
ExprPtr LowerTensorRingAllReduceRule(const CallPtr& call, const std::vector<ExprPtr>& args,
                                     LoweringBuilder& b);

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
  const int64_t element_bytes = static_cast<int64_t>(target_type->dtype_.GetByte());
  INTERNAL_CHECK_SPAN(element_bytes > 0, span)
      << "pld.tensor.allreduce target dtype has no storage width: " << target_type->dtype_.ToString();
  const int64_t chunk_elements = kAllReduceChunkBytes / element_bytes;
  INTERNAL_CHECK_SPAN(chunk_elements > 0, span)
      << "pld.tensor.allreduce dtype is wider than the mesh chunk byte budget";
  INTERNAL_CHECK_SPAN(kPTOTileAlignmentBytes % element_bytes == 0, span)
      << "pld.tensor.allreduce dtype width must divide the tile alignment";
  const int64_t alignment_elements = kPTOTileAlignmentBytes / element_bytes;
  auto alignment_elements_idx = std::make_shared<ConstInt>(alignment_elements, DataType::INDEX, span);
  auto alignment_minus_one_idx = std::make_shared<ConstInt>(alignment_elements - 1, DataType::INDEX, span);
  auto max_chunk_cols = std::make_shared<ConstInt>(chunk_elements, DataType::INDEX, span);

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
  } else if (auto flat_extent = As<ConstInt>(flat_valid_shape[1]);
             flat_extent && flat_extent->value_ > 0 && flat_extent->value_ < chunk_elements) {
    // Do not reserve a full 16-KiB tile for a statically small allreduce. PTO
    // tiles require a 32-byte-aligned row, so use the smallest legal physical
    // width that covers the logical extent. Every chunk-local tile inherits
    // this width, preserving the caller's remaining VEC UB budget.
    const int64_t aligned_extent = ((flat_extent->value_ - 1) / alignment_elements + 1) * alignment_elements;
    chunk_cols = std::make_shared<ConstInt>(aligned_extent, DataType::INDEX, span);
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

  const int64_t element_bytes = static_cast<int64_t>(target_type->dtype_.GetByte());
  INTERNAL_CHECK_SPAN(element_bytes > 0, span)
      << "pld.tensor.allreduce mode=ring target dtype has no storage width: "
      << target_type->dtype_.ToString();
  const int64_t max_chunk_elements = kAllReduceChunkBytes / element_bytes;
  INTERNAL_CHECK_SPAN(max_chunk_elements > 0, span)
      << "pld.tensor.allreduce mode=ring dtype is wider than the chunk byte budget";
  INTERNAL_CHECK_SPAN(kPTOTileAlignmentBytes % element_bytes == 0, span)
      << "pld.tensor.allreduce mode=ring dtype width must divide the tile alignment";
  const int64_t alignment_elements = kPTOTileAlignmentBytes / element_bytes;
  auto alignment_elements_idx = std::make_shared<ConstInt>(alignment_elements, DataType::INDEX, span);
  auto alignment_minus_one_idx = std::make_shared<ConstInt>(alignment_elements - 1, DataType::INDEX, span);

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

  ExprPtr chunk_cols = std::make_shared<ConstInt>(max_chunk_elements, DataType::INDEX, span);
  if (auto segment_const = As<ConstInt>(max_segment_cols);
      segment_const && segment_const->value_ > 0 && segment_const->value_ < max_chunk_elements) {
    const int64_t aligned_segment =
        ((segment_const->value_ - 1) / alignment_elements + 1) * alignment_elements;
    chunk_cols = std::make_shared<ConstInt>(aligned_segment, DataType::INDEX, span);
  }
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
  int64_t rows_val = 1;
  for (size_t d = 0; d + 1 < target_type->shape_.size(); ++d) {
    auto dim_c = As<ConstInt>(target_type->shape_[d]);
    INTERNAL_CHECK_SPAN(dim_c, span) << "broadcast target shape must be static";
    rows_val *= dim_c->value_;
  }
  auto last_dim_c = As<ConstInt>(target_type->shape_.back());
  INTERNAL_CHECK_SPAN(last_dim_c, span) << "broadcast target shape must be static";
  int64_t cols_val = last_dim_c->value_;

  auto rows_expr = std::make_shared<ConstInt>(rows_val, DataType::INDEX, span);
  auto cols_expr = std::make_shared<ConstInt>(cols_val, DataType::INDEX, span);
  auto stage_shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{rows_expr, cols_expr}, span);

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
//   0.  tile.create [1, SIZE] VEC — staging tile for auto-chunking
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
  // pld.tile.put auto-chunks when SIZE exceeds the staging-tile capacity, so a
  // single [1, SIZE] VEC staging tile suffices regardless of SIZE.
  auto put_stage =
      b.Bind("ag_stage",
             reg.Create("tile.create", {chunk_shape},
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

// ============================================================================
// ``pld.tensor.barrier`` lowering rule
//
// Cross-rank barrier: notify-all (AtomicAdd 1) then wait-all (Ge generation).
// Pure synchronisation — no data movement.  Returns the signal expression so
// the rebind idiom (``sig = pld.tensor.barrier(sig)``) matches allreduce; the
// barrier restarts at generation 1 on every call (self-clearing credit protocol).
// ============================================================================

ExprPtr LowerTensorBarrierRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 1, span) << "pld.tensor.barrier rule expects 1 arg, got " << args.size();
  const auto& signal = args[0];
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  INTERNAL_CHECK_SPAN(signal_type, span)
      << "pld.tensor.barrier signal must be DistributedTensorType (deducer-rejected otherwise)";
  ValidateMeshSignalShape(signal_type, "pld.tensor.barrier", span);

  auto comm = b.EmitCommSetup(signal, span);

  // ---- AtomicAdd cell[my_rank, 0] on each peer, then wait cell[src, 0] >= gen ----
  const int64_t generation = b.EmitBarrier(signal, comm, "", span);

  // Self-clearing epilogue: exactly one credit per peer this call.
  auto total_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // Rebind: return the signal so the LHS Var retains the DistributedTensor view.
  return signal;
}

// ============================================================================
// ``pld.tensor.all_to_all`` lowering rule
//
// Push-based symmetric all-to-all: every rank sends a distinct chunk to every
// other rank.  2-phase decomposition:
//
//   Phase 1 (push): for dest in 0..NR-1:
//       pld.tile.put(dst=target, peer=dest, src=input, stage,   // push row to peer
//                    dst_offsets=[my_rank, 0],
//                    src_offsets=[dest, 0],
//                    shape=[1, SIZE], atomic=None)
//
//   Phase 2 (barrier):
//       notify-all (AtomicAdd 1)
//       wait-all   (Ge generation)
//
//   Result: target (window-as-result).  After the barrier, target[src, :]
//           holds the chunk received from rank src.
//
// Input layout:  input[dest, :] = chunk destined for rank dest.
//
// Emits tile.create + pld.tile.put directly (the tensor-level pld.tensor.put
// has no codegen and ConvertTensorToTileOps runs before this pass — same
// reason broadcast/allgather emit pld.tile.get directly). The HCCL TPUT engine
// streams input[dest, :] through the shared VEC staging tile into the peer's
// window row [my_rank, 0], so a row larger than the staging tile is auto-chunked
// by pto-isa. The self-rank case (peer == my_rank) falls out of the same TPUT
// path via HCCL identity mapping (CommRemotePtr returns the local ptr), so no
// separate self-copy branch is needed.
// ============================================================================

ExprPtr LowerTensorAllToAllRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 3, span)
      << "pld.tensor.all_to_all rule expects 3 args (input, target, signal), got " << args.size();
  const auto& input = args[0];
  const auto& target = args[1];
  const auto& signal = args[2];

  auto input_type = As<TensorType>(input->GetType());
  INTERNAL_CHECK_SPAN(input_type, span)
      << "pld.tensor.all_to_all input must be TensorType, got " << input->GetType()->TypeName();
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.all_to_all target must be DistributedTensorType (deducer-rejected otherwise)";
  INTERNAL_CHECK_SPAN(target_type->shape_.size() == 2, span)
      << "pld.tensor.all_to_all target must be 2D [NR, SIZE]";
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  ValidateMeshSignalShape(signal_type, "pld.tensor.all_to_all", span);

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  // Per-chunk shape: [1, SIZE] where SIZE = target.shape[1].
  auto size_expr = target_type->shape_[1];
  auto chunk_shape = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span), size_expr}, span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  // Offsets for the push target: write at [my_rank, 0] on the peer's window.
  // Every rank r writes its per-destination chunk to slot [r, 0] on every
  // peer's window, so after the barrier, rank r sees target[src, :] = chunk
  // sent from src to r.
  auto my_rank_offsets = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{comm.my_rank, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

  // ---- Phase 1: push — write each per-destination row directly into the
  //      peer's window via pld.tile.put (TPUT-based). The HCCL TPUT engine
  //      streams input[dest, :] through the shared VEC staging tile, so a row
  //      larger than the stage is auto-chunked. The self-rank case (peer ==
  //      my_rank) falls out of the same path via HCCL identity mapping.
  //
  // One shared [1, SIZE] VEC staging tile is reused across all destinations,
  // mirroring allgather's per-peer pld.tile.get.
  auto put_stage =
      b.Bind("aa_stage",
             reg.Create("tile.create", {chunk_shape},
                        {{"dtype", target_type->dtype_}, {"target_memory", MemorySpace::Vec}}, span),
             span);

  b.EmitFor(
      "dest", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& dest_var) {
        auto dest_row_offsets = std::make_shared<MakeTuple>(
            std::vector<ExprPtr>{dest_var, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

        // pld.tile.put(dst, peer, src, stage, dst_offsets, src_offsets, shape):
        // read input[dest, :] and write it to the peer's window row [my_rank, 0].
        body.Bind(
            "aa_put",
            reg.Create("pld.tile.put",
                       {target, dest_var, input, put_stage, my_rank_offsets, dest_row_offsets, chunk_shape},
                       {{"atomic", static_cast<int>(AtomicType::kNone)}}, span),
            span);
      },
      span);

  // ---- Phase 2: barrier ----
  const int64_t generation = b.EmitBarrier(signal, comm, "", span);

  // Self-clearing epilogue: exactly one credit per peer this call.
  auto total_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // Window-as-result: target[src, :] now holds the chunk from rank src.
  // No read-back phase or post-barrier needed — the barrier guarantees all
  // peer writes are complete, and no peer reads the window afterwards.
  return target;
}

// ============================================================================
// LowerTensorAllToAllVRule — pld.tensor.all_to_all_v (variable-size all-to-all)
//
// Variable-size all-to-all (MPI_Alltoallv pattern).  Each rank pushes a full
// MAX_RECV-row capacity block to every peer via a single static-shape
// pld.tile.put per destination; only ``min(send_counts[dest], MAX_RECV)`` of
// those rows are logically valid.  ``send_counts[dest]`` is a *runtime*,
// data-dependent count read from device data during the exchange, but it
// does not change the transfer size (PTOAS requires static partition-view
// dims for pto.comm.tput).  The 5-arg API signature (input, target, signal,
// send_counts, recv_counts) extends the symmetric all_to_all's
// window-as-result pattern: the intrinsic returns target, and the caller
// reads back from the window with tile.load.  During the push phase each
// rank also publishes the *clamped* ``clamp(send_counts[dest], 0, MAX_RECV)``
// into peer ``dest``'s ``recv_counts[my_rank, 0]`` via ``pld.system.notify``
// (Set) — MPI_Alltoallv recvcounts — so after the barrier the receiver knows
// which rows of each source's MAX_RECV slot are logically valid. Notify
// writes a scalar INT32 cell (same path as the barrier signal), so
// ``recv_counts`` stays ``[NR, 1]`` and no post-convert ``tensor.create``
// scratch is needed (ConvertTensorToTileOps already ran before this pass).
//
// 2-phase push-based decomposition:
//
//   Phase 1 (push):
//     For each dest ∈ [0, NR):
//       rows = clamp(send_counts[dest], 0, MAX_RECV)   // runtime scalar read
//       notify(recv_counts, dest, [my_rank, 0], rows, Set)  // clamped count
//       // Single pld.tile.put per destination: contiguous [rows, SIZE] block
//       // at input[dest*MAX_RECV, :] → target[my_rank*MAX_RECV, :]. The
//       // transfer shape is the runtime [rows, SIZE] (PTOAS accepts dynamic
//       // partition-view dims on pto.comm.tput).  A [1, SIZE] staging tile
//       // feeds the TPUT engine, which 2-D-slides the transfer through it.
//
//   Phase 2: self-clearing credit barrier
//     EmitBarrier() — AtomicAdd(+1) on every peer cell, then Wait(Ge 1)
//     EmitEpilogueReset(-1) — subtracts the credit back to zero after the call
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
  INTERNAL_CHECK_SPAN(args.size() == 5, span) << "pld.tensor.all_to_all_v rule expects 5 args "
                                                 "(input, target, signal, send_counts, recv_counts), got "
                                              << args.size();
  const auto& input = args[0];
  const auto& target = args[1];
  const auto& signal = args[2];
  const auto& send_counts = args[3];
  const auto& recv_counts = args[4];

  // The composite rail expands into point-to-point primitives inside one
  // kernel, so it is single-core by construction. A multi-AIV request belongs
  // on the managed CHIP/L2 rail, which submits a gang of AIV blocks instead.
  const auto core_num = call->GetKwarg<int>("core_num", 1);
  CHECK_SPAN(core_num == 1, span)
      << "InCore pld.tensor.all_to_all_v requires core_num=1, got core_num=" << core_num
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

  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);

  // SIZE = target[1].
  auto size_expr = target_type->shape_[1];

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  // MAX_RECV = target[0] / NR.  NR is extracted from signal[0]
  // (deducer-enforced compile-time constant).  Signal is required to be 2D
  // [NR, 1] so MakeSignalOffsets(rank) → [rank, 0] matches notify/wait.
  auto total_rows_c = As<ConstInt>(target_type->shape_[0]);
  INTERNAL_CHECK_SPAN(total_rows_c, span) << "target dim 0 must be a compile-time constant";
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  INTERNAL_CHECK_SPAN(signal_type, span) << "signal must be DistributedTensorType";
  ValidateMeshSignalShape(signal_type, "pld.tensor.all_to_all_v", span);
  auto nr_c = As<ConstInt>(signal_type->shape_[0]);
  INTERNAL_CHECK_SPAN(nr_c, span) << "signal dim 0 (NR) must be a compile-time constant";
  int64_t max_recv_value = total_rows_c->value_ / nr_c->value_;
  INTERNAL_CHECK_SPAN(max_recv_value * nr_c->value_ == total_rows_c->value_, span)
      << "target dim 0 (" << total_rows_c->value_ << ") must be divisible by NR (" << nr_c->value_ << ")";
  auto max_recv_expr = std::make_shared<ConstInt>(max_recv_value, DataType::INDEX, span);

  // Per-destination staging tile: static [1, SIZE] — pto-isa auto-chunks the
  // transfer through it. The stage stays static (UB is allocated statically)
  // even though the transfer extent is dynamic.
  auto stage_shape = std::make_shared<MakeTuple>(std::vector<ExprPtr>{one_idx, size_expr}, span);

  // ---- Phase 1: push per-destination blocks to peer windows ----
  // One shared [1, SIZE] VEC staging tile reused across all destinations;
  // a single pld.tile.put per destination transfers [rows, SIZE], where rows is
  // the runtime send count clamped to [0, MAX_RECV] — so only the payload
  // crosses the wire, not the full capacity block.
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
        // (``tensor.read`` → ``pto.load_scalar``) and clamped to the
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

        // Publish the *clamped* transfer count into peer dest's
        // recv_counts[my_rank, 0] via TNOTIFY Set — same scalar-cell path as
        // the barrier signal, including self (peer offset is 0 for self).
        // Emitted UNCONDITIONALLY, outside the rows > 0 guard below: a
        // destination receiving zero rows still needs recv_counts = 0 published,
        // or it would read a stale count from a previous invocation.
        auto count_i32 = body.Bind("aav_count_i32", MakeCast(rows, DataType::INT32, span), span);
        body.Bind("aav_count_notify",
                  reg.Create("pld.system.notify", {recv_counts, dest_var, my_recv_offsets, count_i32},
                             {{"op", static_cast<int>(NotifyOp::kSet)}}, span),
                  span);

        // Single pld.tile.put per destination transferring exactly the rows
        // being sent. The [1, SIZE] VEC staging tile feeds the TPUT engine,
        // which 2-D-slides the larger transfer through it.
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
        // [1, SIZE] staging tile, and ValidateStageFitsTransfer skips dynamic
        // dims because the runtime extent bounds them.
        auto transfer_shape = std::make_shared<MakeTuple>(std::vector<ExprPtr>{rows, size_expr}, span);
        // Skip the push entirely for a destination getting no rows — a
        // zero-extent transfer has no defined TPUT behaviour. The count TNOTIFY
        // above stays outside this guard on purpose.
        body.EmitIf(
            body.Gt(rows, zero_idx, span),
            [&](LoweringBuilder& then_body) {
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

  // ---- Phase 2: self-clearing credit barrier ----
  const int64_t generation = b.EmitBarrier(signal, comm, "", span);

  // Self-clearing epilogue: exactly one credit per peer this call.
  auto total_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // Window-as-result: target[src*MAX_RECV+r, :] now holds the chunk from
  // rank src, offset r (full MAX_RECV capacity). The caller reads back from
  // the window with tile.load, using recv_counts[src] (clamped to MAX_RECV
  // at publish time) to identify valid rows and skip capacity holes.
  return target;
}

// ----------------------------------------------------------------------------
// Composite-op dispatch table.
//
// ``LowerCompositeOps`` is a generic dispatcher: it rewrites a ``var = Call(...)``
// AssignStmt (or a composite-op Call embedded directly in a ReturnStmt) only
// when the callee name appears here. Adding a new composite op = add a rule
// function above + one row in ``kRules``; the mutator below needs no change.
// A new ``pld.tensor.*`` collective must additionally be listed in
// ``LowerCompositeOpsMutator::IsTensorCollective`` so it inherits the HOST
// deferral, must barrier through ``LoweringBuilder::EmitBarrier`` so it shares
// the self-clearing credit-barrier protocol instead of rolling a one-off
// notify/wait pair, and must call ``EmitEpilogueReset`` exactly once with the
// total credit count it issued so the signal returns to all-zero after the
// call.
//
// Today the rules are ``tile.sin`` / ``tile.cos``, ``tile.tquant_mx``, and
// ``pld.tensor.*`` distributed collectives. Host-level allreduce is skipped here
// and lowered later by LowerHostTensorCollectives. The pass is idempotent
// provided each rule emits only ops not listed here.
//
// When the table grows past a handful of entries — or a rule wants its own
// translation unit — promote this back to a standalone registry under
// ``src/ir/transforms/composite_ops/``.
// ----------------------------------------------------------------------------
CompositeLoweringFn LookupCompositeRule(const std::string& op_name) {
  static const std::unordered_map<std::string, CompositeLoweringFn> kRules = {
      {"tile.sin", &LowerSinRule},
      {"tile.cos", &LowerCosRule},
      // tile.tquant_mx → tile.tquant_mx_raw + tile.tmov_x2zz (value-returning SSA).
      // Scratch tiles are created with MemorySpace::Vec before InferTileMemorySpace.
      {"tile.tquant_mx", &LowerTileTQuantMxRule},
      {"pld.tensor.allreduce", &LowerTensorAllReduceRule},
      {"pld.tensor.allgather", &LowerTensorAllGatherRule},
      {"pld.tensor.reduce_scatter", &LowerTensorReduceScatterRule},
      {"pld.tensor.barrier", &LowerTensorBarrierRule},
      {"pld.tensor.broadcast", &LowerTensorBroadcastRule},
      {"pld.tensor.all_to_all", &LowerTensorAllToAllRule},
      {"pld.tensor.all_to_all_v", &LowerTensorAllToAllVRule},
  };
  auto it = kRules.find(op_name);
  return it == kRules.end() ? nullptr : it->second;
}

// ============================================================================
// LowerCompositeOpsMutator
//
// Generic dispatcher: for every ``var = Call(...)`` AssignStmt (or composite-op
// Call embedded directly in a ReturnStmt), look up a lowering rule via
// ``LookupCompositeRule`` and, if found, replace the statement with a SeqStmts
// containing the rule's primitive decomposition. All other statements pass
// through to the base IRMutator, so the pass is a structural no-op on programs
// that contain no registered composite ops.
//
// The pass is idempotent provided each rule emits only ops that are not
// themselves registered (see the dispatch-table comment above).
// ============================================================================
class LowerCompositeOpsMutator : public IRMutator {
 public:
  explicit LowerCompositeOpsMutator(bool skip_managed_collectives = false)
      : skip_managed_collectives_(skip_managed_collectives) {}

  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    auto tuple = VisitExpr(op->tuple_);
    // Prefer composite-produced MakeTuples recorded privately — do not rely on
    // global var_remap_ for arbitrary `v = (a, b)` assignments.
    if (auto values = ResolveCompositeTuple(tuple)) {
      INTERNAL_CHECK_SPAN(op->index_ >= 0 && static_cast<size_t>(op->index_) < values->elements_.size(),
                          op->span_)
          << "Tuple index out of range: " << op->index_;
      return values->elements_[static_cast<size_t>(op->index_)];
    }
    if (tuple.get() != op->tuple_.get()) {
      return std::make_shared<TupleGetItemExpr>(tuple, op->index_, op->span_);
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) {
      auto visited = IRMutator::VisitStmt_(op);
      auto assign = As<AssignStmt>(visited);
      // Propagate aliases of composite-produced tuples (e.g. alias = pair).
      // Prefer composite_tuples_; if VisitExpr already expanded the RHS to a
      // MakeTuple via var_remap_, also seed var_remap_ for the alias Var so
      // later projections and ConvertToSSA see a concrete tuple.
      if (assign) {
        if (auto mt = ResolveCompositeTuple(assign->value_)) {
          composite_tuples_[op->var_.get()] = mt;
          if (As<MakeTuple>(assign->value_)) {
            var_remap_[op->var_.get()] = assign->value_;
          }
        }
      }
      return visited;
    }
    CompositeLoweringFn rule = LookupRule(call);
    if (!rule) {
      return IRMutator::VisitStmt_(op);
    }

    // Apply var_remap_ (if any) to operand expressions before handing them
    // to the rule.
    std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);

    LoweringBuilder builder(op->var_->name_hint_, temp_counter_);
    ExprPtr result = rule(call, visited_args, builder);
    if (auto mt = As<MakeTuple>(result)) {
      // Record privately for TupleGetItem folding, and also seed var_remap_ so
      // SSA aliases (`alias = pair`) expand through VisitExpr_(Var) without
      // needing a global "any MakeTuple" remap.
      composite_tuples_[op->var_.get()] = mt;
      var_remap_[op->var_.get()] = result;
    }

    auto stmts = builder.TakeStmts();
    // Bind the final result to the original target Var (preserves uses
    // downstream — original AssignStmt's var keeps its name and identity).
    auto final_assign = MutableCopy(op);
    final_assign->value_ = result;
    stmts.push_back(std::move(final_assign));

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    CompositeLoweringFn rule = call ? LookupRule(call) : nullptr;
    if (!rule) {
      return IRMutator::VisitStmt_(op);
    }

    std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);

    LoweringBuilder builder("eval", temp_counter_);
    static_cast<void>(rule(call, visited_args, builder));

    auto stmts = builder.TakeStmts();
    if (stmts.empty()) return op;
    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

  // In SSA form (which LowerCompositeOps assumes), every Call is bound to an
  // AssignStmt and ReturnStmt::value_ holds only Vars — the override above is
  // the sole rewrite site. Standalone / pre-SSA invocations of the pass can
  // still surface a composite-op Call directly inside ReturnStmt::value_
  // (e.g. ``return pl.tile.sin(x)``); without this override those would slip
  // through unlowered. The override lifts each registered Call into a SeqStmts
  // whose last statement is the (possibly mutated) ReturnStmt referencing
  // fresh result Vars.
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<StmtPtr> prelude;
    std::vector<ExprPtr> new_values;
    new_values.reserve(op->value_.size());
    bool changed = false;

    for (std::size_t i = 0; i < op->value_.size(); ++i) {
      INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "ReturnStmt has null value at index " << i;
      ExprPtr value = op->value_[i];
      auto call = As<Call>(value);
      CompositeLoweringFn rule = call ? LookupRule(call) : nullptr;
      if (rule) {
        std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);
        const std::string base = "ret" + std::to_string(i);
        LoweringBuilder builder(base, temp_counter_);
        ExprPtr decomposed = rule(call, visited_args, builder);
        // Bind the decomposed result to a fresh Var so ReturnStmt::value_
        // continues to hold a Var (matches the SSA invariant the rest of the
        // pipeline expects). The Bind appends to the same builder, so a single
        // TakeStmts() drains the rule's prelude + the result binding.
        auto result_var = builder.Bind("result", decomposed, call->span_);
        for (auto& s : builder.TakeStmts()) prelude.push_back(std::move(s));
        new_values.push_back(result_var);
        changed = true;
      } else {
        ExprPtr new_expr = VisitExpr(value);
        INTERNAL_CHECK_SPAN(new_expr, op->span_) << "ReturnStmt value at index " << i << " mutated to null";
        new_values.push_back(new_expr);
        if (new_expr.get() != value.get()) {
          changed = true;
        }
      }
    }

    if (!changed) return op;

    auto new_return = MutableCopy(op);
    new_return->value_ = std::move(new_values);
    if (prelude.empty()) return new_return;
    prelude.push_back(std::move(new_return));
    return std::make_shared<SeqStmts>(std::move(prelude), op->span_);
  }

 private:
  /// True for every ``pld.tensor.*`` cross-rank collective, via the shared
  /// predicate — the CHIP rail's post-condition and the orchestration-reference
  /// verifier must agree with this set, so all three read one list.
  [[nodiscard]] static bool IsTensorCollective(const CallPtr& call) {
    return call && op_predicates::IsManagedTensorCollective(call->op_);
  }

  [[nodiscard]] static bool ShouldSkipManagedCollective(const CallPtr& call) {
    // Managed vs InCore is a function-context property, decided authoritatively
    // by the outer skip_managed_collectives_ flag (set for every Orchestration
    // function), not by arg count or arg[0] type.  Every collective is skipped
    // uniformly here so the flag alone governs which functions defer lowering.
    return IsTensorCollective(call);
  }

  [[nodiscard]] CompositeLoweringFn LookupRule(const CallPtr& call) const {
    if (skip_managed_collectives_ && ShouldSkipManagedCollective(call)) {
      return nullptr;
    }
    return call && call->op_ ? LookupCompositeRule(call->op_->name_) : nullptr;
  }

  std::vector<ExprPtr> VisitArgs(const std::vector<ExprPtr>& args, const Span& span) {
    std::vector<ExprPtr> out;
    out.reserve(args.size());
    for (const auto& arg : args) {
      auto visited = VisitExpr(arg);
      INTERNAL_CHECK_SPAN(visited, span) << "Call argument mutated to null during composite-op lowering";
      out.push_back(std::move(visited));
    }
    return out;
  }

  /// Resolve an expression to a MakeTuple produced by this pass (including
  /// aliases recorded in composite_tuples_). Returns nullptr if not found.
  MakeTuplePtr ResolveCompositeTuple(const ExprPtr& expr) const {
    if (auto mt = As<MakeTuple>(expr)) return mt;
    if (auto var = AsVarLike(expr)) {
      auto it = composite_tuples_.find(var.get());
      if (it != composite_tuples_.end()) return it->second;
    }
    return nullptr;
  }

  std::size_t temp_counter_ = 0;
  bool skip_managed_collectives_{false};
  /// MakeTuples produced by composite lowering rules (and their SSA aliases).
  /// Used only to fold TupleGetItem projections; does not affect global var_remap_.
  std::unordered_map<const Expr*, MakeTuplePtr> composite_tuples_;
};

/// A managed collective is one written in an *orchestration* body — HOST/L3 or
/// CHIP/L2. Neither is tile-lowered, so the composite expansion (which emits
/// tensor-level put/notify/wait) would be illegal there; both defer to their
/// own rail: LowerHostTensorCollectives for HOST, LowerL2TensorCollectives for
/// CHIP. Only an InCore body still expands here.
FunctionPtr TransformLowerCompositeOps(const FunctionPtr& func) {
  const bool skip_managed_collectives =
      func && (func->func_type_ == FunctionType::Orchestration ||
               (func->role_.has_value() && *func->role_ == Role::Orchestrator));
  LowerCompositeOpsMutator mutator(skip_managed_collectives);
  return mutator.VisitFunction(func);
}

}  // namespace

namespace pass {

Pass LowerCompositeOps() {
  return CreateFunctionPass(TransformLowerCompositeOps, "LowerCompositeOps", kLowerCompositeOpsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
