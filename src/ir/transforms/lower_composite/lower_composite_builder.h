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

#ifndef SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_BUILDER_H_
#define SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_BUILDER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace lower_composite {

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
// call-local (see the self-clearing credit-barrier protocol in
// lower_composite_ops_pass.cpp), so each LoweringBuilder instance — one per
// top-level composite-op call — owns its own ``barrier_count_`` that always
// starts at 0.
// ============================================================================
class LoweringBuilder {
 public:
  /// @param base_name    Name hint to derive temp names from (typically the
  ///                     AssignStmt's LHS ``Var`` name).
  /// @param temp_counter Reference to a mutator-owned counter; bumped per Bind.
  LoweringBuilder(std::string base_name, std::size_t& temp_counter);

  LoweringBuilder(std::string base_name, std::size_t& temp_counter, bool nested);

  /// Append an ``AssignStmt`` binding a fresh ``Var`` to ``expr`` and return
  /// the new ``Var`` so it can be used as input to subsequent ops. The
  /// ``qualifier`` is woven into the temp name for debuggability.
  ExprPtr Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span);

  /// Append a side-effecting expression without manufacturing an unused SSA
  /// result. This is used by destination-passing ops whose output buffers are
  /// explicit operands.
  void EmitEval(const ExprPtr& expr, const Span& span);

  // Primitive op builders -- type deduction is delegated to OpRegistry so the
  // result preserves the input TileType's shape/layout/dtype.
  ExprPtr Muls(const ExprPtr& x, float c, const Span& span);
  ExprPtr Adds(const ExprPtr& x, float c, const Span& span);
  ExprPtr Add(const ExprPtr& a, const ExprPtr& b, const Span& span);
  ExprPtr Sub(const ExprPtr& a, const ExprPtr& b, const Span& span);
  ExprPtr Mul(const ExprPtr& a, const ExprPtr& b, const Span& span);
  ExprPtr Reduce(ReduceOp op, const ExprPtr& a, const ExprPtr& b, const Span& span);
  ExprPtr Cast(const ExprPtr& x, DataType to, int mode, const Span& span);

  // ---- Scalar comparison helpers (yield BOOL-typed expressions, suitable as
  //      IfStmt conditions or loop guards). Delegated to the scalar_expr
  //      Make* helpers so operand promotion stays consistent with parser
  //      output.
  ExprPtr NotEq(const ExprPtr& left, const ExprPtr& right, const Span& span);

  ExprPtr Gt(const ExprPtr& left, const ExprPtr& right, const Span& span);

  // ---- Collective-op helpers (DRY extraction for barrier/broadcast/allgather/
  //      reduce_scatter/allreduce) ----

  /// Emit comm-setup preamble: get_comm_ctx, nranks, rank.
  /// Returns a CommSetup struct with the bound expressions for use in
  /// subsequent phases.
  CommSetup EmitCommSetup(const ExprPtr& comm_target, const Span& span);

  /// Emit notify-all loop: for peer in 0..nranks: if peer != my_rank: notify(...)
  /// @param signal       The signal DistributedTensor
  /// @param nranks_idx   Loop bound (INDEX-typed)
  /// @param my_rank      This rank's ID (INT32)
  /// @param notify_op    NotifyOp::kSet or NotifyOp::kAtomicAdd
  /// @param value        Value to notify (e.g., one_i32)
  /// @param suffix       Suffix for loop variable names (e.g., "" or "2" for re-notify)
  /// @param span         Source span for error reporting
  void EmitNotifyAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                     NotifyOp notify_op, const ExprPtr& value, const std::string& suffix, const Span& span);

  /// Overload for 2D signal matrices (e.g. ring allreduce [2*(NR-1), NR]).
  /// @param row_offset   Row index expression for the 2D signal (e.g. ring step var)
  void EmitNotifyAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                     const ExprPtr& row_offset, NotifyOp notify_op, const ExprPtr& value,
                     const std::string& suffix, const Span& span);

  /// Emit wait-all loop: for src in 0..nranks: if src != my_rank: wait(...)
  /// @param signal       The signal DistributedTensor
  /// @param nranks_idx   Loop bound (INDEX-typed)
  /// @param my_rank      This rank's ID (INT32)
  /// @param expected     Expected signal value — the barrier generation (INT32)
  /// @param suffix       Suffix for loop variable names (e.g., "" or "2" for re-wait)
  /// @param span         Source span for error reporting
  void EmitWaitAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                   const ExprPtr& expected, const std::string& suffix, const Span& span);

  /// Overload for 2D signal matrices (e.g. ring allreduce [2*(NR-1), NR]).
  /// @param row_offset   Row index expression for the 2D signal (e.g. ring step var)
  void EmitWaitAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                   const ExprPtr& row_offset, const ExprPtr& expected, const std::string& suffix,
                   const Span& span);

  // ---- Self-clearing credit barrier protocol (see lower_composite_ops_pass.cpp's file-header comment) ----

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
                      const Span& span);

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
                         const Span& span);

  /// 2D-signal overload (ring allreduce, ``[2*(NR-1), NR]``): subtract
  /// ``total_per_row`` from every non-self cell of every one of ``num_rows``
  /// rows. Ring credits every row independently (one per round / sub-chunk
  /// sequence), and every row's sub-chunk loop shares the same bound, so one
  /// symbolic ``total_per_row`` resets all rows uniformly.
  void EmitEpilogueReset(const ExprPtr& signal, const CommSetup& comm, const ExprPtr& num_rows,
                         const ExprPtr& total_per_row, const Span& span);

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
               const Span& span);

  /// Emit a reducing ``for`` loop with one loop-carried accumulator. The
  /// body callback receives a nested builder, the loop variable, and the
  /// accumulator (typed via ``init_value``); it returns the next iteration's
  /// accumulator value. The method returns an expression holding the
  /// post-loop accumulator, ready to feed into subsequent ops.
  ExprPtr EmitForReduce(const std::string& loop_var_name, const ExprPtr& start, const ExprPtr& stop,
                        const ExprPtr& step, const ExprPtr& init_value,
                        const std::function<ExprPtr(LoweringBuilder&, const VarPtr&, const VarPtr&)>& body_fn,
                        const Span& span);

  /// Emit a side-effect-only ``if`` statement:
  ///
  ///     if cond:
  ///         <then_fn stmts>
  ///     [else:
  ///         <else_fn stmts>]
  ///
  /// Pass ``nullptr`` for ``else_fn`` when there is no else branch.
  void EmitIf(const ExprPtr& cond, const std::function<void(LoweringBuilder&)>& then_fn,
              const std::function<void(LoweringBuilder&)>& else_fn, const Span& span);

  /// Emit a value-producing ``if`` statement. Both branches must yield a
  /// value (via their body_fn's ExprPtr return); the method returns an
  /// expression holding the chosen value, ready to feed into subsequent ops.
  ExprPtr EmitIfExpr(const ExprPtr& cond, const std::function<ExprPtr(LoweringBuilder&)>& then_fn,
                     const std::function<ExprPtr(LoweringBuilder&)>& else_fn, const Span& span);

  /// Drain accumulated statements (called by the mutator after the rule
  /// returns).
  std::vector<StmtPtr> TakeStmts();

 private:
  std::string MakeTempName(const std::string& qualifier);

  // Wrap a sequence of body stmts into a single StmtPtr: pass through a sole
  // stmt, wrap multiple into a SeqStmts, and synthesise an empty SeqStmts
  // when the body is empty (a no-op body is still a valid loop / if branch).
  static StmtPtr WrapBodyStmts(std::vector<StmtPtr> body_stmts, const Span& span);

  std::string base_name_;
  std::size_t& temp_counter_;
  bool nested_ = false;
  int64_t barrier_count_ = 0;  ///< Call-local generation counter; see EmitBarrier.
  std::vector<StmtPtr> stmts_;
};

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_BUILDER_H_
