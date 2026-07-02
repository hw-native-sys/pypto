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
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"

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
// The temp counter is borrowed from the mutator so distinct composite-op calls
// in the same function get distinct temp names.
// ============================================================================
class LoweringBuilder {
 public:
  /// @param base_name    Name hint to derive temp names from (typically the
  ///                     AssignStmt's LHS ``Var`` name).
  /// @param temp_counter Reference to a mutator-owned counter; bumped per Bind.
  LoweringBuilder(std::string base_name, std::size_t& temp_counter)
      : base_name_(std::move(base_name)), temp_counter_(temp_counter) {}

  /// Append an ``AssignStmt`` binding a fresh ``Var`` to ``expr`` and return
  /// the new ``Var`` so it can be used as input to subsequent ops. The
  /// ``qualifier`` is woven into the temp name for debuggability.
  ExprPtr Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span) {
    auto var = std::make_shared<Var>(MakeTempName(qualifier), expr->GetType(), span);
    stmts_.push_back(std::make_shared<AssignStmt>(var, expr, span));
    return var;
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

  /// Emit wait-all loop: for src in 0..nranks: if src != my_rank: wait(...)
  /// @param signal       The signal DistributedTensor
  /// @param nranks_idx   Loop bound (INDEX-typed)
  /// @param my_rank      This rank's ID (INT32)
  /// @param expected     Expected signal value (e.g., one_i32 or two_i32)
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
    LoweringBuilder body_builder(base_name_, temp_counter_);
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
    LoweringBuilder body_builder(base_name_, temp_counter_);
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
    LoweringBuilder then_builder(base_name_, temp_counter_);
    then_fn(then_builder);
    auto then_body = WrapBodyStmts(then_builder.TakeStmts(), span);

    std::optional<StmtPtr> else_body = std::nullopt;
    if (else_fn) {
      LoweringBuilder else_builder(base_name_, temp_counter_);
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
    LoweringBuilder then_builder(base_name_, temp_counter_);
    ExprPtr then_val = then_fn(then_builder);
    INTERNAL_CHECK_SPAN(then_val, span) << "EmitIfExpr then_fn must return the yielded value";
    then_builder.stmts_.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{then_val}, span));
    auto then_body = WrapBodyStmts(then_builder.TakeStmts(), span);

    LoweringBuilder else_builder(base_name_, temp_counter_);
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
// ``pld.tensor.allreduce`` lowering rule
//
// In-place all-reduce of a window-bound DistributedTensor across every rank
// of its comm group. Expands the single composite Call into the 4-phase
// decomposition validated by the hand-written reference in
// ``tests/st/distributed/test_l3_allreduce.py`` (``reduce_step``):
//
//   Phase 2a: for peer in 0..nranks:
//               if peer != my_rank:
//                 pld.system.notify(signal, peer, [my_rank, 0], 1, op=AtomicAdd)
//   Phase 2b: for src  in 0..nranks:
//               if src != my_rank:
//                 pld.system.wait(signal, [src, 0], 1, cmp=Ge)
//   Phase 3 : acc = tile.load(target, [0..], shape)
//             for peer in 0..nranks:
//               if peer != my_rank:
//                 recv = pld.tile.remote_load(target, peer, [0..], shape)
//                 acc = tile.add(acc, recv)
//               else:
//                 acc = acc
//   Phase 4 : tile.store(acc, [0..], target)
//
// The loop bound ``nranks`` is read at runtime via
// ``pld.system.nranks(pld.system.get_comm_ctx(target))`` so the lowering does
// not depend on CommGroup materialisation (which runs later in the pipeline).
// First-version implementation: ``ReduceOp::kSum`` only — the deducer rejects
// other variants before the rule is invoked, so the rule asserts that
// invariant rather than dispatching.
//
// The Call's source-level form is the in-place rebind idiom shared with
// ``pl.store``:
//
//     pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum)
//
// so the rule returns the (post-reduce) ``target`` ExprPtr and lets the
// mutator bind it to the AssignStmt's LHS Var.
// ============================================================================

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

  // First-version constraint — Max / Min / Prod lowerings not yet implemented.
  auto op_value = GetRequiredKwarg<int>(call->kwargs_, "op", "pld.tensor.allreduce");
  INTERNAL_CHECK_SPAN(op_value == static_cast<int>(ReduceOp::kSum), span)
      << "pld.tensor.allreduce lowering supports ReduceOp::kSum only (got int " << op_value
      << ") — deducer should have rejected this";

  const std::size_t ndim = target_type->shape_.size();

  // ---- Pre-build expressions shared across phases ----
  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  // Loop bounds: INDEX (must agree across start/stop/step). Notify's `value`
  // and wait's `expected` are INT32 per the Python builder's int_dtype
  // override — keep separate constants for those distinct slots.
  //
  // Signal scheme: hybrid Set / AtomicAdd, both waits use ``WaitCmp::kGe``.
  // Phase 2a uses ``Set value=1`` (race-free: each cell has exactly one
  // writer, the corresponding peer); Phase 2b waits for ``>= 1``. Phase 3.5
  // uses ``AtomicAdd 1`` for the post-reduce barrier (cell 1 → 2) with wait
  // for ``>= 2``.
  //
  // ``kGe`` (not ``kEq``) is load-bearing. The cell is monotonically
  // increasing within a single call, but the observer (the waiting rank)
  // is NOT guaranteed to read each intermediate value: if a faster peer
  // races ahead — e.g. rank A pauses between its own Phase 2a and 2b
  // while rank B completes 2a, 2b, the full Phase 3 (remote loads,
  // microseconds), and Phase 3.5a — then by the time A polls its
  // cell[B], it has already been advanced from 1 (B's 2a Set) to 2
  // (B's 3.5a AtomicAdd). ``kEq(==1)`` would never unblock; ``kGe(>=1)``
  // does. The hand-written reference at
  // ``tests/st/distributed/test_l3_allreduce.py`` uses ``Ge(1)`` for
  // exactly this reason and survives the same race window.
  //
  // The cells end the call at 2, so the buffer is **not reusable** across
  // multiple allreduce calls without per-call reallocation — a stale ``2``
  // would let the next call's ``Ge(1)`` Phase 2b pass before any peer
  // notifies, breaking the barrier. The symmetric ``Set value=0`` reset
  // path would let the buffer self-clear, but on-board ``TWAIT(==0)`` did
  // not unblock reliably in our trials (P=4 deadlocked on AICPU stream
  // sync — see PTOAS issue #797). Until that runtime path is verified,
  // callers needing back-to-back allreduces must allocate a fresh signal
  // buffer per call. The user-facing DSL docstring repeats this contract.
  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);
  auto two_i32 = std::make_shared<ConstInt>(2, DataType::INT32, span);
  auto zero_offsets = tile_conversion_utils::MakeZeroOffsets(ndim, span);
  auto shape_tuple = tile_conversion_utils::MakeShapeTuple(target_type->shape_, span);

  // ---- Phase 2a: notify all peers (Set cell[my_rank, 0] on each peer to 1) ----
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kSet, one_i32, "", span);

  // ---- Phase 2b: wait on every peer's signal slot (cell[src, 0] >= 1) ----
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, one_i32, "", span);

  // ---- Phase 3: acc = load(target); for peer != my_rank: acc += remote_load(peer) ----
  // tile.load needs the valid_shapes arg (same as shapes when omitted) plus
  // target_memory / transpose kwargs — mirrors `pl.load(...)`.
  auto acc_initial = b.Bind("acc_initial",
                            reg.Create("tile.load", {target, zero_offsets, shape_tuple, shape_tuple},
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
                                                   {target, peer, zero_offsets, shape_tuple}, {}, span),
                  span);
              // Bind the add result so codegen sees a named tile buffer to
              // write into (pto.tadd lowers to `ins(...) outs(<bound>)`);
              // yielding a raw Call leaves outs() empty and MLIR rejects it.
              return then_body.Bind("acc_next", then_body.Add(acc, recv, span), span);
            },
            [&](LoweringBuilder& /*else_body*/) -> ExprPtr {
              // peer == my_rank: pass acc through unchanged.
              return acc;
            },
            span);
      },
      span);

  // ---- Phase 3.5: post-reduce barrier (AtomicAdd 1 → wait >= 2) ----
  // Phase 4 overwrites every rank's slot of ``target``; without this barrier,
  // a fast rank could write its reduced value back to ``target`` while a slow
  // rank is still in Phase 3 reading the original Phase-1 staged data from
  // the same slot via ``pld.tile.remote_load`` — a write-after-read hazard
  // that surfaces as wrong sums on slower ranks.
  //
  // Reuse the same signal cells: each peer atomic-adds 1 again, raising my
  // cell from 1 to 2, so the second wait checks ``cell >= 2``. ``kGe`` (not
  // ``kEq``) is required for the same race-window reason as Phase 2b: a
  // faster peer can advance the cell past 2 before the slower rank gets
  // around to polling — but since the buffer is single-shot per call and
  // no peer adds more than ``+1`` here, the cell tops out at 2 and
  // ``>= 2`` is both safe and tight. See the prelude comment block above
  // and the hand-written reference at
  // ``tests/st/distributed/test_l3_allreduce.py``.
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32, "2", span);
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, two_i32, "2", span);

  // ---- Phase 4: store the reduced accumulator back into the target slot ----
  b.Bind("store_ret", reg.Create("tile.store", {acc_final, zero_offsets, target}, {}, span), span);

  // In-place semantics: the rebind LHS receives the (post-reduce) target view.
  return target;
}

// ============================================================================
// ``pld.tensor.broadcast`` lowering rule
//
// Broadcast root rank's data to every rank:
//   Phase 2a: notify-all (Set 1)
//   Phase 2b: wait-all  (Ge 1)
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

  auto root_value = GetRequiredKwarg<int>(call->kwargs_, "root", "pld.tensor.broadcast");

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);
  auto root_expr = std::make_shared<ConstInt>(root_value, DataType::INT32, span);

  // ---- Phase 2a: notify-all ----
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kSet, one_i32, "", span);

  // ---- Phase 2b: wait-all ----
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, one_i32, "", span);

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

  // In-place rebind: return target so the LHS Var holds the post-broadcast view.
  return target;
}

// ============================================================================
// ``pld.tensor.allgather`` lowering rule
//
// All-gather: gather data from all ranks and write the concatenated result
// into a user-provided output Tensor.  Fully N-rank general — NR is read from
// the target's compile-time shape at lowering time.
//
//   arg[0] = local_data  — Tensor [1, SIZE] (plain) or Tile [1, SIZE], this rank's chunk
//   arg[1] = target      — DistributedTensor [NR, SIZE], staging window
//   arg[2] = signal      — DistributedTensor INT32, cross-rank barrier
//   arg[3] = out         — Tensor [1, NR*SIZE], output buffer
//
// Phases (aligned with simpler allgather_distributed reference):
//   0.  tile.load(local_data, [0,0], [1,SIZE])   — when local_data is a
//       Tensor (emit a Tile from the plain input); skipped when Tile
//   1.  tile.store(stage_tile, [0, 0], target)    — stage-in (each rank's
//       private HCCL window starts at local row 0)
//   2a. notify-all (Set 1)
//   2b. wait-all  (Ge 1)
//   3.  for r in 0..NR-1:
//         pld.tile.get(out, peer=r, target, stage,
//                      dst_offsets=[0, r*SIZE], src_offsets=[0,0], shape=[1,SIZE])
//       — transfer each peer's chunk directly into out at column offset
//         [0, r*SIZE]; one shared [1, SIZE] VEC staging tile, no tile.concat
//       return out  (Tensor rebind)
//
// Self-read falls out of the same pld.tile.get path via HCCL identity
// mapping (CommRemotePtr returns local ptr for peer == my_rank).
// ============================================================================

ExprPtr LowerTensorAllGatherRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 4, span)
      << "pld.tensor.allgather rule expects 4 args (local_data, target, signal, out), got " << args.size();
  const auto& local_data = args[0];
  const auto& target = args[1];
  const auto& signal = args[2];
  const auto& out = args[3];

  // local_data may be a Tile (pre-loaded by user) or a Tensor (intrinsic
  // handles the load internally).  Record which so we can emit tile.load
  // after the shared setup expressions are available.
  bool is_tensor_input = (As<TensorType>(local_data->GetType()) != nullptr);
  INTERNAL_CHECK_SPAN(As<TileType>(local_data->GetType()) || is_tensor_input, span)
      << "pld.tensor.allgather local_data must be TileType or TensorType, got "
      << local_data->GetType()->TypeName();
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.allgather target must be DistributedTensorType (deducer-rejected otherwise)";
  INTERNAL_CHECK_SPAN(target_type->shape_.size() == 2, span)
      << "pld.tensor.allgather target must be 2D [NR, SIZE]";

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);

  // Per-chunk shape: [1, SIZE] where SIZE = target.shape[1].
  auto size_expr = target_type->shape_[1];
  auto chunk_shape = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span), size_expr}, span);

  // Per the simpler allgather reference (examples/workers/l3/allgather_distributed/),
  // every rank stores its chunk at local offset [0, 0] of its private HCCL
  // window.  The DistributedTensor shape [NR, SIZE] is a logical view;
  // physically each rank holds one row at local row 0.
  auto zero_row_offsets =
      std::make_shared<MakeTuple>(std::vector<ExprPtr>{std::make_shared<ConstInt>(0, DataType::INDEX, span),
                                                       std::make_shared<ConstInt>(0, DataType::INDEX, span)},
                                  span);

  // ---- Phase 0: load (Tensor → Tile, when local_data is a Tensor) ----
  ExprPtr stage_tile;
  if (is_tensor_input) {
    stage_tile = b.Bind("local_tile",
                        reg.Create("tile.load", {local_data, zero_row_offsets, chunk_shape, chunk_shape},
                                   {{"target_memory", MemorySpace::Vec}}, span),
                        span);
  } else {
    stage_tile = local_data;
  }

  // ---- Phase 1: stage-in (stage_tile → target[0, 0]) ----
  b.Bind("stage_in", reg.Create("tile.store", {stage_tile, zero_row_offsets, target}, {}, span), span);

  // ---- Phase 2a: notify-all ----
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kSet, one_i32, "", span);

  // ---- Phase 2b: wait-all ----
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, one_i32, "", span);

  // ---- Phase 3: gather — per-peer pld.tile.get directly into output Tensor ----
  // Each peer's chunk [1, SIZE] is transferred from the window buffer directly
  // into the output Tensor at a rank-ordered column offset [0, r*SIZE] via
  // pld.tile.get (subregion form).  A single VEC staging tile [1, SIZE] is
  // shared across all peers — no tile.concat, each intermediate fits in UB
  // regardless of NR or SIZE.
  //
  // The loop bound is the runtime nranks (comm.nranks_idx), matching the
  // notify/wait phases.  This keeps the gather consistent with the barrier
  // phases regardless of the comm-group size, avoiding a mismatch between
  // compile-time target.shape[0] and the actual world size.

  auto gather_stage =
      b.Bind("ag_stage",
             reg.Create("tile.create", {chunk_shape},
                        {{"dtype", target_type->dtype_}, {"target_memory", MemorySpace::Vec}}, span),
             span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  b.EmitFor(
      "r", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& r_var) {
        // Column dst offset: [0, r*SIZE] — rank-ordered placement.
        auto col_offset_expr = MakeMul(r_var, size_expr, span);
        auto dst_offsets = std::make_shared<MakeTuple>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(0, DataType::INDEX, span), col_offset_expr},
            span);

        body.Bind("get",
                  reg.Create("pld.tile.get",
                             {out, r_var, target, gather_stage, dst_offsets, zero_row_offsets, chunk_shape},
                             {}, span),
                  span);
      },
      span);

  // Return the output Tensor — the lowering writes directly into it.
  return out;
}

// ============================================================================
// ``pld.tensor.reduce_scatter`` lowering rule
//
// Reduce-scatter: each rank holds NR chunks; rank r receives reduced chunk r.
// Target shape [NR, SIZE].  5-phase decomposition matching allreduce:
//   Phase 2a:  notify-all (Set 1)
//   Phase 2b:  wait-all (Ge 1)
//   Phase 3:   acc = load(target, [my_rank, 0], [1, SIZE])
//              for peer != my_rank:
//                  recv = remote_load(target, peer, [my_rank, 0], [1, SIZE])
//                  acc = add(acc, recv)
//   Phase 3.5a: notify-all (AtomicAdd 1)  — WAR prevention
//   Phase 3.5b: wait-all (Ge 2)
//   Phase 4:   tile.store(acc, [my_rank, 0], target)
// Returns target (in-place rebind).  kSum only (first version).
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

  auto op_value = GetRequiredKwarg<int>(call->kwargs_, "op", "pld.tensor.reduce_scatter");
  INTERNAL_CHECK_SPAN(op_value == static_cast<int>(ReduceOp::kSum), span)
      << "pld.tensor.reduce_scatter lowering supports ReduceOp::kSum only (got int " << op_value << ")";

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);
  auto two_i32 = std::make_shared<ConstInt>(2, DataType::INT32, span);

  // Per-chunk shape: [1, SIZE] where SIZE = target.shape[1].
  auto size_expr = target_type->shape_[1];
  auto chunk_shape = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span), size_expr}, span);

  // Helper: data offset [my_rank, 0] — each rank reads/writes its own row.
  auto my_data_offsets = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{comm.my_rank, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

  // ---- Phase 2a: notify-all ----
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kSet, one_i32, "", span);

  // ---- Phase 2b: wait-all ----
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, one_i32, "", span);

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
              return then_body.Bind("acc_next", then_body.Add(acc, recv, span), span);
            },
            [&](LoweringBuilder&) -> ExprPtr { return acc; }, span);
      },
      span);

  // ---- Phase 3.5: post-reduce barrier (AtomicAdd 1 → wait Ge 2) ----
  // Same WAR hazard as allreduce: fast rank could overwrite its row before
  // slow rank reads it.  See allreduce lowering for full rationale.
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32, "2", span);
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, two_i32, "2", span);

  // ---- Phase 4: store reduced chunk back into target[my_rank, 0] ----
  b.Bind("store_ret", reg.Create("tile.store", {acc_final, my_data_offsets, target}, {}, span), span);

  return target;
}

// ============================================================================
// ``pld.tensor.barrier`` lowering rule
//
// Cross-rank barrier: notify-all (Set 1) then wait-all (Ge 1).  Pure
// synchronisation — no data movement.  Returns the signal expression so the
// rebind idiom (``sig = pld.tensor.barrier(sig)``) matches allreduce.
// ============================================================================

ExprPtr LowerTensorBarrierRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 1, span) << "pld.tensor.barrier rule expects 1 arg, got " << args.size();
  const auto& signal = args[0];
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  INTERNAL_CHECK_SPAN(signal_type, span)
      << "pld.tensor.barrier signal must be DistributedTensorType (deducer-rejected otherwise)";

  auto comm = b.EmitCommSetup(signal, span);

  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);

  // ---- Phase 1: notify-all (Set cell[my_rank, 0] on each peer to 1) ----
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kSet, one_i32, "", span);

  // ---- Phase 2: wait-all (cell[src, 0] >= 1) ----
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, one_i32, "", span);

  // Rebind: return the signal so the LHS Var retains the DistributedTensor view.
  return signal;
}

// ============================================================================
// ``pld.tensor.all_to_all`` lowering rule
//
// Symmetric all-to-all: every rank sends a distinct chunk to every other rank.
// 3-phase decomposition matching the simpler all_to_all_distributed reference:
//
//   Phase 1 (stage-in): for dest in 0..NR-1:
//       tile.load(input, [dest, 0], [1, SIZE]) → stage_tile
//       tile.store(stage_tile, [dest, 0], target)  // local window row dest
//
//   Phase 2a: notify-all (Set 1)
//   Phase 2b: wait-all  (Ge 1)
//
//   Phase 3 (exchange): for src in 0..NR-1:
//       pld.tile.get(out, peer=src, target, stage_tile,
//                    dst_offsets=[src, 0], src_offsets=[my_rank, 0],
//                    shape=[1, SIZE])
//
// Input layout:  input[dest, :] = chunk destined for rank dest.
// Output layout: output[src, :] = chunk received from rank src.
// ============================================================================

ExprPtr LowerTensorAllToAllRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 4, span)
      << "pld.tensor.all_to_all rule expects 4 args (input, target, signal, out), got " << args.size();
  const auto& input = args[0];
  const auto& target = args[1];
  const auto& signal = args[2];
  const auto& out = args[3];

  // input must be a Tensor [NR, SIZE].
  INTERNAL_CHECK_SPAN(As<TensorType>(input->GetType()), span)
      << "pld.tensor.all_to_all input must be TensorType, got " << input->GetType()->TypeName();
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.all_to_all target must be DistributedTensorType (deducer-rejected otherwise)";
  INTERNAL_CHECK_SPAN(target_type->shape_.size() == 2, span)
      << "pld.tensor.all_to_all target must be 2D [NR, SIZE]";

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);

  // Per-chunk shape: [1, SIZE] where SIZE = target.shape[1].
  auto size_expr = target_type->shape_[1];
  auto chunk_shape = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span), size_expr}, span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  // ---- Phase 1: stage-in — copy each destination chunk into the local
  //      window.  rank r writes input[dest, :] to target[dest, 0] so
  //      every peer can later read the slice destined for them at
  //      target[my_rank, 0].
  b.EmitFor(
      "dest", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& dest_var) {
        auto dest_row_offsets = std::make_shared<MakeTuple>(
            std::vector<ExprPtr>{dest_var, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

        auto stage_tile =
            body.Bind("aa_load",
                      reg.Create("tile.load", {input, dest_row_offsets, chunk_shape, chunk_shape},
                                 {{"target_memory", MemorySpace::Vec}}, span),
                      span);

        body.Bind("aa_stage", reg.Create("tile.store", {stage_tile, dest_row_offsets, target}, {}, span),
                  span);
      },
      span);

  // ---- Phase 2a: notify-all ----
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kSet, one_i32, "", span);

  // ---- Phase 2b: wait-all ----
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, one_i32, "", span);

  // ---- Phase 3: exchange — read chunk my_rank from every peer's scratch.
  //      Peer src staged the chunk for me at target[my_rank, 0] in its own
  //      local window.  I read that slot and write it to out[src, 0].
  auto recv_stage =
      b.Bind("aa_recv",
             reg.Create("tile.create", {chunk_shape},
                        {{"dtype", target_type->dtype_}, {"target_memory", MemorySpace::Vec}}, span),
             span);

  // Offsets for the peer-read: read at [my_rank, 0] on the peer's window.
  auto my_rank_offsets = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{comm.my_rank, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

  b.EmitFor(
      "src", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& src_var) {
        // Write to out[src, 0] — rank-ordered output.
        auto dst_offsets = std::make_shared<MakeTuple>(
            std::vector<ExprPtr>{src_var, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

        body.Bind("aa_get",
                  reg.Create("pld.tile.get",
                             {out, src_var, target, recv_stage, dst_offsets, my_rank_offsets, chunk_shape},
                             {}, span),
                  span);
      },
      span);

  // ---- Phase 3.5: post-exchange barrier (AtomicAdd 1 → wait >= 2) ----
  // Phase 3 writes exchange results into out, which is a plain tensor.  However
  // target (the HCCL window) is still accessible to peers via pld.tile.remote_load
  // during Phase 3.  Without this barrier, a fast rank that returns from
  // all_to_all and immediately starts a second collective reusing target
  // could overwrite the staging slots while a slow peer is still remote-reading
  // them — a write-after-read hazard matching allreduce's Phase 3.5 rationale.
  auto two_i32 = std::make_shared<ConstInt>(2, DataType::INT32, span);
  b.EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32, "2", span);
  b.EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, two_i32, "2", span);

  return out;
}

// ----------------------------------------------------------------------------
// Composite-op dispatch table.
//
// ``LowerCompositeOps`` is a generic dispatcher: it rewrites a ``var = Call(...)``
// AssignStmt (or a composite-op Call embedded directly in a ReturnStmt) only
// when the callee name appears here. Adding a new composite op = add a rule
// function above + one row in ``kRules``; the mutator below needs no change.
//
// Today the rules are ``tile.sin`` / ``tile.cos`` and ``pld.tensor.*``
// distributed collectives. Host-level allreduce is skipped here and lowered
// later by LowerHostTensorCollectives. The pass is idempotent provided each
// rule emits only ops not listed here.
//
// When the table grows past a handful of entries — or a rule wants its own
// translation unit — promote this back to a standalone registry under
// ``src/ir/transforms/composite_ops/``.
// ----------------------------------------------------------------------------
CompositeLoweringFn LookupCompositeRule(const std::string& op_name) {
  static const std::unordered_map<std::string, CompositeLoweringFn> kRules = {
      {"tile.sin", &LowerSinRule},
      {"tile.cos", &LowerCosRule},
      {"pld.tensor.allreduce", &LowerTensorAllReduceRule},
      {"pld.tensor.allgather", &LowerTensorAllGatherRule},
      {"pld.tensor.reduce_scatter", &LowerTensorReduceScatterRule},
      {"pld.tensor.barrier", &LowerTensorBarrierRule},
      {"pld.tensor.broadcast", &LowerTensorBroadcastRule},
      {"pld.tensor.all_to_all", &LowerTensorAllToAllRule},
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
  explicit LowerCompositeOpsMutator(bool skip_host_collectives = false)
      : skip_host_collectives_(skip_host_collectives) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) {
      return IRMutator::VisitStmt_(op);
    }
    CompositeLoweringFn rule = LookupRule(call);
    if (!rule) {
      return IRMutator::VisitStmt_(op);
    }
    CheckAllReduceLoopUse(call);

    // Apply var_remap_ (if any) to operand expressions before handing them
    // to the rule.
    std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);

    LoweringBuilder builder(op->var_->name_hint_, temp_counter_);
    ExprPtr result = rule(call, visited_args, builder);

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
    CheckAllReduceLoopUse(call);

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
        CheckAllReduceLoopUse(call);
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

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    ++repeating_scope_depth_;
    auto result = IRMutator::VisitStmt_(op);
    --repeating_scope_depth_;
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    ++repeating_scope_depth_;
    auto result = IRMutator::VisitStmt_(op);
    --repeating_scope_depth_;
    return result;
  }

 private:
  [[nodiscard]] static bool ShouldSkipHostCollective(const CallPtr& call) {
    if (!call || !call->op_) return false;
    // pld.tensor.allgather overloads: skip only the 2-arg HOST builtin form;
    // the 4-arg InCore composite form must still be lowered by this pass.
    if (IsOp(call, "pld.tensor.allgather")) {
      return call->args_.size() == 2;
    }
    return IsOp(call, "pld.tensor.allreduce") || IsOp(call, "pld.tensor.barrier") ||
           IsOp(call, "pld.tensor.broadcast") || IsOp(call, "pld.tensor.reduce_scatter");
  }

  [[nodiscard]] CompositeLoweringFn LookupRule(const CallPtr& call) const {
    if (skip_host_collectives_ && ShouldSkipHostCollective(call)) {
      return nullptr;
    }
    return call && call->op_ ? LookupCompositeRule(call->op_->name_) : nullptr;
  }

  void CheckAllReduceLoopUse(const CallPtr& call) const {
    if (!call || !call->op_ || !IsOp(call, "pld.tensor.allreduce")) return;
    CHECK_SPAN(repeating_scope_depth_ == 0, call->span_)
        << "pld.tensor.allreduce is not supported inside a for/while loop. "
           "The signal protocol is single-use and cannot reuse a signal across dynamic invocations.";
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

  std::size_t temp_counter_ = 0;
  int repeating_scope_depth_ = 0;
  bool skip_host_collectives_{false};
};

FunctionPtr TransformLowerCompositeOps(const FunctionPtr& func) {
  const bool skip_host_collectives = func && func->level_.has_value() && *func->level_ == Level::HOST &&
                                     (func->func_type_ == FunctionType::Orchestration ||
                                      (func->role_.has_value() && *func->role_ == Role::Orchestrator));
  LowerCompositeOpsMutator mutator(skip_host_collectives);
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
