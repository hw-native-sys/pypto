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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// LoweringBuilder
//
// Per-call scratchpad handed to a composite-lowering rule. A rule appends one
// ``AssignStmt`` per intermediate temp via ``Bind`` and returns the final
// result ``ExprPtr``; the mutator wraps that result in the original target
// ``Var`` (or a fresh result ``Var`` for ``ReturnStmt`` calls) before splicing
// the accumulated statements into the surrounding sequence.
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

  /// Drain accumulated statements (called by the mutator after the rule
  /// returns).
  std::vector<StmtPtr> TakeStmts() { return std::move(stmts_); }

 private:
  std::string MakeTempName(const std::string& qualifier) {
    return auto_name::BuildName(auto_name::GetBaseName(base_name_), qualifier, "tmp",
                                static_cast<int>(temp_counter_++));
  }

  std::string base_name_;
  std::size_t& temp_counter_;
  std::vector<StmtPtr> stmts_;
};

// Signature for a composite-lowering rule.
//
// @param args     Visited operand expressions (var-remap already applied).
// @param span     Source location of the original call.
// @param builder  Scratchpad: rule appends intermediate temps via builder.Bind
//                 and returns the final result expression.
// @return Final result expression. The mutator binds this to the target ``Var``
//         and splices the builder's accumulated statements before it.
using CompositeLoweringFn = ExprPtr (*)(const std::vector<ExprPtr>& args, const Span& span,
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

ExprPtr LowerSinRule(const std::vector<ExprPtr>& args, const Span& span, LoweringBuilder& builder) {
  ValidateTrigArgs(args, span, "tile.sin");
  return LowerSinCos(args[0], /*is_cos=*/false, builder, span);
}

ExprPtr LowerCosRule(const std::vector<ExprPtr>& args, const Span& span, LoweringBuilder& builder) {
  ValidateTrigArgs(args, span, "tile.cos");
  return LowerSinCos(args[0], /*is_cos=*/true, builder, span);
}

// ----------------------------------------------------------------------------
// Composite-op dispatch table.
//
// ``LowerCompositeOps`` is a generic dispatcher: it rewrites a ``var = Call(...)``
// AssignStmt (or a composite-op Call embedded directly in a ReturnStmt) only
// when the callee name appears here. Adding a new composite op = add a rule
// function above + one row in ``kRules``; the mutator below needs no change.
//
// Today the only rules are ``tile.sin`` / ``tile.cos``. The pass is idempotent
// provided each rule emits only ops not listed here (the sin/cos recipe emits
// only ``tile.muls`` / ``tile.adds`` / ``tile.add`` / ``tile.sub`` / ``tile.mul`` /
// ``tile.cast``).
//
// When the table grows past a handful of entries — or a rule wants its own
// translation unit — promote this back to a standalone registry under
// ``src/ir/transforms/composite_ops/``.
// ----------------------------------------------------------------------------
CompositeLoweringFn LookupCompositeRule(const std::string& op_name) {
  static const std::unordered_map<std::string, CompositeLoweringFn> kRules = {
      {"tile.sin", &LowerSinRule},
      {"tile.cos", &LowerCosRule},
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
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) {
      return IRMutator::VisitStmt_(op);
    }
    CompositeLoweringFn rule = LookupCompositeRule(call->op_->name_);
    if (!rule) {
      return IRMutator::VisitStmt_(op);
    }

    // Apply var_remap_ (if any) to operand expressions before handing them
    // to the rule.
    std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);

    LoweringBuilder builder(op->var_->name_hint_, temp_counter_);
    ExprPtr result = rule(visited_args, call->span_, builder);

    auto stmts = builder.TakeStmts();
    // Bind the final result to the original target Var (preserves uses
    // downstream — original AssignStmt's var keeps its name and identity).
    auto final_assign = MutableCopy(op);
    final_assign->value_ = result;
    stmts.push_back(std::move(final_assign));

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
      CompositeLoweringFn rule = call ? LookupCompositeRule(call->op_->name_) : nullptr;
      if (rule) {
        std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);
        const std::string base = "ret" + std::to_string(i);
        LoweringBuilder builder(base, temp_counter_);
        ExprPtr decomposed = rule(visited_args, call->span_, builder);
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
};

FunctionPtr TransformLowerCompositeOps(const FunctionPtr& func) {
  LowerCompositeOpsMutator mutator;
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
