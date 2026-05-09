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
#include <utility>
#include <vector>

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
// FP32 constants for Cody-Waite range reduction + degree-9 odd Horner.
// Values are the verbatim CANN/PyPTO recipe used by the framework reference at
// gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h. They
// are single-precision FP32 literals.
// ============================================================================

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

// ============================================================================
// LowerMathOpsMutator
//
// Walks a function body, replaces every ``var = tile.sin(x)`` or
// ``var = tile.cos(x)`` AssignStmt with a SeqStmts containing the primitive
// decomposition (Cody-Waite range reduction + degree-9 odd Horner polynomial).
// All other statements pass through to the base IRMutator, so the pass is a
// structural no-op on programs that contain no sin/cos.
//
// The lowering is idempotent: the resulting SeqStmts only contains primitive
// tile ops (tile.muls, tile.adds, tile.add, tile.sub, tile.mul, tile.cast),
// none of which the mutator rewrites. Running the pass twice yields the same
// IR.
// ============================================================================
class LowerMathOpsMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) {
      return IRMutator::VisitStmt_(op);
    }
    const std::string& name = call->op_->name_;
    const bool is_sin = (name == "tile.sin");
    const bool is_cos = (name == "tile.cos");
    if (!is_sin && !is_cos) {
      return IRMutator::VisitStmt_(op);
    }

    ValidateTrigCall(call);

    // Recurse on the input expression (preserves any var_remap_ substitution).
    ExprPtr x = VisitExpr(call->args_[0]);

    std::vector<StmtPtr> stmts;
    ExprPtr result = LowerSinCos(x, is_cos, op->var_->name_hint_, call->span_, stmts);

    // Bind the final result to the original target Var (preserves uses
    // downstream — original AssignStmt's var keeps its name and identity).
    auto final_assign = MutableCopy(op);
    final_assign->value_ = result;
    stmts.push_back(std::move(final_assign));

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

  // In SSA form (which LowerMathOps assumes), every Call is bound to an
  // AssignStmt and ReturnStmt::value_ holds only Vars — the override above is
  // the sole rewrite site. Standalone / pre-SSA invocations of the pass can
  // still surface a tile.sin / tile.cos Call directly inside ReturnStmt::value_
  // (e.g. ``return pl.tile.sin(x)``); without this override those would slip
  // through unlowered. The override lifts each trig Call into a SeqStmts whose
  // last statement is the (possibly mutated) ReturnStmt referencing fresh
  // result Vars.
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<StmtPtr> prelude;
    std::vector<ExprPtr> new_values;
    new_values.reserve(op->value_.size());
    bool changed = false;

    for (size_t i = 0; i < op->value_.size(); ++i) {
      INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "ReturnStmt has null value at index " << i;
      ExprPtr value = op->value_[i];
      auto call = As<Call>(value);
      const bool is_trig = call && (call->op_->name_ == "tile.sin" || call->op_->name_ == "tile.cos");
      if (is_trig) {
        ValidateTrigCall(call);
        const bool is_cos = (call->op_->name_ == "tile.cos");
        ExprPtr x = VisitExpr(call->args_[0]);
        const std::string base = "ret" + std::to_string(i);
        ExprPtr decomposed = LowerSinCos(x, is_cos, base, call->span_, prelude);
        // Bind the decomposed result to a fresh Var so ReturnStmt::value_
        // continues to hold a Var (matches the SSA invariant the rest of the
        // pipeline expects).
        auto result_var = Bind(prelude, base, "result", decomposed, call->span_);
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

    StmtPtr new_return;
    if (prelude.empty()) {
      auto copy = MutableCopy(op);
      copy->value_ = std::move(new_values);
      new_return = copy;
    } else {
      auto copy = MutableCopy(op);
      copy->value_ = std::move(new_values);
      prelude.push_back(copy);
      new_return = std::make_shared<SeqStmts>(std::move(prelude), op->span_);
    }
    return new_return;
  }

 private:
  // Shared validator for tile.sin / tile.cos Calls — keeps the AssignStmt and
  // ReturnStmt overrides honest about input shape, type, and dtype.
  static void ValidateTrigCall(const CallPtr& call) {
    const std::string& name = call->op_->name_;
    INTERNAL_CHECK_SPAN(call->args_.size() == 1, call->span_)
        << name << " requires exactly 1 argument, got " << call->args_.size();
    auto in_tile_type = As<TileType>(call->args_[0]->GetType());
    INTERNAL_CHECK_SPAN(in_tile_type, call->span_)
        << name << " requires a TileType argument, got " << call->args_[0]->GetType()->TypeName();
    INTERNAL_CHECK_SPAN(in_tile_type->dtype_ == DataType::FP32, call->span_)
        << name << " is FP32-only, got dtype " << in_tile_type->dtype_.ToString();
  }

  // Mint a unique temp name keyed off the user's target name + a qualifier.
  std::string MakeTempName(const std::string& base, const std::string& qualifier) {
    return auto_name::BuildName(auto_name::GetBaseName(base), qualifier, "tmp", static_cast<int>(temp_id_++));
  }

  // Bind ``call`` to a fresh Var with name ``base + qualifier`` and append the
  // resulting AssignStmt to ``stmts``. Returns the new Var as an ExprPtr so it
  // can be used as input to subsequent ops.
  ExprPtr Bind(std::vector<StmtPtr>& stmts, const std::string& base, const std::string& qualifier,
               const ExprPtr& call_expr, const Span& span) {
    auto var = std::make_shared<Var>(MakeTempName(base, qualifier), call_expr->GetType(), span);
    stmts.push_back(std::make_shared<AssignStmt>(var, call_expr, span));
    return var;
  }

  // -- Primitive op builders --------------------------------------------------
  // Each helper returns a freshly-constructed Call expression. Type deduction
  // is delegated to the OpRegistry so the result preserves the input
  // TileType's shape/layout/dtype.

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

  // -- High-level recipe ------------------------------------------------------
  //
  // ``out = sin(x)`` or ``out = cos(x)`` lowers to:
  //   1. range-reduce x to t in [-pi/2, pi/2], deriving the integer multiple k
  //      of pi (sin) or k of pi-shifted-by-pi/2 (cos)
  //   2. compute sign = (-1)^k = floor(k/2)*4 - 2*k + 1
  //   3. evaluate degree-9 odd Horner polynomial P(t^2) approximating sin(t)/t
  //   4. final = sign * t * P(t^2)
  //
  // The cos path differs from sin in only two places:
  //   - initial k uses round(x/pi + 0.5) with cast mode RINT (vs ROUND for sin)
  //   - +pi/2 head/tail terms are interleaved into the range-reduction subtractions
  ExprPtr LowerSinCos(const ExprPtr& x, bool is_cos, const std::string& base, const Span& span,
                      std::vector<StmtPtr>& stmts) {
    // ---- Step 1: range reduction --------------------------------------------
    ExprPtr k_f;  // FP32 tile holding the integer multiple as a float
    ExprPtr t;    // FP32 tile holding the reduced argument

    if (is_cos) {
      // k_f = float(rint(x * PI_INV + 0.5))
      auto pi_inv_x = Bind(stmts, base, "pi_inv_x", Muls(x, kPiInv, span), span);
      auto k_pre = Bind(stmts, base, "k_pre", Adds(pi_inv_x, kHalf, span), span);
      auto k_i = Bind(stmts, base, "k_i", Cast(k_pre, DataType::INT32, kCastModeRint, span), span);
      k_f = Bind(stmts, base, "k_f", Cast(k_i, DataType::FP32, kCastModeNone, span), span);
    } else {
      // k_f = float(round(x * PI_INV))
      auto pi_inv_x = Bind(stmts, base, "pi_inv_x", Muls(x, kPiInv, span), span);
      auto k_i = Bind(stmts, base, "k_i", Cast(pi_inv_x, DataType::INT32, kCastModeRound, span), span);
      k_f = Bind(stmts, base, "k_f", Cast(k_i, DataType::FP32, kCastModeNone, span), span);
    }

    // t = x - k_f * pi (4-part Cody-Waite). For cos, +pi/2 head/tail are
    // interleaved between PI_C1 and PI_C2, and after PI_C4 respectively.
    auto kpv2 = Bind(stmts, base, "k_pi_v2", Muls(k_f, kPiV2, span), span);
    t = Bind(stmts, base, "t0", Sub(x, kpv2, span), span);
    auto kpc1 = Bind(stmts, base, "k_pi_c1", Muls(k_f, kPiC1, span), span);
    t = Bind(stmts, base, "t1", Sub(t, kpc1, span), span);
    if (is_cos) {
      t = Bind(stmts, base, "t1h", Adds(t, kPiHalfHead, span), span);
    }
    auto kpc2 = Bind(stmts, base, "k_pi_c2", Muls(k_f, kPiC2, span), span);
    t = Bind(stmts, base, "t2", Sub(t, kpc2, span), span);
    auto kpc3 = Bind(stmts, base, "k_pi_c3", Muls(k_f, kPiC3, span), span);
    t = Bind(stmts, base, "t3", Sub(t, kpc3, span), span);
    auto kpc4 = Bind(stmts, base, "k_pi_c4", Muls(k_f, kPiC4, span), span);
    t = Bind(stmts, base, "t4", Sub(t, kpc4, span), span);
    if (is_cos) {
      t = Bind(stmts, base, "t4t", Adds(t, kPiHalfTail, span), span);
    }

    // ---- Step 2: sign = floor(k_f / 2) * 4 + k_f * (-2) + 1 ------------------
    auto half_k = Bind(stmts, base, "half_k", Muls(k_f, kHalf, span), span);
    auto floor_hk_i =
        Bind(stmts, base, "floor_hk_i", Cast(half_k, DataType::INT32, kCastModeFloor, span), span);
    auto floor_hk_f =
        Bind(stmts, base, "floor_hk_f", Cast(floor_hk_i, DataType::FP32, kCastModeNone, span), span);
    auto floor_x4 = Bind(stmts, base, "floor_x4", Muls(floor_hk_f, kM4, span), span);
    auto neg2_k = Bind(stmts, base, "neg2_k", Muls(k_f, kNeg2, span), span);
    auto sign_pre = Bind(stmts, base, "sign_pre", Add(floor_x4, neg2_k, span), span);
    auto sign = Bind(stmts, base, "sign", Adds(sign_pre, kOne, span), span);

    // ---- Step 3: Horner P(t^2) = (((R0*t^2 + R1)*t^2 + R2)*t^2 + R3)*t^2 + 1
    auto t2 = Bind(stmts, base, "t2sq", Mul(t, t, span), span);
    auto p = Bind(stmts, base, "p_r0", Muls(t2, kR0, span), span);
    p = Bind(stmts, base, "p_r1", Adds(p, kR1, span), span);
    p = Bind(stmts, base, "p_t2_r1", Mul(p, t2, span), span);
    p = Bind(stmts, base, "p_r2", Adds(p, kR2, span), span);
    p = Bind(stmts, base, "p_t2_r2", Mul(p, t2, span), span);
    p = Bind(stmts, base, "p_r3", Adds(p, kR3, span), span);
    p = Bind(stmts, base, "p_t2_r3", Mul(p, t2, span), span);
    p = Bind(stmts, base, "p_one", Adds(p, kOne, span), span);

    // ---- Step 4: out = sign * t * P(t^2) -------------------------------------
    auto t_p = Bind(stmts, base, "t_p", Mul(t, p, span), span);
    return Mul(sign, t_p, span);
  }

  size_t temp_id_ = 0;
};

FunctionPtr TransformLowerMathOps(const FunctionPtr& func) {
  LowerMathOpsMutator mutator;
  return mutator.VisitFunction(func);
}

}  // namespace

namespace pass {

Pass LowerMathOps() {
  return CreateFunctionPass(TransformLowerMathOps, "LowerMathOps", kLowerMathOpsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
