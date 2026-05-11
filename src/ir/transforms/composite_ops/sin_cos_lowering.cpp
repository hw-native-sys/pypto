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

// FP32-only ``tile.sin`` / ``tile.cos`` lowering rules. Registered with
// ``CompositeLoweringRegistry`` so the generic ``LowerCompositeOps`` pass
// dispatches by op name.
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

#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/composite_lowering_registry.h"
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
  ExprPtr k_f;  // FP32 tile holding the integer multiple as a float
  ExprPtr t;    // FP32 tile holding the reduced argument

  if (is_cos) {
    // k_f = float(rint(x * PI_INV + 0.5))
    auto pi_inv_x = b.Bind("pi_inv_x", b.Muls(x, kPiInv, span), span);
    auto k_pre = b.Bind("k_pre", b.Adds(pi_inv_x, kHalf, span), span);
    auto k_i = b.Bind("k_i", b.Cast(k_pre, DataType::INT32, kCastModeRint, span), span);
    k_f = b.Bind("k_f", b.Cast(k_i, DataType::FP32, kCastModeNone, span), span);
  } else {
    // k_f = float(round(x * PI_INV))
    auto pi_inv_x = b.Bind("pi_inv_x", b.Muls(x, kPiInv, span), span);
    auto k_i = b.Bind("k_i", b.Cast(pi_inv_x, DataType::INT32, kCastModeRound, span), span);
    k_f = b.Bind("k_f", b.Cast(k_i, DataType::FP32, kCastModeNone, span), span);
  }

  // t = x - k_f * pi (4-part Cody-Waite). For cos, +pi/2 head/tail are
  // interleaved between PI_C1 and PI_C2, and after PI_C4 respectively.
  auto kpv2 = b.Bind("k_pi_v2", b.Muls(k_f, kPiV2, span), span);
  t = b.Bind("t0", b.Sub(x, kpv2, span), span);
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

}  // namespace

void RegisterSinCosLoweringRules(CompositeLoweringRegistry& reg) {
  reg.Register("tile.sin", &LowerSinRule);
  reg.Register("tile.cos", &LowerCosRule);
}

}  // namespace ir
}  // namespace pypto
