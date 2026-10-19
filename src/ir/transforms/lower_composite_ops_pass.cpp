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
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
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
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {

namespace {

using lower_composite::CommSetup;
using lower_composite::CompositeLoweringFn;
using lower_composite::LoweringBuilder;

// Every pld.tensor.* collective now lives in its own translation unit under
// lower_composite/ (plan 70). What stays here is the dispatch table, the
// mutator, and the elementwise / MX rules that share no code with them.
using lower_composite::LowerTensorAllGatherRule;
using lower_composite::LowerTensorAllReduceRule;
using lower_composite::LowerTensorAllToAllRule;
using lower_composite::LowerTensorAllToAllVRule;
using lower_composite::LowerTensorBarrierRule;
using lower_composite::LowerTensorBroadcastRule;
using lower_composite::LowerTensorReduceScatterRule;

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
// ``tile.select`` lowering — packed predicate mask + TSEL / TSELS.
//
// `tile.select(cond, on_true, on_false)` is the scratch-free selection surface;
// `tile.sel` / `tile.sels` stay the 1:1 PTO forms where the caller owns the
// scratch buffer.
//
// `cond` is already the packed predicate mask both TSEL and TSELS want --
// `DeduceTileSelectType` accepts nothing else -- so the rule only has to pick
// the PTO form and synthesize the scratch. TSELS covers `mask ? tile : scalar`
// in one instruction; everything else materializes the scalar with `tile.full`
// and uses TSEL, which is also where the dtypes with no TSELS form land (bf16 on
// either arch, 8-bit integers on A2/A3).
// ============================================================================

/// Rebuild `scalar` as `dtype` when it is a literal, so the TSELS operand can
/// satisfy the dtype pairing `tile.sels` requires. Returns nullptr for a
/// non-literal of the wrong dtype, which routes the caller to the TSEL path.
ExprPtr RetypeSelectScalar(const ExprPtr& scalar, DataType dtype, const Span& span) {
  auto scalar_type = As<ScalarType>(scalar->GetType());
  if (scalar_type && scalar_type->dtype_ == dtype) return scalar;
  if (auto c = As<ConstInt>(scalar)) {
    return std::make_shared<ConstInt>(c->value_, dtype, span);
  }
  if (auto c = As<ConstFloat>(scalar)) {
    return std::make_shared<ConstFloat>(c->value_, dtype, span);
  }
  return nullptr;
}

/// Materialize `scalar` as a tile shaped like `model`, the other (shaped) branch.
///
/// `tile.full` is the only broadcast the tile pipeline can actually lower today:
/// `tile.expands` has no PTO emitter (`pl.expands` fails codegen on main), so it
/// is not an option here. That constrains the operand to a literal and the shape
/// to static extents; `DeduceTileSelectType` enforces both at the user's call
/// site, so reaching this function with anything else is a compiler bug.
///
/// `tile.full` stamps its result fully valid, which would widen a tail-narrowed
/// select — `tile.sel` takes the result's valid extents from its `lhs`. Re-type
/// the created call with `model`'s view so the materialized branch carries the
/// same valid region as the branch it is standing in for.
ExprPtr MaterializeSelectScalar(const ExprPtr& scalar, const std::shared_ptr<const TileType>& model,
                                const std::string& name, LoweringBuilder& b, const Span& span) {
  auto value = RetypeSelectScalar(scalar, model->dtype_, span);
  INTERNAL_CHECK_SPAN(value, span)
      << "Internal error: tile.select scalar operand is not a literal of the selected dtype";
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", model->dtype_}};
  auto shape = tile_conversion_utils::MakeShapeTuple(model->shape_, span);
  auto created = As<Call>(OpRegistry::GetInstance().Create("tile.full", {shape, value}, kwargs, span));
  INTERNAL_CHECK_SPAN(created, span) << "Internal error: tile.full did not produce a Call";
  TileView view;
  view.valid_shape = GetValidShape(model);
  InheritTileViewLayout(view, model);
  // Override only the view; every other field stays as tile.full deduced it, so
  // the rewritten call still round-trips through print -> parse.
  auto deduced = As<TileType>(created->GetType());
  INTERNAL_CHECK_SPAN(deduced, span) << "Internal error: tile.full did not produce a TileType";
  auto typed = std::make_shared<TileType>(deduced->shape_, deduced->dtype_, deduced->memref_, view,
                                          deduced->memory_space_);
  return b.Bind(
      name,
      std::make_shared<Call>(created->op_, created->args_, created->kwargs_, created->attrs_, typed, span),
      span);
}

/// Bind a freshly created Vec scratch tile matching `spec`.
ExprPtr BindSelectScratch(const backend::TileScratchSpec& spec, LoweringBuilder& b, const Span& span) {
  std::vector<ExprPtr> dims = {std::make_shared<ConstInt>(spec.rows, DataType::INDEX, span),
                               std::make_shared<ConstInt>(spec.cols, DataType::INDEX, span)};
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", spec.dtype},
                                                          {"target_memory", MemorySpace::Vec}};
  auto shape = tile_conversion_utils::MakeShapeTuple(dims, span);
  return b.Bind("sel_tmp", OpRegistry::GetInstance().Create("tile.create", {shape}, kwargs, span), span);
}

ExprPtr LowerTileSelectRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span span = call->span_;
  auto& reg = OpRegistry::GetInstance();

  INTERNAL_CHECK_SPAN(args.size() == 3, span)
      << "Internal error: tile.select lowering expects 3 args, got " << args.size();
  auto result_type = As<TileType>(call->GetType());
  INTERNAL_CHECK_SPAN(result_type, span) << "Internal error: tile.select lowering requires a TileType result";

  // Type deduction already required cond to be the packed predicate mask for
  // this result, which is what both TSEL and TSELS take.
  const ExprPtr& mask = args[0];

  // ---- TSELS where it applies, TSEL otherwise ------------------------------
  auto true_tile = As<TileType>(args[1]->GetType());
  auto false_tile = As<TileType>(args[2]->GetType());
  const auto* handler = tile_conversion_utils::ActiveBackendHandler();

  if (true_tile && !false_tile) {
    // dst = mask ? src : scalar is exactly TSELS -- if the operands are rank 2,
    // the target has a form for this dtype, the source column count is static
    // (A2/A3 sizes the scratch against one physical source row), and the scalar
    // can carry the dtype `tile.sels` pairs with the source.
    //
    // The rank gate is needed because this pass runs BEFORE FlattenTileNdTo2D
    // (slot 13 vs 14), so an ND operand is still ND here while `DeduceTileSelsType`
    // hard-requires rank 2 -- its mask-coverage arithmetic indexes valid_shape[0]
    // and [1] directly. `tile.full` and `tile.sel` have no such restriction, so ND
    // takes the path below and the flattener collapses it with everything else.
    const bool rank_ok = true_tile->shape_.size() == 2;
    auto src_cols = As<ConstInt>(true_tile->shape_.back());
    const bool dtype_ok = handler == nullptr || handler->SupportsTselsDataType(true_tile->dtype_);
    ExprPtr scalar = RetypeSelectScalar(args[2], GetTselsScalarDataType(true_tile->dtype_), span);
    if (rank_ok && dtype_ok && src_cols && scalar) {
      auto tmp = BindSelectScratch(
          tile_conversion_utils::TselsScratchSpec(true_tile->dtype_, src_cols->value_), b, span);
      return reg.Create("tile.sels", {mask, args[1], tmp, scalar}, span);
    }
  }

  // Everything below lowers through TSEL, so a dtype it has no form for has no
  // lowering at all -- `tile.full`'s own `pto.texpands` stops at the same width.
  // A2/A3 8-bit is the case in practice: tsel, tsels and texpands are all 16-bit
  // and up there, so report it here instead of emitting IR ptoas will reject.
  const auto& shaped = true_tile ? true_tile : false_tile;
  CHECK_SPAN(handler == nullptr || handler->SupportsTselDataType(shaped->dtype_), span)
      << "tile.select cannot select " << shaped->dtype_.ToString() << " values on backend '"
      << (handler != nullptr ? handler->GetPtoTargetArch() : "unknown")
      << "': neither pto.tsel nor pto.tsels has a form for that element type there. Cast the operands "
         "to a wider type before selecting";
  ExprPtr on_true = true_tile ? args[1] : MaterializeSelectScalar(args[1], shaped, "sel_true", b, span);
  ExprPtr on_false = false_tile ? args[2] : MaterializeSelectScalar(args[2], shaped, "sel_false", b, span);
  auto tmp = BindSelectScratch(tile_conversion_utils::TselScratchSpec(), b, span);
  return reg.Create("tile.sel", {mask, on_true, on_false, tmp}, span);
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

// ----------------------------------------------------------------------------
// Composite-op dispatch table.
//
// ``LowerCompositeOps`` is a generic dispatcher: it rewrites a ``var = Call(...)``
// AssignStmt (or a composite-op Call embedded directly in a ReturnStmt) only
// when the callee name appears here. Adding a new composite op = add a rule
// function above + one row in ``kRules``; the mutator below needs no change.
// A new ``pld.tensor.*`` collective must additionally be listed in
// ``LowerCompositeOpsMutator::IsTensorCollective`` so it inherits the HOST
// deferral. Use ``EmitBarrier`` for a full rendezvous, or ``EmitNotifyAll`` plus
// call-local waits for split readiness (all_to_all_v); its final Ge(2) wait
// still uses ``EmitWaitAll``. Call ``EmitEpilogueReset`` exactly once with the
// total credits issued, retiring this call's credits while preserving any
// early next-call credits.
//
// Today the rules are ``tile.sin`` / ``tile.cos``, ``tile.tquant_mx``,
// ``tile.select``, and ``pld.tensor.*`` distributed collectives. Host-level
// allreduce is skipped here
// and lowered later by LowerHostTensorCollectives. The pass is idempotent
// provided each rule emits only ops not listed here.
//
// Every ``pld.tensor.*`` collective already has its own translation unit under
// ``src/ir/transforms/lower_composite/`` (plan 70): add the rule there, declare
// it in ``lower_composite_rules.h``, list the new .cpp in ``CMakeLists.txt``,
// and add one row below. The ``tile.*`` rules above are still file-local because
// they share no code with the collectives.
// ----------------------------------------------------------------------------
CompositeLoweringFn LookupCompositeRule(const std::string& op_name) {
  static const std::unordered_map<std::string, CompositeLoweringFn> kRules = {
      {"tile.sin", &LowerSinRule},
      {"tile.cos", &LowerCosRule},
      // tile.tquant_mx → tile.tquant_mx_raw + tile.tmov_x2zz (value-returning SSA).
      // Scratch tiles are created with MemorySpace::Vec before InferTileMemorySpace.
      {"tile.tquant_mx", &LowerTileTQuantMxRule},
      // tile.select → [tile.cmps] + tile.create(scratch) + tile.sel|tile.sels.
      // Scratch tiles are created with MemorySpace::Vec before InferTileMemorySpace.
      {"tile.select", &LowerTileSelectRule},
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
