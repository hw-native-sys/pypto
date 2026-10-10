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
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_common.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {

namespace {

using lower_composite::CommSetup;
using lower_composite::CompositeLoweringFn;
using lower_composite::LoweringBuilder;
using lower_composite::MakeChunkGeometry;
using lower_composite::MakeCollectiveStageShape;
using lower_composite::ValidateMeshSignalShape;

// Rules that already have their own translation unit (plan 70 phase 2).
using lower_composite::LowerTensorAllGatherRule;
using lower_composite::LowerTensorAllReduceRule;

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
  // The stage is a bounded bounce buffer, not a copy of the transfer, so the
  // target shape no longer has to be static: a dynamic extent simply takes the
  // chunk bound. pld.tile.get slides the full extent through it.
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.broadcast");
  auto stage_shape_tuple =
      MakeCollectiveStageShape(target_type->shape_, chunk_geometry, span, "pld.tensor.broadcast");

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
  // One shared VEC staging tile is reused across all destinations, mirroring
  // allgather's. It is capped to one chunk rather than sized from SIZE: the
  // stage is a bounce buffer the transfer slides through, so a [1, SIZE] stage
  // would only waste UB. chunk_shape stays the transfer extent.
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.all_to_all");
  auto stage_shape =
      MakeCollectiveStageShape({one_idx, size_expr}, chunk_geometry, span, "pld.tensor.all_to_all");
  auto put_stage =
      b.Bind("aa_stage",
             reg.Create("tile.create", {stage_shape},
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
// Variable-size all-to-all (MPI_Alltoallv pattern). Each rank pushes a
// runtime-sized block to every peer via one pld.tile.put per destination.
// ``send_counts[dest]`` is read from device data and clamped to
// ``[0, MAX_RECV]``; that clamped value becomes the dynamic transfer row count.
// The 5-arg API signature (input, target, signal,
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
//       // partition-view dims on pto.comm.tput). A bounded
//       // [stage_rows, stage_cols] tile feeds the TPUT engine, which
//       // 2-D-slides the transfer through it.
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

  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);

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

  // Per-destination staging tile, capped to one chunk — pto-isa slides the
  // static [MAX_RECV, SIZE] transfer through it, so the stage need not (and
  // should not) be sized from SIZE.
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.all_to_all_v");
  auto stage_shape =
      MakeCollectiveStageShape({max_recv_expr, size_expr}, chunk_geometry, span, "pld.tensor.all_to_all_v");

  // ---- Phase 1: push per-destination blocks to peer windows ----
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
