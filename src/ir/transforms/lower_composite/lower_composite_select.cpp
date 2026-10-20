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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/op_rules.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
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
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {
namespace lower_composite {

namespace {

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

}  // namespace

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
    const auto scratch = tile_conversion_utils::PlanSelectScratch(
        "tile.sels", {true_tile->dtype_, src_cols ? std::optional<int64_t>(src_cols->value_) : std::nullopt});
    ExprPtr scalar = RetypeSelectScalar(args[2], GetTselsScalarDataType(true_tile->dtype_), span);
    if (rank_ok && src_cols && scalar && scratch.status == backend::OpScratchStatus::Required) {
      INTERNAL_CHECK_SPAN(scratch.spec.has_value(), span) << "Internal error: missing TSELS scratch spec";
      auto tmp = BindSelectScratch(*scratch.spec, b, span);
      return reg.Create("tile.sels", {mask, args[1], tmp, scalar}, span);
    }
  }

  // Everything below lowers through TSEL, so a dtype it has no form for has no
  // lowering at all -- `tile.full`'s own `pto.texpands` stops at the same width.
  // A2/A3 8-bit is the case in practice: tsel, tsels and texpands are all 16-bit
  // and up there, so report it here instead of emitting IR ptoas will reject.
  const auto& shaped = true_tile ? true_tile : false_tile;
  const auto scratch = tile_conversion_utils::PlanSelectScratch("tile.sel", {shaped->dtype_, std::nullopt});
  CHECK_SPAN(scratch.status == backend::OpScratchStatus::Required, span)
      << "tile.select cannot select " << shaped->dtype_.ToString() << " values on backend '"
      << (handler != nullptr ? handler->GetPtoTargetArch() : "unknown")
      << "': neither pto.tsel nor pto.tsels has a form for that element type there. Cast the operands "
         "to a wider type before selecting";
  ExprPtr on_true = true_tile ? args[1] : MaterializeSelectScalar(args[1], shaped, "sel_true", b, span);
  ExprPtr on_false = false_tile ? args[2] : MaterializeSelectScalar(args[2], shaped, "sel_false", b, span);
  INTERNAL_CHECK_SPAN(scratch.spec.has_value(), span) << "Internal error: missing TSEL scratch spec";
  auto tmp = BindSelectScratch(*scratch.spec, b, span);
  return reg.Create("tile.sel", {mask, on_true, on_false, tmp}, span);
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
