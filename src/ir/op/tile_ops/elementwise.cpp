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

/**
 * @file elementwise.cpp
 * @brief Element-wise tile operations (Mul, Add, Div, Sub, and scalar variants)
 *
 * This file implements element-wise tile operations that support
 * 2D tiles (at most 2 dimensions) with 2D broadcasting.
 * Operations are divided into:
 * - Tile-Tile operations (mul, add, div, sub): TileType + TileType
 * - Tile-Scalar operations (muls, adds, divs, subs): TileType + ScalarType
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

static ExprPtr MakeIndexConst(int64_t value, const Span& span = Span::unknown()) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

static ExprPtr MakeCeilDivIndex(const ExprPtr& value, int64_t divisor) {
  if (auto const_value = As<ConstInt>(value)) {
    return MakeIndexConst((const_value->value_ + divisor - 1) / divisor, value->span_);
  }
  return MakeFloorDiv(MakeAdd(value, MakeIndexConst(divisor - 1, value->span_), value->span_),
                      MakeIndexConst(divisor, value->span_), value->span_);
}

static ExprPtr MakeRoundUpIndex(const ExprPtr& value, int64_t alignment) {
  if (auto const_value = As<ConstInt>(value)) {
    const int64_t rounded = ((const_value->value_ + alignment - 1) / alignment) * alignment;
    return MakeIndexConst(rounded, value->span_);
  }
  return MakeMul(MakeCeilDivIndex(value, alignment), MakeIndexConst(alignment, value->span_), value->span_);
}

// Build the packed-predicate mask TileType for tile.cmp / tile.cmps. ``logical_view``
// carries the LOGICAL valid_shape (already agreed across the value operands by
// DeduceBroadcastElementwiseTileView) and the inherited layout; this function packs
// the col axis (ceil_div by 8, aligned) on both the physical shape and the valid
// extent. The col-axis valid packing is applied to the AGREED logical valid, so the
// rhs valid region of a 2-tile cmp participates (issue: previously only lhs was
// consulted). ``logical_view.valid_shape`` has rank == ``logical_shape.size()``
// (helper guarantee), so indexing ``col_axis`` is safe.
static std::shared_ptr<TileType> MakePackedPredicateTileType(const std::vector<ExprPtr>& logical_shape,
                                                             const TileView& logical_view) {
  INTERNAL_CHECK(!logical_shape.empty())
      << "tile.cmp/tile.cmps require a non-empty tile shape for packed predicate mask inference";

  constexpr int64_t kA2A3PredicateBitsPerByte = 8;
  constexpr int64_t kA2A3PredicateColAlignment = 32;

  const size_t col_axis = logical_shape.size() - 1;
  std::vector<ExprPtr> mask_shape = logical_shape;
  mask_shape[col_axis] = MakeRoundUpIndex(
      MakeCeilDivIndex(logical_shape[col_axis], kA2A3PredicateBitsPerByte), kA2A3PredicateColAlignment);

  TileView tile_view = logical_view;  // agreed logical valid_shape + inherited layout
  tile_view.valid_shape[col_axis] =
      MakeCeilDivIndex(tile_view.valid_shape[col_axis], kA2A3PredicateBitsPerByte);
  return std::make_shared<TileType>(mask_shape, DataType::UINT8, std::nullopt, tile_view);
}

// Forward declarations: trem/trems reuse the tmp-carrying deducers defined
// further down (also used by xor/xors), so they must be visible here.
TypePtr DeduceTileOpTernaryType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name, bool require_int);
TypePtr DeduceTileOpTileScalarTileType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name);

TypePtr DeduceTileOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                                          const std::string& op_name, bool require_int = false,
                                          bool partial_combine = false) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Both arguments must be TileType
  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());

  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();

  if (require_int) {
    CHECK(tile_type1->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_type1->dtype_.ToString();
    CHECK(tile_type2->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_type2->dtype_.ToString();
  }

  // Use broadcasting
  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << FormatShape(tile_type1->shape_) << " and "
                                  << FormatShape(tile_type2->shape_);

  TileView tile_view;
  if (partial_combine) {
    // Partial-combine ops (tile.part_add/part_mul/part_max/part_min) copy whichever
    // source is valid where only one is, so the result is valid wherever EITHER
    // source is — the origin-anchored UNION of the two valid regions, NOT the
    // strict agreement of true elementwise ops. Reuse the assemble-union rule with a
    // zero offset (it rejects a non-representable L-shaped union per the North Star,
    // and is symmetric in the two operands so operand order does not matter).
    const std::vector<ExprPtr> src0_valid = LiftValidShapeForBroadcast(
        tile_type1->shape_, GetValidShape(tile_type1), broadcast_result.shape, args[0]->span_, op_name);
    const std::vector<ExprPtr> src1_valid = LiftValidShapeForBroadcast(
        tile_type2->shape_, GetValidShape(tile_type2), broadcast_result.shape, args[0]->span_, op_name);
    const std::vector<ExprPtr> zero_offset(broadcast_result.shape.size(),
                                           std::make_shared<ConstInt>(0, DataType::INDEX, args[0]->span_));
    tile_view.valid_shape = ComputeAssembleUnionValidShape(src0_valid, src1_valid, zero_offset,
                                                           broadcast_result.shape, args[0]->span_, op_name);
    if (auto layout_source = PickElementwiseLayoutSource(broadcast_result.shape, {tile_type1, tile_type2})) {
      InheritFreshTileComputeLayout(tile_view, layout_source);
    } else {
      tile_view.blayout = tile_view_semantics::InferImplicitTileLayoutFromShape(broadcast_result.shape);
    }
  } else {
    // True elementwise binary: operands must AGREE on their valid regions (Q2 —
    // valid_shape never broadcasts); a broadcast operand is exempt in its broadcast
    // dim and the layout comes from the shaping operand, not blindly from arg0.
    tile_view = DeduceBroadcastElementwiseTileView(broadcast_result.shape, {tile_type1, tile_type2},
                                                   args[0]->span_, op_name);
  }
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

// Tile-tile shift ops (shl, shr): RHS is the shift amount, result type equals LHS tile type,
// consistent with scalar variants (shls/shrs) which preserve the LHS tile dtype.
TypePtr DeduceTileOpShiftBinaryType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(tile_type1->dtype_.IsInt()) << "The operator " << op_name << " requires integer tile dtype, but got "
                                    << tile_type1->dtype_.ToString();
  CHECK(tile_type2->dtype_.IsInt()) << "The operator " << op_name
                                    << " requires integer shift tile dtype, but got "
                                    << tile_type2->dtype_.ToString();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

  // Shift is elementwise (dst[i] = lhs[i] << rhs[i]): lhs and the shift-amount rhs
  // must agree on their valid regions (Q2 — valid_shape never broadcasts). Result
  // dtype is the lhs tile dtype (see below).
  TileView tile_view = DeduceBroadcastElementwiseTileView(broadcast_result.shape, {tile_type1, tile_type2},
                                                          args[0]->span_, op_name);
  return std::make_shared<TileType>(broadcast_result.shape, tile_type1->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileOpScalarBinaryType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument MUST be ScalarType
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();

  // Result preserves the tile's element type. The hardware scalar instructions
  // (e.g. pto.tmuls) require src and dst to share the same element type; the
  // scalar operand is implicitly narrowed to match the tile dtype at runtime.
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  InheritFreshTileComputeLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileOpIntScalarBinaryType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs,
                                        const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument must be TileType with integer dtype.
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
  CHECK(tile_type->dtype_.IsInt()) << "The operator " << op_name << " requires integer tile dtype, but got "
                                   << tile_type->dtype_.ToString();

  // Second argument must be ScalarType with an integer dtype per ISA spec:
  //   %dst = tshls/tshrs/tands/tors %src, %scalar : !pto.tile<...>, i32
  // The IR allows any integer width (INT8/16/32/64, UINT variants); codegen casts to i32.
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();
  CHECK(scalar_type->dtype_.IsInt()) << "The operator " << op_name
                                     << " requires shift/bitwise scalar to be an integer type, but got "
                                     << scalar_type->dtype_.ToString();

  // Result has the same shape and dtype as the input tile; the shift amount does not change element type.
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  InheritFreshTileComputeLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

// ============================================================================
// Op Registration
// ============================================================================

REGISTER_OP("tile.mul")
    .set_op_category("TileOp")
    .set_description("Element-wise multiplication of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.mul");
    });

REGISTER_OP("tile.add")
    .set_op_category("TileOp")
    .set_description("Element-wise addition of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.add");
    });

REGISTER_OP("tile.div")
    .set_op_category("TileOp")
    .set_description("Element-wise division of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.div");
    });

REGISTER_OP("tile.sub")
    .set_op_category("TileOp")
    .set_description("Element-wise subtraction of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.sub");
    });

REGISTER_OP("tile.maximum")
    .set_op_category("TileOp")
    .set_description("Element-wise maximum of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.maximum");
    });

REGISTER_OP("tile.minimum")
    .set_op_category("TileOp")
    .set_description("Element-wise minimum of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.minimum");
    });

REGISTER_OP("tile.rem")
    .set_op_category("TileOp")
    .set_description("Element-wise remainder (modulo) of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("tmp", "Temporary tile (TileType) required by the hardware")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTernaryType(args, kwargs, "tile.rem", false);
    });

// Partial-combine binary ops: elementwise over dst's valid region, but where
// only one source is valid at an element the result copies that source. Shape
// is identical to a plain elementwise binary op; the "partial" behaviour is a
// runtime valid-region effect, so type deduction is the standard binary one.
REGISTER_OP("tile.part_add")
    .set_op_category("TileOp")
    .set_description("Partial element-wise add of two tiles (copies the only valid input)")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.part_add", /*require_int=*/false,
                                               /*partial_combine=*/true);
    });

REGISTER_OP("tile.part_mul")
    .set_op_category("TileOp")
    .set_description("Partial element-wise multiply of two tiles (copies the only valid input)")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.part_mul", /*require_int=*/false,
                                               /*partial_combine=*/true);
    });

REGISTER_OP("tile.part_max")
    .set_op_category("TileOp")
    .set_description("Partial element-wise max of two tiles (copies the only valid input)")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.part_max", /*require_int=*/false,
                                               /*partial_combine=*/true);
    });

REGISTER_OP("tile.part_min")
    .set_op_category("TileOp")
    .set_description("Partial element-wise min of two tiles (copies the only valid input)")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.part_min", /*require_int=*/false,
                                               /*partial_combine=*/true);
    });

REGISTER_OP("tile.fmod")
    .set_op_category("TileOp")
    .set_description("Element-wise floating-point remainder of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // The hardware kernel overwrites src0 mid-computation (dst=src0/src1) but
    // still needs the original src0 for the final subtraction, so dst must not
    // alias any input buffer.
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.fmod");
    });

REGISTER_OP("tile.muls")
    .set_op_category("TileOp")
    .set_description("Element-wise multiplication of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.muls");
    });

REGISTER_OP("tile.adds")
    .set_op_category("TileOp")
    .set_description("Element-wise addition of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.adds");
    });

REGISTER_OP("tile.divs")
    .set_op_category("TileOp")
    .set_description("Element-wise division of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.divs");
    });

REGISTER_OP("tile.subs")
    .set_op_category("TileOp")
    .set_description("Element-wise subtraction of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.subs");
    });

REGISTER_OP("tile.rems")
    .set_op_category("TileOp")
    .set_description("Element-wise remainder (modulo) of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("tmp", "Temporary tile (TileType) required by the hardware")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTileScalarTileType(args, kwargs, "tile.rems");
    });

REGISTER_OP("tile.fmods")
    .set_op_category("TileOp")
    .set_description("Element-wise floating-point remainder of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // Same hazard as tile.fmod: the kernel clobbers src0 before its final use,
    // so dst must not alias the input buffer.
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.fmods");
    });

REGISTER_OP("tile.shl")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise left shift of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpShiftBinaryType(args, kwargs, "tile.shl");
    });

REGISTER_OP("tile.shls")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise left shift of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpIntScalarBinaryType(args, kwargs, "tile.shls");
    });

REGISTER_OP("tile.shr")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise right shift of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpShiftBinaryType(args, kwargs, "tile.shr");
    });

REGISTER_OP("tile.shrs")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise right shift of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpIntScalarBinaryType(args, kwargs, "tile.shrs");
    });

REGISTER_OP("tile.maximums")
    .set_op_category("TileOp")
    .set_description("Element-wise maximum of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.maximums");
    });

REGISTER_OP("tile.minimums")
    .set_op_category("TileOp")
    .set_description("Element-wise minimum of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.minimums");
    });

REGISTER_OP("tile.and")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise AND of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.and", true);
    });

REGISTER_OP("tile.ands")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise AND of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpIntScalarBinaryType(args, kwargs, "tile.ands");
    });

REGISTER_OP("tile.or")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise OR of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.or", true);
    });

REGISTER_OP("tile.ors")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise OR of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpIntScalarBinaryType(args, kwargs, "tile.ors");
    });

// Tile-tile ternary ops with a tmp buffer as the third argument.
// When require_int is true (bitwise ops like xor), both tile dtypes must be integer.
TypePtr DeduceTileOpTernaryType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name, bool require_int = false) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(As<TileType>(args[2]->GetType()))
      << "The operator " << op_name << " requires third argument (tmp) to be a TileType, but got "
      << args[2]->GetType()->TypeName();

  if (require_int) {
    CHECK(tile_type1->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_type1->dtype_.ToString();
    CHECK(tile_type2->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_type2->dtype_.ToString();
  }

  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";
  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

  // Only args 0 and 1 are value tiles (arg 2 is scratch tmp) — they must AGREE on
  // their valid regions (Q2 — valid_shape never broadcasts).
  TileView tile_view = DeduceBroadcastElementwiseTileView(broadcast_result.shape, {tile_type1, tile_type2},
                                                          args[0]->span_, op_name);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

// All three tiles are real inputs (addc, subc): promote dtype and broadcast shape across all three.
TypePtr DeduceTileOpTriTileType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());
  auto tile_type3 = As<TileType>(args[2]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(tile_type3) << "The operator " << op_name << " requires third argument to be a TileType, but got "
                    << args[2]->GetType()->TypeName();

  auto result_dtype12 = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype12) << "The operator " << op_name << " requires compatible data types";
  auto result_dtype = PromoteDataTypes(*result_dtype12, tile_type3->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";

  auto broadcast12 = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast12.success) << "The operator " << op_name << " requires compatible shapes";
  auto broadcast_result = BroadcastShapes(broadcast12.shape, tile_type3->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

  // All three tiles are real inputs — they must AGREE on their valid regions (Q2 —
  // valid_shape never broadcasts).
  TileView tile_view = DeduceBroadcastElementwiseTileView(
      broadcast_result.shape, {tile_type1, tile_type2, tile_type3}, args[0]->span_, op_name);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

// (Tile, Scalar, Tile) pattern (addsc, subsc): any scalar type, promote output from all three inputs.
TypePtr DeduceTileOpTileScalarTileType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();

  auto tile_type2 = As<TileType>(args[2]->GetType());
  CHECK(tile_type2) << "The operator " << op_name << " requires third argument to be a TileType, but got "
                    << args[2]->GetType()->TypeName();

  auto result_dtype12 = PromoteDataTypes(tile_type1->dtype_, scalar_type->dtype_);
  CHECK(result_dtype12) << "The operator " << op_name << " requires compatible data types";
  auto result_dtype = PromoteDataTypes(*result_dtype12, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

  // The two value tiles (args 0 and 2; arg 1 is a scalar with no valid region) must
  // AGREE on their valid regions (Q2 — valid_shape never broadcasts).
  TileView tile_view = DeduceBroadcastElementwiseTileView(broadcast_result.shape, {tile_type1, tile_type2},
                                                          args[0]->span_, op_name);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileOpXorScalarType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
  CHECK(tile_type->dtype_.IsInt()) << "The operator " << op_name << " requires integer tile dtype, but got "
                                   << tile_type->dtype_.ToString();

  // Second argument must be ScalarType with an integer dtype per ISA spec:
  //   %dst = txors %src, %scalar : !pto.tile<...>, i32
  // The IR allows any integer width (INT8/16/32/64, UINT variants); codegen casts to i32.
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();
  CHECK(scalar_type->dtype_.IsInt()) << "The operator " << op_name
                                     << " requires scalar to be an integer type, but got "
                                     << scalar_type->dtype_.ToString();

  CHECK(As<TileType>(args[2]->GetType()))
      << "The operator " << op_name << " requires third argument to be a TileType, but got "
      << args[2]->GetType()->TypeName();

  // Result has the same shape and dtype as the input tile; bitwise ops do not change element type.
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  InheritFreshTileComputeLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.xor")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise XOR of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTernaryType(args, kwargs, "tile.xor", true);
    });

REGISTER_OP("tile.xors")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise XOR of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpXorScalarType(args, kwargs, "tile.xors");
    });

REGISTER_OP("tile.prelu")
    .set_op_category("TileOp")
    .set_description("Element-wise parametric ReLU of a tile with slope tile and temporary buffer")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("slope", "Slope tile (TileType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTernaryType(args, kwargs, "tile.prelu");
    });

REGISTER_OP("tile.addc")
    .set_op_category("TileOp")
    .set_description("Element-wise addition of three tiles (lhs + rhs + rhs2) with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTriTileType(args, kwargs, "tile.addc");
    });

REGISTER_OP("tile.subc")
    .set_op_category("TileOp")
    .set_description("Element-wise subtraction of three tiles (lhs - rhs - rhs2) with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTriTileType(args, kwargs, "tile.subc");
    });

REGISTER_OP("tile.addsc")
    .set_op_category("TileOp")
    .set_description("Element-wise addition of tile, scalar, and tile (lhs + scalar + rhs2)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTileScalarTileType(args, kwargs, "tile.addsc");
    });

REGISTER_OP("tile.subsc")
    .set_op_category("TileOp")
    .set_description("Element-wise subtraction of tile, scalar, and tile (lhs - scalar - rhs2)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTileScalarTileType(args, kwargs, "tile.subsc");
    });

REGISTER_OP("tile.lrelu")
    .set_op_category("TileOp")
    .set_description("Element-wise leaky ReLU of a tile with scalar slope (max(x, slope*x))")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("slope", "Scalar slope for negative values (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.lrelu");
    });

// Type deduction for tile.sel (MaskTile x Tile x Tile x TmpTile -> Tile)
// The mask tile encodes per-element predicates in a target-defined layout; its dtype/shape
// do not influence the output type.  Output type is derived from lhs and rhs only.
TypePtr DeduceTileSelType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                          const std::string& op_name) {
  CHECK(args.size() == 4) << "The operator " << op_name << " requires exactly 4 arguments, but got "
                          << args.size();

  CHECK(As<TileType>(args[0]->GetType()))
      << "The operator " << op_name << " requires first argument (mask) to be a TileType, but got "
      << args[0]->GetType()->TypeName();

  auto tile_type1 = As<TileType>(args[1]->GetType());
  auto tile_type2 = As<TileType>(args[2]->GetType());
  CHECK(tile_type1) << "The operator " << op_name
                    << " requires second argument (lhs) to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name
                    << " requires third argument (rhs) to be a TileType, but got "
                    << args[2]->GetType()->TypeName();
  CHECK(As<TileType>(args[3]->GetType()))
      << "The operator " << op_name << " requires fourth argument (tmp) to be a TileType, but got "
      << args[3]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tile_type1->dtype_.ToString() << " and " << tile_type2->dtype_.ToString();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << FormatShape(tile_type1->shape_) << " and "
                                  << FormatShape(tile_type2->shape_);

  // The two value tiles (args 1 and 2) must AGREE on their valid regions (Q2 —
  // valid_shape never broadcasts). The predicate mask (arg 0) and tmp scratch
  // (arg 3) are excluded — the mask's valid columns are ceil_div-packed and cannot
  // agree with a value tile.
  TileView tile_view = DeduceBroadcastElementwiseTileView(broadcast_result.shape, {tile_type1, tile_type2},
                                                          args[1]->span_, op_name);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

REGISTER_OP("tile.sel")
    .set_op_category("TileOp")
    .set_description(
        "Per-element selection between two tiles using a predicate mask tile. "
        "Maps to the TSEL hardware intrinsic.")
    .add_argument("mask", "Predicate mask tile; encoding is target-defined (TileType)")
    .add_argument("lhs", "Source tile 0, selected where mask is true (TileType)")
    .add_argument("rhs", "Source tile 1, selected where mask is false (TileType)")
    .add_argument("tmp", "Scratch tile required by TSEL (TileType UINT8 [1, 32])")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_input_memory(3, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // The TSEL intrinsic reads the predicate mask (arg 0) and the tmp scratch
    // (arg 3) while writing dst, so dst must not share their buffers — even
    // though tile.sel is otherwise in-place-safe w.r.t. its lhs/rhs value
    // operands. (The mask/tmp also differ in dtype/shape from dst, so codegen
    // could not reinterpret them in place anyway.)
    .forbid_output_alias(0)
    .forbid_output_alias(3)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSelType(args, kwargs, "tile.sel");
    });

// Type deduction for tile.sels (Tile x Tile x Scalar -> Tile)
TypePtr DeduceTileSelScalarType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());
  CHECK(tile_type1) << "The operator " << op_name
                    << " requires first argument (lhs) to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name
                    << " requires second argument (rhs) to be a TileType, but got "
                    << args[1]->GetType()->TypeName();

  CHECK(As<ScalarType>(args[2]->GetType()))
      << "The operator " << op_name << " requires third argument (select_mode) to be a ScalarType, but got "
      << args[2]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tile_type1->dtype_.ToString() << " and " << tile_type2->dtype_.ToString();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << FormatShape(tile_type1->shape_) << " and "
                                  << FormatShape(tile_type2->shape_);

  // The two value tiles (args 0 and 1; arg 2 is the scalar select mode) must AGREE
  // on their valid regions (Q2 — valid_shape never broadcasts).
  TileView tile_view = DeduceBroadcastElementwiseTileView(broadcast_result.shape, {tile_type1, tile_type2},
                                                          args[0]->span_, op_name);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

REGISTER_OP("tile.sels")
    .set_op_category("TileOp")
    .set_description("Select between two tiles based on a scalar mode. Maps to the TSELS hardware intrinsic.")
    .add_argument("lhs", "Source tile 0 (TileType)")
    .add_argument("rhs", "Source tile 1 (TileType)")
    .add_argument("select_mode", "Scalar select mode (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSelScalarType(args, kwargs, "tile.sels");
    });

// Type deduction for tile.cmp and tile.cmps (comparison operations)
TypePtr DeduceTileCmpType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                          const std::string& op_name, bool is_scalar_rhs = false) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Validate cmp_type attribute exists
  bool has_cmp_type = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "cmp_type") {
      has_cmp_type = true;
      break;
    }
  }
  CHECK(has_cmp_type) << "The operator " << op_name << " requires 'cmp_type' attribute";

  // First argument must be TileType
  auto tile_type1 = As<TileType>(args[0]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

  if (is_scalar_rhs) {
    // Second argument MUST be ScalarType
    auto scalar_type = As<ScalarType>(args[1]->GetType());
    CHECK(scalar_type) << "The operator " << op_name
                       << " requires second argument to be a ScalarType, but got "
                       << args[1]->GetType()->TypeName();

    // Only the lhs tile has a valid region (rhs is scalar); the logical view is just
    // its own valid_shape + layout, packed on the col axis by MakePackedPredicateTileType.
    TileView logical_view =
        DeduceBroadcastElementwiseTileView(tile_type1->shape_, {tile_type1}, args[0]->span_, op_name);
    return MakePackedPredicateTileType(tile_type1->shape_, logical_view);
  } else {
    // Second argument must be TileType
    auto tile_type2 = As<TileType>(args[1]->GetType());
    CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                      << args[1]->GetType()->TypeName();

    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                    << FormatShape(tile_type1->shape_) << " and "
                                    << FormatShape(tile_type2->shape_);

    // Agree on the LOGICAL valid region across BOTH tiles first (Q2 — valid_shape
    // never broadcasts); MakePackedPredicateTileType then packs the col axis. The
    // rhs valid region now participates (previously only lhs was consulted).
    TileView logical_view = DeduceBroadcastElementwiseTileView(
        broadcast_result.shape, {tile_type1, tile_type2}, args[0]->span_, op_name);
    return MakePackedPredicateTileType(broadcast_result.shape, logical_view);
  }
}

REGISTER_OP("tile.cmp")
    .set_op_category("TileOp")
    .set_description("Element-wise comparison of two tiles (returns a packed predicate mask tile)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_attr<int>("cmp_type")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCmpType(args, kwargs, "tile.cmp", false);
    });

REGISTER_OP("tile.cmps")
    .set_op_category("TileOp")
    .set_description("Element-wise comparison of tile and scalar (returns a packed predicate mask tile)")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_attr<int>("cmp_type")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCmpType(args, kwargs, "tile.cmps", true);
    });

REGISTER_OP("tile.fillpad")
    .set_op_category("TileOp")
    .set_description("Fill destination tile with source tile data and pad remaining elements")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.fillpad requires exactly 1 argument, but got "
                              << args.size();

      // Argument must be TileType
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type) << "The operator tile.fillpad requires first argument to be a TileType, but got "
                       << args[0]->GetType()->TypeName();

      // Get pad_value from kwargs, default to PadValue::zero
      PadValue pad_value = PadValue::zero;
      for (const auto& kv : kwargs) {
        if (kv.first == "pad_value") {
          pad_value = std::any_cast<PadValue>(kv.second);
          CHECK(pad_value != PadValue::null)
              << "tile.fillpad requires pad_value to be zero/max/min, not null";
        }
      }

      // Return TileType with pad value set in tile_view
      // After fillpad, the entire tile is valid (padding region is now filled with pad_value)
      TileView tile_view;
      tile_view.valid_shape = tile_type->shape_;  // Expand valid_shape to full shape
      InheritTileViewLayout(tile_view, tile_type);
      tile_view.pad = pad_value;
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_, tile_view,
                                        tile_type->memory_space_);
    });

REGISTER_OP("tile.fillpad_inplace")
    .set_op_category("TileOp")
    .set_description("Fill padding elements of input tile in place with specified pad value")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_output_reuses_input(0)
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.fillpad_inplace requires exactly 1 argument, but got "
                              << args.size();

      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type)
          << "The operator tile.fillpad_inplace requires first argument to be a TileType, but got "
          << args[0]->GetType()->TypeName();

      PadValue pad_value = PadValue::zero;
      for (const auto& kv : kwargs) {
        if (kv.first == "pad_value") {
          pad_value = std::any_cast<PadValue>(kv.second);
          CHECK(pad_value != PadValue::null)
              << "tile.fillpad_inplace requires pad_value to be zero/max/min, not null";
        }
      }

      TileView tile_view;
      tile_view.valid_shape = tile_type->shape_;
      InheritTileViewLayout(tile_view, tile_type);
      tile_view.pad = pad_value;
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_, tile_view,
                                        tile_type->memory_space_);
    });

REGISTER_OP("tile.fillpad_expand")
    .set_op_category("TileOp")
    .set_description("Copy a smaller source tile into a larger destination tile, padding the remainder")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("shape", "Destination shape (Tuple of ConstInt), each dim >= source dim")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // The destination is a fresh, larger, differently-strided tile (NOT a view of
    // the source like tile.fillpad). The intrinsic reads the full source while
    // writing dst at a wider row stride, so aliasing dst onto src would corrupt
    // the strided copy — same hazard class as the arg-reduction family.
    .not_inplace_safe()
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      // tile.fillpad_expand(src, shape) — TFILLPAD_EXPAND: dst may be larger than
      // src. The valid region of src is copied into the top-left of dst, and all
      // other dst elements are filled with pad_value. The shape tuple is carried
      // for type deduction only (the codegen reads dst extents from the result
      // type, not from this operand).
      CHECK(args.size() == 2)
          << "The operator tile.fillpad_expand requires exactly 2 arguments (tile, shape), but got "
          << args.size();

      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type)
          << "The operator tile.fillpad_expand requires first argument to be a TileType, but got "
          << args[0]->GetType()->TypeName();

      auto shape_tuple = As<MakeTuple>(args[1]);
      CHECK(shape_tuple) << "tile.fillpad_expand shape must be a literal tuple of constants, but got "
                         << args[1]->GetType()->TypeName();
      const std::vector<ExprPtr>& new_shape = shape_tuple->elements_;
      CHECK(new_shape.size() == tile_type->shape_.size())
          << "tile.fillpad_expand shape rank (" << new_shape.size() << ") must match source rank ("
          << tile_type->shape_.size() << ")";

      // When both source and destination dims are static, the destination must
      // not be smaller than the source in any dimension (expand-only).
      for (size_t i = 0; i < new_shape.size(); ++i) {
        auto dst_dim = As<ConstInt>(new_shape[i]);
        CHECK(dst_dim) << "tile.fillpad_expand shape dimension " << i << " must be a constant integer";
        CHECK(dst_dim->value_ > 0) << "tile.fillpad_expand shape dimension " << i << " must be positive, got "
                                   << dst_dim->value_;
        if (auto src_dim = As<ConstInt>(tile_type->shape_[i])) {
          CHECK(dst_dim->value_ >= src_dim->value_)
              << "tile.fillpad_expand destination dimension " << i << " (" << dst_dim->value_
              << ") must be >= source dimension (" << src_dim->value_ << ")";
        }
      }

      PadValue pad_value = PadValue::zero;
      for (const auto& kv : kwargs) {
        if (kv.first == "pad_value") {
          pad_value = std::any_cast<PadValue>(kv.second);
          CHECK(pad_value != PadValue::null)
              << "tile.fillpad_expand requires pad_value to be zero/max/min, not null";
        }
      }

      // After expand the entire destination tile is valid (the padding region is
      // filled). This is a fresh allocation: retain only production
      // layout/fractal constraints, never source addressing or alias identity.
      TileView tile_view;
      tile_view.valid_shape = new_shape;
      InheritFreshTileComputeLayout(tile_view, tile_type);
      tile_view.pad = pad_value;
      return std::make_shared<TileType>(new_shape, tile_type->dtype_, std::nullopt, tile_view,
                                        MemorySpace::Vec);
    });

}  // namespace ir
}  // namespace pypto
