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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// Extract valid_shape from a TileType's TileView, falling back to static shape.
static std::vector<ExprPtr> GetValidShape(const std::shared_ptr<const TileType>& tile_type) {
  if (tile_type->tile_view_ && !tile_type->tile_view_->valid_shape.empty()) {
    return tile_type->tile_view_->valid_shape;
  }
  return tile_type->shape_;
}

TypePtr DeduceTileOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                                          const std::string& op_name, bool require_int = false) {
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

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shapes (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
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

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shapes (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
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

  // Result has same shape as tile, with promoted dtype
  auto result_dtype = PromoteDataTypes(tile_type->dtype_, scalar_type->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tile_type->dtype_.ToString() << " and " << scalar_type->dtype_.ToString();

  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  return std::make_shared<TileType>(tile_type->shape_, *result_dtype, std::nullopt, tile_view);
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
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.rem");
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
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.rems");
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

REGISTER_OP("tile.maxs")
    .set_op_category("TileOp")
    .set_description("Element-wise maximum of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.maxs");
    });

REGISTER_OP("tile.mins")
    .set_op_category("TileOp")
    .set_description("Element-wise minimum of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.mins");
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

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shapes (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
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

  // TODO(YunjiQin): assumes all src tiles have the same valid_shape; may need refinement
  // for cases where tiles have different valid_shapes (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
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

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs tiles have different valid_shapes (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
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

// Type deduction for tile.sel (MaskTile x Tile x Tile -> Tile)
// The mask tile encodes per-element predicates in a target-defined layout; its dtype/shape
// do not influence the output type.  Output type is derived from lhs and rhs only.
TypePtr DeduceTileSelType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                          const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
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

  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tile_type1->dtype_.ToString() << " and " << tile_type2->dtype_.ToString();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << FormatShape(tile_type1->shape_) << " and "
                                  << FormatShape(tile_type2->shape_);

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shapes (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
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
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
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

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shapes (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
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

    // Result has same shape as tile, with promoted dtype
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, scalar_type->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                        << tile_type1->dtype_.ToString() << " and " << scalar_type->dtype_.ToString();

    TileView tile_view;
    tile_view.valid_shape = GetValidShape(tile_type1);
    return std::make_shared<TileType>(tile_type1->shape_, *result_dtype, std::nullopt, tile_view);
  } else {
    // Second argument must be TileType
    auto tile_type2 = As<TileType>(args[1]->GetType());
    CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                      << args[1]->GetType()->TypeName();

    // Use broadcasting
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
    CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                        << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                    << FormatShape(tile_type1->shape_) << " and "
                                    << FormatShape(tile_type2->shape_);

    // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
    // for cases where lhs and rhs have different valid_shapes (e.g. after broadcasting).
    TileView tile_view;
    tile_view.valid_shape = GetValidShape(tile_type1);
    return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
  }
}

REGISTER_OP("tile.cmp")
    .set_op_category("TileOp")
    .set_description("Element-wise comparison of two tiles (returns boolean tile)")
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
    .set_description("Element-wise comparison of tile and scalar (returns boolean tile)")
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
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.fillpad requires exactly 1 argument, but got "
                              << args.size();

      // Argument must be TileType
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type) << "The operator tile.fillpad requires first argument to be a TileType, but got "
                       << args[0]->GetType()->TypeName();

      // Get pad_value from kwargs, default to TilePad::zero
      TilePad pad_value = TilePad::zero;
      for (const auto& kv : kwargs) {
        if (kv.first == "pad_value") {
          pad_value = std::any_cast<TilePad>(kv.second);
          CHECK(pad_value != TilePad::null) << "tile.fillpad requires pad_value to be zero/max/min, not null";
        }
      }

      // Return TileType with pad value set in tile_view
      // After fillpad, the entire tile is valid (padding region is now filled with pad_value)
      TileView tile_view;
      tile_view.valid_shape = tile_type->shape_;  // Expand valid_shape to full shape
      tile_view.pad = pad_value;
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_, tile_view);
    });

}  // namespace ir
}  // namespace pypto
