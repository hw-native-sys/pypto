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
 * @brief Element-wise block operations (Mul, Add, Div, Sub, and scalar variants)
 *
 * This file implements element-wise block operations that support
 * 2D tiles (at most 2 dimensions) with 2D broadcasting.
 * Operations are divided into:
 * - Tile-Tile operations (mul, add, div, sub): TileType + TileType
 * - Tile-Scalar operations (muls, adds, divs, subs): TileType + ScalarType
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
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

TypePtr DeduceBlockOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
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
TypePtr DeduceBlockOpShiftBinaryType(const std::vector<ExprPtr>& args,
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

TypePtr DeduceBlockOpScalarBinaryType(const std::vector<ExprPtr>& args,
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

TypePtr DeduceBlockOpIntScalarBinaryType(const std::vector<ExprPtr>& args,
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

REGISTER_OP("block.mul")
    .set_op_category("BlockOp")
    .set_description("Element-wise multiplication of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.mul");
    });

REGISTER_OP("block.add")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.add");
    });

REGISTER_OP("block.div")
    .set_op_category("BlockOp")
    .set_description("Element-wise division of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.div");
    });

REGISTER_OP("block.sub")
    .set_op_category("BlockOp")
    .set_description("Element-wise subtraction of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.sub");
    });

REGISTER_OP("block.maximum")
    .set_op_category("BlockOp")
    .set_description("Element-wise maximum of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.maximum");
    });

REGISTER_OP("block.minimum")
    .set_op_category("BlockOp")
    .set_description("Element-wise minimum of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.minimum");
    });

REGISTER_OP("block.rem")
    .set_op_category("BlockOp")
    .set_description("Element-wise remainder (modulo) of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.rem");
    });

REGISTER_OP("block.muls")
    .set_op_category("BlockOp")
    .set_description("Element-wise multiplication of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.muls");
    });

REGISTER_OP("block.adds")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.adds");
    });

REGISTER_OP("block.divs")
    .set_op_category("BlockOp")
    .set_description("Element-wise division of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.divs");
    });

REGISTER_OP("block.subs")
    .set_op_category("BlockOp")
    .set_description("Element-wise subtraction of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.subs");
    });

REGISTER_OP("block.rems")
    .set_op_category("BlockOp")
    .set_description("Element-wise remainder (modulo) of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.rems");
    });

REGISTER_OP("block.shl")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise left shift of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpShiftBinaryType(args, kwargs, "block.shl");
    });

REGISTER_OP("block.shls")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise left shift of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpIntScalarBinaryType(args, kwargs, "block.shls");
    });

REGISTER_OP("block.shr")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise right shift of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpShiftBinaryType(args, kwargs, "block.shr");
    });

REGISTER_OP("block.shrs")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise right shift of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpIntScalarBinaryType(args, kwargs, "block.shrs");
    });

REGISTER_OP("block.maxs")
    .set_op_category("BlockOp")
    .set_description("Element-wise maximum of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.maxs");
    });

REGISTER_OP("block.mins")
    .set_op_category("BlockOp")
    .set_description("Element-wise minimum of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.mins");
    });

REGISTER_OP("block.and")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise AND of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.and", true);
    });

REGISTER_OP("block.ands")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise AND of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpIntScalarBinaryType(args, kwargs, "block.ands");
    });

REGISTER_OP("block.or")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise OR of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.or", true);
    });

REGISTER_OP("block.ors")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise OR of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpIntScalarBinaryType(args, kwargs, "block.ors");
    });

// Tile-tile ternary ops with a tmp buffer as the third argument.
// When require_int is true (bitwise ops like xor), both tile dtypes must be integer.
TypePtr DeduceBlockOpTernaryType(const std::vector<ExprPtr>& args,
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
TypePtr DeduceBlockOpTriTileType(const std::vector<ExprPtr>& args,
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
TypePtr DeduceBlockOpTileScalarTileType(const std::vector<ExprPtr>& args,
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

TypePtr DeduceBlockOpXorScalarType(const std::vector<ExprPtr>& args,
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

REGISTER_OP("block.xor")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise XOR of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpTernaryType(args, kwargs, "block.xor", true);
    });

REGISTER_OP("block.xors")
    .set_op_category("BlockOp")
    .set_description("Element-wise bitwise XOR of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpXorScalarType(args, kwargs, "block.xors");
    });

REGISTER_OP("block.prelu")
    .set_op_category("BlockOp")
    .set_description("Element-wise parametric ReLU of a tile with slope tile and temporary buffer")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("slope", "Slope tile (TileType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpTernaryType(args, kwargs, "block.prelu");
    });

REGISTER_OP("block.addc")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of three tiles (lhs + rhs + rhs2) with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpTriTileType(args, kwargs, "block.addc");
    });

REGISTER_OP("block.subc")
    .set_op_category("BlockOp")
    .set_description("Element-wise subtraction of three tiles (lhs - rhs - rhs2) with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpTriTileType(args, kwargs, "block.subc");
    });

REGISTER_OP("block.addsc")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of tile, scalar, and tile (lhs + scalar + rhs2)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpTileScalarTileType(args, kwargs, "block.addsc");
    });

REGISTER_OP("block.subsc")
    .set_op_category("BlockOp")
    .set_description("Element-wise subtraction of tile, scalar, and tile (lhs - scalar - rhs2)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpTileScalarTileType(args, kwargs, "block.subsc");
    });

REGISTER_OP("block.lrelu")
    .set_op_category("BlockOp")
    .set_description("Element-wise leaky ReLU of a tile with scalar slope (max(x, slope*x))")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("slope", "Scalar slope for negative values (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.lrelu");
    });

// Type deduction for block.sel (MaskTile x Tile x Tile -> Tile)
// The mask tile encodes per-element predicates in a target-defined layout; its dtype/shape
// do not influence the output type.  Output type is derived from lhs and rhs only.
TypePtr DeduceBlockSelType(const std::vector<ExprPtr>& args,
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

REGISTER_OP("block.sel")
    .set_op_category("BlockOp")
    .set_description(
        "Per-element selection between two tiles using a predicate mask tile. "
        "Maps to the TSEL hardware intrinsic.")
    .add_argument("mask", "Predicate mask tile; encoding is target-defined (TileType)")
    .add_argument("lhs", "Source tile 0, selected where mask is true (TileType)")
    .add_argument("rhs", "Source tile 1, selected where mask is false (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockSelType(args, kwargs, "block.sel");
    });

// Type deduction for block.sels (Tile x Tile x Scalar -> Tile)
TypePtr DeduceBlockSelScalarType(const std::vector<ExprPtr>& args,
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

REGISTER_OP("block.sels")
    .set_op_category("BlockOp")
    .set_description("Select between two tiles based on a scalar mode. Maps to the TSELS hardware intrinsic.")
    .add_argument("lhs", "Source tile 0 (TileType)")
    .add_argument("rhs", "Source tile 1 (TileType)")
    .add_argument("select_mode", "Scalar select mode (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockSelScalarType(args, kwargs, "block.sels");
    });

// Type deduction for block.cmp and block.cmps (comparison operations)
TypePtr DeduceBlockCmpType(const std::vector<ExprPtr>& args,
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

REGISTER_OP("block.cmp")
    .set_op_category("BlockOp")
    .set_description("Element-wise comparison of two tiles (returns boolean tile)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockCmpType(args, kwargs, "block.cmp", false);
    });

REGISTER_OP("block.cmps")
    .set_op_category("BlockOp")
    .set_description("Element-wise comparison of tile and scalar (returns boolean tile)")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockCmpType(args, kwargs, "block.cmps", true);
    });

REGISTER_OP("block.fillpad")
    .set_op_category("BlockOp")
    .set_description("Fill destination tile with source tile data and pad remaining elements")
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator block.fillpad requires exactly 1 argument, but got "
                              << args.size();

      // Argument must be TileType
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type) << "The operator block.fillpad requires first argument to be a TileType, but got "
                       << args[0]->GetType()->TypeName();

      // Return same TileType
      TileView tile_view;
      tile_view.valid_shape = GetValidShape(tile_type);
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
    });

}  // namespace ir
}  // namespace pypto
