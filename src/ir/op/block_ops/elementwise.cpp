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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                                           const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Both arguments must be TileType
  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());

  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
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

  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
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

  return std::make_shared<TileType>(tile_type->shape_, *result_dtype);
}

// ============================================================================
// PTO Codegen Helpers (return MLIR string; caller does Emit)
// ============================================================================

// Binary tile-tile: build MLIR line using PTOCodegen API, return string (caller Emits)
auto MakeBinaryTileTileCodegenPTO(const std::string& pto_op) {
  return [pto_op](const CallPtr& op, codegen::PTOCodegen& codegen) -> std::string {
    auto lhs = As<ir::Var>(op->args_[0]);
    auto rhs = As<ir::Var>(op->args_[1]);
    INTERNAL_CHECK(lhs && rhs) << "Both arguments must be Var";

    std::string lhs_buf = codegen.GetVarName(lhs);
    std::string rhs_buf = codegen.GetVarName(rhs);
    std::string result_buf = codegen.GetCurrentResultTarget();
    if (result_buf.empty()) {
      result_buf = "RESULT_BUF";
    }

    std::ostringstream mlir;
    mlir << pto_op << " ins(" << lhs_buf;
    mlir << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    mlir << "blayout=row_major, slayout=none_box, fractal=512, pad=0>, " << rhs_buf;
    mlir << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    mlir << "blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(";
    mlir << result_buf;
    mlir << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    mlir << "blayout=row_major, slayout=none_box, fractal=512, pad=0>)";
    return mlir.str();
  };
}

// Tile-scalar: use GetOrEmitFloatConstant for constant (emits to constants section), return instruction line
// only
auto MakeBinaryTileScalarCodegenPTO(const std::string& pto_op) {
  return [pto_op](const CallPtr& op, codegen::PTOCodegen& codegen) -> std::string {
    auto tile = As<ir::Var>(op->args_[0]);
    INTERNAL_CHECK(tile) << "First argument must be Var";

    std::string tile_buf = codegen.GetVarName(tile);
    std::string result_buf = codegen.GetCurrentResultTarget();
    if (result_buf.empty()) {
      result_buf = "RESULT_BUF";
    }

    double scalar_val = 0.0;
    std::string scalar_type = "f32";
    if (auto const_float = As<ir::ConstFloat>(op->args_[1])) {
      scalar_val = const_float->value_;
      scalar_type = "f32";
    } else if (auto const_int = As<ir::ConstInt>(op->args_[1])) {
      scalar_val = static_cast<double>(const_int->value_);
      scalar_type = "i32";
    } else if (auto scalar_t = As<ScalarType>(op->args_[1]->GetType())) {
      scalar_type = codegen.GetTypeString(scalar_t->dtype_);
    }

    std::string scalar_const = codegen.GetOrEmitFloatConstant(scalar_val, scalar_type);

    std::ostringstream mlir;
    mlir << pto_op << " ins(" << tile_buf;
    mlir << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    mlir << "blayout=row_major, slayout=none_box, fractal=512, pad=0>, " << scalar_const;
    mlir << " : " << scalar_type << ") outs(" << result_buf;
    mlir << " : !pto.tile_buf<loc=ub, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, ";
    mlir << "blayout=row_major, slayout=none_box, fractal=512, pad=0>)";
    return mlir.str();
  };
}

// ============================================================================
// Registration Function for Block Element-wise Operations
// ============================================================================

// Helper lambda factory for binary elementwise operations
auto MakeBinaryElementwiseCodegenCCE(const std::string& isa_name) {
  return [isa_name](const CallPtr& op, codegen::CCECodegen& codegen) -> std::string {
    std::string target_var = codegen.GetCurrentResultTarget();
    std::ostringstream args_str;
    args_str << target_var;
    for (const auto& arg : op->args_) {
      args_str << ", " << codegen.GetExprAsCode(arg);
    }
    codegen.Emit(isa_name + "(" + args_str.str() + ");");
    return target_var;
  };
}

REGISTER_OP("block.mul")
    .set_op_category("BlockOp")
    .set_description("Element-wise multiplication of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.mul");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TMUL"))
    .f_codegen_pto(MakeBinaryTileTileCodegenPTO("pto.tmul"));

REGISTER_OP("block.add")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.add");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TADD"))
    .f_codegen_pto(MakeBinaryTileTileCodegenPTO("pto.taddc"));

REGISTER_OP("block.div")
    .set_op_category("BlockOp")
    .set_description("Element-wise division of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.div");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TDIV"))
    .f_codegen_pto(MakeBinaryTileTileCodegenPTO("pto.tdiv"));

REGISTER_OP("block.sub")
    .set_op_category("BlockOp")
    .set_description("Element-wise subtraction of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.sub");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TSUB"))
    .f_codegen_pto(MakeBinaryTileTileCodegenPTO("pto.tsub"));

REGISTER_OP("block.maximum")
    .set_op_category("BlockOp")
    .set_description("Element-wise maximum of two tiles with broadcasting")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.maximum");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TMAX"))
    .f_codegen_pto(MakeBinaryTileTileCodegenPTO("pto.tmax"));

REGISTER_OP("block.muls")
    .set_op_category("BlockOp")
    .set_description("Element-wise multiplication of tile and scalar")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.muls");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TMULS"))
    .f_codegen_pto(MakeBinaryTileScalarCodegenPTO("pto.tmuls"));

REGISTER_OP("block.adds")
    .set_op_category("BlockOp")
    .set_description("Element-wise addition of tile and scalar")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.adds");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TADDS"))
    .f_codegen_pto(MakeBinaryTileScalarCodegenPTO("pto.tadds"));

REGISTER_OP("block.divs")
    .set_op_category("BlockOp")
    .set_description("Element-wise division of tile and scalar")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.divs");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TDIVS"))
    .f_codegen_pto(MakeBinaryTileScalarCodegenPTO("pto.tdivs"));

REGISTER_OP("block.subs")
    .set_op_category("BlockOp")
    .set_description("Element-wise subtraction of tile and scalar")
    .set_pipe(PipeType::V)
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpScalarBinaryType(args, kwargs, "block.subs");
    })
    .f_codegen_cce(MakeBinaryElementwiseCodegenCCE("TSUBS"))
    .f_codegen_pto(MakeBinaryTileScalarCodegenPTO("pto.tsubs"));

}  // namespace ir
}  // namespace pypto
