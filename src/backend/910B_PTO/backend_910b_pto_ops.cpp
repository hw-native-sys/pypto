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

#include <sstream>
#include <string>

#include "pypto/backend/910B_PTO/backend_910b_pto.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/common.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

using ir::As;
using ir::CallPtr;
using ir::PipeType;
using ir::TensorType;
using ir::Var;

// ============================================================================
// Helper Functions for PTO Code Generation
// ============================================================================

// Helper function for binary tile-tile operations
static std::string MakeBinaryTileTileCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                                codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string result = codegen.GetCurrentResultTarget();
  std::ostringstream oss;
  oss << result << " = " << pto_op_name << "(" << lhs << ", " << rhs << ")";
  return oss.str();
}

// Helper function for binary tile-scalar operations
static std::string MakeBinaryTileScalarCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                                  codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string result = codegen.GetCurrentResultTarget();
  std::ostringstream oss;
  oss << result << " = " << pto_op_name << "(" << lhs << ", " << rhs << ")";
  return oss.str();
}

// block.load: emit pto.subview + pto.tload (same format as original IR layer codegen)
static std::string MakeBlockLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tensor = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tensor) << "block.load first argument must be a Var";

  int64_t row_off = codegen.GetConstIntValue(op->args_[1]);
  int64_t col_off = codegen.GetConstIntValue(op->args_[2]);
  int64_t height = codegen.GetConstIntValue(op->args_[3]);
  int64_t width = codegen.GetConstIntValue(op->args_[4]);

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "block.load tensor argument must have TensorType";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK(!tile_buf.empty()) << "block.load requires assignment target (tile_buf)";

  std::string tile_view = codegen.NewTemp();
  std::ostringstream subview_line;
  subview_line << tile_view << " = pto.subview " << tensor_view;
  subview_line << ", offsets = [" << codegen.GetIndexConstant(row_off) << ", ";
  subview_line << codegen.GetIndexConstant(col_off) << "]";
  subview_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", ";
  subview_line << codegen.GetIndexConstant(width) << "]";
  subview_line << " : !pto.tensor_view<2x" << dtype_str << "> -> !pto.tile_view<";
  subview_line << height << "x" << width << "x" << dtype_str << ">";
  codegen.Emit(subview_line.str());

  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << tile_view;
  tload_line << " : !pto.tile_view<" << height << "x" << width << "x" << dtype_str << ">) outs(";
  tload_line << tile_buf << " : !pto.tile_buf<loc=ub, dtype=" << dtype_str;
  tload_line << ", rows=" << height << ", cols=" << width;
  tload_line << ", v_row=" << height << ", v_col=" << width;
  tload_line << ", blayout=row_major, slayout=none_box, fractal=512, pad=0>)";
  codegen.Emit(tload_line.str());
  return "";  // Multi-line emission
}

// block.store: emit pto.subview + pto.tstore (same format as original IR layer codegen)
static std::string MakeBlockStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tile = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tile) << "block.store first argument must be a Var";

  int64_t row_off = codegen.GetConstIntValue(op->args_[1]);
  int64_t col_off = codegen.GetConstIntValue(op->args_[2]);
  int64_t height = codegen.GetConstIntValue(op->args_[3]);
  int64_t width = codegen.GetConstIntValue(op->args_[4]);
  auto output_tensor = As<Var>(op->args_[5]);
  INTERNAL_CHECK(output_tensor) << "block.store output_tensor must be a Var";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "block.store output_tensor must have TensorType";

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);
  std::string tile_view = codegen.NewTemp();

  std::ostringstream subview_line;
  subview_line << tile_view << " = pto.subview " << tensor_view;
  subview_line << ", offsets = [" << codegen.GetIndexConstant(row_off) << ", ";
  subview_line << codegen.GetIndexConstant(col_off) << "]";
  subview_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", ";
  subview_line << codegen.GetIndexConstant(width) << "]";
  subview_line << " : !pto.tensor_view<2x" << dtype_str << "> -> !pto.tile_view<";
  subview_line << height << "x" << width << "x" << dtype_str << ">";
  codegen.Emit(subview_line.str());

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  tstore_line << " : !pto.tile_buf<loc=ub, dtype=" << dtype_str << ", rows=" << height;
  tstore_line << ", cols=" << width << ", v_row=" << height << ", v_col=" << width;
  tstore_line << ", blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(";
  tstore_line << tile_view << " : !pto.tile_view<" << height << "x" << width << "x" << dtype_str << ">)";
  codegen.Emit(tstore_line.str());
  return "";  // Multi-line emission
}

// Helper function for block.alloc (no-op: allocation handled elsewhere)
static std::string MakeBlockAllocCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No MLIR emission - pto.alloc_tile generated from MemRefs in TileTypes
}

// ============================================================================
// Elementwise Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileTileCodegenPTO("pto.tmul", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileTileCodegenPTO("pto.taddc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileTileCodegenPTO("pto.tdiv", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileTileCodegenPTO("pto.tsub", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.maximum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileTileCodegenPTO("pto.tmax", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileScalarCodegenPTO("pto.tmuls", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileScalarCodegenPTO("pto.tadds", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.divs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileScalarCodegenPTO("pto.tdivs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryTileScalarCodegenPTO("pto.tsubs", op, codegen);
    });

// ============================================================================
// Memory Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.load")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockLoadCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.store")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockStoreCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.alloc")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockAllocCodegenPTO(op, codegen);
    });

}  // namespace backend
}  // namespace pypto
