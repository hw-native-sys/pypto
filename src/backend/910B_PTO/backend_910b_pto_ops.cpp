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

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/backend/910B_PTO/backend_910b_pto.h"
#include "pypto/backend/common/backend.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
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

const std::vector<std::string> cmp_modes = {"EQ", "NE", "LT", "LE", "GT", "GE"};
const std::vector<std::string> round_modes = {"NONE", "RINT",  "ROUND", "FLOOR",
                                              "CEIL", "TRUNC", "ODD",   "CAST_RINT"};

// Helper function for input & output generation (with type annotations)
static std::string GenerateInsOutsClause(const CallPtr& op, codegen::PTOCodegen& codegen,
                                         const std::string& config_attr = "") {
  size_t args_num = op->args_.size();
  std::ostringstream oss;

  // Build ins clause with operand names
  oss << "ins(";
  for (size_t input_idx = 0; input_idx < args_num; ++input_idx) {
    std::string operand = codegen.GetExprAsCode(op->args_[input_idx]);
    if (input_idx == 0) {
      oss << operand;
    } else {
      oss << ", " << operand;
    }
  }

  // Add type annotations after colon
  std::string type_annot;
  for (size_t input_idx = 0; input_idx < args_num; ++input_idx) {
    std::string annot = codegen.GetExprTypeAnnotation(op->args_[input_idx]);
    if (!annot.empty()) {
      if (!type_annot.empty()) type_annot += ", ";
      type_annot += annot;
    }
  }
  if (!type_annot.empty()) {
    oss << " : " << type_annot;
  }

  if (!config_attr.empty()) {
    oss << config_attr;
  }

  // Build outs clause with type annotation
  std::string result_target = codegen.GetCurrentResultTarget();
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  oss << ") outs(" << result_target;
  if (!result_type.empty()) {
    oss << " : " << result_type;
  }
  oss << ")";
  return oss.str();
}

// Helper function for Unary operations
static std::string MakeUnaryCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Operation:[" << pto_op_name << "] requires 1 argument.";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

// Helper function for Binary operations
static std::string MakeBinaryCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments.";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

// Helper function for Ternary operations
static std::string MakeTernaryCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "Operation:[" << pto_op_name << "] requires 3 arguments.";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

// Helper function for Quaternary operations
static std::string MakeQuaternaryCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                            codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "Operation:[" << pto_op_name << "] requires 4 arguments.";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

// Helper function for Quinary operations
static std::string MakeQuinaryCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 5) << "Operation:[" << pto_op_name << "] requires 5 arguments.";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

// Helper function for Binary Tile cmp operations
static std::string MakeTileCmpCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Tile_cmp requires 2 arguments.";
  int mode = op->GetKwarg<int>("mode");
  CHECK(mode >= 0 && mode < static_cast<int>(cmp_modes.size())) << "Tile cmp mode out of range: " << mode;
  std::string config_attr = "{cmpMode = #pto<cmp " + cmp_modes.at(mode) + ">}";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

// Helper function for Tile cvt operations
static std::string MakeTileCvtCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Tile_cvt op requires 1 argument.";
  int mode = op->GetKwarg<int>("mode");
  CHECK(mode >= 0 && mode < static_cast<int>(round_modes.size())) << "Round mode out of range: " << mode;
  std::string config_attr = "{rmode = #pto<round_mode " + round_modes.at(mode) + ">}";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

// Helper function for full op
static std::string MakeFullCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "full op requires 2 arguments, got " << op->args_.size();
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  std::string scalar_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
  std::ostringstream oss;
  oss << pto_op_name << " ins(" << scalar;
  if (!scalar_type.empty()) oss << " : " << scalar_type;
  oss << ") outs(" << dst;
  if (!dst_type.empty()) oss << " : " << dst_type;
  oss << ")";
  codegen.Emit(oss.str());
  return "";
}

// block.load: emit pto.partition_view + pto.tload
static std::string MakeBlockLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tensor = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tensor) << "block.load first argument must be a Var";

  // Extract offsets tuple
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "block.load second argument must be a tuple (offsets)";

  // Extract shapes tuple
  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "block.load third argument must be a tuple (shapes)";

  // Extract 2D offset and size values from tuples
  int64_t row_off = codegen.GetConstIntValue(offsets_tuple->elements_[0]);
  int64_t col_off = codegen.GetConstIntValue(offsets_tuple->elements_[1]);
  int64_t height = codegen.GetConstIntValue(shapes_tuple->elements_[0]);
  int64_t width = codegen.GetConstIntValue(shapes_tuple->elements_[1]);

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "block.load tensor argument must have TensorType";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK(!tile_buf.empty()) << "block.load requires assignment target (tile_buf)";

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();
  std::string partition_type = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                               std::to_string(width) + "x" + dtype_str + ">";

  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [" << codegen.GetIndexConstant(row_off) << ", ";
  partition_line << codegen.GetIndexConstant(col_off) << "]";
  partition_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", ";
  partition_line << codegen.GetIndexConstant(width) << "]";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << partition_view << " : " << partition_type << ") outs(";
  tload_line << tile_buf << " : " << tile_buf_type << ")";
  codegen.Emit(tload_line.str());
  return "";  // Multi-line emission
}

// block.store: emit pto.partition_view + pto.tstore
static std::string MakeBlockStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tile = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tile) << "block.store first argument must be a Var";

  // Extract offsets tuple
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "block.store second argument must be a tuple (offsets)";

  // Extract shapes tuple
  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "block.store third argument must be a tuple (shapes)";

  // Extract 2D offset and size values from tuples
  int64_t row_off = codegen.GetConstIntValue(offsets_tuple->elements_[0]);
  int64_t col_off = codegen.GetConstIntValue(offsets_tuple->elements_[1]);
  int64_t height = codegen.GetConstIntValue(shapes_tuple->elements_[0]);
  int64_t width = codegen.GetConstIntValue(shapes_tuple->elements_[1]);
  auto output_tensor = As<Var>(op->args_[3]);
  INTERNAL_CHECK(output_tensor) << "block.store output_tensor must be a Var";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "block.store output_tensor must have TensorType";

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string partition_type = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                               std::to_string(width) + "x" + dtype_str + ">";

  // Get tile_buf type from the tile variable's TileType
  std::string tile_buf_type;
  if (auto tile_type = As<ir::TileType>(tile->GetType())) {
    if (tile_type->memref_.has_value()) {
      tile_buf_type = codegen.GetTileBufTypeString(tile_type->memref_.value().get());
    }
  }

  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [" << codegen.GetIndexConstant(row_off) << ", ";
  partition_line << codegen.GetIndexConstant(col_off) << "]";
  partition_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", ";
  partition_line << codegen.GetIndexConstant(width) << "]";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  if (!tile_buf_type.empty()) {
    tstore_line << " : " << tile_buf_type;
  }
  tstore_line << ") outs(" << partition_view << " : " << partition_type << ")";
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
// Tile x Tile Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tadd", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tsub", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tmul", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tdiv", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.rem")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.trem", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.and")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tand", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.or")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tor", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.xor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.txor", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.shl")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tshl", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.shr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tshr", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.maximum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tmax", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.minimum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tmin", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.prelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tprelu", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.cmp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCmpCodegenPTO("pto.tcmp", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.tabs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.texp", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.tlog", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.tsqrt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.rsqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.trsqrt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.recip")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.trecip", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.tneg", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.not")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.tnot", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.trelu", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCvtCodegenPTO("pto.tcvt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.addc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTernaryCodegenPTO("pto.taddc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.subc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTernaryCodegenPTO("pto.tsubc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.sel")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTernaryCodegenPTO("pto.tsel", op, codegen);
    });

// ============================================================================
// Tile x Scalar Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tadds", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tsubs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tmuls", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.divs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tdivs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.rems")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.trems", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.ands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tands", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.ors")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tors", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.xors")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.txors", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.shls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tshls", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.shrs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tshrs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.maxs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tmaxs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.mins")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tmins", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeFullCodegenPTO("pto.texpands", op, codegen);
    });

// Not Implemented: tlrelu tcmps taddsc tsubsc tsels

// ============================================================================
// Matrix Multiplication Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.matmul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tmatmul", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.matmul_mx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeQuaternaryCodegenPTO("pto.tmatmul.mx", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.matmul_mx_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeQuinaryCodegenPTO("pto.tmatmul.mx.acc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.matmul_mx_bias")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeQuinaryCodegenPTO("pto.tmatmul.mx.bias", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.matmul_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTernaryCodegenPTO("pto.tmatmul.acc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.matmul_bias")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTernaryCodegenPTO("pto.tmatmul.bias", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.gemv")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.tgemv", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.gemv_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTernaryCodegenPTO("pto.tgemv.acc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.gemv_bias")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTernaryCodegenPTO("pto.tgemv.bias", op, codegen);
    });

// ============================================================================
// Data Movement/Layout Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTernaryCodegenPTO("pto.ttrans", op, codegen);
    });

// ============================================================================
// Axis reduction/expansion Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.row_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.trowsum", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.row_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.trowmax", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.row_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.trowmin", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.row_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.trowexpanddiv", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.row_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.trowexpandmul", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.row_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryCodegenPTO("pto.trowexpandsub", op, codegen);
    });

// ============================================================================
// Padding Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.fillpad")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenPTO("pto.tfillpad", op, codegen);
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
