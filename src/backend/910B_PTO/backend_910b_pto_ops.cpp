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
#include <memory>
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
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

using ir::As;
using ir::CallPtr;
using ir::TensorType;
using ir::Var;

// As<Var> uses exact ObjectKind match and won't match IterArg.
// This helper matches both Var and IterArg (which inherits from Var).
inline std::shared_ptr<const Var> AsVarLike(const ir::ExprPtr& expr) {
  if (auto v = As<Var>(expr)) return v;
  if (auto ia = As<ir::IterArg>(expr)) return ia;
  return nullptr;
}

// ============================================================================
// Helper Functions for PTO Code Generation
// ============================================================================

const std::vector<std::string> cmp_modes = {"eq", "ne", "lt", "le", "gt", "ge"};
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

  if (!config_attr.empty()) {
    oss << config_attr;
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

// Helper function for N-ary operations (unary, binary, ternary, etc.)
static std::string MakeNaryCodegenPTO(const std::string& pto_op_name, size_t arity, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == arity) << "Operation:[" << pto_op_name << "] requires " << arity << " argument"
                                   << (arity != 1 ? "s" : "") << ", but got " << op->args_.size();
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

// Helper function for StoreFP
static std::string MakeStoreFPCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "Operation:[" << pto_op_name << "] requires 3 arguments, but got "
                               << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string fp = codegen.GetExprAsCode(op->args_[1]);
  std::string mem = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(pto_op_name + " ins(" + src + ", " + fp + ") outs(" + mem + ")");
  return "";
}

// Helper function for Binary Tile cmp operations
static std::string MakeTileCmpCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
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
  CHECK(op->args_.size() == 1) << "Operation:[" << pto_op_name << "] requires 1 argument, but got "
                               << op->args_.size();
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
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
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

// Helper function for cmps
static std::string MakeCmpsCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
  int mode = op->GetKwarg<int>("mode");
  CHECK(mode >= 0 && mode < static_cast<int>(cmp_modes.size())) << "Tile cmp mode out of range: " << mode;
  std::string config_attr = "{cmpMode = #pto<cmp " + cmp_modes.at(mode) + ">}";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

// Helper function for Assign
static std::string MakeAssignCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string addr = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit(pto_op_name + " ins(" + tile + ", " + addr + ")");
  return "";
}

// Helper function for Ci
static std::string MakeCiCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                    codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Operation:[" << pto_op_name << "] requires 1 argument, but got "
                               << op->args_.size();
  bool descending = op->GetKwarg<bool>("descending");
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string config_attr = descending ? "{descending = true}" : "{descending = false}";
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(pto_op_name + " ins(" + src + " " + config_attr + ") outs(" + dst + ")");
  return "";
}

// TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
// Helper function for Sort32
static std::string MakeSort32CodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Operation:[" << pto_op_name << "] requires 1 argument, but got "
                               << op->args_.size();
  // std::string src = codegen.GetExprAsCode(op->args_[0]);
  // std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(pto_op_name);
  return "";
}

// TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
// Helper function for MrgSort
static std::string MakeMrgSortCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
  // std::string src = codegen.GetExprAsCode(op->args_[0]);
  // std::string blockLen = codegen.GetExprAsCode(op->args_[1]);
  // std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(pto_op_name);
  return "";
}

// Helper function for Print
static std::string MakePrintCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Operation:" << pto_op_name << "] requires 1 argument, but got "
                               << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  codegen.Emit(pto_op_name + " ins(" + src + " | !pto.partition_tensor_view<MxNxdtype>)");
  return "";
}

// tile.load: emit pto.subview + pto.tload (same format as original IR layer codegen)
static std::string MakeTileLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tensor = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(tensor) << "tile.load first argument must be a Var or IterArg";

  // Extract offsets tuple
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "tile.load second argument must be a tuple (offsets)";

  // Extract shapes tuple
  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "tile.load third argument must be a tuple (shapes)";

  // Extract 2D offset and size values from tuples
  auto row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  auto col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  int64_t height = codegen.GetConstIntValue(shapes_tuple->elements_[0]);
  int64_t width = codegen.GetConstIntValue(shapes_tuple->elements_[1]);

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "tile.load tensor argument must have TensorType";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK(!tile_buf.empty()) << "tile.load requires assignment target (tile_buf)";

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();
  std::string partition_type = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                               std::to_string(width) + "x" + dtype_str + ">";

  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [" << row_off << ", " << col_off << "]";
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

// tile.store: emit pto.partition_view + pto.tstore
static std::string MakeTileStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(tile) << "tile.store first argument must be a Var or IterArg";

  // Extract offsets tuple
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "tile.store second argument must be a tuple (offsets)";

  // Get tile type and extract shape for height/width
  // The tile's valid_shape is required and used to determine the store dimensions.
  auto tile_type = As<ir::TileType>(tile->GetType());
  INTERNAL_CHECK(tile_type) << "tile.store first argument must have TileType";
  INTERNAL_CHECK(tile_type->tile_view_.has_value()) << "tile.store tile must have TileView with valid_shape";
  auto& valid_shape = tile_type->tile_view_->valid_shape;
  INTERNAL_CHECK(valid_shape.size() == 2) << "tile.store tile valid_shape must be 2D";

  // Extract 2D offset values
  auto row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  auto col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);

  // Extract height/width from tile's valid shape (may be static or dynamic)
  auto height_code = codegen.GetExprAsCode(valid_shape[0]);
  auto width_code = codegen.GetExprAsCode(valid_shape[1]);

  auto output_tensor = AsVarLike(op->args_[2]);
  INTERNAL_CHECK(output_tensor) << "tile.store output_tensor must be a Var or IterArg";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "tile.store output_tensor must have TensorType";

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);

  // Build partition_type: use concrete dims for static shapes, ? for dynamic
  std::string height_dim = "?", width_dim = "?";
  if (auto h = As<ir::ConstInt>(valid_shape[0])) height_dim = std::to_string(h->value_);
  if (auto w = As<ir::ConstInt>(valid_shape[1])) width_dim = std::to_string(w->value_);
  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string partition_type =
      "!pto.partition_tensor_view<" + height_dim + "x" + width_dim + "x" + dtype_str + ">";

  // Get tile_buf type via GetExprTypeAnnotation which correctly handles
  // dynamically-allocated buffers (e.g., reshape outputs in extra_tile_buf_types_)
  std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [" << row_off << ", " << col_off << "]";
  partition_line << ", sizes = [" << height_code << ", " << width_code << "]";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  if (!tile_buf_type.empty()) {
    tstore_line << " : " << tile_buf_type;
  }
  tstore_line << ") outs(" << partition_view << " : " << partition_type << ")";
  codegen.Emit(tstore_line.str());

  // Register the store result variable to the same tensor view, so that
  // subsequent yields can reference it correctly in scf.for/scf.if.
  std::string result_var = codegen.GetCurrentResultTarget();
  if (!result_var.empty()) {
    codegen.RegisterTensorView(result_var, tensor_view);
    codegen.RegisterVarToMlir(result_var, tensor_view);
  }

  return "";
}

// Helper function for tile.alloc (no-op: allocation handled elsewhere)
static std::string MakeTileAllocCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No MLIR emission - pto.alloc_tile generated from MemRefs in TileTypes
}

static std::string MakeTensorDimCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.dim requires 2 arguments, but got " << op->args_.size();
  auto input_tensor = ir::As<ir::TensorType>(op->args_[0]->GetType());
  CHECK(input_tensor) << "tensor.dim need TensorType for first arg, but got "
                      << op->args_[0]->GetType()->TypeName();
  auto axis = codegen.GetConstIntValue(op->args_[1]);
  CHECK(axis >= 0 && static_cast<size_t>(axis) < input_tensor->shape_.size())
      << "tensor.dim axis " << axis << " out of range for tensor with rank " << input_tensor->shape_.size();
  auto shape = input_tensor->shape_[axis];
  std::string shape_name;
  // dynamic shape
  if (auto dyn_shape = ir::As<ir::Var>(shape)) {
    shape_name = codegen.GetVarName(dyn_shape);
  } else if (auto static_shape = ir::As<ir::ConstInt>(shape)) {  // constant shape
    shape_name = codegen.GetIndexConstant(static_shape->value_);
  } else {
    INTERNAL_CHECK(false) << "Internal error: tensor.dim shape is neither Var nor ConstInt";
  }
  // register target var to shape name so later uses (e.g., pl.range(M)) resolve correctly
  auto target_var_name = codegen.GetCurrentResultTarget();
  if (!target_var_name.empty() && !shape_name.empty()) {
    codegen.RegisterVarToMlir(target_var_name, shape_name);
  }

  return "";
}

// ============================================================================
// Table-driven registration for simple N-ary operations
// ============================================================================

struct SimpleOpEntry {
  const char* op_name;
  const char* pto_op_name;
  size_t arity;
};

// clang-format off
static const SimpleOpEntry kSimpleOps[] = {
    // Tile utility operations
    {"tile.getval",          "pto.tgetval",          2},
    {"tile.setval",          "pto.tsetval",          2},
    // Memory operations
    {"tile.mgather",         "pto.tmgather",         2},
    {"tile.mscatter",        "pto.tmscatter",        2},
    // Tile x Tile arithmetic operations
    {"tile.add",             "pto.tadd",             2},
    {"tile.sub",             "pto.tsub",             2},
    {"tile.mul",             "pto.tmul",             2},
    {"tile.div",             "pto.tdiv",             2},
    {"tile.rem",             "pto.trem",             2},
    // Tile x Tile bitwise operations
    {"tile.and",             "pto.tand",             2},
    {"tile.or",              "pto.tor",              2},
    {"tile.xor",             "pto.txor",             2},
    {"tile.shl",             "pto.tshl",             2},
    {"tile.shr",             "pto.tshr",             2},
    // Tile x Tile comparison/selection operations
    {"tile.maximum",         "pto.tmax",             2},
    {"tile.minimum",         "pto.tmin",             2},
    {"tile.prelu",           "pto.tprelu",           2},
    // Unary operations
    {"tile.abs",             "pto.tabs",             1},
    {"tile.exp",             "pto.texp",             1},
    {"tile.log",             "pto.tlog",             1},
    {"tile.sqrt",            "pto.tsqrt",            1},
    {"tile.rsqrt",           "pto.trsqrt",           1},
    {"tile.recip",           "pto.trecip",           1},
    {"tile.neg",             "pto.tneg",             1},
    {"tile.not",             "pto.tnot",             1},
    {"tile.relu",            "pto.trelu",            1},
    // Ternary operations (tile x tile + carry/select)
    {"tile.addc",            "pto.taddc",            3},
    {"tile.subc",            "pto.tsubc",            3},
    {"tile.sel",             "pto.tsel",             3},
    // Tile x Scalar operations
    {"tile.adds",            "pto.tadds",            2},
    {"tile.subs",            "pto.tsubs",            2},
    {"tile.muls",            "pto.tmuls",            2},
    {"tile.divs",            "pto.tdivs",            2},
    {"tile.rems",            "pto.trems",            2},
    {"tile.ands",            "pto.tands",            2},
    {"tile.ors",             "pto.tors",             2},
    {"tile.xors",            "pto.txors",            2},
    {"tile.shls",            "pto.tshls",            2},
    {"tile.shrs",            "pto.tshrs",            2},
    {"tile.maxs",            "pto.tmaxs",            2},
    {"tile.mins",            "pto.tmins",            2},
    {"tile.lrelu",           "pto.tlrelu",           2},
    // Ternary scalar operations (tile x scalar + carry/select)
    {"tile.addsc",           "pto.taddsc",           3},
    {"tile.subsc",           "pto.tsubsc",           3},
    {"tile.selc",            "pto.tselc",            3},
    // Axis reduction/expansion operations
    {"tile.row_sum",         "pto.trowsum",          2},
    {"tile.row_max",         "pto.trowmax",          2},
    {"tile.row_min",         "pto.trowmin",          2},
    {"tile.row_expand",      "pto.trowexpand",       1},
    {"tile.col_sum",         "pto.tcolsum",          1},
    {"tile.col_max",         "pto.tcolmax",          1},
    {"tile.col_min",         "pto.tcolmin",          1},
    {"tile.col_expand",      "pto.tcolexpand",       2},
    {"tile.row_expand_div",  "pto.trowexpanddiv",    2},
    {"tile.row_expand_mul",  "pto.trowexpandmul",    2},
    {"tile.row_expand_sub",  "pto.trowexpandsub",    2},
    // Padding operations
    {"tile.fillpad",         "pto.tfillpad",         1},
    // Matrix multiplication operations (PipeType::M → CUBE/AIC core)
    {"tile.matmul",          "pto.tmatmul",          2},
    {"tile.matmul_mx",       "pto.tmatmul.mx",       4},
    {"tile.matmul_mx_acc",   "pto.tmatmul.mx.acc",   5},
    {"tile.matmul_mx_bias",  "pto.tmatmul.mx.bias",  5},
    {"tile.matmul_acc",      "pto.tmatmul.acc",      3},
    {"tile.matmul_bias",     "pto.tmatmul.bias",     3},
    {"tile.gemv",            "pto.tgemv",            2},
    {"tile.gemv_acc",        "pto.tgemv.acc",        3},
    {"tile.gemv_bias",       "pto.tgemv.bias",       3},
    // Data movement/layout operations (PipeType::MTE1 → memory transfer, not V/M)
    {"tile.move",            "pto.tmov",             1},
    {"tile.move_fp",         "pto.tmov.fp",          2},
    {"tile.transpose",       "pto.ttrans",           3},
    {"tile.extract",         "pto.textract",         3},
    // Gather/scatter operations
    {"tile.gather",          "pto.tgather",          2},
    {"tile.gatherb",         "pto.tgatherb",         2},
    {"tile.scatter",         "pto.tscatter",         2},
    // Partial reduction operations
    {"tile.partadd",         "pto.tpartadd",         2},
    {"tile.partmax",         "pto.tpartmax",         2},
    {"tile.partmin",         "pto.tpartmin",         2},
};
// clang-format on

static void RegisterSimpleOps() {
  for (const auto& entry : kSimpleOps) {
    std::string pto_op = entry.pto_op_name;
    size_t arity = entry.arity;
    Backend910B_PTO::Instance()
        .RegisterOp(entry.op_name)
        .f_codegen([pto_op, arity](const CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeNaryCodegenPTO(pto_op, arity, op, codegen);
        });
  }
}

static const bool kSimpleOpsRegistered = [] {
  RegisterSimpleOps();
  return true;
}();

// ============================================================================
// Operations with custom codegen logic
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.load")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileLoadCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.store")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileStoreCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.alloc")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileAllocCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.create_tile")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      (void)op;
      (void)codegen_base;
      return std::string("");  // No MLIR emission - tile allocation handled by pto.alloc_tile
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.store_fp")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeStoreFPCodegenPTO("pto.tstore.fp", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.cmp")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCmpCodegenPTO("pto.tcmp", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.cast")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCvtCodegenPTO("pto.tcvt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.full")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeFullCodegenPTO("pto.texpands", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.cmps")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCmpsCodegenPTO("pto.tcmps", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.assign")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeAssignCodegenPTO("pto.tassign", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.ci")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCiCodegenPTO("pto.tci", op, codegen);
    });

// TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
REGISTER_BACKEND_OP(Backend910B_PTO, "tile.sort32")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSort32CodegenPTO("pto.tsort32", op, codegen);
    });

// TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
REGISTER_BACKEND_OP(Backend910B_PTO, "tile.mrgsort")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeMrgSortCodegenPTO("pto.tmrgsort", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.print")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakePrintCodegenPTO("pto.tprint", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tensor.dim")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorDimCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tile.reshape")
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
      CHECK(op->args_.size() == 2) << "Operation:[tile.reshape] requires 2 arguments (tile, shape), but got "
                                   << op->args_.size();
      // Only use the first argument (source tile); shape tuple is metadata
      std::string src = codegen.GetExprAsCode(op->args_[0]);
      std::string result_target = codegen.GetCurrentResultTarget();
      std::string result_type = codegen.GetCurrentResultTileBufTypeString();
      // Get the correct input type directly from the source variable's TileType,
      // bypassing the memref_to_tile_type_ lookup which may return the wrong shape
      // when input and output share the same MemRef.
      std::string src_type;
      if (auto src_var = AsVarLike(op->args_[0])) {
        if (auto tile_type = ir::As<ir::TileType>(src_var->GetType())) {
          if (tile_type->memref_.has_value()) {
            src_type = codegen.GetTileBufTypeStringFromTileType(tile_type);
          }
        }
      }
      // PTO bytecode requires distinct tile buffers for reshape input/output.
      // When both resolve to the same buffer (shared MemRef), allocate a new result variable.
      if (src == result_target && !result_type.empty()) {
        result_target = codegen.NewTemp();
        codegen.SetCurrentResultBuf(result_target);
      }
      // Register the result variable's type so GetExprTypeAnnotation can find it
      if (!result_type.empty()) {
        codegen.RegisterTileBufType(result_target, result_type);
      }
      // New format: %result = pto.treshape %src : input_type -> output_type
      std::ostringstream oss;
      oss << result_target << " = pto.treshape " << src;
      if (!src_type.empty() && !result_type.empty()) {
        oss << " : " << src_type << " -> " << result_type;
      }
      codegen.Emit(oss.str());
      return std::string("");
    });

}  // namespace backend
}  // namespace pypto
