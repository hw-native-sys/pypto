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
 * @file pto_ops_common.cpp
 * @brief Shared PTO op registration for all PTO-based backends
 *
 * Provides RegisterPTOOps() which registers the full set of standard PTO
 * operator codegen functions to any backend instance. Backends that need to
 * override specific ops can pass those op names in the exclude_ops set and
 * register their own implementations before calling this function.
 */

#include "pypto/backend/common/pto_ops_common.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

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
using ir::AsVarLike;
using ir::CallPtr;
using ir::TensorType;
using ir::Var;

static bool RequiresRowMajorElementwiseLayout(std::string_view op_name) {
  static const std::unordered_set<std::string_view> kRowMajorElementwiseOps = {
      "tile.add", "tile.and", "tile.div", "tile.maximum", "tile.minimum", "tile.mul", "tile.or",
      "tile.rem", "tile.sel", "tile.shl", "tile.shr",     "tile.sub",     "tile.xor",
  };
  return kRowMajorElementwiseOps.count(op_name) > 0;
}

// Validate that a string is a safe MLIR identifier (alphanumeric + underscores).
// Prevents injection of arbitrary MLIR via crafted buffer/function names.
static void CheckSafeIdentifier(const std::string& value, const std::string& attr_name) {
  for (char c : value) {
    CHECK(c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
        << attr_name << " contains invalid character '" << c
        << "'; only alphanumeric and underscore are allowed";
  }
}

// ============================================================================
// Helper Functions for PTO Code Generation
// ============================================================================

static const std::vector<std::string> cmp_modes = {"eq", "ne", "lt", "le", "gt", "ge"};
static const std::vector<std::string> round_modes = {"NONE", "RINT",  "ROUND", "FLOOR",
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

// Helper function for tile.assemble → pto.tinsert
// Inserts source tile into target tile at a given row/col offset (DPS pattern).
// pto.tinsert semantics: dst[i+row, j+col] = src[i, j]
// Arguments: args[0] = target (destination base), args[1] = source, args[2] = offset MakeTuple
static std::string MakeTileAssembleCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.assemble requires 3 arguments (target, source, offset), got "
                               << op->args_.size();

  std::string target = codegen.GetExprAsCode(op->args_[0]);
  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  auto offset_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(offset_tuple) << "tile.assemble third argument must be a tuple (offset)";
  INTERNAL_CHECK(offset_tuple->elements_.size() >= 2)
      << "tile.assemble offset tuple must have at least 2 elements (row, col), got "
      << offset_tuple->elements_.size();
  std::string row_off = codegen.GetExprAsCode(offset_tuple->elements_[0]);
  std::string col_off = codegen.GetExprAsCode(offset_tuple->elements_[1]);

  // pto.tinsert writes src into dst at (row, col) in place — dst must already
  // contain target's data.  When target and dst are different buffers (i.e.
  // memory reuse did not merge them), copy target → dst first.
  if (target != dst) {
    std::string target_type = codegen.GetExprTypeAnnotation(op->args_[0]);
    std::ostringstream mov;
    mov << "pto.tmov ins(" << target;
    if (!target_type.empty()) mov << " : " << target_type;
    mov << ") outs(" << dst;
    if (!dst_type.empty()) mov << " : " << dst_type;
    mov << ")";
    codegen.Emit(mov.str());
  }

  // Emit pto.tinsert ins(src, row, col) outs(dst)
  std::ostringstream oss;
  oss << "pto.tinsert ins(" << src << ", " << row_off << ", " << col_off;
  if (!src_type.empty()) {
    oss << " : " << src_type << ", index, index";
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) oss << " : " << dst_type;
  oss << ")";
  codegen.Emit(oss.str());
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

// tile.load: emit pto.subview + pto.tload
static std::string MakeTileLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tensor = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(tensor) << "tile.load first argument must be a Var or IterArg";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "tile.load second argument must be a tuple (offsets)";

  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "tile.load third argument must be a tuple (shapes)";

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "tile.load tensor argument must have TensorType";

  const size_t ndim = shapes_tuple->elements_.size();
  INTERNAL_CHECK(ndim >= 1) << "tile.load shapes tuple must have at least one element";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK(!tile_buf.empty()) << "tile.load requires assignment target (tile_buf)";

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();
  // Build partition type with all ND dimensions to match the sizes attribute.
  std::string partition_type = "!pto.partition_tensor_view<";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) partition_type += "x";
    partition_type += std::to_string(codegen.GetConstIntValue(shapes_tuple->elements_[i]));
  }
  partition_type += "x" + dtype_str + ">";

  std::string partition_view = codegen.NewNamedTemp(tensor->name_hint_ + "_pview");
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  // Use all offsets/sizes elements to match the tensor_view rank (handles ND tensors)
  partition_line << ", offsets = [";
  for (size_t i = 0; i < offsets_tuple->elements_.size(); ++i) {
    if (i > 0) partition_line << ", ";
    partition_line << codegen.GetExprAsCode(offsets_tuple->elements_[i]);
  }
  partition_line << "]";
  partition_line << ", sizes = [";
  for (size_t i = 0; i < shapes_tuple->elements_.size(); ++i) {
    if (i > 0) partition_line << ", ";
    partition_line << codegen.GetIndexConstant(codegen.GetConstIntValue(shapes_tuple->elements_[i]));
  }
  partition_line << "]";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << partition_view << " : " << partition_type << ") outs(";
  tload_line << tile_buf << " : " << tile_buf_type << ")";
  codegen.Emit(tload_line.str());

  // Emit pto.set_validshape after tload only when a fillpad consumer exists.
  // Physical dims were used for alloc_tile (correct DMA stride); set_validshape
  // sets the actual valid region before fillpad pads the rest.
  auto result_var = codegen.GetCurrentResultVar();
  if (result_var) {
    auto tile_type = ir::As<ir::TileType>(result_var->GetType());
    if (tile_type && tile_type->tile_view_.has_value() && codegen.HasFillpadConsumer(result_var.get())) {
      const auto& tv = tile_type->tile_view_.value();
      bool has_dynamic = false;
      std::string vr, vc;

      // Extract valid_row SSA
      if (tv.valid_shape.size() >= 1) {
        if (auto var = ir::As<ir::Var>(tv.valid_shape[0])) {
          std::string mlir_name = codegen.GetVarName(var);
          vr = codegen.EmitCastToIndex(var, mlir_name);
          has_dynamic = true;
        } else if (auto c = ir::As<ir::ConstInt>(tv.valid_shape[0])) {
          vr = codegen.GetIndexConstant(c->value_);
        }
      }

      // Extract valid_col SSA
      if (tv.valid_shape.size() >= 2) {
        if (auto var = ir::As<ir::Var>(tv.valid_shape[1])) {
          std::string mlir_name = codegen.GetVarName(var);
          vc = codegen.EmitCastToIndex(var, mlir_name);
          has_dynamic = true;
        } else if (auto c = ir::As<ir::ConstInt>(tv.valid_shape[1])) {
          vc = codegen.GetIndexConstant(c->value_);
        }
      }

      if (has_dynamic && !vr.empty() && !vc.empty()) {
        codegen.Emit("pto.set_validshape " + tile_buf + ", " + vr + ", " + vc + " : " + tile_buf_type);
      }
    }
  }

  return "";
}

// tile.store: emit pto.partition_view + pto.tstore
static std::string MakeTileStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(tile) << "tile.store first argument must be a Var or IterArg";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "tile.store second argument must be a tuple (offsets)";

  auto tile_type = As<ir::TileType>(tile->GetType());
  INTERNAL_CHECK(tile_type) << "tile.store first argument must have TileType";
  INTERNAL_CHECK(tile_type->tile_view_.has_value()) << "tile.store tile must have TileView with valid_shape";
  auto& valid_shape = tile_type->tile_view_->valid_shape;
  INTERNAL_CHECK(valid_shape.size() == 2) << "tile.store tile valid_shape must be 2D";

  auto height_code = codegen.GetExprAsCode(valid_shape[0]);
  auto width_code = codegen.GetExprAsCode(valid_shape[1]);

  auto output_tensor = AsVarLike(op->args_[2]);
  INTERNAL_CHECK(output_tensor) << "tile.store output_tensor must be a Var or IterArg";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "tile.store output_tensor must have TensorType";

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::string partition_view = codegen.NewNamedTemp(output_tensor->name_hint_ + "_pview");
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  // Use all offsets elements to match tensor_view rank (handles ND tensors)
  partition_line << ", offsets = [";
  for (size_t i = 0; i < offsets_tuple->elements_.size(); ++i) {
    if (i > 0) partition_line << ", ";
    partition_line << codegen.GetExprAsCode(offsets_tuple->elements_[i]);
  }
  partition_line << "]";
  partition_line << ", sizes = [";

  // Build partition_type and sizes to match the tensor rank so they are consistent.
  std::string partition_type;
  const size_t tensor_rank = tensor_type->shape_.size();
  if (tensor_rank > 2) {
    // Use the explicit shapes tuple (args[3]) injected by FlattenTileNdTo2D.
    // Signature: (tile, offsets, output_tensor[, shapes]) — shapes at args[3]
    // when 4 args total.
    INTERNAL_CHECK(op->args_.size() > 3) << "tile.store on ND tensor requires shapes tuple (args[3])";
    auto shapes_tuple = As<ir::MakeTuple>(op->args_[3]);
    INTERNAL_CHECK(shapes_tuple) << "tile.store args[3] must be a shapes MakeTuple";
    partition_type = "!pto.partition_tensor_view<";
    for (size_t i = 0; i < shapes_tuple->elements_.size(); ++i) {
      if (i > 0) partition_line << ", ";
      if (auto c = As<ir::ConstInt>(shapes_tuple->elements_[i])) {
        partition_line << codegen.GetIndexConstant(c->value_);
        if (i > 0) partition_type += "x";
        partition_type += std::to_string(c->value_);
      } else {
        partition_line << codegen.GetExprAsCode(shapes_tuple->elements_[i]);
        if (i > 0) partition_type += "x";
        partition_type += "?";
      }
    }
    partition_type += "x" + dtype_str + ">";
  } else {
    std::string height_dim = "?", width_dim = "?";
    if (auto h = As<ir::ConstInt>(valid_shape[0])) height_dim = std::to_string(h->value_);
    if (auto w = As<ir::ConstInt>(valid_shape[1])) width_dim = std::to_string(w->value_);
    partition_type = "!pto.partition_tensor_view<" + height_dim + "x" + width_dim + "x" + dtype_str + ">";
    partition_line << height_code << ", " << width_code;
  }
  partition_line << "]";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  if (!tile_buf_type.empty()) {
    tstore_line << " : " << tile_buf_type;
  }
  tstore_line << ") outs(" << partition_view << " : " << partition_type << ")";
  codegen.Emit(tstore_line.str());

  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
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

// Compute a row-major flat offset string from a MakeTuple of indices and the shape of the container.
static std::string ComputeFlatOffsetPTO(const ir::MakeTuplePtr& indices_tuple,
                                        const std::vector<ir::ExprPtr>& shape, codegen::PTOCodegen& codegen) {
  const auto& indices = indices_tuple->elements_;
  INTERNAL_CHECK(indices.size() == shape.size())
      << "Index count (" << indices.size() << ") must match shape rank (" << shape.size() << ")";

  std::ostringstream idx_oss;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) idx_oss << " + ";
    idx_oss << codegen.GetExprAsCode(indices[i]);
    for (size_t j = i + 1; j < shape.size(); ++j) {
      idx_oss << " * " << codegen.GetExprAsCode(shape[j]);
    }
  }
  return idx_oss.str();
}

// Get or emit a flat offset SSA value for a MakeTuple of indices and shape.
static std::string GetFlatOffsetSSA(const ir::MakeTuplePtr& indices_tuple,
                                    const std::vector<ir::ExprPtr>& shape, codegen::PTOCodegen& codegen) {
  const auto& indices = indices_tuple->elements_;

  int64_t flat_offset = 0;
  bool all_constant = true;
  for (size_t i = 0; i < indices.size() && all_constant; ++i) {
    auto idx_val = As<ir::ConstInt>(indices[i]);
    if (!idx_val) {
      all_constant = false;
      break;
    }

    int64_t stride = 1;
    for (size_t j = i + 1; j < shape.size(); ++j) {
      auto dim_val = As<ir::ConstInt>(shape[j]);
      if (!dim_val) {
        all_constant = false;
        break;
      }
      stride *= dim_val->value_;
    }
    if (!all_constant) break;
    flat_offset += idx_val->value_ * stride;
  }

  if (all_constant) {
    return codegen.GetIndexConstant(flat_offset);
  }

  return ComputeFlatOffsetPTO(indices_tuple, shape, codegen);
}

// Helper function for tile.read (indices -> flat offset -> pto.tgetval)
static std::string MakeTileReadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.read requires 2 arguments, but got " << op->args_.size();

  auto tile_type = As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK(tile_type) << "tile.read first argument must be TileType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(indices_tuple) << "tile.read second argument must be MakeTuple (indices)";

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string result = codegen.GetCurrentResultTarget();
  std::string scalar_type = codegen.GetTypeString(tile_type->dtype_);

  std::string off = GetFlatOffsetSSA(indices_tuple, tile_type->shape_, codegen);

  std::ostringstream oss;
  oss << result << " = pto.tgetval ins(" << src << ", " << off;
  if (!src_type.empty()) {
    oss << " : " << src_type << ", index";
  } else {
    oss << " : , index";
  }
  oss << ") outs : " << scalar_type;
  codegen.Emit(oss.str());
  return "";
}

// Helper function for tile.write (indices -> flat offset -> pto.tsetval)
static std::string MakeTileWriteCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.write requires 3 arguments, but got " << op->args_.size();

  auto tile_type = As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK(tile_type) << "tile.write first argument must be TileType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(indices_tuple) << "tile.write second argument must be MakeTuple (indices)";

  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tile_type_str = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string value = codegen.GetExprAsCode(op->args_[2]);
  std::string value_type = codegen.GetExprTypeAnnotation(op->args_[2]);

  std::string off = GetFlatOffsetSSA(indices_tuple, tile_type->shape_, codegen);

  std::ostringstream oss;
  oss << "pto.tsetval ins(" << off << ", " << value;
  oss << " : index";
  if (!value_type.empty()) oss << ", " << value_type;
  oss << ") outs(" << tile;
  if (!tile_type_str.empty()) oss << " : " << tile_type_str;
  oss << ")";
  codegen.Emit(oss.str());

  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
    codegen.RegisterVarToMlir(result_var, tile);
  }
  return "";
}

static std::string MakeTensorReadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.read requires 2 arguments, but got " << op->args_.size();

  auto tensor_type_ptr = As<ir::TensorType>(op->args_[0]->GetType());
  INTERNAL_CHECK(tensor_type_ptr) << "tensor.read first argument must be TensorType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(indices_tuple) << "tensor.read second argument must be MakeTuple (indices)";

  auto scalar_type_ptr = As<ir::ScalarType>(op->GetType());
  INTERNAL_CHECK(scalar_type_ptr) << "tensor.read result must be ScalarType";
  std::string scalar_type = codegen.GetTypeString(scalar_type_ptr->dtype_);

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string result = codegen.GetCurrentResultTarget();

  if (src_type.empty()) {
    src_type = "!pto.ptr<" + codegen.GetTypeString(tensor_type_ptr->dtype_) + ">";
  }

  std::string off = GetFlatOffsetSSA(indices_tuple, tensor_type_ptr->shape_, codegen);

  std::ostringstream oss;
  oss << result << " = pto.load_scalar " << src << "[" << off << "]";
  if (!src_type.empty()) {
    oss << " : " << src_type;
  }
  oss << " -> " << scalar_type;
  codegen.Emit(oss.str());
  return "";
}

static std::string MakeTensorWriteCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tensor.write requires 3 arguments, but got " << op->args_.size();

  auto tensor_type_ptr = As<ir::TensorType>(op->args_[0]->GetType());
  INTERNAL_CHECK(tensor_type_ptr) << "tensor.write first argument must be TensorType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(indices_tuple) << "tensor.write second argument must be MakeTuple (indices)";

  std::string tensor = codegen.GetExprAsCode(op->args_[0]);
  std::string tensor_type_str = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string value = codegen.GetExprAsCode(op->args_[2]);
  std::string value_type = codegen.GetExprTypeAnnotation(op->args_[2]);

  if (tensor_type_str.empty()) {
    tensor_type_str = "!pto.ptr<" + codegen.GetTypeString(tensor_type_ptr->dtype_) + ">";
  }

  std::string off = GetFlatOffsetSSA(indices_tuple, tensor_type_ptr->shape_, codegen);

  std::ostringstream oss;
  oss << "pto.store_scalar " << value << ", " << tensor << "[" << off << "]";
  if (!tensor_type_str.empty() || !value_type.empty()) {
    oss << " : ";
    if (!tensor_type_str.empty()) oss << tensor_type_str;
    if (!tensor_type_str.empty() && !value_type.empty()) oss << ", ";
    if (!value_type.empty()) oss << value_type;
  }
  codegen.Emit(oss.str());

  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
    codegen.RegisterTensorView(result_var, tensor);
    codegen.RegisterVarToMlir(result_var, tensor);
  }
  return "";
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
  if (auto dyn_shape = ir::As<ir::Var>(shape)) {
    shape_name = codegen.GetVarName(dyn_shape);
  } else if (auto static_shape = ir::As<ir::ConstInt>(shape)) {
    shape_name = codegen.GetIndexConstant(static_shape->value_);
  } else {
    INTERNAL_CHECK(false) << "Internal error: tensor.dim shape is neither Var nor ConstInt";
  }
  auto target_var = codegen.GetCurrentResultVar();
  if (target_var != nullptr && !shape_name.empty()) {
    codegen.RegisterVarToMlir(target_var, shape_name);
  }

  return "";
}

// ============================================================================
// Cross-Core Communication Operations (TPUSH/TPOP)
// ============================================================================

// tile.tpush_to_aiv: Push tile from Cube to Vector
static std::string MakeTpushToAivCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 1) << "tpush_to_aiv requires 1 argument (tile), got " << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(tile) << "tpush_to_aiv first argument must be a Var or IterArg";

  const int split = op->GetKwarg<int>("split", -1);
  CHECK(split >= 0 && split <= 2)
      << "tpush_to_aiv requires 'split' attribute (0=none, 1=up-down, 2=left-right), got " << split;

  std::string tile_buf = codegen.GetVarName(tile);
  std::string tile_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::ostringstream oss;
  oss << "pto.tpush_to_aiv(" << tile_buf;
  if (!tile_type.empty()) {
    oss << " : " << tile_type;
  }
  oss << ") {split = " << split << "}";
  codegen.Emit(oss.str());

  return "";
}

// tile.tpush_to_aic: Push tile from Vector to Cube
static std::string MakeTpushToAicCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 1) << "tpush_to_aic requires 1 argument (tile), got " << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(tile) << "tpush_to_aic first argument must be a Var or IterArg";

  const int split = op->GetKwarg<int>("split", -1);
  CHECK(split >= 0 && split <= 2)
      << "tpush_to_aic requires 'split' attribute (0=none, 1=up-down, 2=left-right), got " << split;

  std::string tile_buf = codegen.GetVarName(tile);
  std::string tile_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::ostringstream oss;
  oss << "pto.tpush_to_aic(" << tile_buf;
  if (!tile_type.empty()) {
    oss << " : " << tile_type;
  }
  oss << ") {split = " << split << "}";
  codegen.Emit(oss.str());

  return "";
}

// tile.tpop_from_aic: Pop tile from Cube into Vector
static std::string MakeTpopFromAicCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 0) << "tpop_from_aic takes no arguments, got " << op->args_.size();

  const int split = op->GetKwarg<int>("split", 0);
  CHECK(split >= 0 && split <= 2)
      << "tpop_from_aic requires 'split' attribute (0=none, 1=up-down, 2=left-right), got " << split;

  std::string result_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK(!result_buf.empty()) << "tpop_from_aic requires assignment target (tile_buf)";
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << result_buf << " = pto.tpop_from_aic {split = " << split << "}";
  if (!result_type.empty()) {
    oss << " -> " << result_type;
  }
  codegen.Emit(oss.str());

  return "";
}

// tile.tpop_from_aiv: Pop tile from Vector into Cube
static std::string MakeTpopFromAivCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 0) << "tpop_from_aiv takes no arguments, got " << op->args_.size();

  const int split = op->GetKwarg<int>("split", 0);
  CHECK(split >= 0 && split <= 2)
      << "tpop_from_aiv requires 'split' attribute (0=none, 1=up-down, 2=left-right), got " << split;

  std::string result_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK(!result_buf.empty()) << "tpop_from_aiv requires assignment target (tile_buf)";
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << result_buf << " = pto.tpop_from_aiv {split = " << split << "}";
  if (!result_type.empty()) {
    oss << " -> " << result_type;
  }
  codegen.Emit(oss.str());

  return "";
}

/// tfree codegen for system.tfree_to_aic: emits pto.tfree_from_aic {split = N}
static std::string MakeTfreeToAicCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 1) << "tfree_to_aic requires 1 argument (tile from tpop), got "
                               << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(tile) << "tfree_to_aic first argument must be a Var or IterArg";
  int split = codegen.GetValidatedTpopSplit(tile.get(), "tile.tpop_from_aic", "system.tfree_to_aic");

  std::ostringstream oss;
  oss << "pto.tfree_from_aic {split = " << split << "}";
  codegen.Emit(oss.str());

  return "";
}

// tfree codegen for system.tfree_to_aiv: emits pto.tfree_from_aiv {split = N}
static std::string MakeTfreeToAivCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 1) << "tfree_to_aiv requires 1 argument (tile from tpop), got "
                               << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(tile) << "tfree_to_aiv first argument must be a Var or IterArg";

  int split = codegen.GetValidatedTpopSplit(tile.get(), "tile.tpop_from_aiv", "system.tfree_to_aiv");

  std::ostringstream oss;
  oss << "pto.tfree_from_aiv {split = " << split << "}";
  codegen.Emit(oss.str());

  return "";
}

// system.aic_initialize_pipe: Initialize cross-core pipe on Cube side
static std::string MakeAicInitializePipeCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  const int dir_mask = op->GetKwarg<int>("dir_mask", -1);
  const int slot_size = op->GetKwarg<int>("slot_size", -1);
  CHECK(dir_mask >= 0) << "aic_initialize_pipe requires 'dir_mask' attribute";
  CHECK(slot_size > 0) << "aic_initialize_pipe requires 'slot_size' attribute";

  // AIC (Cube): consumer for V2C (reserve), producer for C2V (import)
  std::string v2c_ssa = codegen.GetReserveBufferSSA();
  std::string c2v_ssa = codegen.GetImportBufferSSA();
  if (v2c_ssa.empty()) v2c_ssa = codegen.GetOrEmitI32Constant(0);
  if (c2v_ssa.empty()) c2v_ssa = codegen.GetOrEmitI32Constant(0);

  std::ostringstream oss;
  oss << "pto.aic_initialize_pipe {dir_mask = " << dir_mask << ", slot_size = " << slot_size << "}"
      << " (c2v_consumer_buf = " << c2v_ssa << " : i32"
      << ", v2c_consumer_buf = " << v2c_ssa << " : i32)";
  codegen.Emit(oss.str());

  return "";
}

// system.aiv_initialize_pipe: Initialize cross-core pipe on Vector side
static std::string MakeAivInitializePipeCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  const int dir_mask = op->GetKwarg<int>("dir_mask", -1);
  const int slot_size = op->GetKwarg<int>("slot_size", -1);
  CHECK(dir_mask >= 0) << "aiv_initialize_pipe requires 'dir_mask' attribute";
  CHECK(slot_size > 0) << "aiv_initialize_pipe requires 'slot_size' attribute";

  // AIV (Vector): consumer for C2V (reserve), producer for V2C (import)
  std::string c2v_ssa = codegen.GetReserveBufferSSA();
  std::string v2c_ssa = codegen.GetImportBufferSSA();
  if (c2v_ssa.empty()) c2v_ssa = codegen.GetOrEmitI32Constant(0);
  if (v2c_ssa.empty()) v2c_ssa = codegen.GetOrEmitI32Constant(0);

  std::ostringstream oss;
  oss << "pto.aiv_initialize_pipe {dir_mask = " << dir_mask << ", slot_size = " << slot_size << "}"
      << " (c2v_consumer_buf = " << c2v_ssa << " : i32"
      << ", v2c_consumer_buf = " << v2c_ssa << " : i32)";
  codegen.Emit(oss.str());

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
    {"tile.col_expand_mul",  "pto.tcolexpandmul",    2},
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
    // tile.matmul_acc and tile.gemv_acc have custom codegen (in-place accumulation)
    {"tile.matmul_bias",     "pto.tmatmul.bias",     3},
    {"tile.gemv",            "pto.tgemv",            2},
    // tile.gemv_acc has custom codegen (in-place accumulation)
    {"tile.gemv_bias",       "pto.tgemv.bias",       3},
    // Data movement/layout operations
    {"tile.concat",          "pto.tconcat",          2},
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

// ============================================================================
// RegisterPTOOps: Register all standard PTO ops to the given backend
// ============================================================================

void RegisterPTOOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  // Register simple N-ary ops
  for (const auto& entry : kSimpleOps) {
    if (exclude_ops.count(entry.op_name) > 0) continue;
    std::string pto_op = entry.pto_op_name;
    size_t arity = entry.arity;
    auto reg_entry = backend.RegisterOp(entry.op_name);
    reg_entry.f_codegen([pto_op, arity](const CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeNaryCodegenPTO(pto_op, arity, op, codegen);
    });
    if (RequiresRowMajorElementwiseLayout(entry.op_name)) {
      for (size_t i = 0; i < arity; ++i) {
        reg_entry.set_input_layout(i, ir::TileLayout::row_major);
      }
      reg_entry.set_output_layout(ir::TileLayout::row_major);
    }
  }

  // Register ops with custom codegen logic
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  reg("tile.read", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileReadCodegenPTO(op, codegen);
  });
  reg("tile.write", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileWriteCodegenPTO(op, codegen);
  });
  reg("tensor.read", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTensorReadCodegenPTO(op, codegen);
  });
  reg("tensor.write", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTensorWriteCodegenPTO(op, codegen);
  });
  reg("tile.load", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileLoadCodegenPTO(op, codegen);
  });
  reg("tile.store", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileStoreCodegenPTO(op, codegen);
  });
  reg("tile.alloc", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileAllocCodegenPTO(op, codegen);
  });
  reg("tile.create", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    (void)op;
    (void)codegen_base;
    return std::string("");  // No MLIR emission - tile allocation handled by pto.alloc_tile
  });
  reg("tile.store_fp", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeStoreFPCodegenPTO("pto.tstore.fp", op, codegen);
  });
  reg("tile.cmp", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileCmpCodegenPTO("pto.tcmp", op, codegen);
  });
  reg("tile.cast", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileCvtCodegenPTO("pto.tcvt", op, codegen);
  });
  reg("tile.full", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeFullCodegenPTO("pto.texpands", op, codegen);
  });
  reg("tile.cmps", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeCmpsCodegenPTO("pto.tcmps", op, codegen);
  });
  reg("tile.assign", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeAssignCodegenPTO("pto.tassign", op, codegen);
  });
  reg("tile.ci", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeCiCodegenPTO("pto.tci", op, codegen);
  });
  // TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
  reg("tile.sort32", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeSort32CodegenPTO("pto.tsort32", op, codegen);
  });
  // TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
  reg("tile.mrgsort", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeMrgSortCodegenPTO("pto.tmrgsort", op, codegen);
  });
  reg("tile.print", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakePrintCodegenPTO("pto.tprint", op, codegen);
  });

  // In-place accumulation ops (matmul_acc, gemv_acc): ptoas expects the
  // accumulator in ins() to be the same SSA value as outs().  InitMemRef
  // guarantees that the output shares the MemRef of the accumulator input
  // (via set_output_reuses_input), so we use the result buffer (dst) as the
  // accumulator operand instead of the IR-level input arg.
  auto make_acc_codegen = [](const std::string& pto_op) {
    return [pto_op](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
      auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
      CHECK(op->args_.size() == 3) << pto_op << " requires 3 arguments: acc, lhs, rhs";

      std::string dst = codegen.GetCurrentResultTarget();
      std::string lhs = codegen.GetExprAsCode(op->args_[1]);
      std::string rhs = codegen.GetExprAsCode(op->args_[2]);
      std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
      std::string lhs_type = codegen.GetExprTypeAnnotation(op->args_[1]);
      std::string rhs_type = codegen.GetExprTypeAnnotation(op->args_[2]);

      std::ostringstream acc_inst;
      acc_inst << pto_op << " ins(" << dst << ", " << lhs << ", " << rhs;
      std::vector<std::string> ins_type_parts;
      for (const auto& t : {dst_type, lhs_type, rhs_type}) {
        if (!t.empty()) ins_type_parts.push_back(t);
      }
      if (!ins_type_parts.empty()) {
        acc_inst << " : ";
        for (size_t i = 0; i < ins_type_parts.size(); ++i) {
          if (i > 0) acc_inst << ", ";
          acc_inst << ins_type_parts[i];
        }
      }
      acc_inst << ") outs(" << dst;
      if (!dst_type.empty()) acc_inst << " : " << dst_type;
      acc_inst << ")";
      codegen.Emit(acc_inst.str());
      return "";
    };
  };

  reg("tile.matmul_acc", make_acc_codegen("pto.tmatmul.acc"));
  reg("tile.gemv_acc", make_acc_codegen("pto.tgemv.acc"));

  reg("tensor.dim", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTensorDimCodegenPTO(op, codegen);
  });
  reg("tile.tpush_to_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpushToAivCodegenPTO(op, codegen);
  });
  reg("tile.tpop_from_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpopFromAivCodegenPTO(op, codegen);
  });
  reg("tile.tpush_to_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpushToAicCodegenPTO(op, codegen);
  });
  reg("tile.tpop_from_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTpopFromAicCodegenPTO(op, codegen);
  });
  reg("system.tfree_to_aic", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTfreeToAicCodegenPTO(op, codegen);
  });
  reg("system.tfree_to_aiv", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTfreeToAivCodegenPTO(op, codegen);
  });
  reg("system.aic_initialize_pipe", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeAicInitializePipeCodegenPTO(op, codegen);
  });
  reg("system.aiv_initialize_pipe", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeAivInitializePipeCodegenPTO(op, codegen);
  });
  reg("system.reserve_buffer", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 0) << "reserve_buffer takes no arguments, got " << op->args_.size();

    const auto name = op->GetKwarg<std::string>("name");
    const int size = op->GetKwarg<int>("size", -1);
    const int base = op->GetKwarg<int>("base", -1);
    CHECK(!name.empty()) << "reserve_buffer requires 'name' attribute";
    CHECK(size > 0) << "reserve_buffer requires positive 'size' attribute, got " << size;
    CHECK(base >= 0)
        << "reserve_buffer requires AllocateMemoryAddr to resolve 'base' before PTO emission, got " << base;
    CheckSafeIdentifier(name, "reserve_buffer 'name'");

    std::string ssa_name = codegen.GetCurrentResultTarget();
    if (ssa_name.empty()) {
      // EvalStmt context — derive SSA name from buffer name hint
      ssa_name = codegen.NewNamedTemp(name);
    }

    std::string location;
    if (codegen.IsAICFunction()) {
      location = "mat";
    } else if (codegen.IsAIVFunction()) {
      location = "vec";
    } else {
      location = "undefined";
    }

    std::ostringstream oss;
    oss << ssa_name << " = pto.reserve_buffer {name = \"" << name << "\", size = " << size
        << ", location = #pto.address_space<" << location << ">, auto = false, base = " << base;
    oss << "} -> i32";
    codegen.Emit(oss.str());
    codegen.RecordReserveBufferSSA(ssa_name);

    return std::string("");
  });
  reg("system.import_peer_buffer", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 0) << "import_peer_buffer takes no arguments, got " << op->args_.size();

    const auto name = op->GetKwarg<std::string>("name");
    const auto peer_func = op->GetKwarg<std::string>("peer_func");
    CHECK(!name.empty()) << "import_peer_buffer requires 'name' attribute";
    CHECK(!peer_func.empty()) << "import_peer_buffer requires 'peer_func' attribute";
    CheckSafeIdentifier(name, "import_peer_buffer 'name'");
    CheckSafeIdentifier(peer_func, "import_peer_buffer 'peer_func'");

    std::string ssa_name = codegen.GetCurrentResultTarget();
    if (ssa_name.empty()) {
      ssa_name = codegen.NewNamedTemp(name + "_import");
    }

    std::ostringstream oss;
    oss << ssa_name << " = pto.import_reserved_buffer {name = \"" << name << "\", peer_func = @" << peer_func
        << "} -> i32";
    codegen.Emit(oss.str());
    codegen.RecordImportBufferSSA(ssa_name);

    return std::string("");
  });
  reg("tile.slice", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 3 || op->args_.size() == 4)
        << "Operation:[tile.slice] requires 3 or 4 arguments (tile, shape, offset[, valid_shape]), but got "
        << op->args_.size();

    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);

    auto offset_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    INTERNAL_CHECK(offset_tuple) << "tile.slice third argument must be a tuple (offset)";
    INTERNAL_CHECK(offset_tuple->elements_.size() >= 2)
        << "tile.slice offset tuple must have at least 2 elements (row, col), got "
        << offset_tuple->elements_.size();
    std::string row_off = codegen.GetExprAsCode(offset_tuple->elements_[0]);
    std::string col_off = codegen.GetExprAsCode(offset_tuple->elements_[1]);

    std::string result_target = codegen.GetCurrentResultTarget();
    std::string result_type = codegen.GetCurrentResultTileBufTypeStringFromTileType();

    // With per-var alloc model, prefer the pre-declared alloc SSA if its type
    // matches the slice result type
    auto existing_type = codegen.GetSSATileBufType(result_target);
    if (!result_type.empty() && existing_type != result_type) {
      result_target = codegen.AllocNewTileBuf(result_type, "slice_buf");
      codegen.SetCurrentResultBuf(result_target);
    } else if (!result_type.empty()) {
      codegen.RegisterTileBufType(result_target, result_type);
    }

    std::ostringstream oss;
    oss << "pto.textract ins(" << src << ", " << row_off << ", " << col_off;
    if (!src_type.empty()) {
      oss << " : " << src_type << ", index, index";
    }
    oss << ") outs(" << result_target;
    if (!result_type.empty()) {
      oss << " : " << result_type;
    }
    oss << ")";
    codegen.Emit(oss.str());
    return std::string("");
  });
  reg("tile.assemble", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileAssembleCodegenPTO(op, codegen);
  });
  reg("tile.reshape", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "Operation:[tile.reshape] requires 2 arguments (tile, shape), but got "
                                 << op->args_.size();
    std::string result_target = codegen.GetCurrentResultTarget();
    std::string result_type = codegen.GetCurrentResultTileBufTypeStringFromTileType();

    // With per-var alloc model, the result variable already has a pre-declared
    // alloc_tile with the correct reshaped type and shared addr. If the types
    // match, the reshape is a no-op at the PTO level.
    auto existing_type = codegen.GetSSATileBufType(result_target);
    if (!existing_type.empty() && existing_type == result_type) {
      return std::string("");
    }

    // Fallback: emit pto.treshape for cases without pre-declared alloc
    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string src_type;
    if (auto src_var = AsVarLike(op->args_[0])) {
      if (auto tile_type = ir::As<ir::TileType>(src_var->GetType())) {
        if (tile_type->memref_.has_value()) {
          src_type = codegen.GetTileBufTypeStringFromTileType(tile_type);
        }
      }
    }

    if (!result_type.empty()) {
      result_target = codegen.NewNamedTemp("reshape_buf");
      codegen.SetCurrentResultBuf(result_target);
      codegen.RegisterTileBufType(result_target, result_type);
    }
    std::ostringstream oss;
    oss << result_target << " = pto.treshape " << src;
    if (!src_type.empty() && !result_type.empty()) {
      oss << " : " << src_type << " -> " << result_type;
    }
    codegen.Emit(oss.str());
    return std::string("");
  });
}

}  // namespace backend
}  // namespace pypto
