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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/distributed/comm_layout.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

using ir::As;
using ir::AsTensorTypeLike;
using ir::AsVarLike;
using ir::CallPtr;
using ir::ExprPtr;
using ir::ScalarType;
using ir::TensorType;
using ir::Var;

static bool RequiresRowMajorLayout(std::string_view op_name) {
  static const std::unordered_set<std::string_view> kRowMajorOps = {
      // Tile x Tile binary ops
      "tile.add",
      "tile.and",
      "tile.div",
      "tile.maximum",
      "tile.minimum",
      "tile.mul",
      "tile.or",
      "tile.rem",
      "tile.shl",
      "tile.shr",
      "tile.sub",
      "tile.xor",
      // Unary ops
      "tile.abs",
      "tile.exp",
      "tile.log",
      "tile.sqrt",
      "tile.recip",
      "tile.not",
      "tile.relu",
      // Tile x Scalar ops
      "tile.adds",
      "tile.muls",
      "tile.divs",
      "tile.maximums",
      "tile.lrelu",
      // Ternary scalar ops (Tile x Scalar x Tile)
      "tile.addsc",
      "tile.subsc",
  };
  return kRowMajorOps.count(op_name) > 0;
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
// Mask pattern names for pto.tgather mask form (index 0 unused, patterns 1-7)
static const std::vector<std::string> mask_patterns = {"",      "P0101", "P1010", "P0001",
                                                       "P0010", "P0100", "P1000", "P1111"};

// Build a partition_tensor_view type string from dimension strings and element dtype.
static std::string MakePartitionTensorViewType(const std::vector<std::string>& dims,
                                               const std::string& dtype_str) {
  std::ostringstream oss;
  oss << "!pto.partition_tensor_view<";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) oss << "x";
    oss << dims[i];
  }
  oss << "x" << dtype_str << ">";
  return oss.str();
}

// Convert expressions to MLIR operand strings.
static std::vector<std::string> GetExprCodes(const std::vector<ir::ExprPtr>& exprs,
                                             codegen::PTOCodegen& codegen) {
  std::vector<std::string> codes;
  codes.reserve(exprs.size());
  for (const auto& expr : exprs) {
    codes.push_back(codegen.GetExprAsCode(expr));
  }
  return codes;
}

// Convert statically-known dimensions to plain integer strings for MLIR types.
static std::vector<std::string> GetStaticDimStrings(const std::vector<ir::ExprPtr>& exprs,
                                                    codegen::PTOCodegen& codegen) {
  std::vector<std::string> dims;
  dims.reserve(exprs.size());
  for (const auto& expr : exprs) {
    dims.push_back(std::to_string(codegen.GetConstIntValue(expr)));
  }
  return dims;
}

// Convert statically-known dimensions to index-typed MLIR constants.
static std::vector<std::string> GetStaticIndexCodes(const std::vector<ir::ExprPtr>& exprs,
                                                    codegen::PTOCodegen& codegen) {
  std::vector<std::string> codes;
  codes.reserve(exprs.size());
  for (const auto& expr : exprs) {
    codes.push_back(codegen.GetOrEmitConstant(codegen.GetConstIntValue(expr), DataType::INDEX));
  }
  return codes;
}

// Get dimension strings handling both static (ConstInt) and dynamic dimensions.
// Static dims become numeric strings; dynamic dims become "?".
static std::vector<std::string> GetDimStrings(const std::vector<ir::ExprPtr>& exprs) {
  std::vector<std::string> dims;
  dims.reserve(exprs.size());
  for (const auto& expr : exprs) {
    if (auto c = As<ir::ConstInt>(expr)) {
      dims.push_back(std::to_string(c->value_));
    } else {
      dims.emplace_back("?");
    }
  }
  return dims;
}

// Convert expressions to MLIR size codes, using constants when available and
// GetExprAsCode for dynamic values.
static std::vector<std::string> GetSizeCodes(const std::vector<ir::ExprPtr>& exprs,
                                             codegen::PTOCodegen& codegen) {
  std::vector<std::string> codes;
  codes.reserve(exprs.size());
  for (const auto& expr : exprs) {
    if (auto c = As<ir::ConstInt>(expr)) {
      codes.push_back(codegen.GetOrEmitConstant(c->value_, DataType::INDEX));
    } else {
      codes.push_back(codegen.GetExprAsCode(expr));
    }
  }
  return codes;
}

static bool ExprsEquivalentForSubview(const ir::ExprPtr& lhs, const ir::ExprPtr& rhs) {
  if (lhs.get() == rhs.get()) return true;
  auto lhs_const = As<ir::ConstInt>(lhs);
  auto rhs_const = As<ir::ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

static codegen::TileTypeComponents InferSubviewTileTypeComponents(const ir::TileType& source_tile_type,
                                                                  const ir::MakeTuple& shape_tuple,
                                                                  const ir::MakeTuple& offset_tuple,
                                                                  const std::string& dtype_str) {
  codegen::TileTypeComponents c;
  c.dtype_str = dtype_str;

  auto rows_const = As<ir::ConstInt>(shape_tuple.elements_[0]);
  auto cols_const = As<ir::ConstInt>(shape_tuple.elements_[1]);
  INTERNAL_CHECK(rows_const && cols_const) << "Subview shape must be static for PTO type inference";
  c.rows = rows_const->value_;
  c.cols = cols_const->value_;

  const auto tv = ir::tile_view_semantics::GetEffectiveTileView(source_tile_type);
  c.blayout = tv.blayout;
  c.slayout = tv.slayout;
  c.fractal = tv.fractal;
  c.pad = tv.pad;

  c.v_row = c.rows;
  c.v_col = c.cols;
  c.v_row_dynamic = true;
  c.v_col_dynamic = true;

  // PTOAS infers the subview result's static valid from the slice's `sizes`
  // whenever the parent's static type string carries `v_row=?, v_col=?`.
  // Non-subview tile types always render that way (see ExtractTileTypeInfo in
  // pto_type_utils.cpp) — even when the IR's `tile_view_.valid_shape` is set —
  // so reading the parent's IR valid here can diverge from PTOAS. The case
  // that surfaces in practice is SplitVectorKernel's lane1 [0, 0] sentinel
  // (set by WithZeroValidShape on cloned lane1 ops): the parent prints as `?`
  // but its IR valid is `[0, 0]`, producing a `v_row=0, v_col=0` result that
  // PTOAS rejects (issue #1507).
  //
  // Special-case only the all-zero sentinel; for every other narrow valid
  // (e.g., parent subviews whose deducer propagated a real `v_row/v_col`)
  // keep the existing inference so nested subviews still type correctly.
  std::vector<ir::ExprPtr> source_valid = source_tile_type.shape_;
  if (source_tile_type.tile_view_.has_value() && source_tile_type.tile_view_->valid_shape.size() >= 2) {
    const auto& parent_valid = source_tile_type.tile_view_->valid_shape;
    const bool is_zero_sentinel =
        std::all_of(parent_valid.begin(), parent_valid.end(), [](const ir::ExprPtr& e) {
          auto c = As<ir::ConstInt>(e);
          return c && c->value_ == 0;
        });
    if (!is_zero_sentinel) {
      source_valid = parent_valid;
    }
  }

  auto infer_dim = [&](size_t dim_idx, int64_t size, int64_t* out_value, bool* out_dynamic) {
    auto offset_const = As<ir::ConstInt>(offset_tuple.elements_[dim_idx]);
    auto valid_const = dim_idx < source_valid.size() ? As<ir::ConstInt>(source_valid[dim_idx]) : nullptr;
    if (offset_const && valid_const) {
      int64_t remain = valid_const->value_ - offset_const->value_;
      if (remain < 0) remain = 0;
      *out_value = std::min<int64_t>(size, remain);
      *out_dynamic = false;
      return;
    }
    // offset=0 and slice shape matches the full source valid extent
    if (offset_const && offset_const->value_ == 0 && dim_idx < source_valid.size() &&
        ExprsEquivalentForSubview(shape_tuple.elements_[dim_idx], source_valid[dim_idx])) {
      *out_value = size;
      *out_dynamic = false;
      return;
    }
    // Any valid offset leaves exactly `size` valid elements in this dimension,
    // so v_row/v_col is statically known regardless of the offset value.
    if (valid_const && valid_const->value_ >= size) {
      *out_value = size;
      *out_dynamic = false;
      return;
    }
  };

  infer_dim(0, c.rows, &c.v_row, &c.v_row_dynamic);
  infer_dim(1, c.cols, &c.v_col, &c.v_col_dynamic);

  // Match ExtractTileTypeInfo: PTOAS rejects mixed static/dynamic valid dims
  // on 2D tiles, so promote both to dynamic when either is dynamic.
  if (c.v_row_dynamic || c.v_col_dynamic) {
    c.v_row_dynamic = true;
    c.v_col_dynamic = true;
  }
  return c;
}

// Emit a pto.partition_view op and return the generated SSA name.
static std::string EmitPartitionViewPTO(const std::string& name_hint, const std::string& tensor_view,
                                        const std::string& tensor_view_type,
                                        const std::string& partition_type,
                                        const std::vector<std::string>& offset_codes,
                                        const std::vector<std::string>& size_codes,
                                        codegen::PTOCodegen& codegen) {
  std::string partition_view = codegen.NewNamedTemp(name_hint + "_pview");
  std::ostringstream oss;
  oss << partition_view << " = pto.partition_view " << tensor_view;
  oss << ", offsets = [";
  for (size_t i = 0; i < offset_codes.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << offset_codes[i];
  }
  oss << "]";
  oss << ", sizes = [";
  for (size_t i = 0; i < size_codes.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << size_codes[i];
  }
  oss << "]";
  oss << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(oss.str());
  return partition_view;
}

// Emit SSA ops that compute a row-major flat offset from lowered SSA index
// values and the container shape. This mirrors GetFlatOffsetSSA(...) but uses
// explicit arith.muli/arith.addi so the result is valid MLIR SSA, not a raw
// textual expression.
static std::string EmitFlatOffsetSSAFromValues(const std::vector<std::string>& indices,
                                               const std::vector<ir::ExprPtr>& shape,
                                               codegen::PTOCodegen& codegen, const std::string& name_hint) {
  INTERNAL_CHECK(indices.size() == shape.size())
      << "Index count (" << indices.size() << ") must match shape rank (" << shape.size() << ")";

  INTERNAL_CHECK(!indices.empty()) << "EmitFlatOffsetSSAFromValues requires at least one index";
  if (indices.size() == 1) {
    return indices[0];
  }

  std::string acc = indices[0];
  for (size_t i = 1; i < indices.size(); ++i) {
    std::string dim_ssa;
    if (auto c = As<ir::ConstInt>(shape[i])) {
      dim_ssa = codegen.GetOrEmitConstant(c->value_, DataType::INDEX);
    } else {
      dim_ssa = codegen.EmitCastToIndex(shape[i], codegen.GetExprAsCode(shape[i]));
    }

    std::string mul = codegen.NewNamedTemp(name_hint + "_mul");
    codegen.Emit(mul + " = arith.muli " + acc + ", " + dim_ssa + " : index");

    std::string add = codegen.NewNamedTemp(name_hint);
    codegen.Emit(add + " = arith.addi " + mul + ", " + indices[i] + " : index");
    acc = add;
  }
  return acc;
}

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
    oss << " " << config_attr;
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

static std::string MaterializeSubviewOperandIfNeeded(const ir::ExprPtr& expr, codegen::PTOCodegen& codegen,
                                                     const std::string& name_hint) {
  auto var = ir::As<ir::Var>(expr);
  if (!var) return codegen.GetExprAsCode(expr);

  std::string operand = codegen.GetExprAsCode(expr);
  auto* mat = codegen.GetSubviewMaterialization(operand);
  if (!mat) return operand;
  if (mat->emitted) return mat->materialize_target_ssa;

  auto result_type = mat->materialize_target_type;
  std::ostringstream extract;
  extract << "pto.textract ins(" << mat->source_ssa << ", " << mat->row_off_ssa << ", " << mat->col_off_ssa;
  if (!mat->source_type.empty()) {
    extract << " : " << mat->source_type << ", index, index";
  }
  extract << ") outs(" << mat->materialize_target_ssa;
  if (!result_type.empty()) extract << " : " << result_type;
  extract << ")";
  codegen.Emit(extract.str());
  mat->emitted = true;
  return mat->materialize_target_ssa;
}

// Helper function for N-ary operations (unary, binary, ternary, etc.)
static std::string MakeNaryCodegenPTO(const std::string& pto_op_name, size_t arity, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == arity) << "Operation:[" << pto_op_name << "] requires " << arity << " argument"
                                   << (arity != 1 ? "s" : "") << ", but got " << op->args_.size();
  // pto.tcolexpandmul requires materialized tile data — its hardware lowering
  // reads physical tile rows/cols from the operand type, which is incorrect for
  // a pto.subview alias.  Other tile ops (tmov, tfillpad, tadd, ...) accept
  // subview SSAs natively, so only tcolexpandmul needs eager materialization.
  if (pto_op_name == "pto.tcolexpandmul") {
    auto lhs_operand = MaterializeSubviewOperandIfNeeded(op->args_[0], codegen, "colexpandmul_mat");
    auto rhs_operand = MaterializeSubviewOperandIfNeeded(op->args_[1], codegen, "colexpandmul_vec");
    std::string lhs_orig = codegen.GetExprAsCode(op->args_[0]);
    std::string rhs_orig = codegen.GetExprAsCode(op->args_[1]);
    if (lhs_operand != lhs_orig || rhs_operand != rhs_orig) {
      // Resolve type annotations: use the materialized target type when
      // the operand was a subview, otherwise use the original annotation.
      std::string lhs_type = codegen.GetExprTypeAnnotation(op->args_[0]);
      auto* lhs_mat = codegen.GetSubviewMaterialization(lhs_orig);
      if (lhs_mat) lhs_type = lhs_mat->materialize_target_type;

      std::string rhs_type = codegen.GetExprTypeAnnotation(op->args_[1]);
      auto* rhs_mat = codegen.GetSubviewMaterialization(rhs_orig);
      if (rhs_mat) rhs_type = rhs_mat->materialize_target_type;

      std::ostringstream oss;
      oss << "pto.tcolexpandmul ins(" << lhs_operand << ", " << rhs_operand;
      if (!lhs_type.empty() && !rhs_type.empty()) {
        oss << " : " << lhs_type << ", " << rhs_type;
      }
      std::string result_target = codegen.GetCurrentResultTarget();
      std::string result_type = codegen.GetCurrentResultTileBufTypeString();
      oss << ") outs(" << result_target;
      if (!result_type.empty()) oss << " : " << result_type;
      oss << ")";
      codegen.Emit(oss.str());
      return "";
    }
  }
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

static std::string MakeTileSelCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "Operation:[pto.tsel] requires 4 arguments, but got " << op->args_.size();
  codegen.Emit("pto.tsel " + GenerateInsOutsClause(op, codegen));
  return "";
}

// pto.ttrans ins(%src, %tmp : tile_type, tile_type). IR form: tile.transpose(src, axis0, axis1, tmp).
// tmp is pre-allocated by an IR-level tile.create so the memory allocator gives it a real UB
// address before codegen (required at --pto-level=level3).
static std::string MakeTileTransposeCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "tile.transpose requires 4 arguments (src, axis0, axis1, tmp), got "
                               << op->args_.size();

  std::string src_ssa = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string tmp_ssa = codegen.GetExprAsCode(op->args_[3]);
  // Fall back to tmp's annotation when src lacks a MemRef (ForStmt result var, tile.reshape view).
  if (src_type.empty()) {
    src_type = codegen.GetExprTypeAnnotation(op->args_[3]);
  }

  std::string result_target = codegen.GetCurrentResultTarget();
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << "pto.ttrans ins(" << src_ssa << ", " << tmp_ssa;
  if (!src_type.empty()) {
    oss << " : " << src_type << ", " << src_type;
  }
  oss << ") outs(" << result_target;
  if (!result_type.empty()) {
    oss << " : " << result_type;
  }
  oss << ")";
  codegen.Emit(oss.str());
  return std::string("");
}

// pto.tcolexpand takes only the column vector in ins(); output shape comes from outs().
// IR tile.col_expand(target, col_vec) keeps target for shape/type inference only.
static std::string MakeColExpandCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.col_expand requires 2 arguments, got " << op->args_.size();
  const ir::ExprPtr& col_vec = op->args_[1];
  std::string operand = codegen.GetExprAsCode(col_vec);
  std::string in_type = codegen.GetExprTypeAnnotation(col_vec);
  std::string result_target = codegen.GetCurrentResultTarget();
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  std::ostringstream oss;
  oss << "pto.tcolexpand ins(" << operand;
  if (!in_type.empty()) {
    oss << " : " << in_type;
  }
  oss << ") outs(" << result_target;
  if (!result_type.empty()) {
    oss << " : " << result_type;
  }
  oss << ")";
  codegen.Emit(oss.str());
  return "";
}

// pto.trowexpand takes only the row vector in ins(); output shape comes from outs().
// IR tile.row_expand(target, row_vec) keeps target for shape/type inference only.
static std::string MakeRowExpandCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.row_expand requires 2 arguments, got " << op->args_.size();
  const ir::ExprPtr& row_vec = op->args_[1];
  std::string operand = codegen.GetExprAsCode(row_vec);
  std::string in_type = codegen.GetExprTypeAnnotation(row_vec);
  std::string result_target = codegen.GetCurrentResultTarget();
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  std::ostringstream oss;
  oss << "pto.trowexpand ins(" << operand;
  if (!in_type.empty()) {
    oss << " : " << in_type;
  }
  oss << ") outs(" << result_target;
  if (!result_type.empty()) {
    oss << " : " << result_type;
  }
  oss << ")";
  codegen.Emit(oss.str());
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
  int mode = op->GetKwarg<int>("cmp_type");
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
  int mode = op->GetKwarg<int>("cmp_type");
  CHECK(mode >= 0 && mode < static_cast<int>(cmp_modes.size())) << "Tile cmp mode out of range: " << mode;
  std::string config_attr = "{cmpMode = #pto<cmp " + cmp_modes.at(mode) + ">}";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

// Verify that two TileTypes share the strict "same tile config" required by
// pto.subview: identical dtype, identical TileView (blayout, slayout, fractal,
// pad), and pad must be null since pto.subview is a pure view and does not
// pad.  Memory-space equality is enforced separately (via memory_inherit
// rules on the op definition); this helper checks the tile_view fields that
// must be byte-for-byte compatible for a subview to be legal.
static void CheckSubviewTileCompat(const ir::TileType& source, const ir::TileType& result,
                                   const std::string& op_name) {
  CHECK(source.dtype_ == result.dtype_) << op_name << ": source and result must share dtype, got "
                                        << source.dtype_.ToString() << " vs " << result.dtype_.ToString();

  const auto src_v = ir::tile_view_semantics::GetEffectiveTileView(source);
  const auto res_v = ir::tile_view_semantics::GetEffectiveTileView(result);
  CHECK(src_v.blayout == res_v.blayout)
      << op_name
      << ": blayout mismatch between source and result; pto.subview requires identical block layout";
  CHECK(src_v.slayout == res_v.slayout)
      << op_name
      << ": slayout mismatch between source and result; pto.subview requires identical scatter layout";
  CHECK(src_v.fractal == res_v.fractal) << op_name << ": fractal mismatch (" << src_v.fractal << " vs "
                                        << res_v.fractal << "); pto.subview requires identical fractal";
  CHECK(src_v.pad == res_v.pad)
      << op_name << ": pad mismatch between source and result; pto.subview requires identical pad mode";
  CHECK(src_v.pad == ir::PadValue::null)
      << op_name << ": pto.subview does not support pad_value (" << static_cast<int>(src_v.pad)
      << "); apply tile.fillpad on the result tile instead of carrying a pad on the slice/assemble window";
}

// Helper function for tile.assemble → pto.subview + pto.tmov
// Writes source tile into target tile at a given row/col offset.  Lowering:
//   1. (optional) pto.tmov target → dst when buffer reuse did not merge them
//      (preserves any data outside the insertion window).
//   2. %dst_view = pto.subview %dst[row, col] sizes [src.rows, src.cols] : ... -> ...
//   3. pto.tmov ins(%src) outs(%dst_view)
// Arguments: args[0] = target (destination base), args[1] = source, args[2] = offset MakeTuple
static std::string MakeTileAssembleCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.assemble requires 3 arguments (target, source, offset), got "
                               << op->args_.size();

  auto target_tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  auto source_tile_type = ir::As<ir::TileType>(op->args_[1]->GetType());
  INTERNAL_CHECK_SPAN(target_tile_type && source_tile_type, op->span_)
      << "tile.assemble target and source must both be TileType";
  CheckSubviewTileCompat(*target_tile_type, *source_tile_type, "tile.assemble");

  std::string target = codegen.GetExprAsCode(op->args_[0]);
  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  auto offset_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offset_tuple, op->span_) << "tile.assemble third argument must be a tuple (offset)";
  INTERNAL_CHECK_SPAN(offset_tuple->elements_.size() >= 2, op->span_)
      << "tile.assemble offset tuple must have at least 2 elements (row, col), got "
      << offset_tuple->elements_.size();
  std::string row_off = codegen.GetExprAsCode(offset_tuple->elements_[0]);
  std::string col_off = codegen.GetExprAsCode(offset_tuple->elements_[1]);

  // pto.subview is a view, so writing into the dst_view only affects the
  // [row, col]+sizes window.  Data outside that window must already be present
  // in dst — when target and dst are different buffers (memory reuse did not
  // merge them), copy target → dst first to preserve target's outer data.
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

  // Build %dst_view = pto.subview %dst[%row, %col] sizes [R, C] valid [Vr, Vc] : <dst_type> -> <view_type>
  // The subview "sizes" attribute is the source tile's physical shape, while
  // the explicit `valid [...]` operands must match the source tile's logical
  // valid_shape. PTOAS v0.32 validates that the result tile_buf type's
  // v_row/v_col agree with those explicit valid operands, so the result type
  // must be static when source valid_shape is static, and dynamic only when the
  // source valid_shape itself is dynamic.
  const auto& src_shape = source_tile_type->shape_;
  INTERNAL_CHECK_SPAN(src_shape.size() >= 2, op->span_)
      << "tile.assemble source must have at least 2 dimensions for pto.subview";
  auto rows_const = ir::As<ir::ConstInt>(src_shape[0]);
  auto cols_const = ir::As<ir::ConstInt>(src_shape[1]);
  INTERNAL_CHECK_SPAN(rows_const && cols_const, op->span_)
      << "tile.assemble source shape must be compile-time constant for pto.subview sizes attribute";

  ir::ExprPtr valid_row_expr = src_shape[0];
  ir::ExprPtr valid_col_expr = src_shape[1];
  const auto src_valid = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type).valid_shape;
  if (src_valid.size() >= 1 && src_valid[0]) valid_row_expr = src_valid[0];
  if (src_valid.size() >= 2 && src_valid[1]) valid_col_expr = src_valid[1];

  auto valid_row_const = ir::As<ir::ConstInt>(valid_row_expr);
  auto valid_col_const = ir::As<ir::ConstInt>(valid_col_expr);
  std::string valid_rows = valid_row_const
                               ? codegen.GetOrEmitConstant(valid_row_const->value_, DataType::INDEX)
                               : codegen.GetExprAsCode(valid_row_expr);
  std::string valid_cols = valid_col_const
                               ? codegen.GetOrEmitConstant(valid_col_const->value_, DataType::INDEX)
                               : codegen.GetExprAsCode(valid_col_expr);

  INTERNAL_CHECK_SPAN(source_tile_type->memory_space_.has_value(), op->span_)
      << "tile.assemble source must carry a memory space for pto.subview result typing";
  const auto source_memory_space = source_tile_type->memory_space_.value_or(ir::MemorySpace::DDR);
  auto view_type_info =
      codegen::ExtractTileTypeInfo(*source_tile_type, codegen.GetTypeString(source_tile_type->dtype_));
  if (valid_row_const) {
    view_type_info.v_row = valid_row_const->value_;
    view_type_info.v_row_dynamic = false;
  }
  if (valid_col_const) {
    view_type_info.v_col = valid_col_const->value_;
    view_type_info.v_col_dynamic = false;
  }
  std::string view_type = codegen::FormatTileBufTypeString(
      codegen::MemorySpaceToMLIR(source_memory_space), view_type_info.dtype_str, view_type_info.rows,
      view_type_info.cols, view_type_info.blayout, view_type_info.slayout, view_type_info.fractal,
      view_type_info.pad, view_type_info.v_row, view_type_info.v_col, view_type_info.v_row_dynamic,
      view_type_info.v_col_dynamic);

  std::string dst_view = codegen.NewNamedTemp("assemble_view");
  std::ostringstream sv;
  sv << dst_view << " = pto.subview " << dst << "[" << row_off << ", " << col_off << "] sizes ["
     << rows_const->value_ << ", " << cols_const->value_ << "]";
  sv << " valid [" << valid_rows << ", " << valid_cols << "]";
  if (!dst_type.empty() && !view_type.empty()) {
    sv << " : " << dst_type << " -> " << view_type;
  }
  codegen.Emit(sv.str());
  if (!view_type.empty()) {
    codegen.RegisterTileBufType(dst_view, view_type);
  }

  // Emit pto.tmov ins(%src) outs(%dst_view) — the actual data transfer.
  std::ostringstream tmov;
  tmov << "pto.tmov ins(" << src;
  if (!src_type.empty()) tmov << " : " << src_type;
  tmov << ") outs(" << dst_view;
  if (!view_type.empty()) tmov << " : " << view_type;
  tmov << ")";
  codegen.Emit(tmov.str());
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
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name
                               << "] requires 2 arguments (start, shape), but got " << op->args_.size();
  bool descending = op->GetKwarg<bool>("descending");
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string config_attr = descending ? "{descending = true}" : "{descending = false}";
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
  std::ostringstream oss;
  oss << pto_op_name << " ins(" << src << " " << config_attr;
  if (!src_type.empty()) {
    oss << " : " << src_type;
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";
  codegen.Emit(oss.str());
  return "";
}

// Helper function for Sort32: emits pto.tsort32
// PTOAS expects: ins(src, idx : src_type, idx_type) outs(dst : dst_type)
static std::string MakeSort32CodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name
                               << "] requires 2 arguments (src, idx), but got " << op->args_.size();

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string idx = codegen.GetExprAsCode(op->args_[1]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string idx_type = codegen.GetExprTypeAnnotation(op->args_[1]);

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << pto_op_name;
  // ins clause: src, idx
  oss << " ins(" << src << ", " << idx;
  if (!src_type.empty() || !idx_type.empty()) {
    oss << " : " << src_type << ", " << idx_type;
  }
  // outs clause: dst only (idx is modified in-place by hardware)
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper function for GatherMask: emits pto.tgather with maskPattern attribute
// PTOAS expects: ins(src, {maskPattern = #pto.mask_pattern<Pxxxx>} : src_type) outs(dst : dst_type)
static std::string MakeGatherMaskCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "tile.gather_mask requires 1 argument (src), but got " << op->args_.size();

  int pattern = op->GetKwarg<int>("mask_pattern");
  CHECK(pattern >= 1 && pattern < static_cast<int>(mask_patterns.size()))
      << "mask_pattern out of range: " << pattern;

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << "pto.tgather ins(" << src << ", {maskPattern = #pto.mask_pattern<" << mask_patterns.at(pattern)
      << ">}";
  if (!src_type.empty()) {
    oss << " : " << src_type;
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper for tile.gather_compare (TGATHER compare-form, two outputs):
//   pto.tgather ins(src, kvalue, tmp {cmpMode = #pto<cmp eq>, offset = N : i32}
//                   : src_ty, kv_ty, tmp_ty)
//               outs(dst, cdst : dst_ty, cdst_ty)
//
// Op surface: 3 inputs / TupleType{dst_TileType, cdst_TileType} output. DPS
// dst/cdst buffers are bound by downstream `<element> = tuple_var[i]`
// AssignStmts (parser desugaring of `dst, cdst = ...`). Because the framework
// only pre-binds `fs_.current_result_*` for TileType LHS, multi-output ops
// must resolve their own DPS targets — done via ResolveTupleResultElements.
static std::string MakeGatherCompareCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.gather_compare requires 3 arguments (src, kvalue, tmp), but got "
                               << op->args_.size();

  ir::VarPtr tuple_var = codegen.GetCurrentResultVar();
  INTERNAL_CHECK_SPAN(tuple_var, op->span_)
      << "Internal error: tile.gather_compare codegen requires current_result_var";

  auto element_vars = codegen.ResolveTupleResultElements(tuple_var, /*arity=*/2);
  INTERNAL_CHECK_SPAN(element_vars[0] && element_vars[1], op->span_)
      << "Internal error: tile.gather_compare expects two TupleGetItemExpr consumers (dst, cdst), got "
      << (element_vars[0] ? "dst-yes" : "dst-no") << "/" << (element_vars[1] ? "cdst-yes" : "cdst-no");

  // Eagerly emit alloc_tile for dst/cdst; the later `dst = tuple_var[i]`
  // AssignStmts skip re-emission via fs_.emitted_tile_alloc_vars.
  std::array<std::shared_ptr<const ir::TileType>, 2> elem_types;
  for (size_t i = 0; i < 2; ++i) {
    elem_types[i] = ir::GetTileTypeWithMemRef(element_vars[i]->GetType());
    INTERNAL_CHECK_SPAN(elem_types[i], element_vars[i]->span_)
        << "Internal error: tile.gather_compare element var " << i
        << " must have TileType with MemRef set by InitMemRef";
    codegen.EmitAllocTileForVar(element_vars[i], elem_types[i]);
  }

  int cmp_mode = op->GetKwarg<int>("cmp_mode");
  CHECK(cmp_mode >= 0 && cmp_mode < 6) << "tile.gather_compare cmp_mode out of range: " << cmp_mode;
  static constexpr const char* kCmpNames[] = {"eq", "ne", "lt", "le", "gt", "ge"};
  int offset = op->GetKwarg<int>("offset", 0);

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string kvalue = codegen.GetExprAsCode(op->args_[1]);
  std::string tmp = codegen.GetExprAsCode(op->args_[2]);
  std::string src_ty = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string kv_ty = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string tmp_ty = codegen.GetExprTypeAnnotation(op->args_[2]);
  std::string dst = codegen.GetVarName(element_vars[0]);
  std::string cdst = codegen.GetVarName(element_vars[1]);
  std::string dst_ty = codegen.GetTileBufTypeStringFromTileType(elem_types[0]);
  std::string cdst_ty = codegen.GetTileBufTypeStringFromTileType(elem_types[1]);

  std::ostringstream oss;
  oss << "pto.tgather ins(" << src << ", " << kvalue << ", " << tmp;
  if (!src_ty.empty() || !kv_ty.empty() || !tmp_ty.empty()) {
    oss << " : " << src_ty << ", " << kv_ty << ", " << tmp_ty;
  }
  oss << ") outs(" << dst << ", " << cdst;
  if (!dst_ty.empty() || !cdst_ty.empty()) {
    oss << " : " << dst_ty << ", " << cdst_ty;
  }
  oss << ") {cmpMode = #pto<cmp " << kCmpNames[cmp_mode] << ">, offset = " << offset << " : i32}";

  codegen.Emit(oss.str());
  return "";
}

// Helper for tile.scatter (TSCATTER index form, DPS):
//   pto.tscatter ins(%src, %indexes : src_ty, idx_ty) outs(%dst : dst_ty)
//
// IR surface: 3-input op (dst, src, indexes) marked
// set_output_reuses_input(0) — the AssignStmt LHS aliases `dst` so
// GetCurrentResultTarget() returns the same SSA as args_[0].
static std::string MakeScatterCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.scatter requires 3 arguments (dst, src, indexes), but got "
                               << op->args_.size();

  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string idx = codegen.GetExprAsCode(op->args_[2]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string idx_type = codegen.GetExprTypeAnnotation(op->args_[2]);

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  // DPS in-place contract: tile.scatter is set_output_reuses_input(0), so the
  // result buffer must alias the `dst` input tile (args_[0]). Otherwise the
  // tscatter writes a freshly-allocated tile and the rows it does not touch are
  // never initialized with `dst`'s values. PTOCodegen guarantees this by
  // binding the result var to the input SSA (ShouldAliasScatterResultToInput).
  std::string input_ssa = codegen.GetExprAsCode(op->args_[0]);
  INTERNAL_CHECK(!dst.empty() && dst == input_ssa)
      << "Internal error: tile.scatter result SSA must alias the dst input tile SSA, got dst=" << dst
      << ", input=" << input_ssa;

  std::ostringstream oss;
  oss << "pto.tscatter ins(" << src << ", " << idx;
  // Emit the type clause only when both annotations are present; printing one
  // alone would produce malformed PTOAS (": , idx" or ": src, ").
  if (!src_type.empty() && !idx_type.empty()) {
    oss << " : " << src_type << ", " << idx_type;
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper for tile.scatter_mask (TSCATTER mask form, DPS):
//   pto.tscatter ins(%src : src_ty) outs(%dst : dst_ty)
//                {maskPattern = #pto.mask_pattern<Pxxxx>}
//
// Unlike pto.tgather's mask form (which carries maskPattern inside ins()),
// pto.tscatter only accepts SSA operands in ins() — the maskPattern must be a
// trailing op attribute dict (same placement as gather_compare's cmpMode).
//
// IR surface: 2-input op (dst, src) + mask_pattern attr; dst aliased via
// set_output_reuses_input(0). Mask form is targeted at A3 / CPU-sim style
// backends; A5 rejects it on the PTOAS side.
static std::string MakeScatterMaskCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.scatter_mask requires 2 arguments (dst, src), but got "
                               << op->args_.size();

  int pattern = op->GetKwarg<int>("mask_pattern");
  CHECK(pattern >= 1 && pattern < static_cast<int>(mask_patterns.size()))
      << "tile.scatter_mask mask_pattern out of range: " << pattern;

  std::string src = codegen.GetExprAsCode(op->args_[1]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  // DPS in-place contract (mirror of tile.scatter): result must alias the `dst`
  // input tile (args_[0]) so mask-marked columns are written in-place and the
  // unselected columns keep `dst`'s values.
  std::string input_ssa = codegen.GetExprAsCode(op->args_[0]);
  INTERNAL_CHECK(!dst.empty() && dst == input_ssa)
      << "Internal error: tile.scatter_mask result SSA must alias the dst input tile SSA, got dst=" << dst
      << ", input=" << input_ssa;

  std::ostringstream oss;
  oss << "pto.tscatter ins(" << src;
  if (!src_type.empty()) {
    oss << " : " << src_type;
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ") {maskPattern = #pto.mask_pattern<" << mask_patterns.at(pattern) << ">}";

  codegen.Emit(oss.str());
  return "";
}

// Helper function for MrgSort format2: emits pto.tmrgsort
// Supports 2-4 way merge. tmp is the last ins operand and carries the
// {exhausted} attribute; outs holds dst plus a synthesized executed vector
// (vector<4xi16>) — the executed status is not an IR-level tile operand:
//   2-way: ins(src0, src1, tmp {exhausted} : src_types..., tmp_type)
//          outs(dst, executed : dst_type, vector<4xi16>)
//   3-way: ins(src0, src1, src2, tmp {exhausted} : ...) outs(dst, executed : ...)
//   4-way: ins(src0, src1, src2, src3, tmp {exhausted} : ...) outs(dst, executed : ...)
static std::string MakeMrgSortCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() >= 3 && op->args_.size() <= 5)
      << "Operation:[" << pto_op_name << "] requires 3-5 arguments (2-4 srcs + tmp), but got "
      << op->args_.size();

  size_t n_srcs = op->args_.size() - 1;

  std::vector<std::string> srcs, src_types;
  for (size_t i = 0; i < n_srcs; ++i) {
    srcs.push_back(codegen.GetExprAsCode(op->args_[i]));
    src_types.push_back(codegen.GetExprTypeAnnotation(op->args_[i]));
  }

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
  std::string tmp = codegen.GetExprAsCode(op->args_[n_srcs]);
  std::string tmp_type = codegen.GetExprTypeAnnotation(op->args_[n_srcs]);
  std::string executed_vec = codegen.NewNamedTemp("executed_vec");
  codegen.Emit(executed_vec + " = arith.constant dense<0> : vector<4xi16>");

  bool exhausted = op->GetKwarg<bool>("exhausted", false);
  std::string exhausted_attr = exhausted ? "{exhausted = true}" : "{exhausted = false}";

  std::ostringstream oss;
  oss << pto_op_name << " ins(";
  for (size_t i = 0; i < n_srcs; ++i) {
    oss << srcs[i] << ", ";
  }
  oss << tmp << " " << exhausted_attr;

  bool has_types = !tmp_type.empty();
  for (const auto& t : src_types) {
    if (!t.empty()) {
      has_types = true;
      break;
    }
  }
  if (has_types) {
    oss << " : ";
    for (size_t i = 0; i < n_srcs; ++i) {
      oss << src_types[i] << ", ";
    }
    oss << tmp_type;
  }

  oss << ") outs(" << dst << ", " << executed_vec;
  if (!dst_type.empty()) {
    oss << " : " << dst_type << ", vector<4xi16>";
  }
  oss << ")";

  codegen.Emit(oss.str());
  return "";
}

// Helper function for MrgSort1 format1: emits pto.tmrgsort
// format1: ins(src, blockLen : src_type, i32) outs(dst : dst_type)
static std::string MakeMrgSort1CodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                          codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name
                               << "] requires 2 arguments (src, block_len), but got " << op->args_.size();

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  // blockLen must be i32 per PTO ISA. Constants use the optimized dedup path;
  // runtime variables (e.g., loop-carried block_len) go through GetExprAsCode + cast.
  std::string block_len;
  if (auto const_int = ir::As<ir::ConstInt>(op->args_[1])) {
    block_len = codegen.GetOrEmitConstant(static_cast<int64_t>(static_cast<int32_t>(const_int->value_)),
                                          DataType::INT32);
  } else {
    block_len = codegen.GetExprAsCode(op->args_[1]);
    block_len = codegen.EmitCastToI32(op->args_[1], block_len);
  }

  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream oss;
  oss << pto_op_name << " ins(" << src << ", " << block_len;
  if (!src_type.empty()) {
    oss << " : " << src_type << ", i32";
  }
  oss << ") outs(" << dst;
  if (!dst_type.empty()) {
    oss << " : " << dst_type;
  }
  oss << ")";

  codegen.Emit(oss.str());
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
  INTERNAL_CHECK_SPAN(tensor, op->span_) << "tile.load first argument must be a Var or IterArg";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "tile.load second argument must be a tuple (offsets)";

  INTERNAL_CHECK_SPAN(op->args_.size() >= 3, op->span_)
      << "tile.load expects at least 3 arguments (tensor, offsets, shapes), but got " << op->args_.size();

  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(shapes_tuple, op->span_) << "tile.load third argument must be a tuple (shapes)";

  // valid_shapes is optional: when omitted (callers built before the 4-arg
  // signature was introduced, or hand-written IR), fall back to shapes so the
  // partition_view covers the entire physical region — equivalent to the DSL
  // behavior `pl.load(..., valid_shapes=None)`.
  auto valid_shapes_tuple = shapes_tuple;
  if (op->args_.size() >= 4) {
    valid_shapes_tuple = As<ir::MakeTuple>(op->args_[3]);
    INTERNAL_CHECK_SPAN(valid_shapes_tuple, op->span_)
        << "tile.load fourth argument must be a tuple (valid_shapes)";
  }

  auto tensor_type = AsTensorTypeLike(tensor->GetType());
  INTERNAL_CHECK_SPAN(tensor_type, op->span_) << "tile.load tensor argument must have TensorType";

  const size_t ndim = shapes_tuple->elements_.size();
  INTERNAL_CHECK_SPAN(ndim >= 1, op->span_) << "tile.load shapes tuple must have at least one element";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK_SPAN(!tile_buf.empty(), op->span_) << "tile.load requires assignment target (tile_buf)";

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();

  // RFC #1300 P7: the IR's offsets / shapes / valid_shapes are already in
  // canonical coordinates (matching the source TensorType's shape). There is
  // no implicit dn_swap here — ``LowerTransposeLoadParamLayout`` (P6) is
  // responsible for ensuring all coordinate systems match before codegen.
  const auto& valid_elems = valid_shapes_tuple->elements_;
  const auto& offset_elems = offsets_tuple->elements_;

  std::string partition_type = MakePartitionTensorViewType(GetDimStrings(valid_elems), dtype_str);
  std::string partition_view =
      EmitPartitionViewPTO(tensor->name_hint_, tensor_view, tensor_view_type, partition_type,
                           GetExprCodes(offset_elems, codegen), GetSizeCodes(valid_elems, codegen), codegen);

  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << partition_view << " : " << partition_type << ") outs(";
  tload_line << tile_buf << " : " << tile_buf_type << ")";
  codegen.Emit(tload_line.str());

  // No follow-up `pto.set_validshape` is emitted: every `pto.alloc_tile`
  // already carries the desired `valid_row` / `valid_col` operands, and the
  // partition_view above already reflects the same valid region.

  return "";
}

// tile.store: emit pto.partition_view + pto.tstore
static std::string MakeTileStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(tile, op->span_) << "tile.store first argument must be a Var or IterArg";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "tile.store second argument must be a tuple (offsets)";

  auto tile_type = As<ir::TileType>(tile->GetType());
  INTERNAL_CHECK_SPAN(tile_type, op->span_) << "tile.store first argument must have TileType";
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*tile_type);
  const auto& valid_shape = tile_view.valid_shape;
  INTERNAL_CHECK_SPAN(valid_shape.size() == 2, op->span_) << "tile.store tile valid_shape must be 2D";

  auto height_code = codegen.GetExprAsCode(valid_shape[0]);
  auto width_code = codegen.GetExprAsCode(valid_shape[1]);

  auto output_tensor = AsVarLike(op->args_[2]);
  INTERNAL_CHECK_SPAN(output_tensor, op->span_) << "tile.store output_tensor must be a Var or IterArg";

  auto tensor_type = AsTensorTypeLike(output_tensor->GetType());
  INTERNAL_CHECK_SPAN(tensor_type, op->span_) << "tile.store output_tensor must have TensorType";

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::string partition_view;
  std::string partition_type;
  const size_t tensor_rank = tensor_type->shape_.size();

  // RFC #1300 P7: the IR's offsets / shapes are already in canonical
  // coordinates (matching the source TensorType's shape). No implicit
  // dn_swap here — the IR-level lowering passes (P6 + canonical TensorView)
  // are responsible for ensuring all coordinate systems match before codegen.

  // Check if FlattenTileNdTo2D injected an explicit shapes tuple as args[3].
  ir::MakeTuplePtr shapes_tuple;
  if (tensor_rank > 2 && op->args_.size() > 3) {
    shapes_tuple = As<ir::MakeTuple>(op->args_[3]);
  }

  if (shapes_tuple) {
    // N-rank partition path: use the explicit shapes tuple from FlattenTileNdTo2D.
    const auto& shape_elems = shapes_tuple->elements_;
    const auto& offset_elems = offsets_tuple->elements_;
    partition_type = MakePartitionTensorViewType(GetDimStrings(shape_elems), dtype_str);
    partition_view = EmitPartitionViewPTO(output_tensor->name_hint_, tensor_view, tensor_view_type,
                                          partition_type, GetExprCodes(offset_elems, codegen),
                                          GetSizeCodes(shape_elems, codegen), codegen);
  } else {
    // Standard 1D/2D path
    std::string height_dim = "?", width_dim = "?";
    if (auto h = As<ir::ConstInt>(valid_shape[0])) height_dim = std::to_string(h->value_);
    if (auto w = As<ir::ConstInt>(valid_shape[1])) width_dim = std::to_string(w->value_);
    partition_type = MakePartitionTensorViewType({height_dim, width_dim}, dtype_str);
    partition_view = EmitPartitionViewPTO(output_tensor->name_hint_, tensor_view, tensor_view_type,
                                          partition_type, GetExprCodes(offsets_tuple->elements_, codegen),
                                          {height_code, width_code}, codegen);
  }

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  if (!tile_buf_type.empty()) {
    tstore_line << " : " << tile_buf_type;
  }
  tstore_line << ") outs(" << partition_view << " : " << partition_type << ")";

  // Optional atomic-add combine mode (split-K accumulation into GM). The attr
  // is emitted only for atomic_add — a plain store omits it so non-atomic
  // codegen stays byte-identical (pto.tstore's atomicType defaults to none).
  const int atomic_int = op->GetKwarg<int>("atomic", 0);
  INTERNAL_CHECK_SPAN(atomic_int == static_cast<int>(ir::AtomicType::kNone) ||
                          atomic_int == static_cast<int>(ir::AtomicType::kAdd),
                      op->span_)
      << "tile.store atomic kwarg must encode AtomicType::kNone or kAdd, got " << atomic_int;
  if (atomic_int == static_cast<int>(ir::AtomicType::kAdd)) {
    tstore_line << " {atomicType = #pto<atomic_type atomic_add>}";
  }
  codegen.Emit(tstore_line.str());

  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
    codegen.RegisterTensorView(result_var, tensor_view);
    codegen.RegisterVarToMlir(result_var, tensor_view);
    codegen.RegisterBasePtr(result_var, codegen.GetTensorBasePtr(output_tensor));
  }

  return "";
}

// tile.mscatter(src, idx, output_tensor) -> pto.mscatter
// Generates:
//   %pview = pto.partition_view %tensor_view, offsets=[0,...], sizes=[d0,...] : ... -> ...
//   pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
//                outs(%pview : !pto.partition_tensor_view<...>)
static std::string MakeTileMscatterCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  INTERNAL_CHECK(op->args_.size() == 3)
      << "tile.mscatter requires 3 arguments (src, idx, output_tensor), got " << op->args_.size();

  auto src = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(src) << "tile.mscatter src must be a Var or IterArg";
  auto idx = AsVarLike(op->args_[1]);
  INTERNAL_CHECK(idx) << "tile.mscatter idx must be a Var or IterArg";
  auto output_tensor = AsVarLike(op->args_[2]);
  INTERNAL_CHECK(output_tensor) << "tile.mscatter output_tensor must be a Var or IterArg";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "tile.mscatter output_tensor must have TensorType";

  std::string src_name = codegen.GetVarName(src);
  std::string idx_name = codegen.GetVarName(idx);
  std::string src_type_annot = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string idx_type_annot = codegen.GetExprTypeAnnotation(op->args_[1]);

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());

  // Build pto.partition_view covering the entire tensor (mscatter uses per-element
  // indices, so the partition is the whole tensor — offsets all zero, sizes = shape).
  std::string partition_view = codegen.NewNamedTemp(output_tensor->name_hint_ + "_pview");
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [";
  for (size_t i = 0; i < tensor_type->shape_.size(); ++i) {
    if (i > 0) partition_line << ", ";
    partition_line << codegen.GetOrEmitConstant(static_cast<int64_t>(0), DataType::INDEX);
  }
  partition_line << "], sizes = [";
  std::string partition_type = "!pto.partition_tensor_view<";
  for (size_t i = 0; i < tensor_type->shape_.size(); ++i) {
    if (i > 0) {
      partition_line << ", ";
      partition_type += "x";
    }
    if (auto c = As<ir::ConstInt>(tensor_type->shape_[i])) {
      partition_line << codegen.GetOrEmitConstant(c->value_, DataType::INDEX);
      partition_type += std::to_string(c->value_);
    } else {
      partition_line << codegen.GetExprAsCode(tensor_type->shape_[i]);
      partition_type += "?";
    }
  }
  partition_line << "]";
  partition_type += "x" + dtype_str + ">";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  // Emit pto.mscatter with partition_view in outs()
  std::ostringstream mscatter_line;
  mscatter_line << "pto.mscatter ins(" << src_name << ", " << idx_name;
  if (!src_type_annot.empty() && !idx_type_annot.empty()) {
    mscatter_line << " : " << src_type_annot << ", " << idx_type_annot;
  }
  mscatter_line << ") outs(" << partition_view << " : " << partition_type << ")";
  codegen.Emit(mscatter_line.str());

  // Propagate tensor_view to the result var so downstream ops see the updated tensor
  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
    codegen.RegisterTensorView(result_var, tensor_view);
    codegen.RegisterVarToMlir(result_var, tensor_view);
    codegen.RegisterBasePtr(result_var, codegen.GetTensorBasePtr(output_tensor));
  }

  return "";
}

// Helper function for tile.alloc (no-op: allocation handled elsewhere)
static std::string MakeTileAllocCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No MLIR emission - pto.alloc_tile generated from MemRefs in TileTypes
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
    return codegen.GetOrEmitConstant(flat_offset, DataType::INDEX);
  }

  std::vector<std::string> index_ssa;
  index_ssa.reserve(indices.size());
  for (const auto& index : indices) {
    if (auto c = As<ir::ConstInt>(index)) {
      index_ssa.push_back(codegen.GetOrEmitConstant(c->value_, DataType::INDEX));
      continue;
    }
    index_ssa.push_back(codegen.EmitCastToIndex(index, codegen.GetExprAsCode(index)));
  }
  return EmitFlatOffsetSSAFromValues(index_ssa, shape, codegen, "flat_offset");
}

// Helper function for tile.read (indices -> flat offset -> pto.tgetval)
static std::string MakeTileReadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.read requires 2 arguments, but got " << op->args_.size();

  auto tile_type = As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tile_type, op->span_) << "tile.read first argument must be TileType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(indices_tuple, op->span_) << "tile.read second argument must be MakeTuple (indices)";

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
  INTERNAL_CHECK_SPAN(tile_type, op->span_) << "tile.write first argument must be TileType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(indices_tuple, op->span_) << "tile.write second argument must be MakeTuple (indices)";

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

  auto tensor_type_ptr = AsTensorTypeLike(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tensor_type_ptr, op->span_) << "tensor.read first argument must be TensorType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(indices_tuple, op->span_) << "tensor.read second argument must be MakeTuple (indices)";

  auto scalar_type_ptr = As<ir::ScalarType>(op->GetType());
  INTERNAL_CHECK_SPAN(scalar_type_ptr, op->span_) << "tensor.read result must be ScalarType";
  std::string scalar_type = codegen.GetTypeString(scalar_type_ptr->dtype_);

  // store_scalar/load_scalar need the base !pto.ptr; resolve via the tensor var
  // even after a slice-assign rebound it to a tensor_view (issue #1493).
  std::string src = codegen.GetTensorBasePtr(AsVarLike(op->args_[0]));
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

  auto tensor_type_ptr = AsTensorTypeLike(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tensor_type_ptr, op->span_) << "tensor.write first argument must be TensorType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(indices_tuple, op->span_) << "tensor.write second argument must be MakeTuple (indices)";

  // store_scalar needs the base !pto.ptr; resolve via the tensor var even after
  // a prior slice-assign rebound it to a tensor_view (issue #1493).
  std::string tensor = codegen.GetTensorBasePtr(AsVarLike(op->args_[0]));
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
    codegen.RegisterBasePtr(result_var, tensor);
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
    shape_name = codegen.GetOrEmitConstant(static_shape->value_, DataType::INDEX);
  } else {
    INTERNAL_CHECK_SPAN(false, op->span_) << "Internal error: tensor.dim shape is neither Var nor ConstInt";
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

static std::string EmitIndexOperand(codegen::PTOCodegen& codegen, const ExprPtr& expr,
                                    std::string_view context) {
  INTERNAL_CHECK(expr) << "Internal error: " << context << " requires a non-null index operand";
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return codegen.GetOrEmitConstant(const_int->value_, DataType::INDEX);
  }

  std::string ssa = codegen.GetExprAsCode(expr);
  if (auto scalar_type = As<ScalarType>(expr->GetType())) {
    if (scalar_type->dtype_ == DataType::INDEX) {
      return ssa;
    }
    CHECK(scalar_type->dtype_.IsInt()) << context << " operand must be integer or index type, got "
                                       << codegen.GetTypeString(scalar_type->dtype_);
    std::string idx = codegen.NewTemp();
    codegen.Emit(idx + " = arith.index_cast " + ssa + " : " + codegen.GetTypeString(scalar_type->dtype_) +
                 " to index");
    return idx;
  }
  return ssa;
}

static bool IsSameDimExpr(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto lhs_const = As<ir::ConstInt>(lhs);
  auto rhs_const = As<ir::ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

static std::shared_ptr<const ir::TileType> GetTpushTileType(const ExprPtr& tile_expr) {
  if (auto tile_type = ir::GetTileTypeWithMemRef(tile_expr->GetType())) {
    return tile_type;
  }
  return As<ir::TileType>(tile_expr->GetType());
}

static bool EmitSplitTpushTransportValidShape(const CallPtr& op, codegen::PTOCodegen& codegen,
                                              const std::string& tile_buf, const std::string& tile_type,
                                              int split) {
  if (split == 0 || tile_buf.empty() || tile_type.empty()) {
    return false;
  }

  auto source_tile_type = GetTpushTileType(op->args_[0]);
  if (!source_tile_type || source_tile_type->shape_.size() < 2) {
    return false;
  }
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type);
  if (tile_view.valid_shape.size() < 2) {
    return false;
  }

  // The transport must carry the full box for both subblocks to receive
  // complete data: each subblock reads its half of the slot regardless of the
  // user-declared valid_shape. Narrowing valid_shape on the producer side
  // before tpush would leave the non-split axis under-written and (on LR
  // splits in particular) make even subblock 0 see zeros for the cells the
  // producer skipped. Localization back to the user's logical valid happens
  // on the consumer side via LocalizeValidDimForSplit.
  const auto& shape = source_tile_type->shape_;
  const auto& valid_shape = tile_view.valid_shape;
  ExprPtr transport_row = shape[0];
  ExprPtr transport_col = shape[1];

  if (IsSameDimExpr(transport_row, valid_shape[0]) && IsSameDimExpr(transport_col, valid_shape[1])) {
    return false;
  }

  std::string row = EmitIndexOperand(codegen, transport_row, "tpush transport valid_row");
  std::string col = EmitIndexOperand(codegen, transport_col, "tpush transport valid_col");
  codegen.Emit("pto.set_validshape " + tile_buf + ", " + row + ", " + col + " : " + tile_type);
  return true;
}

static void EmitLogicalTpushValidShapeRestore(const CallPtr& op, codegen::PTOCodegen& codegen,
                                              const std::string& tile_buf, const std::string& tile_type) {
  auto source_tile_type = GetTpushTileType(op->args_[0]);
  INTERNAL_CHECK(source_tile_type) << "Internal error: tpush validShape restore requires a TileType source";
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*source_tile_type);
  INTERNAL_CHECK(tile_view.valid_shape.size() >= 2)
      << "Internal error: tpush validShape restore requires rank-2 validShape";
  const auto& valid_shape = tile_view.valid_shape;
  std::string row = EmitIndexOperand(codegen, valid_shape[0], "tpush logical valid_row");
  std::string col = EmitIndexOperand(codegen, valid_shape[1], "tpush logical valid_col");
  codegen.Emit("pto.set_validshape " + tile_buf + ", " + row + ", " + col + " : " + tile_type);
}

static std::string FormatFrontendPipeAttrs(const CallPtr& op, int split) {
  std::ostringstream oss;
  oss << "{";
  if (op->HasKwarg("id")) {
    const int id = op->GetKwarg<int>("id", 0);
    CHECK(id >= 0) << "Frontend pipe 'id' attribute must be non-negative, got " << id;
    oss << "id = " << id << ", ";
  }
  oss << "split = " << split << "}";
  return oss.str();
}

static std::string FormatFrontendPipeAttrs(std::optional<int> pipe_id, int split) {
  std::ostringstream oss;
  oss << "{";
  if (pipe_id.has_value()) {
    CHECK(pipe_id.value() >= 0) << "Frontend pipe 'id' attribute must be non-negative, got "
                                << pipe_id.value();
    oss << "id = " << pipe_id.value() << ", ";
  }
  oss << "split = " << split << "}";
  return oss.str();
}

static std::string FormatInitializePipeAttrs(const CallPtr& op, int dir_mask, int slot_size) {
  std::ostringstream oss;
  oss << "{";
  if (op->HasKwarg("id")) {
    const int id = op->GetKwarg<int>("id", 0);
    CHECK(id >= 0) << "Frontend initialize_pipe 'id' attribute must be non-negative, got " << id;
    oss << "id = " << id << ", ";
  }
  oss << "dir_mask = " << dir_mask << ", slot_size = " << slot_size << "}";
  return oss.str();
}

// tile.tpush_to_aiv: Push tile from Cube to Vector
static std::string MakeTpushToAivCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 1) << "tpush_to_aiv requires 1 argument (tile), got " << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(tile, op->span_) << "tpush_to_aiv first argument must be a Var or IterArg";

  const int split = op->GetKwarg<int>("split", -1);
  CHECK(split >= 0 && split <= 2)
      << "tpush_to_aiv requires 'split' attribute (0=none, 1=up-down, 2=left-right), got " << split;

  std::string tile_buf = codegen.GetExprAsCode(op->args_[0]);
  std::string tile_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  const bool restore_valid_shape = EmitSplitTpushTransportValidShape(op, codegen, tile_buf, tile_type, split);

  std::ostringstream oss;
  oss << "pto.tpush_to_aiv(" << tile_buf;
  if (!tile_type.empty()) {
    oss << " : " << tile_type;
  }
  oss << ") " << FormatFrontendPipeAttrs(op, split);
  codegen.Emit(oss.str());
  if (restore_valid_shape) {
    EmitLogicalTpushValidShapeRestore(op, codegen, tile_buf, tile_type);
  }

  return "";
}

// tile.tpush_to_aic: Push tile from Vector to Cube
static std::string MakeTpushToAicCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 1) << "tpush_to_aic requires 1 argument (tile), got " << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(tile, op->span_) << "tpush_to_aic first argument must be a Var or IterArg";

  const int split = op->GetKwarg<int>("split", -1);
  CHECK(split >= 0 && split <= 2)
      << "tpush_to_aic requires 'split' attribute (0=none, 1=up-down, 2=left-right), got " << split;

  std::string tile_buf = codegen.GetExprAsCode(op->args_[0]);
  std::string tile_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  const bool restore_valid_shape = EmitSplitTpushTransportValidShape(op, codegen, tile_buf, tile_type, split);

  std::ostringstream oss;
  oss << "pto.tpush_to_aic(" << tile_buf;
  if (!tile_type.empty()) {
    oss << " : " << tile_type;
  }
  oss << ") " << FormatFrontendPipeAttrs(op, split);
  codegen.Emit(oss.str());
  if (restore_valid_shape) {
    EmitLogicalTpushValidShapeRestore(op, codegen, tile_buf, tile_type);
  }

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
  INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_)
      << "tpop_from_aic requires assignment target (tile_buf)";
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  auto [valid_row, valid_col] = codegen.GetCurrentResultTpopValidShapeOperands();

  std::ostringstream oss;
  oss << result_buf << " = pto.tpop_from_aic";
  if (!valid_row.empty() || !valid_col.empty()) {
    INTERNAL_CHECK_SPAN(!valid_row.empty() && !valid_col.empty(), op->span_)
        << "Internal error: tpop_from_aic dynamic valid_shape requires both valid_row and valid_col";
    oss << "(" << valid_row << ", " << valid_col << ")";
  }
  oss << " " << FormatFrontendPipeAttrs(op, split);
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
  INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_)
      << "tpop_from_aiv requires assignment target (tile_buf)";
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  auto [valid_row, valid_col] = codegen.GetCurrentResultTpopValidShapeOperands();

  std::ostringstream oss;
  oss << result_buf << " = pto.tpop_from_aiv";
  if (!valid_row.empty() || !valid_col.empty()) {
    INTERNAL_CHECK_SPAN(!valid_row.empty() && !valid_col.empty(), op->span_)
        << "Internal error: tpop_from_aiv dynamic valid_shape requires both valid_row and valid_col";
    oss << "(" << valid_row << ", " << valid_col << ")";
  }
  oss << " " << FormatFrontendPipeAttrs(op, split);
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
  INTERNAL_CHECK_SPAN(tile, op->span_) << "tfree_to_aic first argument must be a Var or IterArg";
  const auto& tpop_info =
      codegen.GetValidatedTpopInfo(tile.get(), "tile.tpop_from_aic", "system.tfree_to_aic");
  if (op->HasKwarg("id")) {
    const int tfree_id = op->GetKwarg<int>("id", 0);
    const int tpop_id = tpop_info.pipe_id.value_or(0);
    CHECK(tpop_id == tfree_id) << "system.tfree_to_aic pipe id " << tfree_id
                               << " does not match originating tile.tpop_from_aic pipe id " << tpop_id;
  }
  const std::optional<int> pipe_id =
      op->HasKwarg("id") ? std::optional<int>(op->GetKwarg<int>("id", 0)) : tpop_info.pipe_id;

  std::ostringstream oss;
  oss << "pto.tfree_from_aic " << FormatFrontendPipeAttrs(pipe_id, tpop_info.split);
  codegen.Emit(oss.str());

  return "";
}

// tfree codegen for system.tfree_to_aiv: emits pto.tfree_from_aiv {split = N}
static std::string MakeTfreeToAivCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 1) << "tfree_to_aiv requires 1 argument (tile from tpop), got "
                               << op->args_.size();
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(tile, op->span_) << "tfree_to_aiv first argument must be a Var or IterArg";

  const auto& tpop_info =
      codegen.GetValidatedTpopInfo(tile.get(), "tile.tpop_from_aiv", "system.tfree_to_aiv");
  if (op->HasKwarg("id")) {
    const int tfree_id = op->GetKwarg<int>("id", 0);
    const int tpop_id = tpop_info.pipe_id.value_or(0);
    CHECK(tpop_id == tfree_id) << "system.tfree_to_aiv pipe id " << tfree_id
                               << " does not match originating tile.tpop_from_aiv pipe id " << tpop_id;
  }
  const std::optional<int> pipe_id =
      op->HasKwarg("id") ? std::optional<int>(op->GetKwarg<int>("id", 0)) : tpop_info.pipe_id;

  std::ostringstream oss;
  oss << "pto.tfree_from_aiv " << FormatFrontendPipeAttrs(pipe_id, tpop_info.split);
  codegen.Emit(oss.str());

  return "";
}

static bool ExprIsI32Scalar(const ir::ExprPtr& expr) {
  if (auto st = As<ScalarType>(expr->GetType())) {
    return st->dtype_ == DataType::INT32;
  }
  return false;
}

// Pipe buffer operands are i32 SSA. GetExprAsCode(ConstInt) uses index constants; use i32 here.
static std::string GetPipeBufOperandI32SSA(codegen::PTOCodegen& codegen, const ir::ExprPtr& expr) {
  if (auto c = As<ir::ConstInt>(expr)) {
    return codegen.GetOrEmitConstant(static_cast<int64_t>(static_cast<int32_t>(c->value_)), DataType::INT32);
  }
  INTERNAL_CHECK_SPAN(ExprIsI32Scalar(expr), expr->span_)
      << "Initialize-pipe buffer operand must be INT32 scalar SSA or integral ConstInt placeholder";
  return codegen.GetExprAsCode(expr);
}

// Helper to format initialize_pipe operand list
static void EmitInitializePipeOperands(std::ostringstream& oss, const std::string& gm_ssa,
                                       const std::string& c2v_ssa, const std::string& v2c_ssa) {
  if (!gm_ssa.empty()) {
    oss << "\n      (gm_slot_buffer = " << gm_ssa << " : !pto.ptr<f32>"
        << ", c2v_consumer_buf = " << c2v_ssa << " : i32"
        << ", v2c_consumer_buf = " << v2c_ssa << " : i32)";
  } else {
    oss << " (c2v_consumer_buf = " << c2v_ssa << " : i32"
        << ", v2c_consumer_buf = " << v2c_ssa << " : i32)";
  }
}

// system.aic_initialize_pipe: Initialize cross-core pipe on Cube side
static std::string MakeAicInitializePipeCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 2)
      << "aic_initialize_pipe requires 2 arguments (c2v_consumer_buf, v2c_consumer_buf), got "
      << op->args_.size();
  const int dir_mask = op->GetKwarg<int>("dir_mask", -1);
  const int slot_size = op->GetKwarg<int>("slot_size", -1);
  CHECK(dir_mask >= 0) << "aic_initialize_pipe requires 'dir_mask' attribute";
  CHECK(slot_size > 0) << "aic_initialize_pipe requires 'slot_size' attribute";

  // AIC (Cube): operands are explicit i32 SSAs (validated by MixedKernelExpanded verifier).
  std::string c2v_ssa = GetPipeBufOperandI32SSA(codegen, op->args_[0]);
  std::string v2c_ssa = GetPipeBufOperandI32SSA(codegen, op->args_[1]);
  CHECK(!c2v_ssa.empty() && !v2c_ssa.empty())
      << "aic_initialize_pipe: failed to lower buffer operands to SSA names";

  std::ostringstream oss;
  oss << "pto.aic_initialize_pipe " << FormatInitializePipeAttrs(op, dir_mask, slot_size);
  const int pipe_id = op->GetKwarg<int>("id", 0);
  EmitInitializePipeOperands(oss, codegen.GetGMSlotBufferSSAForPipe(pipe_id, dir_mask), c2v_ssa, v2c_ssa);
  codegen.Emit(oss.str());

  return "";
}

// system.aiv_initialize_pipe: Initialize cross-core pipe on Vector side
static std::string MakeAivInitializePipeCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  CHECK(op->args_.size() == 2)
      << "aiv_initialize_pipe requires 2 arguments (c2v_consumer_buf, v2c_consumer_buf), got "
      << op->args_.size();
  const int dir_mask = op->GetKwarg<int>("dir_mask", -1);
  const int slot_size = op->GetKwarg<int>("slot_size", -1);
  CHECK(dir_mask >= 0) << "aiv_initialize_pipe requires 'dir_mask' attribute";
  CHECK(slot_size > 0) << "aiv_initialize_pipe requires 'slot_size' attribute";

  std::string c2v_ssa = GetPipeBufOperandI32SSA(codegen, op->args_[0]);
  std::string v2c_ssa = GetPipeBufOperandI32SSA(codegen, op->args_[1]);
  CHECK(!c2v_ssa.empty() && !v2c_ssa.empty())
      << "aiv_initialize_pipe: failed to lower buffer operands to SSA names";

  std::ostringstream oss;
  oss << "pto.aiv_initialize_pipe " << FormatInitializePipeAttrs(op, dir_mask, slot_size);
  const int pipe_id = op->GetKwarg<int>("id", 0);
  EmitInitializePipeOperands(oss, codegen.GetGMSlotBufferSSAForPipe(pipe_id, dir_mask), c2v_ssa, v2c_ssa);
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
    // tile.rsqrt is registered with a custom codegen handler below (supports 1 or 2 args).
    {"tile.recip",           "pto.trecip",           1},
    {"tile.neg",             "pto.tneg",             1},
    {"tile.not",             "pto.tnot",             1},
    {"tile.relu",            "pto.trelu",            1},
    // Ternary operations (tile x tile + carry/select)
    {"tile.addc",            "pto.taddc",            3},
    {"tile.subc",            "pto.tsubc",            3},
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
    {"tile.maximums",        "pto.tmaxs",            2},
    {"tile.minimums",        "pto.tmins",            2},
    {"tile.lrelu",           "pto.tlrelu",           2},
    // Ternary scalar operations (tile x scalar + carry/select)
    {"tile.addsc",           "pto.taddsc",           3},
    {"tile.subsc",           "pto.tsubsc",           3},
    {"tile.selc",            "pto.tselc",            3},
    // Axis reduction/expansion operations
    {"tile.row_sum",         "pto.trowsum",          2},
    {"tile.row_max",         "pto.trowmax",          2},
    {"tile.row_min",         "pto.trowmin",          2},
    {"tile.col_max",         "pto.tcolmax",          1},
    {"tile.col_min",         "pto.tcolmin",          1},
    {"tile.col_expand_mul",  "pto.tcolexpandmul",    2},
    {"tile.row_expand_add",  "pto.trowexpandadd",    2},
    {"tile.row_expand_div",  "pto.trowexpanddiv",    2},
    {"tile.row_expand_mul",  "pto.trowexpandmul",    2},
    {"tile.row_expand_sub",  "pto.trowexpandsub",    2},
    // Padding operations
    {"tile.fillpad",         "pto.tfillpad",         1},
    // Inplace variant: set_output_reuses_input(0) makes src/dst share UB addr.
    {"tile.fillpad_inplace", "pto.tfillpad",         1},
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
    // tile.move has custom codegen (no-op elision for same-space same-address moves)
    {"tile.move_fp",         "pto.tmov.fp",          2},
    // tile.transpose has custom codegen (MakeTileTransposeCodegenPTO): pto.ttrans needs
    // ins(%src, %tmp : tile_type, tile_type) where %tmp is a scratch workspace tile, NOT
    // the axis-index integers that tile.transpose(src, axis0, axis1) carries in the IR.
    // tile.extract has custom codegen (see reg("tile.extract") below): the IR carries the
    // shape tuple as args_[3] purely for type deduction, so the generic N-ary lowering
    // would emit the tuple as a PTO operand — not what pto.textract expects.
    // Gather/scatter operations
    {"tile.gather",          "pto.tgather",          3},
    {"tile.gatherb",         "pto.tgatherb",         2},
    // tile.scatter and tile.scatter_mask are registered with custom codegen
    // handlers below (DPS — dst is `args_[0]`, aliased to the result via
    // set_output_reuses_input(0)).
    // Partial reduction operations
    {"tile.partadd",         "pto.tpartadd",         2},
    {"tile.partmax",         "pto.tpartmax",         2},
    {"tile.partmin",         "pto.tpartmin",         2},
};
// clang-format on

// ============================================================================
// Distributed N6: pld.tile.remote_load / pld.system.notify / pld.system.wait
// ============================================================================

namespace {

// Lower a tile-op offsets/shape MakeTuple to a vector of MLIR SSA strings (one
// per dimension). Constants are materialised via GetOrEmitConstant; dynamic
// dims fall back to GetExprAsCode + EmitCastToIndex when needed.
std::vector<std::string> LowerTupleToIndexSSA(const ir::MakeTuplePtr& tuple, codegen::PTOCodegen& codegen) {
  std::vector<std::string> ssa_names;
  ssa_names.reserve(tuple->elements_.size());
  for (const auto& elem : tuple->elements_) {
    if (auto ci = As<ir::ConstInt>(elem)) {
      ssa_names.push_back(codegen.GetOrEmitConstant(ci->value_, DataType::INDEX));
    } else {
      ssa_names.push_back(codegen.EmitCastToIndex(elem, codegen.GetExprAsCode(elem)));
    }
  }
  return ssa_names;
}

// Resolve a DistributedTensor argument to its parameter Var + matching
// CommContext SSA. The argument is expected to be a Var directly bound to a
// function parameter (no aliasing); the verifier on remote_load / notify /
// wait already requires DistributedTensorType, but additionally checks that
// the var has a CommContext ptr threaded through the func.func signature.
struct DistTensorBinding {
  ir::VarPtr var;
  std::shared_ptr<const ir::DistributedTensorType> type;
  std::string local_ptr_ssa;
  std::string ctx_ssa;
};

DistTensorBinding ResolveDistTensorBinding(const ExprPtr& arg, codegen::PTOCodegen& codegen,
                                           const char* op_name) {
  auto var = AsVarLike(arg);
  CHECK(var) << op_name << " expects DistributedTensor argument to be a Var-like expression, got "
             << arg->TypeName();
  auto dist_type = As<ir::DistributedTensorType>(var->GetType());
  CHECK(dist_type) << op_name << " expects DistributedTensorType, got " << var->GetType()->TypeName();
  std::string local_ptr = codegen.GetVarName(var);
  std::string ctx_ssa = codegen.GetCommCtxSSAFor(var.get());
  CHECK(!ctx_ssa.empty()) << op_name << " requires a CommContext pointer arg threaded for DistributedTensor '"
                          << var->name_hint_ << "', but none was found in the function signature";
  return {var, dist_type, std::move(local_ptr), std::move(ctx_ssa)};
}

// Emit:
//   (1) a single ``func.call`` to the per-dtype module-level
//       ``@CommRemoteOffset_<dtype>`` helper (see
//       ``PTOCodegen::EmitCommRemoteOffsetHelpers``) — returns the
//       peer-vs-local **element offset** (``index``);
//   (2) a ``pto.addptr`` against the local DistributedTensor pointer, and
//   (3) a ``pto.make_tensor_view`` rooted at the resulting peer pointer.
//
// Steps (2) and (3) live at the call site (i.e. inside the user kernel's
// ``func.func``) for two intertwined PTOAS constraints:
//
// * ``pto.addptr`` must feed ``pto.make_tensor_view`` /
//   ``initialize_l2g2l_pipe(gm_addr)`` / ``load|store_scalar`` *within
//   the same func.func*. A helper that ended with ``addptr → return``
//   would only feed ``func.return``, which PTOAS rejects.
// * ``pto.make_tensor_view`` always lowers to ``memref<…, strided<[?,
//   ?], offset: ?>>`` when strides are passed as operands, but
//   ``!pto.tensor_view<…>`` source syntax cannot carry a strided layout
//   suffix — so the view cannot be returned across a func boundary
//   either.
//
// Both forbidden ops therefore have to live in the user kernel. The
// helper still pulls its weight: it bundles the CommContext field reads
// and the byte→element division (which depends on dtype), so multiple
// remote ops share that work via ``func.call`` without duplicating the
// scalar arithmetic at each call site.
//
// Generated MLIR (2-D example, ``DistributedTensor[[1, 64], FP32]``):
//
//   %peer_idx = arith.index_cast %peer : i32 to index
//   %delems = func.call @CommRemoteOffset_f32(%ctx, %peer_idx)
//           : (!pto.ptr<i64>, index) -> index
//   %peer_ptr = pto.addptr %local_ptr, %delems
//             : !pto.ptr<f32> -> !pto.ptr<f32>
//   %peer_view = pto.make_tensor_view %peer_ptr,
//                   shape = [%c1, %c64], strides = [%c64, %c1]
//                   {layout = #pto.layout<nd>}
//                   : !pto.tensor_view<?x?xf32>
struct PeerViewInfo {
  std::string ssa;
  std::string view_type_str;
};

PeerViewInfo EmitCommRemoteView(const DistTensorBinding& target, const ExprPtr& peer_expr,
                                codegen::PTOCodegen& codegen) {
  const auto& shape = target.type->shape_;
  const size_t rank = shape.size();
  CHECK(rank >= 1) << "DistributedTensor must have rank >= 1 for peer view emission";
  const std::string dtype_str = codegen.GetTypeString(target.type->dtype_);
  const std::string ptr_type = "!pto.ptr<" + dtype_str + ">";

  // Peer rank may be any scalar int; the helper takes it as ``index``, so
  // normalise here. Constants and i32/i64 values flow through
  // EmitCastToIndex (no-op when already index-typed).
  std::string peer_ssa = codegen.EmitCastToIndex(peer_expr, codegen.GetExprAsCode(peer_expr));

  // (1) Call the per-dtype offset helper. Registering here causes the helper
  //     definition to be emitted at module-flush time — any new op that calls
  //     EmitCommRemoteView is wired up automatically, no codegen-side opt-in.
  const std::string func_name = codegen.RegisterCommRemoteOffsetHelper(target.type->dtype_);
  std::string delems = codegen.NewTemp();
  codegen.Emit(delems + " = func.call @" + func_name + "(" + target.ctx_ssa + ", " + peer_ssa +
               ") : (!pto.ptr<i64>, index) -> index");

  // (2) addptr from the local pointer by the returned element offset.
  std::string peer_ptr = codegen.NewTemp();
  codegen.Emit(peer_ptr + " = pto.addptr " + target.local_ptr_ssa + ", " + delems + " : " + ptr_type +
               " -> " + ptr_type);

  // (3) make_tensor_view at the call site. Same shape/stride emission as
  // ``EmitMakeTensorViews``: row-major strides, ``{layout = #pto.layout<nd>}``
  // attribute, dynamic-shape result type (``?x?x…xT``). ``addptr``'s
  // direct consumer is this ``make_tensor_view`` in the same func →
  // PTOAS's per-func lowering rule is satisfied.
  std::vector<std::string> shape_ssa(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (auto ci = As<ir::ConstInt>(shape[i])) {
      shape_ssa[i] = codegen.GetOrEmitConstant(ci->value_, DataType::INDEX);
    } else {
      shape_ssa[i] = codegen.EmitCastToIndex(shape[i], codegen.GetExprAsCode(shape[i]));
    }
  }
  std::vector<std::string> stride_ssa(rank);
  stride_ssa[rank - 1] = codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
  for (size_t j = rank - 1; j > 0; --j) {
    std::string mul = codegen.NewTemp();
    codegen.Emit(mul + " = arith.muli " + stride_ssa[j] + ", " + shape_ssa[j] + " : index");
    stride_ssa[j - 1] = mul;
  }

  std::string peer_view = codegen.NewTemp();
  std::ostringstream view_type;
  view_type << "!pto.tensor_view<";
  for (size_t i = 0; i < rank; ++i) {
    if (i > 0) view_type << "x";
    view_type << "?";
  }
  view_type << "x" << dtype_str << ">";

  std::ostringstream mv;
  mv << peer_view << " = pto.make_tensor_view " << peer_ptr << ", shape = [";
  for (size_t i = 0; i < rank; ++i) {
    if (i > 0) mv << ", ";
    mv << shape_ssa[i];
  }
  mv << "], strides = [";
  for (size_t i = 0; i < rank; ++i) {
    if (i > 0) mv << ", ";
    mv << stride_ssa[i];
  }
  mv << "] {layout = #pto.layout<nd>} : " << view_type.str();
  codegen.Emit(mv.str());

  return {peer_view, view_type.str()};
}

}  // namespace

// pld.tile.remote_load(target, peer, offsets, shape) — load a peer's slice of
// a window-bound DistributedTensor into a local tile. Lowers to:
//   delems = func.call @CommRemoteOffset_<dtype>(ctx, peer) : ... -> index
//   peer_ptr = pto.addptr local_ptr, delems
//   peer_view = pto.make_tensor_view peer_ptr, shape=..., strides=...
//   pto.partition_view peer_view, offsets=..., sizes=<shape>
//   pto.tload ins(<pview>) outs(<tile>)
static std::string MakeRemoteLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "pld.tile.remote_load requires 4 arguments (target, peer, offsets, "
                                  "shape), got "
                               << op->args_.size();

  auto binding = ResolveDistTensorBinding(op->args_[0], codegen, "pld.tile.remote_load");
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "pld.tile.remote_load offsets must be MakeTuple";
  auto shapes_tuple = As<ir::MakeTuple>(op->args_[3]);
  INTERNAL_CHECK_SPAN(shapes_tuple, op->span_) << "pld.tile.remote_load shape must be MakeTuple";

  auto peer_view = EmitCommRemoteView(binding, op->args_[1], codegen);

  const std::string dtype_str = codegen.GetTypeString(binding.type->dtype_);
  const auto& offset_elems = offsets_tuple->elements_;
  const auto& shape_elems = shapes_tuple->elements_;
  std::string partition_type = MakePartitionTensorViewType(GetDimStrings(shape_elems), dtype_str);
  std::string partition_view = EmitPartitionViewPTO(
      binding.var->name_hint_ + "_peer", peer_view.ssa, peer_view.view_type_str, partition_type,
      GetExprCodes(offset_elems, codegen), GetSizeCodes(shape_elems, codegen), codegen);

  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK_SPAN(!tile_buf.empty(), op->span_)
      << "pld.tile.remote_load requires assignment target (tile_buf)";
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream tload;
  tload << "pto.tload ins(" << partition_view << " : " << partition_type << ") outs(" << tile_buf << " : "
        << tile_buf_type << ")";
  codegen.Emit(tload.str());
  return "";
}

// pld.system.notify(target, peer, offsets, value, *, op) — atomically signal a
// peer rank's slot in a DistributedTensor signal matrix.
//   delems = func.call @CommRemoteOffset_<dtype>(ctx, peer) : ... -> index
//   peer_ptr = pto.addptr local_ptr, delems
//   peer_view = pto.make_tensor_view peer_ptr, shape=..., strides=...
//   pto.partition_view peer_view, sizes=[1, ..., 1]
//   pto.comm.tnotify(<pview>, <value>) {notifyOp = #pto<notify_op (set|atomic_add)>}
static std::string MakeNotifyCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "pld.system.notify requires 4 arguments (target, peer, offsets, "
                                  "value), got "
                               << op->args_.size();

  auto binding = ResolveDistTensorBinding(op->args_[0], codegen, "pld.system.notify");
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "pld.system.notify offsets must be MakeTuple";

  const int notify_op_int = op->GetKwarg<int>("op", 0);
  CHECK(notify_op_int == static_cast<int>(ir::NotifyOp::kAtomicAdd) ||
        notify_op_int == static_cast<int>(ir::NotifyOp::kSet))
      << "pld.system.notify op kwarg must encode NotifyOp::kAtomicAdd or kSet, got " << notify_op_int;
  const std::string notify_attr =
      notify_op_int == static_cast<int>(ir::NotifyOp::kAtomicAdd) ? "atomic_add" : "set";

  auto peer_view = EmitCommRemoteView(binding, op->args_[1], codegen);

  // Notify slot is a single signal cell — partition_view sizes are 1 per dim.
  const std::string dtype_str = codegen.GetTypeString(binding.type->dtype_);
  const size_t rank = binding.type->shape_.size();
  std::vector<std::string> one_dims(rank, "1");
  std::vector<std::string> one_size_ssa(rank,
                                        codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX));
  std::string partition_type = MakePartitionTensorViewType(one_dims, dtype_str);
  std::string partition_view = EmitPartitionViewPTO(
      binding.var->name_hint_ + "_peer", peer_view.ssa, peer_view.view_type_str, partition_type,
      GetExprCodes(offsets_tuple->elements_, codegen), one_size_ssa, codegen);

  // PTOAS contract: tnotify value's MLIR type must match the signal element
  // type. Emit using the value's own ScalarType — mismatched IR-level dtypes
  // surface here as a PTOAS verifier diagnostic rather than as silently
  // garbled DMA.
  std::string value_ssa = codegen.GetExprAsCode(op->args_[3]);
  auto value_scalar = As<ir::ScalarType>(op->args_[3]->GetType());
  CHECK(value_scalar) << "pld.system.notify value must have ScalarType, got "
                      << op->args_[3]->GetType()->TypeName();
  std::string value_type = codegen.GetTypeString(value_scalar->dtype_);
  std::ostringstream tnotify;
  tnotify << "pto.comm.tnotify(" << partition_view << ", " << value_ssa << " : " << partition_type << ", "
          << value_type << ") {notifyOp = #pto<notify_op " << notify_attr << ">}";
  codegen.Emit(tnotify.str());
  return "";
}

// pld.system.wait(signal, offsets, expected, *, cmp) — block until local signal
// slot satisfies cmp against expected. wait is local (no peer arithmetic):
//   pto.partition_view <local_view>, offsets=..., sizes=[1,..,1]
//   pto.comm.twait(<pview>, <expected>) {cmp = #pto<wait_cmp (eq|ge)>}
static std::string MakeWaitCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "pld.system.wait requires 3 arguments (signal, offsets, expected), got "
                               << op->args_.size();

  auto signal_var = AsVarLike(op->args_[0]);
  CHECK(signal_var) << "pld.system.wait signal must be a Var-like expression";
  auto dist_type = As<ir::DistributedTensorType>(signal_var->GetType());
  CHECK(dist_type) << "pld.system.wait signal must be DistributedTensorType, got "
                   << signal_var->GetType()->TypeName();

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "pld.system.wait offsets must be MakeTuple";

  const int cmp_int = op->GetKwarg<int>("cmp", 0);
  CHECK(cmp_int == static_cast<int>(ir::WaitCmp::kEq) || cmp_int == static_cast<int>(ir::WaitCmp::kGe))
      << "pld.system.wait cmp kwarg must encode WaitCmp::kEq or kGe, got " << cmp_int;
  const std::string cmp_attr = cmp_int == static_cast<int>(ir::WaitCmp::kEq) ? "eq" : "ge";

  // Reuse the local tensor_view created by EmitMakeTensorViews — wait only
  // touches the local signal slot. The view's MLIR type must match the
  // emit-time form (all dims printed as ``?``), not the IR-level concrete
  // shape, otherwise the SSA value picks up two incompatible types when other
  // uses (tile.load, etc.) reference the same view. Mirrors tile.load at
  // line 1192 above.
  std::string local_view = codegen.GetOrCreateTensorView(signal_var);
  std::string local_view_type = codegen.GetTensorViewTypeString(dist_type.get());
  const std::string dtype_str = codegen.GetTypeString(dist_type->dtype_);
  const size_t rank = dist_type->shape_.size();

  std::vector<std::string> one_dims(rank, "1");
  std::vector<std::string> one_size_ssa(rank,
                                        codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX));
  std::string partition_type = MakePartitionTensorViewType(one_dims, dtype_str);
  std::string partition_view =
      EmitPartitionViewPTO(signal_var->name_hint_ + "_local", local_view, local_view_type, partition_type,
                           GetExprCodes(offsets_tuple->elements_, codegen), one_size_ssa, codegen);

  // PTOAS contract: twait expected value's MLIR type must match the signal
  // element type. Emit using the expected value's own ScalarType — see notify
  // codegen above for the rationale.
  std::string expected_ssa = codegen.GetExprAsCode(op->args_[2]);
  auto expected_scalar = As<ir::ScalarType>(op->args_[2]->GetType());
  CHECK(expected_scalar) << "pld.system.wait expected must have ScalarType, got "
                         << op->args_[2]->GetType()->TypeName();
  std::string expected_type = codegen.GetTypeString(expected_scalar->dtype_);
  std::ostringstream twait;
  twait << "pto.comm.twait(" << partition_view << ", " << expected_ssa << " : " << partition_type << ", "
        << expected_type << ") {cmp = #pto<wait_cmp " << cmp_attr << ">}";
  codegen.Emit(twait.str());
  return "";
}

// pld.tile.put(dst, peer, src, stage[, dst_offsets, src_offsets, shape],
//              *, atomic) - synchronous cross-rank bulk write of the local
// slice `src` into the peer rank's slice of `dst`. `stage` is a VEC scratch
// TileType pre-allocated by an IR-level `tile.create` (so the memory allocator
// gives it a UB address before codegen at --pto-level=level3).
// Lowers to:
//   delems   = func.call @CommRemoteOffset_<dtype>(ctx, peer) : ... -> index
//   dst_ptr  = pto.addptr <dst_local_ptr>, delems
//   dst_view = pto.make_tensor_view dst_ptr, shape=..., strides=...
//   dst_pv   = pto.partition_view dst_view, offsets=<dst_offsets>, sizes=<transfer_shape>
//   src_pv   = pto.partition_view <src_local_view>, offsets=<src_offsets>, sizes=<transfer_shape>
//   pto.comm.tput(dst_pv, src_pv, buf(%stage)
//       : <ptype>, <ptype>, <stage_type>) {atomicType = #pto<atomic_type (atomic_none|atomic_add)>}
//
// Full-slice tile.put (4 args) uses zero offsets and the full dst/src shape.
// Subregion tile.put (7 args) uses the explicit offsets and transfer shape that
// ConvertTensorToTileOps forwarded from user-facing pld.tensor.put. PTOAS
// validates the stage tile type against the partition views, so the IR verifier
// also checks that stage elements equal prod(transfer_shape).
static std::string MakePutCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 4 || op->args_.size() == 7)
      << "pld.tile.put requires 4 arguments (dst, peer, src, stage) or 7 arguments "
         "(dst, peer, src, stage, dst_offsets, src_offsets, shape), got "
      << op->args_.size();

  // dst: remote (peer-addressed) DistributedTensor destination.
  auto dst_binding = ResolveDistTensorBinding(op->args_[0], codegen, "pld.tile.put");

  // src: local DistributedTensor source — reuses the local tensor_view created
  // by EmitMakeTensorViews (no peer arithmetic, like wait's signal).
  auto src_var = AsVarLike(op->args_[2]);
  CHECK(src_var) << "pld.tile.put src must be a Var-like expression";
  auto src_dist = As<ir::DistributedTensorType>(src_var->GetType());
  CHECK(src_dist) << "pld.tile.put src must be DistributedTensorType, got " << src_var->GetType()->TypeName();

  const int atomic_int = op->GetKwarg<int>("atomic", 0);
  CHECK(atomic_int == static_cast<int>(ir::AtomicType::kNone) ||
        atomic_int == static_cast<int>(ir::AtomicType::kAdd))
      << "pld.tile.put atomic kwarg must encode AtomicType::kNone or kAdd, got " << atomic_int;
  const std::string atomic_attr =
      atomic_int == static_cast<int>(ir::AtomicType::kAdd) ? "atomic_add" : "atomic_none";

  const auto& shape = dst_binding.type->shape_;
  const size_t rank = shape.size();
  INTERNAL_CHECK_SPAN(rank >= 1, op->span_) << "pld.tile.put requires rank >= 1";
  const std::string dtype_str = codegen.GetTypeString(dst_binding.type->dtype_);

  std::vector<std::string> dst_offsets;
  std::vector<std::string> src_offsets;
  std::vector<std::string> size_ssa;
  std::vector<ExprPtr> transfer_shape;

  if (op->args_.size() == 4) {
    // Full-slice partition views: offsets all-zero, sizes = full shape. dst and
    // src share the same partition_tensor_view type (same dtype + static shape).
    std::string c0 = codegen.GetOrEmitConstant(static_cast<int64_t>(0), DataType::INDEX);
    dst_offsets.assign(rank, c0);
    src_offsets.assign(rank, c0);
    transfer_shape = shape;
    size_ssa = GetSizeCodes(transfer_shape, codegen);
  } else {
    // Subregion partition views: user-facing tensor.put supplied the two
    // offset tuples plus a shared static transfer shape. The explicit stage
    // tile was sized to this transfer shape by ConvertTensorToTileOps.
    auto dst_offsets_tuple = As<ir::MakeTuple>(op->args_[4]);
    auto src_offsets_tuple = As<ir::MakeTuple>(op->args_[5]);
    auto shape_tuple = As<ir::MakeTuple>(op->args_[6]);
    INTERNAL_CHECK_SPAN(dst_offsets_tuple, op->span_) << "pld.tile.put dst_offsets must be MakeTuple";
    INTERNAL_CHECK_SPAN(src_offsets_tuple, op->span_) << "pld.tile.put src_offsets must be MakeTuple";
    INTERNAL_CHECK_SPAN(shape_tuple, op->span_) << "pld.tile.put shape must be MakeTuple";
    INTERNAL_CHECK_SPAN(dst_offsets_tuple->elements_.size() == rank, op->span_)
        << "pld.tile.put dst_offsets rank must match tensor rank";
    INTERNAL_CHECK_SPAN(src_offsets_tuple->elements_.size() == rank, op->span_)
        << "pld.tile.put src_offsets rank must match tensor rank";
    INTERNAL_CHECK_SPAN(shape_tuple->elements_.size() == rank, op->span_)
        << "pld.tile.put shape rank must match tensor rank";
    dst_offsets = GetExprCodes(dst_offsets_tuple->elements_, codegen);
    src_offsets = GetExprCodes(src_offsets_tuple->elements_, codegen);
    transfer_shape = shape_tuple->elements_;
    size_ssa = GetSizeCodes(transfer_shape, codegen);
  }

  std::string partition_type = MakePartitionTensorViewType(GetDimStrings(transfer_shape), dtype_str);

  // dst: CommRemoteOffset + addptr + make_tensor_view at the call site, then
  // a full-slice or subregion partition_view.
  auto dst_peer_view = EmitCommRemoteView(dst_binding, op->args_[1], codegen);
  std::string dst_pview =
      EmitPartitionViewPTO(dst_binding.var->name_hint_ + "_peer", dst_peer_view.ssa,
                           dst_peer_view.view_type_str, partition_type, dst_offsets, size_ssa, codegen);

  // src: local tensor_view + full-slice or subregion partition_view (no peer arithmetic).
  // Use the shared helper for the source view type so it matches the dynamic-dim
  // tensor_view SSA that GetOrCreateTensorView emits (mirroring dst's peer view
  // and every other tensor-view op in this file); a hand-rolled static-shape
  // string would mismatch that SSA's type.
  std::string src_local_view = codegen.GetOrCreateTensorView(src_var);
  std::string src_view_type = codegen.GetTensorViewTypeString(src_dist.get());
  std::string src_pview = EmitPartitionViewPTO(src_var->name_hint_ + "_local", src_local_view, src_view_type,
                                               partition_type, src_offsets, size_ssa, codegen);

  std::string stage = codegen.GetExprAsCode(op->args_[3]);
  std::string stage_type = codegen.GetExprTypeAnnotation(op->args_[3]);
  INTERNAL_CHECK_SPAN(!stage_type.empty(), op->span_)
      << "Internal error: pld.tile.put stage tile " << stage << " has no tile_buf type annotation";

  // The VEC staging tile is not synthesized here: pld.tensor.put has already
  // been lowered to tile.create + pld.tile.put so the allocator can assign the
  // stage tile a real UB address before this PTO emission step.

  // TPUT reads the local source GM through MTE2. If the caller populated that
  // source via an immediately preceding TSTORE, order the store before TPUT's
  // source read; otherwise one rank can observe stale zeros while another wins
  // the timing race.
  codegen.Emit("pto.barrier <PIPE_ALL>");

  std::ostringstream tput;
  tput << "pto.comm.tput(" << dst_pview << ", " << src_pview << ", buf(" << stage << ") : " << partition_type
       << ", " << partition_type << ", " << stage_type << ") {atomicType = #pto<atomic_type " << atomic_attr
       << ">}";
  codegen.Emit(tput.str());
  return "";
}

// pld.tensor.get(dst, peer, src) — synchronous cross-rank bulk read from the
// peer rank's slice of `src` into the local slice `dst`. dst and src share
// dtype and (verified) static shape. Lowers to:
//   delems   = func.call @CommRemoteOffset_<dtype>(ctx, peer) : ... -> index
//   src_ptr  = pto.addptr <src_local_ptr>, delems
//   src_view = pto.make_tensor_view src_ptr, shape=..., strides=...
//   src_pv   = pto.partition_view src_view,  offsets=[0,..], sizes=<shape>
//   dst_pv   = pto.partition_view <dst_local_view>, offsets=[0,..], sizes=<shape>
//   %stage   = pto.alloc_tile : !pto.tile_buf<loc=vec, ...>
//   pto.comm.tget(dst_pv, src_pv, buf(%stage) : <ptype>, <ptype>, <stage_type>)
//
// The VEC staging tile is synthesised here (the user never sees it): TGET
// copies GM->GM through a VEC bounce buffer, so codegen sizes one ping tile to
// the (2-D flattened) transfer extent.
static constexpr uint64_t kTgetVecStagingFractal = 512;

static std::string MakeGetCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "pld.tensor.get requires 3 arguments (dst, peer, src), got "
                               << op->args_.size();

  // dst: local DistributedTensor destination — reuses the local tensor_view
  // created by EmitMakeTensorViews (no peer arithmetic, like wait's signal).
  auto dst_var = AsVarLike(op->args_[0]);
  CHECK(dst_var) << "pld.tensor.get dst must be a Var-like expression";
  auto dst_dist = As<ir::DistributedTensorType>(dst_var->GetType());
  CHECK(dst_dist) << "pld.tensor.get dst must be DistributedTensorType, got "
                  << dst_var->GetType()->TypeName();

  // src: remote (peer-addressed) DistributedTensor source.
  auto src_binding = ResolveDistTensorBinding(op->args_[2], codegen, "pld.tensor.get");
  auto peer_scalar = As<ir::ScalarType>(op->args_[1]->GetType());
  CHECK(peer_scalar) << "pld.tensor.get peer must be ScalarType at codegen, got "
                     << op->args_[1]->GetType()->TypeName();

  const auto& shape = dst_dist->shape_;
  const size_t rank = shape.size();
  const std::string dtype_str = codegen.GetTypeString(dst_dist->dtype_);

  // Full-slice partition views: offsets all-zero, sizes = full shape. dst and
  // src share the same partition_tensor_view type (same dtype + static shape).
  std::string c0 = codegen.GetOrEmitConstant(static_cast<int64_t>(0), DataType::INDEX);
  std::vector<std::string> zero_offsets(rank, c0);
  std::vector<std::string> size_ssa = GetSizeCodes(shape, codegen);
  std::string partition_type = MakePartitionTensorViewType(GetDimStrings(shape), dtype_str);

  // dst: local tensor_view + full-slice partition_view.
  std::string dst_local_view = codegen.GetOrCreateTensorView(dst_var);
  std::string dst_view_type = codegen.GetTensorViewTypeString(dst_dist.get());
  std::string dst_pview = EmitPartitionViewPTO(dst_var->name_hint_ + "_local", dst_local_view, dst_view_type,
                                               partition_type, zero_offsets, size_ssa, codegen);

  // src: CommRemoteOffset + addptr + make_tensor_view at the call site, then
  // a full-slice partition_view.
  auto src_peer_view = EmitCommRemoteView(src_binding, op->args_[1], codegen);
  std::string src_pview =
      EmitPartitionViewPTO(src_binding.var->name_hint_ + "_peer", src_peer_view.ssa,
                           src_peer_view.view_type_str, partition_type, zero_offsets, size_ssa, codegen);

  // Synthesise a VEC staging tile_buf sized to the 2-D-flattened transfer:
  // rows = product of leading dims, cols = innermost dim (rank-1 -> 1xN).
  int64_t cols = codegen.GetConstIntValue(shape[rank - 1]);
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < rank; ++i) {
    rows *= codegen.GetConstIntValue(shape[i]);
  }
  std::string stage_type = codegen::FormatTileBufTypeString(
      "vec", dtype_str, rows, cols, ir::TileLayout::row_major, ir::TileLayout::none_box,
      kTgetVecStagingFractal, ir::PadValue::null, /*v_row=*/rows, /*v_col=*/cols);
  std::string stage = codegen.AllocNewTileBuf(stage_type, "tget_stage");

  std::ostringstream tget;
  tget << "pto.comm.tget(" << dst_pview << ", " << src_pview << ", buf(" << stage << ") : " << partition_type
       << ", " << partition_type << ", " << stage_type << ")";
  codegen.Emit(tget.str());
  return "";
}

// ============================================================================
// On-core array (ArrayType) -> !pto.local_array lowering helpers
// ============================================================================

// Emit a value SSA whose MLIR type matches the array's element dtype. The IR's
// array.update_element verifier permits an `index`-typed value into an integer
// array (and vice-versa), so the C++ orchestration path relies on implicit
// conversion. PTO/MLIR is strictly typed, so any dtype mismatch is bridged with
// an explicit arith cast here (index_cast for index<->int, trunci/extsi/extui
// for int width changes).
static std::string EmitLocalArrayValue(codegen::PTOCodegen& codegen, const ir::ExprPtr& value,
                                       DataType target) {
  std::string ssa = codegen.GetExprAsCode(value);
  auto value_type = ir::As<ScalarType>(value->GetType());
  if (!value_type || value_type->dtype_ == target) {
    return ssa;
  }
  DataType src = value_type->dtype_;
  std::string mlir_op;
  if (src == DataType::INDEX || target == DataType::INDEX) {
    mlir_op = "arith.index_cast";
  } else if (src.GetBit() > target.GetBit()) {
    mlir_op = "arith.trunci";
  } else if (src.GetBit() < target.GetBit()) {
    mlir_op = src.IsUnsignedInt() ? "arith.extui" : "arith.extsi";
  } else {
    // Same bit width but distinct dtype (signed vs unsigned, e.g. i32 vs ui32):
    // no arith width/index cast applies, yet the operand type must still match
    // the element dtype. Bridge with the MLIR escape-hatch cast. Unreachable for
    // verifier-valid IR (array.update_element requires equal dtypes for
    // non-index values), but keeps the operand well-typed rather than silently
    // emitting a mistyped value.
    mlir_op = "builtin.unrealized_conversion_cast";
  }
  std::string out = codegen.NewTemp();
  codegen.Emit(out + " = " + mlir_op + " " + ssa + " : " + codegen.GetTypeString(src) + " to " +
               codegen.GetTypeString(target));
  return out;
}

// ============================================================================
// RegisterPTOOps: Register all standard PTO ops to the given backend
// ============================================================================

// Emit ``%rk_pair = pto.load_scalar %ctx[%slot] : !pto.ptr<i64> -> i64`` for
// the (rankId, rankNum) u64 slot. Shared by ``pld.system.rank`` (low 32 bits)
// and ``pld.system.nranks`` (high 32 bits) — see comm_layout.h for the static
// asserts that anchor rankNum at rankId + 4 in the same i64 slot.
static std::string EmitLoadRankPair(codegen::PTOCodegen& cg, const std::string& ctx_ssa) {
  namespace cl = codegen::distributed::comm_layout;
  constexpr int64_t kRankSlotIdx = static_cast<int64_t>(cl::kRankIdOffset / cl::kWindowSlotStride);
  std::string slot_c = cg.GetOrEmitConstant(kRankSlotIdx, DataType::INDEX);
  std::string rk_pair = cg.NewTemp();
  cg.Emit(rk_pair + " = pto.load_scalar " + ctx_ssa + "[" + slot_c + "] : !pto.ptr<i64> -> i64");
  return rk_pair;
}

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
    if (RequiresRowMajorLayout(entry.op_name)) {
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

  // On-core arrays (ArrayType) -> PTOAS stack-local arrays. The IR's
  // SSA-functional update_element semantics are realized in place: PTOCodegen's
  // AssignStmt dispatch aliases an array.update_element result Var to the input
  // array's SSA name BEFORE invoking the codegen below, so the emitted
  // pto.local_array_set mutates the same `pto.declare_local_array` storage.
  reg("array.create", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "array.create requires 1 argument (extent)";
    auto array_type = ir::As<ir::ArrayType>(op->GetType());
    CHECK(array_type) << "array.create must return ArrayType";
    std::string result = codegen.GetCurrentResultTarget();
    INTERNAL_CHECK_SPAN(!result.empty(), op->span_) << "array.create requires an assignment target";
    codegen.Emit(result + " = pto.declare_local_array -> " +
                 codegen::FormatLocalArrayTypeString(*array_type));
    return std::string("");
  });

  reg("array.get_element", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 2) << "array.get_element requires 2 arguments (array, index)";
    auto array_type = ir::As<ir::ArrayType>(op->args_[0]->GetType());
    CHECK(array_type) << "array.get_element first argument must be an ArrayType";
    std::string result = codegen.GetCurrentResultTarget();
    INTERNAL_CHECK_SPAN(!result.empty(), op->span_) << "array.get_element requires an assignment target";
    std::string arr = codegen.GetExprAsCode(op->args_[0]);
    std::string idx = EmitIndexOperand(codegen, op->args_[1], "array.get_element index");
    codegen.Emit(result + " = pto.local_array_get " + arr + "[" + idx +
                 "] : " + codegen::FormatLocalArrayTypeString(*array_type) + " -> " +
                 codegen::DataTypeToMLIR(array_type->dtype_));
    return std::string("");
  });

  reg("array.update_element", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 3) << "array.update_element requires 3 arguments (array, index, value)";
    auto array_type = ir::As<ir::ArrayType>(op->args_[0]->GetType());
    CHECK(array_type) << "array.update_element first argument must be an ArrayType";
    // arr resolves to the input array's SSA; the AssignStmt dispatch has already
    // aliased the result Var to this name, so the write is in place.
    std::string arr = codegen.GetExprAsCode(op->args_[0]);
    std::string idx = EmitIndexOperand(codegen, op->args_[1], "array.update_element index");
    std::string value = EmitLocalArrayValue(codegen, op->args_[2], array_type->dtype_);
    codegen.Emit("pto.local_array_set " + arr + "[" + idx + "], " + value + " : " +
                 codegen::FormatLocalArrayTypeString(*array_type) + ", " +
                 codegen::DataTypeToMLIR(array_type->dtype_));
    return std::string("");
  });

  // Helper for zero-arg i64 query ops that need index_cast (get_subblock_idx, get_block_idx, etc.)
  auto reg_i64_to_index_op = [&](const char* tile_op, const char* pto_op) {
    reg(tile_op, [tile_op, pto_op](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
      CHECK(op->args_.empty()) << tile_op << " takes no arguments, got " << op->args_.size();
      std::string result = codegen.GetCurrentResultTarget();
      INTERNAL_CHECK_SPAN(!result.empty(), op->span_) << tile_op << " requires assignment target";
      std::string i64_tmp = codegen.NewTemp();
      codegen.Emit(i64_tmp + " = " + pto_op);
      codegen.Emit(result + " = arith.index_cast " + i64_tmp + " : i64 to index");
      return std::string("");
    });
  };
  reg_i64_to_index_op("tile.get_subblock_idx", "pto.get_subblock_idx");

  // SPMD block identity ops read from synthetic i32 %arg prefix params that
  // PTOCodegen prepends to the func.func signature whenever the function
  // body contains tile.get_block_idx / tile.get_block_num. The kernel
  // wrapper resolves the runtime values from intrinsic.h::get_block_idx(args) /
  // get_block_num(args) and forwards them as the first two call args.
  auto reg_spmd_block_op = [&](const char* tile_op, std::string (codegen::PTOCodegen::*getter)() const) {
    reg(tile_op, [tile_op, getter](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
      CHECK(op->args_.empty()) << tile_op << " takes no arguments, got " << op->args_.size();
      std::string result = codegen.GetCurrentResultTarget();
      INTERNAL_CHECK_SPAN(!result.empty(), op->span_) << tile_op << " requires assignment target";
      std::string arg_ssa = (codegen.*getter)();
      INTERNAL_CHECK_SPAN(!arg_ssa.empty(), op->span_)
          << tile_op << " requires PTOCodegen SPMD prefix params to be initialised";
      codegen.Emit(result + " = arith.index_cast " + arg_ssa + " : i32 to index");
      return std::string("");
    });
  };
  reg_spmd_block_op("tile.get_block_idx", &codegen::PTOCodegen::GetSpmdBlockIdxArgSSA);
  reg_spmd_block_op("tile.get_block_num", &codegen::PTOCodegen::GetSpmdBlockNumArgSSA);

  // tile.move → pto.tmov with no-op elision.
  // When MemoryReuse inserts a tile.move between two MemRefs that end up at the
  // same physical address after AllocateMemoryAddr (e.g. acc→acc at the same Acc
  // offset), the move is a no-op. Elide it to avoid emitting pto.tmov with
  // unsupported same-space address pairs (fixes #1310).
  reg("tile.move", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "tile.move requires 1 argument, got " << op->args_.size();

    auto src_var = AsVarLike(op->args_[0]);
    auto dst_var = codegen.GetCurrentResultVar();
    if (src_var && dst_var) {
      auto src_tile = As<ir::TileType>(src_var->GetType());
      auto dst_tile = As<ir::TileType>(dst_var->GetType());
      if (src_tile && dst_tile && src_tile->memref_.has_value() && dst_tile->memref_.has_value()) {
        auto src_space = src_tile->GetMemorySpace();
        auto dst_space = dst_tile->GetMemorySpace();
        if (src_space.has_value() && dst_space.has_value() && *src_space == *dst_space) {
          auto src_offset = As<ir::ConstInt>((*src_tile->memref_)->byte_offset_);
          auto dst_offset = As<ir::ConstInt>((*dst_tile->memref_)->byte_offset_);
          if (src_offset && dst_offset && src_offset->value_ == dst_offset->value_) {
            // Alias the destination to the source SSA value so downstream
            // references use the source's defined buffer, not the destination's
            // alloc_tile (which would be unwritten after eliding the tmov).
            codegen.SetCurrentResultBuf(codegen.GetExprAsCode(op->args_[0]));
            return std::string("");  // no-op: same space, same address
          }
        }
      }
    }

    codegen.Emit("pto.tmov " + GenerateInsOutsClause(op, codegen));
    return std::string("");
  });

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
  // ``tensor.as_layout`` (RFC #1300 §3.3): pure metadata reinterpret over the
  // same physical buffer. In InCore code, ``LowerTransposeLoadParamLayout``
  // prepends ``b_dn = tensor.as_layout(b, DN)`` at the top of the body for
  // each ``tile.load(transpose=True)``-loaded param, then rewrites the body
  // to use ``b_dn`` (a Var with TensorType ``[..., b, a] DN`` and explicit
  // canonical strides) in place of the original param ``b``.
  //
  // Codegen lowers this to a fresh ``pto.make_tensor_view`` bound to the
  // input's underlying buffer (the function parameter SSA), using the LHS's
  // own ``(shape, stride, layout)`` from its TensorType. Downstream
  // ``tile.load`` lookups via ``GetOrCreateTensorView`` find the LHS through
  // the ``RegisterTensorView`` call below.
  reg("tensor.as_layout", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "tensor.as_layout requires 1 arg (input)";
    auto input_var = AsVarLike(op->args_[0]);
    CHECK(input_var) << "tensor.as_layout input must be a Var/IterArg";

    auto lhs_var = codegen.GetCurrentResultVar();
    INTERNAL_CHECK_SPAN(static_cast<bool>(lhs_var), op->span_)
        << "Internal error: tensor.as_layout result var must be set by VisitStmt_(AssignStmt)";
    auto lhs_type = As<ir::TensorType>(lhs_var->GetType());
    CHECK(lhs_type) << "tensor.as_layout output must be TensorType, got " << lhs_var->GetType()->TypeName();
    INTERNAL_CHECK_SPAN(lhs_type->tensor_view_.has_value(), op->span_)
        << "Internal error: tensor.as_layout output must have an explicit TensorView "
           "(set by DeduceTensorAsLayoutType + CanonicalizeView)";

    const size_t rank = lhs_type->shape_.size();
    const auto& view = lhs_type->tensor_view_.value();
    INTERNAL_CHECK_SPAN(view.stride.size() == rank, op->span_)
        << "Internal error: tensor.as_layout output stride rank " << view.stride.size()
        << " does not match shape rank " << rank;

    // The result SSA name (auto-allocated by VisitStmt_(AssignStmt) for the
    // backend-dispatched RHS Call) doubles as the tensor_view SSA name —
    // register it in tensor_to_view so downstream tile.load lookups resolve.
    std::string result_buf = codegen.GetCurrentResultTarget();
    INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_) << "Internal error: result buf must be set";
    codegen.RegisterTensorView(lhs_var, result_buf);

    // Materialize shape and stride SSA names.
    auto emit_dim = [&](const ir::ExprPtr& dim) -> std::string {
      if (auto c = As<ir::ConstInt>(dim)) {
        return codegen.GetOrEmitConstant(c->value_, DataType::INDEX);
      }
      return codegen.EmitCastToIndex(dim, codegen.GetExprAsCode(dim));
    };
    std::vector<std::string> shape_dim_names(rank);
    for (size_t j = 0; j < rank; ++j) shape_dim_names[j] = emit_dim(lhs_type->shape_[j]);
    std::vector<std::string> stride_names(rank);
    for (size_t j = 0; j < rank; ++j) stride_names[j] = emit_dim(view.stride[j]);

    std::string layout_str = "nd";
    switch (view.layout) {
      case ir::TensorLayout::DN:
        layout_str = "dn";
        break;
      case ir::TensorLayout::NZ:
        layout_str = "nz";
        break;
      case ir::TensorLayout::ND:
        break;
    }

    std::ostringstream oss;
    oss << result_buf << " = pto.make_tensor_view " << codegen.GetVarName(input_var) << ", shape = [";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) oss << ", ";
      oss << shape_dim_names[j];
    }
    oss << "], strides = [";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) oss << ", ";
      oss << stride_names[j];
    }
    oss << "] {layout = #pto.layout<" << layout_str << ">}";
    oss << ": !pto.tensor_view<";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) oss << "x";
      oss << "?";
    }
    oss << "x" << codegen.GetTypeString(lhs_type->dtype_) << ">";
    return oss.str();
  });

  reg("tile.load", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileLoadCodegenPTO(op, codegen);
  });
  reg("tile.store", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileStoreCodegenPTO(op, codegen);
  });
  // Distributed N6 ops — cross-rank tile load + per-rank signal notify/wait +
  // synchronous bulk get/put. See MakeRemoteLoadCodegenPTO /
  // MakeNotifyCodegenPTO / MakeWaitCodegenPTO / MakeGetCodegenPTO /
  // MakePutCodegenPTO for the emitted MLIR shape.
  // Cross-rank ops lower to a single
  // ``func.call @CommRemoteOffset_<dtype>`` against a module-level helper
  // emitted by PTOCodegen::EmitCommRemoteOffsetHelpers; the helper returns
  // the peer-vs-local element offset (``index``) and the call site emits
  // ``pto.addptr`` + ``pto.make_tensor_view`` locally so PTOAS's per-func
  // "addptr must feed make_tensor_view" check is satisfied. The helper's
  // byte-offset literals are pinned to ``comm_layout::k*`` constants
  // (PyPTO compile-time static_asserts catch any CommContext ABI drift).
  reg("pld.tile.remote_load", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeRemoteLoadCodegenPTO(op, codegen);
  });
  reg("pld.system.notify",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeNotifyCodegenPTO(op, codegen); });
  reg("pld.system.wait",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeWaitCodegenPTO(op, codegen); });

  // Distributed N7 ops — CommContext accessor lowering.
  //
  // ``pld.system.get_comm_ctx(dist_t) -> CommCtxType``: pure SSA alias. No
  // MLIR is emitted; the ctx-ptr arg slot that PTOCodegen's
  // ``GenerateFunction`` appended for ``dist_t`` (see
  // ``fs_.dist_tensor_to_ctx`` / ``GetCommCtxSSAFor``) is published as the
  // current expression value, which the surrounding ``VisitStmt_(AssignStmt)``
  // then binds to the LHS Var. Downstream ``pld.system.rank(ctx)`` /
  // ``pld.system.nranks(ctx)`` codegen resolves ``ctx`` via the standard
  // ``GetExprAsCode(call->args_[0])`` path.
  reg("pld.system.get_comm_ctx",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
        auto& cg = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
        CHECK(op->args_.size() == 1) << "pld.system.get_comm_ctx expects exactly 1 argument, got "
                                     << op->args_.size();
        auto var = ir::AsVarLike(op->args_[0]);
        CHECK(var) << "pld.system.get_comm_ctx expects a Var (DistributedTensor param), got "
                   << op->args_[0]->TypeName();
        std::string ctx_ssa = cg.GetCommCtxSSAFor(var.get());
        CHECK(!ctx_ssa.empty())
            << "No CommContext ptr arg threaded for DistributedTensor '" << var->name_hint_
            << "' — ensure the func.func ctx segment was emitted (PTOCodegen::GenerateFunction)";
        if (auto lhs = cg.GetCurrentResultVar()) {
          cg.RegisterVarToMlir(lhs, ctx_ssa);
        }
        cg.SetCurrentExprValue(ctx_ssa);
        return "";
      });

  // ``pld.system.rank(ctx) -> i32``: low 32 bits of the (rankId, rankNum)
  // u64 slot. Emits ``pto.load_scalar`` (via EmitLoadRankPair) + arith.trunci.
  reg("pld.system.rank", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
    auto& cg = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "pld.system.rank expects exactly 1 argument, got " << op->args_.size();
    std::string ctx_ssa = cg.GetExprAsCode(op->args_[0]);
    std::string rk_pair = EmitLoadRankPair(cg, ctx_ssa);
    std::string rk = cg.GetCurrentResultTarget();
    cg.Emit(rk + " = arith.trunci " + rk_pair + " : i64 to i32");
    cg.SetCurrentExprValue(rk);
    return "";
  });

  // ``pld.system.nranks(ctx) -> i32``: high 32 bits of the same slot —
  // ``kRankNumOffset == kRankIdOffset + 4`` lets us shift the already-loaded
  // i64 right by 32 instead of issuing a second pto.load_scalar.
  reg("pld.system.nranks", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
    auto& cg = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 1) << "pld.system.nranks expects exactly 1 argument, got " << op->args_.size();
    namespace cl = codegen::distributed::comm_layout;
    static_assert(cl::kRankNumOffset == cl::kRankIdOffset + 4,
                  "pld.system.nranks codegen assumes rankNum sits in the high 32 bits of rankId's i64 slot");
    std::string ctx_ssa = cg.GetExprAsCode(op->args_[0]);
    std::string rk_pair = EmitLoadRankPair(cg, ctx_ssa);
    std::string c32 = cg.GetOrEmitConstant(static_cast<int64_t>(32), DataType::INT64);
    std::string rn_i64 = cg.NewTemp();
    cg.Emit(rn_i64 + " = arith.shrui " + rk_pair + ", " + c32 + " : i64");
    std::string rn = cg.GetCurrentResultTarget();
    cg.Emit(rn + " = arith.trunci " + rn_i64 + " : i64 to i32");
    cg.SetCurrentExprValue(rn);
    return "";
  });
  reg("pld.tensor.get",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeGetCodegenPTO(op, codegen); });
  reg("pld.tile.put",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakePutCodegenPTO(op, codegen); });
  reg("tile.transpose", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileTransposeCodegenPTO(op, codegen);
  });
  if (exclude_ops.count("tile.sel") == 0) {
    backend.RegisterOp("tile.sel")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeTileSelCodegenPTO(op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_input_layout(2, ir::TileLayout::row_major)
        .set_input_layout(3, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }
  // tile.mscatter: src and idx must be row_major (MTE3 DMA reads UB linearly)
  if (exclude_ops.count("tile.mscatter") == 0) {
    backend.RegisterOp("tile.mscatter")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeTileMscatterCodegenPTO(op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major);
  }
  reg("tile.alloc", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileAllocCodegenPTO(op, codegen);
  });
  reg("tile.create", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    (void)op;
    (void)codegen_base;
    return std::string("");  // No MLIR emission - tile allocation handled by pto.alloc_tile
  });
  reg("tile.col_expand", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeColExpandCodegenPTO(op, codegen);
  });
  reg("tile.row_expand", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeRowExpandCodegenPTO(op, codegen);
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
  // tile.rsqrt accepts 1 arg (basic) or 2 args (high-precision with tmp workspace).
  // Both forms emit pto.trsqrt with the appropriate ins() arity. Per ISA, both
  // inputs (when present) and the output must be row_major.
  if (exclude_ops.count("tile.rsqrt") == 0) {
    backend.RegisterOp("tile.rsqrt")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          size_t arity = op->args_.size();
          CHECK(arity == 1 || arity == 2) << "tile.rsqrt requires 1 or 2 arguments, but got " << arity;
          return MakeNaryCodegenPTO("pto.trsqrt", arity, op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }
  // tile.col_sum (TCOLSUM): accepts 1 arg (sequential) or 2 args (tile + tmp for binary-tree).
  // PTOAS pairs tmp operand with isBinary attribute; both present or both absent.
  if (exclude_ops.count("tile.col_sum") == 0) {
    backend.RegisterOp("tile.col_sum")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
          auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
          CHECK(op->args_.size() == 1 || op->args_.size() == 2)
              << "tile.col_sum requires 1 or 2 arguments, but got " << op->args_.size();
          std::string config_attr = op->args_.size() == 2 ? " {isBinary = true}" : "";
          codegen.Emit("pto.tcolsum " + GenerateInsOutsClause(op, codegen, config_attr));
          return std::string("");
        });
  }
  // tile.full (TEXPANDS): output is row_major per ISA
  if (exclude_ops.count("tile.full") == 0) {
    backend.RegisterOp("tile.full")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeFullCodegenPTO("pto.texpands", op, codegen);
        })
        .set_output_layout(ir::TileLayout::row_major);
  }
  // tile.cmps (TCMPS): tile input and output must be row_major per ISA
  if (exclude_ops.count("tile.cmps") == 0) {
    backend.RegisterOp("tile.cmps")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeCmpsCodegenPTO("pto.tcmps", op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }
  reg("tile.assign", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeAssignCodegenPTO("pto.tassign", op, codegen);
  });
  reg("tile.ci", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeCiCodegenPTO("pto.tci", op, codegen);
  });
  // tile.sort32 (TSORT32): all inputs and output must be row_major per ISA
  if (exclude_ops.count("tile.sort32") == 0) {
    backend.RegisterOp("tile.sort32")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeSort32CodegenPTO("pto.tsort32", op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }
  // tile.gather_mask (TGATHER mask form): only src operand + maskPattern attribute
  reg("tile.gather_mask", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeGatherMaskCodegenPTO(op, codegen);
  });
  // tile.gather_compare (TGATHER compare form, two outputs):
  // 3-input op returning TupleType{dst, cdst}; outs() bound to downstream
  // TupleGetItemExpr consumers (parser desugars `dst, cdst = ...`).
  reg("tile.gather_compare", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeGatherCompareCodegenPTO(op, codegen);
  });
  // tile.scatter (TSCATTER index form, DPS): 3-input op (dst, src, indexes).
  reg("tile.scatter", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeScatterCodegenPTO(op, codegen);
  });
  // tile.scatter_mask (TSCATTER mask form, DPS): 2-input op (dst, src) +
  // maskPattern attr. A3 / CPU-sim style backends only.
  reg("tile.scatter_mask", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeScatterMaskCodegenPTO(op, codegen);
  });
  // tile.mrgsort_format2 (TMRGSORT format2): all inputs and output must be row_major per ISA
  if (exclude_ops.count("tile.mrgsort_format2") == 0) {
    backend.RegisterOp("tile.mrgsort_format2")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeMrgSortCodegenPTO("pto.tmrgsort", op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major)
        .set_input_layout(2, ir::TileLayout::row_major)
        .set_input_layout(3, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }
  // tile.mrgsort_format1 (TMRGSORT format1): src and output must be row_major per ISA
  if (exclude_ops.count("tile.mrgsort_format1") == 0) {
    backend.RegisterOp("tile.mrgsort_format1")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeMrgSort1CodegenPTO("pto.tmrgsort", op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_output_layout(ir::TileLayout::row_major);
  }
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

    return std::string("");
  });
  reg("tile.slice", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    // 3-5 args: (tile, shape, offset[, valid_shape[, drop_dims]]). The optional
    // 5th `drop_dims` operand only affects the result type's rank (already
    // reflected in the result tile-buf type) — the pto.subview sizes/offset come
    // from the full-rank shape/offset tuples, so codegen ignores it. An empty
    // 4th MakeTuple is the "no valid_shape" sentinel that pairs with drop_dims.
    CHECK(op->args_.size() >= 3 && op->args_.size() <= 5)
        << "Operation:[tile.slice] requires 3-5 arguments (tile, shape, offset[, valid_shape[, "
           "drop_dims]]), but got "
        << op->args_.size();

    auto source_tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    INTERNAL_CHECK_SPAN(source_tile_type, op->span_) << "tile.slice source must be TileType";

    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);

    auto offset_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    INTERNAL_CHECK_SPAN(offset_tuple, op->span_) << "tile.slice third argument must be a tuple (offset)";
    INTERNAL_CHECK_SPAN(offset_tuple->elements_.size() >= 2, op->span_)
        << "tile.slice offset tuple must have at least 2 elements (row, col), got "
        << offset_tuple->elements_.size();
    std::string row_off = codegen.GetExprAsCode(offset_tuple->elements_[0]);
    std::string col_off = codegen.GetExprAsCode(offset_tuple->elements_[1]);

    auto shape_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
    INTERNAL_CHECK_SPAN(shape_tuple, op->span_) << "tile.slice shape must be a literal tuple";
    INTERNAL_CHECK_SPAN(shape_tuple->elements_.size() >= 2, op->span_)
        << "tile.slice shape must have at least 2 elements (rows, cols)";
    auto rows_const = ir::As<ir::ConstInt>(shape_tuple->elements_[0]);
    auto cols_const = ir::As<ir::ConstInt>(shape_tuple->elements_[1]);
    INTERNAL_CHECK_SPAN(rows_const && cols_const, op->span_)
        << "tile.slice shape must be compile-time constant for pto.subview sizes attribute";

    std::string valid_row;
    std::string valid_col;
    // valid_shape is the optional 4th operand; an empty MakeTuple means "none"
    // (the form used when only drop_dims is supplied).
    auto valid_tuple = op->args_.size() >= 4 ? ir::As<ir::MakeTuple>(op->args_[3]) : nullptr;
    bool has_explicit_valid_shape = valid_tuple != nullptr && !valid_tuple->elements_.empty();
    if (has_explicit_valid_shape) {
      INTERNAL_CHECK_SPAN(valid_tuple, op->span_) << "tile.slice valid_shape must be a literal tuple";
      INTERNAL_CHECK_SPAN(valid_tuple->elements_.size() >= 2, op->span_)
          << "tile.slice valid_shape must have at least 2 elements";
      valid_row = codegen.GetExprAsCode(valid_tuple->elements_[0]);
      valid_col = codegen.GetExprAsCode(valid_tuple->elements_[1]);
    }

    std::string result_target = codegen.GetCurrentResultTarget();
    std::string result_type = codegen.GetCurrentResultTileBufTypeString();
    INTERNAL_CHECK_SPAN(!result_target.empty(), op->span_) << "tile.slice requires assignment target";

    auto row_off_const = ir::As<ir::ConstInt>(offset_tuple->elements_[0]);
    auto col_off_const = ir::As<ir::ConstInt>(offset_tuple->elements_[1]);
    bool full_window = row_off_const && col_off_const && row_off_const->value_ == 0 &&
                       col_off_const->value_ == 0 && source_tile_type->shape_.size() >= 2 &&
                       ExprsEquivalentForSubview(shape_tuple->elements_[0], source_tile_type->shape_[0]) &&
                       ExprsEquivalentForSubview(shape_tuple->elements_[1], source_tile_type->shape_[1]);

    // Skip the tmov fast-path when the source is already a subview: tmov
    // materializes a copy, which breaks view/alias semantics that tile.slice
    // must preserve.  Fall through to the normal subview emission instead.
    bool source_is_subview = codegen.GetSubviewMaterialization(src) != nullptr;
    if (full_window && !source_is_subview) {
      std::ostringstream mov;
      mov << "pto.tmov ins(" << src;
      if (!src_type.empty()) mov << " : " << src_type;
      mov << ") outs(" << result_target;
      if (!result_type.empty()) mov << " : " << result_type;
      mov << ")";
      codegen.Emit(mov.str());
      if (has_explicit_valid_shape) {
        std::ostringstream set_validshape;
        set_validshape << "pto.set_validshape " << result_target << ", " << valid_row << ", " << valid_col;
        if (!result_type.empty()) {
          set_validshape << " : " << result_type;
        }
        codegen.Emit(set_validshape.str());
      }
      return std::string("");
    }

    // Emit a pto.subview and register its type; returns (view_ssa, view_type).
    auto emit_subview = [&]() -> std::pair<std::string, std::string> {
      auto view_type_info = InferSubviewTileTypeComponents(*source_tile_type, *shape_tuple, *offset_tuple,
                                                           codegen.GetTypeString(source_tile_type->dtype_));
      INTERNAL_CHECK_SPAN(source_tile_type->memory_space_.has_value(), op->span_)
          << "tile.slice source must carry a memory space for pto.subview result typing";
      std::string view_type = codegen::FormatTileBufTypeString(
          codegen::MemorySpaceToMLIR(*source_tile_type->memory_space_), view_type_info.dtype_str,
          view_type_info.rows, view_type_info.cols, view_type_info.blayout, view_type_info.slayout,
          view_type_info.fractal, view_type_info.pad, view_type_info.v_row, view_type_info.v_col,
          view_type_info.v_row_dynamic, view_type_info.v_col_dynamic);
      std::string view_ssa = codegen.NewNamedTemp("slice_view");
      std::ostringstream oss;
      oss << view_ssa << " = pto.subview " << src << "[" << row_off << ", " << col_off << "] sizes ["
          << rows_const->value_ << ", " << cols_const->value_ << "]";
      if (!src_type.empty() && !view_type.empty()) {
        oss << " : " << src_type << " -> " << view_type;
      }
      codegen.Emit(oss.str());
      codegen.RegisterTileBufType(view_ssa, view_type);
      return {view_ssa, view_type};
    };

    if (!has_explicit_valid_shape) {
      auto [view_ssa, view_type] = emit_subview();
      codegen::PTOCodegen::SubviewMaterializationInfo mat_info;
      mat_info.source_ssa = src;
      mat_info.source_type = src_type;
      mat_info.row_off_ssa = row_off;
      mat_info.col_off_ssa = col_off;
      mat_info.materialize_target_ssa = result_target;
      mat_info.materialize_target_type = result_type;
      codegen.RegisterSubviewMaterialization(view_ssa, mat_info);
      // For pure static-window slices, keep the result as a true pto.subview
      // SSA instead of materializing a copy. This preserves view semantics and
      // avoids backend C++ lowering reconstructing the TMOV source tile with
      // the base tile's physical cols/rows.
      codegen.SetCurrentResultBuf(view_ssa);
      return std::string("");
    }

    auto [producer, producer_type] = emit_subview();

    std::ostringstream mov;
    mov << "pto.tmov ins(" << producer;
    if (!producer_type.empty()) mov << " : " << producer_type;
    mov << ") outs(" << result_target;
    if (!result_type.empty()) mov << " : " << result_type;
    mov << ")";
    codegen.Emit(mov.str());
    std::ostringstream set_validshape;
    set_validshape << "pto.set_validshape " << result_target << ", " << valid_row << ", " << valid_col;
    if (!result_type.empty()) {
      set_validshape << " : " << result_type;
    }
    codegen.Emit(set_validshape.str());
    return std::string("");
  });
  reg("tile.assemble", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileAssembleCodegenPTO(op, codegen);
  });
  reg("tile.extract", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 4)
        << "tile.extract requires 4 arguments (src, index_row, index_col, shape), but got "
        << op->args_.size();

    std::string src = codegen.GetExprAsCode(op->args_[0]);
    std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
    std::string row_off = codegen.GetExprAsCode(op->args_[1]);
    std::string col_off = codegen.GetExprAsCode(op->args_[2]);
    // Use the actual offset SSA dtype (`index` / `i64` / `i32` ...) — the IR
    // type-check accepts any IndexLike scalar, so don't hardcode `index`.
    std::string row_type = codegen.GetExprTypeAnnotation(op->args_[1]);
    std::string col_type = codegen.GetExprTypeAnnotation(op->args_[2]);
    if (row_type.empty()) row_type = "index";
    if (col_type.empty()) col_type = "index";
    // args_[3] is the shape tuple: type-deduction only, no PTO operand.

    std::string result_target = codegen.GetCurrentResultTarget();
    std::string result_type = codegen.GetCurrentResultTileBufTypeStringFromTileType();

    auto existing_type = codegen.GetSSATileBufType(result_target);
    if (!result_type.empty() && existing_type != result_type) {
      result_target = codegen.AllocNewTileBuf(result_type, "extract_buf");
      codegen.SetCurrentResultBuf(result_target);
    } else if (!result_type.empty()) {
      codegen.RegisterTileBufType(result_target, result_type);
    }

    std::ostringstream oss;
    oss << "pto.textract ins(" << src << ", " << row_off << ", " << col_off;
    if (!src_type.empty()) {
      oss << " : " << src_type << ", " << row_type << ", " << col_type;
    }
    oss << ") outs(" << result_target;
    if (!result_type.empty()) {
      oss << " : " << result_type;
    }
    oss << ")";
    codegen.Emit(oss.str());
    return std::string("");
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
  reg("tile.set_validshape", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
    CHECK(op->args_.size() == 3)
        << "tile.set_validshape requires 3 arguments (tile, valid_rows, valid_cols), but got "
        << op->args_.size();

    std::string tile_buf = codegen.GetExprAsCode(op->args_[0]);
    std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);
    if (tile_buf.empty()) {
      tile_buf = codegen.GetCurrentResultTarget();
    }
    if (tile_buf_type.empty()) {
      tile_buf_type = codegen.GetCurrentResultTileBufTypeStringFromTileType();
    }

    auto emit_index_arg = [&](const ir::ExprPtr& arg) -> std::string {
      if (auto var = ir::As<ir::Var>(arg)) {
        std::string mlir_name = codegen.GetVarName(var);
        return codegen.EmitCastToIndex(var, mlir_name);
      }
      if (auto c = ir::As<ir::ConstInt>(arg)) {
        return codegen.GetOrEmitConstant(c->value_, DataType::INDEX);
      }
      std::string ssa = codegen.GetExprAsCode(arg);
      if (auto st = ir::As<ir::ScalarType>(arg->GetType())) {
        if (st->dtype_ != DataType::INDEX) {
          std::string src_type = codegen.GetTypeString(st->dtype_);
          std::string idx = codegen.NewTemp();
          codegen.Emit(idx + " = arith.index_cast " + ssa + " : " + src_type + " to index");
          return idx;
        }
      }
      return ssa;
    };

    std::string vr = emit_index_arg(op->args_[1]);
    std::string vc = emit_index_arg(op->args_[2]);

    codegen.RegisterTileBufType(tile_buf, tile_buf_type);
    codegen.SetCurrentResultBuf(tile_buf);
    codegen.Emit("pto.set_validshape " + tile_buf + ", " + vr + ", " + vc + " : " + tile_buf_type);
    return std::string("");
  });
}

}  // namespace backend
}  // namespace pypto
