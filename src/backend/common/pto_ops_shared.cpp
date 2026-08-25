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
 * @file pto_ops_shared.cpp
 * @brief Definitions of the pto_ops_detail shared codegen helper toolkit.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "src/backend/common/pto_ops_internal.h"

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

namespace pto_ops_detail {

// Validate that a string is a safe MLIR identifier (alphanumeric + underscores).
// Prevents injection of arbitrary MLIR via crafted buffer/function names.
void CheckSafeIdentifier(const std::string& value, const std::string& attr_name) {
  for (char c : value) {
    CHECK(c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
        << attr_name << " contains invalid character '" << c
        << "'; only alphanumeric and underscore are allowed";
  }
}

// ============================================================================
// Helper Functions for PTO Code Generation
// ============================================================================

// Every op codegen handler receives the abstract CodegenBase and downcasts to
// PTOCodegen. Centralize the cast so call sites read `auto& cg = AsPto(base);`.
codegen::PTOCodegen& AsPto(codegen::CodegenBase& codegen_base) {
  return dynamic_cast<codegen::PTOCodegen&>(codegen_base);
}

// Validate a call's arity with the canonical "Operation:[<pto_op>] requires N
// argument(s), but got M" message shared by the simple-op handlers.
void CheckArity(const CallPtr& op, std::string_view pto_op_name, size_t arity) {
  INTERNAL_CHECK_SPAN(op->args_.size() == arity, op->span_)
      << "Operation:[" << pto_op_name << "] requires " << arity << " argument" << (arity != 1 ? "s" : "")
      << ", but got " << op->args_.size();
}

const std::vector<std::string> cmp_modes = {"eq", "ne", "lt", "le", "gt", "ge"};
const std::vector<std::string> round_modes = {"NONE", "RINT",  "ROUND", "FLOOR",
                                              "CEIL", "TRUNC", "ODD",   "CAST_RINT"};
// Mask pattern names for pto.tgather mask form (index 0 unused, patterns 1-7)
const std::vector<std::string> mask_patterns = {"",      "P0101", "P1010", "P0001",
                                                "P0010", "P0100", "P1000", "P1111"};

// Build a partition_tensor_view type string from dimension strings and element dtype.
std::string MakePartitionTensorViewType(const std::vector<std::string>& dims, const std::string& dtype_str) {
  std::ostringstream oss;
  oss << "!pto.partition_tensor_view<";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) oss << "x";
    oss << dims[i];
  }
  oss << "x" << dtype_str << ">";
  return oss.str();
}

// Coerce each offset operand to `index`. A
// pto.partition_view offset MUST be `index`; a bare i32 offset (e.g. a
// data-dependent row from a scalar param or pl.read) would otherwise reach
// PTOAS as i32 and fail type unification ('index' vs 'i32'). ConstInts are
// emitted as index constants; loop vars are already index (EmitCastToIndex is
// a no-op). Used for every pto.partition_view offset (load/store/remote_store/
// notify/wait/put), so all paths coerce offsets identically.
//
// After coercion, every runtime offset is clamped to non-negative via
// arith.maxsi(offset, 0). A negative offset (e.g. a ring-neighbour computation
// that passes through -1 before a conditional adjust) would otherwise reach
// PTOAS as a signed index value that partition_view does not handle, silently
// dropping or misdirecting the access. ConstInt offsets are clamped at emit
// time so compile-time provably-negative offsets also become safe codegen
// (the verifier rejects those independently).
std::vector<std::string> GetIndexOffsetCodes(const std::vector<ir::ExprPtr>& exprs,
                                             codegen::PTOCodegen& codegen) {
  std::vector<std::string> codes;
  codes.reserve(exprs.size());
  for (const auto& expr : exprs) {
    std::string offset_idx;
    if (auto ci = As<ir::ConstInt>(expr)) {
      // Clamp compile-time negative constants to zero at codegen time.
      offset_idx = codegen.GetOrEmitConstant(std::max(ci->value_, static_cast<int64_t>(0)), DataType::INDEX);
    } else {
      // For runtime expressions, insert an arith.maxsi(offset, 0) clamp so
      // that a negative dynamic value cannot reach pto.partition_view, where
      // it would silently misbehave.
      auto raw_idx = codegen.EmitCastToIndex(expr, codegen.GetExprAsCode(expr));
      auto zero_idx = codegen.GetOrEmitConstant(static_cast<int64_t>(0), DataType::INDEX);
      auto clamped = codegen.NewTemp();
      codegen.Emit(clamped + " = arith.maxsi " + raw_idx + ", " + zero_idx + " : index");
      offset_idx = clamped;
    }
    codes.push_back(std::move(offset_idx));
  }
  return codes;
}

// Get dimension strings handling both static (ConstInt) and dynamic dimensions.
// Static dims become numeric strings; dynamic dims become "?".
std::vector<std::string> GetDimStrings(const std::vector<ir::ExprPtr>& exprs) {
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

// Convert expressions to MLIR size codes. partition_view sizes are index-typed,
// so dynamic integer scalars need the same coercion as offsets.
std::vector<std::string> GetSizeCodes(const std::vector<ir::ExprPtr>& exprs, codegen::PTOCodegen& codegen) {
  std::vector<std::string> codes;
  codes.reserve(exprs.size());
  for (const auto& expr : exprs) {
    if (auto c = As<ir::ConstInt>(expr)) {
      codes.push_back(codegen.GetOrEmitConstant(c->value_, DataType::INDEX));
    } else {
      codes.push_back(codegen.EmitCastToIndex(expr, codegen.GetExprAsCode(expr)));
    }
  }
  return codes;
}

bool ExprsEquivalentForSubview(const ir::ExprPtr& lhs, const ir::ExprPtr& rhs) {
  if (lhs.get() == rhs.get()) return true;
  auto lhs_const = As<ir::ConstInt>(lhs);
  auto rhs_const = As<ir::ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

codegen::TileTypeComponents InferSubviewTileTypeComponents(const ir::TileType& source_tile_type,
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
  c.compact = tv.compact;

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
std::string EmitPartitionViewPTO(const std::string& name_hint, const std::string& tensor_view,
                                 const std::string& tensor_view_type, const std::string& partition_type,
                                 const std::vector<std::string>& offset_codes,
                                 const std::vector<std::string>& size_codes, codegen::PTOCodegen& codegen) {
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
std::string EmitFlatOffsetSSAFromValues(const std::vector<std::string>& indices,
                                        const std::vector<ir::ExprPtr>& shape, codegen::PTOCodegen& codegen,
                                        const std::string& name_hint) {
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
std::string GenerateInsOutsClause(const CallPtr& op, codegen::PTOCodegen& codegen,
                                  const std::string& config_attr) {
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

namespace {

std::shared_ptr<const ir::TileType> GetTileTypeFromExpr(const ir::ExprPtr& expr) {
  if (auto var = ir::As<ir::Var>(expr)) {
    return ir::As<ir::TileType>(var->GetType());
  }
  if (auto iter_arg = ir::As<ir::IterArg>(expr)) {
    return ir::As<ir::TileType>(iter_arg->GetType());
  }
  return nullptr;
}

}  // namespace

bool HasStaticValidShape(const std::shared_ptr<const ir::TileType>& tile_type) {
  if (!tile_type) return false;
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*tile_type);
  const auto& valid = tile_view.valid_shape;
  if (valid.empty()) return false;
  return std::all_of(valid.begin(), valid.end(),
                     [](const ir::ExprPtr& dim) { return ir::As<ir::ConstInt>(dim) != nullptr; });
}

void RequireStaticValidShapeForPtoas(const std::shared_ptr<const ir::TileType>& tile_type,
                                     std::string_view op_name, std::string_view role, const ir::Span& span) {
  CHECK_SPAN(HasStaticValidShape(tile_type), span)
      << op_name << " on A2/A3 requires compile-time static valid_shape on " << role
      << " for PTOAS level3 tmp verification";
}

std::string GetTileViewTypeAnnotation(const ir::ExprPtr& expr, codegen::PTOCodegen& codegen) {
  if (auto tile_type = GetTileTypeFromExpr(expr)) {
    if (tile_type->memref_.has_value()) {
      return codegen.GetViewTileBufTypeStringFromTileType(tile_type);
    }
  }
  return "";
}

std::string EnsureTileViewSsa(const ir::ExprPtr& expr,
                              const std::shared_ptr<const ir::TileType>& view_tile_type,
                              codegen::PTOCodegen& codegen, const std::string& temp_prefix) {
  INTERNAL_CHECK(view_tile_type) << "Internal error: EnsureTileViewSsa requires a TileType view";
  const std::string view_type = codegen.GetViewTileBufTypeStringFromTileType(view_tile_type);
  std::string current_ssa = codegen.GetExprAsCode(expr);
  const std::string current_type = codegen.GetExprTypeAnnotation(expr);
  if (!current_type.empty() && current_type == view_type) {
    return current_ssa;
  }
  std::string view_ssa = codegen.NewNamedTemp(temp_prefix);
  codegen.RegisterTileBufType(view_ssa, view_type);
  codegen.RegisterTileViewName(view_ssa);
  std::ostringstream oss;
  oss << view_ssa << " = pto.treshape " << current_ssa;
  if (!current_type.empty()) {
    oss << " : " << current_type << " -> " << view_type;
  }
  codegen.Emit(oss.str());
  return view_ssa;
}

std::string EnsureStaticViewTileSsa(const ir::ExprPtr& expr, codegen::PTOCodegen& codegen,
                                    const std::string& temp_prefix) {
  auto tile_type = GetTileTypeFromExpr(expr);
  INTERNAL_CHECK(tile_type) << "Internal error: EnsureStaticViewTileSsa requires a TileType expression";
  return EnsureTileViewSsa(expr, tile_type, codegen, temp_prefix);
}

void EmitInsOutsWithViewTypes(codegen::PTOCodegen& codegen, std::string_view pto_op_name,
                              const std::vector<std::pair<std::string, std::string>>& ins,
                              const std::string& result_ssa,
                              const std::shared_ptr<const ir::TileType>& result_tile_type,
                              const std::string& config_attr) {
  std::ostringstream oss;
  oss << pto_op_name << " ins(";
  std::string type_annot;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << ins[i].first;
    if (!ins[i].second.empty()) {
      if (!type_annot.empty()) type_annot += ", ";
      type_annot += ins[i].second;
    }
  }
  if (!config_attr.empty()) {
    oss << " " << config_attr;
  }
  if (!type_annot.empty()) {
    oss << " : " << type_annot;
  }
  const std::string result_type = codegen.GetViewTileBufTypeStringFromTileType(result_tile_type);
  oss << ") outs(" << result_ssa;
  if (!result_type.empty()) {
    oss << " : " << result_type;
  }
  oss << ")";
  codegen.Emit(oss.str());
}

// Emit `<pto_op> ins(<op0>, <op1>, ... [: <types>]) outs(<result> [: <result_type>])`
// for handlers that build the operand list explicitly rather than mapping
// op->args_ 1:1 (that case is GenerateInsOutsClause). Each `ins` entry is
// (ssa, type_annotation); an empty type_annotation contributes no operand type,
// and when every entry's type is empty the whole `: <types>` clause is omitted
// (matching the hand-written `if (!type.empty())` idiom). The result target and
// type come from the current AssignStmt context.
void EmitInsOuts(codegen::PTOCodegen& codegen, std::string_view pto_op_name,
                 const std::vector<std::pair<std::string, std::string>>& ins) {
  std::ostringstream oss;
  oss << pto_op_name << " ins(";
  std::string type_annot;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << ins[i].first;
    if (!ins[i].second.empty()) {
      if (!type_annot.empty()) type_annot += ", ";
      type_annot += ins[i].second;
    }
  }
  if (!type_annot.empty()) {
    oss << " : " << type_annot;
  }
  std::string result_target = codegen.GetCurrentResultTarget();
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  oss << ") outs(" << result_target;
  if (!result_type.empty()) {
    oss << " : " << result_type;
  }
  oss << ")";
  codegen.Emit(oss.str());
}

std::string MaterializeSubviewOperandIfNeeded(const ir::ExprPtr& expr, codegen::PTOCodegen& codegen,
                                              const std::string& name_hint) {
  auto var = ir::As<ir::Var>(expr);
  if (!var) return codegen.GetExprAsCode(expr);

  std::string operand = codegen.GetExprAsCode(expr);
  auto* mat = codegen.GetSubviewMaterialization(operand);
  if (!mat) return operand;
  if (mat->emitted) return mat->materialize_target_ssa;

  INTERNAL_CHECK_SPAN(
      !mat->source_memory_space.has_value() || *mat->source_memory_space != ir::MemorySpace::Mat, expr->span_)
      << "Internal error: lazy materialization of a Mat-resident pto.subview "
         "would produce an unsupported Mat->Mat pto.textract (no L1->L1 DMA); "
         "the consumer should accept the subview SSA directly";

  // The materialize target is the slice's own buffer, which inherits — and sits
  // inside — the source allocation. The pto.textract therefore repacks in place
  // over its own still-live input, and only survives that when it is an identity
  // copy: the destination address must be right *and* its dense layout must
  // match the source window's. CanonicalizeTileSlice rewrites every slice that
  // fails either condition into a tile.extract with a fresh buffer, so reaching
  // here with such a slice means it escaped that pass (e.g. it carries a
  // valid_shape / drop_dims and is not a plain 3-arg window). Fail loudly rather
  // than emit silently-wrong data.
  //
  // Address: a dynamic offset cannot be folded into the inherited buffer's
  // ConstInt address, so the destination falls back to the bare source base and
  // the extracted window lands on the source's row 0 (#1640).
  INTERNAL_CHECK_SPAN(mat->const_offset, expr->span_)
      << "Internal error: lazy pto.textract materialization of a dynamic-offset tile.slice would write the "
         "extracted window onto its own live source's row 0: the source-inherited destination buffer cannot "
         "encode a dynamic offset and falls back to the source base. CanonicalizeTileSlice must rewrite it "
         "into a tile.extract with a fresh buffer";

  // Layout: the destination is dense (row pitch = view cols) while the source
  // window is strided (row pitch = source cols). These coincide only for a
  // contiguous window — a single row, or one spanning every column. A column
  // slice of a multi-row tile repacks strided -> dense on top of its source and
  // destroys it (#2010).
  //
  // Note the check reads the *immediate* source's cols; for a slice of a slice
  // the effective pitch is the root tile's. CanonicalizeTileSlice peels such
  // chains, so a non-contiguous chain never reaches this path.  source_cols == 0
  // means the source columns were not statically known — stand down rather than
  // reject a shape we cannot reason about.
  INTERNAL_CHECK_SPAN(mat->source_cols == 0 || mat->view_rows == 1 || mat->view_cols == mat->source_cols,
                      expr->span_)
      << "Internal error: lazy pto.textract materialization of a non-contiguous tile.slice window ("
      << mat->view_rows << "x" << mat->view_cols << " of a tile with " << mat->source_cols
      << " columns) would repack strided -> dense on top of its own live source and corrupt it. "
         "CanonicalizeTileSlice must rewrite this slice into a tile.extract with a fresh buffer";

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

// Verify that two TileTypes share the strict "same tile config" required by
// pto.subview: identical dtype, identical TileView (blayout, slayout, fractal,
// pad, compact), and pad must be null since pto.subview is a pure view and does not
// pad.  Memory-space equality is enforced separately (via memory_inherit
// rules on the op definition); this helper checks the tile_view fields that
// must be byte-for-byte compatible for a subview to be legal.
void CheckSubviewTileCompat(const ir::TileType& source, const ir::TileType& result,
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
  // An Acc compact mismatch is reachable from ordinary DSL: a matmul whose lhs carries a
  // narrowed valid row count produces a compact L0C tile (stride ceil(validRow/16)*16), while a
  // plain `tile.create(target_memory=Acc)` window stays non-compact (stride = physical rows).
  // The two really do address L0C at different pitches, so the copy is not expressible as a
  // subview — say so in the user's terms rather than in `pto.subview`'s (#2470).
  const bool acc_windows =
      source.GetMemorySpace() == ir::MemorySpace::Acc && result.GetMemorySpace() == ir::MemorySpace::Acc;
  CHECK(src_v.compact == res_v.compact)
      << op_name << ": compact mismatch between source and result; pto.subview requires identical "
      << "compact mode"
      << (acc_windows ? "; the two accumulator windows disagree on their L0C fractal stride. A matmul "
                        "whose lhs carries narrowed valid rows writes L0C at ceil(validRow/16)*16, "
                        "and no wider full-height accumulator window can view that layout. Drop the "
                        "narrowing from the matmul operand and compute the full tile."
                      : "");
  CHECK(src_v.pad == ir::PadValue::null)
      << op_name << ": pto.subview does not support pad_value (" << static_cast<int>(src_v.pad)
      << "); apply tile.fillpad on the result tile instead of carrying a pad on the slice/assemble window";
}

std::string EmitIndexOperand(codegen::PTOCodegen& codegen, const ExprPtr& expr, std::string_view context) {
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

}  // namespace pto_ops_detail

}  // namespace backend
}  // namespace pypto
