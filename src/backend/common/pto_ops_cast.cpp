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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/op_rules.h"
#include "pypto/backend/common/pto_ops_common.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/cast_saturation.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/backend/common/pto_ops_internal.h"

namespace pypto::backend {
namespace {

using ir::As;
using ir::CallPtr;
using ir::ExprPtr;
using pto_ops_detail::AsPto;
using pto_ops_detail::EmitInsOutsWithViewTypes;
using pto_ops_detail::EnsureStaticViewTileSsa;
using pto_ops_detail::GenerateInsOutsClause;
using pto_ops_detail::GetTileViewTypeAnnotation;
using pto_ops_detail::RequireStaticValidShapeForPtoas;
using pto_ops_detail::round_modes;

// Target differences belong to this operator and are captured at registration.
enum class CastTarget { A2A3, A5 };

// PTOAS v0.60 makeTCvtTmpType head block: 4 bytes * 64 cols * min(cols/64, 255).
constexpr int64_t kTcvtHeadBlockCols = 64;
constexpr int64_t kTcvtHeadBlockBytes = 4;
constexpr int64_t kTcvtHeadMaxBlocks = 255;
constexpr int64_t kTcvtTailRowBytes = 32;
constexpr int64_t kTcvtFp16HalfToI8Base = 128;
constexpr int64_t kTcvtMinScratchBytes = 32;
constexpr int64_t kTcvtScratchAlignBytes = 32;

int64_t CeilDivI64(int64_t num, int64_t den) { return (num + den - 1) / den; }

OpScratchPlan PlanCastScratch(CastTarget target, const OpScratchRequest& request) {
  const auto& src_tile = request.source_tile;
  if (src_tile && request.source_dtype && src_tile->dtype_ != *request.source_dtype) {
    return {OpScratchStatus::Unsupported, std::nullopt, "TCVT source dtype disagrees with source tile"};
  }
  if (target == CastTarget::A5 || request.saturating.value_or(false)) {
    return {OpScratchStatus::NotRequired, std::nullopt, ""};
  }
  const auto source_dtype = src_tile ? std::optional<DataType>(src_tile->dtype_) : request.source_dtype;
  if (!source_dtype || !request.target_dtype) {
    return {OpScratchStatus::Unknown, std::nullopt, "TCVT source and target dtypes are required"};
  }
  const DataType dst = *request.target_dtype;
  // A2/A3 emulates these OFF conversions with a vector helper. INT4 uses a
  // native saturating vconv; the other pairs do not need compiler scratch.
  const bool helper_pair = (*source_dtype == DataType::FP32 && dst == DataType::INT16) ||
                           (*source_dtype == DataType::FP16 &&
                            (dst == DataType::INT16 || dst == DataType::INT8 || dst == DataType::UINT8));
  if (!helper_pair) return {OpScratchStatus::NotRequired, std::nullopt, ""};
  if (!request.saturating) {
    return {OpScratchStatus::Unknown, std::nullopt, "TCVT effective saturation mode is required"};
  }
  if (!src_tile) {
    return {OpScratchStatus::Unknown, std::nullopt, "TCVT scratch sizing requires source tile geometry"};
  }
  const auto& src_shape = src_tile->shape_;
  const auto valid_shape = ir::GetValidShape(src_tile);
  if (src_shape.size() != 2 || valid_shape.size() != 2) {
    return {OpScratchStatus::Unsupported, std::nullopt,
            "A2/A3 narrowing tile.cast scratch requires a 2D source tile"};
  }
  const auto rows_ci = ir::As<ir::ConstInt>(valid_shape[0]);
  const auto cols_ci = ir::As<ir::ConstInt>(valid_shape[1]);
  const auto src_cols_ci = ir::As<ir::ConstInt>(src_shape[1]);
  if (!rows_ci || !cols_ci || !src_cols_ci) {
    return {OpScratchStatus::Unknown, std::nullopt,
            "A2/A3 non-saturating narrowing tile.cast requires static source shape "
            "and valid_shape to size its scratch tile"};
  }
  // Match PTOAS makeTCvtTmpType. INT8 [1, bytes] expresses the byte capacity
  // as element geometry; it does not constrain a caller-provided tmp tile.
  const int64_t rows = rows_ci->value_;
  const int64_t cols = cols_ci->value_;
  const int64_t src_cols = src_cols_ci->value_;
  int64_t bytes = 0;
  if (*source_dtype == DataType::FP32) {
    if (rows > 0 && cols > 0) {
      const int64_t head = kTcvtHeadBlockBytes * kTcvtHeadBlockCols *
                           std::min<int64_t>(cols / kTcvtHeadBlockCols, kTcvtHeadMaxBlocks);
      const int64_t remainder = cols % kTcvtHeadBlockCols;
      const int64_t tail =
          remainder == 0
              ? 0
              : kTcvtTailRowBytes * ((std::min<int64_t>(rows, kTcvtHeadMaxBlocks) - 1) * (src_cols / 8) +
                                     CeilDivI64(remainder, 8));
      bytes = std::max(head, tail);
    }
  } else if (*source_dtype == DataType::FP16 && cols > 0) {
    const int64_t width = std::min<int64_t>(cols, kTcvtHeadBlockCols);
    const int64_t half_to_i16 = kTcvtTailRowBytes * CeilDivI64(width, 8);
    const int64_t half_to_i8 =
        std::max(half_to_i16, kTcvtFp16HalfToI8Base + kTcvtTailRowBytes * CeilDivI64(width, 16));
    bytes = (dst == DataType::INT8 || dst == DataType::UINT8) ? half_to_i8 : half_to_i16;
  }
  bytes = std::max<int64_t>(kTcvtMinScratchBytes,
                            CeilDivI64(bytes, kTcvtScratchAlignBytes) * kTcvtScratchAlignBytes);
  return {OpScratchStatus::Required, TileScratchSpec{DataType::INT8, 1, bytes}, ""};
}
// Direct FP4 TCVT requires static valid columns, while alloc_tile deliberately
// carries both valid axes as operands. A full-size, zero-offset subview refines
// only that metadata: PTOAS resolves it to the same address and retains the
// runtime valid-row operand. A treshape cannot carry that operand, and
// set_validshape requires both axes dynamic, so neither can express this view.
static std::pair<std::string, std::string> MakeFp4CastValidView(const ExprPtr& expr,
                                                                codegen::PTOCodegen& codegen,
                                                                const std::string& hint) {
  const auto tile_type = As<ir::TileType>(expr->GetType());
  INTERNAL_CHECK_SPAN(tile_type && tile_type->shape_.size() == 2 && tile_type->memory_space_.has_value(),
                      expr->span_)
      << "Internal error: direct FP4 cast requires a 2D tile with a memory space";
  const auto rows = As<ir::ConstInt>(tile_type->shape_[0]);
  const auto cols = As<ir::ConstInt>(tile_type->shape_[1]);
  INTERNAL_CHECK_SPAN(rows && cols, expr->span_)
      << "Internal error: direct FP4 cast requires static physical tile dimensions";
  auto components = codegen::ExtractTileTypeInfo(*tile_type, codegen.GetTypeString(tile_type->dtype_));
  const auto valid_shape = ir::GetValidShape(tile_type);
  INTERNAL_CHECK_SPAN(valid_shape.size() == 2, expr->span_)
      << "Internal error: direct FP4 cast requires two valid extents";
  const auto valid_row = As<ir::ConstInt>(valid_shape[0]);
  const auto valid_col = As<ir::ConstInt>(valid_shape[1]);
  CHECK_SPAN(valid_col && valid_col->value_ > 0, expr->span_)
      << "Direct FP4-to-FP8 cast requires a positive compile-time valid column extent";
  INTERNAL_CHECK_SPAN(!valid_row || valid_row->value_ > 0, expr->span_)
      << "Internal error: empty direct FP4 casts must be handled before valid-view emission";
  components.v_row_dynamic = valid_row == nullptr;
  if (valid_row) components.v_row = valid_row->value_;
  components.v_col_dynamic = false;
  components.v_col = valid_col->value_;
  // All packed source geometry already counts byte carriers. Only the
  // numerical cast boundary expands the destination into FP8 values.
  const auto view_type = codegen::FormatTileBufTypeString(
      codegen::MemorySpaceToMLIR(*tile_type->memory_space_), components.dtype_str, components.rows,
      components.cols, components.blayout, components.slayout, components.fractal, components.pad,
      components.compact, components.v_row, components.v_col, components.v_row_dynamic,
      components.v_col_dynamic);
  const auto source = codegen.GetExprAsCode(expr);
  const auto source_type = codegen.GetExprTypeAnnotation(expr);
  const auto row_operand =
      valid_row ? codegen.GetOrEmitConstant(components.v_row, DataType::INDEX)
                : codegen.EmitCastToIndex(valid_shape[0], codegen.GetExprAsCode(valid_shape[0]));
  const auto col_operand = codegen.GetOrEmitConstant(components.v_col, DataType::INDEX);
  const auto zero = codegen.GetOrEmitConstant(int64_t{0}, DataType::INDEX);
  const auto view = codegen.NewNamedTemp(hint);
  codegen.Emit(view + " = pto.subview " + source + "[" + zero + ", " + zero + "] sizes [" +
               std::to_string(components.rows) + ", " + std::to_string(components.cols) + "] valid [" +
               row_operand + ", " + col_operand + "] : " + source_type + " -> " + view_type);
  codegen.RegisterTileBufType(view, view_type);
  codegen.RegisterTileViewName(view);
  return {view, view_type};
}

// A2/A3 scratch-backed tcvt requires static validity in both forms: level3
// verifies an explicit tmp, while level2 sizes an implicit tmp in PlanMemory.
// alloc_tile types keep v_row=?, v_col=?, so bridge src/dst without changing
// their allocation handles or allocating scratch on behalf of level2.
//
// Both forms carry the same config attr-dict, so the rounding mode and the
// destination saturation are rendered once, before the form splits. `satmode` is
// always stamped: the IR records only a deviation from the destination's default,
// and `GetEmittedSaturationMode` resolves both the default and the "whatever the
// target does" case a float destination leaves open. Omitting the attribute
// instead would hand the decision to the assembler, whose own default for an
// omitted `satmode` flipped OFF -> ON in PTOAS v0.63.
std::string MakeTcvtCodegenPTO(const Backend& backend, CastTarget target, const CallPtr& op,
                               codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK_SPAN(op->args_.size() == 1 || op->args_.size() == 2, op->span_)
      << "tile.cast requires 1 or 2 arguments (src[, tmp]), but got " << op->args_.size();

  if (codegen.EmitTileAddr() && ir::op_predicates::CastRequiresDisjointSource(
                                    op, codegen.GetBackendHandler()->GetPtoTargetArch() == "a2a3")) {
    const auto source_type = As<ir::TileType>(op->args_[0]->GetType());
    const auto result = codegen.GetCurrentResultVar();
    const auto result_type = result ? As<ir::TileType>(result->GetType()) : nullptr;
    INTERNAL_CHECK_SPAN(source_type && result_type, op->span_)
        << "Internal error: tile.cast requires source and result TileTypes before PTO codegen";
    const auto source_memref = source_type->memref_.value_or(nullptr);
    const auto result_memref = result_type->memref_.value_or(nullptr);
    INTERNAL_CHECK_SPAN(result_memref, op->span_)
        << "Internal error: tile.cast requires a result MemRef before PTO codegen";
    // TPOP tiles and their views are backed by reserved pipe storage, not a
    // general-pool MemRef. Their runtime-owned address cannot be compared here.
    if (source_memref) {
      // At level3 offsets are physical addresses, even for distinct allocation
      // bases. Compare complete byte windows, not just base or pointer equality.
      ir::arith::Analyzer analyzer;
      const auto distance = As<ir::ConstInt>(analyzer.Simplify(std::make_shared<ir::Sub>(
          result_memref->byte_offset_, source_memref->byte_offset_, DataType::INT64, op->span_)));
      bool overlaps = ir::MemRef::MayAlias(source_memref, result_memref);
      if (distance) {
        // Subtracting a shared symbolic address also exposes a constant distance.
        overlaps = distance->value_ >= 0
                       ? static_cast<uint64_t>(distance->value_) < source_memref->size_
                       : static_cast<uint64_t>(-(distance->value_ + 1)) + 1 < result_memref->size_;
      }
      CHECK_SPAN(!overlaps, op->span_)
          << "tile.cast from " << source_type->dtype_.ToString() << " to " << result_type->dtype_.ToString()
          << " requires non-overlapping source and destination memory on "
          << codegen.GetBackendHandler()->GetPtoTargetArch();
    }
  }

  const int mode = op->GetKwarg<int>("mode");
  INTERNAL_CHECK_SPAN(mode >= 0 && mode < static_cast<int>(round_modes.size()), op->span_)
      << "Internal error: tile.cast round mode out of range: " << mode;
  std::string config_attr = "{rmode = #pto<round_mode " + round_modes.at(mode) + ">";
  const int saturation_mode = ir::GetEmittedSaturationMode(op);
  INTERNAL_CHECK_SPAN(ir::IsValidSaturationMode(saturation_mode), op->span_)
      << "Internal error: tile.cast saturation_mode out of range: " << saturation_mode;
  config_attr += ", satmode = #pto<saturation_mode " + ir::SaturationModeToPTOString(saturation_mode) + ">";
  config_attr += "}";

  const auto source_type = As<ir::TileType>(op->args_[0]->GetType());
  const auto result = codegen.GetCurrentResultVar();
  const auto result_type = result ? As<ir::TileType>(result->GetType()) : nullptr;
  if (op->args_.size() == 1 && codegen.GetBackendHandler()->SupportsFp4CastToNZ() && source_type &&
      result_type && source_type->dtype_ == DataType::FP4E2M1X2 &&
      result_type->dtype_ == DataType::FP8E4M3FN) {
    const auto source_valid = ir::GetValidShape(source_type);
    const auto result_valid = ir::GetValidShape(result_type);
    INTERNAL_CHECK_SPAN(source_valid.size() == 2 && result_valid.size() == 2, op->span_)
        << "Internal error: direct FP4 cast requires two source and result valid extents";
    ir::arith::Analyzer analyzer;
    const auto source_row = As<ir::ConstInt>(analyzer.Simplify(source_valid[0]));
    const auto result_row = As<ir::ConstInt>(analyzer.Simplify(result_valid[0]));
    const bool source_empty = source_row && source_row->value_ == 0;
    const bool result_empty = result_row && result_row->value_ == 0;
    if (source_empty || result_empty) {
      INTERNAL_CHECK_SPAN(source_empty && result_empty, op->span_)
          << "Internal error: direct FP4 cast source and result must agree on zero valid rows";
      // A statically empty conversion reads and writes nothing. Keep its
      // allocated handles for downstream consumers, avoiding an ISA Tile with
      // unsupported staticRow=0. Runtime rows, including zero, stay dynamic.
      return "";
    }
    const auto source_view = MakeFp4CastValidView(op->args_[0], codegen, "tcvt_src_view");
    const auto result_view = MakeFp4CastValidView(result, codegen, "tcvt_dst_view");
    codegen.Emit("pto.tcvt ins(" + source_view.first + " " + config_attr + " : " + source_view.second +
                 ") outs(" + result_view.first + " : " + result_view.second + ")");
    return "";
  }

  auto src_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(src_type, op->span_) << "Internal error: tile.cast source must be TileType";
  const bool explicit_tmp = op->args_.size() == 2;
  const OpScratchRequest request{src_type->dtype_, std::nullopt, op->GetKwarg<DataType>("target_type"),
                                 saturation_mode == static_cast<int>(ir::SaturationMode::kOn), src_type};
  // Synthesis geometry does not constrain caller-owned scratch. In particular,
  // rank-one sources with explicit tmp are emitted as PTO [1, N] tiles.
  const auto plan = explicit_tmp ? std::nullopt : backend.PlanOpScratch("tile.cast", request);
  if (!explicit_tmp) {
    INTERNAL_CHECK_SPAN(plan, op->span_) << "Internal error: tile.cast scratch rule is not registered";
    CHECK_SPAN(plan->status != OpScratchStatus::Unsupported, op->span_) << plan->reason;
  }
  // Unknown is never a no-scratch verdict. The static-view checks below retain
  // source-located diagnostics for runtime validity; explicit tmp remains caller-owned.
  const bool needs_static_views =
      explicit_tmp ? target == CastTarget::A2A3 : plan->status != OpScratchStatus::NotRequired;
  if (needs_static_views) {
    auto dst_var = codegen.GetCurrentResultVar();
    auto dst_type = dst_var ? ir::As<ir::TileType>(dst_var->GetType()) : nullptr;
    INTERNAL_CHECK_SPAN(dst_type, op->span_) << "Internal error: tile.cast result must be TileType";
    RequireStaticValidShapeForPtoas(src_type, "tile.cast", "src", op->span_);
    RequireStaticValidShapeForPtoas(dst_type, "tile.cast", "dst", op->span_);
    if (!explicit_tmp) {
      CHECK_SPAN(plan->status == OpScratchStatus::Required, op->span_) << plan->reason;
    }

    const std::string src_ssa = EnsureStaticViewTileSsa(op->args_[0], codegen, "tcvt_src_view");
    std::vector<std::pair<std::string, std::string>> ins = {
        {src_ssa, GetTileViewTypeAnnotation(op->args_[0], codegen)}};
    if (explicit_tmp) {
      auto tmp_type = ir::As<ir::TileType>(op->args_[1]->GetType());
      INTERNAL_CHECK_SPAN(tmp_type, op->span_) << "Internal error: tile.cast tmp must be TileType";
      RequireStaticValidShapeForPtoas(tmp_type, "tile.cast", "tmp", op->args_[1]->span_);
      const std::string tmp_ssa = EnsureStaticViewTileSsa(op->args_[1], codegen, "tcvt_tmp_view");
      ins.emplace_back(tmp_ssa, GetTileViewTypeAnnotation(op->args_[1], codegen));
    }
    const std::string dst_ssa = EnsureStaticViewTileSsa(dst_var, codegen, "tcvt_dst_view");
    EmitInsOutsWithViewTypes(codegen, "pto.tcvt", ins, dst_ssa, dst_type, config_attr);
    return "";
  }
  codegen.Emit("pto.tcvt " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

}  // namespace

void RegisterCastOps(Backend& backend, BackendType backend_type) {
  CastTarget target;
  switch (backend_type) {
    case BackendType::Ascend910:
      target = CastTarget::A2A3;
      break;
    case BackendType::Ascend950:
      target = CastTarget::A5;
      break;
    default:
      INTERNAL_CHECK(false) << "Internal error: no cast rules for backend";
      return;
  }
  // pto.tcvt mis-orders col_major source elements; both sides must be row_major.
  backend.RegisterOp("tile.cast")
      .f_plan_scratch([target](const OpScratchRequest& request) { return PlanCastScratch(target, request); })
      .f_codegen([&backend, target](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeTcvtCodegenPTO(backend, target, op, codegen);
      })
      .set_input_layout(0, ir::TileLayout::row_major)
      .set_output_layout(ir::TileLayout::row_major);

  // Direct NZ casts share conversion semantics but must not inherit the
  // ordinary cast's row-major destination constraint.
  backend.RegisterOp("tile.cast_to_nz")
      .f_codegen([&backend, target](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        CHECK_SPAN(AsPto(codegen).GetBackendHandler()->SupportsFp4CastToNZ(), op->span_)
            << "tile.cast_to_nz is unsupported by this backend";
        return MakeTcvtCodegenPTO(backend, target, op, codegen);
      })
      .set_input_layout(0, ir::TileLayout::row_major);
}

}  // namespace pypto::backend
