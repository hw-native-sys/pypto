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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/op_rules.h"
#include "pypto/backend/common/pto_ops_common.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/type.h"
#include "src/backend/common/pto_ops_internal.h"

namespace pypto::backend {
namespace {

// These differences are local to the select family. They are captured during
// backend registration, rather than queried via a growing handler interface.
enum class SelectTarget { A2A3, A5 };
enum class SelectForm { Tile, Scalar };

bool SupportsSelect(SelectTarget target, SelectForm form, DataType dtype) {
  if (dtype == DataType::BF16) return form == SelectForm::Tile;
  if (dtype == DataType::INT8 || dtype == DataType::UINT8) return target == SelectTarget::A5;
  return dtype == DataType::INT16 || dtype == DataType::UINT16 || dtype == DataType::INT32 ||
         dtype == DataType::UINT32 || dtype == DataType::FP16 || dtype == DataType::FP32;
}

OpScratchPlan PlanSelectScratch(SelectTarget target, SelectForm form, const OpScratchRequest& request) {
  if (request.source_dtype && !SupportsSelect(target, form, *request.source_dtype)) {
    return {OpScratchStatus::Unsupported, std::nullopt, "Select source dtype is not supported"};
  }
  if (form == SelectForm::Tile) {
    const auto spec = target == SelectTarget::A5 ? TileScratchSpec{DataType::UINT8, 1, 32}
                                                 : TileScratchSpec{DataType::UINT32, 1, 16};
    return {OpScratchStatus::Required, spec, ""};
  }
  if (!request.source_dtype) {
    return {OpScratchStatus::Unknown, std::nullopt, "TSELS source dtype is required"};
  }
  if (target == SelectTarget::A5) {
    return {OpScratchStatus::Required, TileScratchSpec{DataType::UINT8, 1, 32}, ""};
  }
  if (!request.source_cols) {
    return {OpScratchStatus::Unknown, std::nullopt, "A2/A3 TSELS requires physical source columns"};
  }
  if (*request.source_cols <= 0) {
    return {OpScratchStatus::Unsupported, std::nullopt, "Physical source columns must be positive"};
  }
  return {OpScratchStatus::Required, TileScratchSpec{*request.source_dtype, 1, *request.source_cols}, ""};
}

void ValidateSels(SelectTarget target, const ir::CallPtr& op, const ir::TypePtr& result_type) {
  pto_ops_detail::CheckArity(op, "pto.tsels", 4);
  const auto mask = ir::As<ir::TileType>(op->args_[0]->GetType());
  const auto src = ir::As<ir::TileType>(op->args_[1]->GetType());
  const auto tmp = ir::As<ir::TileType>(op->args_[2]->GetType());
  const auto dst = ir::As<ir::TileType>(result_type);
  INTERNAL_CHECK_SPAN(mask && src && tmp && dst, op->span_)
      << "Internal error: tile.sels requires tile operands and an assignment target";
  CHECK_SPAN(SupportsSelect(target, SelectForm::Scalar, src->dtype_), op->span_)
      << "tile.sels with integer src dtype " << src->dtype_.ToString()
      << " is only supported on the 'a5' backend; A2/A3 supports 16/32-bit integers, FP16, and FP32";

  const std::vector<std::pair<std::string_view, ir::TileTypePtr>> operands = {
      {"mask", mask}, {"src", src}, {"tmp", tmp}, {"dst", dst}};
  // Extract each region immediately after checking its optional MemRef. Later
  // alias checks operate only on these validated regions.
  std::vector<ir::MemRefPtr> regions;
  regions.reserve(operands.size());
  for (const auto& [name, type] : operands) {
    INTERNAL_CHECK_SPAN(type->memref_.has_value(), op->span_)
        << "Internal error: tile.sels " << name << " must carry a MemRef before PTO codegen";
    regions.push_back(*type->memref_);
  }
  CHECK_SPAN(!ir::MemRef::MayAlias(regions[0], regions[3]), op->span_)
      << "tile.sels requires mask and dst to use non-overlapping memory regions";
  if (target == SelectTarget::A2A3) {
    for (const size_t other : {size_t{0}, size_t{1}}) {
      CHECK_SPAN(!ir::MemRef::MayAlias(regions[2], regions[other]), op->span_)
          << "tile.sels on A2/A3 requires tmp not to overlap mask or src, but tmp overlaps "
          << operands[other].first;
    }
  }
  // Preserve caller-owned scratch verbatim. The synthesis recipe above is not
  // an exact-shape/dtype requirement on explicit placeholders (in particular,
  // A5 accepts other scratch dtypes). Existing PTOAS geometry checks remain at
  // the assembler boundary; this refactor moves the existing compiler checks.
}

void RegisterSelect(Backend& backend, SelectTarget target, SelectForm form) {
  const bool scalar = form == SelectForm::Scalar;
  auto entry = backend.RegisterOp(scalar ? "tile.sels" : "tile.sel");
  entry.f_plan_scratch(
      [target, form](const OpScratchRequest& request) { return PlanSelectScratch(target, form, request); });
  if (scalar) {
    entry.f_validate([target](const ir::CallPtr& op, const ir::TypePtr& result_type) {
      ValidateSels(target, op, result_type);
    });
  }
  // TSEL had no target-specific emission check. Its candidate dtype gate is
  // now part of scratch planning in composite lowering; do not add unrelated
  // restrictions to direct TSEL calls while moving the existing rule source.
  entry.f_codegen([scalar](const ir::CallPtr& op, codegen::CodegenBase& base) {
    auto& codegen = pto_ops_detail::AsPto(base);
    const std::string name = scalar ? "pto.tsels" : "pto.tsel";
    pto_ops_detail::CheckArity(op, name, 4);
    codegen.Emit(name + " " + pto_ops_detail::GenerateInsOutsClause(op, codegen));
    return std::string{};
  });
  for (size_t i = 0; i < 4; ++i) entry.set_input_layout(i, ir::TileLayout::row_major);
  entry.set_output_layout(ir::TileLayout::row_major);
}

}  // namespace

void RegisterSelectOps(Backend& backend, BackendType backend_type) {
  SelectTarget target;
  switch (backend_type) {
    case BackendType::Ascend910:
      target = SelectTarget::A2A3;
      break;
    case BackendType::Ascend950:
      target = SelectTarget::A5;
      break;
    default:
      INTERNAL_CHECK(false) << "Internal error: no select rules for backend";
      return;
  }
  RegisterSelect(backend, target, SelectForm::Tile);
  RegisterSelect(backend, target, SelectForm::Scalar);
}

}  // namespace pypto::backend
