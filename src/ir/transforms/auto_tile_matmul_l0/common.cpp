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
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/auto_tile_matmul_l0/internal.h"

namespace pypto {
namespace ir {
namespace auto_tile_matmul_l0 {

int64_t AlignStaticExtent(int64_t extent, int64_t alignment, const Span& span) {
  INTERNAL_CHECK_SPAN(extent > 0 && alignment > 0, span)
      << "Internal error: tile extent/alignment must be positive, got " << extent << "/" << alignment;
  const int64_t remainder = extent % alignment;
  const int64_t increment = remainder == 0 ? 0 : alignment - remainder;
  INTERNAL_CHECK_SPAN(extent <= std::numeric_limits<int64_t>::max() - increment, span)
      << "Internal error: box-aligning tile extent " << extent << " by " << alignment << " overflows int64";
  return extent + increment;
}

/// True if `tile`'s 2D shape is static and its memory space is one of
/// `allowed`.  Operand-source residency check for the L0 tiling rewrite:
///
///   * The right (B) operand must be ``Mat`` — it is loaded from DDR into L1
///     and fed into L0B.
///   * The left (A) operand may be ``Mat`` (the QK pattern) *or* ``Vec`` (the
///     fused-attention ``score·V`` / PV pattern, where the softmax/``exp``
///     output crosses the cube↔vector boundary resident in ``Vec`` rather
///     than ``Mat``).
///
/// This is purely a residency/static-shape check.  A ``Vec`` left operand is
/// not extracted directly: ``BuildKLoopRewrite`` stages it into ``Mat`` first
/// via ``BuildMove(..., Mat)``, so the per-iter ``tile.extract`` always slices from
/// a ``Mat`` source regardless of the original operand space.
bool IsStatic2DInSpaces(const TileTypePtr& tile, std::initializer_list<MemorySpace> allowed, int64_t& out_d0,
                        int64_t& out_d1) {
  if (!tile || tile->shape_.size() != 2) return false;
  auto mem = tile->GetMemorySpace();
  if (!mem.has_value()) return false;
  bool space_ok = false;
  for (auto space : allowed) {
    if (*mem == space) {
      space_ok = true;
      break;
    }
  }
  if (!space_ok) return false;
  auto a = As<ConstInt>(tile->shape_[0]);
  auto b = As<ConstInt>(tile->shape_[1]);
  if (!a || !b) return false;
  out_d0 = a->value_;
  out_d1 = b->value_;
  return true;
}

bool IsStatic2DTuple(const ExprPtr& expr, int64_t dim0, int64_t dim1) {
  auto tuple = As<MakeTuple>(expr);
  if (!tuple || tuple->elements_.size() != 2) return false;
  auto first = As<ConstInt>(tuple->elements_[0]);
  auto second = As<ConstInt>(tuple->elements_[1]);
  return first && second && first->value_ == dim0 && second->value_ == dim1;
}

/// Element width in bytes for a tile dtype.  Returns 0 for sub-byte types
/// (INT4, UINT4, HF4) which the cube path does not support; the caller emits
/// a ``PerfHint`` and skips in that case.
uint32_t DTypeBytes(const DataType& dt) {
  size_t bits = dt.GetBit();
  if (bits % 8 != 0) return 0;
  return static_cast<uint32_t>(bits / 8);
}

void SiblingUseCounter::VisitVarLike_(const VarPtr& op) {
  vars.emplace(op.get(), op);
  ++counts[op.get()];
  if (in_matmul_operand_) ++matmul_operand_uses[op.get()];
}

void SiblingUseCounter::VisitStmt_(const AssignStmtPtr& op) { VisitExpr(op->value_); }

void SiblingUseCounter::VisitExpr_(const CallPtr& op) {
  const bool is_mm = IsOp(op, "tile.matmul");
  const bool is_acc = IsOp(op, "tile.matmul_acc");
  const bool is_bias = IsOp(op, "tile.matmul_bias");
  for (size_t i = 0; i < op->args_.size(); ++i) {
    const bool operand_pos = ((is_mm || is_bias) && (i == 0 || i == 1)) || (is_acc && (i == 1 || i == 2));
    const bool prev = in_matmul_operand_;
    in_matmul_operand_ = operand_pos && (AsVarLike(op->args_[i]) != nullptr);
    VisitExpr(op->args_[i]);
    in_matmul_operand_ = prev;
  }
}

}  // namespace auto_tile_matmul_l0
}  // namespace ir
}  // namespace pypto
