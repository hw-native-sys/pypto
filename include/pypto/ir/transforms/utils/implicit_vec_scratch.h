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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_IMPLICIT_VEC_SCRATCH_H_
#define PYPTO_IR_TRANSFORMS_UTILS_IMPLICIT_VEC_SCRATCH_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/op_rules.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/cast_saturation.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto::ir {

struct ImplicitVecScratch {
  std::vector<ExprPtr> shape;
  DataType dtype;
  const char* name;
};

// A diagnostic means the target cannot size this workspace yet. Allocation
// requires a definite answer; layout selection can conservatively reserve UB.
using VecScratchQuery = std::variant<std::monostate, ImplicitVecScratch, std::string>;

inline VecScratchQuery QueryImplicitVecScratch(const CallPtr& call, const backend::Backend& target) {
  if (IsOp(call, "tile.cast") && call->args_.size() == 1) {
    auto src = As<TileType>(call->args_[0]->GetType());
    INTERNAL_CHECK(src) << "tile.cast source must be a tile";
    const bool saturating = GetEmittedSaturationMode(call) == static_cast<int>(SaturationMode::kOn);
    const auto plan = target.PlanOpScratch(
        call->op_->name_,
        {src->dtype_, std::nullopt, call->GetKwarg<DataType>("target_type"), saturating, src});
    INTERNAL_CHECK(plan) << "tile.cast scratch rule is not registered";
    if (plan->status == backend::OpScratchStatus::NotRequired) return {};
    if (plan->status != backend::OpScratchStatus::Required) return plan->reason;
    INTERNAL_CHECK(plan->spec) << "required tile.cast scratch has no geometry";
    const auto& spec = *plan->spec;
    return ImplicitVecScratch{
        {MakeIndexConst(spec.rows, call->span_), MakeIndexConst(spec.cols, call->span_)}, spec.dtype, "tcvt"};
  }
  if (IsOp(call, "tile.ci") && call->args_.size() == 2 && target.GetHandler()->RequiresLevel3TmpScratch()) {
    const auto dst = As<TileType>(call->GetType());
    INTERNAL_CHECK(dst) << "tile.ci result must be a tile";
    // A2/A3 level-3 TCI workspace width in FP32 columns.
    const int64_t cols = dst->dtype_.GetBit() == 32 ? 192 : 448;
    return ImplicitVecScratch{
        {MakeIndexConst(1, call->span_), MakeIndexConst(cols, call->span_)}, DataType::FP32, "ci"};
  }
  if (IsOp(call, "tile.sort32") && call->args_.size() == 2) {
    const auto src = As<TileType>(call->args_[0]->GetType());
    INTERNAL_CHECK(src) << "tile.sort32 source must be a tile";
    const auto valid = GetValidShape(src);
    if (valid.size() != 2) return std::string("tile.sort32 source needs a 2D valid_shape");
    const auto cols = As<ConstInt>(valid[1]);
    if (!cols || cols->value_ % 32 != 0) return ImplicitVecScratch{src->shape_, src->dtype_, "sort32"};
  }
  return {};
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_IMPLICIT_VEC_SCRATCH_H_
