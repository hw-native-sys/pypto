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

#include "pypto/backend/common/buffer_transpose_recipe.h"

#include <cstdint>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto::backend {
namespace {

const char* OperandError(const std::vector<ir::BufferTypePtr>& types) {
  for (const auto& type : types) {
    if (!IsAlignedDenseVecBuffer(type) || type->dtype_ != DataType::FP32 ||
        type->valid_shape_ != type->shape_) {
      return "requires aligned full-valid rank-2 row-major Vec FP32 operands";
    }
  }
  // TTRANS uses the source type for both ins operands. A larger scratch type
  // cannot be substituted just because it has enough bytes: extents drive the
  // native transpose algorithm. Full validity also excludes undefined padding.
  if (types[1]->shape_ != types[0]->shape_) return "requires scratch matching the source shape";
  if (types[2]->shape_[0] != types[0]->shape_[1] || types[2]->shape_[1] != types[0]->shape_[0]) {
    return "requires destination shape to swap the source dimensions";
  }
  return nullptr;
}

}  // namespace

bool SupportsLogicalBufferTranspose(const ir::CallPtr& call) {
  if (call->args_.size() != 4 || !call->kwargs_.empty()) return false;
  const auto first = ir::As<ir::ConstInt>(call->args_[1]);
  const auto second = ir::As<ir::ConstInt>(call->args_[2]);
  // FlattenTileNdTo2D has already produced a 2D swap; retain both axis orders
  // and the public negative-axis spelling, without accepting arbitrary axes.
  const auto normalized = [](int64_t axis) { return axis < 0 ? axis + 2 : axis; };
  if (!first || !second) return false;
  const auto a = normalized(first->value_);
  const auto b = normalized(second->value_);
  if (!((a == 0 && b == 1) || (a == 1 && b == 0))) return false;
  return OperandError({StaticLogicalBufferDescriptor(call->args_[0]->GetType()),
                       StaticLogicalBufferDescriptor(call->args_[3]->GetType()),
                       StaticLogicalBufferDescriptor(call->GetType())}) == nullptr;
}

void ValidateBufferTransposeOperands(const std::vector<ir::ExprPtr>& args) {
  // Operand count is a compiler invariant; malformed calls may have no source.
  INTERNAL_CHECK_SPAN(args.size() == 3,
                      !args.empty() && args.front() ? args.front()->span_ : ir::Span::unknown())
      << "buffer.transpose requires source, scratch and destination";
  std::vector<ir::BufferTypePtr> types;
  for (const auto& arg : args) {
    CHECK(arg) << "buffer.transpose operands must not be null";
    types.push_back(ir::As<ir::BufferType>(arg->GetType()));
  }
  const auto* error = OperandError(types);
  CHECK(!error) << "buffer.transpose " << error;
}

}  // namespace pypto::backend
