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
#include "pypto/ir/transforms/utils/window_view_utils.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

WindowViewFitResult EvaluateWindowViewFit(const DistributedTensorType& view, const WindowBuffer& slot) {
  WindowViewFitResult result;

  // The slot's logical size is the addressable extent. Physical allocations
  // are rounded up to a cache line, but that padding is not part of a view.
  const auto slot_bytes = arith::TryEvaluateIntConstant(slot.size_);
  if (slot_bytes && *slot_bytes >= 0) result.slot_bytes = slot_bytes;

  std::vector<ExprPtr> strides;
  TensorLayout layout = TensorLayout::ND;
  if (view.tensor_view_) {
    layout = view.tensor_view_->layout;
    strides = view.tensor_view_->stride;
  }
  if (strides.empty()) strides = tensor_view_semantics::BuildLogicalStridesFromLayout(view.shape_, layout);
  if (strides.size() != view.shape_.size()) return result;

  const int64_t dtype_bytes = static_cast<int64_t>(view.dtype_.GetByte());
  if (dtype_bytes <= 0) return result;

  int64_t elements = 1;
  for (size_t i = 0; i < view.shape_.size(); ++i) {
    const auto dim = arith::TryEvaluateIntConstant(view.shape_[i]);
    const auto stride = arith::TryEvaluateIntConstant(strides[i]);
    // A negative stride is a reversed view whose span this row-major formula
    // does not model; leave it to the runtime guard. A zero stride is a
    // broadcast/alias dimension and contributes no span.
    if (!dim || !stride || *dim <= 0 || *stride < 0) return result;
    int64_t term = 0;
    if (__builtin_mul_overflow(*dim - 1, *stride, &term)) return result;
    if (__builtin_add_overflow(elements, term, &elements)) return result;
  }

  int64_t view_bytes = 0;
  if (__builtin_mul_overflow(elements, dtype_bytes, &view_bytes)) return result;
  result.view_bytes = view_bytes;

  if (!result.slot_bytes) {
    result.fit = WindowViewFit::kUnverifiable;
    return result;
  }
  result.fit = view_bytes > *result.slot_bytes ? WindowViewFit::kViolation : WindowViewFit::kFits;
  return result;
}

std::string DescribeWindowViewFit(const std::string& view_name, const WindowBuffer& slot,
                                  const WindowViewFitResult& result) {
  const std::string subject = view_name.empty() ? "window view" : "window view '" + view_name + "'";
  if (result.fit == WindowViewFit::kViolation && result.view_bytes && result.slot_bytes) {
    return subject + " of " + std::to_string(*result.view_bytes) + " bytes exceeds backing window slot '" +
           slot.name_hint_ + "' of " + std::to_string(*result.slot_bytes) + " bytes";
  }
  if (result.fit == WindowViewFit::kUnverifiable) {
    return subject + " cannot be statically bounded by backing window slot '" + slot.name_hint_ + "'";
  }
  return "";
}

}  // namespace ir
}  // namespace pypto
