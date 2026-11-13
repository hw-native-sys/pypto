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

#include <cstdint>
#include <string>

#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/program.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

WindowViewFitResult EvaluateWindowViewFit(const DistributedTensorType& view, const WindowBuffer& slot) {
  WindowViewFitResult result;

  // The slot's logical size is the addressable extent. Physical allocations
  // are rounded up to a cache line, but that padding is not part of a view.
  const auto slot_bytes = arith::TryEvaluateIntConstant(slot.size_);
  if (slot_bytes) {
    if (*slot_bytes < 0) {
      result.fit = WindowViewFit::kViolation;
      result.invalid_reason = "has a negative backing window slot '" + slot.name_hint_ + "' size";
      return result;
    }
    result.slot_bytes = slot_bytes;
  }

  const int64_t dtype_bytes = static_cast<int64_t>(view.dtype_.GetByte());
  if (dtype_bytes <= 0) {
    result.fit = WindowViewFit::kViolation;
    result.invalid_reason = "has a non-positive element size";
    return result;
  }

  // pld.window creates a dense view from shape and dtype only.
  int64_t elements = 1;
  for (const auto& extent : view.shape_) {
    const auto dim = arith::TryEvaluateIntConstant(extent);
    if (!dim) return result;  // symbolic: a host-side guard must check it
    if (*dim <= 0) {
      result.fit = WindowViewFit::kViolation;
      result.invalid_reason = "has a non-positive static extent";
      return result;
    }
    if (__builtin_mul_overflow(elements, *dim, &elements)) return result;
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
  if (!result.invalid_reason.empty()) return subject + " " + result.invalid_reason;
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
