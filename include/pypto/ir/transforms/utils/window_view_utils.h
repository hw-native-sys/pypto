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
#ifndef PYPTO_IR_TRANSFORMS_UTILS_WINDOW_VIEW_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_WINDOW_VIEW_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>

namespace pypto {
namespace ir {

class DistributedTensorType;
class WindowBuffer;

/// Result of comparing a distributed window view's accessible byte span with
/// the logical size of the allocation that backs it.
enum class WindowViewFit {
  kFits,         ///< The view's span is statically known to fit the slot.
  kViolation,    ///< The view's span is statically known to exceed the slot.
  kUnverifiable  ///< The span or slot is symbolic; a host-side guard must check it.
};

struct WindowViewFitResult {
  WindowViewFit fit = WindowViewFit::kUnverifiable;
  std::optional<int64_t> view_bytes;
  std::optional<int64_t> slot_bytes;
};

/**
 * @brief Compare a window view's accessible byte span with its backing slot.
 *
 * The span includes stride gaps:
 * ``(1 + sum((shape_i - 1) * stride_i)) * dtype_bytes``. A canonical packed
 * view therefore reduces to ``product(shape) * dtype_bytes``. Static
 * expressions are const-folded with checked int64 arithmetic. The comparison
 * uses the slot's logical ``size_``, not the cache-line-aligned physical
 * allocation: alignment padding is not addressable through a window view.
 *
 * @param view Distributed view type; ``tensor_view_->stride`` is used when
 *             present, otherwise strides are derived from the layout.
 * @param slot Backing window slot whose logical ``size_`` bounds the view.
 * @return Fit result with the computed view/slot bytes where statically known.
 */
[[nodiscard]] WindowViewFitResult EvaluateWindowViewFit(const DistributedTensorType& view,
                                                        const WindowBuffer& slot);

/**
 * @brief Human-readable description of a window-view fit result.
 *
 * Returns the violation diagnostic for ``kViolation`` and a short
 * "cannot be statically bounded" description for ``kUnverifiable``. The string
 * is empty for ``kFits``.
 */
[[nodiscard]] std::string DescribeWindowViewFit(const std::string& view_name, const WindowBuffer& slot,
                                                const WindowViewFitResult& result);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_WINDOW_VIEW_UTILS_H_
