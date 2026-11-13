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

/// Result of comparing a dense window view's byte size with
/// the logical size of the allocation that backs it.
enum class WindowViewFit {
  kFits,         ///< The view's size is statically known to fit the slot.
  kViolation,    ///< The view is invalid or statically exceeds the slot.
  kUnverifiable  ///< The view or slot is symbolic; a host-side guard must check it.
};

struct WindowViewFitResult {
  WindowViewFit fit = WindowViewFit::kUnverifiable;
  std::optional<int64_t> view_bytes;
  std::optional<int64_t> slot_bytes;
  /// Set when the view or slot itself is malformed (non-positive extent/size).
  std::string invalid_reason;
};

/**
 * @brief Compare a dense window view's byte size with its backing slot.
 *
 * ``pld.window`` declares a dense view, so its size is
 * ``product(shape) * dtype_bytes``. Static expressions are const-folded with
 * checked int64 arithmetic. The slot's logical size excludes physical padding.
 *
 * @param view Dense distributed window view type.
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
