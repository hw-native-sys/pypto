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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_REMAINDER_SCRATCH_H_
#define PYPTO_IR_TRANSFORMS_UTILS_REMAINDER_SCRATCH_H_

#include <cstdint>
#include <optional>

namespace pypto {
namespace ir {

/**
 * @brief Required valid columns of the A2/A3 TREM/TREMS tmp scratch.
 *
 * PTO-ISA lays the scratch out as one flat row of 32-bit elements from the tmp
 * base: the data area(s), a packed compare mask rounded up to a 32-byte block,
 * and an 8-element address buffer. The INT32 paths keep the float-domain
 * dividend, divisor and quotient chain in tmp, so they need three (TREM) or
 * two (TREMS) data areas instead of one.
 *
 * `row_stride` is the destination tile's physical row stride in elements.
 * Returns std::nullopt when the row stride is not strictly positive, so
 * callers can fall back to a symbolic check.
 */
inline std::optional<int64_t> RemainderTmpRequiredCols(bool is_trem, bool is_int32, int64_t row_stride) {
  if (row_stride <= 0) return std::nullopt;

  const int64_t mask_bytes = (row_stride + 7) / 8;           // 1 bit per element
  const int64_t mask_floats = ((mask_bytes + 31) / 32) * 8;  // 32-byte blocks
  return (is_int32 ? (is_trem ? 3 : 2) : 1) * row_stride + mask_floats + 8;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_REMAINDER_SCRATCH_H_
