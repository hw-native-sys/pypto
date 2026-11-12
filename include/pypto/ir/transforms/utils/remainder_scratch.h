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

#include <algorithm>
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
 * two (TREMS) data areas instead of one. The FP32 high-precision TREM path
 * additionally stages its divisor, quotient and fused-subtraction buffers,
 * whose capacity scales with the number of 64-element vectors per row.
 *
 * `row_stride` is the destination tile's physical row stride in elements and
 * `valid_cols` its valid column count, if statically known. Returns
 * std::nullopt when the row stride is not strictly positive, so callers can
 * fall back to a symbolic check. A symbolic valid-column count is bounded by
 * the physical row stride, so the conservative row-stride value is used for
 * the high-precision budget.
 */
inline std::optional<int64_t> RemainderTmpRequiredCols(bool is_trem, bool is_int32, bool high_precision,
                                                       int64_t row_stride,
                                                       std::optional<int64_t> valid_cols = std::nullopt) {
  if (row_stride <= 0) return std::nullopt;

  const int64_t mask_bytes = (row_stride + 7) / 8;           // 1 bit per element
  const int64_t mask_floats = ((mask_bytes + 31) / 32) * 8;  // 32-byte blocks
  int64_t required = (is_int32 ? (is_trem ? 3 : 2) : 1) * row_stride + mask_floats + 8;

  if (high_precision && !is_int32) {
    const int64_t cap_vecs = std::max<int64_t>(valid_cols.value_or(row_stride) / 64, 1);  // 256-byte vectors
    const int64_t hp_mask_bytes = cap_vecs * 8;
    const int64_t hp_mask_floats = ((hp_mask_bytes + 31) / 32) * 8;
    const int64_t chunk = std::min<int64_t>(cap_vecs, 8);  // TREM_HP_FUSED_CHUNK_A2A3
    const int64_t hp = (is_trem ? 192 : 128) * cap_vecs + hp_mask_floats + 8 + (cap_vecs == 1 ? 64 : 0) +
                       int64_t{8} * 64 * chunk;
    required = std::max(required, hp);
  }
  return required;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_REMAINDER_SCRATCH_H_
