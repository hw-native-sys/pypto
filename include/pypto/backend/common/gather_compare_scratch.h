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

#ifndef PYPTO_BACKEND_COMMON_GATHER_COMPARE_SCRATCH_H_
#define PYPTO_BACKEND_COMMON_GATHER_COMPARE_SCRATCH_H_

#include <cstdint>
#include <limits>
#include <optional>

namespace pypto::backend {

/// A2/A3 TGATHER places indices and converted thresholds immediately after its
/// mask descriptor. The caller must reserve all three regions, while the native
/// mask descriptor itself must retain one aligned mask row per source row.
struct GatherCompareScratch {
  int64_t rows;
  int64_t mask_cols;
  int64_t bytes;
};

inline std::optional<GatherCompareScratch> PlanGatherCompareScratch(int64_t rows, int64_t cols,
                                                                    uint64_t element_bytes) {
  constexpr auto limit = std::numeric_limits<int64_t>::max();
  if (rows <= 0 || cols <= 0 || (element_bytes != 2 && element_bytes != 4) || cols > limit - 255) {
    return std::nullopt;
  }
  const auto width = static_cast<int64_t>(element_bytes);
  const int64_t mask_cols = ((cols + 255) / 256) * 32;
  // The converted threshold vector has a complete 32-byte final block.
  if (cols > (limit - mask_cols - 32) / 4 ||
      rows > (limit - mask_cols + 1) / (mask_cols + cols * 4 + width)) {
    return std::nullopt;
  }
  const int64_t raw_bytes = rows * (mask_cols + cols * 4 + width);
  const int64_t bytes = ((raw_bytes + mask_cols - 1) / mask_cols) * mask_cols;
  return GatherCompareScratch{rows, mask_cols, bytes};
}

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_GATHER_COMPARE_SCRATCH_H_
