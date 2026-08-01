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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_L0C_FOOTPRINT_H_
#define PYPTO_IR_TRANSFORMS_UTILS_L0C_FOOTPRINT_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto::ir::utils {

/// Physical L0C elements occupied by a logical ``[m, n]`` accumulator.
///
/// L0C may reserve more M rows than the logical cube work shape. Invalid
/// dimensions and arithmetic overflow return nullopt so capacity/allocation
/// callers cannot accidentally guess low.
inline std::optional<uint64_t> L0cPhysicalElements(int64_t m, int64_t n, int64_t m_alignment) {
  if (m <= 0 || n <= 0 || m_alignment <= 0) return std::nullopt;
  const uint64_t um = static_cast<uint64_t>(m);
  const uint64_t un = static_cast<uint64_t>(n);
  const uint64_t ua = static_cast<uint64_t>(m_alignment);
  const uint64_t remainder = um % ua;
  const uint64_t increment = remainder == 0 ? 0 : ua - remainder;
  if (um > std::numeric_limits<uint64_t>::max() - increment) return std::nullopt;
  const uint64_t physical_m = um + increment;
  if (physical_m > std::numeric_limits<uint64_t>::max() / un) return std::nullopt;
  return physical_m * un;
}

/// Physical bytes occupied by a logical ``[m, n]`` L0C accumulator.
inline std::optional<uint64_t> L0cPhysicalBytes(int64_t m, int64_t n, uint64_t element_bytes,
                                                int64_t m_alignment) {
  if (element_bytes == 0) return std::nullopt;
  auto elements = L0cPhysicalElements(m, n, m_alignment);
  if (!elements || *elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
    return std::nullopt;
  }
  return *elements * element_bytes;
}

/// Static physical allocation bytes for a shaped value in ``space``.
///
/// Non-L0C spaces use their logical element count. L0C aligns the logical M
/// dimension (the penultimate dimension; leading dimensions are batch axes)
/// according to the active backend. Dynamic/non-positive shapes and overflow
/// return nullopt.
inline std::optional<uint64_t> StaticPhysicalAllocationBytes(const ShapedTypePtr& type, MemorySpace space,
                                                             const backend::BackendHandler* handler) {
  if (!type) return std::nullopt;
  std::vector<uint64_t> extents;
  extents.reserve(type->shape_.size());
  for (const auto& dim : type->shape_) {
    auto value = As<ConstInt>(dim);
    if (!value || value->value_ <= 0) return std::nullopt;
    extents.push_back(static_cast<uint64_t>(value->value_));
  }

  if (space == MemorySpace::Acc) {
    // L0C is matrix storage. Do not guess a linear footprint for an invalid or
    // not-yet-lowered rank; all supported cube accumulators are at least 2D.
    if (extents.size() < 2) return std::nullopt;
    // Bare pass tests historically support allocation without configuring a
    // backend. Use the strictest alignment among the current backends rather
    // than guessing low: Ascend910B INT32 accumulators occupy 32 physical M
    // rows, while every other currently supported combination uses 16.
    const int64_t fallback_alignment = type->dtype_ == DataType::INT32 ? 32 : 16;
    const int64_t alignment = handler ? handler->GetL0cMAlignment(type->dtype_) : fallback_alignment;
    const uint64_t m = extents[extents.size() - 2];
    const uint64_t n = extents.back();
    if (m > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return std::nullopt;
    }
    auto matrix_bytes =
        L0cPhysicalBytes(static_cast<int64_t>(m), static_cast<int64_t>(n), type->dtype_.GetByte(), alignment);
    if (!matrix_bytes) return std::nullopt;
    uint64_t bytes = *matrix_bytes;
    for (size_t i = 0; i + 2 < extents.size(); ++i) {
      if (bytes > std::numeric_limits<uint64_t>::max() / extents[i]) return std::nullopt;
      bytes *= extents[i];
    }
    return bytes;
  }

  uint64_t elements = 1;
  for (const uint64_t extent : extents) {
    if (elements > std::numeric_limits<uint64_t>::max() / extent) return std::nullopt;
    elements *= extent;
  }
  const uint64_t element_bytes = type->dtype_.GetByte();
  if (element_bytes == 0 || elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
    return std::nullopt;
  }
  return elements * element_bytes;
}

}  // namespace pypto::ir::utils

#endif  // PYPTO_IR_TRANSFORMS_UTILS_L0C_FOOTPRINT_H_
