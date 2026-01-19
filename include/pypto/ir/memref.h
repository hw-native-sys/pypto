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

#ifndef PYPTO_IR_MEMREF_H_
#define PYPTO_IR_MEMREF_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declarations
class Expr;
using ExprPtr = std::shared_ptr<const Expr>;

/**
 * @brief Memory space enumeration
 *
 * Defines the available memory spaces in the hardware hierarchy:
 * - DDR: Double Data Rate memory (off-chip)
 * - UB: Unified Buffer (on-chip shared memory)
 * - L1: L1 cache
 * - L0A: L0A buffer (matrix A operand)
 * - L0B: L0B buffer (matrix B operand)
 * - L0C: L0C buffer (matrix C/result)
 */
enum class MemorySpace {
  DDR,  ///< DDR memory (off-chip)
  UB,   ///< Unified Buffer (on-chip)
  L1,   ///< L1 cache
  L0A,  ///< L0A buffer
  L0B,  ///< L0B buffer
  L0C   ///< L0C buffer
};

/**
 * @brief Memory reference for shaped types (tensor and tile)
 *
 * Represents memory allocation information for ShapedType instances.
 * Tracks memory space, address, and size.
 * This is a plain struct (not an IRNode) that is embedded in ShapedType.
 */
struct MemRef {
  MemorySpace memory_space_;  ///< Memory space (DDR, UB, L1, etc.)
  ExprPtr addr_;              ///< Starting address (expression)
  uint64_t size_;             ///< Size in bytes (64-bit unsigned)

  /**
   * @brief Default constructor for aggregate initialization
   */
  MemRef() = default;

  /**
   * @brief Constructor with all parameters
   *
   * @param memory_space Memory space (DDR, UB, L1, etc.)
   * @param addr Starting address expression
   * @param size Size in bytes
   */
  MemRef(MemorySpace memory_space, ExprPtr addr, uint64_t size)
      : memory_space_(memory_space), addr_(std::move(addr)), size_(size) {}

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(reflection::UsualField(&MemRef::memory_space_, "memory_space"),
                           reflection::UsualField(&MemRef::addr_, "addr"),
                           reflection::UsualField(&MemRef::size_, "size"));
  }
};

/**
 * @brief Convert MemorySpace enum to string
 *
 * @param space Memory space enum value
 * @return String representation
 */
std::string MemorySpaceToString(MemorySpace space);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMREF_H_
