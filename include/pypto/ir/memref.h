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
#include <string>
#include <tuple>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

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
 * @brief Convert MemorySpace enum to string
 *
 * @param space Memory space enum value
 * @return String representation
 */
std::string MemorySpaceToString(MemorySpace space);

/**
 * @brief Memory reference variable for shaped types (tensor and tile)
 *
 * Represents a memory allocation with metadata (space, address, size, id).
 * Inherits from Var, making it a first-class IR expression that can be
 * declared and referenced like other variables.
 *
 * Memory references have auto-generated names based on their ID (e.g., "mem_123")
 * and MemRefType as their type.
 */
class MemRef : public Var {
 public:
  MemorySpace memory_space_;  ///< Memory space (DDR, UB, L1, etc.)
  ExprPtr addr_;              ///< Starting address expression
  uint64_t size_;             ///< Size in bytes (64-bit unsigned)
  uint64_t id_;               ///< Unique identifier (used for name generation)

  /**
   * @brief Constructor with all parameters including explicit ID
   *
   * Generates a variable name from the ID (e.g., "mem_123") and creates
   * a MemRefType for the type. Calls Var constructor with these values.
   *
   * @param memory_space Memory space (DDR, UB, L1, etc.)
   * @param addr Starting address expression
   * @param size Size in bytes
   * @param id Unique identifier (used to generate variable name)
   * @param span Source location (defaults to Span::unknown())
   */
  MemRef(MemorySpace memory_space, ExprPtr addr, uint64_t size, uint64_t id, Span span = Span::unknown());

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MemRef; }
  [[nodiscard]] std::string TypeName() const override { return "MemRef"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Var::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&MemRef::memory_space_, "memory_space"),
                                          reflection::UsualField(&MemRef::addr_, "addr"),
                                          reflection::UsualField(&MemRef::size_, "size"),
                                          reflection::UsualField(&MemRef::id_, "id")));
  }
};

using MemRefPtr = std::shared_ptr<const MemRef>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMREF_H_
