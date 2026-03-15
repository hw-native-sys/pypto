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
#include "pypto/ir/memory_space.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

/**
 * @brief Memory reference variable for shaped types (tensor and tile)
 *
 * Represents a memory allocation with metadata (address, size, id).
 * Inherits from Var, making it a first-class IR expression that can be
 * declared and referenced like other variables.
 *
 * Memory references have auto-generated names based on their ID (e.g.,
 * "mem_123") and MemRefType as their type. Generated alloc buffers may use a
 * memory-space hint in the name (for example, "mem_vec_7"), but the memory
 * space itself is not stored on MemRef.
 */
class MemRef : public Var {
 public:
  ExprPtr addr_;   ///< Starting address expression
  uint64_t size_;  ///< Size in bytes (64-bit unsigned)
  uint64_t id_;    ///< Unique identifier (used for name generation)

  /**
   * @brief Constructor with auto-generated generic name
   *
   * Generates a variable name from the ID (e.g., "mem_123") and creates
   * a MemRefType for the type. Calls Var constructor with these values.
   *
   * @param addr Starting address expression
   * @param size Size in bytes
   * @param id Unique identifier (used to generate variable name)
   * @param span Source location (defaults to Span::unknown())
   */
  MemRef(ExprPtr addr, uint64_t size, uint64_t id, Span span = Span::unknown());

  /**
   * @brief Constructor with space-specific generated name
   *
   * This is intended for generated alloc buffers whose name should retain the
   * memory-space hint (e.g., "mem_vec_7"). The memory space is used only for
   * naming; it is not stored on MemRef.
   *
   * @param naming_space Memory space used to build the generated variable name
   * @param addr Starting address expression
   * @param size Size in bytes
   * @param id Unique identifier (used to generate variable name)
   * @param span Source location (defaults to Span::unknown())
   */
  MemRef(MemorySpace naming_space, ExprPtr addr, uint64_t size, uint64_t id, Span span = Span::unknown());

  /**
   * @brief Constructor with explicit variable name
   *
   * Used by deserialization and transforms that need to preserve an existing
   * MemRef variable name exactly.
   *
   * @param name Explicit variable name
   * @param addr Starting address expression
   * @param size Size in bytes
   * @param id Unique identifier associated with the MemRef
   * @param span Source location (defaults to Span::unknown())
   */
  MemRef(std::string name, ExprPtr addr, uint64_t size, uint64_t id, Span span = Span::unknown());

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MemRef; }
  [[nodiscard]] std::string TypeName() const override { return "MemRef"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Var::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&MemRef::addr_, "addr"),
                                          reflection::UsualField(&MemRef::size_, "size"),
                                          reflection::UsualField(&MemRef::id_, "id")));
  }
};

using MemRefPtr = std::shared_ptr<const MemRef>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMREF_H_
