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
 * @brief Memory reference variable for shaped types (tensor and tile)
 *
 * Represents a memory reference combining an allocation identity (base Ptr),
 * a byte offset within that allocation, and a size.
 *
 * - base_: VarPtr to the Ptr variable from tile.alloc/tensor.alloc (allocation identity)
 * - byte_offset_: byte offset from base (0 for root alloc, computed for views)
 * - size_: size in bytes of this memory region
 *
 * Aliasing is determined by comparing base_ pointers (SameAllocation) and
 * checking for overlapping byte ranges (MayAlias).
 */
class MemRef : public Var {
 public:
  VarPtr base_;          ///< Ptr variable from alloc — allocation identity token
  ExprPtr byte_offset_;  ///< Byte offset from base (0 for full alloc, view offset for views)
  uint64_t size_;        ///< Size in bytes of this MemRef

  /**
   * @brief Construct MemRef from base pointer, expression offset, and size.
   * Name is derived from the base Ptr's name.
   */
  MemRef(VarPtr base, ExprPtr byte_offset, uint64_t size, Span span = Span::unknown());

  /**
   * @brief Convenience: construct with integer byte_offset (auto-wrapped in ConstInt).
   */
  MemRef(VarPtr base, int64_t byte_offset, uint64_t size, Span span = Span::unknown());

  /**
   * @brief Construct with explicit variable name. Used by deserialization and
   * address allocation where the name must be preserved exactly.
   */
  MemRef(std::string name, VarPtr base, ExprPtr byte_offset, uint64_t size, Span span = Span::unknown());

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MemRef; }
  [[nodiscard]] std::string TypeName() const override { return "MemRef"; }

  /// Are two MemRefs from the same allocation? (compare base_ Ptr identity)
  static bool SameAllocation(const MemRefPtr& a, const MemRefPtr& b) {
    return a->base_.get() == b->base_.get();
  }

  /// Do two MemRefs potentially alias? (same base + overlapping byte ranges)
  static bool MayAlias(const MemRefPtr& a, const MemRefPtr& b);

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Var::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&MemRef::base_, "base"),
                                          reflection::UsualField(&MemRef::byte_offset_, "byte_offset"),
                                          reflection::UsualField(&MemRef::size_, "size")));
  }
};

using MemRefPtr = std::shared_ptr<const MemRef>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMREF_H_
