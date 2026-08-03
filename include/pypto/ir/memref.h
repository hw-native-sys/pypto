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

  /// An allocation the author declared (`pl.MemRef("name")`) rather than one the
  /// compiler derived: it is sized to its members and nothing else is packed into
  /// it. True only between the parser and `InitMemRef`, which resolves it into an
  /// ordinary MemRef (real size, flag cleared) plus a `pinned=True` alloc that
  /// carries the isolation from there on. Printed as `pl.MemRef("name")` — the
  /// one-argument form — so the distinction survives a print/parse round trip
  /// without being inferred from `size_`.
  bool is_pinned_;

  /// How many equally-sized slots the allocation holds
  /// (`pl.MemRef("name", slots=N)`); 1 for an unsubscripted declaration.
  uint64_t slot_count_;

  /// Which slot of the allocation this MemRef denotes (`l0c[k]`), as an
  /// expression so the index may be a runtime value (`l0c[i & 1]`). Null when the
  /// declaration is unsubscripted.
  ///
  /// `InitMemRef` resolves both fields into geometry: it sizes one slot to the
  /// largest tile bound to any slot, sizes the allocation to `slot_count_ * slot`,
  /// and turns this index into `byte_offset_ = slot_index_ * slot` (folded to a
  /// `ConstInt` when the index is constant). It does **not** drop them —
  /// `byte_offset_` says where the slot lands, these two still say *which slot of
  /// what* it is, which is what PTO codegen needs to emit ptoas
  /// `pto.alloc_multi_tile` / `pto.multi_tile_get` rather than N unrelated allocs.
  /// `is_pinned_` does clear, so "still to resolve" and "is a slot" stay separate
  /// questions.
  ///
  /// The two are related, not independent: `byte_offset_` is derived from this
  /// index and `AllocateMemoryAddr` may rebase it onto a physical address, so the
  /// index is the author's selection and the offset is its resolved location.
  ///
  /// Because the index may name SSA values, it is the one MemRef field that
  /// substitution must rewrite; `CloneTypeWithMemRefAndRemapExprs` does so, and
  /// only for a pinned MemRef. That restriction keeps rebuilds out of the passes
  /// that key on MemRef pointer identity (`AllocateMemoryAddr`). After InitMemRef
  /// the index is exposed to the same staleness as the SSA values already inside
  /// `byte_offset_`, which substitution likewise leaves alone.
  std::optional<ExprPtr> slot_index_;

  /**
   * @brief Construct MemRef from base pointer, expression offset, and size.
   * Name is derived from the base Ptr's name.
   */
  MemRef(VarPtr base, ExprPtr byte_offset, uint64_t size, Span span = Span::unknown(), bool is_pinned = false,
         uint64_t slot_count = 1, std::optional<ExprPtr> slot_index = std::nullopt);

  /**
   * @brief Convenience: construct with integer byte_offset (auto-wrapped in ConstInt).
   */
  MemRef(VarPtr base, int64_t byte_offset, uint64_t size, Span span = Span::unknown(), bool is_pinned = false,
         uint64_t slot_count = 1, std::optional<ExprPtr> slot_index = std::nullopt);

  /**
   * @brief Construct with explicit variable name. Used by deserialization and
   * address allocation where the name must be preserved exactly.
   */
  MemRef(std::string name, VarPtr base, ExprPtr byte_offset, uint64_t size, Span span = Span::unknown(),
         bool is_pinned = false, uint64_t slot_count = 1, std::optional<ExprPtr> slot_index = std::nullopt);

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
                                          reflection::UsualField(&MemRef::size_, "size"),
                                          reflection::UsualField(&MemRef::is_pinned_, "is_pinned"),
                                          reflection::UsualField(&MemRef::slot_count_, "slot_count"),
                                          reflection::UsualField(&MemRef::slot_index_, "slot_index")));
  }
};

using MemRefPtr = std::shared_ptr<const MemRef>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMREF_H_
