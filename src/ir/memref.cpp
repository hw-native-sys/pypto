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

#include "pypto/ir/memref.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

bool IsTileMoveEverSupported(MemorySpace src, MemorySpace dst) {
  // Mirrors PTOAS's `TMovOp::verify` address-space table, unioned over targets.
  // PTOAS is authoritative; this copy exists so IR-level code can reject an
  // unimplementable move against the user's own span, and because type
  // deduction runs while parsing, before a backend is selected.
  //
  // Deliberately NOT derived from `SoC::GetMemoryGraph()`. That graph models the
  // memory hierarchy for `Backend::FindMemPath` (which has no production
  // callers) and is not a tmov-legality table: it omits `Acc -> Vec` on
  // Ascend910B, a move this pipeline emits and PTOAS accepts. The two agree on
  // the row that matters here -- nothing moves into `Acc` -- and
  // tests/ut/ir/test_memory_space.py pins that agreement.
  switch (src) {
    case MemorySpace::Mat:
      // MTE1 feeds the cube operand and scale buffers. Mat -> Mat is absent:
      // there is no L1 -> L1 tmov.
      return dst == MemorySpace::Left || dst == MemorySpace::Right || dst == MemorySpace::Bias ||
             dst == MemorySpace::LeftScale || dst == MemorySpace::RightScale;
    case MemorySpace::Vec:
      // Vec -> Mat is A5-only; included because this is the union over targets.
      return dst == MemorySpace::Vec || dst == MemorySpace::Mat;
    case MemorySpace::Acc:
      // FIXPIPE drains L0C outward only.
      return dst == MemorySpace::Mat || dst == MemorySpace::Vec;
    default:
      return false;
  }
  // Note the absent row: nothing has `dst == MemorySpace::Acc`.
}

bool IsTileMoveEverPossibleInto(MemorySpace dst) {
  // Every space that can hold a tile is a candidate source; if none of them
  // reaches `dst`, no pass can ever bridge into it with a `tile.move` and the
  // value has to be created there instead.
  for (MemorySpace src :
       {MemorySpace::Vec, MemorySpace::Mat, MemorySpace::Acc, MemorySpace::Left, MemorySpace::Right,
        MemorySpace::Bias, MemorySpace::LeftScale, MemorySpace::RightScale}) {
    if (IsTileMoveEverSupported(src, dst)) return true;
  }
  return false;
}

std::string MemorySpaceToString(MemorySpace space) {
  switch (space) {
    case MemorySpace::DDR:
      return "DDR";
    case MemorySpace::Vec:
      return "Vec";
    case MemorySpace::Mat:
      return "Mat";
    case MemorySpace::Left:
      return "Left";
    case MemorySpace::Right:
      return "Right";
    case MemorySpace::Acc:
      return "Acc";
    case MemorySpace::Bias:
      return "Bias";
    case MemorySpace::LeftScale:
      return "LeftScale";
    case MemorySpace::RightScale:
      return "RightScale";
    case MemorySpace::ScalarLocal:
      return "ScalarLocal";
    default:
      return "Unknown";
  }
}

MemorySpace StringToMemorySpace(const std::string& str) {
  if (str == "DDR") return MemorySpace::DDR;
  if (str == "Vec") return MemorySpace::Vec;
  if (str == "Mat") return MemorySpace::Mat;
  if (str == "Left") return MemorySpace::Left;
  if (str == "Right") return MemorySpace::Right;
  if (str == "Acc") return MemorySpace::Acc;
  if (str == "Bias") return MemorySpace::Bias;
  if (str == "LeftScale") return MemorySpace::LeftScale;
  if (str == "RightScale") return MemorySpace::RightScale;
  if (str == "ScalarLocal") return MemorySpace::ScalarLocal;
  throw pypto::ValueError("Unknown MemorySpace: " + str);
}

// MemRef implementation
MemRef::MemRef(VarPtr base, ExprPtr byte_offset, uint64_t size, Span span, bool is_pinned,
               uint64_t slot_count, std::optional<ExprPtr> slot_index)
    : Var(base->name_hint_, GetMemRefType(), std::move(span)),
      base_(std::move(base)),
      byte_offset_(std::move(byte_offset)),
      size_(size),
      is_pinned_(is_pinned),
      slot_count_(slot_count),
      slot_index_(std::move(slot_index)) {}

MemRef::MemRef(VarPtr base, int64_t byte_offset, uint64_t size, Span span, bool is_pinned,
               uint64_t slot_count, std::optional<ExprPtr> slot_index)
    // INT64 dtype matches AllocateMemoryAddrPass (which materializes the final
    // concrete address) and the PTOAS dialect's `i64` requirement on the
    // alloc_tile addr operand. Codegen reads dtype from the ConstInt 1:1.
    : MemRef(std::move(base), std::make_shared<ConstInt>(byte_offset, DataType::INT64, Span::unknown()), size,
             std::move(span), is_pinned, slot_count, std::move(slot_index)) {}

MemRef::MemRef(std::string name, VarPtr base, ExprPtr byte_offset, uint64_t size, Span span, bool is_pinned,
               uint64_t slot_count, std::optional<ExprPtr> slot_index)
    : Var(std::move(name), GetMemRefType(), std::move(span)),
      base_(std::move(base)),
      byte_offset_(std::move(byte_offset)),
      size_(size),
      is_pinned_(is_pinned),
      slot_count_(slot_count),
      slot_index_(std::move(slot_index)) {}

bool MemRef::MayAlias(const MemRefPtr& a, const MemRefPtr& b) {
  if (a->base_.get() != b->base_.get()) return false;

  auto off_a = As<ConstInt>(a->byte_offset_);
  auto off_b = As<ConstInt>(b->byte_offset_);
  if (off_a && off_b) {
    int64_t end_a = off_a->value_ + static_cast<int64_t>(a->size_);
    int64_t end_b = off_b->value_ + static_cast<int64_t>(b->size_);
    return off_a->value_ < end_b && off_b->value_ < end_a;
  }
  return true;  // same base, symbolic offsets → conservatively alias
}

}  // namespace ir
}  // namespace pypto
