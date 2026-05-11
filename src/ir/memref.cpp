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
  if (str == "ScalarLocal") return MemorySpace::ScalarLocal;
  throw pypto::ValueError("Unknown MemorySpace: " + str);
}

// MemRef implementation
MemRef::MemRef(VarPtr base, ExprPtr byte_offset, uint64_t size, Span span)
    : Var(base->name_hint_, GetMemRefType(), std::move(span)),
      base_(std::move(base)),
      byte_offset_(std::move(byte_offset)),
      size_(size) {}

MemRef::MemRef(VarPtr base, int64_t byte_offset, uint64_t size, Span span)
    // INT64 dtype matches AllocateMemoryAddrPass (which materializes the final
    // concrete address) and the PTOAS dialect's `i64` requirement on the
    // alloc_tile addr operand. Codegen reads dtype from the ConstInt 1:1.
    : MemRef(std::move(base), std::make_shared<ConstInt>(byte_offset, DataType::INT64, Span::unknown()), size,
             std::move(span)) {}

MemRef::MemRef(std::string name, VarPtr base, ExprPtr byte_offset, uint64_t size, Span span)
    : Var(std::move(name), GetMemRefType(), std::move(span)),
      base_(std::move(base)),
      byte_offset_(std::move(byte_offset)),
      size_(size) {}

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
