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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <utility>

#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

std::string MemorySpaceToString(MemorySpace space) {
  switch (space) {
    case MemorySpace::DDR:
      return "DDR";
    case MemorySpace::UB:
      return "UB";
    case MemorySpace::L1:
      return "L1";
    case MemorySpace::L0A:
      return "L0A";
    case MemorySpace::L0B:
      return "L0B";
    case MemorySpace::L0C:
      return "L0C";
    default:
      return "Unknown";
  }
}

// Helper function to convert string to lowercase
static std::string ToLowerCase(const std::string& str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

// MemRef implementation
MemRef::MemRef(MemorySpace memory_space, ExprPtr addr, uint64_t size, uint64_t id, Span span)
    : Var("mem_" + ToLowerCase(MemorySpaceToString(memory_space)) + "_" + std::to_string(id), GetMemRefType(),
          std::move(span)),
      memory_space_(memory_space),
      addr_(std::move(addr)),
      size_(size),
      id_(id) {}

}  // namespace ir
}  // namespace pypto
