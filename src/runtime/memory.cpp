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

#include "pypto/runtime/memory.h"

namespace pypto {
namespace runtime {

void SharedMemory::Write(int64_t addr, const Value& value) {
  std::lock_guard<std::mutex> lock(mutex_);
  memory_[addr] = value;
}

Value SharedMemory::Read(int64_t addr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = memory_.find(addr);
  if (it == memory_.end()) {
    return std::monostate{};  // Return empty value if not found
  }
  return it->second;
}

bool SharedMemory::Contains(int64_t addr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return memory_.find(addr) != memory_.end();
}

void SharedMemory::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  memory_.clear();
}

}  // namespace runtime
}  // namespace pypto
