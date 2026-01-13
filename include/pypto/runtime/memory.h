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

#ifndef PYPTO_RUNTIME_MEMORY_H_
#define PYPTO_RUNTIME_MEMORY_H_

#include <cstdint>
#include <map>
#include <mutex>

#include "pypto/runtime/instruction.h"

namespace pypto {
namespace runtime {

/**
 * @brief Shared memory for communication between AICPU and AICORE
 *
 * This is a simple flat memory space simulated as a map.
 * Thread-safe for concurrent access from multiple threads.
 */
class SharedMemory {
 public:
  SharedMemory() = default;

  /**
   * @brief Write a value to memory
   * @param addr Memory address
   * @param value Value to write
   */
  void Write(int64_t addr, const Value& value);

  /**
   * @brief Read a value from memory
   * @param addr Memory address
   * @return Value at address, or monostate if not found
   */
  Value Read(int64_t addr) const;

  /**
   * @brief Check if address has been written
   * @param addr Memory address
   * @return true if address exists in memory
   */
  bool Contains(int64_t addr) const;

  /**
   * @brief Clear all memory
   */
  void Clear();

 private:
  mutable std::mutex mutex_;
  std::map<int64_t, Value> memory_;
};

}  // namespace runtime
}  // namespace pypto

#endif  // PYPTO_RUNTIME_MEMORY_H_
