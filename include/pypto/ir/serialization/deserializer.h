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

#ifndef PYPTO_IR_SERIALIZATION_DESERIALIZER_H_
#define PYPTO_IR_SERIALIZATION_DESERIALIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/core.h"

namespace pypto {
namespace ir {
namespace serialization {

/**
 * @brief Deserializer for IR AST nodes from MessagePack format
 *
 * Deserializes IR AST nodes while preserving pointer sharing and identity.
 * Uses a reference table to track already-deserialized nodes and restore
 * shared pointers correctly.
 */
class IRDeserializer {
 public:
  IRDeserializer();
  ~IRDeserializer();

  /**
   * @brief Deserialize an IR node from MessagePack bytes
   *
   * @param data Vector of bytes containing the MessagePack-encoded data
   * @return The deserialized IR node
   */
  IRNodePtr Deserialize(const std::vector<uint8_t>& data);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Deserialize an IR node from MessagePack bytes
 *
 * Convenience function that creates a deserializer and deserializes the data.
 *
 * @param data Vector of bytes containing the MessagePack-encoded data
 * @return The deserialized IR node
 */
IRNodePtr Deserialize(const std::vector<uint8_t>& data);

/**
 * @brief Deserialize an IR node from a file
 *
 * @param path Path to the input file
 * @return The deserialized IR node
 */
IRNodePtr DeserializeFromFile(const std::string& path);

}  // namespace serialization
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_SERIALIZATION_DESERIALIZER_H_
