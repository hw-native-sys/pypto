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

#ifndef PYPTO_IR_SERIALIZATION_SERIALIZER_H_
#define PYPTO_IR_SERIALIZATION_SERIALIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "pypto/ir/core.h"

namespace pypto {
namespace ir {

// Forward declarations
class Expr;
class Stmt;
class Type;
class Op;
using ExprPtr = std::shared_ptr<const Expr>;
using StmtPtr = std::shared_ptr<const Stmt>;
using TypePtr = std::shared_ptr<const Type>;
using OpPtr = std::shared_ptr<const Op>;

namespace serialization {

/**
 * @brief Serializer for IR AST nodes to MessagePack format
 *
 * Serializes IR AST nodes while preserving pointer sharing and identity.
 * Uses a reference table to track already-serialized nodes and emit references
 * for subsequent occurrences of the same pointer.
 */
class IRSerializer {
 public:
  IRSerializer();
  ~IRSerializer();

  // Allow internal field visitor helper to access Impl
  friend class FieldSerializerVisitor;

  /**
   * @brief Serialize an IR node to MessagePack bytes
   *
   * @param node The IR node to serialize
   * @return Vector of bytes containing the MessagePack-encoded data
   */
  std::vector<uint8_t> Serialize(const IRNodePtr& node);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Serialize an IR node to MessagePack bytes
 *
 * Convenience function that creates a serializer and serializes the node.
 *
 * @param node The IR node to serialize
 * @return Vector of bytes containing the MessagePack-encoded data
 */
std::vector<uint8_t> Serialize(const IRNodePtr& node);

/**
 * @brief Serialize an IR node to a file
 *
 * @param node The IR node to serialize
 * @param path Path to the output file
 */
void SerializeToFile(const IRNodePtr& node, const std::string& path);

}  // namespace serialization
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_SERIALIZATION_SERIALIZER_H_
