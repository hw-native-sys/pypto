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

#ifndef PYPTO_IR_SERIALIZATION_TYPE_REGISTRY_H_
#define PYPTO_IR_SERIALIZATION_TYPE_REGISTRY_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace serialization {

// Forward declaration
class IRDeserializer;

// Deserialization context interface exposed for type deserializers.
namespace detail {
class DeserializerContext {
 public:
  virtual ~DeserializerContext() = default;

  virtual Span DeserializeSpan(const msgpack::object& obj) = 0;
  virtual TypePtr DeserializeType(const msgpack::object& obj, msgpack::zone& zone) = 0;
  virtual OpPtr DeserializeOp(const msgpack::object& obj) = 0;
  virtual IRNodePtr DeserializeNode(const msgpack::object& obj, msgpack::zone& zone) = 0;
  virtual msgpack::object GetFieldObj(const msgpack::object& fields_obj, const std::string& field_name) = 0;

  template <typename T>
  T GetField(const msgpack::object& fields_obj, const std::string& field_name) {
    msgpack::object field_obj = GetFieldObj(fields_obj, field_name);
    T value;
    field_obj.convert(value);
    return value;
  }
};
}  // namespace detail

/**
 * @brief Registry mapping IR node type names to deserializer functions
 *
 * This registry allows the deserializer to create the correct IR node type
 * based on the type name in the serialized data.
 */
class TypeRegistry {
 public:
  using DeserializerFunc =
      std::function<IRNodePtr(const msgpack::object&, msgpack::zone&, detail::DeserializerContext&)>;

  /**
   * @brief Get the singleton instance of the type registry
   */
  static TypeRegistry& Instance();

  /**
   * @brief Register a deserializer function for a type name
   *
   * @param type_name The type name (e.g., "Add", "Var", "Function")
   * @param func The deserializer function
   */
  void Register(const std::string& type_name, DeserializerFunc func);

  /**
   * @brief Create an IR node from serialized data
   *
   * @param type_name The type name
   * @param obj The MessagePack object containing the node data
   * @param zone MessagePack zone for memory management
   * @param ctx Deserializer context
   * @return The deserialized IR node
   */
  IRNodePtr Create(const std::string& type_name, const msgpack::object& obj, msgpack::zone& zone,
                   detail::DeserializerContext& ctx);

  /**
   * @brief Check if a type is registered
   *
   * @param type_name The type name to check
   * @return true if the type is registered
   */
  [[nodiscard]] bool IsRegistered(const std::string& type_name) const;

 private:
  TypeRegistry() = default;
  std::unordered_map<std::string, DeserializerFunc> registry_;
};

/**
 * @brief RAII helper for registering a type deserializer
 *
 * Use this at global scope to automatically register a type deserializer.
 */
class TypeRegistrar {
 public:
  TypeRegistrar(const std::string& type_name, TypeRegistry::DeserializerFunc func) {
    TypeRegistry::Instance().Register(type_name, std::move(func));
  }
};

// Macro to simplify type registration
#define REGISTER_IR_TYPE(TypeName, DeserializerFunc) \
  static ::pypto::ir::serialization::TypeRegistrar _type_registrar_##TypeName(#TypeName, DeserializerFunc)

}  // namespace serialization
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_SERIALIZATION_TYPE_REGISTRY_H_
