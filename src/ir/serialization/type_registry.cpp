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

#include "pypto/ir/serialization/type_registry.h"

#include <string>
#include <utility>

#include "pypto/core/error.h"

namespace pypto {
namespace ir {
namespace serialization {

TypeRegistry& TypeRegistry::Instance() {
  static TypeRegistry instance;
  return instance;
}

void TypeRegistry::Register(const std::string& type_name, DeserializerFunc func) {
  auto [it, inserted] = registry_.insert({type_name, std::move(func)});
  if (!inserted) {
    // Type already registered, overwrite silently
    it->second = std::move(func);
  }
}

IRNodePtr TypeRegistry::Create(const std::string& type_name, const msgpack::object& obj, msgpack::zone& zone,
                               detail::DeserializerContext& ctx) {
  auto it = registry_.find(type_name);
  if (it == registry_.end()) {
    throw TypeError("Unknown IR node type in deserialization: " + type_name);
  }
  return it->second(obj, zone, ctx);
}

bool TypeRegistry::IsRegistered(const std::string& type_name) const {
  return registry_.find(type_name) != registry_.end();
}

}  // namespace serialization
}  // namespace ir
}  // namespace pypto
