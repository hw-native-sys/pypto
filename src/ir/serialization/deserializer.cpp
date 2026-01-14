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

#include "pypto/ir/serialization/deserializer.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/serialization/type_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace serialization {

/**
 * @brief Implementation class for IRDeserializer
 */
class IRDeserializer::Impl : public detail::DeserializerContext {
 public:
  Impl() = default;

  IRNodePtr Deserialize(const std::vector<uint8_t>& data) {
    id_to_ptr_.clear();

    try {
      msgpack::object_handle oh = msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size());
      msgpack::object obj = oh.get();
      return DeserializeNode(obj, *oh.zone());
    } catch (const msgpack::parse_error& e) {
      throw RuntimeError(std::string("MessagePack parse error: ") + e.what());
    } catch (const msgpack::type_error& e) {
      throw RuntimeError(std::string("MessagePack type error: ") + e.what());
    }
  }

  IRNodePtr DeserializeNode(const msgpack::object& obj, msgpack::zone& zone) override {
    CHECK(obj.type == msgpack::type::MAP) << "Expected map for IR node";

    msgpack::object_kv* p = obj.via.map.ptr;
    msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;

    // Check if this is a reference
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == "ref") {
        uint64_t id;
        p->val.convert(id);
        auto it = id_to_ptr_.find(id);
        CHECK(it != id_to_ptr_.end()) << "Invalid reference ID: " << id;
        return it->second;
      }
    }

    // Parse full node
    uint64_t id = 0;
    std::string type_name;
    msgpack::object fields_obj;
    bool has_id = false;
    bool has_type = false;
    bool has_fields = false;

    p = obj.via.map.ptr;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == "id") {
        p->val.convert(id);
        has_id = true;
      } else if (key == "type") {
        p->val.convert(type_name);
        has_type = true;
      } else if (key == "fields") {
        fields_obj = p->val;
        has_fields = true;
      }
    }

    INTERNAL_CHECK(has_id && has_type && has_fields)
        << "Missing required fields (id, type, or fields) in node";

    // Use type registry to create the node
    IRNodePtr node = TypeRegistry::Instance().Create(type_name, fields_obj, zone, *this);

    // Store in reference table
    id_to_ptr_[id] = node;

    return node;
  }

  Span DeserializeSpan(const msgpack::object& obj) override {
    CHECK(obj.type == msgpack::type::MAP) << "Expected map for Span";
    std::string filename;
    int begin_line = -1, begin_column = -1, end_line = -1, end_column = -1;

    msgpack::object_kv* p = obj.via.map.ptr;
    msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == "filename") {
        p->val.convert(filename);
      } else if (key == "begin_line") {
        p->val.convert(begin_line);
      } else if (key == "begin_column") {
        p->val.convert(begin_column);
      } else if (key == "end_line") {
        p->val.convert(end_line);
      } else if (key == "end_column") {
        p->val.convert(end_column);
      }
    }

    return Span(filename, begin_line, begin_column, end_line, end_column);
  }

  TypePtr DeserializeType(const msgpack::object& obj, msgpack::zone& zone) override {
    CHECK(obj.type == msgpack::type::MAP) << "Expected map for Type";

    std::string type_kind;
    uint8_t dtype_code = 0;
    std::vector<ExprPtr> shape;

    msgpack::object_kv* p = obj.via.map.ptr;
    msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == "type_kind") {
        p->val.convert(type_kind);
      } else if (key == "dtype") {
        p->val.convert(dtype_code);
      } else if (key == "shape") {
        if (p->val.type == msgpack::type::ARRAY) {
          for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
            shape.push_back(
                std::static_pointer_cast<const Expr>(DeserializeNode(p->val.via.array.ptr[i], zone)));
          }
        }
      }
    }

    if (type_kind == "ScalarType") {
      return std::make_shared<ScalarType>(DataType(dtype_code));
    } else if (type_kind == "TensorType") {
      return std::make_shared<TensorType>(DataType(dtype_code), shape);
    } else if (type_kind == "UnknownType") {
      return GetUnknownType();
    } else {
      throw RuntimeError("Unknown Type kind: " + type_kind);
    }
  }

  OpPtr DeserializeOp(const msgpack::object& obj) override {
    CHECK(obj.type == msgpack::type::MAP) << "Expected map for Op";

    std::string name;
    bool is_global_var = false;

    msgpack::object_kv* p = obj.via.map.ptr;
    msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == "name") {
        p->val.convert(name);
      } else if (key == "is_global_var") {
        p->val.convert(is_global_var);
      }
    }

    if (is_global_var) {
      return std::make_shared<GlobalVar>(name);
    } else {
      return std::make_shared<Op>(name);
    }
  }

  msgpack::object GetFieldObj(const msgpack::object& fields_obj, const std::string& field_name) override {
    CHECK(fields_obj.type == msgpack::type::MAP) << "Expected map for fields";
    msgpack::object_kv* p = fields_obj.via.map.ptr;
    msgpack::object_kv* const pend = fields_obj.via.map.ptr + fields_obj.via.map.size;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == field_name) {
        return p->val;
      }
    }
    UNREACHABLE << "Missing field: " << field_name;
  }

 private:
  std::unordered_map<uint64_t, IRNodePtr> id_to_ptr_;
};

// IRDeserializer implementation

IRDeserializer::IRDeserializer() : impl_(std::make_unique<Impl>()) {}

IRDeserializer::~IRDeserializer() = default;

IRNodePtr IRDeserializer::Deserialize(const std::vector<uint8_t>& data) { return impl_->Deserialize(data); }

// Public API functions

IRNodePtr Deserialize(const std::vector<uint8_t>& data) {
  IRDeserializer deserializer;
  return deserializer.Deserialize(data);
}

IRNodePtr DeserializeFromFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << "Failed to open file for reading: " + path;
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  CHECK(!file.fail() && !file.eof()) << "Failed to read from file: " + path;

  return Deserialize(data);
}

}  // namespace serialization
}  // namespace ir
}  // namespace pypto
