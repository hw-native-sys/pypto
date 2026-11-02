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

#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/serialization/type_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace serialization {

DataType detail::DataTypeFromSerializedCode(uint8_t code) {
  CHECK(code != 0x30) << "Serialized logical FP4 dtype code 0x30 is no longer supported; use FP4E2M1X2 or "
                         "FP4E1M2X2 with byte-carrier shapes, strides, and offsets";
  return DataType(code);
}

namespace {

using TypeFields = std::unordered_map<std::string, msgpack::object>;

// Reuse enum conversion while preserving TileView's field-specific ValueError
// contract; generic enum-valued kwargs deliberately report TypeError instead.
TileLayout ReadTileViewLayout(const std::string& value, const char* field) {
  try {
    return StringToTileLayout(value);
  } catch (const TypeError&) {
    throw ValueError("Unknown TileLayout for " + std::string(field) + ": " + value);
  }
}

TypeFields ReadRequiredTypeFields(const msgpack::object& obj, const std::string& type_kind,
                                  std::initializer_list<const char*> required_fields) {
  TypeFields fields;
  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const auto& entry = obj.via.map.ptr[i];
    std::string key = entry.key.as<std::string>();
    CHECK(fields.emplace(key, entry.val).second) << type_kind << " has duplicate field '" << key << "'";
  }
  for (const char* key : required_fields) {
    CHECK(fields.count(key) != 0) << type_kind << " is missing required field '" << key << "'";
  }
  CHECK(fields.size() == required_fields.size())
      << type_kind << " has unexpected fields in its type descriptor";
  return fields;
}

template <typename Integer>
Integer ReadTypeInteger(const msgpack::object& obj, const std::string& field) {
  CHECK(obj.type == msgpack::type::POSITIVE_INTEGER || obj.type == msgpack::type::NEGATIVE_INTEGER)
      << "Type field '" << field << "' must be an integer";
  // Check the wire value before conversion: narrowing enums/dtypes can turn
  // corrupt large integers into valid-looking descriptor values.
  if (obj.type == msgpack::type::POSITIVE_INTEGER) {
    CHECK(obj.via.u64 <= static_cast<uint64_t>(std::numeric_limits<Integer>::max()))
        << "Type field '" << field << "' is out of range";
  } else {
    if constexpr (std::is_signed_v<Integer>) {
      CHECK(obj.via.i64 >= std::numeric_limits<Integer>::min())
          << "Type field '" << field << "' is out of range";
    } else {
      CHECK(false) << "Type field '" << field << "' is out of range";
    }
  }
  return obj.as<Integer>();
}

std::vector<int64_t> ReadBufferShape(const msgpack::object& obj, const std::string& field) {
  CHECK(obj.type == msgpack::type::ARRAY) << "BufferType field '" << field << "' must be an integer array";
  std::vector<int64_t> result;
  result.reserve(obj.via.array.size);
  for (uint32_t i = 0; i < obj.via.array.size; ++i) {
    result.push_back(ReadTypeInteger<int64_t>(obj.via.array.ptr[i], field));
  }
  return result;
}

}  // namespace

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

    // Use type registry to create the node. Per-type Stmt deserializers read
    // "leading_comments" from fields_obj and pass it to the constructor —
    // symmetric with how they read "span".
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

  std::optional<MemRefPtr> DeserializeMemRef(const msgpack::object& obj, msgpack::zone& zone) {
    if (obj.is_nil()) {
      return std::nullopt;
    }

    CHECK(obj.type == msgpack::type::MAP) << "Expected map for MemRef";

    std::string base_name;
    VarPtr base_node;
    ExprPtr byte_offset = nullptr;
    uint64_t size = 0;
    // Absent in blobs written before declared allocations existed; those hold only
    // compiler allocations, which is exactly what `false` means.
    bool is_pinned = false;
    // Slot fields default to "one slot, index 0" — what an unsubscripted
    // declaration and every compiler allocation mean.
    uint64_t slot_count = 1;
    std::optional<ExprPtr> slot_index = std::nullopt;
    bool has_base = false;
    bool has_byte_offset = false;
    bool has_size = false;

    msgpack::object_kv* p = obj.via.map.ptr;
    msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == "base") {
        p->val.convert(base_name);
        has_base = true;
      } else if (key == "base_node") {
        if (!p->val.is_nil()) {
          base_node = As<Var>(DeserializeNode(p->val, zone));
        }
      } else if (key == "byte_offset") {
        byte_offset = std::static_pointer_cast<const Expr>(DeserializeNode(p->val, zone));
        has_byte_offset = true;
      } else if (key == "size") {
        p->val.convert(size);
        has_size = true;
      } else if (key == "is_pinned") {
        p->val.convert(is_pinned);
      } else if (key == "slot_count") {
        p->val.convert(slot_count);
      } else if (key == "slot_index") {
        if (!p->val.is_nil()) {
          slot_index = std::static_pointer_cast<const Expr>(DeserializeNode(p->val, zone));
        }
      }
    }

    CHECK(has_base && has_byte_offset && has_size)
        << "MemRef missing required fields (base, byte_offset, or size)";

    // Prefer the serialized base node: it comes from the shared node graph, so it
    // is the very same Var object every other reference resolves to — including
    // the alloc statement that defines the Ptr. Allocation identity in the IR is
    // base_ *pointer* identity (MemRef::SameAllocation, InitMemRef's
    // DeclaredAllocMap key, the address allocator's base groups and its
    // declared-size lookup), so a base rebuilt from a name would not match its own
    // allocation and the group would be sized as if it held one MemRef.
    //
    // Blobs written before `base_node` existed carry only the name. Interning one
    // Var per name recovers the sharing *among the MemRefs* — the best that is
    // available without the node — so those blobs at least keep their slots on one
    // allocation.
    VarPtr base = base_node;
    if (!base) {
      auto& interned = memref_bases_[base_name];
      if (!interned) interned = std::make_shared<Var>(base_name, GetPtrType(), Span::unknown());
      base = interned;
    }
    return std::make_shared<MemRef>(base, byte_offset, size, Span::unknown(), is_pinned, slot_count,
                                    std::move(slot_index));
  }

  std::optional<TileView> DeserializeTileView(const msgpack::object& obj, msgpack::zone& zone) {
    if (obj.is_nil()) {
      return std::nullopt;
    }

    CHECK(obj.type == msgpack::type::MAP) << "Expected map for TileView";

    TileView tile_view;

    msgpack::object_kv* p = obj.via.map.ptr;
    msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == "valid_shape") {
        if (p->val.type == msgpack::type::ARRAY) {
          for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
            tile_view.valid_shape.push_back(
                std::static_pointer_cast<const Expr>(DeserializeNode(p->val.via.array.ptr[i], zone)));
          }
        }
      } else if (key == "stride") {
        if (p->val.type == msgpack::type::ARRAY) {
          for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
            tile_view.stride.push_back(
                std::static_pointer_cast<const Expr>(DeserializeNode(p->val.via.array.ptr[i], zone)));
          }
        }
      } else if (key == "start_offset") {
        if (!p->val.is_nil()) {
          tile_view.start_offset = std::static_pointer_cast<const Expr>(DeserializeNode(p->val, zone));
        }
      } else if (key == "blayout") {
        std::string blayout_str;
        p->val.convert(blayout_str);
        tile_view.blayout = ReadTileViewLayout(blayout_str, "blayout");
      } else if (key == "slayout") {
        std::string slayout_str;
        p->val.convert(slayout_str);
        tile_view.slayout = ReadTileViewLayout(slayout_str, "slayout");
      } else if (key == "fractal") {
        p->val.convert(tile_view.fractal);
      } else if (key == "pad") {
        std::string pad_str;
        p->val.convert(pad_str);
        auto pad = TryStringToPadValue(pad_str);
        CHECK(pad.has_value()) << "Unknown PadValue: " << pad_str;
        tile_view.pad = *pad;
      } else if (key == "compact") {
        std::string compact_str;
        p->val.convert(compact_str);
        tile_view.compact = StringToCompactMode(compact_str);
      }
    }

    return tile_view;
  }

  std::optional<TensorView> DeserializeTensorView(const msgpack::object& obj, msgpack::zone& zone) {
    if (obj.is_nil()) {
      return std::nullopt;
    }

    CHECK(obj.type == msgpack::type::MAP) << "Expected map for TensorView";

    TensorView tensor_view;

    msgpack::object_kv* p = obj.via.map.ptr;
    msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == "stride") {
        if (p->val.type == msgpack::type::ARRAY) {
          for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
            tensor_view.stride.push_back(
                std::static_pointer_cast<const Expr>(DeserializeNode(p->val.via.array.ptr[i], zone)));
          }
        }
      } else if (key == "valid_shape") {
        if (p->val.type == msgpack::type::ARRAY) {
          for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
            tensor_view.valid_shape.push_back(
                std::static_pointer_cast<const Expr>(DeserializeNode(p->val.via.array.ptr[i], zone)));
          }
        }
      } else if (key == "layout") {
        std::string layout_str;
        p->val.convert(layout_str);
        if (layout_str == "ND") {
          tensor_view.layout = TensorLayout::ND;
        } else if (layout_str == "DN") {
          tensor_view.layout = TensorLayout::DN;
        } else if (layout_str == "NZ") {
          tensor_view.layout = TensorLayout::NZ;
        } else if (layout_str == "MX_A_ZZ") {
          tensor_view.layout = TensorLayout::MX_A_ZZ;
        } else if (layout_str == "MX_B_NN") {
          tensor_view.layout = TensorLayout::MX_B_NN;
        } else {
          CHECK(false) << "Unknown TensorLayout: " << layout_str;
        }
      } else if (key == "pad") {
        std::string pad_str;
        p->val.convert(pad_str);
        auto pad = TryStringToPadValue(pad_str);
        CHECK(pad.has_value()) << "Unknown PadValue: " << pad_str;
        tensor_view.pad = *pad;
      }
      // Older serialized IR may omit "valid_shape" or "pad"; defaults stay
      // fully valid and PadValue::null, respectively.
    }

    return tensor_view;
  }

  TypePtr DeserializeType(const msgpack::object& obj, msgpack::zone& zone) override {
    CHECK(obj.type == msgpack::type::MAP) << "Expected map for Type";

    std::string type_kind;
    bool has_type_kind = false;
    for (uint32_t i = 0; i < obj.via.map.size; ++i) {
      const auto& entry = obj.via.map.ptr[i];
      if (entry.key.as<std::string>() == "type_kind") {
        CHECK(!has_type_kind) << "Type has duplicate field 'type_kind'";
        entry.val.convert(type_kind);
        has_type_kind = true;
      }
    }
    // Decode final buffer descriptors before the legacy shape-expression path.
    // Every descriptor field is required; malformed blobs must not acquire
    // default dtypes, layouts, or storage identities during deserialization.
    if (type_kind == "RegType") {
      const auto fields = ReadRequiredTypeFields(obj, type_kind, {"type_kind", "shape", "dtype", "kind"});
      return std::make_shared<RegType>(
          ReadTypeInteger<int64_t>(fields.at("shape"), "shape"),
          detail::DataTypeFromSerializedCode(ReadTypeInteger<uint8_t>(fields.at("dtype"), "dtype")),
          static_cast<RegKind>(ReadTypeInteger<uint8_t>(fields.at("kind"), "kind")));
    }
    if (type_kind == "VoidType") {
      ReadRequiredTypeFields(obj, type_kind, {"type_kind"});
      return GetVoidType();
    }
    if (type_kind == "BufferType") {
      const auto fields =
          ReadRequiredTypeFields(obj, type_kind,
                                 {"type_kind", "physical_shape", "dtype", "memory_space", "valid_shape",
                                  "blayout", "slayout", "fractal", "pad", "compact"});
      return std::make_shared<BufferType>(
          ReadBufferShape(fields.at("physical_shape"), "physical_shape"),
          detail::DataTypeFromSerializedCode(ReadTypeInteger<uint8_t>(fields.at("dtype"), "dtype")),
          static_cast<MemorySpace>(ReadTypeInteger<uint8_t>(fields.at("memory_space"), "memory_space")),
          ReadBufferShape(fields.at("valid_shape"), "valid_shape"),
          static_cast<TileLayout>(ReadTypeInteger<uint8_t>(fields.at("blayout"), "blayout")),
          static_cast<TileLayout>(ReadTypeInteger<uint8_t>(fields.at("slayout"), "slayout")),
          ReadTypeInteger<uint64_t>(fields.at("fractal"), "fractal"),
          static_cast<PadValue>(ReadTypeInteger<uint8_t>(fields.at("pad"), "pad")),
          static_cast<CompactMode>(ReadTypeInteger<uint8_t>(fields.at("compact"), "compact")));
    }
    if (type_kind == "MultiBufferType") {
      const auto fields = ReadRequiredTypeFields(obj, type_kind, {"type_kind", "element_type", "slot_count"});
      auto element_type = As<BufferType>(DeserializeType(fields.at("element_type"), zone));
      CHECK(element_type) << "MultiBufferType element_type must be a BufferType";
      return std::make_shared<MultiBufferType>(
          element_type, ReadTypeInteger<int64_t>(fields.at("slot_count"), "slot_count"));
    }

    uint8_t dtype_code = 0;
    std::vector<ExprPtr> shape;
    std::vector<TypePtr> types;
    msgpack::object memref_obj;
    msgpack::object tile_view_obj;
    msgpack::object tensor_view_obj;
    msgpack::object window_buffer_obj;
    uint8_t memory_space_code = 0;
    bool has_memref = false;
    bool has_tile_view = false;
    bool has_tensor_view = false;
    bool has_window_buffer = false;
    bool has_memory_space = false;

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
      } else if (key == "types") {
        if (p->val.type == msgpack::type::ARRAY) {
          for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
            types.push_back(DeserializeType(p->val.via.array.ptr[i], zone));
          }
        }
      } else if (key == "memref") {
        memref_obj = p->val;
        has_memref = true;
      } else if (key == "tile_view") {
        tile_view_obj = p->val;
        has_tile_view = true;
      } else if (key == "tensor_view") {
        tensor_view_obj = p->val;
        has_tensor_view = true;
      } else if (key == "window_buffer") {
        window_buffer_obj = p->val;
        has_window_buffer = true;
      } else if (key == "memory_space") {
        p->val.convert(memory_space_code);
        has_memory_space = true;
      }
    }

    if (type_kind == "ScalarType") {
      return std::make_shared<ScalarType>(detail::DataTypeFromSerializedCode(dtype_code));
    } else if (type_kind == "TensorType") {
      std::optional<MemRefPtr> memref;
      std::optional<TensorView> tensor_view;

      if (has_memref) {
        memref = DeserializeMemRef(memref_obj, zone);
      }
      if (has_tensor_view) {
        tensor_view = DeserializeTensorView(tensor_view_obj, zone);
      }

      return std::make_shared<TensorType>(shape, detail::DataTypeFromSerializedCode(dtype_code), memref,
                                          tensor_view);
    } else if (type_kind == "DistributedTensorType") {
      std::optional<MemRefPtr> memref;
      std::optional<TensorView> tensor_view;
      std::optional<WindowBufferPtr> window_buffer;
      if (has_memref) {
        memref = DeserializeMemRef(memref_obj, zone);
      }
      if (has_tensor_view) {
        tensor_view = DeserializeTensorView(tensor_view_obj, zone);
      }
      if (has_window_buffer) {
        window_buffer =
            std::static_pointer_cast<const WindowBuffer>(DeserializeNode(window_buffer_obj, zone));
      }
      return std::make_shared<DistributedTensorType>(shape, detail::DataTypeFromSerializedCode(dtype_code),
                                                     memref, tensor_view, window_buffer);
    } else if (type_kind == "TileType") {
      std::optional<MemRefPtr> memref;
      std::optional<TileView> tile_view;
      std::optional<MemorySpace> memory_space;

      if (has_memref) {
        memref = DeserializeMemRef(memref_obj, zone);
      }
      if (has_tile_view) {
        tile_view = DeserializeTileView(tile_view_obj, zone);
      }
      if (has_memory_space) {
        memory_space = static_cast<MemorySpace>(memory_space_code);
      }

      return std::make_shared<TileType>(shape, detail::DataTypeFromSerializedCode(dtype_code), memref,
                                        tile_view, memory_space);
    } else if (type_kind == "ArrayType") {
      CHECK(shape.size() == 1) << "ArrayType must have rank-1 shape, got " << shape.size();
      return std::make_shared<ArrayType>(detail::DataTypeFromSerializedCode(dtype_code), shape[0]);
    } else if (type_kind == "TupleType") {
      return std::make_shared<TupleType>(types);
    } else if (type_kind == "WindowBufferType") {
      return GetWindowBufferType();
    } else if (type_kind == "CommCtxType") {
      return GetCommCtxType();
    } else if (type_kind == "PrefetchAsyncContextType") {
      return GetPrefetchAsyncContextType();
    } else if (type_kind == "AsyncEventType") {
      return GetAsyncEventType();
    } else if (type_kind == "AsyncSessionType") {
      return GetAsyncSessionType();
    } else if (type_kind == "MemRefType") {
      return GetMemRefType();
    } else if (type_kind == "Ptr") {
      return GetPtrType();
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
    throw RuntimeError("Missing required field: " + field_name);
  }

  bool HasField(const msgpack::object& fields_obj, const std::string& field_name) override {
    if (fields_obj.type != msgpack::type::MAP) {
      return false;
    }
    msgpack::object_kv* p = fields_obj.via.map.ptr;
    msgpack::object_kv* const pend = fields_obj.via.map.ptr + fields_obj.via.map.size;
    for (; p < pend; ++p) {
      std::string key;
      p->key.convert(key);
      if (key == field_name) {
        return true;
      }
    }
    return false;
  }

 private:
  std::unordered_map<uint64_t, IRNodePtr> id_to_ptr_;

  /// Base Ptr per allocation name, so every MemRef naming one allocation shares
  /// one Var. An embedded MemRef serializes its base by name, so without this the
  /// reader mints a fresh Var per MemRef and the allocation identity — which the
  /// IR carries as base_ *pointer* identity — is lost on the way back in.
  std::unordered_map<std::string, VarPtr> memref_bases_;
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
  CHECK(!file.fail()) << "Failed to read from file: " + path;

  return Deserialize(data);
}

}  // namespace serialization
}  // namespace ir
}  // namespace pypto
