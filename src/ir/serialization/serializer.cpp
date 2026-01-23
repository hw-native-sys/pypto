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

#include "pypto/ir/serialization/serializer.h"

#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace serialization {

/**
 * @brief Field visitor for serialization
 *
 * Visits all fields of an IR node and serializes them to MessagePack format.
 */
class FieldSerializerVisitor {
 public:
  using result_type = msgpack::object;

  explicit FieldSerializerVisitor(msgpack::zone& zone, class IRSerializer::Impl& ctx)
      : zone_(zone), ctx_(ctx) {}

  [[nodiscard]] result_type InitResult() const;

  // Visit IRNode pointer fields
  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const IRNodePtrType& field);

  // Visit optional IRNode pointer fields
  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const std::optional<IRNodePtrType>& field);

  // Visit vector of IRNode pointers
  template <typename IRNodePtrType>
  result_type VisitIRNodeVectorField(const std::vector<IRNodePtrType>& field);

  // Visit map of IRNode pointers
  template <typename KeyType, typename ValueType, typename Compare>
  result_type VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& field);

  // Visit leaf fields
  result_type VisitLeafField(const int& field);
  result_type VisitLeafField(const double& field);
  result_type VisitLeafField(const bool& field);
  result_type VisitLeafField(const std::string& field);
  result_type VisitLeafField(const DataType& field);
  result_type VisitLeafField(const TypePtr& field);
  result_type VisitLeafField(const OpPtr& field);
  result_type VisitLeafField(const Span& field);
  result_type VisitLeafField(const std::vector<TypePtr>& field);
  result_type VisitLeafField(const std::vector<std::pair<std::string, std::any>>& field);

  // Field kind hooks
  template <typename FVisitOp>
  void VisitIgnoreField(FVisitOp&& visit_op) {
    visit_op();
  }

  template <typename FVisitOp>
  void VisitDefField(FVisitOp&& visit_op) {
    visit_op();
  }

  template <typename FVisitOp>
  void VisitUsualField(FVisitOp&& visit_op) {
    visit_op();
  }

  // Combine field results into a map
  template <typename Desc>
  void CombineResult(result_type& acc, result_type field_result, const Desc& desc);

 private:
  msgpack::zone& zone_;
  class IRSerializer::Impl& ctx_;
  std::map<std::string, msgpack::object> fields_;
};

/**
 * @brief Implementation class for IRSerializer
 */
class IRSerializer::Impl {
 public:
  Impl() = default;

  std::vector<uint8_t> Serialize(const IRNodePtr& node) {
    ptr_to_id_.clear();
    next_id_ = 0;

    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> packer(buffer);

    msgpack::zone zone;
    auto obj = SerializeNode(node, zone);
    packer.pack(obj);

    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  msgpack::object SerializeNode(const IRNodePtr& node, msgpack::zone& zone) {
    INTERNAL_CHECK(node) << "Cannot serialize null IR node";

    // Check if we've already serialized this pointer
    auto it = ptr_to_id_.find(node.get());
    if (it != ptr_to_id_.end()) {
      // Return a reference to the already-serialized node
      std::map<std::string, msgpack::object> ref_map;
      ref_map["ref"] = msgpack::object(it->second, zone);
      return msgpack::object(ref_map, zone);
    }

    // Assign a new ID to this node
    uint64_t id = next_id_++;
    ptr_to_id_[node.get()] = id;

    // Serialize the node with its ID and type
    std::map<std::string, msgpack::object> node_map;
    node_map["id"] = msgpack::object(id, zone);
    node_map["type"] = msgpack::object(node->TypeName(), zone);

    // Serialize fields using field visitor
    node_map["fields"] = SerializeFields(node, zone);

    return msgpack::object(node_map, zone);
  }

  msgpack::object SerializeFields(const IRNodePtr& node, msgpack::zone& zone) {
#define SERIALIZE_FIELDS(Type)                                \
  if (auto p = std::dynamic_pointer_cast<const Type>(node)) { \
    return SerializeFieldsGeneric(p, zone);                   \
  }

    SERIALIZE_FIELDS(IterArg);
    SERIALIZE_FIELDS(Var);
    SERIALIZE_FIELDS(ConstInt);
    SERIALIZE_FIELDS(ConstFloat);
    SERIALIZE_FIELDS(ConstBool);
    SERIALIZE_FIELDS(Call);
    SERIALIZE_FIELDS(TupleGetItemExpr);
    SERIALIZE_FIELDS(BinaryExpr);
    SERIALIZE_FIELDS(UnaryExpr);
    SERIALIZE_FIELDS(AssignStmt);
    SERIALIZE_FIELDS(IfStmt);
    SERIALIZE_FIELDS(YieldStmt);
    SERIALIZE_FIELDS(ReturnStmt);
    SERIALIZE_FIELDS(ForStmt);
    SERIALIZE_FIELDS(SeqStmts);
    SERIALIZE_FIELDS(OpStmts);
    SERIALIZE_FIELDS(EvalStmt);
    SERIALIZE_FIELDS(Function);
    SERIALIZE_FIELDS(Program);

#undef SERIALIZE_FIELDS

    INTERNAL_UNREACHABLE << "Unknown IR node type in serialization: " << node->TypeName();
  }

  msgpack::object SerializeSpan(const Span& span, msgpack::zone& zone) {
    std::map<std::string, msgpack::object> span_map;
    span_map["filename"] = msgpack::object(span.filename_, zone);
    span_map["begin_line"] = msgpack::object(span.begin_line_, zone);
    span_map["begin_column"] = msgpack::object(span.begin_column_, zone);
    span_map["end_line"] = msgpack::object(span.end_line_, zone);
    span_map["end_column"] = msgpack::object(span.end_column_, zone);
    return msgpack::object(span_map, zone);
  }

  msgpack::object SerializeMemRef(const std::optional<MemRefPtr>& memref_opt, msgpack::zone& zone) {
    if (!memref_opt.has_value()) {
      return msgpack::object();  // null
    }

    const auto& memref = *memref_opt.value();
    std::map<std::string, msgpack::object> memref_map;
    memref_map["memory_space"] = msgpack::object(static_cast<uint8_t>(memref.memory_space_), zone);
    memref_map["addr"] = SerializeNode(memref.addr_, zone);
    memref_map["size"] = msgpack::object(memref.size_, zone);
    return msgpack::object(memref_map, zone);
  }

  msgpack::object SerializeTileView(const std::optional<TileView>& tile_view, msgpack::zone& zone) {
    if (!tile_view.has_value()) {
      return msgpack::object();  // null
    }

    std::map<std::string, msgpack::object> tv_map;

    // Serialize valid_shape
    std::vector<msgpack::object> valid_shape_vec;
    for (const auto& dim : tile_view->valid_shape) {
      valid_shape_vec.push_back(SerializeNode(dim, zone));
    }
    tv_map["valid_shape"] = msgpack::object(valid_shape_vec, zone);

    // Serialize stride
    std::vector<msgpack::object> stride_vec;
    for (const auto& dim : tile_view->stride) {
      stride_vec.push_back(SerializeNode(dim, zone));
    }
    tv_map["stride"] = msgpack::object(stride_vec, zone);

    // Serialize start_offset
    tv_map["start_offset"] = SerializeNode(tile_view->start_offset, zone);

    return msgpack::object(tv_map, zone);
  }

  msgpack::object SerializeType(const TypePtr& type, msgpack::zone& zone) {
    INTERNAL_CHECK(type) << "Cannot serialize null Type";

    std::map<std::string, msgpack::object> type_map;
    type_map["type_kind"] = msgpack::object(type->TypeName(), zone);

    if (auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(type)) {
      type_map["dtype"] = msgpack::object(scalar_type->dtype_.Code(), zone);
    } else if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(type)) {
      type_map["dtype"] = msgpack::object(tensor_type->dtype_.Code(), zone);

      std::vector<msgpack::object> shape_vec;
      for (const auto& dim : tensor_type->shape_) {
        shape_vec.push_back(SerializeNode(dim, zone));
      }
      type_map["shape"] = msgpack::object(shape_vec, zone);

      // Serialize memref if present
      if (tensor_type->memref_.has_value()) {
        type_map["memref"] = SerializeMemRef(tensor_type->memref_, zone);
      }
    } else if (auto tile_type = std::dynamic_pointer_cast<const TileType>(type)) {
      type_map["dtype"] = msgpack::object(tile_type->dtype_.Code(), zone);

      std::vector<msgpack::object> shape_vec;
      for (const auto& dim : tile_type->shape_) {
        shape_vec.push_back(SerializeNode(dim, zone));
      }
      type_map["shape"] = msgpack::object(shape_vec, zone);

      // Serialize memref if present
      if (tile_type->memref_.has_value()) {
        type_map["memref"] = SerializeMemRef(tile_type->memref_, zone);
      }

      // Serialize tile_view if present
      if (tile_type->tile_view_.has_value()) {
        type_map["tile_view"] = SerializeTileView(tile_type->tile_view_, zone);
      }
    } else if (auto tuple_type = std::dynamic_pointer_cast<const TupleType>(type)) {
      std::vector<msgpack::object> types_vec;
      for (const auto& t : tuple_type->types_) {
        types_vec.push_back(SerializeType(t, zone));
      }
      type_map["types"] = msgpack::object(types_vec, zone);
    } else if (std::dynamic_pointer_cast<const UnknownType>(type)) {
      // UnknownType has no additional fields
    } else {
      INTERNAL_UNREACHABLE << "Unknown Type subclass: " << type->TypeName();
    }

    return msgpack::object(type_map, zone);
  }

  msgpack::object SerializeDataType(const DataType& dtype, msgpack::zone& zone) {
    std::map<std::string, msgpack::object> dtype_map;
    dtype_map["type"] = msgpack::object("DataType", zone);
    dtype_map["code"] = msgpack::object(dtype.Code(), zone);
    return msgpack::object(dtype_map, zone);
  }

  msgpack::object SerializeOp(const OpPtr& op, msgpack::zone& zone) {
    INTERNAL_CHECK(op) << "Cannot serialize null Op";

    std::map<std::string, msgpack::object> op_map;
    op_map["name"] = msgpack::object(op->name_, zone);

    // Check if it's a GlobalVar
    if (std::dynamic_pointer_cast<const GlobalVar>(op)) {
      op_map["is_global_var"] = msgpack::object(true, zone);
    } else {
      op_map["is_global_var"] = msgpack::object(false, zone);
    }

    return msgpack::object(op_map, zone);
  }

  template <typename NodePtr>
  msgpack::object SerializeFieldsGeneric(const NodePtr& node, msgpack::zone& zone) {
    using NodeType = typename NodePtr::element_type;
    auto descriptors = NodeType::GetFieldDescriptors();

    FieldSerializerVisitor visitor(zone, *this);
    return std::apply(
        [&](auto&&... descs) {
          return reflection::FieldIterator<NodeType, FieldSerializerVisitor, decltype(descs)...>::Visit(
              *node, visitor, descs...);
        },
        descriptors);
  }

 private:
  uint64_t next_id_;
  std::unordered_map<const IRNode*, uint64_t> ptr_to_id_;
};

// FieldSerializerVisitor implementation

msgpack::object FieldSerializerVisitor::InitResult() const { return msgpack::object(fields_, zone_); }

template <typename IRNodePtrType>
msgpack::object FieldSerializerVisitor::VisitIRNodeField(const IRNodePtrType& field) {
  return ctx_.SerializeNode(field, zone_);
}

// Overload for std::optional<IRNodePtr>
template <typename IRNodePtrType>
msgpack::object FieldSerializerVisitor::VisitIRNodeField(const std::optional<IRNodePtrType>& field) {
  if (field.has_value() && *field) {
    return ctx_.SerializeNode(*field, zone_);
  } else {
    // Return null object for empty optional
    return msgpack::object();
  }
}

template <typename IRNodePtrType>
msgpack::object FieldSerializerVisitor::VisitIRNodeVectorField(const std::vector<IRNodePtrType>& field) {
  std::vector<msgpack::object> vec;
  for (const auto& item : field) {
    vec.push_back(ctx_.SerializeNode(item, zone_));
  }
  return msgpack::object(vec, zone_);
}

template <typename KeyType, typename ValueType, typename Compare>
msgpack::object FieldSerializerVisitor::VisitIRNodeMapField(
    const std::map<KeyType, ValueType, Compare>& field) {
  // Serialize map as array of {key, value} pairs
  std::vector<msgpack::object> entries;
  for (const auto& [key, value] : field) {
    std::map<std::string, msgpack::object> entry;
    entry["key"] = ctx_.SerializeOp(key, zone_);
    entry["value"] = ctx_.SerializeNode(value, zone_);
    entries.emplace_back(entry, zone_);
  }
  return msgpack::object(entries, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const int& field) {
  return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const double& field) {
  return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const bool& field) {
  return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const std::string& field) {
  return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const DataType& field) {
  return ctx_.SerializeDataType(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const TypePtr& field) {
  return ctx_.SerializeType(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const OpPtr& field) {
  return ctx_.SerializeOp(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const Span& field) {
  return ctx_.SerializeSpan(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const std::vector<TypePtr>& field) {
  std::vector<msgpack::object> vec;
  vec.reserve(field.size());
  for (const auto& type : field) {
    vec.push_back(ctx_.SerializeType(type, zone_));
  }
  return msgpack::object(vec, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(
    const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // Use vector to preserve order (msgpack will serialize as array of [key, value] pairs)
  std::vector<msgpack::object> kwargs_msgs;

  auto make_pair = [this](const std::string& key, const msgpack::object& value) -> msgpack::object {
    std::map<std::string, msgpack::object> pair_map;
    pair_map["key"] = msgpack::object(key, zone_);
    pair_map["value"] = value;
    return msgpack::object(pair_map, zone_);
  };

  for (const auto& [key, value] : kwargs) {
    // Serialize common types
    if (value.type() == typeid(int)) {
      kwargs_msgs.push_back(make_pair(key, VisitLeafField(AnyCast<int>(value, "serializing kwarg: " + key))));
    } else if (value.type() == typeid(bool)) {
      kwargs_msgs.push_back(
          make_pair(key, VisitLeafField(AnyCast<bool>(value, "serializing kwarg: " + key))));
    } else if (value.type() == typeid(std::string)) {
      kwargs_msgs.push_back(
          make_pair(key, VisitLeafField(AnyCast<std::string>(value, "serializing kwarg: " + key))));
    } else if (value.type() == typeid(double)) {
      kwargs_msgs.push_back(
          make_pair(key, VisitLeafField(AnyCast<double>(value, "serializing kwarg: " + key))));
    } else if (value.type() == typeid(float)) {
      kwargs_msgs.push_back(
          make_pair(key, VisitLeafField(AnyCast<float>(value, "serializing kwarg: " + key))));
    } else if (value.type() == typeid(DataType)) {
      kwargs_msgs.push_back(
          make_pair(key, VisitLeafField(AnyCast<DataType>(value, "serializing kwarg: " + key))));
    } else {
      throw TypeError("Invalid kwarg type for key: " + key +
                      ", expected int, bool, std::string, double, float, or DataType, but got " +
                      DemangleTypeName(value.type().name()));
    }
  }

  return msgpack::object(kwargs_msgs, zone_);
}

template <typename Desc>
void FieldSerializerVisitor::CombineResult(result_type& acc, result_type field_result, const Desc& desc) {
  fields_[desc.name] = field_result;
  acc = msgpack::object(fields_, zone_);
}

// IRSerializer implementation

IRSerializer::IRSerializer() : impl_(std::make_unique<Impl>()) {}

IRSerializer::~IRSerializer() = default;

std::vector<uint8_t> IRSerializer::Serialize(const IRNodePtr& node) { return impl_->Serialize(node); }

// Public API functions

std::vector<uint8_t> Serialize(const IRNodePtr& node) {
  IRSerializer serializer;
  return serializer.Serialize(node);
}

void SerializeToFile(const IRNodePtr& node, const std::string& path) {
  auto data = Serialize(node);
  std::ofstream file(path, std::ios::binary);
  CHECK(file) << "Failed to open file for writing: " + path;
  file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  CHECK(file) << "Failed to write to file: " + path;
}

}  // namespace serialization
}  // namespace ir
}  // namespace pypto
