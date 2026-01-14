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

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/function.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Hash combine using Boost-inspired algorithm
 */
inline uint64_t hash_combine(uint64_t seed, uint64_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief Structural hasher for IR nodes
 *
 * Computes hash based on IR node tree structure, ignoring Span (source location).
 * Also serves as a FieldVisitor for the reflection-based field iteration.
 */
class StructuralHasher {
 public:
  using result_type = uint64_t;

  explicit StructuralHasher(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}

  result_type operator()(const IRNodePtr& node) { return HashNode(node); }

  // FieldVisitor interface methods
  [[nodiscard]] result_type InitResult() const { return 0; }

  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const IRNodePtrType& field) {
    INTERNAL_CHECK(field) << "structural_hash encountered null IR node field";
    return HashNode(field);
  }

  template <typename IRNodePtrType>
  result_type VisitIRNodeVectorField(const std::vector<IRNodePtrType>& fields) {
    result_type h = 0;
    for (size_t i = 0; i < fields.size(); ++i) {
      INTERNAL_CHECK(fields[i]) << "structural_hash encountered null IR node in vector at index " << i;
      h = hash_combine(h, HashNode(fields[i]));
    }
    return h;
  }

  template <typename FVisitOp>
  void VisitIgnoreField([[maybe_unused]] FVisitOp&& visit_op) {
    // Ignore field, do nothing
  }
  template <typename FVisitOp>
  void VisitDefField(FVisitOp&& visit_op) {
    bool enable_auto_mapping = true;
    std::swap(enable_auto_mapping, enable_auto_mapping_);
    visit_op();
    std::swap(enable_auto_mapping, enable_auto_mapping_);
  }
  template <typename FVisitOp>
  void VisitUsualField(FVisitOp&& visit_op) {
    visit_op();
  }

  result_type VisitLeafField(const int& field) { return static_cast<result_type>(std::hash<int>{}(field)); }

  result_type VisitLeafField(const std::string& field) {
    return static_cast<result_type>(std::hash<std::string>{}(field));
  }

  result_type VisitLeafField(const OpPtr& field) {
    return static_cast<result_type>(std::hash<std::string>{}(field->name_));
  }

  result_type VisitLeafField(const DataType& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(field.Code()));
  }

  result_type VisitLeafField(const TypePtr& field) {
    INTERNAL_CHECK(field) << "structural_hash encountered null TypePtr field";
    return HashType(field);
  }

  result_type VisitLeafField(const std::vector<TypePtr>& fields) {
    result_type h = 0;
    for (size_t i = 0; i < fields.size(); ++i) {
      INTERNAL_CHECK(fields[i]) << "structural_hash encountered null TypePtr in vector at index " << i;
      h = hash_combine(h, HashType(fields[i]));
    }
    return h;
  }

  result_type VisitLeafField(const Span& field) {
    INTERNAL_UNREACHABLE << "structural_hash should not visit Span field";
  }

  template <typename Desc>
  void CombineResult(result_type& accumulator, result_type field_hash, const Desc& /*descriptor*/) {
    accumulator = hash_combine(accumulator, field_hash);
  }

 private:
  result_type HashNode(const IRNodePtr& node);
  result_type HashVar(const VarPtr& op);
  result_type HashType(const TypePtr& type);

  template <typename NodePtr>
  result_type HashNodeImpl(const NodePtr& node);

  bool enable_auto_mapping_;
  std::unordered_map<IRNodePtr, result_type> hash_value_map_;
  int64_t free_var_counter_ = 0;
};

template <typename NodePtr>
StructuralHasher::result_type StructuralHasher::HashNodeImpl(const NodePtr& node) {
  using NodeType = typename NodePtr::element_type;

  // Start with type discriminator
  result_type h = static_cast<result_type>(std::hash<std::string>{}(node->TypeName()));

  // Visit all fields using reflection
  auto descriptors = NodeType::GetFieldDescriptors();

  result_type fields_hash = std::apply(
      [&](auto&&... descs) {
        return reflection::FieldIterator<NodeType, StructuralHasher, decltype(descs)...>::Visit(*node, *this,
                                                                                                descs...);
      },
      descriptors);

  return hash_combine(h, fields_hash);
}

StructuralHasher::result_type StructuralHasher::HashVar(const VarPtr& op) {
  result_type h = HashNodeImpl(op);
  if (enable_auto_mapping_) {
    // Auto-mapping: map Var pointers to sequential IDs for structural comparison
    h = hash_combine(h, free_var_counter_++);
  } else {
    // Without auto-mapping: hash the VarPtr itself (pointer-based)
    h = hash_combine(h, static_cast<result_type>(std::hash<VarPtr>{}(op)));
  }
  return h;
}

StructuralHasher::result_type StructuralHasher::HashType(const TypePtr& type) {
  INTERNAL_CHECK(type) << "structural_hash encountered null TypePtr";
  result_type h = static_cast<result_type>(std::hash<std::string>{}(type->TypeName()));
  if (auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(type)) {
    h = hash_combine(h, static_cast<result_type>(std::hash<uint8_t>{}(scalar_type->dtype_.Code())));
  } else if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(type)) {
    h = hash_combine(h, static_cast<result_type>(std::hash<uint8_t>{}(tensor_type->dtype_.Code())));
    h = hash_combine(h, static_cast<result_type>(tensor_type->shape_.size()));
    for (const auto& dim : tensor_type->shape_) {
      INTERNAL_CHECK(dim) << "structural_hash encountered null shape dimension in TypePtr";
      h = hash_combine(h, HashNode(dim));
    }
  } else if (std::dynamic_pointer_cast<const UnknownType>(type)) {
    // UnknownType has no fields, so only hash the type name (already done above)
  } else {
    INTERNAL_CHECK(false) << "HashType encountered unhandled Type: " << type->TypeName();
  }
  return h;
}

// Type dispatch macro
#define HASH_DISPATCH(Type)                                                                      \
  if (auto p = std::dynamic_pointer_cast<const Type>(node)) {                                    \
    INTERNAL_CHECK(dispatched == false) << "HashNodeImpl already dispatched for type " << #Type; \
    hash_value = HashNodeImpl(p);                                                                \
    dispatched = true;                                                                           \
  }

StructuralHasher::result_type StructuralHasher::HashNode(const IRNodePtr& node) {
  INTERNAL_CHECK(node) << "structural_hash received null IR node";

  auto it = hash_value_map_.find(node);
  if (it != hash_value_map_.end()) {
    return it->second;
  }

  result_type hash_value = 0;
  bool dispatched = false;

  HASH_DISPATCH(Var)
  HASH_DISPATCH(ConstInt)
  HASH_DISPATCH(Call)
  HASH_DISPATCH(BinaryExpr)
  HASH_DISPATCH(UnaryExpr)
  HASH_DISPATCH(AssignStmt)
  HASH_DISPATCH(IfStmt)
  HASH_DISPATCH(YieldStmt)
  HASH_DISPATCH(ForStmt)
  HASH_DISPATCH(SeqStmts)
  HASH_DISPATCH(OpStmts)
  HASH_DISPATCH(Function)

  // Free Var types that may be mapped to other free vars
  if (auto var = std::dynamic_pointer_cast<const Var>(node)) {
    if (enable_auto_mapping_) {
      // Auto-mapping: map Var pointers to sequential IDs for structural comparison
      hash_value = hash_combine(hash_value, free_var_counter_++);
    } else {
      // Without auto-mapping: hash the VarPtr itself (pointer-based)
      hash_value = hash_combine(hash_value, static_cast<result_type>(std::hash<VarPtr>{}(var)));
    }
  }

  if (!dispatched) {
    // Unknown IR node type
    throw pypto::TypeError("Unknown IR node type in StructuralHasher::HashNode");
  } else {
    hash_value_map_.emplace(node, hash_value);
    return hash_value;
  }
}

#undef HASH_DISPATCH

// Public API
uint64_t structural_hash(const IRNodePtr& node, bool enable_auto_mapping) {
  StructuralHasher hasher(enable_auto_mapping);
  return hasher(node);
}

}  // namespace ir
}  // namespace pypto
