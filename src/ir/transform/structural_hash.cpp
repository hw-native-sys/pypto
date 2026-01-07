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
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/functor.h"
#include "pypto/ir/transform/transformers.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Hash combine using Boost-inspired algorithm
 */
int64_t hash_combine(int64_t seed, int64_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief Field visitor for structural hashing
 */
class HashFieldVisitor;

/**
 * @brief Structural hasher for IR nodes
 *
 * Computes hash based on IR node tree structure, ignoring Span (source location).
 */
class StructuralHasher {
 public:
  explicit StructuralHasher(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}

  int64_t operator()(const IRNodePtr& node) { return HashNode(node); }

 private:
  friend class HashFieldVisitor;
  int64_t HashNode(const IRNodePtr& node);
  int64_t HashVar(const VarPtr& op);
  int64_t HashType(const TypePtr& type);

  template <typename NodePtr>
  int64_t HashNodeImpl(const NodePtr& node);

  bool enable_auto_mapping_;
  std::unordered_map<const Var*, int64_t> var_map_;
  int64_t free_var_counter_ = 0;
};

class HashFieldVisitor {
 public:
  using result_type = int64_t;

  explicit HashFieldVisitor(StructuralHasher* parent) : parent_(parent) {}

  [[nodiscard]] result_type InitResult() const { return 0; }

  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const IRNodePtrType& field) {
    INTERNAL_CHECK(field) << "structural_hash encountered null IR node field";
    return (*parent_)(field);
  }

  template <typename IRNodePtrType>
  result_type VisitIRNodeVectorField(const std::vector<IRNodePtrType>& fields) {
    int64_t h = 0;
    for (size_t i = 0; i < fields.size(); ++i) {
      INTERNAL_CHECK(fields[i]) << "structural_hash encountered null IR node in vector at index " << i;
      h = hash_combine(h, (*parent_)(fields[i]));
    }
    return h;
  }

  result_type VisitScalarField(const int& field) { return static_cast<int64_t>(std::hash<int>{}(field)); }

  result_type VisitScalarField(const std::string& field) {
    return static_cast<int64_t>(std::hash<std::string>{}(field));
  }

  result_type VisitScalarField(const OpPtr& field) {
    return static_cast<int64_t>(std::hash<std::string>{}(field->name_));
  }

  result_type VisitScalarField(const DataType& field) {
    return static_cast<int64_t>(std::hash<uint8_t>{}(field.Code()));
  }

  result_type VisitScalarField(const TypePtr& field) {
    INTERNAL_CHECK(field) << "structural_hash encountered null TypePtr field";
    return parent_->HashType(field);
  }

  template <typename Desc>
  void AccumulateResult(result_type& accumulator, result_type field_hash, const Desc& /*descriptor*/) {
    accumulator = hash_combine(accumulator, field_hash);
  }

 private:
  StructuralHasher* parent_;
};

template <typename NodePtr>
int64_t StructuralHasher::HashNodeImpl(const NodePtr& node) {
  using NodeType = typename NodePtr::element_type;

  // Start with type discriminator
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}(node->TypeName()));

  // Visit all fields using reflection
  HashFieldVisitor visitor(this);
  auto descriptors = NodeType::GetFieldDescriptors();

  int64_t fields_hash = std::apply(
      [&](auto&&... descs) {
        return reflection::FieldIterator<NodeType, HashFieldVisitor, decltype(descs)...>::Visit(
            *node, visitor, descs...);
      },
      descriptors);

  return hash_combine(h, fields_hash);
}

int64_t StructuralHasher::HashType(const TypePtr& type) {
  INTERNAL_CHECK(type) << "structural_hash encountered null TypePtr";
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}(type->TypeName()));
  if (auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(type)) {
    h = hash_combine(h, static_cast<int64_t>(std::hash<uint8_t>{}(scalar_type->dtype_.Code())));
  } else if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(type)) {
    h = hash_combine(h, static_cast<int64_t>(std::hash<uint8_t>{}(tensor_type->dtype_.Code())));
    h = hash_combine(h, static_cast<int64_t>(tensor_type->shape_.size()));
    for (const auto& dim : tensor_type->shape_) {
      INTERNAL_CHECK(dim) << "structural_hash encountered null shape dimension in TypePtr";
      h = hash_combine(h, (*this)(dim));
    }
  } else if (std::dynamic_pointer_cast<const UnknownType>(type)) {
    // UnknownType has no fields, so only hash the type name (already done above)
  } else {
    INTERNAL_CHECK(false) << "HashType encountered unhandled Type: " << type->TypeName();
  }
  return h;
}

int64_t StructuralHasher::HashVar(const VarPtr& op) {
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}("Var"));
  if (enable_auto_mapping_) {
    // Auto-mapping: map Var pointers to sequential IDs for structural comparison
    auto [it, inserted] = var_map_.try_emplace(op.get(), free_var_counter_);
    if (inserted) {
      free_var_counter_++;
    }
    h = hash_combine(h, it->second);

    // Hash type information
    h = hash_combine(h, HashType(op->type_));
  } else {
    // Without auto-mapping: hash the VarPtr itself (pointer-based)
    h = hash_combine(h, static_cast<int64_t>(std::hash<VarPtr>{}(op)));
  }
  return h;
}

// Type dispatch macro
#define HASH_DISPATCH(Type)                                   \
  if (auto p = std::dynamic_pointer_cast<const Type>(node)) { \
    return HashNodeImpl(p);                                   \
  }

int64_t StructuralHasher::HashNode(const IRNodePtr& node) {
  INTERNAL_CHECK(node) << "structural_hash received null IR node";

  // Dispatch to type-specific handlers using dynamic_cast
  // Check types that require special handling first
  if (auto var = std::dynamic_pointer_cast<const Var>(node)) {
    return HashVar(var);
  }

  // All other types use generic field-based hashing
  HASH_DISPATCH(ConstInt)
  HASH_DISPATCH(Call)
  HASH_DISPATCH(BinaryExpr)
  HASH_DISPATCH(UnaryExpr)
  HASH_DISPATCH(Stmt)

  // Unknown IR node type
  throw pypto::TypeError("Unknown IR node type in StructuralHasher::HashNode");
}

#undef HASH_DISPATCH

}  // namespace

// Public API
int64_t structural_hash(const IRNodePtr& node, bool enable_auto_mapping) {
  StructuralHasher hasher(enable_auto_mapping);
  return hasher(node);
}

}  // namespace ir
}  // namespace pypto
