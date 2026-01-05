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
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tensor_expr.h"
#include "pypto/ir/transform/base/functor.h"
#include "pypto/ir/transform/transformers.h"

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
 * @brief Structural hasher for expressions
 *
 * Computes hash based on expression tree structure, ignoring Span (source location).
 */
class StructuralHasher {
 public:
  explicit StructuralHasher(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}

  int64_t operator()(const ExprPtr& expr) { return HashExpr(expr); }

 private:
  int64_t HashExpr(const ExprPtr& expr);
  int64_t HashVar(const VarPtr& op);

  template <typename NodePtr>
  int64_t HashNode(const NodePtr& node);

  bool enable_auto_mapping_;
  std::unordered_map<const Var*, int64_t> var_map_;
  int64_t free_var_counter_ = 0;
};

/**
 * @brief Field visitor for structural hashing
 */
class HashFieldVisitor {
 public:
  using result_type = int64_t;

  explicit HashFieldVisitor(StructuralHasher* parent) : parent_(parent) {}

  [[nodiscard]] result_type InitResult() const { return 0; }

  template <typename ExprPtrType>
  result_type VisitExprField(const ExprPtrType& field) {
    INTERNAL_CHECK(field) << "structural_hash encountered null expression field";
    return (*parent_)(field);
  }

  template <typename ExprPtrType>
  result_type VisitExprVectorField(const std::vector<ExprPtrType>& fields) {
    int64_t h = 0;
    for (size_t i = 0; i < fields.size(); ++i) {
      INTERNAL_CHECK(fields[i]) << "structural_hash encountered null expression in vector at index " << i;
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

  template <typename Desc>
  void AccumulateResult(result_type& accumulator, result_type field_hash, const Desc& /*descriptor*/) {
    accumulator = hash_combine(accumulator, field_hash);
  }

 private:
  StructuralHasher* parent_;
};

template <typename NodePtr>
int64_t StructuralHasher::HashNode(const NodePtr& node) {
  using NodeType = typename NodePtr::element_type;

  // Start with type discriminator
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}(node->type_name()));

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

int64_t StructuralHasher::HashVar(const VarPtr& op) {
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}("Var"));
  if (enable_auto_mapping_) {
    // Auto-mapping: map Var pointers to sequential IDs for structural comparison
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) {
      h = hash_combine(h, it->second);
    } else {
      var_map_[op.get()] = free_var_counter_++;
      h = hash_combine(h, free_var_counter_);
    }
  } else {
    // Without auto-mapping: hash the VarPtr itself (pointer-based)
    h = hash_combine(h, static_cast<int64_t>(std::hash<VarPtr>{}(op)));
  }
  return h;
}

// Type dispatch macro
#define HASH_DISPATCH(Type)                                   \
  if (auto p = std::dynamic_pointer_cast<const Type>(expr)) { \
    return HashNode(p);                                       \
  }

int64_t StructuralHasher::HashExpr(const ExprPtr& expr) {
  INTERNAL_CHECK(expr) << "structural_hash received null expression";

  // Special case: Var needs auto-mapping logic
  if (auto var = std::dynamic_pointer_cast<const Var>(expr)) {
    return HashVar(var);
  }

  // All other scalar expr types use generic field-based hashing
  HASH_DISPATCH(ConstInt)
  HASH_DISPATCH(Call)

  // Binary operations
  if (auto binary = std::dynamic_pointer_cast<const BinaryExpr>(expr)) {
    return HashNode(binary);
  }
  // Unary operations
  if (auto unary = std::dynamic_pointer_cast<const UnaryExpr>(expr)) {
    return HashNode(unary);
  }

  // Tensor expressions
  HASH_DISPATCH(TensorVar)

  // Unknown type - return hash of type name
  throw pypto::TypeError("Unknown expression type in StructuralHasher::HashExpr");
}

#undef HASH_DISPATCH

}  // namespace

// Public API
int64_t structural_hash(const ExprPtr& expr, bool enable_auto_mapping) {
  StructuralHasher hasher(enable_auto_mapping);
  return hasher(expr);
}

}  // namespace ir
}  // namespace pypto
