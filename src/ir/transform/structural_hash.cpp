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

#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transform/base/functor.h"
#include "pypto/ir/transform/transformers.h"

namespace pypto {
namespace ir {

// Forward declaration
class StructuralHash;

/**
 * @brief Field visitor for structural hashing
 *
 * Implements field-by-field hashing logic for the reflection system.
 */
class StructuralHashFieldVisitor {
 public:
  using result_type = int64_t;

  explicit StructuralHashFieldVisitor(StructuralHash* parent) : parent_(parent) {}

  [[nodiscard]] result_type InitResult() const { return 0; }

  template<typename ExprPtrType>
  result_type VisitExprField(const ExprPtrType& field);

  template<typename ExprPtrType>
  result_type VisitExprVectorField(const std::vector<ExprPtrType>& fields);

  // Visit scalar fields
  result_type VisitScalarField(const int& field) { return static_cast<int64_t>(std::hash<int>{}(field)); }

  result_type VisitScalarField(const std::string& field) {
    return static_cast<int64_t>(std::hash<std::string>{}(field));
  }

  result_type VisitScalarField(const OpPtr& field) {
    return static_cast<int64_t>(std::hash<std::string>{}(field->name_));
  }

  result_type VisitScalarField(const DataType& field) {
    return static_cast<int64_t>(std::hash<int>{}(static_cast<int>(field)));
  }

  // Accumulate results by combining hashes
  template <typename Desc>
  void AccumulateResult(result_type& accumulator, result_type field_hash, const Desc& descriptor) {
    accumulator = hash_combine(accumulator, field_hash);
  }

 private:
  StructuralHash* parent_;

  static int64_t hash_combine(int64_t seed, int64_t value) {
    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
  }
};

/**
 * @brief Internal implementation: Structural hash functor for expressions
 *
 * Computes hash based on expression tree structure, ignoring Span (source location).
 * This class is not part of the public API - use structural_hash() function instead.
 */
class StructuralHash : public ExprFunctor<int64_t> {
 public:
  explicit StructuralHash(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}
  int64_t operator()(const ExprPtr& expr);
  int64_t VisitExpr(const ExprPtr& expr) override;

 protected:
  // Leaf nodes
  int64_t VisitExpr_(const VarPtr& op) override;
  int64_t VisitExpr_(const ConstIntPtr& op) override;
  int64_t VisitExpr_(const CallPtr& op) override;

  // Binary operations
  int64_t VisitExpr_(const AddPtr& op) override;
  int64_t VisitExpr_(const SubPtr& op) override;
  int64_t VisitExpr_(const MulPtr& op) override;
  int64_t VisitExpr_(const FloorDivPtr& op) override;
  int64_t VisitExpr_(const FloorModPtr& op) override;
  int64_t VisitExpr_(const FloatDivPtr& op) override;
  int64_t VisitExpr_(const MinPtr& op) override;
  int64_t VisitExpr_(const MaxPtr& op) override;
  int64_t VisitExpr_(const PowPtr& op) override;
  int64_t VisitExpr_(const EqPtr& op) override;
  int64_t VisitExpr_(const NePtr& op) override;
  int64_t VisitExpr_(const LtPtr& op) override;
  int64_t VisitExpr_(const LePtr& op) override;
  int64_t VisitExpr_(const GtPtr& op) override;
  int64_t VisitExpr_(const GePtr& op) override;
  int64_t VisitExpr_(const AndPtr& op) override;
  int64_t VisitExpr_(const OrPtr& op) override;
  int64_t VisitExpr_(const XorPtr& op) override;
  int64_t VisitExpr_(const BitAndPtr& op) override;
  int64_t VisitExpr_(const BitOrPtr& op) override;
  int64_t VisitExpr_(const BitXorPtr& op) override;
  int64_t VisitExpr_(const BitShiftLeftPtr& op) override;
  int64_t VisitExpr_(const BitShiftRightPtr& op) override;

  // Unary operations
  int64_t VisitExpr_(const AbsPtr& op) override;
  int64_t VisitExpr_(const NegPtr& op) override;
  int64_t VisitExpr_(const NotPtr& op) override;
  int64_t VisitExpr_(const BitNotPtr& op) override;

 private:
  /**
   * @brief Generic field-based hashing for IR nodes
   *
   * Uses field descriptors to hash all fields of a node generically.
   *
   * @tparam NodePtr Shared pointer type to the node
   * @param op The node to hash
   * @return Hash value
   */
  template <typename NodePtr>
  int64_t VisitExprWithFields(const NodePtr& op) {
    using NodeType = typename NodePtr::element_type;

    // Start with type discriminator
    int64_t h = static_cast<int64_t>(std::hash<std::string>{}(op->type_name()));

    // Visit all fields using descriptors
    StructuralHashFieldVisitor visitor(this);
    auto descriptors = NodeType::GetFieldDescriptors();

    int64_t fields_hash = std::apply(
        [&](auto&&... descs) {
          return reflection::FieldIterator<NodeType, StructuralHashFieldVisitor, decltype(descs)...>::Visit(
              *op, visitor, descs...);
        },
        descriptors);

    return hash_combine(h, fields_hash);
  }

  static int64_t hash_combine(int64_t seed, int64_t value);

  bool enable_auto_mapping_;
  std::unordered_map<const Var*, int64_t> var_map_;
  int64_t free_var_counter_ = 0;

  friend class StructuralHashFieldVisitor;
};

int64_t StructuralHash::operator()(const ExprPtr& expr) { return VisitExpr(expr); }

int64_t StructuralHash::VisitExpr(const ExprPtr& expr) { return ExprFunctor::VisitExpr(expr); }

// Hash combine using Boost-inspired algorithm
int64_t StructuralHash::hash_combine(int64_t seed, int64_t value) {
  // Magic number from boost::hash_combine for good distribution
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

// StructuralHashFieldVisitor template method implementations
template<typename ExprPtrType>
int64_t StructuralHashFieldVisitor::VisitExprField(const ExprPtrType& field) {
  return parent_->VisitExpr(field);
}

template<typename ExprPtrType>
int64_t StructuralHashFieldVisitor::VisitExprVectorField(const std::vector<ExprPtrType>& fields) {
  int64_t h = 0;
  for (const auto& field : fields) {
    h = hash_combine(h, parent_->VisitExpr(field));
  }
  return h;
}

// Leaf nodes - Var requires special handling for auto-mapping
int64_t StructuralHash::VisitExpr_(const VarPtr& op) {
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}("Var"));  // Type discriminator
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

int64_t StructuralHash::VisitExpr_(const ConstIntPtr& op) { return VisitExprWithFields(op); }

int64_t StructuralHash::VisitExpr_(const CallPtr& op) { return VisitExprWithFields(op); }

// Binary operations - now use generic field-based hashing
#define DEFINE_BINARY_HASH(OpType) \
  int64_t StructuralHash::VisitExpr_(const OpType##Ptr& op) { return VisitExprWithFields(op); }

// Binary operations
DEFINE_BINARY_HASH(Add)
DEFINE_BINARY_HASH(Sub)
DEFINE_BINARY_HASH(Mul)
DEFINE_BINARY_HASH(FloorDiv)
DEFINE_BINARY_HASH(FloorMod)
DEFINE_BINARY_HASH(FloatDiv)
DEFINE_BINARY_HASH(Min)
DEFINE_BINARY_HASH(Max)
DEFINE_BINARY_HASH(Pow)
DEFINE_BINARY_HASH(Eq)
DEFINE_BINARY_HASH(Ne)
DEFINE_BINARY_HASH(Lt)
DEFINE_BINARY_HASH(Le)
DEFINE_BINARY_HASH(Gt)
DEFINE_BINARY_HASH(Ge)
DEFINE_BINARY_HASH(And)
DEFINE_BINARY_HASH(Or)
DEFINE_BINARY_HASH(Xor)
DEFINE_BINARY_HASH(BitAnd)
DEFINE_BINARY_HASH(BitOr)
DEFINE_BINARY_HASH(BitXor)
DEFINE_BINARY_HASH(BitShiftLeft)
DEFINE_BINARY_HASH(BitShiftRight)

#undef DEFINE_BINARY_HASH

// Unary operations - now use generic field-based hashing
#define DEFINE_UNARY_HASH(OpType) \
  int64_t StructuralHash::VisitExpr_(const OpType##Ptr& op) { return VisitExprWithFields(op); }

// Unary operations
DEFINE_UNARY_HASH(Abs)
DEFINE_UNARY_HASH(Neg)
DEFINE_UNARY_HASH(Not)
DEFINE_UNARY_HASH(BitNot)

#undef DEFINE_UNARY_HASH

// Public API implementation
int64_t structural_hash(const ExprPtr& expr, bool enable_auto_mapping) {
  StructuralHash hasher(enable_auto_mapping);
  return hasher(expr);
}

}  // namespace ir
}  // namespace pypto
