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

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/hash.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

DataType CanonicalizeForSyntaxScalarDtype(const DataType& dtype) {
  if (dtype == DataType::INT64 || dtype == DataType::INDEX) {
    return DataType::INDEX;
  }
  return dtype;
}

}  // namespace

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

  result_type operator()(const TypePtr& type) { return HashType(type); }

  // FieldVisitor interface methods
  [[nodiscard]] result_type InitResult() const { return 0; }

  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const IRNodePtrType& field) {
    INTERNAL_CHECK(field) << "structural_hash encountered null IR node field";
    return HashNode(field);
  }

  // Specialization for std::optional<IRNodePtr>
  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const std::optional<IRNodePtrType>& field) {
    if (field.has_value() && *field) {
      return HashNode(*field);
    } else {
      // Hash empty optional as 0
      return 0;
    }
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

  template <typename KeyType, typename ValueType, typename Compare>
  result_type VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& field) {
    result_type h = 0;
    for (const auto& [key, value] : field) {
      INTERNAL_CHECK(key) << "structural_hash encountered null key in map";
      INTERNAL_CHECK(value) << "structural_hash encountered null value in map";
      // Hash key by name (keys are Op types, not IRNode)
      h = hash_combine(h, static_cast<result_type>(std::hash<std::string>{}(key->name_)));
      // Hash value (values are IRNode types)
      h = hash_combine(h, HashNode(value));
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

  void PushFieldName(const char* name) {
    if (transparent_depth_ == 0) {
      field_name_stack_.emplace_back(name);
    }
  }
  void PopFieldName() {
    if (transparent_depth_ == 0) {
      field_name_stack_.pop_back();
    }
  }

  result_type VisitLeafField(const int& field) { return static_cast<result_type>(std::hash<int>{}(field)); }

  result_type VisitLeafField(const int64_t& field) {
    return static_cast<result_type>(std::hash<int64_t>{}(field));
  }

  result_type VisitLeafField(const uint64_t& field) {
    return static_cast<result_type>(std::hash<uint64_t>{}(field));
  }

  result_type VisitLeafField(const double& field) {
    return static_cast<result_type>(std::hash<double>{}(field));
  }

  result_type VisitLeafField(const std::string& field) {
    return static_cast<result_type>(std::hash<std::string>{}(field));
  }

  result_type VisitLeafField(const OpPtr& field) {
    return static_cast<result_type>(std::hash<std::string>{}(field->name_));
  }

  result_type VisitLeafField(const DataType& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(field.Code()));
  }

  result_type VisitLeafField(const FunctionType& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const ForKind& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const InlineLanguage& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const ChunkPolicy& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const std::optional<ChunkConfig>& field) {
    if (!field.has_value()) return 0;
    result_type h = VisitIRNodeField(field->size);
    return hash_combine(h, VisitLeafField(field->policy));
  }

  result_type VisitLeafField(const ScopeKind& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const Level& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const Role& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const std::optional<Level>& field) {
    if (field.has_value()) {
      return hash_combine(1, VisitLeafField(*field));
    }
    return static_cast<result_type>(0);
  }

  result_type VisitLeafField(const std::optional<Role>& field) {
    if (field.has_value()) {
      return hash_combine(1, VisitLeafField(*field));
    }
    return static_cast<result_type>(0);
  }

  result_type VisitLeafField(const SplitMode& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const std::optional<SplitMode>& field) {
    if (field.has_value()) {
      return hash_combine(1, VisitLeafField(*field));
    }
    return static_cast<result_type>(0);
  }

  result_type VisitLeafField(const std::optional<int>& field) {
    if (field.has_value()) {
      return hash_combine(1, static_cast<result_type>(std::hash<int>{}(*field)));
    }
    return static_cast<result_type>(0);
  }

  result_type VisitLeafField(const std::optional<bool>& field) {
    if (field.has_value()) {
      return hash_combine(1, static_cast<result_type>(std::hash<bool>{}(*field)));
    }
    return static_cast<result_type>(0);
  }

  result_type VisitLeafField(const ParamDirection& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const std::vector<ParamDirection>& field) {
    result_type h = 0;
    for (const auto& dir : field) {
      h = hash_combine(h, VisitLeafField(dir));
    }
    return h;
  }

  result_type VisitLeafField(const ArgDirection& field) {
    return static_cast<result_type>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
  }

  result_type VisitLeafField(const std::vector<ArgDirection>& field) {
    result_type h = 0;
    for (const auto& dir : field) {
      h = hash_combine(h, VisitLeafField(dir));
    }
    return h;
  }

  result_type VisitLeafField(const MemorySpace& field) {
    return static_cast<result_type>(std::hash<int>{}(static_cast<int>(field)));
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

  // Required by the template instantiation path for Stmt::leading_comments_
  // (IgnoreField). VisitIgnoreField discards the lambda at runtime, so this
  // overload should never be called — guard it like the Span overload.
  result_type VisitLeafField(const std::vector<std::string>& /*field*/) {
    INTERNAL_UNREACHABLE << "structural_hash should not visit leading_comments field";
    return 0;  // Never reached
  }

  // Hash kwargs (vector of pairs - order is preserved and matters)
  result_type VisitLeafField(const std::vector<std::pair<std::string, std::any>>& kwargs) {
    result_type h = 0;
    // Hash keys and values in order (no need to sort since order is preserved)
    for (const auto& [key, value] : kwargs) {
      h = hash_combine(h, std::hash<std::string>{}(key));

      // Hash value based on type
      if (value.type() == typeid(int)) {
        h = hash_combine(h, std::hash<int>{}(AnyCast<int>(value, "hashing kwarg: " + key)));
      } else if (value.type() == typeid(bool)) {
        h = hash_combine(h, std::hash<bool>{}(AnyCast<bool>(value, "hashing kwarg: " + key)));
      } else if (value.type() == typeid(std::string)) {
        h = hash_combine(h, std::hash<std::string>{}(AnyCast<std::string>(value, "hashing kwarg: " + key)));
      } else if (value.type() == typeid(double)) {
        h = hash_combine(h, std::hash<double>{}(AnyCast<double>(value, "hashing kwarg: " + key)));
      } else if (value.type() == typeid(float)) {
        h = hash_combine(h, std::hash<float>{}(AnyCast<float>(value, "hashing kwarg: " + key)));
      } else if (value.type() == typeid(DataType)) {
        h = hash_combine(h, std::hash<uint8_t>{}(AnyCast<DataType>(value, "hashing kwarg: " + key).Code()));
      } else if (value.type() == typeid(LoopOrigin)) {
        h = hash_combine(h, std::hash<uint8_t>{}(
                                static_cast<uint8_t>(AnyCast<LoopOrigin>(value, "hashing kwarg: " + key))));
      } else if (value.type() == typeid(MemorySpace)) {
        h = hash_combine(h, std::hash<uint8_t>{}(
                                static_cast<uint8_t>(AnyCast<MemorySpace>(value, "hashing kwarg: " + key))));
      } else if (value.type() == typeid(TensorLayout)) {
        h = hash_combine(h, std::hash<uint8_t>{}(
                                static_cast<uint8_t>(AnyCast<TensorLayout>(value, "hashing kwarg: " + key))));
      } else if (value.type() == typeid(TileLayout)) {
        h = hash_combine(h, std::hash<uint8_t>{}(
                                static_cast<uint8_t>(AnyCast<TileLayout>(value, "hashing kwarg: " + key))));
      } else if (value.type() == typeid(PadValue)) {
        h = hash_combine(
            h, std::hash<uint8_t>{}(static_cast<uint8_t>(AnyCast<PadValue>(value, "hashing kwarg: " + key))));
      } else if (value.type() == typeid(std::vector<ArgDirection>)) {
        h = hash_combine(h,
                         VisitLeafField(AnyCast<std::vector<ArgDirection>>(value, "hashing kwarg: " + key)));
      } else {
        throw TypeError("Unsupported kwarg type for key: " + key + ": " +
                        DemangleTypeName(value.type().name()));
      }
    }
    return h;
  }

  result_type VisitLeafField(const Span& field) {
    INTERNAL_UNREACHABLE_SPAN(field) << "structural_hash should not visit Span field";
    return 0;
  }

  template <typename Desc>
  void CombineResult(result_type& accumulator, result_type field_hash, const Desc& /*descriptor*/) {
    accumulator = hash_combine(accumulator, field_hash);
  }

 private:
  result_type HashNode(const IRNodePtr& node);
  result_type HashType(const TypePtr& type);
  bool IsLoopVarFieldContext() const {
    return !field_name_stack_.empty() && field_name_stack_.back() == "loop_var";
  }
  bool IsConstIntTypeContext() const {
    return !node_type_stack_.empty() && node_type_stack_.back() == "ConstInt" && !field_name_stack_.empty() &&
           field_name_stack_.back() == "type";
  }

  template <typename NodePtr>
  result_type HashNodeImpl(const NodePtr& node);

  bool enable_auto_mapping_;
  std::unordered_map<IRNodePtr, result_type> hash_value_map_;
  int64_t free_var_counter_ = 0;
  std::vector<std::string> field_name_stack_;
  std::vector<std::string> node_type_stack_;
  int transparent_depth_ = 0;
};

template <typename NodePtr>
StructuralHasher::result_type StructuralHasher::HashNodeImpl(const NodePtr& node) {
  using NodeType = typename NodePtr::element_type;

  // Start with type discriminator
  result_type h = static_cast<result_type>(std::hash<std::string>{}(node->TypeName()));
  node_type_stack_.emplace_back(node->TypeName());

  // Mirror EQUAL_DISPATCH / EQUAL_DISPATCH_TRANSPARENT from structural_equal.cpp:
  // - Transparent containers (Program, SeqStmts) suppress their own field names by
  //   incrementing transparent_depth_, so PushFieldName skips them.
  // - Non-transparent nodes reset transparent_depth_ to 0 so their fields are always
  //   tracked, even when visited from within a transparent container.
  constexpr bool is_transparent = std::is_same_v<NodeType, Program> || std::is_same_v<NodeType, SeqStmts>;
  int saved_depth = transparent_depth_;
  if constexpr (is_transparent) {
    transparent_depth_++;
  } else {
    transparent_depth_ = 0;
  }

  // Visit all fields using reflection
  auto descriptors = NodeType::GetFieldDescriptors();

  result_type fields_hash = std::apply(
      [&](auto&&... descs) {
        return reflection::FieldIterator<NodeType, StructuralHasher, decltype(descs)...>::Visit(*node, *this,
                                                                                                descs...);
      },
      descriptors);

  transparent_depth_ = saved_depth;
  node_type_stack_.pop_back();

  return hash_combine(h, fields_hash);
}

StructuralHasher::result_type StructuralHasher::HashType(const TypePtr& type) {
  INTERNAL_CHECK(type) << "structural_hash encountered null TypePtr";
  result_type h = static_cast<result_type>(std::hash<std::string>{}(type->TypeName()));
  if (auto scalar_type = As<ScalarType>(type)) {
    DataType dtype = scalar_type->dtype_;
    if (IsLoopVarFieldContext() || IsConstIntTypeContext()) {
      dtype = CanonicalizeForSyntaxScalarDtype(dtype);
    }
    h = hash_combine(h, static_cast<result_type>(std::hash<uint8_t>{}(dtype.Code())));
  } else if (auto tensor_type = As<TensorType>(type)) {
    h = hash_combine(h, static_cast<result_type>(std::hash<uint8_t>{}(tensor_type->dtype_.Code())));
    h = hash_combine(h, static_cast<result_type>(tensor_type->shape_.size()));
    for (const auto& dim : tensor_type->shape_) {
      INTERNAL_CHECK(dim) << "structural_hash encountered null shape dimension in TypePtr";
      h = hash_combine(h, HashNode(dim));
    }
    // Hash tensor_view if present
    if (tensor_type->tensor_view_.has_value()) {
      const auto& tv = tensor_type->tensor_view_.value();
      h = hash_combine(h, static_cast<result_type>(1));  // indicate presence
      // Hash valid_shape
      h = hash_combine(h, static_cast<result_type>(tv.valid_shape.size()));
      for (const auto& dim : tv.valid_shape) {
        INTERNAL_CHECK(dim) << "structural_hash encountered null valid_shape dimension in TensorView";
        h = hash_combine(h, HashNode(dim));
      }
      // Hash stride
      h = hash_combine(h, static_cast<result_type>(tv.stride.size()));
      for (const auto& dim : tv.stride) {
        INTERNAL_CHECK(dim) << "structural_hash encountered null stride dimension in TensorView";
        h = hash_combine(h, HashNode(dim));
      }
      // Hash layout
      h = hash_combine(h, static_cast<result_type>(tv.layout));
    } else {
      h = hash_combine(h, static_cast<result_type>(0));  // indicate absence
    }
  } else if (auto tile_type = As<TileType>(type)) {
    // Hash dtype
    h = hash_combine(h, static_cast<result_type>(std::hash<uint8_t>{}(tile_type->dtype_.Code())));
    // Hash shape size and dimensions
    h = hash_combine(h, static_cast<result_type>(tile_type->shape_.size()));
    for (const auto& dim : tile_type->shape_) {
      INTERNAL_CHECK(dim) << "structural_hash encountered null shape dimension in TileType";
      h = hash_combine(h, HashNode(dim));
    }
    // tile_view_ is already canonical (TileType ctor canonicalizes implicit
    // views to nullopt), so direct hashing yields one hash per semantic state.
    const auto& tile_view = tile_type->tile_view_;
    if (tile_view.has_value()) {
      const auto& tv = tile_view.value();
      h = hash_combine(h, static_cast<result_type>(1));  // indicate presence
      // Hash valid_shape
      h = hash_combine(h, static_cast<result_type>(tv.valid_shape.size()));
      for (const auto& dim : tv.valid_shape) {
        INTERNAL_CHECK(dim) << "structural_hash encountered null valid_shape dimension in TileView";
        h = hash_combine(h, HashNode(dim));
      }
      // Hash stride
      h = hash_combine(h, static_cast<result_type>(tv.stride.size()));
      for (const auto& dim : tv.stride) {
        INTERNAL_CHECK(dim) << "structural_hash encountered null stride dimension in TileView";
        h = hash_combine(h, HashNode(dim));
      }
      // Hash start_offset
      INTERNAL_CHECK(tv.start_offset) << "structural_hash encountered null start_offset in TileView";
      h = hash_combine(h, HashNode(tv.start_offset));
      // Hash blayout
      h = hash_combine(h, static_cast<result_type>(tv.blayout));
      // Hash slayout
      h = hash_combine(h, static_cast<result_type>(tv.slayout));
      // Hash fractal
      h = hash_combine(h, static_cast<result_type>(tv.fractal));
      // Hash pad
      h = hash_combine(h, static_cast<result_type>(tv.pad));
    } else {
      h = hash_combine(h, static_cast<result_type>(0));  // indicate absence
    }
    // Hash memory_space
    if (tile_type->memory_space_.has_value()) {
      h = hash_combine(h, static_cast<result_type>(1));  // indicate presence
      h = hash_combine(h, static_cast<result_type>(tile_type->memory_space_.value()));
    } else {
      h = hash_combine(h, static_cast<result_type>(0));  // indicate absence
    }
  } else if (auto tuple_type = As<TupleType>(type)) {
    h = hash_combine(h, static_cast<result_type>(tuple_type->types_.size()));
    for (const auto& t : tuple_type->types_) {
      INTERNAL_CHECK(t) << "structural_hash encountered null type in TupleType";
      h = hash_combine(h, HashType(t));
    }
  } else if (IsA<MemRefType>(type) || IsA<UnknownType>(type) || IsA<PtrType>(type)) {
    // MemRefType, PtrType, and UnknownType have no fields, only hash type name (already done above)
  } else {
    INTERNAL_CHECK(false) << "HashType encountered unhandled Type: " << type->TypeName();
  }
  return h;
}

// Type dispatch macro
#define HASH_DISPATCH(Type)                                      \
  if (auto p = As<Type>(node)) {                                 \
    INTERNAL_CHECK_SPAN(dispatched == false, node->span_)        \
        << "HashNodeImpl already dispatched for type " << #Type; \
    hash_value = HashNodeImpl(p);                                \
    dispatched = true;                                           \
  }

// Dispatch macro for abstract base classes
#define HASH_DISPATCH_BASE(Type)                                 \
  if (auto p = As<Type>(node)) {                                 \
    INTERNAL_CHECK_SPAN(dispatched == false, node->span_)        \
        << "HashNodeImpl already dispatched for type " << #Type; \
    hash_value = HashNodeImpl(p);                                \
    dispatched = true;                                           \
  }

StructuralHasher::result_type StructuralHasher::HashNode(const IRNodePtr& node) {
  INTERNAL_CHECK(node) << "structural_hash received null IR node";

  auto it = hash_value_map_.find(node);
  if (it != hash_value_map_.end()) {
    return it->second;
  }

  result_type hash_value = 0;
  bool dispatched = false;

  // MemRef needs special handling: dispatch for fields, then add Var mapping
  HASH_DISPATCH(MemRef)
  // IterArg needs special handling: dispatch for fields, then add Var mapping
  HASH_DISPATCH(IterArg)
  HASH_DISPATCH(Var)
  HASH_DISPATCH(ConstInt)
  HASH_DISPATCH(ConstFloat)
  HASH_DISPATCH(ConstBool)
  HASH_DISPATCH(Call)
  HASH_DISPATCH(MakeTuple)
  HASH_DISPATCH(TupleGetItemExpr)

  // BinaryExpr and UnaryExpr are abstract base classes, use dynamic_pointer_cast
  HASH_DISPATCH_BASE(BinaryExpr)
  HASH_DISPATCH_BASE(UnaryExpr)

  HASH_DISPATCH(AssignStmt)
  HASH_DISPATCH(IfStmt)
  HASH_DISPATCH(YieldStmt)
  HASH_DISPATCH(ReturnStmt)
  HASH_DISPATCH(ForStmt)
  HASH_DISPATCH(WhileStmt)
  HASH_DISPATCH(InCoreScopeStmt)
  HASH_DISPATCH(AutoInCoreScopeStmt)
  HASH_DISPATCH(ClusterScopeStmt)
  HASH_DISPATCH(HierarchyScopeStmt)
  HASH_DISPATCH(SpmdScopeStmt)
  HASH_DISPATCH(RuntimeScopeStmt)
  HASH_DISPATCH(SeqStmts)
  HASH_DISPATCH(EvalStmt)
  HASH_DISPATCH(BreakStmt)
  HASH_DISPATCH(ContinueStmt)
  HASH_DISPATCH(InlineStmt)
  HASH_DISPATCH(Function)
  HASH_DISPATCH(Program)

  // Free Var types (including MemRef and IterArg) that may be mapped to other free vars.
  // These have already been dispatched above for field hashing;
  // here we add the variable identity hash.
  auto hash_var_identity = [&](uint64_t unique_id) {
    if (enable_auto_mapping_) {
      hash_value = hash_combine(hash_value, free_var_counter_++);
    } else {
      hash_value = hash_combine(hash_value, unique_id);
    }
  };

  auto kind = node->GetKind();
  if (kind == ObjectKind::MemRef || kind == ObjectKind::IterArg || kind == ObjectKind::Var) {
    hash_var_identity(static_cast<const Var*>(node.get())->UniqueId());
  }

  if (!dispatched) {
    throw pypto::TypeError("Unknown IR node type in StructuralHasher::HashNode");
  }

  hash_value_map_.emplace(node, hash_value);
  return hash_value;
}

#undef HASH_DISPATCH
#undef HASH_DISPATCH_BASE

// Public API
uint64_t structural_hash(const IRNodePtr& node, bool enable_auto_mapping) {
  StructuralHasher hasher(enable_auto_mapping);
  return hasher(node);
}

uint64_t structural_hash(const TypePtr& type, bool enable_auto_mapping) {
  StructuralHasher hasher(enable_auto_mapping);
  return hasher(type);
}

}  // namespace ir
}  // namespace pypto
