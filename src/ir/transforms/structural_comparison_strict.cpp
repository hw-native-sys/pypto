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

#include <any>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/structural_comparison.h"

namespace pypto::ir {
namespace {

// Exact interning, not hash-based equality. Each record contains only scalar
// payloads and child IDs; shared DAGs are never expanded into trees. Ordered
// indexing costs O(N log N) including all fields/string bytes, with O(N) space.
// Digests use child digests, NOT arena IDs, so independent invocations agree.
using Id = size_t;
using Fields = std::vector<std::pair<std::string, Id>>;
using Key = std::tuple<std::string, std::string, Fields>;

uint64_t Combine(uint64_t seed, uint64_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct Record {
  Key key;
  uint64_t hash;
};

class Arena {
 public:
  Arena() { Intern("null", "", {}); }

  Id Intern(std::string tag, std::string atom, Fields fields = {}) {
    Key key{std::move(tag), std::move(atom), std::move(fields)};
    auto found = index_.find(key);
    if (found != index_.end()) return found->second;
    uint64_t hash =
        Combine(std::hash<std::string>{}(std::get<0>(key)), std::hash<std::string>{}(std::get<1>(key)));
    for (const auto& [name, child] : std::get<2>(key)) {
      hash = Combine(hash, std::hash<std::string>{}(name));
      hash = Combine(hash, records_[child].hash);
    }
    Id id = records_.size();
    index_.emplace(key, id);
    records_.push_back({std::move(key), hash});
    return id;
  }

  uint64_t Hash(Id id) const { return records_[id].hash; }

  std::string Mismatch(Id lhs, Id rhs) const {
    std::string path = "root";
    while (true) {
      const auto& [lt, la, lf] = records_[lhs].key;
      const auto& [rt, ra, rf] = records_[rhs].key;
      if (lt != rt || la != ra || lf.size() != rf.size()) {
        return "Strict structural mismatch at " + path + ": " + lt + "(" + la.substr(0, 128) + ", " +
               std::to_string(lf.size()) + " fields) != " + rt + "(" + ra.substr(0, 128) + ", " +
               std::to_string(rf.size()) + " fields)";
      }
      size_t i = 0;
      while (i < lf.size() && lf[i] == rf[i]) ++i;
      INTERNAL_CHECK(i < lf.size()) << "Different interned records must have a differing field";
      if (lf[i].first != rf[i].first) {
        return "Strict structural mismatch at " + path + ": field " + lf[i].first.substr(0, 128) +
               " != " + rf[i].first.substr(0, 128);
      }
      if (path.size() < 1024) path += "." + lf[i].first.substr(0, 128);
      lhs = lf[i].second;
      rhs = rf[i].second;
    }
  }

 private:
  std::map<Key, Id> index_;
  std::vector<Record> records_;
};

struct Binding {
  std::string category;
  uint64_t id;
};

// One reflection projection supplies strict equality, hashing and diagnostics.
// The first traversal reserves DefField identities. The second traversal uses
// the frozen bindings, with independent memo tables for definitions/references.
// Function contexts are independent even when the input shares node pointers.
class Projection {
 public:
  using result_type = Fields;

  Projection(Arena& arena, bool map_free) : arena_(arena), map_free_(map_free) {}

  template <typename Root>
  Id Run(const Root& root) {
    PrepareFunction(root);
    collecting_ = true;
    Value(root);
    collecting_ = false;
    node_cache_ = {};
    type_cache_.clear();
    return Value(root);
  }

  result_type InitResult() const { return {}; }
  void PushFieldName(const char* name) { path_.emplace_back(name); }
  void PopFieldName() { path_.pop_back(); }

  template <typename F>
  void VisitIgnoreField(F&&) {}
  template <typename F>
  void VisitUsualField(F&& visit) {
    visit();
  }
  template <typename F>
  void VisitDefField(F&& visit) {
    const bool saved = defining_;
    defining_ = true;
    visit();
    defining_ = saved;
  }
  template <typename T>
  result_type VisitIRNodeField(const T& field) {
    return {{"", Value(field)}};
  }
  template <typename T>
  result_type VisitIRNodeVectorField(const T& field) {
    return {{"", Value(field)}};
  }
  template <typename T>
  result_type VisitIRNodeMapField(const T& field) {
    return {{"", Value(field)}};
  }
  template <typename T>
  result_type VisitLeafField(const T& field) {
    return {{"", Value(field)}};
  }
  template <typename Desc>
  void CombineResult(result_type& result, result_type field, const Desc& desc) {
    result.emplace_back(desc.name, field.front().second);
  }

 private:
  [[noreturn]] void Unsupported(const std::string& reason) const {
    std::string path = "root";
    for (const auto& part : path_) {
      if (path.size() >= 1024) break;
      path += "." + part.substr(0, 128);
    }
    throw StructuralComparisonUnsupported(reason, path);
  }

  Id Intern(std::string tag, std::string atom = "", Fields fields = {}) {
    return collecting_ ? 0 : arena_.Intern(std::move(tag), std::move(atom), std::move(fields));
  }

  template <typename T>
  Id Reflect(const T& value, const std::string& tag) {
    auto fields = std::apply(
        [&](auto&&... descriptors) {
          return reflection::FieldIterator<T, Projection, decltype(descriptors)...>::Visit(value, *this,
                                                                                           descriptors...);
        },
        T::GetFieldDescriptors());
    return Intern(tag, "", std::move(fields));
  }

  void Reserve(const VarPtr& var, const std::string& category) {
    if (!bindings_.count(var.get())) {
      uint64_t id = category == "bound" ? next_bound_++ : next_symbol_++;
      bindings_.emplace(var.get(), Binding{category, id});
    }
  }

  void CollectShapeSymbols(const ExprPtr& expr, bool bind, std::unordered_set<const Expr*>& visited) {
    std::vector<ExprPtr> pending{expr};
    while (!pending.empty()) {
      auto current = std::move(pending.back());
      pending.pop_back();
      if (!current || !visited.insert(current.get()).second) continue;
      if (auto var = AsVarLike(current)) {
        auto scalar = As<ScalarType>(var->GetType());
        if (!scalar || (scalar->dtype_ != DataType::INDEX && scalar->dtype_ != DataType::INT64)) {
          Unsupported("UnsupportedSignatureSymbol");
        }
        if (bind)
          Reserve(var, "symbol");
        else if (!bindings_.count(var.get()))
          Unsupported("UnboundValidShapeSymbol");
      } else if (auto binary = As<BinaryExpr>(current)) {
        pending.push_back(binary->right_);
        pending.push_back(binary->left_);
      } else if (auto unary = As<UnaryExpr>(current)) {
        pending.push_back(unary->operand_);
      } else if (auto call = As<Call>(current)) {
        pending.insert(pending.end(), call->args_.rbegin(), call->args_.rend());
      } else if (auto item = As<TupleGetItemExpr>(current)) {
        pending.push_back(item->tuple_);
      }
    }
  }

  template <typename T>
  void PrepareFunction(const std::shared_ptr<const T>& root) {
    if constexpr (std::is_base_of_v<IRNode, T>) {
      function_ = As<Function>(std::static_pointer_cast<const IRNode>(root));
      if (!function_) return;
      // Reserve every parameter before looking at any type (forward references
      // to scalar parameters are legal in parameter shape annotations).
      for (const auto& param : function_->params_) Reserve(param, "bound");
      std::unordered_set<const Expr*> shapes;
      for (const auto& param : function_->params_) {
        if (auto tensor = AsTensorTypeLike(param->GetType())) {
          for (const auto& dim : tensor->shape_) CollectShapeSymbols(dim, true, shapes);
        }
      }
      // Bare valid-shape slots are runtime-readable sources. A compound slot
      // must use an already bound symbol, as in MaterializeValidShapeSymbols.
      for (const auto& param : function_->params_) {
        if (auto tensor = AsTensorTypeLike(param->GetType()); tensor && tensor->tensor_view_) {
          for (const auto& dim : tensor->tensor_view_->valid_shape) {
            if (auto var = AsVarLike(dim)) {
              std::unordered_set<const Expr*> seen;
              CollectShapeSymbols(var, true, seen);
            }
          }
        }
      }
      std::unordered_set<const Expr*> valid;
      for (const auto& param : function_->params_) {
        if (auto tensor = AsTensorTypeLike(param->GetType()); tensor && tensor->tensor_view_) {
          for (const auto& dim : tensor->tensor_view_->valid_shape) CollectShapeSymbols(dim, false, valid);
        }
      }
    }
  }

  Id Variable(const VarPtr& var, bool definition) {
    if (collecting_ && definition) {
      if (!definitions_.insert(var.get()).second) Unsupported("RepeatedDefinition");
      Reserve(var, "bound");
    }
    Binding binding{"free_identity", var->UniqueId()};
    auto found = bindings_.find(var.get());
    if (found != bindings_.end()) {
      binding = found->second;
    } else if (!collecting_ && map_free_) {
      auto [it, inserted] = free_ids_.emplace(var.get(), free_ids_.size());
      binding = {"free_mapped", it->second};
    }
    Fields fields{{"type", Value(var->GetType())}};
    fields.emplace_back("binding", Intern(binding.category, std::to_string(binding.id)));
    // A use references the IterArg binder; its initial value belongs to the
    // enclosing loop's definition, not to every use of that IterArg.
    if (auto iter = As<IterArg>(var); iter && (definition || found == bindings_.end())) {
      path_.emplace_back("initValue");
      fields.emplace_back("initValue", Value(iter->initValue_));
      path_.pop_back();
    }
    return Intern(var->TypeName(), "", std::move(fields));
  }

  Id Node(const IRNodePtr& node) {
    if (!node) return 0;
    if (node->GetKind() == ObjectKind::MemRef || node->GetKind() == ObjectKind::WindowBuffer) {
      Unsupported("UnsupportedStorage");
    }
    if (auto func = As<Function>(node); func && func != function_) {
      if (collecting_) return 0;
      Projection child(arena_, map_free_);
      return child.Run(func);
    }
    const bool definition = defining_;
    defining_ = false;
    // Repeated definitions must be checked before memoization.
    const auto variable = AsVarLike(As<Expr>(node));
    if (collecting_ && definition && variable && definitions_.count(variable.get())) {
      Unsupported("RepeatedDefinition");
    }
    auto& cache = node_cache_[definition ? 1 : 0];
    auto cached = cache.find(node.get());
    if (cached != cache.end()) {
      if (collecting_ && As<Stmt>(node)) Unsupported("SharedStatement");
      defining_ = definition;
      return cached->second;
    }
    if (depth_ >= 512) Unsupported("DepthLimitExceeded");
    if (!active_.insert(node.get()).second) Unsupported("CyclicStructure");
    ++depth_;
    Id result;
    if (variable)
      result = Variable(variable, definition);
    else
      result = DispatchNode(node);
    --depth_;
    active_.erase(node.get());
    cache.emplace(node.get(), result);
    defining_ = definition;
    return result;
  }

  Id DispatchNode(const IRNodePtr& node) {
#define PROJECT_NODE(T)                       \
  if (auto value = As<T>(node)) {             \
    return Reflect(*value, node->TypeName()); \
  }
    PROJECT_NODE(ConstInt)
    PROJECT_NODE(ConstFloat)
    PROJECT_NODE(ConstBool)
    PROJECT_NODE(Call)
    PROJECT_NODE(Submit)
    PROJECT_NODE(MakeTuple)
    PROJECT_NODE(TupleGetItemExpr)
    PROJECT_NODE(BinaryExpr)
    PROJECT_NODE(UnaryExpr)
    PROJECT_NODE(AssignStmt)
    PROJECT_NODE(IfStmt)
    PROJECT_NODE(YieldStmt)
    PROJECT_NODE(ReturnStmt)
    PROJECT_NODE(ForStmt)
    PROJECT_NODE(WhileStmt)
    PROJECT_NODE(InCoreScopeStmt)
    PROJECT_NODE(ClusterScopeStmt)
    PROJECT_NODE(HierarchyScopeStmt)
    PROJECT_NODE(SpmdScopeStmt)
    PROJECT_NODE(SplitAivScopeStmt)
    PROJECT_NODE(RuntimeScopeStmt)
    PROJECT_NODE(CommDomainScopeStmt)
    PROJECT_NODE(GraphScopeStmt)
    PROJECT_NODE(SeqStmts)
    PROJECT_NODE(EvalStmt)
    PROJECT_NODE(BreakStmt)
    PROJECT_NODE(ContinueStmt)
    PROJECT_NODE(InlineStmt)
    PROJECT_NODE(Function)
    PROJECT_NODE(Program)
#undef PROJECT_NODE
    Unsupported("UnsupportedNodeKind: " + node->TypeName());
  }

  Id Type(const TypePtr& type) {
    if (!type) Unsupported("NullType");
    switch (type->GetKind()) {
      case ObjectKind::MemRefType:
      case ObjectKind::PtrType:
      case ObjectKind::WindowBufferType:
      case ObjectKind::CommCtxType:
        Unsupported("UnsupportedStorage");
      case ObjectKind::UnknownType:
        Unsupported("UnresolvedType");
      default:
        break;
    }
    auto cached = type_cache_.find(type.get());
    if (cached != type_cache_.end()) return cached->second;
    if (depth_ >= 512) Unsupported("DepthLimitExceeded");
    if (!active_.insert(type.get()).second) Unsupported("CyclicStructure");
    ++depth_;
    Id result = DispatchType(type);
    --depth_;
    active_.erase(type.get());
    type_cache_.emplace(type.get(), result);
    return result;
  }

  Id DispatchType(const TypePtr& type) {
#define PROJECT_TYPE(T)                       \
  if (auto value = As<T>(type)) {             \
    return Reflect(*value, type->TypeName()); \
  }
    PROJECT_TYPE(VoidType)
    PROJECT_TYPE(ScalarType)
    PROJECT_TYPE(TensorType)
    PROJECT_TYPE(DistributedTensorType)
    PROJECT_TYPE(TileType)
    PROJECT_TYPE(ArrayType)
    PROJECT_TYPE(TupleType)
    PROJECT_TYPE(BufferType)
    PROJECT_TYPE(MultiBufferType)
    PROJECT_TYPE(PrefetchAsyncContextType)
    PROJECT_TYPE(AsyncEventType)
    PROJECT_TYPE(AsyncSessionType)
    PROJECT_TYPE(ShapedType)
#undef PROJECT_TYPE
    Unsupported("UnsupportedTypeKind: " + type->TypeName());
  }

  template <typename T>
  Id Value(const std::shared_ptr<const T>& value) {
    if constexpr (std::is_base_of_v<IRNode, T>) {
      return Node(value);
    } else if constexpr (std::is_base_of_v<ir::Type, T>) {
      return Type(value);
    } else if constexpr (std::is_base_of_v<Op, T>) {
      if (!value) Unsupported("NullOperator");
      return Intern(value->TypeName(), value->name_, {{"kind", Value(value->GetKind())}});
    } else {
      Unsupported("UnsupportedPointerField");
    }
  }

  template <typename T>
  Id Value(const T& value) {
    if constexpr (std::is_enum_v<T>) {
      return Intern(typeid(T).name(), std::to_string(static_cast<int64_t>(value)));
    } else if constexpr (std::is_integral_v<T>) {
      return Intern(typeid(T).name(), std::to_string(value));
    } else if constexpr (std::is_floating_point_v<T>) {
      using Bits = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
      Bits bits;
      static_assert(sizeof(bits) == sizeof(value));
      std::memcpy(&bits, &value, sizeof(bits));
      std::ostringstream out;
      out << "0x" << std::hex << bits;
      return Intern(typeid(T).name(), out.str());
    } else {
      Unsupported("UnsupportedFieldType: " + DemangleTypeName(typeid(T).name()));
    }
  }

  Id Value(const std::string& value) { return Intern("string", value); }
  Id Value(const DataType& value) { return Intern("dtype", value.ToString()); }
  Id Value(const TensorView& value) { return Reflect(value, "TensorView"); }
  Id Value(const TileView& value) { return Reflect(value, "TileView"); }

  template <typename T>
  Id Value(const std::optional<T>& value) {
    return value ? Intern("some", "", {{"value", Value(*value)}}) : Intern("none");
  }

  template <typename T>
  Id Value(const std::vector<T>& values) {
    Fields fields;
    fields.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      std::string index = "[" + std::to_string(i) + "]";
      path_.push_back(index);
      fields.emplace_back(index, Value(values[i]));
      path_.pop_back();
    }
    return Intern("vector", "", std::move(fields));
  }

  template <typename A, typename B>
  Id Value(const std::pair<A, B>& value) {
    return Intern("pair", "", {{"first", Value(value.first)}, {"second", Value(value.second)}});
  }

  template <typename K, typename V, typename C>
  Id Value(const std::map<K, V, C>& values) {
    Fields fields;
    for (const auto& [key, value] : values) {
      path_.push_back(key->name_);
      fields.emplace_back(key->name_, Intern("entry", "", {{"key", Value(key)}, {"value", Value(value)}}));
      path_.pop_back();
    }
    return Intern("map", "", std::move(fields));
  }

  Id Value(const std::vector<std::pair<std::string, std::any>>& attrs) {
    std::map<std::string, const std::any*> sorted;
    for (const auto& [key, value] : attrs) {
      if (!sorted.emplace(key, &value).second) Unsupported("DuplicateAttribute: " + key);
    }
    Fields fields;
    for (const auto& [key, value] : sorted) {
      path_.push_back(key);
      fields.emplace_back(key, Value(*value));
      path_.pop_back();
    }
    return Intern("attrs", "", std::move(fields));
  }

  Id Value(const std::any& value) {
#define PROJECT_ANY(...)                                                                              \
  if (value.type() == typeid(__VA_ARGS__)) {                                                          \
    return Intern("any", #__VA_ARGS__, {{"value", Value(std::any_cast<const __VA_ARGS__&>(value))}}); \
  }
    PROJECT_ANY(int)
    PROJECT_ANY(bool)
    PROJECT_ANY(std::string)
    PROJECT_ANY(double)
    PROJECT_ANY(float)
    PROJECT_ANY(DataType)
    PROJECT_ANY(MemorySpace)
    PROJECT_ANY(TensorLayout)
    PROJECT_ANY(TileLayout)
    PROJECT_ANY(PadValue)
    PROJECT_ANY(ArgDirection)
    PROJECT_ANY(std::vector<ArgDirection>)
    PROJECT_ANY(std::vector<int32_t>)
    PROJECT_ANY(std::vector<VarPtr>)
    PROJECT_ANY(std::vector<std::pair<VarPtr, int>>)
    PROJECT_ANY(std::vector<std::pair<int32_t, int>>)
    PROJECT_ANY(VarPtr)
    PROJECT_ANY(ExprPtr)
#undef PROJECT_ANY
    Unsupported("UnsupportedAttributeType: " + DemangleTypeName(value.type().name()));
  }

  Arena& arena_;
  bool map_free_;
  bool collecting_ = false;
  bool defining_ = false;
  size_t depth_ = 0;
  FunctionPtr function_;
  uint64_t next_bound_ = 0;
  uint64_t next_symbol_ = 0;
  std::unordered_map<const Var*, Binding> bindings_;
  std::unordered_map<const Var*, uint64_t> free_ids_;
  std::unordered_set<const Var*> definitions_;
  std::array<std::unordered_map<const IRNode*, Id>, 2> node_cache_;
  std::unordered_map<const ir::Type*, Id> type_cache_;
  std::unordered_set<const void*> active_;
  std::vector<std::string> path_;
};

template <typename Root>
bool EqualStrict(const Root& lhs, const Root& rhs, bool map_free, bool assert_equal) {
  Arena arena;
  Projection left(arena, map_free);
  Projection right(arena, map_free);
  Id l = left.Run(lhs);
  Id r = right.Run(rhs);
  if (l != r && assert_equal) throw ValueError(arena.Mismatch(l, r));
  return l == r;
}

template <typename Root>
uint64_t HashStrict(const Root& root, bool map_free) {
  Arena arena;
  Projection projection(arena, map_free);
  return arena.Hash(projection.Run(root));
}

void CheckMode(StructuralComparisonMode mode) {
  CHECK(mode == StructuralComparisonMode::Default || mode == StructuralComparisonMode::Strict)
      << "Invalid structural comparison mode";
}

}  // namespace

#define DEFINE_OPTIONS_OVERLOADS(T)                                                                      \
  uint64_t structural_hash(const T& node, const StructuralComparisonOptions& options) {                  \
    CheckMode(options.mode);                                                                             \
    if (options.mode == StructuralComparisonMode::Default) {                                             \
      return structural_hash(node, options.enable_auto_mapping);                                         \
    }                                                                                                    \
    return HashStrict(node, options.enable_auto_mapping);                                                \
  }                                                                                                      \
  bool structural_equal(const T& lhs, const T& rhs, const StructuralComparisonOptions& options) {        \
    CheckMode(options.mode);                                                                             \
    if (options.mode == StructuralComparisonMode::Default) {                                             \
      return structural_equal(lhs, rhs, options.enable_auto_mapping);                                    \
    }                                                                                                    \
    return EqualStrict(lhs, rhs, options.enable_auto_mapping, false);                                    \
  }                                                                                                      \
  void assert_structural_equal(const T& lhs, const T& rhs, const StructuralComparisonOptions& options) { \
    CheckMode(options.mode);                                                                             \
    if (options.mode == StructuralComparisonMode::Default) {                                             \
      assert_structural_equal(lhs, rhs, options.enable_auto_mapping);                                    \
    } else {                                                                                             \
      EqualStrict(lhs, rhs, options.enable_auto_mapping, true);                                          \
    }                                                                                                    \
  }

DEFINE_OPTIONS_OVERLOADS(IRNodePtr)
DEFINE_OPTIONS_OVERLOADS(TypePtr)
#undef DEFINE_OPTIONS_OVERLOADS

}  // namespace pypto::ir
