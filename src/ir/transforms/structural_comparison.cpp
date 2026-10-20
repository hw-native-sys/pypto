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

#include "pypto/ir/transforms/structural_comparison.h"

#include <algorithm>
#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/signature_symbols.h"
#include "pypto/ir/type.h"
#include "utils/structural_comparison_arena.h"
#include "utils/structural_inspection.h"

namespace pypto::ir {
namespace {

using structural_comparison_detail::Fields;
using structural_comparison_detail::Id;
using Arena = structural_comparison_detail::ExactArena<>;

std::string Mismatch(const Arena& arena, Id lhs, Id rhs) {
  std::string path = "root";
  while (true) {
    const auto& [lt, la, lf] = arena.GetKey(lhs);
    const auto& [rt, ra, rf] = arena.GetKey(rhs);
    if (lt != rt || la != ra || lf.size() != rf.size()) {
      return "Structural mismatch at " + path + ": " + lt + "(" + la.substr(0, 128) + ", " +
             std::to_string(lf.size()) + " fields) != " + rt + "(" + ra.substr(0, 128) + ", " +
             std::to_string(rf.size()) + " fields)";
    }
    size_t i = 0;
    while (i < lf.size() && lf[i] == rf[i]) ++i;
    INTERNAL_CHECK(i < lf.size()) << "Different interned records must have a differing field";
    if (lf[i].first != rf[i].first) {
      return "Structural mismatch at " + path + ": field " + lf[i].first.substr(0, 128) +
             " != " + rf[i].first.substr(0, 128);
    }
    if (path.size() < 1024) path += "." + lf[i].first.substr(0, 128);
    lhs = lf[i].second;
    rhs = rf[i].second;
  }
}

struct Binding {
  std::string category;
  uint64_t id;
};

// One reflection projection supplies equality, hashing and diagnostics.
// Preparation reserves DefField identities; the independent verifier also
// checks lexical use intervals.
// Projection uses frozen bindings, with separate definition/reference caches.
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
    node_cache_ = {};
    type_cache_.clear();
    if (verify_bindings_) {
      checking_visibility_ = true;
      Value(root);
      CheckVisibility();
      checking_visibility_ = false;
    }
    collecting_ = false;
    node_cache_ = {};
    type_cache_.clear();
    return Value(root);
  }

  void Verify(const IRNodePtr& root) {
    verify_bindings_ = true;
    Run(root);
  }

  void Inspect(const FunctionPtr& function, structural_comparison_detail::Inspection* inspection) {
    inspection_ = inspection;
    Run(function);
  }

  [[nodiscard]] result_type InitResult() const { return {}; }
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
  bool IsBodyContext() const {
    if (std::find(path_.begin(), path_.end(), "body") == path_.end()) return false;
    for (const auto& field : path_) {
      if (field == "attrs" || field == "kwargs" || field == "type" || field == "return_types") return false;
    }
    return true;
  }

  std::string CurrentPath() const {
    std::string path = "root";
    for (const auto& part : path_) {
      if (path.size() >= 1024) break;
      path += "." + part.substr(0, 128);
    }
    return path.substr(0, 1024);
  }

  [[noreturn]] void Unsupported(const std::string& reason) const {
    throw StructuralComparisonUnsupported(reason, CurrentPath());
  }

  Id Intern(std::string tag, std::string atom = "", Fields fields = {}) {
    if (inspection_) {
      inspection_->weight += 1 + tag.size() + atom.size() + fields.size();
      for (const auto& field : fields) inspection_->weight += field.first.size();
    }
    return collecting_ ? 0 : arena_.Intern(std::move(tag), std::move(atom), std::move(fields));
  }

  template <typename T>
  Id Reflect(const T& value, const std::string& tag, bool skip_body = false) {
    auto fields = std::apply(
        [&](auto&&... descriptors) {
          Fields result;
          auto visit = [&](const auto& desc) {
            // VisitUsualField runs before PushFieldName, so inspect the current
            // descriptor here. Only this scope's body is deferred; metadata and
            // any nested reflected values retain their normal traversal.
            if (skip_body && std::string(desc.name) == "body") return;
            auto field = reflection::FieldIterator<T, Projection, std::decay_t<decltype(desc)>>::Visit(
                value, *this, desc);
            result.insert(result.end(), field.begin(), field.end());
          };
          (visit(descriptors), ...);
          return result;
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

  template <typename T>
  void PrepareFunction(const std::shared_ptr<const T>& root) {
    if constexpr (std::is_base_of_v<IRNode, T>) {
      function_ = As<Function>(std::static_pointer_cast<const IRNode>(root));
      if (!function_) return;
      for (const auto& param : function_->params_) Reserve(param, "bound");
      auto symbols = ClassifySignatureSymbols(function_);
      auto reserve_symbol = [&](const VarPtr& var) {
        auto scalar = As<ScalarType>(var->GetType());
        if (verify_bindings_ &&
            (!scalar || (scalar->dtype_ != DataType::INDEX && scalar->dtype_ != DataType::INT64))) {
          Unsupported("UnsupportedSignatureSymbol");
        }
        Reserve(var, "symbol");
      };
      for (const auto& var : symbols.bound) {
        if (!bindings_.count(var.get())) reserve_symbol(var);
      }
      for (const auto& symbol : symbols.valid_shape) reserve_symbol(symbol.symbol);
      if (verify_bindings_ && !symbols.unbound.empty()) Unsupported("UnboundValidShapeSymbol");
      for (const auto& [var, binding] : bindings_) lifetimes_[var] = {0, kForever};
    }
  }

  static constexpr size_t kForever = std::numeric_limits<size_t>::max();
  struct UseRange {
    size_t first;
    size_t last;
  };
  struct UseEdge {
    size_t child;
    std::string suffix;
  };
  struct UseRecord {
    UseRange range;
    const Var* variable;
    std::vector<UseEdge> children;
    std::string first_path;
    std::string last_path;
  };

  // Expressions/types have no binding side effects. Propagate the earliest and
  // latest use through their DAG in reverse postorder. A lexical binding is one
  // contiguous interval, so checking those two endpoints proves visibility at
  // every use without rescanning a shared expression at each statement.
  size_t BeginUses(const void* object, bool definition, const Var* variable) {
    auto [it, inserted] = use_index_.emplace(std::make_pair(object, definition), uses_.size());
    size_t id = it->second;
    const std::string path = CurrentPath();
    if (inserted) {
      uses_.push_back({{position_, position_}, definition ? nullptr : variable, {}, path, path});
    } else {
      uses_[id].range.last = position_;
      uses_[id].last_path = path;
    }
    if (use_parent_ != kForever) {
      auto& parent = uses_[use_parent_];
      // Children are traversed only on the parent's first use. Retain the
      // relative field path so later uses can report the same leaf accurately.
      parent.children.push_back({id, path.substr(parent.first_path.size())});
    }
    use_parent_ = id;
    return id;
  }

  void CheckVisibility() {
    for (auto it = use_postorder_.rbegin(); it != use_postorder_.rend(); ++it) {
      const auto& source = uses_[*it];
      for (const auto& edge : source.children) {
        auto& target = uses_[edge.child];
        if (source.range.first < target.range.first) {
          target.range.first = source.range.first;
          target.first_path = (source.first_path + edge.suffix).substr(0, 1024);
        }
        if (source.range.last > target.range.last) {
          target.range.last = source.range.last;
          target.last_path = (source.last_path + edge.suffix).substr(0, 1024);
        }
      }
    }
    for (const auto& use : uses_) {
      auto binding = bindings_.find(use.variable);
      if (binding == bindings_.end()) continue;  // True open capture, not a local binder.
      auto lifetime = lifetimes_.find(use.variable);
      if (lifetime == lifetimes_.end() || use.range.first < lifetime->second.first) {
        throw StructuralComparisonUnsupported("BindingNotVisible", use.first_path);
      }
      if (use.range.last >= lifetime->second.last) {
        throw StructuralComparisonUnsupported("BindingNotVisible", use.last_path);
      }
    }
  }

  template <typename T>
  void CheckField(const char* name, const T& value, bool definition = false) {
    path_.emplace_back(name);
    defining_ = definition;
    Value(value);
    defining_ = false;
    path_.pop_back();
  }

  void Activate(const VarPtr& var) {
    lifetimes_[var.get()] = {++position_, kForever};
    scope_definitions_.push_back(var.get());
  }

  void RestoreScope(size_t checkpoint) {
    ++position_;
    while (scope_definitions_.size() > checkpoint) {
      lifetimes_[scope_definitions_.back()].last = position_;
      scope_definitions_.pop_back();
    }
  }

  void DefineResults(const std::vector<VarPtr>& results) {
    CheckField("return_vars", results, true);
    for (const auto& result : results) Activate(result);
  }

  Id CheckStatement(const StmtPtr& stmt) {
    ++position_;
    if (auto assign = As<AssignStmt>(stmt)) {
      CheckField("value", assign->value_);
      CheckField("var", assign->var_, true);
      Activate(assign->var_);
    } else if (auto loop = As<ForStmt>(stmt)) {
      CheckField("attrs", loop->attrs_);
      CheckField("start", loop->start_);
      CheckField("stop", loop->stop_);
      CheckField("step", loop->step_);
      CheckField("loop_var", loop->loop_var_, true);
      CheckField("iter_args", loop->iter_args_, true);
      size_t checkpoint = scope_definitions_.size();
      Activate(loop->loop_var_);
      for (const auto& iter : loop->iter_args_) Activate(iter);
      CheckField("body", loop->body_);
      RestoreScope(checkpoint);
      DefineResults(loop->return_vars_);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      CheckField("iter_args", loop->iter_args_, true);
      size_t checkpoint = scope_definitions_.size();
      for (const auto& iter : loop->iter_args_) Activate(iter);
      CheckField("condition", loop->condition_);
      CheckField("body", loop->body_);
      RestoreScope(checkpoint);
      DefineResults(loop->return_vars_);
    } else if (auto branch = As<IfStmt>(stmt)) {
      CheckField("condition", branch->condition_);
      size_t checkpoint = scope_definitions_.size();
      CheckField("then_body", branch->then_body_);
      RestoreScope(checkpoint);
      CheckField("else_body", branch->else_body_);
      RestoreScope(checkpoint);
      DefineResults(branch->return_vars_);
    } else if (auto scope = As<ScopeStmt>(stmt)) {
      // Scope metadata is evaluated on entry; these straight-line regions do
      // not hide definitions from subsequent statements (see ScopeStmt).
      DispatchNode(stmt, true);
      CheckField("body", scope->body_);
    } else {
      DispatchNode(stmt);
    }
    return 0;
  }

  Id Variable(const VarPtr& var, bool definition) {
    if (collecting_ && !checking_visibility_ && definition) {
      if (!definitions_.insert(var.get()).second && verify_bindings_) Unsupported("RepeatedDefinition");
      Reserve(var, "bound");
    }
    Binding binding{"free_identity", var->UniqueId()};
    auto found = bindings_.find(var.get());
    if (found != bindings_.end()) {
      binding = found->second;
    } else if (!collecting_ &&
               (implicit_allocations_.count(var.get()) || implicit_storage_.count(var.get()))) {
      auto [it, inserted] = implicit_ids_.emplace(var.get(), implicit_ids_.size());
      binding = {implicit_allocations_.count(var.get()) ? "allocation" : "storage", it->second};
    } else if (!collecting_ && map_free_) {
      auto [it, inserted] = free_ids_.emplace(var.get(), free_ids_.size());
      binding = {"free_mapped", it->second};
    }
    if (inspection_ && !collecting_ && found == bindings_.end()) {
      inspection_->free_variables.push_back(var);
    }
    path_.emplace_back("type");
    Fields fields{{"type", Value(var->GetType())}};
    path_.pop_back();
    fields.emplace_back("binding", Intern(binding.category, std::to_string(binding.id)));
    // A use references the IterArg binder; its initial value belongs to the
    // enclosing loop's definition, not to every use of that IterArg.
    if (auto iter = As<IterArg>(var); iter && (definition || found == bindings_.end())) {
      path_.emplace_back("initValue");
      fields.emplace_back("initValue", Value(iter->initValue_));
      path_.pop_back();
    }
    if (var->GetKind() == ObjectKind::WindowBuffer) {
      fields.emplace_back("storage", DispatchNode(var));
    }
    return Intern(var->TypeName(), "", std::move(fields));
  }

  Id Node(const IRNodePtr& node) {
    if (!node) return 0;
    if (auto func = As<Function>(node); func && func != function_) {
      if (collecting_) return 0;
      Projection child(arena_, map_free_);
      child.verify_bindings_ = verify_bindings_;
      return child.Run(func);
    }
    const bool definition = defining_;
    defining_ = false;
    // Repeated definitions must be checked before memoization.
    VarPtr variable = AsVarLike(As<Expr>(node));
    if (node->GetKind() == ObjectKind::MemRef || node->GetKind() == ObjectKind::WindowBuffer) {
      // MemRef is a value descriptor: allocation identity is its base, not
      // the descriptor object's address. WindowBuffer is a bound resource.
      if (node->GetKind() == ObjectKind::WindowBuffer) {
        variable = std::static_pointer_cast<const Var>(node);
      }
      if (collecting_ && function_) {
        // Storage descriptors in a function declare implicit resource identities.
        // Preserve their alias pattern across serialization and renaming.
        const auto base = node->GetKind() == ObjectKind::MemRef
                              ? std::static_pointer_cast<const MemRef>(node)->base_
                              : std::static_pointer_cast<const WindowBuffer>(node)->base_;
        implicit_allocations_.insert(base.get());
        if (variable) implicit_storage_.insert(variable.get());
      }
    }
    if (verify_bindings_ && collecting_ && !checking_visibility_ && definition && variable &&
        definitions_.count(variable.get())) {
      Unsupported("RepeatedDefinition");
    }
    const size_t saved_parent = use_parent_;
    size_t use_id = kForever;
    if (checking_visibility_) {
      if (As<Expr>(node)) {
        use_id = BeginUses(node.get(), definition, variable.get());
      } else {
        use_parent_ = kForever;
      }
    }
    // Inspection must observe both uses when one DAG node appears in executable
    // code and in metadata: the latter reference cannot be rewritten. Equality
    // and binding verification retain their original two-cache behavior.
    const size_t context = inspection_ && !collecting_ && !IsBodyContext() ? 2 : 0;
    if (inspection_ && !collecting_) {
      inspection_->node_uses.push_back({node, inspection_parent_, IsBodyContext()});
    }
    auto& cache = node_cache_[context + (definition ? 1 : 0)];
    auto cached = cache.find(node.get());
    if (cached != cache.end()) {
      if (verify_bindings_ && collecting_ && As<Stmt>(node)) Unsupported("SharedStatement");
      defining_ = definition;
      use_parent_ = saved_parent;
      return cached->second;
    }
    if (inspection_ && !collecting_) inspection_->nodes.push_back(node);
    if (depth_ >= 512) Unsupported("DepthLimitExceeded");
    if (!active_.insert(node.get()).second) Unsupported("CyclicStructure");
    ++depth_;
    const auto saved_inspection_parent = inspection_parent_;
    const bool saved_inspection_body = inspection_parent_in_body_;
    if (inspection_ && !collecting_) {
      inspection_parent_ = node;
      inspection_parent_in_body_ = IsBodyContext();
    }
    Id result;
    if (variable) {
      result = Variable(variable, definition);
    } else if (checking_visibility_ && As<Stmt>(node)) {
      result = CheckStatement(As<Stmt>(node));
    } else {
      result = DispatchNode(node);
    }
    inspection_parent_ = saved_inspection_parent;
    inspection_parent_in_body_ = saved_inspection_body;
    if (use_id != kForever) use_postorder_.push_back(use_id);
    use_parent_ = saved_parent;
    --depth_;
    active_.erase(node.get());
    cache.emplace(node.get(), result);
    defining_ = definition;
    return result;
  }

  Id DispatchNode(const IRNodePtr& node, bool skip_body = false) {
#define PROJECT_NODE(T)                                  \
  if (auto value = As<T>(node)) {                        \
    return Reflect(*value, node->TypeName(), skip_body); \
  }
    PROJECT_NODE(MemRef)
    PROJECT_NODE(WindowBuffer)
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
    if (!type) return 0;
    if (inspection_ && !collecting_) {
      auto owner = As<Expr>(inspection_parent_);
      if (path_.empty() || path_.back() != "type" || !owner || owner->GetType() != type) owner = nullptr;
      inspection_->type_uses.push_back({type, owner, inspection_parent_in_body_});
    }
    const size_t saved_parent = use_parent_;
    size_t use_id = checking_visibility_ ? BeginUses(type.get(), false, nullptr) : kForever;
    auto cached = type_cache_.find(type.get());
    if (cached != type_cache_.end()) {
      use_parent_ = saved_parent;
      return cached->second;
    }
    if (depth_ >= 512) Unsupported("DepthLimitExceeded");
    if (!active_.insert(type.get()).second) Unsupported("CyclicStructure");
    ++depth_;
    if (inspection_ && !collecting_) inspection_->types.push_back(type);
    Id result = DispatchType(type);
    if (use_id != kForever) use_postorder_.push_back(use_id);
    use_parent_ = saved_parent;
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
    PROJECT_TYPE(UnknownType)
    PROJECT_TYPE(MemRefType)
    PROJECT_TYPE(PtrType)
    PROJECT_TYPE(WindowBufferType)
    PROJECT_TYPE(CommCtxType)
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
      if (inspection_ && !collecting_) {
        const bool body_callee = !path_.empty() && path_.back() == "op" && IsBodyContext();
        inspection_->operators.emplace_back(value, body_callee);
      }
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
  Id Value(const DataType& value) { return Intern("dtype", std::to_string(value.Code())); }
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

  structural_comparison_detail::Inspection* inspection_ = nullptr;
  IRNodePtr inspection_parent_;
  bool inspection_parent_in_body_ = false;
  Arena& arena_;
  bool map_free_;
  bool verify_bindings_ = false;
  bool collecting_ = false;
  bool checking_visibility_ = false;
  size_t position_ = 0;
  size_t use_parent_ = kForever;
  std::map<std::pair<const void*, bool>, size_t> use_index_;
  std::vector<UseRecord> uses_;
  std::vector<size_t> use_postorder_;
  std::unordered_map<const Var*, UseRange> lifetimes_;
  std::vector<const Var*> scope_definitions_;
  bool defining_ = false;
  size_t depth_ = 0;
  FunctionPtr function_;
  uint64_t next_bound_ = 0;
  uint64_t next_symbol_ = 0;
  std::unordered_map<const Var*, Binding> bindings_;
  std::unordered_map<const Var*, uint64_t> free_ids_;
  std::unordered_map<const Var*, uint64_t> implicit_ids_;
  std::unordered_set<const Var*> implicit_allocations_;
  std::unordered_set<const Var*> implicit_storage_;
  std::unordered_set<const Var*> definitions_;
  std::array<std::unordered_map<const IRNode*, Id>, 4> node_cache_;
  std::unordered_map<const ir::Type*, Id> type_cache_;
  std::unordered_set<const void*> active_;
  std::vector<std::string> path_;
};

template <typename Root>
bool EqualStructure(const Root& lhs, const Root& rhs, bool map_free, bool assert_equal) {
  Arena arena;
  Projection left(arena, map_free);
  Projection right(arena, map_free);
  Id l = left.Run(lhs);
  Id r = right.Run(rhs);
  if (l != r && assert_equal) throw ValueError(Mismatch(arena, l, r));
  return l == r;
}

template <typename Root>
uint64_t HashStructure(const Root& root, bool map_free) {
  Arena arena;
  Projection projection(arena, map_free);
  return arena.Hash(projection.Run(root));
}

}  // namespace

structural_comparison_detail::Inspection structural_comparison_detail::Inspect(const FunctionPtr& function) {
  Arena arena;
  Projection projection(arena, false);
  Inspection inspection;
  projection.Inspect(function, &inspection);
  return inspection;
}

void verify_structural_bindings(const IRNodePtr& node) {
  Arena arena;
  Projection projection(arena, false);
  projection.Verify(node);
}

#define DEFINE_COMPARISON_OVERLOADS(T)                                                 \
  uint64_t structural_hash(const T& node, bool enable_auto_mapping) {                  \
    return HashStructure(node, enable_auto_mapping);                                   \
  }                                                                                    \
  bool structural_equal(const T& lhs, const T& rhs, bool enable_auto_mapping) {        \
    return EqualStructure(lhs, rhs, enable_auto_mapping, false);                       \
  }                                                                                    \
  void assert_structural_equal(const T& lhs, const T& rhs, bool enable_auto_mapping) { \
    EqualStructure(lhs, rhs, enable_auto_mapping, true);                               \
  }

DEFINE_COMPARISON_OVERLOADS(IRNodePtr)
DEFINE_COMPARISON_OVERLOADS(TypePtr)
#undef DEFINE_COMPARISON_OVERLOADS

}  // namespace pypto::ir
