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
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Complexity: the identity graph and IR walks are O(N + E). Coverage uses shared
// canonical interval snapshots, not copied dispatch histories. Shared type DAGs
// are indexed once; graph tuple fields omit ordinary scalar-only positions and
// unchanged functions bypass mutation. Each distinct
// snapshot union costs O(I_left + I_right), plus ordered canonical-key comparison
// costs; direct dispatch intervals accumulate in O(D log D), and worker IDs are
// enumerated only in the emitted domain lists. Identical snapshot pairs are
// cached. Distinct overlapping collective unions can still take quadratic work
// in IR size even when final output is small; canonical comparisons also inspect
// interval keys. This residual coverage exception needs explicit approval before
// merge; it is not a universal O(N log N) bound.
using ValueId = size_t;

// Window identity follows storage, never an arbitrary tensor computation.
// Registry-backed helpers are shared with the other alias/return analyses.
std::optional<size_t> WindowSourceArg(const CallPtr& call) {
  if (auto index = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size())) return index;
  if (op_predicates::IsBufferAliasingViewOp(call->op_->name_) && !call->args_.empty()) return 0;
  if ((IsOp(call, "pld.tensor.allreduce") || IsOp(call, "pld.tensor.broadcast") ||
       IsOp(call, "pld.tensor.reduce_scatter")) &&
      !call->args_.empty())
    return 0;
  if ((IsOp(call, "pld.tensor.all_to_all") || IsOp(call, "pld.tensor.allgather") ||
       IsOp(call, "pld.tensor.all_to_all_v")) &&
      call->args_.size() > 1)
    return 1;
  return std::nullopt;
}

/// One bounded graph for the entire program, including tuple positions and
/// declared returns. Each identity changes from unknown to one object once;
/// every edge/consensus input is processed once. No function-order dependency,
/// recursive callee searches, per-value lineage walks, or body fixed points.
class WindowIdentityGraph : public IRVisitor {
 public:
  explicit WindowIdentityGraph(const ProgramPtr& program) {
    for (const auto& [gvar, func] : program->functions_) {
      current_function_ = func.get();
      Facts facts{func,
                  transform_utils::TrailingCommCtxParamCount(func),
                  {},
                  return_lineage::ExplicitReturnedParamIndices(func)};
      for (const auto& type : func->return_types_) facts.returns.push_back(NewValue(type, func->span_));
      // Annotation-free DSL functions retain an empty declared signature even
      // when their calls have body-derived result types. Match that ABI: one
      // position per ReturnStmt value, including a packed Tuple as one value.
      // These analysis-only positions must not populate return_types_ later.
      if (func->return_types_.empty()) {
        if (auto returned = return_lineage::FindFirstReturn(func->body_)) {
          for (const auto& value : returned->value_) {
            facts.returns.push_back(NewValue(value->GetType(), value->span_));
          }
        }
      }
      for (ValueId result : facts.returns) function_roots_[func.get()].insert(result);
      functions_.emplace(func->name_, std::move(facts));
      for (const auto& param : func->params_) Value(param);
    }
    for (const auto& [gvar, func] : program->functions_) {
      current_ = &functions_.at(func->name_);
      current_function_ = func.get();
      // Function attributes are live references at parameter binding, including
      // expression trees whose aliases and tuple results exist only in attrs.
      for (const auto& [key, value] : func->attrs_) {
        ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
      }
      VisitStmt(func->body_);
    }
    current_ = nullptr;
    current_function_ = nullptr;
  }

  bool FunctionNeedsRewrite(const FunctionPtr& func) const { return changed_functions_.count(func.get()); }

  void SeedLocal(const ExprPtr& expr, const WindowBufferPtr& window) {
    local_windows_.insert(window.get());
    Set(Value(expr), window);
  }

  void Propagate() {
    // Queue entries are indices rather than references: constructing the graph
    // may grow values_, whereas propagation never adds graph vertices/edges.
    for (size_t next = 0; next < ready_.size(); ++next) {
      ValueId source = ready_[next];
      auto window = values_[source].window;
      for (ValueId target : values_[source].users) Set(target, window);
      for (size_t join_id : values_[source].joins) {
        auto& join = joins_[join_id];
        CHECK_SPAN(!join.window || join.window.get() == window.get(), values_[join.target].span)
            << "MaterializeCommDomainScopes: conflicting WindowBuffer identities at a branch/loop merge";
        join.window = window;
        if (--join.remaining == 0) Set(join.target, window);
      }
    }
    std::vector<bool> changed(values_.size(), false);
    for (ValueId id = 0; id < values_.size(); ++id) {
      const auto& value = values_[id];
      if (!value.window) continue;
      CHECK_SPAN(local_windows_.count(value.window.get()) || owners_.count(value.window.get()), value.span)
          << "MaterializeCommDomainScopes: orphan WindowBuffer identity has no owning communication scope";
      auto original = As<DistributedTensorType>(value.type);
      changed[id] =
          !original->window_buffer_.has_value() || original->window_buffer_->get() != value.window.get();
    }
    // Containment children are allocated after their parent. Compute each root
    // flag once, then follow real function/root uses; never expand a tuple's
    // descendants once per function sharing that same native expression DAG.
    for (size_t id = values_.size(); id-- > 0;) {
      for (const auto& [position, child] : values_[id].fields) {
        changed[id] = changed[id] || changed[child];
      }
    }
    // A scalar/CommCtx expression may contain a Call whose DT result changed.
    // Propagate that mutation requirement through the shared expression DAG,
    // separately from tuple type containment, before consulting function users.
    std::vector<ValueId> changed_roots;
    for (ValueId id = 0; id < changed.size(); ++id) {
      if (changed[id]) changed_roots.push_back(id);
    }
    for (size_t next = 0; next < changed_roots.size(); ++next) {
      auto parents = expression_parents_.find(changed_roots[next]);
      if (parents == expression_parents_.end()) continue;
      for (ValueId parent : parents->second) {
        if (changed[parent]) continue;
        changed[parent] = true;
        changed_roots.push_back(parent);
      }
    }
    for (const auto& [function, roots] : function_roots_) {
      for (ValueId root : roots) {
        if (changed[root]) {
          changed_functions_.insert(function);
          break;
        }
      }
    }
  }

  WindowBufferPtr Identity(const ExprPtr& expr) const {
    auto found = expr_values_.find(expr.get());
    return found == expr_values_.end() ? nullptr : values_[found->second].window;
  }

  TypePtr RewriteType(const ExprPtr& expr, const TypePtr& type) const {
    auto found = expr_values_.find(expr.get());
    return found == expr_values_.end() ? type : RewriteValueType(found->second, type);
  }

  std::vector<TypePtr> RewriteReturns(const FunctionPtr& func) const {
    const auto& returns = functions_.at(func->name_).returns;
    std::vector<TypePtr> types;
    for (size_t i = 0; i < func->return_types_.size(); ++i) {
      types.push_back(RewriteValueType(returns[i], func->return_types_[i]));
    }
    return types;
  }

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr) return;
    ValueId id = Value(expr);  // Record this function's use before the global memo.
    if (active_parent_) expression_parents_[id].push_back(*active_parent_);
    if (!visited_.insert(expr.get()).second) return;
    auto saved = active_parent_;
    active_parent_ = id;
    IRVisitor::VisitExpr(expr);
    active_parent_ = saved;
  }

 private:
  struct ValueInfo {
    TypePtr type;
    Span span;
    WindowBufferPtr window;
    std::unordered_map<size_t, ValueId> fields;  // Original tuple positions; scalar fields are absent.
    std::vector<ValueId> users;
    std::vector<size_t> joins;
  };
  struct Join {
    ValueId target;
    size_t remaining;
    WindowBufferPtr window;
  };
  struct Facts {
    FunctionPtr func;
    size_t ctx;
    std::vector<ValueId> returns;
    std::vector<std::optional<size_t>> explicit_params;
  };

  bool ContainsDistributedTensor(const TypePtr& type) {
    auto found = type_relevance_.find(type.get());
    if (found != type_relevance_.end()) return found->second;
    bool relevant = IsA<DistributedTensorType>(type);
    if (auto tuple = As<TupleType>(type)) {
      std::vector<size_t> positions;
      for (size_t i = 0; i < tuple->types_.size(); ++i) {
        if (ContainsDistributedTensor(tuple->types_[i])) positions.push_back(i);
      }
      relevant = !positions.empty();
      tuple_positions_.emplace(type.get(), std::move(positions));
    }
    type_relevance_.emplace(type.get(), relevant);
    return relevant;
  }

  ValueId NewValue(const TypePtr& type, const Span& span) {
    ValueId id = values_.size();
    values_.push_back({type, span, nullptr, {}, {}, {}});
    if (!ContainsDistributedTensor(type)) return id;
    if (auto dt = As<DistributedTensorType>(type)) {
      if (dt->window_buffer_.has_value()) Set(id, *dt->window_buffer_);
    } else if (auto tuple = As<TupleType>(type)) {
      for (size_t position : tuple_positions_.at(type.get())) {
        ValueId child = NewValue(tuple->types_[position], span);
        values_[id].fields.emplace(position, child);
      }
    }
    return id;
  }

  ValueId Value(const ExprPtr& expr) {
    auto found = expr_values_.find(expr.get());
    ValueId id;
    if (found != expr_values_.end()) {
      id = found->second;
    } else {
      id = NewValue(expr->GetType(), expr->span_);
      expr_values_.emplace(expr.get(), id);
    }
    if (current_function_) function_roots_[current_function_].insert(id);
    return id;
  }

  void Set(ValueId id, const WindowBufferPtr& window) {
    if (!window || !As<DistributedTensorType>(values_[id].type)) return;
    auto& value = values_[id];
    CHECK_SPAN(!value.window || value.window.get() == window.get(), value.span)
        << "MaterializeCommDomainScopes: conflicting WindowBuffer identities for one value/formal";
    if (!value.window) {
      value.window = window;
      ready_.push_back(id);
    }
  }

  void Connect(ValueId source, ValueId target) {
    if (As<DistributedTensorType>(values_[source].type) && As<DistributedTensorType>(values_[target].type)) {
      values_[source].users.push_back(target);
    } else {
      for (const auto& [position, child] : values_[source].fields) {
        auto found = values_[target].fields.find(position);
        if (found != values_[target].fields.end()) Connect(child, found->second);
      }
    }
  }

  void Consensus(ValueId left, ValueId right, ValueId target) {
    if (As<DistributedTensorType>(values_[target].type) && As<DistributedTensorType>(values_[left].type) &&
        As<DistributedTensorType>(values_[right].type)) {
      size_t id = joins_.size();
      joins_.push_back({target, 2, nullptr});
      values_[left].joins.push_back(id);
      values_[right].joins.push_back(id);
    } else {
      for (const auto& [position, child] : values_[target].fields) {
        auto l = values_[left].fields.find(position);
        auto r = values_[right].fields.find(position);
        if (l != values_[left].fields.end() && r != values_[right].fields.end()) {
          Consensus(l->second, r->second, child);
        }
      }
    }
  }

  TypePtr RewriteValueType(ValueId id, const TypePtr& type) const {
    const auto& value = values_[id];
    if (auto dt = As<DistributedTensorType>(type); dt && value.window) {
      if (dt->window_buffer_.has_value() && dt->window_buffer_->get() == value.window.get()) return type;
      // Only identity changes. In particular, retain the callee's own MemRef,
      // view, dtype and shape so actual type disagreements remain observable.
      return std::make_shared<const DistributedTensorType>(dt->shape_, dt->dtype_, dt->memref_,
                                                           dt->tensor_view_, value.window);
    }
    auto tuple = As<TupleType>(type);
    if (!tuple || value.fields.empty()) return type;
    std::vector<TypePtr> types;
    for (const auto& [position, child] : value.fields) {
      if (position >= tuple->types_.size()) continue;
      auto field = RewriteValueType(child, tuple->types_[position]);
      if (field.get() == tuple->types_[position].get()) continue;
      if (types.empty()) types = tuple->types_;
      types[position] = std::move(field);
    }
    return types.empty() ? type : std::make_shared<const TupleType>(std::move(types));
  }

  std::vector<std::pair<size_t, ValueId>> Results(const ExprPtr& expr, size_t count) {
    ValueId id = Value(expr);
    if (!As<Submit>(expr) && count == 1) return {{0, id}};
    auto tuple = As<TupleType>(expr->GetType());
    if (!tuple || tuple->types_.size() < count) return {};
    std::vector<std::pair<size_t, ValueId>> results;
    results.reserve(values_[id].fields.size());
    for (const auto& [position, child] : values_[id].fields) {
      if (position < count) results.emplace_back(position, child);
    }
    return results;
  }

  void BindCall(const ExprPtr& expr) {
    auto call = transform_utils::AsCallOrSubmitView(expr);
    auto gvar = As<GlobalVar>(call->op_);
    if (!gvar) {
      if (auto source = WindowSourceArg(call)) Connect(Value(call->args_[*source]), Value(expr));
      return;
    }
    auto found = functions_.find(gvar->name_);
    if (found == functions_.end()) return;
    const auto& facts = found->second;
    const auto& callee = facts.func;
    if (call->args_.size() < facts.ctx || call->args_.size() > callee->params_.size()) return;
    if (!As<Submit>(expr) && call->args_.size() != callee->params_.size()) return;
    const size_t supplied = call->args_.size() - facts.ctx;
    // Traverse actual args, never all formal params for each call. The shared
    // binding helper receives the once-per-callee ctx count; omitted Out params
    // have no caller provenance and cannot consume an argument from the suffix.
    for (size_t i = 0; i < supplied; ++i) {
      auto actual = transform_utils::CallerSuppliedArg(call, callee, i, facts.ctx);
      Connect(Value(actual), Value(callee->params_[i]));
    }
    auto outputs = Results(expr, facts.returns.size());
    for (const auto& [i, output] : outputs) {
      Connect(facts.returns[i], output);
      if (i < facts.explicit_params.size() && facts.explicit_params[i].has_value()) {
        auto actual = transform_utils::CallerSuppliedArg(call, callee, *facts.explicit_params[i], facts.ctx);
        if (actual) Connect(Value(actual), output);
      }
    }
  }

  void VisitExpr_(const CallPtr& op) override {
    BindCall(op);
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubmitPtr& op) override {
    BindCall(op);
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const IterArgPtr& op) override {
    Connect(Value(op->initValue_), Value(op));
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const MakeTuplePtr& op) override {
    ValueId tuple = Value(op);
    // Value() can append to values_; do not retain a field-map reference across it.
    for (size_t i = 0; i < op->elements_.size(); ++i) {
      auto found = values_[tuple].fields.find(i);
      if (found == values_[tuple].fields.end()) continue;
      ValueId child = found->second;
      Connect(Value(op->elements_[i]), child);
    }
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const TupleGetItemExprPtr& op) override {
    ValueId tuple = Value(op->tuple_);
    auto found = values_[tuple].fields.find(static_cast<size_t>(op->index_));
    if (op->index_ >= 0 && found != values_[tuple].fields.end()) {
      ValueId child = found->second;
      Connect(child, Value(op));
    }
    IRVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const AssignStmtPtr& op) override {
    Connect(Value(op->value_), Value(op->var_));
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ReturnStmtPtr& op) override {
    if (current_) {
      if (op->value_.size() == current_->returns.size()) {
        for (size_t i = 0; i < op->value_.size(); ++i) Connect(Value(op->value_[i]), current_->returns[i]);
      } else if (op->value_.size() == 1) {
        ValueId packed = Value(op->value_[0]);
        auto tuple = As<TupleType>(op->value_[0]->GetType());
        if (tuple && tuple->types_.size() == current_->returns.size()) {
          for (const auto& [position, child] : values_[packed].fields) {
            Connect(child, current_->returns[position]);
          }
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }
  template <typename LoopPtr>
  void BindLoop(const LoopPtr& loop) {
    auto yield = transform_utils::GetLastYieldStmt(loop->body_);
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      ValueId init = Value(loop->iter_args_[i]->initValue_);
      ValueId carry = Value(loop->iter_args_[i]);
      Connect(init, carry);
      if (yield && i < yield->value_.size()) {
        ValueId yielded = Value(yield->value_[i]);
        Connect(yielded, carry);
        if (i < loop->return_vars_.size()) Consensus(init, yielded, Value(loop->return_vars_[i]));
      }
    }
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    BindLoop(op);
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    BindLoop(op);
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    auto left = transform_utils::GetLastYieldStmt(op->then_body_);
    auto right = op->else_body_ ? transform_utils::GetLastYieldStmt(*op->else_body_) : nullptr;
    if (left && right) {
      size_t count = std::min({left->value_.size(), right->value_.size(), op->return_vars_.size()});
      for (size_t i = 0; i < count; ++i) {
        Consensus(Value(left->value_[i]), Value(right->value_[i]), Value(op->return_vars_[i]));
      }
    }
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    for (const auto& slot : op->slots_) {
      auto [found, inserted] = owners_.emplace(slot.get(), op.get());
      CHECK_SPAN(inserted || found->second == op.get(), op->span_)
          << "MaterializeCommDomainScopes: WindowBuffer has multiple owning scopes";
    }
    IRVisitor::VisitStmt_(op);
  }

  std::vector<ValueInfo> values_;
  std::vector<Join> joins_;
  std::vector<ValueId> ready_;
  std::unordered_map<const Expr*, ValueId> expr_values_;
  std::unordered_set<const Expr*> visited_;
  std::unordered_map<std::string, Facts> functions_;
  std::unordered_map<const WindowBuffer*, const CommDomainScopeStmt*> owners_;
  std::unordered_set<const WindowBuffer*> local_windows_;
  std::unordered_map<const Type*, bool> type_relevance_;
  std::unordered_map<const Type*, std::vector<size_t>> tuple_positions_;
  std::unordered_map<const Function*, std::unordered_set<ValueId>> function_roots_;
  std::unordered_map<ValueId, std::vector<ValueId>> expression_parents_;
  std::optional<ValueId> active_parent_;
  std::unordered_set<const Function*> changed_functions_;
  const Function* current_function_ = nullptr;
  Facts* current_ = nullptr;
};

/// Device coverage descriptor inferred from a dispatch ``worker=`` expression.
struct DeviceDescriptor {
  bool is_all = false;
  // Sorted, disjoint inclusive intervals. Numeric loop bounds do not enumerate
  // workers until one final device list is emitted for each distinct domain.
  std::vector<std::pair<int64_t, int64_t>> intervals;

  bool operator<(const DeviceDescriptor& other) const {
    if (is_all != other.is_all) return is_all < other.is_all;
    return !is_all && intervals < other.intervals;
  }

  void Merge(const DeviceDescriptor& other) {
    if (is_all || other.is_all) {
      is_all = true;
      intervals.clear();
      return;
    }
    std::vector<std::pair<int64_t, int64_t>> merged;
    merged.reserve(intervals.size() + other.intervals.size());
    size_t left = 0;
    size_t right = 0;
    while (left < intervals.size() || right < other.intervals.size()) {
      auto next = right == other.intervals.size() ||
                          (left < intervals.size() && intervals[left] < other.intervals[right])
                      ? intervals[left++]
                      : other.intervals[right++];
      if (!merged.empty() && (merged.back().second == std::numeric_limits<int64_t>::max() ||
                              next.first <= merged.back().second + 1)) {
        merged.back().second = std::max(merged.back().second, next.second);
      } else {
        merged.push_back(next);
      }
    }
    intervals = std::move(merged);
  }

  [[nodiscard]] std::vector<int64_t> ToDevices() const {
    std::vector<int64_t> devices;
    if (is_all) return devices;
    for (const auto& [first, last] : intervals) {
      for (int64_t worker = first;; ++worker) {
        devices.push_back(worker);
        if (worker == last) break;  // Also avoids incrementing INT64_MAX.
      }
    }
    return devices;
  }
};

/// Direct dispatches accumulate mutably: sparse worker annotations must not copy
/// a growing immutable prefix at every dispatch. Erased intervals are visited
/// only once, giving O(D log D) accumulation for D direct interval insertions.
struct DispatchCoverage {
  bool observed = false;
  bool is_all = false;
  std::map<int64_t, int64_t> intervals;

  void Add(const DeviceDescriptor& descriptor) {
    observed = true;
    if (is_all) return;
    if (descriptor.is_all) {
      is_all = true;
      intervals.clear();
      return;
    }
    for (auto [first, last] : descriptor.intervals) {
      auto next = intervals.lower_bound(first);
      if (next != intervals.begin()) {
        auto previous = std::prev(next);
        if (previous->second >= first - 1) {
          first = previous->first;
          last = std::max(last, previous->second);
          next = intervals.erase(previous);
        }
      }
      while (next != intervals.end() &&
             (last == std::numeric_limits<int64_t>::max() || next->first <= last + 1)) {
        last = std::max(last, next->second);
        next = intervals.erase(next);
      }
      intervals.emplace_hint(next, first, last);
    }
  }

  DeviceDescriptor Freeze() const {
    DeviceDescriptor descriptor;
    descriptor.is_all = is_all;
    descriptor.intervals.assign(intervals.begin(), intervals.end());
    return descriptor;
  }
};

/// Immutable canonical snapshots share large identical coverage sets. A cached
/// union is keyed by snapshots, not allocation pairs: a later A -> S consumer
/// must observe a new A snapshot after an intervening B -> A consumer.
class CoverageSnapshots {
 public:
  CoverageSnapshots() { Intern({}); }  // ID zero is the empty set.

  size_t Intern(DeviceDescriptor descriptor) {
    auto [entry, inserted] = canonical_.try_emplace(std::move(descriptor), snapshots_.size());
    if (inserted) snapshots_.push_back(&entry->first);
    return entry->second;
  }

  size_t Merge(size_t left, size_t right) {
    if (left == right || right == 0) return left;
    if (left == 0) return right;
    if (Get(left).is_all) return left;
    if (Get(right).is_all) return right;
    auto key = std::make_pair(std::min(left, right), std::max(left, right));
    auto cached = unions_.find(key);
    if (cached != unions_.end()) return cached->second;
    DeviceDescriptor merged = Get(left);
    merged.Merge(Get(right));
    size_t result = Intern(std::move(merged));
    unions_.emplace(key, result);
    return result;
  }

  const DeviceDescriptor& Get(size_t id) const { return *snapshots_[id]; }

 private:
  std::map<DeviceDescriptor, size_t> canonical_;
  std::vector<const DeviceDescriptor*> snapshots_;
  std::map<std::pair<size_t, size_t>, size_t> unions_;
};

/// Per-alloc bookkeeping populated during the host_orch scan.
struct AllocRecord {
  CallPtr alloc_call;                  ///< pld.tensor.alloc_window_buffer Call
  VarPtr ptr_var;                      ///< AssignStmt LHS (Var of PtrType)
  ExprPtr size_expr;                   ///< alloc_call->args_[0]
  std::string name;                    ///< from alloc_call attr "name"
  Span span;                           ///< alloc_call->span_ (const fields → emplaced)
  DispatchCoverage dispatch_coverage;  ///< mutable direct-dispatch evidence
  std::optional<size_t> coverage;      ///< immutable snapshot ID; nullopt means no consumer
  WindowBufferPtr wb;                  ///< filled after construction

  AllocRecord(CallPtr ac, VarPtr pv, ExprPtr sz, std::string nm, Span sp)
      : alloc_call(std::move(ac)),
        ptr_var(std::move(pv)),
        size_expr(std::move(sz)),
        name(std::move(nm)),
        span(std::move(sp)) {}
};

/// Per-pld.tensor.window result Var: maps the LHS Var pointer back to its alloc.
struct WindowRecord {
  CallPtr window_call;
  VarPtr old_view_var;
  AllocRecord* alloc;
};

struct CollectiveConsumer {
  AllocRecord* data_alloc;  ///< nullptr for barrier-only consumers
  AllocRecord* signal_alloc;
  Span span;
};

/// Scans a host_orch function body once and records every
/// ``pld.tensor.alloc_window_buffer`` and ``pld.tensor.window`` assignment.
class AllocAndWindowCollector : public IRVisitor {
 public:
  // Existing scopes already own their slots. Inherited formals and allocations
  // under an existing scope must never be materialized as fresh local slots.
  void VisitStmt_(const CommDomainScopeStmtPtr&) override {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    auto var = AsVarLike(op->var_);
    auto call = As<Call>(op->value_);
    if (var && call && call->op_) {
      if (IsOp(call, "pld.tensor.alloc_window_buffer")) {
        INTERNAL_CHECK_SPAN(call->args_.size() == 1, call->span_)
            << "MaterializeCommDomainScopes: pld.tensor.alloc_window_buffer expects exactly one arg (size)";
        // The parser injects ``name`` as a kwarg derived from the assignment
        // LHS — not as an ``attrs`` entry — so use GetKwarg here.
        auto name = call->GetKwarg<std::string>("name");
        INTERNAL_CHECK_SPAN(!name.empty(), call->span_)
            << "MaterializeCommDomainScopes: pld.tensor.alloc_window_buffer missing 'name' kwarg";
        auto rec = std::make_unique<AllocRecord>(call, var, call->args_[0], name, call->span_);
        ptr_to_alloc[var.get()] = rec.get();
        allocs.push_back(std::move(rec));
      } else if (IsOp(call, "pld.tensor.window") && !call->args_.empty()) {
        // AsVarLike (not As<Var>): the windowed buffer may be an SSA loop carry
        // (IterArg); identity propagation uses the same Var-like distinction.
        auto ptr_arg_var = AsVarLike(call->args_[0]);
        if (ptr_arg_var) {
          auto it = ptr_to_alloc.find(ptr_arg_var.get());
          if (it != ptr_to_alloc.end()) {
            WindowRecord wr{call, var, it->second};
            windows.push_back(wr);
          }
        }
      }
    }
    // Record the AssignStmt def for every Var so ResolveDeviceDescriptor can
    // follow ``for r in pl.range(<var>)`` back to ``<var> = pld.system.world_size()``
    // (CSE / NormalizeStmtStructure hoists such calls out into a temp).
    // AsVarLike (not As<Var>): an SSA loop-carry LHS is an IterArg.
    if (auto var = AsVarLike(op->var_)) {
      var_defs[var.get()] = op->value_;
    }
    IRVisitor::VisitStmt_(op);
  }

  std::vector<std::unique_ptr<AllocRecord>> allocs;
  std::unordered_map<const Var*, AllocRecord*> ptr_to_alloc;
  std::vector<WindowRecord> windows;
  std::unordered_map<const Var*, ExprPtr> var_defs;
};

/// Detects whether a Call resolves to a chip-level Orchestration function (i.e.
/// the host_orch is dispatching down one level). Such calls carry the
/// ``worker=`` attr written by the N3 parser pass.
[[nodiscard]] bool IsAdjacentOrchDispatch(const CallPtr& op,
                                          const std::map<std::string, FunctionPtr>& chip_orchs) {
  if (!op || !op->op_) return false;
  auto gvar = As<GlobalVar>(op->op_);
  if (!gvar) return false;
  return chip_orchs.find(gvar->name_) != chip_orchs.end();
}

/// Resolve SSA aliases once per function. Path compression prevents repeated
/// dispatches sharing long alias chains from rescanning them quadratically.
/// Definitions are immutable during DispatchAnalyzer's traversal; unknown or
/// cyclic chains remain unresolved rather than being treated as full coverage.
[[nodiscard]] ExprPtr UnwrapStopExpr(const ExprPtr& stop,
                                     const std::unordered_map<const Var*, ExprPtr>& var_defs,
                                     std::unordered_map<const Var*, ExprPtr>& cache) {
  ExprPtr cur = stop;
  std::vector<const Var*> path;
  std::unordered_set<const Var*> visited;
  while (auto v = As<Var>(cur)) {
    if (auto hit = cache.find(v.get()); hit != cache.end()) {
      cur = hit->second;
      break;
    }
    if (!visited.insert(v.get()).second) break;
    path.push_back(v.get());
    auto it = var_defs.find(v.get());
    if (it == var_defs.end() || !it->second) break;
    cur = it->second;
  }
  for (const auto* var : path) cache[var] = cur;
  return cur;
}

/// Resolves the device descriptor for a ``worker=`` Expr in the context of a
/// stack of enclosing ForStmt scopes. Throws pypto::ValueError on unsupported
/// forms (the user's parser is meant to restrict ``worker=`` to ConstInt or
/// the induction var of an enclosing pl.range loop).
DeviceDescriptor ResolveDeviceDescriptor(const ExprPtr& device,
                                         const std::unordered_map<const Var*, ForStmtPtr>& active_loops,
                                         const std::unordered_map<const Var*, ExprPtr>& var_defs,
                                         std::unordered_map<const Var*, ExprPtr>& bound_cache,
                                         const Span& dispatch_span, bool host_owner) {
  DeviceDescriptor desc;
  if (auto ci = As<ConstInt>(device)) {
    INTERNAL_CHECK_SPAN(ci->value_ >= 0, dispatch_span)
        << "MaterializeCommDomainScopes: worker= ConstInt must be non-negative, got " << ci->value_;
    desc.intervals.emplace_back(ci->value_, ci->value_);
    return desc;
  }
  if (auto v = As<Var>(device)) {
    auto loop = active_loops.find(v.get());
    if (loop != active_loops.end()) {
      const auto& fs = loop->second;
      ExprPtr stop = UnwrapStopExpr(fs->stop_, var_defs, bound_cache);
      if (auto stop_call = As<Call>(stop)) {
        if (IsOp(stop_call, "pld.system.world_size") || IsOp(stop_call, "pld.system.num_workers")) {
          CHECK_SPAN(host_owner || !IsOp(stop_call, "pld.system.world_size"), dispatch_span)
              << "MaterializeCommDomainScopes: world_size is HOST-only; use num_workers at CLUSTER_0";
          // A dynamic child-count stop covers all children only for [0, N)
          // with unit step. In particular, an unproven start/step cannot be
          // accepted merely because the stop is a topology query.
          auto start = As<ConstInt>(UnwrapStopExpr(fs->start_, var_defs, bound_cache));
          auto step = As<ConstInt>(UnwrapStopExpr(fs->step_, var_defs, bound_cache));
          CHECK_SPAN(start && start->value_ == 0, dispatch_span)
              << "MaterializeCommDomainScopes: dynamic full coverage requires loop start=0";
          CHECK_SPAN(step && step->value_ == 1, dispatch_span)
              << "MaterializeCommDomainScopes: dynamic full coverage requires loop step=1";
          desc.is_all = true;
          return desc;
        }
      }
      if (auto stop_ci = As<ConstInt>(stop)) {
        auto start_ci = As<ConstInt>(UnwrapStopExpr(fs->start_, var_defs, bound_cache));
        INTERNAL_CHECK_SPAN(start_ci, dispatch_span)
            << "MaterializeCommDomainScopes: worker=r loop start must unwrap to ConstInt";
        int64_t start = start_ci->value_;
        auto step_ci = As<ConstInt>(UnwrapStopExpr(fs->step_, var_defs, bound_cache));
        INTERNAL_CHECK_SPAN(step_ci, dispatch_span)
            << "MaterializeCommDomainScopes: worker=r loop step must unwrap to ConstInt";
        int64_t step = step_ci->value_;
        INTERNAL_CHECK_SPAN(step == 1, dispatch_span)
            << "MaterializeCommDomainScopes: worker=r over a non-unit-step loop is not supported (step="
            << step << ")";
        INTERNAL_CHECK_SPAN(start >= 0 && stop_ci->value_ >= start, dispatch_span)
            << "MaterializeCommDomainScopes: worker=r loop range must be [0, N) with N>=0";
        if (start < stop_ci->value_) desc.intervals.emplace_back(start, stop_ci->value_ - 1);
        return desc;
      }
      throw pypto::ValueError(
          "MaterializeCommDomainScopes: worker=r loop bound must be ConstInt, pld.system.world_size() "
          "or pld.system.num_workers()");
    }
    throw pypto::ValueError(
        "MaterializeCommDomainScopes: worker= Var is not the induction variable of any enclosing "
        "pl.range "
        "loop");
  }
  throw pypto::ValueError(
      "MaterializeCommDomainScopes: worker= expression must be ConstInt or the induction var of "
      "pl.range; "
      "got "
      "an unsupported expression at " +
      dispatch_span.to_string());
}

/// Walks a host_orch body, maintaining a stack of enclosing ForStmts, and for
/// every chip_orch dispatch Call records the inferred device descriptor against
/// each view Var passed positionally.
class DispatchAnalyzer : public IRVisitor {
 public:
  DispatchAnalyzer(const WindowIdentityGraph& identities,
                   const std::unordered_map<const WindowBuffer*, AllocRecord*>& local_allocs,
                   const std::map<std::string, FunctionPtr>& chip_orchs,
                   const std::unordered_map<const Var*, ExprPtr>& var_defs, bool host_owner)
      : identities_(identities),
        local_allocs_(local_allocs),
        chip_orchs_(chip_orchs),
        var_defs_(var_defs),
        host_owner_(host_owner) {}

  void VisitStmt_(const ForStmtPtr& op) override {
    // Index active induction variables so deeply nested dispatches do not scan
    // the entire enclosing loop stack at every call. Restore shadowed entries.
    auto previous = active_loops_.find(op->loop_var_.get());
    ForStmtPtr saved = previous == active_loops_.end() ? nullptr : previous->second;
    active_loops_[op->loop_var_.get()] = op;
    ++repeating_scope_depth_;
    IRVisitor::VisitStmt_(op);
    --repeating_scope_depth_;
    if (saved) {
      active_loops_[op->loop_var_.get()] = std::move(saved);
    } else {
      active_loops_.erase(op->loop_var_.get());
    }
  }

  // WhileStmt carries no device-descriptor role (unlike ForStmt, which
  // active_loops_ tracks for ResolveDeviceDescriptor), but it must still count
  // toward repeating_scope_depth_ so the all_to_all_v loop-use guard in
  // AnalyzeCollective rejects a call nested in a while loop too.
  void VisitStmt_(const WhileStmtPtr& op) override {
    ++repeating_scope_depth_;
    IRVisitor::VisitStmt_(op);
    --repeating_scope_depth_;
  }

  void AnalyzeDispatch(const CallPtr& op) {
    if (!IsAdjacentOrchDispatch(op, chip_orchs_)) return;
    ExprPtr device;
    for (const auto& [k, v] : op->attrs_) {
      if (k == kAttrWorker) {
        // attrs["worker"] is stored as ExprPtr by N3 parser.
        if (const auto* p = std::any_cast<ExprPtr>(&v)) device = *p;
        break;
      }
    }
    if (!device) return;
    DeviceDescriptor desc =
        ResolveDeviceDescriptor(device, active_loops_, var_defs_, bound_cache_, op->span_, host_owner_);
    for (const auto& arg : op->args_) {
      if (auto* alloc = LocalAllocation(arg)) alloc->dispatch_coverage.Add(desc);
    }
  }

  [[nodiscard]] AllocRecord* ResolveWindowAlloc(const ExprPtr& expr, const std::string& op_name,
                                                const char* role) {
    auto* alloc = LocalAllocation(expr);
    INTERNAL_CHECK_SPAN(alloc, expr->span_) << "MaterializeCommDomainScopes: " << op_name << " " << role
                                            << " must be produced by pld.tensor.window";
    return alloc;
  }

  // Nullable variant of ResolveWindowAlloc for operands that are window-bound
  // on the builtin rails but may legally stay plain Tensors on the InCore
  // composite rail (send_counts today): returns nullptr instead of failing
  // when the expression is not a pld.tensor.window view.
  [[nodiscard]] AllocRecord* TryResolveWindowAlloc(const ExprPtr& expr) { return LocalAllocation(expr); }

  void AnalyzeCollective(const CallPtr& op) {
    if (!op || !op->op_) return;

    if (IsOp(op, "pld.tensor.allreduce") || IsOp(op, "pld.tensor.broadcast") ||
        IsOp(op, "pld.tensor.reduce_scatter")) {
      const auto& op_name = op->op_->name_;
      INTERNAL_CHECK_SPAN(op->args_.size() == 2, op->span_)
          << "MaterializeCommDomainScopes: " << op_name << " expects exactly two args";
      collective_consumers.push_back({ResolveWindowAlloc(op->args_[0], op_name, "data/target"),
                                      ResolveWindowAlloc(op->args_[1], op_name, "signal"), op->span_});
      return;
    }

    if (IsOp(op, "pld.tensor.barrier")) {
      INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.barrier expects exactly one arg";
      collective_consumers.push_back(
          {nullptr, ResolveWindowAlloc(op->args_[0], "pld.tensor.barrier", "signal"), op->span_});
      return;
    }

    if (IsOp(op, "pld.tensor.allgather")) {
      // 3-arg push-based form: pld.tensor.allgather(input, target, signal).
      // Track target + signal for device-coverage inheritance; input coverage
      // comes from the caller's own dispatch site.
      INTERNAL_CHECK_SPAN(op->args_.size() == 3, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.allgather expects exactly 3 args";
      collective_consumers.push_back({ResolveWindowAlloc(op->args_[1], "pld.tensor.allgather", "target"),
                                      ResolveWindowAlloc(op->args_[2], "pld.tensor.allgather", "signal"),
                                      op->span_});
      return;
    }

    if (IsOp(op, "pld.tensor.all_to_all")) {
      // 3-arg push-based form: pld.tensor.all_to_all(input, target, signal)
      // args[0] = input  (Tensor or DistributedTensor — on HOST path this
      //                    must be a DISTINCT window from args[1], never the
      //                    same one; on InCore it is a plain Tensor.)
      // args[1] = target (DistributedTensor, window-bound)
      // args[2] = signal (DistributedTensor, window-bound)
      INTERNAL_CHECK_SPAN(op->args_.size() == 3, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.all_to_all expects exactly 3 args";
      auto* data_alloc = ResolveWindowAlloc(op->args_[1], "pld.tensor.all_to_all", "target");
      auto* signal_alloc = ResolveWindowAlloc(op->args_[2], "pld.tensor.all_to_all", "signal");
      // Device-coverage inheritance from data to signal is handled generically
      // in the Phase 3 loop below (every collective consumer with a data_alloc
      // triggers the merge automatically).  No manual merge needed here.
      collective_consumers.push_back({data_alloc, signal_alloc, op->span_});
      return;
    }

    if (IsOp(op, "pld.tensor.all_to_all_v")) {
      // 6-arg push-based form:
      // pld.tensor.all_to_all_v(input, target, signal, send_counts, recv_counts, core_num)
      // args[0] = input        (window-bound on HOST path; distinct from target)
      // args[1] = target       (DistributedTensor, window-bound, window-as-result)
      // args[2] = signal       (DistributedTensor, window-bound, barrier)
      // args[3] = send_counts  (window-bound on the builtin rails; peers READ
      //                          it remotely via the counts pull, so — like
      //                          `signal` and `recv_counts` — it inherits
      //                          target's device coverage; conditional,
      //                          because the InCore composite rail may pass a
      //                          plain Tensor, written locally only)
      // args[4] = recv_counts  (DistributedTensor, window-bound; published
      //                          cross-rank via notify, so needs target's
      //                          device coverage exactly like `signal` does)
      // args[5] = core_num     (Scalar[INDEX], not window-bound — irrelevant
      //                          to this pass's comm-domain scoping)
      INTERNAL_CHECK_SPAN(op->args_.size() == 6, op->span_)
          << "MaterializeCommDomainScopes: pld.tensor.all_to_all_v expects exactly 6 args";
      // HOST-orch calls still need loop-carried comm-domain window lifetime
      // management, even though the credit barrier supports rank-uniform
      // InCore loops. Track HOST repetition across both ForStmt and WhileStmt
      // with repeating_scope_depth_ (see the VisitStmt_ overrides above).
      CHECK_SPAN(repeating_scope_depth_ == 0, op->span_)
          << "pld.tensor.all_to_all_v is not supported inside a for/while loop in a HOST "
             "orchestrator yet: dynamic re-invocation needs loop-carried comm-domain window "
             "lifetime management this compiler does not model (the credit-based signal "
             "itself is reusable across sequential calls and rank-uniform InCore loops).";
      auto* data_alloc = ResolveWindowAlloc(op->args_[1], "pld.tensor.all_to_all_v", "target");
      auto* signal_alloc = ResolveWindowAlloc(op->args_[2], "pld.tensor.all_to_all_v", "signal");
      auto* recv_counts_alloc = ResolveWindowAlloc(op->args_[4], "pld.tensor.all_to_all_v", "recv_counts");
      // Consumer entries sharing the same data_alloc: signal, recv_counts and
      // (when window-bound) send_counts inherit target's device coverage
      // (Phase 3 loop below merges generically per entry).
      collective_consumers.push_back({data_alloc, signal_alloc, op->span_});
      collective_consumers.push_back({data_alloc, recv_counts_alloc, op->span_});
      if (auto* send_counts_alloc = TryResolveWindowAlloc(op->args_[3])) {
        collective_consumers.push_back({data_alloc, send_counts_alloc, op->span_});
      }
      return;
    }
  }

  void VisitExpr_(const CallPtr& op) override {
    AnalyzeDispatch(op);
    if (host_owner_) AnalyzeCollective(op);
    IRVisitor::VisitExpr_(op);
  }

  // A captured (`as tid`) chip-orch dispatch is a Submit; the ``worker=`` attr
  // it carries is analysed identically through the Call-shaped view.
  void VisitExpr_(const SubmitPtr& op) override {
    AnalyzeDispatch(SubmitToCallView(op));
    IRVisitor::VisitExpr_(op);
  }

  std::vector<CollectiveConsumer> collective_consumers;

 private:
  AllocRecord* LocalAllocation(const ExprPtr& expr) const {
    auto identity = identities_.Identity(expr);
    auto found = local_allocs_.find(identity.get());
    return found == local_allocs_.end() ? nullptr : found->second;
  }

  const WindowIdentityGraph& identities_;
  const std::unordered_map<const WindowBuffer*, AllocRecord*>& local_allocs_;
  const std::map<std::string, FunctionPtr>& chip_orchs_;
  const std::unordered_map<const Var*, ExprPtr>& var_defs_;
  std::unordered_map<const Var*, ForStmtPtr> active_loops_;
  std::unordered_map<const Var*, ExprPtr> bound_cache_;
  int repeating_scope_depth_ = 0;
  bool host_owner_;
};

/// A host-orchestration function in PyPTO is declared as either
/// ``@pl.function(type=FunctionType.Orchestration, level=Level.HOST)`` or
/// (more common in distributed programs) ``@pl.function(level=Level.HOST,
/// role=Role.Orchestrator)`` where ``func_type_`` may stay ``Opaque``. Accept
/// either form so the pass works with the conventional host_orch declaration
/// idiom used in distributed tests.
[[nodiscard]] bool IsHostOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::HOST) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

[[nodiscard]] bool IsChipOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::CHIP) return false;
  // A Graph function carries {CHIP, Orchestrator} exactly like a chip entry, so
  // the role disjunct below matches it even though its func_type_ does not. It
  // is a task launched *by* a chip entry, never a host dispatch target, so
  // exclude it explicitly — narrowing the func_type_ term alone would not help.
  if (func->func_type_ == FunctionType::Graph) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

[[nodiscard]] bool IsLocalDomainOwner(const FunctionPtr& func) {
  if (!func || !func->level_.has_value()) return false;
  if (*func->level_ != Level::HOST && *func->level_ != Level::CLUSTER_0) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

struct PendingDomain {
  DeviceDescriptor desc;
  std::vector<WindowBufferPtr> slots;
  std::set<std::string> names;
  Span span;
};

struct LocalWindowPlan {
  AllocAndWindowCollector collector;
  std::vector<PendingDomain> domains;
};

// Local allocation collection and scope planning are separate from rewriting.
// Graph propagation can therefore see all owners before any child signature is
// rebuilt; lexical Program order cannot select a first caller or clone a scope.
void PrepareLocalWindows(const FunctionPtr& func, LocalWindowPlan& plan) {
  plan.collector.VisitStmt(func->body_);
  std::unordered_set<const AllocRecord*> viewed;
  for (const auto& window : plan.collector.windows) viewed.insert(window.alloc);
  for (const auto& rec : plan.collector.allocs) {
    INTERNAL_CHECK_SPAN(viewed.count(rec.get()), rec->span)
        << "MaterializeCommDomainScopes: pld.tensor.alloc_window_buffer '" << rec->name
        << "' has no pld.tensor.window materialisation (dead allocation)";
    rec->wb = std::make_shared<const WindowBuffer>(rec->ptr_var, rec->size_expr, false, false, rec->span);
  }
}

void InferLocalDomains(const FunctionPtr& func, LocalWindowPlan& plan,
                       const std::map<std::string, FunctionPtr>& children,
                       const WindowIdentityGraph& identities) {
  auto& collector = plan.collector;
  if (collector.allocs.empty()) return;
  std::unordered_map<const WindowBuffer*, AllocRecord*> local_allocs;
  for (const auto& rec : collector.allocs) local_allocs.emplace(rec->wb.get(), rec.get());
  CoverageSnapshots coverage;
  DispatchAnalyzer analyzer(identities, local_allocs, children, collector.var_defs,
                            *func->level_ == Level::HOST);
  analyzer.VisitStmt(func->body_);
  for (const auto& rec : collector.allocs) {
    if (rec->dispatch_coverage.observed) rec->coverage = coverage.Intern(rec->dispatch_coverage.Freeze());
  }

  // Signal synthesis and collective-domain alignment remain HOST-only. Each
  // entry is a real local slot; inherited formals never enter this index.
  for (const auto& consumer : analyzer.collective_consumers) {
    INTERNAL_CHECK_SPAN(consumer.signal_alloc, consumer.span)
        << "MaterializeCommDomainScopes: invalid collective consumer bookkeeping";
    if (consumer.data_alloc) {
      INTERNAL_CHECK_SPAN(consumer.data_alloc->coverage.has_value(), consumer.span)
          << "MaterializeCommDomainScopes: collective data/target buffer has no inferred comm-domain "
             "coverage to share with its signal buffer";
      consumer.signal_alloc->coverage =
          coverage.Merge(consumer.signal_alloc->coverage.value_or(0), *consumer.data_alloc->coverage);
    } else {
      CHECK_SPAN(consumer.signal_alloc->coverage.has_value(), consumer.span)
          << "MaterializeCommDomainScopes: pld.tensor.barrier signal buffer has no inferred "
             "comm-domain coverage — add a worker= annotation or ensure the signal window "
             "is consumed by a device-tagged chip dispatch";
    }
  }

  // Canonical IDs group identical descriptors without comparing large sets for
  // every allocation. Declaration order within each group remains unchanged.
  std::unordered_map<size_t, size_t> desc_to_index;
  for (const auto& rec : collector.allocs) {
    INTERNAL_CHECK_SPAN(rec->coverage.has_value(), rec->span)
        << "MaterializeCommDomainScopes: pld.tensor.alloc_window_buffer '" << rec->name
        << "' is not consumed by any chip_orch dispatch or adjacent HOST dispatch";
    auto [found, inserted] = desc_to_index.emplace(*rec->coverage, plan.domains.size());
    if (inserted) plan.domains.push_back({coverage.Get(*rec->coverage), {}, {}, rec->span});
    auto& domain = plan.domains[found->second];
    INTERNAL_CHECK_SPAN(domain.names.insert(rec->name).second, rec->span)
        << "MaterializeCommDomainScopes: duplicate allocation name '" << rec->name
        << "' within the same comm domain";
    domain.slots.push_back(rec->wb);
  }
}

/// One final mutation uses the shared IRMutator's substitution/type plumbing.
/// Call/Submit attrs, deps, launch operands and predicates are rewritten by the
/// base mutator. Only the result type is changed here; no Submit becomes a Call.
/// As in the previous DefiningCallRetyper, result shape/dtype/view come from the
/// defining expression itself, so identity propagation cannot hide a mismatch.
class WindowIdentityRewriter : public IRMutator {
 public:
  WindowIdentityRewriter(const WindowIdentityGraph& identities,
                         const std::unordered_map<const Function*, std::unique_ptr<LocalWindowPlan>>& plans)
      : identities_(identities), plans_(plans) {}

  FunctionPtr VisitFunction(const FunctionPtr& func) override {
    auto plan = plans_.find(func.get());
    if (!identities_.FunctionNeedsRewrite(func) && (plan == plans_.end() || plan->second->domains.empty()))
      return func;
    auto params = func->params_;
    bool changed = false;
    for (auto& param : params) {
      auto fresh = AsVarLike(VisitExpr(param));
      changed |= fresh.get() != param.get();
      param = std::move(fresh);
    }
    auto body = VisitStmt(func->body_);
    auto types = identities_.RewriteReturns(func);
    for (size_t i = 0; i < types.size(); ++i) {
      types[i] = RemapTypeViaVisitor(types[i]);
      changed |= types[i].get() != func->return_types_[i].get();
    }
    auto found = plans_.find(func.get());
    if (found != plans_.end()) {
      const auto& domains = found->second->domains;
      for (size_t i = domains.size(); i-- > 0;) {
        const auto& domain = domains[i];
        body = std::make_shared<const CommDomainScopeStmt>(domain.desc.ToDevices(), domain.slots,
                                                           "comm_d" + std::to_string(i), body, domain.span);
      }
    }
    changed |= body.get() != func->body_.get();
    // Match the base mutator's type-based attribute handling after parameter
    // substitutions are known. Preserve every ordinary attribute unchanged.
    auto attrs = func->attrs_;
    for (auto& [key, value] : attrs) {
      auto remapped = MapAttrExprs(value, [this](const ExprPtr& expr) { return VisitExpr(expr); });
      if (remapped.has_value()) {
        value = std::move(*remapped);
        changed = true;
      }
    }
    if (!changed) return func;
    auto result = MutableCopy(func);
    result->params_ = std::move(params);
    result->return_types_ = std::move(types);
    result->body_ = std::move(body);
    result->attrs_ = std::move(attrs);
    return result;
  }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto found = var_remap_.find(op.get());
    if (found != var_remap_.end()) return found->second;
    auto type = identities_.RewriteType(op, RemapTypeViaVisitor(op->GetType()));
    if (type.get() == op->GetType().get()) return op;
    auto fresh = std::make_shared<const Var>(op->name_hint_, type, op->span_);
    var_remap_[op.get()] = fresh;
    return fresh;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto found = var_remap_.find(op.get());
    if (found != var_remap_.end()) return found->second;
    auto init = VisitExpr(op->initValue_);
    auto type = identities_.RewriteType(op, RemapTypeViaVisitor(op->GetType()));
    if (init.get() == op->initValue_.get() && type.get() == op->GetType().get()) return op;
    auto fresh = std::make_shared<const IterArg>(op->name_hint_, type, init, op->span_);
    var_remap_[op.get()] = fresh;
    return fresh;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto result = As<Call>(IRMutator::VisitExpr_(op));
    auto type = identities_.RewriteType(op, result->GetType());
    if (type.get() == result->GetType().get()) return result;
    return std::make_shared<const Call>(result->op_, result->args_, result->kwargs_, result->attrs_,
                                        std::move(type), result->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto result = As<Submit>(IRMutator::VisitExpr_(op));
    auto type = identities_.RewriteType(op, result->GetType());
    if (type.get() == result->GetType().get()) return result;
    return std::make_shared<const Submit>(result->op_, result->args_, result->deps_, result->kwargs_,
                                          result->attrs_, std::move(type), result->span_, result->core_num_,
                                          result->sync_start_, result->allow_early_resolve_,
                                          result->predicate_);
  }

 private:
  const WindowIdentityGraph& identities_;
  const std::unordered_map<const Function*, std::unique_ptr<LocalWindowPlan>>& plans_;
};
}  // namespace

namespace pass {

Pass MaterializeCommDomainScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Index adjacent child dispatches once. A CHIP Graph remains a task of a
    // CHIP entry, not a HOST dispatch target.
    std::map<std::string, FunctionPtr> chip_orchs;
    std::map<std::string, FunctionPtr> host_orchs;
    std::unordered_map<const Function*, std::unique_ptr<LocalWindowPlan>> plans;
    for (const auto& [gv, func] : program->functions_) {
      if (IsChipOrch(func)) chip_orchs[func->name_] = func;
      if (IsHostOrch(func)) host_orchs[func->name_] = func;
      if (IsLocalDomainOwner(func)) {
        auto plan = std::make_unique<LocalWindowPlan>();
        PrepareLocalWindows(func, *plan);
        plans.emplace(func.get(), std::move(plan));
      }
    }
    WindowIdentityGraph identities(program);
    for (const auto& [gvar, func] : program->functions_) {
      auto found = plans.find(func.get());
      if (found == plans.end()) continue;
      for (const auto& window : found->second->collector.windows) {
        identities.SeedLocal(window.old_view_var, window.alloc->wb);
        identities.SeedLocal(window.window_call, window.alloc->wb);
      }
    }
    identities.Propagate();
    for (const auto& [gvar, func] : program->functions_) {
      auto found = plans.find(func.get());
      if (found == plans.end()) continue;
      InferLocalDomains(func, *found->second, *func->level_ == Level::HOST ? chip_orchs : host_orchs,
                        identities);
    }
    WindowIdentityRewriter rewriter(identities, plans);
    return rewriter.VisitProgram(program);
  };

  return CreateProgramPass(pass_func, "MaterializeCommDomainScopes", kMaterializeCommDomainScopesProperties);
}

}  // namespace pass

// ============================================================================
// CommDomainScopesMaterialized property verifier
//
// Walks every function signature/body looking for window uses and owner scopes;
// asserts: (a) ``slots_`` is non-empty (an empty domain would emit
// ``window_size=0`` and the runtime would reject it); (b) every slot is
// non-null; (c) each ``WindowBuffer`` appears as a slot in at most one
// scope program-wide (cross-scope identity uniqueness); (d) every non-null
// DistributedTensorType window reference has a scope owner, including formals.
// ============================================================================

class CommDomainScopeCollector : public IRVisitor {
 public:
  std::vector<CommDomainScopeStmtPtr> scopes;
  std::vector<std::pair<WindowBufferPtr, Span>> referenced_windows;

  void VisitFunction(const FunctionPtr& func) override {
    IRVisitor::VisitFunction(func);
    for (const auto& type : func->return_types_) CollectType(type, func->span_);
  }

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr || !visited_exprs_.insert(expr.get()).second) return;
    CollectType(expr->GetType(), expr->span_);
    IRVisitor::VisitExpr(expr);
  }

  void VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    scopes.push_back(op);
    VisitScopeAttrs(op);
    if (op->body_) VisitStmt(op->body_);
  }

 private:
  void CollectType(const TypePtr& type, const Span& span) {
    if (!type || !visited_types_.insert(type.get()).second) return;
    if (auto dt = As<DistributedTensorType>(type)) {
      if (dt->window_buffer_.has_value() && *dt->window_buffer_) {
        referenced_windows.emplace_back(*dt->window_buffer_, span);
      }
    } else if (auto tuple = As<TupleType>(type)) {
      for (const auto& field : tuple->types_) CollectType(field, span);
    }
  }

  // Types and SSA values are shared DAG nodes. Memoize both so signatures,
  // tuple projections and repeated uses do not repeatedly traverse metadata.
  std::unordered_set<const Expr*> visited_exprs_;
  std::unordered_set<const Type*> visited_types_;
};

class CommDomainScopesMaterializedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "CommDomainScopesMaterialized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    CommDomainScopeCollector collector;
    for (const auto& [gvar, func] : program->functions_) {
      if (func) collector.VisitFunction(func);
    }
    std::unordered_map<const WindowBuffer*, const CommDomainScopeStmt*> first_seen;
    for (const auto& scope : collector.scopes) {
      if (!scope) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                                 "null CommDomainScopeStmt in IR", Span::unknown());
        continue;
      }
      if (scope->slots_.empty()) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                                 "CommDomainScopeStmt '" + scope->name_hint_ +
                                     "' has no slots — every comm-domain scope must carry at least one "
                                     "WindowBuffer",
                                 scope->span_);
        continue;
      }
      for (const auto& slot : scope->slots_) {
        if (!slot) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                                   "CommDomainScopeStmt '" + scope->name_hint_ + "' has a null slot",
                                   scope->span_);
          continue;
        }
        auto [it, inserted] = first_seen.emplace(slot.get(), scope.get());
        if (!inserted) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                                   "WindowBuffer '" + slot->name_hint_ +
                                       "' appears in multiple CommDomainScopeStmts ('" +
                                       it->second->name_hint_ + "' and '" + scope->name_hint_ + "')",
                                   slot->span_);
        }
      }
    }
    // Resolve references only after indexing every scope in the program: a
    // child's typed formal may precede its owner's function in lexical order.
    // Unbound nullable formals deliberately contribute no reference here.
    for (const auto& [window, span] : collector.referenced_windows) {
      if (first_seen.count(window.get())) continue;
      diagnostics.emplace_back(DiagnosticSeverity::Error, "CommDomainScopesMaterialized", 0,
                               "DistributedTensorType references orphan WindowBuffer '" + window->name_hint_ +
                                   "' with no owning CommDomainScopeStmt",
                               span);
    }
  }
};

PropertyVerifierPtr CreateCommDomainScopesMaterializedPropertyVerifier() {
  return std::make_shared<CommDomainScopesMaterializedPropertyVerifierImpl>();
}
}  // namespace ir
}  // namespace pypto
