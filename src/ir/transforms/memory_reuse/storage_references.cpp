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
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/storage_access_index.h"
#include "pypto/ir/transforms/utils/storage_reference_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/memory_reuse/internal.h"
#include "src/ir/transforms/memory_reuse/storage.h"

namespace pypto::ir {
namespace {

using memory_reuse::SamePhysicalWindow;
using memory_reuse::SameStorageDescriptor;

bool SameReferenceDescriptor(const TileTypePtr& source, const TileTypePtr& target) {
  // A selector may rebind validity, while padding remains part of its typed
  // native handle. Storage reuse alone has a weaker padding contract.
  return source && target &&
         tile_view_semantics::GetEffectiveTileView(*source).pad ==
             tile_view_semantics::GetEffectiveTileView(*target).pad &&
         SameStorageDescriptor(source, target);
}

struct ReferenceNode {
  VarPtr value;
  std::vector<const Var*> sources;
  std::vector<const Var*> users;
  size_t definition = 0;
  bool defined = false;
  bool region = false;
  bool alias = false;
  bool endpoint = false;
  bool writer = false;       // A fresh payload version, despite aliasing storage.
  bool loop_result = false;  // Final iteration's logical value (or entry value).
  bool branch_result = false;
};

struct ReferenceIf {
  IfStmtPtr statement;
  std::array<std::pair<size_t, size_t>, 2> ranges;
  std::array<YieldStmtPtr, 2> yields;
};

struct ReferenceLoop {
  StmtPtr statement;
  size_t begin = 0;
  size_t end = 0;
  std::vector<IterArgPtr> arguments;
  std::vector<VarPtr> results;
  YieldStmtPtr yield;
};

const OpRegistryEntry* ReferenceOpEntry(const CallPtr& call) {
  const auto& registry = OpRegistry::GetInstance();
  if (!call || !call->op_ || !registry.IsRegistered(call->op_->name_)) return nullptr;
  return &registry.GetEntry(call->op_->name_);
}

bool ReferenceCallHasPayload(const CallPtr& call) {
  const auto* entry = ReferenceOpEntry(call);
  return call &&
         (!entry || entry->GetExecutionMemoryAccessEvidence() != ExecutionMemoryAccessEvidence::NoAccess);
}

// Every reference edge is stored once. Forward propagation handles nested
// selections and loop backedges with a bounded worklist, rather than expanding
// a root set at each phi (which is quadratic for a chain of selections).
class ReferenceGraph : public IRVisitor {
 public:
  std::map<const Var*, ReferenceNode> nodes;
  std::vector<ReferenceIf> branches;
  std::vector<ReferenceLoop> loops;
  std::map<const Stmt*, size_t> statement_order;
  std::set<const Var*> pinned;
  std::set<const Var*> declared_bases;
  std::set<const Var*> owned_bases;

  ReferenceGraph(const StmtPtr& body, const std::vector<VarPtr>& parameters) {
    for (const auto& parameter : parameters) {
      if (As<TileType>(parameter->GetType())) {
        auto& node = Node(parameter);
        node.defined = true;
        node.endpoint = true;
        if (auto tile = GetTileTypeWithMemRef(parameter->GetType())) {
          pinned.insert(GetDefinedMemRef(tile)->base_.get());
        }
      }
      pinned.insert(parameter.get());
    }
    VisitStmt(body);
  }

  std::set<const Var*> Propagate(std::set<const Var*> marked) const {
    std::vector<const Var*> pending(marked.begin(), marked.end());
    for (size_t i = 0; i < pending.size(); ++i) {
      for (const auto* user : nodes.at(pending[i]).users) {
        if (marked.insert(user).second) pending.push_back(user);
      }
    }
    return marked;
  }

  const Var* AliasOrigin(const Var* value) const {
    std::vector<const Var*> path;
    while (true) {
      if (auto found = alias_origins_.find(value); found != alias_origins_.end()) {
        value = found->second;
        break;
      }
      const auto& node = nodes.at(value);
      if (!node.alias || node.sources.size() != 1) break;
      path.push_back(value);
      value = node.sources.front();
    }
    for (const auto* member : path) alias_origins_[member] = value;
    return value;
  }

  bool FixedResult(const VarPtr& value) const {
    auto tile = GetTileTypeWithMemRef(value->GetType());
    auto memory = tile ? GetDefinedMemRef(tile) : nullptr;
    return memory && (memory->is_pinned_ || declared_bases.count(memory->base_.get()) ||
                      owned_bases.count(memory->base_.get()));
  }

 protected:
  void VisitExpr_(const IterArgPtr&) override {}

  void VisitStmt_(const AssignStmtPtr& statement) override {
    const size_t order = ++clock_;
    statement_order[statement.get()] = order;
    auto call = As<Call>(statement->value_);
    if (const auto base = GetPinnedAllocBase(statement)) {
      pinned.insert(base.get());
      declared_bases.insert(base.get());
    }
    if (!As<TileType>(statement->var_->GetType())) return;
    auto& node = Node(statement->var_);
    node.definition = order;
    node.defined = true;
    auto source = AsVarLike(statement->value_);
    if (!source && call) {
      auto reuse = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
      if (reuse) source = AsVarLike(call->args_[*reuse]);
      if (!source && !call->args_.empty() && op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
        source = AsVarLike(call->args_[0]);
      }
    }
    if (source) {
      node.alias = true;
      if (ReferenceCallHasPayload(call)) {
        const auto* entry = ReferenceOpEntry(call);
        node.writer = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size()).has_value() ||
                      (entry && entry->GetIRStage() == OpIRStage::Functional && entry->MayWriteArg(0));
      }
      Edge(statement->var_, source);
    } else {
      // A normal native producer needs its owned allocation. Only registered
      // cross-core pops supply a bufferless native handle by their own contract.
      node.endpoint =
          call && (GetTileTypeWithMemRef(statement->var_->GetType()) || op_predicates::IsTPop(call));
      if (auto tile = GetTileTypeWithMemRef(statement->var_->GetType())) {
        owned_bases.insert(GetDefinedMemRef(tile)->base_.get());
      }
    }
  }

  void VisitStmt_(const IfStmtPtr& statement) override {
    statement_order[statement.get()] = ++clock_;
    ReferenceIf info;
    info.statement = statement;
    const std::array<StmtPtr, 2> bodies{statement->then_body_, statement->else_body_.value_or(nullptr)};
    for (size_t arm = 0; arm < bodies.size(); ++arm) {
      info.ranges[arm].first = clock_ + 1;
      if (bodies[arm]) VisitStmt(bodies[arm]);
      info.ranges[arm].second = clock_;
      info.yields[arm] = transform_utils::GetLastYieldStmt(bodies[arm]);
    }
    const size_t order = ++clock_;
    for (size_t slot = 0; slot < statement->return_vars_.size(); ++slot) {
      const auto& result = statement->return_vars_[slot];
      if (!As<TileType>(result->GetType())) continue;
      auto& node = Node(result);
      node.definition = order;
      node.defined = true;
      node.region = true;
      node.branch_result = true;
      for (const auto& yield : info.yields) {
        if (yield && slot < yield->value_.size()) Edge(result, AsVarLike(yield->value_[slot]));
      }
    }
    branches.push_back(std::move(info));
  }

  template <typename LoopPtr>
  void Loop(const LoopPtr& loop) {
    const size_t order = ++clock_;
    statement_order[loop.get()] = order;
    const auto yield = transform_utils::GetLastYieldStmt(loop->body_);
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      if (!As<TileType>(argument->GetType())) continue;
      auto& node = Node(argument);
      node.definition = order;
      node.defined = true;
      node.region = true;
      Edge(argument, AsVarLike(argument->initValue_));
      if (yield && i < yield->value_.size()) Edge(argument, AsVarLike(yield->value_[i]));
      if (i < loop->return_vars_.size()) {
        auto& result = Node(loop->return_vars_[i]);
        result.defined = true;
        result.region = true;
        result.loop_result = true;
        result.definition = order;
        Edge(loop->return_vars_[i], argument);
      }
    }
    VisitStmt(loop->body_);
    const size_t end = ++clock_;
    for (const auto& result : loop->return_vars_) {
      if (As<TileType>(result->GetType())) Node(result).definition = end;
    }
    loops.push_back({loop, order, end, loop->iter_args_, loop->return_vars_, yield});
  }

  void VisitStmt_(const ForStmtPtr& loop) override { Loop(loop); }
  void VisitStmt_(const WhileStmtPtr& loop) override { Loop(loop); }

  void VisitStmt_(const EvalStmtPtr& statement) override { statement_order[statement.get()] = ++clock_; }
  void VisitStmt_(const YieldStmtPtr& statement) override { statement_order[statement.get()] = ++clock_; }
  void VisitStmt_(const ReturnStmtPtr& statement) override { statement_order[statement.get()] = ++clock_; }

 private:
  size_t clock_ = 0;
  mutable std::map<const Var*, const Var*> alias_origins_;

  ReferenceNode& Node(const VarPtr& value) {
    auto& node = nodes[value.get()];
    node.value = value;
    return node;
  }

  void Edge(const VarPtr& destination, const VarPtr& source) {
    if (!source || !As<TileType>(source->GetType())) return;
    Node(destination).sources.push_back(source.get());
    Node(source).users.push_back(destination.get());
  }
};

// Replace each reference variable exactly once, including IterArgs. The SSA
// definition and every occurrence must share the same immutable Var object.
class ReferenceRetyping : public IRMutator {
 public:
  ReferenceRetyping(const std::set<const Var*>& references, const std::vector<VarPtr>& parameters)
      : references_(references) {
    // Caller-provided intermediate Tile values retain their declared types.
    // Rewriting a body must keep the function's exact parameter objects.
    for (const auto& parameter : parameters) replacements_.emplace(parameter.get(), parameter);
  }

 protected:
  ExprPtr VisitExpr_(const VarPtr& value) override {
    if (!references_.count(value.get())) return IRMutator::VisitExpr_(value);
    auto& replacement = replacements_[value.get()];
    if (!replacement) {
      auto type = RemapTypeViaVisitor(CloneTypeWithoutMemRef(value->GetType()));
      replacement = std::make_shared<Var>(value->name_hint_, std::move(type), value->span_);
    }
    return replacement;
  }

  ExprPtr VisitExpr_(const IterArgPtr& value) override {
    if (auto found = var_remap_.find(value.get()); found != var_remap_.end()) {
      return found->second;
    }
    if (auto found = replacements_.find(value.get()); found != replacements_.end()) {
      return found->second;
    }
    auto initial = VisitExpr(value->initValue_);
    auto type = RemapTypeViaVisitor(value->GetType());
    if (references_.count(value.get())) type = CloneTypeWithoutMemRef(type);
    if (initial == value->initValue_ && type == value->GetType()) {
      replacements_[value.get()] = value;
      return value;
    }
    auto replacement = std::make_shared<IterArg>(value->name_hint_, type, initial, value->span_);
    replacements_[value.get()] = replacement;
    return replacement;
  }

 private:
  const std::set<const Var*>& references_;
  std::map<const Var*, VarPtr> replacements_;
};

// This union-find is a protection overapproximation, not an allocation plan.
// Region alternatives and all windows of a concrete base may alias here, but
// their allocations retain independent identities and independent addresses.
class ReferenceComponents {
 public:
  explicit ReferenceComponents(const ReferenceGraph& graph) {
    for (const auto& [value, node] : graph.nodes) {
      (void)node;
      indices_.emplace(value, parents_.size());
      parents_.push_back(parents_.size());
    }
    ranks_.resize(parents_.size());
    std::map<const Var*, const Var*> bases;
    for (const auto& [value, node] : graph.nodes) {
      for (const auto* source : node.sources) Join(value, source);
      if (auto type = GetTileTypeWithMemRef(node.value->GetType())) {
        const auto* base = GetDefinedMemRef(type)->base_.get();
        auto [prior, inserted] = bases.emplace(base, value);
        if (!inserted) Join(value, prior->second);
      }
    }
  }

  size_t Find(const Var* value) {
    auto found = indices_.find(value);
    if (found == indices_.end()) return std::numeric_limits<size_t>::max();
    size_t index = found->second;
    while (parents_[index] != index) {
      parents_[index] = parents_[parents_[index]];
      index = parents_[index];
    }
    return index;
  }

 private:
  std::map<const Var*, size_t> indices_;
  std::vector<size_t> parents_;
  std::vector<size_t> ranks_;

  void Join(const Var* lhs, const Var* rhs) {
    size_t left = Find(lhs);
    size_t right = Find(rhs);
    if (left == right) return;
    if (ranks_[left] < ranks_[right]) std::swap(left, right);
    parents_[right] = left;
    if (ranks_[left] == ranks_[right]) ++ranks_[left];
  }
};

// A rename/view keeps the old logical value even when its SSA definition is
// later. A native writer creates a new version on the same storage. Selecting
// a value creates a new joined version. Its demanded incoming values are
// observed at each reachable yield below, preserving branch correlation.
// Propagation settles every node once: O((N+E) log N), including loops.
std::map<const Var*, size_t> ReferenceVersionAges(const ReferenceGraph& graph) {
  using Version = std::pair<size_t, const Var*>;
  std::priority_queue<Version, std::vector<Version>, std::greater<>> pending;
  auto fresh = [](const ReferenceNode& node) { return node.endpoint || node.writer || node.region; };
  for (const auto& [value, node] : graph.nodes) {
    if (fresh(node)) pending.emplace(node.definition, value);
  }
  std::map<const Var*, size_t> ages;
  while (!pending.empty()) {
    const auto [age, value] = pending.top();
    pending.pop();
    if (!ages.emplace(value, age).second) continue;
    for (const auto* user : graph.nodes.at(value).users) {
      if (!fresh(graph.nodes.at(user))) pending.emplace(age, user);
    }
  }
  return ages;
}

std::map<const Var*, size_t> AccContinuationOrders(const ReferenceGraph& graph) {
  using Definition = std::pair<size_t, const Var*>;
  std::priority_queue<Definition> pending;
  for (const auto& [value, node] : graph.nodes) {
    const auto tile = GetTileTypeWithMemRef(node.value->GetType());
    if (node.writer && tile && tile->GetMemorySpace() == MemorySpace::Acc) {
      pending.emplace(node.definition, value);
    }
  }
  std::map<const Var*, size_t> orders;
  while (!pending.empty()) {
    const auto [order, value] = pending.top();
    pending.pop();
    if (!orders.emplace(value, order).second) continue;
    for (const auto* user : graph.nodes.at(value).users) {
      // A native writer begins its own continuation; aliases and region
      // forwarding preserve it. Settle each node once in descending order.
      if (!graph.nodes.at(user).writer) pending.emplace(order, user);
    }
  }
  return orders;
}

struct ReferenceRead {
  VarPtr value;
  size_t age = 0;
  size_t order = 0;
  size_t reverse = 0;
};

struct ReferenceWrite {
  VarPtr value;
  size_t order = 0;
  size_t reverse = 0;
  const Span* span = nullptr;  // Owned by the frozen statement snapshot.
  size_t age_boundary = 0;
};

struct ReferenceNoAlias {
  VarPtr input;
  VarPtr output;
  const Span* span = nullptr;
};

// During the age sweep each observation is inserted once. Compress primary
// coordinates in descending order and retain a maximum reverse coordinate in
// a Fenwick tree. A suffix query then tests one real observation in both branch
// orders, without mixing independent maxima from mutually exclusive arms.
class ReferenceObservationIndex {
 public:
  void Reserve(size_t order) { coordinates_.push_back(order); }

  void Build() {
    std::sort(coordinates_.begin(), coordinates_.end());
    coordinates_.erase(std::unique(coordinates_.begin(), coordinates_.end()), coordinates_.end());
    observations_.resize(coordinates_.size() + 1);
  }

  void Insert(const ReferenceRead& read) {
    size_t position =
        coordinates_.end() - std::lower_bound(coordinates_.begin(), coordinates_.end(), read.order);
    for (; position < observations_.size(); position += position & -position) {
      const auto*& observation = observations_[position];
      if (!observation || observation->reverse < read.reverse) observation = &read;
    }
  }

  [[nodiscard]] const ReferenceRead* After(const ReferenceWrite& write) const {
    size_t position =
        coordinates_.end() - std::upper_bound(coordinates_.begin(), coordinates_.end(), write.order);
    const ReferenceRead* result = nullptr;
    for (; position; position -= position & -position) {
      const auto* observation = observations_[position];
      if (observation && (!result || result->reverse < observation->reverse)) result = observation;
    }
    return result && result->reverse > write.reverse ? result : nullptr;
  }

 private:
  std::vector<size_t> coordinates_;
  std::vector<const ReferenceRead*> observations_;
};

// Concrete windows only share an observation index when their real byte/slot
// ranges may overlap. Unlike selector components this never connects distinct
// allocation bases. Static interval components prove separation in one scan;
// an unknown address or slot conservatively protects its complete real base.
std::map<const Var*, size_t> ReferenceWindowGroups(const ReferenceGraph& graph) {
  struct Window {
    const Var* value;
    MemRefPtr memory;
  };
  struct KnownWindow {
    const Var* value;
    uint64_t slot;
    storage_access_detail::ByteRange bytes;
  };
  std::map<const Var*, std::vector<Window>> bases;
  for (const auto& [value, node] : graph.nodes) {
    if (const auto tile = GetTileTypeWithMemRef(node.value->GetType())) {
      const auto memory = GetDefinedMemRef(tile);
      bases[memory->base_.get()].push_back({value, memory});
    }
  }
  std::map<const Var*, size_t> groups;
  size_t next_group = 0;
  for (auto& [base, windows] : bases) {
    (void)base;
    const auto slot_count = windows.front().memory->slot_count_;
    bool known = true;
    std::vector<KnownWindow> known_windows;
    known_windows.reserve(windows.size());
    for (const auto& window : windows) {
      const auto slot = storage_access_detail::StaticSlot(window.memory);
      const auto bytes = storage_access_detail::StaticBytes(window.memory);
      if (window.memory->slot_count_ != slot_count || !slot || !bytes) {
        known = false;
        break;
      }
      known_windows.push_back({window.value, *slot, *bytes});
    }
    if (!known) {
      for (const auto& window : windows) groups.emplace(window.value, next_group);
      ++next_group;
      continue;
    }
    std::sort(known_windows.begin(), known_windows.end(), [](const KnownWindow& lhs, const KnownWindow& rhs) {
      if (lhs.slot != rhs.slot) return lhs.slot < rhs.slot;
      return lhs.bytes.begin < rhs.bytes.begin;
    });
    std::optional<uint64_t> current_slot;
    storage_access_detail::Offset end = 0;
    for (const auto& window : known_windows) {
      const auto slot = window.slot;
      const auto bytes = window.bytes;
      if (!current_slot || *current_slot != slot || bytes.begin >= end) {
        ++next_group;
        current_slot = slot;
        end = bytes.end;
      } else {
        end = std::max(end, bytes.end);
      }
      groups.emplace(window.value, next_group);
    }
    ++next_group;
  }
  return groups;
}

class ReferenceEffects : public IRVisitor {
 public:
  std::vector<ReferenceRead> reads;
  std::vector<ReferenceWrite> writes;
  std::vector<ReferenceNoAlias> no_alias;

  ReferenceEffects(const ReferenceGraph& graph, const std::map<const Var*, size_t>& ages, const StmtPtr& body)
      : graph_(graph), ages_(ages) {
    reverse_.Run(body);
    for (const auto& loop : graph.loops) loops_.emplace(loop.statement.get(), &loop);
    using CarryOrigin = std::pair<size_t, const Var*>;
    std::priority_queue<CarryOrigin, std::vector<CarryOrigin>, std::greater<>> pending;
    for (const auto& loop : graph.loops) {
      if (!memory_reuse::LoopMayExecute(loop.statement, true)) continue;
      for (const auto& argument : loop.arguments) {
        if (As<TileType>(argument->GetType()) && !GetTileTypeWithMemRef(argument->GetType())) {
          pending.emplace(loop.begin, argument.get());
        }
      }
    }
    std::set<const Var*> visited;
    while (!pending.empty()) {
      const auto [begin, value] = pending.top();
      pending.pop();
      if (!visited.insert(value).second) continue;
      const auto& node = graph.nodes.at(value);
      if (auto type = GetTileTypeWithMemRef(node.value->GetType())) {
        auto [found, inserted] = carried_origins_.emplace(GetDefinedMemRef(type)->base_.get(), begin);
        if (!inserted) found->second = std::min(found->second, begin);
      }
      for (const auto* source : node.sources) pending.emplace(begin, source);
    }
    for (const auto& [value, node] : graph.nodes) {
      (void)value;
      if (auto type = GetTileTypeWithMemRef(node.value->GetType()); type && (node.endpoint || node.writer)) {
        const auto* base = GetDefinedMemRef(type)->base_.get();
        auto [found, inserted] = first_definitions_.emplace(base, node.definition);
        if (!inserted) found->second = std::min(found->second, node.definition);
      }
    }
    VisitStmt(body);
    AddDemandedForwardingReads();
  }

 protected:
  void VisitVarLike_(const VarPtr& value) override {
    if (!As<TileType>(value->GetType()) || !graph_.nodes.count(value.get())) return;
    if (read_) Read(value);
    if (write_) writes.push_back({value, order_, reverse_order_, span_, order_});
  }

  void VisitExpr_(const IterArgPtr& value) override { VisitVarLike_(value); }

  void VisitExpr_(const CallPtr& call) override {
    const auto* entry = ReferenceOpEntry(call);
    const bool payload = ReferenceCallHasPayload(call);
    // Functional registry arguments default to Read. Missing native-access
    // evidence is not a declaration that every argument can be mutated;
    // reserve that conservative bound for opaque, unregistered operations.
    const bool unknown = !entry;
    const auto writeback = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
    const bool outer_read = read_;
    const bool outer_write = write_;
    for (size_t i = 0; i < call->args_.size(); ++i) {
      ArgEffect effect = ArgEffect::Read;
      if (entry && entry->GetIRStage() == OpIRStage::Functional) {
        effect = entry->GetArgEffect(i, call->kwargs_);
      }
      read_ = payload && ArgEffectReads(effect);
      write_ = payload && (unknown || ArgEffectWrites(effect) || (writeback && *writeback == i));
      VisitExpr(call->args_[i]);
    }
    read_ = true;
    write_ = false;
    for (const auto& [key, value] : call->attrs_) {
      (void)key;
      ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
    }
    read_ = outer_read;
    write_ = outer_write;
  }

  void VisitStmt_(const AssignStmtPtr& statement) override {
    At(statement);
    const auto call = As<Call>(statement->value_);
    if (!call && As<TileType>(statement->var_->GetType())) return;  // A rename does not read payloads.
    VisitExpr(statement->value_);
    const auto definition = graph_.nodes.find(statement->var_.get());
    // Registered native writebacks were already recorded through their real
    // destination operand above. Their fresh result version aliases that same
    // write; it is not another unknown destination when its MemRef is absent.
    if (definition != graph_.nodes.end() && ReferenceCallHasPayload(call) && definition->second.endpoint) {
      size_t boundary = order_;
      const auto memory = GetTileTypeWithMemRef(statement->var_->GetType());
      if (definition->second.endpoint && memory &&
          first_definitions_.at(GetDefinedMemRef(memory)->base_.get()) == order_) {
        // Initializing a distinct owned allocation does not overwrite the
        // selector's other alternatives. In a repeated loop it can overwrite
        // a prior carried instance, but never a fresh sibling value produced
        // earlier in this same iteration. Keep those version cases separate.
        const auto carried = carried_origins_.find(GetDefinedMemRef(memory)->base_.get());
        boundary = carried == carried_origins_.end() || carried->second >= order_ ? 0 : carried->second + 1;
      }
      writes.push_back({statement->var_, order_, reverse_order_, &statement->span_, boundary});
    }
    const auto* entry = ReferenceOpEntry(call);
    if (!entry || !As<TileType>(statement->var_->GetType()) || !ReferenceCallHasPayload(call)) return;
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const bool forbidden =
          !entry->IsInplaceSafe() || entry->ForbidOutputAliasArgs().count(i) ||
          (i == 0 && op_predicates::CastRequiresDisjointSource(call, !memory_reuse::IsA5Target())) ||
          (i == 1 && IsOp(call, "tile.assemble"));
      if (!forbidden) continue;
      const auto input = AsVarLike(call->args_[i]);
      if (input && As<TileType>(input->GetType())) {
        no_alias.push_back({input, statement->var_, &statement->span_});
      }
    }
  }

  void VisitStmt_(const EvalStmtPtr& statement) override {
    At(statement);
    VisitExpr(statement->expr_);
  }

  void VisitStmt_(const IfStmtPtr& statement) override {
    At(statement);
    VisitExpr(statement->condition_);
    const auto constant = As<ConstBool>(statement->condition_);
    const std::array<StmtPtr, 2> arms{statement->then_body_, statement->else_body_.value_or(nullptr)};
    for (size_t arm = 0; arm < arms.size(); ++arm) {
      if (!arms[arm] || (constant && constant->value_ != (arm == 0))) continue;
      VisitStmt(arms[arm]);
      const auto yield = transform_utils::GetLastYieldStmt(arms[arm]);
      if (!yield) continue;
      At(yield);
      for (size_t slot = 0; slot < statement->return_vars_.size() && slot < yield->value_.size(); ++slot) {
        ForwardedRead(yield->value_[slot], statement->return_vars_[slot]);
      }
    }
  }

  void VisitStmt_(const YieldStmtPtr&) override {}  // A handle binding is not a payload read.

  void VisitStmt_(const ReturnStmtPtr& statement) override {
    At(statement);
    for (const auto& value : statement->value_) VisitExpr(value);
  }

  void VisitStmt_(const ForStmtPtr& loop) override { Loop(loop); }
  void VisitStmt_(const WhileStmtPtr& loop) override { Loop(loop); }

 private:
  const ReferenceGraph& graph_;
  const std::map<const Var*, size_t>& ages_;
  std::map<const Stmt*, const ReferenceLoop*> loops_;
  std::map<const Var*, size_t> first_definitions_;
  std::map<const Var*, size_t> carried_origins_;
  std::vector<const ReferenceLoop*> repeated_;
  memory_reuse::ReverseBranchOrderVisitor reverse_;
  std::vector<std::pair<const Var*, ReferenceRead>> forwarded_reads_;
  size_t order_ = 0;
  size_t reverse_order_ = 0;
  const Span* span_ = nullptr;
  bool read_ = true;
  bool write_ = false;

  void At(const StmtPtr& statement) {
    const auto found = graph_.statement_order.find(statement.get());
    if (found != graph_.statement_order.end()) order_ = found->second;
    reverse_order_ = reverse_.stmt_order.at(statement.get());
    span_ = &statement->span_;
  }

  [[nodiscard]] std::optional<ReferenceRead> ReadEvent(const VarPtr& value) const {
    const auto age = ages_.find(value.get());
    if (age == ages_.end()) return std::nullopt;  // The property verifier diagnoses malformed references.
    size_t last = order_;
    size_t reverse = reverse_order_;
    // Re-reading an externally defined version in another iteration is a
    // future observation even when its textual read precedes the writer.
    // Current IterArgs and writer results have fresh iteration-local ages.
    const auto repeated =
        std::upper_bound(repeated_.begin(), repeated_.end(), age->second,
                         [](size_t birth, const ReferenceLoop* loop) { return birth < loop->begin; });
    if (repeated != repeated_.end()) {
      last = std::max(last, (*repeated)->end + 1);
      reverse = std::max(reverse, reverse_.stmt_end_order.at((*repeated)->statement.get()) + 1);
    }
    return ReferenceRead{value, age->second, last, reverse};
  }

  void Read(const VarPtr& value) {
    if (const auto event = ReadEvent(value)) reads.push_back(*event);
  }

  void ForwardedRead(const ExprPtr& value, const VarPtr& demanded) {
    const auto variable = AsVarLike(value);
    if (!variable || !As<TileType>(variable->GetType())) return;
    if (const auto event = ReadEvent(variable)) forwarded_reads_.emplace_back(demanded.get(), *event);
  }

  static bool MaySkip(const StmtPtr& statement) {
    if (const auto loop = As<ForStmt>(statement)) {
      if (As<ConstInt>(loop->start_) && As<ConstInt>(loop->stop_) && As<ConstInt>(loop->step_)) {
        return !memory_reuse::LoopMayExecute(loop, false);
      }
    }
    if (const auto loop = As<WhileStmt>(statement)) {
      if (const auto condition = As<ConstBool>(loop->condition_)) return !condition->value_;
    }
    return true;
  }

  void AddDemandedForwardingReads() {
    // Demand follows value forwarding, not storage sharing or native writers.
    // In particular, an unused one-trip backedge does not observe its payload.
    std::map<const Var*, std::vector<const Var*>> sources;
    for (const auto& [value, node] : graph_.nodes) {
      if (!node.writer) sources.emplace(value, node.sources);
    }
    for (const auto& info : graph_.loops) {
      const bool enters = memory_reuse::LoopMayExecute(info.statement, false);
      const bool repeats = memory_reuse::LoopMayExecute(info.statement, true);
      const bool skips = MaySkip(info.statement);
      const auto& results = info.results;
      for (size_t slot = 0; slot < info.arguments.size(); ++slot) {
        const auto& argument = info.arguments[slot];
        if (!As<TileType>(argument->GetType())) continue;
        auto& carry = sources[argument.get()];
        carry.clear();
        const auto initial = AsVarLike(argument->initValue_);
        const auto backedge =
            info.yield && slot < info.yield->value_.size() ? AsVarLike(info.yield->value_[slot]) : nullptr;
        if (enters && initial) carry.push_back(initial.get());
        if (repeats && backedge) carry.push_back(backedge.get());
        if (slot < results.size()) {
          auto& output = sources[results[slot].get()];
          output.clear();
          if (skips && initial) output.push_back(initial.get());
          if (enters && backedge) output.push_back(backedge.get());
        }
      }
    }
    std::set<const Var*> demanded;
    std::vector<const Var*> pending;
    for (const auto& read : reads) {
      if (demanded.insert(read.value.get()).second) pending.push_back(read.value.get());
    }
    for (size_t i = 0; i < pending.size(); ++i) {
      const auto found = sources.find(pending[i]);
      if (found == sources.end()) continue;
      for (const auto* source : found->second) {
        if (demanded.insert(source).second) pending.push_back(source);
      }
    }
    for (const auto& [value, read] : forwarded_reads_) {
      if (demanded.count(value)) reads.push_back(read);
    }
  }

  template <typename LoopPtr>
  void Loop(const LoopPtr& loop) {
    At(loop);
    const auto& info = *loops_.at(loop.get());
    if constexpr (std::is_same_v<LoopPtr, ForStmtPtr>) {
      VisitExpr(loop->start_);
      VisitExpr(loop->stop_);
      VisitExpr(loop->step_);
      for (const auto& [key, value] : loop->attrs_) {
        (void)key;
        ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
      }
    }
    const bool enters = memory_reuse::LoopMayExecute(loop, false);
    const bool repeats = memory_reuse::LoopMayExecute(loop, true);
    for (size_t slot = 0; slot < loop->iter_args_.size(); ++slot) {
      const auto& argument = loop->iter_args_[slot];
      if (enters) ForwardedRead(argument->initValue_, argument);
      if (MaySkip(loop) && slot < loop->return_vars_.size()) {
        ForwardedRead(argument->initValue_, loop->return_vars_[slot]);
      }
    }
    if (repeats) repeated_.push_back(&info);
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) {
      VisitExpr(loop->condition_);
    }
    if (enters) {
      VisitStmt(loop->body_);
      order_ = info.end;
      reverse_order_ = reverse_.stmt_end_order.at(loop.get()) + 1;
      span_ = &loop->span_;
      // A carried output must name the version produced by this iteration.
      // Observe backedge values before treating the loop result as the fresh
      // final version. This catches yielding an old carry after mutating it.
      if (info.yield) {
        for (size_t slot = 0; slot < info.yield->value_.size(); ++slot) {
          const auto& value = info.yield->value_[slot];
          if (repeats && slot < loop->iter_args_.size()) ForwardedRead(value, loop->iter_args_[slot]);
          if (slot < loop->return_vars_.size()) ForwardedRead(value, loop->return_vars_[slot]);
        }
      }
    }
    if (repeats) repeated_.pop_back();
  }
};

}  // namespace

StorageReferenceAnalysis AnalyzeStorageReferences(const StmtPtr& body,
                                                  const std::vector<VarPtr>& parameters) {
  const ReferenceGraph graph(body, parameters);
  StorageReferenceAnalysis result;
  std::set<const Var*> origins;
  std::set<const Var*> invalid;
  std::vector<const Var*> reachable;
  for (const auto& [value, node] : graph.nodes) {
    if (GetTileTypeWithMemRef(node.value->GetType()) || node.endpoint) origins.insert(value);
    if (!node.defined || (!node.endpoint && node.sources.empty())) invalid.insert(value);
    if (node.region) {
      const size_t expected = node.loop_result ? 1 : 2;
      if (node.sources.size() != expected) invalid.insert(value);
      for (const auto* source : node.sources) {
        if (!SameReferenceDescriptor(As<TileType>(graph.nodes.at(source).value->GetType()),
                                     As<TileType>(node.value->GetType()))) {
          invalid.insert(value);
        }
      }
    }
    if (node.region && !GetTileTypeWithMemRef(node.value->GetType())) reachable.push_back(value);
  }
  for (const auto& branch : graph.branches) {
    for (const auto& yield : branch.yields) {
      if (yield && yield->value_.size() == branch.statement->return_vars_.size()) continue;
      for (const auto& value : branch.statement->return_vars_) {
        if (graph.nodes.count(value.get())) invalid.insert(value.get());
      }
    }
  }
  for (const auto& loop : graph.loops) {
    const auto& results = loop.results;
    if (loop.arguments.size() == results.size() && loop.yield &&
        loop.yield->value_.size() == loop.arguments.size()) {
      continue;
    }
    for (const auto& argument : loop.arguments) {
      if (graph.nodes.count(argument.get())) invalid.insert(argument.get());
    }
    for (const auto& value : results) {
      if (graph.nodes.count(value.get())) invalid.insert(value.get());
    }
  }
  origins = graph.Propagate(std::move(origins));
  invalid = graph.Propagate(std::move(invalid));
  std::set<const Var*> visited;
  for (size_t i = 0; i < reachable.size(); ++i) {
    const auto* value = reachable[i];
    if (!visited.insert(value).second) continue;
    const auto& node = graph.nodes.at(value);
    if (!origins.count(value) || invalid.count(value)) continue;
    if (auto tile = GetTileTypeWithMemRef(node.value->GetType())) {
      result.source_bases.insert(GetDefinedMemRef(tile)->base_.get());
    } else if (node.region || node.alias || node.endpoint) {
      // Caller-provided intermediate Tile parameters and registered TPop
      // results are known storage sources. Their missing MemRef belongs to
      // this proven reference graph rather than an unallocated producer.
      result.references.insert(value);
    }
    reachable.insert(reachable.end(), node.sources.begin(), node.sources.end());
  }
  // Required aliases of a selector are references too, even if they are only
  // consumed by a native operation and never yield into another region.
  auto aliases = graph.Propagate(result.references);
  for (const auto* value : aliases) {
    const auto& node = graph.nodes.at(value);
    if (!GetTileTypeWithMemRef(node.value->GetType()) && origins.count(value) && !invalid.count(value)) {
      result.references.insert(value);
    }
  }
  return result;
}

StmtPtr NormalizeStorageReferences(const StmtPtr& body, const std::vector<VarPtr>& parameters,
                                   const std::set<const Var*>& required_references) {
  const ReferenceGraph graph(body, parameters);
  auto existing = AnalyzeStorageReferences(body, parameters);
  std::set<const Var*> references = std::move(existing.references);
  references.insert(required_references.begin(), required_references.end());
  const auto acc_continuations = AccContinuationOrders(graph);
  std::map<size_t, const Var*> acc_writers;
  for (const auto& [value, node] : graph.nodes) {
    if (node.writer) acc_writers.emplace(node.definition, value);
  }
  for (const auto& branch : graph.branches) {
    using IncomingPair = std::pair<const Var*, const Var*>;
    std::array<std::map<const Var*, IncomingPair>, 2> shared_sources;
    std::array<std::set<const Var*>, 2> divergent_sources;
    std::vector<std::pair<const Var*, std::array<const Var*, 2>>> incoming_slots;
    for (size_t slot = 0; slot < branch.statement->return_vars_.size(); ++slot) {
      const auto& result = branch.statement->return_vars_[slot];
      if (!As<TileType>(result->GetType()) || graph.FixedResult(result)) continue;
      const auto result_tile = As<TileType>(result->GetType());
      const auto space = result_tile->GetMemorySpace();
      // Copyable spaces retain established destination coalescing and its
      // capacity guarantees. A new reference is necessary for spaces such as
      // Mat that cannot reconcile independently placed windows by copying.
      // Existing reference SSA still propagates through every memory space.
      if (result_tile->memref_.has_value() && space && IsTileMoveEverSupported(*space, *space)) continue;
      std::array<VarPtr, 2> sources;
      for (size_t arm = 0; arm < sources.size(); ++arm) {
        const auto& yield = branch.yields[arm];
        if (yield && slot < yield->value_.size()) sources[arm] = AsVarLike(yield->value_[slot]);
      }
      if (!sources[0] || !sources[1]) continue;
      // A handle selection cannot change the physical element/space/layout
      // contract. Such an edge remains on the established real-transfer path.
      if (!SameReferenceDescriptor(As<TileType>(sources[0]->GetType()), result_tile) ||
          !SameReferenceDescriptor(As<TileType>(sources[1]->GetType()), result_tile)) {
        continue;
      }
      bool native_acc_continuation = false;
      if (space == MemorySpace::Acc && result_tile->memref_.has_value()) {
        const Var* owner = nullptr;
        bool compatible = true;
        for (size_t arm = 0; arm < sources.size(); ++arm) {
          const auto writer = acc_continuations.find(sources[arm].get());
          const auto [begin, end] = branch.ranges[arm];
          if (writer == acc_continuations.end() || writer->second < begin || writer->second > end) continue;
          const auto* incoming_owner = graph.AliasOrigin(acc_writers.at(writer->second));
          compatible &=
              (!owner || owner == incoming_owner) && graph.AliasOrigin(sources[arm].get()) == incoming_owner;
          owner = incoming_owner;
        }
        for (size_t arm = 0; owner && arm < sources.size(); ++arm) {
          const auto* origin = graph.AliasOrigin(sources[arm].get());
          if (origin == owner) continue;
          const auto& producer = graph.nodes.at(origin);
          const auto [begin, end] = branch.ranges[arm];
          const auto tile = GetTileTypeWithMemRef(producer.value->GetType());
          const auto memory = tile ? GetDefinedMemRef(tile) : nullptr;
          compatible &= producer.endpoint && begin <= producer.definition && producer.definition <= end &&
                        memory && !memory->is_pinned_ && !graph.pinned.count(memory->base_.get());
        }
        native_acc_continuation = owner && compatible;
      }
      // Split-K matmul/matmul_acc joins are a native accumulator continuation.
      // Keep its existing coupled destination and old-observer proofs rather
      // than turning the correlated updated/unchanged arms into alternatives
      // in the conservative generic reference-effects analysis. A readonly
      // external Acc selection has no writer inside its arms and remains a phi.
      // An unrelated external seed cannot join this continuation: keep its
      // independent allocation and select its handle instead.
      if (native_acc_continuation) continue;
      if (space && !IsTileMoveEverSupported(*space, *space)) {
        const IncomingPair incoming{graph.AliasOrigin(sources[0].get()), graph.AliasOrigin(sources[1].get())};
        // One producer can feed duplicate result slots without a copy only if
        // the other arm agrees on that equivalence. For (x,x)/(y,z), selecting
        // references is necessary even though every producer is branch-local.
        const std::array<const Var*, 2> origins{incoming.first, incoming.second};
        incoming_slots.emplace_back(result.get(), origins);
        for (size_t arm = 0; arm < origins.size(); ++arm) {
          auto [previous, inserted] = shared_sources[arm].emplace(origins[arm], incoming);
          if (!inserted && previous->second != incoming) divergent_sources[arm].insert(origins[arm]);
        }
      }
      const auto first = GetTileTypeWithMemRef(sources[0]->GetType());
      const auto second = GetTileTypeWithMemRef(sources[1]->GetType());
      // Exact common storage needs no reference selection or transfer. The
      // existing common-window proof still protects independent old observers.
      if (first && second && first->GetMemorySpace() == second->GetMemorySpace() &&
          SamePhysicalWindow(GetDefinedMemRef(first), GetDefinedMemRef(second))) {
        continue;
      }
      bool needs_reference = false;
      for (size_t arm = 0; arm < sources.size(); ++arm) {
        const auto& node = graph.nodes.at(sources[arm].get());
        const auto& origin = graph.nodes.at(graph.AliasOrigin(sources[arm].get()));
        const auto [begin, end] = branch.ranges[arm];
        const auto tile = GetTileTypeWithMemRef(sources[arm]->GetType());
        const auto memory = tile ? GetDefinedMemRef(tile) : nullptr;
        needs_reference |= !node.defined || node.definition < begin || node.definition > end || !memory ||
                           memory->is_pinned_ || graph.pinned.count(memory->base_.get()) || !origin.defined ||
                           origin.definition < begin || origin.definition > end;
        if (node.alias && !node.sources.empty()) {
          const auto source_type =
              GetTileTypeWithMemRef(graph.nodes.at(node.sources.front()).value->GetType());
          needs_reference |=
              source_type && memory && !SamePhysicalWindow(memory, GetDefinedMemRef(source_type));
        }
      }
      if (needs_reference) references.insert(result.get());
    }
    // A second scan marks every duplicate, including slots equal to the first
    // incoming pair after a later conflicting pair. No group is rescanned per
    // peer, so arbitrarily many duplicate results remain O(N log N).
    for (const auto& [result, sources] : incoming_slots) {
      if (divergent_sources[0].count(sources[0]) || divergent_sources[1].count(sources[1])) {
        references.insert(result);
      }
    }
  }
  for (const auto& loop : graph.loops) {
    if (!memory_reuse::LoopMayExecute(loop.statement, false) || !loop.yield) continue;
    for (size_t slot = 0; slot < loop.arguments.size() && slot < loop.yield->value_.size(); ++slot) {
      const auto& argument = loop.arguments[slot];
      const auto type = As<TileType>(argument->GetType());
      const auto space = type ? type->GetMemorySpace() : std::nullopt;
      if (!space || IsTileMoveEverSupported(*space, *space)) continue;
      const auto source = AsVarLike(loop.yield->value_[slot]);
      if (!source) continue;
      const auto initial = AsVarLike(argument->initValue_);
      const auto continuation = acc_continuations.find(source.get());
      if (space == MemorySpace::Acc && type->memref_.has_value() && continuation != acc_continuations.end() &&
          loop.begin < continuation->second && continuation->second < loop.end) {
        const auto* owner = graph.AliasOrigin(acc_writers.at(continuation->second));
        if (graph.AliasOrigin(source.get()) == owner &&
            (owner == argument.get() || (initial && owner == graph.AliasOrigin(initial.get())))) {
          continue;
        }
      }
      const auto initial_type = initial ? GetTileTypeWithMemRef(initial->GetType()) : nullptr;
      const auto source_type = GetTileTypeWithMemRef(source->GetType());
      if (initial_type && source_type &&
          SamePhysicalWindow(GetDefinedMemRef(initial_type), GetDefinedMemRef(source_type))) {
        continue;
      }
      const auto& origin = graph.nodes.at(graph.AliasOrigin(source.get()));
      // An external backedge or a different carried slot selects existing
      // storage. A loop-local native producer retains destination retargeting.
      if (!source_type || !origin.defined || origin.definition <= loop.begin ||
          origin.definition >= loop.end) {
        references.insert(argument.get());
      }
    }
  }
  references = graph.Propagate(std::move(references));
  // Explicit destinations cannot silently become address selections. Reject a
  // reference crossing such a boundary until its payload can be materialized.
  for (const auto* value : references) {
    const auto& node = graph.nodes.at(value);
    // Alias and carry types inherit their source window in InitMemRef. That
    // inherited binding is not an explicit destination; only a separately
    // declared pinned allocation constrains such forwarding values.
    const auto tile = GetTileTypeWithMemRef(node.value->GetType());
    const auto memory = tile ? GetDefinedMemRef(tile) : nullptr;
    bool inherited = false;
    if (memory && !node.branch_result) {
      for (const auto* source : node.sources) {
        const auto type = GetTileTypeWithMemRef(graph.nodes.at(source).value->GetType());
        if (type && SamePhysicalWindow(memory, GetDefinedMemRef(type))) {
          inherited = true;
          break;
        }
      }
    }
    CHECK_SPAN(!memory || inherited || !graph.FixedResult(node.value), node.value->span_)
        << "A dynamic storage reference cannot satisfy an explicit fixed result destination";
  }
  return references.empty() ? body : ReferenceRetyping(references, parameters).VisitStmt(body);
}

void ValidateStorageReferenceWrites(const StmtPtr& body, const std::vector<VarPtr>& parameters,
                                    const std::vector<ParamDirection>& directions,
                                    bool include_concrete_acc_regions) {
  const auto references = AnalyzeStorageReferences(body, parameters);
  if (references.references.empty() && !include_concrete_acc_regions) return;
  const ReferenceGraph graph(body, parameters);
  ReferenceComponents components(graph);
  std::set<size_t> protected_components;
  for (const auto* reference : references.references) protected_components.insert(components.Find(reference));
  if (include_concrete_acc_regions) {
    // Once canonical planning settles Acc If destinations, validate their
    // writes before fixup can request an unavailable Acc-to-Acc edge copy.
    // Connected carries share the If component automatically. Standalone
    // native loops keep their precise window and updated-row overwrite proofs;
    // actual reference carries are already protected above.
    for (const auto& [value, node] : graph.nodes) {
      const auto tile = GetTileTypeWithMemRef(node.value->GetType());
      if (node.branch_result && tile && tile->GetMemorySpace() == MemorySpace::Acc) {
        protected_components.insert(components.Find(value));
      }
    }
  }
  if (protected_components.empty()) return;
  const auto ages = ReferenceVersionAges(graph);
  ReferenceEffects effects(graph, ages, body);
  const auto window_groups = ReferenceWindowGroups(graph);
  std::vector<StorageAccessIndex::Event> readonly_windows;
  std::set<size_t> readonly_components;
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (i < directions.size() && directions[i] != ParamDirection::In) continue;
    if (auto type = GetTileTypeWithMemRef(parameters[i]->GetType())) {
      readonly_windows.push_back({GetDefinedMemRef(type), 1, 1});
      readonly_components.insert(components.Find(parameters[i].get()));
    } else if (As<TileType>(parameters[i]->GetType())) {
      readonly_components.insert(components.Find(parameters[i].get()));
    }
  }
  StorageAccessIndex readonly_storage;
  readonly_storage.Build(readonly_windows);

  // A missing MemRef on the selected input must not erase an operator's
  // no-alias restriction. Whole-base components conservatively protect partial
  // windows too; placement never coalesces these components as allocations.
  for (const auto& pair : effects.no_alias) {
    const auto input_memory = GetTileTypeWithMemRef(pair.input->GetType());
    const auto output_memory = GetTileTypeWithMemRef(pair.output->GetType());
    if (input_memory && output_memory &&
        !MemRef::MayAlias(GetDefinedMemRef(input_memory), GetDefinedMemRef(output_memory))) {
      continue;  // A selector linking both roots does not make the roots alias.
    }
    const auto input = components.Find(pair.input.get());
    const auto output = components.Find(pair.output.get());
    CHECK_SPAN(input != output || !protected_components.count(input), *pair.span)
        << "A native operation requires its output to be disjoint from a possible selected storage source";
  }

  // Sweep versions in birth order. At write t, the index contains exactly
  // observations of versions older than t. Query a real observation after t
  // in both branch orders. Every event enters once: O((N+E) log N).
  std::sort(effects.reads.begin(), effects.reads.end(),
            [](const ReferenceRead& lhs, const ReferenceRead& rhs) { return lhs.age < rhs.age; });
  std::sort(effects.writes.begin(), effects.writes.end(),
            [](const ReferenceWrite& lhs, const ReferenceWrite& rhs) {
              return lhs.age_boundary < rhs.age_boundary;
            });
  // Dynamic writers can select any origin in their component. A concrete
  // writer instead queries dynamic observations plus its real spatial group,
  // so a selector linking A/B never makes a concrete write to A clobber B.
  using ObservationKey = std::pair<unsigned, size_t>;
  std::map<ObservationKey, ReferenceObservationIndex> observations;
  const auto read_keys = [&](const ReferenceRead& read) {
    const auto component = components.Find(read.value.get());
    const auto concrete = window_groups.find(read.value.get());
    return std::array<ObservationKey, 2>{
        ObservationKey{0, component},
        concrete == window_groups.end() ? ObservationKey{1, component} : ObservationKey{2, concrete->second}};
  };
  for (const auto& read : effects.reads) {
    for (const auto& key : read_keys(read)) observations[key].Reserve(read.order);
  }
  for (auto& [key, index] : observations) {
    (void)key;
    index.Build();
  }
  size_t next_read = 0;
  for (const auto& write : effects.writes) {
    while (next_read < effects.reads.size() && effects.reads[next_read].age < write.age_boundary) {
      const auto& read = effects.reads[next_read++];
      for (const auto& key : read_keys(read)) observations.at(key).Insert(read);
    }
    const size_t component = components.Find(write.value.get());
    if (!protected_components.count(component)) continue;
    const auto written_type = GetTileTypeWithMemRef(write.value->GetType());
    const bool readonly = written_type
                              ? readonly_storage.MayOverlapAfter(GetDefinedMemRef(written_type), 0, 0)
                              : readonly_components.count(component);
    CHECK_SPAN(!readonly, *write.span)
        << "Writing a selected storage reference may modify a readonly caller buffer";
    const auto after = [&](const ObservationKey& key) {
      const auto index = observations.find(key);
      return index == observations.end() ? nullptr : index->second.After(write);
    };
    const auto* old = after({written_type ? 1U : 0U, component});
    if (!old && written_type) old = after({2, window_groups.at(write.value.get())});
    CHECK_SPAN(!old, *write.span)
        << "Writing selected storage may overwrite an independently observed older value version ('"
        << old->value->name_hint_
        << "'). Preserve that value in independent storage or produce the new value directly in its "
           "destination";
  }
}

}  // namespace pypto::ir
