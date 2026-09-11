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

#include "pypto/ir/transforms/dsa/reuse_penalty_recognizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {
namespace {

// Complexity. Let V be the statements of one InCore function, E its
// dependency edges, A its recorded allocation accesses, k the fixed number of
// abstract access resources (`kResourceCount`), and P the number of
// lifetime-compatible cross-resource allocation pairs this function actually
// contains.
//
//   access collection + ordering index   O(k * (V + E))
//   completion frontiers                 O(A * k)
//   pair enumeration (indexed sweep)     O(B log B + k^2 * P)
//
// Ordering is answered in O(1) by a chain-cover reachability index rather than
// by per-statement transitive predecessor sets, and the sweep only visits
// allocation pairs that already share a memory space, have compatible
// lifetimes, and carry two different resources. The analysis is therefore
// O(N log N) in the size of the IR; the residual `P` term is the size of the
// model's own output. The explicit pairwise DSA-RP model is output-sensitive
// by construction: a kernel can genuinely contain Theta(B^2) penalty pairs for
// B reusable buffers, and no pair enumeration can be cheaper than the pairs it
// must report. Nothing here performs a nested scan over IR nodes.

enum class AccessKind : uint8_t {
  Read,
  Write,
};

/// Portable memory classes used to identify an access route. L0A/L0B/L0C stay
/// separate physical arenas; they are merged only in this taxonomy.
enum class MemoryClass : uint8_t {
  External,
  Ub,
  L1,
  L0,
  Scalar,
};

/// Abstract execution resource of one access. Two accesses on distinct
/// resources may execute asynchronously, so physical reuse across them needs a
/// completion handoff.
enum class AccessResource : uint8_t {
  InboundDma,
  OutboundDma,
  L0ToExternal,
  L1ToL0,
  L0ToL1,
  UbToL1,
  L1ToUb,
  UbToL0,
  L0ToUb,
  VectorCompute,
  MatrixCompute,
  ScalarAccess,
};

constexpr size_t kResourceCount = static_cast<size_t>(AccessResource::ScalarAccess) + 1;

size_t ResourceIndex(AccessResource resource) { return static_cast<size_t>(resource); }

struct AccessRoute {
  MemoryClass source = MemoryClass::Scalar;
  MemoryClass destination = MemoryClass::Scalar;
  AccessResource resource = AccessResource::ScalarAccess;
};

struct BranchChoice {
  size_t id = 0;
  bool alternative = false;
  size_t loop_depth = 0;

  bool operator<(const BranchChoice& other) const {
    return std::tie(id, alternative, loop_depth) < std::tie(other.id, other.alternative, other.loop_depth);
  }

  bool operator==(const BranchChoice& other) const {
    return id == other.id && alternative == other.alternative && loop_depth == other.loop_depth;
  }
};

struct AccessEndpoint {
  /// Ordering-index node of the hosting statement.
  size_t node = 0;
  const Stmt* statement = nullptr;
  size_t global_order = 0;
  AccessRoute route;
  MemorySpace memory_space = MemorySpace::ScalarLocal;
  AccessKind access_kind = AccessKind::Read;
  std::vector<BranchChoice> branch_path;
  std::vector<size_t> loop_stack;
  uint64_t byte_offset = 0;
  uint64_t byte_size = 0;
  bool range_known = false;
  bool full_allocation = false;
};

struct AllocationAccessSummary {
  std::vector<AccessEndpoint> accesses;
  /// Accesses whose range, evidence, or backend support is not established.
  /// A single one disqualifies the whole allocation from promotion.
  size_t unsupported_accesses = 0;
};

// ---------------------------------------------------------------------------
// Chain-cover reachability
// ---------------------------------------------------------------------------

/**
 * @brief O(1) happens-before queries over the statement dependency DAG.
 *
 * Every access-bearing statement carries exactly one abstract resource, and a
 * resource is modelled as one completion-ordered issue chain. Adding that
 * chain order to the dependency edges makes the resources a chain cover of the
 * happens-before order, so one vector clock per node decides an ordering query
 * by reading a single component. Building the index costs O(k * (V + E)).
 *
 * Nodes are created in traversal order and every edge runs from a lower node
 * id to a higher one, so node id order is a linear extension and one forward
 * relaxation pass suffices.
 */
class OrderIndex {
 public:
  static constexpr size_t kNoChain = std::numeric_limits<size_t>::max();

  size_t AddNode() {
    clocks_.emplace_back();
    clocks_.back().fill(0);
    successors_.emplace_back();
    chain_.push_back(kNoChain);
    chain_index_.push_back(0);
    return clocks_.size() - 1;
  }

  void AddEdge(size_t from, size_t to) {
    INTERNAL_CHECK(from < to && to < clocks_.size())
        << "Internal error: DSA ordering edge " << from << " -> " << to << " is not forward";
    successors_[from].push_back(to);
  }

  /// Place a statement on its resource chain. Returns false when the node is
  /// already on a different chain, which the caller must treat as unsupported.
  bool PlaceOnChain(size_t node, size_t chain, size_t index) {
    INTERNAL_CHECK(node < clocks_.size() && chain < kResourceCount)
        << "Internal error: invalid DSA chain placement";
    if (chain_[node] != kNoChain) return chain_[node] == chain;
    chain_[node] = chain;
    chain_index_[node] = index;
    return true;
  }

  void Finalize() {
    for (size_t node = 0; node < clocks_.size(); ++node) {
      if (chain_[node] != kNoChain) {
        auto& own = clocks_[node][chain_[node]];
        own = std::max<uint64_t>(own, chain_index_[node] + 1);
      }
      for (size_t successor : successors_[node]) {
        for (size_t component = 0; component < kResourceCount; ++component) {
          clocks_[successor][component] = std::max(clocks_[successor][component], clocks_[node][component]);
        }
      }
    }
  }

  /// True when the access at `earlier` is guaranteed to finish before the
  /// access at `later` begins. Accesses of one statement are never ordered.
  [[nodiscard]] bool HappensBefore(size_t earlier, size_t later) const {
    if (earlier == later) return false;
    const size_t chain = chain_[earlier];
    if (chain == kNoChain) return false;
    return clocks_[later][chain] > chain_index_[earlier];
  }

  [[nodiscard]] size_t ChainOf(size_t node) const { return chain_[node]; }
  [[nodiscard]] uint64_t ChainIndexOf(size_t node) const { return chain_index_[node]; }
  [[nodiscard]] uint64_t ClockComponent(size_t node, size_t component) const {
    return clocks_[node][component];
  }

 private:
  std::vector<std::array<uint64_t, kResourceCount>> clocks_;
  std::vector<std::vector<size_t>> successors_;
  std::vector<size_t> chain_;
  std::vector<uint64_t> chain_index_;
};

// ---------------------------------------------------------------------------
// Route classification
// ---------------------------------------------------------------------------

std::optional<MemorySpace> GetMemorySpace(const TypePtr& type) {
  if (!type) return std::nullopt;
  const auto shaped = As<ShapedType>(type);
  return shaped ? shaped->GetMemorySpace() : std::nullopt;
}

std::optional<MemorySpace> GetMemorySpace(const VarPtr& var) {
  return var ? GetMemorySpace(var->GetType()) : std::nullopt;
}

MemoryClass ClassifyMemory(MemorySpace space) {
  switch (space) {
    case MemorySpace::DDR:
      return MemoryClass::External;
    case MemorySpace::Vec:
      return MemoryClass::Ub;
    case MemorySpace::Mat:
      return MemoryClass::L1;
    case MemorySpace::Left:
    case MemorySpace::Right:
    case MemorySpace::Acc:
    case MemorySpace::Bias:
    case MemorySpace::LeftScale:
    case MemorySpace::RightScale:
      return MemoryClass::L0;
    case MemorySpace::ScalarLocal:
      return MemoryClass::Scalar;
  }
  throw pypto::ValueError("Unknown memory space in DSA route classifier");
}

std::optional<AccessRoute> LookupTransferRoute(MemoryClass source, MemoryClass destination) {
  using Memory = MemoryClass;
  using Resource = AccessResource;
  if (source == Memory::External && (destination == Memory::Ub || destination == Memory::L1)) {
    return AccessRoute{source, destination, Resource::InboundDma};
  }
  if ((source == Memory::Ub || source == Memory::L1) && destination == Memory::External) {
    return AccessRoute{source, destination, Resource::OutboundDma};
  }
  if (source == Memory::L0 && destination == Memory::External) {
    return AccessRoute{source, destination, Resource::L0ToExternal};
  }
  if (source == Memory::L1 && destination == Memory::L0) {
    return AccessRoute{source, destination, Resource::L1ToL0};
  }
  if (source == Memory::L0 && destination == Memory::L1) {
    return AccessRoute{source, destination, Resource::L0ToL1};
  }
  if (source == Memory::Ub && destination == Memory::L1) {
    return AccessRoute{source, destination, Resource::UbToL1};
  }
  if (source == Memory::L1 && destination == Memory::Ub) {
    return AccessRoute{source, destination, Resource::L1ToUb};
  }
  if (source == Memory::Ub && destination == Memory::L0) {
    return AccessRoute{source, destination, Resource::UbToL0};
  }
  if (source == Memory::L0 && destination == Memory::Ub) {
    return AccessRoute{source, destination, Resource::L0ToUb};
  }
  return std::nullopt;
}

bool SameAllocation(const VarPtr& first, const VarPtr& second) {
  const auto first_tile = first ? As<TileType>(first->GetType()) : nullptr;
  const auto second_tile = second ? As<TileType>(second->GetType()) : nullptr;
  if (!first_tile || !second_tile || !first_tile->memref_ || !second_tile->memref_) return false;
  return GetDefinedMemRef(first_tile)->base_.get() == GetDefinedMemRef(second_tile)->base_.get();
}

ArgEffect GetArgumentEffect(const CallPtr& call, size_t argument_index) {
  const auto& registry = OpRegistry::GetInstance();
  if (!call || !call->op_ || !registry.IsRegistered(call->op_->name_)) return ArgEffect::Read;
  return registry.GetEntry(call->op_->name_).GetArgEffect(argument_index, call->kwargs_);
}

bool ResultWriteIsRepresentedByArgument(const VarPtr& result, const std::vector<VarPtr>& arguments) {
  return result && std::any_of(arguments.begin(), arguments.end(),
                               [&](const VarPtr& argument) { return SameAllocation(result, argument); });
}

void CollectMemoryClasses(const TypePtr& type, std::set<MemoryClass>* classes) {
  if (!type || classes == nullptr) return;
  if (const auto space = GetMemorySpace(type)) {
    classes->insert(ClassifyMemory(*space));
    return;
  }
  if (const auto tuple = As<TupleType>(type)) {
    for (const TypePtr& element : tuple->types_) CollectMemoryClasses(element, classes);
  }
}

/// Identify the abstract source/destination route of one operation. The route
/// is target independent: it names which engine class must complete, not which
/// hardware pipe the selected SoC assigns.
std::optional<AccessRoute> ClassifyOperationRoute(const CallPtr& call, const std::vector<VarPtr>& results) {
  if (!call || !call->op_) return std::nullopt;
  using Memory = MemoryClass;
  using Resource = AccessResource;

  std::set<Memory> result_classes;
  for (const VarPtr& result : results) {
    CollectMemoryClasses(result ? result->GetType() : nullptr, &result_classes);
  }
  if (result_classes.empty()) CollectMemoryClasses(call->GetType(), &result_classes);
  std::vector<std::pair<VarPtr, Memory>> inputs;
  bool has_scalar_input = false;
  for (size_t argument_index = 0; argument_index < call->args_.size(); ++argument_index) {
    const VarPtr var = AsVarLike(call->args_[argument_index]);
    const auto space = GetMemorySpace(var);
    const ArgEffect effect = GetArgumentEffect(call, argument_index);
    if (space && ArgEffectWrites(effect)) result_classes.insert(ClassifyMemory(*space));
    if (space && ArgEffectReads(effect)) {
      inputs.emplace_back(var, ClassifyMemory(*space));
    } else if (ArgEffectReads(effect) && var && As<ScalarType>(var->GetType())) {
      has_scalar_input = true;
    }
  }
  const std::optional<Memory> result_class =
      result_classes.size() == 1 ? std::optional<Memory>(*result_classes.begin()) : std::nullopt;

  // A cube-to-vector push reads an accumulator but returns no local value, so
  // result-type inference alone cannot classify this real source access.
  if (IsOp(call, "tile.tpush_to_aiv") && results.empty() && inputs.size() == 1 &&
      inputs.front().second == Memory::L0) {
    return AccessRoute{Memory::L0, Memory::Ub, Resource::L0ToUb};
  }

  const auto scalar_result = std::find_if(results.begin(), results.end(), [](const VarPtr& result) {
    return result && As<ScalarType>(result->GetType());
  });
  if (scalar_result != results.end()) {
    const auto local = std::find_if(inputs.begin(), inputs.end(), [](const auto& input) {
      return input.second != Memory::External && input.second != Memory::Scalar;
    });
    if (local != inputs.end()) {
      return AccessRoute{local->second, Memory::Scalar, Resource::ScalarAccess};
    }
  }

  if (result_class && *result_class != Memory::External && has_scalar_input &&
      std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) {
        return input.second == *result_class &&
               std::any_of(results.begin(), results.end(),
                           [&](const VarPtr& result) { return SameAllocation(input.first, result); });
      })) {
    return AccessRoute{Memory::Scalar, *result_class, Resource::ScalarAccess};
  }

  if (result_class && *result_class != Memory::External) {
    if (std::any_of(inputs.begin(), inputs.end(),
                    [](const auto& input) { return input.second == Memory::External; })) {
      return LookupTransferRoute(Memory::External, *result_class);
    }
    std::set<Memory> local_inputs;
    for (const auto& [var, memory] : inputs) {
      static_cast<void>(var);
      if (memory != Memory::External && memory != Memory::Scalar) local_inputs.insert(memory);
    }
    // A mutating operation such as tile.assemble reads its destination and
    // returns an updated value in that same memory class. The destination-side
    // read does not identify the transfer engine.
    std::set<Memory> transfer_sources = local_inputs;
    transfer_sources.erase(*result_class);
    if (transfer_sources.size() == 1) {
      if (const auto transfer = LookupTransferRoute(*transfer_sources.begin(), *result_class)) {
        return transfer;
      }
    }
    // A move within one memory class is a copy, not a compute operation: the
    // vector unit performs it even between two L0 buffers. Reading the class
    // alone would call it matrix compute and hide a real handoff.
    if (IsOp(call, "tile.move") && transfer_sources.empty() && !local_inputs.empty()) {
      return AccessRoute{*result_class, *result_class, Resource::VectorCompute};
    }
    const bool all_ub =
        *result_class == Memory::Ub && std::all_of(local_inputs.begin(), local_inputs.end(),
                                                   [](Memory memory) { return memory == Memory::Ub; });
    if (all_ub) return AccessRoute{Memory::Ub, Memory::Ub, Resource::VectorCompute};
    const bool all_l0 =
        *result_class == Memory::L0 && std::all_of(local_inputs.begin(), local_inputs.end(),
                                                   [](Memory memory) { return memory == Memory::L0; });
    if (all_l0) return AccessRoute{Memory::L0, Memory::L0, Resource::MatrixCompute};
  }

  if (result_class && *result_class == Memory::External) {
    std::set<Memory> local_inputs;
    for (const auto& [var, memory] : inputs) {
      static_cast<void>(var);
      if (memory != Memory::External && memory != Memory::Scalar) local_inputs.insert(memory);
    }
    if (local_inputs.size() == 1) return LookupTransferRoute(*local_inputs.begin(), Memory::External);
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Control-flow compatibility
// ---------------------------------------------------------------------------

bool ControlPathsCompatible(const AccessEndpoint& first, const AccessEndpoint& second,
                            std::optional<size_t> crossed_loop_depth) {
  for (const BranchChoice& first_choice : first.branch_path) {
    for (const BranchChoice& second_choice : second.branch_path) {
      if (first_choice.id != second_choice.id || first_choice.alternative == second_choice.alternative) {
        continue;
      }
      // Branches inside a crossed loop may choose different arms in different
      // iterations. Branches outside that loop remain mutually exclusive.
      if (!crossed_loop_depth || first_choice.loop_depth < *crossed_loop_depth) return false;
    }
  }
  return true;
}

std::vector<std::pair<size_t, size_t>> SharedLoopContexts(const AccessEndpoint& first,
                                                          const AccessEndpoint& second) {
  std::vector<std::pair<size_t, size_t>> result;
  const size_t limit = std::min(first.loop_stack.size(), second.loop_stack.size());
  for (size_t index = 0; index < limit && first.loop_stack[index] == second.loop_stack[index]; ++index) {
    result.emplace_back(first.loop_stack[index], index + 1);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Access collection
// ---------------------------------------------------------------------------

using TupleResultElements = std::unordered_map<const Var*, std::map<int, VarPtr>>;

class TupleResultCollector : public IRVisitor {
 public:
  const TupleResultElements& Elements() const { return elements_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (const auto get_item = As<TupleGetItemExpr>(op->value_)) {
      if (const VarPtr tuple = AsVarLike(get_item->tuple_); tuple && get_item->index_ >= 0) {
        elements_[tuple.get()][get_item->index_] = op->var_;
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  TupleResultElements elements_;
};

class AccessCollector : public IRVisitor {
 public:
  AccessCollector(const AllocationPlan& plan, std::unordered_map<const Var*, size_t> interval_by_base,
                  TupleResultElements tuple_results, const backend::Backend* backend)
      : plan_(plan),
        interval_by_base_(std::move(interval_by_base)),
        tuple_results_(std::move(tuple_results)),
        backend_(backend),
        summaries_(plan.intervals.size()) {}

  void Collect(const StmtPtr& body) {
    VisitRegion(body);
    order_.Finalize();
  }

  const std::vector<AllocationAccessSummary>& Summaries() const { return summaries_; }
  const OrderIndex& Order() const { return order_; }

 protected:
  void VisitStmt_(const SeqStmtsPtr& op) override { VisitSequence(op); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    RecordValue(op->value_, op->var_);
    ++global_order_;
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    RecordValue(op->expr_, nullptr);
    ++global_order_;
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    for (const ExprPtr& value : op->value_) {
      RecordValue(value, nullptr);
      ++global_order_;
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    const size_t branch_id = next_control_id_++;
    branch_path_.push_back({branch_id, false, loop_stack_.size()});
    VisitRegion(op->then_body_);
    branch_path_.pop_back();
    if (op->else_body_) {
      branch_path_.push_back({branch_id, true, loop_stack_.size()});
      VisitRegion(*op->else_body_);
      branch_path_.pop_back();
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override { VisitLoop(op->body_); }

  void VisitStmt_(const WhileStmtPtr& op) override { VisitLoop(op->body_); }

 private:
  struct CompoundFrame {
    size_t entry = 0;
    std::vector<size_t> children;
  };

  void VisitLoop(const StmtPtr& body) {
    const size_t loop_id = next_control_id_++;
    loop_stack_.push_back(loop_id);
    VisitRegion(body);
    loop_stack_.pop_back();
  }

  void VisitRegion(const StmtPtr& region) {
    if (const auto seq = As<SeqStmts>(region)) {
      VisitSequence(seq);
      return;
    }
    VisitRegionStatement(region, nullptr);
  }

  void VisitSequence(const SeqStmtsPtr& op) {
    const stmt_dep::StmtDependencyGraph graph = stmt_dep::BuildStmtDependencyGraph(op);
    for (const StmtPtr& stmt : op->stmts_) {
      const auto found = graph.predecessors.find(stmt.get());
      VisitRegionStatement(stmt, found == graph.predecessors.end() ? nullptr : &found->second);
    }
  }

  size_t RepresentativeOf(const Stmt* statement) const {
    const auto found = representative_of_stmt_.find(statement);
    INTERNAL_CHECK(found != representative_of_stmt_.end())
        << "Internal error: DSA ordering node is missing for a visited statement";
    return found->second;
  }

  /// Give one statement an ordering node, wire the edges that place it in the
  /// happens-before order, and let its own body statements nest below it.
  void VisitRegionStatement(const StmtPtr& stmt, const std::unordered_set<const Stmt*>* predecessors) {
    const size_t node = order_.AddNode();
    node_statement_[node] = stmt.get();
    if (predecessors != nullptr) {
      for (const Stmt* predecessor : *predecessors) {
        const auto found = representative_of_stmt_.find(predecessor);
        if (found == representative_of_stmt_.end()) continue;
        order_.AddEdge(found->second, node);
      }
    }
    // Anything ordered before the enclosing statement is ordered before every
    // statement of its body.
    if (!compound_stack_.empty()) order_.AddEdge(compound_stack_.back().entry, node);

    node_stack_.push_back(node);
    compound_stack_.push_back({node, {}});
    VisitStmt(stmt);
    const CompoundFrame frame = std::move(compound_stack_.back());
    compound_stack_.pop_back();
    node_stack_.pop_back();

    // Everything inside a compound statement completes before anything the
    // compound itself precedes, so its body feeds one exit node that later
    // dependants observe in place of the compound.
    size_t representative = node;
    if (!frame.children.empty()) {
      representative = order_.AddNode();
      node_statement_[representative] = stmt.get();
      order_.AddEdge(node, representative);
      for (size_t child : frame.children) order_.AddEdge(child, representative);
    }
    representative_of_stmt_[stmt.get()] = representative;
    if (!compound_stack_.empty()) compound_stack_.back().children.push_back(representative);
  }

  std::optional<size_t> FindInterval(const VarPtr& var) const {
    if (!var) return std::nullopt;
    const auto tile = As<TileType>(var->GetType());
    if (!tile || !tile->memref_) return std::nullopt;
    const MemRefPtr memref = GetDefinedMemRef(tile);
    const auto found = interval_by_base_.find(memref->base_.get());
    return found == interval_by_base_.end() ? std::nullopt : std::optional<size_t>(found->second);
  }

  std::vector<VarPtr> ResolveCallResults(const VarPtr& result) const {
    if (!result) return {};
    if (!As<TupleType>(result->GetType())) return {result};
    const auto found = tuple_results_.find(result.get());
    if (found == tuple_results_.end()) return {};
    std::vector<VarPtr> results;
    results.reserve(found->second.size());
    for (const auto& [index, element] : found->second) {
      static_cast<void>(index);
      results.push_back(element);
    }
    return results;
  }

  struct AccessRange {
    uint64_t offset = 0;
    uint64_t size = 0;
    bool known = false;
    bool full_allocation = false;
  };

  AccessRange GetAccessRange(const VarPtr& var, size_t interval) const {
    AccessRange range;
    const auto tile = As<TileType>(var->GetType());
    if (!tile || !tile->memref_) return range;
    const MemRefPtr memref = GetDefinedMemRef(tile);
    const auto offset = As<ConstInt>(memref->byte_offset_);
    if (!offset || offset->value_ < 0) return range;
    range.offset = static_cast<uint64_t>(offset->value_);
    range.size = memref->size_;
    range.known = true;
    // A view that covers the allocation's bytes but only part of its logical
    // window is still a partial access.
    range.full_allocation = range.offset == 0 &&
                            range.size == static_cast<uint64_t>(plan_.intervals[interval].size) &&
                            IsFullyValidTile(tile);
    return range;
  }

  static bool IsFullyValidTile(const TileTypePtr& tile) {
    return tile && AreExprVectorsEqual(GetValidShape(tile), tile->shape_);
  }

  static bool IsZeroOffsetTuple(const ExprPtr& expression, size_t expected_rank) {
    const auto offsets = As<MakeTuple>(expression);
    if (!offsets || offsets->elements_.size() != expected_rank) return false;
    return std::all_of(offsets->elements_.begin(), offsets->elements_.end(), [](const ExprPtr& offset) {
      const auto value = As<ConstInt>(offset);
      return value && value->value_ == 0;
    });
  }

  static bool IsProvablyWholeStore(const CallPtr& call) {
    if (!IsOp(call, "tile.store") || call->args_.size() != 3) return false;
    return IsFullyValidTile(As<TileType>(call->args_[0]->GetType()));
  }

  static bool IsProvablyWholeAssemble(const CallPtr& call, const std::vector<VarPtr>& results) {
    if (!IsOp(call, "tile.assemble") || call->args_.size() != 3 || results.size() != 1) return false;

    const VarPtr target_var = AsVarLike(call->args_[0]);
    const VarPtr source_var = AsVarLike(call->args_[1]);
    const VarPtr& result_var = results.front();
    const auto target = target_var ? As<TileType>(target_var->GetType()) : nullptr;
    const auto source = source_var ? As<TileType>(source_var->GetType()) : nullptr;
    const auto result = result_var ? As<TileType>(result_var->GetType()) : nullptr;
    if (!target || !source || !result || !target->memref_ || !source->memref_ || !result->memref_) {
      return false;
    }
    if (!AreExprVectorsEqual(target->shape_, source->shape_) ||
        !AreExprVectorsEqual(target->shape_, result->shape_)) {
      return false;
    }
    if (!IsFullyValidTile(target) || !IsFullyValidTile(source) || !IsFullyValidTile(result)) return false;
    if (!IsZeroOffsetTuple(call->args_[2], target->shape_.size())) return false;
    return SameAllocation(target_var, result_var);
  }

  static ExecutionMemoryAccessEvidence ResolveAccessEvidence(const CallPtr& call,
                                                             const std::vector<VarPtr>& results) {
    const auto& registry = OpRegistry::GetInstance();
    if (!registry.IsRegistered(call->op_->name_)) return ExecutionMemoryAccessEvidence::Unknown;

    const ExecutionMemoryAccessEvidence registered =
        registry.GetEntry(call->op_->name_).GetExecutionMemoryAccessEvidence();
    if (registered != ExecutionMemoryAccessEvidence::Unknown) return registered;

    // Destination-passing and subrange operations remain Unknown by default.
    // These two are promoted locally only when their operands prove an exact
    // whole-window access.
    if (IsProvablyWholeStore(call) || IsProvablyWholeAssemble(call, results)) {
      return ExecutionMemoryAccessEvidence::Functional;
    }
    return ExecutionMemoryAccessEvidence::Unknown;
  }

  void Poison(const std::vector<std::pair<size_t, VarPtr>>& reads,
              const std::vector<std::pair<size_t, VarPtr>>& writes) {
    std::unordered_set<size_t> touched;
    for (const auto& [interval, var] : reads) {
      static_cast<void>(var);
      touched.insert(interval);
    }
    for (const auto& [interval, var] : writes) {
      static_cast<void>(var);
      touched.insert(interval);
    }
    for (size_t interval : touched) ++summaries_[interval].unsupported_accesses;
  }

  void RecordAccess(size_t interval, AccessEndpoint endpoint, const VarPtr& var) {
    const auto memory_space = GetMemorySpace(var);
    if (!memory_space) {
      ++summaries_[interval].unsupported_accesses;
      return;
    }
    const AccessRange range = GetAccessRange(var, interval);
    endpoint.memory_space = *memory_space;
    endpoint.byte_offset = range.offset;
    endpoint.byte_size = range.size;
    endpoint.range_known = range.known;
    endpoint.full_allocation = range.full_allocation;
    // A partial or symbolic access leaves the allocation-level completion
    // frontier incomplete: a different full-range access must not then promote
    // a whole-buffer penalty pair.
    if (!range.full_allocation) ++summaries_[interval].unsupported_accesses;
    summaries_[interval].accesses.push_back(std::move(endpoint));
  }

  void RecordValue(const ExprPtr& value, const VarPtr& result) {
    if (const auto submit = As<Submit>(value)) {
      RecordSubmit(submit);
      return;
    }
    RecordCall(As<Call>(value), result);
  }

  /// A task launch is not modelled by the structural catalog. Its arguments
  /// and TaskId dependencies are real uses, so every allocation it touches is
  /// left unsupported rather than silently unconstrained.
  void RecordSubmit(const SubmitPtr& submit) {
    std::unordered_set<size_t> touched;
    const auto note = [&](const ExprPtr& operand) {
      if (const auto interval = FindInterval(AsVarLike(operand))) touched.insert(*interval);
    };
    for (const ExprPtr& argument : submit->args_) note(argument);
    for (const ExprPtr& dependency : submit->deps_) note(dependency);
    for (size_t interval : touched) ++summaries_[interval].unsupported_accesses;
  }

  void RecordCall(const CallPtr& call, const VarPtr& result) {
    if (!call || !call->op_) return;

    const std::vector<VarPtr> results = ResolveCallResults(result);
    const bool whole_assemble = IsProvablyWholeAssemble(call, results);
    std::vector<std::pair<size_t, VarPtr>> reads;
    std::vector<std::pair<size_t, VarPtr>> writes;
    std::vector<VarPtr> write_arguments;
    for (size_t argument_index = 0; argument_index < call->args_.size(); ++argument_index) {
      const VarPtr var = AsVarLike(call->args_[argument_index]);
      const auto interval = FindInterval(var);
      if (!interval) continue;
      const ArgEffect effect = GetArgumentEffect(call, argument_index);
      // A whole-window assemble overwrites its target, so the target's old
      // contents are not an input access.
      if (ArgEffectReads(effect) && !(whole_assemble && argument_index == 0)) {
        reads.emplace_back(*interval, var);
      }
      if (ArgEffectWrites(effect)) {
        writes.emplace_back(*interval, var);
        write_arguments.push_back(var);
      }
    }
    for (const VarPtr& output : results) {
      if (ResultWriteIsRepresentedByArgument(output, write_arguments)) continue;
      if (const auto interval = FindInterval(output)) writes.emplace_back(*interval, output);
    }
    if (reads.empty() && writes.empty()) return;

    const ExecutionMemoryAccessEvidence evidence = ResolveAccessEvidence(call, results);
    if (evidence == ExecutionMemoryAccessEvidence::NoAccess) return;
    if (evidence == ExecutionMemoryAccessEvidence::Unknown) {
      Poison(reads, writes);
      return;
    }

    const std::optional<AccessRoute> route = ClassifyOperationRoute(call, results);
    if (!route.has_value()) {
      Poison(reads, writes);
      return;
    }
    // The route names the abstract engine class; the backend decides whether
    // the selected SoC can actually perform this operation's transfer. An
    // operation it cannot classify stays unsupported.
    if (backend_ != nullptr && !backend_->TryInferPipe(call).has_value()) {
      Poison(reads, writes);
      return;
    }

    INTERNAL_CHECK(!node_stack_.empty()) << "Internal error: access statement has no ordering node";
    const size_t node = node_stack_.back();
    const size_t resource = ResourceIndex(route->resource);
    if (!order_.PlaceOnChain(node, resource, next_resource_index_[resource])) {
      // One statement carrying two different resources would break the
      // single-chain model, so its allocations stay unsupported.
      Poison(reads, writes);
      return;
    }
    if (order_.ChainIndexOf(node) == next_resource_index_[resource]) {
      auto& last = last_node_on_resource_[resource];
      if (last.has_value()) order_.AddEdge(*last, node);
      last = node;
      ++next_resource_index_[resource];
    }

    AccessEndpoint read_endpoint;
    read_endpoint.node = node;
    read_endpoint.statement = node_statement_.at(node);
    read_endpoint.global_order = global_order_;
    read_endpoint.route = *route;
    read_endpoint.access_kind = AccessKind::Read;
    read_endpoint.branch_path = branch_path_;
    read_endpoint.loop_stack = loop_stack_;
    AccessEndpoint write_endpoint = read_endpoint;
    write_endpoint.access_kind = AccessKind::Write;
    for (const auto& [interval, var] : reads) RecordAccess(interval, read_endpoint, var);
    for (const auto& [interval, output] : writes) RecordAccess(interval, write_endpoint, output);
  }

  const AllocationPlan& plan_;
  std::unordered_map<const Var*, size_t> interval_by_base_;
  TupleResultElements tuple_results_;
  const backend::Backend* backend_;
  std::vector<AllocationAccessSummary> summaries_;
  OrderIndex order_;
  /// Node a later dependant observes in place of a statement: its own node,
  /// or its exit node when it has a body.
  std::unordered_map<const Stmt*, size_t> representative_of_stmt_;
  std::unordered_map<size_t, const Stmt*> node_statement_;
  std::vector<size_t> node_stack_;
  std::vector<CompoundFrame> compound_stack_;
  std::array<std::optional<size_t>, kResourceCount> last_node_on_resource_;
  std::array<size_t, kResourceCount> next_resource_index_{};
  std::vector<BranchChoice> branch_path_;
  std::vector<size_t> loop_stack_;
  size_t global_order_ = 0;
  size_t next_control_id_ = 0;
};

// ---------------------------------------------------------------------------
// Completion frontiers
// ---------------------------------------------------------------------------

struct AllocationFrontier {
  /// Every access of this allocation is classified, backend supported, and
  /// covers the full allocation.
  bool complete = false;
  /// A minimal access is not a write, so axiom A3's initialization guarantee
  /// is unproven and the allocation must not be reused as a penalty target.
  bool conservative_initial_anchor = false;
  std::vector<AccessEndpoint> terminal;
  std::vector<AccessEndpoint> initial_write;
  /// Resources appearing in `terminal` / `initial_write`, for the sweep index.
  std::vector<size_t> terminal_resources;
  std::vector<size_t> initial_resources;
};

/// Extreme values of one clock component over a set, with the node attaining
/// them, so an element can be compared against the set excluding itself.
struct ComponentExtremes {
  uint64_t best = 0;
  size_t best_node = 0;
  uint64_t second = 0;
  bool has_best = false;
  bool has_second = false;
};

/// Maximal accesses under happens-before: those no other access follows.
std::vector<AccessEndpoint> BuildTerminalFrontier(const std::vector<AccessEndpoint>& accesses,
                                                  const OrderIndex& order) {
  // One representative per (resource, control path, byte range): the latest on
  // its chain dominates the others by chain order.
  using FrontierKey = std::tuple<AccessResource, MemorySpace, std::vector<BranchChoice>, std::vector<size_t>,
                                 bool, uint64_t, uint64_t>;
  std::map<FrontierKey, AccessEndpoint> representatives;
  for (const AccessEndpoint& access : accesses) {
    const FrontierKey key{access.route.resource, access.memory_space, access.branch_path, access.loop_stack,
                          access.range_known,    access.byte_offset,  access.byte_size};
    const auto found = representatives.find(key);
    if (found == representatives.end() ||
        order.ChainIndexOf(found->second.node) <= order.ChainIndexOf(access.node)) {
      representatives[key] = access;
    }
  }

  std::vector<AccessEndpoint> candidates;
  candidates.reserve(representatives.size());
  for (auto& [key, access] : representatives) {
    static_cast<void>(key);
    candidates.push_back(std::move(access));
  }

  // `candidate` is dominated when some other candidate's clock already covers
  // the candidate's own chain position, which is one component lookup.
  std::array<ComponentExtremes, kResourceCount> maxima{};
  for (const AccessEndpoint& candidate : candidates) {
    for (size_t component = 0; component < kResourceCount; ++component) {
      const uint64_t value = order.ClockComponent(candidate.node, component);
      ComponentExtremes& extremes = maxima[component];
      if (!extremes.has_best || value > extremes.best) {
        if (extremes.has_best && extremes.best_node != candidate.node) {
          extremes.second = extremes.best;
          extremes.has_second = true;
        }
        extremes.best = value;
        extremes.best_node = candidate.node;
        extremes.has_best = true;
      } else if (candidate.node != extremes.best_node && (!extremes.has_second || value > extremes.second)) {
        extremes.second = value;
        extremes.has_second = true;
      }
    }
  }

  std::vector<AccessEndpoint> maximal;
  maximal.reserve(candidates.size());
  for (const AccessEndpoint& candidate : candidates) {
    const size_t chain = order.ChainOf(candidate.node);
    INTERNAL_CHECK(chain != OrderIndex::kNoChain)
        << "Internal error: a recorded DSA access has no resource chain";
    const ComponentExtremes& extremes = maxima[chain];
    const uint64_t reach = extremes.best_node == candidate.node ? (extremes.has_second ? extremes.second : 0)
                                                                : (extremes.has_best ? extremes.best : 0);
    if (reach <= order.ChainIndexOf(candidate.node)) maximal.push_back(candidate);
  }
  return maximal;
}

/// Minimal accesses under happens-before. Axiom A3 requires each of them to
/// define the allocation; a minimal read leaves initialization unproven.
std::vector<AccessEndpoint> BuildInitialWriteFrontier(const std::vector<AccessEndpoint>& accesses,
                                                      const OrderIndex& order, bool* conservative_anchor) {
  if (conservative_anchor != nullptr) *conservative_anchor = false;
  if (accesses.empty()) return {};

  std::array<ComponentExtremes, kResourceCount> minima{};
  for (const AccessEndpoint& access : accesses) {
    const size_t chain = order.ChainOf(access.node);
    INTERNAL_CHECK(chain != OrderIndex::kNoChain)
        << "Internal error: a recorded DSA access has no resource chain";
    const uint64_t index = order.ChainIndexOf(access.node);
    ComponentExtremes& extremes = minima[chain];
    if (!extremes.has_best || index < extremes.best) {
      if (extremes.has_best && extremes.best_node != access.node) {
        extremes.second = extremes.best;
        extremes.has_second = true;
      }
      extremes.best = index;
      extremes.best_node = access.node;
      extremes.has_best = true;
    } else if (access.node != extremes.best_node && (!extremes.has_second || index < extremes.second)) {
      extremes.second = index;
      extremes.has_second = true;
    }
  }

  // A source-order "first access" is not sufficient inside structured control:
  // two writes in opposite branches are both minimal. Keep the whole antichain.
  std::vector<AccessEndpoint> minimal;
  for (const AccessEndpoint& access : accesses) {
    bool has_predecessor = false;
    for (size_t component = 0; component < kResourceCount && !has_predecessor; ++component) {
      const ComponentExtremes& extremes = minima[component];
      if (!extremes.has_best) continue;
      const bool self = extremes.best_node == access.node;
      if (self && !extremes.has_second) continue;
      const uint64_t index = self ? extremes.second : extremes.best;
      has_predecessor = order.ClockComponent(access.node, component) > index;
    }
    if (!has_predecessor) minimal.push_back(access);
  }

  INTERNAL_CHECK(!minimal.empty()) << "Internal error: a finite access order has no minimal access";
  if (conservative_anchor != nullptr) {
    *conservative_anchor = std::any_of(minimal.begin(), minimal.end(), [](const AccessEndpoint& access) {
      return access.access_kind != AccessKind::Write;
    });
  }
  return minimal;
}

std::vector<size_t> DistinctResources(const std::vector<AccessEndpoint>& accesses) {
  std::set<size_t> resources;
  for (const AccessEndpoint& access : accesses) resources.insert(ResourceIndex(access.route.resource));
  return {resources.begin(), resources.end()};
}

// ---------------------------------------------------------------------------
// Pair promotion
// ---------------------------------------------------------------------------

/// True when giving `next` the bytes of `prior` requires a completion handoff
/// between two different resources. `prior`'s maximal accesses must finish
/// before `next`'s first write begins, and the compiler cannot prove that
/// ordering across resources at this point in the pipeline.
bool RequiresHandoff(const AllocationFrontier& prior, const AllocationFrontier& next, bool loop_carried) {
  if (!prior.complete || !next.complete || next.conservative_initial_anchor) return false;
  for (const AccessEndpoint& terminal : prior.terminal) {
    for (const AccessEndpoint& initial : next.initial_write) {
      if (terminal.route.resource == initial.route.resource) continue;
      // Two endpoints of one operation are an alias-contract question, not an
      // optional reuse.
      if (terminal.statement == initial.statement) continue;
      if (!loop_carried) {
        if (terminal.global_order > initial.global_order) continue;
        if (ControlPathsCompatible(terminal, initial, std::nullopt)) return true;
        continue;
      }
      // A shared address inside a repeated loop also creates the cyclic
      // handoff from the later value in iteration k to the earlier value in
      // iteration k+1.
      if (terminal.global_order <= initial.global_order) continue;
      for (const auto& [loop_id, depth] : SharedLoopContexts(terminal, initial)) {
        static_cast<void>(loop_id);
        if (ControlPathsCompatible(terminal, initial, depth)) return true;
      }
    }
  }
  return false;
}

bool LifetimesPermitReuse(const LifetimeInterval& first, const LifetimeInterval& second) {
  return first.last_use_point <= second.def_point || second.last_use_point <= first.def_point;
}

struct PairKey {
  size_t first = 0;
  size_t second = 0;

  bool operator==(const PairKey& other) const { return first == other.first && second == other.second; }
};

struct PairKeyHash {
  size_t operator()(const PairKey& pair) const {
    const size_t first_hash = std::hash<size_t>{}(pair.first);
    const size_t second_hash = std::hash<size_t>{}(pair.second);
    return first_hash ^ (second_hash + 0x9e3779b9U + (first_hash << 6U) + (first_hash >> 2U));
  }
};

PairKey NormalizePair(size_t first, size_t second) {
  return first < second ? PairKey{first, second} : PairKey{second, first};
}

/// Shared promotion decision for one ordered allocation pair.
class PairPromoter {
 public:
  PairPromoter(const AllocationPlan& plan, const std::vector<AllocationFrontier>& frontiers)
      : plan_(plan), frontiers_(frontiers) {
    // A correctness separation removes the pair from the model entirely, and a
    // pipeline-only separation is performance intent owned by the no-fit
    // relaxation. Neither may also carry a hazard penalty.
    for (const AllocationSeparation& separation : plan.separations) {
      separated_.insert(NormalizePair(separation.first, separation.second));
    }
  }

  /// Evaluate one unordered allocation pair exactly once.
  void Consider(size_t first, size_t second, std::vector<RecognizedReusePenalty>* penalties) {
    if (first == second) return;
    const PairKey pair = NormalizePair(first, second);
    if (!evaluated_.insert(pair).second) return;
    if (separated_.count(pair) != 0) return;

    const LifetimeInterval& first_lifetime = plan_.intervals[pair.first];
    const LifetimeInterval& second_lifetime = plan_.intervals[pair.second];
    if (first_lifetime.memory_space != second_lifetime.memory_space ||
        !LifetimesPermitReuse(first_lifetime, second_lifetime)) {
      return;
    }

    size_t earlier = pair.first;
    size_t later = pair.second;
    if (second_lifetime.last_use_point <= first_lifetime.def_point) std::swap(earlier, later);

    if (RequiresHandoff(frontiers_[earlier], frontiers_[later], false) ||
        RequiresHandoff(frontiers_[later], frontiers_[earlier], true)) {
      penalties->push_back({pair.first, pair.second, 1});
    }
  }

 private:
  const AllocationPlan& plan_;
  const std::vector<AllocationFrontier>& frontiers_;
  std::unordered_set<PairKey, PairKeyHash> separated_;
  std::unordered_set<PairKey, PairKeyHash> evaluated_;
};

/// Readable specification of the pair universe: every allocation pair. Used by
/// tests to pin the indexed sweep below.
std::vector<RecognizedReusePenalty> EnumerateAllPairs(const AllocationPlan& plan,
                                                      const std::vector<AllocationFrontier>& frontiers) {
  PairPromoter promoter(plan, frontiers);
  std::vector<RecognizedReusePenalty> penalties;
  for (size_t first = 0; first < frontiers.size(); ++first) {
    for (size_t second = first + 1; second < frontiers.size(); ++second) {
      promoter.Consider(first, second, &penalties);
    }
  }
  return penalties;
}

/// Output-sensitive enumeration. Allocations are swept in lifetime order and
/// indexed by the resources on their frontiers, so a pair is examined only
/// when it already shares a memory space, has compatible lifetimes, and offers
/// two different resources.
std::vector<RecognizedReusePenalty> EnumerateIndexedSweep(const AllocationPlan& plan,
                                                          const std::vector<AllocationFrontier>& frontiers) {
  PairPromoter promoter(plan, frontiers);
  std::vector<RecognizedReusePenalty> penalties;

  std::map<MemorySpace, std::vector<size_t>> active_by_space;
  for (size_t index = 0; index < frontiers.size(); ++index) {
    if (!frontiers[index].complete) continue;
    active_by_space[plan.intervals[index].memory_space].push_back(index);
  }

  for (auto& [space, active] : active_by_space) {
    static_cast<void>(space);
    std::vector<size_t> by_definition = active;
    std::vector<size_t> by_end = active;
    const auto definition_key = [&](size_t index) {
      return std::pair{plan.intervals[index].def_point, index};
    };
    const auto end_key = [&](size_t index) { return std::pair{plan.intervals[index].last_use_point, index}; };
    std::sort(by_definition.begin(), by_definition.end(),
              [&](size_t lhs, size_t rhs) { return definition_key(lhs) < definition_key(rhs); });
    std::sort(by_end.begin(), by_end.end(),
              [&](size_t lhs, size_t rhs) { return end_key(lhs) < end_key(rhs); });

    // Allocations whose lifetime has ended, bucketed by the resources they
    // expose on each side of a handoff.
    std::array<std::vector<size_t>, kResourceCount> terminal_bucket;
    std::array<std::vector<size_t>, kResourceCount> initial_bucket;
    size_t end_cursor = 0;
    for (size_t current : by_definition) {
      const int current_definition = plan.intervals[current].def_point;
      while (end_cursor < by_end.size() &&
             plan.intervals[by_end[end_cursor]].last_use_point <= current_definition) {
        const size_t closed = by_end[end_cursor++];
        for (size_t resource : frontiers[closed].terminal_resources) {
          terminal_bucket[resource].push_back(closed);
        }
        if (!frontiers[closed].conservative_initial_anchor) {
          for (size_t resource : frontiers[closed].initial_resources) {
            initial_bucket[resource].push_back(closed);
          }
        }
      }

      const AllocationFrontier& frontier = frontiers[current];
      // Forward handoff: a closed allocation hands its bytes to `current`.
      if (!frontier.conservative_initial_anchor) {
        for (size_t initial_resource : frontier.initial_resources) {
          for (size_t resource = 0; resource < kResourceCount; ++resource) {
            if (resource == initial_resource) continue;
            for (size_t prior : terminal_bucket[resource]) {
              promoter.Consider(prior, current, &penalties);
            }
          }
        }
      }
      // Loop-carried handoff: `current` hands its bytes back to a closed
      // allocation's first write in the next iteration.
      for (size_t terminal_resource : frontier.terminal_resources) {
        for (size_t resource = 0; resource < kResourceCount; ++resource) {
          if (resource == terminal_resource) continue;
          for (size_t next : initial_bucket[resource]) {
            promoter.Consider(next, current, &penalties);
          }
        }
      }
    }
  }
  return penalties;
}

}  // namespace

std::vector<RecognizedReusePenalty> RecognizeReusePenalties(const FunctionPtr& func,
                                                            const AllocationPlan& allocation_plan,
                                                            const backend::Backend& backend,
                                                            ReuseEnumeration enumeration) {
  if (!func || allocation_plan.intervals.empty()) return {};

  std::unordered_map<const Var*, size_t> interval_by_base;
  for (size_t index = 0; index < allocation_plan.intervals.size(); ++index) {
    const auto tile = As<TileType>(allocation_plan.intervals[index].variable->GetType());
    if (!tile || !tile->memref_) continue;
    interval_by_base.emplace(GetDefinedMemRef(tile)->base_.get(), index);
  }

  TupleResultCollector tuple_result_collector;
  tuple_result_collector.VisitStmt(func->body_);
  AccessCollector collector(allocation_plan, std::move(interval_by_base), tuple_result_collector.Elements(),
                            &backend);
  collector.Collect(func->body_);

  std::vector<AllocationFrontier> frontiers(collector.Summaries().size());
  for (size_t index = 0; index < collector.Summaries().size(); ++index) {
    const AllocationAccessSummary& summary = collector.Summaries()[index];
    AllocationFrontier& frontier = frontiers[index];
    if (summary.unsupported_accesses != 0 || summary.accesses.empty()) continue;
    frontier.complete = true;
    frontier.terminal = BuildTerminalFrontier(summary.accesses, collector.Order());
    frontier.initial_write =
        BuildInitialWriteFrontier(summary.accesses, collector.Order(), &frontier.conservative_initial_anchor);
    frontier.terminal_resources = DistinctResources(frontier.terminal);
    frontier.initial_resources = DistinctResources(frontier.initial_write);
  }

  std::vector<RecognizedReusePenalty> penalties = enumeration == ReuseEnumeration::ReferenceAllPairs
                                                      ? EnumerateAllPairs(allocation_plan, frontiers)
                                                      : EnumerateIndexedSweep(allocation_plan, frontiers);
  std::sort(penalties.begin(), penalties.end(),
            [](const RecognizedReusePenalty& lhs, const RecognizedReusePenalty& rhs) {
              return std::tie(lhs.first_interval, lhs.second_interval) <
                     std::tie(rhs.first_interval, rhs.second_interval);
            });
  return penalties;
}

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto
