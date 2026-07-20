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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {
namespace {

enum class AccessKind : uint8_t {
  Read,
  Write,
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
  size_t region = 0;
  size_t statement_index = 0;
  size_t global_order = 0;
  const Stmt* statement = nullptr;
  RecognizedAccessRoute route;
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
  size_t unsupported_accesses = 0;
};

using FrontierKey =
    std::tuple<RecognizedMemoryClass, RecognizedMemoryClass, RecognizedAccessResource, MemorySpace,
               std::vector<BranchChoice>, std::vector<size_t>, bool, uint64_t, uint64_t>;

FrontierKey GetFrontierKey(const AccessEndpoint& endpoint) {
  return {endpoint.route.source, endpoint.route.destination, endpoint.route.resource,
          endpoint.memory_space, endpoint.branch_path,       endpoint.loop_stack,
          endpoint.range_known,  endpoint.byte_offset,       endpoint.byte_size};
}

std::vector<AccessEndpoint> BuildTerminalFrontier(const AllocationAccessSummary& summary) {
  std::map<FrontierKey, AccessEndpoint> terminal;
  for (const AccessEndpoint& endpoint : summary.accesses) {
    const FrontierKey key = GetFrontierKey(endpoint);
    const auto found = terminal.find(key);
    if (found == terminal.end() || found->second.global_order <= endpoint.global_order) {
      terminal[key] = endpoint;
    }
  }
  std::vector<AccessEndpoint> result;
  result.reserve(terminal.size());
  for (const auto& [key, endpoint] : terminal) {
    static_cast<void>(key);
    result.push_back(endpoint);
  }
  return result;
}

std::vector<AccessEndpoint> BuildInitialWriteFrontier(const AllocationAccessSummary& summary) {
  std::map<FrontierKey, AccessEndpoint> initial;
  for (const AccessEndpoint& endpoint : summary.accesses) {
    if (endpoint.access_kind != AccessKind::Write) continue;
    const FrontierKey key = GetFrontierKey(endpoint);
    const auto found = initial.find(key);
    if (found == initial.end() || endpoint.global_order < found->second.global_order) {
      initial[key] = endpoint;
    }
  }
  std::vector<AccessEndpoint> result;
  result.reserve(initial.size());
  for (const auto& [key, endpoint] : initial) {
    static_cast<void>(key);
    result.push_back(endpoint);
  }
  return result;
}

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

std::optional<MemorySpace> GetMemorySpace(const TypePtr& type) {
  if (!type) return std::nullopt;
  const auto shaped = As<ShapedType>(type);
  return shaped ? shaped->GetMemorySpace() : std::nullopt;
}

std::optional<MemorySpace> GetMemorySpace(const VarPtr& var) {
  return var ? GetMemorySpace(var->GetType()) : std::nullopt;
}

RecognizedMemoryClass ClassifyMemory(MemorySpace space) {
  switch (space) {
    case MemorySpace::DDR:
      return RecognizedMemoryClass::External;
    case MemorySpace::Vec:
      return RecognizedMemoryClass::Ub;
    case MemorySpace::Mat:
      return RecognizedMemoryClass::L1;
    case MemorySpace::Left:
    case MemorySpace::Right:
    case MemorySpace::Acc:
    case MemorySpace::Bias:
      return RecognizedMemoryClass::L0;
    case MemorySpace::ScalarLocal:
      return RecognizedMemoryClass::Scalar;
  }
  throw pypto::ValueError("Unknown memory space in DSA route classifier");
}

std::optional<RecognizedAccessRoute> LookupTransferRoute(RecognizedMemoryClass source,
                                                         RecognizedMemoryClass destination) {
  using Memory = RecognizedMemoryClass;
  using Resource = RecognizedAccessResource;
  if (source == Memory::External && (destination == Memory::Ub || destination == Memory::L1)) {
    return RecognizedAccessRoute{source, destination, Resource::InboundDma};
  }
  if ((source == Memory::Ub || source == Memory::L1) && destination == Memory::External) {
    return RecognizedAccessRoute{source, destination, Resource::OutboundDma};
  }
  if (source == Memory::L0 && destination == Memory::External) {
    return RecognizedAccessRoute{source, destination, Resource::L0ToExternal};
  }
  if (source == Memory::L1 && destination == Memory::L0) {
    return RecognizedAccessRoute{source, destination, Resource::L1ToL0};
  }
  if (source == Memory::L0 && destination == Memory::L1) {
    return RecognizedAccessRoute{source, destination, Resource::L0ToL1};
  }
  if (source == Memory::Ub && destination == Memory::L1) {
    return RecognizedAccessRoute{source, destination, Resource::UbToL1};
  }
  if (source == Memory::L1 && destination == Memory::Ub) {
    return RecognizedAccessRoute{source, destination, Resource::L1ToUb};
  }
  if (source == Memory::Ub && destination == Memory::L0) {
    return RecognizedAccessRoute{source, destination, Resource::UbToL0};
  }
  if (source == Memory::L0 && destination == Memory::Ub) {
    return RecognizedAccessRoute{source, destination, Resource::L0ToUb};
  }
  return std::nullopt;
}

bool SameAllocation(const VarPtr& first, const VarPtr& second) {
  const auto first_tile = first ? As<TileType>(first->GetType()) : nullptr;
  const auto second_tile = second ? As<TileType>(second->GetType()) : nullptr;
  if (!first_tile || !second_tile || !first_tile->memref_ || !second_tile->memref_) return false;
  return GetDefinedMemRef(first_tile)->base_.get() == GetDefinedMemRef(second_tile)->base_.get();
}

bool HasExecutionMemoryAccess(const CallPtr& call) {
  if (!call || !call->op_) return false;
  const auto& registry = OpRegistry::GetInstance();
  return !registry.IsRegistered(call->op_->name_) ||
         registry.GetEntry(call->op_->name_).HasExecutionMemoryAccess();
}

void CollectMemoryClasses(const TypePtr& type, std::set<RecognizedMemoryClass>* classes) {
  if (!type || classes == nullptr) return;
  if (const auto space = GetMemorySpace(type)) {
    classes->insert(ClassifyMemory(*space));
    return;
  }
  if (const auto tuple = As<TupleType>(type)) {
    for (const TypePtr& element : tuple->types_) CollectMemoryClasses(element, classes);
  }
}

std::optional<RecognizedAccessRoute> ClassifyOperationRoute(const CallPtr& call,
                                                            const std::vector<VarPtr>& results) {
  if (!call || !HasExecutionMemoryAccess(call)) return std::nullopt;
  using Memory = RecognizedMemoryClass;
  using Resource = RecognizedAccessResource;

  std::set<Memory> result_classes;
  for (const VarPtr& result : results) {
    CollectMemoryClasses(result ? result->GetType() : nullptr, &result_classes);
  }
  if (result_classes.empty()) CollectMemoryClasses(call->GetType(), &result_classes);
  const std::optional<Memory> result_class =
      result_classes.size() == 1 ? std::optional<Memory>(*result_classes.begin()) : std::nullopt;
  std::vector<std::pair<VarPtr, Memory>> inputs;
  bool has_scalar_input = false;
  for (const ExprPtr& argument : call->args_) {
    const VarPtr var = AsVarLike(argument);
    const auto space = GetMemorySpace(var);
    if (space) {
      inputs.emplace_back(var, ClassifyMemory(*space));
    } else if (var && As<ScalarType>(var->GetType())) {
      has_scalar_input = true;
    }
  }

  const auto scalar_result = std::find_if(results.begin(), results.end(), [](const VarPtr& result) {
    return result && As<ScalarType>(result->GetType());
  });
  if (scalar_result != results.end()) {
    const auto local = std::find_if(inputs.begin(), inputs.end(), [](const auto& input) {
      return input.second != Memory::External && input.second != Memory::Scalar;
    });
    if (local != inputs.end()) {
      return RecognizedAccessRoute{local->second, Memory::Scalar, Resource::ScalarAccess};
    }
  }

  if (result_class && *result_class != Memory::External && has_scalar_input &&
      std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) {
        return input.second == *result_class &&
               std::any_of(results.begin(), results.end(),
                           [&](const VarPtr& result) { return SameAllocation(input.first, result); });
      })) {
    return RecognizedAccessRoute{Memory::Scalar, *result_class, Resource::ScalarAccess};
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
    if (local_inputs.size() == 1 && *local_inputs.begin() != *result_class) {
      return LookupTransferRoute(*local_inputs.begin(), *result_class);
    }
    const bool all_ub =
        *result_class == Memory::Ub && std::all_of(local_inputs.begin(), local_inputs.end(),
                                                   [](Memory memory) { return memory == Memory::Ub; });
    if (all_ub) return RecognizedAccessRoute{Memory::Ub, Memory::Ub, Resource::VectorCompute};
    const bool all_l0 =
        *result_class == Memory::L0 && std::all_of(local_inputs.begin(), local_inputs.end(),
                                                   [](Memory memory) { return memory == Memory::L0; });
    if (all_l0) return RecognizedAccessRoute{Memory::L0, Memory::L0, Resource::MatrixCompute};
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
                  TupleResultElements tuple_results)
      : plan_(plan),
        interval_by_base_(std::move(interval_by_base)),
        tuple_results_(std::move(tuple_results)),
        summaries_(plan.intervals.size()) {}

  void Collect(const StmtPtr& body) { VisitStmt(body); }

  const std::vector<AllocationAccessSummary>& Summaries() const { return summaries_; }

  bool TransitivelyOrdered(const Stmt* earlier, const Stmt* later) {
    auto cached = transitive_predecessors_.find(later);
    if (cached == transitive_predecessors_.end()) {
      std::unordered_set<const Stmt*> ancestors;
      std::vector<const Stmt*> worklist{later};
      while (!worklist.empty()) {
        const Stmt* current = worklist.back();
        worklist.pop_back();
        const auto found = predecessors_.find(current);
        if (found == predecessors_.end()) continue;
        for (const Stmt* predecessor : found->second) {
          if (ancestors.insert(predecessor).second) worklist.push_back(predecessor);
        }
      }
      cached = transitive_predecessors_.emplace(later, std::move(ancestors)).first;
    }
    return cached->second.count(earlier) != 0;
  }

 protected:
  void VisitStmt_(const SeqStmtsPtr& op) override {
    const size_t previous_region = current_region_;
    const size_t previous_index = current_statement_index_;
    const bool previous_region_supported = current_region_supported_;
    current_region_ = next_region_++;
    current_region_supported_ = std::all_of(op->stmts_.begin(), op->stmts_.end(), [](const StmtPtr& stmt) {
      return As<AssignStmt>(stmt) || As<EvalStmt>(stmt) || As<YieldStmt>(stmt) || As<ReturnStmt>(stmt);
    });

    if (current_region_supported_) {
      const stmt_dep::StmtDependencyGraph graph = stmt_dep::BuildStmtDependencyGraph(op);
      for (const auto& [statement, predecessors] : graph.predecessors) {
        predecessors_[statement].insert(predecessors.begin(), predecessors.end());
      }
    }

    for (size_t index = 0; index < op->stmts_.size(); ++index) {
      current_statement_index_ = index;
      VisitStmt(op->stmts_[index]);
    }
    current_region_ = previous_region;
    current_statement_index_ = previous_index;
    current_region_supported_ = previous_region_supported;
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    const auto call = As<Call>(op->value_);
    RecordCall(call, op->var_, op.get());
    ++global_order_;
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    RecordCall(As<Call>(op->expr_), nullptr, op.get());
    ++global_order_;
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    for (const ExprPtr& value : op->value_) {
      RecordCall(As<Call>(value), nullptr, op.get());
      ++global_order_;
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    const size_t branch_id = next_control_id_++;
    branch_path_.push_back({branch_id, false, loop_stack_.size()});
    VisitStmt(op->then_body_);
    branch_path_.pop_back();
    if (op->else_body_) {
      branch_path_.push_back({branch_id, true, loop_stack_.size()});
      VisitStmt(*op->else_body_);
      branch_path_.pop_back();
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    const size_t loop_id = next_control_id_++;
    loop_stack_.push_back(loop_id);
    VisitStmt(op->body_);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    const size_t loop_id = next_control_id_++;
    loop_stack_.push_back(loop_id);
    VisitStmt(op->body_);
    loop_stack_.pop_back();
  }

 private:
  const AllocationPlan& plan_;
  std::unordered_map<const Var*, size_t> interval_by_base_;
  TupleResultElements tuple_results_;
  std::vector<AllocationAccessSummary> summaries_;
  std::unordered_map<const Stmt*, std::unordered_set<const Stmt*>> predecessors_;
  std::unordered_map<const Stmt*, std::unordered_set<const Stmt*>> transitive_predecessors_;
  size_t next_region_ = 0;
  size_t current_region_ = 0;
  size_t current_statement_index_ = 0;
  size_t global_order_ = 0;
  bool current_region_supported_ = false;
  size_t next_control_id_ = 0;
  std::vector<BranchChoice> branch_path_;
  std::vector<size_t> loop_stack_;

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
    range.full_allocation =
        range.offset == 0 && range.size == static_cast<uint64_t>(plan_.intervals[interval].size);
    return range;
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
    summaries_[interval].accesses.push_back(std::move(endpoint));
  }

  void RecordCall(const CallPtr& call, const VarPtr& result, const Stmt* statement) {
    if (!call || !HasExecutionMemoryAccess(call)) return;
    std::vector<std::pair<size_t, VarPtr>> reads;
    for (const ExprPtr& argument : call->args_) {
      const VarPtr var = AsVarLike(argument);
      const auto interval = FindInterval(var);
      if (interval) reads.emplace_back(*interval, var);
    }
    const std::vector<VarPtr> results = ResolveCallResults(result);
    std::vector<std::pair<size_t, VarPtr>> writes;
    for (const VarPtr& output : results) {
      if (const auto interval = FindInterval(output)) writes.emplace_back(*interval, output);
    }
    if (reads.empty() && writes.empty()) return;

    const std::optional<RecognizedAccessRoute> route = ClassifyOperationRoute(call, results);
    if (!route.has_value()) {
      for (const auto& [interval, var] : reads) {
        static_cast<void>(var);
        ++summaries_[interval].unsupported_accesses;
      }
      for (const auto& [interval, output] : writes) {
        static_cast<void>(output);
        ++summaries_[interval].unsupported_accesses;
      }
      return;
    }

    AccessEndpoint read_endpoint;
    read_endpoint.region = current_region_;
    read_endpoint.statement_index = current_statement_index_;
    read_endpoint.global_order = global_order_;
    read_endpoint.statement = statement;
    read_endpoint.route = *route;
    read_endpoint.access_kind = AccessKind::Read;
    read_endpoint.branch_path = branch_path_;
    read_endpoint.loop_stack = loop_stack_;
    AccessEndpoint write_endpoint = read_endpoint;
    write_endpoint.access_kind = AccessKind::Write;
    for (const auto& [interval, var] : reads) {
      RecordAccess(interval, read_endpoint, var);
    }
    for (const auto& [interval, output] : writes) RecordAccess(interval, write_endpoint, output);
  }
};

bool LifetimesPermitReuse(const LifetimeInterval& first, const LifetimeInterval& second) {
  return first.last_use_point <= second.def_point || second.last_use_point <= first.def_point;
}

}  // namespace

std::string RecognizedMemoryClassToString(RecognizedMemoryClass memory_class) {
  switch (memory_class) {
    case RecognizedMemoryClass::External:
      return "external";
    case RecognizedMemoryClass::Ub:
      return "ub";
    case RecognizedMemoryClass::L1:
      return "l1";
    case RecognizedMemoryClass::L0:
      return "l0";
    case RecognizedMemoryClass::Scalar:
      return "scalar";
  }
  throw pypto::ValueError("Unknown DSA recognizer memory class");
}

std::string RecognizedAccessResourceToString(RecognizedAccessResource resource) {
  switch (resource) {
    case RecognizedAccessResource::InboundDma:
      return "inbound_dma";
    case RecognizedAccessResource::OutboundDma:
      return "outbound_dma";
    case RecognizedAccessResource::L0ToExternal:
      return "l0_to_external";
    case RecognizedAccessResource::L1ToL0:
      return "l1_to_l0";
    case RecognizedAccessResource::L0ToL1:
      return "l0_to_l1";
    case RecognizedAccessResource::UbToL1:
      return "ub_to_l1";
    case RecognizedAccessResource::L1ToUb:
      return "l1_to_ub";
    case RecognizedAccessResource::UbToL0:
      return "ub_to_l0";
    case RecognizedAccessResource::L0ToUb:
      return "l0_to_ub";
    case RecognizedAccessResource::VectorCompute:
      return "vector_compute";
    case RecognizedAccessResource::MatrixCompute:
      return "matrix_compute";
    case RecognizedAccessResource::ScalarAccess:
      return "scalar_access";
  }
  throw pypto::ValueError("Unknown DSA recognizer access resource");
}

std::string RecognizedAccessRouteToString(const RecognizedAccessRoute& route) {
  return RecognizedMemoryClassToString(route.source) + "->" +
         RecognizedMemoryClassToString(route.destination) + "@" +
         RecognizedAccessResourceToString(route.resource);
}

ReusePenaltyRecognition RecognizeReusePenaltyCandidates(const FunctionPtr& func,
                                                        const AllocationPlan& allocation_plan,
                                                        DsaReusePenaltyRecognizer recognizer) {
  ReusePenaltyRecognition result;
  if (recognizer == DsaReusePenaltyRecognizer::Disabled || !func) return result;

  std::unordered_map<const Var*, size_t> interval_by_base;
  for (size_t index = 0; index < allocation_plan.intervals.size(); ++index) {
    const auto tile = As<TileType>(allocation_plan.intervals[index].variable->GetType());
    if (!tile || !tile->memref_) continue;
    interval_by_base.emplace(GetDefinedMemRef(tile)->base_.get(), index);
  }
  TupleResultCollector tuple_result_collector;
  tuple_result_collector.VisitStmt(func->body_);
  AccessCollector collector(allocation_plan, std::move(interval_by_base), tuple_result_collector.Elements());
  collector.Collect(func->body_);
  const auto& summaries = collector.Summaries();
  result.supported_allocations = static_cast<size_t>(
      std::count_if(summaries.begin(), summaries.end(), [](const AllocationAccessSummary& summary) {
        return summary.unsupported_accesses == 0 && !summary.accesses.empty();
      }));
  result.partially_supported_allocations = static_cast<size_t>(
      std::count_if(summaries.begin(), summaries.end(), [](const AllocationAccessSummary& summary) {
        return summary.unsupported_accesses != 0 && !summary.accesses.empty();
      }));
  for (const AllocationAccessSummary& summary : summaries) {
    for (const AccessEndpoint& access : summary.accesses) result.observed_routes.push_back(access.route);
  }
  std::sort(result.observed_routes.begin(), result.observed_routes.end(),
            [](const RecognizedAccessRoute& lhs, const RecognizedAccessRoute& rhs) {
              return std::tie(lhs.source, lhs.destination, lhs.resource) <
                     std::tie(rhs.source, rhs.destination, rhs.resource);
            });
  result.observed_routes.erase(
      std::unique(result.observed_routes.begin(), result.observed_routes.end(),
                  [](const RecognizedAccessRoute& lhs, const RecognizedAccessRoute& rhs) {
                    return std::tie(lhs.source, lhs.destination, lhs.resource) ==
                           std::tie(rhs.source, rhs.destination, rhs.resource);
                  }),
      result.observed_routes.end());

  std::vector<std::vector<AccessEndpoint>> terminal_frontiers;
  std::vector<std::vector<AccessEndpoint>> initial_write_frontiers;
  terminal_frontiers.reserve(summaries.size());
  initial_write_frontiers.reserve(summaries.size());
  for (const AllocationAccessSummary& summary : summaries) {
    terminal_frontiers.push_back(BuildTerminalFrontier(summary));
    initial_write_frontiers.push_back(BuildInitialWriteFrontier(summary));
  }

  std::set<std::pair<size_t, size_t>> separated;
  for (const AllocationSeparation& separation : allocation_plan.separations) {
    separated.insert(std::minmax(separation.first, separation.second));
  }

  std::set<std::pair<size_t, size_t>> candidate_pairs;
  std::set<std::pair<size_t, size_t>> ordered_pairs;
  auto emit_handoff = [&](size_t prior, size_t next, const AccessEndpoint& terminal,
                          const AccessEndpoint& initial, bool require_adjacent,
                          std::optional<std::pair<size_t, size_t>> crossed_loop) {
    const bool loop_carried = crossed_loop.has_value();
    if (!ControlPathsCompatible(terminal, initial,
                                loop_carried ? std::optional<size_t>(crossed_loop->second) : std::nullopt)) {
      return;
    }
    if (!loop_carried && terminal.global_order > initial.global_order) return;
    if (loop_carried && terminal.global_order <= initial.global_order) {
      return;
    }
    if (require_adjacent &&
        (terminal.region != initial.region || terminal.statement_index + 1 != initial.statement_index)) {
      return;
    }

    const auto canonical_pair = std::minmax(prior, next);
    candidate_pairs.insert(canonical_pair);
    const bool same_operation = terminal.statement == initial.statement;
    const bool ordered = !same_operation && terminal.region == initial.region &&
                         collector.TransitivelyOrdered(terminal.statement, initial.statement);
    if (ordered) ordered_pairs.insert(canonical_pair);
    const RecognizedReuseHazard hazard = terminal.route.resource == initial.route.resource
                                             ? RecognizedReuseHazard::SameResource
                                             : RecognizedReuseHazard::CrossResource;
    const RecognizedReuseDependence dependence = terminal.access_kind == AccessKind::Read
                                                     ? RecognizedReuseDependence::WriteAfterRead
                                                     : RecognizedReuseDependence::WriteAfterWrite;
    const bool nested_control = !terminal.branch_path.empty() || !initial.branch_path.empty() ||
                                !terminal.loop_stack.empty() || !initial.loop_stack.empty();
    const bool partial_access = !terminal.full_allocation || !initial.full_allocation;
    const bool incomplete_access_set =
        summaries[prior].unsupported_accesses != 0 || summaries[next].unsupported_accesses != 0;
    result.candidates.push_back({canonical_pair.first,
                                 canonical_pair.second,
                                 prior,
                                 next,
                                 hazard,
                                 dependence,
                                 terminal.route,
                                 initial.route,
                                 terminal.memory_space,
                                 initial.memory_space,
                                 terminal.global_order,
                                 initial.global_order,
                                 terminal.byte_offset,
                                 terminal.byte_size,
                                 initial.byte_offset,
                                 initial.byte_size,
                                 loop_carried ? crossed_loop->first : std::numeric_limits<size_t>::max(),
                                 ordered,
                                 same_operation,
                                 partial_access,
                                 incomplete_access_set,
                                 nested_control,
                                 !terminal.loop_stack.empty() || !initial.loop_stack.empty(),
                                 loop_carried});
  };

  auto consider = [&](size_t first, size_t second, bool require_adjacent) {
    if (first == second || separated.count(std::minmax(first, second)) != 0) return;
    const LifetimeInterval& first_lifetime = allocation_plan.intervals[first];
    const LifetimeInterval& second_lifetime = allocation_plan.intervals[second];
    if (first_lifetime.memory_space != second_lifetime.memory_space ||
        !LifetimesPermitReuse(first_lifetime, second_lifetime)) {
      return;
    }

    size_t earlier = first;
    size_t later = second;
    if (second_lifetime.last_use_point <= first_lifetime.def_point) std::swap(earlier, later);
    for (const AccessEndpoint& terminal : terminal_frontiers[earlier]) {
      for (const AccessEndpoint& initial : initial_write_frontiers[later]) {
        emit_handoff(earlier, later, terminal, initial, require_adjacent, std::nullopt);
      }
    }

    if (require_adjacent) return;
    // A shared address inside a repeated loop also creates the cyclic handoff
    // from the later value in iteration k to the earlier value in iteration
    // k+1. This is the scratchpad analogue of SIRA's distance-one reuse edge.
    for (const AccessEndpoint& terminal : terminal_frontiers[later]) {
      for (const AccessEndpoint& initial : initial_write_frontiers[earlier]) {
        for (const auto& context : SharedLoopContexts(terminal, initial)) {
          emit_handoff(later, earlier, terminal, initial, false, context);
        }
      }
    }
  };

  INTERNAL_CHECK(recognizer == DsaReusePenaltyRecognizer::Quadratic)
      << "Internal error: unrecognized DSA reuse-penalty recognizer";
  // Explicitly approved research mode: compare all allocation pairs. This is
  // intentionally not the default compiler path.
  for (size_t first = 0; first < summaries.size(); ++first) {
    for (size_t second = first + 1; second < summaries.size(); ++second) {
      consider(first, second, false);
    }
  }

  result.candidate_pairs = candidate_pairs.size();
  result.already_ordered_pairs = ordered_pairs.size();

  std::sort(
      result.candidates.begin(), result.candidates.end(),
      [](const RecognizedReuseCandidate& lhs, const RecognizedReuseCandidate& rhs) {
        return std::tie(lhs.first_interval, lhs.second_interval, lhs.prior_interval, lhs.next_interval,
                        lhs.prior_route.source, lhs.prior_route.destination, lhs.prior_route.resource,
                        lhs.next_route.source, lhs.next_route.destination, lhs.next_route.resource,
                        lhs.prior_memory_space, lhs.next_memory_space, lhs.dependence,
                        lhs.ordered_by_logical_dag, lhs.prior_access_order, lhs.next_access_order,
                        lhs.prior_byte_offset, lhs.prior_byte_size, lhs.next_byte_offset, lhs.next_byte_size,
                        lhs.loop_id, lhs.requires_alias_contract, lhs.partial_access,
                        lhs.incomplete_access_set, lhs.nested_control, lhs.in_loop, lhs.loop_carried) <
               std::tie(rhs.first_interval, rhs.second_interval, rhs.prior_interval, rhs.next_interval,
                        rhs.prior_route.source, rhs.prior_route.destination, rhs.prior_route.resource,
                        rhs.next_route.source, rhs.next_route.destination, rhs.next_route.resource,
                        rhs.prior_memory_space, rhs.next_memory_space, rhs.dependence,
                        rhs.ordered_by_logical_dag, rhs.prior_access_order, rhs.next_access_order,
                        rhs.prior_byte_offset, rhs.prior_byte_size, rhs.next_byte_offset, rhs.next_byte_size,
                        rhs.loop_id, rhs.requires_alias_contract, rhs.partial_access,
                        rhs.incomplete_access_set, rhs.nested_control, rhs.in_loop, rhs.loop_carried);
      });
  result.candidates.erase(
      std::unique(
          result.candidates.begin(), result.candidates.end(),
          [](const RecognizedReuseCandidate& lhs, const RecognizedReuseCandidate& rhs) {
            return std::tie(lhs.first_interval, lhs.second_interval, lhs.prior_interval, lhs.next_interval,
                            lhs.prior_route.source, lhs.prior_route.destination, lhs.prior_route.resource,
                            lhs.next_route.source, lhs.next_route.destination, lhs.next_route.resource,
                            lhs.prior_memory_space, lhs.next_memory_space, lhs.dependence,
                            lhs.ordered_by_logical_dag, lhs.prior_access_order, lhs.next_access_order,
                            lhs.prior_byte_offset, lhs.prior_byte_size, lhs.next_byte_offset,
                            lhs.next_byte_size, lhs.loop_id, lhs.requires_alias_contract, lhs.partial_access,
                            lhs.incomplete_access_set, lhs.nested_control, lhs.in_loop, lhs.loop_carried) ==
                   std::tie(rhs.first_interval, rhs.second_interval, rhs.prior_interval, rhs.next_interval,
                            rhs.prior_route.source, rhs.prior_route.destination, rhs.prior_route.resource,
                            rhs.next_route.source, rhs.next_route.destination, rhs.next_route.resource,
                            rhs.prior_memory_space, rhs.next_memory_space, rhs.dependence,
                            rhs.ordered_by_logical_dag, rhs.prior_access_order, rhs.next_access_order,
                            rhs.prior_byte_offset, rhs.prior_byte_size, rhs.next_byte_offset,
                            rhs.next_byte_size, rhs.loop_id, rhs.requires_alias_contract, rhs.partial_access,
                            rhs.incomplete_access_set, rhs.nested_control, rhs.in_loop, rhs.loop_carried);
          }),
      result.candidates.end());
  for (const RecognizedReuseCandidate& candidate : result.candidates) {
    if (candidate.hazard == RecognizedReuseHazard::CrossResource) {
      ++result.cross_resource_candidates;
    } else {
      ++result.same_resource_candidates;
    }
    if (candidate.dependence == RecognizedReuseDependence::WriteAfterRead) {
      ++result.write_after_read_candidates;
    } else {
      ++result.write_after_write_candidates;
    }
    if (candidate.ordered_by_logical_dag) ++result.ordered_evidence_candidates;
    if (candidate.requires_alias_contract) ++result.alias_contract_candidates;
    if (candidate.partial_access) ++result.partial_access_candidates;
    if (candidate.incomplete_access_set) ++result.incomplete_access_candidates;
    if (candidate.nested_control) ++result.nested_control_candidates;
    if (candidate.in_loop) ++result.in_loop_candidates;
    if (candidate.loop_carried) ++result.loop_carried_candidates;
  }
  return result;
}

void ApplyExperimentalUnitPenaltyPolicy(ReusePenaltyRecognition* recognition) {
  if (recognition == nullptr) return;
  recognition->penalties.clear();
  std::set<std::pair<size_t, size_t>> promoted;
  for (const RecognizedReuseCandidate& candidate : recognition->candidates) {
    if (candidate.hazard != RecognizedReuseHazard::CrossResource || candidate.ordered_by_logical_dag ||
        candidate.requires_alias_contract || candidate.partial_access || candidate.incomplete_access_set) {
      continue;
    }
    if (candidate.nested_control || candidate.loop_carried) continue;
    if (!promoted.emplace(candidate.first_interval, candidate.second_interval).second) continue;
    recognition->penalties.push_back(
        {candidate.first_interval, candidate.second_interval, 1, candidate.hazard});
  }
}

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto
