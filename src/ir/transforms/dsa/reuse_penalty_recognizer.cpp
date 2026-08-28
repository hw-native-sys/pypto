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
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {
namespace {

// Complexity is O(N log N + E), where N is IR/allocation input size and E is
// the number of cross-pipe candidate relations examined. Access collection
// and summary construction are linear. Structured control paths receive
// constant-size IDs during traversal, so sorting lifetimes and path-bucket
// maintenance cost O(N log N). Candidate enumeration visits only
// lifetime-compatible allocations in a different backend pipe bucket. E can be
// quadratic in the number of reusable allocations. This output-sensitive
// exception is inherent in the opt-in explicit pairwise DSA-RP model: a kernel
// can genuinely produce Theta(B^2) penalty edges for B buffers. No access-pair
// antichain or per-pair dependency walk is performed beyond materializing those
// relations.

enum class AccessKind : uint8_t {
  Read,
  Write,
};

constexpr size_t kResourceCount = static_cast<size_t>(PipeType::ALL) + 1;

size_t ResourceIndex(PipeType resource) { return static_cast<size_t>(resource); }

struct StructuredPath {
  size_t id = 0;
  bool in_loop = false;

  bool operator==(const StructuredPath& other) const { return id == other.id; }

  bool operator!=(const StructuredPath& other) const { return !(*this == other); }

  bool operator<(const StructuredPath& other) const { return id < other.id; }
};

struct AccessEndpoint {
  size_t global_order = 0;
  PipeType resource = PipeType::S;
  AccessKind access_kind = AccessKind::Read;
  StructuredPath path;
  bool full_allocation = false;
};

struct AllocationAccessSummary {
  std::vector<AccessEndpoint> accesses;
  bool has_unknown_access = false;
};

struct CompactAccessSummary {
  StructuredPath path;
  AccessEndpoint initial_write;
  std::array<std::optional<AccessEndpoint>, kResourceCount> terminal_by_resource;
};

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

std::optional<MemorySpace> GetMemorySpace(const TypePtr& type) {
  if (!type) return std::nullopt;
  const auto shaped = As<ShapedType>(type);
  return shaped ? shaped->GetMemorySpace() : std::nullopt;
}

std::optional<MemorySpace> GetMemorySpace(const VarPtr& var) {
  return var ? GetMemorySpace(var->GetType()) : std::nullopt;
}

bool SameAllocation(const VarPtr& first, const VarPtr& second) {
  const auto first_tile = first ? As<TileType>(first->GetType()) : nullptr;
  const auto second_tile = second ? As<TileType>(second->GetType()) : nullptr;
  if (!first_tile || !second_tile || !first_tile->memref_ || !second_tile->memref_) return false;
  return GetDefinedMemRef(first_tile)->base_.get() == GetDefinedMemRef(second_tile)->base_.get();
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
                  TupleResultElements tuple_results, const backend::Backend& backend)
      : plan_(plan),
        interval_by_base_(std::move(interval_by_base)),
        tuple_results_(std::move(tuple_results)),
        backend_(backend),
        summaries_(plan.intervals.size()) {}

  void Collect(const StmtPtr& body) { VisitStmt(body); }

  const std::vector<AllocationAccessSummary>& Summaries() const { return summaries_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    RecordCall(As<Call>(op->value_), op->var_);
    ++global_order_;
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    RecordCall(As<Call>(op->expr_), nullptr);
    ++global_order_;
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    for (const ExprPtr& value : op->value_) {
      RecordCall(As<Call>(value), nullptr);
    }
    ++global_order_;
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    const StructuredPath parent = current_path_;
    current_path_ = {next_path_id_++, parent.in_loop};
    VisitStmt(op->then_body_);
    current_path_ = parent;
    if (op->else_body_) {
      current_path_ = {next_path_id_++, parent.in_loop};
      VisitStmt(*op->else_body_);
      current_path_ = parent;
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    const StructuredPath parent = current_path_;
    current_path_ = {next_path_id_++, true};
    VisitStmt(op->body_);
    current_path_ = parent;
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    const StructuredPath parent = current_path_;
    current_path_ = {next_path_id_++, true};
    VisitStmt(op->body_);
    current_path_ = parent;
  }

 private:
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

  bool IsFullAllocationAccess(const VarPtr& var, size_t interval) const {
    const auto tile = As<TileType>(var->GetType());
    if (!tile || !tile->memref_) return false;
    const MemRefPtr memref = GetDefinedMemRef(tile);
    const auto offset = As<ConstInt>(memref->byte_offset_);
    return offset && offset->value_ == 0 &&
           memref->size_ == static_cast<uint64_t>(plan_.intervals[interval].size) &&
           AreExprVectorsEqual(GetValidShape(tile), tile->shape_);
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
    // These two cases are promoted locally only when their operands prove an
    // exact whole-window access.
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
    for (size_t interval : touched) summaries_[interval].has_unknown_access = true;
  }

  void RecordAccess(size_t interval, AccessEndpoint endpoint, const VarPtr& var) {
    const auto memory_space = GetMemorySpace(var);
    if (!memory_space) {
      summaries_[interval].has_unknown_access = true;
      return;
    }
    endpoint.full_allocation = IsFullAllocationAccess(var, interval);
    summaries_[interval].accesses.push_back(endpoint);
  }

  void RecordCall(const CallPtr& call, const VarPtr& result) {
    if (!call || !call->op_) return;

    const std::vector<VarPtr> results = ResolveCallResults(result);
    const bool whole_assemble = IsProvablyWholeAssemble(call, results);
    std::vector<std::pair<size_t, VarPtr>> reads;
    std::vector<std::pair<size_t, VarPtr>> writes;
    const auto& registry = OpRegistry::GetInstance();
    const OpRegistryEntry* entry =
        registry.IsRegistered(call->op_->name_) ? &registry.GetEntry(call->op_->name_) : nullptr;
    for (size_t argument_index = 0; argument_index < call->args_.size(); ++argument_index) {
      // A whole-window assemble overwrites the target. Its old contents are
      // not an input access; partial assemble remains Unknown and is poisoned.
      const VarPtr var = AsVarLike(call->args_[argument_index]);
      const auto interval = FindInterval(var);
      if (!interval) continue;
      const ArgEffect effect = entry ? entry->GetArgEffect(argument_index, call->kwargs_) : ArgEffect::Read;
      if (ArgEffectReads(effect) && !(whole_assemble && argument_index == 0)) {
        reads.emplace_back(*interval, var);
      }
      if (ArgEffectWrites(effect)) writes.emplace_back(*interval, var);
    }
    for (const VarPtr& output : results) {
      if (const auto interval = FindInterval(output)) writes.emplace_back(*interval, output);
    }
    if (reads.empty() && writes.empty()) return;

    const ExecutionMemoryAccessEvidence evidence = ResolveAccessEvidence(call, results);
    if (evidence == ExecutionMemoryAccessEvidence::NoAccess) return;
    if (evidence == ExecutionMemoryAccessEvidence::Unknown) {
      Poison(reads, writes);
      return;
    }

    const std::optional<PipeType> resource = backend_.TryInferPipe(call);
    if (!resource) {
      Poison(reads, writes);
      return;
    }

    AccessEndpoint read_endpoint;
    read_endpoint.global_order = global_order_;
    read_endpoint.resource = *resource;
    read_endpoint.access_kind = AccessKind::Read;
    read_endpoint.path = current_path_;
    AccessEndpoint write_endpoint = read_endpoint;
    write_endpoint.access_kind = AccessKind::Write;
    for (const auto& [interval, var] : reads) RecordAccess(interval, read_endpoint, var);
    for (const auto& [interval, output] : writes) RecordAccess(interval, write_endpoint, output);
  }

  const AllocationPlan& plan_;
  std::unordered_map<const Var*, size_t> interval_by_base_;
  TupleResultElements tuple_results_;
  const backend::Backend& backend_;
  std::vector<AllocationAccessSummary> summaries_;
  size_t global_order_ = 0;
  size_t next_path_id_ = 1;
  StructuredPath current_path_;
};

std::optional<CompactAccessSummary> CompactSummary(const AllocationAccessSummary& summary) {
  if (summary.has_unknown_access || summary.accesses.empty()) return std::nullopt;

  const StructuredPath& path = summary.accesses.front().path;
  if (std::any_of(summary.accesses.begin(), summary.accesses.end(), [&](const AccessEndpoint& endpoint) {
        return endpoint.path != path || !endpoint.full_allocation;
      })) {
    return std::nullopt;
  }

  const auto earliest = std::min_element(summary.accesses.begin(), summary.accesses.end(),
                                         [](const AccessEndpoint& lhs, const AccessEndpoint& rhs) {
                                           return lhs.global_order < rhs.global_order;
                                         });
  INTERNAL_CHECK(earliest != summary.accesses.end()) << "Non-empty access summary has no first access";
  const size_t earliest_order = earliest->global_order;
  std::optional<AccessEndpoint> initial_write;
  for (const AccessEndpoint& endpoint : summary.accesses) {
    if (endpoint.global_order != earliest_order) continue;
    if (endpoint.access_kind != AccessKind::Write) return std::nullopt;
    if (initial_write && initial_write->resource != endpoint.resource) return std::nullopt;
    initial_write = endpoint;
  }
  if (!initial_write) return std::nullopt;

  CompactAccessSummary compact;
  compact.path = path;
  compact.initial_write = *initial_write;
  for (const AccessEndpoint& endpoint : summary.accesses) {
    std::optional<AccessEndpoint>& terminal = compact.terminal_by_resource[ResourceIndex(endpoint.resource)];
    if (!terminal || terminal->global_order < endpoint.global_order ||
        (terminal->global_order == endpoint.global_order && terminal->access_kind == AccessKind::Read &&
         endpoint.access_kind == AccessKind::Write)) {
      terminal = endpoint;
    }
  }
  return compact;
}

struct ResourceBuckets {
  std::array<std::vector<size_t>, kResourceCount> terminal;
  std::array<std::vector<size_t>, kResourceCount> initial;
};

}  // namespace

std::vector<RecognizedReusePenalty> RecognizeReusePenalties(const FunctionPtr& func,
                                                            const AllocationPlan& allocation_plan,
                                                            const backend::Backend& backend) {
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
                            backend);
  collector.Collect(func->body_);

  std::vector<std::optional<CompactAccessSummary>> compact_summaries;
  compact_summaries.reserve(collector.Summaries().size());
  for (const AllocationAccessSummary& summary : collector.Summaries()) {
    compact_summaries.push_back(CompactSummary(summary));
  }

  std::unordered_set<PairKey, PairKeyHash> separated;
  separated.reserve(allocation_plan.separations.size());
  for (const AllocationSeparation& separation : allocation_plan.separations) {
    separated.insert(NormalizePair(separation.first, separation.second));
  }

  std::vector<RecognizedReusePenalty> penalties;
  std::unordered_set<PairKey, PairKeyHash> recognized;
  auto emit = [&](size_t first, size_t second) {
    if (first == second) return;
    const PairKey pair = NormalizePair(first, second);
    if (separated.count(pair) != 0 || !recognized.insert(pair).second) return;
    penalties.push_back({pair.first, pair.second, 1});
  };

  std::map<MemorySpace, std::vector<size_t>> intervals_by_space;
  for (size_t index = 0; index < allocation_plan.intervals.size(); ++index) {
    intervals_by_space[allocation_plan.intervals[index].memory_space].push_back(index);
  }

  for (auto& [space, indices] : intervals_by_space) {
    static_cast<void>(space);
    std::vector<size_t> by_definition = indices;
    std::vector<size_t> by_end = std::move(indices);
    const auto def_key = [&](size_t index) {
      return std::pair{allocation_plan.intervals[index].def_point, index};
    };
    const auto end_key = [&](size_t index) {
      return std::pair{allocation_plan.intervals[index].last_use_point, index};
    };
    std::sort(by_definition.begin(), by_definition.end(),
              [&](size_t lhs, size_t rhs) { return def_key(lhs) < def_key(rhs); });
    std::sort(by_end.begin(), by_end.end(),
              [&](size_t lhs, size_t rhs) { return end_key(lhs) < end_key(rhs); });

    size_t end_cursor = 0;
    std::map<StructuredPath, ResourceBuckets> buckets_by_path;
    for (size_t current : by_definition) {
      const int current_definition = allocation_plan.intervals[current].def_point;
      while (end_cursor < by_end.size() &&
             allocation_plan.intervals[by_end[end_cursor]].last_use_point <= current_definition) {
        const size_t reusable = by_end[end_cursor++];
        if (!compact_summaries[reusable]) continue;
        const CompactAccessSummary& summary = *compact_summaries[reusable];
        ResourceBuckets& buckets = buckets_by_path[summary.path];
        buckets.initial[ResourceIndex(summary.initial_write.resource)].push_back(reusable);
        for (size_t resource = 0; resource < kResourceCount; ++resource) {
          if (summary.terminal_by_resource[resource]) buckets.terminal[resource].push_back(reusable);
        }
      }

      if (!compact_summaries[current]) continue;
      const CompactAccessSummary& current_summary = *compact_summaries[current];
      const auto path_buckets = buckets_by_path.find(current_summary.path);
      if (path_buckets == buckets_by_path.end()) continue;
      const ResourceBuckets& buckets = path_buckets->second;

      const size_t initial_resource = ResourceIndex(current_summary.initial_write.resource);
      for (size_t terminal_resource = 0; terminal_resource < kResourceCount; ++terminal_resource) {
        if (terminal_resource == initial_resource) continue;
        for (size_t prior : buckets.terminal[terminal_resource]) {
          if (prior == current || !compact_summaries[prior]) continue;
          const AccessEndpoint& terminal = *compact_summaries[prior]->terminal_by_resource[terminal_resource];
          if (terminal.global_order <= current_summary.initial_write.global_order) emit(prior, current);
        }
      }

      // A later value in iteration k can hand storage back to an earlier value
      // in iteration k+1. Exact structured-path equality is a conservative,
      // deterministic proof that both endpoints cross the same loop backedge.
      if (!current_summary.path.in_loop) continue;
      for (size_t terminal_resource = 0; terminal_resource < kResourceCount; ++terminal_resource) {
        const std::optional<AccessEndpoint>& terminal =
            current_summary.terminal_by_resource[terminal_resource];
        if (!terminal) continue;
        for (size_t prior_initial_resource = 0; prior_initial_resource < kResourceCount;
             ++prior_initial_resource) {
          if (terminal_resource == prior_initial_resource) continue;
          for (size_t prior : buckets.initial[prior_initial_resource]) {
            if (prior == current || !compact_summaries[prior]) continue;
            const AccessEndpoint& initial = compact_summaries[prior]->initial_write;
            if (terminal->global_order > initial.global_order) emit(prior, current);
          }
        }
      }
    }
  }
  return penalties;
}

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto
