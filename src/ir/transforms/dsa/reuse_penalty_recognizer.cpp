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
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
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

struct AccessEndpoint {
  size_t region = 0;
  size_t statement_index = 0;
  size_t global_order = 0;
  const Stmt* statement = nullptr;
  PipeType pipe = PipeType::ALL;
  AccessKind access_kind = AccessKind::Read;
  bool nested_control = false;
};

struct AllocationAccessSummary {
  bool supported = true;
  std::optional<AccessEndpoint> first_write;
  std::optional<AccessEndpoint> last_access;
};

bool IsVectorOperation(const CallPtr& call) {
  return IsOp(call, "tile.add") || IsOp(call, "tile.sub") || IsOp(call, "tile.mul") ||
         IsOp(call, "tile.div") || IsOp(call, "tile.exp") || IsOp(call, "tile.adds") ||
         IsOp(call, "tile.subs") || IsOp(call, "tile.muls") || IsOp(call, "tile.divs") ||
         IsOp(call, "tile.cast") || IsOp(call, "tile.row_expand_add") || IsOp(call, "tile.row_expand_sub") ||
         IsOp(call, "tile.row_expand_mul") || IsOp(call, "tile.row_expand_div") || IsOp(call, "tile.full");
}

bool IsMatrixOperation(const CallPtr& call) {
  return IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_acc") || IsOp(call, "tile.matmul_bias") ||
         IsOp(call, "tile.gemv") || IsOp(call, "tile.gemv_acc") || IsOp(call, "tile.gemv_bias");
}

std::optional<MemorySpace> GetMemorySpace(const VarPtr& var) {
  if (!var) return std::nullopt;
  const auto tile = As<TileType>(var->GetType());
  return tile ? tile->GetMemorySpace() : std::nullopt;
}

std::optional<PipeType> ClassifyOperation(const CallPtr& call, const VarPtr& result) {
  if (!call) return std::nullopt;
  if (IsVectorOperation(call)) {
    bool saw_tile = false;
    for (const ExprPtr& argument : call->args_) {
      const auto space = GetMemorySpace(AsVarLike(argument));
      if (!space) continue;
      saw_tile = true;
      if (*space != MemorySpace::Vec) return std::nullopt;
    }
    const auto result_space = GetMemorySpace(result);
    if (result_space) {
      saw_tile = true;
      if (*result_space != MemorySpace::Vec) return std::nullopt;
    }
    return saw_tile ? std::optional<PipeType>(PipeType::V) : std::nullopt;
  }
  if (IsMatrixOperation(call)) {
    const auto result_space = GetMemorySpace(result);
    return result_space && *result_space == MemorySpace::Acc ? std::optional<PipeType>(PipeType::M)
                                                             : std::nullopt;
  }

  if (IsOp(call, "tile.load") || IsOp(call, "tile.read")) {
    const auto destination = GetMemorySpace(result);
    if (!destination) return std::nullopt;
    if (*destination == MemorySpace::Vec || *destination == MemorySpace::Mat) return PipeType::MTE2;
    if (*destination == MemorySpace::Left || *destination == MemorySpace::Right) return PipeType::MTE1;
    return std::nullopt;
  }

  if (IsOp(call, "tile.store") || IsOp(call, "tile.write")) {
    for (const ExprPtr& argument : call->args_) {
      const auto source = GetMemorySpace(AsVarLike(argument));
      if (!source) continue;
      if (*source == MemorySpace::Vec) return PipeType::MTE3;
      if (*source == MemorySpace::Acc) return PipeType::FIX;
    }
    return std::nullopt;
  }

  if (IsOp(call, "tile.move")) {
    if (call->args_.empty()) return std::nullopt;
    const auto source = GetMemorySpace(AsVarLike(call->args_.front()));
    const auto destination = GetMemorySpace(result);
    if (!source || !destination) return std::nullopt;
    if (*source == MemorySpace::Mat &&
        (*destination == MemorySpace::Left || *destination == MemorySpace::Right)) {
      return PipeType::MTE1;
    }
    if (*source == MemorySpace::Acc) return PipeType::FIX;
    if (*source == MemorySpace::Vec && *destination == MemorySpace::Vec) return PipeType::V;
  }
  return std::nullopt;
}

class AccessCollector : public IRVisitor {
 public:
  AccessCollector(const AllocationPlan& plan, std::unordered_map<const Var*, size_t> interval_by_base,
                  bool allow_nested_control)
      : plan_(plan),
        interval_by_base_(std::move(interval_by_base)),
        summaries_(plan.intervals.size()),
        allow_nested_control_(allow_nested_control) {}

  void Collect(const StmtPtr& body) { VisitStmt(body); }

  const std::vector<AllocationAccessSummary>& Summaries() const { return summaries_; }

  bool DirectlyOrdered(const Stmt* earlier, const Stmt* later) const {
    const auto found = predecessors_.find(later);
    return found != predecessors_.end() && found->second.count(earlier) != 0;
  }

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
    current_region_supported_ =
        (allow_nested_control_ || control_depth_ == 0) &&
        std::all_of(op->stmts_.begin(), op->stmts_.end(), [](const StmtPtr& stmt) {
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

  void VisitStmt_(const IfStmtPtr& op) override {
    ++control_depth_;
    IRVisitor::VisitStmt_(op);
    --control_depth_;
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    ++control_depth_;
    IRVisitor::VisitStmt_(op);
    --control_depth_;
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    ++control_depth_;
    IRVisitor::VisitStmt_(op);
    --control_depth_;
  }

 private:
  const AllocationPlan& plan_;
  std::unordered_map<const Var*, size_t> interval_by_base_;
  std::vector<AllocationAccessSummary> summaries_;
  std::unordered_map<const Stmt*, std::unordered_set<const Stmt*>> predecessors_;
  std::unordered_map<const Stmt*, std::unordered_set<const Stmt*>> transitive_predecessors_;
  size_t next_region_ = 0;
  size_t current_region_ = 0;
  size_t current_statement_index_ = 0;
  size_t global_order_ = 0;
  bool current_region_supported_ = false;
  size_t control_depth_ = 0;
  bool allow_nested_control_ = false;

  std::optional<size_t> FindInterval(const VarPtr& var) const {
    if (!var) return std::nullopt;
    const auto tile = As<TileType>(var->GetType());
    if (!tile || !tile->memref_) return std::nullopt;
    const MemRefPtr memref = GetDefinedMemRef(tile);
    const auto found = interval_by_base_.find(memref->base_.get());
    return found == interval_by_base_.end() ? std::nullopt : std::optional<size_t>(found->second);
  }

  bool IsFullAllocationAccess(const VarPtr& var, size_t interval) const {
    if (plan_.intervals[interval].alias_members.size() != 1) return false;
    const auto tile = As<TileType>(var->GetType());
    if (!tile || !tile->memref_) return false;
    const MemRefPtr memref = GetDefinedMemRef(tile);
    const auto offset = As<ConstInt>(memref->byte_offset_);
    return memref->size_ == plan_.intervals[interval].size && offset && offset->value_ == 0;
  }

  void RecordAccess(size_t interval, const AccessEndpoint& endpoint) {
    AllocationAccessSummary& summary = summaries_[interval];
    const bool is_write = endpoint.access_kind == AccessKind::Write;
    if (is_write && (!summary.first_write || endpoint.global_order < summary.first_write->global_order)) {
      summary.first_write = endpoint;
    }
    if (!summary.last_access || endpoint.global_order >= summary.last_access->global_order) {
      summary.last_access = endpoint;
    }
  }

  void RecordCall(const CallPtr& call, const VarPtr& result, const Stmt* statement) {
    std::vector<std::pair<size_t, VarPtr>> reads;
    if (call) {
      for (const ExprPtr& argument : call->args_) {
        const VarPtr var = AsVarLike(argument);
        const auto interval = FindInterval(var);
        if (interval) reads.emplace_back(*interval, var);
      }
    }
    const auto result_interval = FindInterval(result);
    if (reads.empty() && !result_interval) return;

    const std::optional<PipeType> pipe = ClassifyOperation(call, result);
    bool supported = current_region_supported_ && pipe.has_value();
    for (const auto& [interval, var] : reads) {
      supported = supported && IsFullAllocationAccess(var, interval);
    }
    if (result_interval) supported = supported && IsFullAllocationAccess(result, *result_interval);

    if (!supported) {
      for (const auto& [interval, var] : reads) {
        static_cast<void>(var);
        summaries_[interval].supported = false;
      }
      if (result_interval) summaries_[*result_interval].supported = false;
      return;
    }

    const AccessEndpoint read_endpoint{
        current_region_, current_statement_index_, global_order_,      statement,
        *pipe,           AccessKind::Read,         control_depth_ != 0};
    const AccessEndpoint write_endpoint{
        current_region_, current_statement_index_, global_order_,      statement,
        *pipe,           AccessKind::Write,        control_depth_ != 0};
    std::set<size_t> recorded_reads;
    for (const auto& [interval, var] : reads) {
      static_cast<void>(var);
      if (recorded_reads.insert(interval).second) RecordAccess(interval, read_endpoint);
    }
    if (result_interval) RecordAccess(*result_interval, write_endpoint);
  }
};

bool LifetimesPermitReuse(const LifetimeInterval& first, const LifetimeInterval& second) {
  return first.last_use_point <= second.def_point || second.last_use_point <= first.def_point;
}

}  // namespace

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
  AccessCollector collector(allocation_plan, std::move(interval_by_base),
                            recognizer == DsaReusePenaltyRecognizer::Quadratic);
  collector.Collect(func->body_);
  const auto& summaries = collector.Summaries();
  result.supported_allocations = static_cast<size_t>(
      std::count_if(summaries.begin(), summaries.end(), [](const AllocationAccessSummary& summary) {
        return summary.supported && summary.first_write && summary.last_access;
      }));

  std::set<std::pair<size_t, size_t>> separated;
  for (const AllocationSeparation& separation : allocation_plan.separations) {
    separated.insert(std::minmax(separation.first, separation.second));
  }

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
    const AllocationAccessSummary& earlier_summary = summaries[earlier];
    const AllocationAccessSummary& later_summary = summaries[later];
    if (!earlier_summary.supported || !later_summary.supported || !earlier_summary.last_access ||
        !later_summary.first_write) {
      return;
    }
    const AccessEndpoint& terminal = *earlier_summary.last_access;
    const AccessEndpoint& initial = *later_summary.first_write;
    if (terminal.region != initial.region || terminal.statement_index >= initial.statement_index) return;
    if (require_adjacent && terminal.statement_index + 1 != initial.statement_index) return;

    ++result.candidate_pairs;
    const bool ordered = require_adjacent
                             ? collector.DirectlyOrdered(terminal.statement, initial.statement)
                             : collector.TransitivelyOrdered(terminal.statement, initial.statement);
    if (ordered) {
      ++result.already_ordered_pairs;
      return;
    }
    const RecognizedReuseHazard hazard =
        terminal.pipe == initial.pipe ? RecognizedReuseHazard::SamePipe : RecognizedReuseHazard::CrossPipe;
    const RecognizedReuseDependence dependence = terminal.access_kind == AccessKind::Read
                                                     ? RecognizedReuseDependence::WriteAfterRead
                                                     : RecognizedReuseDependence::WriteAfterWrite;
    const RecognizedReuseCandidate candidate{std::min(first, second), std::max(first, second), hazard,
                                             dependence, terminal.nested_control};
    result.candidates.push_back(candidate);
  };

  if (recognizer == DsaReusePenaltyRecognizer::Linear) {
    std::map<std::pair<size_t, size_t>, std::vector<size_t>> terminal_by_location;
    for (size_t index = 0; index < summaries.size(); ++index) {
      const auto& summary = summaries[index];
      if (!summary.supported || !summary.last_access) continue;
      terminal_by_location[{summary.last_access->region, summary.last_access->statement_index}].push_back(
          index);
    }
    for (size_t later = 0; later < summaries.size(); ++later) {
      const auto& summary = summaries[later];
      if (!summary.supported || !summary.first_write || summary.first_write->statement_index == 0) continue;
      const auto found =
          terminal_by_location.find({summary.first_write->region, summary.first_write->statement_index - 1});
      if (found == terminal_by_location.end()) continue;
      for (size_t earlier : found->second) consider(earlier, later, true);
    }
  } else {
    // Explicitly approved research mode: compare all allocation pairs. This is
    // intentionally not the default compiler path.
    for (size_t first = 0; first < summaries.size(); ++first) {
      for (size_t second = first + 1; second < summaries.size(); ++second) {
        consider(first, second, false);
      }
    }
  }

  std::sort(result.candidates.begin(), result.candidates.end(),
            [](const RecognizedReuseCandidate& lhs, const RecognizedReuseCandidate& rhs) {
              return std::tie(lhs.first_interval, lhs.second_interval) <
                     std::tie(rhs.first_interval, rhs.second_interval);
            });
  result.candidates.erase(
      std::unique(result.candidates.begin(), result.candidates.end(),
                  [](const RecognizedReuseCandidate& lhs, const RecognizedReuseCandidate& rhs) {
                    return lhs.first_interval == rhs.first_interval &&
                           lhs.second_interval == rhs.second_interval;
                  }),
      result.candidates.end());
  for (const RecognizedReuseCandidate& candidate : result.candidates) {
    if (candidate.hazard == RecognizedReuseHazard::CrossPipe) {
      ++result.cross_pipe_candidates;
    } else {
      ++result.same_pipe_candidates;
    }
    if (candidate.dependence == RecognizedReuseDependence::WriteAfterRead) {
      ++result.write_after_read_candidates;
    } else {
      ++result.write_after_write_candidates;
    }
    if (candidate.nested_control) ++result.nested_control_candidates;
  }
  return result;
}

void ApplyExperimentalUnitPenaltyPolicy(ReusePenaltyRecognition* recognition) {
  if (recognition == nullptr) return;
  recognition->penalties.clear();
  for (const RecognizedReuseCandidate& candidate : recognition->candidates) {
    if (candidate.hazard != RecognizedReuseHazard::CrossPipe || candidate.nested_control) continue;
    recognition->penalties.push_back(
        {candidate.first_interval, candidate.second_interval, 1, candidate.hazard});
  }
}

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto
