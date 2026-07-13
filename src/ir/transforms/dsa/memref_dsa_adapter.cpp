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

#include "pypto/ir/transforms/dsa/memref_dsa_adapter.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dsa/first_fit_solver.h"
#include "dsa/model.h"
#include "dsa/solver.h"
#include "dsa/structured_problem.h"
#include "dsa/validator.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace dsa_adapter {
namespace {

::dsa::PoolId ToPoolId(MemorySpace space) { return static_cast<::dsa::PoolId>(space); }

std::vector<::dsa::Interval> ConvertAndMergeIntervals(const LifetimeInterval& lifetime) {
  std::vector<VariableLifetime> source = lifetime.live_ranges;
  if (source.empty()) source.push_back({lifetime.def_point, lifetime.last_use_point});

  std::vector<::dsa::Interval> intervals;
  intervals.reserve(source.size());
  for (const VariableLifetime& range : source) {
    INTERNAL_CHECK(range.def_point >= 0 && range.last_use_point >= range.def_point)
        << "Invalid PyPTO variable lifetime [" << range.def_point << ", " << range.last_use_point << "]";

    // Split each statement point into a read sub-point (2*p) and a write
    // sub-point (2*p+1). An input last read at p ends exactly where an output
    // defined at p begins, preserving PyPTO's read-before-write reuse rule.
    const int64_t lower = 2 * static_cast<int64_t>(range.def_point) + 1;
    const int64_t last_read_end = 2 * static_cast<int64_t>(range.last_use_point) + 1;
    // A definition with no later use still occupies the write sub-point. All
    // other ranges end immediately after their final read sub-point.
    const int64_t upper = std::max(lower + 1, last_read_end);
    intervals.push_back({lower, upper});
  }

  std::sort(intervals.begin(), intervals.end(),
            [](const ::dsa::Interval& first, const ::dsa::Interval& second) {
              return first.lower != second.lower ? first.lower < second.lower : first.upper < second.upper;
            });
  std::vector<::dsa::Interval> merged;
  for (const ::dsa::Interval& interval : intervals) {
    if (merged.empty() || interval.lower > merged.back().upper) {
      merged.push_back(interval);
    } else {
      merged.back().upper = std::max(merged.back().upper, interval.upper);
    }
  }
  return merged;
}

std::string CorpusFileStem(const std::string& instance) {
  std::ostringstream output;
  output << "pypto_";
  for (const char raw_character : instance) {
    const auto character = static_cast<unsigned char>(raw_character);
    if (std::isalnum(character) != 0 || character == '-' || character == '_' || character == '.') {
      output << static_cast<char>(character);
    } else {
      output << '_' << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(character)
             << std::dec;
    }
  }
  if (instance.empty()) output << "unnamed";
  return output.str();
}

}  // namespace

ExportedProblem BuildStructuredProblem(const FunctionPtr& func, const AllocationPlan& allocation_plan,
                                       const MemoryAllocatorPolicy& policy,
                                       const std::unordered_map<MemorySpace, uint64_t>& reserved_end_by_space,
                                       const std::unordered_map<MemorySpace, uint64_t>& pool_caps) {
  INTERNAL_CHECK(func != nullptr) << "BuildStructuredProblem cannot analyze a null function";

  ExportedProblem exported;
  exported.document.profile = ::dsa::BenchmarkProfile::kPyptoStructured;
  exported.document.instance = func->name_;
  exported.document.metadata = {
      {"lifetime_ordering", "pypto_read_before_write"},
      {"memory_space_ids", "pypto_memory_space_enum_v1"},
      {"producer", "pypto"},
      {"solver_input", "pre_memory_reuse"},
  };
  exported.document.problem.pools.clear();
  exported.document.problem.objective = ::dsa::MinimizePeakObjective();

  std::map<MemorySpace, ::dsa::Pool> pools;
  std::vector<std::optional<::dsa::BufferId>> buffer_id_by_interval(allocation_plan.intervals.size());
  for (size_t index = 0; index < allocation_plan.intervals.size(); ++index) {
    const LifetimeInterval& lifetime = allocation_plan.intervals[index];
    if (lifetime.memory_space == MemorySpace::DDR || !policy.ShouldAllocate(lifetime.memory_space)) continue;
    INTERNAL_CHECK(index <= std::numeric_limits<::dsa::BufferId>::max())
        << "Too many PyPTO allocations for the standalone DSA BufferId type";

    const auto tile_type = As<TileType>(lifetime.variable->GetType());
    INTERNAL_CHECK_SPAN(tile_type != nullptr && tile_type->memref_.has_value(), lifetime.variable->span_)
        << "DSA export expected representative '" << lifetime.variable->name_hint_ << "' to carry a MemRef";
    const MemRefPtr memref = GetDefinedMemRef(tile_type);

    const auto id = static_cast<::dsa::BufferId>(index);
    ::dsa::Buffer buffer;
    buffer.id = id;
    buffer.name = memref->base_->name_hint_;
    buffer.size = lifetime.size;
    buffer.alignment = std::max<uint64_t>(1, policy.AlignAddress(1, lifetime.memory_space));
    buffer.live_intervals = ConvertAndMergeIntervals(lifetime);
    buffer.allowed_pools = {ToPoolId(lifetime.memory_space)};
    exported.document.problem.buffers.push_back(std::move(buffer));
    buffer_id_by_interval[index] = id;

    const auto insertion = exported.buffer_id_by_base.emplace(memref->base_.get(), id);
    INTERNAL_CHECK_SPAN(insertion.second, lifetime.variable->span_)
        << "DSA export produced duplicate allocation identity for base '" << memref->base_->name_hint_ << "'";

    ::dsa::Pool& pool = pools[lifetime.memory_space];
    pool.id = ToPoolId(lifetime.memory_space);
    pool.name = MemorySpaceToString(lifetime.memory_space);
    const auto cap = pool_caps.find(lifetime.memory_space);
    if (cap != pool_caps.end() && cap->second > 0) pool.capacity = cap->second;
    const auto reserved = reserved_end_by_space.find(lifetime.memory_space);
    if (reserved != reserved_end_by_space.end() && reserved->second > 0) {
      pool.reserved_ranges = {{0, reserved->second}};
    }
  }

  for (auto& [space, pool] : pools) {
    static_cast<void>(space);
    exported.document.problem.pools.push_back(std::move(pool));
  }

  std::set<std::pair<::dsa::BufferId, ::dsa::BufferId>> separations;
  for (const auto& [first_index, second_index] : allocation_plan.separations) {
    INTERNAL_CHECK(first_index < buffer_id_by_interval.size() && second_index < buffer_id_by_interval.size())
        << "DSA allocation separation references an out-of-range lifetime index";
    const auto& first_buffer = buffer_id_by_interval[first_index];
    if (!first_buffer.has_value()) continue;
    const auto& second_buffer = buffer_id_by_interval[second_index];
    if (!second_buffer.has_value()) continue;
    auto first = first_buffer.value();
    auto second = second_buffer.value();
    if (second < first) std::swap(first, second);
    if (first != second) separations.emplace(first, second);
  }
  for (const auto& [first, second] : separations) {
    exported.document.problem.separations.push_back({first, second});
  }

  return exported;
}

std::filesystem::path WriteProblemJson(const ExportedProblem& exported,
                                       const std::filesystem::path& directory) {
  CHECK(!directory.empty()) << "DSA export directory must not be empty";
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    throw pypto::RuntimeError("Failed to create DSA export directory '" + directory.string() +
                              "': " + error.message());
  }

  const std::filesystem::path output = directory / (CorpusFileStem(exported.document.instance) + ".dsa.json");
  try {
    ::dsa::WriteStructuredProblemJsonFile(output, exported.document);
  } catch (const std::exception& exception) {
    throw pypto::RuntimeError("Failed to export DSA problem to '" + output.string() +
                              "': " + exception.what());
  }
  return output;
}

SolverRun SolveWithFirstFit(const ExportedProblem& exported) {
  SolverRun run;
  run.problem_errors = ::dsa::ValidateStructuredProblemDocument(exported.document);
  if (!run.problem_errors.empty()) {
    run.result.status = ::dsa::SolveStatus::kInvalidProblem;
    run.result.diagnostics = run.problem_errors;
    return run;
  }

  ::dsa::FirstFitSolver solver;
  run.compatibility = ::dsa::CheckSolverCompatibility(exported.document.problem, solver.Capabilities());
  if (!run.compatibility.Compatible()) {
    run.result.status = ::dsa::SolveStatus::kUnsupported;
    run.result.diagnostics = run.compatibility.unsupported_features;
    run.result.diagnostics.insert(run.result.diagnostics.end(),
                                  run.compatibility.unsupported_objectives.begin(),
                                  run.compatibility.unsupported_objectives.end());
    return run;
  }

  run.result = solver.Solve(exported.document.problem);
  if (run.result.solution) {
    run.solution_errors = ::dsa::ValidateSolution(exported.document.problem, *run.result.solution);
  }
  return run;
}

std::vector<std::pair<const MemRef*, MemRefPtr>> BuildMemRefReplacements(
    const ExportedProblem& exported, const ::dsa::DsaSolution& solution,
    const std::vector<MemRefWithSpace>& memrefs, const MemoryAllocatorPolicy& policy) {
  std::vector<std::pair<const MemRef*, MemRefPtr>> replacements;
  replacements.reserve(memrefs.size());
  for (const auto& [old_memref, memory_space] : memrefs) {
    if (memory_space == MemorySpace::DDR || !policy.ShouldAllocate(memory_space)) continue;

    const auto buffer = exported.buffer_id_by_base.find(old_memref->base_.get());
    INTERNAL_CHECK_SPAN(buffer != exported.buffer_id_by_base.end(), old_memref->span_)
        << "DSA writeback could not find allocation base '" << old_memref->base_->name_hint_ << "'";
    const ::dsa::Placement* placement = solution.Find(buffer->second);
    INTERNAL_CHECK_SPAN(placement != nullptr, old_memref->span_)
        << "DSA writeback has no placement for buffer " << buffer->second;
    INTERNAL_CHECK_SPAN(placement->pool == ToPoolId(memory_space), old_memref->span_)
        << "DSA writeback changed fixed memory pool for buffer " << buffer->second;

    int64_t relative_offset = 0;
    if (const auto relative = As<ConstInt>(old_memref->byte_offset_)) relative_offset = relative->value_;
    INTERNAL_CHECK_SPAN(relative_offset >= 0, old_memref->span_)
        << "DSA writeback encountered a negative relative MemRef offset";
    const uint64_t relative = static_cast<uint64_t>(relative_offset);
    INTERNAL_CHECK_SPAN(
        placement->offset <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - relative,
        old_memref->span_)
        << "DSA writeback address exceeds PyPTO's signed INT64 address representation";

    auto address = std::make_shared<ConstInt>(static_cast<int64_t>(placement->offset + relative),
                                              DataType::INT64, Span::unknown());
    auto new_memref = std::make_shared<MemRef>(old_memref->name_hint_, old_memref->base_, std::move(address),
                                               old_memref->size_, old_memref->span_);
    replacements.emplace_back(old_memref.get(), std::move(new_memref));
  }

  std::sort(replacements.begin(), replacements.end(),
            [](const std::pair<const MemRef*, MemRefPtr>& first,
               const std::pair<const MemRef*, MemRefPtr>& second) {
              const auto first_offset = As<ConstInt>(first.second->byte_offset_);
              const auto second_offset = As<ConstInt>(second.second->byte_offset_);
              INTERNAL_CHECK(first_offset != nullptr && second_offset != nullptr)
                  << "DSA writeback produced a non-constant address";
              return first_offset->value_ != second_offset->value_
                         ? first_offset->value_ < second_offset->value_
                         : first.second->name_hint_ < second.second->name_hint_;
            });
  return replacements;
}

}  // namespace dsa_adapter
}  // namespace ir
}  // namespace pypto
