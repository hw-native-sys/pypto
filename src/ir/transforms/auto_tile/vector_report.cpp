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

#include "src/ir/transforms/auto_tile/vector_report.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {
namespace {

struct TileExtent {
  int64_t rows = 0;
  int64_t cols = 0;
};

struct ReportInput {
  size_t tensor = 0;
  size_t first_use = 0;
  size_t last_use = 0;
  size_t use_count = 0;
  TileExtent logical;
  TileExtent physical;
};

struct ReportPhase {
  VectorPhase phase = VectorPhase::Body;
  int64_t first_chunk = 0;
  int64_t trip_count = 0;
  int pipeline_stages = 1;
  TileExtent frame;
  TileExtent tail_frame;
  std::vector<size_t> ops;
  std::vector<ReportInput> inputs;
  std::vector<ReportInput> tail_inputs;
  std::string generated_algorithm;
};

struct VectorScheduleReport {
  const VectorGraph* graph = nullptr;
  const VectorSchedulePlan* plan = nullptr;
  std::array<ReportPhase, 4> phases;
};

const char* PhaseName(VectorPhase phase) {
  switch (phase) {
    case VectorPhase::Body:
      return "body";
    case VectorPhase::Stats:
      return "stats";
    case VectorPhase::Apply:
      return "apply";
    case VectorPhase::Finalize:
      return "finalize";
  }
  return "unknown";
}

int64_t AlignUp(int64_t value, int64_t granule) {
  INTERNAL_CHECK(value >= 0 && granule > 0) << "Internal error: invalid AutoTile report alignment";
  return ((value + granule - 1) / granule) * granule;
}

TileExtent PhaseFrame(const VectorGraph& graph, const VectorSchedulePlan& plan, VectorPhase phase,
                      int64_t reduced_extent) {
  if (phase == VectorPhase::Body) {
    if (plan.kind == VectorScheduleKind::Materialized) return {plan.tile_h, plan.tile_w};
    if (plan.kind == VectorScheduleKind::PointwiseStream) return {plan.strip_h, plan.strip_w};
    return {};
  }
  if (phase == VectorPhase::Finalize) reduced_extent = 1;
  if (reduced_extent <= 0 || graph.reduced_axis == 0) return {};
  return graph.reduced_axis == 1 ? TileExtent{plan.free_tile, reduced_extent}
                                 : TileExtent{reduced_extent, plan.free_tile};
}

TileExtent InputLogicalExtent(const VectorTensor& tensor, const TileExtent& frame) {
  return {tensor.rows == 1 ? 1 : frame.rows, tensor.cols == 1 ? 1 : frame.cols};
}

TileExtent InputPhysicalExtent(const VectorTensor& tensor, const TileExtent& frame, int64_t granule,
                               bool reduction_layout) {
  TileExtent logical = InputLogicalExtent(tensor, frame);
  return {tensor.rows == 1 ? 1 : (reduction_layout ? AlignUp(logical.rows, granule) : logical.rows),
          tensor.cols == 1 ? 1 : AlignUp(logical.cols, granule)};
}

std::vector<ReportInput> BuildInputs(const VectorGraph& graph, const VectorSchedulePlan& plan,
                                     const VectorPhasePlan& phase, const TileExtent& frame,
                                     bool reduction_layout) {
  std::vector<ReportInput> result;
  result.reserve(phase.inputs.size());
  for (const VectorInputLifetime& input : phase.inputs) {
    INTERNAL_CHECK(input.tensor < graph.tensors.size() && input.tensor < plan.tensor_element_granules.size())
        << "Internal error: AutoTile report input is outside the selected plan";
    const VectorTensor& tensor = graph.tensors[input.tensor];
    const int64_t granule = plan.tensor_element_granules[input.tensor];
    result.push_back({input.tensor, input.first_use, input.last_use, input.uses.size(),
                      InputLogicalExtent(tensor, frame),
                      InputPhysicalExtent(tensor, frame, granule, reduction_layout)});
  }
  return result;
}

VectorScheduleReport BuildReport(const VectorGraph& graph, const VectorSchedulePlan& plan) {
  VectorScheduleReport report{&graph, &plan, {}};
  for (size_t i = 0; i < report.phases.size(); ++i) {
    const VectorPhase phase = static_cast<VectorPhase>(i);
    const VectorPhasePlan& source = plan.phases[i];
    ReportPhase& target = report.phases[i];
    target.phase = phase;
    target.first_chunk = source.first_chunk;
    target.trip_count = source.trip_count;
    target.pipeline_stages = source.pipeline_stages;
    target.ops = source.ops;
    const int64_t full_extent =
        (phase == VectorPhase::Body || phase == VectorPhase::Finalize) ? 0 : plan.chunk;
    target.frame = PhaseFrame(graph, plan, phase, full_extent);
    if (plan.tail > 0 && (phase == VectorPhase::Stats || phase == VectorPhase::Apply)) {
      target.tail_frame = PhaseFrame(graph, plan, phase, plan.tail);
    }
    if (source.generated_algorithm.has_value())
      target.generated_algorithm = GeneratedAlgorithmName(*source.generated_algorithm);
    const bool reduction_layout = graph.reduced_axis != 0 && (phase != VectorPhase::Body ||
                                                              plan.kind == VectorScheduleKind::Materialized);
    target.inputs = BuildInputs(graph, plan, source, target.frame, reduction_layout);
    if (target.tail_frame.rows > 0 && target.tail_frame.cols > 0) {
      target.tail_inputs = BuildInputs(graph, plan, source, target.tail_frame, reduction_layout);
    }
  }
  return report;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ch) << std::dec
              << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  out << '"';
  return out.str();
}

std::string EncodeFilename(const std::string& value) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string result;
  for (unsigned char ch : value) {
    const bool ascii_alnum = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
    if (ascii_alnum || ch == '-' || ch == '_' || ch == '.') {
      result.push_back(static_cast<char>(ch));
    } else {
      result.push_back('%');
      result.push_back(kHex[ch >> 4]);
      result.push_back(kHex[ch & 0x0F]);
    }
  }
  return result.empty() ? "unnamed" : result;
}

void WriteExtentJson(std::ostringstream& out, const TileExtent& extent) {
  out << '[' << extent.rows << ',' << extent.cols << ']';
}

void WriteDouble(std::ostringstream& out, double value) {
  out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
}

std::string TensorName(const VectorTensor& tensor) { return auto_name::GetBaseName(tensor.var->name_hint_); }

std::string RenderJson(const VectorScheduleReport& report) {
  const VectorGraph& graph = *report.graph;
  const VectorSchedulePlan& plan = *report.plan;
  std::ostringstream out;
  out << "{\n";
  out << "  \"schema_version\":1,\n";
  out << "  \"function\":" << JsonEscape(graph.function->name_) << ",\n";
  out << "  \"backend\":\"Ascend910B\",\n";
  out << "  \"kernel_kind\":\"vector\",\n";
  out << "  \"schedule\":" << JsonEscape(ScheduleKindName(plan.kind)) << ",\n";
  out << "  \"grid\":{\"rows\":" << plan.m_partition.parts << ",\"cols\":" << plan.n_partition.parts
      << ",\"work_units\":" << plan.work_units << "},\n";
  out << "  \"partitions\":{\"rows\":{\"small\":" << plan.m_partition.small
      << ",\"big\":" << plan.m_partition.big << ",\"num_big\":" << plan.m_partition.num_big
      << "},\"cols\":{\"small\":" << plan.n_partition.small << ",\"big\":" << plan.n_partition.big
      << ",\"num_big\":" << plan.n_partition.num_big << "}},\n";
  out << "  \"region\":{\"logical_max\":[" << plan.tile_h << ',' << plan.tile_w << "],\"strip\":["
      << plan.strip_h << ',' << plan.strip_w << "],\"row_strips\":" << plan.row_strips
      << ",\"width_strips\":" << plan.width_strips << "},\n";
  out << "  \"reduction\":{\"axis\":" << graph.reduced_axis << ",\"free_tile\":" << plan.free_tile
      << ",\"free_tile_alloc\":" << plan.free_tile_alloc << ",\"extent\":" << plan.reduced_extent
      << ",\"chunk\":" << plan.chunk << ",\"full_chunks\":" << plan.full_chunks << ",\"tail\":" << plan.tail
      << "},\n";
  out << "  \"memory\":{\"dma_alignment_bytes\":" << plan.dma_alignment_bytes
      << ",\"full_peak_ub_bytes\":" << plan.full_peak_ub_bytes
      << ",\"stream_peak_ub_bytes\":" << plan.chunk_peak_ub_bytes << "},\n";
  out << "  \"cost\":{\"modeled_cycles\":";
  WriteDouble(out, plan.modeled_cycles);
  out << ",\"compute_cycles\":";
  WriteDouble(out, plan.modeled_compute_cycles);
  out << ",\"transfer_cycles\":";
  WriteDouble(out, plan.modeled_transfer_cycles);
  out << ",\"reduction_model\":" << JsonEscape(plan.used_reduction_fallback ? "legacy_fallback" : "grounded")
      << ",\"pointwise_model\":" << JsonEscape(PointwiseCostModelName(plan)) << "},\n";
  out << "  \"tensors\":[\n";
  for (size_t i = 0; i < graph.tensors.size(); ++i) {
    const VectorTensor& tensor = graph.tensors[i];
    out << "    {\"id\":" << i << ",\"name\":" << JsonEscape(TensorName(tensor)) << ",\"shape\":["
        << tensor.rows << ',' << tensor.cols << "],\"dtype\":" << JsonEscape(tensor.dtype.ToString())
        << ",\"element_bytes\":" << DTypeBytes(tensor.dtype)
        << ",\"physical_shape_class\":" << tensor.physical_shape_class
        << ",\"physical_element_granule\":" << plan.tensor_element_granules.at(i)
        << ",\"boundary_input\":" << (tensor.boundary_input ? "true" : "false")
        << ",\"required_output\":" << (tensor.required_output ? "true" : "false") << '}'
        << (i + 1 == graph.tensors.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"operations\":[\n";
  for (size_t i = 0; i < graph.ops.size(); ++i) {
    const VectorOp& op = graph.ops[i];
    out << "    {\"id\":" << i << ",\"operation\":" << JsonEscape(op.emission_op) << ",\"inputs\":[";
    for (size_t arg = 0; arg < op.inputs.size(); ++arg) {
      if (arg != 0) out << ',';
      out << op.inputs[arg];
    }
    out << "],\"output\":" << op.output << '}' << (i + 1 == graph.ops.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"phases\":[\n";
  for (size_t i = 0; i < report.phases.size(); ++i) {
    const ReportPhase& phase = report.phases[i];
    out << "    {\"name\":" << JsonEscape(PhaseName(phase.phase)) << ",\"first_chunk\":" << phase.first_chunk
        << ",\"trip_count\":" << phase.trip_count << ",\"pipeline_stages\":" << phase.pipeline_stages
        << ",\"frame\":";
    WriteExtentJson(out, phase.frame);
    out << ",\"tail_frame\":";
    WriteExtentJson(out, phase.tail_frame);
    out << ",\"generated_algorithm\":" << JsonEscape(phase.generated_algorithm) << ",\"operations\":[";
    for (size_t op = 0; op < phase.ops.size(); ++op) {
      if (op != 0) out << ',';
      out << phase.ops[op];
    }
    out << "],\"inputs\":[";
    for (size_t input = 0; input < phase.inputs.size(); ++input) {
      const ReportInput& value = phase.inputs[input];
      if (input != 0) out << ',';
      out << "{\"tensor\":" << value.tensor << ",\"first_use\":" << value.first_use
          << ",\"last_use\":" << value.last_use << ",\"use_count\":" << value.use_count
          << ",\"logical_tile\":";
      WriteExtentJson(out, value.logical);
      out << ",\"physical_tile\":";
      WriteExtentJson(out, value.physical);
      out << '}';
    }
    out << "],\"tail_inputs\":[";
    for (size_t input = 0; input < phase.tail_inputs.size(); ++input) {
      const ReportInput& value = phase.tail_inputs[input];
      if (input != 0) out << ',';
      out << "{\"tensor\":" << value.tensor << ",\"logical_tile\":";
      WriteExtentJson(out, value.logical);
      out << ",\"physical_tile\":";
      WriteExtentJson(out, value.physical);
      out << '}';
    }
    out << "],\"compute_cycles\":";
    WriteDouble(out, plan.modeled_phase_compute_cycles[i]);
    out << ",\"transfer_cycles\":";
    WriteDouble(out, plan.modeled_phase_transfer_cycles[i]);
    out << ",\"input_bytes\":";
    WriteDouble(out, plan.modeled_phase_input_bytes[i]);
    out << ",\"output_bytes\":";
    WriteDouble(out, plan.modeled_phase_output_bytes[i]);
    out << '}' << (i + 1 == report.phases.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

std::string TensorLabel(const VectorGraph& graph, size_t tensor) {
  INTERNAL_CHECK(tensor < graph.tensors.size()) << "Internal error: invalid AutoTile report tensor";
  return TensorName(graph.tensors[tensor]) + "(t" + std::to_string(tensor) + ")";
}

std::string OperationLine(const VectorGraph& graph, size_t op_index) {
  INTERNAL_CHECK(op_index < graph.ops.size()) << "Internal error: invalid AutoTile report operation";
  const VectorOp& op = graph.ops[op_index];
  std::ostringstream out;
  out << TensorLabel(graph, op.output) << " = " << op.emission_op << '(';
  for (size_t i = 0; i < op.inputs.size(); ++i) {
    if (i != 0) out << ", ";
    out << TensorLabel(graph, op.inputs[i]);
  }
  if (op.call->args_.size() > op.inputs.size()) {
    if (!op.inputs.empty()) out << ", ";
    out << "scalar_args...";
  }
  out << ')';
  return out.str();
}

void RenderPhaseInputs(std::ostringstream& out, const VectorScheduleReport& report, const ReportPhase& phase,
                       const std::string& indent, bool tail = false) {
  const VectorGraph& graph = *report.graph;
  const std::vector<ReportInput>& inputs = tail ? phase.tail_inputs : phase.inputs;
  for (const ReportInput& input : inputs) {
    out << indent << TensorLabel(graph, input.tensor) << " = GM.load(logical=[" << input.logical.rows << 'x'
        << input.logical.cols << "], physical=[" << input.physical.rows << 'x' << input.physical.cols
        << "])\n";
  }
}

void RenderPhaseOperations(std::ostringstream& out, const VectorScheduleReport& report,
                           const ReportPhase& phase, const std::string& indent) {
  const VectorGraph& graph = *report.graph;
  if (phase.generated_algorithm == "online_softmax_update") {
    out << indent << "local_max = row_max(input_tile)\n";
    out << indent << "next_max = maximum(running_max, local_max)\n";
    out << indent << "local_sum = row_sum(exp(input_tile - next_max))\n";
    out << indent << "running_sum = running_sum * exp(running_max - next_max) + local_sum\n";
    out << indent << "running_max = next_max\n";
    return;
  }
  std::vector<size_t> last_use(graph.tensors.size(), 0);
  std::vector<bool> used(graph.tensors.size(), false);
  for (size_t step = 0; step < phase.ops.size(); ++step) {
    const VectorOp& op = graph.ops.at(phase.ops[step]);
    for (size_t tensor : op.inputs) {
      last_use[tensor] = step;
      used[tensor] = true;
    }
  }
  for (size_t step = 0; step < phase.ops.size(); ++step) {
    const size_t op_index = phase.ops[step];
    const VectorOp& op = graph.ops.at(op_index);
    out << indent << OperationLine(graph, op_index) << '\n';
    std::vector<size_t> ending;
    for (size_t tensor : op.inputs) {
      if (used[tensor] && last_use[tensor] == step &&
          std::find(ending.begin(), ending.end(), tensor) == ending.end()) {
        ending.push_back(tensor);
      }
    }
    if (!ending.empty()) {
      out << indent << "lifetime ends: ";
      for (size_t i = 0; i < ending.size(); ++i) {
        if (i != 0) out << ", ";
        out << TensorLabel(graph, ending[i]);
      }
      out << '\n';
    }
  }
}

void RenderStores(std::ostringstream& out, const VectorScheduleReport& report, const std::string& indent) {
  const VectorGraph& graph = *report.graph;
  for (size_t tensor : graph.required_outputs) {
    out << indent << "GM.store(" << TensorLabel(graph, tensor) << ")\n";
  }
}

void RenderLoopPhase(std::ostringstream& out, const VectorScheduleReport& report, const ReportPhase& phase,
                     const std::string& label, bool stores) {
  if (phase.trip_count <= 0) return;
  out << "  " << label << ": for iter in range(" << phase.first_chunk << ", "
      << phase.first_chunk + phase.trip_count << ')';
  if (phase.pipeline_stages > 1) out << " pipeline(" << phase.pipeline_stages << ')';
  out << " frame=[" << phase.frame.rows << 'x' << phase.frame.cols << "]:\n";
  const std::string indent = "    ";
  if (phase.pipeline_stages > 1) out << indent << "slot = iter % " << phase.pipeline_stages << "\n";
  RenderPhaseInputs(out, report, phase, indent);
  RenderPhaseOperations(out, report, phase, indent);
  if (stores) RenderStores(out, report, indent);
}

std::string RenderPseudocode(const VectorScheduleReport& report) {
  const VectorGraph& graph = *report.graph;
  const VectorSchedulePlan& plan = *report.plan;
  std::ostringstream out;
  out << "AutoTile schedule for " << graph.function->name_ << "\n";
  out << "backend: Ascend910B vector (AIV)\n";
  out << "schedule: " << ScheduleKindName(plan.kind) << "\n";
  out << "grid: " << plan.m_partition.parts << 'x' << plan.n_partition.parts << " = " << plan.work_units
      << " work units\n";
  out << "representative logical region: [" << plan.tile_h << 'x' << plan.tile_w << "]\n";
  out << "UB peak: " << plan.chunk_peak_ub_bytes << " bytes streamed, " << plan.full_peak_ub_bytes
      << " bytes full-region\n";
  out << "modeled cost: ";
  WriteDouble(out, plan.modeled_cycles);
  out << " cycles (compute=";
  WriteDouble(out, plan.modeled_compute_cycles);
  out << ", transfer=";
  WriteDouble(out, plan.modeled_transfer_cycles);
  out << ")\n\n";
  out << "spmd " << plan.work_units << " work units:\n";
  out << "  region = balanced_partition(block_idx)\n";

  const ReportPhase& body = report.phases[PhaseIndex(VectorPhase::Body)];
  const ReportPhase& stats = report.phases[PhaseIndex(VectorPhase::Stats)];
  const ReportPhase& apply = report.phases[PhaseIndex(VectorPhase::Apply)];
  const ReportPhase& finalize = report.phases[PhaseIndex(VectorPhase::Finalize)];
  if (plan.kind == VectorScheduleKind::Materialized) {
    out << "  body: materialize logical=[" << body.frame.rows << 'x' << body.frame.cols << "]\n";
    RenderPhaseInputs(out, report, body, "    ");
    RenderPhaseOperations(out, report, body, "    ");
    RenderStores(out, report, "    ");
  } else if (plan.kind == VectorScheduleKind::PointwiseStream) {
    RenderLoopPhase(out, report, body, "body strips", true);
  } else {
    out << "  stats init: chunk 0, frame=[" << stats.frame.rows << 'x' << stats.frame.cols << "]\n";
    RenderPhaseInputs(out, report, stats, "    ");
    RenderPhaseOperations(out, report, stats, "    ");
    out << "    persistent state: ";
    if (plan.kind == VectorScheduleKind::Softmax) {
      out << "running_max, running_sum\n";
    } else {
      out << TensorLabel(graph, graph.ops.at(graph.reduction_op).output) << " accumulator\n";
    }
    RenderLoopPhase(out, report, stats, "stats rolled chunks", false);
    if (plan.tail > 0) {
      out << "  stats tail: serial frame=[" << stats.tail_frame.rows << 'x' << stats.tail_frame.cols << "]\n";
      RenderPhaseInputs(out, report, stats, "    ", true);
      RenderPhaseOperations(out, report, stats, "    ");
    }
    if (!finalize.ops.empty()) {
      out << "  finalize: serial\n";
      RenderPhaseOperations(out, report, finalize, "    ");
    }
    if (plan.kind == VectorScheduleKind::ReductionFolded) {
      RenderStores(out, report, "  ");
    } else {
      RenderLoopPhase(out, report, apply, "apply chunks", true);
      if (plan.tail > 0) {
        out << "  apply tail: serial frame=[" << apply.tail_frame.rows << 'x' << apply.tail_frame.cols
            << "]\n";
        RenderPhaseInputs(out, report, apply, "    ", true);
        RenderPhaseOperations(out, report, apply, "    ");
        RenderStores(out, report, "    ");
      }
    }
  }
  out << "\nphysical tensor granules (elements):\n";
  for (size_t tensor = 0; tensor < graph.tensors.size(); ++tensor) {
    out << "  " << TensorLabel(graph, tensor) << ": logical=[" << graph.tensors[tensor].rows << 'x'
        << graph.tensors[tensor].cols << "], dtype=" << graph.tensors[tensor].dtype.ToString()
        << ", element_bytes=" << DTypeBytes(graph.tensors[tensor].dtype)
        << ", granule=" << plan.tensor_element_granules.at(tensor) << '\n';
  }
  return out.str();
}

std::optional<std::filesystem::path> CurrentReportDirectory() {
  const PassContext* context = PassContext::Current();
  if (context == nullptr) return std::nullopt;
  for (const auto& instrument : context->GetInstruments()) {
    if (const auto* report = dynamic_cast<const ReportInstrument*>(instrument.get())) {
      return std::filesystem::path(report->GetOutputDir());
    }
  }
  return std::nullopt;
}

void WriteAtomically(const std::filesystem::path& path, const std::string& content, const Span& span) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    CHECK_SPAN(output.is_open(), span) << "AutoTile could not open schedule report " << temporary.string();
    output << content;
    output.close();
    CHECK_SPAN(output.good(), span) << "AutoTile could not write schedule report " << temporary.string();
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  CHECK_SPAN(!error, span) << "AutoTile could not publish schedule report " << path.string() << ": "
                           << error.message();
}

}  // namespace

std::optional<std::string> WriteVectorScheduleReport(const VectorGraph& graph,
                                                     const VectorSchedulePlan& plan) {
  const std::optional<std::filesystem::path> report_root = CurrentReportDirectory();
  if (!report_root.has_value()) return std::nullopt;
  INTERNAL_CHECK(graph.function != nullptr) << "Internal error: AutoTile report has no source function";
  const VectorScheduleReport report = BuildReport(graph, plan);
  const std::filesystem::path directory = *report_root / "auto_tile";
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  CHECK_SPAN(!error, graph.function->span_) << "AutoTile could not create schedule report directory "
                                            << directory.string() << ": " << error.message();
  const std::string stem = EncodeFilename(graph.function->name_);
  const std::filesystem::path json_path = directory / (stem + ".json");
  const std::filesystem::path text_path = directory / (stem + ".txt");
  WriteAtomically(json_path, RenderJson(report), graph.function->span_);
  WriteAtomically(text_path, RenderPseudocode(report), graph.function->span_);
  return text_path.string();
}

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto
