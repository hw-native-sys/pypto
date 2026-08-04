/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "src/ir/transforms/auto_tile/vector_emit.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {
namespace {

ExprPtr Index(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr IndexTuple(std::initializer_list<int64_t> values, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(values.size());
  for (int64_t value : values) elements.push_back(Index(value, span));
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

ExprPtr Pair(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{lhs, rhs}, span);
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

StmtPtr WrapSpmd(const VarPtr& index, std::vector<StmtPtr> body, int64_t work_units, const std::string& name,
                 const Span& span) {
  body.insert(body.begin(),
              std::make_shared<AssignStmt>(
                  index, OpRegistry::GetInstance().Create("tile.get_block_idx", {}, span), span));
  auto kernel = std::make_shared<InCoreScopeStmt>(SplitMode::None, name,
                                                  SeqStmts::Flatten(std::move(body), span), span);
  return std::make_shared<SpmdScopeStmt>(Index(work_units, span), false, name + "_spmd", kernel, span);
}

std::vector<std::pair<std::string, std::any>> WithoutAutoTile(
    const std::vector<std::pair<std::string, std::any>>& attrs) {
  std::vector<std::pair<std::string, std::any>> result;
  result.reserve(attrs.size());
  for (const auto& attr : attrs)
    if (attr.first != "auto_tile") result.push_back(attr);
  return result;
}

class VectorEmitter {
 public:
  VectorEmitter(const VectorGraph& graph, const VectorSchedulePlan& plan, bool lift_outputs)
      : graph_(graph), plan_(plan), lift_outputs_(lift_outputs), span_(graph.function->span_) {
    for (size_t op = 0; op < graph_.ops.size(); ++op) producer_.emplace(graph_.ops[op].output, op);
  }

  FunctionPtr Emit() {
    INTERNAL_CHECK_SPAN(plan_.feasible, span_) << "Internal error: AutoTile cannot emit an infeasible plan";
    ValidatePlan();
    CreateOutputs();

    std::vector<StmtPtr> body = std::move(prologue_);
    switch (plan_.kind) {
      case VectorScheduleKind::Materialized:
        EmitMaterialized(body);
        break;
      case VectorScheduleKind::PointwiseStream:
        EmitPointwiseStream(body);
        break;
      case VectorScheduleKind::ReductionFolded:
      case VectorScheduleKind::ReductionSpanning:
        EmitReduction(body);
        break;
      case VectorScheduleKind::Softmax:
        EmitSoftmax(body);
        break;
    }
    body.push_back(OriginalReturn());

    std::vector<VarPtr> params = graph_.function->params_;
    std::vector<ParamDirection> directions = graph_.function->param_directions_;
    if (lift_outputs_) {
      for (size_t tensor : graph_.required_outputs) {
        params.push_back(outputs_.at(tensor));
        directions.push_back(ParamDirection::Out);
      }
    }
    return std::make_shared<Function>(
        graph_.function->name_, std::move(params), std::move(directions), graph_.function->return_types_,
        SeqStmts::Flatten(std::move(body), span_), graph_.function->span_, graph_.function->func_type_,
        graph_.function->level_, graph_.function->role_, WithoutAutoTile(graph_.function->attrs_),
        graph_.function->requires_runtime_binding_);
  }

 private:
  const VectorGraph& graph_;
  const VectorSchedulePlan& plan_;
  bool lift_outputs_;
  Span span_;
  std::unordered_map<size_t, size_t> producer_;
  std::unordered_map<size_t, VarPtr> outputs_;
  std::vector<StmtPtr> prologue_;
  int unique_ = 0;

  std::string Fresh(const std::string& suffix) { return "__auto_tile_" + suffix + std::to_string(unique_++); }

  ReturnStmtPtr OriginalReturn() const {
    if (auto sequence = As<SeqStmts>(graph_.function->body_)) {
      for (const StmtPtr& stmt : sequence->stmts_)
        if (auto ret = As<ReturnStmt>(stmt)) return ret;
    }
    return As<ReturnStmt>(graph_.function->body_);
  }

  void CreateOutputs() {
    auto& registry = OpRegistry::GetInstance();
    for (size_t tensor : graph_.required_outputs) {
      const VectorTensor& output = graph_.tensors[tensor];
      auto create = registry.Create("tensor.create", {IndexTuple({output.rows, output.cols}, span_)},
                                    {{"dtype", output.dtype}, {"layout", TensorLayout::ND}}, span_);
      auto buffer = std::make_shared<Var>(output.var->name_hint_ + "_out", create->GetType(), span_);
      outputs_.emplace(tensor, buffer);
      if (!lift_outputs_) prologue_.push_back(std::make_shared<AssignStmt>(buffer, create, span_));
    }
  }

  void ValidatePlan() const {
    const auto validate_partition = [&](const AxisPartition& partition, int64_t extent, const char* axis) {
      INTERNAL_CHECK_SPAN(partition.parts > 0 && partition.small > 0 && partition.num_big >= 0 &&
                              partition.num_big < partition.parts &&
                              partition.big == partition.small + (partition.num_big != 0 ? 1 : 0) &&
                              partition.small * partition.parts + partition.num_big == extent,
                          span_)
          << "Internal error: AutoTile vector " << axis << " partition is not balanced";
    };
    validate_partition(plan_.m_partition, graph_.iteration_rows, "row");
    validate_partition(plan_.n_partition, graph_.iteration_cols, "column");
    INTERNAL_CHECK_SPAN(plan_.work_units == plan_.m_partition.parts * plan_.n_partition.parts, span_)
        << "Internal error: AutoTile vector work-unit count disagrees with its grid";
    INTERNAL_CHECK_SPAN(plan_.tile_h == plan_.m_partition.big && plan_.tile_w == plan_.n_partition.big, span_)
        << "Internal error: AutoTile vector tile disagrees with its balanced partitions";
    INTERNAL_CHECK_SPAN(plan_.dma_alignment_bytes > 0, span_)
        << "Internal error: AutoTile vector plan has no DMA alignment";
    INTERNAL_CHECK_SPAN(plan_.full_peak_ub_bytes > 0 && plan_.chunk_peak_ub_bytes > 0, span_)
        << "Internal error: AutoTile vector plan has an empty UB footprint";
    for (const VectorPhasePlan& phase : plan_.phases) ValidatePhase(phase);

    const VectorPhasePlan& body = plan_.phases[PhaseIndex(VectorPhase::Body)];
    const VectorPhasePlan& stats = plan_.phases[PhaseIndex(VectorPhase::Stats)];
    const VectorPhasePlan& apply = plan_.phases[PhaseIndex(VectorPhase::Apply)];
    const VectorPhasePlan& finalize = plan_.phases[PhaseIndex(VectorPhase::Finalize)];
    const auto is_empty = [](const VectorPhasePlan& phase) {
      return phase.ops.empty() && phase.inputs.empty() && phase.first_chunk == 0 && phase.trip_count == 0 &&
             phase.pipeline_stages == 1;
    };
    const auto expected_stages = [](int64_t trips) { return trips >= 2 ? 2 : 1; };
    switch (plan_.kind) {
      case VectorScheduleKind::Materialized:
        INTERNAL_CHECK_SPAN(body.ops.size() == graph_.ops.size() && is_empty(stats) && is_empty(apply) &&
                                is_empty(finalize) && plan_.row_strips == 1 && plan_.width_strips == 1 &&
                                plan_.strip_h == plan_.tile_h && plan_.strip_w == plan_.tile_w,
                            span_)
            << "Internal error: AutoTile materialized descriptor is incomplete";
        break;
      case VectorScheduleKind::PointwiseStream:
        INTERNAL_CHECK_SPAN(graph_.reduced_axis == 0 && !graph_.required_outputs.empty() &&
                                body.ops.size() == graph_.ops.size() &&
                                body.trip_count == plan_.row_strips * plan_.width_strips &&
                                body.trip_count > 1 && body.pipeline_stages == 2 &&
                                (plan_.row_strips == 1 || plan_.width_strips == 1) &&
                                plan_.row_strips * plan_.strip_h >= plan_.tile_h &&
                                (plan_.row_strips - 1) * plan_.strip_h < plan_.tile_h &&
                                plan_.width_strips * plan_.strip_w >= plan_.tile_w &&
                                (plan_.width_strips - 1) * plan_.strip_w < plan_.tile_w && is_empty(stats) &&
                                is_empty(apply) && is_empty(finalize),
                            span_)
            << "Internal error: AutoTile pointwise-stream descriptor is incomplete";
        break;
      case VectorScheduleKind::ReductionFolded:
      case VectorScheduleKind::ReductionSpanning:
      case VectorScheduleKind::Softmax:
        INTERNAL_CHECK_SPAN(graph_.reduced_axis != 0 && graph_.required_outputs.size() == 1 &&
                                is_empty(body) && plan_.chunk > 0 && plan_.chunk <= plan_.reduced_extent &&
                                plan_.full_chunks > 0 &&
                                plan_.full_chunks * plan_.chunk + plan_.tail == plan_.reduced_extent &&
                                stats.first_chunk == 1 && stats.trip_count == plan_.full_chunks - 1 &&
                                stats.pipeline_stages == expected_stages(stats.trip_count),
                            span_)
            << "Internal error: AutoTile reduction descriptor is incomplete";
        if (plan_.kind == VectorScheduleKind::ReductionFolded) {
          INTERNAL_CHECK_SPAN(is_empty(apply), span_)
              << "Internal error: folded reduction unexpectedly has an apply pass";
        } else {
          INTERNAL_CHECK_SPAN(
              !apply.ops.empty() && apply.first_chunk == 0 && apply.trip_count == plan_.full_chunks &&
                  apply.pipeline_stages == expected_stages(apply.trip_count) && is_empty(finalize),
              span_)
              << "Internal error: spanning reduction descriptor is incomplete";
        }
        if (plan_.kind == VectorScheduleKind::Softmax) {
          INTERNAL_CHECK_SPAN(graph_.softmax.matched && stats.ops.empty() && stats.inputs.size() == 1, span_)
              << "Internal error: online-softmax descriptor is incomplete";
        } else {
          INTERNAL_CHECK_SPAN(!stats.ops.empty(), span_)
              << "Internal error: reduction stats descriptor has no operations";
        }
        break;
    }
    if (plan_.reduction_split.present) {
      INTERNAL_CHECK_SPAN(
          plan_.kind == VectorScheduleKind::Materialized && graph_.reduced_axis == 2 &&
              plan_.n_partition.num_big == 0 && plan_.reduction_split.factor > 1 &&
              plan_.reduction_split.partial_extent * plan_.reduction_split.factor == graph_.iteration_rows &&
              plan_.reduction_split.seed_work_units == plan_.work_units,
          span_)
          << "Internal error: AutoTile column-split descriptor is inconsistent";
    } else {
      INTERNAL_CHECK_SPAN(plan_.reduction_split.factor == 1, span_)
          << "Internal error: AutoTile non-split descriptor has a split factor";
    }
  }

  void ValidatePhase(const VectorPhasePlan& phase) const {
    std::unordered_map<size_t, std::vector<VectorInputUse>> expected;
    std::unordered_map<size_t, size_t> step_by_op;
    for (size_t step = 0; step < phase.ops.size(); ++step) {
      const size_t op = phase.ops[step];
      INTERNAL_CHECK_SPAN(op < graph_.ops.size() && step_by_op.emplace(op, step).second, span_)
          << "Internal error: AutoTile phase contains an invalid or duplicate operation";
    }
    for (const VectorInputLifetime& input : phase.inputs) {
      INTERNAL_CHECK_SPAN(
          input.tensor < graph_.tensors.size() && expected.emplace(input.tensor, input.uses).second, span_)
          << "Internal error: AutoTile phase contains a duplicate boundary lifetime";
      INTERNAL_CHECK_SPAN(graph_.tensors[input.tensor].boundary_input && !input.uses.empty(), span_)
          << "Internal error: AutoTile phase contains an invalid boundary lifetime";
      size_t first = phase.ops.size();
      size_t last = 0;
      for (const VectorInputUse& use : input.uses) {
        INTERNAL_CHECK_SPAN(use.op < graph_.ops.size() && use.arg < graph_.ops[use.op].inputs.size() &&
                                graph_.ops[use.op].inputs[use.arg] == input.tensor,
                            span_)
            << "Internal error: AutoTile phase lifetime does not match the source DAG";
        if (!phase.ops.empty()) {
          auto step = step_by_op.find(use.op);
          INTERNAL_CHECK_SPAN(step != step_by_op.end(), span_)
              << "Internal error: AutoTile phase lifetime references an operation outside the phase";
          first = std::min(first, step->second);
          last = std::max(last, step->second);
        }
      }
      if (!phase.ops.empty()) {
        INTERNAL_CHECK_SPAN(input.first_use == first && input.last_use == last, span_)
            << "Internal error: AutoTile phase lifetime bounds disagree with its uses";
      }
    }
  }

  ExprPtr PartitionOffset(const ExprPtr& index, const AxisPartition& partition) const {
    ExprPtr result = MakeMul(index, Index(partition.small, span_), span_);
    const int64_t extra = partition.big - partition.small;
    if (extra != 0 && partition.num_big != 0) {
      result = MakeAdd(result,
                       MakeMul(MakeMin(index, Index(partition.num_big, span_)), Index(extra, span_), span_),
                       span_);
    }
    return result;
  }

  std::pair<ExprPtr, ExprPtr> RegionOffset(const VarPtr& index) const {
    const ExprPtr m_index = MakeFloorDiv(index, Index(plan_.n_partition.parts, span_), span_);
    const ExprPtr n_index = MakeFloorMod(index, Index(plan_.n_partition.parts, span_), span_);
    ExprPtr row = PartitionOffset(m_index, plan_.m_partition);
    ExprPtr col = PartitionOffset(n_index, plan_.n_partition);
    if (plan_.m_partition.parts * plan_.tile_h > graph_.iteration_rows)
      row = MakeMin(row, Index(graph_.iteration_rows - plan_.tile_h, span_), span_);
    if (plan_.n_partition.parts * plan_.tile_w > graph_.iteration_cols)
      col = MakeMin(col, Index(graph_.iteration_cols - plan_.tile_w, span_), span_);
    return {row, col};
  }

  VarPtr SliceInput(size_t tensor, int64_t rows, int64_t cols, const ExprPtr& row_offset,
                    const ExprPtr& col_offset, bool reduction_layout, std::vector<StmtPtr>& body) {
    auto& registry = OpRegistry::GetInstance();
    const VectorTensor& input = graph_.tensors[tensor];
    const int64_t granule = std::max<int64_t>(1, plan_.dma_alignment_bytes / DTypeBytes(input.dtype));
    const bool broadcast_row = input.rows == 1;
    const bool broadcast_col = input.cols == 1;
    const int64_t valid_rows = broadcast_row ? 1 : rows;
    const int64_t valid_cols = broadcast_col ? 1 : cols;
    const int64_t alloc_rows = broadcast_row ? 1 : (reduction_layout ? AlignUp(rows, granule) : rows);
    const int64_t alloc_cols = broadcast_col ? 1 : AlignUp(cols, granule);
    const ExprPtr input_row = broadcast_row ? Index(0, span_) : row_offset;
    const ExprPtr input_col = broadcast_col ? Index(0, span_) : col_offset;
    std::vector<ExprPtr> args{input.var, IndexTuple({alloc_rows, alloc_cols}, span_),
                              Pair(input_row, input_col, span_)};
    if (alloc_rows != valid_rows || alloc_cols != valid_cols)
      args.push_back(IndexTuple({valid_rows, valid_cols}, span_));
    auto slice = registry.Create("tensor.slice", args, span_);
    auto value = std::make_shared<Var>(Fresh("in"), slice->GetType(), span_);
    body.push_back(std::make_shared<AssignStmt>(value, slice, span_));
    return value;
  }

  VarPtr Replay(const VectorPhasePlan& phase, int64_t rows, int64_t cols, const ExprPtr& row_offset,
                const ExprPtr& col_offset, bool reduction_layout, std::vector<StmtPtr>& body,
                std::unordered_map<size_t, VarPtr>& onchip,
                const std::unordered_map<size_t, VarPtr>& substitutions = {}) {
    auto& registry = OpRegistry::GetInstance();
    onchip = substitutions;
    std::unordered_map<size_t, VarPtr> input_cache;
    std::unordered_map<size_t, std::vector<VectorInputUse>> observed;
    VarPtr result;
    for (size_t op_index : phase.ops) {
      const VectorOp& op = graph_.ops.at(op_index);
      auto substitution = substitutions.find(op.output);
      if (substitution != substitutions.end()) {
        onchip[op.output] = substitution->second;
        result = substitution->second;
        continue;
      }
      std::vector<ExprPtr> args;
      size_t tensor_arg = 0;
      for (const ExprPtr& source_arg : op.call->args_) {
        if (As<TensorType>(source_arg->GetType()) == nullptr) {
          args.push_back(source_arg);
          continue;
        }
        INTERNAL_CHECK_SPAN(tensor_arg < op.inputs.size(), op.stmt->span_)
            << "Internal error: AutoTile tensor operand count changed during emission";
        const size_t tensor = op.inputs[tensor_arg];
        auto resident = onchip.find(tensor);
        if (resident != onchip.end()) {
          args.push_back(resident->second);
        } else {
          INTERNAL_CHECK_SPAN(graph_.tensors[tensor].boundary_input, op.stmt->span_)
              << "Internal error: AutoTile phase omitted a producer required by replay";
          auto cached = input_cache.find(tensor);
          if (cached == input_cache.end()) {
            VarPtr input = SliceInput(tensor, rows, cols, row_offset, col_offset, reduction_layout, body);
            cached = input_cache.emplace(tensor, input).first;
          }
          args.push_back(cached->second);
          observed[tensor].push_back({op_index, tensor_arg});
        }
        ++tensor_arg;
      }
      INTERNAL_CHECK_SPAN(tensor_arg == op.inputs.size(), op.stmt->span_)
          << "Internal error: AutoTile tensor operand count changed during emission";
      if (op.swap_operands) {
        INTERNAL_CHECK_SPAN(args.size() == 2, op.stmt->span_)
            << "Internal error: AutoTile broadcast normalization expected two operands";
        std::swap(args[0], args[1]);
      }
      auto call = registry.Create(op.emission_op, args, op.call->kwargs_, span_);
      result = std::make_shared<Var>(Fresh("v"), call->GetType(), span_);
      body.push_back(std::make_shared<AssignStmt>(result, call, span_));
      onchip[op.output] = result;
    }
    for (const VectorInputLifetime& expected : phase.inputs) {
      INTERNAL_CHECK_SPAN(observed[expected.tensor] == expected.uses, span_)
          << "Internal error: AutoTile emission did not realize the planned input-use lifetime";
    }
    INTERNAL_CHECK_SPAN(observed.size() == phase.inputs.size(), span_)
        << "Internal error: AutoTile emission observed an unplanned boundary input";
    INTERNAL_CHECK_SPAN(result != nullptr || phase.ops.empty(), span_)
        << "Internal error: AutoTile phase produced no value";
    return result;
  }

  ExprPtr OutputOffset(size_t tensor, const ExprPtr& row, const ExprPtr& col) const {
    const VectorTensor& output = graph_.tensors[tensor];
    return Pair(output.rows == 1 ? Index(0, span_) : row, output.cols == 1 ? Index(0, span_) : col, span_);
  }

  VarPtr Assemble(size_t tensor, const VarPtr& tile, const ExprPtr& row, const ExprPtr& col,
                  std::vector<StmtPtr>& body, const ExprPtr& target = nullptr, int atomic = 0) {
    auto& registry = OpRegistry::GetInstance();
    const ExprPtr buffer = target == nullptr ? ExprPtr(outputs_.at(tensor)) : target;
    std::vector<std::pair<std::string, std::any>> kwargs;
    if (atomic != 0) kwargs.emplace_back("atomic", atomic);
    auto assemble =
        registry.Create("tensor.assemble", {buffer, tile, OutputOffset(tensor, row, col)}, kwargs, span_);
    VarPtr result = atomic == 0 ? graph_.tensors[tensor].var
                                : std::make_shared<Var>(Fresh("atomic"), assemble->GetType(), span_);
    body.push_back(std::make_shared<AssignStmt>(result, assemble, span_));
    return result;
  }

  void EmitMaterialized(std::vector<StmtPtr>& outer) {
    if (plan_.reduction_split.present) {
      EmitColumnSplit(outer);
      return;
    }
    auto index = std::make_shared<Var>(Fresh("region"), std::make_shared<ScalarType>(DataType::INDEX), span_);
    const auto [row, col] = RegionOffset(index);
    std::vector<StmtPtr> body;
    std::unordered_map<size_t, VarPtr> onchip;
    Replay(plan_.phases[PhaseIndex(VectorPhase::Body)], plan_.tile_h, plan_.tile_w, row, col,
           graph_.reduced_axis != 0, body, onchip);
    for (size_t tensor : graph_.required_outputs) {
      auto value = onchip.find(tensor);
      INTERNAL_CHECK_SPAN(value != onchip.end(), span_)
          << "Internal error: AutoTile did not keep a returned value live through its store";
      Assemble(tensor, value->second, row, col, body);
    }
    outer.push_back(WrapSpmd(index, std::move(body), plan_.work_units, graph_.function->name_, span_));
  }

  void EmitPointwiseStream(std::vector<StmtPtr>& outer) {
    auto index = std::make_shared<Var>(Fresh("region"), std::make_shared<ScalarType>(DataType::INDEX), span_);
    const auto [region_row, region_col] = RegionOffset(index);
    const int64_t strips = plan_.row_strips * plan_.width_strips;
    auto strip = std::make_shared<Var>(Fresh("strip"), std::make_shared<ScalarType>(DataType::INDEX), span_);
    std::vector<IterArgPtr> output_iters;
    output_iters.reserve(graph_.required_outputs.size());
    for (size_t output : graph_.required_outputs) {
      output_iters.push_back(std::make_shared<IterArg>(Fresh("out_it"), outputs_.at(output)->GetType(),
                                                       ExprPtr(outputs_.at(output)), span_));
    }
    ExprPtr row_index = plan_.width_strips == 1
                            ? ExprPtr(strip)
                            : MakeFloorDiv(strip, Index(plan_.width_strips, span_), span_);
    ExprPtr local_row = MakeMul(row_index, Index(plan_.strip_h, span_), span_);
    if (plan_.row_strips * plan_.strip_h > plan_.tile_h)
      local_row = MakeMin(local_row, Index(plan_.tile_h - plan_.strip_h, span_), span_);
    ExprPtr local_col = Index(0, span_);
    if (plan_.width_strips > 1) {
      local_col = MakeMul(MakeFloorMod(strip, Index(plan_.width_strips, span_), span_),
                          Index(plan_.strip_w, span_), span_);
      if (plan_.width_strips * plan_.strip_w > plan_.tile_w)
        local_col = MakeMin(local_col, Index(plan_.tile_w - plan_.strip_w, span_), span_);
    }
    ExprPtr row = MakeAdd(region_row, local_row, span_);
    ExprPtr col = MakeAdd(region_col, local_col, span_);
    std::vector<StmtPtr> loop_body;
    std::unordered_map<size_t, VarPtr> onchip;
    Replay(plan_.phases[PhaseIndex(VectorPhase::Body)], plan_.strip_h, plan_.strip_w, row, col, false,
           loop_body, onchip);
    std::vector<ExprPtr> yielded;
    std::vector<VarPtr> results;
    yielded.reserve(graph_.required_outputs.size());
    results.reserve(graph_.required_outputs.size());
    for (size_t i = 0; i < graph_.required_outputs.size(); ++i) {
      const size_t output = graph_.required_outputs[i];
      auto tile = onchip.find(output);
      INTERNAL_CHECK_SPAN(tile != onchip.end(), span_)
          << "Internal error: AutoTile did not keep a streamed live-out through its store";
      auto assemble = OpRegistry::GetInstance().Create(
          "tensor.assemble",
          {ExprPtr(output_iters[i]), ExprPtr(tile->second), OutputOffset(output, row, col)}, span_);
      auto next = std::make_shared<Var>(Fresh("out_next"), assemble->GetType(), span_);
      loop_body.push_back(std::make_shared<AssignStmt>(next, assemble, span_));
      yielded.push_back(next);
      results.push_back(graph_.tensors[output].var);
    }
    loop_body.push_back(std::make_shared<YieldStmt>(std::move(yielded), span_));
    std::vector<std::pair<std::string, std::any>> attrs;
    const bool pipelined = plan_.phases[PhaseIndex(VectorPhase::Body)].pipeline_stages == 2;
    if (pipelined) attrs.emplace_back(kPipelineStagesAttr, 2);
    auto loop = std::make_shared<ForStmt>(
        strip, Index(0, span_), Index(strips, span_), Index(1, span_), std::move(output_iters),
        SeqStmts::Flatten(std::move(loop_body), span_), std::move(results), span_,
        pipelined ? ForKind::Pipeline : ForKind::Sequential, std::move(attrs));
    outer.push_back(WrapSpmd(index, {loop}, plan_.work_units, graph_.function->name_, span_));
  }

  VarPtr PreserveValid(const VarPtr& value, const TypePtr& carried, std::vector<StmtPtr>& body) {
    auto type = As<TensorType>(carried);
    if (type == nullptr || !type->tensor_view_.has_value() || type->tensor_view_->valid_shape.empty())
      return value;
    const auto& valid = type->tensor_view_->valid_shape;
    auto call = OpRegistry::GetInstance().Create("tensor.set_validshape", {value, valid[0], valid[1]}, span_);
    auto result = std::make_shared<Var>(Fresh("valid"), call->GetType(), span_);
    body.push_back(std::make_shared<AssignStmt>(result, call, span_));
    return result;
  }

  void AppendLoop(std::vector<StmtPtr>& outer, const VectorPhasePlan& descriptor, const VarPtr& index,
                  std::vector<IterArgPtr> carries, std::vector<StmtPtr> body, std::vector<VarPtr> results) {
    INTERNAL_CHECK_SPAN(descriptor.trip_count > 0, span_)
        << "Internal error: AutoTile attempted to emit an empty planned loop";
    std::vector<std::pair<std::string, std::any>> attrs;
    const bool pipelined = descriptor.pipeline_stages == 2;
    if (pipelined) attrs.emplace_back(kPipelineStagesAttr, 2);
    outer.push_back(std::make_shared<ForStmt>(
        index, Index(descriptor.first_chunk, span_),
        Index(descriptor.first_chunk + descriptor.trip_count, span_), Index(1, span_), std::move(carries),
        SeqStmts::Flatten(std::move(body), span_), std::move(results), span_,
        pipelined ? ForKind::Pipeline : ForKind::Sequential, std::move(attrs)));
  }

  size_t ReductionOp() const {
    INTERNAL_CHECK_SPAN(graph_.reduction_op < graph_.ops.size(), span_)
        << "Internal error: AutoTile reduction plan has no reduction operation";
    return graph_.reduction_op;
  }

  VarPtr ReplayReducedChunk(const VectorPhasePlan& phase, int64_t extent, const ExprPtr& reduced_offset,
                            const ExprPtr& free_offset, std::vector<StmtPtr>& body,
                            std::unordered_map<size_t, VarPtr>& onchip,
                            const std::unordered_map<size_t, VarPtr>& substitutions = {}) {
    return graph_.reduced_axis == 1 ? Replay(phase, plan_.free_tile, extent, free_offset, reduced_offset,
                                             true, body, onchip, substitutions)
                                    : Replay(phase, extent, plan_.free_tile, reduced_offset, free_offset,
                                             true, body, onchip, substitutions);
  }

  ExprPtr FreeOffset(const VarPtr& region) const {
    return graph_.reduced_axis == 1 ? RegionOffset(region).first : RegionOffset(region).second;
  }

  ExprPtr ReducedOutputOffset(const ExprPtr& free) const {
    return graph_.reduced_axis == 1 ? Pair(free, Index(0, span_), span_) : Pair(Index(0, span_), free, span_);
  }

  void EmitReduction(std::vector<StmtPtr>& outer) {
    INTERNAL_CHECK_SPAN(graph_.required_outputs.size() == 1, span_)
        << "Internal error: streamed reduction AutoTile requires one output";
    const size_t output = graph_.required_outputs.front();
    const size_t reduction_op = ReductionOp();
    const size_t reduction_tensor = graph_.ops[reduction_op].output;
    const bool is_max = graph_.ops[reduction_op].kind == VectorOpKind::RowMax ||
                        graph_.ops[reduction_op].kind == VectorOpKind::ColMax;
    const std::string merge_name = is_max ? "tensor.maximum" : "tensor.add";
    auto region =
        std::make_shared<Var>(Fresh("region"), std::make_shared<ScalarType>(DataType::INDEX), span_);
    ExprPtr free = FreeOffset(region);
    std::vector<StmtPtr> body;

    std::unordered_map<size_t, VarPtr> onchip;
    VarPtr accumulator = ReplayReducedChunk(plan_.phases[PhaseIndex(VectorPhase::Stats)], plan_.chunk,
                                            Index(0, span_), free, body, onchip);
    const VectorPhasePlan& stats = plan_.phases[PhaseIndex(VectorPhase::Stats)];
    if (stats.trip_count > 0) {
      auto chunk_index =
          std::make_shared<Var>(Fresh("chunk"), std::make_shared<ScalarType>(DataType::INDEX), span_);
      auto acc_iter = std::make_shared<IterArg>(Fresh("acc_it"), accumulator->GetType(), accumulator, span_);
      std::vector<StmtPtr> loop_body;
      std::unordered_map<size_t, VarPtr> chunk_values;
      VarPtr partial =
          ReplayReducedChunk(stats, plan_.chunk, MakeMul(chunk_index, Index(plan_.chunk, span_), span_), free,
                             loop_body, chunk_values);
      auto merge = OpRegistry::GetInstance().Create(merge_name, {ExprPtr(acc_iter), partial}, span_);
      auto next = std::make_shared<Var>(Fresh("acc_next"), merge->GetType(), span_);
      loop_body.push_back(std::make_shared<AssignStmt>(next, merge, span_));
      VarPtr valid = PreserveValid(next, acc_iter->GetType(), loop_body);
      loop_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{valid}, span_));
      auto result = std::make_shared<Var>(Fresh("acc"), accumulator->GetType(), span_);
      AppendLoop(body, stats, chunk_index, {acc_iter}, std::move(loop_body), {result});
      accumulator = result;
    }
    if (plan_.tail > 0) {
      std::unordered_map<size_t, VarPtr> tail_values;
      VarPtr partial = ReplayReducedChunk(stats, plan_.tail, Index(plan_.full_chunks * plan_.chunk, span_),
                                          free, body, tail_values);
      auto merge = OpRegistry::GetInstance().Create(merge_name, {accumulator, partial}, span_);
      auto next = std::make_shared<Var>(Fresh("acc_tail"), merge->GetType(), span_);
      body.push_back(std::make_shared<AssignStmt>(next, merge, span_));
      accumulator = PreserveValid(next, accumulator->GetType(), body);
    }

    if (plan_.kind == VectorScheduleKind::ReductionFolded) {
      VarPtr result = accumulator;
      const VectorPhasePlan& finalize = plan_.phases[PhaseIndex(VectorPhase::Finalize)];
      if (!finalize.ops.empty()) {
        std::unordered_map<size_t, VarPtr> finalized;
        result = ReplayReducedChunk(finalize, 1, Index(0, span_), free, body, finalized,
                                    {{reduction_tensor, accumulator}});
      }
      auto assemble = OpRegistry::GetInstance().Create(
          "tensor.assemble", {outputs_.at(output), result, ReducedOutputOffset(free)}, span_);
      body.push_back(std::make_shared<AssignStmt>(graph_.tensors[output].var, assemble, span_));
    } else {
      EmitApplyPass(output, free, {{reduction_tensor, accumulator}}, body);
    }
    outer.push_back(WrapSpmd(region, std::move(body), plan_.work_units, graph_.function->name_, span_));
  }

  void EmitApplyPass(size_t output, const ExprPtr& free,
                     const std::unordered_map<size_t, VarPtr>& substitutions, std::vector<StmtPtr>& body) {
    const VectorPhasePlan& apply = plan_.phases[PhaseIndex(VectorPhase::Apply)];
    ExprPtr current = outputs_.at(output);
    if (apply.trip_count > 0) {
      auto chunk_index =
          std::make_shared<Var>(Fresh("apply"), std::make_shared<ScalarType>(DataType::INDEX), span_);
      auto output_iter =
          std::make_shared<IterArg>(Fresh("out_it"), outputs_.at(output)->GetType(), current, span_);
      ExprPtr offset = MakeMul(chunk_index, Index(plan_.chunk, span_), span_);
      std::vector<StmtPtr> loop_body;
      std::unordered_map<size_t, VarPtr> onchip;
      VarPtr tile = ReplayReducedChunk(apply, plan_.chunk, offset, free, loop_body, onchip, substitutions);
      ExprPtr store_offset = graph_.reduced_axis == 1 ? Pair(free, offset, span_) : Pair(offset, free, span_);
      auto assemble = OpRegistry::GetInstance().Create("tensor.assemble",
                                                       {ExprPtr(output_iter), tile, store_offset}, span_);
      auto next = std::make_shared<Var>(Fresh("out_next"), assemble->GetType(), span_);
      loop_body.push_back(std::make_shared<AssignStmt>(next, assemble, span_));
      loop_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{next}, span_));
      const bool has_tail = plan_.tail > 0;
      VarPtr result = has_tail
                          ? std::make_shared<Var>(Fresh("out_full"), outputs_.at(output)->GetType(), span_)
                          : graph_.tensors[output].var;
      AppendLoop(body, apply, chunk_index, {output_iter}, std::move(loop_body), {result});
      current = result;
    }
    if (plan_.tail > 0) {
      const ExprPtr offset = Index(plan_.full_chunks * plan_.chunk, span_);
      std::unordered_map<size_t, VarPtr> onchip;
      VarPtr tile = ReplayReducedChunk(apply, plan_.tail, offset, free, body, onchip, substitutions);
      ExprPtr store_offset = graph_.reduced_axis == 1 ? Pair(free, offset, span_) : Pair(offset, free, span_);
      auto assemble =
          OpRegistry::GetInstance().Create("tensor.assemble", {current, tile, store_offset}, span_);
      body.push_back(std::make_shared<AssignStmt>(graph_.tensors[output].var, assemble, span_));
    }
  }

  std::pair<VarPtr, VarPtr> EmitSoftmaxChunk(int64_t extent, const ExprPtr& offset, const ExprPtr& free,
                                             const VarPtr& old_max, const VarPtr& old_sum,
                                             std::vector<StmtPtr>& body) {
    auto& registry = OpRegistry::GetInstance();
    const size_t input = graph_.softmax.input;
    VarPtr x = graph_.reduced_axis == 1
                   ? SliceInput(input, plan_.free_tile, extent, free, offset, true, body)
                   : SliceInput(input, extent, plan_.free_tile, offset, free, true, body);
    const VectorOp& max_op = graph_.ops[graph_.softmax.max_op];
    const VectorOp& sum_op = graph_.ops[graph_.softmax.sum_op];
    auto local_max_call = registry.Create(max_op.call->op_->name_, {x}, max_op.call->kwargs_, span_);
    auto local_max = std::make_shared<Var>(Fresh("local_max"), local_max_call->GetType(), span_);
    body.push_back(std::make_shared<AssignStmt>(local_max, local_max_call, span_));
    VarPtr new_max = local_max;
    if (old_max != nullptr) {
      auto call = registry.Create("tensor.maximum", {old_max, local_max}, span_);
      new_max = std::make_shared<Var>(Fresh("max"), call->GetType(), span_);
      body.push_back(std::make_shared<AssignStmt>(new_max, call, span_));
    }
    auto shifted_call = registry.Create("tensor.sub", {x, new_max}, span_);
    auto shifted = std::make_shared<Var>(Fresh("shift"), shifted_call->GetType(), span_);
    body.push_back(std::make_shared<AssignStmt>(shifted, shifted_call, span_));
    auto exp_call = registry.Create("tensor.exp", {shifted}, span_);
    auto exp = std::make_shared<Var>(Fresh("exp"), exp_call->GetType(), span_);
    body.push_back(std::make_shared<AssignStmt>(exp, exp_call, span_));
    auto local_sum_call = registry.Create(sum_op.call->op_->name_, {exp}, sum_op.call->kwargs_, span_);
    auto local_sum = std::make_shared<Var>(Fresh("local_sum"), local_sum_call->GetType(), span_);
    body.push_back(std::make_shared<AssignStmt>(local_sum, local_sum_call, span_));
    VarPtr new_sum = local_sum;
    if (old_sum != nullptr) {
      auto delta_call = registry.Create("tensor.sub", {old_max, new_max}, span_);
      auto delta = std::make_shared<Var>(Fresh("delta"), delta_call->GetType(), span_);
      body.push_back(std::make_shared<AssignStmt>(delta, delta_call, span_));
      auto correction_call = registry.Create("tensor.exp", {delta}, span_);
      auto correction = std::make_shared<Var>(Fresh("correction"), correction_call->GetType(), span_);
      body.push_back(std::make_shared<AssignStmt>(correction, correction_call, span_));
      auto scaled_call = registry.Create("tensor.mul", {old_sum, correction}, span_);
      auto scaled = std::make_shared<Var>(Fresh("scaled_sum"), scaled_call->GetType(), span_);
      body.push_back(std::make_shared<AssignStmt>(scaled, scaled_call, span_));
      auto sum_call = registry.Create("tensor.add", {scaled, local_sum}, span_);
      new_sum = std::make_shared<Var>(Fresh("sum"), sum_call->GetType(), span_);
      body.push_back(std::make_shared<AssignStmt>(new_sum, sum_call, span_));
    }
    return {new_max, new_sum};
  }

  void EmitSoftmax(std::vector<StmtPtr>& outer) {
    INTERNAL_CHECK_SPAN(graph_.softmax.matched && graph_.required_outputs.size() == 1, span_)
        << "Internal error: AutoTile softmax descriptor is incomplete";
    const size_t output = graph_.required_outputs.front();
    auto region =
        std::make_shared<Var>(Fresh("region"), std::make_shared<ScalarType>(DataType::INDEX), span_);
    ExprPtr free = FreeOffset(region);
    std::vector<StmtPtr> body;
    auto [running_max, running_sum] =
        EmitSoftmaxChunk(plan_.chunk, Index(0, span_), free, nullptr, nullptr, body);
    const VectorPhasePlan& stats = plan_.phases[PhaseIndex(VectorPhase::Stats)];
    if (stats.trip_count > 0) {
      auto chunk_index =
          std::make_shared<Var>(Fresh("chunk"), std::make_shared<ScalarType>(DataType::INDEX), span_);
      auto max_iter = std::make_shared<IterArg>(Fresh("max_it"), running_max->GetType(), running_max, span_);
      auto sum_iter = std::make_shared<IterArg>(Fresh("sum_it"), running_sum->GetType(), running_sum, span_);
      std::vector<StmtPtr> loop_body;
      auto [next_max, next_sum] =
          EmitSoftmaxChunk(plan_.chunk, MakeMul(chunk_index, Index(plan_.chunk, span_), span_), free,
                           max_iter, sum_iter, loop_body);
      next_max = PreserveValid(next_max, max_iter->GetType(), loop_body);
      next_sum = PreserveValid(next_sum, sum_iter->GetType(), loop_body);
      loop_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{next_max, next_sum}, span_));
      auto max_result = std::make_shared<Var>(Fresh("max"), running_max->GetType(), span_);
      auto sum_result = std::make_shared<Var>(Fresh("sum"), running_sum->GetType(), span_);
      AppendLoop(body, stats, chunk_index, {max_iter, sum_iter}, std::move(loop_body),
                 {max_result, sum_result});
      running_max = max_result;
      running_sum = sum_result;
    }
    if (plan_.tail > 0) {
      auto [next_max, next_sum] = EmitSoftmaxChunk(plan_.tail, Index(plan_.full_chunks * plan_.chunk, span_),
                                                   free, running_max, running_sum, body);
      running_max = PreserveValid(next_max, running_max->GetType(), body);
      running_sum = PreserveValid(next_sum, running_sum->GetType(), body);
    }
    EmitApplyPass(output, free,
                  {{graph_.ops[graph_.softmax.max_op].output, running_max},
                   {graph_.ops[graph_.softmax.sum_op].output, running_sum}},
                  body);
    outer.push_back(WrapSpmd(region, std::move(body), plan_.work_units, graph_.function->name_, span_));
  }

  void EmitColumnSplit(std::vector<StmtPtr>& outer) {
    INTERNAL_CHECK_SPAN(graph_.required_outputs.size() == 1 && graph_.reduced_axis == 2, span_)
        << "Internal error: AutoTile column split has an invalid graph";
    const size_t output = graph_.required_outputs.front();
    const VectorTensor& result = graph_.tensors[output];
    auto& registry = OpRegistry::GetInstance();

    auto seed_index =
        std::make_shared<Var>(Fresh("seed"), std::make_shared<ScalarType>(DataType::INDEX), span_);
    ExprPtr seed_col = MakeMul(seed_index, Index(plan_.tile_w, span_), span_);
    auto zero = std::make_shared<ConstFloat>(0.0, result.dtype, span_);
    auto full = registry.Create("tensor.full", {IndexTuple({1, plan_.tile_w}, span_), zero},
                                {{"dtype", result.dtype}}, span_);
    auto zero_tile = std::make_shared<Var>(Fresh("zero"), full->GetType(), span_);
    auto seed_store = registry.Create(
        "tensor.assemble", {outputs_.at(output), zero_tile, Pair(Index(0, span_), seed_col, span_)}, span_);
    auto seeded = std::make_shared<Var>(Fresh("seeded"), seed_store->GetType(), span_);
    std::vector<StmtPtr> seed_body{std::make_shared<AssignStmt>(zero_tile, full, span_),
                                   std::make_shared<AssignStmt>(seeded, seed_store, span_)};
    outer.push_back(WrapSpmd(seed_index, std::move(seed_body), plan_.reduction_split.seed_work_units,
                             graph_.function->name_ + "_seed", span_));

    auto index = std::make_shared<Var>(Fresh("split"), std::make_shared<ScalarType>(DataType::INDEX), span_);
    ExprPtr split_index = MakeFloorMod(index, Index(plan_.reduction_split.factor, span_), span_);
    ExprPtr free_index = MakeFloorDiv(index, Index(plan_.reduction_split.factor, span_), span_);
    ExprPtr row = MakeMul(split_index, Index(plan_.reduction_split.partial_extent, span_), span_);
    ExprPtr col = PartitionOffset(free_index, plan_.n_partition);
    std::vector<StmtPtr> body;
    std::unordered_map<size_t, VarPtr> onchip;
    VarPtr partial = Replay(plan_.phases[PhaseIndex(VectorPhase::Body)], plan_.reduction_split.partial_extent,
                            plan_.tile_w, row, col, true, body, onchip);
    auto atomic = registry.Create("tensor.assemble", {seeded, partial, Pair(Index(0, span_), col, span_)},
                                  {{"atomic", 1}}, span_);
    body.push_back(std::make_shared<AssignStmt>(graph_.tensors[output].var, atomic, span_));
    outer.push_back(WrapSpmd(index, std::move(body), plan_.work_units * plan_.reduction_split.factor,
                             graph_.function->name_, span_));
  }
};

}  // namespace

FunctionPtr EmitVectorSchedule(const VectorGraph& graph, const VectorSchedulePlan& plan,
                               const std::unordered_set<std::string>& called_functions) {
  const bool lift_outputs = called_functions.count(graph.function->name_) == 0;
  return VectorEmitter(graph, plan, lift_outputs).Emit();
}

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto
