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
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/auto_layout/extractor.h"
#include "pypto/ir/transforms/auto_layout/solver.h"
#include "pypto/ir/transforms/auto_layout/writer.h"
#include "pypto/ir/transforms/passes.h"

namespace ir = pypto::ir;
namespace al = ir::auto_layout;
using pypto::DataType;

namespace {

std::string SignatureString(const std::unordered_map<std::string, al::LayoutSpec>& value_layouts) {
  std::vector<std::string> keys;
  keys.reserve(value_layouts.size());
  for (const auto& [k, _] : value_layouts) {
    keys.push_back(k);
  }
  std::sort(keys.begin(), keys.end());
  std::ostringstream oss;
  for (const auto& k : keys) {
    const auto& v = value_layouts.at(k);
    oss << k << '=' << v.ShortName() << ';';
  }
  return oss.str();
}

// Exhaustive reference for small extracted problems, independent of solver search.
std::vector<al::CostBreakdown> EnumerateAssignments(const al::LayoutProblemInstance& instance) {
  if (instance.operations.empty()) {
    return {al::EvaluateAssignment(instance, al::LayoutAssignment{})};
  }

  std::vector<al::CostBreakdown> results;
  std::unordered_map<std::string, al::LayoutSpec> val_map;
  for (const auto& [name, node] : instance.values) {
    if (!node.producer_op_id.has_value() && node.memory_space != ir::MemorySpace::DDR) {
      val_map.emplace(name, node.current_layout);
    }
  }
  std::unordered_map<std::string, al::OperatorConfig> op_map;

  const auto& ops = instance.operations;
  auto search = [&](auto& self, size_t idx) -> void {
    if (idx == ops.size()) {
      results.push_back(al::EvaluateAssignment(instance, al::LayoutAssignment{val_map, op_map}));
      return;
    }
    const auto& op = ops[idx];
    for (const auto& cfg : op.configs) {
      bool consistent = true;
      std::vector<std::pair<std::string, al::LayoutSpec>> newly_bound;
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        const std::string& inp_name = op.inputs[i];
        const al::LayoutSpec& req_lay = cfg.input_layouts[i];
        const al::ValueNode* inp_node = instance.FindValue(inp_name);
        if (inp_node == nullptr) {
          consistent = false;
          break;
        }
        auto existing_it = val_map.find(inp_name);
        if (existing_it != val_map.end()) {
          const al::LayoutSpec& existing = existing_it->second;
          const bool can_relayout =
              inp_node->memory_space == ir::MemorySpace::Vec &&
              (op.op_name == "tile.move" ||
               (op.op_name.rfind("tile.", 0) == 0 &&
                instance.cost_model.SupportsVecTileConversion(*inp_node, existing, req_lay)));
          if (!can_relayout && existing != req_lay) {
            consistent = false;
            break;
          }
        } else {
          // Only bind if not already added earlier in this same op's input loop
          auto dup_it = std::find_if(newly_bound.begin(), newly_bound.end(),
                                     [&](const auto& p) { return p.first == inp_name; });
          if (dup_it != newly_bound.end()) {
            const al::LayoutSpec& existing = dup_it->second;
            const bool can_relayout =
                inp_node->memory_space == ir::MemorySpace::Vec &&
                (op.op_name == "tile.move" ||
                 (op.op_name.rfind("tile.", 0) == 0 &&
                  instance.cost_model.SupportsVecTileConversion(*inp_node, existing, req_lay)));
            if (!can_relayout && existing != req_lay) {
              consistent = false;
              break;
            }
          } else {
            newly_bound.emplace_back(inp_name, req_lay);
          }
        }
      }
      if (!consistent) {
        continue;
      }

      auto out_it = val_map.find(op.output);
      if (out_it != val_map.end() && out_it->second != cfg.output_layout) {
        continue;
      }
      bool out_dup_conflict = false;
      for (const auto& [k, v] : newly_bound) {
        if (k == op.output && v != cfg.output_layout) {
          out_dup_conflict = true;
          break;
        }
      }
      if (out_dup_conflict) {
        continue;
      }

      std::vector<std::string> inserted_keys;
      for (const auto& [k, v] : newly_bound) {
        if (val_map.emplace(k, v).second) {
          inserted_keys.push_back(k);
        }
      }
      if (val_map.emplace(op.output, cfg.output_layout).second) {
        inserted_keys.push_back(op.output);
      }
      op_map[op.op_id] = cfg;

      self(self, idx + 1);

      op_map.erase(op.op_id);
      for (const auto& k : inserted_keys) {
        val_map.erase(k);
      }
    }
  };
  search(search, 0);

  std::unordered_map<std::string, size_t> best_idx_by_sig;
  std::vector<al::CostBreakdown> deduped;
  for (auto& bd : results) {
    std::string sig = SignatureString(bd.assignment.value_layouts);
    auto it = best_idx_by_sig.find(sig);
    if (it == best_idx_by_sig.end()) {
      best_idx_by_sig.emplace(std::move(sig), deduped.size());
      deduped.push_back(std::move(bd));
    } else if (bd.total_cost_cycles < deduped[it->second].total_cost_cycles) {
      deduped[it->second] = std::move(bd);
    }
  }
  results = std::move(deduped);

  std::stable_sort(results.begin(), results.end(),
                   [](const al::CostBreakdown& a, const al::CostBreakdown& b) {
                     return a.total_cost_cycles < b.total_cost_cycles;
                   });
  return results;
}

enum class HostMatrixDispatch { Implicit, RankZero, UnknownRank, PartialSlice, FullSlice, Mutable };

void TestHostMatrixLayout(HostMatrixDispatch dispatch, std::optional<int64_t> world_size,
                          bool expect_selectable, bool wide = false,
                          pypto::backend::BackendType backend = pypto::backend::BackendType::Ascend910) {
  const auto span = ir::Span::unknown();
  using Exprs = std::vector<ir::ExprPtr>;
  using Stmts = std::vector<ir::StmtPtr>;
  using Attrs = std::vector<std::pair<std::string, std::any>>;
  const auto index = [&](int64_t n) { return std::make_shared<ir::ConstInt>(n, DataType::INDEX, span); };
  const auto tuple = [&](Exprs values) { return std::make_shared<ir::MakeTuple>(std::move(values), span); };
  const auto tensor = [](std::vector<int64_t> dims) {
    return std::make_shared<ir::TensorType>(std::move(dims), DataType::BF16, std::nullopt);
  };
  const int64_t rows = wide ? 152064 : 64;
  const int64_t cols = wide ? 5120 : 32;
  const auto matrix = tensor({rows, cols});
  const auto weight = std::make_shared<ir::Var>("weight", matrix, span);
  const auto device_type = dispatch == HostMatrixDispatch::PartialSlice ? tensor({rows / 2, cols}) : matrix;
  const auto chip_param = std::make_shared<ir::Var>("chip_weight", device_type, span);
  const auto kernel_param = std::make_shared<ir::Var>("kernel_weight", device_type, span);
  const auto load_shape = wide ? tuple({index(192), index(512)}) : tuple(device_type->shape_);
  const auto offsets = wide ? tuple({index(192), index(512)}) : tuple({index(0), index(0)});
  const auto& registry = ir::OpRegistry::GetInstance();
  const auto load = registry.Create("tile.load", {kernel_param, offsets, load_shape, load_shape},
                                    {{"target_memory", ir::MemorySpace::Mat}}, span);
  const auto loaded = std::make_shared<ir::Var>("loaded", load->GetType(), span);
  const auto function = [&](const char* name, std::vector<ir::VarPtr> params,
                            std::vector<ir::ParamDirection> directions, Stmts body, ir::FunctionType type,
                            ir::Level level, ir::Role role) {
    return std::make_shared<ir::Function>(
        name, std::move(params), std::move(directions), std::vector<ir::TypePtr>{},
        std::make_shared<ir::SeqStmts>(std::move(body), span), span, type, level, role);
  };
  const auto in = ir::ParamDirection::In;
  const auto kernel =
      function("load_weight", {kernel_param}, {in}, {std::make_shared<ir::AssignStmt>(loaded, load, span)},
               ir::FunctionType::InCore, ir::Level::CHIP_DIE, ir::Role::SubWorker);
  const auto kernel_call =
      std::make_shared<ir::Call>(std::make_shared<ir::GlobalVar>(kernel->name_), Exprs{chip_param}, span);
  const auto chip = function("chip", {chip_param},
                             {dispatch == HostMatrixDispatch::Mutable ? ir::ParamDirection::InOut : in},
                             {std::make_shared<ir::EvalStmt>(kernel_call, span)},
                             ir::FunctionType::Orchestration, ir::Level::CHIP, ir::Role::Orchestrator);
  Stmts host_body;
  ir::VarPtr argument = weight;
  if (dispatch == HostMatrixDispatch::PartialSlice || dispatch == HostMatrixDispatch::FullSlice) {
    const auto slice = registry.Create(
        "tensor.slice", {weight, tuple(device_type->shape_), tuple({index(0), index(0)})}, {}, span);
    argument = std::make_shared<ir::Var>("view", slice->GetType(), span);
    host_body.push_back(std::make_shared<ir::AssignStmt>(argument, slice, span));
  }
  std::vector<ir::VarPtr> params{weight};
  std::vector<ir::ParamDirection> directions{in};
  Attrs attrs;
  if (dispatch == HostMatrixDispatch::UnknownRank) {
    const auto rank =
        std::make_shared<ir::Var>("rank", std::make_shared<ir::ScalarType>(DataType::INDEX), span);
    params.push_back(rank);
    directions.push_back(in);
    attrs.emplace_back(ir::kAttrDevice, ir::ExprPtr{rank});
  } else if (dispatch == HostMatrixDispatch::RankZero) {
    attrs.emplace_back(ir::kAttrDevice, ir::ExprPtr{index(0)});
  }
  const auto forward = std::make_shared<ir::Var>("forwarded", argument->GetType(), span);
  host_body.push_back(std::make_shared<ir::AssignStmt>(forward, argument, span));
  const auto chip_call =
      std::make_shared<ir::Call>(std::make_shared<ir::GlobalVar>(chip->name_), Exprs{forward}, Attrs{},
                                 std::move(attrs), ir::GetUnknownType(), span);
  host_body.push_back(std::make_shared<ir::EvalStmt>(chip_call, span));
  const auto host = function("host", std::move(params), std::move(directions), std::move(host_body),
                             ir::FunctionType::Opaque, ir::Level::HOST, ir::Role::Orchestrator);
  const auto program =
      std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel, chip, host}, "test", span);
  ir::pass::AutoLayoutOptions options;
  options.nz_inputs = {"weight"};
  options.reusable_tensors = {"weight"};
  options.cost_model.host_world_size = world_size;
  options.cost_model.backend_type = backend;
  // Force NZ where legal so this checks writeback as well as extraction.
  options.cost_model.gm_pack_fixed_cycles = 0;
  options.cost_model.gm_pack_bytes_per_cycle = 1e12;
  const auto rewritten = ir::pass::AutoLayout(options)(program);
  const auto host_out = rewritten->GetFunction("host");
  assert(al::detail::TensorTypeLayout(ir::As<ir::TensorType>(host_out->params_[0]->GetType())) ==
         ir::TensorLayout::ND);
  assert(host_out->GetAttr<std::string>(ir::kAttrAutoLayoutHostInputs) ==
         (expect_selectable ? "v1;0=NZ" : "v1;0=UNSUPPORTED"));
  const auto host_statements = ir::As<ir::SeqStmts>(host_out->body_);
  const auto emitted_call = ir::As<ir::Call>(ir::As<ir::EvalStmt>(host_statements->stmts_.back())->expr_);
  const auto rank = emitted_call->GetAttr<ir::ExprPtr>(ir::kAttrDevice, nullptr);
  if (expect_selectable) {
    assert(ir::IsConstValue(rank, 0));
  } else {
    assert(rank.get() == chip_call->GetAttr<ir::ExprPtr>(ir::kAttrDevice, nullptr).get());
  }
  const auto chip_out = rewritten->GetFunction("chip");
  const auto chip_type = ir::As<ir::TensorType>(chip_out->params_[0]->GetType());
  assert(al::detail::TensorTypeLayout(chip_type) ==
         (expect_selectable ? ir::TensorLayout::NZ : ir::TensorLayout::ND));
}

void TestHostInputProjectionCost(bool same_rank, int64_t layers, int64_t configured_ranks = 0,
                                 int64_t bank_ranks = 4) {
  const auto span = ir::Span::unknown();
  using Exprs = std::vector<ir::ExprPtr>;
  using Stmts = std::vector<ir::StmtPtr>;
  using Attrs = std::vector<std::pair<std::string, std::any>>;
  auto index = [&](int64_t n) -> ir::ExprPtr {
    return std::make_shared<ir::ConstInt>(n, DataType::INDEX, span);
  };
  auto tuple = [&](Exprs values) { return std::make_shared<ir::MakeTuple>(std::move(values), span); };
  auto tensor = [&](std::vector<int64_t> dims) {
    return std::make_shared<ir::TensorType>(std::move(dims), DataType::FP16, std::nullopt);
  };
  const auto matrix_type = tensor({64, 32});
  const auto formal = std::make_shared<ir::Var>("matrix", matrix_type, span);
  const auto shard_type = layers > 1 ? tensor({layers, 64, 32}) : matrix_type;
  const auto chip_param = std::make_shared<ir::Var>("shard", shard_type, span);
  const auto bank_type = layers > 1 ? tensor({bank_ranks, layers, 64, 32}) : tensor({bank_ranks, 64, 32});
  const auto bank = std::make_shared<ir::Var>("weight", bank_type, span);
  const auto& registry = ir::OpRegistry::GetInstance();
  auto select = [&](const ir::VarPtr& source, Exprs shape, Exprs offsets) {
    return registry.Create(
        "tensor.slice",
        {source, tuple(std::move(shape)), tuple(std::move(offsets)), tuple({}), tuple({index(0)})}, {}, span);
  };
  auto load = registry.Create(
      "tile.load",
      {formal, tuple({index(0), index(0)}), tuple(matrix_type->shape_), tuple(matrix_type->shape_)},
      {{"target_memory", ir::MemorySpace::Mat}}, span);
  auto loaded = std::make_shared<ir::Var>("loaded", load->GetType(), span);
  using FuncType = ir::FunctionType;
  auto function = [&](const char* name, const ir::VarPtr& param, Stmts body, FuncType type, ir::Level level,
                      ir::Role role) {
    return std::make_shared<ir::Function>(
        name, std::vector<ir::VarPtr>{param}, std::vector<ir::ParamDirection>{ir::ParamDirection::In},
        std::vector<ir::TypePtr>{}, std::make_shared<ir::SeqStmts>(std::move(body), span), span, type, level,
        role);
  };
  auto kernel = function("load_matrix", formal, {std::make_shared<ir::AssignStmt>(loaded, load, span)},
                         FuncType::InCore, ir::Level::CHIP_DIE, ir::Role::SubWorker);
  Stmts chip_body;
  for (int64_t layer = 0; layer < layers; ++layer) {
    auto matrix = chip_param;
    if (layers > 1) {
      const auto slice =
          select(chip_param, {index(1), index(64), index(32)}, {index(layer), index(0), index(0)});
      matrix = std::make_shared<ir::Var>("plane_" + std::to_string(layer), slice->GetType(), span);
      chip_body.push_back(std::make_shared<ir::AssignStmt>(matrix, slice, span));
    }
    auto call =
        std::make_shared<ir::Call>(std::make_shared<ir::GlobalVar>(kernel->name_), Exprs{matrix}, span);
    chip_body.push_back(std::make_shared<ir::EvalStmt>(call, span));
  }
  auto chip = function("chip", chip_param, std::move(chip_body), FuncType::Orchestration, ir::Level::CHIP,
                       ir::Role::Orchestrator);
  Stmts dispatches;
  auto dispatch = [&](const ir::ExprPtr& rank, int64_t i) {
    Exprs extents{index(1)};
    extents.insert(extents.end(), shard_type->shape_.begin(), shard_type->shape_.end());
    Exprs offsets(extents.size(), index(0));
    offsets[0] = rank;
    auto slice = select(bank, std::move(extents), std::move(offsets));
    auto shard = std::make_shared<ir::Var>("shard_" + std::to_string(i), slice->GetType(), span);
    dispatches.push_back(std::make_shared<ir::AssignStmt>(shard, slice, span));
    auto call =
        std::make_shared<ir::Call>(std::make_shared<ir::GlobalVar>(chip->name_), Exprs{shard}, Attrs{},
                                   Attrs{{ir::kAttrDevice, rank}}, ir::GetUnknownType(), span);
    dispatches.push_back(std::make_shared<ir::EvalStmt>(call, span));
  };
  if (configured_ranks > 0) {
    const auto world_size = registry.Create("pld.system.world_size", {}, {}, span);
    const auto stop = std::make_shared<ir::Var>("world_size", world_size->GetType(), span);
    const auto rank = std::make_shared<ir::Var>("rank", world_size->GetType(), span);
    dispatch(rank, 0);
    auto loop = std::make_shared<ir::ForStmt>(rank, index(0), stop, index(1), std::vector<ir::IterArgPtr>{},
                                              std::make_shared<ir::SeqStmts>(std::move(dispatches), span),
                                              std::vector<ir::VarPtr>{}, span);
    dispatches = {std::make_shared<ir::AssignStmt>(stop, world_size, span), loop};
  } else {
    for (int64_t i = 0; i < 2; ++i) dispatch(index(same_rank ? 0 : i), i);
  }
  auto host = function("host", bank, std::move(dispatches), FuncType::Opaque, ir::Level::HOST,
                       ir::Role::Orchestrator);
  auto program =
      std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel, chip, host}, "test", span);
  al::LayoutCostModel model;
  model.expected_uses = 8;
  if (configured_ranks > 0) model.host_world_size = configured_ranks;
  auto instance = al::ExtractLayoutInstance(program, model, {"weight"}, {"weight"});
  const auto nd = al::LayoutSpec::ND();
  const auto nz = al::LayoutSpec::NZ();
  if (configured_ranks > 0 && configured_ranks != bank_ranks) {
    assert(instance.values.at("weight").admissible_layouts == std::vector<al::LayoutSpec>{nd});
    return;
  }
  constexpr double matrix_bytes = 64 * 32 * 2;
  assert(instance.values.at("weight").ByteSize() == bank_ranks * layers * matrix_bytes);
  assert(instance.operations.size() == 1);
  const auto& op = instance.operations.front();
  const int64_t launches = configured_ranks > 0 ? configured_ranks : 2;
  assert(op.invocations == launches * layers * model.expected_uses);
  assert(op.projected_mte2_source_bytes == (same_rank ? 1 : launches) * layers * matrix_bytes);
  const auto solutions = EnumerateAssignments(instance);
  assert(solutions.size() == 2);
  for (const auto& solution : solutions) {
    const auto selected = solution.assignment.value_layouts.at("weight");
    const double pack = selected == nz ? model.gm_pack_fixed_cycles + bank_ranks * layers * matrix_bytes /
                                                                          model.gm_pack_bytes_per_cycle
                                       : 0;
    assert(std::abs(solution.conversion_cost_cycles - pack) < 1e-9);
    const double loads =
        layers * model.expected_uses * (same_rank ? 1 + model.mte2_l2_reread_discount : launches);
    const double bandwidth =
        selected == nd ? model.mte2_nd2nz_contiguous_bytes_per_cycle : model.mte2_nz2nz_bytes_per_cycle;
    assert(std::abs(solution.operator_cost_cycles - loads * matrix_bytes / bandwidth) < 1e-9);
  }
  assert(solutions[0].assignment.value_layouts.at("weight") !=
         solutions[1].assignment.value_layouts.at("weight"));
}

void TestDistinctTileDefinitionsWithRepeatedHints() {
  const auto span = ir::Span::unknown();
  const auto width = std::make_shared<ir::ConstInt>(16, DataType::INDEX, span);
  const auto shape = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{width, width}, span);
  const auto& registry = ir::OpRegistry::GetInstance();
  std::vector<ir::StmtPtr> stmts;
  auto bind = [&](const std::string& hint, const ir::CallPtr& call) {
    auto var = std::make_shared<ir::Var>(hint, call->GetType(), span);
    stmts.push_back(std::make_shared<ir::AssignStmt>(var, call, span));
    return var;
  };
  auto full = [&]() {
    auto fill = std::make_shared<ir::ConstInt>(1, DataType::INT32, span);
    return registry.Create("tile.full", {shape, fill}, {{"dtype", DataType::INT32}}, span);
  };
  bind("t_3", full());  // User-provided hints can also look like generated names.
  auto first = bind("t", full());
  auto second = bind("t", full());
  for (int i = 0; i < 2; ++i) {
    bind("t", registry.Create("tile.create", {shape},
                              {{"dtype", DataType::INT32}, {"target_memory", ir::MemorySpace::Vec}}, span));
  }
  bind("sum", registry.Create("tile.add", {first, second}, {}, span));
  stmts.push_back(std::make_shared<ir::ReturnStmt>(span));
  auto func = std::make_shared<ir::Function>("repeated_hints", std::vector<ir::VarPtr>{},
                                             std::vector<ir::ParamDirection>{}, std::vector<ir::TypePtr>{},
                                             std::make_shared<ir::SeqStmts>(stmts, span), span,
                                             ir::FunctionType::InCore);
  auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{func}, "test", span);
  const auto instance = al::ExtractLayoutInstance(program);
  assert(instance.values.size() == 6);
  assert(instance.operations.size() == 4);
  for (const auto& op : instance.operations) {
    assert(instance.values.at(op.output).producer_op_id == op.op_id);
  }
  const auto& inputs = instance.operations.back().inputs;
  assert((inputs == std::vector<std::string>{instance.operations[1].output, instance.operations[2].output}));
  assert(instance.operations[1].output != instance.operations[2].output);
}

void TestValueOnlyWritebackWithRepeatedHints(const std::string& first_hint, const std::string& second_hint) {
  const auto span = ir::Span::unknown();
  const std::vector<int64_t> dims{32, 32};
  const auto dn = std::make_shared<ir::TensorType>(
      dims, DataType::FP32, std::nullopt, ir::TensorView(std::vector<int64_t>{1, 32}, ir::TensorLayout::DN));
  const auto nd = std::make_shared<ir::TensorType>(dims, DataType::FP32, std::nullopt);
  const auto shape = std::make_shared<ir::MakeTuple>(nd->shape_, span);
  const auto zero = std::make_shared<ir::ConstInt>(0, DataType::INDEX, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{zero, zero}, span);
  auto first = std::make_shared<ir::Var>("first", dn, span);
  auto second = std::make_shared<ir::Var>("second", nd, span);
  auto load = [&](const ir::VarPtr& input, const std::string& hint) {
    auto call = ir::OpRegistry::GetInstance().Create("tile.load", {input, offsets, shape, shape},
                                                     {{"target_memory", ir::MemorySpace::Vec}}, span);
    return std::make_shared<ir::AssignStmt>(std::make_shared<ir::Var>(hint, call->GetType(), span), call,
                                            span);
  };
  auto body = std::make_shared<ir::SeqStmts>(
      std::vector<ir::StmtPtr>{load(first, first_hint), load(second, second_hint),
                               std::make_shared<ir::ReturnStmt>(span)},
      span);
  auto kernel =
      std::make_shared<ir::Function>("kernel", std::vector<ir::VarPtr>{first, second},
                                     std::vector<ir::ParamDirection>(2, ir::ParamDirection::In),
                                     std::vector<ir::TypePtr>{}, body, span, ir::FunctionType::InCore);
  auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "probe", span);
  auto assignment = al::SolveAutoLayout(al::ExtractLayoutInstance(program)).WritebackAssignment();
  for (bool values_only : {false, true}) {
    if (values_only) assignment.op_configs.clear();
    const auto result = al::ApplyLayoutAssignment(program, assignment);
    auto loads = ir::As<ir::SeqStmts>(result->GetFunction("kernel")->body_);
    auto layout = [&](size_t i) {
      auto tile = ir::As<ir::AssignStmt>(loads->stmts_.at(i))->var_;
      return al::EffectiveTileLayout(ir::As<ir::TileType>(tile->GetType()));
    };
    assert(layout(0) == al::LayoutSpec::ColMajorFlat());
    assert(layout(1) == al::LayoutSpec::RowMajorFlat());
  }
}

void TestGenericVecOutputLayoutAssignment() {
  const auto span = ir::Span::unknown();
  const std::vector<int64_t> dims{32, 64};
  ir::TileView column;
  column.blayout = ir::TileLayout::col_major;
  const auto col = al::LayoutSpec::ColMajorFlat();
  for (const auto& [op_name, column_input, accepts_column] :
       std::array<std::tuple<std::string, bool, bool>, 3>{
           {{"tile.fillpad", true, true}, {"tile.fillpad", false, false}, {"tile.abs", true, false}}}) {
    const auto input_type = std::make_shared<ir::TileType>(
        dims, DataType::FP32, std::nullopt, column_input ? column : ir::TileView{}, ir::MemorySpace::Vec);
    const auto input = std::make_shared<ir::Var>("input", input_type, span);
    const auto call = ir::OpRegistry::GetInstance().Create(op_name, {input}, {}, span);
    const auto result = std::make_shared<ir::Var>("result", call->GetType(), span);
    const auto body = std::make_shared<ir::AssignStmt>(result, call, span);
    const auto kernel = std::make_shared<ir::Function>(
        "kernel", std::vector<ir::VarPtr>{input}, std::vector<ir::ParamDirection>{ir::ParamDirection::In},
        std::vector<ir::TypePtr>{}, body, span, ir::FunctionType::InCore);
    const auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "generic", span);
    const auto instance = al::ExtractLayoutInstance(program);
    assert(instance.operations.size() == 1);
    const auto& op = instance.operations.front();
    assert(std::any_of(op.configs.begin(), op.configs.end(),
                       [&](const auto& cfg) { return cfg.output_layout == col; }) == accepts_column);
    al::LayoutAssignment assignment;
    assignment.value_layouts.emplace(op.output, col);
    for (bool supply_config : {false, true}) {
      if (supply_config) {
        auto cfg = op.configs.front();
        cfg.output_layout = col;
        assignment.op_configs.emplace(op.op_id, cfg);
      }
      try {
        const auto written = al::ApplyLayoutAssignment(program, assignment);
        assert(accepts_column && "unsupported output layout silently accepted");
        const auto stmt = ir::As<ir::AssignStmt>(written->GetFunction("kernel")->body_);
        const auto emitted = ir::As<ir::Call>(stmt->value_);
        assert(emitted->args_.front() == input);
        assert(al::EffectiveTileLayout(ir::As<ir::TileType>(emitted->GetType())) == col);
        assert(al::EffectiveTileLayout(ir::As<ir::TileType>(stmt->var_->GetType())) == col);
      } catch (const pypto::ValueError& error) {
        assert(!accepts_column);
        assert(std::string(error.what()).find("for 'result' at " + op_name) != std::string::npos);
      }
    }

    if (op_name == "tile.abs") {
      const auto row = al::LayoutSpec::RowMajorFlat();
      al::LayoutAssignment compatible;
      compatible.value_layouts.emplace(op.output, row);
      const auto written = al::ApplyLayoutAssignment(program, compatible);
      const auto seq = ir::As<ir::SeqStmts>(written->GetFunction("kernel")->body_);
      assert(seq && seq->stmts_.size() > 1);
      const auto stmt = ir::As<ir::AssignStmt>(seq->stmts_.back());
      const auto emitted = ir::As<ir::Call>(stmt->value_);
      assert(emitted->op_ == call->op_ && emitted->args_.front() != input);
      assert(al::EffectiveTileLayout(ir::As<ir::TileType>(emitted->args_.front()->GetType())) == row);
      assert(al::EffectiveTileLayout(ir::As<ir::TileType>(stmt->var_->GetType())) == row);
    }

    const auto& producer = op.configs.front();
    assignment.op_configs.at(op.op_id) = producer;
    assignment.value_layouts.at(op.output) =
        producer.output_layout == col ? al::LayoutSpec::RowMajorFlat() : col;
    try {
      al::ApplyLayoutAssignment(program, assignment);
      assert(false && "conflicting value and producer layouts silently accepted");
    } catch (const pypto::ValueError& error) {
      assert(std::string(error.what()).find("contradicts its producer configuration") != std::string::npos);
    }
  }
}

void TestStoreBindingAcrossUnmodeledStore(bool return_store) {
  const auto span = ir::Span::unknown();
  const std::vector<int64_t> dims{32, 32};
  const auto nd = std::make_shared<ir::TensorType>(dims, DataType::FP32, std::nullopt);
  const auto dn = std::make_shared<ir::TensorType>(
      dims, DataType::FP32, std::nullopt, ir::TensorView(std::vector<int64_t>{1, 32}, ir::TensorLayout::DN));
  const auto input = std::make_shared<ir::Var>("input", nd, span);
  const auto mask_output = std::make_shared<ir::Var>("mask_output", nd, span);
  const auto data_output = std::make_shared<ir::Var>("data_output", dn, span);
  const auto shape = std::make_shared<ir::MakeTuple>(nd->shape_, span);
  const auto zero = std::make_shared<ir::ConstInt>(0, DataType::INDEX, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{zero, zero}, span);
  const auto& registry = ir::OpRegistry::GetInstance();
  const auto load = registry.Create("tile.load", {input, offsets, shape, shape},
                                    {{"target_memory", ir::MemorySpace::Vec}}, span);
  const auto data = std::make_shared<ir::Var>("data", load->GetType(), span);
  const auto diagonal = std::make_shared<ir::ConstInt>(0, DataType::INT32, span);
  const auto tri = registry.Create("tile.tri", {diagonal, shape}, {{"dtype", DataType::FP32}}, span);
  const auto mask = std::make_shared<ir::Var>("mask", tri->GetType(), span);
  const auto mask_store = registry.Create("tile.store", {mask, offsets, mask_output}, {}, span);
  const auto data_store = registry.Create("tile.store", {data, offsets, data_output}, {}, span);
  std::vector<ir::StmtPtr> stmts{std::make_shared<ir::AssignStmt>(data, load, span),
                                 std::make_shared<ir::AssignStmt>(mask, tri, span),
                                 std::make_shared<ir::EvalStmt>(mask_store, span)};
  if (!return_store) stmts.push_back(std::make_shared<ir::EvalStmt>(data_store, span));
  stmts.push_back(std::make_shared<ir::ReturnStmt>(
      std::vector<ir::ExprPtr>{return_store ? ir::ExprPtr(data_store) : data_output}, span));
  const auto kernel = std::make_shared<ir::Function>(
      "kernel", std::vector<ir::VarPtr>{input, mask_output, data_output},
      std::vector<ir::ParamDirection>{ir::ParamDirection::In, ir::ParamDirection::Out,
                                      ir::ParamDirection::Out},
      std::vector<ir::TypePtr>{dn}, std::make_shared<ir::SeqStmts>(stmts, span), span,
      ir::FunctionType::InCore);
  const auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "stores", span);
  // The extractor models the ND load and its DN store. The intervening ND
  // mask store must receive the mask itself, even when it precedes a returned store.
  const auto instance = al::ExtractLayoutInstance(program);
  assert(instance.operations.size() == 2);
  for (bool values_only : {false, true}) {
    auto assignment = al::SolveAutoLayout(instance).WritebackAssignment();
    if (values_only) assignment.op_configs.clear();
    auto rewritten = al::ApplyLayoutAssignment(program, assignment);
    const auto body = ir::As<ir::SeqStmts>(rewritten->GetFunction("kernel")->body_);
    size_t stores = 0;
    for (const auto& stmt : body->stmts_) {
      ir::CallPtr store;
      if (const auto eval = ir::As<ir::EvalStmt>(stmt)) store = ir::As<ir::Call>(eval->expr_);
      if (const auto ret = ir::As<ir::ReturnStmt>(stmt)) store = ir::As<ir::Call>(ret->value_.front());
      if (!ir::IsOp(store, "tile.store")) continue;
      if (stores++ == 0) {
        assert(store->args_[0] == mask);
        assert(store->args_[2] == mask_output);
      } else {
        assert(al::EffectiveTileLayout(ir::As<ir::TileType>(store->args_[0]->GetType())) ==
               al::LayoutSpec::ColMajorFlat());
        assert(store->args_[2] == data_output);
      }
    }
    assert(stores == 2);
  }
}

void TestCallSiteAndFunctionNamesKeepTheirOwnLayouts(bool values_only) {
  const auto span = ir::Span::unknown();
  const auto tensor =
      std::make_shared<ir::TensorType>(std::vector<int64_t>{32, 32}, DataType::FP32, std::nullopt);
  const auto shape = std::make_shared<ir::MakeTuple>(tensor->shape_, span);
  const auto zero = std::make_shared<ir::ConstInt>(0, DataType::INDEX, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{zero, zero}, span);
  const auto& registry = ir::OpRegistry::GetInstance();
  auto make_kernel = [&](const std::string& name) {
    auto input = std::make_shared<ir::Var>("input", tensor, span);
    auto load = registry.Create("tile.load", {input, offsets, shape, shape},
                                {{"target_memory", ir::MemorySpace::Vec}}, span);
    auto loaded = std::make_shared<ir::Var>("loaded", load->GetType(), span);
    auto create = registry.Create("tensor.create", {shape}, {{"dtype", DataType::FP32}}, span);
    auto scratch = std::make_shared<ir::Var>("scratch", create->GetType(), span);
    auto scratch_load = registry.Create("tile.load", {scratch, offsets, shape, shape},
                                        {{"target_memory", ir::MemorySpace::Vec}}, span);
    auto second = std::make_shared<ir::Var>("loaded", scratch_load->GetType(), span);
    auto move = registry.Create("tile.move", {second}, {{"target_memory", ir::MemorySpace::Vec}}, span);
    auto moved = std::make_shared<ir::Var>("moved", move->GetType(), span);
    auto body = std::make_shared<ir::SeqStmts>(
        std::vector<ir::StmtPtr>{std::make_shared<ir::AssignStmt>(loaded, load, span),
                                 std::make_shared<ir::AssignStmt>(scratch, create, span),
                                 std::make_shared<ir::AssignStmt>(second, scratch_load, span),
                                 std::make_shared<ir::AssignStmt>(moved, move, span),
                                 std::make_shared<ir::ReturnStmt>(span)},
        span);
    return std::make_shared<ir::Function>(name, std::vector<ir::VarPtr>{input},
                                          std::vector<ir::ParamDirection>{ir::ParamDirection::In},
                                          std::vector<ir::TypePtr>{}, body, span, ir::FunctionType::InCore);
  };
  const auto kernel = make_kernel("kernel");
  const auto other_kernel = make_kernel("kernel_2");
  const auto reserved_symbol = make_kernel("kernel_site2");
  std::vector<ir::StmtPtr> stmts;
  auto add_site = [&](const std::string& callee, const std::string& allocation) {
    auto create = registry.Create("tensor.create", {shape}, {{"dtype", DataType::FP32}}, span);
    auto buffer = std::make_shared<ir::Var>(allocation, create->GetType(), span);
    stmts.push_back(std::make_shared<ir::AssignStmt>(buffer, create, span));
    auto call = std::make_shared<ir::Call>(std::make_shared<ir::GlobalVar>(callee),
                                           std::vector<ir::ExprPtr>{buffer}, span);
    stmts.push_back(std::make_shared<ir::EvalStmt>(call, span));
  };
  add_site("kernel", "first");
  add_site("kernel", "second");
  add_site("kernel", "column_source");
  add_site("kernel_2", "independent");
  add_site("kernel_site2", "reserved");
  add_site("kernel", "tile_variant");
  stmts.push_back(std::make_shared<ir::ReturnStmt>(span));
  const auto entry = std::make_shared<ir::Function>(
      "entry", std::vector<ir::VarPtr>{}, std::vector<ir::ParamDirection>{}, std::vector<ir::TypePtr>{},
      std::make_shared<ir::SeqStmts>(stmts, span), span, ir::FunctionType::Orchestration);
  const auto program = std::make_shared<ir::Program>(
      std::vector<ir::FunctionPtr>{kernel, other_kernel, reserved_symbol, entry}, "call_sites", span);
  const auto instance = al::ExtractLayoutInstance(program);

  // Select different legal configurations with the same assignment hint at
  // site 2 of "kernel" and in the separately declared "kernel_2".
  assert(instance.operations.size() == 18);
  al::LayoutAssignment assignment;
  const auto row = al::LayoutSpec::RowMajorFlat();
  const auto col = al::LayoutSpec::ColMajorFlat();
  // The fourth site uses the same ND parameter and scratch layouts as the
  // first, but selects a column-major representation for its private move.
  const auto tile_variant = al::detail::TileLayoutPrefix(program, "kernel", true, 3) + "moved";
  for (const auto& op : instance.operations) {
    const bool move = op.op_name == "tile.move";
    assert((move || op.op_name == "tile.load") && op.inputs.size() == 1);
    const auto& source = op.inputs.front();
    const bool column = source == "column_source" || op.output == tile_variant;
    const auto output = column ? col : row;
    const auto input = move ? output : (column ? al::LayoutSpec::DN() : al::LayoutSpec::ND());
    const auto config = std::find_if(op.configs.begin(), op.configs.end(), [&](const auto& cfg) {
      return cfg.input_layouts.front() == input && cfg.output_layout == output;
    });
    assert(config != op.configs.end());
    assignment.op_configs.emplace(op.op_id, *config);
    assignment.value_layouts.emplace(source, input);
    assignment.value_layouts.emplace(op.output, output);
  }
  if (values_only) assignment.op_configs.clear();
  const auto rewritten = al::ApplyLayoutAssignment(program, assignment);
  const auto body = ir::As<ir::SeqStmts>(rewritten->GetFunction("entry")->body_);
  std::vector<std::string> callees;
  callees.reserve(6);
  for (const auto& stmt : body->stmts_) {
    if (auto eval = ir::As<ir::EvalStmt>(stmt)) {
      auto call = ir::As<ir::Call>(eval->expr_);
      callees.push_back(ir::As<ir::GlobalVar>(call->op_)->name_);
    }
  }
  assert(callees.size() == 6);
  assert(callees[0] == "kernel" && callees[1] == "kernel");
  assert(callees[2] != "kernel" && callees[2] != "kernel_2" && callees[2] != "kernel_site2");
  assert(callees[3] == "kernel_2" && callees[4] == "kernel_site2");
  assert(callees[5] != callees[0] && callees[5] != callees[2]);

  auto load_layout = [&](const std::string& callee) {
    const auto seq = ir::As<ir::SeqStmts>(rewritten->GetFunction(callee)->body_);
    // Even when the private move materializes a transpose, the scratch load
    // itself retains the row-major layout of its ND source.
    const auto moved = ir::As<ir::AssignStmt>(seq->stmts_[seq->stmts_.size() - 2]);
    assert(moved->var_->name_hint_ == "moved");
    assert(al::EffectiveTileLayout(ir::As<ir::TileType>(moved->var_->GetType())) ==
           (callee == callees[5] ? col : row));
    const auto second = ir::As<ir::AssignStmt>(seq->stmts_.at(2));
    assert(al::EffectiveTileLayout(ir::As<ir::TileType>(second->var_->GetType())) == row);
    const auto load = ir::As<ir::AssignStmt>(seq->stmts_.front());
    assert(ir::IsOp(ir::As<ir::Call>(load->value_), "tile.load"));
    return al::EffectiveTileLayout(ir::As<ir::TileType>(load->var_->GetType()));
  };
  assert(load_layout(callees[0]) == row);
  assert(load_layout(callees[2]) == col);
  assert(load_layout(callees[3]) == row);
  assert(load_layout(callees[4]) == row);
  assert(load_layout(callees[5]) == row);
}

void TestValueOnlyJoinCapacity(int64_t columns, bool matching_branches,
                               std::optional<size_t> expected_transposes) {
  const auto span = ir::Span::unknown();
  const auto tensor =
      std::make_shared<ir::TensorType>(std::vector<int64_t>{32, columns}, DataType::FP32, std::nullopt);
  const auto shape = std::make_shared<ir::MakeTuple>(tensor->shape_, span);
  const auto zero = std::make_shared<ir::ConstInt>(0, DataType::INDEX, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{zero, zero}, span);
  std::vector<ir::VarPtr> params;
  std::vector<ir::StmtPtr> stmts;
  auto load = [&](const std::string& name) {
    auto input = std::make_shared<ir::Var>(name, tensor, span);
    params.push_back(input);
    auto call = ir::OpRegistry::GetInstance().Create("tile.load", {input, offsets, shape, shape},
                                                     {{"target_memory", ir::MemorySpace::Vec}}, span);
    auto tile = std::make_shared<ir::Var>(name + "_tile", call->GetType(), span);
    stmts.push_back(std::make_shared<ir::AssignStmt>(tile, call, span));
    return tile;
  };
  const auto a = load("a");
  const auto b = load("b");
  const auto result = std::make_shared<ir::Var>("result", a->GetType(), span);
  const auto condition =
      std::make_shared<ir::Var>("condition", std::make_shared<ir::ScalarType>(DataType::BOOL), span);
  params.push_back(condition);
  stmts.push_back(std::make_shared<ir::IfStmt>(
      condition, std::make_shared<ir::YieldStmt>(std::vector<ir::ExprPtr>{a}, span),
      std::make_shared<ir::YieldStmt>(std::vector<ir::ExprPtr>{b}, span), std::vector<ir::VarPtr>{result},
      span));
  stmts.push_back(std::make_shared<ir::ReturnStmt>(span));
  const auto kernel = std::make_shared<ir::Function>(
      "kernel", params, std::vector<ir::ParamDirection>(params.size(), ir::ParamDirection::In),
      std::vector<ir::TypePtr>{}, std::make_shared<ir::SeqStmts>(stmts, span), span,
      ir::FunctionType::InCore);
  const auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "join", span);

  // Omit the result layout. Different branches fall back to the row-major
  // result. Two column-major branches can carry their common layout directly.
  al::LayoutAssignment assignment;
  assignment.value_layouts.emplace("a", al::LayoutSpec::DN());
  if (matching_branches) assignment.value_layouts.emplace("b", al::LayoutSpec::DN());
  ir::ProgramPtr rewritten;
  try {
    rewritten = al::ApplyLayoutAssignment(program, assignment);
  } catch (const pypto::ValueError& e) {
    assert(!expected_transposes && std::string(e.what()).find("Vec capacity") != std::string::npos);
    return;
  }
  assert(expected_transposes);
  const auto body = ir::As<ir::SeqStmts>(rewritten->GetFunction("kernel")->body_);
  size_t transposes = 0;
  for (const auto& stmt : body->stmts_) {
    if (const auto assign = ir::As<ir::AssignStmt>(stmt);
        assign && ir::IsOp(ir::As<ir::Call>(assign->value_), "tile.transpose")) {
      ++transposes;
    }
  }
  assert(transposes == *expected_transposes);
  const auto join = ir::As<ir::IfStmt>(body->stmts_[body->stmts_.size() - 2]);
  assert(al::EffectiveTileLayout(ir::As<ir::TileType>(join->return_vars_.front()->GetType())) ==
         (matching_branches ? al::LayoutSpec::ColMajorFlat() : al::LayoutSpec::RowMajorFlat()));
}

enum class VecPressureScope { StraightLine, ForLoop, WhileLoop, Reserved, UnmodeledGenerator };

void TestVecCopyLifetime(int64_t a_columns, int64_t b_columns, bool overlap, bool convert_b,
                         size_t expected_copies, VecPressureScope scope = VecPressureScope::StraightLine) {
  const auto span = ir::Span::unknown();
  const auto& registry = ir::OpRegistry::GetInstance();
  const auto zero = std::make_shared<ir::ConstInt>(0, DataType::INDEX, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{zero, zero}, span);
  std::vector<ir::VarPtr> params;
  std::vector<ir::ParamDirection> directions;
  std::vector<ir::StmtPtr> allocations;
  std::vector<ir::ExprPtr> arguments;
  auto make_tile = [&](const std::string& name, int64_t columns) {
    const auto type =
        std::make_shared<ir::TensorType>(std::vector<int64_t>{32, columns}, DataType::FP32, std::nullopt);
    const auto shape = std::make_shared<ir::MakeTuple>(type->shape_, span);
    auto input = std::make_shared<ir::Var>(name + "_input", type, span);
    auto output = std::make_shared<ir::Var>(name + "_output", type, span);
    params.insert(params.end(), {input, output});
    directions.insert(directions.end(), {ir::ParamDirection::InOut, ir::ParamDirection::Out});
    auto load = registry.Create("tile.load", {input, offsets, shape, shape},
                                {{"target_memory", ir::MemorySpace::Vec}}, span);
    if (scope == VecPressureScope::UnmodeledGenerator && name == "b") {
      auto diagonal = std::make_shared<ir::ConstInt>(0, DataType::INT32, span);
      load = registry.Create("tile.tri", {diagonal, shape}, {{"dtype", DataType::FP32}}, span);
    }
    auto tile = std::make_shared<ir::Var>(name, load->GetType(), span);
    auto store = registry.Create("tile.store", {tile, offsets, output}, {}, span);
    auto store_original = registry.Create("tile.store", {tile, offsets, input}, {}, span);
    for (const auto& suffix : {"_source", "_buffer"}) {
      auto alloc = registry.Create("tensor.create", {shape}, {{"dtype", DataType::FP32}}, span);
      auto gm = std::make_shared<ir::Var>(name + suffix, alloc->GetType(), span);
      allocations.push_back(std::make_shared<ir::AssignStmt>(gm, alloc, span));
      arguments.push_back(gm);
    }
    return std::array<ir::StmtPtr, 3>{std::make_shared<ir::AssignStmt>(tile, load, span),
                                      std::make_shared<ir::EvalStmt>(store, span),
                                      std::make_shared<ir::EvalStmt>(store_original, span)};
  };
  const auto [load_a, store_a, store_original_a] = make_tile("a", a_columns);
  const auto [load_b, store_b, store_original_b] = make_tile("b", b_columns);
  std::vector<ir::StmtPtr> stmts;
  if (scope == VecPressureScope::Reserved) {
    // Tile storage starts after the end of the reservation, including its explicit base.
    auto reserve =
        registry.Create("system.reserve_buffer", {},
                        {{"name", std::string("pipe")}, {"size", 8 * 1024}, {"base", 16 * 1024}}, span);
    stmts.push_back(std::make_shared<ir::EvalStmt>(reserve, span));
  }
  stmts.push_back(load_a);
  std::vector<ir::StmtPtr> uses;
  if (overlap) uses.push_back(load_b);
  uses.insert(uses.end(), {store_a, store_original_a});
  if (!overlap) uses.push_back(load_b);
  uses.insert(uses.end(), {store_b, store_original_b});
  if (scope == VecPressureScope::ForLoop || scope == VecPressureScope::WhileLoop) {
    auto body = std::make_shared<ir::SeqStmts>(uses, span);
    if (scope == VecPressureScope::ForLoop) {
      auto one = std::make_shared<ir::ConstInt>(1, DataType::INDEX, span);
      auto two = std::make_shared<ir::ConstInt>(2, DataType::INDEX, span);
      auto induction = std::make_shared<ir::Var>("i", zero->GetType(), span);
      stmts.push_back(std::make_shared<ir::ForStmt>(induction, zero, two, one, std::vector<ir::IterArgPtr>{},
                                                    body, std::vector<ir::VarPtr>{}, span));
    } else {
      auto condition = std::make_shared<ir::ConstBool>(true, span);
      stmts.push_back(std::make_shared<ir::WhileStmt>(condition, std::vector<ir::IterArgPtr>{}, body,
                                                      std::vector<ir::VarPtr>{}, span));
    }
  } else {
    stmts.insert(stmts.end(), uses.begin(), uses.end());
  }
  stmts.push_back(std::make_shared<ir::ReturnStmt>(span));
  const auto kernel = std::make_shared<ir::Function>("kernel", params, directions, std::vector<ir::TypePtr>{},
                                                     std::make_shared<ir::SeqStmts>(stmts, span), span,
                                                     ir::FunctionType::InCore);
  auto call = std::make_shared<ir::Call>(std::make_shared<ir::GlobalVar>("kernel"), arguments, span);
  allocations.push_back(std::make_shared<ir::EvalStmt>(call, span));
  allocations.push_back(std::make_shared<ir::ReturnStmt>(span));
  const auto entry = std::make_shared<ir::Function>(
      "entry", std::vector<ir::VarPtr>{}, std::vector<ir::ParamDirection>{}, std::vector<ir::TypePtr>{},
      std::make_shared<ir::SeqStmts>(allocations, span), span, ir::FunctionType::Orchestration);
  const auto program =
      std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel, entry}, "pressure", span);
  auto instance = al::ExtractLayoutInstance(program);
  if (scope == VecPressureScope::UnmodeledGenerator) {
    // B is only used by skipped stores. It still owns storage at the last modeled store of A.
    assert(!instance.FindValue("b"));
    const auto& region = instance.vec_memory_regions.at(0);
    assert(region.original_bytes.at(region.operation_positions.back()) == 32 * 4 * (a_columns + b_columns));
  }
  const auto row = al::LayoutSpec::RowMajorFlat();
  const auto col = al::LayoutSpec::ColMajorFlat();
  assert(instance.cost_model.SupportsVecTileConversion(instance.values.at("a"), row, col));
  if (convert_b) {
    assert(instance.cost_model.SupportsVecTileConversion(instance.values.at("b"), row, col));
  }
  // Make the transpose free and make the store prefer it when capacity allows.
  al::LayoutAssignment preferred;
  for (auto& op : instance.operations) {
    if (op.op_name == "tile.load") {
      for (auto& cfg : op.configs) cfg.op_cost_cycles = cfg.output_layout == row ? 0 : 10000;
    }
    if (op.op_name != "tile.store") continue;
    const auto& tile = op.inputs.front();
    const auto target = op.inputs.size() == 1 && (tile == "a" || convert_b) ? col : row;
    instance.custom_conversion_costs[tile][{row, col}] = 0;
    for (auto& cfg : op.configs) cfg.op_cost_cycles = cfg.input_layouts.front() == target ? 0 : 1000;
    const auto selected = std::find_if(op.configs.begin(), op.configs.end(),
                                       [&](const auto& cfg) { return cfg.input_layouts.front() == target; });
    assert(selected != op.configs.end());
    preferred.op_configs.emplace(op.op_id, *selected);
  }
  const auto conflicts = al::CheckVecConversionCapacity(instance, preferred);
  const size_t requested_copies = convert_b ? 2 : 1;
  assert(conflicts.size() == requested_copies - expected_copies);
  // A values-only caller can request the same copies by choosing DN for the
  // destination buffers. Check the inferred store demands before returning IR.
  al::LayoutAssignment only_buffers;
  only_buffers.value_layouts.emplace("a_buffer", al::LayoutSpec::DN());
  if (convert_b) only_buffers.value_layouts.emplace("b_buffer", al::LayoutSpec::DN());
  bool rejected = false;
  try {
    al::ApplyLayoutAssignment(program, only_buffers);
  } catch (const pypto::ValueError& e) {
    assert(std::string(e.what()).find("Vec capacity") != std::string::npos);
    rejected = true;
  }
  assert(rejected == !conflicts.empty());
  for (const auto& result : {al::SolveAutoLayout(instance), al::SolveTreewidthDP(instance),
                             al::SolveGreedy(instance), al::SolveGreedyFM(instance)}) {
    assert(result.writeback_issues.empty());
    assert(result.conversions.size() == expected_copies);
    assert(al::CheckVecConversionCapacity(instance, result.Assignment()).empty());
  }
  const auto result = al::SolveAutoLayout(instance);
  const auto rewritten = al::ApplyLayoutAssignment(program, result.WritebackAssignment());
  size_t transposes = 0;
  auto count = [&](auto& self, const ir::StmtPtr& stmt) -> void {
    if (const auto assign = ir::As<ir::AssignStmt>(stmt);
        assign && ir::IsOp(ir::As<ir::Call>(assign->value_), "tile.transpose")) {
      ++transposes;
    }
    for (const auto& child : al::detail::IterChildStmts(stmt)) self(self, child);
  };
  count(count, rewritten->GetFunction("kernel")->body_);
  assert(transposes == expected_copies);
}

void TestMutableVecPartialAssignment(bool project_tuple) {
  const auto span = ir::Span::unknown();
  const auto tensor =
      std::make_shared<ir::TensorType>(std::vector<int64_t>{32, 64}, DataType::FP32, std::nullopt);
  const auto zero = std::make_shared<ir::ConstInt>(0, DataType::INDEX, span);
  const auto one = std::make_shared<ir::ConstInt>(1, DataType::INDEX, span);
  ir::TensorView dn;
  dn.layout = ir::TensorLayout::DN;
  dn.stride = {one, tensor->shape_[0]};
  const auto dn_tensor = std::make_shared<ir::TensorType>(tensor->shape_, DataType::FP32, std::nullopt, dn);
  const auto input = std::make_shared<ir::Var>("input", tensor, span);
  const auto before = std::make_shared<ir::Var>("before", dn_tensor, span);
  const auto after = std::make_shared<ir::Var>("after", dn_tensor, span);
  const auto shape = std::make_shared<ir::MakeTuple>(tensor->shape_, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{zero, zero}, span);
  const auto& registry = ir::OpRegistry::GetInstance();
  const auto load = registry.Create("tile.load", {input, offsets, shape, shape},
                                    {{"target_memory", ir::MemorySpace::Vec}}, span);
  const auto tile = std::make_shared<ir::Var>("tile", load->GetType(), span);
  const auto patch = std::make_shared<ir::Var>("patch", load->GetType(), span);
  auto store = [&](const ir::VarPtr& dst) {
    return std::make_shared<ir::EvalStmt>(registry.Create("tile.store", {tile, offsets, dst}, {}, span),
                                          span);
  };
  std::vector<ir::StmtPtr> statements{std::make_shared<ir::AssignStmt>(tile, load, span),
                                      std::make_shared<ir::AssignStmt>(patch, load, span), store(before)};
  ir::VarPtr target = tile;
  if (project_tuple) {
    // A nested projection aliases only its selected tile, including through
    // a separate SSA binding of the outer tuple.
    auto pair = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{patch, tile}, span);
    auto outer = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{patch, pair}, span);
    auto tuple = std::make_shared<ir::Var>("tuple", outer->GetType(), span);
    auto tuple_alias = std::make_shared<ir::Var>("tuple_alias", outer->GetType(), span);
    auto nested = std::make_shared<ir::TupleGetItemExpr>(tuple_alias, 1, span);
    auto projection = std::make_shared<ir::TupleGetItemExpr>(nested, 1, span);
    target = std::make_shared<ir::Var>("target", projection->GetType(), span);
    statements.insert(statements.end(), {std::make_shared<ir::AssignStmt>(tuple, outer, span),
                                         std::make_shared<ir::AssignStmt>(tuple_alias, tuple, span),
                                         std::make_shared<ir::AssignStmt>(target, projection, span)});
  }
  const auto update = registry.Create("tile.assemble", {target, patch, offsets}, {}, span);
  statements.insert(statements.end(), {std::make_shared<ir::EvalStmt>(update, span), store(after),
                                       std::make_shared<ir::ReturnStmt>(span)});
  auto body = std::make_shared<ir::SeqStmts>(std::move(statements), span);
  auto kernel = std::make_shared<ir::Function>(
      "kernel", std::vector<ir::VarPtr>{input, before, after},
      std::vector<ir::ParamDirection>{ir::ParamDirection::In, ir::ParamDirection::Out,
                                      ir::ParamDirection::Out},
      std::vector<ir::TypePtr>{}, body, span, ir::FunctionType::InCore);
  auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "mutable_vec", span);
  const auto instance = al::ExtractLayoutInstance(program);
  assert(instance.values.at("tile").has_mutable_vec_backing);
  assert(!instance.values.at("patch").has_mutable_vec_backing);
  if (project_tuple) assert(instance.values.at("target").has_mutable_vec_backing);
  const auto& load_op = instance.operations.front();
  for (bool partial : {false, true}) {
    al::LayoutAssignment assignment;
    if (partial) assignment.op_configs.emplace(load_op.op_id, load_op.configs.front());
    const auto written = al::ApplyLayoutAssignment(program, assignment);
    std::vector<std::string> order;
    auto collect = [&](auto& self, const ir::StmtPtr& stmt) -> void {
      if (auto assign = ir::As<ir::AssignStmt>(stmt);
          assign && ir::IsOp(ir::As<ir::Call>(assign->value_), "tile.transpose")) {
        order.push_back("transpose");
      }
      if (auto eval = ir::As<ir::EvalStmt>(stmt)) {
        auto call = ir::As<ir::Call>(eval->expr_);
        if (ir::IsOp(call, "tile.store")) order.push_back("store");
        if (ir::IsOp(call, "tile.assemble")) order.push_back("write");
      }
      for (const auto& child : al::detail::IterChildStmts(stmt)) self(self, child);
    };
    collect(collect, written->GetFunction("kernel")->body_);
    assert((order == std::vector<std::string>{"transpose", "store", "write", "transpose", "store"}));
  }
}

void TestVecConversionPressure() {
  const auto span = ir::Span::unknown();
  const auto tensor =
      std::make_shared<ir::TensorType>(std::vector<int64_t>{32, 384}, DataType::FP32, std::nullopt);
  const auto input = std::make_shared<ir::Var>("input", tensor, span);
  const auto zero = std::make_shared<ir::ConstInt>(0, DataType::INDEX, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{zero, zero}, span);
  const auto dims = std::make_shared<ir::MakeTuple>(tensor->shape_, span);
  const auto& registry = ir::OpRegistry::GetInstance();
  auto load = [&]() {
    return registry.Create("tile.load", {input, offsets, dims, dims},
                           {{"target_memory", ir::MemorySpace::Vec}}, span);
  };
  auto load_a = load();
  auto load_b = load();
  auto a = std::make_shared<ir::Var>("a", load_a->GetType(), span);
  auto b = std::make_shared<ir::Var>("b", load_b->GetType(), span);
  auto add = registry.Create("tile.add", {a, b}, {}, span);
  auto sum = std::make_shared<ir::Var>("sum", add->GetType(), span);
  auto body = std::make_shared<ir::SeqStmts>(
      std::vector<ir::StmtPtr>{std::make_shared<ir::AssignStmt>(a, load_a, span),
                               std::make_shared<ir::AssignStmt>(b, load_b, span),
                               std::make_shared<ir::AssignStmt>(sum, add, span),
                               std::make_shared<ir::ReturnStmt>(span)},
      span);
  auto kernel = std::make_shared<ir::Function>(
      "pressure", std::vector<ir::VarPtr>{input}, std::vector<ir::ParamDirection>{ir::ParamDirection::In},
      std::vector<ir::TypePtr>{}, body, span, ir::FunctionType::InCore);
  const auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "pressure", span);
  const auto instance = al::ExtractLayoutInstance(program);
  const auto& first = instance.values.at("a");
  const auto& second = instance.values.at("b");
  const auto rm = al::LayoutSpec::RowMajorFlat();
  const auto cm = al::LayoutSpec::ColMajorFlat();
  // Each 48 KiB conversion needs both a destination and scratch. It fits
  // with one live original tile, but exceeds Vec capacity with two.
  assert(instance.cost_model.SupportsVecTileConversion(first, rm, cm));
  assert(!instance.cost_model.SupportsVecTileConversion(second, rm, cm));
}

void TestSubviewStore(bool narrowed_pitch, bool project_tuple = false, bool inline_store = false) {
  const auto span = ir::Span::unknown();
  const auto source_dims = narrowed_pitch ? std::vector<int64_t>{16, 64} : std::vector<int64_t>{32, 32};
  const auto source_type = std::make_shared<ir::TensorType>(source_dims, DataType::INT32, std::nullopt);
  const auto dest_type =
      std::make_shared<ir::TensorType>(std::vector<int64_t>{16, 64}, DataType::INT32, std::nullopt);
  const auto source = std::make_shared<ir::Var>("source", source_type, span);
  const auto dest = std::make_shared<ir::Var>("dest", dest_type, span);
  auto tuple = [&](int64_t first, int64_t second) {
    return std::make_shared<ir::MakeTuple>(
        std::vector<ir::ExprPtr>{std::make_shared<ir::ConstInt>(first, DataType::INDEX, span),
                                 std::make_shared<ir::ConstInt>(second, DataType::INDEX, span)},
        span);
  };
  const auto offsets = tuple(0, 0);
  const auto source_shape = tuple(source_dims[0], source_dims[1]);
  const auto slice_shape = tuple(16, 32);
  const auto& registry = ir::OpRegistry::GetInstance();
  const auto load = registry.Create("tile.load", {source, offsets, source_shape, source_shape},
                                    {{"target_memory", ir::MemorySpace::Vec}}, span);
  const auto tile = std::make_shared<ir::Var>("backing_tile", load->GetType(), span);
  const auto slice = registry.Create("tile.slice", {tile, slice_shape, offsets}, {}, span);
  const auto slice_var = std::make_shared<ir::Var>("first_view", slice->GetType(), span);
  const auto nested = registry.Create("tile.slice", {slice_var, slice_shape, offsets}, {}, span);
  const auto nested_var = std::make_shared<ir::Var>("nested_view", nested->GetType(), span);
  std::vector<ir::StmtPtr> stmts{std::make_shared<ir::AssignStmt>(tile, load, span),
                                 std::make_shared<ir::AssignStmt>(slice_var, slice, span),
                                 std::make_shared<ir::AssignStmt>(nested_var, nested, span)};
  ir::VarPtr source_view = nested_var;
  ir::ExprPtr inline_view;
  if (project_tuple) {
    auto pair = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{tile, nested_var}, span);
    auto pair_var = std::make_shared<ir::Var>("pair", pair->GetType(), span);
    stmts.push_back(std::make_shared<ir::AssignStmt>(pair_var, pair, span));
    auto projection = std::make_shared<ir::TupleGetItemExpr>(pair_var, 1, span);
    inline_view = projection;
    source_view = std::make_shared<ir::Var>("projected_view", projection->GetType(), span);
    stmts.push_back(std::make_shared<ir::AssignStmt>(source_view, projection, span));
  }
  const auto store = registry.Create("tile.store", {source_view, offsets, dest}, {}, span);
  stmts.push_back(std::make_shared<ir::EvalStmt>(store, span));
  if (project_tuple) {
    auto move = registry.Create(
        "tile.move", {source_view},
        {{"target_memory", ir::MemorySpace::Vec}, {"blayout", ir::TileLayout::col_major}}, span);
    stmts.push_back(std::make_shared<ir::AssignStmt>(
        std::make_shared<ir::Var>("moved", move->GetType(), span), move, span));
    // A later read through an inline projection keeps the entire backing live.
    // An inline store also observes its destination's declared format.
    auto inline_read = inline_store ? registry.Create("tile.store", {inline_view, offsets, dest}, {}, span)
                                    : registry.Create("tile.read", {inline_view, offsets}, {}, span);
    stmts.push_back(std::make_shared<ir::EvalStmt>(inline_read, span));
  }
  stmts.push_back(std::make_shared<ir::ReturnStmt>(span));
  const auto kernel_body = std::make_shared<ir::SeqStmts>(std::move(stmts), span);
  const auto kernel = std::make_shared<ir::Function>(
      "store_view", std::vector<ir::VarPtr>{source, dest},
      std::vector<ir::ParamDirection>{ir::ParamDirection::In, ir::ParamDirection::Out},
      std::vector<ir::TypePtr>{}, kernel_body, span, ir::FunctionType::InCore);
  const auto create = registry.Create("tensor.create", {tuple(16, 64)}, {{"dtype", DataType::INT32}}, span);
  const auto buffer = std::make_shared<ir::Var>("buffer", create->GetType(), span);
  const auto input = std::make_shared<ir::Var>("input", source_type, span);
  const auto call = std::make_shared<ir::Call>(std::make_shared<ir::GlobalVar>("store_view"),
                                               std::vector<ir::ExprPtr>{input, buffer}, span);
  const auto entry_body = std::make_shared<ir::SeqStmts>(
      std::vector<ir::StmtPtr>{std::make_shared<ir::AssignStmt>(buffer, create, span),
                               std::make_shared<ir::EvalStmt>(call, span),
                               std::make_shared<ir::ReturnStmt>(span)},
      span);
  const auto entry = std::make_shared<ir::Function>(
      "entry", std::vector<ir::VarPtr>{input}, std::vector<ir::ParamDirection>{ir::ParamDirection::In},
      std::vector<ir::TypePtr>{}, entry_body, span, ir::FunctionType::Orchestration);
  const auto program =
      std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel, entry}, "subview_store", span);

  const auto instance = al::ExtractLayoutInstance(program, {}, {});
  assert(instance.values.at("first_view").is_strided_subview == narrowed_pitch);
  assert(instance.values.at("nested_view").is_strided_subview == narrowed_pitch);
  // Neither slice releases its backing or allocates a second tile. The
  // conversion after each named view sees the same remaining Vec storage.
  const auto& backing_headroom = instance.values.at("backing_tile").vec_conversion_headroom;
  assert(backing_headroom.has_value());
  assert(instance.values.at("first_view").vec_conversion_headroom == backing_headroom);
  assert(instance.values.at("nested_view").vec_conversion_headroom == backing_headroom);
  if (project_tuple) {
    const auto& view = instance.values.at("projected_view");
    assert(view.is_vec_slice && view.is_strided_subview == narrowed_pitch);
    assert(view.vec_conversion_headroom == backing_headroom);
    assert(!instance.values.at("backing_tile").is_strided_subview);
    assert(instance.vec_memory_regions.front().original_bytes.back() ==
           instance.values.at("backing_tile").ByteSize());
    const auto& move = instance.operations.back();
    assert(move.op_name == "tile.move");
    const bool offers_transpose = std::any_of(move.configs.begin(), move.configs.end(), [](const auto& cfg) {
      return cfg.input_layouts.front() == al::LayoutSpec::RowMajorFlat() &&
             cfg.output_layout == al::LayoutSpec::ColMajorFlat();
    });
    assert(offers_transpose == !narrowed_pitch);
  }
  const auto result = al::SolveAutoLayout(instance);
  // DN makes the write contiguous, but requires a compact column-major tile.
  // An equal-pitch window can convert as before. A nested view of a narrowed
  // pitch inherits the original stride even with the same shape as its parent.
  const bool choose_dn = !narrowed_pitch && !inline_store;
  const auto expected_layout = choose_dn ? al::LayoutSpec::DN() : al::LayoutSpec::ND();
  assert(result.Assignment().value_layouts.at("buffer") == expected_layout);
  assert(result.conversions.size() == (choose_dn ? 1 : 0));
}

void TestVecLoopConversions(bool column_carry, int64_t trip_count, bool while_loop = false,
                            bool column_input = false) {
  const auto span = ir::Span::unknown();
  auto index = [&](int64_t value) { return std::make_shared<ir::ConstInt>(value, DataType::INDEX, span); };
  const auto shape = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{index(32), index(32)}, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{index(0), index(0)}, span);
  const std::vector<int64_t> dims{32, 32};
  const auto input_type = std::make_shared<ir::TensorType>(
      dims, DataType::FP32, std::nullopt,
      column_input
          ? std::optional<ir::TensorView>(ir::TensorView(std::vector<int64_t>{1, 32}, ir::TensorLayout::DN))
          : std::nullopt);
  const auto input = std::make_shared<ir::Var>("input", input_type, span);
  const auto& registry = ir::OpRegistry::GetInstance();
  auto bind = [&](const std::string& name, const ir::CallPtr& call) {
    auto var = std::make_shared<ir::Var>(name, call->GetType(), span);
    return std::make_shared<ir::AssignStmt>(var, call, span);
  };
  const auto seed = bind("seed", registry.Create("tile.load", {input, offsets, shape, shape},
                                                 {{"target_memory", ir::MemorySpace::Vec}}, span));
  const auto initializer =
      column_carry ? bind("column", registry.Create("tile.transpose_view", {seed->var_}, {}, span)) : seed;
  const auto carried_type = initializer->var_->GetType();
  const auto iter = std::make_shared<ir::IterArg>("carry", carried_type, initializer->var_, span);
  const auto result = std::make_shared<ir::Var>("exit", carried_type, span);
  const auto scale = std::make_shared<ir::ConstFloat>(2.0, DataType::FP32, span);
  auto mul = [&](const ir::VarPtr& tile, const std::string& name) {
    return bind(name, registry.Create("tile.muls", {tile, scale}, {}, span));
  };
  const auto next = mul(iter, "next");
  const auto result_use = mul(result, "result_use");
  std::vector<ir::IterArgPtr> iter_args{iter};
  std::vector<ir::VarPtr> results{result};
  std::vector<ir::ExprPtr> yielded{next->var_};
  ir::ExprPtr condition;
  if (while_loop) {
    const auto start = std::make_shared<ir::ConstBool>(trip_count != 0, span);
    auto flag = std::make_shared<ir::IterArg>("flag", start->GetType(), start, span);
    iter_args.push_back(flag);
    results.push_back(std::make_shared<ir::Var>("flag_exit", start->GetType(), span));
    yielded.push_back(std::make_shared<ir::ConstBool>(false, span));
    condition = trip_count == 0 ? ir::ExprPtr(start) : ir::ExprPtr(flag);
  }
  const auto loop_body = std::make_shared<ir::SeqStmts>(
      std::vector<ir::StmtPtr>{next, std::make_shared<ir::YieldStmt>(yielded, span)}, span);
  ir::StmtPtr loop;
  if (while_loop) {
    loop = std::make_shared<ir::WhileStmt>(condition, iter_args, loop_body, results, span);
  } else {
    auto induction = std::make_shared<ir::Var>("i", index(0)->GetType(), span);
    loop = std::make_shared<ir::ForStmt>(induction, index(0), index(trip_count), index(1), iter_args,
                                         loop_body, results, span);
  }
  std::vector<ir::StmtPtr> stmts{seed};
  if (column_carry) stmts.push_back(initializer);
  stmts.insert(stmts.end(), {loop, result_use, std::make_shared<ir::ReturnStmt>(span)});
  auto kernel = std::make_shared<ir::Function>(
      "vec_loop", std::vector<ir::VarPtr>{input}, std::vector<ir::ParamDirection>{ir::ParamDirection::In},
      std::vector<ir::TypePtr>{}, std::make_shared<ir::SeqStmts>(stmts, span), span,
      ir::FunctionType::InCore);
  const auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "vec_loop", span);
  const auto instance = al::ExtractLayoutInstance(program);
  const auto row = al::LayoutSpec::RowMajorFlat();
  const auto col = al::LayoutSpec::ColMajorFlat();
  const auto carried_layout = column_carry ? col : row;
  const auto& entry = instance.values.at("carry");
  const auto& exit = instance.values.at("exit");
  assert(entry.conversion_invocations == trip_count);
  assert(exit.conversion_invocations == 1);
  const auto solution = al::SolveAutoLayout(instance);
  assert(solution.writeback_issues.empty());
  assert(solution.Assignment().value_layouts.at("carry") == carried_layout);
  assert(solution.Assignment().value_layouts.at("exit") == carried_layout);

  const auto rewritten = al::ApplyLayoutAssignment(program, solution.WritebackAssignment());
  int64_t emitted_inside = 0;
  int64_t emitted_outside = 0;
  auto layout_of = [](const ir::ExprPtr& expr) {
    return al::EffectiveTileLayout(ir::As<ir::TileType>(expr->GetType()));
  };
  auto count_transposes = [&](auto& self, const ir::StmtPtr& stmt, bool inside) -> void {
    auto check_loop = [&](const auto& rewritten_loop) {
      assert(layout_of(rewritten_loop->iter_args_[0]->initValue_) == carried_layout);
      assert(layout_of(rewritten_loop->iter_args_[0]) == carried_layout);
      assert(layout_of(rewritten_loop->return_vars_[0]) == carried_layout);
      self(self, rewritten_loop->body_, true);
    };
    if (const auto fs = ir::As<ir::ForStmt>(stmt)) return check_loop(fs);
    if (const auto ws = ir::As<ir::WhileStmt>(stmt)) return check_loop(ws);
    if (const auto ys = ir::As<ir::YieldStmt>(stmt)) {
      assert(layout_of(ys->value_[0]) == carried_layout);
    }
    if (const auto assign = ir::As<ir::AssignStmt>(stmt)) {
      if (ir::IsOp(ir::As<ir::Call>(assign->value_), "tile.transpose")) {
        ++(inside ? emitted_inside : emitted_outside);
      }
    }
    for (const auto& child : al::detail::IterChildStmts(stmt)) self(self, child, inside);
  };
  count_transposes(count_transposes, rewritten->GetFunction("vec_loop")->body_, false);
  assert(emitted_inside == (column_carry ? 2 : 0));
  assert(emitted_outside == (column_carry || column_input ? 1 : 0));
  const auto one_transpose = instance.EstimateConversionCost(instance.values.at("seed"), row, col);
  const auto expected = one_transpose * (emitted_inside * trip_count + emitted_outside);
  assert(solution.breakdown.conversion_cost_cycles == expected);
  assert(al::EvaluateAssignment(instance, solution.Assignment()).conversion_cost_cycles == expected);
}

void TestUnmodeledVecLoopConversion(bool peel_first_iteration) {
  const auto span = ir::Span::unknown();
  const auto index = [&](int64_t value) {
    return std::make_shared<ir::ConstInt>(value, DataType::INDEX, span);
  };
  const auto shape = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{index(32), index(32)}, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{index(0), index(0)}, span);
  const auto dn =
      std::make_shared<ir::TensorType>(std::vector<int64_t>{32, 32}, DataType::FP32, std::nullopt,
                                       ir::TensorView(std::vector<int64_t>{1, 32}, ir::TensorLayout::DN));
  const auto output = std::make_shared<ir::Var>("output", dn, span);
  const auto& registry = ir::OpRegistry::GetInstance();
  const auto diagonal = std::make_shared<ir::ConstInt>(0, DataType::INT32, span);
  const auto tri = registry.Create("tile.tri", {diagonal, shape}, {{"dtype", DataType::FP32}}, span);
  const auto mask = std::make_shared<ir::Var>("mask", tri->GetType(), span);
  // The generic operation first introduces the unmodeled mask to the solver.
  // The DN store requests its column-major representation.
  const auto scale = std::make_shared<ir::ConstFloat>(2.0, DataType::FP32, span);
  const auto use_mask = [&](const char* name) {
    const auto mul = registry.Create("tile.muls", {mask, scale}, {}, span);
    const auto store = registry.Create("tile.store", {mask, offsets, output}, {}, span);
    const auto scaled = std::make_shared<ir::Var>(name, mul->GetType(), span);
    return std::vector<ir::StmtPtr>{std::make_shared<ir::AssignStmt>(scaled, mul, span),
                                    std::make_shared<ir::EvalStmt>(store, span)};
  };
  std::vector<ir::StmtPtr> stmts{std::make_shared<ir::AssignStmt>(mask, tri, span)};
  if (peel_first_iteration) {
    auto first_iteration = use_mask("first_scaled");
    stmts.insert(stmts.end(), first_iteration.begin(), first_iteration.end());
  }
  constexpr int64_t trip_count = 4;
  const auto induction = std::make_shared<ir::Var>("i", index(0)->GetType(), span);
  const auto loop_body = std::make_shared<ir::SeqStmts>(use_mask("scaled"), span);
  stmts.push_back(std::make_shared<ir::ForStmt>(induction, index(peel_first_iteration ? 1 : 0),
                                                index(trip_count), index(1), std::vector<ir::IterArgPtr>{},
                                                loop_body, std::vector<ir::VarPtr>{}, span));
  stmts.push_back(std::make_shared<ir::ReturnStmt>(span));
  const auto kernel = std::make_shared<ir::Function>(
      "kernel", std::vector<ir::VarPtr>{output}, std::vector<ir::ParamDirection>{ir::ParamDirection::Out},
      std::vector<ir::TypePtr>{}, std::make_shared<ir::SeqStmts>(stmts, span), span,
      ir::FunctionType::InCore);
  const auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "tri_loop", span);
  const auto instance = al::ExtractLayoutInstance(program);
  const auto& value = instance.values.at("mask");
  const auto row = al::LayoutSpec::RowMajorFlat();
  const auto col = al::LayoutSpec::ColMajorFlat();
  const auto one_transpose = instance.EstimateConversionCost(value, row, col);
  const auto solution = al::SolveAutoLayout(instance);
  // Peeling changes where the mask is first registered. Both programs define
  // the mask just once and reuse its converted copy for every store.
  assert(value.conversion_invocations == 1);
  assert(solution.conversions.size() == 1);
  assert(solution.breakdown.conversion_cost_cycles == one_transpose);
  assert(al::EvaluateAssignment(instance, solution.Assignment()).conversion_cost_cycles == one_transpose);
}

}  // namespace

int main() {
  pypto::backend::BackendConfig::SetBackendType(pypto::backend::BackendType::Ascend910);
  TestHostMatrixLayout(HostMatrixDispatch::Implicit, 1, true);
  TestHostMatrixLayout(HostMatrixDispatch::RankZero, 1, true);
  TestHostMatrixLayout(HostMatrixDispatch::Implicit, std::nullopt, false);
  TestHostMatrixLayout(HostMatrixDispatch::RankZero, 2, false);
  TestHostMatrixLayout(HostMatrixDispatch::UnknownRank, 1, false);
  TestHostMatrixLayout(HostMatrixDispatch::PartialSlice, 1, false);
  TestHostMatrixLayout(HostMatrixDispatch::FullSlice, 1, false);
  TestHostMatrixLayout(HostMatrixDispatch::Mutable, 1, false);
  // A Qwen-size NZ source needs multiple bursts with a gap over 16 bits.
  TestHostMatrixLayout(HostMatrixDispatch::Implicit, 1, true, true);
  TestHostMatrixLayout(HostMatrixDispatch::Implicit, 1, false, true, pypto::backend::BackendType::Ascend950);
  TestHostInputProjectionCost(false, 1);
  TestHostInputProjectionCost(true, 1);
  TestHostInputProjectionCost(false, 3);
  TestHostInputProjectionCost(true, 3);
  TestHostInputProjectionCost(false, 3, 1, 1);
  TestHostInputProjectionCost(false, 3, 2, 2);
  TestHostInputProjectionCost(false, 3, 2);
  TestHostInputProjectionCost(false, 3, 5);
  const auto span = ir::Span::unknown();
  const std::vector<int64_t> dims{64, 64};
  const auto tensor = std::make_shared<ir::TensorType>(dims, DataType::FP16, std::nullopt);
  const auto weight = std::make_shared<ir::Var>("weight", tensor, span);
  const auto activation = std::make_shared<ir::Var>("activation", tensor, span);
  const auto zero = std::make_shared<ir::ConstInt>(0, DataType::INDEX, span);
  const auto offsets = std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{zero, zero}, span);
  const auto shape = std::make_shared<ir::MakeTuple>(tensor->shape_, span);
  auto load = [&](const ir::VarPtr& input, ir::MemorySpace memory, const char* name) {
    auto call = ir::OpRegistry::GetInstance().Create("tile.load", {input, offsets, shape, shape},
                                                     {{"target_memory", memory}}, span);
    auto tile = std::make_shared<ir::Var>(name, call->GetType(), span);
    return std::make_shared<ir::AssignStmt>(tile, call, span);
  };
  auto body = std::make_shared<ir::SeqStmts>(
      std::vector<ir::StmtPtr>{load(weight, ir::MemorySpace::Mat, "w_tile"),
                               load(activation, ir::MemorySpace::Vec, "a_tile"),
                               std::make_shared<ir::ReturnStmt>(span)},
      span);
  auto kernel = std::make_shared<ir::Function>(
      "kernel", std::vector<ir::VarPtr>{weight, activation},
      std::vector<ir::ParamDirection>{ir::ParamDirection::In, ir::ParamDirection::In},
      std::vector<ir::TypePtr>{}, body, span, ir::FunctionType::InCore);
  auto program = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{kernel}, "loads", span);

  ir::pass::AutoLayoutOptions opts;
  opts.reusable_tensors.insert("weight");
  auto baseline = al::ExtractLayoutInstance(program, opts.cost_model, opts.reusable_tensors);
  assert(baseline.operations.size() == 2);
  assert(baseline.values.at("weight").admissible_layouts ==
         std::vector<al::LayoutSpec>{al::LayoutSpec::ND()});

  // The caller can supply prepacked NZ weights. DN still requires an internally
  // owned buffer, even for the aligned external Vec input.
  opts.cost_model.allow_external_input_relayout = true;
  auto choices = al::ExtractLayoutInstance(program, opts.cost_model, opts.reusable_tensors);
  const auto& activation_layouts = choices.values.at("activation").admissible_layouts;
  assert(std::find(activation_layouts.begin(), activation_layouts.end(), al::LayoutSpec::DN()) ==
         activation_layouts.end());
  auto weight_type = [](const ir::ProgramPtr& prog) {
    return ir::AsTensorTypeLike(prog->GetFunction("kernel")->params_.front()->GetType());
  };
  opts.custom_conversion_costs["weight"][{al::LayoutSpec::ND(), al::LayoutSpec::NZ()}] = 1e6;
  assert(al::detail::TensorTypeLayout(weight_type(ir::pass::AutoLayout(opts)(program))) ==
         ir::TensorLayout::ND);
  opts.custom_conversion_costs["weight"][{al::LayoutSpec::ND(), al::LayoutSpec::NZ()}] = 0.0;
  const auto rewritten = ir::pass::AutoLayout(opts)(program);
  assert(al::detail::TensorTypeLayout(weight_type(rewritten)) == ir::TensorLayout::NZ);
  const auto blocked = ir::pass::BlockNzTensorViews()(rewritten);
  assert(weight_type(blocked)->shape_.size() == 5);
  TestDistinctTileDefinitionsWithRepeatedHints();
  TestGenericVecOutputLayoutAssignment();
  TestValueOnlyWritebackWithRepeatedHints("t", "t");
  TestValueOnlyWritebackWithRepeatedHints("t__ssa_v0", "t__ssa_v1");
  TestValueOnlyWritebackWithRepeatedHints("t_1", "t_2");
  TestStoreBindingAcrossUnmodeledStore(false);
  TestStoreBindingAcrossUnmodeledStore(true);
  TestCallSiteAndFunctionNamesKeepTheirOwnLayouts(false);
  TestCallSiteAndFunctionNamesKeepTheirOwnLayouts(true);
  TestValueOnlyJoinCapacity(256, false, 1);
  TestValueOnlyJoinCapacity(512, false, std::nullopt);  // Three 64 KiB tiles exceed Vec capacity.
  TestValueOnlyJoinCapacity(512, true, 0);
  TestVecCopyLifetime(384, 768, true, false, 0);  // A later allocation exceeds 184 KiB with the copy.
  TestVecCopyLifetime(384, 576, true, false, 1);  // Transpose scratch is free by the later allocation.
  TestVecCopyLifetime(320, 320, true, true, 1);   // A second transpose exceeds capacity with the first copy.
  TestVecCopyLifetime(320, 320, false, true, 2);  // The same copies fit if their lifetimes are disjoint.
  // Both representations of A must survive B for the next iteration.
  TestVecCopyLifetime(384, 768, false, false, 0, VecPressureScope::ForLoop);
  TestVecCopyLifetime(384, 768, false, false, 0, VecPressureScope::WhileLoop);
  TestVecCopyLifetime(384, 576, true, false, 0, VecPressureScope::Reserved);
  TestVecCopyLifetime(384, 768, true, false, 0, VecPressureScope::UnmodeledGenerator);
  TestVecConversionPressure();
  TestMutableVecPartialAssignment(false);
  TestMutableVecPartialAssignment(true);
  TestSubviewStore(true);
  TestSubviewStore(false);
  TestSubviewStore(true, true);
  TestSubviewStore(false, true);
  TestSubviewStore(false, true, /*inline_store=*/true);
  TestVecLoopConversions(true, 2);
  TestVecLoopConversions(true, 0);
  TestVecLoopConversions(false, 2);
  TestVecLoopConversions(true, 1, /*while_loop=*/true);
  TestVecLoopConversions(true, 0, /*while_loop=*/true);
  TestVecLoopConversions(false, 2, /*while_loop=*/false, /*column_input=*/true);
  TestUnmodeledVecLoopConversion(false);
  TestUnmodeledVecLoopConversion(true);
}
