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
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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
    const auto result = ir::pass::AutoLayoutFromAssignment(assignment)(program);
    auto loads = ir::As<ir::SeqStmts>(result->GetFunction("kernel")->body_);
    auto layout = [&](size_t i) {
      auto tile = ir::As<ir::AssignStmt>(loads->stmts_.at(i))->var_;
      return al::EffectiveTileLayout(ir::As<ir::TileType>(tile->GetType()));
    };
    assert(layout(0) == al::LayoutSpec::ColMajorFlat());
    assert(layout(1) == al::LayoutSpec::RowMajorFlat());
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
    auto rewritten = ir::pass::AutoLayoutFromAssignment(assignment)(program);
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

void TestSubviewStore(bool narrowed_pitch) {
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
  const auto store = registry.Create("tile.store", {nested_var, offsets, dest}, {}, span);
  const auto kernel_body = std::make_shared<ir::SeqStmts>(
      std::vector<ir::StmtPtr>{std::make_shared<ir::AssignStmt>(tile, load, span),
                               std::make_shared<ir::AssignStmt>(slice_var, slice, span),
                               std::make_shared<ir::AssignStmt>(nested_var, nested, span),
                               std::make_shared<ir::EvalStmt>(store, span),
                               std::make_shared<ir::ReturnStmt>(span)},
      span);
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
  const auto result = al::SolveAutoLayout(instance);
  // DN makes the write contiguous, but requires a compact column-major tile.
  // An equal-pitch window can convert as before. A nested view of a narrowed
  // pitch inherits the original stride even with the same shape as its parent.
  const auto expected_layout = narrowed_pitch ? al::LayoutSpec::ND() : al::LayoutSpec::DN();
  assert(result.Assignment().value_layouts.at("buffer") == expected_layout);
  assert(result.conversions.size() == (narrowed_pitch ? 0 : 1));
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

}  // namespace

int main() {
  pypto::backend::BackendConfig::SetBackendType(pypto::backend::BackendType::Ascend910);
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
  TestSubviewStore(true);
  TestSubviewStore(false);
  TestVecLoopConversions(true, 2);
  TestVecLoopConversions(true, 0);
  TestVecLoopConversions(false, 2);
  TestVecLoopConversions(true, 1, /*while_loop=*/true);
  TestVecLoopConversions(true, 0, /*while_loop=*/true);
  TestVecLoopConversions(false, 2, /*while_loop=*/false, /*column_input=*/true);
}
