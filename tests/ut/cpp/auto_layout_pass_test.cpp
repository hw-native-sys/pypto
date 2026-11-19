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
#include <cassert>
#include <memory>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/auto_layout/extractor.h"
#include "pypto/ir/transforms/auto_layout/solver.h"
#include "pypto/ir/transforms/passes.h"

namespace ir = pypto::ir;
namespace al = ir::auto_layout;
using pypto::DataType;

namespace {

void TestVecConversionPressure() {
  const auto span = ir::Span::unknown();
  const auto tensor =
      std::make_shared<ir::TensorType>(std::vector<int64_t>{32, 512}, DataType::FP32, std::nullopt);
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
  // With one 64 KiB tile live a copy fits. With two live tiles it exceeds Vec capacity.
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
  const auto result = al::SolveAutoLayout(instance);
  // DN makes the write contiguous, but requires a compact column-major tile.
  // An equal-pitch window can convert as before. A nested view of a narrowed
  // pitch inherits the original stride even with the same shape as its parent.
  const auto expected_layout = narrowed_pitch ? al::LayoutSpec::ND() : al::LayoutSpec::DN();
  assert(result.Assignment().value_layouts.at("buffer") == expected_layout);
  assert(result.conversions.size() == (narrowed_pitch ? 0 : 1));
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
  TestVecConversionPressure();
  TestSubviewStore(true);
  TestSubviewStore(false);
}
