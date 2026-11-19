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
#include "pypto/ir/transforms/passes.h"

namespace ir = pypto::ir;
namespace al = ir::auto_layout;
using pypto::DataType;

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
}
