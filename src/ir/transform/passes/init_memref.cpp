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

#include "pypto/ir/transform/passes/init_memref.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transform/base/mutator.h"
#include "pypto/ir/transform/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Visitor to identify variables that should be in DDR memory space
class MemRefUsageVisitor : public IRVisitor {
 public:
  const std::set<std::string>& GetDdrVars() const { return ddr_vars_; }

  void VisitExpr_(const CallPtr& op) override {
    if (op->op_->name_ == "block.load") {
      // block.load(tensor, ...) -> tensor is source (DDR)
      if (!op->args_.empty()) {
        if (auto v = std::dynamic_pointer_cast<const Var>(op->args_[0])) {
          ddr_vars_.insert(v->name_);
        }
      }
    } else if (op->op_->name_ == "block.store") {
      // block.store(..., output_tensor) -> output_tensor is dest (DDR)
      // Signature: store(tile, row, col, h, w, output) -> index 5
      if (op->args_.size() > 5) {
        if (auto v = std::dynamic_pointer_cast<const Var>(op->args_[5])) {
          ddr_vars_.insert(v->name_);
        }
      }
    }
    // Continue visiting arguments
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::set<std::string> ddr_vars_;  // Store variable names that should be DDR
};

// Mutator to initialize MemRef for variables
class InitMemRefMutator : public IRMutator {
 public:
  explicit InitMemRefMutator(const std::set<std::string>& ddr_vars) : ddr_vars_(ddr_vars) {}

  // Helper to calculate size and create MemRef
  std::optional<MemRef> CreateMemRef(const ShapedTypePtr& type, const std::string& var_name) {
    uint64_t size_bytes = 0;
    bool is_static = true;
    uint64_t num_elements = 1;

    for (const auto& dim : type->shape_) {
      if (auto const_dim = std::dynamic_pointer_cast<const ConstInt>(dim)) {
        num_elements *= const_dim->value_;
      } else {
        is_static = false;
        break;
      }
    }

    if (is_static) {
      size_t bits = type->dtype_.GetBit();
      // Round up to bytes
      size_t bytes = (bits + 7) / 8;
      size_bytes = num_elements * bytes;
    }

    MemorySpace space = MemorySpace::UB;
    if (ddr_vars_.count(var_name)) {
      space = MemorySpace::DDR;
    }

    // Addr is always 0
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());

    return MemRef(space, addr, size_bytes);
  }

  // Create a new Var with MemRef initialized
  VarPtr GetNewVar(const VarPtr& old_var) {
    // Check if already mapped
    auto it = var_map_.find(old_var->name_);
    if (it != var_map_.end()) {
      return it->second;
    }

    // Create new var
    TypePtr new_type = old_var->GetType();

    // Process Type if it is ShapedType (TensorType or TileType)
    if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(old_var->GetType())) {
      auto memref = CreateMemRef(tensor_type, old_var->name_);
      new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, memref);
    } else if (auto tile_type = std::dynamic_pointer_cast<const TileType>(old_var->GetType())) {
      auto memref = CreateMemRef(tile_type, old_var->name_);
      new_type =
          std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, memref, tile_type->tile_view_);
    }

    VarPtr new_var;
    if (auto iter_arg = std::dynamic_pointer_cast<const IterArg>(old_var)) {
      // For IterArg, we also need to visit the initValue
      auto new_init = VisitExpr(iter_arg->initValue_);
      new_var = std::make_shared<IterArg>(iter_arg->name_, new_type, new_init, iter_arg->span_);
    } else {
      new_var = std::make_shared<Var>(old_var->name_, new_type, old_var->span_);
    }

    var_map_[old_var->name_] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const VarPtr& op) override { return GetNewVar(op); }

  ExprPtr VisitExpr_(const IterArgPtr& op) override { return GetNewVar(op); }

 private:
  const std::set<std::string>& ddr_vars_;
  std::unordered_map<std::string, VarPtr> var_map_;
};

}  // namespace

FunctionPtr InitMemRefPass::Run(const FunctionPtr& func) {
  // Step 1: Analyze usage to find DDR variables
  MemRefUsageVisitor visitor;
  visitor.VisitStmt(func->body_);

  // Step 2: Mutate variables
  InitMemRefMutator mutator(visitor.GetDdrVars());

  // Process params first to define them in the map
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  for (const auto& param : func->params_) {
    // Cast ExprPtr back to VarPtr for GetNewVar
    auto new_param_expr = mutator.GetNewVar(param);
    auto new_param = std::dynamic_pointer_cast<const Var>(new_param_expr);
    INTERNAL_CHECK(new_param) << "Failed to cast mutated param to Var";
    new_params.push_back(new_param);
  }

  // Process body
  auto new_body = mutator.VisitStmt(func->body_);

  // Reconstruct function
  return std::make_shared<Function>(func->name_, new_params, func->return_types_, new_body, func->span_);
}

}  // namespace ir
}  // namespace pypto
