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

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Visitor to identify variables that should be in DDR memory space
class MemRefUsageVisitor : public IRVisitor {
 public:
  // Initialize visitor with function parameters (all params should be in DDR)
  explicit MemRefUsageVisitor(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      ddr_vars_.insert(param->name_);
    }
  }

  [[nodiscard]] const std::set<std::string>& GetDdrVars() const { return ddr_vars_; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // Check if the right-hand side is a block.store call
    if (auto call = std::dynamic_pointer_cast<const Call>(op->value_)) {
      if (call->op_->name_ == "block.store") {
        // block.store returns the output_tensor (6th argument)
        // So the variable receiving the return value should also be DDR
        ddr_vars_.insert(op->var_->name_);
      }
    }
    // Continue with default traversal
    if (op->var_) {
      VisitExpr(op->var_);
    }
    if (op->value_) {
      VisitExpr(op->value_);
    }
  }

  void VisitExpr_(const CallPtr& op) override {
    if (op->op_->name_ == "block.load") {
      // block.load(tensor, ...) -> tensor is source (DDR)
      if (!op->args_.empty()) {
        if (auto v = As<Var>(op->args_[0])) {
          ddr_vars_.insert(v->name_);
        }
      }
    } else if (op->op_->name_ == "block.store") {
      // block.store(..., output_tensor) -> output_tensor is dest (DDR)
      // Signature: store(tile, row, col, h, w, output) -> index 5
      if (op->args_.size() > 5) {
        if (auto v = As<Var>(op->args_[5])) {
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
  explicit InitMemRefMutator(const std::set<std::string>& ddr_vars) : ddr_vars_(ddr_vars), next_id_(0) {}

  // Helper to calculate size and create MemRef
  std::optional<MemRefPtr> CreateMemRef(const ShapedTypePtr& type, const std::string& var_name) {
    uint64_t size_bytes = 0;
    bool is_static = true;
    uint64_t num_elements = 1;

    for (const auto& dim : type->shape_) {
      if (auto const_dim = As<ConstInt>(dim)) {
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

    // Generate unique ID for this MemRef
    uint64_t id = next_id_++;

    return std::make_shared<MemRef>(space, addr, size_bytes, id);
  }

  // Create a new Var with MemRef initialized
  VarPtr GetNewVar(const VarPtr& old_var) {
    // Check if already mapped
    auto it = var_map_.find(old_var->name_);
    if (it != var_map_.end()) {
      return it->second;
    }

    // Special handling for IterArg: should inherit MemRef from initValue
    VarPtr new_var;
    if (auto iter_arg = As<IterArg>(std::static_pointer_cast<const IRNode>(old_var))) {
      // First visit the initValue to get its updated MemRef
      auto new_init = VisitExpr(iter_arg->initValue_);

      // Extract MemRef from the initValue's type
      TypePtr new_type = old_var->GetType();
      if (auto init_tensor_type = As<TensorType>(new_init->GetType())) {
        // IterArg inherits the MemRef from its initValue
        new_type = std::make_shared<TensorType>(init_tensor_type->shape_, init_tensor_type->dtype_,
                                                init_tensor_type->memref_);
      } else if (auto init_tile_type = As<TileType>(new_init->GetType())) {
        new_type = std::make_shared<TileType>(init_tile_type->shape_, init_tile_type->dtype_,
                                              init_tile_type->memref_, init_tile_type->tile_view_);
      }

      new_var = std::make_shared<IterArg>(iter_arg->name_, new_type, new_init, iter_arg->span_);
    } else {
      // Normal Var: create new MemRef based on usage analysis
      TypePtr new_type = old_var->GetType();

      // Process Type if it is ShapedType (TensorType or TileType)
      if (auto tensor_type = As<TensorType>(old_var->GetType())) {
        auto memref = CreateMemRef(tensor_type, old_var->name_);
        new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, memref);
      } else if (auto tile_type = As<TileType>(old_var->GetType())) {
        auto memref = CreateMemRef(tile_type, old_var->name_);
        new_type =
            std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, memref, tile_type->tile_view_);
      }

      new_var = std::make_shared<Var>(old_var->name_, new_type, old_var->span_);
    }

    var_map_[old_var->name_] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const VarPtr& op) override { return GetNewVar(op); }

  ExprPtr VisitExpr_(const IterArgPtr& op) override { return GetNewVar(op); }

  // Handle block.store specially: return value should share the same MemRef as the 6th argument
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // First visit the value (RHS)
    auto new_value = VisitExpr(op->value_);

    // Check if the RHS is a block.store call
    if (auto call = std::dynamic_pointer_cast<const Call>(op->value_)) {
      if (call->op_->name_ == "block.store" && call->args_.size() > 5) {
        // Get the 6th argument (output tensor) after mutation
        auto new_call = std::dynamic_pointer_cast<const Call>(new_value);
        if (new_call && new_call->args_.size() > 5) {
          auto output_tensor_arg = new_call->args_[5];

          // Extract MemRef from the output tensor
          std::optional<MemRefPtr> shared_memref = std::nullopt;
          if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(output_tensor_arg->GetType())) {
            shared_memref = tensor_type->memref_;
          }

          // Create new variable with the shared MemRef
          if (shared_memref.has_value()) {
            TypePtr new_type = op->var_->GetType();
            if (auto var_tensor_type = std::dynamic_pointer_cast<const TensorType>(op->var_->GetType())) {
              // Reuse the MemRef from the 6th argument
              new_type = std::make_shared<TensorType>(var_tensor_type->shape_, var_tensor_type->dtype_,
                                                      shared_memref);
            }

            VarPtr new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
            var_map_[op->var_->name_] = new_var;

            return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
          }
        }
      }
    }

    // Default case: visit the variable normally
    auto new_var = GetNewVar(op->var_);
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

 private:
  const std::set<std::string>& ddr_vars_;
  std::unordered_map<std::string, VarPtr> var_map_;
  uint64_t next_id_ = 0;  // Counter for generating unique MemRef IDs
};

/**
 * @brief Transform a function by initializing MemRef for all variables
 *
 * This transformation initializes the MemRef field for all Var nodes in the function.
 * It sets memory space to UB by default, or DDR for variables used in
 * block.load/block.store operations.
 */
FunctionPtr TransformInitMemRef(const FunctionPtr& func) {
  // Step 1: Analyze usage to find DDR variables
  // All function parameters should be in DDR (main memory)
  MemRefUsageVisitor visitor(func->params_);
  visitor.VisitStmt(func->body_);

  // Step 2: Mutate variables
  InitMemRefMutator mutator(visitor.GetDdrVars());

  // Process params first to define them in the map
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  for (const auto& param : func->params_) {
    // GetNewVar returns a VarPtr directly
    auto new_param = mutator.GetNewVar(param);
    INTERNAL_CHECK(new_param) << "Failed to get new param";
    new_params.push_back(new_param);
  }

  // Process body
  auto new_body = mutator.VisitStmt(func->body_);

  // Reconstruct function
  return std::make_shared<Function>(func->name_, new_params, func->return_types_, new_body, func->span_);
}

}  // namespace

// Factory function
namespace pass {
Pass InitMemRef() { return CreateFunctionPass(TransformInitMemRef, "InitMemRef"); }
}  // namespace pass
}  // namespace ir
}  // namespace pypto
