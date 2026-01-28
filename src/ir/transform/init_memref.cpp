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

#include "pypto/ir/transform/init_memref.h"

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/mutator.h"
#include "pypto/ir/transform/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Helper to extract target_space from Call kwargs
// Returns the memory space based on unified mapping: 0=UB, 1=L1, 2=L0A, 3=L0B
MemorySpace ExtractTargetSpace(const CallPtr& call) {
  // Search for target_space attribute in kwargs
  for (const auto& [key, value] : call->kwargs_) {
    if (key == "target_space") {
      try {
        int space_val = AnyCast<int>(value, "target_space");
        switch (space_val) {
          case 0:
            return MemorySpace::UB;
          case 1:
            return MemorySpace::L1;
          case 2:
            return MemorySpace::L0A;
          case 3:
            return MemorySpace::L0B;
          default:
            LOG_WARN << "Invalid target_space value: " << space_val << ", defaulting to UB";
            return MemorySpace::UB;
        }
      } catch (const pypto::TypeError& e) {
        LOG_WARN << "Failed to cast 'target_space' attribute: " << e.what() << ". Defaulting to UB.";
        return MemorySpace::UB;
      }
    }
  }
  // If target_space not found, default to UB
  return MemorySpace::UB;
}

// Visitor to identify memory space for each variable
// Tracks memory spaces based on function parameters and block operations
class InitMemrefVisitor : public IRVisitor {
 public:
  // Initialize visitor with function parameters (all params should be in DDR)
  explicit InitMemrefVisitor(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      var_memory_spaces_[param.get()] = MemorySpace::DDR;
    }
  }

  const std::map<const Var*, MemorySpace>& GetVarMemorySpaces() const { return var_memory_spaces_; }

  void VisitExpr_(const CallPtr& op) override {
    // Continue visiting arguments
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // Capture assignments for block operations to track memory spaces
    if (auto call = As<Call>(op->value_)) {
      if (call->op_->name_ == "block.load") {
        // 1. Assigned variable -> target_space (from kwargs)
        MemorySpace space = ExtractTargetSpace(call);
        var_memory_spaces_[op->var_.get()] = space;

        // 2. First argument (source tensor) -> DDR
        if (!call->args_.empty()) {
          if (auto source_var = As<Var>(call->args_[0])) {
            var_memory_spaces_[source_var.get()] = MemorySpace::DDR;
          }
        }
      } else if (call->op_->name_ == "block.move") {
        // Assigned variable -> target_space (from kwargs)
        MemorySpace space = ExtractTargetSpace(call);
        var_memory_spaces_[op->var_.get()] = space;
      } else if (call->op_->name_ == "block.store") {
        // 1. Assigned variable -> DDR (return value is output tensor)
        var_memory_spaces_[op->var_.get()] = MemorySpace::DDR;

        // 2. Sixth argument (output tensor) -> DDR
        constexpr size_t BLOCK_STORE_OUTPUT_ARG_INDEX = 5;
        if (call->args_.size() > BLOCK_STORE_OUTPUT_ARG_INDEX) {
          if (auto output_var = As<Var>(call->args_[BLOCK_STORE_OUTPUT_ARG_INDEX])) {
            var_memory_spaces_[output_var.get()] = MemorySpace::DDR;
          }
        }
      } else if (call->op_->name_ == "block.matmul") {
        // Return value -> L0C
        var_memory_spaces_[op->var_.get()] = MemorySpace::L0C;
      } else if (call->op_->name_ == "block.matmul_acc") {
        // Return value -> L0C
        var_memory_spaces_[op->var_.get()] = MemorySpace::L0C;

        // First argument (acc) -> L0C
        if (!call->args_.empty()) {
          if (auto acc_var = As<Var>(call->args_[0])) {
            var_memory_spaces_[acc_var.get()] = MemorySpace::L0C;
          }
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::map<const Var*, MemorySpace> var_memory_spaces_;
};

// Mutator to initialize MemRef for variables
class InitMemRefMutator : public IRMutator {
 public:
  explicit InitMemRefMutator(const std::map<const Var*, MemorySpace>& var_memory_spaces)
      : var_memory_spaces_(var_memory_spaces), next_id_(0) {}

  // Helper to calculate size and create MemRef
  std::optional<MemRefPtr> CreateMemRef(const ShapedTypePtr& type, const Var* old_var_ptr) {
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

    // Query memory space from var_memory_spaces_ map
    MemorySpace space = MemorySpace::UB;  // Default to UB
    auto it = var_memory_spaces_.find(old_var_ptr);
    if (it != var_memory_spaces_.end()) {
      space = it->second;
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
    const Var* key = old_var.get();
    auto it = var_map_.find(key);
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
        auto memref = CreateMemRef(tensor_type, key);
        new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, memref);
      } else if (auto tile_type = As<TileType>(old_var->GetType())) {
        auto memref = CreateMemRef(tile_type, key);
        new_type =
            std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, memref, tile_type->tile_view_);
      }

      new_var = std::make_shared<Var>(old_var->name_, new_type, old_var->span_);
    }

    var_map_[key] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const VarPtr& op) override { return GetNewVar(op); }

  ExprPtr VisitExpr_(const IterArgPtr& op) override { return GetNewVar(op); }

  // Handle block.store specially: return value should share the same MemRef as the 6th argument
  StmtPtr VisitStmt_(const AssignStmtPtr& op) {
    // First visit the value (RHS)
    auto new_value = VisitExpr(op->value_);

    // Check if the RHS is a block.store call
    constexpr size_t BLOCK_STORE_OUTPUT_ARG_INDEX = 5;
    if (auto call = As<Call>(op->value_)) {
      if (call->op_->name_ == "block.store" && call->args_.size() > BLOCK_STORE_OUTPUT_ARG_INDEX) {
        // Get the 6th argument (output tensor) after mutation
        auto new_call = As<Call>(new_value);
        if (new_call && new_call->args_.size() > BLOCK_STORE_OUTPUT_ARG_INDEX) {
          auto output_tensor_arg = new_call->args_[BLOCK_STORE_OUTPUT_ARG_INDEX];

          // Extract MemRef from the output tensor
          std::optional<MemRefPtr> shared_memref = std::nullopt;
          if (auto tensor_type = As<TensorType>(output_tensor_arg->GetType())) {
            shared_memref = tensor_type->memref_;
          }

          // Create new variable with the shared MemRef
          if (shared_memref.has_value()) {
            TypePtr new_type = op->var_->GetType();
            if (auto var_tensor_type = As<TensorType>(op->var_->GetType())) {
              // Reuse the MemRef from the 6th argument
              new_type = std::make_shared<TensorType>(var_tensor_type->shape_, var_tensor_type->dtype_,
                                                      shared_memref);
            }

            VarPtr new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
            var_map_[op->var_.get()] = new_var;

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
  const std::map<const Var*, MemorySpace>& var_memory_spaces_;
  std::map<const Var*, VarPtr> var_map_;
  uint64_t next_id_;  // Counter for generating unique MemRef IDs
};

}  // namespace

FunctionPtr InitMemRefPass::Run(const FunctionPtr& func) {
  // Step 1: Analyze usage to determine memory space for each variable
  // All function parameters are in DDR (main memory)
  InitMemrefVisitor visitor(func->params_);
  visitor.VisitStmt(func->body_);

  // Step 2: Mutate variables to initialize their MemRef
  InitMemRefMutator mutator(visitor.GetVarMemorySpaces());

  // Process params first to define them in the map
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  for (const auto& param : func->params_) {
    // Cast ExprPtr back to VarPtr for GetNewVar
    auto new_param_expr = mutator.GetNewVar(param);
    auto new_param = As<Var>(std::static_pointer_cast<const IRNode>(new_param_expr));
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
