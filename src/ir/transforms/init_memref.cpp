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
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Helper to extract target_memory from Call kwargs
MemorySpace ExtractTargetMemory(const CallPtr& call) {
  for (const auto& [key, value] : call->kwargs_) {
    if (key == "target_memory") {
      try {
        int memory_val = AnyCast<int>(value, "target_memory");
        // Validate range: MemorySpace enum values are 0-5 (DDR, UB, L1, L0A, L0B, L0C)
        if (memory_val < 0 || memory_val > 5) {
          LOG_ERROR << "Invalid target_memory value: " << memory_val << ", defaulting to UB";
          return MemorySpace::UB;
        }
        return static_cast<MemorySpace>(memory_val);
      } catch (const std::exception& e) {
        LOG_ERROR << "Failed to cast 'target_memory' attribute: " << e.what() << ". Defaulting to UB.";
        return MemorySpace::UB;
      }
    }
  }
  // If target_memory not found, default to UB
  return MemorySpace::UB;
}

// Return value memory space rules for block operators
const std::map<std::string, std::optional<MemorySpace>> kBlockOpMemoryRules = {
    {"block.create_tile", std::nullopt},     // Extract from target_memory
    {"block.load", std::nullopt},            // Extract from target_memory
    {"block.move", std::nullopt},            // Extract from target_memory
    {"block.store", MemorySpace::DDR},       // Fixed DDR
    {"block.matmul", MemorySpace::L0C},      // Fixed L0C
    {"block.matmul_acc", MemorySpace::L0C},  // Fixed L0C
};

// Visitor to identify memory space for each variable
class MemRefUsageVisitor : public IRVisitor {
 public:
  // Initialize visitor with function parameters (all params should be in DDR)
  explicit MemRefUsageVisitor(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      var_memory_spaces_[param] = MemorySpace::DDR;
    }
  }

  [[nodiscard]] const std::map<VarPtr, MemorySpace>& GetVarMemorySpaces() const { return var_memory_spaces_; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = std::dynamic_pointer_cast<const Call>(op->value_)) {
      // Check if this is a block operation (op name starts with "block.")
      const std::string& op_name = call->op_->name_;
      if (op_name.rfind("block.", 0) == 0) {
        // Look up memory assignment rules for this operator
        auto it = kBlockOpMemoryRules.find(op_name);
        MemorySpace space;

        if (it != kBlockOpMemoryRules.end()) {
          // Operator in rules table
          const auto& mem_space_opt = it->second;
          if (mem_space_opt.has_value()) {
            // Fixed memory space
            space = mem_space_opt.value();
          } else {
            // Extract from target_memory kwarg
            space = ExtractTargetMemory(call);
          }
        } else {
          // Block operation not in rules table, default to UB
          space = MemorySpace::UB;
        }

        var_memory_spaces_[op->var_] = space;
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

 private:
  std::map<VarPtr, MemorySpace> var_memory_spaces_;
};

// Mutator to initialize MemRef for variables
class InitMemRefMutator : public IRMutator {
 public:
  explicit InitMemRefMutator(const std::map<VarPtr, MemorySpace>& var_memory_spaces)
      : var_memory_spaces_(var_memory_spaces) {}

  // Helper to calculate size and create MemRef
  std::optional<MemRefPtr> CreateMemRef(const ShapedTypePtr& type, const VarPtr& var) {
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
    MemorySpace space = MemorySpace::DDR;  // Default to DDR
    auto it = var_memory_spaces_.find(var);
    if (it != var_memory_spaces_.end()) {
      space = it->second;
    }

    // Addr is always 0
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());

    // Generate unique ID for this MemRef
    uint64_t id = next_id_++;

    return std::make_shared<MemRef>(space, addr, size_bytes, id);
  }

  // Clone a type with specified MemRef (handles TensorType and TileType)
  TypePtr CloneTypeWithMemRef(const TypePtr& original_type, const std::optional<MemRefPtr>& memref) {
    if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(original_type)) {
      return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, memref);
    }

    if (auto tile_type = std::dynamic_pointer_cast<const TileType>(original_type)) {
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, memref, tile_type->tile_view_);
    }

    // For non-ShapedTypes, return as-is
    return original_type;
  }

  // Extract MemRef from ShapedType (TensorType or TileType)
  std::optional<MemRefPtr> ExtractMemRefFromType(const TypePtr& type) {
    if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(type)) {
      return tensor_type->memref_;
    }

    if (auto tile_type = std::dynamic_pointer_cast<const TileType>(type)) {
      return tile_type->memref_;
    }

    return std::nullopt;
  }

  // Process IterArg variable (inherits MemRef from initValue)
  VarPtr ProcessIterArg(const VarPtr& old_var) {
    auto iter_arg = std::static_pointer_cast<const IterArg>(old_var);

    // Visit initValue to get its updated MemRef
    auto new_init = VisitExpr(iter_arg->initValue_);

    // Extract MemRef from initValue and create new type
    auto memref = ExtractMemRefFromType(new_init->GetType());
    auto old_var_expr = std::static_pointer_cast<const Expr>(old_var);
    TypePtr new_type = CloneTypeWithMemRef(old_var_expr->GetType(), memref);

    return std::make_shared<IterArg>(iter_arg->name_, new_type, new_init, iter_arg->span_);
  }

  // Process normal Var variable (creates new MemRef based on usage)
  VarPtr ProcessNormalVar(const VarPtr& var) {
    auto var_expr = std::static_pointer_cast<const Expr>(var);
    TypePtr new_type = var_expr->GetType();

    // Process Type if it is ShapedType (TensorType or TileType)
    if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(var_expr->GetType())) {
      auto memref = CreateMemRef(shaped_type, var);
      new_type = CloneTypeWithMemRef(var_expr->GetType(), memref);
    }

    return std::make_shared<Var>(var->name_, new_type, var->span_);
  }

  // Create a new Var with MemRef initialized
  VarPtr GetNewVar(const VarPtr& old_var) {
    // Check cache first to prevent infinite recursion
    auto it = var_map_.find(old_var);
    if (it != var_map_.end()) {
      return it->second;
    }

    // Dispatch based on variable type
    VarPtr new_var;
    if (std::dynamic_pointer_cast<const IterArg>(old_var)) {
      new_var = ProcessIterArg(old_var);
    } else {
      new_var = ProcessNormalVar(old_var);
    }

    var_map_[old_var] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    return std::static_pointer_cast<const Expr>(GetNewVar(op));
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // IterArg extends Var, so cast to VarPtr for processing
    auto var_ptr = std::static_pointer_cast<const Var>(op);
    return std::static_pointer_cast<const Expr>(GetNewVar(var_ptr));
  }

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
          auto shared_memref = ExtractMemRefFromType(output_tensor_arg->GetType());

          // Create new variable with the shared MemRef
          if (shared_memref.has_value()) {
            TypePtr new_type = CloneTypeWithMemRef(op->var_->GetType(), shared_memref);

            VarPtr new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
            var_map_[op->var_] = new_var;

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
  const std::map<VarPtr, MemorySpace>& var_memory_spaces_;
  std::map<VarPtr, VarPtr> var_map_;
  uint64_t next_id_ = 0;  // Counter for generating unique MemRef IDs
};

/**
 * @brief Initialize MemRef for all variables in a function
 *
 * This transformation initializes the MemRef field for all Var nodes in the function.
 * Memory space assignment rules:
 * - Function parameters -> DDR
 * - block.load/block.move return values -> Extract from target_memory kwarg (default UB)
 * - block.store return values -> DDR
 * - block.matmul/block.matmul_acc return values -> L0C
 * - Other block operations (not in rules table) -> UB
 * - Other variables -> DDR (default)
 */
FunctionPtr TransformInitMemRef(const FunctionPtr& func) {
  // Step 1: Analyze usage to determine memory space for each variable
  // All function parameters are in DDR (main memory)
  MemRefUsageVisitor visitor(func->params_);
  visitor.VisitStmt(func->body_);

  // Step 2: Mutate variables to initialize their MemRef
  InitMemRefMutator mutator(visitor.GetVarMemorySpaces());

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
  return std::make_shared<Function>(func->name_, new_params, func->return_types_, new_body, func->span_,
                                    func->func_type_);
}

}  // namespace

// Factory function
namespace pass {
Pass InitMemRef() { return CreateFunctionPass(TransformInitMemRef, "InitMemRef"); }
}  // namespace pass
}  // namespace ir
}  // namespace pypto
