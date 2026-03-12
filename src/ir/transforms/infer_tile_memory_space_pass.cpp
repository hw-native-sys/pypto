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

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Cube ops whose output is always in Acc memory space
const std::unordered_set<std::string> kCubeOps = {
    "tile.matmul",    "tile.matmul_acc", "tile.matmul_bias",   "tile.gemv",           "tile.gemv_acc",
    "tile.gemv_bias", "tile.matmul_mx",  "tile.matmul_mx_acc", "tile.matmul_mx_bias", "tile.batch_matmul"};

// Ops that read target_memory from their kwarg
const std::unordered_set<std::string> kTargetMemoryKwargOps = {"tile.load", "tile.move", "tile.create"};

// Ops that inherit target_memory from their first tile-typed input (view/transform ops)
const std::unordered_set<std::string> kInheritFromInputOps = {"tile.reshape"};

// Extract target_memory kwarg from a Call, defaulting to Vec
MemorySpace ExtractTargetMemoryKwarg(const CallPtr& call) {
  for (const auto& [key, value] : call->kwargs_) {
    if (key == "target_memory") {
      return AnyCast<MemorySpace>(value, "target_memory");
    }
  }
  return MemorySpace::Vec;
}

// ============================================================================
// Phase 1: Analyze - infer memory_space for each tile variable
// ============================================================================

class TileMemorySpaceAnalyzer : public IRVisitor {
 public:
  explicit TileMemorySpaceAnalyzer(const std::vector<VarPtr>& params) {
    for (const auto& var : params) {
      CHECK(!As<TileType>(var->GetType())) << "InCore function parameter '" << var->name_
                                           << "' has TileType, but InCore parameters must be TensorType";
    }
  }

  [[nodiscard]] const std::map<VarPtr, MemorySpace>& GetVarMemory() const { return var_memory_; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op->var_ || !As<TileType>(op->var_->GetType())) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    if (auto call = As<Call>(op->value_)) {
      const std::string& op_name = call->op_->name_;
      if (op_name.rfind("tile.", 0) == 0) {
        var_memory_[op->var_] = InferFromOp(op_name, call);
      } else {
        // Non-tile ops producing TileType (e.g., system.tpop_from_aiv): default to Vec
        var_memory_[op->var_] = MemorySpace::Vec;
      }
    }

    IRVisitor::VisitStmt_(op);
  }

 private:
  std::map<VarPtr, MemorySpace> var_memory_;

  MemorySpace InferFromOp(const std::string& op_name, const CallPtr& call) {
    // Cube ops -> Acc
    if (kCubeOps.count(op_name) > 0) {
      return MemorySpace::Acc;
    }
    // Ops with target_memory kwarg
    if (kTargetMemoryKwargOps.count(op_name) > 0) {
      return ExtractTargetMemoryKwarg(call);
    }
    // View/transform ops: inherit from first tile-typed input
    if (kInheritFromInputOps.count(op_name) > 0) {
      return InheritFromInput(call);
    }
    // All other tile ops: default to Vec
    return MemorySpace::Vec;
  }

  MemorySpace InheritFromInput(const CallPtr& call) {
    for (const auto& arg : call->args_) {
      if (auto var = As<Var>(arg)) {
        auto it = var_memory_.find(var);
        if (it != var_memory_.end()) {
          return it->second;
        }
      }
    }
    return MemorySpace::Vec;
  }
};

// ============================================================================
// Phase 2: Mutate - set memory_space_ on TileType for each variable
// ============================================================================

class TileMemorySpaceMutator : public IRMutator {
 public:
  explicit TileMemorySpaceMutator(const std::map<VarPtr, MemorySpace>& var_memory)
      : var_memory_(var_memory) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) {
      return it->second;
    }

    auto tile_type = As<TileType>(op->GetType());
    auto mem_it = var_memory_.find(op);

    if (tile_type && mem_it != var_memory_.end()) {
      auto new_type = std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_,
                                                 tile_type->tile_view_, mem_it->second);
      auto new_var = std::make_shared<Var>(op->name_, std::move(new_type), op->span_);
      var_cache_[op] = new_var;
      return new_var;
    }

    var_cache_[op] = op;
    return op;
  }

 private:
  const std::map<VarPtr, MemorySpace>& var_memory_;
  std::map<VarPtr, ExprPtr> var_cache_;
};

// ============================================================================
// Transform: combine analysis and mutation for a single InCore function
// ============================================================================

FunctionPtr TransformInferTileMemorySpace(const FunctionPtr& func) {
  // Phase 1: Analyze
  TileMemorySpaceAnalyzer analyzer(func->params_);
  analyzer.VisitStmt(func->body_);

  const auto& var_memory = analyzer.GetVarMemory();
  if (var_memory.empty()) {
    return func;
  }

  // Phase 2: Mutate
  TileMemorySpaceMutator mutator(var_memory);
  auto new_body = mutator.VisitStmt(func->body_);

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_);
}

}  // namespace

// ============================================================================
// Pass factory function
// ============================================================================

namespace pass {

Pass InferTileMemorySpace() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        new_functions[gvar] = TransformInferTileMemorySpace(func);
      } else {
        new_functions[gvar] = func;
      }
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "InferTileMemorySpace", kInferTileMemorySpaceProperties);
}

}  // namespace pass

// ============================================================================
// TileMemoryInferred property verifier
// ============================================================================

namespace {

class TileMemoryInferredVerifier : public IRVisitor {
 public:
  explicit TileMemoryInferredVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) {
      auto tile_type = As<TileType>(op->var_->GetType());
      if (tile_type && !tile_type->memory_space_.has_value()) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileMemoryInferred", 0,
                                  "InCore function '" + func_name_ + "': TileType variable '" +
                                      op->var_->name_ + "' has no memory_space set",
                                  op->var_->span_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

}  // namespace

class TileMemoryInferredPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TileMemoryInferred"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::InCore) continue;
      TileMemoryInferredVerifier verifier(diagnostics, func->name_);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateTileMemoryInferredPropertyVerifier() {
  return std::make_shared<TileMemoryInferredPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
