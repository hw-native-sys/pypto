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
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using transform_utils::GetLastYieldStmt;

namespace {

// Unregistered cube ops (not yet registered via REGISTER_OP but still need Acc output)
const std::unordered_set<std::string> kUnregisteredCubeOps = {"tile.matmul_mx", "tile.matmul_mx_acc",
                                                              "tile.matmul_mx_bias"};

// Look up input constraints for an op. Returns nullptr if none.
const std::vector<std::vector<MemorySpace>>* GetInputConstraints(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return nullptr;
  const auto& spec_opt = registry.GetEntry(op_name).GetMemorySpec();
  if (!spec_opt.has_value()) return nullptr;
  return &spec_opt->input_constraints;
}

// ============================================================================
// Phase 1: Analyze - infer memory_space for each tile variable
// ============================================================================

class TileMemorySpaceAnalyzer : public IRVisitor {
 public:
  explicit TileMemorySpaceAnalyzer(const std::vector<VarPtr>& params) {
    for (const auto& var : params) {
      CHECK(!As<TileType>(var->GetType())) << "InCore function parameter '" << var->name_hint_
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
        // Non-tile ops producing TileType: default to Vec
        var_memory_[op->var_] = MemorySpace::Vec;
      }
    }

    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);

    if (op->return_vars_.empty()) return;

    auto yield_stmt = GetLastYieldStmt(op->body_);
    if (!yield_stmt) return;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!As<TileType>(op->return_vars_[i]->GetType())) continue;
      if (i < yield_stmt->value_.size()) {
        if (auto yield_var = As<Var>(yield_stmt->value_[i])) {
          auto it = var_memory_.find(yield_var);
          if (it != var_memory_.end()) {
            var_memory_[op->return_vars_[i]] = it->second;
          }
        }
      }
    }
  }

 private:
  std::map<VarPtr, MemorySpace> var_memory_;

  MemorySpace InferFromOp(const std::string& op_name, const CallPtr& call) {
    auto& registry = OpRegistry::GetInstance();

    // Handle unregistered ops (backward compat)
    if (!registry.IsRegistered(op_name)) {
      if (kUnregisteredCubeOps.count(op_name) > 0) return MemorySpace::Acc;
      return MemorySpace::Vec;
    }

    const auto& spec_opt = registry.GetEntry(op_name).GetMemorySpec();
    if (!spec_opt.has_value() || !spec_opt->deduce_output_memory) {
      // no_memory_spec ops (e.g. tile.tpop_*): read memory_space from Call return type
      if (auto tile_type = As<TileType>(call->GetType())) {
        if (tile_type->memory_space_.has_value() && *tile_type->memory_space_ != MemorySpace::DDR) {
          return *tile_type->memory_space_;
        }
      }
      return MemorySpace::Vec;
    }

    auto result = spec_opt->deduce_output_memory(call->kwargs_);
    if (result.has_value()) {
      return *result;
    }
    // nullopt -> inherit from first tile-typed input (view ops)
    return InheritFromInput(call);
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
// Phase 2: Collect needed tile.move insertions for input constraint mismatches
// ============================================================================

// Key: (producer variable, target memory space)
using MoveKey = std::pair<VarPtr, MemorySpace>;
struct MoveKeyLess {
  bool operator()(const MoveKey& a, const MoveKey& b) const {
    if (a.first != b.first) return a.first < b.first;
    return static_cast<int>(a.second) < static_cast<int>(b.second);
  }
};

class MoveCollector : public IRVisitor {
 public:
  explicit MoveCollector(const std::map<VarPtr, MemorySpace>& var_memory) : var_memory_(var_memory) {}

  [[nodiscard]] const std::set<MoveKey, MoveKeyLess>& GetNeededMoves() const { return needed_moves_; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_)) {
      CheckInputConstraints(call);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  const std::map<VarPtr, MemorySpace>& var_memory_;
  std::set<MoveKey, MoveKeyLess> needed_moves_;

  void CheckInputConstraints(const CallPtr& call) {
    const auto* constraints = GetInputConstraints(call->op_->name_);
    if (!constraints) return;

    for (size_t i = 0; i < constraints->size() && i < call->args_.size(); ++i) {
      const auto& allowed_spaces = (*constraints)[i];
      if (allowed_spaces.empty()) continue;

      auto var = As<Var>(call->args_[i]);
      if (!var) continue;
      auto it = var_memory_.find(var);
      if (it == var_memory_.end()) continue;

      bool allowed =
          std::find(allowed_spaces.begin(), allowed_spaces.end(), it->second) != allowed_spaces.end();
      if (!allowed) {
        needed_moves_.insert({var, allowed_spaces[0]});
      }
    }
  }
};

// ============================================================================
// Phase 3: Mutate - set memory_space_, insert tile.move, substitute args
// ============================================================================

class TileMemorySpaceMutator : public IRMutator {
 public:
  TileMemorySpaceMutator(const std::map<VarPtr, MemorySpace>& var_memory,
                         const std::set<MoveKey, MoveKeyLess>& needed_moves)
      : var_memory_(var_memory), needed_moves_(needed_moves) {}

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
      auto new_var = std::make_shared<Var>(op->name_hint_, std::move(new_type), op->span_);
      var_cache_[op] = new_var;
      return new_var;
    }

    var_cache_[op] = op;
    return op;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    const auto* constraints = GetInputConstraints(op->op_->name_);

    std::vector<ExprPtr> new_args;
    bool changed = false;
    new_args.reserve(op->args_.size());

    for (size_t i = 0; i < op->args_.size(); ++i) {
      bool substituted = false;
      if (constraints && i < constraints->size() && !(*constraints)[i].empty()) {
        if (auto var = As<Var>(op->args_[i])) {
          MoveKey key = {var, (*constraints)[i][0]};
          auto move_it = created_moves_.find(key);
          if (move_it != created_moves_.end()) {
            new_args.push_back(move_it->second);
            changed = true;
            substituted = true;
          }
        }
      }
      if (!substituted) {
        auto new_arg = IRMutator::VisitExpr(op->args_[i]);
        new_args.push_back(new_arg);
        if (new_arg.get() != op->args_[i].get()) changed = true;
      }
    }

    if (!changed) return op;
    return std::make_shared<Call>(op->op_, std::move(new_args), op->kwargs_, op->GetType(), op->span_);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    bool changed = false;
    auto new_stmts = VisitAndInsertMoves(op->stmts_, changed);
    if (!changed) return op;
    return std::make_shared<SeqStmts>(std::move(new_stmts), op->span_);
  }

 private:
  const std::map<VarPtr, MemorySpace>& var_memory_;
  const std::set<MoveKey, MoveKeyLess>& needed_moves_;
  std::map<VarPtr, ExprPtr> var_cache_;
  std::map<MoveKey, ExprPtr, MoveKeyLess> created_moves_;

  std::vector<StmtPtr> VisitAndInsertMoves(const std::vector<StmtPtr>& stmts, bool& changed) {
    std::vector<StmtPtr> new_stmts;
    for (const auto& stmt : stmts) {
      auto new_stmt = IRMutator::VisitStmt(stmt);
      if (new_stmt.get() != stmt.get()) changed = true;
      new_stmts.push_back(new_stmt);

      // After producer AssignStmt, insert tile.move stmts if needed
      if (auto assign = As<AssignStmt>(stmt)) {
        for (const auto& key : needed_moves_) {
          if (key.first == assign->var_) {
            InsertMoveStmt(new_stmts, assign->var_, key.second, assign->span_);
            changed = true;
          }
        }
      }
    }
    return new_stmts;
  }

  void InsertMoveStmt(std::vector<StmtPtr>& stmts, const VarPtr& original_var, MemorySpace target,
                      const Span& span) {
    // Get the mutated producer var
    auto cache_it = var_cache_.find(original_var);
    INTERNAL_CHECK(cache_it != var_cache_.end()) << "Internal error: producer var not in cache";
    auto mutated_producer = cache_it->second;

    // Create tile.move call via OpRegistry
    auto& op_reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", std::any(target)}};
    auto move_call = op_reg.Create("tile.move", {mutated_producer}, kwargs, span);

    // Create moved var with memory_space_ set
    auto move_type = As<TileType>(move_call->GetType());
    INTERNAL_CHECK(move_type) << "Internal error: tile.move return type is not TileType";
    auto moved_type = std::make_shared<TileType>(move_type->shape_, move_type->dtype_, move_type->memref_,
                                                 move_type->tile_view_, target);
    auto mutated_producer_var = As<Var>(mutated_producer);
    INTERNAL_CHECK(mutated_producer_var) << "Internal error: mutated producer is not a Var";
    auto moved_var = std::make_shared<Var>(
        mutated_producer_var->name_hint_ + "_" + MemorySpaceToString(target), std::move(moved_type), span);

    // Register for substitution and in var_cache_ so VisitExpr_(VarPtr) returns it as-is
    MoveKey key = {original_var, target};
    created_moves_[key] = moved_var;
    var_cache_[moved_var] = moved_var;

    stmts.push_back(std::make_shared<AssignStmt>(moved_var, move_call, span));
  }
};

// ============================================================================
// Transform: combine analysis, move collection, and mutation
// ============================================================================

FunctionPtr TransformInferTileMemorySpace(const FunctionPtr& func) {
  // Phase 1: Analyze — infer memory space for each tile variable
  TileMemorySpaceAnalyzer analyzer(func->params_);
  analyzer.VisitStmt(func->body_);

  const auto& var_memory = analyzer.GetVarMemory();
  if (var_memory.empty()) {
    return func;
  }

  // Phase 2: Collect needed tile.move insertions
  MoveCollector collector(var_memory);
  collector.VisitStmt(func->body_);

  // Phase 3: Mutate — set memory_space_ on types, insert moves, substitute args
  TileMemorySpaceMutator mutator(var_memory, collector.GetNeededMoves());
  auto new_body = mutator.VisitStmt(func->body_);

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_);
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
                                      op->var_->name_hint_ + "' has no memory_space set",
                                  op->var_->span_);
      }
    }

    // Verify input memory space constraints
    if (auto call = As<Call>(op->value_)) {
      VerifyInputConstraints(call);
    }

    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;

  void VerifyInputConstraints(const CallPtr& call) {
    const auto* constraints = GetInputConstraints(call->op_->name_);
    if (!constraints) return;

    for (size_t i = 0; i < constraints->size() && i < call->args_.size(); ++i) {
      const auto& allowed_spaces = (*constraints)[i];
      if (allowed_spaces.empty()) continue;

      auto var = As<Var>(call->args_[i]);
      if (!var) continue;
      auto tile_type = As<TileType>(var->GetType());
      if (!tile_type || !tile_type->memory_space_.has_value()) continue;

      MemorySpace actual = *tile_type->memory_space_;
      bool allowed = std::find(allowed_spaces.begin(), allowed_spaces.end(), actual) != allowed_spaces.end();
      if (!allowed) {
        std::string allowed_str;
        for (size_t j = 0; j < allowed_spaces.size(); ++j) {
          if (j > 0) allowed_str += "/";
          allowed_str += MemorySpaceToString(allowed_spaces[j]);
        }
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileMemoryInferred", 0,
                                  "InCore function '" + func_name_ + "': Op '" + call->op_->name_ +
                                      "' input " + std::to_string(i) + " ('" + var->name_hint_ +
                                      "') requires " + allowed_str + " but is in " +
                                      MemorySpaceToString(actual),
                                  var->span_);
      }
    }
  }
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
