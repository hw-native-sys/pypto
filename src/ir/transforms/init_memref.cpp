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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Resolve memory space for tile op output using registry metadata.
// tile.store is special-cased since it returns TensorType (DDR), not TileType.
MemorySpace ResolveMemorySpace(const std::string& op_name, const CallPtr& call) {
  if (op_name == "tile.store") return MemorySpace::DDR;

  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return MemorySpace::Vec;

  const auto& spec_opt = registry.GetEntry(op_name).GetMemorySpec();
  if (!spec_opt.has_value() || !spec_opt->deduce_output_memory) {
    return MemorySpace::Vec;
  }

  auto result = spec_opt->deduce_output_memory(call->kwargs_);
  return result.value_or(MemorySpace::Vec);
}

// Check if operation is a view operation (zero-copy metadata transform)
// using the registry: deduce_output_memory returning nullopt = view op.
bool IsViewOperation(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return false;

  const auto& spec_opt = registry.GetEntry(op_name).GetMemorySpec();
  if (!spec_opt.has_value() || !spec_opt->deduce_output_memory) return false;

  return !spec_opt->deduce_output_memory({}).has_value();
}

// Helper to find the YieldStmt inside a statement body (searches through SeqStmts/OpStmts)
YieldStmtPtr FindYieldStmt(const StmtPtr& body) {
  if (auto yield = As<YieldStmt>(body)) return yield;
  if (auto seq = As<SeqStmts>(body)) {
    for (const auto& child : seq->stmts_) {
      auto result = FindYieldStmt(child);
      if (result) return result;
    }
  }
  if (auto ops = As<OpStmts>(body)) {
    for (const auto& child : ops->stmts_) {
      auto result = FindYieldStmt(child);
      if (result) return result;
    }
  }
  return nullptr;
}

// Visitor to identify memory space for each variable
class MemRefUsageVisitor : public IRVisitor {
 public:
  // Initialize visitor with function parameters (all params should be in DDR)
  explicit MemRefUsageVisitor(const std::vector<VarPtr>& params,
                              const std::vector<ParamDirection>& /*param_directions*/) {
    for (const auto& var : params) {
      var_memory_spaces_[var] = MemorySpace::DDR;
    }
  }

  [[nodiscard]] const std::map<VarPtr, MemorySpace>& GetVarMemorySpaces() const { return var_memory_spaces_; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = std::dynamic_pointer_cast<const Call>(op->value_)) {
      // Check if this is a tile operation (op name starts with "tile.")
      const std::string& op_name = call->op_->name_;
      if (op_name.rfind("tile.", 0) == 0) {
        var_memory_spaces_[op->var_] = ResolveMemorySpace(op_name, call);
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

  void VisitStmt_(const IfStmtPtr& op) override {
    // Visit bodies first to populate yield values' memory spaces
    IRVisitor::VisitStmt_(op);

    // Propagate memory spaces from yield values to return_vars
    if (op->return_vars_.empty()) return;

    // Collect memory spaces from both branches
    auto then_yield = FindYieldStmt(op->then_body_);
    auto else_yield = op->else_body_.has_value() ? FindYieldStmt(op->else_body_.value()) : nullptr;
    if (!then_yield && !else_yield) return;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      // Try then branch first
      if (then_yield && i < then_yield->value_.size()) {
        if (auto yield_var = As<Var>(then_yield->value_[i])) {
          auto it = var_memory_spaces_.find(yield_var);
          if (it != var_memory_spaces_.end()) {
            var_memory_spaces_[op->return_vars_[i]] = it->second;
            continue;
          }
        }
      }
      // Fall back to else branch
      if (else_yield && i < else_yield->value_.size()) {
        if (auto yield_var = As<Var>(else_yield->value_[i])) {
          auto it = var_memory_spaces_.find(yield_var);
          if (it != var_memory_spaces_.end()) {
            var_memory_spaces_[op->return_vars_[i]] = it->second;
          }
        }
      }
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    // Visit body first to populate yield values' memory spaces
    IRVisitor::VisitStmt_(op);

    // Propagate memory spaces from yield values to return_vars
    if (op->return_vars_.empty()) return;

    auto yield_stmt = FindYieldStmt(op->body_);
    if (!yield_stmt) return;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (i < yield_stmt->value_.size()) {
        if (auto yield_var = As<Var>(yield_stmt->value_[i])) {
          auto it = var_memory_spaces_.find(yield_var);
          if (it != var_memory_spaces_.end()) {
            var_memory_spaces_[op->return_vars_[i]] = it->second;
          }
        }
      }
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

    // Addr is -1 (unallocated)
    auto addr = std::make_shared<ConstInt>(-1, DataType::INDEX, Span::unknown());

    // Generate unique ID for this MemRef
    uint64_t id = next_id_++;

    return std::make_shared<MemRef>(space, addr, size_bytes, id);
  }

  // Clone a type with specified MemRef (handles TensorType and TileType)
  TypePtr CloneTypeWithMemRef(const TypePtr& original_type, const std::optional<MemRefPtr>& memref) {
    if (auto tensor_type = std::dynamic_pointer_cast<const TensorType>(original_type)) {
      return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, memref,
                                          tensor_type->tensor_view_);
    }

    if (auto tile_type = std::dynamic_pointer_cast<const TileType>(original_type)) {
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, memref, tile_type->tile_view_,
                                        tile_type->memory_space_);
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

  // Handle tile.store specially: return value should share the same MemRef as the 3rd argument
  // (output_tensor)
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // First visit the value (RHS)
    auto new_value = VisitExpr(op->value_);

    // Check if the RHS is a Call expression
    if (auto call = std::dynamic_pointer_cast<const Call>(op->value_)) {
      LOG_DEBUG << "Processing AssignStmt for " << op->var_->name_ << " with call to " << call->op_->name_;

      // Handle view operations: output should share MemRef with input tile
      if (IsViewOperation(call->op_->name_) && call->args_.size() > 0) {
        LOG_DEBUG << "Detected view operation: " << call->op_->name_;
        // Get the input tile (first argument) after mutation
        auto new_call = std::dynamic_pointer_cast<const Call>(new_value);
        if (new_call) {
          auto input_tile_arg = new_call->args_[0];

          // Extract MemRef from input tile
          auto shared_memref = ExtractMemRefFromType(input_tile_arg->GetType());

          // Create new variable with shared MemRef
          if (shared_memref.has_value()) {
            LOG_DEBUG << "Sharing MemRef from input tile to " << op->var_->name_;
            TypePtr new_type = CloneTypeWithMemRef(op->var_->GetType(), shared_memref);
            VarPtr new_var = std::make_shared<Var>(op->var_->name_, new_type, op->var_->span_);
            var_map_[op->var_] = new_var;

            return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
          } else {
            LOG_DEBUG << "Input tile has no MemRef yet";
          }
        }
      }

      // Check if the RHS is a tile.store call
      if (call->op_->name_ == "tile.store") {
        // Get the 3rd argument (output tensor) after mutation
        auto new_call = std::dynamic_pointer_cast<const Call>(new_value);
        if (new_call && new_call->args_.size() > 2) {
          auto output_tensor_arg = new_call->args_[2];

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

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Let the default mutator process iter_args, body, and return_vars
    auto result = IRMutator::VisitStmt_(op);
    auto new_for = As<ForStmt>(result);
    if (!new_for || new_for->iter_args_.empty() || new_for->return_vars_.empty()) {
      return result;
    }

    // Make each return_var share the same MemRef as the corresponding iter_arg
    // (MLIR scf.for requires result types to match iter_arg types)
    bool changed = false;
    std::vector<VarPtr> patched_return_vars;
    patched_return_vars.reserve(new_for->return_vars_.size());

    for (size_t i = 0; i < new_for->return_vars_.size(); ++i) {
      if (i >= new_for->iter_args_.size()) {
        patched_return_vars.push_back(new_for->return_vars_[i]);
        continue;
      }

      auto rv_tile = As<TileType>(new_for->return_vars_[i]->GetType());
      auto ia_tile = As<TileType>(new_for->iter_args_[i]->GetType());
      if (rv_tile && ia_tile && ia_tile->memref_.has_value()) {
        auto new_type = CloneTypeWithMemRef(new_for->return_vars_[i]->GetType(), ia_tile->memref_);
        auto new_rv =
            std::make_shared<Var>(new_for->return_vars_[i]->name_, new_type, new_for->return_vars_[i]->span_);
        // Update the cache so downstream references use the patched var
        var_map_[op->return_vars_[i]] = new_rv;
        patched_return_vars.push_back(new_rv);
        changed = true;
      } else {
        patched_return_vars.push_back(new_for->return_vars_[i]);
      }
    }

    if (!changed) return result;

    return std::make_shared<ForStmt>(new_for->loop_var_, new_for->start_, new_for->stop_, new_for->step_,
                                     new_for->iter_args_, new_for->body_, std::move(patched_return_vars),
                                     new_for->span_, new_for->kind_, new_for->chunk_size_,
                                     new_for->chunk_policy_, new_for->loop_origin_);
  }

 private:
  const std::map<VarPtr, MemorySpace>& var_memory_spaces_;
  std::map<VarPtr, VarPtr> var_map_;
  uint64_t next_id_ = 0;  // Counter for generating unique MemRef IDs
};

// Visitor to collect unique non-DDR MemRef objects from TileType variables
class NonDDRMemRefCollector : public IRVisitor {
 public:
  [[nodiscard]] const std::vector<MemRefPtr>& GetMemRefs() const { return memrefs_; }

  void VisitExpr_(const VarPtr& op) override {
    if (auto tile_type = As<TileType>(op->GetType())) {
      if (tile_type->memref_.has_value()) {
        AddMemRefIfUnique(tile_type->memref_.value());
      }
    }
  }

  void VisitExpr_(const IterArgPtr& op) override {
    if (auto tile_type = As<TileType>(op->GetType())) {
      if (tile_type->memref_.has_value()) {
        AddMemRefIfUnique(tile_type->memref_.value());
      }
    }
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const MemRef*> seen_ptrs_;

  void AddMemRefIfUnique(const MemRefPtr& memref) {
    if (memref->memory_space_ == MemorySpace::DDR) return;
    const MemRef* raw_ptr = memref.get();
    if (seen_ptrs_.find(raw_ptr) == seen_ptrs_.end()) {
      memrefs_.push_back(memref);
      seen_ptrs_.insert(raw_ptr);
    }
  }
};

// Create tile.alloc AssignStmt for a MemRef with addr=-1 (unallocated)
StmtPtr CreateAllocStatement(const MemRefPtr& memref) {
  auto alloc_op = std::make_shared<Op>("tile.alloc");

  auto memspace_expr = std::make_shared<ConstInt>(static_cast<int64_t>(memref->memory_space_),
                                                  DataType::INDEX, Span::unknown());
  ExprPtr addr_expr = memref->addr_;
  auto size_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(memref->size_), DataType::INDEX, Span::unknown());
  auto id_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(memref->id_), DataType::INDEX, Span::unknown());

  std::vector<ExprPtr> alloc_args = {memspace_expr, addr_expr, size_expr, id_expr};
  auto alloc_call = std::make_shared<Call>(alloc_op, alloc_args, GetMemRefType(), Span::unknown());

  return std::make_shared<AssignStmt>(memref, alloc_call, Span::unknown());
}

// Insert alloc statements at the beginning of the first OpStmts in a SeqStmts body
StmtPtr InsertAllocsIntoBody(const StmtPtr& body, const std::vector<StmtPtr>& alloc_stmts) {
  if (alloc_stmts.empty()) return body;

  auto seq = As<SeqStmts>(body);
  if (!seq || seq->stmts_.empty()) return body;

  std::vector<StmtPtr> new_seq_stmts;
  bool inserted = false;

  for (const auto& child : seq->stmts_) {
    if (!inserted) {
      if (auto op_stmts = As<OpStmts>(child)) {
        // Prepend alloc statements into the first OpStmts
        std::vector<StmtPtr> merged = alloc_stmts;
        merged.insert(merged.end(), op_stmts->stmts_.begin(), op_stmts->stmts_.end());
        new_seq_stmts.push_back(std::make_shared<OpStmts>(merged, child->span_));
        inserted = true;
        continue;
      }
    }
    new_seq_stmts.push_back(child);
  }

  if (!inserted) {
    // No OpStmts found — create one at the beginning
    auto new_op_stmts = std::make_shared<OpStmts>(alloc_stmts, Span::unknown());
    new_seq_stmts.insert(new_seq_stmts.begin(), new_op_stmts);
  }

  return std::make_shared<SeqStmts>(new_seq_stmts, body->span_);
}

/**
 * @brief Initialize MemRef for all variables in a function
 *
 * This transformation:
 * 1. Normalizes statement structure (ensures SeqStmts/OpStmts)
 * 2. Initializes the MemRef field for all Var nodes
 * 3. Creates tile.alloc operations for non-DDR MemRefs (addr=-1, unallocated)
 *
 * Memory space assignment:
 * - Function parameters -> DDR
 * - tile.store return values -> DDR (special-cased, returns TensorType)
 * - Other tile ops -> resolved via OpRegistry memory specs (see OpMemorySpaceSpec)
 * - Non-tile variables -> DDR (default)
 */
FunctionPtr TransformInitMemRef(const FunctionPtr& func) {
  // Step 1: Normalize statement structure to ensure SeqStmts/OpStmts
  auto normalized_func = NormalizeStmtStructure(func);

  // Step 2: Analyze usage to determine memory space for each variable
  MemRefUsageVisitor visitor(normalized_func->params_, normalized_func->param_directions_);
  visitor.VisitStmt(normalized_func->body_);

  // Step 3: Mutate variables to initialize their MemRef
  InitMemRefMutator mutator(visitor.GetVarMemorySpaces());

  std::vector<VarPtr> new_params;
  new_params.reserve(normalized_func->params_.size());
  for (const auto& var : normalized_func->params_) {
    auto new_param = mutator.GetNewVar(var);
    INTERNAL_CHECK(new_param) << "Failed to get new param";
    new_params.push_back(new_param);
  }

  auto new_body = mutator.VisitStmt(normalized_func->body_);

  auto result_func = std::make_shared<Function>(
      normalized_func->name_, new_params, normalized_func->param_directions_, normalized_func->return_types_,
      new_body, normalized_func->span_, normalized_func->func_type_);

  // Step 4: Collect non-DDR MemRefs and create alloc statements
  NonDDRMemRefCollector collector;
  for (const auto& param : new_params) {
    collector.VisitExpr(param);
  }
  collector.VisitStmt(new_body);

  const auto& memrefs = collector.GetMemRefs();
  if (memrefs.empty()) return result_func;

  std::vector<StmtPtr> alloc_stmts;
  alloc_stmts.reserve(memrefs.size());
  for (const auto& memref : memrefs) {
    alloc_stmts.push_back(CreateAllocStatement(memref));
  }

  // Step 5: Insert alloc statements into the first OpStmts
  auto final_body = InsertAllocsIntoBody(new_body, alloc_stmts);

  return std::make_shared<Function>(result_func->name_, new_params, result_func->param_directions_,
                                    result_func->return_types_, final_body, result_func->span_,
                                    result_func->func_type_);
}

}  // namespace

// Factory function
namespace pass {
Pass InitMemRef() { return CreateFunctionPass(TransformInitMemRef, "InitMemRef", kInitMemRefProperties); }
}  // namespace pass

// ============================================================================
// HasMemRefs property verifier
// ============================================================================

namespace {

/**
 * @brief Checks all TileType variables have MemRef initialized.
 */
class HasMemRefsVerifier : public IRVisitor {
 public:
  explicit HasMemRefsVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    CheckVarMemRef(op->var_);
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckVarMemRef(const VarPtr& var) {
    if (!var || !var->GetType()) return;
    auto tile_type = std::dynamic_pointer_cast<const TileType>(var->GetType());
    if (tile_type && !tile_type->memref_.has_value()) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "HasMemRefs", 0,
                                "TileType variable '" + var->name_ + "' has no MemRef initialized",
                                var->span_);
    }
  }

  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class HasMemRefsPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "HasMemRefs"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      HasMemRefsVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateHasMemRefsPropertyVerifier() {
  return std::make_shared<HasMemRefsPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
