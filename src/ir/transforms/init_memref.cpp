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
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
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
#include "pypto/ir/transforms/utils/memref_utils.h"
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
    // No deduction logic — check if the Call's return type already carries a memory_space
    // (e.g., tpop ops whose return TileType has memory_space set by ExpandMixedKernel).
    if (auto tile_type = std::dynamic_pointer_cast<const TileType>(call->GetType())) {
      if (tile_type->memory_space_.has_value()) {
        return tile_type->memory_space_.value();
      }
    }
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

  [[nodiscard]] std::optional<MemorySpace> ResolveTileMemorySpace(const TypePtr& type, const VarPtr& var,
                                                                  bool default_to_ddr = false) const {
    if (var) {
      auto it = var_memory_spaces_.find(var);
      if (it != var_memory_spaces_.end()) {
        return it->second;
      }
    }

    if (auto tile_type = std::dynamic_pointer_cast<const TileType>(type)) {
      if (tile_type->memory_space_.has_value()) {
        return tile_type->memory_space_;
      }
    }

    if (default_to_ddr) {
      return MemorySpace::DDR;
    }
    return std::nullopt;
  }

  // Helper to calculate size and create MemRef
  std::optional<MemRefPtr> CreateMemRef(const ShapedTypePtr& type, const VarPtr& var) {
    const std::string var_name = var ? var->name_hint_ : "<anonymous>";
    uint64_t size_bytes = 0;
    if (As<TileType>(type)) {
      uint64_t num_elements = 1;
      for (size_t i = 0; i < type->shape_.size(); ++i) {
        auto const_dim = As<ConstInt>(type->shape_[i]);
        CHECK(const_dim) << "InitMemRef requires static shape for variable '" << var_name
                         << "', but shape element " << i
                         << " is dynamic. Fix the upstream op to keep TileType.shape static and put runtime "
                            "extent in TileView.valid_shape instead.";
        CHECK(const_dim->value_ > 0) << "InitMemRef requires positive shape for variable '" << var_name
                                     << "', but shape element " << i << " is " << const_dim->value_;
        num_elements *= static_cast<uint64_t>(const_dim->value_);
      }

      size_t bits = type->dtype_.GetBit();
      // Round up to bytes
      size_t bytes = (bits + 7) / 8;
      size_bytes = num_elements * bytes;
    } else {
      bool is_static = true;
      uint64_t num_elements = 1;
      for (const auto& dim : type->shape_) {
        if (auto const_dim = As<ConstInt>(dim)) {
          num_elements *= static_cast<uint64_t>(const_dim->value_);
        } else {
          is_static = false;
          break;
        }
      }
      if (is_static) {
        size_t bits = type->dtype_.GetBit();
        size_t bytes = (bits + 7) / 8;
        size_bytes = num_elements * bytes;
      }
    }

    // Query memory space: var_memory_spaces_ map > TileType's own memory_space > DDR default
    auto memory_space = ResolveTileMemorySpace(type, var, /*default_to_ddr=*/true);
    INTERNAL_CHECK(memory_space.has_value())
        << "Internal error: ResolveTileMemorySpace must return a value when default_to_ddr is enabled";
    MemorySpace space = *memory_space;

    // Addr is -1 (unallocated)
    auto addr = std::make_shared<ConstInt>(-1, DataType::INDEX, Span::unknown());

    // Generate unique ID for this MemRef
    uint64_t id = next_id_++;

    return std::make_shared<MemRef>(space, addr, size_bytes, id);
  }

  std::optional<MemorySpace> ExtractMemorySpaceFromType(const TypePtr& type) {
    auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(type);
    if (!shaped_type) {
      return std::nullopt;
    }
    return shaped_type->GetMemorySpace();
  }

  // Process IterArg variable (inherits MemRef from initValue)
  VarPtr ProcessIterArg(const VarPtr& old_var) {
    auto iter_arg = std::static_pointer_cast<const IterArg>(old_var);

    // Visit initValue to get its updated MemRef
    auto new_init = VisitExpr(iter_arg->initValue_);

    // Extract MemRef from initValue and create new type
    auto memref = GetTypeMemRef(new_init->GetType());
    auto old_var_expr = std::static_pointer_cast<const Expr>(old_var);
    auto source_memory_space = ExtractMemorySpaceFromType(new_init->GetType());
    TypePtr new_type = CloneTypeWithMemRefAndRemapExprs(
        old_var_expr->GetType(), memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
        source_memory_space);

    return std::make_shared<IterArg>(iter_arg->name_hint_, new_type, new_init, iter_arg->span_);
  }

  // Process normal Var variable (creates new MemRef based on usage)
  VarPtr ProcessNormalVar(const VarPtr& var) {
    auto var_expr = std::static_pointer_cast<const Expr>(var);
    TypePtr new_type = var_expr->GetType();

    // Process Type if it is ShapedType (TensorType or TileType)
    if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(var_expr->GetType())) {
      auto memref = CreateMemRef(shaped_type, var);
      new_type = CloneTypeWithMemRefAndRemapExprs(
          var_expr->GetType(), memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
          ResolveTileMemorySpace(var_expr->GetType(), var, /*default_to_ddr=*/true));
    }

    return std::make_shared<Var>(var->name_hint_, new_type, var->span_);
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
      LOG_DEBUG << "Processing AssignStmt for " << op->var_->name_hint_ << " with call to "
                << call->op_->name_;

      // Handle view operations: output should share MemRef with input tile
      if (IsViewOperation(call->op_->name_) && call->args_.size() > 0) {
        LOG_DEBUG << "Detected view operation: " << call->op_->name_;
        // Get the input tile (first argument) after mutation
        auto new_call = std::dynamic_pointer_cast<const Call>(new_value);
        if (new_call) {
          auto input_tile_arg = new_call->args_[0];

          // Extract MemRef from input tile
          auto shared_memref = GetTypeMemRef(input_tile_arg->GetType());

          // Create new variable with shared MemRef
          if (shared_memref.has_value()) {
            LOG_DEBUG << "Sharing MemRef from input tile to " << op->var_->name_hint_;
            auto source_memory_space = ExtractMemorySpaceFromType(input_tile_arg->GetType());
            TypePtr new_type = CloneTypeWithMemRefAndRemapExprs(
                op->var_->GetType(), shared_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
                source_memory_space);
            VarPtr new_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
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
          auto shared_memref = GetTypeMemRef(output_tensor_arg->GetType());

          // Create new variable with the shared MemRef
          if (shared_memref.has_value()) {
            TypePtr new_type = CloneTypeWithMemRefAndRemapExprs(
                op->var_->GetType(), shared_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
                ResolveTileMemorySpace(op->var_->GetType(), op->var_));

            VarPtr new_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
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
    // Manual traversal of ForStmt fields to ensure correct MemRef assignment:
    // - iter_args inherit MemRef from initValue (via ProcessIterArg)
    // - return_vars share MemRef with corresponding yield values

    // Step 1: Visit loop bounds and loop_var
    auto new_loop_var_expr = VisitExpr(op->loop_var_);
    auto new_loop_var = As<Var>(new_loop_var_expr);
    INTERNAL_CHECK(new_loop_var) << "Internal error: ForStmt loop_var is not a Var after mutation";
    auto new_start = VisitExpr(op->start_);
    auto new_stop = VisitExpr(op->stop_);
    auto new_step = VisitExpr(op->step_);

    // Step 2: Process iter_args (inherits MemRef from initValue via ProcessIterArg)
    std::vector<IterArgPtr> new_iter_args;
    new_iter_args.reserve(op->iter_args_.size());
    for (const auto& ia : op->iter_args_) {
      auto new_ia_expr = VisitExpr(ia);
      auto new_ia =
          std::dynamic_pointer_cast<const IterArg>(std::static_pointer_cast<const IRNode>(new_ia_expr));
      INTERNAL_CHECK(new_ia) << "Internal error: ForStmt iter_arg is not an IterArg after mutation";
      new_iter_args.push_back(new_ia);
    }

    // Register old->new IterArg mappings so body references are substituted
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (new_iter_args[i].get() != op->iter_args_[i].get()) {
        var_remap_[op->iter_args_[i].get()] = new_iter_args[i];
      }
    }

    // Step 3: Visit body
    auto new_body = VisitStmt(op->body_);

    // Clean up IterArg remappings
    for (const auto& old_iter_arg : op->iter_args_) {
      var_remap_.erase(old_iter_arg.get());
    }

    // Step 4: Visit return_vars
    std::vector<VarPtr> new_return_vars;
    new_return_vars.reserve(op->return_vars_.size());
    for (const auto& rv : op->return_vars_) {
      auto new_rv_expr = VisitExpr(rv);
      auto new_rv = As<Var>(new_rv_expr);
      INTERNAL_CHECK(new_rv) << "Internal error: ForStmt return_var is not a Var after mutation";
      new_return_vars.push_back(new_rv);
    }

    // Visit chunk_size if present
    std::optional<ExprPtr> new_chunk_size = op->chunk_size_;
    if (op->chunk_size_.has_value()) {
      new_chunk_size = VisitExpr(*op->chunk_size_);
    }

    auto new_for = std::make_shared<ForStmt>(new_loop_var, new_start, new_stop, new_step, new_iter_args,
                                             new_body, new_return_vars, op->span_, op->kind_, new_chunk_size,
                                             op->chunk_policy_, op->loop_origin_);

    // Step 5: Patch return_vars to share yield value MemRefs.
    // return_var receives the last yield value, so they must share the same MemRef.
    auto yield_stmt = FindYieldStmt(new_body);
    if (!yield_stmt || new_for->iter_args_.empty() || new_for->return_vars_.empty()) {
      return new_for;
    }

    bool changed = false;
    std::vector<VarPtr> patched_return_vars;
    patched_return_vars.reserve(new_for->return_vars_.size());

    for (size_t i = 0; i < new_for->return_vars_.size(); ++i) {
      if (i >= yield_stmt->value_.size()) {
        patched_return_vars.push_back(new_for->return_vars_[i]);
        continue;
      }

      auto rv_tile = As<TileType>(new_for->return_vars_[i]->GetType());
      auto yield_var = As<Var>(yield_stmt->value_[i]);
      auto yield_tile = yield_var ? As<TileType>(yield_var->GetType()) : nullptr;
      if (rv_tile && yield_tile && yield_tile->memref_.has_value()) {
        auto new_type = CloneTypeWithMemRefAndRemapExprs(
            new_for->return_vars_[i]->GetType(), yield_tile->memref_,
            [this](const ExprPtr& expr) { return VisitExpr(expr); }, yield_tile->GetMemorySpace());
        auto new_rv = std::make_shared<Var>(new_for->return_vars_[i]->name_hint_, new_type,
                                            new_for->return_vars_[i]->span_);
        var_map_[op->return_vars_[i]] = new_rv;
        patched_return_vars.push_back(new_rv);
        changed = true;
      } else {
        patched_return_vars.push_back(new_for->return_vars_[i]);
      }
    }

    if (!changed) return new_for;

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
  using MemRefAlloc = std::pair<MemRefPtr, MemorySpace>;

  [[nodiscard]] const std::vector<MemRefAlloc>& GetMemRefs() const { return memrefs_; }

  void VisitVarLike_(const VarPtr& op) override {
    if (auto tile_type = GetTileTypeWithMemRef(op->GetType())) {
      AddMemRefIfUnique(tile_type);
    }
  }

 private:
  std::vector<MemRefAlloc> memrefs_;
  std::map<const MemRef*, MemorySpace> seen_ptrs_;

  void AddMemRefIfUnique(const std::shared_ptr<const TileType>& tile_type) {
    auto memory_space = tile_type->GetMemorySpace();
    CHECK(memory_space.has_value())
        << "TileType with MemRef must have memory_space before emitting tile.alloc";
    CHECK(tile_type->memref_.has_value()) << "TileType must carry MemRef before emitting tile.alloc";
    const MemorySpace canonical_space = memory_space.value();
    if (canonical_space == MemorySpace::DDR) return;

    const auto& memref = tile_type->memref_.value();
    if (TryRegisterUniqueMemRef(memref, canonical_space, seen_ptrs_)) {
      memrefs_.emplace_back(memref, canonical_space);
    }
  }
};

// Create tile.alloc AssignStmt for a MemRef with addr=-1 (unallocated)
StmtPtr CreateAllocStatement(const MemRefPtr& memref, MemorySpace memory_space) {
  auto alloc_op = std::make_shared<Op>("tile.alloc");

  auto memspace_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(memory_space), DataType::INDEX, Span::unknown());
  ExprPtr addr_expr = memref->addr_;
  auto size_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(memref->size_), DataType::INDEX, Span::unknown());
  auto id_expr =
      std::make_shared<ConstInt>(static_cast<int64_t>(memref->id_), DataType::INDEX, Span::unknown());

  std::vector<ExprPtr> alloc_args = {memspace_expr, addr_expr, size_expr, id_expr};
  auto alloc_call = std::make_shared<Call>(alloc_op, alloc_args, GetMemRefType(), Span::unknown());

  return std::make_shared<AssignStmt>(memref, alloc_call, Span::unknown());
}

// Insert alloc statements at the beginning of a function body.
StmtPtr InsertAllocsIntoBody(const StmtPtr& body, const std::vector<StmtPtr>& alloc_stmts) {
  if (alloc_stmts.empty()) return body;

  std::vector<StmtPtr> new_seq_stmts;
  new_seq_stmts.insert(new_seq_stmts.end(), alloc_stmts.begin(), alloc_stmts.end());

  const Span& span = body ? body->span_ : alloc_stmts.front()->span_;
  if (body) {
    if (auto seq = As<SeqStmts>(body)) {
      new_seq_stmts.insert(new_seq_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
    } else {
      new_seq_stmts.push_back(body);
    }
  }

  return SeqStmts::Flatten(std::move(new_seq_stmts), span);
}

/**
 * @brief Initialize MemRef for all variables in a function
 *
 * This transformation:
 * 1. Normalizes statement structure (ensures SeqStmts)
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
  // Step 1: Normalize statement structure to ensure SeqStmts
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
      new_body, normalized_func->span_, normalized_func->func_type_, normalized_func->level_,
      normalized_func->role_);

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
  for (const auto& [memref, memory_space] : memrefs) {
    alloc_stmts.push_back(CreateAllocStatement(memref, memory_space));
  }

  // Step 5: Insert alloc statements at the beginning of the function body
  auto final_body = InsertAllocsIntoBody(new_body, alloc_stmts);

  return std::make_shared<Function>(result_func->name_, new_params, result_func->param_directions_,
                                    result_func->return_types_, final_body, result_func->span_,
                                    result_func->func_type_, result_func->level_, result_func->role_);
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
                                "TileType variable '" + var->name_hint_ + "' has no MemRef initialized",
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
