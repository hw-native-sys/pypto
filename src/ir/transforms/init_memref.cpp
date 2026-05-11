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
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
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
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Check if operation is a view operation (zero-copy metadata transform).
// A view op is one registered with set_output_memory_inherit_input() — its
// output reuses the input's MemRef view. Delegates to the shared registry
// predicate so InferTileMemorySpace and InitMemRef agree on the set.
bool IsViewOperation(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return false;
  return registry.GetEntry(op_name).OutputMemoryInheritsInput();
}

// Check if an operation's output should reuse the MemRef of a specific input argument.
// Returns the input arg index whose MemRef to share, or nullopt.
std::optional<size_t> GetOutputReusesInputArg(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return std::nullopt;
  return registry.GetEntry(op_name).GetOutputReusesInputArg();
}

// Mutator to initialize MemRef for variables
class InitMemRefMutator : public IRMutator {
 public:
  InitMemRefMutator() = default;

  // Resolve memory space from TileType::memory_space_ field (set by InferTileMemorySpace),
  // falling back to DDR when default_to_ddr is true.
  [[nodiscard]] static std::optional<MemorySpace> ResolveTileMemorySpace(const TypePtr& type,
                                                                         bool default_to_ddr = false) {
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

  // Calculate allocation size and create MemRef with the given memory space.
  std::optional<MemRefPtr> CreateMemRef(const ShapedTypePtr& type, const VarPtr& var,
                                        std::optional<MemorySpace> memory_space) {
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

      size_bytes = num_elements * ((type->dtype_.GetBit() + 7) / 8);
    } else {
      uint64_t num_elements = 1;
      bool is_static = true;
      for (const auto& dim : type->shape_) {
        if (auto const_dim = As<ConstInt>(dim)) {
          num_elements *= static_cast<uint64_t>(const_dim->value_);
        } else {
          is_static = false;
          break;
        }
      }
      if (is_static) {
        size_bytes = num_elements * ((type->dtype_.GetBit() + 7) / 8);
      }
    }

    const Span& err_span = var ? var->span_ : Span::unknown();
    INTERNAL_CHECK_SPAN(memory_space.has_value(), err_span)
        << "Internal error: memory_space must be resolved before CreateMemRef";

    auto base =
        std::make_shared<Var>(BuildBasePtrName(*memory_space, next_id_++), GetPtrType(), Span::unknown());
    return std::make_shared<MemRef>(base, static_cast<int64_t>(0), size_bytes);
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

    // ArrayType lives on the on-core scalar register file / C stack and never
    // needs a runtime MemRef — codegen lowers it to a stack array directly.
    // Leave the var untouched so the original ArrayType (no memref_) propagates.
    if (std::dynamic_pointer_cast<const ArrayType>(var_expr->GetType())) {
      return var;
    }

    if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(var_expr->GetType())) {
      // Resolve memory space once, pass to both CreateMemRef and CloneType
      auto memory_space = ResolveTileMemorySpace(var_expr->GetType(), /*default_to_ddr=*/true);
      auto memref = CreateMemRef(shaped_type, var, memory_space);
      new_type = CloneTypeWithMemRefAndRemapExprs(
          var_expr->GetType(), memref, [this](const ExprPtr& expr) { return VisitExpr(expr); }, memory_space);
    } else {
      // Non-shaped types (e.g. ScalarType for dynamic-shape dimensions like M, N)
      // don't need MemRef initialization — return the original Var to preserve
      // pointer identity across all type annotations that reference it.
      return var;
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

  /**
   * @brief Create a view MemRef from a source expression for the LHS variable of an assignment.
   *
   * Creates a NEW MemRef with the same base_ Ptr as the source's MemRef,
   * accumulated byte offset, and computed size from the output shape.
   * Returns nullptr if the source has no MemRef.
   */
  StmtPtr ShareMemRefFrom(const ExprPtr& source, const AssignStmtPtr& op, const ExprPtr& new_value) {
    auto parent_memref_opt = GetTypeMemRef(source->GetType());
    if (!parent_memref_opt.has_value()) return nullptr;
    const auto& parent_memref = *parent_memref_opt;

    // Compute additional byte offset from the view op (if applicable)
    ExprPtr additional_offset = MakeZeroByteOffset();
    if (auto call = As<Call>(new_value)) {
      additional_offset = ComputeViewByteOffset(call, source->GetType());
    }

    // Accumulate: total_offset = parent.byte_offset + additional_offset
    ExprPtr total_offset = AddByteOffsets(parent_memref->byte_offset_, additional_offset);

    // Compute view size from the output type
    uint64_t view_size = parent_memref->size_;  // default: same size as parent
    if (auto out_shaped = std::dynamic_pointer_cast<const ShapedType>(op->var_->GetType())) {
      uint64_t num_elements = 1;
      bool is_static = true;
      for (const auto& dim : out_shaped->shape_) {
        if (auto const_dim = As<ConstInt>(dim)) {
          num_elements *= static_cast<uint64_t>(const_dim->value_);
        } else {
          is_static = false;
          break;
        }
      }
      if (is_static) {
        view_size = num_elements * ((out_shaped->dtype_.GetBit() + 7) / 8);
      }
    }

    // For pure aliases (no offset change, same size), share the SAME MemRef shared_ptr
    // to preserve base_ Ptr identity. MemoryReuse builds sharing groups by base_ Ptr —
    // variables sharing the same base_ are merged into one lifetime group.
    bool is_pure_alias = (view_size == parent_memref->size_);
    if (is_pure_alias) {
      if (auto const_offset = As<ConstInt>(additional_offset)) {
        is_pure_alias = (const_offset->value_ == 0);
      } else {
        is_pure_alias = false;
      }
    }
    MemRefPtr view_memref = is_pure_alias
                                ? parent_memref
                                : std::make_shared<MemRef>(parent_memref->base_, total_offset, view_size);

    auto source_ms = ExtractMemorySpaceFromType(source->GetType());
    std::optional<MemRefPtr> view_opt = view_memref;
    TypePtr new_type = CloneTypeWithMemRefAndRemapExprs(
        op->var_->GetType(), view_opt, [this](const ExprPtr& e) { return VisitExpr(e); }, source_ms);
    VarPtr new_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    var_map_[op->var_] = new_var;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

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
          auto result = ShareMemRefFrom(new_call->args_[0], op, new_value);
          if (result) {
            LOG_DEBUG << "Sharing MemRef from input tile to " << op->var_->name_hint_;
            return result;
          }
          LOG_DEBUG << "Input tile has no MemRef yet";
        }
      }

      // Handle ops whose output reuses a specific input arg's MemRef (registry-based)
      auto reuse_arg_idx = GetOutputReusesInputArg(call->op_->name_);
      if (reuse_arg_idx.has_value()) {
        auto new_call = std::dynamic_pointer_cast<const Call>(new_value);
        if (new_call && *reuse_arg_idx < new_call->args_.size()) {
          auto result = ShareMemRefFrom(new_call->args_[*reuse_arg_idx], op, new_value);
          if (result) {
            LOG_DEBUG << "Reusing MemRef from input arg " << *reuse_arg_idx << " for "
                      << op->var_->name_hint_;
            return result;
          }
        }
      }
    }

    // Tile alias: a = b where b is a tile Var — share b's MemRef.
    // Without this, the alias gets a fresh MemRef that is never written to,
    // breaking IfStmt return_vars that inherit the alias's empty MemRef.
    if (auto value_var = As<Var>(new_value)) {
      if (As<TileType>(op->var_->GetType())) {
        auto result = ShareMemRefFrom(value_var, op, new_value);
        if (result) return result;
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
    INTERNAL_CHECK_SPAN(new_loop_var, op->span_)
        << "Internal error: ForStmt loop_var is not a Var after mutation";
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
      INTERNAL_CHECK_SPAN(new_ia, op->span_)
          << "Internal error: ForStmt iter_arg is not an IterArg after mutation";
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
      INTERNAL_CHECK_SPAN(new_rv, op->span_)
          << "Internal error: ForStmt return_var is not a Var after mutation";
      new_return_vars.push_back(new_rv);
    }

    // Visit chunk_size if present
    std::optional<ChunkConfig> new_chunk_config = op->chunk_config_;
    if (op->chunk_config_.has_value()) {
      new_chunk_config = ChunkConfig{VisitExpr(op->chunk_config_->size), op->chunk_config_->policy};
    }

    auto new_for = MutableCopy(op);
    new_for->loop_var_ = new_loop_var;
    new_for->start_ = new_start;
    new_for->stop_ = new_stop;
    new_for->step_ = new_step;
    new_for->iter_args_ = new_iter_args;
    new_for->body_ = new_body;
    new_for->return_vars_ = new_return_vars;
    new_for->chunk_config_ = new_chunk_config;

    // Patch return_vars so each shares its iter_arg's MemRef (inherited from initValue).
    // This establishes the invariant that initValue/iter_arg/return_var all share the
    // same MemRef buffer — the loop accumulator lives in one place for the whole loop.
    // Any yield-vs-buffer mismatch is the concern of downstream passes.
    if (new_for->iter_args_.empty() || new_for->return_vars_.empty()) {
      return new_for;
    }

    auto get_iter_arg_var = [&](size_t i) -> VarPtr {
      if (i >= new_for->iter_args_.size()) return nullptr;
      return std::static_pointer_cast<const Var>(new_for->iter_args_[i]);
    };
    auto [patched, changed] =
        PatchReturnVarsFromYield(new_for->return_vars_, op->return_vars_, get_iter_arg_var);

    if (!changed) return new_for;

    new_for->return_vars_ = std::move(patched);
    return new_for;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    auto new_if = As<IfStmt>(result);
    if (!new_if || new_if->return_vars_.empty()) return result;

    auto then_yield = FindYieldStmt(new_if->then_body_);
    auto else_yield = new_if->else_body_.has_value() ? FindYieldStmt(new_if->else_body_.value()) : nullptr;
    if (!then_yield && !else_yield) return result;

    auto get_yield_var = [&](size_t i) -> VarPtr {
      VarPtr var = nullptr;
      if (then_yield && i < then_yield->value_.size()) var = As<Var>(then_yield->value_[i]);
      if (!var && else_yield && i < else_yield->value_.size()) var = As<Var>(else_yield->value_[i]);
      return var;
    };
    auto [patched, changed] = PatchReturnVarsFromYield(new_if->return_vars_, op->return_vars_, get_yield_var);

    if (!changed) return result;

    auto patched_if = MutableCopy(new_if);
    patched_if->return_vars_ = std::move(patched);
    return patched_if;
  }

 private:
  // Shared logic for ForStmt/IfStmt: patch each return_var to share its yield value's MemRef.
  // get_yield_var(i) resolves the yield variable for the i-th return_var.
  using YieldVarResolver = std::function<VarPtr(size_t)>;
  std::pair<std::vector<VarPtr>, bool> PatchReturnVarsFromYield(const std::vector<VarPtr>& new_return_vars,
                                                                const std::vector<VarPtr>& old_return_vars,
                                                                const YieldVarResolver& get_yield_var) {
    bool changed = false;
    std::vector<VarPtr> patched;
    patched.reserve(new_return_vars.size());

    for (size_t i = 0; i < new_return_vars.size(); ++i) {
      auto yield_var = get_yield_var(i);
      auto yield_tile = yield_var ? GetTileTypeWithMemRef(yield_var->GetType()) : nullptr;
      if (As<TileType>(new_return_vars[i]->GetType()) && yield_tile) {
        auto new_type = CloneTypeWithMemRef(new_return_vars[i]->GetType(), yield_tile->memref_,
                                            yield_tile->GetMemorySpace());
        auto new_rv =
            std::make_shared<Var>(new_return_vars[i]->name_hint_, new_type, new_return_vars[i]->span_);
        var_map_[old_return_vars[i]] = new_rv;
        patched.push_back(new_rv);
        changed = true;
      } else {
        patched.push_back(new_return_vars[i]);
      }
    }
    return {std::move(patched), changed};
  }

  std::map<VarPtr, VarPtr> var_map_;
  uint64_t next_id_ = 0;
};

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
 * Memory space is read from TileType::memory_space_ (set by InferTileMemorySpace).
 * Variables without memory_space default to DDR.
 */
FunctionPtr TransformInitMemRef(const FunctionPtr& func) {
  // Step 1: Normalize statement structure to ensure SeqStmts
  auto normalized_func = NormalizeStmtStructure(func);

  // Step 2: Mutate variables to initialize their MemRef
  InitMemRefMutator mutator;

  std::vector<VarPtr> new_params;
  new_params.reserve(normalized_func->params_.size());
  for (const auto& var : normalized_func->params_) {
    auto new_param = mutator.GetNewVar(var);
    INTERNAL_CHECK_SPAN(new_param, var->span_) << "Failed to get new param";
    new_params.push_back(new_param);
  }

  auto new_body = mutator.VisitStmt(normalized_func->body_);

  auto result_func = MutableCopy(normalized_func);
  result_func->params_ = new_params;
  result_func->body_ = new_body;

  // Step 3: Collect ALL MemRefs (DDR gets tensor.alloc, on-chip gets tile.alloc)
  memref_collectors::MemRefWithSpaceCollector collector(/*skip_ddr=*/false);
  for (const auto& param : new_params) {
    collector.VisitExpr(param);
  }
  collector.VisitStmt(new_body);

  const auto& memrefs = collector.memrefs;
  if (memrefs.empty()) return result_func;

  // Deduplicate by base_ Ptr — one alloc per unique base
  std::set<const Var*> seen_bases;
  std::vector<StmtPtr> alloc_stmts;
  alloc_stmts.reserve(memrefs.size());
  for (const auto& [memref, memory_space] : memrefs) {
    if (seen_bases.insert(memref->base_.get()).second) {
      alloc_stmts.push_back(CreateAllocStatement(memref, memory_space));
    }
  }

  // Step 4: Insert alloc statements at the beginning of the function body
  auto final_body = InsertAllocsIntoBody(new_body, alloc_stmts);

  result_func->body_ = final_body;
  return result_func;
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
