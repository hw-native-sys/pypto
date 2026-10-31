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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/code_emitter.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "src/codegen/orchestration/orchestration_stmt_codegen.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void OrchestrationStmtCodegen::GenerateTensorOpCode(const CallPtr& call, const std::string& result_var,
                                                    const VarPtr& assign_var) {
  for (const auto& arg : call->args_) ValidateValueUse(arg, call->span_);
  const std::string& op_name = call->op_->name_;

  auto& registry = OrchestrationOpRegistry::GetInstance();
  auto codegen_func = registry.Get(op_name);
  INTERNAL_CHECK_SPAN(codegen_func.has_value(), call->span_)
      << "Misplaced tensor op '" << op_name << "' in Orchestration function (should be inside InCore block)";

  if (IsOp(call, "tensor.create") && assign_var &&
      (declared_var_ptrs_.count(assign_var.get()) || param_name_set_.count(GetVarName(assign_var)))) {
    return;
  }

  std::string emit_var = result_var;
  if (IsOp(call, "tensor.create") && assign_var) {
    declared_var_ptrs_.insert(assign_var.get());
    emit_var = ReserveVarEmitName(assign_var.get());
  }

  if (AsTensorTypeLike(call->GetType()) && !IsOp(call, "tensor.create")) {
    for (const auto& arg : call->args_) {
      if (AsTensorTypeLike(arg->GetType())) {
        RecordTensorSource(emit_var, GetExternalTensorName(TryGetVarName(arg)));
      }
    }
  }
  current_result_var_ = emit_var;

  std::string gen_code = (*codegen_func)(call, *this);

  std::istringstream iss(gen_code);
  std::string line;
  while (std::getline(iss, line)) {
    if (!line.empty()) {
      EmitIndentedLine(line);
    }
  }

  if (IsOp(call, "tensor.create")) {
    EmitAllocBatch({emit_var});
  }
}

bool OrchestrationStmtCodegen::IsInjectedGMPipeCreateVar(const VarPtr& var) {
  return alloc_batching::IsInjectedGMPipeCreateVar(var);
}

int64_t OrchestrationStmtCodegen::GetGMPipeWorkspaceElements(const FunctionPtr& root_func) {
  INTERNAL_CHECK(root_func != nullptr) << "Internal error: GM pipe workspace root function must not be null";
  auto it = gm_pipe_workspace_elements_by_callee_.find(root_func->name_);
  if (it == gm_pipe_workspace_elements_by_callee_.end()) {
    const int64_t elements = ComputeGMPipeWorkspaceElements(program_, root_func);
    it = gm_pipe_workspace_elements_by_callee_.emplace(root_func->name_, elements).first;
  }
  return it->second;
}

std::optional<OrchestrationStmtCodegen::GMPipeCreateUse> OrchestrationStmtCodegen::ResolveGMPipeCreateUse(
    const std::vector<StmtPtr>& stmts, size_t create_stmt_idx, const VarPtr& create_var) const {
  if (!create_var) return std::nullopt;

  for (size_t i = create_stmt_idx + 1; i < stmts.size(); ++i) {
    auto assign = As<AssignStmt>(stmts[i]);
    if (assign && assign->var_ && assign->var_.get() == create_var.get()) {
      break;
    }

    // The consumer may be a Submit (e.g. pl.spmd_submit of a mixed kernel
    // that receives the injected gm_pipe_buffer create); view it as a Call.
    CallPtr call;
    if (assign) {
      call = AsCallOrSubmitView(assign->value_);
    } else if (auto eval = As<EvalStmt>(stmts[i])) {
      call = AsCallOrSubmitView(eval->expr_);
    }
    if (!call) continue;

    bool uses_create_var = false;
    for (const auto& arg : call->args_) {
      auto arg_var = AsVarLike(arg);
      if (arg_var && arg_var.get() == create_var.get()) {
        uses_create_var = true;
        break;
      }
    }
    if (!uses_create_var) continue;

    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return std::nullopt;
    auto callee_func = program_->GetFunction(gv->name_);
    if (!callee_func) return std::nullopt;

    // Resolve core_num from the same source as the launch spec: pl.spmd_submit
    // carries it on the dispatch call's attrs, while scope-based pl.spmd /
    // Group wrappers carry it on the callee function's attrs. Sizing the
    // GM-pipe workspace from callee attrs alone would under-allocate for a
    // direct ``pl.spmd_submit(self.aic_or_aiv_kernel, ..., core_num=N)``
    // (N blocks launched, 1-block workspace).
    auto core_num_expr = EffectiveLaunchSpec(call, callee_func).first;
    std::string rendered_core_num;
    if (core_num_expr) {
      rendered_core_num = RenderLaunchCoreNum(core_num_expr);
    }
    return GMPipeCreateUse{callee_func, rendered_core_num, core_num_expr};
  }
  return std::nullopt;
}

bool OrchestrationStmtCodegen::ExprRefsAnyOf(const ExprPtr& expr,
                                             const std::unordered_set<const Var*>& vars) {
  return alloc_batching::ExprRefsAnyOf(expr, vars);
}

bool OrchestrationStmtCodegen::ShapeDependsOnLocalVars(
    const CallPtr& call, const std::unordered_set<const Var*>& locally_defined) const {
  auto result_type = AsTensorTypeLike(call->GetType());
  if (!result_type) {
    return false;
  }
  for (const auto& dim : result_type->shape_) {
    if (ExprRefsAnyOf(dim, locally_defined)) {
      return true;
    }
  }
  return false;
}

void OrchestrationStmtCodegen::EmitAllocBatch(const std::vector<std::string>& emit_names) {
  std::string alloc_var = "alloc_" + std::to_string(alloc_counter_++);
  std::ostringstream alloc;
  alloc << "TaskOutputTensors " << alloc_var << " = alloc_tensors(";
  for (size_t i = 0; i < emit_names.size(); i++) {
    if (i > 0) {
      alloc << ", ";
    }
    alloc << emit_names[i] << "_ci";
  }
  alloc << ");";
  EmitIndentedLine(alloc.str());

  for (size_t i = 0; i < emit_names.size(); i++) {
    RecordTensorAllocation(emit_names[i]);
    EmitIndentedLine("const Tensor& " + emit_names[i] + " = " + alloc_var + ".get_ref(" + std::to_string(i) +
                     ");");
  }
}

void OrchestrationStmtCodegen::EmitBatchedAllocTensors(const std::vector<StmtPtr>& stmts) {
  struct PendingCreate {
    std::string emit_name;
    CallPtr call;
  };
  std::vector<PendingCreate> creates;

  std::unordered_set<const Var*> locally_defined;

  for (size_t stmt_idx = 0; stmt_idx < stmts.size(); ++stmt_idx) {
    const auto& stmt = stmts[stmt_idx];
    auto assign = As<AssignStmt>(stmt);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    if (!call || !IsOp(call, "tensor.create")) {
      // Track every name this level defines, not just AssignStmt targets: an
      // IfStmt / ForStmt / WhileStmt return_var is a phi whose C++ declaration
      // is emitted where the statement sits, so a create sized from one is
      // hoist-ineligible exactly like a create sized from a plain local.
      for (const auto* def : var_collectors::CollectStmtDefinedVars(stmt)) {
        locally_defined.insert(def);
      }
      continue;
    }
    if (declared_var_ptrs_.count(assign->var_.get()) || param_name_set_.count(GetVarName(assign->var_))) {
      continue;
    }
    if (ShapeDependsOnLocalVars(call, locally_defined)) {
      locally_defined.insert(assign->var_.get());
      continue;
    }

    declared_var_ptrs_.insert(assign->var_.get());
    std::string emit_var = ReserveVarEmitName(assign->var_.get());
    if (IsInjectedGMPipeCreateVar(assign->var_)) {
      auto create_use = ResolveGMPipeCreateUse(stmts, stmt_idx, assign->var_);
      INTERNAL_CHECK(create_use.has_value())
          << "Internal error: injected gm_pipe_buffer tensor.create is not passed to a callee";
      int64_t workspace_elems = GetGMPipeWorkspaceElements(create_use->callee);
      INTERNAL_CHECK(workspace_elems > 0)
          << "Internal error: injected gm_pipe_buffer tensor.create found without initialize_pipe ops";
      std::string size_expr = std::to_string(workspace_elems);
      const std::string& core_num_expr = create_use->core_num_expr;
      if (!core_num_expr.empty()) {
        size_expr = "static_cast<uint32_t>((" + size_expr + ") * (" + core_num_expr + "))";
      }
      tensor_create_size_expr_by_emit_name_[emit_var] = size_expr;
      // Keep the create in body order when core_num is computed from a body-local.
      if (ExprRefsAnyOf(create_use->core_num_node, locally_defined)) {
        declared_var_ptrs_.erase(assign->var_.get());
        locally_defined.insert(assign->var_.get());
        continue;
      }
    }
    creates.push_back({emit_var, call});
    batched_create_stmts_.insert(stmt.get());
  }
  if (creates.empty()) {
    return;
  }

  auto& registry = OrchestrationOpRegistry::GetInstance();
  auto handler = registry.Get("tensor.create");
  if (!handler.has_value()) {
    INTERNAL_CHECK_SPAN(false, creates[0].call->span_)
        << "Internal error: tensor.create handler not registered";
  }
  const auto& tensor_create_handler = *handler;

  // Manual-scope allocation hoisting (issue #1697). When this SeqStmts is the
  // direct body of a ``pl.manual_scope``, the alloc batch is declared in the
  // enclosing scope (routed into ``scope_hoist_sink_``) rather than at the
  // deep block indent — so a buffer created inside the scope but read by a
  // task placed AFTER it stays in C++ scope. A manual_scope is a scheduling
  // region, not a storage scope: an ``alloc_tensors`` has no scheduling
  // dependency, so emitting it one level out is semantically inert. The batch
  // is enclosing-scope-valid by construction — ShapeDependsOnLocalVars already
  // excluded any create whose shape references a scope-local value (those fall
  // to the per-op path and stay put). The ``+ 4`` guard restricts this to the
  // scope's own body, so a create nested in a for/if *within* the manual scope
  // is left in place. Erasing the hoisted names from ``scope_local_names_``
  // then lets a kernel output that aliases such a buffer remap to it
  // (EmitTensorAlias / IsEnclosingScopeValid).
  const bool hoist_batch =
      scope_hoist_allocations_ && scope_hoist_sink_ != nullptr && IsAtRuntimeScopeBodyIndent();
  CodeEmitter batch_emitter;
  CodeEmitter* saved_active = active_emitter_;
  const int saved_indent_level = Active().GetIndentLevel();
  if (hoist_batch) {
    active_emitter_ = &batch_emitter;
    batch_emitter.SetIndentLevel(saved_indent_level - 1);
  }

  for (size_t batch_start = 0; batch_start < creates.size(); batch_start += kMaxAllocTensorsArgs) {
    size_t batch_end = std::min(batch_start + kMaxAllocTensorsArgs, creates.size());

    for (size_t i = batch_start; i < batch_end; i++) {
      current_result_var_ = creates[i].emit_name;
      std::string gen_code = tensor_create_handler(creates[i].call, *this);
      std::istringstream iss(gen_code);
      std::string line;
      while (std::getline(iss, line)) {
        if (!line.empty()) {
          EmitIndentedLine(line);
        }
      }
    }

    std::vector<std::string> batch_names;
    for (size_t i = batch_start; i < batch_end; i++) {
      batch_names.push_back(creates[i].emit_name);
    }
    EmitAllocBatch(batch_names);
  }

  if (hoist_batch) {
    active_emitter_ = saved_active;
    scope_hoist_sink_->push_back(batch_emitter.GetCode());
    // The hoisted buffers now live in the enclosing scope: drop them from this
    // scope's local set so an output that aliases one remaps to it. If the
    // enclosing scope is itself a manual scope, the buffer's decl landed in
    // *its* body, so it is scope-local one level up — record it there, or a
    // reader after the enclosing scope would wrongly treat it as enclosing-
    // valid (nested manual scopes).
    for (const auto& c : creates) {
      scope_local_names_->erase(c.emit_name);
      if (enclosing_scope_local_names_ != nullptr) {
        enclosing_scope_local_names_->insert(c.emit_name);
      }
    }
  }
}

std::vector<size_t> OrchestrationStmtCodegen::CollectOutIndices(const FunctionPtr& callee) {
  const auto& dirs = callee->param_directions_;
  std::vector<size_t> out_indices;
  for (size_t i = 0; i < dirs.size(); ++i) {
    if (dirs[i] == ParamDirection::Out || dirs[i] == ParamDirection::InOut) {
      out_indices.push_back(i);
    }
  }
  return out_indices;
}

// Return-position -> callee param-index map, read straight off the callee's
// ReturnStmt (pointer identity) and memoized per callee. Pipeline IR
// satisfies IRProperty::ReturnParamsExplicit, which is what makes this a
// lookup rather than an analysis. Empty when the callee has no ReturnStmt —
// callers then fall back to the direction-based tail heuristic.
const std::vector<std::optional<size_t>>& OrchestrationStmtCodegen::GetReturnedParamIndices(
    const FunctionPtr& callee) {
  auto it = returned_param_indices_cache_.find(callee.get());
  if (it != returned_param_indices_cache_.end()) return it->second;
  auto inserted = returned_param_indices_cache_.emplace(
      callee.get(), ir::return_lineage::ExplicitReturnedParamIndices(callee));
  return inserted.first->second;
}

// Decide whether the precise return->param map is trustworthy for this call.
// It must be non-empty AND every return position whose declared type is a
// tensor must have resolved to a param. If a tensor writeback failed to
// trace, we keep the legacy heuristic to avoid regressing shapes the tracer
// does not model (a missed tensor alias would emit an undeclared symbol).
bool OrchestrationStmtCodegen::IsReturnedParamMapPrecise(const std::vector<std::optional<size_t>>& ret_map,
                                                         const CallPtr& call) {
  if (ret_map.empty()) return false;
  auto tuple_ty = As<TupleType>(call->GetType());
  if (!tuple_ty) return false;
  // Kernel-result positions to validate. For a submit call the trailing tuple
  // element is the producer TASK_ID, not a kernel result, so it has no
  // ret_map entry — exclude it from the expected count.
  size_t expected = tuple_ty->types_.size();
  if (IsSubmitCall(call) && expected > 0) --expected;
  // A short map leaves trailing tensor outputs unchecked: treat as imprecise
  // so the caller falls back to the heuristic instead of silently skipping an
  // unmapped tensor alias.
  if (ret_map.size() < expected) return false;
  for (size_t j = 0; j < expected; ++j) {
    if (AsTensorTypeLike(tuple_ty->types_[j]) && !ret_map[j].has_value()) {
      return false;
    }
  }
  return true;
}

void OrchestrationStmtCodegen::EmitTensorAlias(const Var* result_var, const std::string& alias_name,
                                               const CallPtr& call, size_t arg_idx) {
  std::string out_arg = TryGetVarName(call->args_[arg_idx]);
  if (out_arg.empty() || alias_name == out_arg) {
    return;
  }
  std::string out_name = GetExternalTensorName(out_arg);
  const bool mutable_alias = IsMutableTensorNameInCurrentScope(alias_name);

  // A caller-allocated kernel/submit output aliases an arg it writes in place
  // — it is the *same physical tensor* as that arg. Rather than mint a
  // ``const Tensor& <result> = <source>;`` rename, we remap the result Var's
  // emit name to the source, so every reference resolves directly to the
  // source name. This is the strategy ``tensor.assemble`` already uses
  // (HandleTensorAssembleAssign); applying it uniformly drops the redundant
  // alias decls and, for ``pl.manual_scope``, fixes the dead-alias bug at its
  // root: a task placed AFTER the block references the enclosing-scope source
  // name, with no manual-scope-internal identifier to fall out of C++ scope
  // (issue #1697).
  //
  // AUTO scopes retain their existing output remapping. MANUAL scopes
  // additionally require an enclosing-scope-valid source so a reader placed
  // after the block resolves it (IsEnclosingScopeValid).
  //
  // A phi reassignment (``mutable_alias``) is excluded: it rebinds an lvalue
  // the enclosing if/loop owns, so remapping it would erase the merge point and
  // break loop carries. It keeps its ``<name> = <src>;`` reassignment. A
  // manual-scope-local source that could not be hoisted also keeps the decl
  // path (remapping to it would not help an after-scope reader).
  const bool source_in_scope = in_manual_scope_depth_ == 0 || IsEnclosingScopeValid(out_arg);
  if (result_var != nullptr && !mutable_alias && source_in_scope) {
    emit_name_map_[result_var] = out_name;
    return;
  }

  RecordTensorSource(alias_name, out_name);
  if (mutable_alias) {
    EmitIndentedLine(alias_name + " = " + out_name + ";");
  } else {
    EmitIndentedLine("const Tensor& " + alias_name + " = " + out_name + ";");
  }
}

/// Validate an operand at a code-emission site, never while reserving or
/// querying a definition's name. All Tensor use paths share this check;
/// unrelated value types retain their existing validation.
void OrchestrationStmtCodegen::ValidateValueUse(const ExprPtr& value, const Span& use_span,
                                                bool read_storage) const {
  auto var = AsVarLike(value);
  if (!var || !AsTensorTypeLike(var->GetType())) return;
  auto it = emit_name_map_.find(var.get());
  if (it == emit_name_map_.end()) return;
  CHECK_SPAN(closed_auto_scope_names_.count(it->second) == 0, use_span)
      << "Tensor '" << var->name_hint_ << "' is used after its AUTO runtime scope has closed. "
      << "Allocate the tensor in an enclosing scope, or move this use inside its pl.scope().";
  const auto storage = tensor_values_.find(GetExternalTensorName(it->second));
  if (read_storage && storage != tensor_values_.end()) {
    if (auto_scope_carry_values_.count(storage->second)) {
      ValidateTensorStorage(storage->second, var, use_span);
    }
    tensor_read_values_.insert(storage->second);
  }
}

void OrchestrationStmtCodegen::ValidateTensorStorage(size_t value, const VarPtr& var,
                                                     const Span& use_span) const {
  CHECK_SPAN(invalid_tensor_storage_.count(value) == 0, use_span)
      << "Tensor '" << var->name_hint_ << "' may reference an allocation from a closed runtime scope. "
      << "Allocate its backing buffer in an enclosing scope, or consume it inside its pl.scope().";
}

/// True when ``name`` (a tensor emit name) is valid in the C++ scope that
/// encloses the active runtime scope — i.e. a runtime-scope output may safely
/// remap to it / a hoisted decl may reference it. A name is scope-local iff it
/// was first reserved *inside* the block and not subsequently hoisted out of
/// it (EmitBatchedAllocTensors erases hoisted ``alloc_tensors`` names from
/// ``scope_local_names_``); anything else — a function param, a parent-scope
/// tensor, or a hoisted in-scope buffer — is enclosing-scope-valid.
bool OrchestrationStmtCodegen::IsEnclosingScopeValid(const std::string& name) const {
  return scope_local_names_ != nullptr && scope_local_names_->count(name) == 0;
}

/// True when the current emit indent is exactly the direct body of a
/// runtime scope — one nesting level (``+ 4`` spaces) deeper than where
/// the scope-hoist sink lands (``scope_hoist_indent_level_``). Used to restrict
/// runtime-scope hoisting / carry-collapse to the scope's own body, so anything
/// nested in a for/if *within* the scope is left in place.
bool OrchestrationStmtCodegen::IsAtRuntimeScopeBodyIndent() const {
  return Active().GetIndentLevel() == scope_hoist_indent_level_ + 1;
}

void OrchestrationStmtCodegen::RegisterMutableTensorName(const std::string& cpp_type,
                                                         const std::string& emit_name) {
  if (cpp_type == "Tensor") {
    mutable_tensor_name_scopes_.back().insert(emit_name);
  }
}

bool OrchestrationStmtCodegen::IsMutableTensorName(const std::string& name) const {
  return std::any_of(mutable_tensor_name_scopes_.begin(), mutable_tensor_name_scopes_.end(),
                     [&](const auto& names) { return names.count(name) != 0; });
}

/// Hoist an initialized Tensor carry/phi from a runtime scope's direct body
/// when its initializer is enclosing-scope-valid. Its yield assignments stay
/// inside the block, while a subsequent loop yield can name the hoisted handle.
/// Track the original body indent to preserve snapshots inside nested loops.
void OrchestrationStmtCodegen::EmitMutableTensorCarryDecl(const std::string& name,
                                                          const std::string& init_expr) {
  RecordTensorSource(name, init_expr);
  if (scope_hoist_sink_ != nullptr && IsAtRuntimeScopeBodyIndent() && IsEnclosingScopeValid(init_expr)) {
    size_t target = tensor_hoist_frames_.size() - 1;
    // Consecutive AUTO wrappers introduce no control flow. An immutable
    // enclosing initializer can seed a descriptor outside all of them.
    // Stop at control flow, MANUAL scopes, or a scope-local initializer;
    // moving a mutable initializer could change the value being captured.
    const bool mutable_init = IsMutableTensorName(init_expr);
    while (target > 0 && !mutable_init) {
      const auto& frame = tensor_hoist_frames_[target];
      const auto& parent = tensor_hoist_frames_[target - 1];
      if (frame.scope->manual_ || parent.scope->manual_ || frame.parent_indent != parent.parent_indent + 1 ||
          parent.local_names->count(init_expr)) {
        break;
      }
      --target;
    }
    const auto& frame = tensor_hoist_frames_[target];
    if (!tensor_hoist_frames_.back().scope->manual_) {
      PropagateTensorFlag(GetTensorValue(name), &auto_scope_carry_values_);
    }
    frame.sink->push_back(IndentAtLevel(frame.parent_indent) + "Tensor " + name + " = " + init_expr + ";\n");
    mutable_tensor_name_scopes_[frame.parent_cpp_scope].insert(name);
    hoisted_carry_body_indents_[name] = scope_hoist_indent_level_ + 1;
    scope_local_names_->erase(name);
    if (frame.enclosing_local_names) frame.enclosing_local_names->insert(name);
  } else {
    EmitIndentedLine("Tensor " + name + " = " + init_expr + ";");

    RegisterMutableTensorName("Tensor", name);
  }
}

void OrchestrationStmtCodegen::RecordTensorAllocation(const std::string& name) {
  if (tensor_hoist_frames_.empty()) return;
  size_t owner = tensor_hoist_frames_.size() - 1;
  // A MANUAL allocation batch is emitted in its enclosing scheduling scope.
  const auto& frame = tensor_hoist_frames_[owner];
  if (frame.scope->manual_ && Active().GetIndentLevel() == frame.parent_indent) {
    if (owner == 0) return;
    --owner;
  }
  tensor_hoist_frames_[owner].allocations.push_back(GetTensorValue(name));
}

size_t OrchestrationStmtCodegen::NewTensorValue() {
  const size_t value = tensor_dependents_.size();
  tensor_dependents_.emplace_back();
  return value;
}

size_t OrchestrationStmtCodegen::GetTensorValue(const std::string& name) {
  const auto [it, inserted] = tensor_values_.try_emplace(name, tensor_dependents_.size());
  if (inserted) NewTensorValue();
  return it->second;
}

// Descriptor copies retain the source value, not future assignments to its
// C++ name. Propagate each flag once per value/edge; loop backedges are added
// explicitly by the ForStmt emitter, while IfStmt yields merge into its phi.
void OrchestrationStmtCodegen::PropagateTensorFlag(size_t value, std::unordered_set<size_t>* values) {
  std::vector<size_t> pending{value};
  while (!pending.empty()) {
    const size_t current = pending.back();
    pending.pop_back();
    if (!values->insert(current).second) continue;
    const auto& dependents = tensor_dependents_[current];
    pending.insert(pending.end(), dependents.begin(), dependents.end());
  }
}

void OrchestrationStmtCodegen::RecordTensorSource(size_t value, size_t source) {
  if (value == source) return;
  tensor_dependents_[source].insert(value);
  if (invalid_tensor_storage_.count(source)) PropagateTensorFlag(value, &invalid_tensor_storage_);
  if (auto_scope_carry_values_.count(source)) PropagateTensorFlag(value, &auto_scope_carry_values_);
}

void OrchestrationStmtCodegen::RecordTensorSource(const std::string& name, const std::string& source) {
  if (name.empty() || source.empty() || name == source) return;
  RecordTensorSource(GetTensorValue(name), GetTensorValue(source));
}

bool OrchestrationStmtCodegen::IsMutableTensorNameInCurrentScope(const std::string& emit_name) const {
  return !mutable_tensor_name_scopes_.empty() && mutable_tensor_name_scopes_.back().count(emit_name);
}

void OrchestrationStmtCodegen::GenerateSingleReturnAlias(const Var* result_var, const CallPtr& call,
                                                         const std::string& var_name) {
  FunctionPtr callee = program_->GetFunction(call->op_->name_);
  if (!callee) return;
  auto out_indices = CollectOutIndices(callee);
  if (out_indices.empty()) return;
  // Find the Out/InOut parameter that the callee's ReturnStmt actually
  // returns. When the kernel declares multiple Out params (e.g., a real
  // result plus per-block GM scratch tensors used by a pl.spmd-dispatched
  // mixed kernel), guessing wrong silently routes every downstream consumer
  // into the wrong buffer (#1702). Pipeline IR satisfies ReturnParamsExplicit
  // so this is a pointer-identity lookup; the fallback is reachable only for
  // IR with a single output, where it is unambiguous.
  const auto& ret_map = GetReturnedParamIndices(callee);
  auto returned_idx = ret_map.empty() ? std::nullopt : ret_map[0];
  INTERNAL_CHECK_SPAN(returned_idx.has_value() || out_indices.size() == 1, call->span_)
      << "Internal error: cannot map return of callee '" << callee->name_ << "' to one of its "
      << out_indices.size()
      << " Out/InOut params; its ReturnStmt does not reference a param directly, so the IR violates "
         "IRProperty::ReturnParamsExplicit (run NormalizeReturnOrder). Aliasing would be a guess";
  size_t param_idx = returned_idx.value_or(out_indices[0]);
  EmitTensorAlias(result_var, var_name, call, param_idx);
}

void OrchestrationStmtCodegen::GenerateTupleReturnAliases(const CallPtr& call, const Var* result_var) {
  auto tuple_key_it = tuple_var_to_key_.find(result_var);
  if (tuple_key_it == tuple_var_to_key_.end()) return;
  auto elements_it = tuple_var_to_elements_.find(tuple_key_it->second);
  if (elements_it == tuple_var_to_elements_.end()) return;
  FunctionPtr callee = program_->GetFunction(call->op_->name_);
  if (!callee) return;

  // Prefer the precise return-position -> callee param map derived from the
  // callee's ReturnStmt. It correctly handles a kernel that takes an InOut
  // param written in place but NOT returned (issue #1573) — the legacy
  // tail-alignment heuristic puts that param in ``out_indices`` and shifts
  // every carry to the wrong source tensor. Fall back to the heuristic when
  // the map is not fully trustworthy (Group/Spmd wrappers with no traceable
  // top-level ReturnStmt, or return shapes the tracer cannot model).
  const auto& ret_param_map = GetReturnedParamIndices(callee);
  const bool precise = IsReturnedParamMapPrecise(ret_param_map, call);

  // Legacy heuristic state — only computed when the precise map is unusable.
  // Classify output slots by the callee's ``ParamDirection`` (not the
  // call-site ``ArgDirection``): a ``pl.no_dep(t)`` rewrites a slot's
  // ArgDirection to ``NoDep`` even though the callee declared it Out/InOut,
  // and the return tuple still carries the post-call value at that position.
  std::vector<size_t> out_indices;
  size_t tuple_out_base = 0;
  if (!precise) {
    const auto& effective_dirs = callee->param_directions_;
    for (size_t i = 0; i < effective_dirs.size(); ++i) {
      if (effective_dirs[i] == ParamDirection::Out || effective_dirs[i] == ParamDirection::InOut) {
        out_indices.push_back(i);
      }
    }
    int max_tuple_index = -1;
    for (const auto& elem : elements_it->second) {
      max_tuple_index = std::max(max_tuple_index, elem.index);
    }
    size_t tuple_arity = max_tuple_index >= 0 ? static_cast<size_t>(max_tuple_index + 1) : 0;
    tuple_out_base = tuple_arity >= out_indices.size() ? (tuple_arity - out_indices.size()) : 0;
  }

  for (const auto& elem : elements_it->second) {
    if (elem.index < 0) continue;
    size_t elem_pos = static_cast<size_t>(elem.index);

    // Resolve the callee param index this tuple element writes back to.
    std::optional<size_t> param_idx_opt;
    if (precise) {
      if (elem_pos < ret_param_map.size()) param_idx_opt = ret_param_map[elem_pos];
    } else if (elem_pos >= tuple_out_base) {
      size_t out_pos = elem_pos - tuple_out_base;
      if (out_pos < out_indices.size()) param_idx_opt = out_indices[out_pos];
    }

    if (!param_idx_opt) {
      // Not a param writeback: the position carries no runtime output.
      //
      // A Scalar here has no carrier at all -- the runtime returns only
      // tensors. IRProperty::NoScalarKernelReturn makes that unreachable: the
      // scope outliner lifts a caller-computable scalar out of the scope body
      // and rejects the ones it cannot, so no device function reaches codegen
      // with a Scalar return. The old stop-gap emitted ``<ctype> x = 0;`` to
      // keep the generated C++ compiling, which traded a compile error for a
      // silently wrong value (#631) -- do not reinstate it.
      INTERNAL_CHECK_SPAN(!As<ScalarType>(elem.var->GetType()), call->span_)
          << "Internal error: tuple return element " << elem_pos << " of '" << call->op_->name_
          << "' is a Scalar with no param writeback; IRProperty::NoScalarKernelReturn should "
             "have rejected this callee";
      continue;
    }

    size_t param_idx = *param_idx_opt;
    INTERNAL_CHECK_SPAN(param_idx < call->args_.size(), call->span_)
        << "Internal error: resolved param_idx " << param_idx << " out of range for " << call->op_->name_
        << " (has " << call->args_.size() << " args)";

    const ManualTaskIdBinding* tid_binding = ResolveManualTaskIdBinding(result_var);
    if (!tid_binding) {
      auto arg_var = AsVarLike(call->args_[param_idx]);
      if (arg_var) {
        tid_binding = ResolveManualTaskIdBinding(arg_var.get());
      }
    }
    if (tid_binding) {
      manual_task_id_map_[elem.var] = *tid_binding;
      manual_task_id_map_by_key_[TaskIdHoistKey(elem.var)] = *tid_binding;
    }

    if (!effective_uses_.count(elem.var)) {
      continue;
    }
    std::string elem_name = ReserveVarEmitName(elem.var);
    EmitTensorAlias(elem.var, elem_name, call, param_idx);
  }
}

/// A ``pl.submit(...)`` kernel call: a non-builtin Call whose return type is
/// the flat ``TupleType([*<kernel results>, Scalar[TASK_ID]])``. The trailing
/// ``Scalar[TASK_ID]`` element is the producer TaskId the DSL ``pl.submit``
/// construct captures; it distinguishes a submit call from an ordinary
/// multi-output kernel call (which has no TaskId tail element).
bool OrchestrationStmtCodegen::IsSubmitCall(const CallPtr& call) {
  auto tuple_ty = As<TupleType>(call->GetType());
  if (!tuple_ty || tuple_ty->types_.empty()) return false;
  auto last = As<ScalarType>(tuple_ty->types_.back());
  return last != nullptr && last->dtype_ == DataType::TASK_ID;
}

/// Emit aliases for a ``pl.submit(...)`` kernel call. Tuple elements
/// ``0..N-1`` are the kernel's results — element ``N`` (the trailing
/// ``Scalar[TASK_ID]``) is the producer TaskId, bound to
/// ``task_<idx>_outs.task_id()`` and registered in ``manual_task_id_map_``
/// so a downstream ``deps=[tid]`` resolves to it.
///
/// For the Out/InOut tuple elements, the aliasing target depends on which
/// coverage region the callee param falls in (see Submit::args_ in
/// include/pypto/ir/expr.h) — *caller-allocated* (in Submit's original,
/// non-ctx args) or *runtime-allocated* (callee param index >= original
/// arg count):
///   - Caller-allocated (param_idx < original arg count): alias to
///     ``call->args_[param_idx]`` — the original tensor variable the user
///     passed in. The runtime's ``TaskOutputTensors`` stores only
///     ``add_output`` entries (see runtime/.../types.h — "Only
///     runtime-created outputs are stored here"), so ``add_inout`` /
///     in-args ``add_output(Tensor&)`` slots do **not** appear in
///     ``task_<idx>_outs`` and ``get_ref`` would skip past them or assert.
///   - Runtime-allocated (param_idx >= original arg count): alias to
///     ``task_<idx>_outs.get_ref(runtime_out_pos)`` where
///     ``runtime_out_pos = param_idx - original_arg_count`` because
///     ``BuildTaskParams`` appends one synth ``add_output`` per callee Out
///     before the materialized CommCtx suffix, in callee param order.
void OrchestrationStmtCodegen::GenerateSubmitReturnAliases(const CallPtr& call, int task_idx,
                                                           const Var* result_var) {
  auto tuple_key_it = tuple_var_to_key_.find(result_var);
  if (tuple_key_it == tuple_var_to_key_.end()) return;
  auto elements_it = tuple_var_to_elements_.find(tuple_key_it->second);
  if (elements_it == tuple_var_to_elements_.end()) return;

  auto tuple_ty = As<TupleType>(call->GetType());
  INTERNAL_CHECK_SPAN(tuple_ty != nullptr && !tuple_ty->types_.empty(), call->span_)
      << "Internal error: submit call must have a non-empty TupleType return";
  const size_t n_outs = tuple_ty->types_.size() - 1;  // trailing element is the TaskId

  FunctionPtr callee = program_->GetFunction(call->op_->name_);
  INTERNAL_CHECK_SPAN(callee != nullptr, call->span_)
      << "Internal error: submit callee '" << call->op_->name_ << "' not found";
  const size_t ctx_count = CountTrailingCommCtxParams(callee);
  INTERNAL_CHECK_SPAN(call->args_.size() >= ctx_count, call->span_)
      << "Submit to '" << call->op_->name_ << "' has " << call->args_.size()
      << " args, fewer than the materialized CommCtx suffix size " << ctx_count << ".";
  const size_t original_arg_count = call->args_.size() - ctx_count;

  // Prefer the precise return-position -> callee param map (handles an InOut
  // param written in place but not returned — issue #1573). Fall back to the
  // direction-based tail heuristic when the map is not fully trustworthy
  // (Group/Spmd wrappers, or shapes the tracer cannot model).
  const auto& ret_param_map = GetReturnedParamIndices(callee);
  const bool precise = IsReturnedParamMapPrecise(ret_param_map, call);

  // Legacy heuristic state — only computed when the precise map is unusable.
  // Kernel output param positions, in declared order, classified by the
  // callee's ``ParamDirection`` (not the call-site ``ArgDirection``): a
  // ``pl.no_dep(t)`` rewrites a slot's ArgDirection to ``NoDep`` even though
  // the callee declared it Out/InOut, yet the return tuple still carries the
  // post-call value there. Some SPMD wrappers return auxiliary scalars
  // *before* the tensor outputs, so result element ``elem_pos`` is an
  // Out/InOut tensor iff ``elem_pos >= tuple_out_base``.
  std::vector<size_t> out_indices;
  size_t tuple_out_base = 0;
  if (!precise) {
    const auto& effective_dirs = callee->param_directions_;
    for (size_t i = 0; i < effective_dirs.size(); ++i) {
      if (effective_dirs[i] == ParamDirection::Out || effective_dirs[i] == ParamDirection::InOut) {
        out_indices.push_back(i);
      }
    }
    tuple_out_base = n_outs >= out_indices.size() ? (n_outs - out_indices.size()) : 0;
  }

  for (const auto& elem : elements_it->second) {
    if (elem.index < 0) continue;
    size_t elem_pos = static_cast<size_t>(elem.index);
    if (elem_pos == n_outs) {
      // The producer TaskId element. Bind it to ``task_<idx>_outs.task_id()``
      // and register so a downstream ``deps=[tid]`` resolves to the name.
      if (!effective_uses_.count(elem.var)) continue;  // unused (``out, _ = ...``)
      std::string tid_name = BindVarTaskId(elem.var, "task_" + std::to_string(task_idx) + "_outs.task_id()");
      manual_task_id_map_[elem.var] = tid_name;
      continue;
    }
    // A kernel result element. Skip the trailing TaskId / out-of-range.
    if (elem_pos >= n_outs) continue;
    std::optional<size_t> param_idx_opt;
    if (precise) {
      if (elem_pos < ret_param_map.size()) param_idx_opt = ret_param_map[elem_pos];
    } else if (elem_pos >= tuple_out_base) {
      size_t out_pos = elem_pos - tuple_out_base;
      if (out_pos < out_indices.size()) param_idx_opt = out_indices[out_pos];
    }
    if (!param_idx_opt) {
      // Untraced position: no runtime output. A Scalar is unreachable here for
      // the same reason as in GenerateTupleReturnAliases -- see the comment
      // there, and IRProperty::NoScalarKernelReturn.
      INTERNAL_CHECK_SPAN(!As<ScalarType>(elem.var->GetType()), call->span_)
          << "Internal error: submit return element " << elem_pos << " of '" << call->op_->name_
          << "' is a Scalar with no param writeback; IRProperty::NoScalarKernelReturn should "
             "have rejected this callee";
      continue;
    }
    if (!effective_uses_.count(elem.var)) continue;
    size_t param_idx = *param_idx_opt;
    INTERNAL_CHECK_SPAN(param_idx < callee->params_.size(), call->span_)
        << "Internal error: resolved param_idx " << param_idx << " out of range for " << call->op_->name_
        << " (has " << callee->params_.size() << " params)";
    std::string elem_name = ReserveVarEmitName(elem.var);
    if (param_idx < original_arg_count) {
      // Caller-allocated: the param was passed positionally as an arg.
      // Alias to the arg's emit name — runtime tracks producer via the
      // submitted task, but TaskOutputTensors does NOT contain this slot.
      EmitTensorAlias(elem.var, elem_name, call, param_idx);
    } else {
      // Runtime-allocated: BuildTaskParams synthesised an add_output for
      // this param at runtime output position (param_idx - original_arg_count).
      RecordTensorAllocation(elem_name);
      size_t runtime_out_pos = param_idx - original_arg_count;
      std::string source =
          "task_" + std::to_string(task_idx) + "_outs.get_ref(" + std::to_string(runtime_out_pos) + ")";
      if (IsMutableTensorNameInCurrentScope(elem_name)) {
        EmitIndentedLine(elem_name + " = " + source + ";");
      } else {
        EmitIndentedLine("const Tensor& " + elem_name + " = " + source + ";");
      }
    }
  }
}

void OrchestrationStmtCodegen::HandleTensorAssembleAssign(const AssignStmtPtr& assign, const CallPtr& call) {
  INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
      << "Internal error: tensor.assemble expects 3 arguments";

  // An orchestration-level assemble emits no write of its own: it aliases the LHS to
  // the target and relies on the producing InCore kernel having stored straight into
  // the target's view. No orchestration instruction can perform an atomic combine —
  // and an atomic assemble is exactly the case FuseCreateAssembleToSlice refuses to
  // fold (folding would drop the combine mode), so the direct-into-target view is
  // never set up either. The partial products would land in a per-iteration scratch
  // buffer that is then discarded, leaving the output tensor untouched.
  const int atomic = call->GetKwarg<int>("atomic", static_cast<int>(AtomicType::kNone));
  CHECK_SPAN(atomic == static_cast<int>(AtomicType::kNone), call->span_)
      << "pl.assemble(..., atomic=pl.AtomicType.Add) is only supported inside an InCore "
         "function — typically a pl.at(level=pl.Level.CORE_GROUP, ...) scope — where it "
         "lowers to an atomic-add tile.store into global memory. This assemble sits at the "
         "orchestration level, where no atomic-combine instruction exists. Move it inside "
         "the scope that produces the partial result.";

  std::string target_name = GenerateExprString(call->args_[0]);
  target_name = GetExternalTensorName(target_name);
  emit_name_map_[assign->var_.get()] = target_name;
}

}  // namespace codegen
}  // namespace pypto
