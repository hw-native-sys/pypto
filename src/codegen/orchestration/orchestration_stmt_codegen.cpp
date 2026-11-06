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

#include "src/codegen/orchestration/orchestration_stmt_codegen.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/code_emitter.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

OrchestrationStmtCodegen::OrchestrationStmtCodegen(
    const ProgramPtr& prog, KernelRegistrations* registrations,
    std::unordered_map<const Var*, std::string> param_to_emit_name, std::set<std::string> param_name_set,
    std::map<std::string, int> param_name_to_orch_index,
    std::unordered_map<std::string, std::string> dist_param_to_ctx_param)
    : program_(prog),
      registrations_(registrations),
      emit_name_map_(std::move(param_to_emit_name)),
      param_name_set_(std::move(param_name_set)),
      param_name_to_orch_index_(std::move(param_name_to_orch_index)),
      dist_param_to_ctx_param_(std::move(dist_param_to_ctx_param)) {
  declared_var_names_ = param_name_set_;
  // Function ``Scalar[TASK_ID]`` parameters (and lineage aliases seeded into
  // ``emit_name_map_`` before construction) are live for the whole body.
  // Register them in the TaskId binding maps / live-emit set so
  // ``FindClosedScopeTaskId`` does not treat a branch/loop yield of a
  // parameter as a producer local from a closed scope.
  RegisterFunctionTaskIdParams();
  CollectCompilerDepTaskIds(program_);
}

void OrchestrationStmtCodegen::SetCallTupleElements(
    const std::map<std::string, std::vector<TupleElement>>& elements) {
  tuple_var_to_elements_ = elements;
  for (auto& [key, vec] : tuple_var_to_elements_) {
    std::sort(vec.begin(), vec.end(),
              [](const TupleElement& a, const TupleElement& b) { return a.index < b.index; });
  }
}

void OrchestrationStmtCodegen::SetTupleVarToKey(std::map<const Var*, std::string> mapping) {
  tuple_var_to_key_ = std::move(mapping);
}

void OrchestrationStmtCodegen::SetInitialIndent(int indent) {
  INTERNAL_CHECK(indent >= 0 && indent % 4 == 0)
      << "Internal error: initial indent must be a non-negative multiple of 4, got " << indent;
  emitter_.SetIndentLevel(indent / 4);
}

/// Prefix every task-arg variable this instance declares.
///
/// A Graph function is emitted by a *second* codegen instance whose
/// ``task_counter_`` / ``alloc_counter_`` restart at 0. Without a distinct
/// prefix, the entry and each graph helper would each declare ``params_t0``
/// in the same file — legal C++, since they are separate function scopes, but
/// it makes every generated identifier ambiguous to read and to grep.
void OrchestrationStmtCodegen::SetTaskVarPrefix(std::string prefix) { task_var_prefix_ = std::move(prefix); }

void OrchestrationStmtCodegen::SetGraphScalarParams(const std::vector<VarPtr>& params) {
  size_t scalar_index = 0;
  for (const auto& param : params) {
    if (AsTensorTypeLike(param->GetType())) continue;
    if (As<ScalarType>(param->GetType())) {
      graph_scalar_params_.emplace(param.get(), scalar_index);
    }
    ++scalar_index;
  }
}

const std::unordered_set<size_t>& OrchestrationStmtCodegen::GetGraphScalarValueReads() const {
  return graph_scalar_value_reads_;
}

void OrchestrationStmtCodegen::SetGraphConfigScalarSlots(GraphConfigScalarSlots slots) {
  graph_config_scalar_slots_ = std::move(slots);
}

/// Reserve identifiers the emitted C++ function already declares outside the
/// body this instance generates, so no body Var can be handed one of them.
///
/// The entry gets this for free: its ``param_name_set`` seeds
/// ``declared_var_names_``. A Graph function passes an empty set (its
/// parameters must not be ``ext_``-rewritten), so without this its
/// ``const Tensor& <p> = args.tensor(i).ref();`` prologue names are invisible
/// to ``ReserveVarEmitName``. The first body SSA rename of a parameter then
/// takes the parameter's own name — shadowing the prologue decl and, when it
/// happens inside a ``pl.manual_scope``, recording that name in
/// ``scope_local_names_`` as if it were scope-local. ``IsEnclosingScopeValid``
/// then reports the *parameter* as unreachable from outside the block, so
/// every later writeback mints a block-scoped ``const Tensor& <p>__ssa_vN =
/// <p>;`` alias instead of remapping onto the parameter — and a task placed
/// after the block references an identifier that has fallen out of C++ scope
/// (issue #2605).
void OrchestrationStmtCodegen::ReserveDeclaredNames(const std::set<std::string>& names) {
  declared_var_names_.insert(names.begin(), names.end());
}

std::string OrchestrationStmtCodegen::ReserveSyntheticEmitName(const std::string& base_name) {
  std::string emit_name = auto_name::ReserveUniqueName(base_name, declared_var_names_);
  // A name first reserved inside an active manual scope is scope-local; record
  // it so IsEnclosingScopeValid never treats it as hoistable (issue #1697).
  // Mirrors ReserveVarEmitName so the gating does not depend on every
  // synthetic-named tensor happening to be mutable / non-tensor.
  if (scope_local_names_ != nullptr) {
    scope_local_names_->insert(emit_name);
  }
  return emit_name;
}

void OrchestrationStmtCodegen::SetEffectiveUses(std::unordered_set<const Var*> uses) {
  effective_uses_ = std::move(uses);
}

bool OrchestrationStmtCodegen::NeedsVectorInclude() const { return needs_vector_include_; }

std::string OrchestrationStmtCodegen::GetGeneratedCode() const { return emitter_.GetCode(); }

// --- CodegenBase pure virtual implementations ---
std::string OrchestrationStmtCodegen::GetCurrentResultTarget() const { return current_result_var_; }

void OrchestrationStmtCodegen::Emit(const std::string& line) { Active().AppendRaw(line); }

std::string OrchestrationStmtCodegen::GetExprAsCode(const ExprPtr& expr) { return GenerateExprString(expr); }

std::string OrchestrationStmtCodegen::GetTypeString(const DataType& dtype) const {
  return dtype.ToCTypeString();
}

int64_t OrchestrationStmtCodegen::GetConstIntValue(const ExprPtr& expr) const {
  auto ci = As<ConstInt>(expr);
  INTERNAL_CHECK_SPAN(ci, expr->span_) << "Internal error: expected ConstInt expression";
  return ci->value_;
}

std::string OrchestrationStmtCodegen::GetVarName(const VarPtr& var) const {
  auto it = emit_name_map_.find(var.get());
  if (it != emit_name_map_.end()) {
    return it->second;
  }
  return GetSSABaseName(var->name_hint_);
}

std::string OrchestrationStmtCodegen::TryGetVarName(const ir::ExprPtr& expr) const {
  if (auto var = AsVarLike(expr)) {
    return GetVarName(var);
  }
  return CodegenBase::TryGetVarName(expr);
}

std::string OrchestrationStmtCodegen::GenerateExprString(const ExprPtr& expr) const {
  ValidateValueUse(expr, expr->span_);
  if (auto var = AsVarLike(expr); var && graph_scalar_params_.count(var.get())) {
    auto scalar_type = As<ScalarType>(var->GetType());
    INTERNAL_CHECK_SPAN(scalar_type, expr->span_) << "Internal error: Graph scalar must have ScalarType";
    const size_t slot = graph_scalar_params_.at(var.get());
    graph_scalar_value_reads_.insert(slot);
    return "from_u64<" + scalar_type->dtype_.ToCTypeString() + ">(_pypto_graph_config.scalar_" +
           std::to_string(slot) + ")";
  }
  return CodegenBase::GenerateExprString(expr);
}

std::string OrchestrationStmtCodegen::GetTensorShapeDim(const std::string& name, int64_t axis) const {
  std::string physical_dim;
  auto it = param_name_to_orch_index_.find(name);
  if (it != param_name_to_orch_index_.end()) {
    physical_dim = "(int64_t)orch_args.tensor(" + std::to_string(it->second) + ").ref().shapes[" +
                   std::to_string(axis) + "]";
  } else {
    physical_dim = "(int64_t)" + name + ".shapes[" + std::to_string(axis) + "]";
  }
  return physical_dim;
}

std::string OrchestrationStmtCodegen::GetTensorCreateSizeExpr(const std::string& result_var,
                                                              const std::string& default_dim_expr) const {
  auto size_it = tensor_create_size_expr_by_emit_name_.find(result_var);
  if (size_it != tensor_create_size_expr_by_emit_name_.end()) {
    return size_it->second;
  }
  return CodegenBase::GetTensorCreateSizeExpr(result_var, default_dim_expr);
}

void OrchestrationStmtCodegen::VisitStmt_(const AssignStmtPtr& assign) {
  std::string var_name = ReserveVarEmitName(assign->var_.get());
  // Funnel Submit through the existing Call codepath via the synthetic
  // SubmitToCallView adapter (deps_ → attrs[manual_dep_edges]). The IR
  // still has Submit as the canonical form; only this view is consumed
  // by the Call-shaped codegen logic. Tuple keys are looked up by the
  // binding Var (stable), never this transient view's pointer.
  CallPtr call = AsCallOrSubmitView(assign->value_);
  if (call) {
    const std::string& op_name = call->op_->name_;

    // Special-case TaskId ops emitted from the DSL surface. The producer
    // TaskId of a ``pl.submit(...)`` call is handled separately (see
    // ``GenerateSubmitReturnAliases``) — it is a tuple element of the
    // kernel Call, not a standalone op.
    if (IsOp(call, "system.task_dummy")) {
      INTERNAL_CHECK_SPAN(call->GetAttr<bool>(kAttrDummyTask, false), assign->span_)
          << "Internal error: system.task_dummy must be marked with attrs['" << kAttrDummyTask << "']";
      EmitDummyTask(call, var_name);
      manual_task_id_map_[assign->var_.get()] = var_name;
      return;
    }
    if (IsOp(call, "system.task_invalid")) {
      // The Python literal ``None`` in a TaskId position lowers here.
      EmitIndentedLine("TaskId " + var_name + " = TaskId::invalid();");

      // Register so a downstream ``deps=[<this var>]`` resolves to the
      // emitted name (string variant). The ``is_valid()`` guard in
      // ``EmitManualDeps`` skips the invalid sentinel at runtime.
      manual_task_id_map_[assign->var_.get()] = var_name;
      return;
    }
    if (IsOp(call, "system.task_is_valid")) {
      // ``b = task_is_valid(t)`` lowers to ``bool b = <t>.is_valid();``.
      // The argument is any Scalar[TASK_ID] expression — a Var holding a
      // companion id, or an ``array.get_element`` call on a TASK_ID array
      // carry. ``GenerateExprString`` handles both.
      INTERNAL_CHECK_SPAN(call->args_.size() == 1, assign->span_)
          << "Internal error: system.task_is_valid expects exactly 1 argument";
      std::string arg_expr = GenerateExprString(call->args_[0]);
      EmitIndentedLine("bool " + var_name + " = " + arg_expr + ".is_valid();");

      return;
    }
    // Launch-shape queries: the run's device geometry, latched by the runtime
    // at bring-up. Emitted as a plain int32 binding so a downstream
    // ``launch_spec.set_block_num(<var>)`` reads it back (see
    // ``RenderLaunchCoreNum``).
    if (IsOp(call, "system.available_cluster_count") || IsOp(call, "system.available_aiv_count")) {
      EmitIndentedLine("int32_t " + var_name + " = " + RenderAvailableCoreCount(call) + ";");
      return;
    }
    if (IsOp(call, "pld.system.get_comm_ctx")) {
      INTERNAL_CHECK_SPAN(call->args_.size() == 1, assign->span_)
          << "Internal error: pld.system.get_comm_ctx expects exactly 1 argument";
      std::string arg_name = TryGetVarName(call->args_[0]);
      INTERNAL_CHECK_SPAN(!arg_name.empty(), assign->span_)
          << "Internal error: orchestration get_comm_ctx expects a DistributedTensor Var argument";
      auto ctx_it = dist_param_to_ctx_param_.find(arg_name);
      INTERNAL_CHECK_SPAN(ctx_it != dist_param_to_ctx_param_.end(), assign->span_)
          << "Internal error: orchestration get_comm_ctx could not find materialized CommCtx param for "
          << arg_name;
      EmitIndentedLine("uint64_t " + var_name + " = " + ctx_it->second + ";");
      return;
    }

    if (IsTensorOp(op_name)) {
      if (IsOp(call, "tensor.assemble")) {
        HandleTensorAssembleAssign(assign, call);
      } else {
        GenerateTensorOpCode(call, var_name, assign->var_);
      }
    } else if (IsArrayOp(op_name)) {
      // ArrayType ops emit C-stack array operations. ``array.update_element``
      // is SSA-functional: the LHS Var refers to the post-update array. At
      // codegen time we alias the LHS to the input array's emit name so the
      // emitted write lands on the same storage.
      if (IsOp(call, "array.update_element")) {
        HandleArrayUpdateElementAssign(assign, call);
      }
      GenerateTensorOpCode(call, var_name, assign->var_);
      // Register an ``array.create`` result as a backing array so a ForStmt
      // ArrayType iter_arg seeded by it can reuse the same C-stack array
      // instead of allocating a fresh one and copying in slot-by-slot.
      if (IsOp(call, "array.create")) {
        if (auto arr_ty = As<ArrayType>(assign->var_->GetType())) {
          if (auto extent = As<ConstInt>(arr_ty->extent())) {
            RegisterArrayCarry(assign->var_.get(), var_name, extent->value_);
          }
        }
      } else if (IsOp(call, "array.get_element")) {
        auto scalar_ty = As<ScalarType>(assign->var_->GetType());
        if (scalar_ty && scalar_ty->dtype_ == DataType::TASK_ID) {
          manual_task_id_map_[assign->var_.get()] = var_name;
        }
      }
      // ``prev = tids[k]`` reads one slot of a TaskId array into a scalar
      // C++ local (``TaskId prev = arr[k];``). Register the LHS so a
      // downstream ``deps=[prev]`` resolves to this snapshot local (string
      // variant) — mirroring the producer-TaskId registration in
      // ``GenerateSubmitReturnAliases``. Binding to the local (not the
      // ``arr[k]`` slot) preserves snapshot semantics: a later
      // ``arr[k] = ...`` overwrite must not retroactively change ``prev``.
      // Non-TaskId ``get_element`` results are not dependency sources, so
      // they are intentionally left unregistered.
      if (IsOp(call, "array.get_element")) {
        if (auto st = As<ScalarType>(assign->var_->GetType()); st && st->dtype_ == DataType::TASK_ID) {
          manual_task_id_map_[assign->var_.get()] = var_name;
        }
      }
    } else if (!IsBuiltinOp(op_name)) {
      std::string result_key;
      if (As<TupleType>(call->GetType())) {
        // Key on the binding Var (stable for both Call and the Submit view).
        auto it = tuple_var_to_key_.find(assign->var_.get());
        result_key = (it != tuple_var_to_key_.end()) ? it->second : var_name;
      } else {
        result_key = var_name;
      }
      int task_idx_before = task_counter_;
      const bool capture_plain_task_id = NeedsCompilerDepTaskId(assign->var_.get());
      std::vector<const Var*> dep_outputs = CompilerDepOutputArgs(call);
      GenerateFunctionCallCode(call, result_key, capture_plain_task_id || !dep_outputs.empty());

      if (task_counter_ > task_idx_before && capture_plain_task_id) {
        std::string tid_name =
            BindSyntheticTaskId(assign->var_.get(), GetSSABaseName(var_name) + "_tid",
                                "task_" + std::to_string(task_idx_before) + "_outs.task_id()");
        manual_task_id_map_[assign->var_.get()] = tid_name;
        manual_task_id_map_by_key_[TaskIdHoistKey(assign->var_.get())] = tid_name;
      }
      if (task_counter_ > task_idx_before && !dep_outputs.empty()) {
        const std::string value_expr = "task_" + std::to_string(task_idx_before) + "_outs.task_id()";
        for (const Var* var : dep_outputs) {
          if (!var) continue;
          std::string tid_name =
              BindSyntheticTaskId(var, GetSSABaseName(var->name_hint_) + "_tid", value_expr);
          manual_task_id_map_[var] = tid_name;
          manual_task_id_map_by_key_[TaskIdHoistKey(var)] = tid_name;
        }
      } else if (in_manual_scope_depth_ > 0 && task_counter_ > task_idx_before && !capture_plain_task_id) {
        // Bind the LHS Var to the just-emitted ``task_<n>`` so a downstream
        // sibling call inside the same manual scope can ``add_dep`` on it.
        manual_task_id_map_[assign->var_.get()] = task_idx_before;
      }

      if (!As<TupleType>(call->GetType())) {
        if (effective_uses_.count(assign->var_.get())) {
          GenerateSingleReturnAlias(assign->var_.get(), call, var_name);
        }
      } else if (IsSubmitCall(call)) {
        GenerateSubmitReturnAliases(call, task_idx_before, assign->var_.get());
      } else {
        GenerateTupleReturnAliases(call, assign->var_.get());
      }
    } else {
      INTERNAL_CHECK_SPAN(false, assign->span_)
          << "Misplaced builtin op '" << op_name
          << "' in Orchestration function (should be inside InCore block)";
    }
  } else if (As<TupleGetItemExpr>(assign->value_)) {
    // No-op: tuple elements handled via tuple_var_to_elements_
  } else if (auto make_tuple = As<MakeTuple>(assign->value_)) {
    PropagateMakeTupleAssign(assign, make_tuple);
  } else {
    std::string value_expr = GenerateExprString(assign->value_);
    // Drop a no-op `X = X;` that arises when VarLineageCollector has
    // collapsed both LHS and RHS onto the same param-rooted emit name
    // (both Vars alias the same buffer). Without this guard the catch-all
    // emit path produces `auto X = X;`, which gcc rejects with
    // "use of 'X' before deduction of 'auto'".
    //
    // FIXME(#1281): this is a codegen-layer band-aid. The cleaner home is
    // an IR-level copy-propagation pass that drops lineage-redundant
    // Var-RHS AssignStmts so downstream analyses don't have to reason
    // about no-op aliases at all (today's Simplify only does scalar
    // constant propagation, not tensor-Var copy prop). Once such a pass
    // exists, this guard can go.
    const std::string cpp_type = GetCppType(assign->var_->GetType());
    if (auto input_var = AsVarLike(assign->value_)) {
      auto tid_it = manual_task_id_map_.find(input_var.get());
      if (tid_it != manual_task_id_map_.end()) {
        manual_task_id_map_[assign->var_.get()] = tid_it->second;
      }
      if (value_expr == var_name) {
        return;
      }
      // Collapse aliases of immutable enclosing Tensors, MANUAL aliases, and
      // safe copies of hoisted carries/phis in either runtime scope mode. This
      // keeps an after-scope reader from naming a block-local SSA copy.
      // Mutable locals still require snapshots. A hoisted carry's safe
      // post-loop copy site is its ORIGINAL runtime-scope body indent, not
      // the body indent of a nested AUTO scope entered while emitting it.
      const auto hoisted = hoisted_carry_body_indents_.find(value_expr);
      const bool carry_collapse_ok =
          hoisted == hoisted_carry_body_indents_.end() || Active().GetIndentLevel() == hoisted->second;
      const bool collapse_scope = in_manual_scope_depth_ > 0 ||
                                  hoisted != hoisted_carry_body_indents_.end() ||
                                  (scope_hoist_sink_ != nullptr && !IsMutableTensorName(value_expr));
      if (cpp_type == "Tensor" && collapse_scope && IsEnclosingScopeValid(value_expr) &&
          !IsMutableTensorNameInCurrentScope(value_expr) && !IsMutableTensorNameInCurrentScope(var_name) &&
          carry_collapse_ok) {
        emit_name_map_[assign->var_.get()] = value_expr;
        return;
      }
      // A nested runtime scope must keep a carry snapshot distinct, but its
      // C++ declaration must survive for readers after that scope. Reuse the
      // carry hoist and capture the value at the original assignment site;
      // initializing only before the scope would miss updates in its body.
      if (cpp_type == "Tensor" && hoisted != hoisted_carry_body_indents_.end() &&
          scope_hoist_sink_ != nullptr && IsAtRuntimeScopeBodyIndent() && IsEnclosingScopeValid(value_expr) &&
          !IsMutableTensorNameInCurrentScope(var_name)) {
        EmitMutableTensorCarryDecl(var_name, value_expr);
        EmitIndentedLine(var_name + " = " + value_expr + ";");
        return;
      }
    }
    EmitIndentedLine(cpp_type + " " + var_name + " = " + value_expr + ";");

    if (cpp_type == "Tensor" && AsVarLike(assign->value_)) RecordTensorSource(var_name, value_expr);
    RegisterMutableTensorName(cpp_type, var_name);
  }
}

void OrchestrationStmtCodegen::VisitStmt_(const ReturnStmtPtr& ret) {
  // No-op: return tensors are already make_tensor_external
}

void OrchestrationStmtCodegen::VisitStmt_(const SeqStmtsPtr& seq) {
  EmitBatchedAllocTensors(seq->stmts_);
  for (const auto& stmt : seq->stmts_) {
    if (batched_create_stmts_.count(stmt.get())) continue;
    VisitStmt(stmt);
  }
}

void OrchestrationStmtCodegen::VisitStmt_(const EvalStmtPtr& eval) {
  // A fire-and-forget ``pl.submit(...)`` statement (no result binding) is a
  // Submit; view it as a Call so the dispatch is still emitted.
  if (auto call = AsCallOrSubmitView(eval->expr_)) {
    const std::string& op_name = call->op_->name_;
    if (IsTensorOp(op_name) || IsArrayOp(op_name)) {
      GenerateTensorOpCode(call, "", nullptr);
    } else if (!IsBuiltinOp(op_name)) {
      int task_idx_before = task_counter_;
      std::vector<const Var*> dep_outputs = CompilerDepOutputArgs(call);
      GenerateFunctionCallCode(call, "", !dep_outputs.empty());
      if (task_counter_ > task_idx_before) {
        const std::string value_expr = "task_" + std::to_string(task_idx_before) + "_outs.task_id()";
        for (const Var* var : dep_outputs) {
          if (!var) continue;
          std::string tid_name =
              BindSyntheticTaskId(var, GetSSABaseName(var->name_hint_) + "_tid", value_expr);
          manual_task_id_map_[var] = tid_name;
          manual_task_id_map_by_key_[TaskIdHoistKey(var)] = tid_name;
        }
      }
    } else {
      INTERNAL_CHECK_SPAN(false, eval->span_)
          << "Misplaced builtin op '" << op_name
          << "' in Orchestration function (should be inside InCore block)";
    }
  }
}

CodeEmitter& OrchestrationStmtCodegen::Active() { return *active_emitter_; }

const CodeEmitter& OrchestrationStmtCodegen::Active() const { return *active_emitter_; }

void OrchestrationStmtCodegen::EmitIndentedLine(const std::string& line) { Active().EmitLine(line); }

void OrchestrationStmtCodegen::EmitBlankLine() { Active().EmitLine(""); }

std::string OrchestrationStmtCodegen::IndentAtLevel(int level) const {
  return std::string(static_cast<size_t>(level * 4), ' ');
}

std::string OrchestrationStmtCodegen::GetCppType(const TypePtr& type) {
  if (auto scalar_type = As<ScalarType>(type)) {
    // TaskId is an opaque 64-bit handle, not numeric — emit as TaskId.
    if (scalar_type->dtype_ == DataType::TASK_ID) return "TaskId";
    return scalar_type->dtype_.ToCTypeString();
  }
  // TensorType: use ``Tensor`` so default-constructible declarations are
  // legal (C++ rejects ``auto x;`` without init). Yield/Assign downstream
  // will rebind it.
  if (AsTensorTypeLike(type)) return "Tensor";
  if (As<CommCtxType>(type)) return "uint64_t";
  // ArrayType has split declaration syntax (``dtype name[N]``) — there's no
  // single "type expression" that names a C array. No caller should ask for
  // one: an ArrayType Var is either declared by ``array.create``'s op codegen,
  // or is an alias of an existing backing array (an ``array.update_element``
  // result, a ForStmt ArrayType carry, an IfStmt array phi) that needs no
  // declaration at all. If this fires, some declaration path is missing its
  // ArrayType case and would emit invalid C — surface it loudly, and name the
  // *calling* path rather than guessing which one it was.
  INTERNAL_CHECK(!As<ArrayType>(type))
      << "Internal error: GetCppType called for ArrayType — an array Var is declared by "
         "array.create's op codegen or aliased onto an existing backing array, never declared "
         "from its type. The calling declaration path is missing its ArrayType case.";
  return "auto";
}

// Encode a scalar variable for the orchestration API.
// Graph parameters keep their InheritableScalar origin when forwarded.
// Ordinary float variables must be bit-cast via to_u64().
std::string OrchestrationStmtCodegen::EncodeScalarVar(const ExprPtr& expr, const std::string& var_name,
                                                      const std::string& cpp_type) const {
  if (auto var = AsVarLike(expr); var && graph_scalar_params_.count(var.get())) {
    return var_name;
  }
  return cpp_type == "float" ? "to_u64(" + var_name + ")" : var_name;
}

// Encode a scalar constant expression for the orchestration API.
// float literals need to_u64() and an "f" suffix; other types need an explicit (uint64_t) cast.
std::string OrchestrationStmtCodegen::EncodeScalarConst(const std::string& value,
                                                        const std::string& cpp_type) {
  return cpp_type == "float" ? "to_u64(" + value + "f)" : "(uint64_t)" + value;
}

std::string OrchestrationStmtCodegen::GetExternalTensorName(const std::string& name) const {
  if (param_name_set_.count(name)) {
    return "ext_" + name;
  }
  return name;
}

void OrchestrationStmtCodegen::PushCppScope() { mutable_tensor_name_scopes_.emplace_back(); }

void OrchestrationStmtCodegen::PopCppScope() {
  INTERNAL_CHECK(!mutable_tensor_name_scopes_.empty()) << "Internal error: C++ scope stack underflow";
  mutable_tensor_name_scopes_.pop_back();
}

void OrchestrationStmtCodegen::PropagateMakeTupleAssign(const AssignStmtPtr& assign,
                                                        const MakeTuplePtr& make_tuple) {
  auto key_it = tuple_var_to_key_.find(assign->var_.get());
  if (key_it == tuple_var_to_key_.end()) {
    return;
  }
  auto elements_it = tuple_var_to_elements_.find(key_it->second);
  if (elements_it == tuple_var_to_elements_.end()) {
    return;
  }

  for (const auto& elem : elements_it->second) {
    if (elem.index < 0 || elem.index >= static_cast<int>(make_tuple->elements_.size())) {
      continue;
    }
    auto input_var = AsVarLike(make_tuple->elements_[elem.index]);
    if (!input_var) {
      continue;
    }

    auto emit_it = emit_name_map_.find(input_var.get());
    if (emit_it != emit_name_map_.end()) {
      emit_name_map_[elem.var] = emit_it->second;
    }

    auto tid_it = manual_task_id_map_.find(input_var.get());
    if (tid_it != manual_task_id_map_.end()) {
      manual_task_id_map_[elem.var] = tid_it->second;
    }

    auto carry_it = array_carry_vars_.find(input_var.get());
    if (carry_it != array_carry_vars_.end()) {
      array_carry_vars_[elem.var] = carry_it->second;
    }
  }
}

std::string OrchestrationStmtCodegen::ReserveVarEmitName(const Var* var) {
  auto it = emit_name_map_.find(var);
  if (it != emit_name_map_.end()) {
    return it->second;
  }

  auto parsed = auto_name::Parse(var->name_hint_);
  bool preserve_raw_name = parsed.role.has_value() && *parsed.role == "out";
  std::string base_name = GetSSABaseName(var->name_hint_);
  if (preserve_raw_name || declared_var_names_.count(base_name)) {
    base_name = var->name_hint_;
  }

  std::string emit_name = auto_name::ReserveUniqueName(base_name, declared_var_names_);
  emit_name_map_[var] = emit_name;
  // Record names first reserved inside an active runtime scope so
  // IsEnclosingScopeValid can tell a scope-local tensor (not hoistable) from
  // an enclosing-scope one (issue #1697).
  if (scope_local_names_ != nullptr) {
    scope_local_names_->insert(emit_name);
  }
  return emit_name;
}

}  // namespace codegen
}  // namespace pypto
