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

#include <any>
#include <cstddef>
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
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/codegen/orchestration/orchestration_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"
#include "src/codegen/orchestration/orchestration_stmt_codegen.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

/// C++ symbol emitted for a Graph function's recorded body.
///
/// The runtime identifies a graph by a bare function pointer
/// (``rt_graph_function_id`` static_asserts on that), so each Graph function is
/// emitted once as a named file-scope function rather than inlined as a lambda
/// at every call site — a lambda would mint one pointer per syntactic
/// occurrence and burn through the 16-entry Definition cache.
std::string GraphFunctionSymbol(const std::string& func_name) {
  return "pypto_graph_" + auto_name::GetCompatibleBaseName(func_name);
}

namespace {

// AIV functions that run on both vector sub-lanes carry `dual_aiv_dispatch`.
// The post-pass attribute is the dispatch source of truth: DSL split passes
// normalize it from SplitMode, while external declarations may set it directly
// because their hand-written source owns sub-lane partitioning.
bool RequiresDualAivDispatch(const FunctionPtr& aiv_func) {
  if (aiv_func == nullptr) return false;
  return aiv_func->HasAttr(kAttrDualAivDispatch) && aiv_func->GetAttr<bool>(kAttrDualAivDispatch, false);
}

// Returns the opening of a rt_submit_{aic,aiv}_task call.
// Caller appends: func_id << ", " << params << ");".
std::string CoreTypeToSubmitPrefix(CoreType core_type) {
  std::string func = core_type == CoreType::CUBE ? "rt_submit_aic_task" : "rt_submit_aiv_task";
  return func + "(";
}

}  // namespace

int KernelRegistrations::GetOrCreate(const std::string& name, CoreType core_type,
                                     std::vector<std::string> signature) {
  auto key = std::make_tuple(name, core_type, signature);
  const int id = static_cast<int>(entries.size());
  auto [it, inserted] = ids_.try_emplace(std::move(key), id);
  if (inserted) entries.push_back({id, name, core_type, std::move(signature)});
  return it->second;
}

void OrchestrationStmtCodegen::TaskDispatchPlan::Emit(OrchestrationStmtCodegen& cg) && {
  cg.EmitBlankLine();
  cg.EmitIndentedLine(comment);
  for (const auto& line : guard_lines) {
    cg.EmitIndentedLine(line);
  }
  cg.EmitTaskParamsDecl(task_var);
  for (const auto& p : params) {
    cg.EmitIndentedLine(task_var + "." + ArgDirectionToMethodName(p.direction) + "(" + p.value + ");");
  }
  cg.EmitSelectiveDumpCall(task_var, params);
  for (const auto& line : pre_lines) {
    cg.EmitIndentedLine(line);
  }
  if (deps_before_launch) {
    // Direct-call order: deps -> launch_spec -> task metadata.
    cg.EmitManualDeps(call, task_var);
    cg.EmitLaunchSpec(task_var, launch_core_num, launch_sync_start);
    cg.EmitEarlyResolveHint(task_var, call);
    cg.EmitTaskTimingSlot(task_var, call);
    cg.EmitPredicateHint(task_var, call);
  } else {
    // Wrapper (Spmd/Group/Mixed) order: launch_spec -> task metadata -> deps.
    cg.EmitLaunchSpec(task_var, launch_core_num, launch_sync_start);
    cg.EmitEarlyResolveHint(task_var, call);
    cg.EmitTaskTimingSlot(task_var, call);
    cg.EmitPredicateHint(task_var, call);
    cg.EmitManualDeps(call, task_var);
  }
  cg.EmitTaskSubmitAndBind(submit_expr, capture_outputs);
}

// Map an IR ArgDirection directly to the runtime Arg::add_* method name.
// ArgDirection is the single source of truth for codegen.
const char* OrchestrationStmtCodegen::ArgDirectionToMethodName(ArgDirection dir) {
  switch (dir) {
    case ArgDirection::Input:
      return "add_input";
    case ArgDirection::Output:
    case ArgDirection::OutputExisting:
      // The runtime overloads add_output on the argument type:
      //   add_output(TensorCreateInfo&)  -> OUTPUT       (runtime allocates)
      //   add_output(Tensor&)            -> OUTPUT_EXISTING (write-only existing tensor)
      // The codegen pre-allocates via tensor.create + alloc_tensors, so the emitted
      // call site always passes a Tensor& and the OUTPUT_EXISTING overload is selected.
      // We still distinguish the two ArgDirections in the IR to let downstream phases
      // switch to the runtime-allocated form without an IR change.
      return "add_output";
    case ArgDirection::InOut:
      return "add_inout";
    case ArgDirection::NoDep:
      return "add_no_dep";
    case ArgDirection::Scalar:
      return "add_scalar";
  }
  INTERNAL_CHECK(false) << "Internal error: unexpected ArgDirection value";
  return "";
}

/// Reorder a param list so non-scalar (tensor) entries precede scalars,
/// preserving relative order within each group — the new PTOParam ordering
/// invariant (tensors must be added before scalars; see
/// check_add_tensor_capacity() in types.h). A hand-rolled two-pass stable
/// reorder rather than ``std::stable_partition``: libstdc++'s stable_partition
/// allocates a temporary buffer through the C++17-deprecated
/// ``std::get_temporary_buffer`` for a non-trivially-relocatable element type,
/// which trips ``clang-diagnostic-deprecated-declarations`` under
/// ``-warnings-as-errors``. Param lists are short, so the extra pass is free.
std::vector<OrchestrationStmtCodegen::ParamEntry> OrchestrationStmtCodegen::ReorderTensorsBeforeScalars(
    std::vector<ParamEntry> params) {
  // ``params`` is taken by value (callers pass a local about to be destroyed),
  // so the two passes move each element exactly once — pass 1 the non-scalars,
  // pass 2 the scalars — avoiding a copy of each entry's ``std::string value``.
  // ``direction`` is a trivially-copyable enum, so it stays readable on an
  // element whose string was already moved out.
  std::vector<ParamEntry> ordered;
  ordered.reserve(params.size());
  for (auto& p : params) {
    if (p.direction != ArgDirection::Scalar) ordered.push_back(std::move(p));
  }
  for (auto& p : params) {
    if (p.direction == ArgDirection::Scalar) ordered.push_back(std::move(p));
  }
  return ordered;
}

/// Build one ParamEntry from the call-site arg at `arg_idx`. Centralises the
/// var/const/scalar dispatch that BuildTaskParams used to inline, so the
/// per-callee-param iteration (which interleaves args-derived and
/// synthesised entries) can call into it cleanly.
OrchestrationStmtCodegen::ParamEntry OrchestrationStmtCodegen::BuildOneArgParam(
    const CallPtr& call, const std::string& callee_name, const std::vector<ArgDirection>& call_arg_directions,
    size_t arg_idx) {
  const auto& arg = call->args_[arg_idx];
  ValidateValueUse(arg, call->span_);
  std::string var_name = TryGetVarName(arg);
  if (!var_name.empty()) {
    if (IsA<CommCtxType>(arg->GetType())) {
      return {ArgDirection::Scalar, var_name};
    }
    if (auto scalar_type = As<ScalarType>(arg->GetType())) {
      std::string cpp_type = scalar_type->dtype_.ToCTypeString();
      return {ArgDirection::Scalar, EncodeScalarVar(arg, var_name, cpp_type)};
    }
    std::string ext_name = GetExternalTensorName(var_name);
    return {call_arg_directions[arg_idx], ext_name};
  }
  if (auto const_int = As<ConstInt>(arg)) {
    std::string cpp_type = const_int->dtype().ToCTypeString();
    std::string value = FormatConstIntValue(const_int, cpp_type);
    return {ArgDirection::Scalar, "(uint64_t)" + value};
  }
  if (auto const_float = As<ConstFloat>(arg)) {
    std::string cpp_type = const_float->dtype().ToCTypeString();
    std::string value = FormatConstFloatValue(const_float, cpp_type);
    return {ArgDirection::Scalar, EncodeScalarConst(value, cpp_type)};
  }
  if (auto const_bool = As<ConstBool>(arg)) {
    return {ArgDirection::Scalar, const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0"};
  }
  INTERNAL_CHECK_SPAN(false, call->span_) << "Call to '" << callee_name << "' arg " << arg_idx
                                          << " is neither a variable nor a recognized constant literal "
                                          << "(unsupported expression kind for orchestration codegen).";
  return {};  // unreachable
}

/// Collect the set of arg Var pointers a Call marks for selective dump via
/// its ``kAttrDumpVars`` attr (``pl.dump_tag`` / ``dumps=``).
/// Matched by VarPtr identity in ``BuildTaskParams`` — never by name.
std::set<const Var*> OrchestrationStmtCodegen::CollectDumpVarSet(const CallPtr& call) {
  std::set<const Var*> out;
  for (const auto& [k, v] : call->attrs_) {
    if (k != kAttrDumpVars) continue;
    if (const auto* vars = std::any_cast<std::vector<VarPtr>>(&v)) {
      for (const auto& var : *vars) {
        if (var) out.insert(var.get());
      }
    }
  }
  return out;
}

size_t OrchestrationStmtCodegen::CountTrailingCommCtxParams(const FunctionPtr& callee_func) const {
  size_t ctx_count = 0;
  while (ctx_count < callee_func->params_.size() &&
         IsA<CommCtxType>(callee_func->params_[callee_func->params_.size() - 1 - ctx_count]->GetType())) {
    ++ctx_count;
  }
  return ctx_count;
}

/// Return the ``ParamEntry::value`` strings to pass to ``Arg::dump(...)``.
/// An entry is selected when its arg's Var was marked via ``kAttrDumpVars``
/// (``p.dump``, set by VarPtr identity in the param builders). Scalar entries
/// are never dumpable (the runtime rejects them in ``Arg::dump``'s
/// static_assert) and are skipped.
std::vector<std::string> OrchestrationStmtCodegen::CollectSelectiveDumpValues(
    const std::vector<ParamEntry>& params) {
  std::vector<std::string> out;
  for (const auto& p : params) {
    if (p.direction == ArgDirection::Scalar) continue;
    if (!p.dump) continue;
    out.push_back(p.value);
  }
  return out;
}

/// Emit ``<task_var>.dump(v1, v2, ...);`` for the ``params`` entries marked
/// for selective dump (``kAttrDumpVars``). No-op when no param is marked
/// (legacy full-dump path is preserved).
void OrchestrationStmtCodegen::EmitSelectiveDumpCall(const std::string& task_var,
                                                     const std::vector<ParamEntry>& params) {
  auto dump_vals = CollectSelectiveDumpValues(params);
  if (dump_vals.empty()) return;
  std::ostringstream oss;
  oss << task_var << ".dump(";
  for (size_t i = 0; i < dump_vals.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << dump_vals[i];
  }
  oss << ");";
  EmitIndentedLine(oss.str());
}

/// For a Submit's Out param at callee position `param_idx` that is *not*
/// passed in Submit::args_, emit a TensorCreateInfo declaration the runtime
/// can use to allocate the output, and return the ParamEntry that consumes
/// it via `add_output(<ci>)`. The runtime's TaskOutputTensors exposes
/// allocated outputs via `get_ref(i)` in add_output / add_inout order, so
/// this entry's runtime position is determined by where it lands in the
/// caller's `params` vector relative to other tensor entries.
OrchestrationStmtCodegen::ParamEntry OrchestrationStmtCodegen::EmitSubmitSynthOutputEntry(
    const FunctionPtr& callee_func, size_t param_idx) {
  const auto& param = callee_func->params_[param_idx];
  auto tensor_ty = As<TensorType>(param->GetType());
  INTERNAL_CHECK_SPAN(tensor_ty, param->span_)
      << "Submit synthesised output for callee '" << callee_func->name_ << "' param[" << param_idx
      << "] must have TensorType, got " << param->GetType()->TypeName();
  const size_t ndim = tensor_ty->shape_.size();
  std::string ci_var = "params_t" + std::to_string(task_counter_) + "_synth_out_" + std::to_string(param_idx);
  {
    std::ostringstream shapes;
    shapes << "uint32_t " << ci_var << "_shapes[" << ndim << "] = {";
    for (size_t i = 0; i < ndim; ++i) {
      if (i > 0) shapes << ", ";
      std::string dim_str = GenerateExprString(tensor_ty->shape_[i]);
      if (!As<ConstInt>(tensor_ty->shape_[i])) {
        dim_str = "static_cast<uint32_t>(" + dim_str + ")";
      }
      shapes << dim_str;
    }
    shapes << "};";
    EmitIndentedLine(shapes.str());
  }

  // This CI is synthesised for a callee Out param the caller never passed (a
  // runtime-allocated output). Its buffer arrives uninitialised: the runtime
  // dropped TensorCreateInfo's create-info fill, so no orchestration-side
  // pre-fill exists on any tensor.create path either. A kernel that reads
  // before writing must be ordered after a seeding kernel.
  EmitIndentedLine("TensorCreateInfo " + ci_var + "(" + ci_var + "_shapes, " + std::to_string(ndim) + ", " +
                   GetRuntimeDataTypeString(tensor_ty->dtype_) + ");");

  return {ArgDirection::Output, ci_var};
}

// Map an IR ArgDirection to the runtime ArgDirection enum name used by the
// CoreCallable signature (simpler.task_interface.ArgDirection). The runtime
// enum only distinguishes SCALAR / IN / OUT / INOUT; NoDep tensors are
// emitted as INOUT so the tensor dump captures them at both stages.
const char* OrchestrationStmtCodegen::ArgDirectionToRuntimeName(ArgDirection dir) {
  switch (dir) {
    case ArgDirection::Input:
      return "IN";
    case ArgDirection::Output:
    case ArgDirection::OutputExisting:
      return "OUT";
    case ArgDirection::InOut:
    case ArgDirection::NoDep:
      return "INOUT";
    case ArgDirection::Scalar:
      return "SCALAR";
  }
  INTERNAL_CHECK(false) << "Internal error: unexpected ArgDirection value";
  return "";
}

// Compute the registration only after wrapper bridging and direction analysis.
// Scalar payloads have their own store and do not consume tensor signature slots.
int OrchestrationStmtCodegen::RegisterKernel(const std::string& name, CoreType core_type,
                                             const std::vector<ParamEntry>& params) {
  std::vector<std::string> signature;
  for (const auto& param : params) {
    if (param.direction != ArgDirection::Scalar) {
      signature.emplace_back(ArgDirectionToRuntimeName(param.direction));
    }
  }
  return registrations_->GetOrCreate(name, core_type, std::move(signature));
}

std::vector<OrchestrationStmtCodegen::ParamEntry> OrchestrationStmtCodegen::BuildTaskParams(
    const CallPtr& call, const FunctionPtr& callee_func) {
  std::vector<ParamEntry> params;
  const std::string& callee_name = callee_func->name_;

  // Phase-5 invariant: every Call entering codegen must carry an explicit
  // ArgDirection vector (populated by the DeriveCallDirections IR pass).
  // The legacy ParamDirection fallback has been removed.
  auto call_arg_directions = call->GetArgDirections();
  INTERNAL_CHECK_SPAN(call_arg_directions.size() == call->args_.size(), call->span_)
      << "Call to '" << callee_name << "' has arg_directions size " << call_arg_directions.size()
      << " but args size " << call->args_.size()
      << ". DeriveCallDirections must run before orchestration codegen.";

  // Args use positional identity mapping against the callee param list
  // (args_[i] ↔ params_[i]) in both kinds. The difference is *coverage*:
  //   - Call: args_.size() == params_.size() (full coverage).
  //   - Submit: args_.size() <= params_.size(). NOT a plain prefix once
  //     MaterializeDistTensorCtx has run: args_ is the caller-supplied
  //     prefix *plus* the trailing CommCtx suffix, whose params grew
  //     alongside the call site. The gap between them — callee params
  //     [original_arg_count .. original_param_count) below — is the
  //     runtime-allocated outputs, which must be Out; no arg is passed, so
  //     we synth an add_output(TensorCreateInfo) entry for each, in callee
  //     param order, before emitting the CommCtx args. That ordering is
  //     what makes GenerateSubmitReturnAliases' get_ref(param_idx -
  //     original_arg_count) correct.
  // Canonical statement: Submit::args_ in include/pypto/ir/expr.h; see also
  // `.claude/rules/pass-submit-awareness.md` §5.
  // Selective dump (``pl.dump_tag`` / ``dumps=``):
  // ``kAttrDumpVars`` lists the arg Vars to mark via ``Arg::dump``. Match by
  // VarPtr identity against each arg — never by name.
  std::set<const Var*> dump_var_set = CollectDumpVarSet(call);

  params.reserve(callee_func->params_.size());
  auto push_call_arg = [&](size_t arg_idx) {
    params.push_back(BuildOneArgParam(call, callee_name, call_arg_directions, arg_idx));
    // Graph arguments retain their value category: to_u64(float_var) is an
    // rvalue and would declare a changing float static instead of dynamic.
    // The runtime performs the scalar bit encoding itself.
    if (callee_func->func_type_ == FunctionType::Graph && As<ScalarType>(call->args_[arg_idx]->GetType())) {
      if (auto var = AsVarLike(call->args_[arg_idx])) {
        params.back().value = GetVarName(var);
      }
    }
    if (!dump_var_set.empty()) {
      if (auto v = AsVarLike(call->args_[arg_idx])) {
        params.back().dump = dump_var_set.count(v.get()) > 0;
      }
    }
  };
  if (IsSubmitCall(call)) {
    INTERNAL_CHECK_SPAN(call->args_.size() <= callee_func->params_.size(), call->span_)
        << "Submit to '" << callee_name << "' has args_ size " << call->args_.size()
        << " but callee has only " << callee_func->params_.size() << " params.";
    const size_t ctx_count = CountTrailingCommCtxParams(callee_func);
    INTERNAL_CHECK_SPAN(call->args_.size() >= ctx_count, call->span_)
        << "Submit to '" << callee_name << "' has " << call->args_.size()
        << " args, fewer than the materialized CommCtx suffix size " << ctx_count << ".";
    const size_t original_arg_count = call->args_.size() - ctx_count;
    const size_t original_param_count = callee_func->params_.size() - ctx_count;
    for (size_t arg_idx = 0; arg_idx < original_arg_count; ++arg_idx) {
      push_call_arg(arg_idx);
    }
    for (size_t param_idx = original_arg_count; param_idx < original_param_count; ++param_idx) {
      INTERNAL_CHECK_SPAN(callee_func->param_directions_[param_idx] == ParamDirection::Out,
                          callee_func->params_[param_idx]->span_)
          << "Submit to '" << callee_name << "' missing positional arg for callee param[" << param_idx
          << "] which is declared " << ParamDirectionToString(callee_func->param_directions_[param_idx])
          << " (only Out params may be runtime-allocated).";
      params.push_back(EmitSubmitSynthOutputEntry(callee_func, param_idx));
    }
    for (size_t arg_idx = original_arg_count; arg_idx < call->args_.size(); ++arg_idx) {
      push_call_arg(arg_idx);
    }
  } else {
    for (size_t arg_idx = 0; arg_idx < call->args_.size(); ++arg_idx) {
      push_call_arg(arg_idx);
    }
    // Plain Call: full coverage required (args.size() == callee params).
    // The trivial `params.size() == call->args_.size()` invariant holds by
    // construction; the meaningful invariant is callee-side coverage.
    INTERNAL_CHECK_SPAN(call->args_.size() == callee_func->params_.size(), call->span_)
        << "Call to '" << callee_name << "' has " << call->args_.size() << " args but callee declares "
        << callee_func->params_.size()
        << " params (Call requires full coverage; only Submit may be a prefix).";
  }

  // New PTOParam API: tensors must precede scalars (see check_add_tensor_capacity() in types.h)
  return ReorderTensorsBeforeScalars(params);
}

OrchestrationStmtCodegen::WrapperCallInfo OrchestrationStmtCodegen::FindWrapperInnerCall(
    const FunctionPtr& wrapper_func) {
  auto info = ir::FindFirstInnerCall(wrapper_func, program_);
  return {std::move(info.inner_call), std::move(info.inner_callee)};
}

/// Walk the Group function body to find the AIC and AIV callee names
/// and the inner InCore call (needed for param reordering).
OrchestrationStmtCodegen::GroupCalleeInfo OrchestrationStmtCodegen::FindGroupCallees(
    const FunctionPtr& group_func) {
  auto info = ir::FindGroupCallees(group_func, program_);
  return {std::move(info.aic_name), std::move(info.aiv_name), std::move(info.inner_call),
          std::move(info.inner_callee)};
}

/// Build task params for a wrapper function call, reordered to match the
/// inner callee's parameter order.
///
/// Wrapper functions may omit constants or otherwise expose a different
/// parameter order than the callee binary expects. Submit args using the
/// inner callee's order, not the wrapper's order.
std::vector<OrchestrationStmtCodegen::ParamEntry> OrchestrationStmtCodegen::BuildWrapperReorderedParams(
    const CallPtr& outer_call, const FunctionPtr& wrapper_func, const CallPtr& inner_call,
    const WrapperBridge& bridge) {
  std::unordered_map<const Var*, size_t> wrapper_param_to_outer_idx;
  for (size_t i = 0; i < wrapper_func->params_.size(); ++i) {
    wrapper_param_to_outer_idx[wrapper_func->params_[i].get()] = i;
  }

  // Phase-5 invariant: the outer Call must carry explicit arg_directions
  // (populated by DeriveCallDirections). The legacy ParamDirection fallback
  // has been removed from codegen.
  auto outer_arg_directions = outer_call->GetArgDirections();
  INTERNAL_CHECK_SPAN(outer_arg_directions.size() == outer_call->args_.size(), outer_call->span_)
      << "Outer call to wrapper '" << wrapper_func->name_ << "' has arg_directions size "
      << outer_arg_directions.size() << " but args size " << outer_call->args_.size()
      << ". DeriveCallDirections must run before orchestration codegen.";

  // Second hop for the Spmd-wrapped Group case (see WrapperBridge): map each
  // Group actual to its Spmd-wrapper parameter position, which is
  // positionally 1:1 with ``outer_call->args_``. Inline argument handles may
  // leave SSA alias chains in this wrapper; require complete-parameter
  // lineage so a view or a fresh allocation cannot forward the wrong tensor.
  const bool has_bridge = (bridge.bridge_func != nullptr);
  const std::vector<std::optional<size_t>>* bridge_param_indices = nullptr;
  if (has_bridge) {
    // A non-null bridge_func always pairs with a non-null bridge_call (the
    // Group call inside it); guard here so resolve_outer_arg can safely read
    // bridge.bridge_call->span_ below.
    INTERNAL_CHECK(bridge.bridge_call != nullptr)
        << "Internal error: WrapperBridge has a non-null bridge_func but a null bridge_call.";
    auto [it, inserted] = wrapper_bridge_param_indices_cache_.try_emplace(bridge.bridge_func.get());
    if (inserted) {
      it->second = return_lineage::TraceExprsToParamIndices(
          bridge.bridge_call->args_, bridge.bridge_func->body_, bridge.bridge_func->params_, program_);
    }
    bridge_param_indices = &it->second;
  }

  // Resolve a ``wrapper_func`` (Group/Spmd) param position to the concrete
  // orchestration-level arg Expr, and report which ``outer_arg_directions``
  // slot governs it (``*dir_idx``). ``*dir_idx == kNoDir`` means the resolved
  // Expr is a constant baked into the Spmd-wrapper body (no outer direction);
  // the caller's const branches emit it inline as a Scalar.
  //
  // No bridge (plain-Spmd / direct-Group): ``wrapper_func`` IS the called
  // function, so the wrapper param index is the outer arg index directly.
  //
  // With bridge (Spmd-wrapped Group): ``wrapper_idx`` is a Group param index.
  // The Group call inside the Spmd wrapper (``bridge.bridge_call``) passes one
  // expr per Group param at the same position; that expr is a Spmd-wrapper
  // param or a complete-parameter SSA alias (resolved to its outer arg), or
  // a constant.
  constexpr size_t kNoDir = static_cast<size_t>(-1);
  auto resolve_outer_arg = [&](size_t wrapper_idx, size_t* dir_idx) -> ExprPtr {
    if (!has_bridge) {
      INTERNAL_CHECK_SPAN(wrapper_idx < outer_call->args_.size(), outer_call->span_)
          << "Internal error: wrapper param index " << wrapper_idx << " out of range for call to '"
          << wrapper_func->name_ << "' (" << outer_call->args_.size() << " args).";
      *dir_idx = wrapper_idx;
      return outer_call->args_[wrapper_idx];
    }
    INTERNAL_CHECK_SPAN(wrapper_idx < bridge.bridge_call->args_.size(), bridge.bridge_call->span_)
        << "Internal error: Group param index " << wrapper_idx << " out of range for bridge call to '"
        << wrapper_func->name_ << "' (" << bridge.bridge_call->args_.size() << " args).";
    const auto& bridge_arg = bridge.bridge_call->args_[wrapper_idx];
    auto bridge_var = AsVarLike(bridge_arg);
    if (!bridge_var) {
      *dir_idx = kNoDir;  // constant in the Spmd-wrapper body -> emit inline
      return bridge_arg;
    }
    const auto& outer_idx = (*bridge_param_indices)[wrapper_idx];
    INTERNAL_CHECK_SPAN(outer_idx.has_value(), bridge.bridge_call->span_)
        << "Internal error: Spmd-wrapper arg for Group '" << wrapper_func->name_ << "' param " << wrapper_idx
        << " does not map to any Spmd-wrapper parameter (deduped/aliased arg tracking "
        << "is inconsistent).";
    INTERNAL_CHECK_SPAN(*outer_idx < outer_call->args_.size(), outer_call->span_)
        << "Internal error: outer arg index " << *outer_idx << " out of range (" << outer_call->args_.size()
        << " args).";
    *dir_idx = *outer_idx;
    return outer_call->args_[*outer_idx];
  };

  // Per-call selective dump rides on the outer Call's ``kAttrDumpVars``
  // (e.g. a parent ``pl.dump_tag`` transferred onto the wrapper call by
  // InlineFunctions). It can equally ride on the *inner* call's
  // ``kAttrDumpVars`` — a ``pl.dump_tag`` inside the wrapped body (e.g. before
  // a for-form ``pl.spmd`` loop) attaches to the inner InCore scope, which
  // OutlineIncoreScopes carries onto the inner kernel Call. Match against
  // both: outer dump vars are outer-arg Vars, inner dump vars are inner-arg
  // Vars (mapped to the outer arg below via ``wrapper_param_to_outer_idx``).
  std::set<const Var*> dump_var_set = CollectDumpVarSet(outer_call);
  std::set<const Var*> inner_dump_var_set = CollectDumpVarSet(inner_call);

  std::vector<ParamEntry> params;
  for (size_t inner_idx = 0; inner_idx < inner_call->args_.size(); ++inner_idx) {
    const auto& inner_arg = inner_call->args_[inner_idx];
    auto inner_arg_var = AsVarLike(inner_arg);

    // Constant args are inlined in the wrapper body — emit them directly.
    if (!inner_arg_var) {
      if (auto const_int = As<ConstInt>(inner_arg)) {
        std::string cpp_type = const_int->dtype().ToCTypeString();
        std::string value = FormatConstIntValue(const_int, cpp_type);
        params.push_back({ArgDirection::Scalar, "(uint64_t)" + value});
      } else if (auto const_float = As<ConstFloat>(inner_arg)) {
        std::string cpp_type = const_float->dtype().ToCTypeString();
        std::string value = FormatConstFloatValue(const_float, cpp_type);
        params.push_back({ArgDirection::Scalar, EncodeScalarConst(value, cpp_type)});
      } else if (auto const_bool = As<ConstBool>(inner_arg)) {
        params.push_back({ArgDirection::Scalar, const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0"});
      } else {
        INTERNAL_CHECK_SPAN(false, inner_call->span_) << "Internal error: inner call arg " << inner_idx
                                                      << " is neither a variable nor a recognized constant";
      }
      continue;
    }

    auto it = wrapper_param_to_outer_idx.find(inner_arg_var.get());
    if (it == wrapper_param_to_outer_idx.end()) {
      // Some wrapper-expansion paths can leave inner-call scalar ivs that are
      // not part of the user-visible wrapper signature. They should not be
      // forwarded as orchestration args.
      if (As<ScalarType>(inner_arg->GetType()) != nullptr) {
        continue;
      }
      INTERNAL_CHECK_SPAN(false, inner_call->span_)
          << "Internal error: inner call arg " << inner_idx << " does not map to any wrapper parameter";
    }

    size_t outer_idx = it->second;
    size_t dir_idx = kNoDir;
    ExprPtr outer_arg = resolve_outer_arg(outer_idx, &dir_idx);
    ValidateValueUse(outer_arg, outer_arg->span_);
    std::string var_name = TryGetVarName(outer_arg);

    if (!var_name.empty()) {
      if (IsA<CommCtxType>(outer_arg->GetType())) {
        params.push_back({ArgDirection::Scalar, var_name});
        continue;
      }
      if (auto scalar_type = As<ScalarType>(outer_arg->GetType())) {
        std::string cpp_type = scalar_type->dtype_.ToCTypeString();
        params.push_back({ArgDirection::Scalar, EncodeScalarVar(outer_arg, var_name, cpp_type)});
        continue;
      }

      std::string ext_name = GetExternalTensorName(var_name);
      bool is_dump = false;
      if (!dump_var_set.empty()) {
        if (auto v = AsVarLike(outer_arg)) is_dump = dump_var_set.count(v.get()) > 0;
      }
      // A dump tag inside the wrapped body marks the *inner* arg Var.
      if (!is_dump && !inner_dump_var_set.empty()) {
        is_dump = inner_dump_var_set.count(inner_arg_var.get()) > 0;
      }
      // A tensor arg always resolves to a real outer Var (never a baked-in
      // constant), so ``dir_idx`` is a valid ``outer_arg_directions`` slot.
      INTERNAL_CHECK_SPAN(dir_idx < outer_arg_directions.size(), outer_call->span_)
          << "Internal error: resolved direction index " << dir_idx << " out of range for tensor arg of '"
          << wrapper_func->name_ << "' (" << outer_arg_directions.size() << " directions).";
      params.push_back({outer_arg_directions[dir_idx], ext_name, is_dump});
    } else if (auto const_int = As<ConstInt>(outer_arg)) {
      std::string cpp_type = const_int->dtype().ToCTypeString();
      std::string value = FormatConstIntValue(const_int, cpp_type);
      params.push_back({ArgDirection::Scalar, "(uint64_t)" + value});
    } else if (auto const_float = As<ConstFloat>(outer_arg)) {
      std::string cpp_type = const_float->dtype().ToCTypeString();
      std::string value = FormatConstFloatValue(const_float, cpp_type);
      params.push_back({ArgDirection::Scalar, EncodeScalarConst(value, cpp_type)});
    } else if (auto const_bool = As<ConstBool>(outer_arg)) {
      params.push_back({ArgDirection::Scalar, const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0"});
    } else {
      INTERNAL_CHECK_SPAN(false, outer_call->span_)
          << "Outer call to wrapper '" << wrapper_func->name_ << "' arg " << outer_idx
          << " is neither a variable nor a recognized constant literal "
          << "(unsupported expression kind for orchestration codegen).";
    }
  }

  INTERNAL_CHECK_SPAN(params.size() == inner_call->args_.size(), inner_call->span_)
      << "Wrapper '" << wrapper_func->name_ << "' built " << params.size() << " params for "
      << inner_call->args_.size() << " inner-call args (1:1 invariant violated).";

  return ReorderTensorsBeforeScalars(std::move(params));
}

// Render the launched function's core_num attribute as a C++ scalar expression.
// Handles a ConstInt literal, a Var resolving to an orchestration-scope scalar
// variable, or a composite scalar expression (e.g. ``b_dim * 64``,
// ``b_dim // 8``). ``GenerateExprString`` resolves leaf Vars via TryGetVarName
// (so SSA/emit-name mapping is respected) and recurses through arithmetic
// operators — the same path used to render ForStmt bounds.
// The orchestration helper behind a launch-shape query op. Both helpers are
// pure reads of state the runtime latched at bring-up, so the call is safe to
// render either as a standalone binding or inline in a launch spec.
std::string OrchestrationStmtCodegen::RenderAvailableCoreCount(const CallPtr& call) {
  return IsOp(call, "system.available_aiv_count") ? "rt_available_aiv_count()"
                                                  : "rt_available_cluster_count()";
}

std::string OrchestrationStmtCodegen::RenderLaunchCoreNum(const ExprPtr& expr) const {
  if (auto ci = As<ConstInt>(expr)) {
    return std::to_string(ci->value_);
  }
  // A launch-shape query used directly as core_num (`pl.spmd(
  // pl.system.available_cluster_count())`) — no binding statement exists, so
  // render the helper inline.
  if (auto call = As<Call>(expr)) {
    if (IsOp(call, "system.available_cluster_count") || IsOp(call, "system.available_aiv_count")) {
      return RenderAvailableCoreCount(call);
    }
  }
  // Var/IterArg or composite index arithmetic (BinaryExpr/UnaryExpr). Keep a
  // core_num-specific diagnostic here rather than falling through to
  // ``GenerateExprString``'s generic NotImplementedError.
  const bool renderable = AsVarLike(expr) != nullptr ||
                          std::dynamic_pointer_cast<const BinaryExpr>(expr) != nullptr ||
                          std::dynamic_pointer_cast<const UnaryExpr>(expr) != nullptr;
  INTERNAL_CHECK_SPAN(renderable, expr->span_)
      << "Unsupported core_num expression kind for orchestration codegen: "
      << "expected ConstInt, Var, a launch-shape query, or composite index arithmetic, got kind="
      << static_cast<int>(expr->GetKind());
  return GenerateExprString(expr);
}

// Resolve the effective SPMD launch spec for a dispatch. The spec rides the
// launch site: ``core_num`` / ``sync_start`` Call attrs for an outlined
// ``with pl.spmd`` / ``pl.cluster()`` dispatch, or the first-class Submit
// fields for ``pl.spmd_submit`` / an ``as tid`` scope (surfaced as Call attrs
// by SubmitToCallView). The launch-function fallback is legacy: no pass
// produces a Function-level spec any more, but hand-written and deserialized
// IR may still spell a constant one.
std::pair<ExprPtr, bool> OrchestrationStmtCodegen::EffectiveLaunchSpec(const CallPtr& call,
                                                                       const FunctionPtr& launch_func) const {
  ExprPtr core_num = alloc_batching::EffectiveCoreNum(call, launch_func);
  bool sync_start = call->GetAttr<bool>(kAttrSyncStart, false);
  if (!call->GetAttr<ExprPtr>(kAttrCoreNum, nullptr) && launch_func) {
    sync_start = launch_func->GetAttr<bool>(kAttrSyncStart, false);
  }
  return {core_num, sync_start};
}

void OrchestrationStmtCodegen::EmitLaunchSpec(const std::string& task_var, const ExprPtr& core_num_expr,
                                              bool sync_start) {
  if (core_num_expr) {
    const std::string method = pypto::backend::GetBackend()->GetHandler()->GetLaunchSpecCoreCountMethod();
    EmitIndentedLine(task_var + ".launch_spec." + method + "(" + RenderLaunchCoreNum(core_num_expr) + ");");
  }
  if (sync_start) {
    EmitIndentedLine(task_var + ".launch_spec.set_require_sync_start(true);");
  }
}

// Speculative early-dispatch opt-in (pl.submit(..., allow_early_resolve=True)).
// The flag rides on the Submit and is surfaced as the ``allow_early_resolve``
// Call attr by SubmitToCallView; a plain submit / call lacks it, so this is a
// no-op there. Emitted on the producer task's CoreTaskArgs before its rt_submit_* —
// see simpler#1065 ("codegen-side emission of set_allow_early_resolve()").
void OrchestrationStmtCodegen::EmitEarlyResolveHint(const std::string& task_var, const CallPtr& call) {
  if (call->GetAttr<bool>("allow_early_resolve", false)) {
    EmitIndentedLine(task_var + ".set_allow_early_resolve(true);");
  }
}

void OrchestrationStmtCodegen::EmitTaskTimingSlot(const std::string& task_var, const CallPtr& call) {
  if (!call->HasAttr(kAttrTaskTimingSlot)) return;
  const int slot = call->GetAttr<int>(kAttrTaskTimingSlot);
  CHECK_SPAN(slot >= 0 && slot < kRuntimeNumTaskTimingSlots, call->span_)
      << "task_timing_slot must be in 0.." << kRuntimeNumTaskTimingSlots - 1 << ", got " << slot;
  EmitIndentedLine(task_var + ".set_task_timing_slot(" + std::to_string(slot) + ");");
}

// Map an IR comparison node kind onto the runtime PredicateOp enumerator, and
// the flipped form used when the constant sits on the left (``0 < t[i]``).
// Returns nullptr for any other kind — the caller reports that as a user error.
const char* OrchestrationStmtCodegen::RenderPredicateOp(ObjectKind kind, bool flipped) {
  switch (kind) {
    case ObjectKind::Eq:
      return "EQ";
    case ObjectKind::Ne:
      return "NE";
    case ObjectKind::Gt:
      return flipped ? "LT" : "GT";
    case ObjectKind::Lt:
      return flipped ? "GT" : "LT";
    case ObjectKind::Ge:
      return flipped ? "LE" : "GE";
    case ObjectKind::Le:
      return flipped ? "GE" : "LE";
    default:
      return nullptr;
  }
}

// Strip the type-promotion Cast the DSL inserts around a tensor.read whose
// dtype differs from the compared literal's (e.g. INT32 read vs an INDEX
// literal), so the pattern match sees the read itself.
ExprPtr OrchestrationStmtCodegen::StripCast(const ExprPtr& e) {
  auto cur = e;
  while (auto cast = As<Cast>(cur)) cur = cast->operand_;
  return cur;
}

// Dispatch predicate (pl.spmd_submit(..., predicate=(t[i] > 0))). Surfaced as
// the ``predicate`` Call attr (the comparison Expr) by SubmitToCallView.
//
// This is where the IR's general comparison Expr is decomposed into the
// runtime's `operand OP target` triple — the runtime ABI is codegen's
// concern, so the IR keeps the predicate as an ordinary Expr rather than a
// pre-decomposed bundle. Only the shape the runtime can express is accepted:
// a single comparison between a `tensor.read(t, [indices])` and an integer
// constant, in either operand order. Anything else is a user error reported
// here (the DSL parser rejects it earlier with a better message; this is the
// backstop for IR built by other means).
void OrchestrationStmtCodegen::EmitPredicateHint(const std::string& task_var, const CallPtr& call) {
  if (!call->HasAttr(kAttrPredicate)) return;
  auto pred = call->GetAttr<ExprPtr>(kAttrPredicate, nullptr);
  INTERNAL_CHECK_SPAN(pred, call->span_) << "Submit predicate attr is null";
  auto cmp = std::dynamic_pointer_cast<const BinaryExpr>(pred);
  CHECK_SPAN(cmp && RenderPredicateOp(pred->GetKind(), false) != nullptr, pred->span_)
      << "Submit dispatch predicate must be a single comparison (==, !=, >, <, >=, <=) between a "
         "tensor element and an integer literal, e.g. predicate=(t[i] > 0)";

  // Orientation: `read OP const`, or the mirrored `const OP read` — the
  // runtime always stores the tensor as the operand, so flip the operator.
  auto lhs = StripCast(cmp->left_);
  auto rhs = StripCast(cmp->right_);
  auto read = As<Call>(lhs);
  auto konst = As<ConstInt>(rhs);
  bool flipped = false;
  if (!(read && IsOp(read, "tensor.read") && konst)) {
    read = As<Call>(rhs);
    konst = As<ConstInt>(lhs);
    flipped = true;
  }
  CHECK_SPAN(read && IsOp(read, "tensor.read") && konst, pred->span_)
      << "Submit dispatch predicate must compare a tensor element (a tensor.read, e.g. t[i]) against "
         "an integer literal; got neither side in that form";
  INTERNAL_CHECK_SPAN(read->args_.size() >= 2, read->span_)
      << "tensor.read in a dispatch predicate must have a tensor and an index list";

  // tensor.read(tensor, MakeTuple(i0, i1, ...)) — arg 0 is the tensor, arg 1
  // is the per-axis index list (a MakeTuple, or a single index expression for
  // a rank-1 read).
  const auto& operand = read->args_[0];

  // The runtime reads `elem_size` bytes at the operand address and
  // **sign-extends** to int64 before comparing (DispatchPredicate::pass()).
  // An unsigned operand whose value has the top bit set therefore compares as
  // negative and silently inverts the dispatch decision — e.g. a UINT32
  // row_count of 3'000'000'000 sign-extends to -1'294'967'296, so
  // `row_count[e] > 0` is false and the expert is skipped with no diagnostic.
  // Sub-byte dtypes have no addressable single-element read at all. Both are
  // rejected here: codegen owns the runtime ABI, so it is the backstop for IR
  // that did not come through the DSL parser.
  auto operand_tensor_type = AsTensorTypeLike(operand->GetType());
  CHECK_SPAN(operand_tensor_type, operand->span_) << "Submit dispatch-predicate operand must be a tensor";
  const DataType operand_dtype = operand_tensor_type->dtype_;
  const size_t operand_bits = operand_dtype.GetBit();
  CHECK_SPAN(operand_dtype.IsSignedInt() &&
                 (operand_bits == 8 || operand_bits == 16 || operand_bits == 32 || operand_bits == 64),
             operand->span_)
      << "Submit dispatch-predicate operand must be a signed 8/16/32/64-bit integer tensor, got "
      << operand_dtype.ToString()
      << ". The runtime sign-extends the value it reads, so an unsigned operand can compare as "
         "negative and silently invert the dispatch decision; sub-byte dtypes have no addressable "
         "single-element read.";

  std::vector<ExprPtr> indices;
  if (auto idx_tuple = As<MakeTuple>(read->args_[1])) {
    indices = idx_tuple->elements_;
  } else {
    indices.push_back(read->args_[1]);
  }
  // ``CorePredicateOperand::indices`` is a fixed ``uint32_t[MAX_TENSOR_DIMS]``;
  // emitting more would write past it (the next field is ``op``, so an
  // overflow corrupts the comparison itself).
  CHECK_SPAN(indices.size() <= kRuntimeMaxTensorDims, read->span_)
      << "Submit dispatch-predicate operand has rank " << indices.size()
      << ", exceeding the runtime's maximum of " << kRuntimeMaxTensorDims
      << " indices (CorePredicateOperand::indices is a fixed-size array)";
  ValidateValueUse(operand, read->span_);
  const std::string var_name = TryGetVarName(operand);
  CHECK_SPAN(!var_name.empty(), operand->span_)
      << "Submit dispatch-predicate operand must be a named tensor (a function parameter or a variable "
         "bound to a tensor), got an unnamed expression of kind "
      << static_cast<int>(operand->GetKind())
      << ". Bind the tensor to a variable first and pass that variable.";
  const std::string tname = GetExternalTensorName(var_name);
  const std::string pv = task_var + "_pred";
  EmitIndentedLine("CoreTaskPredicate " + pv + ";");
  EmitIndentedLine(pv + ".operand.tensor = &" + tname + ";");
  EmitIndentedLine(pv + ".operand.ndims = " + std::to_string(indices.size()) + ";");
  for (size_t i = 0; i < indices.size(); ++i) {
    // The runtime index array is ``uint32_t``, so a negative constant would
    // wrap to a huge value and compute an out-of-bounds GM address that the
    // scheduler then reads. Reject the statically-known case; a negative
    // *dynamic* index cannot be caught here and stays the author's
    // responsibility, same as any other out-of-range index in the DSL.
    if (auto const_idx = As<ConstInt>(indices[i])) {
      CHECK_SPAN(const_idx->value_ >= 0, indices[i]->span_)
          << "Submit dispatch-predicate index " << i << " is negative (" << const_idx->value_
          << "); the runtime stores indices as uint32_t, so it would wrap to an out-of-bounds address";
    }
    EmitIndentedLine(pv + ".operand.indices[" + std::to_string(i) + "] = " + GenerateExprString(indices[i]) +
                     ";");
  }
  EmitIndentedLine(pv + ".op = PredicateOp::" + RenderPredicateOp(pred->GetKind(), flipped) + ";");
  EmitIndentedLine(pv + ".target = " + std::to_string(konst->value_) + ";");
  EmitIndentedLine(task_var + ".set_predicate(" + pv + ");");
}

void OrchestrationStmtCodegen::EmitTaskSubmitAndBind(const std::string& submit_expr, bool capture_outputs) {
  if (capture_outputs) {
    // The caller will consume this task's producer TaskId — capture the
    // ``TaskOutputTensors`` handle so ``GenerateSubmitReturnAliases`` can
    // bind it to ``task_<n>_outs.task_id()`` on demand. Orthogonal to
    // scope mode: a ``pl.submit(...)`` call captures the handle whether
    // it appears in a manual or auto scope, since
    // ``Arg::set_dependencies`` is itself orthogonal to OverlapMap
    // tracking (final fanin = auto ∪ explicit).
    const std::string outs_var = "task_" + std::to_string(task_counter_) + "_outs";
    EmitIndentedLine("TaskOutputTensors " + outs_var + " = " + submit_expr + ";");
  } else {
    EmitIndentedLine(submit_expr + ";");
  }
  task_counter_++;
}

/// Emit the per-task ``Arg`` declaration. Dependency edges (if any) are
/// attached separately by ``EmitManualDeps`` via ``set_dependencies``.
///
/// @p args_type selects the ``Arg`` instantiation. A task submitted inside
/// orchestration takes `CoreTaskArgs`; a Graph boundary takes `GraphTaskArgs`,
/// which `rt_submit_graph` requires. The two are *different* instantiations of
/// the same template — `Arg<MAX_TENSOR_ARGS, MAX_SCALAR_ARGS>` versus
/// `Arg<GRAPH_MAX_TENSOR_ARGS, GRAPH_MAX_SCALAR_ARGS>` — so they are not
/// interchangeable, and passing the wrong one is a compile error in the
/// generated file rather than anything a codegen text assertion would catch.
void OrchestrationStmtCodegen::EmitTaskParamsDecl(const std::string& task_var, const std::string& args_type) {
  EmitIndentedLine(args_type + " " + task_var + ";");
}

std::string OrchestrationStmtCodegen::CurrentTaskVarName() const {
  return task_var_prefix_ + "params_t" + std::to_string(task_counter_);
}

OrchestrationStmtCodegen::TaskDispatchPlan OrchestrationStmtCodegen::BuildTaskDispatchPlan(
    const std::string& comment, const CallPtr& call, std::vector<ParamEntry>&& params,
    const ExprPtr& launch_core_num, bool launch_sync_start, const std::string& submit_expr,
    bool capture_outputs, std::vector<std::string>&& pre_lines) {
  TaskDispatchPlan plan;
  plan.comment = comment;
  plan.task_var = CurrentTaskVarName();
  plan.pre_lines = std::move(pre_lines);
  plan.params = std::move(params);
  plan.call = call;
  plan.launch_core_num = launch_core_num;
  plan.launch_sync_start = launch_sync_start;
  plan.submit_expr = submit_expr;
  plan.capture_outputs = capture_outputs;
  return plan;
}

/// Build the submit for a task that dispatches ONE kernel function — a direct
/// call, a Spmd wrapper, or an AIV-only Group.
///
/// An AIV kernel stamped `dual_aiv_dispatch` must run on BOTH vector lanes of a
/// cluster: a `pl.split_aiv` region hands each lane disjoint work selected by
/// `aiv_id`. `rt_submit_aiv_task` fills only the AIV0 slot, so the mask carries one
/// core bit and the runtime schedules an AIV-shape task: one AIV core per block.
/// The second lane never launches, and the lone lane that does reads a
/// `get_sub_block_id()` the runtime documents as meaningless for a single-AIV task
/// (runtime `common/intrinsic.h`) — the scheduler seeds it per core from that core's
/// fixed position in its cluster, so `aiv_id` is that position rather than a
/// per-block lane id. Half the work is then silently dropped (issue #2006).
///
/// Emit an explicit `MixedKernels{INVALID, id, id}` instead: two active AIV slots
/// make it a MIX-shape task, so the scheduler places both lanes of one cluster
/// under the same block_idx and gives them sub_block_id 0 and 1. The cluster's AIC
/// core is simply unused. Non-dual-AIV kernels keep `rt_submit_{aic,aiv}_task`,
/// which dispatches across independent cores.
OrchestrationStmtCodegen::SubmitEmission OrchestrationStmtCodegen::BuildSingleKernelSubmit(
    CoreType core_type, const FunctionPtr& kernel_func, int func_id, const std::string& task_var) {
  if (core_type == CoreType::VECTOR && RequiresDualAivDispatch(kernel_func)) {
    const std::string mixed_var = "mixed_" + std::to_string(task_counter_);
    const std::string id = std::to_string(func_id);
    return {"rt_submit_task(" + mixed_var + ", " + task_var + ")",
            {"MixedKernels " + mixed_var + " = {INVALID_KERNEL_ID, " + id + ", " + id + "};"}};
  }
  return {CoreTypeToSubmitPrefix(core_type) + std::to_string(func_id) + ", " + task_var + ")", {}};
}

OrchestrationStmtCodegen::TaskDispatchPlan OrchestrationStmtCodegen::BuildDirectCallDispatchPlan(
    const CallPtr& call, const FunctionPtr& callee_func, const std::string& callee_name, CoreType core_type,
    int func_id, std::vector<ParamEntry>&& params, bool capture_plain_task_id) {
  std::string task_var = CurrentTaskVarName();
  auto submit = BuildSingleKernelSubmit(core_type, callee_func, func_id, task_var);
  auto [launch_core_num, launch_sync_start] = EffectiveLaunchSpec(call, callee_func);
  TaskDispatchPlan plan = BuildTaskDispatchPlan(
      "// Task " + std::to_string(task_counter_) + ": " + callee_name, call, std::move(params),
      launch_core_num, launch_sync_start, submit.expr, ShouldCaptureTaskOutputs(call, capture_plain_task_id),
      std::move(submit.pre_lines));
  // Direct calls historically emit set_dependencies before the launch spec.
  plan.deps_before_launch = true;
  plan.guard_lines = BuildContiguityGuardLines(call, callee_func);
  return plan;
}

/// Assert that a wrapper-dispatched kernel does not need operand guards.
///
/// ``BuildContiguityGuardLines`` is wired into the direct-call plan only.
/// ``kAttrRequireContiguousOperands`` is written by ``LowerL2TensorCollectives``, which runs
/// after every outlining pass that builds a ``Spmd`` / ``Group`` wrapper, so the kernel it
/// synthesizes cannot be a wrapper's inner kernel and the direct path is the only one that can
/// carry the attr. That is a pipeline-ordering invariant, not a property of this file:
/// reordering the passes would silently drop the guard and let a strided operand reach the
/// kernel again. Fail loudly here instead.
void OrchestrationStmtCodegen::CheckWrapperNeedsNoContiguityGuard(const FunctionPtr& kernel_func,
                                                                  const CallPtr& call) {
  INTERNAL_CHECK_SPAN(!kernel_func->GetAttr<bool>(kAttrRequireContiguousOperands, false), call->span_)
      << "Internal error: kernel '" << kernel_func->name_
      << "' requires contiguous operands but is dispatched through a wrapper plan, which emits no "
         "operand guards; LowerL2TensorCollectives must run after every outlining pass";
}

/// Runtime half of a kernel's ``kAttrRequireContiguousOperands`` contract:
/// one ``is_contiguous`` check per tensor argument, chained with ``else if``
/// so only the first violation is reported.
///
/// The checks latch ``SIMPLER_ERROR_INVALID_ARGS`` and deliberately do not
/// ``return``: this dispatch may sit inside a scope, a loop, or a function
/// whose return type is not ``void``, and the runtime already turns every
/// later ``submit_task`` / scope boundary into a no-op once a fatal is
/// latched, so the guarded submit below never launches.
///
/// ``call->args_[i]`` binds ``callee_func->params_[i]`` for every tensor
/// argument: a Call has full coverage, and a Submit's caller-supplied args are
/// the leading region (its gap and CommCtx suffix carry no tensor operands).
///
/// Backstop for caller-supplied operands; see kAttrRequireContiguousOperands.
std::vector<std::string> OrchestrationStmtCodegen::BuildContiguityGuardLines(const CallPtr& call,
                                                                             const FunctionPtr& callee_func) {
  std::vector<std::string> lines;
  if (!callee_func->GetAttr<bool>(kAttrRequireContiguousOperands, false)) {
    return lines;
  }
  const std::string prefix = "rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, \"" + callee_func->name_ + ": ";
  std::string keyword = "if";
  for (size_t i = 0; i < call->args_.size() && i < callee_func->params_.size(); ++i) {
    const auto& arg = call->args_[i];
    if (!AsTensorTypeLike(arg->GetType())) {
      continue;
    }
    const std::string var_name = TryGetVarName(arg);
    INTERNAL_CHECK_SPAN(!var_name.empty(), call->span_)
        << "Internal error: tensor argument " << i << " of '" << callee_func->name_
        << "' does not resolve to a variable, so its contiguity cannot be checked";
    lines.push_back(keyword + " (!" + GetExternalTensorName(var_name) + ".is_contiguous) {");
    lines.push_back("    " + prefix + callee_func->params_[i]->name_hint_ +
                    " must be a contiguous row-major view\");");
    keyword = "} else if";
  }
  if (!lines.empty()) {
    lines.insert(lines.begin(), "// " + callee_func->name_ +
                                    " reads every operand as dense row-major (require_contiguous_operands)");
    lines.emplace_back("}");
  }
  return lines;
}

/// ``kernel_func`` is the AIC/AIV kernel the wrapper dispatches (``spmd_func`` is
/// the Spmd wrapper itself, which carries the launch spec but no core attrs).
OrchestrationStmtCodegen::TaskDispatchPlan OrchestrationStmtCodegen::BuildSpmdCallDispatchPlan(
    const CallPtr& call, const FunctionPtr& spmd_func, const FunctionPtr& kernel_func,
    const std::string& callee_name, CoreType core_type, int func_id, std::vector<ParamEntry>&& params,
    bool capture_plain_task_id) {
  CheckWrapperNeedsNoContiguityGuard(kernel_func, call);
  std::string task_var = CurrentTaskVarName();
  auto [launch_core_num, launch_sync_start] = EffectiveLaunchSpec(call, spmd_func);
  auto submit = BuildSingleKernelSubmit(core_type, kernel_func, func_id, task_var);
  return BuildTaskDispatchPlan("// Spmd " + spmd_func->name_ + ": " + callee_name, call, std::move(params),
                               launch_core_num, launch_sync_start, submit.expr,
                               ShouldCaptureTaskOutputs(call, capture_plain_task_id),
                               std::move(submit.pre_lines));
}

OrchestrationStmtCodegen::TaskDispatchPlan OrchestrationStmtCodegen::BuildAivOnlyGroupDispatchPlan(
    const CallPtr& call, const FunctionPtr& launch_func, const std::string& group_name,
    const FunctionPtr& aiv_func, int aiv_id, std::vector<ParamEntry>&& params, bool capture_plain_task_id) {
  CheckWrapperNeedsNoContiguityGuard(aiv_func, call);
  std::string task_var = CurrentTaskVarName();
  auto [launch_core_num, launch_sync_start] = EffectiveLaunchSpec(call, launch_func);
  auto submit = BuildSingleKernelSubmit(CoreType::VECTOR, aiv_func, aiv_id, task_var);
  return BuildTaskDispatchPlan("// Group " + group_name + ": AIV-only SPMD", call, std::move(params),
                               launch_core_num, launch_sync_start, submit.expr,
                               ShouldCaptureTaskOutputs(call, capture_plain_task_id),
                               std::move(submit.pre_lines));
}

OrchestrationStmtCodegen::TaskDispatchPlan OrchestrationStmtCodegen::BuildMixedGroupDispatchPlan(
    const CallPtr& call, const FunctionPtr& launch_func, const std::string& group_name, int aic_id,
    int aiv_id, const FunctionPtr& aiv_func, std::vector<ParamEntry>&& params, bool capture_plain_task_id) {
  CheckWrapperNeedsNoContiguityGuard(aiv_func, call);
  std::string task_var = CurrentTaskVarName();
  auto [launch_core_num, launch_sync_start] = EffectiveLaunchSpec(call, launch_func);
  std::string submit_expr = "rt_submit_task(mixed_" + std::to_string(task_counter_) + ", " + task_var + ")";
  // Split AIV groups dispatch the same kernel on both vector lanes.
  std::string third_id = RequiresDualAivDispatch(aiv_func) ? std::to_string(aiv_id) : "INVALID_KERNEL_ID";
  std::vector<std::string> pre_lines{"MixedKernels mixed_" + std::to_string(task_counter_) + " = {" +
                                     std::to_string(aic_id) + ", " + std::to_string(aiv_id) + ", " +
                                     third_id + "};"};
  return BuildTaskDispatchPlan("// Group " + group_name + ": MixedKernels (AIC + AIV lanes)", call,
                               std::move(params), launch_core_num, launch_sync_start, submit_expr,
                               ShouldCaptureTaskOutputs(call, capture_plain_task_id), std::move(pre_lines));
}

void OrchestrationStmtCodegen::GenerateFunctionCallCode(const CallPtr& call, const std::string& result_var,
                                                        bool capture_plain_task_id) {
  const std::string& callee_name = call->op_->name_;

  FunctionPtr callee_func = program_->GetFunction(callee_name);
  INTERNAL_CHECK_SPAN(callee_func != nullptr, call->span_)
      << "Internal error: function '" << callee_name << "' not found after validation.";

  if (callee_func->func_type_ == FunctionType::Spmd) {
    GenerateSpmdCallCode(call, callee_func, capture_plain_task_id);
    return;
  }

  if (callee_func->func_type_ == FunctionType::Group) {
    GenerateGroupCallCode(call, callee_func, callee_func, capture_plain_task_id);
    return;
  }

  // A Graph is a recordable orchestration fragment launched as one task, not
  // a kernel. It must be intercepted before InferFunctionCoreType, which
  // recognises only AIC/AIV and hard-fails on anything else. This replaces the
  // placeholder guard that reported the missing emission path.
  if (callee_func->func_type_ == FunctionType::Graph) {
    GenerateGraphCallCode(call, callee_func, capture_plain_task_id);
    return;
  }

  CoreType core_type = InferFunctionCoreType(callee_func);

  auto params = BuildTaskParams(call, callee_func);
  int func_id = RegisterKernel(callee_name, core_type, params);

  // SPMD launch spec for pl.spmd_submit targeting an AIC/AIV kernel directly
  // (no Spmd-wrapper function). core_num/sync_start ride on the Submit and
  // are surfaced as Call attrs by SubmitToCallView; a plain submit / call
  // has neither, so EffectiveLaunchSpec yields (nullptr, false) → no-op.
  BuildDirectCallDispatchPlan(call, callee_func, callee_name, core_type, func_id, std::move(params),
                              capture_plain_task_id)
      .Emit(*this);
}

/// Emit the launch of a Graph function: one task carrying the whole region.
///
/// Deliberately unlike every other submit here, in two ways:
///
/// * The result is discarded. ``rt_submit_graph`` returns a valid task id only
///   on a cache *hit*; on the recording pass ``graph_begin`` reports an invalid
///   one. Nothing may chain a dependency on it — ordering comes from the
///   boundary tensors' dataflow, which is why LegalizeGraphBoundary rejects a
///   Graph that returns a computed value.
/// * The callee gets no kernel id. A Graph has no ``.cpp`` under ``kernels/``,
///   so registering it in the kernel registrations would put a phantom entry in
///   ``kernel_config.py``'s KERNELS list pointing at a file that does not
///   exist.
///
/// The recording is identified by the emitted function's own address, via the
/// ``rt_submit_graph(function, args)`` overload. The ``GRAPH_KEY("...")``
/// overload exists for callers needing a name stable across builds, but it
/// drops the function pointer from the cache identity, so two functions
/// sharing a key silently replay one recording. One Graph function is one
/// emitted C++ function here, so the address is already the right identity.
void OrchestrationStmtCodegen::GenerateGraphCallCode(const CallPtr& call, const FunctionPtr& callee_func,
                                                     bool capture_plain_task_id) {
  // A graph launch produces no TaskId a consumer may depend on:
  // `rt_submit_graph` returns a valid id only on a cache *hit*, so this
  // function discards the result and declares no `TaskOutputTensors`. The
  // caller, however, keys its dependency binding on `task_counter_` advancing
  // and would then emit `task_<n>_outs.task_id()` naming a variable that was
  // never declared — generated C++ that does not compile.
  //
  // `LegalizeGraphBoundary` rejects an *explicit* dependency on a graph
  // launch; a compiler-derived manual-scope edge is the same situation and is
  // caught here, where the producing edge is known.
  INTERNAL_CHECK_SPAN(!capture_plain_task_id && CompilerDepOutputArgs(call).empty(), call->span_)
      << "Internal error: Graph function '" << callee_func->name_
      << "' is launched as the producer of a task dependency, but rt_submit_graph yields a valid "
         "task id only on a cache hit. Order the graph against its consumers through its boundary "
         "tensors instead of a TaskId edge.";

  auto params = BuildTaskParams(call, callee_func);
  const std::string task_var = CurrentTaskVarName();

  EmitBlankLine();
  EmitIndentedLine("// Graph region: " + callee_func->name_ + " (recorded once, then replayed)");
  // `rt_submit_graph(GraphFunction, const GraphTaskArgs&)` — the boundary
  // container is its own `Arg` instantiation, sized by GRAPH_MAX_*_ARGS.
  EmitTaskParamsDecl(task_var, "GraphTaskArgs");
  const auto slots_it = graph_config_scalar_slots_.find(callee_func->name_);
  INTERNAL_CHECK_SPAN(slots_it != graph_config_scalar_slots_.end(), call->span_)
      << "Internal error: Graph scalar usage was not collected for '" << callee_func->name_ << "'";
  const std::string config_var = task_var + "_config";
  EmitIndentedLine(GraphFunctionSymbol(callee_func->name_) + "Config " + config_var + ";");
  size_t scalar_index = 0;
  for (const auto& p : params) {
    INTERNAL_CHECK_SPAN(p.direction != ArgDirection::Output, call->span_)
        << "Internal error: Graph function '" << callee_func->name_
        << "' has a runtime-allocated output argument; a recorded graph's boundary tensors must "
           "already exist so replay can patch their addresses.";
    const char* method = ArgDirectionToMethodName(p.direction);
    if (p.direction == ArgDirection::Scalar) {
      if (slots_it->second.count(scalar_index)) {
        EmitIndentedLine(config_var + ".scalar_" + std::to_string(scalar_index) + " = to_u64(" + p.value +
                         ");");
      }
      ++scalar_index;
    }
    EmitIndentedLine(task_var + "." + method + "(" + p.value + ");");
  }
  EmitSelectiveDumpCall(task_var, params);
  EmitIndentedLine(config_var + ".boundary = pypto_graph_boundary_layout(" + task_var + ");");
  EmitIndentedLine("rt_submit_graph(&" + GraphFunctionSymbol(callee_func->name_) + ", " + task_var + ", " +
                   config_var + ");");
  ++task_counter_;
}

void OrchestrationStmtCodegen::GenerateSpmdCallCode(const CallPtr& call, const FunctionPtr& spmd_func,
                                                    bool capture_plain_task_id) {
  auto info = FindWrapperInnerCall(spmd_func);
  INTERNAL_CHECK(info.inner_call != nullptr && info.inner_callee != nullptr)
      << "Internal error: no inner call found in Spmd function '" << spmd_func->name_ << "'";

  if (info.inner_callee->func_type_ == FunctionType::Group) {
    // The Group is dispatched THROUGH this Spmd wrapper: ``call`` invokes
    // ``spmd_func``, not the Group. Pass the Group call inside the wrapper
    // (``info.inner_call``) as the bridge: its args are positionally 1:1 with
    // the Group's params, and each arg references a ``spmd_func`` param,
    // complete-parameter alias, or constant — so BuildWrapperReorderedParams can map Group params ->
    // Spmd-wrapper params -> outer args even when an aliased-arg dedup shrank
    // the wrapper's param count below the Group's.
    GenerateGroupCallCode(call, info.inner_callee, spmd_func, capture_plain_task_id,
                          WrapperBridge{info.inner_call, spmd_func});
    return;
  }

  const std::string& callee_name = info.inner_callee->name_;
  CoreType core_type = InferFunctionCoreType(info.inner_callee);

  auto params = BuildWrapperReorderedParams(call, spmd_func, info.inner_call);
  int func_id = RegisterKernel(callee_name, core_type, params);

  BuildSpmdCallDispatchPlan(call, spmd_func, info.inner_callee, callee_name, core_type, func_id,
                            std::move(params), capture_plain_task_id)
      .Emit(*this);
}

void OrchestrationStmtCodegen::GenerateGroupCallCode(const CallPtr& call, const FunctionPtr& group_func,
                                                     const FunctionPtr& launch_func,
                                                     bool capture_plain_task_id,
                                                     const WrapperBridge& bridge) {
  std::string group_name = group_func->name_;

  auto info = FindGroupCallees(group_func);

  // AIV-only Group: pure vector SPMD kernel (no AIC callee), dispatched with
  // core_num/sync_start from the Group. BuildSingleKernelSubmit picks the
  // submit: rt_submit_aiv_task (independent AIV cores) for a plain kernel, or a
  // both-lanes MixedKernels for a `dual_aiv_dispatch` one.
  if (info.aic_name.empty() && !info.aiv_name.empty()) {
    FunctionPtr aiv_func = program_->GetFunction(info.aiv_name);
    INTERNAL_CHECK(aiv_func != nullptr) << "Internal error: AIV function '" << info.aiv_name
                                        << "' not found for Group '" << group_name << "'";

    // Reorder params from wrapper param order to inner kernel arg order.
    INTERNAL_CHECK(info.inner_call != nullptr && info.inner_callee != nullptr)
        << "Internal error: no inner call found in AIV-only Group '" << group_name << "'";
    auto params = BuildWrapperReorderedParams(call, group_func, info.inner_call, bridge);
    int aiv_id = RegisterKernel(info.aiv_name, CoreType::VECTOR, params);

    BuildAivOnlyGroupDispatchPlan(call, launch_func, group_name, aiv_func, aiv_id, std::move(params),
                                  capture_plain_task_id)
        .Emit(*this);
    return;
  }

  INTERNAL_CHECK_SPAN(!info.aic_name.empty(), call->span_)
      << "Internal error: no AIC callee found in Group '" << group_name << "' body";
  INTERNAL_CHECK_SPAN(!info.aiv_name.empty(), call->span_)
      << "Internal error: no AIV callee found in Group '" << group_name << "' body";

  FunctionPtr aic_func = program_->GetFunction(info.aic_name);
  FunctionPtr aiv_func = program_->GetFunction(info.aiv_name);
  INTERNAL_CHECK_SPAN(aic_func != nullptr, call->span_)
      << "Internal error: AIC function '" << info.aic_name << "' not found for Group '" << group_name << "'";
  INTERNAL_CHECK_SPAN(aiv_func != nullptr, call->span_)
      << "Internal error: AIV function '" << info.aiv_name << "' not found for Group '" << group_name << "'";

  // Reorder params from wrapper param order to inner kernel arg order.
  INTERNAL_CHECK(info.inner_call != nullptr && info.inner_callee != nullptr)
      << "Internal error: no inner call found in MixedKernels Group '" << group_name << "'";
  auto params = BuildWrapperReorderedParams(call, group_func, info.inner_call, bridge);
  int aic_id = RegisterKernel(info.aic_name, CoreType::CUBE, params);
  int aiv_id = RegisterKernel(info.aiv_name, CoreType::VECTOR, params);

  BuildMixedGroupDispatchPlan(call, launch_func, group_name, aic_id, aiv_id, aiv_func, std::move(params),
                              capture_plain_task_id)
      .Emit(*this);
}

}  // namespace codegen
}  // namespace pypto
