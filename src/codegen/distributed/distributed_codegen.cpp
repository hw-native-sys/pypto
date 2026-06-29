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

#include "pypto/codegen/distributed/distributed_codegen.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/distributed/distributed_op_registry.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

// Handle naming for emitted `with orch.allocate_domain(...)` blocks. The
// MaterializeCommDomainScopes pass sets each scope's name_hint_ to
// "comm_d<n>" (n = declaration order); codegen prepends "__" to derive the
// Python handle var (`__comm_d<n>`). The handle var and the simpler-side
// ``name=`` argument differ by the leading "__" so the Python identifier
// is name-mangling-friendly while ``name=`` stays a stable runtime tag.
namespace {
constexpr const char kCommDomainHandlePrefix[] = "__";

std::string HandleVarForScope(const ir::CommDomainScopeStmtPtr& scope) {
  return std::string(kCommDomainHandlePrefix) + scope->name_hint_;
}

std::string BuiltinEntrySymbol(const std::string& variant) {
  std::string result = variant;
  for (auto& c : result) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      c = '_';
    }
  }
  return result;
}
}  // namespace

// ========================================================================
// Public API
// ========================================================================

std::string DistributedCodegen::Generate(const ir::ProgramPtr& program) {
  CHECK(program != nullptr) << "Cannot generate code for null program";

  program_ = program;
  emitter_.Clear();
  workers_.clear();
  orchestrators_.clear();
  entry_func_ = nullptr;
  all_funcs_.clear();
  used_levels_.clear();
  hoisted_allocs_.clear();
  host_orch_body_after_hoist_ = false;
  host_orch_var_defs_.clear();
  unwrap_hoisted_var_refs_ = false;
  tuple_element_tensors_.clear();
  emitted_builtin_variants_.clear();
  builtin_next_level_specs_.clear();

  comm_domain_stack_.clear();

  ClassifyFunctions();
  CHECK(!workers_.empty() || !orchestrators_.empty())
      << "Program has no distributed functions (no functions with level/role metadata)";

  EmitImports();
  emitter_.EmitLine("");

  // Emit the highest-level orchestrator as the entry function.
  // In an L3 program, this is the HOST Orchestrator.
  if (!orchestrators_.empty()) {
    // Use the orchestrator with the highest level as the entry
    ir::FunctionPtr best_orch = orchestrators_.begin()->second;
    int best_level = 0;
    for (const auto& [name, func] : orchestrators_) {
      if (func->level_.has_value()) {
        int level = ir::LevelToLinquLevel(*func->level_);
        if (level > best_level) {
          best_level = level;
          best_orch = func;
        }
      }
    }
    // HOST-or-above orchestrator (Linqu level >= 3): split tensor.create
    // allocations into a pre-init `_alloc_intermediates(tensors)` function so
    // the simpler runtime can establish POSIX shared memory before fork.
    const bool is_host_orch =
        best_orch->level_.has_value() &&
        ir::LevelToLinquLevel(*best_orch->level_) >= 3;  // NOLINT(bugprone-unchecked-optional-access)
    if (is_host_orch) {
      CollectHostOrchHoistableAllocs(best_orch);
      EmitAllocIntermediatesFunction(best_orch);
      host_orch_body_after_hoist_ = true;
      EmitFunction(best_orch);
      host_orch_body_after_hoist_ = false;
      hoisted_allocs_.clear();
    } else {
      EmitFunction(best_orch);
    }
    EmitEntryMarker(best_orch->name_);
  } else if (entry_func_) {
    EmitEntryFunction();
    EmitEntryMarker("entry");
  }

  return emitter_.GetCode();
}

ir::CommDomainScopeStmtPtr DistributedCodegen::ScopeForWindowBuffer(const ir::WindowBufferPtr& wb) const {
  // Inner-to-outer scan: a WindowBuffer is always carved out of exactly one
  // scope's window, but if (future work) the same WindowBuffer were ever
  // shared across nested scopes the innermost wins (which matches the
  // codegen handle the dispatch is lexically inside).
  for (auto it = comm_domain_stack_.rbegin(); it != comm_domain_stack_.rend(); ++it) {
    for (const auto& slot : (*it)->slots_) {
      if (slot.get() == wb.get()) return *it;
    }
  }
  INTERNAL_CHECK(false) << "WindowBuffer '" << wb->name_hint_
                        << "' is not a slot of any open CommDomainScopeStmt "
                           "(MaterializeCommDomainScopes pass must run before DistributedCodegen, and the "
                           "dispatch must lexically sit inside its enclosing comm-domain scope)";
  return nullptr;  // unreachable
}

bool DistributedCodegen::MarkBuiltinEmitted(const std::string& variant) {
  return emitted_builtin_variants_.insert(variant).second;
}

std::string DistributedCodegen::NextTaskArgsVar() { return "_ta_" + std::to_string(task_args_counter_++); }

void DistributedCodegen::RecordBuiltinNextLevel(const ir::CallPtr& call, const std::string& variant,
                                                std::map<std::string, std::string> template_vars) {
  INTERNAL_CHECK(call && call->op_) << "Internal error: builtin next-level record needs a valid Call";
  const auto& op_name = call->op_->name_;
  const auto& entry = ir::OpRegistry::GetInstance().GetEntry(op_name);
  const auto& template_dir = entry.GetTemplateDir();
  INTERNAL_CHECK_SPAN(template_dir.has_value(), call->span_)
      << "Internal error: builtin op '" << op_name << "' must declare template_dir";

  BuiltinNextLevelSpec spec;
  spec.op_name = op_name;
  spec.variant = variant;
  spec.entry_symbol = BuiltinEntrySymbol(variant);
  spec.template_dir = template_dir.value();  // NOLINT(bugprone-unchecked-optional-access)
  spec.template_vars = std::move(template_vars);
  builtin_next_level_specs_.push_back(std::move(spec));
}

std::string DistributedCodegen::GetCommDomainHandleVar(const ir::WindowBufferPtr& wb) const {
  return HandleVarForScope(ScopeForWindowBuffer(wb));
}

// ========================================================================
// Function classification
// ========================================================================

void DistributedCodegen::ClassifyFunctions() {
  for (const auto& [gvar, func] : program_->functions_) {
    all_funcs_[func->name_] = func;

    if (!func->level_.has_value() && !func->role_.has_value()) {
      // Functions without level/role: chip-level functions (Orchestration, InCore, etc.)
      // or a true entry function (Opaque with no level/role).
      // Only treat Opaque functions as entry; chip functions are ignored by distributed codegen.
      if (func->func_type_ == ir::FunctionType::Opaque) {
        entry_func_ = func;
      }
      continue;
    }

    if (func->role_.has_value() && *func->role_ == ir::Role::Orchestrator) {
      orchestrators_[func->name_] = func;
    } else {
      // Explicit Worker role or level-only (no role) — treat as worker
      workers_[func->name_] = func;
    }

    if (func->level_.has_value()) {
      used_levels_.insert(ir::LevelToLinquLevel(*func->level_));
    }
  }
}

// ========================================================================
// Topological sort: callees before callers
// ========================================================================

std::vector<ir::FunctionPtr> DistributedCodegen::SortFunctionsByRoleAndLevel() const {
  std::vector<ir::FunctionPtr> funcs;
  for (const auto& [name, func] : all_funcs_) {
    if (func != entry_func_) {
      funcs.push_back(func);
    }
  }

  std::sort(funcs.begin(), funcs.end(), [](const ir::FunctionPtr& a, const ir::FunctionPtr& b) {
    bool a_sub_worker = a->role_.has_value() && *a->role_ == ir::Role::SubWorker;
    bool b_sub_worker = b->role_.has_value() && *b->role_ == ir::Role::SubWorker;
    if (a_sub_worker != b_sub_worker) return a_sub_worker;

    int a_level = a->level_.has_value() ? ir::LevelToLinquLevel(*a->level_) : 0;
    int b_level = b->level_.has_value() ? ir::LevelToLinquLevel(*b->level_) : 0;
    if (a_level != b_level) return a_level < b_level;

    return a->name_ < b->name_;
  });

  return funcs;
}

// ========================================================================
// Code structure emission
// ========================================================================

void DistributedCodegen::EmitImports() {
  emitter_.EmitLine("import torch");
  // The unified strided ``Tensor`` + ``DataType`` are used by DistributedTensor
  // formal emission (host_orch wraps per-rank window-bound regions via
  // ``Tensor.make(..., child_memory=True)``).
  // ``CommBufferSpec`` is the spec list passed to ``orch.allocate_domain``
  // inside host_orch when the program declares at least one comm domain;
  // harmless to import for comm-less L3 programs.
  emitter_.EmitLine(
      "from simpler.task_interface import "
      "CallConfig, CommBufferSpec, DataType, TaskArgs, Tensor, TensorArgType");
  emitter_.EmitLine("from pypto.runtime.tensor_arg import make_tensor_arg");
  // ``_submit_chip`` wraps ``orch.submit_next_level`` to namespace per-rank DFX
  // ``output_prefix`` (``<base>/rank{worker}``); a no-op when DFX is off.
  emitter_.EmitLine("from pypto.runtime.distributed_runner import _submit_chip");
}

void DistributedCodegen::EmitFunction(const ir::FunctionPtr& func) {
  declared_vars_.clear();
  task_args_counter_ = 0;
  current_func_ = func;

  bool is_sub_worker = func->role_.has_value() && *func->role_ == ir::Role::SubWorker;
  is_worker_context_ = is_sub_worker;

  // Build function signature
  // Orchestrators: def func(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size):
  // SubWorkers are not emitted as Python functions (they run on device or as registered callables)
  if (is_sub_worker) {
    is_worker_context_ = false;
    return;
  }

  // ``world_size`` is always present in the signature; the runner fills it
  // with ``len(DistributedConfig.device_ids)``. ``pld.system.world_size()``
  // lowers to a bare reference to this kwarg.
  std::ostringstream sig;
  sig << "def " << func->name_ << "(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size):";
  emitter_.EmitLine(sig.str());
  emitter_.IncreaseIndent();

  // Register parameter names and emit local bindings for scalar params.
  // All orchestrator parameters live in the tensors dict; tensor params are
  // referenced via tensors["name"] at call sites, but scalar params (e.g.
  // pl.Scalar[pl.BOOL]) may appear in bare-name contexts such as ``if``
  // conditions.  Emitting ``name = tensors["name"]`` at the top of the
  // function body ensures the bare name resolves correctly.
  RegisterParamsAndEmitScalarBindings(func);

  // For HOST orchestrators, pre-collect AssignStmt defs so the upcoming
  // ``VisitStmt_(CommDomainScopeStmtPtr)`` can unwrap CSE / SSA-hoisted
  // scalar temps (e.g. ``t = pld.system.world_size()``) when lowering each
  // slot's ``size_`` expression — the ``with orch.allocate_domain(...)``
  // line is emitted before the rest of the body, so those temps are not
  // yet bound in the emitted Python at that point. Chip-level orchestrators
  // don't own comm allocations and skip this step.
  if (func->level_.has_value() && ir::LevelToLinquLevel(*func->level_) >= 3) {
    CollectHostOrchVarDefs(func);
    // Recover pl.dynamic() shape dims into local bindings before the body walk
    // so per-rank host slices that reference a dynamic dim resolve at runtime
    // instead of raising NameError (#1873). Bindings are emitted at the top of
    // the function body (ahead of the comm-domain ``with`` / for-loop), so they
    // are visible everywhere below — Python ``with`` / ``for`` do not open a
    // new scope.
    EmitHostOrchDynamicDimBindings(func);
  }

  // Emit body. The host_orch body is wrapped by MaterializeCommDomainScopes
  // in a chain of CommDomainScopeStmts (one per comm domain); the override
  // of VisitStmt_(CommDomainScopeStmtPtr) below emits the
  // `with orch.allocate_domain(...)` block and pushes the scope onto
  // ``comm_domain_stack_`` so EmitCallToWorker can route each
  // DistributedTensor arg to the matching ``__comm_d<n>`` handle.
  if (func->body_) {
    VisitStmt(func->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("");
  is_worker_context_ = false;
}

namespace {

class HostOrchVarDefCollector : public ir::IRVisitor {
 public:
  explicit HostOrchVarDefCollector(std::unordered_map<const ir::Var*, ir::ExprPtr>& defs) : defs_(defs) {}

  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    if (auto var = ir::As<ir::Var>(op->var_)) {
      defs_[var.get()] = op->value_;
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::unordered_map<const ir::Var*, ir::ExprPtr>& defs_;
};

}  // namespace

void DistributedCodegen::CollectHostOrchVarDefs(const ir::FunctionPtr& func) {
  host_orch_var_defs_.clear();
  if (!func->body_) return;
  HostOrchVarDefCollector collector(host_orch_var_defs_);
  collector.VisitStmt(func->body_);
}

namespace {

// Extract a constant int from a tensor-shape sub-expression, or nullopt.
std::optional<int64_t> ConstIntFromShapeExpr(const ir::ExprPtr& expr) {
  if (auto ci = ir::As<ir::ConstInt>(expr)) return ci->value_;
  return std::nullopt;
}

// True iff ``v`` is the Var ``target``. Raw pointer identity is sound: the IR
// holds the canonical shared_ptr graph, so each Var has exactly one address
// (matching ``CollectVarsFromShapeExpr``'s own ``Var*`` dedup).
bool IsTargetVar(const ir::ExprPtr& v, const ir::VarPtr& target) {
  auto var = ir::As<ir::Var>(v);
  return var && var.get() == target.get();
}

}  // namespace

std::string DistributedCodegen::InvertShapeDimForVar(const ir::ExprPtr& dim_expr,
                                                     const ir::VarPtr& target_var,
                                                     const std::string& shape_access) const {
  // Bare var: shape == var  ->  var = shape_access.
  if (IsTargetVar(dim_expr, target_var)) return shape_access;

  // Add: shape == var + c (either operand order)  ->  var = (shape - c).
  if (auto add = ir::As<ir::Add>(dim_expr)) {
    if (IsTargetVar(add->left_, target_var)) {
      if (auto c = ConstIntFromShapeExpr(add->right_)) {
        return "(" + shape_access + " - " + std::to_string(*c) + ")";
      }
    }
    if (IsTargetVar(add->right_, target_var)) {
      if (auto c = ConstIntFromShapeExpr(add->left_)) {
        return "(" + shape_access + " - " + std::to_string(*c) + ")";
      }
    }
    return "";
  }

  // Sub: shape == var - c  ->  var = (shape + c);
  //      shape == c - var  ->  var = (c - shape).
  if (auto sub = ir::As<ir::Sub>(dim_expr)) {
    if (IsTargetVar(sub->left_, target_var)) {
      if (auto c = ConstIntFromShapeExpr(sub->right_)) {
        return "(" + shape_access + " + " + std::to_string(*c) + ")";
      }
    }
    if (IsTargetVar(sub->right_, target_var)) {
      if (auto c = ConstIntFromShapeExpr(sub->left_)) {
        return "(" + std::to_string(*c) + " - " + shape_access + ")";
      }
    }
    return "";
  }

  // Mul is commutative: shape == var * c (either order)  ->  var = (shape // c).
  // Integer ``//`` (not ``/``) keeps the recovered dim int-typed for slice
  // bounds. ``c == 0`` is rejected (non-invertible).
  if (auto mul = ir::As<ir::Mul>(dim_expr)) {
    if (IsTargetVar(mul->left_, target_var)) {
      if (auto c = ConstIntFromShapeExpr(mul->right_); c && *c != 0) {
        return "(" + shape_access + " // " + std::to_string(*c) + ")";
      }
    }
    if (IsTargetVar(mul->right_, target_var)) {
      if (auto c = ConstIntFromShapeExpr(mul->left_); c && *c != 0) {
        return "(" + shape_access + " // " + std::to_string(*c) + ")";
      }
    }
    return "";
  }

  // FloorDiv is non-commutative: only shape == var // c inverts (to
  // var = shape * c). ``c // var`` is not uniquely invertible against a runtime
  // shape, so ``(c, var)`` is rejected.
  if (auto fdiv = ir::As<ir::FloorDiv>(dim_expr)) {
    if (IsTargetVar(fdiv->left_, target_var)) {
      if (auto c = ConstIntFromShapeExpr(fdiv->right_); c && *c != 0) {
        return "(" + shape_access + " * " + std::to_string(*c) + ")";
      }
    }
    return "";
  }

  return "";
}

void DistributedCodegen::EmitHostOrchDynamicDimBindings(const ir::FunctionPtr& func) {
  // Per-param/dim walk delegates to ``CollectVarsFromShapeExpr`` -- the same C++
  // helper that drives the trailing ``%argN: index`` params on the device
  // ``func.func`` signature -- so host-side dim recovery stays in lockstep with
  // the compiled kernel by construction.
  struct DimSource {
    ir::VarPtr var;
    std::string param_name;  // already SanitizeName-d
    int dim_idx;
    ir::ExprPtr expr;
  };

  // First-seen order + best (invertible-preferred) source per canonical Var*.
  std::vector<const ir::Var*> dyn_var_order;
  std::unordered_map<const ir::Var*, DimSource> dyn_var_best;

  for (const auto& param : func->params_) {
    auto tensor_type = ir::AsTensorTypeLike(param->GetType());
    if (!tensor_type) continue;
    const std::string param_name = SanitizeName(param->name_hint_);
    for (size_t dim_idx = 0; dim_idx < tensor_type->shape_.size(); ++dim_idx) {
      const auto& dim = tensor_type->shape_[dim_idx];
      for (const auto& dyn_var : CollectVarsFromShapeExpr(dim)) {
        const ir::Var* key = dyn_var.get();
        auto it = dyn_var_best.find(key);
        if (it == dyn_var_best.end()) {
          dyn_var_order.push_back(key);
          dyn_var_best.emplace(key, DimSource{dyn_var, param_name, static_cast<int>(dim_idx), dim});
          continue;
        }
        // Upgrade source if the previously-seen dim is non-invertible and this
        // one is, so a symbol first seen in a non-invertible form still gets
        // pinned to a recoverable shape.
        const bool prev_invertible = !InvertShapeDimForVar(it->second.expr, it->second.var, "S").empty();
        const bool cur_invertible = !InvertShapeDimForVar(dim, dyn_var, "S").empty();
        if (!prev_invertible && cur_invertible) {
          it->second = DimSource{dyn_var, param_name, static_cast<int>(dim_idx), dim};
        }
      }
    }
  }

  for (const ir::Var* key : dyn_var_order) {
    const DimSource& src = dyn_var_best.at(key);
    const std::string shape_access =
        "tensors[\"" + src.param_name + "\"].shape[" + std::to_string(src.dim_idx) + "]";
    const std::string value_expr = InvertShapeDimForVar(src.expr, src.var, shape_access);
    // Surface a non-invertible dynamic dim as a documented user-facing
    // limitation (mirrors the device-side ValueError), rather than silently
    // leaving the symbol unbound. ``src.var`` is guaranteed non-null here.
    CHECK_SPAN(!value_expr.empty(), src.var->span_)
        << "Cannot recover dynamic dimension '" << src.var->name_hint_
        << "' for host-orchestration codegen: it only appears inside non-invertible tensor-shape "
        << "expressions. Host-slice dynamic dims must be recoverable from a runtime tensor shape via "
        << "'var' or a single-var affine form (var +/-/*// const_int); at least one tensor parameter "
        << "must expose the variable in one of these forms.";
    emitter_.EmitLine(SanitizeName(src.var->name_hint_) + " = " + value_expr);
  }
}

std::string DistributedCodegen::GetCommSlotSizeAsCode(const ir::ExprPtr& size_expr) {
  unwrap_hoisted_var_refs_ = true;
  std::string result = GetExprAsCode(size_expr);
  unwrap_hoisted_var_refs_ = false;
  return result;
}

void DistributedCodegen::VisitStmt_(const ir::CommDomainScopeStmtPtr& op) {
  // Pass invariant: MaterializeCommDomainScopes only wraps a body when at
  // least one alloc with at least one window-materialisation and at least
  // one consuming dispatch survives clustering — so every emitted scope
  // carries >= 1 slot. An empty `slots_` would also produce
  // `window_size=0`, which the simpler runtime would reject anyway.
  INTERNAL_CHECK_SPAN(!op->slots_.empty(), op->span_)
      << "CommDomainScopeStmt '" << op->name_hint_
      << "' has no slots — MaterializeCommDomainScopes must not emit empty scopes";

  const std::string handle_var = HandleVarForScope(op);
  const std::string& domain_name = op->name_hint_;

  // workers: literal list of worker indices into DistributedConfig.device_ids.
  // Empty devices_ in the IR means "all" — resolved at runtime via world_size.
  std::ostringstream workers;
  workers << "[";
  if (op->devices_.empty()) {
    workers << "*range(world_size)";
  } else {
    for (size_t i = 0; i < op->devices_.size(); ++i) {
      if (i > 0) workers << ", ";
      workers << op->devices_[i];
    }
  }
  workers << "]";

  // Lower each slot's size_ expression to a Python string. Constant sizes
  // become int literals; dynamic sizes (e.g. `pld.world_size() * 4`) lower
  // to the Python equivalent (e.g. `(world_size * 4)`) — `world_size` is
  // bound at the host_orch signature. ``GetCommSlotSizeAsCode`` enables
  // ``unwrap_hoisted_var_refs_`` so any CSE/SSA temp (e.g.
  // ``t = pld.system.world_size(); ... alloc_window_buffer(t * 4, ...)``)
  // unwraps to ``world_size * 4`` — the temp isn't yet bound in the
  // emitted Python at the point we write ``window_size=`` (it comes
  // ahead of the rest of the body).
  std::vector<std::string> slot_nbytes;
  slot_nbytes.reserve(op->slots_.size());
  for (const auto& slot : op->slots_) {
    slot_nbytes.push_back(GetCommSlotSizeAsCode(slot->size_));
  }

  // window_size = sum of all slot byte expressions. Parenthesise each summand
  // to keep operator precedence safe under any sub-expression shape.
  std::ostringstream window_size_expr;
  for (size_t i = 0; i < slot_nbytes.size(); ++i) {
    if (i > 0) window_size_expr << " + ";
    window_size_expr << "(" << slot_nbytes[i] << ")";
  }

  emitter_.EmitLine("with orch.allocate_domain(");
  emitter_.IncreaseIndent();
  emitter_.EmitLine(std::string("name=\"") + domain_name + "\",");
  emitter_.EmitLine("workers=" + workers.str() + ",");
  emitter_.EmitLine("window_size=" + window_size_expr.str() + ",");
  emitter_.EmitLine("buffers=[");
  emitter_.IncreaseIndent();
  for (size_t i = 0; i < op->slots_.size(); ++i) {
    const auto& slot = op->slots_[i];
    const std::string& nbytes = slot_nbytes[i];
    // dtype="opaque" mirrors the manifest-era placeholder: WindowBuffer is
    // intentionally dtype-agnostic (the field is unused by simpler). count is
    // also in opaque bytes so it shares the same expression as nbytes.
    emitter_.EmitLine(std::string("CommBufferSpec(name=\"") + SanitizeName(slot->name_hint_) +
                      "\", dtype=\"opaque\", count=" + nbytes + ", nbytes=" + nbytes + "),");
  }
  emitter_.DecreaseIndent();
  emitter_.EmitLine("],");
  emitter_.DecreaseIndent();
  emitter_.EmitLine(") as " + handle_var + ":");
  emitter_.IncreaseIndent();

  // Push this scope so EmitCallToWorker can resolve DistributedTensor args
  // inside the body to the correct handle var. Pop on exit so a sibling
  // scope's body never sees this one.
  comm_domain_stack_.push_back(op);
  if (op->body_) {
    VisitStmt(op->body_);
  }
  comm_domain_stack_.pop_back();

  emitter_.DecreaseIndent();
}

void DistributedCodegen::EmitEntryFunction() {
  if (!entry_func_) return;

  declared_vars_.clear();
  task_args_counter_ = 0;
  current_func_ = entry_func_;

  // Entry function signature
  emitter_.EmitLine("def entry(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size):");
  emitter_.IncreaseIndent();

  // Register parameter names and emit local bindings for scalar params.
  RegisterParamsAndEmitScalarBindings(entry_func_);

  // Emit body
  if (entry_func_->body_) {
    VisitStmt(entry_func_->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("");
}

void DistributedCodegen::EmitEntryMarker(const std::string& func_name) {
  // The runtime (distributed_runner._load_orch_entry) selects the unique
  // module-level function carrying this sentinel attribute as the dispatch
  // entry. This decouples entry resolution from the user's Python function
  // name, so renaming the @pl.jit.host orchestrator no longer breaks dispatch
  // (issue #1678). Keep the attribute name in sync with `_ENTRY_MARKER` in
  // python/pypto/runtime/distributed_runner.py.
  emitter_.EmitLine(func_name + "._pypto_distributed_entry = True");
  emitter_.EmitLine("");
}

void DistributedCodegen::RegisterParamsAndEmitScalarBindings(const ir::FunctionPtr& func) {
  for (const auto& param : func->params_) {
    std::string name = SanitizeName(param->name_hint_);
    declared_vars_.insert(name);
    if (ir::As<ir::ScalarType>(param->GetType())) {
      emitter_.EmitLine(name + " = tensors[\"" + name + "\"]");
    }
  }
}

// ========================================================================
// Statement visitors
// ========================================================================

bool DistributedCodegen::TryEmitHierarchyCall(const ir::ExprPtr& expr) {
  auto call = std::dynamic_pointer_cast<const ir::Call>(expr);
  if (!call) return false;
  auto gv = std::dynamic_pointer_cast<const ir::GlobalVar>(call->op_);
  if (!gv) return false;
  auto callee = program_->GetFunction(gv->name_);
  if (!callee) return false;

  INTERNAL_CHECK(callee->level_.has_value() && callee->role_.has_value() &&
                 current_func_->level_.has_value() && current_func_->role_.has_value());

  // INTERNAL_CHECK above guarantees the optionals hold values; clang-tidy
  // cannot see through the macro, so suppress its false positives here.
  const ir::Level callee_level = callee->level_.value();  // NOLINT(bugprone-unchecked-optional-access)
  const ir::Role callee_role = callee->role_.value();     // NOLINT(bugprone-unchecked-optional-access)
  const ir::Level current_level =
      current_func_->level_.value();  // NOLINT(bugprone-unchecked-optional-access)

  const bool same_level_sub_worker = callee_role == ir::Role::SubWorker && callee_level == current_level;
  const bool next_level_orch = callee_role == ir::Role::Orchestrator &&
                               static_cast<int>(callee_level) == static_cast<int>(current_level) - 1;

  if (same_level_sub_worker || next_level_orch) {
    EmitCallToWorker(call, callee);
    return true;
  }

  UNREACHABLE << ir::LevelToString(current_level) << "Level Orch func '" << current_func_->name_
              << "' can only call same level sub-worker or next level Orch, but call to func '" << gv->name_
              << "' has level = '" << ir::LevelToString(callee_level) << "', role = '"
              << ir::RoleToString(callee_role) << "'.";
  return false;  // unreachable
}

void DistributedCodegen::VisitStmt_(const ir::AssignStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null AssignStmt";

  std::string var_name = SanitizeName(op->var_->name_hint_);

  // If this AssignStmt was hoisted to _alloc_intermediates(tensors), the
  // value has already been emitted in the alloc function. Register the SSA
  // name for downstream tensors[...] references but emit nothing here.
  if (host_orch_body_after_hoist_ && hoisted_allocs_.count(op.get())) {
    declared_vars_.insert(var_name);
    return;
  }

  // Handle TupleGetItemExpr: `out_rms = tuple_tmp[i]` produced by multi-return
  // function call unpacking.  Resolve each element to its actual Out/InOut
  // parameter tensor recorded in tuple_element_tensors_ during EmitCallToWorker.
  if (auto tge = std::dynamic_pointer_cast<const ir::TupleGetItemExpr>(op->value_)) {
    INTERNAL_CHECK_SPAN(ir::AsVarLike(tge->tuple_) != nullptr, op->span_)
        << "Internal error: TupleGetItemExpr tuple_ must be a Var-like expression "
        << "for distributed codegen tuple-return unpacking";
    VisitExpr(tge->tuple_);
    std::string tuple_var = current_expr_value_;
    current_expr_value_ = "";
    auto it = tuple_element_tensors_.find(std::make_pair(tuple_var, tge->index_));
    INTERNAL_CHECK_SPAN(it != tuple_element_tensors_.end(), op->span_)
        << "Internal error: TupleGetItemExpr unpacking found no Out parameter "
        << "for tuple var '" << tuple_var << "' index " << tge->index_;
    emitter_.EmitLine("tensors[\"" + var_name + "\"] = tensors[\"" + it->second + "\"]");
    declared_vars_.insert(var_name);
    current_target_var_ = "";
    return;
  }

  const auto value_call = ir::As<ir::Call>(op->value_);
  const bool distributed_tensor_alias = ir::As<ir::DistributedTensorType>(op->var_->GetType()) &&
                                        ir::As<ir::DistributedTensorType>(op->value_->GetType()) &&
                                        (!value_call || ir::IsOp(value_call, "pld.tensor.window"));
  if (distributed_tensor_alias) {
    declared_vars_.insert(var_name);
    current_target_var_ = "";
    current_expr_value_ = "";
    return;
  }

  current_target_var_ = var_name;
  current_expr_value_ = "";

  // Check if the value is a Call to a hierarchy function or chip function
  if (TryEmitHierarchyCall(op->value_)) {
    declared_vars_.insert(var_name);
    current_target_var_ = "";
    return;
  }

  // Standard expression
  VisitExpr(op->value_);

  if (!current_expr_value_.empty()) {
    emitter_.EmitLine(var_name + " = " + current_expr_value_);
    declared_vars_.insert(var_name);
    current_expr_value_ = "";
  }

  current_target_var_ = "";
}

void DistributedCodegen::VisitStmt_(const ir::EvalStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";

  current_target_var_ = "";
  current_expr_value_ = "";

  // Check if the value is a Call to a hierarchy function or chip function
  if (TryEmitHierarchyCall(op->expr_)) {
    return;
  }

  // Standard expression
  VisitExpr(op->expr_);

  if (!current_expr_value_.empty()) {
    emitter_.EmitLine(current_expr_value_);
    current_expr_value_ = "";
  }
}

void DistributedCodegen::VisitStmt_(const ir::ReturnStmtPtr& /* op */) {
  // L3 orchestrator functions return via output tensor side effects.
  // No Python return statement is generated.
}

void DistributedCodegen::VisitStmt_(const ir::ForStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ForStmt";

  std::string loop_var = SanitizeName(op->loop_var_->name_hint_);
  declared_vars_.insert(loop_var);

  VisitExpr(op->start_);
  std::string start = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->stop_);
  std::string stop = current_expr_value_;
  current_expr_value_ = "";

  VisitExpr(op->step_);
  std::string step = current_expr_value_;
  current_expr_value_ = "";

  emitter_.EmitLine("for " + loop_var + " in range(" + start + ", " + stop + ", " + step + "):");
  emitter_.IncreaseIndent();

  if (op->body_) {
    VisitStmt(op->body_);
  } else {
    emitter_.EmitLine("pass");
  }

  emitter_.DecreaseIndent();
}

void DistributedCodegen::VisitStmt_(const ir::IfStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";

  VisitExpr(op->condition_);
  std::string condition = current_expr_value_;
  current_expr_value_ = "";

  emitter_.EmitLine("if " + condition + ":");
  emitter_.IncreaseIndent();
  VisitStmt(op->then_body_);
  emitter_.DecreaseIndent();

  if (op->else_body_.has_value()) {
    emitter_.EmitLine("else:");
    emitter_.IncreaseIndent();
    VisitStmt(*op->else_body_);
    emitter_.DecreaseIndent();
  }
}

void DistributedCodegen::VisitStmt_(const ir::SeqStmtsPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null SeqStmts";
  for (const auto& stmt : op->stmts_) {
    VisitStmt(stmt);
  }
}

// ========================================================================
// Expression visitors
// ========================================================================

void DistributedCodegen::VisitExpr_(const ir::CallPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Call";

  // Check if callee is a GlobalVar (program function reference)
  if (auto gv = std::dynamic_pointer_cast<const ir::GlobalVar>(op->op_)) {
    auto callee = program_->GetFunction(gv->name_);
    if (callee) {
      if (callee->role_.has_value() && *callee->role_ == ir::Role::SubWorker) {
        EmitCallToWorker(op, callee);
        return;
      }
      if (callee->role_.has_value() && *callee->role_ == ir::Role::Orchestrator) {
        // Orchestrator-to-orchestrator calls: emit as direct function call
        current_expr_value_ =
            callee->name_ +
            "(orch, _args, config, "
            "tensors=tensors, callables=callables, sub_ids=sub_ids, _keep=_keep, world_size=world_size)";
        return;
      }
      // Chip-level function (Orchestration/InCore with no role) called from HOST orchestrator
      // → treat as submit_next_level (chip dispatch)
      if (callee->func_type_ == ir::FunctionType::Orchestration ||
          callee->func_type_ == ir::FunctionType::InCore) {
        EmitCallToWorker(op, callee);
        return;
      }
    }
    // Regular function call
    current_expr_value_ = gv->name_ + "(" + FormatArgs(op->args_) + ")";
    return;
  }

  // dist.* intrinsic ops
  if (op->op_->name_.rfind("dist.", 0) == 0) {
    EmitDistIntrinsic(op);
    return;
  }

  // tensor.create → orch.alloc() for HOST-level orchestrators
  if (ir::IsOp(op, "tensor.create")) {
    EmitTensorCreate(op);
    return;
  }

  // ``pld.system.world_size()`` lowers to the ``world_size`` kwarg bound in
  // every emitted orchestrator's signature (see EmitFunction / EmitEntryFunction).
  // The runner fills it with len(DistributedConfig.device_ids) — present for
  // comm-less programs too, so this lowering is uniform.
  if (ir::IsOp(op, "pld.system.world_size")) {
    current_expr_value_ = "world_size";
    return;
  }

  // Per-op host_orch codegen registry (mirror of OrchestrationOpRegistry /
  // PTO backend codegen). Handlers that need to emit ``tensors["lhs"] = ...``
  // or skip emission entirely (window-buffer markers) register here and
  // return either the RHS Python expression or the empty string to signal
  // "already emitted, drop the wrapping AssignStmt line".
  auto registered = DistributedOpRegistry::GetInstance().Get(op->op_->name_);
  if (registered) {
    current_expr_value_ = (*registered)(op, *this);
    return;
  }

  // Regular op call
  current_expr_value_ = op->op_->name_ + "(" + FormatArgs(op->args_) + ")";
}

void DistributedCodegen::VisitExpr_(const ir::VarPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Var";
  if (unwrap_hoisted_var_refs_) {
    const ir::Var* cur = op.get();
    std::unordered_set<const ir::Var*> visited;
    while (true) {
      auto it = host_orch_var_defs_.find(cur);
      if (it == host_orch_var_defs_.end() || !it->second) {
        break;
      }
      if (!visited.insert(cur).second) {
        break;
      }
      if (auto next_var = ir::As<ir::Var>(it->second)) {
        cur = next_var.get();
        continue;
      }
      VisitExpr(it->second);
      return;
    }
  }
  current_expr_value_ = SanitizeName(op->name_hint_);
}

void DistributedCodegen::VisitExpr_(const ir::ConstIntPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstInt";
  current_expr_value_ = std::to_string(op->value_);
}

void DistributedCodegen::VisitExpr_(const ir::ConstFloatPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstFloat";
  current_expr_value_ = std::to_string(op->value_);
}

void DistributedCodegen::VisitExpr_(const ir::ConstBoolPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ConstBool";
  current_expr_value_ = op->value_ ? "True" : "False";
}

// ========================================================================
// Call-site lowering
// ========================================================================

void DistributedCodegen::EmitCallToWorker(const ir::CallPtr& call, const ir::FunctionPtr& callee) {
  bool is_sub = IsSubWorker(callee);
  std::string ta_var = "_ta_" + std::to_string(task_args_counter_++);

  // ``device=`` attr (set by N3 parser) is the single source of truth for
  // both the per-rank ``__comm_d0[<r>]`` subscript (used by DistributedTensor
  // arg emit) and the trailing ``worker=<r>`` kwarg on ``submit_next_level``.
  // Empty string means no ``device=`` was set — comm-less L3 dispatch path.
  std::string rank_expr = ResolveRankExpr(call);

  // Build TaskArgs from callee's parameter directions
  emitter_.EmitLine(ta_var + " = TaskArgs()");

  for (size_t i = 0; i < call->args_.size(); ++i) {
    VisitExpr(call->args_[i]);
    std::string arg_str = current_expr_value_;
    current_expr_value_ = "";

    // N7: DistributedTensorType formals route through Tensor.make
    // with ``child_memory=True``. ``As<DistributedTensorType>`` is strict
    // ObjectKind match, so this branch fires only for DistributedTensor —
    // plain TensorType falls through to the existing make_tensor_arg path.
    if (auto dist_type = ir::As<ir::DistributedTensorType>(call->args_[i]->GetType())) {
      INTERNAL_CHECK_SPAN(!rank_expr.empty(), call->span_)
          << "Call passing DistributedTensor args must carry device= attr "
             "(N3 parser writes attrs[\"device\"] on chip-orch dispatch sites)";
      INTERNAL_CHECK_SPAN(dist_type->window_buffer_.has_value(), call->span_)
          << "DistributedTensorType arg must have window_buffer_ populated by N4 pass";
      const auto& window_buffer = dist_type->window_buffer_.value();
      const std::string name = SanitizeName(window_buffer->name_hint_);
      const std::string shape = FormatShapeTuple(dist_type->shape_);
      const std::string dtype_enum = "DataType." + DataTypeToSimplerEnum(dist_type->dtype_);
      std::string tag = "TensorArgType.INOUT";
      if (i < callee->param_directions_.size()) {
        tag = ParamDirectionToTensorArgType(callee->param_directions_[i]);
      }
      const std::string handle_var = HandleVarForScope(ScopeForWindowBuffer(window_buffer));
      emitter_.EmitLine(ta_var + ".add_tensor(Tensor.make(data=" + handle_var + "[" + rank_expr +
                        "].buffer_ptrs[\"" + name + "\"], shapes=" + shape + ", dtype=" + dtype_enum +
                        ", child_memory=True), " + tag + ")");
      continue;
    }

    // Plain TensorType formal — existing path.
    if (std::dynamic_pointer_cast<const ir::TensorType>(call->args_[i]->GetType())) {
      std::string tag = "TensorArgType.INPUT";
      if (i < callee->param_directions_.size()) {
        tag = ParamDirectionToTensorArgType(callee->param_directions_[i]);
      }
      emitter_.EmitLine(ta_var + ".add_tensor(make_tensor_arg(tensors[\"" + arg_str + "\"]), " + tag + ")");
      continue;
    }

    // ScalarType formal — pass-through via ``add_scalar``. Emitted in the
    // call's IR-argument position so the runtime TaskArgs layout matches
    // the callee's parameter list one-for-one (the trailing CommContext
    // pointers appended below for each DistributedTensor formal are
    // synthetic — added by the N7 kernel-signature transform and do not
    // appear in the user-visible signature).
    if (ir::As<ir::ScalarType>(call->args_[i]->GetType())) {
      emitter_.EmitLine(ta_var + ".add_scalar(" + arg_str + ")");
      continue;
    }

    INTERNAL_CHECK_SPAN(false, call->span_) << "EmitCallToWorker: unsupported call arg type at index " << i
                                            << ": " << call->args_[i]->GetType()->TypeName();
  }

  // After all add_tensor lines, append one
  // ``add_scalar(__comm_d<group_idx>[<r>].device_ctx)`` per DistributedTensor
  // arg, in IR-arg order — matches the N6 incore PTO signature's trailing
  // ctx-ptr segment. The group lookup uses the same WindowBuffer-identity
  // map as the add_tensor branch above so two DistributedTensors from
  // different CommGroups route to their respective handles.
  for (const auto& arg : call->args_) {
    auto dist_type = ir::As<ir::DistributedTensorType>(arg->GetType());
    if (!dist_type) continue;
    INTERNAL_CHECK_SPAN(dist_type->window_buffer_.has_value(), call->span_)
        << "DistributedTensorType arg must have window_buffer_ populated by N4 pass";
    const std::string device_ctx_handle =
        HandleVarForScope(ScopeForWindowBuffer(dist_type->window_buffer_.value()));
    emitter_.EmitLine(ta_var + ".add_scalar(" + device_ctx_handle + "[" + rank_expr + "].device_ctx)");
  }

  // If this call has an assignment target (return value) but the callee already
  // has Out/InOut parameters, the output is already covered by those params.
  // Only add the target as OUTPUT_EXISTING if the callee has no explicit Out
  // params. In the no-Out-param branch ``tensors[target]`` may not yet exist
  // (it's only seeded from input parameter names in the runtime), so allocate
  // a fresh shared-memory tensor for it before emitting ``add_tensor``.
  std::string target = current_target_var_;
  if (!target.empty() && !callee->return_types_.empty()) {
    bool has_out_param = false;
    for (const auto& dir : callee->param_directions_) {
      if (dir == ir::ParamDirection::Out || dir == ir::ParamDirection::InOut) {
        has_out_param = true;
        break;
      }
    }
    if (!has_out_param) {
      // Allocate a shared-memory tensor for the return value if absent.
      // share_memory_() is required for fork-based distributed visibility.
      auto ret_type = std::dynamic_pointer_cast<const ir::TensorType>(callee->return_types_.front());
      INTERNAL_CHECK(ret_type) << "Distributed callee return must be TensorType";
      std::string shape = "(";
      for (size_t i = 0; i < ret_type->shape_.size(); ++i) {
        if (i > 0) shape += ", ";
        const auto& dim = ret_type->shape_[i];
        if (auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(dim)) {
          shape += std::to_string(const_int->value_);
        } else if (auto var = std::dynamic_pointer_cast<const ir::Var>(dim)) {
          shape += SanitizeName(var->name_hint_);
        } else {
          CHECK(false) << "Distributed callee return shape dim " << i << " must be ConstInt or Var";
        }
      }
      if (ret_type->shape_.size() == 1) shape += ",";
      shape += ")";
      const std::string torch_dtype = DataTypeToPythonDType(ret_type->dtype_);
      emitter_.EmitLine("if \"" + target + "\" not in tensors:");
      emitter_.EmitLine("    tensors[\"" + target + "\"] = torch.zeros(" + shape + ", dtype=torch." +
                        torch_dtype + ").share_memory_()");
      emitter_.EmitLine(ta_var + ".add_tensor(make_tensor_arg(tensors[\"" + target +
                        "\"]), TensorArgType.OUTPUT_EXISTING)");
    }
  }

  if (is_sub) {
    // HOST Worker = SubWorker: orch.submit_sub(callable_id, task_args)
    emitter_.EmitLine("orch.submit_sub(sub_ids[\"" + callee->name_ + "\"], " + ta_var + ")");
  } else {
    // CHIP Worker: orch.submit_next_level(callable, task_args, config).
    // N7: thread the dispatch ``device=`` attr (N3 parser) into the
    // simpler runtime's ``worker=`` kwarg (see simpler/python/simpler/
    // orchestrator.py — ``-1`` = unconstrained). Empty rank_expr ⇔ no
    // ``device=`` attr → omit the kwarg, byte-compatible with comm-less L3.
    emitter_.EmitLine("_keep.append(" + ta_var + ")");
    if (rank_expr.empty()) {
      emitter_.EmitLine("orch.submit_next_level(callables[\"" + callee->name_ + "\"], " + ta_var +
                        ", config)");
    } else {
      // Rank-pinned dispatch routes through ``_submit_chip`` so DFX artifacts
      // land in a per-rank subdir (``<output_prefix>/rank{r}``); a no-op
      // forward to ``submit_next_level`` when DFX is off.
      emitter_.EmitLine("_submit_chip(orch, callables[\"" + callee->name_ + "\"], " + ta_var + ", config, " +
                        rank_expr + ")");
    }
  }

  // If this call has an assignment target (return value), alias it to the OUT
  // parameter tensor.  In simpler's runtime model, the callee writes in-place
  // to the OUT parameter, so the return value is the same tensor.
  if (!target.empty() && !callee->return_types_.empty()) {
    // Tuple return is expressed in two ways in the IR:
    //   1. pl.Tuple[T1, T2] → return_types_ = [TupleType(T1, T2)]  (single entry)
    //   2. tuple[T1, T2]    → return_types_ = [T1, T2]             (flat entries)
    // Both produce TupleGetItemExpr for unpacking, so handle both.
    bool is_tuple_return =
        std::dynamic_pointer_cast<const ir::TupleType>(callee->return_types_.front()) != nullptr ||
        callee->return_types_.size() > 1;
    if (is_tuple_return) {
      // Tuple return: populate tuple_element_tensors_ so that downstream
      // TupleGetItemExpr AssignStmts resolve each element to its Out param.
      // Do NOT emit a tensors["target"] alias here — the individual
      // TupleGetItemExpr unpacking statements handle per-element aliasing.
      int out_idx = 0;
      for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
        if (callee->param_directions_[i] == ir::ParamDirection::Out ||
            callee->param_directions_[i] == ir::ParamDirection::InOut) {
          VisitExpr(call->args_[i]);
          std::string out_arg = current_expr_value_;
          current_expr_value_ = "";
          tuple_element_tensors_[std::make_pair(target, out_idx)] = out_arg;
          ++out_idx;
        }
      }
    } else {
      // Single return: alias target to the first Out/InOut parameter tensor.
      for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
        if (callee->param_directions_[i] == ir::ParamDirection::Out ||
            callee->param_directions_[i] == ir::ParamDirection::InOut) {
          VisitExpr(call->args_[i]);
          std::string out_arg = current_expr_value_;
          current_expr_value_ = "";
          emitter_.EmitLine("tensors[\"" + target + "\"] = tensors[\"" + out_arg + "\"]");
          break;
        }
      }
    }
  }
}

void DistributedCodegen::EmitDistIntrinsic(const ir::CallPtr& call) {
  const auto& op_name = call->op_->name_;

  if (ir::IsOp(call, "dist.tree_reduce")) {
    EmitTreeReduce(call);
    return;
  }

  current_expr_value_ = op_name + "(" + FormatArgs(call->args_) + ")";
}

void DistributedCodegen::EmitTreeReduce(const ir::CallPtr& call) {
  std::string target = current_target_var_;
  std::ostringstream args;
  for (size_t i = 0; i < call->args_.size(); ++i) {
    if (i > 0) args << ", ";
    VisitExpr(call->args_[i]);
    args << current_expr_value_;
    current_expr_value_ = "";
  }

  if (!target.empty()) {
    current_expr_value_ = "tree_reduce(" + args.str() + ")";
  } else {
    emitter_.EmitLine("tree_reduce(" + args.str() + ")");
  }
}

void DistributedCodegen::EmitTensorCreate(const ir::CallPtr& call) {
  auto result_type = std::dynamic_pointer_cast<const ir::TensorType>(call->GetType());
  INTERNAL_CHECK(result_type) << "tensor.create must return TensorType";

  std::string target = current_target_var_;
  INTERNAL_CHECK(!target.empty()) << "tensor.create must have an assignment target";

  std::string shape = "(";
  for (size_t i = 0; i < result_type->shape_.size(); ++i) {
    if (i > 0) shape += ", ";
    const auto& dim = result_type->shape_[i];
    if (auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(dim)) {
      shape += std::to_string(const_int->value_);
    } else if (auto var = std::dynamic_pointer_cast<const ir::Var>(dim)) {
      // Dynamic shape: reference the parameter name in the generated Python.
      shape += SanitizeName(var->name_hint_);
    } else {
      CHECK(false) << "tensor.create shape dim " << i
                   << " must be a ConstInt or Var (dynamic shape variable), "
                   << "got expression of unsupported kind";
    }
  }
  if (result_type->shape_.size() == 1) shape += ",";  // trailing comma for 1-tuple
  shape += ")";

  std::string torch_dtype = DataTypeToPythonDType(result_type->dtype_);

  // share_memory_() is required for fork-based distributed runtime visibility.
  emitter_.EmitLine("tensors[\"" + target + "\"] = torch.zeros(" + shape + ", dtype=torch." + torch_dtype +
                    ").share_memory_()");

  declared_vars_.insert(target);
  current_expr_value_ = "";
}

// ========================================================================
// HOST orchestrator alloc hoisting
// ========================================================================

namespace {

// Returns true iff @p stmt is an AssignStmt whose value is a Call to op
// `tensor.create`. Used to detect hoistable allocations.
bool IsTensorCreateAssign(const ir::StmtPtr& stmt) {
  auto assign = std::dynamic_pointer_cast<const ir::AssignStmt>(stmt);
  if (!assign) return false;
  auto call = std::dynamic_pointer_cast<const ir::Call>(assign->value_);
  if (!call || !call->op_) return false;
  return ir::IsOp(call, "tensor.create");
}

// Recursively reject any `tensor.create` reachable through @p stmt — pre-init
// hoisting requires unconditional, top-level allocations, so a tensor.create
// nested inside control flow (if/for/scope) is unsupported. @p container
// labels the enclosing construct for the error message.
void RejectNestedTensorCreate(const ir::StmtPtr& stmt, const std::string& container) {
  if (!stmt) return;
  CHECK(!IsTensorCreateAssign(stmt)) << "pl.create_tensor in HOST orchestrator must be a top-level "
                                        "statement (not nested inside "
                                     << container
                                     << "). Pre-init hoisting requires static, unconditional allocation.";
  if (auto if_stmt = std::dynamic_pointer_cast<const ir::IfStmt>(stmt)) {
    RejectNestedTensorCreate(if_stmt->then_body_, "if-then");
    if (if_stmt->else_body_.has_value()) {
      RejectNestedTensorCreate(*if_stmt->else_body_,  // NOLINT(bugprone-unchecked-optional-access)
                               "if-else");
    }
  } else if (auto for_stmt = std::dynamic_pointer_cast<const ir::ForStmt>(stmt)) {
    RejectNestedTensorCreate(for_stmt->body_, "for-loop");
  } else if (auto seq = std::dynamic_pointer_cast<const ir::SeqStmts>(stmt)) {
    for (const auto& s : seq->stmts_) RejectNestedTensorCreate(s, container);
  }
}

// Returns the immediate top-level statements of a function body. The HOST
// orchestrator body is a SeqStmts in practice; the single-stmt case is
// handled for safety.
std::vector<ir::StmtPtr> TopLevelStmts(const ir::StmtPtr& body) {
  if (auto seq = std::dynamic_pointer_cast<const ir::SeqStmts>(body)) {
    return {seq->stmts_.begin(), seq->stmts_.end()};
  }
  return {body};
}

// MaterializeCommDomainScopes wraps the host_orch body in a chain of
// ``CommDomainScopeStmt`` (one per inferred comm domain, outer = first
// declared). Alloc-hoisting needs to scan the original top-level
// statements, not the scope chain itself — so peel any leading
// CommDomainScopeStmts before delegating to TopLevelStmts. The chain is
// always linear (no SeqStmts between scopes by construction), so a simple
// while loop suffices.
ir::StmtPtr UnwrapLeadingCommDomainScopes(const ir::StmtPtr& body) {
  ir::StmtPtr cur = body;
  while (auto scope = std::dynamic_pointer_cast<const ir::CommDomainScopeStmt>(cur)) {
    cur = scope->body_;
  }
  return cur;
}

}  // namespace

void DistributedCodegen::CollectHostOrchHoistableAllocs(const ir::FunctionPtr& host_orch) {
  hoisted_allocs_.clear();
  if (!host_orch->body_) return;

  for (const auto& stmt : TopLevelStmts(UnwrapLeadingCommDomainScopes(host_orch->body_))) {
    if (IsTensorCreateAssign(stmt)) {
      auto assign = std::dynamic_pointer_cast<const ir::AssignStmt>(stmt);
      hoisted_allocs_.insert(assign.get());
    } else {
      // Any tensor.create reached from a non-top-level statement is an error.
      RejectNestedTensorCreate(stmt, "control flow");
    }
  }
}

void DistributedCodegen::EmitAllocIntermediatesFunction(const ir::FunctionPtr& host_orch) {
  emitter_.EmitLine("def _alloc_intermediates(tensors):");
  emitter_.IncreaseIndent();
  if (hoisted_allocs_.empty()) {
    emitter_.EmitLine("pass");
  } else {
    // Walk top-level statements in original order; emit only hoisted allocs.
    // Falls through the normal AssignStmt → Call → EmitTensorCreate path.
    // host_orch_body_after_hoist_ is FALSE here so EmitTensorCreate runs
    // unchanged. The subsequent EmitFunction(host_orch) call resets visitor
    // state (declared_vars_, current_func_, ...), so no save/restore needed.
    current_func_ = host_orch;
    for (const auto& stmt : TopLevelStmts(UnwrapLeadingCommDomainScopes(host_orch->body_))) {
      auto assign = std::dynamic_pointer_cast<const ir::AssignStmt>(stmt);
      if (assign && hoisted_allocs_.count(assign.get())) {
        VisitStmt(stmt);
      }
    }
  }
  emitter_.DecreaseIndent();
  emitter_.EmitLine("");
}

std::string DistributedCodegen::DataTypeToPythonDType(const DataType& dtype) {
  // Most dtype names match torch directly; only fp16/fp32/fp64 differ.
  static const std::unordered_map<std::string, std::string> kRenames = {
      {"fp16", "float16"},
      {"fp32", "float32"},
      {"fp64", "float64"},
  };
  std::string name = dtype.ToString();
  auto it = kRenames.find(name);
  CHECK(name != "unknown") << "Unsupported dtype for distributed tensor create: " << name;
  return it != kRenames.end() ? it->second : name;
}

std::string DistributedCodegen::DataTypeToSimplerEnum(const DataType& dtype) {
  // ``simpler.task_interface.DataType`` exposes the C-style enum names
  // (FLOAT16 / FLOAT32 / BFLOAT16 / INT* / UINT* / BOOL). Map PyPTO's
  // dtype tags to those names so emitted ``Tensor.make(..., dtype=DataType.<X>)``
  // matches at runtime.
  if (dtype == DataType::FP16) return "FLOAT16";
  if (dtype == DataType::FP32) return "FLOAT32";
  if (dtype == DataType::BF16) return "BFLOAT16";
  if (dtype == DataType::INT8) return "INT8";
  if (dtype == DataType::INT16) return "INT16";
  if (dtype == DataType::INT32) return "INT32";
  if (dtype == DataType::INT64) return "INT64";
  if (dtype == DataType::UINT8) return "UINT8";
  if (dtype == DataType::UINT16) return "UINT16";
  if (dtype == DataType::UINT32) return "UINT32";
  if (dtype == DataType::UINT64) return "UINT64";
  if (dtype == DataType::BOOL) return "BOOL";
  CHECK(false) << "Unsupported DistributedTensor dtype for simpler.DataType mapping: " << dtype.ToString();
  return "FLOAT32";
}

std::string DistributedCodegen::ResolveRankExpr(const ir::CallPtr& call) const {
  if (!call->HasAttr(ir::kAttrDevice)) return "";
  auto dev = call->GetAttr<ir::ExprPtr>(ir::kAttrDevice, nullptr);
  INTERNAL_CHECK_SPAN(dev != nullptr, call->span_) << "device= attr must hold a non-null ExprPtr";
  if (auto ci = std::dynamic_pointer_cast<const ir::ConstInt>(dev)) {
    CHECK(ci->value_ >= 0) << "device= ConstInt must be non-negative rank index, got " << ci->value_;
    return std::to_string(ci->value_);
  }
  if (auto v = std::dynamic_pointer_cast<const ir::Var>(dev)) {
    return SanitizeName(v->name_hint_);
  }
  INTERNAL_CHECK_SPAN(false, call->span_)
      << "device= attr must be ConstInt or Var (N3 parser invariant), got " << dev->TypeName();
  return "";
}

std::string DistributedCodegen::FormatShapeTuple(const std::vector<ir::ExprPtr>& shape) {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << GetExprAsCode(shape[i]);
  }
  // Trailing comma on rank-1 so the literal stays a tuple, not a parenthesised scalar.
  if (shape.size() == 1) oss << ",";
  oss << ")";
  return oss.str();
}

// ========================================================================
// Helpers
// ========================================================================

bool DistributedCodegen::IsSubWorker(const ir::FunctionPtr& func) const {
  // A SubWorker callable is specifically a HOST-or-above SubWorker role
  // (runs as Python callable in fork). HOST-or-above Orchestrators are
  // dispatched via ``callables[...]`` not ``sub_ids[...]``, so they must
  // NOT be classified as SubWorker callables. CHIP-level functions and
  // functions without level metadata run via ChipCallable →
  // submit_next_level.
  if (!func->level_.has_value() || !func->role_.has_value()) return false;
  if (*func->role_ != ir::Role::SubWorker) return false;
  return ir::LevelToLinquLevel(*func->level_) >= 3;
}

std::string DistributedCodegen::ParamDirectionToTensorArgType(ir::ParamDirection dir) const {
  switch (dir) {
    case ir::ParamDirection::In:
      return "TensorArgType.INPUT";
    case ir::ParamDirection::Out:
      return "TensorArgType.OUTPUT_EXISTING";
    case ir::ParamDirection::InOut:
      return "TensorArgType.INOUT";
  }
  return "TensorArgType.INPUT";
}

std::string DistributedCodegen::SanitizeName(const std::string& name) const {
  std::string result = name;
  for (auto& c : result) {
    if (c == '.') c = '_';
  }
  return result;
}

std::string DistributedCodegen::FormatArgs(const std::vector<ir::ExprPtr>& args) {
  std::ostringstream oss;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0) oss << ", ";
    VisitExpr(args[i]);
    oss << current_expr_value_;
    current_expr_value_ = "";
  }
  return oss.str();
}

// ========================================================================
// CodegenBase interface implementation
// ========================================================================

void DistributedCodegen::Emit(const std::string& line) { emitter_.EmitLine(line); }

std::string DistributedCodegen::GetExprAsCode(const ir::ExprPtr& expr) {
  VisitExpr(expr);
  std::string result = current_expr_value_;
  current_expr_value_ = "";
  return result;
}

std::string DistributedCodegen::GetTypeString(const DataType& dtype) const { return dtype.ToCTypeString(); }

int64_t DistributedCodegen::GetConstIntValue(const ir::ExprPtr& expr) const {
  auto const_int = std::dynamic_pointer_cast<const ir::ConstInt>(expr);
  CHECK(const_int != nullptr) << "Expected constant integer expression";
  return const_int->value_;
}

std::string DistributedCodegen::GetVarName(const ir::VarPtr& var) const {
  return SanitizeName(var->name_hint_);
}

}  // namespace codegen
}  // namespace pypto
