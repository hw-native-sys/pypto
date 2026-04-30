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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

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
  } else if (entry_func_) {
    EmitEntryFunction();
  }

  return emitter_.GetCode();
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
  emitter_.EmitLine("from simpler.task_interface import TaskArgs, TensorArgType");
  emitter_.EmitLine("from simpler_setup.torch_interop import make_tensor_arg");
}

void DistributedCodegen::EmitFunction(const ir::FunctionPtr& func) {
  declared_vars_.clear();
  task_args_counter_ = 0;
  current_func_ = func;

  bool is_sub_worker = func->role_.has_value() && *func->role_ == ir::Role::SubWorker;
  is_worker_context_ = is_sub_worker;

  // Build function signature
  // Orchestrators: def func(orch, _args, config, *, tensors, callables, sub_ids, _keep):
  // SubWorkers are not emitted as Python functions (they run on device or as registered callables)
  if (is_sub_worker) {
    is_worker_context_ = false;
    return;
  }

  std::ostringstream sig;
  sig << "def " << func->name_ << "(orch, _args, config, *, tensors, callables, sub_ids, _keep):";
  emitter_.EmitLine(sig.str());
  emitter_.IncreaseIndent();

  // Register parameter names
  for (const auto& param : func->params_) {
    declared_vars_.insert(SanitizeName(param->name_hint_));
  }

  // Emit body
  if (func->body_) {
    VisitStmt(func->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("");
  is_worker_context_ = false;
}

void DistributedCodegen::EmitEntryFunction() {
  if (!entry_func_) return;

  declared_vars_.clear();
  task_args_counter_ = 0;
  current_func_ = entry_func_;

  // Entry function signature
  emitter_.EmitLine("def entry(orch, _args, config, *, tensors, callables, sub_ids, _keep):");
  emitter_.IncreaseIndent();

  // Register parameter names
  for (const auto& param : entry_func_->params_) {
    declared_vars_.insert(SanitizeName(param->name_hint_));
  }

  // Emit body
  if (entry_func_->body_) {
    VisitStmt(entry_func_->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("");
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
        current_expr_value_ = callee->name_ +
                              "(orch, _args, config, "
                              "tensors=tensors, callables=callables, sub_ids=sub_ids, _keep=_keep)";
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
  if (op->op_->name_ == "tensor.create") {
    EmitTensorCreate(op);
    return;
  }

  // Regular op call
  current_expr_value_ = op->op_->name_ + "(" + FormatArgs(op->args_) + ")";
}

void DistributedCodegen::VisitExpr_(const ir::VarPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null Var";
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

  // Build TaskArgs from callee's parameter directions
  emitter_.EmitLine(ta_var + " = TaskArgs()");

  for (size_t i = 0; i < call->args_.size(); ++i) {
    VisitExpr(call->args_[i]);
    std::string arg_str = current_expr_value_;
    current_expr_value_ = "";

    // Only tensor args get add_tensor; skip scalar args for now
    if (std::dynamic_pointer_cast<const ir::TensorType>(call->args_[i]->GetType())) {
      std::string tag = "TensorArgType.INPUT";
      if (i < callee->param_directions_.size()) {
        tag = ParamDirectionToTensorArgType(callee->param_directions_[i]);
      }
      emitter_.EmitLine(ta_var + ".add_tensor(make_tensor_arg(tensors[\"" + arg_str + "\"]), " + tag + ")");
    }
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
    // CHIP Worker: orch.submit_next_level(callable, task_args, config)
    emitter_.EmitLine("_keep.append(" + ta_var + ")");
    emitter_.EmitLine("orch.submit_next_level(callables[\"" + callee->name_ + "\"], " + ta_var + ", config)");
  }

  // If this call has an assignment target (return value), alias it to the OUT
  // parameter tensor.  In simpler's runtime model, the callee writes in-place
  // to the OUT parameter, so the return value is the same tensor.
  if (!target.empty() && !callee->return_types_.empty()) {
    // Find the first Out/InOut parameter — that is the tensor the return value aliases
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

void DistributedCodegen::EmitDistIntrinsic(const ir::CallPtr& call) {
  const auto& op_name = call->op_->name_;

  if (op_name == "dist.tree_reduce") {
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
  return call->op_->name_ == "tensor.create";
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

}  // namespace

void DistributedCodegen::CollectHostOrchHoistableAllocs(const ir::FunctionPtr& host_orch) {
  hoisted_allocs_.clear();
  if (!host_orch->body_) return;

  for (const auto& stmt : TopLevelStmts(host_orch->body_)) {
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
    for (const auto& stmt : TopLevelStmts(host_orch->body_)) {
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
