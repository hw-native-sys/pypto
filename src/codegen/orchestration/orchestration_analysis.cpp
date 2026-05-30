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

#include "pypto/codegen/orchestration/orchestration_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

std::string GetSSABaseName(const std::string& name) { return auto_name::GetCompatibleBaseName(name); }

bool IsBuiltinOp(const std::string& op_name) {
  return op_name.find("tile.") == 0 || op_name.find("tensor.") == 0 || op_name.find("system.") == 0 ||
         op_name.find("array.") == 0;
}

bool IsTensorOp(const std::string& op_name) { return op_name.find("tensor.") == 0; }

bool IsArrayOp(const std::string& op_name) { return op_name.find("array.") == 0; }

std::string FormatConstIntValue(const ConstIntPtr& c, const std::string& cpp_type) {
  int64_t v = c->value_;
  if (cpp_type != "int64_t") {
    return "static_cast<" + cpp_type + ">(" + std::to_string(v) + ")";
  }
  return std::to_string(v);
}

std::string FormatConstFloatValue(const ConstFloatPtr& c, const std::string& cpp_type) {
  double v = c->value_;
  if (cpp_type == "float") {
    return std::to_string(static_cast<float>(v));
  }
  return std::to_string(v);  // double
}

int GetOrCreateFuncId(const std::string& func_name, std::map<std::string, int>* func_name_to_id,
                      int* next_func_id) {
  if (func_name_to_id->find(func_name) == func_name_to_id->end()) {
    (*func_name_to_id)[func_name] = (*next_func_id)++;
  }
  return (*func_name_to_id)[func_name];
}

// ---------------------------------------------------------------------------
// OrchestrationInfoCollector
// ---------------------------------------------------------------------------

void OrchestrationInfoCollector::VisitStmt_(const AssignStmtPtr& assign) {
  if (auto call = As<Call>(assign->value_)) {
    if (!IsBuiltinOp(call->op_->name_) && call->op_->name_ != "tensor.create") {
      if (As<TupleType>(call->GetType())) {
        std::string unique_key = "_tc_" + std::to_string(tuple_call_counter_++);
        tuple_var_to_key[assign->var_.get()] = unique_key;
        call_to_tuple_key[call.get()] = unique_key;
      }
    }
  } else if (As<MakeTuple>(assign->value_)) {
    std::string unique_key = "_tc_" + std::to_string(tuple_call_counter_++);
    tuple_var_to_key[assign->var_.get()] = unique_key;
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_ref = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_var_to_key.find(tuple_ref.get());
      if (it != tuple_var_to_key.end()) {
        call_tuple_elements[it->second].push_back({tuple_get->index_, assign->var_.get()});
      }
    }
  }
  IRVisitor::VisitStmt_(assign);
}

// ---------------------------------------------------------------------------
// BufferRootCollector
// ---------------------------------------------------------------------------

BufferRootCollector::BufferRootCollector(ProgramPtr program) : program_(std::move(program)) {}

void BufferRootCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    buffer_roots[param.get()] = param.get();
  }
}

void BufferRootCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = for_stmt->iter_args_[i];
    const Var* root = ResolveExpr(iter_arg->initValue_);
    if (root) {
      buffer_roots[iter_arg.get()] = root;
      if (i < for_stmt->return_vars_.size()) {
        buffer_roots[for_stmt->return_vars_[i].get()] = root;
      }
    }
  }
  IRVisitor::VisitStmt_(for_stmt);
}

void BufferRootCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = while_stmt->iter_args_[i];
    const Var* root = ResolveExpr(iter_arg->initValue_);
    if (root) {
      buffer_roots[iter_arg.get()] = root;
      if (i < while_stmt->return_vars_.size()) {
        buffer_roots[while_stmt->return_vars_[i].get()] = root;
      }
    }
  }
  IRVisitor::VisitStmt_(while_stmt);
}

void BufferRootCollector::VisitStmt_(const AssignStmtPtr& assign) {
  if (auto call = As<Call>(assign->value_)) {
    const std::string& op_name = call->op_->name_;
    if (op_name == "tensor.create" || op_name == "tensor.slice") {
      buffer_roots[assign->var_.get()] = assign->var_.get();
    } else if (op_name == "tensor.assemble") {
      if (call->args_.size() == 3) {
        if (const Var* target_root = ResolveExpr(call->args_[0])) {
          buffer_roots[assign->var_.get()] = target_root;
        }
      }
    } else if (!IsBuiltinOp(op_name)) {
      auto out_roots = CollectCallOutputRoots(call);
      if (As<TupleType>(call->GetType())) {
        tuple_output_roots_[assign->var_.get()] = std::move(out_roots);
      } else if (!out_roots.empty() && out_roots[0]) {
        buffer_roots[assign->var_.get()] = out_roots[0];
      }
    }
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_output_roots_.find(tuple_var.get());
      if (it != tuple_output_roots_.end() && tuple_get->index_ < static_cast<int>(it->second.size()) &&
          it->second[tuple_get->index_]) {
        buffer_roots[assign->var_.get()] = it->second[tuple_get->index_];
      }
    }
  } else if (auto src_var = AsVarLike(assign->value_)) {
    if (const Var* root = ResolveVar(src_var.get())) {
      buffer_roots[assign->var_.get()] = root;
    }
  }
  IRVisitor::VisitStmt_(assign);
}

const Var* BufferRootCollector::ResolveVar(const Var* var) const {
  auto it = buffer_roots.find(var);
  return it != buffer_roots.end() ? it->second : nullptr;
}

const Var* BufferRootCollector::ResolveExpr(const ExprPtr& expr) const {
  if (auto var = AsVarLike(expr)) {
    return ResolveVar(var.get());
  }
  return nullptr;
}

std::vector<const Var*> BufferRootCollector::CollectCallOutputRoots(const CallPtr& call) const {
  auto callee = program_->GetFunction(call->op_->name_);
  if (!callee) return {};

  std::vector<const Var*> roots;
  for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
    if (callee->param_directions_[i] != ParamDirection::Out &&
        callee->param_directions_[i] != ParamDirection::InOut) {
      continue;
    }
    if (auto arg_var = AsVarLike(call->args_[i])) {
      roots.push_back(ResolveVar(arg_var.get()));
    } else {
      roots.push_back(nullptr);
    }
  }
  return roots;
}

// ---------------------------------------------------------------------------
// VarLineageCollector
// ---------------------------------------------------------------------------

VarLineageCollector::VarLineageCollector(ProgramPtr program) : program_(std::move(program)) {}

void VarLineageCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    var_to_param[param.get()] = param.get();
  }
}

void VarLineageCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = for_stmt->iter_args_[i];
    const Var* param = ResolveExpr(iter_arg->initValue_);
    if (param) {
      var_to_param[iter_arg.get()] = param;
      // Only propagate buffer lineage to the return_var for Tensor-type carries.
      // Scalar carries (e.g. a loop counter ``idx = batch_base + inner``) are
      // value-typed: the body may overwrite them with a freshly computed value
      // that has no relationship to the init param.  Propagating param lineage
      // to a Scalar return_var makes FindReturnedParamIndices incorrectly map
      // a Scalar return element to a param index, causing EmitTensorAlias to
      // emit ``const Tensor&`` for an int64_t variable (issue #1580).
      if (i < for_stmt->return_vars_.size() && AsTensorTypeLike(for_stmt->return_vars_[i]->GetType())) {
        var_to_param[for_stmt->return_vars_[i].get()] = param;
      }
    }
  }
  IRVisitor::VisitStmt_(for_stmt);
}

void VarLineageCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = while_stmt->iter_args_[i];
    const Var* param = ResolveExpr(iter_arg->initValue_);
    if (param) {
      var_to_param[iter_arg.get()] = param;
      // Same guard as ForStmt: only propagate to Tensor return_vars.
      if (i < while_stmt->return_vars_.size() && AsTensorTypeLike(while_stmt->return_vars_[i]->GetType())) {
        var_to_param[while_stmt->return_vars_[i].get()] = param;
      }
    }
  }
  IRVisitor::VisitStmt_(while_stmt);
}

void VarLineageCollector::VisitStmt_(const AssignStmtPtr& assign) {
  if (auto src_var = AsVarLike(assign->value_)) {
    const Var* param = ResolveVar(src_var.get());
    if (param) {
      var_to_param[assign->var_.get()] = param;
    }
  } else if (auto call = As<Call>(assign->value_)) {
    // Propagate lineage through function calls: the result inherits lineage
    // from the Out/InOut argument. This covers sequential SPMD submissions
    // like: out = self.kernel(a, b, out) where `out` is the output buffer.
    //
    // Group functions (produced by ScopeOutliner) have all directions set to
    // In, so we trace through their bodies to find the inner kernel call and
    // use its directions mapped back to the Group's parameter positions.
    if (!IsBuiltinOp(call->op_->name_)) {
      auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
      if (callee) {
        std::vector<ParamDirection> effective_dirs = callee->param_directions_;
        if (callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd) {
          effective_dirs = ComputeGroupEffectiveDirections(callee, program_);
        }
        // Prefer tracing through the Out/InOut arg the callee actually
        // returns (multi-Out kernels would otherwise be mis-traced to the
        // first Out, leaking scratch-buffer lineage onto the result Var).
        std::optional<size_t> returned_idx = FindReturnedParamIndex(callee, program_);
        for (size_t i = 0; i < effective_dirs.size() && i < call->args_.size(); ++i) {
          if (effective_dirs[i] != ParamDirection::Out && effective_dirs[i] != ParamDirection::InOut) {
            continue;
          }
          if (returned_idx.has_value() && i != *returned_idx) {
            continue;
          }
          if (auto arg_var = AsVarLike(call->args_[i])) {
            const Var* param = ResolveVar(arg_var.get());
            if (param) {
              var_to_param[assign->var_.get()] = param;
              break;
            }
          }
        }
      }
    }
  }
  IRVisitor::VisitStmt_(assign);
}

const Var* VarLineageCollector::ResolveVar(const Var* var) const {
  auto it = var_to_param.find(var);
  return it != var_to_param.end() ? it->second : nullptr;
}

const Var* VarLineageCollector::ResolveExpr(const ExprPtr& expr) const {
  if (auto var = AsVarLike(expr)) {
    return ResolveVar(var.get());
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// FindReturnedParamIndex
// ---------------------------------------------------------------------------

// Walk the callee body collecting per-var defining AssignStmts and
// ReturnStmt locations. The callee body may not contain a top-level
// ReturnStmt (e.g. Group wrappers produced by the outliner end with the
// inner kernel call); in that case the function falls back to nullopt.
namespace {

class ReturnAndDefCollector : public IRVisitor {
 public:
  ReturnStmtPtr first_return;
  std::unordered_map<const Var*, AssignStmtPtr> var_def;

 protected:
  void VisitStmt_(const ReturnStmtPtr& ret) override {
    if (!first_return) first_return = ret;
  }
  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (assign->var_) {
      var_def.emplace(assign->var_.get(), assign);
    }
    IRVisitor::VisitStmt_(assign);
  }
};

// Forward declaration: lineage walk for a returned value uses both the
// passed-in lineage map and a small inline tracer for builtin
// tensor.assemble / tile.store rules + user-call returned-param recursion.
const Var* TraceReturnedToParam(const ExprPtr& root_expr, const ReturnAndDefCollector& collector,
                                const VarLineageCollector& lineage, const ProgramPtr& program,
                                std::unordered_set<const Var*>& visited);

const Var* TraceVarToParam(const Var* var, const ReturnAndDefCollector& collector,
                           const VarLineageCollector& lineage, const ProgramPtr& program,
                           std::unordered_set<const Var*>& visited) {
  if (!var) return nullptr;
  if (!visited.insert(var).second) return nullptr;

  // VarLineageCollector already handles ForStmt iter args, var-to-var
  // assigns and user-call lineage; check it first.
  auto lin_it = lineage.var_to_param.find(var);
  if (lin_it != lineage.var_to_param.end()) {
    return lin_it->second;
  }

  // Builtin tensor / tile output-side ops produce a fresh SSA var bound
  // to an existing GM tensor (the kernel's Out parameter); follow that
  // target arg manually since VarLineageCollector does not propagate
  // through builtin op calls.
  auto def_it = collector.var_def.find(var);
  if (def_it == collector.var_def.end()) return nullptr;

  const auto& assign = def_it->second;
  if (auto rhs_var = AsVarLike(assign->value_)) {
    return TraceVarToParam(rhs_var.get(), collector, lineage, program, visited);
  }
  if (auto call = As<Call>(assign->value_)) {
    const std::string& op_name = call->op_->name_;
    // tensor.assemble(target, tile, offset) -> aliases target (args[0]).
    if (op_name == "tensor.assemble" && !call->args_.empty()) {
      return TraceReturnedToParam(call->args_[0], collector, lineage, program, visited);
    }
    // tile.store(value, indices, target) -> aliases target (args[2]).
    if (op_name == "tile.store" && call->args_.size() >= 3) {
      return TraceReturnedToParam(call->args_[2], collector, lineage, program, visited);
    }
    // tensor.set_validshape(target, rows, cols) -> aliases target.
    if (op_name == "tensor.set_validshape" && !call->args_.empty()) {
      return TraceReturnedToParam(call->args_[0], collector, lineage, program, visited);
    }
  }
  return nullptr;
}

const Var* TraceReturnedToParam(const ExprPtr& root_expr, const ReturnAndDefCollector& collector,
                                const VarLineageCollector& lineage, const ProgramPtr& program,
                                std::unordered_set<const Var*>& visited) {
  if (auto var = AsVarLike(root_expr)) {
    return TraceVarToParam(var.get(), collector, lineage, program, visited);
  }
  return nullptr;
}

}  // namespace

std::optional<size_t> FindReturnedParamIndex(const FunctionPtr& callee, const ProgramPtr& program) {
  // Cycle guard + memoization: the helper is invoked transitively by
  // VarLineageCollector when a callee body itself contains user-function
  // calls. Each level rebuilds a full ReturnAndDefCollector and
  // VarLineageCollector over the callee body — without memoization the
  // worst-case cost cascades to O(N^d) when nested d levels deep. The
  // result depends only on the callee body (independent of caller
  // context), so a thread_local Function* -> result cache collapses
  // repeat visits and bounds total work to O(sum_i N_i) for distinct
  // callees touched in one top-level invocation. The cache lives
  // alongside call_stack and is cleared when the outermost call
  // unwinds, so it never crosses unrelated compilations on the same
  // thread.
  thread_local std::vector<const Function*> call_stack;
  thread_local std::unordered_map<const Function*, std::optional<size_t>> memo;

  if (!callee) return std::nullopt;
  if (std::find(call_stack.begin(), call_stack.end(), callee.get()) != call_stack.end()) {
    return std::nullopt;
  }
  if (auto it = memo.find(callee.get()); it != memo.end()) {
    return it->second;
  }
  call_stack.push_back(callee.get());
  bool is_top_level = call_stack.size() == 1;
  struct RecursionGuard {
    std::vector<const Function*>& stack;
    std::unordered_map<const Function*, std::optional<size_t>>& memo_ref;
    bool top_level;
    ~RecursionGuard() {
      stack.pop_back();
      if (top_level) memo_ref.clear();
    }
  } guard{call_stack, memo, is_top_level};

  auto record = [&](std::optional<size_t> result) -> std::optional<size_t> {
    memo.emplace(callee.get(), result);
    return result;
  };

  if (!callee->body_) return record(std::nullopt);

  ReturnAndDefCollector collector;
  collector.VisitStmt(callee->body_);
  if (!collector.first_return || collector.first_return->value_.empty()) {
    return record(std::nullopt);
  }

  VarLineageCollector lineage(program);
  lineage.Initialize(callee->params_);
  lineage.VisitStmt(callee->body_);

  std::unordered_set<const Var*> visited;
  const Var* root =
      TraceReturnedToParam(collector.first_return->value_[0], collector, lineage, program, visited);
  if (!root) return record(std::nullopt);

  for (size_t i = 0; i < callee->params_.size(); ++i) {
    if (callee->params_[i].get() == root) return record(i);
  }
  return record(std::nullopt);
}

std::vector<std::optional<size_t>> FindReturnedParamIndices(const FunctionPtr& callee,
                                                            const ProgramPtr& program) {
  // Per-position generalization of FindReturnedParamIndex: trace every element
  // of the callee's top-level ReturnStmt back to a Param. Returns an empty
  // vector when there is no traceable top-level ReturnStmt (e.g. Group/Spmd
  // wrappers whose body ends in the inner kernel call) so callers can fall
  // back to a direction-based heuristic. Each entry maps a return-tuple
  // position to its source ``callee->params_`` index, or nullopt when that
  // position is not a writeback to a param (e.g. an auxiliary scalar).
  if (!callee || !callee->body_) return {};

  ReturnAndDefCollector collector;
  collector.VisitStmt(callee->body_);
  if (!collector.first_return || collector.first_return->value_.empty()) {
    return {};
  }

  VarLineageCollector lineage(program);
  lineage.Initialize(callee->params_);
  lineage.VisitStmt(callee->body_);

  std::vector<std::optional<size_t>> result;
  result.reserve(collector.first_return->value_.size());
  for (const auto& ret_value : collector.first_return->value_) {
    std::unordered_set<const Var*> visited;
    const Var* root = TraceReturnedToParam(ret_value, collector, lineage, program, visited);
    std::optional<size_t> idx;
    if (root) {
      for (size_t i = 0; i < callee->params_.size(); ++i) {
        if (callee->params_[i].get() == root) {
          idx = i;
          break;
        }
      }
    }
    result.push_back(idx);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ComputeGroupEffectiveDirections
// ---------------------------------------------------------------------------

std::vector<ParamDirection> ComputeGroupEffectiveDirections(const FunctionPtr& group_func,
                                                            const ProgramPtr& program) {
  if (!group_func) return {};
  std::vector<ParamDirection> fallback(group_func->params_.size(), ParamDirection::In);
  if (!program) return fallback;

  // Recursive summary for Group/Spmd functions:
  // - walk callsites in function body;
  // - for nested Group/Spmd callees, reuse recursively-computed effective dirs;
  // - merge directions as lattice: InOut > Out > In.
  //
  // This keeps writeback semantics across Group->Group wrappers generated by SPMD/outline passes.
  std::unordered_map<const Function*, std::vector<ParamDirection>> memo;
  std::unordered_set<const Function*> visiting;

  std::function<std::vector<ParamDirection>(const FunctionPtr&)> compute_effective =
      [&](const FunctionPtr& func) -> std::vector<ParamDirection> {
    if (!func) return {};
    auto memo_it = memo.find(func.get());
    if (memo_it != memo.end()) return memo_it->second;

    std::vector<ParamDirection> directions(func->params_.size(), ParamDirection::In);
    if (!func->body_) {
      memo.emplace(func.get(), directions);
      return directions;
    }

    // Cycle guard: fall back to declared directions if a recursion cycle exists.
    if (!visiting.insert(func.get()).second) {
      std::vector<ParamDirection> declared = func->param_directions_;
      if (declared.size() != func->params_.size()) declared.resize(func->params_.size(), ParamDirection::In);
      return declared;
    }

    auto inner_calls = ir::CollectInnerCalls(func, program);
    if (!inner_calls.empty()) {
      std::unordered_map<const Var*, size_t> param_to_index;
      for (size_t i = 0; i < func->params_.size(); ++i) {
        param_to_index[func->params_[i].get()] = i;
      }

      for (const auto& [inner_call, inner_callee] : inner_calls) {
        const auto& inner_args = inner_call->args_;
        std::vector<ParamDirection> inner_dirs;
        if (inner_callee->func_type_ == FunctionType::Group ||
            inner_callee->func_type_ == FunctionType::Spmd) {
          inner_dirs = compute_effective(inner_callee);
        } else {
          inner_dirs = inner_callee->param_directions_;
        }
        for (size_t arg_idx = 0; arg_idx < inner_args.size() && arg_idx < inner_dirs.size(); ++arg_idx) {
          auto var = AsVarLike(inner_args[arg_idx]);
          if (!var) continue;
          auto it = param_to_index.find(var.get());
          if (it == param_to_index.end()) continue;
          ParamDirection d = inner_dirs[arg_idx];
          ParamDirection& merged = directions[it->second];
          if (d == ParamDirection::InOut || (d == ParamDirection::Out && merged == ParamDirection::In)) {
            merged = d;
          }
        }
      }
    }

    visiting.erase(func.get());
    memo.emplace(func.get(), directions);
    return directions;
  };

  return compute_effective(group_func);
}

}  // namespace codegen
}  // namespace pypto
