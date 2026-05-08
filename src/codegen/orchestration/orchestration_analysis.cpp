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
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

std::string GetSSABaseName(const std::string& name) { return auto_name::GetCompatibleBaseName(name); }

bool IsBuiltinOp(const std::string& op_name) {
  return op_name.find("tile.") == 0 || op_name.find("tensor.") == 0 || op_name.find("system.") == 0;
}

bool IsTensorOp(const std::string& op_name) { return op_name.find("tensor.") == 0; }

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
        current_tuple_key_[assign->var_->name_hint_] = unique_key;
        call_to_tuple_key[call.get()] = unique_key;
      }
    }
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    std::string tuple_ref_name;
    if (auto var = As<Var>(tuple_get->tuple_)) {
      tuple_ref_name = var->name_hint_;
    } else if (auto iter_arg = As<IterArg>(tuple_get->tuple_)) {
      tuple_ref_name = iter_arg->name_hint_;
    }

    auto it = current_tuple_key_.find(tuple_ref_name);
    if (it != current_tuple_key_.end()) {
      call_tuple_elements[it->second].push_back({tuple_get->index_, assign->var_.get()});
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
      if (i < for_stmt->return_vars_.size()) {
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
      if (i < while_stmt->return_vars_.size()) {
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
        for (size_t i = 0; i < effective_dirs.size() && i < call->args_.size(); ++i) {
          if (effective_dirs[i] != ParamDirection::Out && effective_dirs[i] != ParamDirection::InOut) {
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

    class InnerCallFinder : public IRVisitor {
     public:
      explicit InnerCallFinder(const ProgramPtr& program) : program_(program) {}
      const ProgramPtr& program_;
      std::vector<std::pair<CallPtr, FunctionPtr>> inner_calls;

     protected:
      void VisitExpr_(const CallPtr& call) override {
        if (auto gv = As<GlobalVar>(call->op_)) {
          auto callee = program_->GetFunction(gv->name_);
          if (callee && callee->func_type_ != FunctionType::Orchestration &&
              callee->func_type_ != FunctionType::Opaque) {
            inner_calls.emplace_back(call, callee);
            return;
          }
        }
        IRVisitor::VisitExpr_(call);
      }
    };

    InnerCallFinder finder(program);
    finder.VisitStmt(func->body_);
    if (!finder.inner_calls.empty()) {
      std::unordered_map<const Var*, size_t> param_to_index;
      for (size_t i = 0; i < func->params_.size(); ++i) {
        param_to_index[func->params_[i].get()] = i;
      }

      for (const auto& [inner_call, inner_callee] : finder.inner_calls) {
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
