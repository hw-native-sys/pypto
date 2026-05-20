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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::codegen::ComputeGroupEffectiveDirections;
using ::pypto::codegen::IsBuiltinOp;

enum class AccessKind { Read, Write, ReadWrite };

struct StorageAccess {
  const Var* root = nullptr;
  AccessKind kind = AccessKind::Read;
  VarPtr task_id_var;
};

bool IsTensorType(const TypePtr& type) { return As<TensorType>(type) != nullptr; }

bool HasTaskIdTail(const CallPtr& call) {
  auto tuple_ty = As<TupleType>(call ? call->GetType() : TypePtr{});
  if (!tuple_ty || tuple_ty->types_.empty()) return false;
  auto scalar_ty = As<ScalarType>(tuple_ty->types_.back());
  return scalar_ty && scalar_ty->dtype_ == DataType::TASK_ID;
}

std::vector<ParamDirection> ResolveCalleeDirections(const ProgramPtr& program, const CallPtr& call,
                                                    const FunctionPtr& callee) {
  if (!callee) return {};
  if (callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd) {
    return ComputeGroupEffectiveDirections(callee, program);
  }
  return callee->param_directions_;
}

std::vector<VarPtr> GetDepAttr(const CallPtr& call, const char* key) {
  if (!call) return {};
  for (const auto& [k, v] : call->attrs_) {
    if (k != key) continue;
    if (const auto* edges = std::any_cast<std::vector<VarPtr>>(&v)) {
      return *edges;
    }
    return {};
  }
  return {};
}

bool ContainsVar(const std::vector<VarPtr>& vars, const VarPtr& candidate) {
  if (!candidate) return false;
  for (const auto& var : vars) {
    if (var && var->UniqueId() == candidate->UniqueId()) return true;
  }
  return false;
}

void AppendUnique(std::vector<VarPtr>* vars, const VarPtr& candidate) {
  if (!vars || !candidate || ContainsVar(*vars, candidate)) return;
  vars->push_back(candidate);
}

bool HasHazard(AccessKind current, AccessKind prior) {
  const bool current_writes = current == AccessKind::Write || current == AccessKind::ReadWrite;
  const bool current_reads = current == AccessKind::Read || current == AccessKind::ReadWrite;
  const bool prior_writes = prior == AccessKind::Write || prior == AccessKind::ReadWrite;
  const bool prior_reads = prior == AccessKind::Read || prior == AccessKind::ReadWrite;
  return (current_reads && prior_writes) || (current_writes && prior_reads) ||
         (current_writes && prior_writes);
}

class StorageRootAnalysis : public IRVisitor {
 public:
  explicit StorageRootAnalysis(ProgramPtr program) : program_(std::move(program)) {}

  void Initialize(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      if (param && IsTensorType(param->GetType())) {
        roots_[param.get()] = param.get();
      }
    }
  }

  const Var* ResolveExpr(const ExprPtr& expr) const {
    auto var = AsVarLike(expr);
    if (!var) return nullptr;
    auto it = roots_.find(var.get());
    return it != roots_.end() ? it->second : nullptr;
  }

 protected:
  void VisitStmt_(const ForStmtPtr& op) override {
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      const Var* root = ResolveExpr(op->iter_args_[i]->initValue_);
      if (!root) continue;
      roots_[op->iter_args_[i].get()] = root;
      if (i < op->return_vars_.size()) {
        roots_[op->return_vars_[i].get()] = root;
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      const Var* root = ResolveExpr(op->iter_args_[i]->initValue_);
      if (!root) continue;
      roots_[op->iter_args_[i].get()] = root;
      if (i < op->return_vars_.size()) {
        roots_[op->return_vars_[i].get()] = root;
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_)) {
      if (As<TupleType>(call->GetType()) && !IsBuiltinOp(call->op_->name_)) {
        tuple_roots_[op->var_.get()] = CollectCallOutputRoots(call);
        IRVisitor::VisitStmt_(op);
        return;
      }
    }

    if (!op->var_ || !IsTensorType(op->var_->GetType())) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    if (auto call = As<Call>(op->value_)) {
      const std::string& op_name = call->op_->name_;
      if (op_name == "tensor.create") {
        roots_[op->var_.get()] = op->var_.get();
      } else if (op_name == "tensor.slice") {
        if (!call->args_.empty()) {
          if (const Var* root = ResolveExpr(call->args_[0])) {
            roots_[op->var_.get()] = root;
          }
        }
      } else if (op_name == "tensor.assemble") {
        if (!call->args_.empty()) {
          if (const Var* root = ResolveExpr(call->args_[0])) {
            roots_[op->var_.get()] = root;
          }
        }
      } else if (!IsBuiltinOp(op_name)) {
        auto out_roots = CollectCallOutputRoots(call);
        if (As<TupleType>(call->GetType())) {
          tuple_roots_[op->var_.get()] = std::move(out_roots);
        } else if (!out_roots.empty() && out_roots[0]) {
          roots_[op->var_.get()] = out_roots[0];
        }
      }
    } else if (auto tuple_get = As<TupleGetItemExpr>(op->value_)) {
      if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
        auto it = tuple_roots_.find(tuple_var.get());
        if (it != tuple_roots_.end() && tuple_get->index_ >= 0 &&
            tuple_get->index_ < static_cast<int>(it->second.size()) && it->second[tuple_get->index_]) {
          roots_[op->var_.get()] = it->second[tuple_get->index_];
        }
      }
    } else if (const Var* root = ResolveExpr(op->value_)) {
      roots_[op->var_.get()] = root;
    }

    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<const Var*> CollectCallOutputRoots(const CallPtr& call) const {
    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) return {};
    auto dirs = ResolveCalleeDirections(program_, call, callee);
    std::vector<const Var*> roots;
    for (size_t i = 0; i < dirs.size() && i < call->args_.size(); ++i) {
      if (dirs[i] != ParamDirection::Out && dirs[i] != ParamDirection::InOut) continue;
      roots.push_back(ResolveExpr(call->args_[i]));
    }
    return roots;
  }

  ProgramPtr program_;
  std::unordered_map<const Var*, const Var*> roots_;
  std::unordered_map<const Var*, std::vector<const Var*>> tuple_roots_;
};

class SubmitTaskIdCollector : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto tuple_get = As<TupleGetItemExpr>(op->value_)) {
      if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
        tuple_get_by_tuple_[tuple_var.get()][tuple_get->index_] = op->var_;
        auto call_it = call_by_tuple_.find(tuple_var.get());
        if (call_it != call_by_tuple_.end()) {
          auto tuple_ty = As<TupleType>(call_it->second->GetType());
          const int task_id_index = static_cast<int>(tuple_ty->types_.size()) - 1;
          if (tuple_get->index_ == task_id_index) {
            task_id_by_call_[call_it->second.get()] = op->var_;
          }
        }
      }
    }

    if (auto call = As<Call>(op->value_)) {
      if (HasTaskIdTail(call)) {
        call_by_tuple_[op->var_.get()] = call;
        auto tuple_ty = As<TupleType>(call->GetType());
        const int task_id_index = static_cast<int>(tuple_ty->types_.size()) - 1;
        auto it = tuple_get_by_tuple_.find(op->var_.get());
        if (it != tuple_get_by_tuple_.end()) {
          auto elem_it = it->second.find(task_id_index);
          if (elem_it != it->second.end()) {
            task_id_by_call_[call.get()] = elem_it->second;
          }
        }
      }
    }

    IRVisitor::VisitStmt_(op);
  }

  const std::unordered_map<const Call*, VarPtr>& task_id_by_call() const { return task_id_by_call_; }

 private:
  std::unordered_map<const Var*, CallPtr> call_by_tuple_;
  std::unordered_map<const Var*, std::unordered_map<int, VarPtr>> tuple_get_by_tuple_;
  std::unordered_map<const Call*, VarPtr> task_id_by_call_;
};

class AutoDepMutator : public IRMutator {
 public:
  AutoDepMutator(ProgramPtr program, const StorageRootAnalysis* storage,
                 const std::unordered_map<const Call*, VarPtr>* task_id_by_call)
      : program_(std::move(program)), storage_(storage), task_id_by_call_(task_id_by_call) {}

 protected:
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    if (!op->manual_) {
      return IRMutator::VisitStmt_(op);
    }

    prior_stack_.emplace_back();
    auto out = IRMutator::VisitStmt_(op);
    prior_stack_.pop_back();
    return out;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    if (!call || prior_stack_.empty()) return base;
    if (IsBuiltinOp(call->op_->name_)) return call;

    VarPtr task_id = LookupTaskId(op.get());
    auto accesses = SummarizeAccesses(call);
    if (accesses.empty()) return call;

    std::vector<VarPtr> compiler_edges;
    auto user_edges = GetDepAttr(call, kAttrManualDepEdges);
    for (const auto& access : accesses) {
      for (const auto& prior : prior_stack_.back()) {
        if (access.root == nullptr || access.root != prior.root) continue;
        if (!HasHazard(access.kind, prior.kind)) continue;
        CHECK(prior.task_id_var)
            << "manual_scope auto-deps requires a producer TaskId for a prior call that writes storage read "
            << "or written by call '" << call->op_->name_
            << "'. Use `out, tid = pl.submit(self.kernel, ...)` for the producer inside manual_scope.";
        if (ContainsVar(user_edges, prior.task_id_var)) continue;
        AppendUnique(&compiler_edges, prior.task_id_var);
      }
    }

    for (auto& access : accesses) {
      access.task_id_var = task_id;
      prior_stack_.back().push_back(std::move(access));
    }

    if (compiler_edges.empty()) {
      return call;
    }

    auto new_attrs = WithCompilerManualDepEdgesAttr(call->attrs_, std::move(compiler_edges));
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

 private:
  VarPtr LookupTaskId(const Call* call) const {
    if (!task_id_by_call_) return nullptr;
    auto it = task_id_by_call_->find(call);
    return it != task_id_by_call_->end() ? it->second : nullptr;
  }

  std::vector<StorageAccess> SummarizeAccesses(const CallPtr& call) const {
    std::vector<StorageAccess> out;
    auto dirs = call->GetArgDirections();
    if (dirs.size() != call->args_.size()) return out;

    for (size_t i = 0; i < dirs.size(); ++i) {
      const Var* root = storage_ ? storage_->ResolveExpr(call->args_[i]) : nullptr;
      if (!root) continue;
      std::optional<AccessKind> kind;
      switch (dirs[i]) {
        case ArgDirection::Input:
          kind = AccessKind::Read;
          break;
        case ArgDirection::Output:
        case ArgDirection::OutputExisting:
          kind = AccessKind::Write;
          break;
        case ArgDirection::InOut:
          kind = AccessKind::ReadWrite;
          break;
        case ArgDirection::NoDep:
        case ArgDirection::Scalar:
          break;
      }
      if (kind.has_value()) {
        out.push_back(StorageAccess{root, *kind, nullptr});
      }
    }
    return out;
  }

  ProgramPtr program_;
  const StorageRootAnalysis* storage_;
  const std::unordered_map<const Call*, VarPtr>* task_id_by_call_;
  std::vector<std::vector<StorageAccess>> prior_stack_;
};

}  // namespace

namespace pass {

Pass AutoDeriveTaskDependencies() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;

    auto new_functions = program->functions_;
    bool changed = false;

    for (auto& [gvar, func] : new_functions) {
      (void)gvar;
      if (!func || !func->body_) continue;

      StorageRootAnalysis storage(program);
      storage.Initialize(func->params_);
      storage.VisitStmt(func->body_);

      SubmitTaskIdCollector task_ids;
      task_ids.VisitStmt(func->body_);

      AutoDepMutator mutator(program, &storage, &task_ids.task_id_by_call());
      auto new_body = mutator.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;

      changed = true;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }

    if (!changed) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "AutoDeriveTaskDependencies", kAutoDeriveTaskDependenciesProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
