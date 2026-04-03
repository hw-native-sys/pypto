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
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
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

void ValidateOrchestrationReferences(const ProgramPtr& program, const FunctionPtr& func) {
  CHECK(func->func_type_ == FunctionType::Orchestration)
      << "ValidateOrchestrationReferences should only be called on Orchestration functions";

  class FunctionCallCollector : public IRVisitor {
   public:
    std::set<std::string> called_functions_;

    void VisitExpr_(const CallPtr& call) override {
      if (!IsBuiltinOp(call->op_->name_)) {
        called_functions_.insert(call->op_->name_);
      }
      IRVisitor::VisitExpr_(call);
    }
  };

  FunctionCallCollector collector;
  collector.VisitStmt(func->body_);

  std::vector<std::string> missing_functions;
  for (const auto& func_name : collector.called_functions_) {
    if (!program->GetFunction(func_name)) {
      missing_functions.push_back(func_name);
    }
  }

  if (!missing_functions.empty()) {
    std::ostringstream oss;
    oss << "Orchestration function '" << func->name_ << "' references undefined functions. "
        << "The Program must contain all functions referenced in orchestration calls.\n"
        << "Missing functions: [";
    for (size_t i = 0; i < missing_functions.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << "'" << missing_functions[i] << "'";
    }
    oss << "]";
    throw pypto::ValueError(oss.str());
  }
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
// AssembleViewOptimizer
// ---------------------------------------------------------------------------

AssembleViewOptimizer::AssembleViewOptimizer(const std::unordered_map<const Var*, const Var*>& buffer_roots)
    : buffer_roots_(buffer_roots) {}

void AssembleViewOptimizer::VisitStmt_(const AssignStmtPtr& assign) {
  tuple_values_.erase(assign->var_.get());
  if (auto tuple_value = As<MakeTuple>(assign->value_)) {
    tuple_values_[assign->var_.get()] = tuple_value;
  } else if (auto call = As<Call>(assign->value_)) {
    if (call->op_->name_ == "tensor.assemble" && call->args_.size() == 3) {
      const Var* source_root = ResolveExpr(call->args_[1]);
      auto offset_tuple = ResolveTupleExpr(call->args_[2]);
      if (source_root && offset_tuple) {
        RecordAssembleViewInfo(source_root, call->args_[0], offset_tuple);
      }
    }
  } else if (auto src_var = AsVarLike(assign->value_)) {
    if (auto it = tuple_values_.find(src_var.get()); it != tuple_values_.end()) {
      tuple_values_[assign->var_.get()] = it->second;
    }
  }
  IRVisitor::VisitStmt_(assign);
}

void AssembleViewOptimizer::RecordAssembleViewInfo(const Var* source_root, const ExprPtr& target_expr,
                                                   const MakeTuplePtr& offset_tuple) {
  if (non_optimizable_roots.count(source_root) > 0) {
    return;
  }
  auto [it, inserted] = assemble_view_infos.emplace(source_root, AssembleViewInfo{target_expr, offset_tuple});
  if (!inserted) {
    assemble_view_infos.erase(source_root);
    non_optimizable_roots.insert(source_root);
  }
}

const Var* AssembleViewOptimizer::ResolveVar(const Var* var) const {
  auto it = buffer_roots_.find(var);
  return it != buffer_roots_.end() ? it->second : nullptr;
}

const Var* AssembleViewOptimizer::ResolveExpr(const ExprPtr& expr) const {
  if (auto var = AsVarLike(expr)) {
    return ResolveVar(var.get());
  }
  return nullptr;
}

MakeTuplePtr AssembleViewOptimizer::ResolveTupleExpr(const ExprPtr& expr) const {
  if (auto tuple = As<MakeTuple>(expr)) {
    return tuple;
  }
  if (auto var = AsVarLike(expr)) {
    auto it = tuple_values_.find(var.get());
    if (it != tuple_values_.end()) {
      return it->second;
    }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// LoopEscapeInfoCollector
// ---------------------------------------------------------------------------

namespace {

class VarReferenceCollector : public IRVisitor {
 public:
  explicit VarReferenceCollector(const Var* target) : target_(target) {}

  bool found = false;

 protected:
  void VisitExpr_(const VarPtr& op) override {
    if (op.get() == target_) {
      found = true;
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IterArgPtr& op) override {
    if (op.get() == target_) {
      found = true;
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  const Var* target_;
};

}  // namespace

void LoopEscapeInfoCollector::VisitStmt_(const SeqStmtsPtr& seq) {
  for (size_t i = 0; i < seq->stmts_.size(); ++i) {
    if (auto for_stmt = As<ForStmt>(seq->stmts_[i])) {
      for (const auto& return_var : for_stmt->return_vars_) {
        if (!return_var) continue;
        if (IsVarReferencedLater(seq, i + 1, return_var.get())) {
          escaping_loop_returns.insert(return_var.get());
        }
      }
    }
    VisitStmt(seq->stmts_[i]);
  }
}

bool LoopEscapeInfoCollector::IsVarReferencedLater(const SeqStmtsPtr& seq, size_t start_index,
                                                   const Var* target) {
  VarReferenceCollector collector(target);
  for (size_t i = start_index; i < seq->stmts_.size() && !collector.found; ++i) {
    collector.VisitStmt(seq->stmts_[i]);
  }
  return collector.found;
}

// ---------------------------------------------------------------------------
// VarLineageCollector
// ---------------------------------------------------------------------------

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

}  // namespace codegen
}  // namespace pypto
