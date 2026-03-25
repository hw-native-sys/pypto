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
#include <vector>

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

  ClassifyFunctions();
  CHECK(!workers_.empty() || !orchestrators_.empty())
      << "Program has no distributed functions (no functions with level/role metadata)";

  EmitIncludes();
  EmitUsingDeclarations();
  emitter_.EmitLine("");

  // Anonymous namespace for internal functions
  emitter_.EmitLine("namespace {");
  emitter_.EmitLine("");

  EmitTopologyConstants();
  EmitTraceHelpers();

  // Emit functions in topological order (callees before callers)
  auto sorted = TopoSortFunctions();
  for (const auto& func : sorted) {
    EmitFunction(func);
  }

  emitter_.EmitLine("}  // namespace");
  emitter_.EmitLine("");

  EmitMain();

  return emitter_.GetCode();
}

// ========================================================================
// Function classification
// ========================================================================

void DistributedCodegen::ClassifyFunctions() {
  for (const auto& [gvar, func] : program_->functions_) {
    all_funcs_[func->name_] = func;

    if (!func->level_.has_value() && !func->role_.has_value()) {
      // Entry function: no level/role metadata
      entry_func_ = func;
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

std::vector<ir::FunctionPtr> DistributedCodegen::TopoSortFunctions() const {
  // Collect non-entry functions
  std::vector<ir::FunctionPtr> funcs;
  for (const auto& [name, func] : all_funcs_) {
    if (func != entry_func_) {
      funcs.push_back(func);
    }
  }

  // Simple sort: workers before orchestrators, lower level before higher level
  std::sort(funcs.begin(), funcs.end(), [](const ir::FunctionPtr& a, const ir::FunctionPtr& b) {
    // Workers before orchestrators
    bool a_worker = a->role_.has_value() && *a->role_ == ir::Role::Worker;
    bool b_worker = b->role_.has_value() && *b->role_ == ir::Role::Worker;
    if (a_worker != b_worker) return a_worker;

    // Lower level before higher level
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

void DistributedCodegen::EmitIncludes() {
  emitter_.EmitLine("#include \"core/tensor.h\"");
  emitter_.EmitLine("#include \"runtime/level_runtime.h\"");
  emitter_.EmitLine("#include \"runtime/tree_reduce.h\"");
  emitter_.EmitLine("");
  emitter_.EmitLine("#include <cstdlib>");
  emitter_.EmitLine("#include <functional>");
  emitter_.EmitLine("#include <future>");
  emitter_.EmitLine("#include <string>");
  emitter_.EmitLine("#include <vector>");
}

void DistributedCodegen::EmitUsingDeclarations() {
  emitter_.EmitLine("");
  emitter_.EmitLine("using linqu::LinquTensor;");
  emitter_.EmitLine("using linqu::LevelRuntime;");
}

void DistributedCodegen::EmitTopologyConstants() {
  // env_int helper
  emitter_.EmitLine("int env_int(const char* name, int fallback) {");
  emitter_.IncreaseIndent();
  emitter_.EmitLine("const char* v = std::getenv(name);");
  emitter_.EmitLine("return v ? std::atoi(v) : fallback;");
  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");
  emitter_.EmitLine("");

  // Topology constants for used levels
  for (int level : used_levels_) {
    std::string var_name = "kNumL" + std::to_string(level);
    std::string env_name = "NUM_L" + std::to_string(level);
    emitter_.EmitLine("const int " + var_name + " = env_int(\"" + env_name + "\", 1);");
  }
  emitter_.EmitLine("");
}

void DistributedCodegen::EmitTraceHelpers() {
  // tpid helper for each used level
  for (int level : used_levels_) {
    std::string func_name = "tpid_l" + std::to_string(level);
    emitter_.EmitLine("int " + func_name + "() { return 0; }");
  }
  emitter_.EmitLine("");
}

void DistributedCodegen::EmitFunction(const ir::FunctionPtr& func) {
  declared_vars_.clear();

  bool is_worker = func->role_.has_value() && *func->role_ == ir::Role::Worker;
  is_worker_context_ = is_worker;

  // Determine return type
  std::string return_type = "void";
  if (!is_worker && !func->return_types_.empty()) {
    return_type = CppTypeForIRType(func->return_types_[0]);
  }

  // Build parameter list
  std::ostringstream params;
  // Add runtime parameter for orchestrators
  if (!is_worker && func->level_.has_value()) {
    int linqu_level = ir::LevelToLinquLevel(*func->level_);
    params << "LevelRuntime& rt_l" << linqu_level;
    if (!func->params_.empty()) {
      params << ", ";
    }
  }

  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    std::string cpp_type = CppTypeForIRType(param->GetType());

    // Workers take tensors by reference, scalars by value
    if (std::dynamic_pointer_cast<const ir::TensorType>(param->GetType())) {
      params << cpp_type << "& " << SanitizeName(param->name_hint_);
    } else {
      params << cpp_type << " " << SanitizeName(param->name_hint_);
    }

    if (i + 1 < func->params_.size()) {
      params << ", ";
    }
  }

  // Emit function signature
  emitter_.EmitLine("static " + return_type + " " + func->name_ + "(" + params.str() + ") {");
  emitter_.IncreaseIndent();

  // Register parameter names
  for (const auto& param : func->params_) {
    declared_vars_.insert(SanitizeName(param->name_hint_));
  }

  // For orchestrators: create runtimes for levels they submit to
  if (!is_worker) {
    std::set<int> needed_runtimes;
    CollectNeededRuntimes(func, needed_runtimes);
    // Remove the function's own level (already received as parameter)
    if (func->level_.has_value()) {
      needed_runtimes.erase(ir::LevelToLinquLevel(*func->level_));
    }
    for (int level : needed_runtimes) {
      std::string rt_var = "rt_l" + std::to_string(level);
      emitter_.EmitLine("LevelRuntime " + rt_var + "(kNumL" + std::to_string(level) + ");");
      declared_vars_.insert(rt_var);
    }
  }

  // Emit body
  if (func->body_) {
    VisitStmt(func->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");
  emitter_.EmitLine("");
  is_worker_context_ = false;
}

void DistributedCodegen::EmitMain() {
  if (!entry_func_) return;

  declared_vars_.clear();

  // Build parameter list
  std::ostringstream params;
  for (size_t i = 0; i < entry_func_->params_.size(); ++i) {
    const auto& param = entry_func_->params_[i];
    std::string cpp_type = CppTypeForIRType(param->GetType());
    params << cpp_type << " " << SanitizeName(param->name_hint_);
    if (i + 1 < entry_func_->params_.size()) {
      params << ", ";
    }
  }

  // Determine return type
  std::string return_type = "void";
  if (!entry_func_->return_types_.empty()) {
    return_type = CppTypeForIRType(entry_func_->return_types_[0]);
  }

  emitter_.EmitLine(return_type + " " + entry_func_->name_ + "(" + params.str() + ") {");
  emitter_.IncreaseIndent();

  // Register parameter names
  for (const auto& param : entry_func_->params_) {
    declared_vars_.insert(SanitizeName(param->name_hint_));
  }

  // Create runtime objects for each used level
  for (int level : used_levels_) {
    std::string rt_var = "rt_l" + std::to_string(level);
    emitter_.EmitLine("LevelRuntime " + rt_var + "(kNumL" + std::to_string(level) + ");");
  }
  if (!used_levels_.empty()) {
    emitter_.EmitLine("");
  }

  // Emit entry function body
  if (entry_func_->body_) {
    VisitStmt(entry_func_->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");
}

// ========================================================================
// Statement visitors
// ========================================================================

void DistributedCodegen::VisitStmt_(const ir::AssignStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null AssignStmt";

  std::string var_name = SanitizeName(op->var_->name_hint_);

  // Set context for expression visitor
  current_target_var_ = var_name;
  current_expr_value_ = "";

  // Check if the value is a Call to a hierarchy function
  if (auto call = std::dynamic_pointer_cast<const ir::Call>(op->value_)) {
    auto gv = std::dynamic_pointer_cast<const ir::GlobalVar>(call->op_);
    if (gv) {
      auto callee = program_->GetFunction(gv->name_);
      if (callee && callee->role_.has_value()) {
        if (*callee->role_ == ir::Role::Worker) {
          EmitCallToWorker(call, callee);
          declared_vars_.insert(var_name);
          current_target_var_ = "";
          return;
        }
        if (*callee->role_ == ir::Role::Orchestrator) {
          EmitCallToOrchestrator(call, callee);
          declared_vars_.insert(var_name);
          current_target_var_ = "";
          return;
        }
      }
    }
  }

  // Standard expression
  VisitExpr(op->value_);

  if (!current_expr_value_.empty()) {
    if (declared_vars_.count(var_name)) {
      emitter_.EmitLine(var_name + " = " + current_expr_value_ + ";");
    } else {
      emitter_.EmitLine("auto " + var_name + " = " + current_expr_value_ + ";");
      declared_vars_.insert(var_name);
    }
    current_expr_value_ = "";
  }

  current_target_var_ = "";
}

void DistributedCodegen::VisitStmt_(const ir::EvalStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";

  current_target_var_ = "";
  current_expr_value_ = "";
  VisitExpr(op->expr_);

  // If the expression produced a value (not emitted as a statement), emit it
  if (!current_expr_value_.empty()) {
    emitter_.EmitLine(current_expr_value_ + ";");
    current_expr_value_ = "";
  }
}

void DistributedCodegen::VisitStmt_(const ir::ReturnStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ReturnStmt";

  // Workers are void — skip return value
  if (is_worker_context_ || op->value_.empty()) {
    return;
  }

  VisitExpr(op->value_[0]);
  emitter_.EmitLine("return " + current_expr_value_ + ";");
  current_expr_value_ = "";
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

  emitter_.EmitLine("for (int " + loop_var + " = " + start + "; " + loop_var + " < " + stop + "; " +
                    loop_var + " += " + step + ") {");
  emitter_.IncreaseIndent();

  if (op->body_) {
    VisitStmt(op->body_);
  }

  emitter_.DecreaseIndent();
  emitter_.EmitLine("}");
}

void DistributedCodegen::VisitStmt_(const ir::IfStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";

  VisitExpr(op->condition_);
  std::string condition = current_expr_value_;
  current_expr_value_ = "";

  emitter_.EmitLine("if (" + condition + ") {");
  emitter_.IncreaseIndent();
  VisitStmt(op->then_body_);
  emitter_.DecreaseIndent();

  if (op->else_body_.has_value()) {
    emitter_.EmitLine("} else {");
    emitter_.IncreaseIndent();
    VisitStmt(*op->else_body_);
    emitter_.DecreaseIndent();
  }

  emitter_.EmitLine("}");
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
    if (callee && callee->role_.has_value()) {
      if (*callee->role_ == ir::Role::Worker) {
        EmitCallToWorker(op, callee);
        return;
      }
      if (*callee->role_ == ir::Role::Orchestrator) {
        EmitCallToOrchestrator(op, callee);
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

  // Regular op call (e.g., add, mul) — emit as function call
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
  current_expr_value_ = op->value_ ? "true" : "false";
}

// ========================================================================
// Call-site lowering
// ========================================================================

void DistributedCodegen::EmitCallToWorker(const ir::CallPtr& call, const ir::FunctionPtr& callee) {
  INTERNAL_CHECK(callee->level_.has_value()) << "Worker function must have a level: " << callee->name_;

  std::string rt_var = RuntimeVarForLevel(*callee->level_);

  // Evaluate all call arguments
  std::vector<std::string> input_names;
  std::vector<std::string> all_arg_strs;

  for (const auto& arg : call->args_) {
    VisitExpr(arg);
    std::string arg_str = current_expr_value_;
    current_expr_value_ = "";
    all_arg_strs.push_back(arg_str);

    // Tensor args are inputs
    if (std::dynamic_pointer_cast<const ir::TensorType>(arg->GetType())) {
      input_names.push_back(arg_str);
    }
  }

  // If there's an assignment target, declare it as the output tensor
  std::string target = current_target_var_;
  std::vector<std::string> output_names;
  if (!target.empty()) {
    // Declare the output variable before submit
    if (!declared_vars_.count(target)) {
      emitter_.EmitLine("LinquTensor " + target + ";");
      declared_vars_.insert(target);
    }
    output_names.push_back(target);
  }

  // Build lambda argument list (include output if present)
  std::ostringstream lambda_args;
  for (size_t i = 0; i < all_arg_strs.size(); ++i) {
    if (i > 0) lambda_args << ", ";
    lambda_args << all_arg_strs[i];
  }

  // Build input/output vectors
  std::ostringstream inputs;
  inputs << "{";
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (i > 0) inputs << ", ";
    inputs << input_names[i];
  }
  inputs << "}";

  std::ostringstream outputs;
  outputs << "{";
  for (size_t i = 0; i < output_names.size(); ++i) {
    if (i > 0) outputs << ", ";
    outputs << output_names[i];
  }
  outputs << "}";

  // Emit: rt_lN.submit_worker("name", [&](){ callee(args); }, {inputs}, {outputs});
  emitter_.EmitLine(rt_var + ".submit_worker(\"" + callee->name_ + "\", [&]() {");
  emitter_.IncreaseIndent();
  emitter_.EmitLine(callee->name_ + "(" + lambda_args.str() + ");");
  emitter_.DecreaseIndent();
  emitter_.EmitLine("}, " + inputs.str() + ", " + outputs.str() + ");");
}

void DistributedCodegen::EmitCallToOrchestrator(const ir::CallPtr& call, const ir::FunctionPtr& callee) {
  INTERNAL_CHECK(callee->level_.has_value()) << "Orchestrator function must have a level: " << callee->name_;

  std::string rt_var = RuntimeVarForLevel(*callee->level_);

  // Build argument list
  std::ostringstream lambda_args;
  // Pass the runtime as first arg to the orchestrator
  lambda_args << rt_var;
  for (const auto& arg : call->args_) {
    VisitExpr(arg);
    lambda_args << ", " << current_expr_value_;
    current_expr_value_ = "";
  }

  // Determine return type
  std::string return_type = "LinquTensor";
  if (!callee->return_types_.empty()) {
    return_type = CppTypeForIRType(callee->return_types_[0]);
  }

  // Emit: [auto target = ] rt_lN.submit_orchestrator("name", [&]()->ReturnType{ return callee(args); });
  std::string target = current_target_var_;
  std::string prefix = target.empty() ? "" : "auto " + target + " = ";
  emitter_.EmitLine(prefix + rt_var + ".submit_orchestrator(\"" + callee->name_ + "\", [&]() -> " +
                    return_type + " {");
  emitter_.IncreaseIndent();
  emitter_.EmitLine("return " + callee->name_ + "(" + lambda_args.str() + ");");
  emitter_.DecreaseIndent();
  emitter_.EmitLine("});");
}

void DistributedCodegen::EmitDistIntrinsic(const ir::CallPtr& call) {
  const auto& op_name = call->op_->name_;

  if (op_name == "dist.tree_reduce") {
    EmitTreeReduce(call);
    return;
  }

  // Fallback for other dist.* ops
  current_expr_value_ = op_name + "(" + FormatArgs(call->args_) + ")";
}

void DistributedCodegen::EmitTreeReduce(const ir::CallPtr& call) {
  // tree_reduce(tensor, reduce_op, level_runtime)
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
    emitter_.EmitLine("tree_reduce(" + args.str() + ");");
  }
}

// ========================================================================
// Helpers
// ========================================================================

namespace {

/// Helper visitor to find GlobalVar calls in a function body
class CallCollector : public ir::IRVisitor {
 public:
  std::set<std::string> called_names_;

  void VisitExpr_(const ir::CallPtr& op) override {
    if (auto gv = std::dynamic_pointer_cast<const ir::GlobalVar>(op->op_)) {
      called_names_.insert(gv->name_);
    }
    // Visit args
    for (const auto& arg : op->args_) {
      VisitExpr(arg);
    }
  }
};

}  // namespace

void DistributedCodegen::CollectNeededRuntimes(const ir::FunctionPtr& func, std::set<int>& needed) const {
  if (!func->body_) return;

  // Walk the function body to find actual calls to program functions
  CallCollector collector;
  collector.VisitStmt(func->body_);

  for (const auto& name : collector.called_names_) {
    auto it = all_funcs_.find(name);
    if (it == all_funcs_.end()) continue;
    const auto& callee = it->second;
    if (callee->level_.has_value()) {
      needed.insert(ir::LevelToLinquLevel(callee->level_.value()));
    }
  }
}

std::string DistributedCodegen::RuntimeVarForLevel(ir::Level level) const {
  return "rt_l" + std::to_string(ir::LevelToLinquLevel(level));
}

std::string DistributedCodegen::CppTypeForIRType(const ir::TypePtr& type) const {
  if (std::dynamic_pointer_cast<const ir::TensorType>(type)) {
    return "LinquTensor";
  }
  if (auto scalar = std::dynamic_pointer_cast<const ir::ScalarType>(type)) {
    return scalar->dtype_.ToCTypeString();
  }
  return "auto";
}

std::string DistributedCodegen::SanitizeName(const std::string& name) const {
  // Replace characters invalid in C++ identifiers (e.g., '.' -> '_')
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
