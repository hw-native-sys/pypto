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

#include "pypto/codegen/orchestration/orchestration_codegen.h"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/dependency_analyzer.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

namespace {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// Format scalar constant as C++ literal/expression for assignment to the given C++ type.
std::string FormatConstIntValue(const ConstIntPtr& c, const std::string& cpp_type) {
  int64_t v = c->value_;
  if (cpp_type == "uint8_t" || cpp_type == "uint16_t" || cpp_type == "uint32_t" || cpp_type == "uint64_t") {
    return "static_cast<" + cpp_type + ">(" + std::to_string(v) + ")";
  }
  if (cpp_type == "int8_t" || cpp_type == "int16_t" || cpp_type == "int32_t") {
    return "static_cast<" + cpp_type + ">(" + std::to_string(v) + ")";
  }
  return std::to_string(v);  // int64_t
}

std::string FormatConstFloatValue(const ConstFloatPtr& c, const std::string& cpp_type) {
  double v = c->value_;
  if (cpp_type == "float") {
    return std::to_string(static_cast<float>(v));
  }
  return std::to_string(v);  // double
}

bool FunctionReturnsTensor(const FunctionPtr& func) {
  if (func->return_types_.empty()) {
    return false;
  }
  return As<TensorType>(func->return_types_[0]) != nullptr;
}

std::string GenerateOrchestrationSignature(const FunctionPtr& func) {
  std::string func_name = "Build";
  if (!func->name_.empty()) {
    func_name += static_cast<char>(std::toupper(static_cast<unsigned char>(func->name_[0])));
    func_name += func->name_.substr(1);
  }
  return "int " + func_name + "(Runtime* runtime, uint64_t* args, int arg_count)";
}

std::string GenerateArgumentValidationCode(const FunctionPtr& func) {
  int tensor_param_count = 0;
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      tensor_param_count++;
    }
  }

  bool has_tensor_return = FunctionReturnsTensor(func);
  int return_tensor_count = has_tensor_return ? 1 : 0;
  int expected_arg_count = (tensor_param_count + return_tensor_count) * 2 + 1;

  std::ostringstream oss;
  oss << "    // Validate argument count\n";
  oss << "    if (arg_count < " << expected_arg_count << ") {\n";
  oss << "        std::cerr << \"Error: Expected at least " << expected_arg_count
      << " args, got \" << arg_count << std::endl;\n";
  oss << "        return -1;\n";
  oss << "    }\n\n";

  return oss.str();
}

std::pair<std::string, std::string> GenerateArgumentExtractionCode(const FunctionPtr& func,
                                                                   bool has_tensor_return,
                                                                   const std::string& return_var_name = "") {
  std::ostringstream oss;
  // Use actual return variable name from IR when available (like params); fallback to "output"
  std::string output_tensor_name = has_tensor_return && !return_var_name.empty() ? return_var_name : "output";

  oss << "    // Extract arguments\n";
  int arg_idx = 0;

  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      oss << "    void* host_" << param->name_ << " = reinterpret_cast<void*>(args[" << arg_idx << "]);\n";
      arg_idx += 1;
    }
  }

  if (has_tensor_return) {
    oss << "    void* host_" << output_tensor_name << " = reinterpret_cast<void*>(args[" << arg_idx
        << "]);\n";
    arg_idx += 1;
  }

  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      oss << "    size_t size_" << param->name_ << " = static_cast<size_t>(args[" << arg_idx << "]);\n";
      arg_idx += 1;
    }
  }

  if (has_tensor_return) {
    oss << "    size_t size_" << output_tensor_name << " = static_cast<size_t>(args[" << arg_idx << "]);\n";
    arg_idx += 1;
  }

  oss << "\n";

  return {oss.str(), output_tensor_name};
}

std::string CalculateTensorSize(const TensorTypePtr& tensor_type) {
  std::ostringstream oss;

  // Calculate total number of elements by multiplying all dimensions
  bool first = true;
  for (const auto& dim : tensor_type->shape_) {
    if (auto const_int = As<ConstInt>(dim)) {
      if (first) {
        oss << const_int->value_;
        first = false;
      } else {
        oss << " * " << const_int->value_;
      }
    } else {
      throw RuntimeError("Orchestration codegen requires constant tensor shapes");
    }
  }

  // If shape is empty, it's a scalar (1 element)
  if (first) {
    oss << "1";
  }

  // Multiply by element size in bytes
  size_t element_bits = tensor_type->dtype_.GetBit();
  size_t element_bytes = (element_bits + 7) / 8;  // Round up to nearest byte
  oss << " * " << element_bytes;

  return oss.str();
}

std::string GenerateDeviceMemoryAllocationCode(
    const ProgramPtr& program, const FunctionPtr& func, const std::set<std::string>& output_tensors,
    bool has_tensor_return, const std::string& output_tensor_name,
    const std::set<std::string>& intermediate_tensors,
    const std::map<std::string, AssignStmtPtr>& output_tensor_assigns) {
  std::ostringstream oss;

  oss << "    // Allocate device memory for parameters\n";
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      oss << "    void* dev_" << param->name_ << " = runtime->host_api.device_malloc(size_" << param->name_
          << ");\n";

      if (output_tensors.count(param->name_)) {
        oss << "    runtime->record_tensor_pair(host_" << param->name_ << ", dev_" << param->name_
            << ", size_" << param->name_ << ");\n";
      } else {
        oss << "    runtime->host_api.copy_to_device(dev_" << param->name_ << ", host_" << param->name_
            << ", size_" << param->name_ << ");\n";
      }
    }
  }

  if (has_tensor_return) {
    oss << "    void* dev_" << output_tensor_name << " = runtime->host_api.device_malloc(size_"
        << output_tensor_name << ");\n";
    oss << "    runtime->record_tensor_pair(host_" << output_tensor_name << ", dev_" << output_tensor_name
        << ", size_" << output_tensor_name << ");\n";
  }

  oss << "\n";

  if (!intermediate_tensors.empty()) {
    oss << "    // Allocate device memory for intermediate tensors\n";
    for (const auto& tensor_name : intermediate_tensors) {
      // Get the AssignStmt to find the Call
      auto it = output_tensor_assigns.find(tensor_name);
      CHECK(it != output_tensor_assigns.end())
          << "Missing assignment info for intermediate tensor: " << tensor_name;

      // Get the Call expression to find the callee function
      auto call = As<Call>(it->second->value_);
      CHECK(call) << "Intermediate tensor assignment must be from a Call: " << tensor_name;

      // Find the called function in the program
      std::string callee_name = call->op_->name_;
      FunctionPtr callee_func = program->GetFunction(callee_name);
      CHECK(callee_func) << "Cannot find called function: " << callee_name;

      // Get the return type from the called function
      CHECK(!callee_func->return_types_.empty()) << "Function " << callee_name << " has no return type";
      auto tensor_type = As<TensorType>(callee_func->return_types_[0]);
      CHECK(tensor_type) << "Function " << callee_name << " must return TensorType, got "
                         << callee_func->return_types_[0]->TypeName();

      std::string size_expr = CalculateTensorSize(tensor_type);
      oss << "    size_t size_" << tensor_name << " = " << size_expr << ";\n";
      oss << "    void* dev_" << tensor_name << " = runtime->host_api.device_malloc(size_" << tensor_name
          << ");\n";
    }
    oss << "\n";
  }

  return oss.str();
}

std::string GenerateSingleTaskCode(const std::string& task_var, const std::vector<std::string>& task_args,
                                   const std::vector<std::string>& task_arg_cpp_types,
                                   const std::string& callee_name, int func_id, CoreType core_type,
                                   int task_counter) {
  std::ostringstream oss;

  oss << "    // Task " << task_counter << ": Call " << callee_name << "\n";
  oss << "    uint64_t args_" << task_var << "[" << task_args.size() << "];\n";
  for (size_t i = 0; i < task_args.size(); ++i) {
    const std::string& cpp_type = task_arg_cpp_types[i];
    const std::string& value = task_args[i];
    if (cpp_type == "void*") {
      oss << "    args_" << task_var << "[" << i << "] = reinterpret_cast<uint64_t>(" << value << ");\n";
    } else {
      oss << "    { union { " << cpp_type << " v; uint64_t u; } _u; _u.v = " << value << "; args_" << task_var
          << "[" << i << "] = _u.u; }\n";
    }
  }
  const std::string core_type_str = core_type == CoreType::CUBE ? "CoreType::AIC" : "CoreType::AIV";
  oss << "    int " << task_var << " = runtime->add_task(args_" << task_var << ", " << task_args.size()
      << ", " << func_id << ", " << core_type_str << ");\n\n";

  return oss.str();
}

void ValidateOrchestrationReferences(const ProgramPtr& program, const FunctionPtr& func) {
  CHECK(func->func_type_ == FunctionType::Orchestration)
      << "ValidateOrchestrationReferences should only be called on Orchestration functions";

  class FunctionCallCollector : public IRVisitor {
   public:
    std::set<std::string> called_functions_;

    void VisitExpr_(const CallPtr& call) override {
      if (call->op_->name_.find("block.") != 0) {
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

CallPtr ExtractCallFromStmt(const StmtPtr& stmt) {
  if (!stmt) {
    return nullptr;
  }
  if (auto assign = As<AssignStmt>(stmt)) {
    return As<Call>(assign->value_);
  }
  if (auto eval = As<EvalStmt>(stmt)) {
    return As<Call>(eval->expr_);
  }
  return nullptr;
}

std::string GenerateTaskDependenciesCode(const FunctionPtr& func,
                                         const std::map<const Call*, int>& call_to_task,
                                         const std::vector<std::string>& task_vars) {
  INTERNAL_CHECK(func) << "Internal error: Cannot analyze dependencies for null function";

  ir::DependencyAnalyzer analyzer;
  auto all_dependencies = analyzer.AnalyzeDependencies(func);

  std::ostringstream oss;
  bool has_deps = false;

  for (const auto& edge : all_dependencies) {
    if (edge.type != ir::DependencyEdge::RAW) {
      continue;
    }

    auto producer_call = ExtractCallFromStmt(edge.producer);
    auto consumer_call = ExtractCallFromStmt(edge.consumer);

    if (producer_call && consumer_call && call_to_task.count(producer_call.get()) &&
        call_to_task.count(consumer_call.get())) {
      int producer_idx = call_to_task.at(producer_call.get());
      int consumer_idx = call_to_task.at(consumer_call.get());

      if (!has_deps) {
        oss << "    // Dependencies (data-flow based)\n";
        has_deps = true;
      }
      oss << "    runtime->add_successor(" << task_vars[producer_idx] << ", " << task_vars[consumer_idx]
          << ");\n";
    }
  }

  if (has_deps) {
    oss << "\n";
  }

  return oss.str();
}

int GetOrCreateFuncId(const std::string& func_name, std::map<std::string, int>* func_name_to_id,
                      int* next_func_id) {
  if (func_name_to_id->find(func_name) == func_name_to_id->end()) {
    (*func_name_to_id)[func_name] = (*next_func_id)++;
  }
  return (*func_name_to_id)[func_name];
}

}  // namespace

CoreType InferFunctionCoreType(const FunctionPtr& func) {
  class CoreTypeCollector : public IRVisitor {
   public:
    std::set<PipeType> pipe_types_;

    void VisitExpr_(const CallPtr& call) override {
      if (auto pipe_type = call->op_->GetPipe()) {
        pipe_types_.insert(*pipe_type);
      }
      IRVisitor::VisitExpr_(call);
    }
  };

  CoreTypeCollector collector;
  collector.VisitStmt(func->body_);

  bool has_m = collector.pipe_types_.count(PipeType::M) > 0;
  bool has_v = collector.pipe_types_.count(PipeType::V) > 0;

  CHECK(!(has_m && has_v)) << "Function " << func->name_
                           << " contains both Matrix (M) and Vector (V) pipe types. "
                           << "A function can only use one core type (CUBE or VECTOR).";

  if (has_m) {
    return CoreType::CUBE;
  }
  if (has_v) {
    return CoreType::VECTOR;
  }
  return CoreType::VECTOR;
}

OrchestrationResult GenerateOrchestration(const ir::ProgramPtr& program, const ir::FunctionPtr& func) {
  using namespace pypto::ir;  // NOLINT(build/namespaces)

  CHECK(program != nullptr) << "Cannot generate orchestration for null program";
  CHECK(func != nullptr) << "Cannot generate orchestration for null function";

  ValidateOrchestrationReferences(program, func);

  std::map<std::string, int> func_name_to_id;
  std::map<std::string, CoreType> func_name_to_core_type;
  int next_func_id = 0;

  std::ostringstream oss;

  bool has_tensor_return = FunctionReturnsTensor(func);

  // Collect info from IR first (return_var, output_tensors, etc.) so we can use actual names
  class OrchestrationInfoCollector : public IRVisitor {
   public:
    std::string return_var;
    std::set<std::string> output_tensors;
    std::vector<CallPtr> function_calls;
    std::map<const Call*, std::string> call_to_result_var;
    std::map<std::string, AssignStmtPtr> output_tensor_assigns;  // Store AssignStmt for type info

    void VisitStmt_(const ReturnStmtPtr& ret) override {
      if (!ret->value_.empty()) {
        if (auto var = As<Var>(ret->value_[0])) {
          return_var = var->name_;
        }
      }
      IRVisitor::VisitStmt_(ret);
    }

    void VisitStmt_(const AssignStmtPtr& assign) override {
      if (auto call = As<Call>(assign->value_)) {
        output_tensors.insert(assign->var_->name_);
        call_to_result_var[call.get()] = assign->var_->name_;
        output_tensor_assigns[assign->var_->name_] = assign;

        if (call->op_->name_.find("block.") != 0) {
          function_calls.push_back(call);
        }
      }
      IRVisitor::VisitStmt_(assign);
    }
  };

  OrchestrationInfoCollector info_collector;
  info_collector.VisitStmt(func->body_);

  std::string return_var_name = info_collector.return_var;

  oss << "#include <cstdint>\n\n";
  oss << "extern \"C\" {\n\n";
  oss << GenerateOrchestrationSignature(func) << " {\n";
  oss << GenerateArgumentValidationCode(func);

  auto [arg_extraction_code, output_tensor_name] =
      GenerateArgumentExtractionCode(func, has_tensor_return, return_var_name);
  oss << arg_extraction_code;
  const std::set<std::string>& output_tensors = info_collector.output_tensors;

  std::set<std::string> param_names;
  for (const auto& param : func->params_) {
    param_names.insert(param->name_);
  }

  std::set<std::string> intermediate_tensors;
  for (const auto& var_name : output_tensors) {
    if (param_names.find(var_name) == param_names.end() && var_name != return_var_name) {
      intermediate_tensors.insert(var_name);
    }
  }

  oss << GenerateDeviceMemoryAllocationCode(program, func, output_tensors, has_tensor_return,
                                            output_tensor_name, intermediate_tensors,
                                            info_collector.output_tensor_assigns);

  const std::vector<CallPtr>& call_ops = info_collector.function_calls;

  if (!call_ops.empty()) {
    oss << "    // Function ID mapping:\n";
    std::set<std::string> seen_functions;
    for (const auto& call : call_ops) {
      std::string callee_name = call->op_->name_;
      if (seen_functions.find(callee_name) == seen_functions.end()) {
        int func_id = GetOrCreateFuncId(callee_name, &func_name_to_id, &next_func_id);
        oss << "    //   " << func_id << ": " << callee_name << "\n";
        seen_functions.insert(callee_name);
      }
    }
    oss << "\n";
  }

  std::vector<std::string> task_vars;
  std::map<const Call*, int> call_to_task;
  int task_counter = 0;

  const std::map<const Call*, std::string>& call_to_result_var = info_collector.call_to_result_var;

  for (const auto& call : call_ops) {
    std::string callee_name = call->op_->name_;

    FunctionPtr callee_func = program->GetFunction(callee_name);
    INTERNAL_CHECK(callee_func != nullptr)
        << "Internal error: function '" << callee_name
        << "' not found after validation. This should have been caught earlier.";
    CoreType core_type = InferFunctionCoreType(callee_func);
    func_name_to_core_type[callee_name] = core_type;

    int func_id = GetOrCreateFuncId(callee_name, &func_name_to_id, &next_func_id);

    std::string result_var;
    if (call_to_result_var.count(call.get())) {
      result_var = call_to_result_var.at(call.get());
    }

    std::vector<std::string> task_args;
    std::vector<std::string> task_arg_cpp_types;
    for (const auto& arg : call->args_) {
      if (auto var = As<Var>(arg)) {
        task_args.emplace_back("dev_" + var->name_);
        task_arg_cpp_types.emplace_back("void*");
      } else if (auto const_int = As<ConstInt>(arg)) {
        std::string cpp_type = const_int->dtype().ToCTypeString();
        task_arg_cpp_types.emplace_back(cpp_type);
        task_args.emplace_back(FormatConstIntValue(const_int, cpp_type));
      } else if (auto const_float = As<ConstFloat>(arg)) {
        std::string cpp_type = const_float->dtype().ToCTypeString();
        task_arg_cpp_types.emplace_back(cpp_type);
        task_args.emplace_back(FormatConstFloatValue(const_float, cpp_type));
      } else if (auto const_bool = As<ConstBool>(arg)) {
        task_arg_cpp_types.emplace_back("bool");
        task_args.emplace_back(const_bool->value_ ? "true" : "false");
      }
    }

    if (!result_var.empty()) {
      if (result_var == return_var_name && has_tensor_return) {
        task_args.emplace_back("dev_" + output_tensor_name);
      } else {
        task_args.emplace_back("dev_" + result_var);
      }
      task_arg_cpp_types.emplace_back("void*");
    }

    std::string task_var = "t" + std::to_string(task_counter);
    task_vars.push_back(task_var);
    call_to_task[call.get()] = task_counter;

    oss << GenerateSingleTaskCode(task_var, task_args, task_arg_cpp_types, callee_name, func_id, core_type,
                                  task_counter);

    task_counter++;
  }

  oss << GenerateTaskDependenciesCode(func, call_to_task, task_vars);

  oss << "    return 0;\n";
  oss << "}\n\n";
  oss << "}  // extern \"C\"\n";

  return OrchestrationResult{oss.str(), std::move(func_name_to_id), std::move(func_name_to_core_type)};
}

}  // namespace codegen
}  // namespace pypto
