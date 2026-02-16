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

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

namespace {

using namespace pypto::ir;  // NOLINT(build/namespaces)

/**
 * @brief Check if an operation is a built-in IR operation (not a user-defined function)
 *
 * Built-in operations include block-level ops (block.*), tensor-level ops (tensor.*),
 * and system ops (system.*). These are handled by specialized codegen paths rather
 * than being dispatched as task graph function calls.
 */
bool IsBuiltinOp(const std::string& op_name) {
  return op_name.find("block.") == 0 || op_name.find("tensor.") == 0 || op_name.find("system.") == 0;
}

/**
 * @brief Check if an operation is a tensor-level IR operation
 *
 * Tensor operations (tensor.create, tensor.read, tensor.view, tensor.reshape)
 * are host-side operations. Only tensor.create and tensor.read require inline C++ codegen;
 * others are metadata-only and expressed through TensorType parameters.
 */
bool IsTensorOp(const std::string& op_name) { return op_name.find("tensor.") == 0; }

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

int CountReturnTensors(const FunctionPtr& func) {
  int count = 0;
  for (const auto& rt : func->return_types_) {
    if (As<TensorType>(rt)) {
      count++;
    }
  }
  return count;
}

std::string CalculateTensorSize(const TensorTypePtr& tensor_type) {
  std::ostringstream oss;

  // Calculate total number of elements by multiplying all dimensions
  bool first = true;
  for (const auto& dim : tensor_type->shape_) {
    if (first) {
      oss << CodegenBase::GenerateExprString(dim);
      first = false;
    } else {
      oss << " * " << CodegenBase::GenerateExprString(dim);
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

// Helper: resolve the TensorType for an intermediate tensor, handling both
// single-return and tuple-return cases.
TensorTypePtr GetIntermediateTensorType(
    const ProgramPtr& program, const std::map<std::string, AssignStmtPtr>& output_tensor_assigns,
    const std::map<std::string, std::pair<std::string, int>>& tuple_element_map,
    const std::string& tensor_name) {
  std::string assign_key;
  int return_index = 0;

  auto tuple_it = tuple_element_map.find(tensor_name);
  if (tuple_it != tuple_element_map.end()) {
    assign_key = tuple_it->second.first;
    return_index = tuple_it->second.second;
  } else {
    assign_key = tensor_name;
  }

  auto assign_it = output_tensor_assigns.find(assign_key);
  CHECK(assign_it != output_tensor_assigns.end()) << "Missing assignment info for tensor: " << assign_key;

  auto call = As<Call>(assign_it->second->value_);
  CHECK(call) << "Tensor assignment must be from a Call: " << assign_key;

  std::string callee_name = call->op_->name_;

  // tensor.create: type comes directly from the call's return type
  if (callee_name == "tensor.create") {
    auto tensor_type = As<TensorType>(call->GetType());
    CHECK(tensor_type) << "tensor.create must return TensorType";
    return tensor_type;
  }

  FunctionPtr callee_func = program->GetFunction(callee_name);
  CHECK(callee_func) << "Cannot find called function: " << callee_name;
  CHECK(return_index < static_cast<int>(callee_func->return_types_.size()))
      << "Return index " << return_index << " out of bounds for function " << callee_name;

  auto tensor_type = As<TensorType>(callee_func->return_types_[return_index]);
  CHECK(tensor_type) << "Function " << callee_name << " return type at index " << return_index
                     << " must be TensorType, got " << callee_func->return_types_[return_index]->TypeName();
  return tensor_type;
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

// Collect metadata from IR (return_vars, output_tensors, tuple info) for memory allocation
class OrchestrationInfoCollector : public IRVisitor {
 public:
  std::vector<std::string> return_vars;
  std::set<std::string> output_tensors;
  std::map<const Call*, std::string> call_to_result_var;
  std::map<std::string, AssignStmtPtr> output_tensor_assigns;            // Store AssignStmt for type info
  std::map<std::string, std::pair<std::string, int>> tuple_element_map;  // var -> (tuple_var, index)
  std::set<std::string> tuple_temp_vars;  // Tuple temporary variables (not real tensors)

  void VisitStmt_(const ReturnStmtPtr& ret) override {
    for (const auto& val : ret->value_) {
      std::string name = CodegenBase::TryGetVarName(val);
      if (!name.empty()) {
        return_vars.push_back(name);
      }
    }
    IRVisitor::VisitStmt_(ret);
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (auto call = As<Call>(assign->value_)) {
      if (call->op_->name_ == "tensor.create") {
        // tensor.create produces a local tensor that needs make_tensor allocation
        std::string var_name = assign->var_->name_;
        output_tensors.insert(var_name);
        output_tensor_assigns[var_name] = assign;
      } else if (!IsBuiltinOp(call->op_->name_)) {
        std::string var_name = assign->var_->name_;

        // Check if this call returns a TupleType
        if (As<TupleType>(call->GetType())) {
          // Tuple-returning call: mark as tuple temp, don't add to output_tensors
          tuple_temp_vars.insert(var_name);
          call_to_result_var[call.get()] = var_name;
          output_tensor_assigns[var_name] = assign;
        } else {
          // Single tensor return (existing behavior)
          output_tensors.insert(var_name);
          call_to_result_var[call.get()] = var_name;
          output_tensor_assigns[var_name] = assign;
        }
      }
    } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
      // Handle: mi = TupleGetItemExpr(_tuple_tmp, 0)
      std::string var_name = assign->var_->name_;
      std::string tuple_var_name = CodegenBase::TryGetVarName(tuple_get->tuple_);

      if (!tuple_var_name.empty() && tuple_temp_vars.count(tuple_var_name)) {
        tuple_element_map[var_name] = {tuple_var_name, tuple_get->index_};
        output_tensors.insert(var_name);
        output_tensor_assigns[var_name] = assign;
      }
    }
    IRVisitor::VisitStmt_(assign);
  }
};

}  // namespace

CoreType InferFunctionCoreType(const FunctionPtr& func) {
  const backend::Backend* backend = backend::GetBackend();
  class CoreTypeCollector : public IRVisitor {
   public:
    explicit CoreTypeCollector(const backend::Backend* backend) : backend_(backend) {}
    std::set<PipeType> pipe_types_;

    void VisitExpr_(const CallPtr& call) override {
      if (call->op_->GetPipe().has_value()) {
        pipe_types_.insert(*call->op_->GetPipe());
      } else if (backend_ != nullptr) {
        const auto* info = backend_->GetOpInfo(call->op_->name_);
        if (info) {
          pipe_types_.insert(info->pipe);
        }
      }
      IRVisitor::VisitExpr_(call);
    }

   private:
    const backend::Backend* backend_;
  };

  CoreTypeCollector collector(backend);
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

namespace {

std::string GenerateIncludes() {
  std::ostringstream oss;
  oss << "#include <stddef.h>\n";
  oss << "#include <stdint.h>\n";
  oss << "#include <stdio.h>\n\n";
  oss << "#include \"pto_orchestration_api.h\"\n\n";
  return oss.str();
}

std::string GenerateArgDefines(const FunctionPtr& func, const std::vector<std::string>& return_var_names) {
  std::ostringstream oss;
  int idx = 0;

  // Pointer defines for input tensor params
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      std::string upper_name = param->name_;
      for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
      oss << "#define ARG_PTR_" << upper_name << " " << idx++ << "\n";
    }
  }
  // Pointer defines for return tensors
  for (const auto& name : return_var_names) {
    std::string upper_name = name;
    for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    oss << "#define ARG_PTR_" << upper_name << " " << idx++ << "\n";
  }

  oss << "\n";

  // Size defines for input tensor params
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      std::string upper_name = param->name_;
      for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
      oss << "#define ARG_SIZE_" << upper_name << " " << idx++ << "\n";
    }
  }
  // Size defines for return tensors
  for (const auto& name : return_var_names) {
    std::string upper_name = name;
    for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    oss << "#define ARG_SIZE_" << upper_name << " " << idx++ << "\n";
  }

  oss << "\n";
  return oss.str();
}

std::string GenerateHelperFunctions() {
  std::ostringstream oss;
  oss << "// Helper to encode float as uint64_t for scalar params\n";
  oss << "static uint64_t float_to_u64(float f) {\n";
  oss << "    union {\n";
  oss << "        float f32;\n";
  oss << "        uint64_t u64;\n";
  oss << "    } conv;\n";
  oss << "    conv.u64 = 0;  // Clear upper bits\n";
  oss << "    conv.f32 = f;\n";
  oss << "    return conv.u64;\n";
  oss << "}\n\n";
  return oss.str();
}

int CountExpectedArgs(const FunctionPtr& func, int return_tensor_count) {
  int tensor_param_count = 0;
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      tensor_param_count++;
    }
  }
  // pointers + sizes for all tensors (params + returns)
  return (tensor_param_count + return_tensor_count) * 2;
}

std::string GenerateConfigFunction(int expected_arg_count) {
  std::ostringstream oss;
  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {\n";
  oss << "    (void)args;\n";
  oss << "    (void)arg_count;\n";
  oss << "    return PTO2OrchestrationConfig{\n";
  oss << "        .expected_arg_count = " << expected_arg_count << ",\n";
  oss << "    };\n";
  oss << "}\n\n";
  return oss.str();
}

std::string CoreTypeToWorker(CoreType core_type) {
  return core_type == CoreType::CUBE ? "PTO2_WORKER_CUBE" : "PTO2_WORKER_VECTOR";
}

}  // namespace

using namespace pypto::ir;  // NOLINT(build/namespaces)

// Record of a generated task for scope analysis
struct TaskRecord {
  int task_id;
  std::string code;                      // Generated C++ code for this task
  std::set<std::string> input_tensors;   // Tensor names read by this task (original names, not ext_)
  std::set<std::string> output_tensors;  // Tensor names written by this task (original names, not ext_)
};

// Statement code generator for orchestration
class OrchestrationStmtCodegen : public CodegenBase {
 public:
  explicit OrchestrationStmtCodegen(const ProgramPtr& prog, std::map<std::string, int>* func_ids,
                                    std::map<std::string, CoreType>* core_types, int* next_id,
                                    const std::set<std::string>& param_names,
                                    const std::set<std::string>& return_names)
      : program_(prog),
        func_name_to_id_(func_ids),
        func_name_to_core_type_(core_types),
        next_func_id_(next_id),
        param_names_(param_names),
        return_names_(return_names) {}

  void SetTupleElementMap(const std::map<std::string, std::pair<std::string, int>>& map) {
    for (const auto& [var_name, pair] : map) {
      const auto& [tuple_var, index] = pair;
      tuple_var_to_elements_[tuple_var].emplace_back(index, var_name);
    }
    for (auto& [key, vec] : tuple_var_to_elements_) {
      std::sort(vec.begin(), vec.end());
    }
  }

  std::string GetGeneratedCode() const { return code_.str(); }

  // Get per-task records for scope analysis
  const std::vector<TaskRecord>& GetTaskRecords() const { return task_records_; }

  // Get non-task code (e.g., for loops, scalar assignments)
  std::string GetNonTaskCode() const { return non_task_code_.str(); }

  // --- CodegenBase pure virtual implementations ---
  [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_result_var_; }
  void Emit(const std::string& line) override { code_ << line; }
  std::string GetExprAsCode(const ExprPtr& expr) override { return GenerateExprString(expr); }
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override {
    return dtype.ToCTypeString();
  }
  int64_t GetConstIntValue(const ExprPtr& expr) override {
    auto ci = As<ConstInt>(expr);
    INTERNAL_CHECK(ci) << "Internal error: expected ConstInt expression";
    return ci->value_;
  }
  std::string GetVarName(const VarPtr& var) override { return var->name_; }
  [[nodiscard]] std::string GetTensorDataPtr(const std::string& name) const override {
    if (param_names_.count(name) || return_names_.count(name)) {
      return "arg_" + name + "_ptr";
    }
    return name + ".data";
  }

  void VisitStmt_(const ForStmtPtr& for_stmt) override {
    std::string loop_var = for_stmt->loop_var_->name_;
    std::string start_expr = GenerateExprString(for_stmt->start_);
    std::string stop_expr = GenerateExprString(for_stmt->stop_);
    std::string step_expr = GenerateExprString(for_stmt->step_);

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      const auto& iter_arg = for_stmt->iter_args_[i];
      const auto& return_var = for_stmt->return_vars_[i];
      std::string init_value = GenerateExprString(iter_arg->initValue_);
      code_ << Indent() << GetCppType(iter_arg->GetType()) << " " << return_var->name_ << " = " << init_value
            << ";\n";
      iter_arg_to_var_[iter_arg.get()] = return_var->name_;
    }

    code_ << Indent() << "for (int64_t " << loop_var << " = " << start_expr << "; " << loop_var << " < "
          << stop_expr << "; " << loop_var << " += " << step_expr << ") {\n";
    indent_ += 4;

    auto saved = current_return_var_names_;
    current_return_var_names_.clear();
    for (const auto& rv : for_stmt->return_vars_) {
      current_return_var_names_.push_back(rv->name_);
    }
    VisitStmt(for_stmt->body_);
    current_return_var_names_ = saved;

    indent_ -= 4;
    code_ << Indent() << "}\n";
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    std::string var_name = assign->var_->name_;

    if (auto call = As<Call>(assign->value_)) {
      const std::string& op_name = call->op_->name_;
      if (IsTensorOp(op_name)) {
        GenerateTensorOpCode(call, var_name);
      } else if (!IsBuiltinOp(op_name)) {
        GenerateFunctionCallCode(call, var_name);
      }
    } else if (As<TupleGetItemExpr>(assign->value_)) {
      // No-op: tuple elements handled via tuple_var_to_elements_
    } else {
      std::string value_expr = GenerateExprString(assign->value_);
      code_ << Indent() << GetCppType(assign->var_->GetType()) << " " << var_name << " = " << value_expr
            << ";\n";
    }
  }

  void VisitStmt_(const ReturnStmtPtr& ret) override {
    // No-op: return tensors are already make_tensor_external
  }

  void VisitStmt_(const SeqStmtsPtr& seq) override {
    for (const auto& stmt : seq->stmts_) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const YieldStmtPtr& yield_stmt) override {
    for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
      std::string value_expr = GenerateExprString(yield_stmt->value_[i]);
      if (i < current_return_var_names_.size()) {
        code_ << Indent() << current_return_var_names_[i] << " = " << value_expr << ";\n";
      }
    }
  }

  void VisitStmt_(const EvalStmtPtr& eval) override {
    if (auto call = As<Call>(eval->expr_)) {
      const std::string& op_name = call->op_->name_;
      if (IsTensorOp(op_name)) {
        GenerateTensorOpCode(call, "");
      } else if (!IsBuiltinOp(op_name)) {
        GenerateFunctionCallCode(call, "");
      }
    }
  }

 private:
  std::string Indent() const { return std::string(indent_, ' '); }

  std::string GetCppType(const TypePtr& type) {
    if (auto scalar_type = As<ScalarType>(type)) {
      return scalar_type->dtype_.ToCTypeString();
    }
    return "auto";
  }

  // Get the external tensor name (ext_ prefix for external tensors)
  std::string GetExternalTensorName(const std::string& name) {
    if (param_names_.count(name) || return_names_.count(name)) {
      return "ext_" + name;
    }
    return name;
  }

  void GenerateTensorOpCode(const CallPtr& call, const std::string& result_var) {
    const std::string& op_name = call->op_->name_;

    auto& registry = OrchestrationOpRegistry::GetInstance();
    auto codegen_func = registry.Get(op_name);
    if (!codegen_func.has_value()) {
      // Metadata-only ops (tensor.view, tensor.reshape, etc.) have no codegen
      return;
    }

    current_result_var_ = result_var;
    std::string gen_code = (*codegen_func)(call, *this);

    // tensor.create declarations are handled by scope-aware emit_tensor_decls, skip here
    if (op_name == "tensor.create") {
      return;
    }

    // Non-task code (tensor.read, tensor.dim) goes to non_task_code_ stream
    std::istringstream iss(gen_code);
    std::string line;
    while (std::getline(iss, line)) {
      if (!line.empty()) {
        non_task_code_ << Indent() << line << "\n";
      }
    }
  }

  void GenerateFunctionCallCode(const CallPtr& call, const std::string& result_var) {
    const std::string& callee_name = call->op_->name_;

    FunctionPtr callee_func = program_->GetFunction(callee_name);
    INTERNAL_CHECK(callee_func != nullptr)
        << "Internal error: function '" << callee_name << "' not found after validation.";
    CoreType core_type = InferFunctionCoreType(callee_func);
    (*func_name_to_core_type_)[callee_name] = core_type;

    int func_id = GetOrCreateFuncId(callee_name, func_name_to_id_, next_func_id_);

    // Collect output tensor names for this call
    std::set<std::string> output_tensor_names;
    auto tuple_it = tuple_var_to_elements_.find(result_var);
    if (tuple_it != tuple_var_to_elements_.end()) {
      for (const auto& [index, elem_var] : tuple_it->second) {
        output_tensor_names.insert(elem_var);
      }
    } else if (!result_var.empty()) {
      output_tensor_names.insert(result_var);
    }

    // Build PTOParam entries
    struct ParamEntry {
      std::string kind;  // "make_input_param", "make_output_param", "make_scalar_param"
      std::string value;
    };
    std::vector<ParamEntry> params;

    // Track tensor names for scope analysis (original names, not ext_ prefixed)
    std::set<std::string> task_input_tensors;
    std::set<std::string> task_output_tensors;

    // Input args
    for (const auto& arg : call->args_) {
      std::string var_name = TryGetVarName(arg);
      if (!var_name.empty()) {
        std::string ext_name = GetExternalTensorName(var_name);
        task_input_tensors.insert(var_name);
        if (output_tensor_names.count(var_name)) {
          // Same tensor appears as both input and output -> inout
          params.push_back({"make_inout_param", ext_name});
        } else {
          params.push_back({"make_input_param", ext_name});
        }
      } else if (auto const_int = As<ConstInt>(arg)) {
        std::string cpp_type = const_int->dtype().ToCTypeString();
        std::string value = FormatConstIntValue(const_int, cpp_type);
        params.push_back({"make_scalar_param", "(uint64_t)" + value});
      } else if (auto const_float = As<ConstFloat>(arg)) {
        std::string cpp_type = const_float->dtype().ToCTypeString();
        std::string value = FormatConstFloatValue(const_float, cpp_type);
        if (cpp_type == "float") {
          params.push_back({"make_scalar_param", "float_to_u64(" + value + "f)"});
        } else {
          params.push_back({"make_scalar_param", "(uint64_t)" + value});
        }
      } else if (auto const_bool = As<ConstBool>(arg)) {
        params.push_back({"make_scalar_param", const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0"});
      }
    }

    // Output args (only those not already added as inout)
    if (tuple_it != tuple_var_to_elements_.end()) {
      for (const auto& [index, elem_var] : tuple_it->second) {
        std::string ext_name = GetExternalTensorName(elem_var);
        task_output_tensors.insert(elem_var);
        // Check if already added as inout
        bool already_added = false;
        for (const auto& p : params) {
          if (p.kind == "make_inout_param" && p.value == ext_name) {
            already_added = true;
            break;
          }
        }
        if (!already_added) {
          params.push_back({"make_output_param", ext_name});
        }
      }
    } else if (!result_var.empty()) {
      std::string ext_name = GetExternalTensorName(result_var);
      task_output_tensors.insert(result_var);
      bool already_added = false;
      for (const auto& p : params) {
        if (p.kind == "make_inout_param" && p.value == ext_name) {
          already_added = true;
          break;
        }
      }
      if (!already_added) {
        params.push_back({"make_output_param", ext_name});
      }
    }

    // Generate PTOParam array and submit_task into a temporary buffer
    std::ostringstream task_code;
    std::string ind = Indent();
    std::string task_var = "params_t" + std::to_string(task_counter_);
    task_code << "\n";
    task_code << ind << "// Task " << task_counter_ << ": " << callee_name << "\n";
    task_code << ind << "PTOParam " << task_var << "[] = {\n";
    for (const auto& p : params) {
      task_code << ind << "    " << p.kind << "(" << p.value << "),\n";
    }
    task_code << ind << "};\n";
    task_code << ind << "pto2_rt_submit_task(rt, " << func_id << ", " << CoreTypeToWorker(core_type) << ", \""
              << callee_name << "\", " << task_var << ", " << params.size() << ");\n";

    // Record task info for scope analysis
    task_records_.push_back(
        {task_counter_, task_code.str(), std::move(task_input_tensors), std::move(task_output_tensors)});

    // Also write to main code stream (for backward compat with GetGeneratedCode)
    code_ << task_code.str();

    task_counter_++;
  }

  const ProgramPtr& program_;
  std::map<std::string, int>* func_name_to_id_;
  std::map<std::string, CoreType>* func_name_to_core_type_;
  int* next_func_id_;
  const std::set<std::string>& param_names_;
  const std::set<std::string>& return_names_;
  std::ostringstream code_;
  std::ostringstream non_task_code_;
  int indent_ = 4;
  std::map<const IterArg*, std::string> iter_arg_to_var_;
  std::string current_result_var_;
  std::vector<std::string> current_return_var_names_;
  int task_counter_ = 0;
  std::map<std::string, std::vector<std::pair<int, std::string>>> tuple_var_to_elements_;
  std::vector<TaskRecord> task_records_;
};

OrchestrationResult GenerateOrchestration(const ir::ProgramPtr& program, const ir::FunctionPtr& func) {
  using namespace pypto::ir;  // NOLINT(build/namespaces)

  CHECK(program != nullptr) << "Cannot generate orchestration for null program";
  CHECK(func != nullptr) << "Cannot generate orchestration for null function";

  ValidateOrchestrationReferences(program, func);

  std::map<std::string, int> func_name_to_id;
  std::map<std::string, CoreType> func_name_to_core_type;
  int next_func_id = 0;

  // Collect metadata from IR
  OrchestrationInfoCollector info_collector;
  info_collector.VisitStmt(func->body_);

  const std::vector<std::string>& return_vars = info_collector.return_vars;

  // Build param and return name sets
  std::set<std::string> param_names;
  for (const auto& param : func->params_) {
    param_names.insert(param->name_);
  }
  std::set<std::string> return_name_set(return_vars.begin(), return_vars.end());

  // Deduplicate: return vars that are also params are inplace tensors,
  // already handled via the params loop. Only return-only vars need
  // separate ARG slots and external tensor declarations.
  std::vector<std::string> unique_return_vars;
  for (const auto& name : return_vars) {
    if (!param_names.count(name)) {
      unique_return_vars.push_back(name);
    }
  }

  // Identify intermediate tensors
  std::set<std::string> intermediate_tensors;
  for (const auto& var_name : info_collector.output_tensors) {
    if (!param_names.count(var_name) && !return_name_set.count(var_name)) {
      intermediate_tensors.insert(var_name);
    }
  }

  int unique_return_count = static_cast<int>(unique_return_vars.size());
  int expected_arg_count = CountExpectedArgs(func, unique_return_count);

  std::ostringstream oss;

  // 1. Includes
  oss << GenerateIncludes();

  // 2. ARG defines
  oss << GenerateArgDefines(func, unique_return_vars);

  // 3. Helper functions
  oss << GenerateHelperFunctions();

  // 4. extern "C" block
  oss << "extern \"C\" {\n\n";

  // 5. Config function
  oss << GenerateConfigFunction(expected_arg_count);

  // 6. Entry function
  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {\n";
  oss << "    (void)arg_count;\n\n";

  // 7. Extract arguments
  oss << "    // Extract device pointers\n";
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      std::string upper_name = param->name_;
      for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
      oss << "    void* arg_" << param->name_ << "_ptr = (void*)(uintptr_t)args[ARG_PTR_" << upper_name
          << "];\n";
    }
  }
  for (const auto& name : unique_return_vars) {
    std::string upper_name = name;
    for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    oss << "    void* arg_" << name << "_ptr = (void*)(uintptr_t)args[ARG_PTR_" << upper_name << "];\n";
  }

  // Extract sizes
  oss << "\n    // Extract sizes\n";
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      std::string upper_name = param->name_;
      for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
      oss << "    size_t size_" << param->name_ << " = (size_t)args[ARG_SIZE_" << upper_name << "];\n";
    }
  }
  for (const auto& name : unique_return_vars) {
    std::string upper_name = name;
    for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    oss << "    size_t size_" << name << " = (size_t)args[ARG_SIZE_" << upper_name << "];\n";
  }

  // 8. External tensors (make_tensor_external)
  oss << "\n    // External tensors\n";
  for (const auto& param : func->params_) {
    if (As<TensorType>(param->GetType())) {
      oss << "    Tensor ext_" << param->name_ << " = make_tensor_external(arg_" << param->name_
          << "_ptr, size_" << param->name_ << ");\n";
    }
  }
  for (const auto& name : unique_return_vars) {
    oss << "    Tensor ext_" << name << " = make_tensor_external(arg_" << name << "_ptr, size_" << name
        << ");\n";
  }

  // 9. Generate task submission code via statement codegen
  OrchestrationStmtCodegen stmt_codegen(program, &func_name_to_id, &func_name_to_core_type, &next_func_id,
                                        param_names, return_name_set);
  stmt_codegen.SetTupleElementMap(info_collector.tuple_element_map);
  stmt_codegen.VisitStmt(func->body_);

  // Emit non-task code (e.g., tensor.read, tensor.dim scalar assignments)
  std::string non_task_code = stmt_codegen.GetNonTaskCode();
  if (!non_task_code.empty()) {
    oss << "\n" << non_task_code;
  }

  // 10. Scope-aware output: classify tasks as outer vs inner scope
  // A task is "outer" if all its input tensors are external (param or return).
  // All other tasks go into an inner PTO2_SCOPE block.
  const auto& task_records = stmt_codegen.GetTaskRecords();
  std::set<std::string> external_names;
  external_names.insert(param_names.begin(), param_names.end());
  external_names.insert(return_name_set.begin(), return_name_set.end());

  std::vector<int> outer_task_ids;
  std::vector<int> inner_task_ids;
  // Also track which intermediate tensors are produced by outer tasks
  // (these need to be declared before the scope, not inside it)
  std::set<std::string> outer_produced_tensors;

  for (const auto& record : task_records) {
    bool all_inputs_external = true;
    for (const auto& input : record.input_tensors) {
      if (!external_names.count(input)) {
        all_inputs_external = false;
        break;
      }
    }
    if (all_inputs_external) {
      outer_task_ids.push_back(record.task_id);
      for (const auto& out : record.output_tensors) {
        outer_produced_tensors.insert(out);
      }
    } else {
      inner_task_ids.push_back(record.task_id);
    }
  }

  // Separate intermediate tensors into outer-produced and inner-only
  std::set<std::string> outer_intermediate_tensors;
  std::set<std::string> inner_intermediate_tensors;
  for (const auto& tensor_name : intermediate_tensors) {
    if (outer_produced_tensors.count(tensor_name)) {
      outer_intermediate_tensors.insert(tensor_name);
    } else {
      inner_intermediate_tensors.insert(tensor_name);
    }
  }

  // Helper to emit make_tensor declarations
  auto emit_tensor_decls = [&](const std::set<std::string>& tensors, const std::string& indent) {
    for (const auto& tensor_name : tensors) {
      auto tensor_type = GetIntermediateTensorType(program, info_collector.output_tensor_assigns,
                                                   info_collector.tuple_element_map, tensor_name);
      std::string size_expr = CalculateTensorSize(tensor_type);
      oss << indent << "Tensor " << tensor_name << " = make_tensor(" << size_expr << ");\n";
    }
  };

  // Emit outer intermediate tensors
  if (!outer_intermediate_tensors.empty()) {
    oss << "\n    // Intermediate tensors (outer scope)\n";
    emit_tensor_decls(outer_intermediate_tensors, "    ");
  }

  // Emit outer tasks
  for (int tid : outer_task_ids) {
    oss << task_records[tid].code;
  }

  // Emit inner scope if there are inner tasks
  if (!inner_task_ids.empty()) {
    oss << "\n    // Inner scope: intermediates released on scope end\n";
    oss << "    PTO2_SCOPE(rt) {\n";

    // Inner intermediate tensor declarations
    if (!inner_intermediate_tensors.empty()) {
      emit_tensor_decls(inner_intermediate_tensors, "        ");
      oss << "\n";
    }

    // Inner tasks (re-indent from 4 to 8 spaces)
    for (int tid : inner_task_ids) {
      // Re-indent the task code: replace leading 4-space indent with 8-space
      std::istringstream iss(task_records[tid].code);
      std::string line;
      while (std::getline(iss, line)) {
        if (line.empty()) {
          oss << "\n";
        } else if (line.substr(0, 4) == "    ") {
          oss << "        " << line.substr(4) << "\n";
        } else {
          oss << "        " << line << "\n";
        }
      }
    }

    oss << "    }  // inner scope ends\n";
  } else if (!intermediate_tensors.empty() && outer_intermediate_tensors.empty()) {
    // All intermediates but no inner tasks — just emit them normally
    oss << "\n    // Intermediate tensors\n";
    emit_tensor_decls(intermediate_tensors, "    ");
  }

  oss << "}\n\n";
  oss << "}  // extern \"C\"\n";

  return OrchestrationResult{oss.str(), std::move(func_name_to_id), std::move(func_name_to_core_type)};
}

}  // namespace codegen
}  // namespace pypto
