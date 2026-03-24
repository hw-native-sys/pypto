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
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

namespace {

using namespace pypto::ir;  // NOLINT(build/namespaces)

/**
 * @brief Extract the semantic base name from an auto-generated IR variable
 *
 * New-style names use `base__qualifier_role_vN`, while legacy names continue to
 * be accepted for compatibility. This helper normalizes both formats so codegen
 * can match input args with output vars for inout parameter detection.
 */
std::string GetSSABaseName(const std::string& name) { return auto_name::GetLegacyCompatibleBaseName(name); }

/**
 * @brief Check if an operation is a built-in IR operation (not a user-defined function)
 *
 * Built-in operations include tile-level ops (tile.*), tensor-level ops (tensor.*),
 * and system ops (system.*). These are handled by specialized codegen paths rather
 * than being dispatched as task graph function calls.
 */
bool IsBuiltinOp(const std::string& op_name) {
  return op_name.find("tile.") == 0 || op_name.find("tensor.") == 0 || op_name.find("system.") == 0;
}

/**
 * @brief Check if an operation is a tensor-level IR operation
 *
 * Tensor operations (tensor.create, tensor.read, tensor.slice, tensor.reshape)
 * are host-side operations. tensor.create, tensor.read, and tensor.slice require inline
 * orchestration C++ codegen; other tensor ops (e.g., tensor.reshape) are metadata-only and
 * expressed through TensorType parameters.
 */
bool IsTensorOp(const std::string& op_name) { return op_name.find("tensor.") == 0; }

// Format scalar constant as C++ literal/expression for assignment to the given C++ type.
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

// Per-call tuple element: stores index, fallback name, and VarPtr for identity resolution
struct TupleElement {
  int index;
  std::string fallback_name;  // GetSSABaseName result — used only if VarPtr resolution unavailable
  const Var* var;             // VarPtr for identity-based emit name resolution
};

// Collect metadata from IR (tuple info) for orchestration codegen
class OrchestrationInfoCollector : public IRVisitor {
 public:
  // Per-call tuple elements: key = unique call key, value = [TupleElement, ...]
  std::map<std::string, std::vector<TupleElement>> call_tuple_elements;
  // Maps Call* to unique key for cross-phase coordination with StmtCodegen
  std::map<const Call*, std::string> call_to_tuple_key;

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (auto call = As<Call>(assign->value_)) {
      if (!IsBuiltinOp(call->op_->name_) && call->op_->name_ != "tensor.create") {
        // Check if this call returns a TupleType
        if (As<TupleType>(call->GetType())) {
          // Generate unique key per tuple-returning call (works with/without SSA)
          std::string unique_key = "_tc_" + std::to_string(tuple_call_counter_++);
          current_tuple_key_[assign->var_->name_hint_] = unique_key;
          call_to_tuple_key[call.get()] = unique_key;
        }
      }
    } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
      // Handle: mi = TupleGetItemExpr(_tuple_tmp, 0)
      std::string var_name = GetSSABaseName(assign->var_->name_hint_);
      std::string tuple_ref_name;
      if (auto var = As<Var>(tuple_get->tuple_)) {
        tuple_ref_name = var->name_hint_;
      } else if (auto iter_arg = As<IterArg>(tuple_get->tuple_)) {
        tuple_ref_name = iter_arg->name_hint_;
      }

      // Find the unique key for the most recent tuple call with this var name
      auto it = current_tuple_key_.find(tuple_ref_name);
      if (it != current_tuple_key_.end()) {
        call_tuple_elements[it->second].push_back({tuple_get->index_, var_name, assign->var_.get()});
      }
    }
    IRVisitor::VisitStmt_(assign);
  }

 private:
  int tuple_call_counter_ = 0;
  // Maps raw var name (from assign->var_->name_hint_) to the unique key of the most recent tuple call
  std::map<std::string, std::string> current_tuple_key_;
};

/**
 * @brief Trace variable lineage from body vars back to function parameters
 *
 * Walks the function body and builds a mapping from every body Var* (including
 * IterArgs, which extend Var) back to its originating function parameter Var*.
 * This enables VarPtr-based identity checks instead of fragile string matching.
 */
class VarLineageCollector : public IRVisitor {
 public:
  std::unordered_map<const Var*, const Var*> var_to_param;

  void Initialize(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      var_to_param[param.get()] = param.get();
    }
  }

 protected:
  void VisitStmt_(const ForStmtPtr& for_stmt) override {
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

  void VisitStmt_(const WhileStmtPtr& while_stmt) override {
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

  // IfStmt lineage is not tracked: orchestration IfStmt return_vars are rare
  // and their lineage requires analyzing yield values across branches.

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (auto src_var = AsVarLike(assign->value_)) {
      const Var* param = ResolveVar(src_var.get());
      if (param) {
        var_to_param[assign->var_.get()] = param;
      }
    }
    IRVisitor::VisitStmt_(assign);
  }

 private:
  const Var* ResolveVar(const Var* var) const {
    auto it = var_to_param.find(var);
    return it != var_to_param.end() ? it->second : nullptr;
  }

  const Var* ResolveExpr(const ExprPtr& expr) const {
    // Use AsVarLike to match both Var and IterArg (As<Var> won't match IterArg)
    if (auto var = AsVarLike(expr)) {
      return ResolveVar(var.get());
    }
    return nullptr;
  }
};

}  // namespace

CoreType InferFunctionCoreType(const FunctionPtr& func) {
  // Fast path: derive core type directly from FunctionType for specialized functions
  if (func->func_type_ == FunctionType::AIC) return CoreType::CUBE;
  if (func->func_type_ == FunctionType::AIV) return CoreType::VECTOR;

  class CoreTypeCollector : public IRVisitor {
   public:
    bool has_cube_ = false;
    bool has_vector_ = false;

    void VisitExpr_(const CallPtr& call) override {
      for (const auto& arg : call->args_) {
        if (auto tile = As<TileType>(arg->GetType())) {
          auto memory_space = tile->GetMemorySpace();
          if (!memory_space.has_value()) {
            continue;
          }
          if (IsCubeMemorySpace(*memory_space)) {
            has_cube_ = true;
          } else if (*memory_space == MemorySpace::Vec) {
            has_vector_ = true;
          }
        }
      }
      IRVisitor::VisitExpr_(call);
    }
  };

  CoreTypeCollector collector;
  collector.VisitStmt(func->body_);

  CHECK(!(collector.has_cube_ && collector.has_vector_))
      << "Function " << func->name_ << " contains both CUBE and VECTOR memory spaces. "
      << "A function can only use one core type.";

  if (collector.has_cube_) {
    return CoreType::CUBE;
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
  for (const auto& var : func->params_) {
    if (As<TensorType>(var->GetType())) {
      std::string name = auto_name::GetCompatibleBaseName(var->name_hint_);
      std::string upper_name = name;
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

const char TENSOR_HELPER_FUNCTION[] = R"(
static inline Tensor make_tensor_external_2d_dn(void* addr,
    const uint64_t shapes[],
    uint64_t ndims,
    DataType dtype = DataType::FLOAT32,
    int32_t version = 0) {
    debug_assert(ndims == 2);
    static uint64_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint64_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    uint64_t raw_shapes[RUNTIME_MAX_TENSOR_DIMS] = {shapes[1], shapes[0]};
    return Tensor(addr, total * get_element_size(dtype),
        raw_shapes, shapes, zero_offsets, ndims, dtype, version);
}

static inline Tensor make_tensor_2d_dn(
    const uint64_t shapes[],
    uint64_t ndims,
    DataType dtype = DataType::FLOAT32,
    int32_t version = 0) {
    debug_assert(ndims == 2);
    static uint64_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint64_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    uint64_t raw_shapes[RUNTIME_MAX_TENSOR_DIMS] = {shapes[1], shapes[0]};
    return Tensor(0, total * get_element_size(dtype),
        raw_shapes, shapes, zero_offsets, ndims, dtype, version);
}
)";

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

  // helper function for make DN tensor
  oss << TENSOR_HELPER_FUNCTION << "\n";
  return oss.str();
}

std::string CoreTypeToSubmitFunc(CoreType core_type) {
  return core_type == CoreType::CUBE ? "pto2_rt_submit_aic_task" : "pto2_rt_submit_aiv_task";
}

// Removed DataTypeToPTO2Enum — now uses DataTypeToString from dtype.h

// Generate make_tensor_external with shape array, ndim, and dtype
std::string GenerateMakeTensorExternal(const std::string& var_name, const std::string& ptr_name,
                                       const TensorTypePtr& tensor_type, const CodegenBase& codegen) {
  std::ostringstream oss;
  size_t ndim = tensor_type->shape_.size();
  size_t shape_arr_len = (ndim == 0) ? 1 : ndim;

  // Declare shape array (minimum size 1 for valid C++)
  oss << "    uint64_t " << var_name << "_shapes[" << shape_arr_len << "] = {";
  if (ndim == 0) {
    oss << "1";
  } else {
    for (size_t i = 0; i < ndim; ++i) {
      if (i > 0) oss << ", ";
      oss << codegen.GenerateExprString(tensor_type->shape_[i]);
    }
  }
  oss << "};\n";

  // check layout DN
  std::string runtime_func = "make_tensor_external";
  if (tensor_type->tensor_view_.has_value() && tensor_type->tensor_view_->layout == TensorLayout::DN) {
    CHECK(ndim == 2) << "only support 2D tensor for DN layout now";
    runtime_func = "make_tensor_external_2d_dn";
  }
  // Generate make_tensor_external call
  oss << "    Tensor ext_" << var_name << " = " << runtime_func << "(" << ptr_name << ", " << var_name
      << "_shapes, " << ndim << ", " << codegen.GetRuntimeDataTypeString(tensor_type->dtype_) << ");\n";

  return oss.str();
}

}  // namespace

using namespace pypto::ir;  // NOLINT(build/namespaces)

// Statement code generator for orchestration
class OrchestrationStmtCodegen : public CodegenBase {
 public:
  explicit OrchestrationStmtCodegen(const ProgramPtr& prog, std::map<std::string, int>* func_ids,
                                    std::map<std::string, CoreType>* core_types, int* next_id,
                                    std::unordered_map<const Var*, std::string> param_to_emit_name,
                                    std::unordered_map<const Var*, const Var*> var_to_param,
                                    std::set<std::string> param_name_set)
      : program_(prog),
        func_name_to_id_(func_ids),
        func_name_to_core_type_(core_types),
        next_func_id_(next_id),
        param_to_emit_name_(std::move(param_to_emit_name)),
        var_to_param_(std::move(var_to_param)),
        param_name_set_(std::move(param_name_set)) {}

  // Set per-call tuple elements using unique keys (avoids cross-call collision)
  void SetCallTupleElements(const std::map<std::string, std::vector<TupleElement>>& elements) {
    tuple_var_to_elements_ = elements;
    for (auto& [key, vec] : tuple_var_to_elements_) {
      std::sort(vec.begin(), vec.end(),
                [](const TupleElement& a, const TupleElement& b) { return a.index < b.index; });
    }
  }

  // Set Call* → unique key mapping for tuple-returning calls
  void SetCallToTupleKey(const std::map<const Call*, std::string>& mapping) { call_to_tuple_key_ = mapping; }

  std::string GetGeneratedCode() const { return code_.str(); }

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
  std::string GetVarName(const VarPtr& var) override { return ResolveVarEmitName(var.get()); }
  [[nodiscard]] std::string TryGetVarName(const ir::ExprPtr& expr) const override {
    if (auto iter_arg = As<IterArg>(expr)) {
      // Check iter_arg_to_var_ first (set during ForStmt processing)
      auto it = iter_arg_to_var_.find(iter_arg.get());
      if (it != iter_arg_to_var_.end()) {
        return it->second;
      }
    }
    // Match both Var and IterArg (As<Var> won't match IterArg)
    if (auto var = AsVarLike(expr)) {
      return ResolveVarEmitName(var.get());
    }
    return CodegenBase::TryGetVarName(expr);
  }
  [[nodiscard]] std::string GetTensorDataPtr(const std::string& name) const override {
    if (param_name_set_.count(name)) {
      return "arg_" + name + "_ptr";
    }
    return name + ".data";
  }

  void VisitStmt_(const ForStmtPtr& for_stmt) override {
    if (for_stmt->kind_ == ForKind::Unroll) {
      LOG_WARN << "ForKind::Unroll loop was not expanded before codegen; "
                  "generating sequential loop as fallback";
    }

    std::string loop_var = GetVarName(for_stmt->loop_var_);
    std::string start_expr = GenerateExprString(for_stmt->start_);
    std::string stop_expr = GenerateExprString(for_stmt->stop_);
    std::string step_expr = GenerateExprString(for_stmt->step_);

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      const auto& iter_arg = for_stmt->iter_args_[i];
      const auto& return_var = for_stmt->return_vars_[i];
      std::string resolved_return = GetVarName(return_var);
      std::string init_value = GenerateExprString(iter_arg->initValue_);
      // Apply ext_ prefix for tensor-type init values referencing params/returns
      if (As<TensorType>(iter_arg->GetType())) {
        init_value = GetExternalTensorName(init_value);
      }
      // Skip iter_arg init when:
      // 1. Variable already declared (e.g., via make_tensor) — would be C++ redeclaration error
      // 2. Self-assignment after SSA name collapse (e.g., "auto oi = oi;") — C++ UB
      // 3. Variable traces to a param (external tensor) — would shadow ext_ declaration
      bool is_param_derived = ResolveToParam(return_var.get()) != nullptr;
      if (!declared_var_names_.count(resolved_return) && !is_param_derived && resolved_return != init_value) {
        code_ << Indent() << GetCppType(iter_arg->GetType()) << " " << resolved_return << " = " << init_value
              << ";\n";
      }
      iter_arg_to_var_[iter_arg.get()] = resolved_return;
    }

    code_ << Indent() << "for (int64_t " << loop_var << " = " << start_expr << "; " << loop_var << " < "
          << stop_expr << "; " << loop_var << " += " << step_expr << ") {\n";
    indent_ += 4;
    code_ << Indent() << "PTO2_SCOPE(rt) {\n";
    indent_ += 4;

    auto saved = current_return_var_names_;
    current_return_var_names_.clear();
    for (const auto& rv : for_stmt->return_vars_) {
      current_return_var_names_.push_back(GetVarName(rv));
    }
    VisitStmt(for_stmt->body_);
    current_return_var_names_ = saved;

    indent_ -= 4;
    code_ << Indent() << "}\n";
    indent_ -= 4;
    code_ << Indent() << "}\n";
  }

  void VisitStmt_(const IfStmtPtr& if_stmt) override {
    std::string cond_expr = GenerateExprString(if_stmt->condition_);

    // Declare return variables before the if block
    for (const auto& rv : if_stmt->return_vars_) {
      code_ << Indent() << GetCppType(rv->GetType()) << " " << GetVarName(rv) << ";\n";
    }

    code_ << Indent() << "if (" << cond_expr << ") {\n";
    indent_ += 4;
    code_ << Indent() << "PTO2_SCOPE(rt) {\n";
    indent_ += 4;

    auto saved = current_return_var_names_;
    current_return_var_names_.clear();
    for (const auto& rv : if_stmt->return_vars_) {
      current_return_var_names_.push_back(GetVarName(rv));
    }
    VisitStmt(if_stmt->then_body_);
    current_return_var_names_ = saved;

    indent_ -= 4;
    code_ << Indent() << "}\n";
    indent_ -= 4;

    if (if_stmt->else_body_.has_value()) {
      code_ << Indent() << "} else {\n";
      indent_ += 4;
      code_ << Indent() << "PTO2_SCOPE(rt) {\n";
      indent_ += 4;

      auto saved2 = current_return_var_names_;
      current_return_var_names_.clear();
      for (const auto& rv : if_stmt->return_vars_) {
        current_return_var_names_.push_back(GetVarName(rv));
      }
      VisitStmt(*if_stmt->else_body_);
      current_return_var_names_ = saved2;

      indent_ -= 4;
      code_ << Indent() << "}\n";
      indent_ -= 4;
    }

    code_ << Indent() << "}\n";
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    std::string var_name = GetVarName(assign->var_);

    if (auto call = As<Call>(assign->value_)) {
      const std::string& op_name = call->op_->name_;
      if (IsTensorOp(op_name)) {
        GenerateTensorOpCode(call, var_name, assign->var_.get());
      } else if (!IsBuiltinOp(op_name)) {
        // For tuple-returning calls, look up the unique key via Call* pointer
        std::string result_key;
        if (As<TupleType>(call->GetType())) {
          auto it = call_to_tuple_key_.find(call.get());
          result_key = (it != call_to_tuple_key_.end()) ? it->second : var_name;
        } else {
          result_key = var_name;
        }
        GenerateFunctionCallCode(call, result_key);

        // Generate alias for InCore call return values.
        // Return value at position i maps to the i-th Out/InOut arg (in call arg order).
        // Emit C++ reference alias when return var name differs from the Out/InOut arg name.
        if (!As<TupleType>(call->GetType())) {
          FunctionPtr callee = program_->GetFunction(call->op_->name_);
          if (callee) {
            for (size_t i = 0; i < callee->param_directions_.size(); ++i) {
              if (callee->param_directions_[i] == ParamDirection::Out ||
                  callee->param_directions_[i] == ParamDirection::InOut) {
                std::string out_arg = TryGetVarName(call->args_[i]);
                if (!out_arg.empty() && var_name != out_arg) {
                  // Use VarPtr for external tensor detection
                  const Var* arg_param = ResolveExprToParam(call->args_[i]);
                  std::string ext_out = arg_param ? "ext_" + GetParamEmitName(arg_param) : out_arg;
                  code_ << Indent() << "Tensor& " << var_name << " = " << ext_out << ";\n";
                }
                break;  // Single return → first Out/InOut
              }
            }
          }
          call_result_var_ptrs_.insert(assign->var_.get());
        } else {
          // For tuple returns: generate aliases for each element
          auto tuple_key_it = call_to_tuple_key_.find(call.get());
          if (tuple_key_it != call_to_tuple_key_.end()) {
            auto elements_it = tuple_var_to_elements_.find(tuple_key_it->second);
            if (elements_it != tuple_var_to_elements_.end()) {
              FunctionPtr callee = program_->GetFunction(call->op_->name_);
              if (callee) {
                // Collect Out/InOut arg indices in order
                std::vector<size_t> out_indices;
                for (size_t i = 0; i < callee->param_directions_.size(); ++i) {
                  if (callee->param_directions_[i] == ParamDirection::Out ||
                      callee->param_directions_[i] == ParamDirection::InOut) {
                    out_indices.push_back(i);
                  }
                }
                // Map each tuple element to corresponding Out/InOut arg
                for (size_t ei = 0; ei < elements_it->second.size() && ei < out_indices.size(); ++ei) {
                  const auto& elem = elements_it->second[ei];
                  // Resolve elem name via VarPtr identity (avoids legacy suffix collisions)
                  std::string elem_name = ResolveVarEmitName(elem.var);
                  std::string out_arg = TryGetVarName(call->args_[out_indices[ei]]);
                  if (!out_arg.empty() && elem_name != out_arg) {
                    const Var* arg_param = ResolveExprToParam(call->args_[out_indices[ei]]);
                    std::string ext_out = arg_param ? "ext_" + GetParamEmitName(arg_param) : out_arg;
                    code_ << Indent() << "Tensor& " << elem_name << " = " << ext_out << ";\n";
                  }
                  call_result_var_ptrs_.insert(elem.var);
                }
              }
            }
          }
        }
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
        // Skip yield when:
        // 1. Self-assignment (e.g., "oi = oi;" for inplace tensor iter_args)
        // 2. Value is a task-submission result (no C++ variable exists for it)
        auto yield_var = AsVarLike(yield_stmt->value_[i]);
        bool is_call_result = yield_var && call_result_var_ptrs_.count(yield_var.get());
        if (current_return_var_names_[i] != value_expr && !is_call_result) {
          code_ << Indent() << current_return_var_names_[i] << " = " << value_expr << ";\n";
        }
      }
    }
  }

  void VisitStmt_(const EvalStmtPtr& eval) override {
    if (auto call = As<Call>(eval->expr_)) {
      const std::string& op_name = call->op_->name_;
      if (IsTensorOp(op_name)) {
        GenerateTensorOpCode(call, "", nullptr);
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
  [[nodiscard]] std::string GetExternalTensorName(const std::string& name) const override {
    if (param_name_set_.count(name)) {
      return "ext_" + name;
    }
    return name;
  }

  struct ParamEntry {
    std::string kind;  // "make_input_param", "make_output_param", "make_inout_param", "make_scalar_param"
    std::string value;
  };

  std::vector<ParamEntry> BuildTaskParams(const CallPtr& call, const FunctionPtr& callee_func) {
    std::vector<ParamEntry> params;
    const std::string& callee_name = callee_func->name_;

    for (size_t arg_idx = 0; arg_idx < call->args_.size(); ++arg_idx) {
      const auto& arg = call->args_[arg_idx];
      std::string var_name = TryGetVarName(arg);
      if (!var_name.empty()) {
        // Check if this is a scalar variable (not a tensor) -> make_scalar_param
        if (auto scalar_type = As<ScalarType>(arg->GetType())) {
          std::string cpp_type = scalar_type->dtype_.ToCTypeString();
          if (cpp_type == "float") {
            params.push_back({"make_scalar_param", "float_to_u64(" + var_name + ")"});
          } else {
            params.push_back({"make_scalar_param", var_name});
          }
          continue;
        }

        // Use VarPtr identity for external tensor detection
        const Var* param = ResolveExprToParam(arg);
        std::string ext_name;
        if (param) {
          ext_name = "ext_" + GetParamEmitName(param);
        } else {
          ext_name = var_name;
        }

        // Classify based on callee's ParamDirection
        INTERNAL_CHECK(arg_idx < callee_func->param_directions_.size())
            << "arg count (" << call->args_.size() << ") exceeds param count ("
            << callee_func->param_directions_.size() << ") for callee '" << callee_name << "'";
        ParamDirection dir = callee_func->param_directions_[arg_idx];
        if (dir == ParamDirection::Out) {
          params.push_back({"make_output_param", ext_name});
        } else if (dir == ParamDirection::InOut) {
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

    return params;
  }

  void GenerateTensorOpCode(const CallPtr& call, const std::string& result_var, const Var* assign_var) {
    const std::string& op_name = call->op_->name_;

    auto& registry = OrchestrationOpRegistry::GetInstance();
    auto codegen_func = registry.Get(op_name);
    if (!codegen_func.has_value()) {
      // Ops without a registered orchestration codegen handler (e.g., tensor.reshape) have no codegen
      return;
    }

    // VarPtr dedup: skip if this exact variable was already declared
    if (op_name == "tensor.create" && assign_var && declared_var_ptrs_.count(assign_var)) {
      return;
    }

    // Skip tensor.create for external tensors (params) —
    // they are already declared via make_tensor_external.
    // Use VarPtr identity instead of string matching.
    if (op_name == "tensor.create" && assign_var && ResolveToParam(assign_var)) {
      return;
    }

    std::string emit_var = result_var;
    if (op_name == "tensor.create" && assign_var) {
      declared_var_ptrs_.insert(assign_var);
      emit_var = ReserveTensorCreateEmitName(assign_var, emit_var);
    }

    current_result_var_ = emit_var;

    std::string gen_code = (*codegen_func)(call, *this);

    std::istringstream iss(gen_code);
    std::string line;
    while (std::getline(iss, line)) {
      if (!line.empty()) {
        code_ << Indent() << line << "\n";
      }
    }
  }

  /// Walk the Group function body to find the AIC and AIV callee names.
  void FindGroupCallees(const FunctionPtr& group_func, std::string& aic_name, std::string& aiv_name) {
    class CalleeFinder : public IRVisitor {
     public:
      explicit CalleeFinder(const ProgramPtr& program) : program_(program) {}
      const ProgramPtr& program_;
      std::string aic_name;
      std::string aiv_name;

     protected:
      void VisitExpr_(const CallPtr& call) override {
        if (auto gv = As<GlobalVar>(call->op_)) {
          auto callee = program_->GetFunction(gv->name_);
          if (callee) {
            if (callee->func_type_ == FunctionType::AIC && aic_name.empty()) {
              aic_name = callee->name_;
            } else if (callee->func_type_ == FunctionType::AIV && aiv_name.empty()) {
              aiv_name = callee->name_;
            }
          }
        }
        IRVisitor::VisitExpr_(call);
      }
    };

    CalleeFinder finder(program_);
    finder.VisitStmt(group_func->body_);
    aic_name = std::move(finder.aic_name);
    aiv_name = std::move(finder.aiv_name);
  }

  void GenerateFunctionCallCode(const CallPtr& call, const std::string& result_var) {
    const std::string& callee_name = call->op_->name_;

    FunctionPtr callee_func = program_->GetFunction(callee_name);
    INTERNAL_CHECK(callee_func != nullptr)
        << "Internal error: function '" << callee_name << "' not found after validation.";

    if (callee_func->func_type_ == FunctionType::Group) {
      GenerateGroupCallCode(call, callee_func);
      return;
    }

    CoreType core_type = InferFunctionCoreType(callee_func);
    (*func_name_to_core_type_)[callee_name] = core_type;

    int func_id = GetOrCreateFuncId(callee_name, func_name_to_id_, next_func_id_);

    auto params = BuildTaskParams(call, callee_func);

    std::string ind = Indent();

    // Generate PTOParam array and submit_task
    std::string task_var = "params_t" + std::to_string(task_counter_);
    code_ << "\n";
    code_ << ind << "// Task " << task_counter_ << ": " << callee_name << "\n";
    code_ << ind << "PTOParam " << task_var << "[] = {\n";
    for (const auto& p : params) {
      code_ << ind << "    " << p.kind << "(" << p.value << "),\n";
    }
    code_ << ind << "};\n";
    code_ << ind << CoreTypeToSubmitFunc(core_type) << "(rt, " << func_id << ", " << task_var << ", "
          << params.size() << ");\n";

    task_counter_++;
  }

  void GenerateGroupCallCode(const CallPtr& call, const FunctionPtr& group_func) {
    std::string group_name = group_func->name_;

    // Resolve AIC/AIV callees by inspecting the Group body rather than name suffix.
    // This handles both synthetic wrappers (name matches InCore) and rewritten Groups
    // (where the Group name differs from the InCore-derived AIC/AIV names).
    std::string aic_name;
    std::string aiv_name;
    FindGroupCallees(group_func, aic_name, aiv_name);
    INTERNAL_CHECK(!aic_name.empty())
        << "Internal error: no AIC callee found in Group '" << group_name << "' body";
    INTERNAL_CHECK(!aiv_name.empty())
        << "Internal error: no AIV callee found in Group '" << group_name << "' body";

    FunctionPtr aic_func = program_->GetFunction(aic_name);
    FunctionPtr aiv_func = program_->GetFunction(aiv_name);
    INTERNAL_CHECK(aic_func != nullptr)
        << "Internal error: AIC function '" << aic_name << "' not found for Group '" << group_name << "'";
    INTERNAL_CHECK(aiv_func != nullptr)
        << "Internal error: AIV function '" << aiv_name << "' not found for Group '" << group_name << "'";

    (*func_name_to_core_type_)[aic_name] = CoreType::CUBE;
    (*func_name_to_core_type_)[aiv_name] = CoreType::VECTOR;

    // AIC and AIV share the same params/directions as the original InCore function
    auto params = BuildTaskParams(call, aic_func);

    int aic_id = GetOrCreateFuncId(aic_name, func_name_to_id_, next_func_id_);
    int aiv_id = GetOrCreateFuncId(aiv_name, func_name_to_id_, next_func_id_);

    std::string ind = Indent();
    std::string task_var = "params_t" + std::to_string(task_counter_);

    code_ << "\n";
    code_ << ind << "// Group " << group_name << ": MixedKernels (AIC + AIV)\n";
    code_ << ind << "PTOParam " << task_var << "[] = {\n";
    for (const auto& p : params) {
      code_ << ind << "    " << p.kind << "(" << p.value << "),\n";
    }
    code_ << ind << "};\n";
    code_ << ind << "MixedKernels mixed_" << task_counter_ << " = {" << aic_id << ", " << aiv_id
          << ", INVALID_KERNEL_ID};\n";
    code_ << ind << "pto2_rt_submit_task(rt, mixed_" << task_counter_ << ", " << task_var << ", "
          << params.size() << ");\n";

    task_counter_++;
  }

  // --- VarPtr-based identity helpers ---

  // Resolve a Var to its C++ emit name: param name for param-derived vars,
  // GetSSABaseName for others, with collision defense against param names.
  std::string ResolveVarEmitName(const Var* var) const {
    auto it = explicit_var_emit_names_.find(var);
    if (it != explicit_var_emit_names_.end()) {
      return it->second;
    }
    const Var* param = ResolveToParam(var);
    if (param) {
      return GetParamEmitName(param);
    }
    std::string base = GetSSABaseName(var->name_hint_);
    if (param_name_set_.count(base)) {
      return var->name_hint_;
    }
    return base;
  }

  const Var* ResolveToParam(const Var* var) const {
    auto it = var_to_param_.find(var);
    return it != var_to_param_.end() ? it->second : nullptr;
  }

  const Var* ResolveExprToParam(const ExprPtr& expr) const {
    // Use AsVarLike to match both Var and IterArg (As<Var> won't match IterArg)
    if (auto var = AsVarLike(expr)) {
      return ResolveToParam(var.get());
    }
    return nullptr;
  }

  const std::string& GetParamEmitName(const Var* param) const {
    auto it = param_to_emit_name_.find(param);
    INTERNAL_CHECK(it != param_to_emit_name_.end()) << "Internal error: param not found in emit name map";
    return it->second;
  }

  std::string ReserveTensorCreateEmitName(const Var* assign_var, const std::string& preferred_name) {
    auto it = explicit_var_emit_names_.find(assign_var);
    if (it != explicit_var_emit_names_.end()) {
      return it->second;
    }

    auto parsed = auto_name::Parse(assign_var->name_hint_);
    bool preserve_raw_name = parsed.role.has_value() && *parsed.role == "out";
    std::string base_name = preferred_name;
    if (preserve_raw_name || declared_var_names_.count(base_name)) {
      base_name = assign_var->name_hint_;
    }

    std::string emit_name = auto_name::ReserveUniqueName(base_name, declared_var_names_);
    explicit_var_emit_names_[assign_var] = emit_name;
    return emit_name;
  }

  const ProgramPtr& program_;
  std::map<std::string, int>* func_name_to_id_;
  std::map<std::string, CoreType>* func_name_to_core_type_;
  int* next_func_id_;
  // VarPtr-based identity data
  std::unordered_map<const Var*, std::string> param_to_emit_name_;
  std::unordered_map<const Var*, const Var*> var_to_param_;
  std::set<std::string> param_name_set_;  // For string-only contexts (op codegen callbacks)
  std::ostringstream code_;
  int indent_ = 4;
  std::map<const IterArg*, std::string> iter_arg_to_var_;
  std::string current_result_var_;
  std::vector<std::string> current_return_var_names_;
  int task_counter_ = 0;
  std::map<std::string, std::vector<TupleElement>> tuple_var_to_elements_;
  std::map<const Call*, std::string> call_to_tuple_key_;  // Call* → unique key for tuple calls
  // VarPtr-based dedup: prevents skipping tensor.create for distinct vars with same base name
  std::unordered_set<const Var*> declared_var_ptrs_;
  // String-based dedup: ensures unique C++ variable names in generated code
  std::set<std::string> declared_var_names_;
  // Stores the final emitted name for declared local vars, keyed by Var identity.
  std::unordered_map<const Var*, std::string> explicit_var_emit_names_;
  // VarPtr-based tracking of task-submission results (no C++ declaration exists, skip in yield)
  std::unordered_set<const Var*> call_result_var_ptrs_;
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

  // Build VarPtr-based identity data
  VarLineageCollector lineage;
  lineage.Initialize(func->params_);
  lineage.VisitStmt(func->body_);

  std::unordered_map<const Var*, std::string> param_to_emit_name;
  std::set<std::string> param_name_set;
  int tensor_param_count = 0;
  for (const auto& var : func->params_) {
    // Use GetCompatibleBaseName: strips auto-name suffixes (e.g., __ssa_v0)
    // but preserves semantic name parts (e.g., _0 in out_0).
    // Unlike GetLegacyCompatibleBaseName, does NOT strip plain numeric suffixes.
    std::string emit_name = auto_name::GetCompatibleBaseName(var->name_hint_);
    param_to_emit_name[var.get()] = emit_name;
    param_name_set.insert(emit_name);
    if (As<TensorType>(var->GetType())) {
      tensor_param_count++;
    }
  }

  // All external tensors come from params (including Out/InOut) — no return inference
  int expected_arg_count = tensor_param_count;

  std::ostringstream oss;

  // 1. Includes
  oss << GenerateIncludes();

  // 2. ARG defines (no separate return vars — all from params)
  oss << GenerateArgDefines(func, {});

  // 3. Helper functions
  oss << GenerateHelperFunctions();

  // 4. extern "C" block
  oss << "extern \"C\" {\n\n";

  // 5. Config function
  oss << GenerateConfigFunction(expected_arg_count);

  // 6. Entry function
  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count, "
         "int orch_thread_num, int orch_thread_index) {\n";
  oss << "    (void)arg_count;\n";
  oss << "    (void)orch_thread_num;\n";
  oss << "    (void)orch_thread_index;\n\n";

  // 7. Extract arguments (all from params — use GetCompatibleBaseName, not legacy stripping)
  oss << "    // Extract device pointers\n";
  for (const auto& var : func->params_) {
    if (As<TensorType>(var->GetType())) {
      std::string name = auto_name::GetCompatibleBaseName(var->name_hint_);
      std::string upper_name = name;
      for (auto& ch : upper_name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
      oss << "    void* arg_" << name << "_ptr = reinterpret_cast<void*>(args[ARG_PTR_" << upper_name
          << "]);\n";
    }
  }

  // Create statement codegen (used for both external tensor generation and task submission)
  OrchestrationStmtCodegen stmt_codegen(program, &func_name_to_id, &func_name_to_core_type, &next_func_id,
                                        std::move(param_to_emit_name), std::move(lineage.var_to_param),
                                        std::move(param_name_set));
  stmt_codegen.SetCallTupleElements(info_collector.call_tuple_elements);
  stmt_codegen.SetCallToTupleKey(info_collector.call_to_tuple_key);

  // 8. External tensors (make_tensor_external with shape/dtype — all from params)
  oss << "\n    // External tensors\n";
  for (const auto& var : func->params_) {
    auto tensor_type = As<TensorType>(var->GetType());
    if (tensor_type) {
      std::string name = auto_name::GetCompatibleBaseName(var->name_hint_);
      oss << GenerateMakeTensorExternal(name, "arg_" + name + "_ptr", tensor_type, stmt_codegen);
    }
  }

  // 9. Generate task submission code via statement codegen
  stmt_codegen.VisitStmt(func->body_);

  // 10. Emit generated code (unified path: always use code_ stream)
  oss << "\n" << stmt_codegen.GetGeneratedCode();

  oss << "}\n\n";
  oss << "}  // extern \"C\"\n";

  return OrchestrationResult{oss.str(), std::move(func_name_to_id), std::move(func_name_to_core_type)};
}

}  // namespace codegen
}  // namespace pypto
