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
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/dtype.h"
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

using namespace pypto::ir;  // NOLINT(build/namespaces)

CoreType InferFunctionCoreType(const FunctionPtr& func) {
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

// ---------------------------------------------------------------------------
// Template / boilerplate generation helpers
// ---------------------------------------------------------------------------

std::string GenerateIncludes(bool include_optional) {
  std::ostringstream oss;
  oss << "#include <stddef.h>\n";
  oss << "#include <stdint.h>\n";
  oss << "#include <stdio.h>\n";
  if (include_optional) {
    oss << "#include <optional>\n";
  }
  oss << "\n";
  oss << "#include \"pto_orchestration_api.h\"\n\n";
  return oss.str();
}

std::string GenerateScalarUnpack(const std::string& var_name, int scalar_index,
                                 const ScalarTypePtr& scalar_type) {
  std::ostringstream oss;
  std::string cpp_type = scalar_type->dtype_.ToCTypeString();
  oss << "    " << cpp_type << " " << var_name << " = from_u64<" << cpp_type << ">(orch_args.scalar("
      << scalar_index << "));\n";
  return oss.str();
}

std::string GenerateConfigFunction(int expected_arg_count) {
  std::ostringstream oss;
  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {\n";
  oss << "    (void)orch_args;\n";
  oss << "    return PTO2OrchestrationConfig{\n";
  oss << "        .expected_arg_count = " << expected_arg_count << ",\n";
  oss << "    };\n";
  oss << "}\n\n";
  return oss.str();
}

bool IsA5Backend() { return pypto::backend::GetBackendType() == pypto::backend::BackendType::Ascend950; }

bool RequiresDualAivDispatch(const FunctionPtr& aiv_func) {
  return aiv_func != nullptr && aiv_func->GetSplitMode().has_value() &&
         aiv_func->GetSplitMode().value() != SplitMode::None;
}

// Returns the opening of a pto2_rt_submit_{aic,aiv}_task call.
// Caller appends: func_id << ", " << params << ");".
std::string CoreTypeToSubmitPrefix(CoreType core_type) {
  std::string func = core_type == CoreType::CUBE ? "pto2_rt_submit_aic_task" : "pto2_rt_submit_aiv_task";
  return func + "(";
}

std::string GenerateMakeTensorExternal(const std::string& var_name, int orch_index,
                                       const TensorTypePtr& tensor_type, const CodegenBase& codegen) {
  std::ostringstream oss;
  oss << "    Tensor ext_" << var_name << " = from_tensor_arg(orch_args.tensor(" << orch_index << "));\n";
  return oss.str();
}

}  // namespace

// Statement code generator for orchestration
class OrchestrationStmtCodegen : public CodegenBase {
 public:
  explicit OrchestrationStmtCodegen(const ProgramPtr& prog, std::map<std::string, int>* func_ids,
                                    std::map<std::string, CoreType>* core_types, int* next_id,
                                    std::unordered_map<const Var*, std::string> param_to_emit_name,
                                    std::set<std::string> param_name_set,
                                    std::map<std::string, int> param_name_to_orch_index)
      : program_(prog),
        func_name_to_id_(func_ids),
        func_name_to_core_type_(core_types),
        next_func_id_(next_id),
        emit_name_map_(std::move(param_to_emit_name)),
        param_name_set_(std::move(param_name_set)),
        param_name_to_orch_index_(std::move(param_name_to_orch_index)) {
    declared_var_names_ = param_name_set_;
  }

  void SetCallTupleElements(const std::map<std::string, std::vector<TupleElement>>& elements) {
    tuple_var_to_elements_ = elements;
    for (auto& [key, vec] : tuple_var_to_elements_) {
      std::sort(vec.begin(), vec.end(),
                [](const TupleElement& a, const TupleElement& b) { return a.index < b.index; });
    }
  }

  void SetCallToTupleKey(const std::map<const Call*, std::string>& mapping) { call_to_tuple_key_ = mapping; }

  void SetInitialIndent(int indent) { indent_ = indent; }

  std::string GetGeneratedCode() const { return code_.str(); }
  // --- CodegenBase pure virtual implementations ---
  [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_result_var_; }
  void Emit(const std::string& line) override { code_ << line; }
  std::string GetExprAsCode(const ExprPtr& expr) override { return GenerateExprString(expr); }
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override {
    return dtype.ToCTypeString();
  }
  int64_t GetConstIntValue(const ExprPtr& expr) const override {
    auto ci = As<ConstInt>(expr);
    INTERNAL_CHECK(ci) << "Internal error: expected ConstInt expression";
    return ci->value_;
  }
  std::string GetVarName(const VarPtr& var) const override {
    auto it = emit_name_map_.find(var.get());
    if (it != emit_name_map_.end()) {
      return it->second;
    }
    return GetSSABaseName(var->name_hint_);
  }
  [[nodiscard]] std::string TryGetVarName(const ir::ExprPtr& expr) const override {
    if (auto var = AsVarLike(expr)) {
      return GetVarName(var);
    }
    return CodegenBase::TryGetVarName(expr);
  }
  [[nodiscard]] std::string GetTensorDataPtr(const std::string& name) const override {
    auto it = param_name_to_orch_index_.find(name);
    if (it != param_name_to_orch_index_.end()) {
      return "orch_args.tensor(" + std::to_string(it->second) + ").data_as<void>()";
    }
    return name + ".data";
  }

  [[nodiscard]] std::string GetTensorShapeDim(const std::string& name, int64_t axis) const override {
    auto it = param_name_to_orch_index_.find(name);
    if (it != param_name_to_orch_index_.end()) {
      return "(int64_t)orch_args.tensor(" + std::to_string(it->second) + ").shapes[" + std::to_string(axis) +
             "]";
    }
    return "(int64_t)" + name + ".shapes[" + std::to_string(axis) + "]";
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
      std::string init_var_name = TryGetVarName(iter_arg->initValue_);
      INTERNAL_CHECK(!init_var_name.empty())
          << "Internal error: ForStmt iter_arg initValue must be a variable, got non-variable expr";
      emit_name_map_[iter_arg.get()] = init_var_name;
      emit_name_map_[return_var.get()] = init_var_name;
    }

    code_ << Indent() << "for (int64_t " << loop_var << " = " << start_expr << "; " << loop_var << " < "
          << stop_expr << "; " << loop_var << " += " << step_expr << ") {\n";
    indent_ += 4;
    code_ << Indent() << "PTO2_SCOPE() {\n";
    indent_ += 4;

    auto saved = current_return_vars_;
    current_return_vars_.clear();
    VisitStmt(for_stmt->body_);
    current_return_vars_ = saved;

    indent_ -= 4;
    code_ << Indent() << "}\n";
    indent_ -= 4;
    code_ << Indent() << "}\n";
  }

  void VisitStmt_(const IfStmtPtr& if_stmt) override {
    std::string cond_expr = GenerateExprString(if_stmt->condition_);

    for (const auto& rv : if_stmt->return_vars_) {
      code_ << Indent() << GetCppType(rv->GetType()) << " " << ReserveVarEmitName(rv.get()) << ";\n";
    }

    code_ << Indent() << "if (" << cond_expr << ") {\n";
    VisitScopedBranchBody(if_stmt->then_body_, if_stmt->return_vars_);

    if (if_stmt->else_body_.has_value()) {
      code_ << Indent() << "} else {\n";
      VisitScopedBranchBody(*if_stmt->else_body_, if_stmt->return_vars_);
    }

    code_ << Indent() << "}\n";
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    std::string var_name = ReserveVarEmitName(assign->var_.get());

    if (auto call = As<Call>(assign->value_)) {
      const std::string& op_name = call->op_->name_;
      if (IsTensorOp(op_name)) {
        if (op_name == "tensor.assemble") {
          HandleTensorAssembleAssign(assign, call);
        } else {
          GenerateTensorOpCode(call, var_name, assign->var_);
        }
      } else if (!IsBuiltinOp(op_name)) {
        std::string result_key;
        if (As<TupleType>(call->GetType())) {
          auto it = call_to_tuple_key_.find(call.get());
          result_key = (it != call_to_tuple_key_.end()) ? it->second : var_name;
        } else {
          result_key = var_name;
        }
        GenerateFunctionCallCode(call, result_key);

        if (!As<TupleType>(call->GetType())) {
          GenerateSingleReturnAlias(call, var_name);
        } else {
          GenerateTupleReturnAliases(call);
        }
      } else {
        INTERNAL_CHECK(false) << "Misplaced builtin op '" << op_name
                              << "' in Orchestration function (should be inside InCore block)";
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
    EmitBatchedAllocTensors(seq->stmts_);
    for (const auto& stmt : seq->stmts_) {
      if (batched_create_stmts_.count(stmt.get())) continue;
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const YieldStmtPtr& yield_stmt) override {
    for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
      std::string value_expr = GenerateExprString(yield_stmt->value_[i]);
      if (i < current_return_vars_.size()) {
        auto yield_var = AsVarLike(yield_stmt->value_[i]);
        if (current_return_vars_[i].get() != yield_var.get()) {
          code_ << Indent() << GetVarName(current_return_vars_[i]) << " = " << value_expr << ";\n";
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
      } else {
        INTERNAL_CHECK(false) << "Misplaced builtin op '" << op_name
                              << "' in Orchestration function (should be inside InCore block)";
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

  // Encode a scalar variable for the orchestration API.
  // float variables must be bit-cast via to_u64(); other types pass through as-is.
  static std::string EncodeScalarVar(const std::string& var_name, const std::string& cpp_type) {
    return cpp_type == "float" ? "to_u64(" + var_name + ")" : var_name;
  }

  // Encode a scalar constant expression for the orchestration API.
  // float literals need to_u64() and an "f" suffix; other types need an explicit (uint64_t) cast.
  static std::string EncodeScalarConst(const std::string& value, const std::string& cpp_type) {
    return cpp_type == "float" ? "to_u64(" + value + "f)" : "(uint64_t)" + value;
  }

  [[nodiscard]] std::string GetExternalTensorName(const std::string& name) const override {
    if (param_name_set_.count(name)) {
      return "ext_" + name;
    }
    return name;
  }

  enum class ParamKind { Input, Output, InOut, Scalar };

  static const char* ParamKindToMethodName(ParamKind kind) {
    switch (kind) {
      case ParamKind::Input:
        return "add_input";
      case ParamKind::Output:
        return "add_output";
      case ParamKind::InOut:
        return "add_inout";
      case ParamKind::Scalar:
        return "add_scalar";
    }
    INTERNAL_CHECK(false) << "Internal error: unexpected ParamKind value";
    return "";
  }

  struct ParamEntry {
    ParamKind kind;
    std::string value;
  };

  std::vector<ParamEntry> BuildTaskParams(const CallPtr& call, const FunctionPtr& callee_func) {
    std::vector<ParamEntry> params;
    const std::string& callee_name = callee_func->name_;

    for (size_t arg_idx = 0; arg_idx < call->args_.size(); ++arg_idx) {
      const auto& arg = call->args_[arg_idx];
      std::string var_name = TryGetVarName(arg);
      if (!var_name.empty()) {
        if (auto scalar_type = As<ScalarType>(arg->GetType())) {
          std::string cpp_type = scalar_type->dtype_.ToCTypeString();
          params.push_back({ParamKind::Scalar, EncodeScalarVar(var_name, cpp_type)});
          continue;
        }

        std::string ext_name = GetExternalTensorName(var_name);

        INTERNAL_CHECK(arg_idx < callee_func->param_directions_.size())
            << "arg count (" << call->args_.size() << ") exceeds param count ("
            << callee_func->param_directions_.size() << ") for callee '" << callee_name << "'";

        ParamDirection dir = callee_func->param_directions_[arg_idx];
        switch (dir) {
          case ParamDirection::Out:
            params.push_back({ParamKind::Output, ext_name});
            break;
          case ParamDirection::InOut:
            params.push_back({ParamKind::InOut, ext_name});
            break;
          case ParamDirection::In:
            params.push_back({ParamKind::Input, ext_name});
            break;
          default:
            INTERNAL_CHECK(false) << "Internal error: unexpected ParamDirection value "
                                  << static_cast<int>(dir);
        }
      } else if (auto const_int = As<ConstInt>(arg)) {
        std::string cpp_type = const_int->dtype().ToCTypeString();
        std::string value = FormatConstIntValue(const_int, cpp_type);
        params.push_back({ParamKind::Scalar, "(uint64_t)" + value});
      } else if (auto const_float = As<ConstFloat>(arg)) {
        std::string cpp_type = const_float->dtype().ToCTypeString();
        std::string value = FormatConstFloatValue(const_float, cpp_type);
        params.push_back({ParamKind::Scalar, EncodeScalarConst(value, cpp_type)});
      } else if (auto const_bool = As<ConstBool>(arg)) {
        params.push_back({ParamKind::Scalar, const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0"});
      }
    }

    // New PTOParam API: tensors must precede scalars (see check_add_tensor_valid() in pto_types.h)
    std::stable_partition(params.begin(), params.end(),
                          [](const ParamEntry& p) { return p.kind != ParamKind::Scalar; });

    return params;
  }

  void GenerateTensorOpCode(const CallPtr& call, const std::string& result_var, const VarPtr& assign_var) {
    const std::string& op_name = call->op_->name_;

    auto& registry = OrchestrationOpRegistry::GetInstance();
    auto codegen_func = registry.Get(op_name);
    INTERNAL_CHECK(codegen_func.has_value()) << "Misplaced tensor op '" << op_name
                                             << "' in Orchestration function (should be inside InCore block)";

    if (op_name == "tensor.create" && assign_var &&
        (declared_var_ptrs_.count(assign_var.get()) || param_name_set_.count(GetVarName(assign_var)))) {
      return;
    }

    std::string emit_var = result_var;
    if (op_name == "tensor.create" && assign_var) {
      declared_var_ptrs_.insert(assign_var.get());
      emit_var = ReserveVarEmitName(assign_var.get());
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

    if (op_name == "tensor.create") {
      EmitAllocBatch({emit_var});
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

  void EmitTaskSubmitAndBind(const std::string& submit_expr) {
    code_ << Indent() << submit_expr << ";\n";
    task_counter_++;
  }

  static constexpr size_t kMaxAllocTensorsArgs = 16;

  static bool ExprRefsAnyOf(const ExprPtr& expr, const std::unordered_set<const Var*>& vars) {
    if (!expr) {
      return false;
    }
    if (auto var = As<Var>(expr)) {
      return vars.count(var.get()) > 0;
    }
    if (auto bin = As<BinaryExpr>(expr)) {
      return ExprRefsAnyOf(bin->left_, vars) || ExprRefsAnyOf(bin->right_, vars);
    }
    if (auto un = As<UnaryExpr>(expr)) {
      return ExprRefsAnyOf(un->operand_, vars);
    }
    if (auto cast_expr = As<Cast>(expr)) {
      return ExprRefsAnyOf(cast_expr->operand_, vars);
    }
    return false;
  }

  bool ShapeDependsOnLocalVars(const CallPtr& call,
                               const std::unordered_set<const Var*>& locally_defined) const {
    auto result_type = As<TensorType>(call->GetType());
    if (!result_type) {
      return false;
    }
    for (const auto& dim : result_type->shape_) {
      if (ExprRefsAnyOf(dim, locally_defined)) {
        return true;
      }
    }
    return false;
  }

  void EmitAllocBatch(const std::vector<std::string>& emit_names) {
    std::string alloc_var = "alloc_" + std::to_string(alloc_counter_++);
    code_ << Indent() << "TaskOutputTensors " << alloc_var << " = alloc_tensors(";
    for (size_t i = 0; i < emit_names.size(); i++) {
      if (i > 0) {
        code_ << ", ";
      }
      code_ << emit_names[i] << "_ci";
    }
    code_ << ");\n";

    for (size_t i = 0; i < emit_names.size(); i++) {
      code_ << Indent() << "const Tensor& " << emit_names[i] << " = " << alloc_var << ".get_ref(" << i
            << ");\n";
    }
  }

  void EmitBatchedAllocTensors(const std::vector<StmtPtr>& stmts) {
    struct PendingCreate {
      std::string emit_name;
      CallPtr call;
    };
    std::vector<PendingCreate> creates;

    std::unordered_set<const Var*> locally_defined;

    for (const auto& stmt : stmts) {
      auto assign = As<AssignStmt>(stmt);
      if (!assign) {
        continue;
      }
      auto call = As<Call>(assign->value_);
      if (!call || call->op_->name_ != "tensor.create") {
        locally_defined.insert(assign->var_.get());
        continue;
      }
      if (declared_var_ptrs_.count(assign->var_.get()) || param_name_set_.count(GetVarName(assign->var_))) {
        continue;
      }
      if (ShapeDependsOnLocalVars(call, locally_defined)) {
        locally_defined.insert(assign->var_.get());
        continue;
      }

      declared_var_ptrs_.insert(assign->var_.get());
      std::string emit_var = ReserveVarEmitName(assign->var_.get());
      creates.push_back({emit_var, call});
      batched_create_stmts_.insert(stmt.get());
    }
    if (creates.empty()) {
      return;
    }

    auto& registry = OrchestrationOpRegistry::GetInstance();
    auto handler = registry.Get("tensor.create");
    INTERNAL_CHECK(handler.has_value()) << "Internal error: tensor.create handler not registered";

    for (size_t batch_start = 0; batch_start < creates.size(); batch_start += kMaxAllocTensorsArgs) {
      size_t batch_end = std::min(batch_start + kMaxAllocTensorsArgs, creates.size());

      for (size_t i = batch_start; i < batch_end; i++) {
        current_result_var_ = creates[i].emit_name;
        std::string gen_code = (*handler)(creates[i].call, *this);
        std::istringstream iss(gen_code);
        std::string line;
        while (std::getline(iss, line)) {
          if (!line.empty()) {
            code_ << Indent() << line << "\n";
          }
        }
      }

      std::vector<std::string> batch_names;
      for (size_t i = batch_start; i < batch_end; i++) {
        batch_names.push_back(creates[i].emit_name);
      }
      EmitAllocBatch(batch_names);
    }
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
    std::string task_var = "params_t" + std::to_string(task_counter_);
    code_ << "\n";
    code_ << ind << "// Task " << task_counter_ << ": " << callee_name << "\n";
    code_ << ind << "Arg " << task_var << ";\n";
    for (const auto& p : params) {
      code_ << ind << task_var << "." << ParamKindToMethodName(p.kind) << "(" << p.value << ");\n";
    }

    std::string submit_expr =
        CoreTypeToSubmitPrefix(core_type) + std::to_string(func_id) + ", " + task_var + ")";
    EmitTaskSubmitAndBind(submit_expr);
  }

  void GenerateGroupCallCode(const CallPtr& call, const FunctionPtr& group_func) {
    std::string group_name = group_func->name_;

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

    auto params = BuildTaskParams(call, aic_func);

    int aic_id = GetOrCreateFuncId(aic_name, func_name_to_id_, next_func_id_);
    int aiv_id = GetOrCreateFuncId(aiv_name, func_name_to_id_, next_func_id_);

    std::string ind = Indent();
    std::string task_var = "params_t" + std::to_string(task_counter_);

    code_ << "\n";
    code_ << ind << "// Group " << group_name << ": MixedKernels (AIC + AIV lanes)\n";
    code_ << ind << "Arg " << task_var << ";\n";
    for (const auto& p : params) {
      code_ << ind << task_var << "." << ParamKindToMethodName(p.kind) << "(" << p.value << ");\n";
    }
    // Split AIV groups dispatch the same kernel on both vector lanes. The
    // kernel body uses tile.get_subblock_idx() to select its lane-local slice.
    std::string third_id = RequiresDualAivDispatch(aiv_func) ? std::to_string(aiv_id) : "INVALID_KERNEL_ID";
    code_ << ind << "MixedKernels mixed_" << task_counter_ << " = {" << aic_id << ", " << aiv_id << ", "
          << third_id << "};\n";

    std::string submit_expr =
        "pto2_rt_submit_task(mixed_" + std::to_string(task_counter_) + ", " + task_var + ")";
    EmitTaskSubmitAndBind(submit_expr);
  }

  // --- Alias generation helpers ---

  static std::vector<size_t> CollectOutIndices(const FunctionPtr& callee) {
    std::vector<size_t> out_indices;
    for (size_t i = 0; i < callee->param_directions_.size(); ++i) {
      if (callee->param_directions_[i] == ParamDirection::Out ||
          callee->param_directions_[i] == ParamDirection::InOut) {
        out_indices.push_back(i);
      }
    }
    return out_indices;
  }

  void EmitTensorAlias(const std::string& alias_name, const CallPtr& call, size_t arg_idx) {
    std::string out_arg = TryGetVarName(call->args_[arg_idx]);
    if (!out_arg.empty() && alias_name != out_arg) {
      code_ << Indent() << "const Tensor& " << alias_name << " = " << GetExternalTensorName(out_arg) << ";\n";
    }
  }

  void GenerateSingleReturnAlias(const CallPtr& call, const std::string& var_name) {
    FunctionPtr callee = program_->GetFunction(call->op_->name_);
    if (!callee) return;
    auto out_indices = CollectOutIndices(callee);
    if (!out_indices.empty()) {
      EmitTensorAlias(var_name, call, out_indices[0]);
    }
  }

  void GenerateTupleReturnAliases(const CallPtr& call) {
    auto tuple_key_it = call_to_tuple_key_.find(call.get());
    if (tuple_key_it == call_to_tuple_key_.end()) return;
    auto elements_it = tuple_var_to_elements_.find(tuple_key_it->second);
    if (elements_it == tuple_var_to_elements_.end()) return;
    FunctionPtr callee = program_->GetFunction(call->op_->name_);
    if (!callee) return;

    auto out_indices = CollectOutIndices(callee);

    for (const auto& elem : elements_it->second) {
      INTERNAL_CHECK(elem.index >= 0 && static_cast<size_t>(elem.index) < out_indices.size())
          << "Internal error: tuple element index " << elem.index << " out of range for " << call->op_->name_
          << " (has " << out_indices.size() << " Out/InOut params)";
      size_t param_idx = out_indices[static_cast<size_t>(elem.index)];
      INTERNAL_CHECK(param_idx < callee->param_directions_.size())
          << "Internal error: resolved param_idx " << param_idx << " out of range for " << call->op_->name_
          << " (has " << callee->param_directions_.size() << " params)";
      if (callee->param_directions_[param_idx] == ParamDirection::InOut) {
        continue;
      }
      std::string elem_name = ReserveVarEmitName(elem.var);
      EmitTensorAlias(elem_name, call, param_idx);
    }
  }

  void VisitScopedBranchBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
    indent_ += 4;
    code_ << Indent() << "PTO2_SCOPE() {\n";
    indent_ += 4;

    auto saved = current_return_vars_;
    current_return_vars_.assign(return_vars.begin(), return_vars.end());
    VisitStmt(body);
    current_return_vars_ = saved;

    indent_ -= 4;
    code_ << Indent() << "}\n";
    indent_ -= 4;
  }

  void HandleTensorAssembleAssign(const AssignStmtPtr& assign, const CallPtr& call) {
    INTERNAL_CHECK(call->args_.size() == 3) << "Internal error: tensor.assemble expects 3 arguments";

    std::string target_name = GenerateExprString(call->args_[0]);
    target_name = GetExternalTensorName(target_name);
    emit_name_map_[assign->var_.get()] = target_name;
  }

  std::string ReserveVarEmitName(const Var* var) {
    auto it = emit_name_map_.find(var);
    if (it != emit_name_map_.end()) {
      return it->second;
    }

    auto parsed = auto_name::Parse(var->name_hint_);
    bool preserve_raw_name = parsed.role.has_value() && *parsed.role == "out";
    std::string base_name = GetSSABaseName(var->name_hint_);
    if (preserve_raw_name || declared_var_names_.count(base_name)) {
      base_name = var->name_hint_;
    }

    std::string emit_name = auto_name::ReserveUniqueName(base_name, declared_var_names_);
    emit_name_map_[var] = emit_name;
    return emit_name;
  }

  std::string ReserveSyntheticEmitName(const std::string& base_name) {
    return auto_name::ReserveUniqueName(base_name, declared_var_names_);
  }

  const ProgramPtr& program_;
  std::map<std::string, int>* func_name_to_id_;
  std::map<std::string, CoreType>* func_name_to_core_type_;
  int* next_func_id_;
  std::unordered_map<const Var*, std::string> emit_name_map_;
  std::set<std::string> declared_var_names_;
  std::set<std::string> param_name_set_;
  std::map<std::string, int> param_name_to_orch_index_;
  std::ostringstream code_;
  int indent_ = 4;
  std::string current_result_var_;
  std::vector<VarPtr> current_return_vars_;
  int task_counter_ = 0;
  int alloc_counter_ = 0;
  std::map<std::string, std::vector<TupleElement>> tuple_var_to_elements_;
  std::map<const Call*, std::string> call_to_tuple_key_;
  std::unordered_set<const Var*> declared_var_ptrs_;
  std::unordered_set<const Stmt*> batched_create_stmts_;
};

OrchestrationResult GenerateOrchestration(const ir::ProgramPtr& program, const ir::FunctionPtr& func) {
  CHECK(program != nullptr) << "Cannot generate orchestration for null program";
  CHECK(func != nullptr) << "Cannot generate orchestration for null function";

  ValidateOrchestrationReferences(program, func);

  std::map<std::string, int> func_name_to_id;
  std::map<std::string, CoreType> func_name_to_core_type;
  int next_func_id = 0;

  OrchestrationInfoCollector info_collector;
  info_collector.VisitStmt(func->body_);

  VarLineageCollector lineage;
  lineage.Initialize(func->params_);
  lineage.VisitStmt(func->body_);

  std::unordered_map<const Var*, std::string> emit_name_map;
  std::set<std::string> param_name_set;
  std::map<std::string, int> param_name_to_orch_index;
  int tensor_param_count = 0;
  struct ScalarParamInfo {
    std::string emit_name;
    ScalarTypePtr scalar_type;
  };
  std::vector<ScalarParamInfo> scalar_params;
  for (const auto& var : func->params_) {
    std::string emit_name = GetSSABaseName(var->name_hint_);
    emit_name_map[var.get()] = emit_name;
    param_name_set.insert(emit_name);
    if (As<TensorType>(var->GetType())) {
      param_name_to_orch_index[emit_name] = tensor_param_count;
      tensor_param_count++;
    } else if (auto stype = As<ScalarType>(var->GetType())) {
      scalar_params.push_back({emit_name, stype});
    }
  }

  for (const auto& [body_var, param_var] : lineage.var_to_param) {
    if (emit_name_map.count(body_var) == 0) {
      auto it = emit_name_map.find(param_var);
      if (it != emit_name_map.end()) {
        emit_name_map[body_var] = it->second;
      }
    }
  }

  int expected_arg_count = tensor_param_count + static_cast<int>(scalar_params.size());

  std::ostringstream oss;

  OrchestrationStmtCodegen stmt_codegen(program, &func_name_to_id, &func_name_to_core_type, &next_func_id,
                                        std::move(emit_name_map), std::move(param_name_set),
                                        std::move(param_name_to_orch_index));
  stmt_codegen.SetCallTupleElements(info_collector.call_tuple_elements);
  stmt_codegen.SetCallToTupleKey(info_collector.call_to_tuple_key);
  stmt_codegen.SetInitialIndent(8);
  stmt_codegen.VisitStmt(func->body_);

  oss << GenerateIncludes(false);

  oss << "extern \"C\" {\n\n";

  oss << GenerateConfigFunction(expected_arg_count);

  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {\n";

  oss << "    // External tensors\n";
  int orch_idx = 0;
  for (const auto& var : func->params_) {
    auto tensor_type = As<TensorType>(var->GetType());
    if (tensor_type) {
      std::string name = auto_name::GetCompatibleBaseName(var->name_hint_);
      oss << GenerateMakeTensorExternal(name, orch_idx, tensor_type, stmt_codegen);
      orch_idx++;
    }
  }

  if (!scalar_params.empty()) {
    oss << "\n    // Scalar params\n";
    for (size_t i = 0; i < scalar_params.size(); ++i) {
      oss << GenerateScalarUnpack(scalar_params[i].emit_name, static_cast<int>(i),
                                  scalar_params[i].scalar_type);
    }
  }

  oss << "\n    PTO2_SCOPE() {\n";
  oss << stmt_codegen.GetGeneratedCode();
  oss << "    }\n";

  oss << "}\n\n";
  oss << "}  // extern \"C\"\n";

  return OrchestrationResult{oss.str(), std::move(func_name_to_id), std::move(func_name_to_core_type)};
}

}  // namespace codegen
}  // namespace pypto
