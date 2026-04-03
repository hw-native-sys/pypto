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

const char TENSOR_HELPER_FUNCTION[] = R"(
static inline Tensor make_tensor_external_2d_dn(void* addr,
    const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    int32_t version = 0) {
    debug_assert(ndims == 2);
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint32_t raw_shapes[2] = {shapes[1], shapes[0]};
    Tensor base = make_tensor_external(addr, raw_shapes, ndims, dtype, false, version);
    uint32_t logical_shapes[2] = {shapes[0], shapes[1]};
    return base.view(logical_shapes, zero_offsets);
}

static inline Tensor make_tensor_2d_dn(
    const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    int32_t version = 0) {
    debug_assert(ndims == 2);
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint32_t raw_shapes[2] = {shapes[1], shapes[0]};
    Tensor base = make_tensor_external(nullptr, raw_shapes, ndims, dtype, false, version);
    uint32_t logical_shapes[2] = {shapes[0], shapes[1]};
    return base.view(logical_shapes, zero_offsets);
}
)";

std::string GenerateConfigFunction(int expected_arg_count) {
  std::ostringstream oss;
  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {\n";
  oss << "    (void)orch_args;\n";
  oss << "    return PTO2OrchestrationConfig{\n";
  oss << "        .expected_arg_count = " << expected_arg_count << ",\n";
  oss << "    };\n";
  oss << "}\n\n";

  oss << TENSOR_HELPER_FUNCTION << "\n";
  return oss.str();
}

bool IsA5Backend() { return pypto::backend::GetBackendType() == pypto::backend::BackendType::Ascend950; }

std::string CoreTypeToSubmitPrefix(CoreType core_type) {
  std::string func = core_type == CoreType::CUBE ? "pto2_rt_submit_aic_task" : "pto2_rt_submit_aiv_task";
  return func + "(";
}

std::string GenerateMakeTensorExternal(const std::string& var_name, int orch_index,
                                       const TensorTypePtr& tensor_type, const CodegenBase& codegen) {
  std::ostringstream oss;

  bool is_dn = tensor_type->IsDNLayout();

  if (is_dn) {
    size_t ndim = tensor_type->shape_.size();
    CHECK(ndim == 2) << "only support 2D tensor for DN layout now";
    oss << "    uint32_t " << var_name << "_shapes[2] = {"
        << "orch_args.tensor(" << orch_index << ").shapes[1], "
        << "orch_args.tensor(" << orch_index << ").shapes[0]};\n";
    oss << "    Tensor ext_" << var_name << " = make_tensor_external_2d_dn("
        << "orch_args.tensor(" << orch_index << ").data_as<void>(), " << var_name << "_shapes, " << ndim
        << ", " << codegen.GetRuntimeDataTypeString(tensor_type->dtype_) << ");\n";
  } else {
    oss << "    Tensor ext_" << var_name << " = from_tensor_arg(orch_args.tensor(" << orch_index << "));\n";
  }

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

  void SetBufferRoots(const std::unordered_map<const Var*, const Var*>& mapping) {
    buffer_root_map_ = mapping;
  }

  void SetAssembleViewInfos(const std::unordered_map<const Var*, AssembleViewInfo>& infos) {
    assemble_view_infos_ = infos;
  }

  void SetNonOptimizableAssembleRoots(const std::unordered_set<const Var*>& roots) {
    non_optimizable_assemble_roots_ = roots;
  }

  void SetEscapingLoopReturns(const std::unordered_set<const Var*>& returns) {
    escaping_loop_returns_ = returns;
  }

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
    auto escaped_it = escaped_loop_return_exprs_.find(var.get());
    if (escaped_it != escaped_loop_return_exprs_.end()) {
      return escaped_it->second;
    }
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
      auto tensor_type = As<TensorType>(return_var->GetType());
      bool is_dn = tensor_type && tensor_type->IsDNLayout();
      auto init_var = AsVarLike(iter_arg->initValue_);
      // Transfer create-pending status from init_var to iter_arg.
      // init_var is only referenced at the loop boundary; iter_arg is what
      // BuildTaskParams will see when the var is passed inside the loop body.
      bool is_create_pending = init_var && tensor_create_var_names_.count(init_var.get()) > 0;
      if (is_create_pending) {
        tensor_create_var_names_.erase(init_var.get());
        tensor_create_var_names_.insert(iter_arg.get());
      }
      if (escaping_loop_returns_.count(return_var.get()) > 0 && is_create_pending && !is_dn) {
        std::string state_name = ReserveSyntheticEmitName(init_var_name + "__loop_state");
        INTERNAL_CHECK(tensor_type) << "Internal error: escaping loop-carried output must be a tensor";
        code_ << Indent() << "Tensor " << state_name << " = make_tensor_external(nullptr, " << init_var_name
              << "_ci_shapes, " << tensor_type->shape_.size() << ", "
              << GetRuntimeDataTypeString(tensor_type->dtype_) << ");\n";
        active_loop_output_states_[init_var_name] = state_name;
        escaped_loop_return_exprs_[return_var.get()] = state_name;
      }
    }

    code_ << Indent() << "for (int64_t " << loop_var << " = " << start_expr << "; " << loop_var << " < "
          << stop_expr << "; " << loop_var << " += " << step_expr << ") {\n";
    indent_ += 4;
    code_ << Indent() << "PTO2_SCOPE() {\n";
    indent_ += 4;

    auto saved = current_return_vars_;
    auto saved_active_loop_output_states = active_loop_output_states_;
    current_return_vars_.clear();
    VisitStmt(for_stmt->body_);
    current_return_vars_ = saved;
    active_loop_output_states_ = saved_active_loop_output_states;

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
    std::string value;    // expression passed to the method
    std::string out_var;  // non-empty for internal Out tensors: the Tensor variable to bind via get_ref
    bool out_var_is_new_decl = false;  // true: emit "const Tensor& var = get_ref()" (non-DN);
                                       // false: emit "var = get_ref()" (DN, pre-declared placeholder)
    const Var* out_var_ptr = nullptr;  // raw pointer into tensor_create_var_names_
  };

  struct InternalOutVar {
    std::string name;
    bool is_new_decl;
    const Var* var_ptr = nullptr;
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
          params.push_back({ParamKind::Scalar, EncodeScalarVar(var_name, cpp_type), ""});
          continue;
        }

        std::string ext_name = GetExternalTensorName(var_name);
        auto arg_var = AsVarLike(arg);

        INTERNAL_CHECK(arg_idx < callee_func->param_directions_.size())
            << "arg count (" << call->args_.size() << ") exceeds param count ("
            << callee_func->param_directions_.size() << ") for callee '" << callee_name << "'";

        // Push an "add_output" entry for a tensor.create-allocated argument.
        auto push_create_output = [&]() {
          auto tt = As<TensorType>(arg->GetType());
          bool is_dn = tt && tt->IsDNLayout();
          params.push_back(
              {ParamKind::Output, var_name + "_ci", var_name, /*out_var_is_new_decl=*/!is_dn, arg_var.get()});
        };

        ParamDirection dir = callee_func->param_directions_[arg_idx];
        switch (dir) {
          case ParamDirection::Out:
            if (arg_var && tensor_create_var_names_.count(arg_var.get())) {
              push_create_output();
            } else {
              params.push_back({ParamKind::InOut, ext_name, "", false});
            }
            break;
          case ParamDirection::InOut:
            params.push_back({ParamKind::InOut, ext_name, ""});
            break;
          case ParamDirection::In:
            if (arg_var && tensor_create_var_names_.count(arg_var.get())) {
              push_create_output();
            } else {
              params.push_back({ParamKind::Input, ext_name, ""});
            }
            break;
          default:
            INTERNAL_CHECK(false) << "Internal error: unexpected ParamDirection value "
                                  << static_cast<int>(dir);
        }
      } else if (auto const_int = As<ConstInt>(arg)) {
        std::string cpp_type = const_int->dtype().ToCTypeString();
        std::string value = FormatConstIntValue(const_int, cpp_type);
        params.push_back({ParamKind::Scalar, "(uint64_t)" + value, ""});
      } else if (auto const_float = As<ConstFloat>(arg)) {
        std::string cpp_type = const_float->dtype().ToCTypeString();
        std::string value = FormatConstFloatValue(const_float, cpp_type);
        params.push_back({ParamKind::Scalar, EncodeScalarConst(value, cpp_type), ""});
      } else if (auto const_bool = As<ConstBool>(arg)) {
        params.push_back({ParamKind::Scalar, const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0", ""});
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
    if (!codegen_func.has_value()) {
      return;
    }

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

    std::string gen_code;
    if (op_name == "tensor.create" && assign_var) {
      auto assemble_view = TryGenerateAssembleViewForCreate(call, assign_var.get(), emit_var);
      if (assemble_view.has_value()) {
        gen_code = *assemble_view;
      } else {
        tensor_create_var_names_.insert(assign_var.get());
      }
    }
    if (gen_code.empty()) {
      gen_code = (*codegen_func)(call, *this);
    }

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

  void EmitTaskSubmitAndBind(const std::string& submit_expr, const std::vector<ParamEntry>& params) {
    std::vector<InternalOutVar> internal_out_vars;
    for (const auto& p : params) {
      if (p.kind == ParamKind::Output && !p.out_var.empty()) {
        internal_out_vars.push_back({p.out_var, p.out_var_is_new_decl, p.out_var_ptr});
      }
    }

    std::string ind = Indent();
    if (internal_out_vars.empty()) {
      code_ << ind << submit_expr << ";\n";
    } else {
      std::string outs_var = "outs_t" + std::to_string(task_counter_);
      code_ << ind << "TaskOutputTensors " << outs_var << " = " << submit_expr << ";\n";
      for (size_t i = 0; i < internal_out_vars.size(); ++i) {
        const auto& ov = internal_out_vars[i];
        if (ov.is_new_decl) {
          code_ << ind << "const Tensor& " << ov.name << " = " << outs_var << ".get_ref(" << i << ");\n";
          auto state_it = active_loop_output_states_.find(ov.name);
          if (state_it != active_loop_output_states_.end()) {
            code_ << ind << state_it->second << " = " << ov.name << ";\n";
          }
        } else {
          code_ << ind << ov.name << " = " << outs_var << ".get_ref(" << i << ");\n";
        }
        if (ov.var_ptr) tensor_create_var_names_.erase(ov.var_ptr);
      }
    }

    task_counter_++;
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
    EmitTaskSubmitAndBind(submit_expr, params);
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
    code_ << ind << "// Group " << group_name << ": MixedKernels (AIC + AIV)\n";
    code_ << ind << "Arg " << task_var << ";\n";
    for (const auto& p : params) {
      code_ << ind << task_var << "." << ParamKindToMethodName(p.kind) << "(" << p.value << ");\n";
    }
    auto split_mode = group_func->GetSplitMode();
    std::string third_id = split_mode.has_value() ? std::to_string(aiv_id) : "INVALID_KERNEL_ID";
    code_ << ind << "MixedKernels mixed_" << task_counter_ << " = {" << aic_id << ", " << aiv_id << ", "
          << third_id << "};\n";

    std::string submit_expr =
        "pto2_rt_submit_task(mixed_" + std::to_string(task_counter_) + ", " + task_var + ")";
    EmitTaskSubmitAndBind(submit_expr, params);
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

  // --- Buffer root / assemble view helpers ---

  const Var* ResolveBufferRoot(const Var* var) const {
    auto it = buffer_root_map_.find(var);
    return it != buffer_root_map_.end() ? it->second : var;
  }

  std::optional<std::string> TryGenerateAssembleViewForCreate(const CallPtr& call, const Var* assign_var,
                                                              const std::string& emit_var) {
    const Var* root = ResolveBufferRoot(assign_var);
    if (root != assign_var) {
      return std::nullopt;
    }
    if (non_optimizable_assemble_roots_.count(root) > 0) {
      return std::nullopt;
    }
    auto it = assemble_view_infos_.find(root);
    if (it == assemble_view_infos_.end()) {
      return std::nullopt;
    }

    auto result_type = As<TensorType>(call->GetType());
    INTERNAL_CHECK(result_type) << "Internal error: tensor.create must return TensorType";

    size_t ndim = result_type->shape_.size();
    size_t array_len = ndim == 0 ? 1 : ndim;
    std::ostringstream oss;
    oss << "uint32_t " << emit_var << "_shapes[" << array_len << "] = {";
    if (ndim == 0) {
      oss << "1";
    } else {
      for (size_t i = 0; i < ndim; ++i) {
        if (i > 0) oss << ", ";
        oss << GenerateExprString(result_type->shape_[i]);
      }
    }
    oss << "};\n";

    INTERNAL_CHECK(it->second.offset_tuple != nullptr)
        << "Internal error: tensor.assemble offset must be MakeTuple";
    oss << "uint32_t " << emit_var << "_offsets[" << array_len << "] = {";
    if (ndim == 0) {
      oss << "0";
    } else {
      for (size_t i = 0; i < ndim; ++i) {
        if (i > 0) oss << ", ";
        INTERNAL_CHECK(i < it->second.offset_tuple->elements_.size())
            << "Internal error: tensor.assemble offset rank mismatch";
        oss << GenerateExprString(it->second.offset_tuple->elements_[i]);
      }
    }
    oss << "};\n";

    std::string target_name = GenerateExprString(it->second.target_expr);
    target_name = GetExternalTensorName(target_name);
    oss << "Tensor " << emit_var << " = " << target_name << ".view(" << emit_var << "_shapes, " << emit_var
        << "_offsets);";

    emitted_assemble_view_roots_.insert(root);
    return oss.str();
  }

  bool HandleTensorAssembleAssign(const AssignStmtPtr& assign, const CallPtr& call) {
    INTERNAL_CHECK(call->args_.size() == 3) << "Internal error: tensor.assemble expects 3 arguments";

    std::string target_name = GenerateExprString(call->args_[0]);
    target_name = GetExternalTensorName(target_name);
    emit_name_map_[assign->var_.get()] = target_name;

    auto source_var = AsVarLike(call->args_[1]);
    if (!source_var) {
      return false;
    }
    const Var* source_root = ResolveBufferRoot(source_var.get());
    if (non_optimizable_assemble_roots_.count(source_root) > 0) {
      return false;
    }
    return emitted_assemble_view_roots_.count(source_root) > 0;
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
  std::unordered_map<const Var*, const Var*> buffer_root_map_;
  std::unordered_map<const Var*, AssembleViewInfo> assemble_view_infos_;
  std::unordered_set<const Var*> non_optimizable_assemble_roots_;
  std::unordered_set<const Var*> emitted_assemble_view_roots_;
  std::unordered_set<const Var*> tensor_create_var_names_;
  std::ostringstream code_;
  int indent_ = 4;
  std::string current_result_var_;
  std::vector<VarPtr> current_return_vars_;
  int task_counter_ = 0;
  std::map<std::string, std::vector<TupleElement>> tuple_var_to_elements_;
  std::map<const Call*, std::string> call_to_tuple_key_;
  std::unordered_set<const Var*> escaping_loop_returns_;
  std::unordered_map<const Var*, std::string> escaped_loop_return_exprs_;
  std::unordered_map<std::string, std::string> active_loop_output_states_;
  std::unordered_set<const Var*> declared_var_ptrs_;
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

  BufferRootCollector buffer_info(program);
  buffer_info.Initialize(func->params_);
  buffer_info.VisitStmt(func->body_);

  AssembleViewOptimizer assemble_opt(buffer_info.buffer_roots);
  assemble_opt.VisitStmt(func->body_);

  LoopEscapeInfoCollector loop_escape_info;
  loop_escape_info.VisitStmt(func->body_);

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
  stmt_codegen.SetBufferRoots(buffer_info.buffer_roots);
  stmt_codegen.SetAssembleViewInfos(assemble_opt.assemble_view_infos);
  stmt_codegen.SetNonOptimizableAssembleRoots(assemble_opt.non_optimizable_roots);
  stmt_codegen.SetEscapingLoopReturns(loop_escape_info.escaping_loop_returns);
  stmt_codegen.SetInitialIndent(8);
  stmt_codegen.VisitStmt(func->body_);

  oss << GenerateIncludes(false);

  oss << "extern \"C\" {\n\n";

  oss << GenerateConfigFunction(expected_arg_count);

  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args, "
         "int orch_thread_num, int orch_thread_index) {\n";
  oss << "    (void)orch_thread_num;\n";
  oss << "    (void)orch_thread_index;\n\n";

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
