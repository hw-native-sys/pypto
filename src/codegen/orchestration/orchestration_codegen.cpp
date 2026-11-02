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

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/codegen_preconditions.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "src/codegen/orchestration/orchestration_stmt_codegen.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

CoreType InferFunctionCoreType(const FunctionPtr& func) {
  // After ExpandMixedKernel runs (part of the Default pipeline), every InCore
  // function reaching codegen has been split into AIC, AIV, or Group / Spmd
  // wrappers. The two callers of this function (GenerateFunctionCallCode and
  // GenerateSpmdCallCode) both filter Spmd / Group out before invoking it.
  // Tests that bypass the pipeline must declare their kernels with the
  // appropriate AIC / AIV type explicitly so codegen sees the concrete core
  // type without re-deriving from body memory spaces.
  switch (func->func_type_) {
    case FunctionType::AIC:
      return CoreType::CUBE;
    case FunctionType::AIV:
      return CoreType::VECTOR;
    default:
      INTERNAL_UNREACHABLE_SPAN(func->span_)
          << "InferFunctionCoreType expects AIC or AIV (Spmd/Group are filtered upstream); got "
          << FunctionTypeToString(func->func_type_) << " on function '" << func->name_
          << "'. Either run ExpandMixedKernel before codegen or declare the function "
          << "with @pl.function(type=pl.FunctionType.AIC|AIV) directly.";
  }
}

namespace {

const char* ParamDirectionToRuntimeName(ParamDirection dir) {
  switch (dir) {
    case ParamDirection::In:
      return "IN";
    case ParamDirection::Out:
      return "OUT";
    case ParamDirection::InOut:
      return "INOUT";
  }
  INTERNAL_CHECK(false) << "Internal error: unexpected ParamDirection value";
  return "";
}

// ---------------------------------------------------------------------------
// Template / boilerplate generation helpers
// ---------------------------------------------------------------------------

std::string GenerateIncludes(bool include_optional, bool include_vector, bool include_graph) {
  std::ostringstream oss;
  oss << "#include <stddef.h>\n";
  oss << "#include <stdint.h>\n";
  oss << "#include <stdio.h>\n";
  if (include_vector) {
    oss << "#include <vector>\n";
  }
  if (include_optional) {
    oss << "#include <optional>\n";
  }
  oss << "\n";
  oss << "#include \"orchestration_api.h\"\n";
  // The per-runtime `tensor.h` shim sits first on the orchestration include
  // path and defines `Tensor` as whichever runtime this is built for
  // (`simpler::tmr::Tensor` / `simpler::hbg::Tensor`). Orchestration works in
  // that type, not in the 72-byte boundary `ChipTensor` (simpler#1974), and
  // `orchestration_api.h` does not reach the shim on its own.
  oss << "#include \"tensor.h\"\n";
  if (include_graph) {
    oss << "#include \"host_build_graph/host/graph_boundary_match.h\"\n";
  }
  oss << "\n";
  return oss.str();
}

std::string GenerateScalarUnpack(const std::string& var_name, int scalar_index,
                                 const ScalarTypePtr& scalar_type) {
  std::ostringstream oss;
  std::string cpp_type = scalar_type->dtype_.ToCTypeString();
  oss << "    " << cpp_type << " " << var_name << " = orch_args.scalar<" << cpp_type << ">(" << scalar_index
      << ");\n";
  return oss.str();
}

std::string GenerateConfigFunction(int expected_arg_count) {
  std::ostringstream oss;
  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs& orch_args) {\n";
  oss << "    (void)orch_args;\n";
  oss << "    return OrchestrationConfig{\n";
  oss << "        .expected_arg_count = " << expected_arg_count << ",\n";
  oss << "    };\n";
  oss << "}\n\n";
  return oss.str();
}

// Whether this parameter was rewritten into the blocked NZ form.
//
// ``BlockNzTensorViews`` rewrites an NZ parameter to the blocked rank-5 shape
// the backend addresses, while the caller keeps allocating and passing the
// logical one — the same bytes under two spellings (see
// ``_arg_shape_as_declared`` on the Python side). The runtime ``Tensor`` that
// arrives therefore carries the logical rank.
bool IsBlockedNzParam(const TensorTypePtr& tensor_type) {
  return tensor_type && tensor_type->tensor_view_.has_value() &&
         tensor_type->tensor_view_->layout == TensorLayout::NZ;
}

std::string GenerateMakeTensorExternal(const std::string& var_name, int orch_index,
                                       const TensorTypePtr& tensor_type, const std::string& logical_var,
                                       const std::string& shape_var) {
  std::ostringstream oss;
  if (!IsBlockedNzParam(tensor_type)) {
    oss << "    const Tensor& ext_" << var_name << " = orch_args.tensor(" << orch_index << ").ref();\n";
    return oss.str();
  }

  // Restate an NZ argument in blocked terms once, at the boundary: the blocked
  // form holds the same elements in the same order, so this is a metadata-only
  // reshape. Every later ``Tensor::view`` clamps its blocked extents against
  // this parent, and clamping them against the logical rank would read past the
  // incoming rank and silently yield an empty view.
  const std::vector<ExprPtr>& blocked_shape = tensor_type->shape_;
  oss << "    const Tensor& " << logical_var << " = orch_args.tensor(" << orch_index << ").ref();\n";
  oss << "    uint32_t " << shape_var << "[" << blocked_shape.size() << "] = {";
  for (size_t i = 0; i < blocked_shape.size(); ++i) {
    if (i > 0) oss << ", ";
    if (auto extent = As<ConstInt>(blocked_shape[i])) {
      oss << extent->value_;
      continue;
    }
    // The blocked form is rank 5, and its four trailing extents come from the
    // fractal plane, which `BlockNzShape` requires static. Leading axes above
    // rank 3 are folded into the batch, which `FoldNzLeadingExtents` requires
    // static too. A dynamic extent is therefore the batch of a rank-3 tensor —
    // its own leading extent, which the incoming logical tensor carries at
    // index 0 and passes through unchanged.
    INTERNAL_CHECK(i == 0 && blocked_shape.size() == 5)
        << "Internal error: NZ parameter '" << var_name << "' has a dynamic blocked extent at axis " << i
        << ", which only the batch of a rank-3 NZ tensor may be";
    oss << logical_var << ".shapes[0]";
  }
  oss << "};\n";
  oss << "    Tensor ext_" << var_name << " = " << logical_var << ".reshape(" << shape_var << ", "
      << blocked_shape.size() << ");\n";
  return oss.str();
}

class CodegenEffectiveUseCollector : public var_collectors::VarDefUseCollector {
 protected:
  void VisitStmt_(const ReturnStmtPtr&) override {}
};

/// Whether `code` mentions `name` as a whole C++ identifier rather than as a
/// substring of a longer one (`M` must not match inside `M_DYN` or `ext_M`).
///
/// Occurrences inside a `//` comment do not count: they are not references, and
/// defining a symbol for one would emit an unused variable (`-Wunused-variable`).
bool ReferencesIdentifier(const std::string& code, const std::string& name) {
  if (name.empty()) return false;
  auto is_ident_char = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_'; };
  for (size_t pos = code.find(name); pos != std::string::npos; pos = code.find(name, pos + 1)) {
    const size_t end = pos + name.size();
    const bool left_ok = pos == 0 || !is_ident_char(code[pos - 1]);
    const bool right_ok = end >= code.size() || !is_ident_char(code[end]);
    if (!left_ok || !right_ok) continue;
    const size_t line_start = code.rfind('\n', pos) + 1;  // npos + 1 == 0 for the first line
    const size_t comment = code.find("//", line_start);
    if (comment == std::string::npos || comment > pos) return true;
  }
  return false;
}

}  // namespace

/// Collect, in deterministic order, the Graph functions this entry launches.
std::vector<FunctionPtr> CollectReferencedGraphFunctions(const ProgramPtr& program,
                                                         const FunctionPtr& entry) {
  class Collector : public IRVisitor {
   public:
    explicit Collector(ProgramPtr program) : program_(std::move(program)) {}
    std::vector<FunctionPtr> found;

   protected:
    void VisitExpr_(const CallPtr& op) override {
      IRVisitor::VisitExpr_(op);
      Record(op->op_);
    }
    void VisitExpr_(const SubmitPtr& op) override {
      IRVisitor::VisitExpr_(op);
      Record(op->op_);
    }

   private:
    void Record(const OpPtr& callee_op) {
      auto gvar = As<GlobalVar>(callee_op);
      if (!gvar || !program_) return;
      auto callee = program_->GetFunction(gvar->name_);
      if (!callee || callee->func_type_ != FunctionType::Graph) return;
      if (!seen_.insert(callee->name_).second) return;  // one definition per graph
      found.push_back(callee);
    }

    ProgramPtr program_;
    std::unordered_set<std::string> seen_;
  };

  Collector collector(program);
  if (entry->body_) collector.VisitStmt(entry->body_);
  return collector.found;
}

/// Emit `int64_t <symbol> = <param>.shape()[axis];` for every dynamic extent a
/// body references.
///
/// A tensor parameter whose shape carries a `Var` extent declares that symbol;
/// the body then reads it as a plain identifier. Both the entry and each Graph
/// helper bind their tensor parameters at the top of their own function, so both
/// need these definitions — a Graph body referencing one without this block
/// names an undeclared variable and the generated file does not compile.
///
/// @p pre_defined seeds the dedup set with names already spoken for (scalar
/// parameters, body-local definitions): a symbol sharing one of those names is
/// not what the body's occurrences refer to, and defining it would shadow the
/// real one. Dedup is by emitted *name* rather than by `Var`, because a symbol
/// carries no emit-name mapping and prints as its name hint.
std::string GenerateDynamicDimDefs(const std::vector<VarPtr>& params, const std::string& body_code,
                                   OrchestrationStmtCodegen* codegen,
                                   std::unordered_set<std::string> pre_defined) {
  std::vector<std::string> defs;
  for (const auto& var : params) {
    auto tensor_type = AsTensorTypeLike(var->GetType());
    if (!tensor_type) continue;
    const std::string param_name = auto_name::GetCompatibleBaseName(var->name_hint_);
    for (size_t axis = 0; axis < tensor_type->shape_.size(); ++axis) {
      auto extent = As<Var>(tensor_type->shape_[axis]);
      if (!extent) continue;
      const std::string symbol_name = codegen->GetVarName(extent);
      if (!pre_defined.insert(symbol_name).second) continue;
      if (!ReferencesIdentifier(body_code, symbol_name)) continue;
      defs.push_back("    int64_t " + symbol_name + " = " +
                     codegen->GetTensorShapeDim(param_name, static_cast<int64_t>(axis)) + ";\n");
    }
  }
  if (defs.empty()) return "";
  std::string out = "\n    // Dynamic-dim symbols (extent of the declaring argument)\n";
  for (const auto& def : defs) out += def;
  return out;
}

/// Emit one named file-scope function per Graph the entry launches.
///
/// The body is generated by a *second* codegen instance. Two settings differ
/// from the entry's and both matter:
///
/// * ``param_name_set`` is empty. `GetExternalTensorName` rewrites any name in
///   that set to ``ext_<name>``, which is right for the entry (whose parameters
///   arrive through ``orch_args``) and wrong here, where they are ordinary
///   function parameters bound at the top of the body. The parameter names are
///   still handed to `ReserveDeclaredNames`, which is the other half of what
///   ``param_name_set`` does for the entry: it keeps a body Var from taking a
///   name the prologue already declares (issue #2605).
/// * A task-var prefix, because this instance's counters restart at 0.
///
/// Preserve the InheritableScalar returned by args.scalar(). It carries the
/// boundary parameter's origin through task argument forwarding. Converting it
/// to an integer would freeze the recording invocation's value on every replay.
std::string GenerateGraphFunctions(const ProgramPtr& program, const FunctionPtr& entry,
                                   KernelRegistrations* registrations,
                                   GraphConfigScalarSlots* static_scalar_slots) {
  std::ostringstream oss;
  const auto graph_functions = CollectReferencedGraphFunctions(program, entry);
  if (graph_functions.empty()) return "";
  // Reuse the runtime's bounded alias-partition algorithm. Absolute addresses and
  // a uniform slide of a partition do not affect the recorded dependency layout.
  // Shapes, storage sizes and relative view offsets do, so they join the Config.
  oss << R"(
static std::vector<uint64_t> pypto_graph_boundary_layout(const GraphTaskArgs& args) {
    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> reps{};
    // The runtime reports malformed boundaries at submission. Do not read
    // uninitialized partition entries while preparing their Config.
    if (args.has_error() || !graph_alias_partition(args, reps.data())) return {};
    std::vector<uint64_t> layout;
    layout.reserve(1 + args.tensor_count() * (8 + 2 * MAX_TENSOR_DIMS));
    layout.push_back(args.tensor_count());
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        const auto& tensor = args.tensor(i).ref();
        const auto& representative = args.tensor(reps[i]).ref();
        layout.insert(layout.end(), {
            tensor.buffer.size, static_cast<uint64_t>(tensor.ndims),
            static_cast<uint64_t>(tensor.dtype), static_cast<uint64_t>(args.tag(i)),
            static_cast<uint64_t>(tensor.manual_dep), static_cast<uint64_t>(tensor.is_contiguous),
            static_cast<uint64_t>(reps[i]), tensor.start_offset - representative.start_offset});
        for (int32_t dim = 0; dim < tensor.ndims; ++dim) {
            layout.push_back(tensor.shapes[dim]);
            layout.push_back(tensor.strides[dim]);
        }
    }
    return layout;
}

)";
  int graph_index = 0;
  for (const auto& graph_func : graph_functions) {
    CodegenEffectiveUseCollector use_collector;
    use_collector.VisitStmt(graph_func->body_);

    // Same analyses the entry runs. Without the tuple mappings,
    // GenerateTupleReturnAliases / GenerateSubmitReturnAliases find an empty map
    // and return early, so a multi-output call or a `pl.submit` inside a Graph
    // binds neither its output tensors nor its TaskId.
    OrchestrationInfoCollector graph_info;
    graph_info.VisitStmt(graph_func->body_);

    // Seeded with the *same* string the parameter declarations below use.
    // Unseeded, `GetVarName` falls back to `GetSSABaseName(name_hint_)`, which is
    // a different normalisation from `auto_name::GetCompatibleBaseName`; wherever
    // the two disagree the body would reference a symbol the header never
    // declared. Seeding makes them agree by construction.
    std::unordered_map<const Var*, std::string> emit_name_map;
    std::set<std::string> param_emit_names;
    for (const auto& param : graph_func->params_) {
      std::string name = auto_name::GetCompatibleBaseName(param->name_hint_);
      emit_name_map[param.get()] = name;
      param_emit_names.insert(std::move(name));
    }
    OrchestrationStmtCodegen body_codegen(program, registrations, std::move(emit_name_map),
                                          /*param_name_set=*/{}, /*param_name_to_orch_index=*/{},
                                          /*dist_param_to_ctx_param=*/{});
    body_codegen.SetTaskVarPrefix("g" + std::to_string(graph_index) + "_");
    body_codegen.SetGraphScalarParams(graph_func->params_);
    // The prologue below declares one C++ name per parameter. They are not in
    // ``param_name_set``, so reserve them explicitly or a body SSA rename will
    // shadow one (issue #2605; see ReserveDeclaredNames).
    body_codegen.ReserveDeclaredNames(param_emit_names);
    body_codegen.SetCallTupleElements(graph_info.call_tuple_elements);
    body_codegen.SetTupleVarToKey(graph_info.tuple_var_to_key);
    body_codegen.SetEffectiveUses(std::move(use_collector.var_uses));
    body_codegen.PrepareCrossScopeTaskIdHoists(graph_func->body_);
    body_codegen.SetInitialIndent(4);
    body_codegen.VisitStmt(graph_func->body_);
    (*static_scalar_slots)[graph_func->name_] = body_codegen.GetGraphScalarValueReads();

    // Construction values belong to Config; boundary scalar handles only forward
    // per-invocation values. Compare the same encoded bits that participate in the
    // key, including signed zero and NaN payloads of floating-point parameters.
    const auto& read_slots = (*static_scalar_slots)[graph_func->name_];
    const std::set<size_t> ordered_slots(read_slots.begin(), read_slots.end());
    const std::string config_type = GraphFunctionSymbol(graph_func->name_) + "Config";
    oss << "struct " << config_type << " : simpler::hbg::ConfigBase {\n";
    oss << "    std::vector<uint64_t> boundary;\n";
    std::string equality = "boundary == other.boundary";
    for (size_t slot : ordered_slots) {
      const std::string field = "scalar_" + std::to_string(slot);
      oss << "    uint64_t " << field << " = 0;\n";
      equality += " && " + field + " == other." + field;
    }
    oss << "    uint64_t hash() const {\n        uint64_t h = 0;\n";
    oss << "        for (uint64_t field : boundary) h = simpler::hbg::config_hash_field(h, field);\n";
    for (size_t slot : ordered_slots) {
      oss << "        h = simpler::hbg::config_hash_field(h, scalar_" << slot << ");\n";
    }
    oss << "        return h;\n    }\n";
    oss << "    bool operator==(const " << config_type << "& other) const { return " << equality
        << "; }\n};\n\n";
    oss << "static void " << GraphFunctionSymbol(graph_func->name_) << "(const GraphTaskArgs& args";
    if (!ordered_slots.empty()) oss << ", const " << config_type << "& _pypto_graph_config";
    oss << ") {\n";

    int tensor_index = 0;
    int scalar_index = 0;
    for (const auto& param : graph_func->params_) {
      const std::string name = auto_name::GetCompatibleBaseName(param->name_hint_);
      if (AsTensorTypeLike(param->GetType())) {
        oss << "    const Tensor& " << name << " = args.tensor(" << tensor_index << ").ref();\n";
        ++tensor_index;
        continue;
      }
      oss << "    const auto " << name << " = args.scalar(" << scalar_index << ");\n";
      ++scalar_index;
    }

    const std::string graph_body_code = body_codegen.GetGeneratedCode();
    // A Graph body may carry a dynamic loop as long as it launches nothing
    // (LegalizeGraphBoundary only forbids a *launch* under a non-constant bound),
    // so the symbol it reads has to be defined here just as it is at the entry.
    std::unordered_set<std::string> graph_defined;
    CodegenEffectiveUseCollector graph_body_vars;
    graph_body_vars.VisitStmt(graph_func->body_);
    for (const auto* def : graph_body_vars.var_defs) graph_defined.insert(GetSSABaseName(def->name_hint_));
    for (const auto& param : graph_func->params_) {
      if (!AsTensorTypeLike(param->GetType())) {
        graph_defined.insert(auto_name::GetCompatibleBaseName(param->name_hint_));
      }
    }
    oss << GenerateDynamicDimDefs(graph_func->params_, graph_body_code, &body_codegen,
                                  std::move(graph_defined));

    oss << graph_body_code;
    oss << "}\n\n";
    ++graph_index;
  }
  return oss.str();
}

OrchestrationResult GenerateOrchestration(const ir::ProgramPtr& program, const ir::FunctionPtr& func) {
  CHECK(program != nullptr) << "Cannot generate orchestration for null program";
  CHECK(func != nullptr) << "Cannot generate orchestration for null function";
  VerifyOrchestrationCodegenPreconditions(program, func);

  KernelRegistrations registrations;

  OrchestrationInfoCollector info_collector;
  info_collector.VisitStmt(func->body_);

  VarLineageCollector lineage(program);
  lineage.Initialize(func->params_);
  lineage.VisitStmt(func->body_);

  CodegenEffectiveUseCollector use_collector;
  use_collector.VisitStmt(func->body_);

  std::unordered_map<const Var*, std::string> emit_name_map;
  std::set<std::string> param_name_set;
  std::set<std::string> external_tensor_names;
  std::map<std::string, int> param_name_to_orch_index;
  int tensor_param_count = 0;
  struct ScalarParamInfo {
    std::string emit_name;
    TypePtr type;
  };
  std::vector<ScalarParamInfo> scalar_params;
  // Names of DistributedTensor params and their materialized CommCtxType params,
  // both in IR-param order. The mapping lets ``pld.system.get_comm_ctx`` alias
  // the explicit uint64_t ctx param already unpacked from ``orch_args.scalar``.
  std::vector<std::string> dist_tensor_param_names;
  std::vector<std::string> comm_ctx_param_names;
  std::vector<std::string> orchestration_signature;
  INTERNAL_CHECK(func->params_.size() == func->param_directions_.size())
      << "Internal error: orchestration function has " << func->params_.size() << " params but "
      << func->param_directions_.size() << " param directions";
  for (size_t param_idx = 0; param_idx < func->params_.size(); ++param_idx) {
    const auto& var = func->params_[param_idx];
    std::string emit_name = GetSSABaseName(var->name_hint_);
    emit_name_map[var.get()] = emit_name;
    param_name_set.insert(emit_name);
    if (auto tensor_type = AsTensorTypeLike(var->GetType())) {
      external_tensor_names.insert("ext_" + auto_name::GetCompatibleBaseName(var->name_hint_));
      param_name_to_orch_index[emit_name] = tensor_param_count;
      tensor_param_count++;
      orchestration_signature.emplace_back(ParamDirectionToRuntimeName(func->param_directions_[param_idx]));
      if (As<DistributedTensorType>(var->GetType())) {
        dist_tensor_param_names.push_back(emit_name);
      }
    } else if (As<ScalarType>(var->GetType()) || IsA<CommCtxType>(var->GetType())) {
      scalar_params.push_back({emit_name, var->GetType()});
      if (IsA<CommCtxType>(var->GetType())) {
        comm_ctx_param_names.push_back(emit_name);
      }
    }
  }

  std::unordered_map<std::string, std::string> dist_param_to_ctx_param;
  INTERNAL_CHECK(dist_tensor_param_names.size() == comm_ctx_param_names.size())
      << "Materialized orchestration signature has " << dist_tensor_param_names.size()
      << " DistributedTensor params but " << comm_ctx_param_names.size() << " CommCtx params.";
  for (size_t i = 0; i < dist_tensor_param_names.size(); ++i) {
    dist_param_to_ctx_param[dist_tensor_param_names[i]] = comm_ctx_param_names[i];
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

  OrchestrationStmtCodegen stmt_codegen(program, &registrations, std::move(emit_name_map),
                                        std::move(param_name_set), std::move(param_name_to_orch_index),
                                        std::move(dist_param_to_ctx_param));
  // Reserve every external reference before allocating NZ temporaries, even
  // those for parameters later in the signature. Reserve the temporaries before
  // visiting the body too, so body variables share the same naming discipline.
  stmt_codegen.ReserveDeclaredNames(external_tensor_names);
  std::ostringstream external_tensors;
  int orch_idx = 0;
  for (const auto& var : func->params_) {
    if (auto tensor_type = AsTensorTypeLike(var->GetType())) {
      const std::string name = auto_name::GetCompatibleBaseName(var->name_hint_);
      std::string logical_var;
      std::string shape_var;
      if (IsBlockedNzParam(tensor_type)) {
        logical_var = stmt_codegen.ReserveSyntheticEmitName("ext_" + name + "_logical");
        shape_var = stmt_codegen.ReserveSyntheticEmitName("ext_" + name + "_nz_shapes");
      }
      external_tensors << GenerateMakeTensorExternal(name, orch_idx, tensor_type, logical_var, shape_var);
      ++orch_idx;
    }
  }
  stmt_codegen.SetCallTupleElements(info_collector.call_tuple_elements);
  stmt_codegen.SetTupleVarToKey(info_collector.tuple_var_to_key);
  stmt_codegen.SetEffectiveUses(std::move(use_collector.var_uses));
  stmt_codegen.PrepareCrossScopeTaskIdHoists(func->body_);
  // MaterializeRuntimeScopes wraps the whole body in an AUTO RuntimeScopeStmt,
  // so the outermost ``SIMPLER_SCOPE()`` is now emitted by the scope handler at the
  // base indent (4) rather than by a hardcoded wrapper below; the body lands at 8.
  stmt_codegen.SetInitialIndent(4);
  // Generate Graph bodies first: their actual value reads determine which
  // boundary slots the entry must declare static for safe replay.
  GraphConfigScalarSlots graph_config_scalar_slots;
  const std::string graph_functions =
      GenerateGraphFunctions(program, func, &registrations, &graph_config_scalar_slots);
  stmt_codegen.SetGraphConfigScalarSlots(std::move(graph_config_scalar_slots));
  stmt_codegen.VisitStmt(func->body_);

  oss << GenerateIncludes(false, stmt_codegen.NeedsVectorInclude(), !graph_functions.empty());

  oss << "extern \"C\" {\n\n";

  oss << GenerateConfigFunction(expected_arg_count);

  // Each Graph function referenced by this entry is emitted once, ahead of the
  // entry that launches it, so the call site can take its address.
  oss << graph_functions;

  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "void aicpu_orchestration_entry(const ChipTaskArgs& orch_args) {\n";

  // Selective vs. full tensor dump is no longer requested from the orch body.
  // simpler#953 removed the ``enable_dump_tensor_selective()`` toggle: the
  // runtime now latches the dump level (off / partial / full) host-side at
  // ``dump_args_init`` from ``DumpDataHeader`` (driven by
  // ``CallConfig.enable_dump_args``), race-free regardless of submit order.
  // Codegen only emits the per-task ``Arg::dump(...)`` markers (see
  // ``EmitSelectiveDumpCall``); partial mode selecting exactly those marked
  // tensors is enabled by ``enable_dump_args == 1``.

  oss << "    // External tensors\n";
  oss << external_tensors.str();

  if (!scalar_params.empty()) {
    oss << "\n    // Scalar params\n";
    for (size_t i = 0; i < scalar_params.size(); ++i) {
      if (auto scalar_type = As<ScalarType>(scalar_params[i].type)) {
        oss << GenerateScalarUnpack(scalar_params[i].emit_name, static_cast<int>(i), scalar_type);
      } else {
        INTERNAL_CHECK(IsA<CommCtxType>(scalar_params[i].type))
            << "Unexpected non-scalar orchestration scalar param type: " << scalar_params[i].type->TypeName();
        oss << "    uint64_t " << scalar_params[i].emit_name << " = orch_args.scalar<uint64_t>(" << i
            << ");\n";
      }
    }
  }

  // A ``pl.dynamic("M")`` symbol names the runtime extent of whatever tensor
  // argument declares it. In a kernel it stays a type-level placeholder, but an
  // orchestration body may use it as a *value* — a loop bound, a
  // ``pl.create_tensor`` extent, or a folded ``pl.tensor.dim`` — so the emitted
  // C++ needs a definition for it: read it from the task-arg descriptor of the
  // first parameter declaring it, the signature being what guarantees every
  // argument carrying the symbol has that same extent.
  //
  // Gate on the generated text, not on an IR walk: a symbol also reaches the IR
  // through value *types* whose shapes are never printed (an external tensor's
  // extents), and defining those emits dead code.
  const std::string body_code = stmt_codegen.GetGeneratedCode();
  // Dedup by emitted *name*, not by Var: a symbol carries no emit-name mapping, so
  // it prints as its name hint, and neither the body nor this scan can distinguish
  // two Vars that share one. Seed with every name already spoken for at entry — a
  // scalar param, or a local the body defines — because a symbol sharing one of
  // those names is not what the body's occurrences refer to, and defining it would
  // shadow the real one inside the scope.
  std::unordered_set<std::string> defined_names;
  for (const auto& scalar : scalar_params) defined_names.insert(scalar.emit_name);
  CodegenEffectiveUseCollector body_vars;
  body_vars.VisitStmt(func->body_);
  for (const auto* def : body_vars.var_defs) defined_names.insert(GetSSABaseName(def->name_hint_));
  // Shared with each Graph helper, which binds its own tensor parameters and
  // needs the same definitions; see GenerateDynamicDimDefs.
  oss << GenerateDynamicDimDefs(func->params_, body_code, &stmt_codegen, std::move(defined_names));

  // The outermost SIMPLER_SCOPE() is now an explicit RuntimeScopeStmt emitted by
  // stmt_codegen (see MaterializeRuntimeScopes); just splice its output in.
  oss << "\n";
  oss << body_code;

  oss << "}\n\n";
  oss << "}  // extern \"C\"\n";

  return OrchestrationResult{oss.str(), std::move(registrations.entries), std::move(orchestration_signature)};
}

}  // namespace codegen
}  // namespace pypto
