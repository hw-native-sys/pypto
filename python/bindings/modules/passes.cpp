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

#include "pypto/ir/transforms/passes.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/reporter/report.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"
#include "pypto/ir/verifier/property_verifier_registry.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"
#include "pypto/ir/verifier/warning_verifier_registry.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindPass(nb::module_& m) {
  // Create a new 'passes' submodule (using 'passes' instead of 'pass' to avoid Python keyword)
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Bind IRProperty enum
  nb::enum_<IRProperty>(passes, "IRProperty", "Verifiable IR properties")
      .value("SSAForm", IRProperty::SSAForm, "IR is in SSA form")
      .value("TypeChecked", IRProperty::TypeChecked, "IR has passed type checking")
      .value("NoNestedCalls", IRProperty::NoNestedCalls, "No nested call expressions")
      .value("NormalizedStmtStructure", IRProperty::NormalizedStmtStructure, "Statement structure normalized")
      .value("NoRedundantBlocks", IRProperty::NoRedundantBlocks, "No single-child or nested SeqStmts")
      .value("SplitIncoreOrch", IRProperty::SplitIncoreOrch, "InCore scopes outlined into separate functions")
      .value("HasMemRefs", IRProperty::HasMemRefs, "MemRef objects initialized on variables")
      .value("IncoreTileOps", IRProperty::IncoreTileOps,
             "InCore functions use tile ops (tile types, load/store)")
      .value("AllocatedMemoryAddr", IRProperty::AllocatedMemoryAddr,
             "All MemRefs have valid addresses within buffer limits")
      .value("MixedKernelExpanded", IRProperty::MixedKernelExpanded,
             "Mixed InCore functions split into AIC+AIV")
      .value("ClusterOutlined", IRProperty::ClusterOutlined, "Cluster scopes outlined into Group functions")
      .value("HierarchyOutlined", IRProperty::HierarchyOutlined,
             "Hierarchy scopes outlined into level/role functions")
      .value("TileOps2D", IRProperty::TileOps2D, "All tile ops use ≤2D tiles")
      .value("TileMemoryInferred", IRProperty::TileMemoryInferred,
             "TileType memory_space populated in InCore functions")
      .value("BreakContinueValid", IRProperty::BreakContinueValid,
             "Break/continue only in sequential/while loops")
      .value("UseAfterDef", IRProperty::UseAfterDef, "All variable uses are dominated by a definition")
      .value("StructuredCtrlFlow", IRProperty::StructuredCtrlFlow,
             "No BreakStmt/ContinueStmt — only structured control flow")
      .value("VectorKernelSplit", IRProperty::VectorKernelSplit,
             "AIV functions with split mode have tpop shapes and store offsets adjusted")
      .value("OutParamNotShadowed", IRProperty::OutParamNotShadowed, "Out/InOut params are not reassigned")
      .value("NoNestedInCore", IRProperty::NoNestedInCore,
             "No nested InCore scopes (ScopeStmt inside ScopeStmt)");

  // Bind IRPropertySet
  nb::class_<IRPropertySet>(passes, "IRPropertySet", "A set of IR properties")
      .def(nb::init<>(), "Create an empty property set")
      .def("insert", &IRPropertySet::Insert, nb::arg("prop"), "Insert a property")
      .def("remove", &IRPropertySet::Remove, nb::arg("prop"), "Remove a property")
      .def("contains", &IRPropertySet::Contains, nb::arg("prop"), "Check if property is in set")
      .def("contains_all", &IRPropertySet::ContainsAll, nb::arg("other"),
           "Check if set contains all of other")
      .def("union_with", &IRPropertySet::Union, nb::arg("other"), "Return union of this and other")
      .def("intersection", &IRPropertySet::Intersection, nb::arg("other"), "Return intersection")
      .def("difference", &IRPropertySet::Difference, nb::arg("other"), "Return this minus other")
      .def("empty", &IRPropertySet::Empty, "Check if empty")
      .def("to_list", &IRPropertySet::ToVector, "Convert to list of properties")
      .def("__str__", &IRPropertySet::ToString)
      .def("__repr__", &IRPropertySet::ToString)
      .def("__eq__", &IRPropertySet::operator==)
      .def("__ne__", &IRPropertySet::operator!=);

  // Bind VerificationMode enum
  nb::enum_<VerificationMode>(passes, "VerificationMode", "Controls when property verification runs")
      .value("NONE", VerificationMode::None, "No automatic verification")
      .value("BEFORE", VerificationMode::Before, "Verify required properties before each pass")
      .value("AFTER", VerificationMode::After, "Verify produced properties after each pass")
      .value("BEFORE_AND_AFTER", VerificationMode::BeforeAndAfter, "Verify both before and after each pass");

  // Bind VerificationLevel enum
  nb::enum_<VerificationLevel>(passes, "VerificationLevel", "Controls automatic verification in PassPipeline")
      .value("NONE", VerificationLevel::None, "No automatic verification (fastest)")
      .value("BASIC", VerificationLevel::Basic, "Verify lightweight properties once per pipeline (default)")
      .value("ROUNDTRIP", VerificationLevel::Roundtrip,
             "BASIC + print→parse structural-equality check after every pass");

  // Bind WarningLevel enum
  nb::enum_<WarningLevel>(passes, "WarningLevel", "Controls automatic warning checks in PassPipeline")
      .value("NONE", WarningLevel::None, "All warnings disabled")
      .value("PRE_PIPELINE", WarningLevel::PrePipeline, "Run once before first pass (default)")
      .value("POST_PASS", WarningLevel::PostPass, "Run after every pass (pass debugging)")
      .value("BOTH", WarningLevel::Both, "Both pre-pipeline and post-pass");

  passes.def("get_default_warning_level", &GetDefaultWarningLevel,
             "Get the default warning level (from PYPTO_WARNING_LEVEL env var, default: PrePipeline)");

  // Bind WarningCheck enum
  nb::enum_<WarningCheck>(passes, "WarningCheck", "Identifies a specific warning check")
      .value("UnusedVariable", WarningCheck::UnusedVariable, "Variable defined but never read")
      .value("UnusedControlFlowResult", WarningCheck::UnusedControlFlowResult,
             "Unused return variable from for/while/if statement");

  // Bind WarningCheckSet
  nb::class_<WarningCheckSet>(passes, "WarningCheckSet", "A set of warning checks")
      .def(nb::init<>(), "Create an empty warning check set")
      .def("insert", &WarningCheckSet::Insert, nb::arg("check"), "Insert a warning check")
      .def("remove", &WarningCheckSet::Remove, nb::arg("check"), "Remove a warning check")
      .def("contains", &WarningCheckSet::Contains, nb::arg("check"), "Check if check is in set")
      .def("empty", &WarningCheckSet::Empty, "Check if empty")
      .def("difference", &WarningCheckSet::Difference, nb::arg("other"), "Return this minus other")
      .def("to_list", &WarningCheckSet::ToVector, "Convert to list of warning checks")
      .def("__str__", &WarningCheckSet::ToString)
      .def("__repr__", &WarningCheckSet::ToString)
      .def("__eq__", &WarningCheckSet::operator==)
      .def("__ne__", &WarningCheckSet::operator!=);

  // Bind WarningVerifierRegistry
  nb::class_<WarningVerifierRegistry>(passes, "WarningVerifierRegistry", "Registry of warning verifiers")
      .def_static(
          "run_checks",
          [](const WarningCheckSet& checks, const ProgramPtr& program) {
            return WarningVerifierRegistry::GetInstance().RunChecks(checks, program);
          },
          nb::arg("checks"), nb::arg("program"), "Run warning checks and collect diagnostics")
      .def_static("get_all_checks", &WarningVerifierRegistry::GetAllChecks,
                  "Get all registered warning checks");

  // Verification functions
  passes.def(
      "get_verified_properties", []() { return GetVerifiedProperties(); },
      "Get the set of properties automatically verified during compilation");
  passes.def("get_default_verification_level", &GetDefaultVerificationLevel,
             "Get the default verification level (from PYPTO_VERIFY_LEVEL env var, default: Basic)");
  passes.def("verify_properties", &pass::VerifyProperties, nb::arg("properties"), nb::arg("program"),
             nb::arg("pass_name"), "Verify properties on a program and throw on errors");

  // Pass class - expose call operators and property accessors
  nb::class_<Pass>(passes, "Pass", "Opaque pass object. Do not instantiate directly - use factory functions.")
      .def("__call__", &Pass::operator(), nb::arg("program"), "Execute pass on program")
      .def("get_name", &Pass::GetName, "Get the name of the pass")
      .def("get_required_properties", &Pass::GetRequiredProperties, "Get required properties")
      .def("get_produced_properties", &Pass::GetProducedProperties, "Get produced properties")
      .def("get_invalidated_properties", &Pass::GetInvalidatedProperties, "Get invalidated properties");

  // PassInstrument base class
  nb::class_<PassInstrument>(passes, "PassInstrument", "Abstract base class for pass instrumentation")
      .def("get_name", &PassInstrument::GetName, "Get the name of this instrument");

  // VerificationInstrument
  nb::class_<VerificationInstrument, PassInstrument>(
      passes, "VerificationInstrument", "Instrument that verifies IR properties before/after passes")
      .def(nb::init<VerificationMode>(), nb::arg("mode"),
           "Create a verification instrument with the given mode");

  // CallbackInstrument
  nb::class_<CallbackInstrument, PassInstrument>(passes, "CallbackInstrument",
                                                 "Instrument that invokes callbacks before/after each pass")
      .def(nb::init<CallbackInstrument::Callback, CallbackInstrument::Callback, std::string>(),
           nb::arg("before_pass") = nullptr, nb::arg("after_pass") = nullptr,
           nb::arg("name") = "CallbackInstrument",
           "Create a callback instrument with optional before/after callbacks");

  // ReportType enum
  nb::enum_<ReportType>(passes, "ReportType", "Type of report to generate")
      .value("Memory", ReportType::Memory, "Memory usage per MemorySpace");

  // ReportInstrument
  nb::class_<ReportInstrument, PassInstrument>(
      passes, "ReportInstrument", "Instrument that generates reports to files after specified passes")
      .def(nb::init<std::string>(), nb::arg("output_dir"), "Create a report instrument with output directory")
      .def("enable_report", &ReportInstrument::EnableReport, nb::arg("type"), nb::arg("trigger_pass"),
           "Enable a report type after a specific pass");

  // WarningInstrument
  nb::class_<WarningInstrument, PassInstrument>(passes, "WarningInstrument",
                                                "Instrument that runs warning checks before/after passes")
      .def(nb::init<WarningLevel, WarningCheckSet>(), nb::arg("phase") = WarningLevel::PrePipeline,
           nb::arg("checks") = WarningVerifierRegistry::GetAllChecks(),
           "Create a warning instrument with optional phase and check set");

  // PassContext
  nb::class_<PassContext>(passes, "PassContext",
                          "Context that holds instruments and pass configuration.\n\n"
                          "When active, Pass.__call__ will run the context's instruments\n"
                          "before/after each pass execution. Also controls automatic\n"
                          "verification and warning levels for PassPipeline.")
      .def(nb::init<std::vector<PassInstrumentPtr>, VerificationLevel, WarningLevel, WarningCheckSet>(),
           nb::arg("instruments"), nb::arg("verification_level") = VerificationLevel::Basic,
           nb::arg("warning_level") = WarningLevel::PrePipeline,
           nb::arg("disabled_warnings") = WarningCheckSet{WarningCheck::UnusedControlFlowResult},
           "Create a PassContext with instruments, verification level, warning level, "
           "and optional disabled warnings")
      .def("__enter__",
           [](PassContext& self) -> PassContext& {
             self.EnterContext();
             return self;
           })
      .def("__exit__", [](PassContext& self, const nb::args&) { self.ExitContext(); })
      .def("get_verification_level", &PassContext::GetVerificationLevel,
           "Get the verification level for this context")
      .def("get_warning_level", &PassContext::GetWarningLevel, "Get the warning level for this context")
      .def("get_disabled_warnings", &PassContext::GetDisabledWarnings, "Get the disabled warning checks")
      .def("get_instruments", &PassContext::GetInstruments, "Get the instruments registered on this context")
      .def_static("current", &PassContext::Current, nb::rv_policy::reference,
                  "Get the currently active context, or None if no context is active");

  // PassPipeline class
  nb::class_<PassPipeline>(passes, "PassPipeline", "A pipeline of passes executed in sequence")
      .def(nb::init<>(), "Create an empty pipeline")
      .def("add_pass", &PassPipeline::AddPass, nb::arg("pass_obj"), "Add a pass to the pipeline")
      .def("run", &PassPipeline::Run, nb::arg("program"), "Execute all passes in sequence")
      .def("get_pass_names", &PassPipeline::GetPassNames, "Get names of all passes");

  // Factory functions with snake_case names
  passes.def("init_mem_ref", &pass::InitMemRef,
             "Create an init memref pass\n\n"
             "Initializes MemRef for all variables in functions.\n"
             "Sets memory space to UB by default, or DDR for tile.load/tile.store operands.");

  passes.def("memory_reuse", &pass::MemoryReuse,
             "Create a memory reuse pass\n\n"
             "Uses lifetime analysis over the full IR to identify memory reuse opportunities.\n"
             "Variables with non-overlapping lifetimes in the same memory space can share MemRef objects.\n"
             "Handles nested control flow (for-loops, if/else branches) for accurate lifetime tracking.");

  passes.def("insert_sync", &pass::InsertSync,
             "Create an insert sync pass\n\n"
             "Analyzes data dependencies and inserts synchronization operations\n"
             "(sync_src, sync_dst, bar_v, bar_m) for correct execution across hardware pipes.\n"
             "Uses the globally configured backend to obtain pipe information.");

  passes.def("legalize_pto_buffer_reuse", &pass::LegalizePTOBufferReuse,
             "Create a PTO buffer reuse legalisation pass\n\n"
             "After generic MemoryReuse, detects illegal cross-type MemRef sharing\n"
             "that PTO codegen cannot express and splits such MemRefs.");

  passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
             "Create an allocate memory address pass\n\n"
             "Allocates real memory addresses for existing alloc operations.\n"
             "Updates MemRef addresses and alloc statement arguments in place.");

  passes.def("fuse_create_assemble_to_slice", &pass::FuseCreateAssembleToSlice,
             "Fuse tensor.create + tensor.assemble into tensor.slice in Orchestration functions\n\n"
             "When a tensor.create result is assembled into a target exactly once,\n"
             "replaces create with tensor.slice(target, shape, offsets) and removes\n"
             "the assemble, enabling orchestration codegen to emit .view() directly.");

  passes.def("normalize_return_order", &pass::NormalizeReturnOrder,
             "Create a return order normalization pass\n\n"
             "Reorders return tuple values in InCore functions so that return[i]\n"
             "corresponds to the i-th Out/InOut parameter in declaration order,\n"
             "and updates TupleGetItemExpr indices at call sites accordingly.");

  // Bind SSAErrorType enum
  nb::enum_<ssa::ErrorType>(passes, "SSAErrorType", "SSA verification error types")
      .value("MULTIPLE_ASSIGNMENT", ssa::ErrorType::MULTIPLE_ASSIGNMENT, "Variable assigned more than once")
      .value("NAME_SHADOWING", ssa::ErrorType::NAME_SHADOWING, "Variable name shadows outer scope variable")
      .value("MISSING_YIELD", ssa::ErrorType::MISSING_YIELD, "ForStmt or IfStmt missing required YieldStmt")
      .value("ITER_ARGS_RETURN_VARS_MISMATCH", ssa::ErrorType::ITER_ARGS_RETURN_VARS_MISMATCH,
             "iter_args count != return_vars count in ForStmt/WhileStmt")
      .value("YIELD_COUNT_MISMATCH", ssa::ErrorType::YIELD_COUNT_MISMATCH,
             "YieldStmt value count != iter_args/return_vars count")
      .value("SCOPE_VIOLATION", ssa::ErrorType::SCOPE_VIOLATION, "Variable used outside its defining scope");

  // Bind TypeCheckErrorType enum
  nb::enum_<typecheck::ErrorType>(passes, "TypeCheckErrorType", "Type checking error types")
      .value("TYPE_KIND_MISMATCH", typecheck::ErrorType::TYPE_KIND_MISMATCH, "Type kind mismatch")
      .value("DTYPE_MISMATCH", typecheck::ErrorType::DTYPE_MISMATCH, "Data type mismatch")
      .value("SHAPE_DIMENSION_MISMATCH", typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH,
             "Shape dimension count mismatch")
      .value("SHAPE_VALUE_MISMATCH", typecheck::ErrorType::SHAPE_VALUE_MISMATCH,
             "Shape dimension value mismatch")
      .value("SIZE_MISMATCH", typecheck::ErrorType::SIZE_MISMATCH, "Vector size mismatch in control flow")
      .value("IF_CONDITION_MUST_BE_SCALAR", typecheck::ErrorType::IF_CONDITION_MUST_BE_SCALAR,
             "IfStmt/WhileStmt condition must be ScalarType")
      .value("FOR_RANGE_MUST_BE_SCALAR", typecheck::ErrorType::FOR_RANGE_MUST_BE_SCALAR,
             "ForStmt range must be ScalarType")
      .value("CONDITION_MUST_BE_BOOL", typecheck::ErrorType::CONDITION_MUST_BE_BOOL,
             "IfStmt/WhileStmt condition dtype must be BOOL");

  // Bind NestedCallErrorType enum
  nb::enum_<nested_call::ErrorType>(passes, "NestedCallErrorType", "Nested call verification error types")
      .value("CALL_IN_CALL_ARGS", nested_call::ErrorType::CALL_IN_CALL_ARGS,
             "Call expression appears in call arguments")
      .value("CALL_IN_IF_CONDITION", nested_call::ErrorType::CALL_IN_IF_CONDITION,
             "Call expression appears in if condition")
      .value("CALL_IN_FOR_RANGE", nested_call::ErrorType::CALL_IN_FOR_RANGE,
             "Call expression appears in for range (start/stop/step)")
      .value("CALL_IN_BINARY_EXPR", nested_call::ErrorType::CALL_IN_BINARY_EXPR,
             "Call expression appears in binary expression operands")
      .value("CALL_IN_UNARY_EXPR", nested_call::ErrorType::CALL_IN_UNARY_EXPR,
             "Call expression appears in unary expression operand");

  // Bind UseAfterDefErrorType enum
  nb::enum_<use_after_def::ErrorType>(passes, "UseAfterDefErrorType",
                                      "Use-after-def verification error types")
      .value("USE_BEFORE_DEF", use_after_def::ErrorType::USE_BEFORE_DEF,
             "Variable used before any definition in scope");

  passes.def("split_chunked_loops", &pass::SplitChunkedLoops,
             "Create a pass that splits chunked loops into nested loops");
  passes.def("interchange_chunk_loops", &pass::InterchangeChunkLoops,
             "Create a pass that interchanges chunk loops and inserts InCore scopes");
  passes.def("unroll_loops", &pass::UnrollLoops, "Create a loop unrolling pass");
  passes.def("ctrl_flow_transform", &pass::CtrlFlowTransform,
             "Create a control flow structuring pass (eliminate break/continue)");
  passes.def("convert_to_ssa", &pass::ConvertToSSA, "Create an SSA conversion pass");
  passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes,
             "Create a pass that outlines InCore scopes into separate functions");
  passes.def("outline_cluster_scopes", &pass::OutlineClusterScopes,
             "Create a pass that outlines Cluster scopes into separate Group functions");
  passes.def("outline_hierarchy_scopes", &pass::OutlineHierarchyScopes,
             "Create a pass that outlines Hierarchy scopes into separate level/role functions");
  passes.def("convert_tensor_to_tile_ops", &pass::ConvertTensorToTileOps,
             "Create a pass that converts tensor ops to tile ops in InCore functions");
  passes.def("optimize_orch_tensors", &pass::OptimizeOrchTensors,
             "Create a pass that optimizes tensor buffer usage in orchestration and InCore functions\n\n"
             "Applies three patterns: iter-arg reuse (merge Out->InOut), assemble parent\n"
             "strides (attach TensorView to Out params), and assemble-loop rewrite\n"
             "(convert tile.assemble loops to tile.store loops).");
  passes.def("flatten_tile_nd_to_2d", &pass::FlattenTileNdTo2D,
             "Create a pass that flattens ND tile ops to 2D in InCore functions\n\n"
             "Merges all dimensions except the last into a single dimension.\n"
             "E.g., tile [A, B, C] becomes [A*B, C]. Only converts 3D+ tiles.");
  passes.def("infer_tile_memory_space", &pass::InferTileMemorySpace,
             "Create a pass that infers memory_space for TileType variables in InCore functions");
  passes.def("resolve_transpose_layout", &pass::ResolveTransposeLayout,
             "Create a pass that resolves transpose layout for tile.load with transpose=True\n\n"
             "Detects tile.load(..., transpose=True) in InCore functions and transforms\n"
             "the source tensor parameter type to the logical transposed shape with DN layout.\n"
             "Propagates the type change to corresponding Orchestration function parameters.");
  passes.def("resolve_backend_op_layouts", &pass::ResolveBackendOpLayouts,
             "Create a pass that repairs backend-required layouts for constrained elementwise tile ops\n\n"
             "Repairs `[N,1]` col-major vector inputs at constrained use-sites by reshaping them\n"
             "into `[1,N]` row-major views before the consumer and reshaping the output back when needed.");
  passes.def("expand_mixed_kernel", &pass::ExpandMixedKernel,
             "Create a pass that expands mixed InCore functions into AIC + AIV + Group");
  passes.def("split_vector_kernel", &pass::SplitVectorKernel,
             "Create a pass that splits vector kernels based on SplitMode "
             "(adjusts tpush/tpop split, halves tpop shapes, adjusts store offsets)");
  passes.def(
      "simplify", &pass::Simplify,
      "Create a pass that simplifies expressions and statements using algebraic rules and bound analysis");
  passes.def("flatten_call_expr", &pass::FlattenCallExpr,
             "Create a pass that flattens nested call expressions");
  passes.def("normalize_stmt_structure", &pass::NormalizeStmtStructure,
             "Create a pass that normalizes statement structure");
  // Bind DiagnosticSeverity enum
  nb::enum_<DiagnosticSeverity>(passes, "DiagnosticSeverity", "Severity level for diagnostics")
      .value("Error", DiagnosticSeverity::Error, "Error that must be fixed")
      .value("Warning", DiagnosticSeverity::Warning, "Warning that should be reviewed");

  // Bind Diagnostic structure
  nb::class_<Diagnostic>(passes, "Diagnostic", "Single diagnostic message from verification")
      .def_ro("severity", &Diagnostic::severity, "Severity level (Error or Warning)")
      .def_ro("rule_name", &Diagnostic::rule_name, "Name of the verification rule")
      .def_ro("error_code", &Diagnostic::error_code, "Specific error code")
      .def_ro("message", &Diagnostic::message, "Human-readable error message")
      .def_ro("span", &Diagnostic::span, "Source location of the issue");

  // Bind PropertyVerifierRegistry
  nb::class_<PropertyVerifierRegistry>(passes, "PropertyVerifierRegistry",
                                       "Registry of property verifiers for IR verification")
      .def_static(
          "verify",
          [](const IRPropertySet& props, const ProgramPtr& program) {
            return PropertyVerifierRegistry::GetInstance().VerifyProperties(props, program);
          },
          nb::arg("properties"), nb::arg("program"), "Verify properties and collect diagnostics")
      .def_static(
          "verify_or_throw",
          [](const IRPropertySet& props, const ProgramPtr& program) {
            PropertyVerifierRegistry::GetInstance().VerifyOrThrow(props, program);
          },
          nb::arg("properties"), nb::arg("program"), "Verify properties and throw on errors")
      .def_static("generate_report", &PropertyVerifierRegistry::GenerateReport, nb::arg("diagnostics"),
                  "Generate formatted report");

  passes.def("get_default_verify_properties", &GetDefaultVerifyProperties,
             "Get default property set for explicit verification");
  passes.def("get_structural_properties", &GetStructuralProperties, "Get structural invariant properties");

  // Bind RunVerifier factory function
  passes.def(
      "run_verifier",
      [](const IRPropertySet* properties) {
        return pass::RunVerifier(properties ? *properties : GetDefaultVerifyProperties());
      },
      nb::arg("properties").none() = nb::none(),
      "Create a verifier pass. Defaults to get_default_verify_properties() if None.");

  // PassProperties struct for Python-defined passes
  nb::class_<PassProperties>(passes, "PassProperties", "Property declarations for a pass")
      .def(nb::init<>(), "Create empty pass properties")
      .def(nb::init<IRPropertySet, IRPropertySet, IRPropertySet>(), nb::arg("required"), nb::arg("produced"),
           nb::arg("invalidated"), "Create pass properties with required/produced/invalidated sets")
      .def_rw("required", &PassProperties::required, "Required properties")
      .def_rw("produced", &PassProperties::produced, "Produced properties")
      .def_rw("invalidated", &PassProperties::invalidated, "Invalidated properties");

  // Pass factory functions for Python-defined transforms
  passes.def("create_function_pass", &pass::CreateFunctionPass, nb::arg("transform"), nb::arg("name") = "",
             nb::arg("properties") = PassProperties{},
             "Create a pass from a Python function-level transform.\n\n"
             "The transform receives a Function and returns a (possibly new) Function.\n"
             "The pass applies this transform to each function in the program.");

  passes.def("create_program_pass", &pass::CreateProgramPass, nb::arg("transform"), nb::arg("name") = "",
             nb::arg("properties") = PassProperties{},
             "Create a pass from a Python program-level transform.\n\n"
             "The transform receives a Program and returns a (possibly new) Program.");

  // Statement dependency analysis submodule (RFC #1026 Phase 1 — issue #1027).
  nb::module_ dep_analysis = passes.def_submodule(
      "stmt_dependency_analysis", "Statement dependency analysis and InOut-use discipline check");

  nb::class_<stmt_dep::StmtDependencyGraph>(dep_analysis, "StmtDependencyGraph",
                                            "Dataflow dependency graph over a region's top-level statements")
      .def_ro("stmts", &stmt_dep::StmtDependencyGraph::stmts, "Top-level stmts of the region in order")
      .def(
          "get_predecessors",
          [](const stmt_dep::StmtDependencyGraph& self, const StmtPtr& stmt) {
            std::vector<StmtPtr> result;
            if (!stmt) return result;
            auto it = self.predecessors.find(stmt.get());
            if (it == self.predecessors.end()) return result;
            // Preserve region order for determinism.
            for (const auto& s : self.stmts) {
              if (it->second.count(s.get())) result.push_back(s);
            }
            return result;
          },
          nb::arg("stmt"), "Return the predecessor stmts of the given stmt in region order");

  dep_analysis.def("build_stmt_dependency_graph", &stmt_dep::BuildStmtDependencyGraph, nb::arg("region"),
                   nb::arg("program").none() = nb::none(),
                   "Build a dataflow dependency graph over a region's top-level stmts. "
                   "When `program` is provided, the InOut-use discipline is checked first "
                   "and any violation raises pypto.Error (VerificationError).");

  dep_analysis.def("check_inout_use_discipline", &stmt_dep::CheckInOutUseDiscipline, nb::arg("region"),
                   nb::arg("program"),
                   "Enforce the InOut-use discipline; raises pypto.Error (VerificationError) "
                   "on any violation so compilation halts rather than proceeding with unsound IR.");
}

}  // namespace python
}  // namespace pypto
