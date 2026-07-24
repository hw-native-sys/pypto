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
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"
#include "pypto/ir/verifier/diagnostic_check_registry.h"
#include "pypto/ir/verifier/property_verifier_registry.h"
#include "pypto/ir/verifier/verification_error.h"

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
             "No nested InCore scopes (ScopeStmt inside ScopeStmt)")
      .value("InOutUseValid", IRProperty::InOutUseValid,
             "No reads of InOut/Out-passed variables after the call (RFC #1026)")
      .value("PipelineLoopValid", IRProperty::PipelineLoopValid,
             "Bidirectional invariant: ForStmt.kind_ == Pipeline ⇔ has pipeline_stages attr")
      .value("PipelineResolved", IRProperty::PipelineResolved,
             "No ForKind::Pipeline survives; produced by CanonicalizeIOOrder")
      .value("UnrollResolved", IRProperty::UnrollResolved,
             "No ForKind::Unroll survives; produced by UnrollLoops")
      .value("CallDirectionsResolved", IRProperty::CallDirectionsResolved,
             "Every non-builtin Call has explicit attrs['arg_directions'] (see Call::GetArgDirections)")
      .value("TileTypeCoherence", IRProperty::TileTypeCoherence,
             "Every TileType has canonical tile_view (implicit views stored as nullopt)")
      .value("InlineFunctionsEliminated", IRProperty::InlineFunctionsEliminated,
             "No FunctionType::Inline functions or Calls to them remain")
      .value("OrchestrationReferencesResolved", IRProperty::OrchestrationReferencesResolved,
             "Every non-builtin Call in an Orchestration function targets a Function in the Program")
      .value("TensorViewCanonical", IRProperty::TensorViewCanonical,
             "Every TensorType.tensor_view_ is canonical per RFC #1300 §2.2")
      .value("ArrayNotEscaped", IRProperty::ArrayNotEscaped,
             "ArrayType never appears as a function parameter or return type")
      .value("CommDomainScopesMaterialized", IRProperty::CommDomainScopesMaterialized,
             "Host_orch bodies are wrapped in CommDomainScopeStmts (one per inferred comm domain) and "
             "pld.tensor.window result types carry DistributedTensorType.window_buffer_ back-references")
      .value("RuntimeScopesMaterialized", IRProperty::RuntimeScopesMaterialized,
             "Orchestration functions carry explicit RuntimeScopeStmt nodes for the function body and "
             "for/if bodies; codegen no longer emits implicit PTO2_SCOPE() wrappers")
      .value("AssignTypeSymmetry", IRProperty::AssignTypeSymmetry,
             "Every AssignStmt has structural_equal(var->GetType(), value->GetType()) — covers dtype, "
             "shape, tile_view/tensor_view, and TileType memory_space (memref excluded as an allocation "
             "detail; memory_space exists only on TileType, not TensorType)")
      .value("ManualDepsOnSubmitOnly", IRProperty::ManualDepsOnSubmitOnly,
             "No plain cross-function Call (GlobalVar callee) carries attrs['manual_dep_edges'] — manual "
             "dependency edges live in the typed Submit::deps_ field. Op calls (system.task_dummy) are "
             "exempt")
      .value("ReturnParamsExplicit", IRProperty::ReturnParamsExplicit,
             "InCore/Group/Spmd tensor returns reference function params by pointer identity, so the "
             "return->param map is a lookup (#1702)")
      .value("AivSplitValid", IRProperty::AivSplitValid,
             "Split-mode AIV/AIC functions (explicit split_aiv marker + non-None split mode) have no "
             "vector reduce op that collapses the split axis (partial-reduction miscompile)")
      .value("IterArgCarryClassified", IRProperty::IterArgCarryClassified,
             "Every ForStmt with iter_args in an Orchestration function carries an "
             "attrs['iter_arg_rebind_<i>'] classification per slot (plus attrs['iter_arg_array_size_<i>'] "
             "for TaskId array carries), so orchestration codegen reads the carry lowering");

  // Bind IRPropertySet
  auto ir_property_set = nb::class_<IRPropertySet>(passes, "IRPropertySet", "A set of IR properties");
  ir_property_set.def(nb::init<>(), "Create an empty property set")
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
  // Unhashable: IRPropertySet is mutable via insert()/remove(); a hashable
  // mutable type would silently corrupt set/dict invariants if mutated after
  // insertion. Setting __hash__ = None is the Python convention.
  ir_property_set.attr("__hash__") = nb::none();

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

  // Bind MemoryPlanner enum
  nb::enum_<MemoryPlanner>(passes, "MemoryPlanner", "Selects who plans on-chip buffer memory")
      .value("PYPTO", MemoryPlanner::PyPTO,
             "PyPTO's AllocateMemoryAddr bakes physical addresses (ptoas --pto-level=level3)")
      .value("PTOAS", MemoryPlanner::PtoAS,
             "Skip pypto allocation passes; ptoas PlanMemory allocates (--pto-level=level2)");

  // Bind DiagnosticPhase enum
  nb::enum_<DiagnosticPhase>(passes, "DiagnosticPhase",
                             "Controls when DiagnosticInstrument runs registered checks "
                             "(warnings + performance hints)")
      .value("NONE", DiagnosticPhase::None, "Disable warnings and performance hints entirely")
      .value("PRE_PIPELINE", DiagnosticPhase::PrePipeline,
             "Run pre-pipeline checks once before first pass (default)")
      .value("POST_PASS", DiagnosticPhase::PostPass, "Run post-pass checks after every pass (debugging)")
      .value("POST_PIPELINE", DiagnosticPhase::PostPipeline,
             "Run post-pipeline checks once after the last pass (default for performance hints)");

  passes.def("get_default_diagnostic_phase", &GetDefaultDiagnosticPhase,
             "Get the default diagnostic phase (from PYPTO_WARNING_LEVEL env var, default: PrePipeline)");

  // Bind DiagnosticCheck enum
  nb::enum_<DiagnosticCheck>(passes, "DiagnosticCheck", "Identifies a specific diagnostic check")
      .value("UnusedVariable", DiagnosticCheck::UnusedVariable, "Variable defined but never read")
      .value("UnusedControlFlowResult", DiagnosticCheck::UnusedControlFlowResult,
             "Unused return variable from for/while/if statement")
      .value("TileInnermostDimGranularity", DiagnosticCheck::TileInnermostDimGranularity,
             "Tile innermost dim below recommended HW memory-access granularity (PH001)");

  // Bind DiagnosticCheckSet
  auto diagnostic_check_set =
      nb::class_<DiagnosticCheckSet>(passes, "DiagnosticCheckSet", "A set of diagnostic checks");
  diagnostic_check_set.def(nb::init<>(), "Create an empty diagnostic check set")
      .def("insert", &DiagnosticCheckSet::Insert, nb::arg("check"), "Insert a diagnostic check")
      .def("remove", &DiagnosticCheckSet::Remove, nb::arg("check"), "Remove a diagnostic check")
      .def("contains", &DiagnosticCheckSet::Contains, nb::arg("check"), "Check if check is in set")
      .def("empty", &DiagnosticCheckSet::Empty, "Check if empty")
      .def("difference", &DiagnosticCheckSet::Difference, nb::arg("other"), "Return this minus other")
      .def("union_with", &DiagnosticCheckSet::Union, nb::arg("other"), "Return union of this and other")
      .def("to_list", &DiagnosticCheckSet::ToVector, "Convert to list of diagnostic checks")
      .def("__str__", &DiagnosticCheckSet::ToString)
      .def("__repr__", &DiagnosticCheckSet::ToString)
      .def("__eq__", &DiagnosticCheckSet::operator==)
      .def("__ne__", &DiagnosticCheckSet::operator!=);
  // Unhashable for the same reason as IRPropertySet — mutable via insert/remove.
  diagnostic_check_set.attr("__hash__") = nb::none();

  // Bind DiagnosticCheckRegistry
  nb::class_<DiagnosticCheckRegistry>(passes, "DiagnosticCheckRegistry",
                                      "Registry of diagnostic checks (warnings + performance hints)")
      .def_static(
          "run_checks",
          [](const DiagnosticCheckSet& checks, DiagnosticPhase phase, const ProgramPtr& program) {
            return DiagnosticCheckRegistry::GetInstance().RunChecks(checks, phase, program);
          },
          nb::arg("checks"), nb::arg("phase"), nb::arg("program"),
          "Run diagnostic checks at the given phase and collect diagnostics")
      .def_static("get_all_checks", &DiagnosticCheckRegistry::GetAllChecks,
                  "Get all registered diagnostic checks")
      .def_static("get_warning_checks", &DiagnosticCheckRegistry::GetWarningChecks,
                  "Get all registered Warning-severity checks")
      .def_static("get_perf_hint_checks", &DiagnosticCheckRegistry::GetPerfHintChecks,
                  "Get all registered PerfHint-severity checks");

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
           "Enable a report type after a specific pass")
      .def("get_output_dir", &ReportInstrument::GetOutputDir,
           "Path of the directory that holds report files (used by perf hints to "
           "persist `perf_hints.log` alongside other reports)");

  // DiagnosticInstrument — unified warnings + performance hints (issue #1180)
  nb::class_<DiagnosticInstrument, PassInstrument>(
      passes, "DiagnosticInstrument",
      "Instrument that runs registered diagnostic checks (warnings + performance hints)")
      .def(nb::init<DiagnosticCheckSet>(), nb::arg("checks") = DiagnosticCheckRegistry::GetAllChecks(),
           "Create a diagnostic instrument running the given check set");

  // PassContext
  nb::class_<PassContext>(passes, "PassContext",
                          "Context that holds instruments and pass configuration.\n\n"
                          "When active, Pass.__call__ will run the context's instruments\n"
                          "before/after each pass execution. Also controls automatic\n"
                          "verification and the diagnostic channel (warnings + performance\n"
                          "hints) for PassPipeline.")
      .def(nb::init<std::vector<PassInstrumentPtr>, VerificationLevel, DiagnosticPhase, DiagnosticCheckSet,
                    MemoryPlanner, bool>(),
           nb::arg("instruments"), nb::arg("verification_level") = VerificationLevel::Basic,
           nb::arg("diagnostic_phase") = DiagnosticPhase::PrePipeline,
           nb::arg("disabled_diagnostics") = DiagnosticCheckSet{DiagnosticCheck::UnusedControlFlowResult},
           nb::arg("memory_planner") = MemoryPlanner::PyPTO,
           nb::arg("enable_pypto_l0c_double_buffer") = false,
           "Create a PassContext with instruments, verification level, diagnostic phase gate, "
           "optional disabled diagnostic checks, memory planner selection, and the experimental "
           "PyPTO-planner L0C double-buffer (dbC=2) opt-in")
      .def("__enter__",
           [](PassContext& self) -> PassContext& {
             self.EnterContext();
             return self;
           })
      .def("__exit__", [](PassContext& self, const nb::args&) { self.ExitContext(); })
      .def("get_verification_level", &PassContext::GetVerificationLevel,
           "Get the verification level for this context")
      .def("get_diagnostic_phase", &PassContext::GetDiagnosticPhase,
           "Get the diagnostic phase gate for this context")
      .def("get_disabled_diagnostics", &PassContext::GetDisabledDiagnostics,
           "Get the diagnostic checks suppressed by this context")
      .def("get_instruments", &PassContext::GetInstruments, "Get the instruments registered on this context")
      .def("get_memory_planner", &PassContext::GetMemoryPlanner,
           "Get the memory planner selection for this context")
      .def("get_enable_pypto_l0c_double_buffer", &PassContext::GetEnablePyptoL0cDoubleBuffer,
           "Whether L0C double-buffering (dbC=2) is enabled under the PyPTO memory planner")
      .def_static("current", &PassContext::Current, nb::rv_policy::reference,
                  "Get the currently active context, or None if no context is active");

  // PassPipeline class
  nb::class_<PassPipeline>(passes, "PassPipeline", "A pipeline of passes executed in sequence")
      .def(nb::init<>(), "Create an empty pipeline")
      .def("add_pass", &PassPipeline::AddPass, nb::arg("pass_obj"), "Add a pass to the pipeline")
      .def("run", &PassPipeline::Run, nb::arg("program"), "Execute all passes in sequence")
      .def("get_pass_names", &PassPipeline::GetPassNames, "Get names of all passes")
      .def("get_passes", &PassPipeline::GetPasses, "Get copies of all passes in execution order");

  // Factory functions with snake_case names
  passes.def("init_mem_ref", &pass::InitMemRef,
             "Create an init memref pass\n\n"
             "Initializes MemRef for all variables in functions.\n"
             "Sets memory space to UB by default, or DDR for tile.load/tile.store operands.");

  passes.def("materialize_semantic_aliases", &pass::MaterializeSemanticAliases,
             "Create the semantic must-alias materialization pass\n\n"
             "Propagates loop-carried iter_arg/initValue MemRefs down the yield/producer chain so\n"
             "accumulator producers write directly into the carried buffer. Split out of MemoryReuse\n"
             "so it can run without the opportunistic lifetime-reuse phase (memory_planner=PTOAS).");

  passes.def("memory_reuse", &pass::MemoryReuse,
             "Create a memory reuse pass\n\n"
             "Uses lifetime analysis over the full IR to identify memory reuse opportunities.\n"
             "Variables with non-overlapping lifetimes in the same memory space can share MemRef objects.\n"
             "Handles nested control flow (for-loops, if/else branches) for accurate lifetime tracking.");

  passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
             "Create an allocate memory address pass\n\n"
             "Allocates real memory addresses for existing alloc operations.\n"
             "Updates MemRef addresses and alloc statement arguments in place.");

  passes.def("fuse_create_assemble_to_slice", &pass::FuseCreateAssembleToSlice,
             "Fuse tensor.create + tensor.assemble into tensor.slice in Orchestration functions\n\n"
             "When a tensor.create result is assembled into a target exactly once,\n"
             "replaces create with tensor.slice(target, shape, offsets) and removes\n"
             "the assemble, enabling orchestration codegen to emit .view() directly.");

  passes.def("fold_no_op_reshape", &pass::FoldNoOpReshape,
             "Fold no-op tile.reshape assignments into Var-to-Var assignments\n\n"
             "Rewrites `lhs = tile.reshape(rhs, shape)` into `lhs = rhs` when both\n"
             "sides share the same MemRef root and produce identical TileBufSignatures,\n"
             "removing the reshape Call so PTO codegen can stay 1:1.");

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
             "IfStmt/WhileStmt condition dtype must be BOOL")
      .value("TENSOR_PADDING_MISMATCH", typecheck::ErrorType::TENSOR_PADDING_MISMATCH,
             "Tensor pad metadata mismatch")
      .value("DISTRIBUTED_WINDOW_IDENTITY_MISMATCH",
             typecheck::ErrorType::DISTRIBUTED_WINDOW_IDENTITY_MISMATCH,
             "Distributed tensors refer to different window buffers")
      .value("TILE_VIEW_MISMATCH", typecheck::ErrorType::TILE_VIEW_MISMATCH,
             "Effective TileView metadata mismatch");

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

  passes.def("unroll_loops", &pass::UnrollLoops, "Create a loop unrolling pass");
  passes.def("skew_cross_core_pipeline", &pass::SkewCrossCorePipeline,
             "Skew cross-core (cube/vector) ``pl.pipeline`` loops; runs immediately before\n"
             "lower_pipeline_loops. A single-round-trip producer-role loop runs the producer\n"
             "one iteration ahead (prologue + Sequential steady loop + epilogue); a consumer-role\n"
             "or multi-round-trip loop demotes to a plain Sequential loop (order-preserving).\n"
             "Output is Sequential with no pipeline marker, so lower_pipeline_loops and\n"
             "canonicalize_io_order leave it alone. Non-cross-core loops are untouched.");
  passes.def("lower_pipeline_loops", &pass::LowerPipelineLoops,
             "Lower ``pl.pipeline(N, stage=F)`` loops at the tile level (triggers on F > 1):\n"
             "replicate the body F times per outer iteration with a bare-SeqStmts remainder\n"
             "(or a cascaded IfStmt dispatch for dynamic bounds) covering N % F when needed.\n"
             "The produced outer loop keeps ``ForKind::Pipeline`` and downgrades\n"
             "``pipeline_stages`` to ``1`` as the post-lowering marker for CanonicalizeIOOrder.\n"
             "Keeping the (kind, attr) pair together preserves PipelineLoopValid and round-trip;\n"
             "re-running the pass sees ``factor == 1`` and skips (idempotent).");
  passes.def("canonicalize_io_order", &pass::CanonicalizeIOOrder,
             "Canonicalize statement order inside SeqStmts that live within a\n"
             "``ForKind::Pipeline`` body using a 4-tier schedule: lift scalar-producing\n"
             "assigns (e.g. address arithmetic) as high as possible, then cluster\n"
             "tile.load near the top, then remaining tile compute, and finally sink\n"
             "tile.store to the bottom — all subject to the SSA dependency graph.\n"
             "Enables symmetric ping-pong buffering by making replicated clones' input\n"
             "and output tiles co-live. On exit the pass demotes the outer pipeline\n"
             "loop's kind from ``ForKind::Pipeline`` to ``ForKind::Sequential`` (and\n"
             "strips any stale ``pipeline_stages`` attr). Non-pipeline loops are left\n"
             "untouched.");
  passes.def("ctrl_flow_transform", &pass::CtrlFlowTransform,
             "Create a control flow structuring pass (eliminate break/continue)");
  passes.def("convert_to_ssa", &pass::ConvertToSSA, "Create an SSA conversion pass");
  passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes,
             "Create a pass that outlines InCore scopes into separate functions");
  passes.def("outline_cluster_scopes", &pass::OutlineClusterScopes,
             "Create a pass that outlines Cluster scopes into Group functions "
             "and standalone Spmd scopes into Spmd functions");
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
  passes.def("auto_tile_matmul_l0", &pass::AutoTileMatmulL0,
             "Create a pass that auto-tiles static 2D tile.matmul / tile.matmul_acc for L0\n\n"
             "The active backend's roofline chooser selects (m,n,k,stationarity,dbC). K-split\n"
             "reductions use a 2-stage pipelined loop and peel a supported non-divisor aligned\n"
             "tail. Plain tile.matmul may also use an M/N grid with direct-GM placement or an\n"
             "on-chip Mat scratch for chained matmul consumers; compatible f32->bf16/f16 rint\n"
             "casts fold into the FIXPIPE writeback. Full-K grids support output-, A-, and\n"
             "B-stationary schedules. dbC=2 is enabled under PTOAS and available as a PyPTO\n"
             "planner opt-in. Already-L0-sized and unsupported regimes are left untouched;\n"
             "useful deferred cases emit PerfHint diagnostics. tile.matmul_bias is deferred.");
  passes.def("canonicalize_tile_slice", &pass::CanonicalizeTileSlice,
             "Create a pass that lowers Mat-resident tile.slice into tile.extract\n\n"
             "A tile.slice whose result tile is Mem.Mat (e.g. a batch-page slice emitted by\n"
             "FlattenTileNdTo2D when unrolling tile.batch_matmul) has no standalone hardware\n"
             "lowering. This pass folds each such slice's offset into its consumer: a\n"
             "tile.extract absorbs the offset and reads the slice's source directly, and a\n"
             "tile.matmul operand is replaced by a tile.extract(target_memory=Left|Right).\n"
             "The dead tile.slice is then dropped, unifying Mat->Left/Right on pto.textract.");
  passes.def("infer_tile_memory_space", &pass::InferTileMemorySpace,
             "Create a pass that infers memory_space for TileType variables in InCore functions");
  passes.def("materialize_tensor_strides", &pass::MaterializeTensorStrides,
             "Create the MaterializeTensorStrides pass (RFC #1300 §2.4).\n\n"
             "Walks every TensorType reachable from the program and rewrites any\n"
             "(view.has_value() && view.stride.empty()) slot to its packed canonical\n"
             "stride per BuildLogicalStridesFromLayout. Bare TensorTypes are left\n"
             "untouched. Idempotent. Produces TensorViewCanonical so the registry\n"
             "auto-verifies after the pass runs.");
  passes.def("resolve_backend_op_layouts", &pass::ResolveBackendOpLayouts,
             "Create a pass that repairs backend-required layouts for constrained elementwise tile ops\n\n"
             "Repairs `[N,1]` col-major vector inputs at constrained use-sites by reshaping them\n"
             "into `[1,N]` row-major views before the consumer and reshaping the output back when needed.");
  passes.def("expand_mixed_kernel", &pass::ExpandMixedKernel,
             "Create a pass that expands mixed InCore functions into AIC + AIV + Group");
  passes.def("lower_auto_vector_split", &pass::LowerAutoVectorSplit,
             "Create a pass that lowers AUTO pl.split mixed InCore functions into the explicit\n"
             "split_aiv form (tile.aiv_shard at C->V, tile.aic_gather at V->C, halved vector\n"
             "sub-region, get_subblock_idx) BEFORE ExpandMixedKernel. This is the live auto-split\n"
             "lowering path; SplitVectorKernel then only stamps attrs for the split_aiv functions.");
  passes.def("inject_gm_pipe_buffer", &pass::InjectGMPipeBuffer,
             "Create a backend-gated pass that injects the __gm_pipe_buffer workspace parameter\n"
             "into functions containing cross-core initialize_pipe ops, propagating the parameter\n"
             "through callers (Orchestration functions materialize the buffer locally instead).\n"
             "No-op on backends that don't require GM-backed pipe slots.");
  passes.def("split_vector_kernel", &pass::SplitVectorKernel,
             "Create a pass that splits vector kernels based on SplitMode "
             "(adjusts tpush/tpop split, halves tpop shapes, adjusts store offsets)");
  passes.def(
      "simplify", &pass::Simplify,
      "Create a pass that simplifies expressions and statements using algebraic rules and bound analysis");
  passes.def("lower_composite_ops", &pass::LowerCompositeOps,
             "Decompose composite tile/distributed ops into primitives via the "
             "composite-lowering registry. Today lowers tile.sin/tile.cos and "
             "explicit-signal InCore pld.tensor.allreduce; host allreduce is "
             "skipped for LowerHostTensorCollectives. FP32-only for trig. Idempotent.");
  passes.def("flatten_call_expr", &pass::FlattenCallExpr,
             "Create a pass that flattens nested call expressions");
  passes.def("inline_functions", &pass::InlineFunctions,
             "Create a pass that eliminates FunctionType::Inline functions by splicing\n"
             "their bodies at every call site. Runs as the first pipeline pass.\n"
             "Detects cycles in the Inline → Inline call graph and raises ValueError.\n"
             "Supports multi-return inline (emits MakeTuple at call site) and nested\n"
             "Inline-calls-Inline (iterates to fixpoint).");
  passes.def("synthesize_allreduce_signals", &pass::SynthesizeAllReduceSignals,
             "Synthesize private signal windows for host-level pld.tensor.allreduce calls that omit "
             "the signal argument. Existing explicit-signal calls are preserved.");
  passes.def("materialize_comm_domain_scopes", &pass::MaterializeCommDomainScopes,
             "Trace pld.tensor.alloc_window_buffer → pld.tensor.window → dispatch(device=r) "
             "chains in each\n"
             "host_orch function, materialise WindowBuffer instances back-referenced from\n"
             "DistributedTensorType.window_buffer_ on view Vars, and wrap the host_orch\n"
             "body in nested CommDomainScopeStmts (one per inferred comm domain). Runs\n"
             "late in the default pipeline after phase-fence expansion and before\n"
             "LowerHostTensorCollectives, while the host dispatch chain is still intact.");
  passes.def("lower_host_tensor_collectives", &pass::LowerHostTensorCollectives,
             "Lower host-level pld.tensor.allreduce calls to builtin tensor collective dispatches.");
  passes.def("materialize_dist_tensor_ctx", &pass::MaterializeDistTensorCtx,
             "Materialize CommCtx parameters and arguments for DistributedTensor function parameters.");
  passes.def("stamp_tfree_split", &pass::StampTfreeSplit,
             "Copy each cross-core tpop's split/pipe-id onto its matching tfree op so codegen\n"
             "reads them from the op directly. Covers mixed-kernel and explicit AIC/AIV tfrees.");
  passes.def("materialize_runtime_scopes", &pass::MaterializeRuntimeScopes,
             "Materialize implicit orchestration scopes as explicit RuntimeScopeStmt nodes.\n\n"
             "For every Orchestration function, inserts AUTO RuntimeScopeStmt (manual_=false)\n"
             "wrapping the function body and each ForStmt / IfStmt branch body (suppressed\n"
             "inside a manual scope). Codegen then emits PTO2_SCOPE only from RuntimeScopeStmt\n"
             "nodes, 1:1 with the IR. Runs last in the pipeline, after the final Simplify.");
  passes.def("classify_iter_arg_carry", &pass::ClassifyIterArgCarry,
             "Classify ForStmt iter_arg carries and size TaskId array carries.\n\n"
             "For every Orchestration function, stamps attrs['iter_arg_rebind_<i>'] (bool, every\n"
             "slot) and attrs['iter_arg_array_size_<i>'] (int, positive extents only) onto each\n"
             "ForStmt, so orchestration codegen reads the carry lowering instead of re-deriving\n"
             "it from an alias fixpoint. Runs last, after materialize_runtime_scopes.");
  passes.def("normalize_stmt_structure", &pass::NormalizeStmtStructure,
             "Create a pass that normalizes statement structure");
  passes.def("derive_call_directions", &pass::DeriveCallDirections,
             "Derive Call attrs['arg_directions'].\n\n"
             "Writes per-argument runtime ArgDirection (Input / Output / InOut /\n"
             "OutputExisting / Scalar) onto every non-builtin Call. Locally allocated\n"
             "Out arguments are promoted to InOut to model WAW dependencies.\n\n"
             "Manual-scope dependency edges (Call.attrs['manual_dep_edges']) are\n"
             "written directly by the parser from a pl.submit(...) deps= kwarg — this\n"
             "pass does not synthesise or lower them.\n\n"
             "Post-condition: ``IRProperty::CallDirectionsResolved``. The integrity of\n"
             "the produced ``Call.attrs['arg_directions']`` is verified automatically by the\n"
             "``CallDirectionsResolved`` PropertyVerifier (no separate verify pass).");
  passes.def(
      "auto_derive_task_dependencies", &pass::AutoDeriveTaskDependencies,
      nb::arg("analyze_auto_scopes") = false,
      "Derive compiler-owned runtime-scope task dependency edges.\n\n"
      "Runs after derive_call_directions and writes "
      "Call.attrs['compiler_manual_dep_edges'] inside analyzed AUTO runtime scopes. "
      "User-written manual scopes are skipped: they do not get compiler deps or "
      "automatic NoDep/OutputExisting direction rewrites. "
      "Pass analyze_auto_scopes=True to analyze AUTO scopes without changing their runtime scope mode. "
      "Unanalyzable hazards keep AUTO tracking with partial compiler deps stripped. "
      "User-provided Call.attrs['manual_dep_edges'] remain separate; orchestration "
      "codegen merges both attrs before emitting Arg::set_dependencies.");
  passes.def("expand_manual_phase_fence", &pass::ExpandManualPhaseFence,
             "Insert dependency-only dummy TaskId barriers for profitable manual_scope "
             "Array[TASK_ID] phase-fence fanout and rewrite covered consumers to depend "
             "on the barrier TaskId.");
  // Bind DiagnosticSeverity enum
  nb::enum_<DiagnosticSeverity>(passes, "DiagnosticSeverity", "Severity level for diagnostics")
      .value("Error", DiagnosticSeverity::Error, "Error that must be fixed")
      .value("Warning", DiagnosticSeverity::Warning, "Warning that should be reviewed")
      .value("PerfHint", DiagnosticSeverity::PerfHint,
             "Advisory performance hint (best-effort, not 100% accurate)");

  // Bind Diagnostic structure
  nb::class_<Diagnostic>(passes, "Diagnostic", "Single diagnostic message from verification")
      .def_ro("severity", &Diagnostic::severity, "Severity level (Error, Warning, or PerfHint)")
      .def_ro("rule_name", &Diagnostic::rule_name, "Name of the verification rule")
      .def_ro("error_code", &Diagnostic::error_code, "Specific error code")
      .def_ro("hint_code", &Diagnostic::hint_code,
              "Stable hint code for PerfHint diagnostics (e.g. \"PH001\"); empty otherwise")
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

  // Direct entry point for the TensorViewCanonical verifier (RFC #1300 P2).
  // Lets tests / debugging tools toggle strict (post-MaterializeTensorStrides)
  // mode without depending on the registry's default weak-mode wiring.
  passes.def(
      "verify_tensor_view_canonical",
      [](const ProgramPtr& program, bool require_materialized) {
        std::vector<Diagnostic> diagnostics;
        auto verifier = CreateTensorViewCanonicalPropertyVerifier(require_materialized);
        verifier->Verify(program, diagnostics);
        return diagnostics;
      },
      nb::arg("program"), nb::arg("require_materialized") = false,
      "Run the TensorViewCanonical verifier directly. require_materialized=False (default) "
      "is the weak mode (empty stride accepted as implicitly packed canonical); "
      "require_materialized=True is the strict codegen-entry contract.");

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

  // L0 tile-size chooser submodule (roofline cost model; consumed by the
  // AutoTileMatmulL0 pass and exposed for testing / inspection).
  nb::module_ l0_tile =
      passes.def_submodule("l0_tile_chooser",
                           "Chooser for the L0 matmul design point (m, n, k, stationarity, dbA/dbB/dbC) "
                           "by roofline cost model: min wall over the legal aligned tile grid");

  nb::enum_<utils::Stationarity>(l0_tile, "Stationarity",
                                 "Which GEMM operand is pinned across the L0 tiling loops")
      .value("OutputStationary", utils::Stationarity::kOutputStationary)
      .value("AStationary", utils::Stationarity::kAStationary)
      .value("BStationary", utils::Stationarity::kBStationary);

  nb::class_<utils::L0TileConfig>(l0_tile, "L0TileConfig",
                                  "Inputs to ChooseL0Tile: problem dims + hardware + schedule knobs")
      .def(nb::init<>())
      .def_rw("M", &utils::L0TileConfig::M)
      .def_rw("N", &utils::L0TileConfig::N)
      .def_rw("K", &utils::L0TileConfig::K)
      .def_rw("l0a_bytes", &utils::L0TileConfig::l0a_bytes)
      .def_rw("l0b_bytes", &utils::L0TileConfig::l0b_bytes)
      .def_rw("l0c_bytes", &utils::L0TileConfig::l0c_bytes)
      .def_rw("bytes_a", &utils::L0TileConfig::bytes_a)
      .def_rw("bytes_b", &utils::L0TileConfig::bytes_b)
      .def_rw("bytes_c", &utils::L0TileConfig::bytes_c)
      .def_rw("min_m", &utils::L0TileConfig::min_m)
      .def_rw("min_n", &utils::L0TileConfig::min_n)
      .def_rw("min_k", &utils::L0TileConfig::min_k)
      .def_rw("align_m", &utils::L0TileConfig::align_m)
      .def_rw("align_n", &utils::L0TileConfig::align_n)
      .def_rw("align_k", &utils::L0TileConfig::align_k)
      .def_rw("allow_a_stationary", &utils::L0TileConfig::allow_a_stationary)
      .def_rw("allow_b_stationary", &utils::L0TileConfig::allow_b_stationary)
      .def_rw("allow_double_buffer_c", &utils::L0TileConfig::allow_double_buffer_c)
      .def_rw("c_read", &utils::L0TileConfig::c_read)
      .def_rw("bw_a", &utils::L0TileConfig::bw_a)
      .def_rw("bw_b", &utils::L0TileConfig::bw_b)
      .def_rw("bw_drain", &utils::L0TileConfig::bw_drain)
      .def_rw("drain_fixed_cycles", &utils::L0TileConfig::drain_fixed_cycles)
      .def_rw("drain_row_cycles", &utils::L0TileConfig::drain_row_cycles)
      .def_rw("drain_penalty_cycles", &utils::L0TileConfig::drain_penalty_cycles)
      .def_rw("drain_c0_bytes", &utils::L0TileConfig::drain_c0_bytes)
      .def_rw("mad_head", &utils::L0TileConfig::mad_head)
      .def_rw("mad_k_fractal_bytes", &utils::L0TileConfig::mad_k_fractal_bytes)
      .def_rw("mad_fp32_passes", &utils::L0TileConfig::mad_fp32_passes)
      .def_rw("allow_padding", &utils::L0TileConfig::allow_padding)
      .def_rw("allow_k_boundary", &utils::L0TileConfig::allow_k_boundary);

  nb::class_<utils::L0TileResult>(l0_tile, "L0TileResult",
                                  "Output of ChooseL0Tile: the chosen (m, n, k) plus diagnostics")
      .def_ro("m", &utils::L0TileResult::m)
      .def_ro("n", &utils::L0TileResult::n)
      .def_ro("k", &utils::L0TileResult::k)
      .def_ro("estimated_traffic_bytes", &utils::L0TileResult::estimated_traffic_bytes)
      .def_ro("estimated_cost_cycles", &utils::L0TileResult::estimated_cost_cycles)
      .def_ro("padded_compute_volume", &utils::L0TileResult::padded_compute_volume)
      .def_ro("stationarity", &utils::L0TileResult::stationarity)
      .def_ro("os_holds_a", &utils::L0TileResult::os_holds_a)
      .def_ro("double_buffer_c", &utils::L0TileResult::double_buffer_c)
      .def_ro("perf_hint", &utils::L0TileResult::perf_hint);

  l0_tile.def("choose_l0_tile", &utils::ChooseL0Tile, nb::arg("config"),
              "Pick the minimum-cost L0 tile shape (m, n, k) under the roofline cost model.");
}

}  // namespace python
}  // namespace pypto
