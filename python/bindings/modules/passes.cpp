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
      .value("CallDirectionsResolved", IRProperty::CallDirectionsResolved,
             "Every non-builtin Call has explicit attrs['arg_directions'] (see Call::GetArgDirections)")
      .value("TileTypeCoherence", IRProperty::TileTypeCoherence,
             "Every TileType has canonical tile_view (implicit views stored as nullopt)")
      .value("InlineFunctionsEliminated", IRProperty::InlineFunctionsEliminated,
             "No FunctionType::Inline functions or Calls to them remain")
      .value("OrchestrationReferencesResolved", IRProperty::OrchestrationReferencesResolved,
             "Every non-builtin Call in an Orchestration function targets a Function in the Program");

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
      .def(nb::init<std::vector<PassInstrumentPtr>, VerificationLevel, DiagnosticPhase, DiagnosticCheckSet>(),
           nb::arg("instruments"), nb::arg("verification_level") = VerificationLevel::Basic,
           nb::arg("diagnostic_phase") = DiagnosticPhase::PrePipeline,
           nb::arg("disabled_diagnostics") = DiagnosticCheckSet{DiagnosticCheck::UnusedControlFlowResult},
           "Create a PassContext with instruments, verification level, diagnostic phase gate, "
           "and optional disabled diagnostic checks")
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
             "Create a pass that auto-tiles Mat-resident tile.matmul / tile.matmul_acc into a\n"
             "C-stationary K-loop\n\n"
             "Rewrites each tile.matmul or tile.matmul_acc whose Mat operands have static 2D\n"
             "shape into a range(0, K, k) loop. For tile.matmul the body branches on `ko == 0`\n"
             "between tile.matmul (fresh accumulator) and tile.matmul_acc (accumulating). For\n"
             "tile.matmul_acc the body is uniform — every iteration is tile.matmul_acc with the\n"
             "iter-arg init = caller's accumulator. The K-loop is marked ForKind::Pipeline +\n"
             "pipeline_stages=2 so LowerPipelineLoops produces a 2-deep ping-pong. Already-L0-\n"
             "sized matmuls are left untouched. tile.matmul_bias is not yet supported. Only K\n"
             "tiling; M/N tiling and K%k!=0 cases emit a PerfHint and skip.");
  passes.def("infer_tile_memory_space", &pass::InferTileMemorySpace,
             "Create a pass that infers memory_space for TileType variables in InCore functions");
  passes.def("resolve_transpose_layout", &pass::ResolveTransposeLayout,
             "Create a pass that resolves transpose layout for tile.load with transpose=True\n\n"
             "For each InCore function, detects tile.load(..., transpose=True) whose source\n"
             "is a function parameter and annotates that parameter's TensorType with the DN\n"
             "(column-major) layout. The shape is preserved -- DN is a codegen hint only.\n"
             "Orchestration and Opaque functions are returned unchanged.");
  passes.def("resolve_backend_op_layouts", &pass::ResolveBackendOpLayouts,
             "Create a pass that repairs backend-required layouts for constrained elementwise tile ops\n\n"
             "Repairs `[N,1]` col-major vector inputs at constrained use-sites by reshaping them\n"
             "into `[1,N]` row-major views before the consumer and reshaping the output back when needed.");
  passes.def("expand_mixed_kernel", &pass::ExpandMixedKernel,
             "Create a pass that expands mixed InCore functions into AIC + AIV + Group");
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
  passes.def("flatten_call_expr", &pass::FlattenCallExpr,
             "Create a pass that flattens nested call expressions");
  passes.def("inline_functions", &pass::InlineFunctions,
             "Create a pass that eliminates FunctionType::Inline functions by splicing\n"
             "their bodies at every call site. Runs as the first pipeline pass.\n"
             "Detects cycles in the Inline → Inline call graph and raises ValueError.\n"
             "Supports multi-return inline (emits MakeTuple at call site) and nested\n"
             "Inline-calls-Inline (iterates to fixpoint).");
  passes.def("normalize_stmt_structure", &pass::NormalizeStmtStructure,
             "Create a pass that normalizes statement structure");
  passes.def("derive_call_directions", &pass::DeriveCallDirections,
             "Derive Call attrs['arg_directions'] from callee param directions and buffer lineage.\n\n"
             "Writes per-argument runtime ArgDirection (Input / Output / InOut /\n"
             "OutputExisting / Scalar) onto every non-builtin Call. Locally allocated\n"
             "Out arguments are promoted to InOut to model WAW dependencies.\n\n"
             "Post-condition: ``IRProperty::CallDirectionsResolved``. The integrity of\n"
             "the produced ``Call.attrs['arg_directions']`` is verified automatically by the\n"
             "``CallDirectionsResolved`` PropertyVerifier (no separate verify pass).");
  passes.def("derive_manual_scope_deps", &pass::DeriveManualScopeDeps,
             "Resolve manual_dep_edges on every kernel Call inside a "
             "RuntimeScopeStmt(manual=true).\n\n"
             "Merges user-supplied edges (deps=[var1, var2] kwarg) with auto-derived "
             "data-flow edges (tensor args referencing prior-call Vars in the same "
             "manual scope) into Call.attrs['manual_dep_edges'].\n\n"
             "Args wrapped in pl.no_dep(...) (ArgDirection.NoDep) are excluded from "
             "auto-derived edges. The pass fails the build if any single submit "
             "would carry more than 16 explicit deps (runtime hard cap).");
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

  // L0 tile-size chooser submodule (closed-form heuristic; consumed by the
  // AutoTileMatmulL0 pass and exposed for testing / inspection).
  nb::module_ l0_tile = passes.def_submodule(
      "l0_tile_chooser",
      "Closed-form chooser for L0 matmul tile shape (m, n, k) under L1->L0 traffic minimisation");

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
      .def_rw("double_buffer_a", &utils::L0TileConfig::double_buffer_a)
      .def_rw("double_buffer_b", &utils::L0TileConfig::double_buffer_b)
      .def_rw("double_buffer_c", &utils::L0TileConfig::double_buffer_c)
      .def_rw("c_read", &utils::L0TileConfig::c_read)
      .def_rw("allow_padding", &utils::L0TileConfig::allow_padding);

  nb::class_<utils::L0TileResult>(l0_tile, "L0TileResult",
                                  "Output of ChooseL0Tile: the chosen (m, n, k) plus diagnostics")
      .def_ro("m", &utils::L0TileResult::m)
      .def_ro("n", &utils::L0TileResult::n)
      .def_ro("k", &utils::L0TileResult::k)
      .def_ro("estimated_traffic_bytes", &utils::L0TileResult::estimated_traffic_bytes)
      .def_ro("padded_compute_volume", &utils::L0TileResult::padded_compute_volume)
      .def_ro("perf_hint", &utils::L0TileResult::perf_hint);

  l0_tile.def("choose_l0_tile", &utils::ChooseL0Tile, nb::arg("config"),
              "Pick an approximately-optimal L0 tile shape (m, n, k) by closed-form heuristic.");
}

}  // namespace python
}  // namespace pypto
