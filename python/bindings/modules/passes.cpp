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
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

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
      .value("FlattenedSingleStmt", IRProperty::FlattenedSingleStmt, "Single-statement blocks flattened")
      .value("SplitIncoreOrch", IRProperty::SplitIncoreOrch, "InCore scopes outlined into separate functions")
      .value("HasMemRefs", IRProperty::HasMemRefs, "MemRef objects initialized on variables")
      .value("IncoreBlockOps", IRProperty::IncoreBlockOps,
             "InCore functions use block ops (tile types, load/store)")
      .value("AllocatedMemoryAddr", IRProperty::AllocatedMemoryAddr,
             "All MemRefs have valid addresses within buffer limits");

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
      .value("BASIC", VerificationLevel::Basic, "Verify lightweight properties once per pipeline (default)");

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

  // PassContext
  nb::class_<PassContext>(passes, "PassContext",
                          "Context that holds instruments and pass configuration.\n\n"
                          "When active, Pass.__call__ will run the context's instruments\n"
                          "before/after each pass execution. Also controls automatic\n"
                          "verification level for PassPipeline.")
      .def(nb::init<std::vector<PassInstrumentPtr>, VerificationLevel>(), nb::arg("instruments"),
           nb::arg("verification_level") = VerificationLevel::Basic,
           "Create a PassContext with instruments and optional verification level")
      .def("__enter__",
           [](PassContext& self) -> PassContext& {
             self.EnterContext();
             return self;
           })
      .def("__exit__", [](PassContext& self, const nb::args&) { self.ExitContext(); })
      .def("get_verification_level", &PassContext::GetVerificationLevel,
           "Get the verification level for this context")
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
             "Sets memory space to UB by default, or DDR for block.load/block.store operands.");

  passes.def("basic_memory_reuse", &pass::BasicMemoryReuse,
             "Create a basic memory reuse pass\n\n"
             "Uses dependency analysis to identify memory reuse opportunities.\n"
             "Variables with non-overlapping lifetimes in the same memory space can share MemRef objects.");

  passes.def("insert_sync", &pass::InsertSync,
             "Create an insert sync pass\n\n"
             "Analyzes data dependencies and inserts synchronization operations\n"
             "(sync_src, sync_dst, bar_v, bar_m) for correct execution across hardware pipes.\n"
             "Uses the globally configured backend to obtain pipe information.");

  passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
             "Create an allocate memory address pass\n\n"
             "Allocates real memory addresses for existing alloc operations.\n"
             "Updates MemRef addresses and alloc statement arguments in place.");

  // Bind SSAErrorType enum
  nb::enum_<ssa::ErrorType>(passes, "SSAErrorType", "SSA verification error types")
      .value("MULTIPLE_ASSIGNMENT", ssa::ErrorType::MULTIPLE_ASSIGNMENT, "Variable assigned more than once")
      .value("NAME_SHADOWING", ssa::ErrorType::NAME_SHADOWING, "Variable name shadows outer scope variable")
      .value("MISSING_YIELD", ssa::ErrorType::MISSING_YIELD, "ForStmt or IfStmt missing required YieldStmt");

  // Bind TypeCheckErrorType enum
  nb::enum_<typecheck::ErrorType>(passes, "TypeCheckErrorType", "Type checking error types")
      .value("TYPE_KIND_MISMATCH", typecheck::ErrorType::TYPE_KIND_MISMATCH, "Type kind mismatch")
      .value("DTYPE_MISMATCH", typecheck::ErrorType::DTYPE_MISMATCH, "Data type mismatch")
      .value("SHAPE_DIMENSION_MISMATCH", typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH,
             "Shape dimension count mismatch")
      .value("SHAPE_VALUE_MISMATCH", typecheck::ErrorType::SHAPE_VALUE_MISMATCH,
             "Shape dimension value mismatch")
      .value("SIZE_MISMATCH", typecheck::ErrorType::SIZE_MISMATCH, "Vector size mismatch in control flow");

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

  passes.def("convert_to_ssa", &pass::ConvertToSSA, "Create an SSA conversion pass");
  passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes,
             "Create a pass that outlines InCore scopes into separate functions");
  passes.def("convert_tensor_to_block_ops", &pass::ConvertTensorToBlockOps,
             "Create a pass that converts tensor ops to block ops in InCore functions");
  passes.def("flatten_call_expr", &pass::FlattenCallExpr,
             "Create a pass that flattens nested call expressions");
  passes.def("normalize_stmt_structure", &pass::NormalizeStmtStructure,
             "Create a pass that normalizes statement structure");
  passes.def("flatten_single_stmt", &pass::FlattenSingleStmt,
             "Create a pass that recursively flattens single-statement blocks");

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

  // Bind IRVerifier class
  nb::class_<IRVerifier>(passes, "IRVerifier",
                         "IR verification system that manages verification rules\n\n"
                         "IRVerifier collects verification rules and applies them to programs.\n"
                         "Rules can be enabled/disabled individually.")
      .def(nb::init<>(), "Create an empty verifier with no rules")
      .def_static("create_default", &IRVerifier::CreateDefault,
                  "Create a verifier with default built-in rules (SSAVerify, TypeCheck)")
      .def("enable_rule", &IRVerifier::EnableRule, nb::arg("name"), "Enable a previously disabled rule")
      .def("disable_rule", &IRVerifier::DisableRule, nb::arg("name"), "Disable a rule")
      .def("is_rule_enabled", &IRVerifier::IsRuleEnabled, nb::arg("name"), "Check if a rule is enabled")
      .def("verify", &IRVerifier::Verify, nb::arg("program"),
           "Verify a program and collect diagnostics (does not throw)")
      .def("verify_or_throw", &IRVerifier::VerifyOrThrow, nb::arg("program"),
           "Verify a program and throw VerificationError if errors are found")
      .def_static("generate_report", &IRVerifier::GenerateReport, nb::arg("diagnostics"),
                  "Generate a formatted report from diagnostics");

  // Bind RunVerifier factory function
  passes.def("run_verifier", &pass::RunVerifier, nb::arg("disabled_rules") = std::vector<std::string>{},
             "Create a verifier pass with configurable rules");
}

}  // namespace python
}  // namespace pypto
