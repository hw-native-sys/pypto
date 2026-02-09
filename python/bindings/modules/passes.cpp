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
#include "pypto/ir/transforms/verification_error.h"
#include "pypto/ir/transforms/verifier.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindPass(nb::module_& m) {
  // Create a new 'passes' submodule (using 'passes' instead of 'pass' to avoid Python keyword)
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Pass class - opaque to Python, only expose call operators
  nb::class_<Pass>(passes, "Pass", "Opaque pass object. Do not instantiate directly - use factory functions.")
      .def("__call__", &Pass::operator(), nb::arg("program"), "Execute pass on program");

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

  passes.def("add_alloc", &pass::AddAlloc,
             "Create an add alloc pass\n\n"
             "This pass traverses all TileType variables in each Function and creates alloc operations\n"
             "for each unique MemRef. The alloc operations are added at the beginning of the function.\n\n"
             "The pass:\n"
             "1. Identifies all TileType variables in the function\n"
             "2. Collects all unique MemRef objects from these TileType variables\n"
             "3. Creates an alloc operation for each unique MemRef\n"
             "4. Prepends these alloc operations to the function body\n\n"
             "Each alloc operation has no input/output arguments but is bound to a MemRef pointer\n"
             "to track memory allocation for that specific buffer.");

  // Bind SSAErrorType enum
  nb::enum_<ssa::ErrorType>(passes, "SSAErrorType", "SSA verification error types")
      .value("MULTIPLE_ASSIGNMENT", ssa::ErrorType::MULTIPLE_ASSIGNMENT, "Variable assigned more than once")
      .value("NAME_SHADOWING", ssa::ErrorType::NAME_SHADOWING, "Variable name shadows outer scope variable")
      .value("MISSING_YIELD", ssa::ErrorType::MISSING_YIELD, "ForStmt or IfStmt missing required YieldStmt");

  passes.def(
      "verify_ssa", &pass::VerifySSA,
      "Create an SSA verification pass\n\n"
      "This pass verifies SSA form of IR by checking:\n"
      "1. Each variable is assigned only once (MULTIPLE_ASSIGNMENT)\n"
      "2. No variable name shadowing across scopes (NAME_SHADOWING)\n"
      "3. ForStmt with iter_args must have YieldStmt as last statement (MISSING_YIELD)\n"
      "4. IfStmt with return_vars must have YieldStmt in both then and else branches (MISSING_YIELD)\n\n"
      "The pass collects all errors and generates a verification report instead of\n"
      "throwing exceptions, allowing detection of all issues in a single run.");

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

  passes.def("type_check", &pass::TypeCheck,
             "Create a type checking pass\n\n"
             "This pass checks type consistency in control flow constructs:\n"
             "1. ForStmt: iter_args initValue, yield values, and return_vars must have matching types\n"
             "2. IfStmt: then and else yield values must have matching types\n"
             "3. Shape consistency for TensorType and TileType\n\n"
             "The pass collects all errors and generates a type checking report instead of\n"
             "throwing exceptions, allowing detection of all issues in a single run.");

  passes.def("convert_to_ssa", &pass::ConvertToSSA,
             "Create an SSA conversion pass\n\n"
             "This pass converts non-SSA IR to SSA form by:\n"
             "1. Renaming variables with version suffixes (x -> x_0, x_1, x_2)\n"
             "2. Adding phi nodes (return_vars + YieldStmt) for IfStmt control flow divergence\n"
             "3. Converting loop-modified variables to iter_args + return_vars pattern\n\n"
             "The pass handles:\n"
             "- Straight-line code: multiple assignments to the same variable\n"
             "- If statements: variables modified in one or both branches\n"
             "- For loops: variables modified inside the loop body\n"
             "- Mixed SSA/non-SSA: preserves existing SSA structure while converting non-SSA parts");

  passes.def("flatten_call_expr", &pass::FlattenCallExpr,
             "Create a pass that flattens nested call expressions into three-address code\n\n"
             "This pass ensures that call expressions do not appear in nested contexts:\n"
             "1. Call arguments cannot be calls\n"
             "2. If conditions cannot be calls\n"
             "3. For loop ranges (start/stop/step) cannot be calls\n"
             "4. Binary/unary expression operands cannot be calls\n\n"
             "Nested calls are extracted into temporary variables (named _t0, _t1, etc.)\n"
             "and inserted as AssignStmt before the statement containing the nested call.\n"
             "For if/for statements, extracted statements are inserted into the last OpStmts\n"
             "before the if/for, or a new OpStmts is created if needed.\n\n"
             "Example transformation:\n"
             "    c = foo(bar(a))  =>  _t0 = bar(a); c = foo(_t0)");

  passes.def("normalize_stmt_structure", &pass::NormalizeStmtStructure,
             "Create a pass that normalizes statement structure\n\n"
             "This pass ensures IR is in a normalized form:\n"
             "1. Function/IfStmt/ForStmt body must be SeqStmts\n"
             "2. Consecutive AssignStmt/EvalStmt in SeqStmts are wrapped in OpStmts\n\n"
             "Example transformations:\n"
             "    Function body = AssignStmt(x, 1)\n"
             "    => Function body = SeqStmts([OpStmts([AssignStmt(x, 1)])])\n\n"
             "    SeqStmts([AssignStmt(a, 1), AssignStmt(b, 2), IfStmt(...)])\n"
             "    => SeqStmts([OpStmts([AssignStmt(a, 1), AssignStmt(b, 2)]), IfStmt(...)])");

  passes.def("flatten_single_stmt", &pass::FlattenSingleStmt,
             "Create a pass that recursively flattens single-statement blocks\n\n"
             "This pass simplifies IR by removing unnecessary nesting:\n"
             "- SeqStmts with only one statement is replaced by that statement\n"
             "- OpStmts with only one statement is replaced by that statement\n"
             "- Process is applied recursively\n\n"
             "Example transformations:\n"
             "    SeqStmts([OpStmts([AssignStmt(x, 1)])])\n"
             "    => AssignStmt(x, 1)\n\n"
             "    SeqStmts([OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])])\n"
             "    => OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])\n\n"
             "Note: This pass does NOT enforce that Function/IfStmt/ForStmt body must be SeqStmts.\n"
             "It will flatten them if they contain only a single statement.");

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
             "Create a verifier pass with configurable rules\n\n"
             "This pass creates an IRVerifier with default rules and allows disabling\n"
             "specific rules. The verifier collects all diagnostics and logs them.\n\n"
             "Args:\n"
             "    disabled_rules: List of rule names to disable (e.g., ['TypeCheck'])\n\n"
             "Returns:\n"
             "    Pass that runs IR verification");
}

}  // namespace python
}  // namespace pypto
