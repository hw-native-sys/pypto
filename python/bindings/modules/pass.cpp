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

#include "pypto/ir/transform/base/pass.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "pypto/ir/transform/add_alloc_pass.h"
#include "pypto/ir/transform/basic_memory_reuse_pass.h"
#include "pypto/ir/transform/identity_pass.h"
#include "pypto/ir/transform/init_memref.h"
#include "pypto/ir/transform/insert_sync_pass.h"
#include "pypto/ir/transform/verify_ssa_pass.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindPass(nb::module_& m) {
  // Create a new 'passes' submodule (using 'passes' instead of 'pass' to avoid Python keyword)
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Pass base class for IR transformations
  nb::class_<Pass>(passes, "Pass", "Base class for IR transformation passes")
      .def("run", nb::overload_cast<const FunctionPtr&>(&Pass::Run), nb::arg("func"),
           "Execute the pass on a function")
      .def("run", nb::overload_cast<const ProgramPtr&>(&Pass::Run), nb::arg("program"),
           "Execute the pass on a program");

  // IdentityPass - a pass that appends a suffix to function name
  nb::class_<IdentityPass, Pass>(passes, "IdentityPass",
                                 "A pass that appends '_identity' suffix to function name for testing")
      .def(nb::init<>(), "Create an identity pass");

  // InitMemRefPass - a pass that initializes memref for variables
  nb::class_<InitMemRefPass, Pass>(passes, "InitMemRefPass", "A pass that initializes memref for variables")
      .def(nb::init<>(), "Create an InitMemRef pass");

  // BasicMemoryReusePass - basic memory reuse based on dependency analysis
  nb::class_<BasicMemoryReusePass, Pass>(passes, "BasicMemoryReusePass",
                                         "A pass for basic memory reuse based on dependency graph")
      .def(nb::init<>(), "Create a BasicMemoryReuse pass");

  // AddAllocPass - a pass that adds alloc operations for MemRef objects
  nb::class_<AddAllocPass, Pass>(
      passes, "AddAllocPass",
      "A pass that adds alloc operations for all MemRef objects in TileType variables")
      .def(nb::init<>(), "Create an AddAlloc pass");

  // InsertSyncPass - a pass that inserts sync operations
  nb::class_<InsertSyncPass, Pass>(passes, "InsertSyncPass",
                                   "A pass that inserts sync operations for pipeline synchronization")
      .def(nb::init<>(), "Create an InsertSync pass");

  // Bind SSAErrorType enum
  nb::enum_<SSAErrorType>(passes, "SSAErrorType", "SSA verification error types")
      .value("MULTIPLE_ASSIGNMENT", SSAErrorType::MULTIPLE_ASSIGNMENT, "Variable assigned more than once")
      .value("NAME_SHADOWING", SSAErrorType::NAME_SHADOWING, "Variable name shadows outer scope variable")
      .value("MISSING_YIELD", SSAErrorType::MISSING_YIELD, "ForStmt or IfStmt missing required YieldStmt")
      .value("CONTROL_FLOW_TYPE_MISMATCH", SSAErrorType::CONTROL_FLOW_TYPE_MISMATCH,
             "Type mismatch in control flow (ForStmt or IfStmt)");

  // Bind SSAError struct
  nb::class_<SSAError>(passes, "SSAError", "SSA verification error information")
      .def_ro("type", &SSAError::type, "Error type")
      .def_ro("message", &SSAError::message, "Error message")
      .def_ro("span", &SSAError::span, "Source location");

  // VerifySSAPass - a pass that verifies SSA form
  nb::class_<VerifySSAPass, Pass>(passes, "VerifySSAPass", "A pass that verifies SSA form of IR")
      .def(nb::init<>(), "Create a VerifySSA pass")
      .def("has_errors", &VerifySSAPass::HasErrors, "Check if any SSA violations were found")
      .def("get_errors", &VerifySSAPass::GetErrors, "Get list of SSA errors")
      .def("get_report", &VerifySSAPass::GetReport, "Get formatted verification report");
}

}  // namespace python
}  // namespace pypto
