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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <vector>

#include "../module.h"
#include "pypto/ir/builder.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindIRBuilder(nb::module_& m) {
  // Get or create ir submodule
  nb::module_ ir = m.attr("ir");

  // IRBuilder class
  nb::class_<IRBuilder>(ir, "IRBuilder",
                        "IR Builder for incremental IR construction with context management.\n\n"
                        "The IRBuilder provides a stateful API for building IR incrementally using\n"
                        "Begin/End patterns. It maintains a context stack to track nested scopes\n"
                        "and validates proper construction.")
      .def(nb::init<>(), "Create a new IR builder")

      // Function building
      .def("BeginFunction", &IRBuilder::BeginFunction, nb::arg("name"), nb::arg("span"),
           "Begin building a function.\n\n"
           "Creates a new function context. Must be closed with EndFunction().\n\n"
           "Args:\n"
           "    name: Function name\n"
           "    span: Source location for function definition\n\n"
           "Raises:\n"
           "    RuntimeError: If already inside a function (nested functions not allowed)")

      .def("FuncArg", &IRBuilder::FuncArg, nb::arg("name"), nb::arg("type"), nb::arg("span"),
           "Add a function parameter.\n\n"
           "Must be called within a function context.\n\n"
           "Args:\n"
           "    name: Parameter name\n"
           "    type: Parameter type\n"
           "    span: Source location for parameter\n\n"
           "Returns:\n"
           "    Var: Variable representing the parameter\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a function context")

      .def("ReturnType", &IRBuilder::ReturnType, nb::arg("type"),
           "Add a return type to the current function.\n\n"
           "Can be called multiple times for multiple return types.\n\n"
           "Args:\n"
           "    type: Return type\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a function context")

      .def("EndFunction", &IRBuilder::EndFunction, nb::arg("end_span"),
           "End building a function.\n\n"
           "Finalizes the function and returns it.\n\n"
           "Args:\n"
           "    end_span: Source location for end of function\n\n"
           "Returns:\n"
           "    Function: The built function\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a function context")

      // For loop building
      .def("BeginForLoop", &IRBuilder::BeginForLoop, nb::arg("loop_var"), nb::arg("start"), nb::arg("stop"),
           nb::arg("step"), nb::arg("span"),
           "Begin building a for loop.\n\n"
           "Creates a new for loop context. Must be closed with EndForLoop().\n\n"
           "Args:\n"
           "    loop_var: Loop variable\n"
           "    start: Start value expression\n"
           "    stop: Stop value expression\n"
           "    step: Step value expression\n"
           "    span: Source location for loop definition\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("AddIterArg", &IRBuilder::AddIterArg, nb::arg("iter_arg"),
           "Add an iteration argument to the current for loop.\n\n"
           "Iteration arguments are loop-carried values (SSA-style).\n\n"
           "Args:\n"
           "    iter_arg: Iteration argument with initial value\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a for loop context")

      .def("AddReturnVar", &IRBuilder::AddReturnVar, nb::arg("var"),
           "Add a return variable to the current for loop.\n\n"
           "Return variables capture the final values of iteration arguments.\n"
           "Must match the number of iteration arguments.\n\n"
           "Args:\n"
           "    var: Return variable\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a for loop context")

      .def("EndForLoop", &IRBuilder::EndForLoop, nb::arg("end_span"),
           "End building a for loop.\n\n"
           "Finalizes the loop and returns it.\n\n"
           "Args:\n"
           "    end_span: Source location for end of loop\n\n"
           "Returns:\n"
           "    ForStmt: The built for statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a for loop context\n"
           "    RuntimeError: If number of return variables doesn't match iteration arguments")

      // If statement building
      .def("BeginIf", &IRBuilder::BeginIf, nb::arg("condition"), nb::arg("span"),
           "Begin building an if statement.\n\n"
           "Creates a new if context. Must be closed with EndIf().\n\n"
           "Args:\n"
           "    condition: Condition expression\n"
           "    span: Source location for if statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("BeginElse", &IRBuilder::BeginElse, nb::arg("span"),
           "Begin the else branch of the current if statement.\n\n"
           "Must be called after building the then branch.\n\n"
           "Args:\n"
           "    span: Source location for else keyword\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside an if context\n"
           "    RuntimeError: If else branch already begun")

      .def("AddIfReturnVar", &IRBuilder::AddIfReturnVar, nb::arg("var"),
           "Add a return variable to the current if statement.\n\n"
           "Return variables are used for SSA phi nodes.\n\n"
           "Args:\n"
           "    var: Return variable\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside an if context")

      .def("EndIf", &IRBuilder::EndIf, nb::arg("end_span"),
           "End building an if statement.\n\n"
           "Finalizes the if statement and returns it.\n\n"
           "Args:\n"
           "    end_span: Source location for end of if\n\n"
           "Returns:\n"
           "    IfStmt: The built if statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside an if context")

      // Statement recording
      .def("Emit", &IRBuilder::Emit, nb::arg("stmt"),
           "Emit a statement in the current context.\n\n"
           "Adds a statement to the current context's statement list.\n\n"
           "Args:\n"
           "    stmt: Statement to emit\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("Assign", &IRBuilder::Assign, nb::arg("var"), nb::arg("value"), nb::arg("span"),
           "Create an assignment statement and emit it.\n\n"
           "Convenience method that creates and emits an assignment.\n\n"
           "Args:\n"
           "    var: Variable to assign to\n"
           "    value: Expression value\n"
           "    span: Source location for assignment\n\n"
           "Returns:\n"
           "    AssignStmt: The created assignment statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("Var", &IRBuilder::Var, nb::arg("name"), nb::arg("type"), nb::arg("span"),
           "Create a variable (does not emit).\n\n"
           "Helper to create a variable. User must create assignment separately.\n\n"
           "Args:\n"
           "    name: Variable name\n"
           "    type: Variable type\n"
           "    span: Source location\n\n"
           "Returns:\n"
           "    Var: The created variable")

      .def("Return", nb::overload_cast<const std::vector<ExprPtr>&, const Span&>(&IRBuilder::Return),
           nb::arg("values"), nb::arg("span"),
           "Create a return statement and emit it.\n\n"
           "Convenience method that creates and emits a return statement.\n\n"
           "Args:\n"
           "    values: List of expressions to return\n"
           "    span: Source location for return statement\n\n"
           "Returns:\n"
           "    ReturnStmt: The created return statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("Return", nb::overload_cast<const Span&>(&IRBuilder::Return), nb::arg("span"),
           "Create an empty return statement and emit it.\n\n"
           "Convenience method that creates and emits an empty return statement.\n\n"
           "Args:\n"
           "    span: Source location for return statement\n\n"
           "Returns:\n"
           "    ReturnStmt: The created return statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      // Context state queries
      .def("InFunction", &IRBuilder::InFunction,
           "Check if currently inside a function.\n\n"
           "Returns:\n"
           "    bool: True if inside a function context")

      .def("InLoop", &IRBuilder::InLoop,
           "Check if currently inside a for loop.\n\n"
           "Returns:\n"
           "    bool: True if inside a for loop context")

      .def("InIf", &IRBuilder::InIf,
           "Check if currently inside an if statement.\n\n"
           "Returns:\n"
           "    bool: True if inside an if statement context");
}

}  // namespace python
}  // namespace pypto
