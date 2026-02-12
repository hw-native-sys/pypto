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
      .def("begin_function", &IRBuilder::BeginFunction, nb::arg("name"), nb::arg("span"),
           nb::arg("type") = FunctionType::Opaque,
           "Begin building a function.\n\n"
           "Creates a new function context. Must be closed with end_function().\n\n"
           "Args:\n"
           "    name: Function name\n"
           "    span: Source location for function definition\n"
           "    type: Function type (default: Opaque)\n\n"
           "Raises:\n"
           "    RuntimeError: If already inside a function (nested functions not allowed)")

      .def("func_arg", &IRBuilder::FuncArg, nb::arg("name"), nb::arg("type"), nb::arg("span"),
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

      .def("return_type", &IRBuilder::ReturnType, nb::arg("type"),
           "Add a return type to the current function.\n\n"
           "Can be called multiple times for multiple return types.\n\n"
           "Args:\n"
           "    type: Return type\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a function context")

      .def("end_function", &IRBuilder::EndFunction, nb::arg("end_span"),
           "End building a function.\n\n"
           "Finalizes the function and returns it.\n\n"
           "Args:\n"
           "    end_span: Source location for end of function\n\n"
           "Returns:\n"
           "    Function: The built function\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a function context")

      // For loop building
      .def("begin_for_loop", &IRBuilder::BeginForLoop, nb::arg("loop_var"), nb::arg("start"), nb::arg("stop"),
           nb::arg("step"), nb::arg("span"), nb::arg("kind") = ForKind::Sequential,
           "Begin building a for loop.\n\n"
           "Creates a new for loop context. Must be closed with end_for_loop().\n\n"
           "Args:\n"
           "    loop_var: Loop variable\n"
           "    start: Start value expression\n"
           "    stop: Stop value expression\n"
           "    step: Step value expression\n"
           "    span: Source location for loop definition\n"
           "    kind: Loop kind (Sequential or Parallel, default: Sequential)\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("add_iter_arg", &IRBuilder::AddIterArg, nb::arg("iter_arg"),
           "Add an iteration argument to the current for loop.\n\n"
           "Iteration arguments are loop-carried values (SSA-style).\n\n"
           "Args:\n"
           "    iter_arg: Iteration argument with initial value\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a for loop context")

      .def("add_return_var", &IRBuilder::AddReturnVar, nb::arg("var"),
           "Add a return variable to the current for loop.\n\n"
           "Return variables capture the final values of iteration arguments.\n"
           "Must match the number of iteration arguments.\n\n"
           "Args:\n"
           "    var: Return variable\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a for loop context")

      .def("end_for_loop", &IRBuilder::EndForLoop, nb::arg("end_span"),
           "End building a for loop.\n\n"
           "Finalizes the loop and returns it.\n\n"
           "Args:\n"
           "    end_span: Source location for end of loop\n\n"
           "Returns:\n"
           "    ForStmt: The built for statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a for loop context\n"
           "    RuntimeError: If number of return variables doesn't match iteration arguments")

      // While loop building
      .def("begin_while_loop", &IRBuilder::BeginWhileLoop, nb::arg("condition"), nb::arg("span"),
           "Begin building a while loop.\n\n"
           "Creates a new while loop context. Must be closed with end_while_loop().\n\n"
           "Args:\n"
           "    condition: Condition expression\n"
           "    span: Source location for loop definition\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("add_while_iter_arg", &IRBuilder::AddWhileIterArg, nb::arg("iter_arg"),
           "Add an iteration argument to the current while loop.\n\n"
           "Iteration arguments are loop-carried values (SSA-style).\n\n"
           "Args:\n"
           "    iter_arg: Iteration argument with initial value\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a while loop context")

      .def("add_while_return_var", &IRBuilder::AddWhileReturnVar, nb::arg("var"),
           "Add a return variable to the current while loop.\n\n"
           "Return variables capture the final values of iteration arguments.\n\n"
           "Args:\n"
           "    var: Return variable\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a while loop context")

      .def("set_while_loop_condition", &IRBuilder::SetWhileLoopCondition, nb::arg("condition"),
           "Set the condition for the current while loop.\n\n"
           "Used to update the loop condition after setting up iter_args. This allows\n"
           "the condition to reference iter_arg variables that are defined in the loop.\n\n"
           "Args:\n"
           "    condition: New condition expression\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a while loop context")

      .def("end_while_loop", &IRBuilder::EndWhileLoop, nb::arg("end_span"),
           "End building a while loop.\n\n"
           "Finalizes the loop and returns it.\n\n"
           "Args:\n"
           "    end_span: Source location for end of loop\n\n"
           "Returns:\n"
           "    WhileStmt: The built while statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a while loop context\n"
           "    RuntimeError: If number of return variables doesn't match iteration arguments")

      // If statement building
      .def("begin_if", &IRBuilder::BeginIf, nb::arg("condition"), nb::arg("span"),
           "Begin building an if statement.\n\n"
           "Creates a new if context. Must be closed with end_if().\n\n"
           "Args:\n"
           "    condition: Condition expression\n"
           "    span: Source location for if statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("begin_else", &IRBuilder::BeginElse, nb::arg("span"),
           "Begin the else branch of the current if statement.\n\n"
           "Must be called after building the then branch.\n\n"
           "Args:\n"
           "    span: Source location for else keyword\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside an if context\n"
           "    RuntimeError: If else branch already begun")

      .def("add_if_return_var", &IRBuilder::AddIfReturnVar, nb::arg("var"),
           "Add a return variable to the current if statement.\n\n"
           "Return variables are used for SSA phi nodes.\n\n"
           "Args:\n"
           "    var: Return variable\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside an if context")

      .def("end_if", &IRBuilder::EndIf, nb::arg("end_span"),
           "End building an if statement.\n\n"
           "Finalizes the if statement and returns it.\n\n"
           "Args:\n"
           "    end_span: Source location for end of if\n\n"
           "Returns:\n"
           "    IfStmt: The built if statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside an if context")

      // Statement recording
      .def("emit", &IRBuilder::Emit, nb::arg("stmt"),
           "Emit a statement in the current context.\n\n"
           "Adds a statement to the current context's statement list.\n\n"
           "Args:\n"
           "    stmt: Statement to emit\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      .def("assign", &IRBuilder::Assign, nb::arg("var"), nb::arg("value"), nb::arg("span"),
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

      .def("var", &IRBuilder::Var, nb::arg("name"), nb::arg("type"), nb::arg("span"),
           "Create a variable (does not emit).\n\n"
           "Helper to create a variable. User must create assignment separately.\n\n"
           "Args:\n"
           "    name: Variable name\n"
           "    type: Variable type\n"
           "    span: Source location\n\n"
           "Returns:\n"
           "    Var: The created variable")

      .def("return_", nb::overload_cast<const std::vector<ExprPtr>&, const Span&>(&IRBuilder::Return),
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

      .def("return_", nb::overload_cast<const Span&>(&IRBuilder::Return), nb::arg("span"),
           "Create an empty return statement and emit it.\n\n"
           "Convenience method that creates and emits an empty return statement.\n\n"
           "Args:\n"
           "    span: Source location for return statement\n\n"
           "Returns:\n"
           "    ReturnStmt: The created return statement\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a valid context")

      // Context state queries
      .def("in_function", &IRBuilder::InFunction,
           "Check if currently inside a function.\n\n"
           "Returns:\n"
           "    bool: True if inside a function context")

      .def("in_loop", &IRBuilder::InLoop,
           "Check if currently inside a for loop.\n\n"
           "Returns:\n"
           "    bool: True if inside a for loop context")

      .def("in_if", &IRBuilder::InIf,
           "Check if currently inside an if statement.\n\n"
           "Returns:\n"
           "    bool: True if inside an if statement context")

      .def("in_program", &IRBuilder::InProgram,
           "Check if currently inside a program.\n\n"
           "Returns:\n"
           "    bool: True if inside a program context")

      // Program building
      .def("begin_program", &IRBuilder::BeginProgram, nb::arg("name"), nb::arg("span"),
           "Begin building a program.\n\n"
           "Creates a new program context. Must be closed with end_program().\n\n"
           "Args:\n"
           "    name: Program name\n"
           "    span: Source location for program definition\n\n"
           "Raises:\n"
           "    RuntimeError: If already inside another program")

      .def("declare_function", &IRBuilder::DeclareFunction, nb::arg("func_name"),
           "Declare a function in the current program.\n\n"
           "Creates a GlobalVar for the function that can be used in Call expressions\n"
           "before the function is fully built. This enables cross-function calls.\n\n"
           "Args:\n"
           "    func_name: Function name to declare\n\n"
           "Returns:\n"
           "    GlobalVar: GlobalVar that can be used in Call expressions\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a program context")

      .def("get_global_var", &IRBuilder::GetGlobalVar, nb::arg("func_name"),
           "Get a GlobalVar for a declared function.\n\n"
           "Retrieves a GlobalVar that was previously declared with declare_function.\n\n"
           "Args:\n"
           "    func_name: Function name\n\n"
           "Returns:\n"
           "    GlobalVar: GlobalVar for the function\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a program context or function not declared")

      .def("add_function", &IRBuilder::AddFunction, nb::arg("func"),
           "Add a completed function to the current program.\n\n"
           "The function must have been previously declared with declare_function.\n\n"
           "Args:\n"
           "    func: Completed function to add\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a program context")

      .def("end_program", &IRBuilder::EndProgram, nb::arg("end_span"),
           "End building a program.\n\n"
           "Finalizes the program and returns it.\n\n"
           "Args:\n"
           "    end_span: Source location for end of program\n\n"
           "Returns:\n"
           "    Program: The built program\n\n"
           "Raises:\n"
           "    RuntimeError: If not inside a program context")

      .def("get_function_return_types", &IRBuilder::GetFunctionReturnTypes, nb::arg("gvar"),
           "Get return types for a function by its GlobalVar.\n\n"
           "Returns the return types for a function if it has been added to the program.\n"
           "Returns empty list if not inside a program or function not yet added.\n\n"
           "Args:\n"
           "    gvar: GlobalVar for the function\n\n"
           "Returns:\n"
           "    List[Type]: Vector of return types");
}

}  // namespace python
}  // namespace pypto
