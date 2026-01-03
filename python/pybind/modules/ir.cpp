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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "../bindings.h"
#include "pypto/ir/core.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transform/printer.h"

namespace py = pybind11;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindIR(py::module_& m) {
  py::module_ ir = m.def_submodule("ir", "PyPTO IR (Intermediate Representation) module");

  // Span - value type, copy semantics
  py::class_<Span>(ir, "Span", "Source location information tracking file, line, and column positions")
      .def(py::init<std::string, int, int, int, int>(), py::arg("filename"), py::arg("begin_line"),
           py::arg("begin_column"), py::arg("end_line") = -1, py::arg("end_column") = -1,
           "Create a source span")
      .def("to_string", &Span::to_string, "Convert span to string representation")
      .def("is_valid", &Span::is_valid, "Check if the span has valid coordinates")
      .def_static("unknown", &Span::unknown,
                  "Create an unknown/invalid span for cases where source location is unavailable")
      .def("__repr__", &Span::to_string)
      .def("__str__", &Span::to_string)
      .def_readonly("filename", &Span::filename_, "Source filename")
      .def_readonly("begin_line", &Span::begin_line_, "Beginning line (1-indexed)")
      .def_readonly("begin_column", &Span::begin_column_, "Beginning column (1-indexed)")
      .def_readonly("end_line", &Span::end_line_, "Ending line (1-indexed)")
      .def_readonly("end_column", &Span::end_column_, "Ending column (1-indexed)");

  // Op - operation/function
  py::class_<Op, std::shared_ptr<Op>>(ir, "Op", "Represents callable operations in the IR")
      .def(py::init<std::string>(), py::arg("name"), "Create an operation with the given name")
      .def_readonly("name", &Op::name_, "Operation name");

  // IRNode - abstract base, const shared_ptr
  py::class_<IRNode, std::shared_ptr<IRNode>>(ir, "IRNode", "Base class for all IR nodes")
      .def_readonly("span", &IRNode::span, "Source location of this IR node");

  // Expr - abstract, const shared_ptr
  py::class_<Expr, IRNode, std::shared_ptr<Expr>>(ir, "Expr", "Base class for all expressions")
      .def(
          "__str__",
          [](const std::shared_ptr<const Expr>& self) {
            ExprPrinter printer;
            return printer.Print(self);
          },
          "String representation of the expression")
      .def(
          "__repr__",
          [](const std::shared_ptr<const Expr>& self) {
            ExprPrinter printer;
            return "<ir." + std::string(self->type_name()) + ": " + printer.Print(self) + ">";
          },
          "Detailed representation of the expression");

  // Var - const shared_ptr
  py::class_<Var, Expr, std::shared_ptr<Var>>(ir, "Var", "Variable reference expression")
      .def(py::init<std::string, Span>(), py::arg("name"), py::arg("span"), "Create a variable reference")
      .def_readonly("name", &Var::name_, "Variable name");

  // ConstInt - const shared_ptr
  py::class_<ConstInt, Expr, std::shared_ptr<ConstInt>>(ir, "ConstInt", "Constant integer expression")
      .def(py::init<int, Span>(), py::arg("value"), py::arg("span"), "Create a constant integer expression")
      .def_readonly("value", &ConstInt::value, "Constant integer value");

  // Call - const shared_ptr
  py::class_<Call, Expr, std::shared_ptr<Call>>(ir, "Call", "Function call expression")
      .def(py::init<OpPtr, std::vector<ExprPtr>, Span>(), py::arg("op"), py::arg("args"), py::arg("span"),
           "Create a function call expression")
      .def_readonly("op", &Call::op_, "Operation/function")
      .def_readonly("args", &Call::args_, "Arguments");

  // BinaryExpr - abstract, const shared_ptr
  py::class_<BinaryExpr, Expr, std::shared_ptr<BinaryExpr>>(ir, "BinaryExpr",
                                                            "Base class for binary operations")
      .def_readonly("left", &BinaryExpr::left_, "Left operand")
      .def_readonly("right", &BinaryExpr::right_, "Right operand");

  // UnaryExpr - abstract, const shared_ptr
  py::class_<UnaryExpr, Expr, std::shared_ptr<UnaryExpr>>(ir, "UnaryExpr", "Base class for unary operations")
      .def_readonly("operand", &UnaryExpr::operand_, "Operand");

// Macro to bind binary expression nodes
#define BIND_BINARY_EXPR(OpName, Description)                                                      \
  py::class_<OpName, BinaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)                \
      .def(py::init<ExprPtr, ExprPtr, Span>(), py::arg("left"), py::arg("right"), py::arg("span"), \
           "Create " Description);

  // Bind all binary expression nodes
  BIND_BINARY_EXPR(Add, "Addition expression (left + right)")
  BIND_BINARY_EXPR(Sub, "Subtraction expression (left - right)")
  BIND_BINARY_EXPR(Mul, "Multiplication expression (left * right)")
  BIND_BINARY_EXPR(FloorDiv, "Floor division expression (left // right)")
  BIND_BINARY_EXPR(FloorMod, "Floor modulo expression (left % right)")
  BIND_BINARY_EXPR(FloatDiv, "Float division expression (left / right)")
  BIND_BINARY_EXPR(Min, "Minimum expression (min(left, right))")
  BIND_BINARY_EXPR(Max, "Maximum expression (max(left, right))")
  BIND_BINARY_EXPR(Pow, "Power expression (left ** right)")
  BIND_BINARY_EXPR(Eq, "Equality expression (left == right)")
  BIND_BINARY_EXPR(Ne, "Inequality expression (left != right)")
  BIND_BINARY_EXPR(Lt, "Less than expression (left < right)")
  BIND_BINARY_EXPR(Le, "Less than or equal to expression (left <= right)")
  BIND_BINARY_EXPR(Gt, "Greater than expression (left > right)")
  BIND_BINARY_EXPR(Ge, "Greater than or equal to expression (left >= right)")
  BIND_BINARY_EXPR(And, "Logical and expression (left and right)")
  BIND_BINARY_EXPR(Or, "Logical or expression (left or right)")
  BIND_BINARY_EXPR(Xor, "Logical xor expression (left xor right)")
  BIND_BINARY_EXPR(BitAnd, "Bitwise and expression (left & right)")
  BIND_BINARY_EXPR(BitOr, "Bitwise or expression (left | right)")
  BIND_BINARY_EXPR(BitXor, "Bitwise xor expression (left ^ right)")
  BIND_BINARY_EXPR(BitShiftLeft, "Bitwise left shift expression (left << right)")
  BIND_BINARY_EXPR(BitShiftRight, "Bitwise right shift expression (left >> right)")

#undef BIND_BINARY_EXPR

// Macro to bind unary expression nodes
#define BIND_UNARY_EXPR(OpName, Description)                                       \
  py::class_<OpName, UnaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description) \
      .def(py::init<ExprPtr, Span>(), py::arg("operand"), py::arg("span"), "Create " Description);

  // Bind all unary expression nodes
  BIND_UNARY_EXPR(Abs, "Absolute value expression (abs(operand))")
  BIND_UNARY_EXPR(Neg, "Negation expression (-operand)")
  BIND_UNARY_EXPR(Not, "Logical not expression (not operand)")
  BIND_UNARY_EXPR(BitNot, "Bitwise not expression (~operand)")

#undef BIND_UNARY_EXPR
}

}  // namespace python
}  // namespace pypto
