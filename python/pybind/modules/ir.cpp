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
#include <tuple>
#include <vector>

#include "../bindings.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tensor_expr.h"
#include "pypto/ir/transform/printer.h"
#include "pypto/ir/transform/transformers.h"

namespace py = pybind11;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)
using pypto::DataType;

// Helper to bind a single field using reflection
template <typename ClassType, typename PyClassType, typename FieldDesc>
void BindField(PyClassType& py_class, const FieldDesc& desc) {
  py_class.def_readonly(desc.name, desc.field_ptr);
}

// Helper to bind all fields from a tuple of field descriptors
template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType& py_class, const DescTuple& descriptors, std::index_sequence<Is...>) {
  (BindField<ClassType>(py_class, std::get<Is>(descriptors)), ...);
}

// Main function to bind all fields using reflection
template <typename ClassType, typename PyClassType>
void BindFields(PyClassType& py_class) {
  constexpr auto descriptors = ClassType::GetFieldDescriptors();
  constexpr auto num_fields = std::tuple_size_v<decltype(descriptors)>;
  BindFieldsImpl<ClassType>(py_class, descriptors, std::make_index_sequence<num_fields>{});
}

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
  auto irnode_class =
      py::class_<IRNode, std::shared_ptr<IRNode>>(ir, "IRNode", "Base class for all IR nodes");
  BindFields<IRNode>(irnode_class);

  // Expr - abstract base, const shared_ptr
  auto expr_class =
      py::class_<Expr, IRNode, std::shared_ptr<Expr>>(ir, "Expr", "Base class for all expressions");
  BindFields<Expr>(expr_class);

  // ScalarExpr - abstract, const shared_ptr
  auto scalar_expr_class = py::class_<ScalarExpr, Expr, std::shared_ptr<ScalarExpr>>(
      ir, "ScalarExpr", "Base class for all scalar expressions");
  BindFields<ScalarExpr>(scalar_expr_class);
  scalar_expr_class
      .def(
          "__str__",
          [](const std::shared_ptr<const ScalarExpr>& self) {
            ExprPrinter printer;
            return printer.Print(self);
          },
          "String representation of the expression")
      .def(
          "__repr__",
          [](const std::shared_ptr<const ScalarExpr>& self) {
            ExprPrinter printer;
            return "<ir." + self->TypeName() + ": " + printer.Print(self) + ">";
          },
          "Detailed representation of the expression");

  // Var - const shared_ptr
  auto var_class =
      py::class_<Var, ScalarExpr, std::shared_ptr<Var>>(ir, "Var", "Variable reference expression");
  var_class.def(py::init([](const std::string& name, DataType dtype, const Span& span) {
                  return std::make_shared<Var>(name, dtype, span);
                }),
                py::arg("name"), py::arg("dtype"), py::arg("span"), "Create a variable reference");
  BindFields<Var>(var_class);

  // ConstInt - const shared_ptr
  auto constint_class = py::class_<ConstInt, ScalarExpr, std::shared_ptr<ConstInt>>(
      ir, "ConstInt", "Constant integer expression");
  constint_class.def(py::init([](int value, DataType dtype, const Span& span) {
                       return std::make_shared<ConstInt>(value, dtype, span);
                     }),
                     py::arg("value"), py::arg("dtype"), py::arg("span"),
                     "Create a constant integer expression");
  BindFields<ConstInt>(constint_class);

  // Call - const shared_ptr
  auto call_class =
      py::class_<Call, ScalarExpr, std::shared_ptr<Call>>(ir, "Call", "Function call expression");
  call_class.def(py::init([](const OpPtr& op, const std::vector<ScalarExprPtr>& args, DataType dtype,
                             const Span& span) { return std::make_shared<Call>(op, args, dtype, span); }),
                 py::arg("op"), py::arg("args"), py::arg("dtype"), py::arg("span"),
                 "Create a function call expression");
  BindFields<Call>(call_class);

  // BinaryExpr - abstract, const shared_ptr
  auto binaryexpr_class = py::class_<BinaryExpr, ScalarExpr, std::shared_ptr<BinaryExpr>>(
      ir, "BinaryExpr", "Base class for binary operations");
  BindFields<BinaryExpr>(binaryexpr_class);

  // UnaryExpr - abstract, const shared_ptr
  auto unaryexpr_class = py::class_<UnaryExpr, ScalarExpr, std::shared_ptr<UnaryExpr>>(
      ir, "UnaryExpr", "Base class for unary operations");
  BindFields<UnaryExpr>(unaryexpr_class);

// Macro to bind binary expression nodes
#define BIND_BINARY_EXPR(OpName, Description)                                                             \
  py::class_<OpName, BinaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)                       \
      .def(py::init([](const ScalarExprPtr& left, const ScalarExprPtr& right, DataType dtype,             \
                       const Span& span) { return std::make_shared<OpName>(left, right, dtype, span); }), \
           py::arg("left"), py::arg("right"), py::arg("dtype"), py::arg("span"), "Create " Description);

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
#define BIND_UNARY_EXPR(OpName, Description)                                             \
  py::class_<OpName, UnaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)       \
      .def(py::init([](const ScalarExprPtr& operand, DataType dtype, const Span& span) { \
             return std::make_shared<OpName>(operand, dtype, span);                      \
           }),                                                                           \
           py::arg("operand"), py::arg("dtype"), py::arg("span"), "Create " Description);

  // Bind all unary expression nodes
  BIND_UNARY_EXPR(Abs, "Absolute value expression (abs(operand))")
  BIND_UNARY_EXPR(Neg, "Negation expression (-operand)")
  BIND_UNARY_EXPR(Not, "Logical not expression (not operand)")
  BIND_UNARY_EXPR(BitNot, "Bitwise not expression (~operand)")

#undef BIND_UNARY_EXPR

  // Bind structural hash and equality functions
  ir.def("structural_hash", &structural_hash, py::arg("node"), py::arg("enable_auto_mapping") = false,
         "Compute structural hash of an IR node. "
         "Ignores source location (Span). Two IR nodes with identical structure hash to the same value. "
         "If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  ir.def("structural_equal", &structural_equal, py::arg("lhs"), py::arg("rhs"),
         py::arg("enable_auto_mapping") = false,
         "Check if two IR nodes are structurally equal. "
         "Ignores source location (Span). Returns True if IR nodes have identical structure. "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  // ========== Tensor Expressions ==========

  // TensorExpr - abstract, const shared_ptr
  auto tensor_expr_class = py::class_<TensorExpr, Expr, std::shared_ptr<TensorExpr>>(
      ir, "TensorExpr", "Base class for all tensor expressions");
  BindFields<TensorExpr>(tensor_expr_class);
  tensor_expr_class
      .def(
          "__str__",
          [](const std::shared_ptr<const TensorExpr>& self) {
            ExprPrinter printer;
            return printer.Print(self);
          },
          "String representation of the expression")
      .def(
          "__repr__",
          [](const std::shared_ptr<const TensorExpr>& self) {
            ExprPrinter printer;
            return "<ir." + self->TypeName() + ": " + printer.Print(self) + ">";
          },
          "Detailed representation of the expression");

  // TensorVar - const shared_ptr
  auto tensor_var_class = py::class_<TensorVar, TensorExpr, std::shared_ptr<TensorVar>>(
      ir, "TensorVar", "Tensor variable reference");
  tensor_var_class.def(
      py::init([](const std::string& name, DataType dtype, const std::vector<ScalarExprPtr>& shape,
                  const Span& span) { return std::make_shared<TensorVar>(name, dtype, shape, span); }),
      py::arg("name"), py::arg("dtype"), py::arg("shape"), py::arg("span"),
      "Create a tensor variable reference");
  BindFields<TensorVar>(tensor_var_class);

  // ========== Statements ==========

  // Stmt - abstract base, const shared_ptr
  auto stmt_class =
      py::class_<Stmt, IRNode, std::shared_ptr<Stmt>>(ir, "Stmt", "Base class for all statements");
  stmt_class.def(py::init([](const Span& span) { return std::make_shared<Stmt>(span); }), py::arg("span"),
                 "Create a statement");
  BindFields<Stmt>(stmt_class);
}

}  // namespace python
}  // namespace pypto
