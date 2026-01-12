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
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../module.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/printer.h"
#include "pypto/ir/transform/transformers.h"
#include "pypto/ir/type.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)
using pypto::DataType;

// Helper to bind a single field using reflection
template <typename ClassType, typename PyClassType, typename FieldDesc>
void BindField(PyClassType& nb_class, const FieldDesc& desc) {
  nb_class.def_ro(desc.name, desc.field_ptr);
}

// Helper to bind all fields from a tuple of field descriptors
template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType& nb_class, const DescTuple& descriptors, std::index_sequence<Is...>) {
  (BindField<ClassType>(nb_class, std::get<Is>(descriptors)), ...);
}

// Main function to bind all fields using reflection
template <typename ClassType, typename PyClassType>
void BindFields(PyClassType& nb_class) {
  constexpr auto descriptors = ClassType::GetFieldDescriptors();
  constexpr auto num_fields = std::tuple_size_v<decltype(descriptors)>;
  BindFieldsImpl<ClassType>(nb_class, descriptors, std::make_index_sequence<num_fields>{});
}

// Helper to bind __str__ and __repr__ methods for IR nodes
template <typename T, typename PyClassType>
void BindStrRepr(PyClassType& nb_class) {
  nb_class
      .def(
          "__str__",
          [](const std::shared_ptr<const T>& self) {
            IRPrinter printer;
            return printer.Print(self);
          },
          "String representation")
      .def(
          "__repr__",
          [](const std::shared_ptr<const T>& self) {
            IRPrinter printer;
            return "<ir." + self->TypeName() + ": " + printer.Print(self) + ">";
          },
          "Detailed representation");
}

void BindIR(nb::module_& m) {
  nb::module_ ir = m.def_submodule("ir", "PyPTO IR (Intermediate Representation) module");

  // Span - value type, copy semantics
  nb::class_<Span>(ir, "Span", "Source location information tracking file, line, and column positions")
      .def(nb::init<std::string, int, int, int, int>(), nb::arg("filename"), nb::arg("begin_line"),
           nb::arg("begin_column"), nb::arg("end_line") = -1, nb::arg("end_column") = -1,
           "Create a source span")
      .def("to_string", &Span::to_string, "Convert span to string representation")
      .def("is_valid", &Span::is_valid, "Check if the span has valid coordinates")
      .def_static("unknown", &Span::unknown,
                  "Create an unknown/invalid span for cases where source location is unavailable")
      .def("__repr__", &Span::to_string)
      .def("__str__", &Span::to_string)
      .def_ro("filename", &Span::filename_, "Source filename")
      .def_ro("begin_line", &Span::begin_line_, "Beginning line (1-indexed)")
      .def_ro("begin_column", &Span::begin_column_, "Beginning column (1-indexed)")
      .def_ro("end_line", &Span::end_line_, "Ending line (1-indexed)")
      .def_ro("end_column", &Span::end_column_, "Ending column (1-indexed)");

  // Op - operation/function
  nb::class_<Op>(ir, "Op", "Represents callable operations in the IR")
      .def(nb::init<std::string>(), nb::arg("name"), "Create an operation with the given name")
      .def_ro("name", &Op::name_, "Operation name");

  // IRNode - abstract base, const shared_ptr
  auto irnode_class = nb::class_<IRNode>(ir, "IRNode", "Base class for all IR nodes");
  BindFields<IRNode>(irnode_class);

  // Expr - abstract base, const shared_ptr
  auto expr_class = nb::class_<Expr, IRNode>(ir, "Expr", "Base class for all expressions");
  BindFields<Expr>(expr_class);

  // ScalarExpr - abstract, const shared_ptr
  auto scalar_expr_class =
      nb::class_<ScalarExpr, Expr>(ir, "ScalarExpr", "Base class for all scalar expressions");
  BindFields<ScalarExpr>(scalar_expr_class);
  BindStrRepr<ScalarExpr>(scalar_expr_class);

  // Type - abstract base, const shared_ptr
  auto type_class = nb::class_<Type>(ir, "Type", "Base class for type representations");
  BindFields<Type>(type_class);

  // UnknownType - const shared_ptr
  auto unknown_type_class =
      nb::class_<UnknownType, Type>(ir, "UnknownType", "Unknown or unspecified type representation");
  unknown_type_class.def(nb::init<>(), "Create an unknown type");
  unknown_type_class.def_static(
      "get", []() { return GetUnknownType(); }, "Get the singleton UnknownType instance");
  BindFields<UnknownType>(unknown_type_class);

  // ScalarType - const shared_ptr
  auto scalar_type_class = nb::class_<ScalarType, Type>(ir, "ScalarType", "Scalar type representation");
  scalar_type_class.def(nb::init<DataType>(), nb::arg("dtype"), "Create a scalar type");
  BindFields<ScalarType>(scalar_type_class);

  // TensorType - const shared_ptr
  auto tensor_type_class = nb::class_<TensorType, Type>(ir, "TensorType", "Tensor type representation");
  tensor_type_class.def(nb::init<DataType, const std::vector<ExprPtr>&>(), nb::arg("dtype"), nb::arg("shape"),
                        "Create a tensor type");
  BindFields<TensorType>(tensor_type_class);

  // Var - const shared_ptr
  auto var_class = nb::class_<Var, Expr>(ir, "Var", "Variable reference expression");
  var_class.def(nb::init<const std::string&, const TypePtr&, const Span&>(), nb::arg("name"), nb::arg("type"),
                nb::arg("span"), "Create a variable reference");
  BindStrRepr<Var>(var_class);
  BindFields<Var>(var_class);

  // ConstInt - const shared_ptr
  auto constint_class = nb::class_<ConstInt, ScalarExpr>(ir, "ConstInt", "Constant integer expression");
  constint_class.def(nb::init<int, DataType, const Span&>(), nb::arg("value"), nb::arg("dtype"),
                     nb::arg("span"), "Create a constant integer expression");
  BindFields<ConstInt>(constint_class);

  // Call - const shared_ptr
  auto call_class = nb::class_<Call, Expr>(ir, "Call", "Function call expression");
  call_class.def(nb::init<const OpPtr&, const std::vector<ExprPtr>&, const Span&>(), nb::arg("op"),
                 nb::arg("args"), nb::arg("span"), "Create a function call expression");
  BindStrRepr<Call>(call_class);
  BindFields<Call>(call_class);

  // BinaryExpr - abstract, const shared_ptr
  auto binaryexpr_class =
      nb::class_<BinaryExpr, ScalarExpr>(ir, "BinaryExpr", "Base class for binary operations");
  BindFields<BinaryExpr>(binaryexpr_class);

  // UnaryExpr - abstract, const shared_ptr
  auto unaryexpr_class =
      nb::class_<UnaryExpr, ScalarExpr>(ir, "UnaryExpr", "Base class for unary operations");
  BindFields<UnaryExpr>(unaryexpr_class);

// Macro to bind binary expression nodes
#define BIND_BINARY_EXPR(OpName, Description)                                                  \
  nb::class_<OpName, BinaryExpr>(ir, #OpName, Description)                                     \
      .def(nb::init<const ExprPtr&, const ExprPtr&, DataType, const Span&>(), nb::arg("left"), \
           nb::arg("right"), nb::arg("dtype"), nb::arg("span"), "Create " Description);

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
#define BIND_UNARY_EXPR(OpName, Description)                                                        \
  nb::class_<OpName, UnaryExpr>(ir, #OpName, Description)                                           \
      .def(nb::init<const ExprPtr&, DataType, const Span&>(), nb::arg("operand"), nb::arg("dtype"), \
           nb::arg("span"), "Create " Description);

  // Bind all unary expression nodes
  BIND_UNARY_EXPR(Abs, "Absolute value expression (abs(operand))")
  BIND_UNARY_EXPR(Neg, "Negation expression (-operand)")
  BIND_UNARY_EXPR(Not, "Logical not expression (not operand)")
  BIND_UNARY_EXPR(BitNot, "Bitwise not expression (~operand)")

#undef BIND_UNARY_EXPR

  // Bind structural hash and equality functions
  ir.def("structural_hash", &structural_hash, nb::arg("node"), nb::arg("enable_auto_mapping") = false,
         "Compute structural hash of an IR node. "
         "Ignores source location (Span). Two IR nodes with identical structure hash to the same value. "
         "If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  ir.def("structural_equal", &structural_equal, nb::arg("lhs"), nb::arg("rhs"),
         nb::arg("enable_auto_mapping") = false,
         "Check if two IR nodes are structurally equal. "
         "Ignores source location (Span). Returns True if IR nodes have identical structure. "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  // ========== Statements ==========

  // Stmt - abstract base, const shared_ptr
  auto stmt_class = nb::class_<Stmt, IRNode>(ir, "Stmt", "Base class for all statements");
  stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a statement");
  BindFields<Stmt>(stmt_class);
  BindStrRepr<Stmt>(stmt_class);

  // AssignStmt - const shared_ptr
  auto assign_stmt_class =
      nb::class_<AssignStmt, Stmt>(ir, "AssignStmt", "Assignment statement: var = value");
  assign_stmt_class.def(nb::init<const VarPtr&, const ExprPtr&, const Span&>(), nb::arg("var"),
                        nb::arg("value"), nb::arg("span"), "Create an assignment statement");
  BindFields<AssignStmt>(assign_stmt_class);
  BindStrRepr<AssignStmt>(assign_stmt_class);

  // IfStmt - const shared_ptr
  auto if_stmt_class = nb::class_<IfStmt, Stmt>(
      ir, "IfStmt", "Conditional statement: if condition then then_body else else_body");
  if_stmt_class.def(
      nb::init<const ExprPtr&, const std::vector<StmtPtr>&, const std::vector<StmtPtr>&, const Span&>(),
      nb::arg("condition"), nb::arg("then_body"), nb::arg("else_body"), nb::arg("span"),
      "Create a conditional statement");
  BindFields<IfStmt>(if_stmt_class);
  BindStrRepr<IfStmt>(if_stmt_class);

  // YieldStmt - const shared_ptr
  auto yield_stmt_class = nb::class_<YieldStmt, Stmt>(ir, "YieldStmt", "Yield statement: yield value");
  yield_stmt_class.def(nb::init<const std::vector<VarPtr>&, const Span&>(), nb::arg("value"), nb::arg("span"),
                       "Create a yield statement with a list of variables");
  yield_stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a yield statement without values");
  BindFields<YieldStmt>(yield_stmt_class);
  BindStrRepr<YieldStmt>(yield_stmt_class);

  // ForStmt - const shared_ptr
  auto for_stmt_class = nb::class_<ForStmt, Stmt>(
      ir, "ForStmt", "For loop statement: for loop_var in range(start, stop, step): body");
  for_stmt_class.def(nb::init<const VarPtr&, const ExprPtr&, const ExprPtr&, const ExprPtr&,
                              const std::vector<StmtPtr>&, const Span&>(),
                     nb::arg("loop_var"), nb::arg("start"), nb::arg("stop"), nb::arg("step"), nb::arg("body"),
                     nb::arg("span"), "Create a for loop statement");
  BindFields<ForStmt>(for_stmt_class);
  BindStrRepr<ForStmt>(for_stmt_class);

  // OpStmts - const shared_ptr
  auto op_stmts_class =
      nb::class_<OpStmts, Stmt>(ir, "OpStmts", "Operation statements: a sequence of statements");
  op_stmts_class.def(nb::init<const std::vector<StmtPtr>&, const Span&>(), nb::arg("stmts"), nb::arg("span"),
                     "Create an operation statements");
  BindFields<OpStmts>(op_stmts_class);
  BindStrRepr<OpStmts>(op_stmts_class);
}

}  // namespace python
}  // namespace pypto
