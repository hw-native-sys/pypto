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
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "../module.h"
#include "pypto/core/common.h"
#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/serialization/deserializer.h"
#include "pypto/ir/serialization/serializer.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/printer.h"
#include "pypto/ir/transform/transformers.h"
#include "pypto/ir/type.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)
using pypto::DataType;

template <typename T>
bool TryConvertAnyToPy(const std::any& value, nb::object& out) {
  if (value.type() != typeid(T)) {
    return false;
  }
  out = nb::cast(std::any_cast<const T&>(value));
  return true;
}

template <typename... Ts>
nb::object AnyToPyObject(const std::any& value, const std::string& key) {
  nb::object out;
  if ((TryConvertAnyToPy<Ts>(value, out) || ...)) {
    return out;
  }
  throw pypto::TypeError("Attribute '" + key + "' has unsupported type");
}

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
// __str__ uses the new PythonPrint for Python-style IR syntax with "pi" prefix
// __repr__ uses the old IRPrinter for compact representation
template <typename T, typename PyClassType>
void BindStrRepr(PyClassType& nb_class) {
  nb_class
      .def(
          "__str__",
          [](const std::shared_ptr<const T>& self) {
            // Use unified PythonPrint API with default "pi" prefix
            return PythonPrint(std::static_pointer_cast<const IRNode>(self), "pi");
          },
          "Python-style string representation")
      .def(
          "__repr__",
          [](const std::shared_ptr<const T>& self) {
            IRPrinter printer;  // Use old printer for compact __repr__
            std::string printed;
            if constexpr (std::is_same_v<T, Function>) {
              printed = printer.Print(std::static_pointer_cast<const Function>(self));
            } else if constexpr (std::is_same_v<T, Program>) {
              printed = printer.Print(std::static_pointer_cast<const Program>(self));
            } else if constexpr (std::is_base_of_v<Expr, T>) {
              printed = printer.Print(std::static_pointer_cast<const Expr>(self));
            } else if constexpr (std::is_base_of_v<Stmt, T>) {
              printed = printer.Print(std::static_pointer_cast<const Stmt>(self));
            } else {
              printed = self->TypeName();
            }
            return "<ir." + self->TypeName() + ": " + printed + ">";
          },
          "Detailed representation (compact style)")
      .def(
          "as_python",
          [](const std::shared_ptr<const T>& self, const std::string& prefix) {
            return PythonPrint(std::static_pointer_cast<const IRNode>(self), prefix);
          },
          nb::arg("prefix") = "pi",
          "Convert to Python-style string representation.\n\n"
          "Args:\n"
          "    prefix: Module prefix (default 'pi' for 'import pypto.ir as pi')");
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
      .def_ro("name", &Op::name_, "Operation name")
      .def(
          "get_attr",
          [](const Op& self, const std::string& key) -> nb::object {
            const auto& value = self.GetAttrAny(key);
            return AnyToPyObject<std::string, int, int64_t, bool, float, double>(value, key);
          },
          nb::arg("key"), "Get an attribute value (automatically determines type)")
      .def("has_attr", &Op::HasAttr, nb::arg("key"), "Check if an attribute exists")
      .def("get_attr_keys", &Op::GetAttrKeys, "Get all attribute keys");

  // GlobalVar - global function reference
  nb::class_<GlobalVar, Op>(ir, "GlobalVar",
                            "Global variable reference for functions in a program. "
                            "Can be used in Call expressions to invoke functions within the same program.")
      .def(nb::init<std::string>(), nb::arg("name"),
           "Create a global variable reference with the given name");

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

  // TileType - const shared_ptr
  auto tile_type_class = nb::class_<TileType, Type>(
      ir, "TileType", "Tile type representation (2D tensor with at most 2 dimensions)");
  tile_type_class.def(nb::init<DataType, const std::vector<ExprPtr>&>(), nb::arg("dtype"), nb::arg("shape"),
                      "Create a tile type (validates shape has at most 2 dimensions)");
  BindFields<TileType>(tile_type_class);

  // Dynamic dimension constant
  ir.attr("DYNAMIC_DIM") = kDynamicDim;

  // OpRegistry
  ir.def(
      "create_op_call",
      [](const std::string& op_name, const std::vector<ExprPtr>& args, const Span& span) {
        return OpRegistry::GetInstance().Create(op_name, args, span);
      },
      nb::arg("op_name"), nb::arg("args"), nb::arg("span"),
      "Create a Call expression for a registered operator with automatic type deduction");

  ir.def(
      "is_op_registered",
      [](const std::string& op_name) { return OpRegistry::GetInstance().IsRegistered(op_name); },
      nb::arg("op_name"), "Check if an operator is registered");

  ir.def(
      "get_op", [](const std::string& op_name) { return OpRegistry::GetInstance().GetOp(op_name); },
      nb::arg("op_name"), "Get an operator instance by name");

  // Var - const shared_ptr
  auto var_class = nb::class_<Var, Expr>(ir, "Var", "Variable reference expression");
  var_class.def(nb::init<const std::string&, const TypePtr&, const Span&>(), nb::arg("name"), nb::arg("type"),
                nb::arg("span"), "Create a variable reference");
  BindStrRepr<Var>(var_class);
  BindFields<Var>(var_class);

  // IterArg - const shared_ptr
  auto iterarg_class = nb::class_<IterArg, Var>(ir, "IterArg", "Iteration argument variable");
  iterarg_class.def(nb::init<const std::string&, const TypePtr&, const ExprPtr&, const Span&>(),
                    nb::arg("name"), nb::arg("type"), nb::arg("initValue"), nb::arg("span"),
                    "Create an iteration argument with initial value");
  BindStrRepr<IterArg>(iterarg_class);
  BindFields<IterArg>(iterarg_class);

  // ConstInt - const shared_ptr
  auto constint_class = nb::class_<ConstInt, ScalarExpr>(ir, "ConstInt", "Constant integer expression");
  constint_class.def(nb::init<int, DataType, const Span&>(), nb::arg("value"), nb::arg("dtype"),
                     nb::arg("span"), "Create a constant integer expression");
  BindFields<ConstInt>(constint_class);

  // Call - const shared_ptr
  auto call_class = nb::class_<Call, Expr>(ir, "Call", "Function call expression");
  call_class.def(nb::init<const OpPtr&, const std::vector<ExprPtr>&, const Span&>(), nb::arg("op"),
                 nb::arg("args"), nb::arg("span"), "Create a function call expression");
  call_class.def(nb::init<const OpPtr&, const std::vector<ExprPtr>&, const TypePtr&, const Span&>(),
                 nb::arg("op"), nb::arg("args"), nb::arg("type"), nb::arg("span"),
                 "Create a function call expression with explicit type");
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

  // Serialization functions
  ir.def(
      "serialize",
      [](const IRNodePtr& node) {
        auto data = serialization::Serialize(node);
        return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
      },
      nb::arg("node"), "Serialize an IR node to MessagePack bytes");

  ir.def(
      "deserialize",
      [](const nb::bytes& data) {
        std::vector<uint8_t> vec(static_cast<const uint8_t*>(data.data()),
                                 static_cast<const uint8_t*>(data.data()) + data.size());
        return serialization::Deserialize(vec);
      },
      nb::arg("data"), "Deserialize an IR node from MessagePack bytes");

  ir.def("serialize_to_file", &serialization::SerializeToFile, nb::arg("node"), nb::arg("path"),
         "Serialize an IR node to a file");

  ir.def("deserialize_from_file", &serialization::DeserializeFromFile, nb::arg("path"),
         "Deserialize an IR node from a file");

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
  if_stmt_class.def(nb::init<const ExprPtr&, const StmtPtr&, const std::optional<StmtPtr>&,
                             const std::vector<VarPtr>&, const Span&>(),
                    nb::arg("condition"), nb::arg("then_body"), nb::arg("else_body").none(),
                    nb::arg("return_vars"), nb::arg("span"),
                    "Create a conditional statement with then and else branches (else_body can be None)");
  BindFields<IfStmt>(if_stmt_class);
  BindStrRepr<IfStmt>(if_stmt_class);

  // YieldStmt - const shared_ptr
  auto yield_stmt_class = nb::class_<YieldStmt, Stmt>(ir, "YieldStmt", "Yield statement: yield value");
  yield_stmt_class.def(nb::init<const std::vector<ExprPtr>&, const Span&>(), nb::arg("value"),
                       nb::arg("span"), "Create a yield statement with a list of expressions");
  yield_stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a yield statement without values");
  BindFields<YieldStmt>(yield_stmt_class);
  BindStrRepr<YieldStmt>(yield_stmt_class);

  // ForStmt - const shared_ptr
  auto for_stmt_class = nb::class_<ForStmt, Stmt>(
      ir, "ForStmt", "For loop statement: for loop_var in range(start, stop, step): body");
  for_stmt_class.def(
      nb::init<const VarPtr&, const ExprPtr&, const ExprPtr&, const ExprPtr&, const std::vector<IterArgPtr>&,
               const StmtPtr&, const std::vector<VarPtr>&, const Span&>(),
      nb::arg("loop_var"), nb::arg("start"), nb::arg("stop"), nb::arg("step"), nb::arg("iter_args"),
      nb::arg("body"), nb::arg("return_vars"), nb::arg("span"), "Create a for loop statement");
  BindFields<ForStmt>(for_stmt_class);
  BindStrRepr<ForStmt>(for_stmt_class);

  // SeqStmts - const shared_ptr
  auto seq_stmts_class =
      nb::class_<SeqStmts, Stmt>(ir, "SeqStmts", "Sequence of statements: a sequence of statements");
  seq_stmts_class.def(nb::init<const std::vector<StmtPtr>&, const Span&>(), nb::arg("stmts"), nb::arg("span"),
                      "Create a sequence of statements");
  BindFields<SeqStmts>(seq_stmts_class);
  BindStrRepr<SeqStmts>(seq_stmts_class);

  // OpStmts - const shared_ptr
  auto op_stmts_class =
      nb::class_<OpStmts, Stmt>(ir, "OpStmts", "Operation statements: a sequence of assignment statements");
  op_stmts_class.def(nb::init<const std::vector<AssignStmtPtr>&, const Span&>(), nb::arg("stmts"),
                     nb::arg("span"), "Create an operation statements");
  BindFields<OpStmts>(op_stmts_class);
  BindStrRepr<OpStmts>(op_stmts_class);

  // Function - const shared_ptr
  auto function_class = nb::class_<Function, IRNode>(
      ir, "Function", "Function definition with name, parameters, return types, and body");
  function_class.def(nb::init<const std::string&, const std::vector<VarPtr>&, const std::vector<TypePtr>&,
                              const StmtPtr&, const Span&>(),
                     nb::arg("name"), nb::arg("params"), nb::arg("return_types"), nb::arg("body"),
                     nb::arg("span"), "Create a function definition");
  BindFields<Function>(function_class);
  BindStrRepr<Function>(function_class);

  // Program - const shared_ptr
  auto program_class =
      nb::class_<Program, IRNode>(ir, "Program",
                                  "Program definition with functions mapped by GlobalVar references. "
                                  "Functions are automatically sorted by name for deterministic ordering.");
  program_class.def(nb::init<const std::vector<FunctionPtr>&, const std::string&, const Span&>(),
                    nb::arg("functions"), nb::arg("name"), nb::arg("span"),
                    "Create a program from a list of functions. "
                    "GlobalVar references are created automatically from function names.");
  program_class.def("get_function", &Program::GetFunction, nb::arg("name"),
                    "Get a function by name, returns None if not found");
  program_class.def("get_global_var", &Program::GetGlobalVar, nb::arg("name"),
                    "Get a GlobalVar by name, returns None if not found");
  // Custom property for functions_ map that converts to Python dict
  program_class.def_prop_ro(
      "functions",
      [](const std::shared_ptr<const Program>& self) {
        nb::dict result;
        for (const auto& [gvar, func] : self->functions_) {
          result[nb::cast(gvar)] = nb::cast(func);
        }
        return result;
      },
      "Map of GlobalVar references to their corresponding functions, sorted by GlobalVar name");
  program_class.def_ro("name", &Program::name_, "Program name");
  program_class.def_ro("span", &Program::span_, "Source location");
  BindStrRepr<Program>(program_class);

  // Python-style printer function - unified API
  ir.def(
      "python_print",
      [](const IRNodePtr& node, const std::string& prefix) { return PythonPrint(node, prefix); },
      nb::arg("node"), nb::arg("prefix") = "pi",
      "Print IR node (Expr, Stmt, Function, or Program) in Python IR syntax.\n\n"
      "Args:\n"
      "    node: IR node to print\n"
      "    prefix: Module prefix (default 'pi' for 'import pypto.ir as pi')");
}

}  // namespace python
}  // namespace pypto
