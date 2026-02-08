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
#include <vector>

#include "../module.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/common.h"
#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/serialization/deserializer.h"
#include "pypto/ir/serialization/serializer.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"
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
  out = nb::cast(AnyCastRef<T>(value, "converting to Python"));
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

// Helper function to convert nb::dict to vector<pair<string, any>>
std::vector<std::pair<std::string, std::any>> ConvertKwargsDict(const nb::dict& kwargs_dict) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  for (auto item : kwargs_dict) {
    std::string key = nb::cast<std::string>(item.first);

    // Try to cast to common types
    // NOTE: Check DataType BEFORE int, and bool BEFORE int (since they can be cast to int in Python)
    if (nb::isinstance<DataType>(item.second)) {
      kwargs.emplace_back(key, nb::cast<DataType>(item.second));
    } else if (nb::isinstance<nb::bool_>(item.second)) {
      kwargs.emplace_back(key, nb::cast<bool>(item.second));
    } else if (nb::isinstance<nb::int_>(item.second)) {
      kwargs.emplace_back(key, nb::cast<int>(item.second));
    } else if (nb::isinstance<nb::str>(item.second)) {
      kwargs.emplace_back(key, nb::cast<std::string>(item.second));
    } else if (nb::isinstance<nb::float_>(item.second)) {
      kwargs.emplace_back(key, nb::cast<double>(item.second));
    } else if (nb::isinstance<PipeType>(item.second)) {
      // Cast enum to int for storage
      kwargs.emplace_back(key, static_cast<int>(nb::cast<PipeType>(item.second)));
    } else if (nb::isinstance<CoreType>(item.second)) {
      // Cast enum to int for storage
      kwargs.emplace_back(key, static_cast<int>(nb::cast<CoreType>(item.second)));
    } else {
      throw pypto::TypeError("Unsupported kwarg type for key: " + key);
    }
  }
  return kwargs;
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
  nb::class_<Op>(ir, "Op",
                 "Represents callable operations in the IR. Stores the schema of allowed kwargs (key -> type "
                 "mapping). Actual kwarg values are stored per-Call instance in Call.kwargs")
      .def(nb::init<std::string>(), nb::arg("name"), "Create an operation with the given name")
      .def_ro("name", &Op::name_, "Operation name")
      .def("has_attr", &Op::HasAttr, nb::arg("key"), "Check if a kwarg is registered in the schema")
      .def("get_attr_keys", &Op::GetAttrKeys, "Get all registered kwarg keys from the schema")
      .def_prop_ro(
          "pipe", [](const Op& self) -> std::optional<PipeType> { return self.GetPipe(); },
          "Pipeline type (optional)");

  // GlobalVar - global function reference
  nb::class_<GlobalVar, Op>(ir, "GlobalVar",
                            "Global variable reference for functions in a program. "
                            "Can be used in Call expressions to invoke functions within the same program.")
      .def(nb::init<std::string>(), nb::arg("name"),
           "Create a global variable reference with the given name");

  // Type - abstract base, const shared_ptr
  auto type_class = nb::class_<Type>(ir, "Type", "Base class for type representations");
  BindFields<Type>(type_class);
  type_class.def(
      "__str__", [](const TypePtr& self) { return PythonPrint(self, "pl"); },
      "Python-style string representation");
  type_class.def(
      "__eq__", [](const TypePtr& self, const TypePtr& other) { return structural_equal(self, other); },
      "Equality comparison");

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

  // IRNode - abstract base, const shared_ptr
  auto irnode_class = nb::class_<IRNode>(ir, "IRNode", "Base class for all IR nodes");
  BindFields<IRNode>(irnode_class);
  irnode_class
      .def(
          "same_as", [](const IRNodePtr& self, const IRNodePtr& other) { return self == other; },
          nb::arg("other"), "Check if this IR node is the same as another IR node.")
      .def(
          "__str__",
          [](const IRNodePtr& self) {
            // Use unified PythonPrint API with default "pl" prefix
            return PythonPrint(self, "pl");
          },
          "Python-style string representation")
      .def(
          "as_python",
          [](const IRNodePtr& self, const std::string& prefix) { return PythonPrint(self, prefix); },
          nb::arg("prefix") = "pl",
          "Convert to Python-style string representation.\n\n"
          "Args:\n"
          "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')");

  // Expr - abstract base, const shared_ptr
  auto expr_class = nb::class_<Expr, IRNode>(ir, "Expr", "Base class for all expressions");
  BindFields<Expr>(expr_class);

  // ShapedType - abstract base for types with shape and optional memref
  auto shaped_type_class =
      nb::class_<ShapedType, Type>(ir, "ShapedType", "Base class for shaped types (tensors and tiles)");
  BindFields<ShapedType>(shaped_type_class);
  shaped_type_class.def(
      "shares_memref_with",
      [](const ShapedTypePtr& self, const ShapedTypePtr& other) {
        if (!self->memref_.has_value() || !other->memref_.has_value()) {
          return false;
        }
        return self->memref_.value().get() == other->memref_.value().get();
      },
      nb::arg("other"), "Check if this ShapedType shares the same MemRef object with another ShapedType");

  // TensorType - const shared_ptr
  auto tensor_type_class = nb::class_<TensorType, ShapedType>(ir, "TensorType", "Tensor type representation");
  tensor_type_class.def(nb::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>>(),
                        nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(),
                        "Create a tensor type");
  tensor_type_class.def(nb::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>>(),
                        nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(),
                        "Create a tensor type");
  BindFields<TensorType>(tensor_type_class);

  // TileType - const shared_ptr
  auto tile_type_class =
      nb::class_<TileType, ShapedType>(ir, "TileType", "Tile type representation (multi-dimensional tensor)");
  tile_type_class.def(
      nb::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>, std::optional<TileView>>(),
      nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(), nb::arg("tile_view") = nb::none(),
      "Create a tile type (supports multi-dimensional tensors; code generation has constraints)");
  tile_type_class.def(
      nb::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>, std::optional<TileView>>(),
      nb::arg("shape"), nb::arg("dtype"), nb::arg("memref") = nb::none(), nb::arg("tile_view") = nb::none(),
      "Create a tile type (supports multi-dimensional tensors; code generation has constraints)");
  BindFields<TileType>(tile_type_class);

  // TupleType - const shared_ptr
  auto tuple_type_class =
      nb::class_<TupleType, Type>(ir, "TupleType", "Tuple type representation (contains multiple types)");
  tuple_type_class.def(nb::init<const std::vector<TypePtr>&>(), nb::arg("types"),
                       "Create a tuple type from a list of types");
  BindFields<TupleType>(tuple_type_class);

  // MemorySpace enum
  nb::enum_<MemorySpace>(ir, "MemorySpace", "Memory space enumeration")
      .value("DDR", MemorySpace::DDR, "DDR memory (off-chip)")
      .value("UB", MemorySpace::UB, "Unified Buffer (on-chip)")
      .value("L1", MemorySpace::L1, "L1 cache")
      .value("L0A", MemorySpace::L0A, "L0A buffer")
      .value("L0B", MemorySpace::L0B, "L0B buffer")
      .value("L0C", MemorySpace::L0C, "L0C buffer")
      .export_values();

  // PipeType enum
  nb::enum_<PipeType>(ir, "PipeType", nb::is_arithmetic(), "Pipeline type enumeration")
      .value("MTE1", PipeType::MTE1, "Memory Transfer Engine 1")
      .value("MTE2", PipeType::MTE2, "Memory Transfer Engine 2")
      .value("MTE3", PipeType::MTE3, "Memory Transfer Engine 3")
      .value("M", PipeType::M, "Matrix Unit")
      .value("V", PipeType::V, "Vector Unit")
      .value("S", PipeType::S, "Scalar Unit")
      .value("FIX", PipeType::FIX, "Fix Pipe")
      .value("ALL", PipeType::ALL, "All Pipes")
      .export_values();

  // CoreType enum
  nb::enum_<CoreType>(ir, "CoreType", nb::is_arithmetic(), "Core type enumeration")
      .value("VECTOR", CoreType::VECTOR, "Vector Core")
      .value("CUBE", CoreType::CUBE, "Cube Core")
      .export_values();

  // TileView - struct for tile view information
  nb::class_<TileView>(ir, "TileView", "Tile view representation with valid shape, stride, and start offset")
      .def(nb::init<>(), "Create an empty tile view")
      .def(nb::init<const std::vector<ExprPtr>&, const std::vector<ExprPtr>&, ExprPtr>(),
           nb::arg("valid_shape"), nb::arg("stride"), nb::arg("start_offset"),
           "Create a tile view with valid_shape, stride, and start_offset")
      .def_rw("valid_shape", &TileView::valid_shape, "Valid shape dimensions")
      .def_rw("stride", &TileView::stride, "Stride for each dimension")
      .def_rw("start_offset", &TileView::start_offset, "Starting offset");

  // Dynamic dimension constant
  ir.attr("DYNAMIC_DIM") = kDynamicDim;

  // OpRegistry
  ir.def(
      "create_op_call",
      [](const std::string& op_name, const std::vector<ExprPtr>& args, const Span& span) {
        return OpRegistry::GetInstance().Create(op_name, args, span);
      },
      nb::arg("op_name"), nb::arg("args"), nb::arg("span"),
      "Create a Call expression (backward compatibility)");

  ir.def(
      "create_op_call",
      [](const std::string& op_name, const std::vector<ExprPtr>& args, const nb::dict& kwargs_dict,
         const Span& span) {
        // Convert Python dict to C++ vector<pair<string, any>> to preserve order
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        return OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
      },
      nb::arg("op_name"), nb::arg("args"), nb::arg("kwargs"), nb::arg("span"),
      "Create a Call expression with args and kwargs");

  ir.def(
      "is_op_registered",
      [](const std::string& op_name) { return OpRegistry::GetInstance().IsRegistered(op_name); },
      nb::arg("op_name"), "Check if an operator is registered");

  ir.def(
      "get_op", [](const std::string& op_name) { return OpRegistry::GetInstance().GetOp(op_name); },
      nb::arg("op_name"), "Get an operator instance by name");

  // Var - const shared_ptr
  auto var_class = nb::class_<Var, Expr>(ir, "Var", "Variable reference expression");

  var_class.def(
      nb::init<const std::string&, const TypePtr&, const Span&>(), nb::arg("name"), nb::arg("type"),
      nb::arg("span"),
      "Create a variable reference (memory reference is stored in ShapedType for Tensor/Tile types)");
  BindFields<Var>(var_class);

  // IterArg - const shared_ptr
  auto iterarg_class = nb::class_<IterArg, Var>(ir, "IterArg", "Iteration argument variable");
  iterarg_class.def(nb::init<const std::string&, const TypePtr&, const ExprPtr&, const Span&>(),
                    nb::arg("name"), nb::arg("type"), nb::arg("initValue"), nb::arg("span"),
                    "Create an iteration argument with initial value");
  BindFields<IterArg>(iterarg_class);

  // MemRef - now inherits from Var (first-class expression)
  auto memref_class =
      nb::class_<MemRef, Var>(ir, "MemRef", "Memory reference variable for shaped types (inherits from Var)");
  memref_class
      .def(nb::init<MemorySpace, ExprPtr, uint64_t, uint64_t, Span>(), nb::arg("memory_space"),
           nb::arg("addr"), nb::arg("size"), nb::arg("id"), nb::arg("span") = Span::unknown(),
           "Create a memory reference with memory_space, addr, size, id, and span")
      .def_rw("memory_space_", &MemRef::memory_space_, "Memory space (DDR, UB, L1, etc.)")
      .def_rw("addr_", &MemRef::addr_, "Starting address expression")
      .def_rw("size_", &MemRef::size_, "Size in bytes (64-bit unsigned)")
      .def_rw("id_", &MemRef::id_, "Unique identifier for this MemRef instance");

  // ConstInt - const shared_ptr
  auto constint_class = nb::class_<ConstInt, Expr>(ir, "ConstInt", "Constant integer expression");
  constint_class.def(nb::init<int64_t, DataType, const Span&>(), nb::arg("value"), nb::arg("dtype"),
                     nb::arg("span"), "Create a constant integer expression");
  BindFields<ConstInt>(constint_class);
  constint_class.def_prop_ro("dtype", &ConstInt::dtype, "Data type of the expression");

  // ConstFloat - const shared_ptr
  auto constfloat_class = nb::class_<ConstFloat, Expr>(ir, "ConstFloat", "Constant float expression");
  constfloat_class.def(nb::init<double, DataType, const Span&>(), nb::arg("value"), nb::arg("dtype"),
                       nb::arg("span"), "Create a constant float expression");
  BindFields<ConstFloat>(constfloat_class);
  constfloat_class.def_prop_ro("dtype", &ConstFloat::dtype, "Data type of the expression");

  // ConstBool - const shared_ptr
  auto constbool_class = nb::class_<ConstBool, Expr>(ir, "ConstBool", "Constant boolean expression");
  constbool_class.def(nb::init<bool, const Span&>(), nb::arg("value"), nb::arg("span"),
                      "Create a constant boolean expression");
  BindFields<ConstBool>(constbool_class);
  constbool_class.def_prop_ro("dtype", &ConstBool::dtype, "Data type of the expression (always BOOL)");

  // Call - const shared_ptr
  auto call_class = nb::class_<Call, Expr>(ir, "Call", "Function call expression");

  // Constructors without kwargs (backward compatibility)
  call_class.def(nb::init<const OpPtr&, const std::vector<ExprPtr>&, const Span&>(), nb::arg("op"),
                 nb::arg("args"), nb::arg("span"), "Create a function call expression");
  call_class.def(nb::init<const OpPtr&, const std::vector<ExprPtr>&, const TypePtr&, const Span&>(),
                 nb::arg("op"), nb::arg("args"), nb::arg("type"), nb::arg("span"),
                 "Create a function call expression with explicit type");

  // Constructors with kwargs (using nb::dict) - use factory functions
  call_class.def(
      "__init__",
      [](Call* self, const OpPtr& op, const std::vector<ExprPtr>& args, const nb::dict& kwargs_dict,
         const Span& span) {
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        new (self) Call(op, args, kwargs, span);
      },
      nb::arg("op"), nb::arg("args"), nb::arg("kwargs"), nb::arg("span"),
      "Create a function call expression with kwargs");

  call_class.def(
      "__init__",
      [](Call* self, const OpPtr& op, const std::vector<ExprPtr>& args, const nb::dict& kwargs_dict,
         const TypePtr& type, const Span& span) {
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        new (self) Call(op, args, kwargs, type, span);
      },
      nb::arg("op"), nb::arg("args"), nb::arg("kwargs"), nb::arg("type"), nb::arg("span"),
      "Create a function call expression with kwargs and explicit type");

  BindFields<Call>(call_class);

  // Expose kwargs as a read-only property
  call_class.def_prop_ro(
      "kwargs",
      [](const CallPtr& self) {
        nb::dict result;
        for (const auto& [key, value] : self->kwargs_) {
          if (value.type() == typeid(int)) {
            result[key.c_str()] = AnyCast<int>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(bool)) {
            result[key.c_str()] = AnyCast<bool>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(std::string)) {
            result[key.c_str()] = AnyCast<std::string>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(double)) {
            result[key.c_str()] = AnyCast<double>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(float)) {
            result[key.c_str()] = AnyCast<float>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(DataType)) {
            result[key.c_str()] = AnyCast<DataType>(value, "converting to Python: " + key);
          }
        }
        return result;
      },
      "Keyword arguments (metadata) for this call");

  // MakeTuple - const shared_ptr
  auto make_tuple_class = nb::class_<MakeTuple, Expr>(ir, "MakeTuple", "Tuple construction expression");
  make_tuple_class.def(nb::init<const std::vector<ExprPtr>&, const Span&>(), nb::arg("elements"),
                       nb::arg("span"), "Create a tuple construction expression");
  BindFields<MakeTuple>(make_tuple_class);

  // TupleGetItemExpr - const shared_ptr
  auto tuple_get_item_class =
      nb::class_<TupleGetItemExpr, Expr>(ir, "TupleGetItemExpr", "Tuple element access expression");
  tuple_get_item_class.def(nb::init<const ExprPtr&, int, const Span&>(), nb::arg("tuple"), nb::arg("index"),
                           nb::arg("span"), "Create a tuple element access expression");
  BindFields<TupleGetItemExpr>(tuple_get_item_class);

  // BinaryExpr - abstract, const shared_ptr
  auto binaryexpr_class = nb::class_<BinaryExpr, Expr>(ir, "BinaryExpr", "Base class for binary operations");
  BindFields<BinaryExpr>(binaryexpr_class);

  // UnaryExpr - abstract, const shared_ptr
  auto unaryexpr_class = nb::class_<UnaryExpr, Expr>(ir, "UnaryExpr", "Base class for unary operations");
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
  BIND_UNARY_EXPR(Cast, "Cast expression (cast operand to dtype)")

#undef BIND_UNARY_EXPR

  // Bind structural hash and equality functions
  ir.def("structural_hash", static_cast<uint64_t (*)(const IRNodePtr&, bool)>(&structural_hash),
         nb::arg("node"), nb::arg("enable_auto_mapping") = false,
         "Compute structural hash of an IR node. "
         "Ignores source location (Span). Two IR nodes with identical structure hash to the same value. "
         "If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");
  ir.def("structural_hash", static_cast<uint64_t (*)(const TypePtr&, bool)>(&structural_hash),
         nb::arg("type"), nb::arg("enable_auto_mapping") = false,
         "Compute structural hash of a type. "
         "Ignores source location (Span). Two types with identical structure hash to the same value. "
         "If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  ir.def("structural_equal",
         static_cast<bool (*)(const IRNodePtr&, const IRNodePtr&, bool)>(&structural_equal), nb::arg("lhs"),
         nb::arg("rhs"), nb::arg("enable_auto_mapping") = false,
         "Check if two IR nodes are structurally equal. "
         "Ignores source location (Span). Returns True if IR nodes have identical structure. "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");
  ir.def("structural_equal", static_cast<bool (*)(const TypePtr&, const TypePtr&, bool)>(&structural_equal),
         nb::arg("lhs"), nb::arg("rhs"), nb::arg("enable_auto_mapping") = false,
         "Check if two types are structurally equal. "
         "Ignores source location (Span). Returns True if types have identical structure. "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  ir.def("assert_structural_equal",
         static_cast<void (*)(const IRNodePtr&, const IRNodePtr&, bool)>(&assert_structural_equal),
         nb::arg("lhs"), nb::arg("rhs"), nb::arg("enable_auto_mapping") = false,
         "Assert two IR nodes are structurally equal. "
         "Raises ValueError with detailed error message showing the first mismatch location if they differ. "
         "Ignores source location (Span). "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");
  ir.def("assert_structural_equal",
         static_cast<void (*)(const TypePtr&, const TypePtr&, bool)>(&assert_structural_equal),
         nb::arg("lhs"), nb::arg("rhs"), nb::arg("enable_auto_mapping") = false,
         "Assert two types are structurally equal. "
         "Raises ValueError with detailed error message showing the first mismatch location if they differ. "
         "Ignores source location (Span). "
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
  BindFields<Stmt>(stmt_class);

  // AssignStmt - const shared_ptr
  auto assign_stmt_class =
      nb::class_<AssignStmt, Stmt>(ir, "AssignStmt", "Assignment statement: var = value");
  assign_stmt_class.def(nb::init<const VarPtr&, const ExprPtr&, const Span&>(), nb::arg("var"),
                        nb::arg("value"), nb::arg("span"), "Create an assignment statement");
  BindFields<AssignStmt>(assign_stmt_class);

  // IfStmt - const shared_ptr
  auto if_stmt_class = nb::class_<IfStmt, Stmt>(
      ir, "IfStmt", "Conditional statement: if condition then then_body else else_body");
  if_stmt_class.def(nb::init<const ExprPtr&, const StmtPtr&, const std::optional<StmtPtr>&,
                             const std::vector<VarPtr>&, const Span&>(),
                    nb::arg("condition"), nb::arg("then_body"), nb::arg("else_body") = nb::none(),
                    nb::arg("return_vars"), nb::arg("span"),
                    "Create a conditional statement with then and else branches (else_body can be None)");
  BindFields<IfStmt>(if_stmt_class);

  // YieldStmt - const shared_ptr
  auto yield_stmt_class = nb::class_<YieldStmt, Stmt>(ir, "YieldStmt", "Yield statement: yield value");
  yield_stmt_class.def(nb::init<const std::vector<ExprPtr>&, const Span&>(), nb::arg("value"),
                       nb::arg("span"), "Create a yield statement with a list of expressions");
  yield_stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a yield statement without values");
  BindFields<YieldStmt>(yield_stmt_class);

  // ReturnStmt - const shared_ptr
  auto return_stmt_class = nb::class_<ReturnStmt, Stmt>(ir, "ReturnStmt", "Return statement: return value");
  return_stmt_class.def(nb::init<const std::vector<ExprPtr>&, const Span&>(), nb::arg("value"),
                        nb::arg("span"), "Create a return statement with a list of expressions");
  return_stmt_class.def(nb::init<const Span&>(), nb::arg("span"), "Create a return statement without values");
  BindFields<ReturnStmt>(return_stmt_class);

  // ForStmt - const shared_ptr
  auto for_stmt_class = nb::class_<ForStmt, Stmt>(
      ir, "ForStmt", "For loop statement: for loop_var in range(start, stop, step): body");
  for_stmt_class.def(
      nb::init<const VarPtr&, const ExprPtr&, const ExprPtr&, const ExprPtr&, const std::vector<IterArgPtr>&,
               const StmtPtr&, const std::vector<VarPtr>&, const Span&>(),
      nb::arg("loop_var"), nb::arg("start"), nb::arg("stop"), nb::arg("step"), nb::arg("iter_args"),
      nb::arg("body"), nb::arg("return_vars"), nb::arg("span"), "Create a for loop statement");
  BindFields<ForStmt>(for_stmt_class);

  // SeqStmts - const shared_ptr
  auto seq_stmts_class =
      nb::class_<SeqStmts, Stmt>(ir, "SeqStmts", "Sequence of statements: a sequence of statements");
  seq_stmts_class.def(nb::init<const std::vector<StmtPtr>&, const Span&>(), nb::arg("stmts"), nb::arg("span"),
                      "Create a sequence of statements");
  BindFields<SeqStmts>(seq_stmts_class);

  // OpStmts - const shared_ptr
  auto op_stmts_class = nb::class_<OpStmts, Stmt>(
      ir, "OpStmts", "Operation statements: a sequence of assignment and/or evaluation statements");
  op_stmts_class.def(nb::init<const std::vector<StmtPtr>&, const Span&>(), nb::arg("stmts"), nb::arg("span"),
                     "Create an operation statements");
  BindFields<OpStmts>(op_stmts_class);

  // EvalStmt - const shared_ptr
  auto eval_stmt_class = nb::class_<EvalStmt, Stmt>(ir, "EvalStmt", "Evaluation statement: expr");
  eval_stmt_class.def(nb::init<const ExprPtr&, const Span&>(), nb::arg("expr"), nb::arg("span"),
                      "Create an evaluation statement");
  BindFields<EvalStmt>(eval_stmt_class);

  // FunctionType enum
  nb::enum_<FunctionType>(ir, "FunctionType", "Function type classification")
      .value("Opaque", FunctionType::Opaque, "Unspecified function type (default)")
      .value("Orchestration", FunctionType::Orchestration, "Host/AICPU control and coordination")
      .value("InCore", FunctionType::InCore, "AICore sub-graph execution")
      .export_values();

  // Function - const shared_ptr
  auto function_class = nb::class_<Function, IRNode>(
      ir, "Function", "Function definition with name, parameters, return types, and body");
  function_class.def(nb::init<const std::string&, const std::vector<VarPtr>&, const std::vector<TypePtr>&,
                              const StmtPtr&, const Span&, FunctionType>(),
                     nb::arg("name"), nb::arg("params"), nb::arg("return_types"), nb::arg("body"),
                     nb::arg("span"), nb::arg("type") = FunctionType::Opaque, "Create a function definition");
  BindFields<Function>(function_class);

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

  // Python-style printer function - unified API for IRNode
  ir.def(
      "python_print",
      [](const IRNodePtr& node, const std::string& prefix) { return PythonPrint(node, prefix); },
      nb::arg("node"), nb::arg("prefix") = "pl",
      "Print IR node (Expr, Stmt, Function, or Program) in Python IR syntax.\n\n"
      "Args:\n"
      "    node: IR node to print\n"
      "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')");

  // Python-style printer function for Type objects - use separate name to avoid overload ambiguity
  ir.def(
      "python_print_type",
      [](const TypePtr& type, const std::string& prefix) { return PythonPrint(type, prefix); },
      nb::arg("type"), nb::arg("prefix") = "pl",
      "Print Type object in Python IR syntax.\n\n"
      "Args:\n"
      "    type: Type to print\n"
      "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')");

  // operator functions for Var (wrapped in Python for span capture and normalization)
  // Using standalone C++ API functions from scalar_expr.h
  // Note: first parameter (self) is implicit when binding as method
  ir.def("add", &MakeAdd, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Addition operator");
  ir.def("sub", &MakeSub, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Subtraction operator");
  ir.def("mul", &MakeMul, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Multiplication operator");
  ir.def("truediv", &MakeFloatDiv, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "True division operator");
  ir.def("floordiv", &MakeFloorDiv, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Floor division operator");
  ir.def("mod", &MakeFloorMod, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Modulo operator");
  ir.def("pow", &MakePow, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Power operator");
  ir.def("eq", &MakeEq, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Equality operator");
  ir.def("ne", &MakeNe, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Inequality operator");
  ir.def("lt", &MakeLt, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Less than operator");
  ir.def("le", &MakeLe, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Less than or equal operator");
  ir.def("gt", &MakeGt, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Greater than operator");
  ir.def("ge", &MakeGe, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Greater than or equal operator");
  ir.def("neg", &MakeNeg, nb::arg("operand"), nb::arg("span") = Span::unknown(), "Negation operator");
  ir.def("cast", &MakeCast, nb::arg("operand"), nb::arg("dtype"), nb::arg("span") = Span::unknown(),
         "Cast operator");
  ir.def("bit_and", &MakeBitAnd, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Bitwise and operator");
  ir.def("bit_or", &MakeBitOr, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Bitwise or operator");
  ir.def("bit_xor", &MakeBitXor, nb::arg("lhs"), nb::arg("rhs"), nb::arg("span") = Span::unknown(),
         "Bitwise xor operator");
  ir.def("bit_shift_left", &MakeBitShiftLeft, nb::arg("lhs"), nb::arg("rhs"),
         nb::arg("span") = Span::unknown(), "Bitwise left shift operator");
  ir.def("bit_shift_right", &MakeBitShiftRight, nb::arg("lhs"), nb::arg("rhs"),
         nb::arg("span") = Span::unknown(), "Bitwise right shift operator");
  ir.def("bit_not", &MakeBitNot, nb::arg("operand"), nb::arg("span") = Span::unknown(),
         "Bitwise not operator");

  // ParentStmtAnalysis - utility class for analyzing statement parent relationships
  auto parent_stmt_analysis_class = nb::class_<ParentStmtAnalysis>(
      ir, "ParentStmtAnalysis",
      "Utility class for analyzing parent-child relationships in statement trees.\n\n"
      "This class builds a mapping from each statement to its parent statement within\n"
      "a function's body. It is useful for passes that need to traverse upward in the\n"
      "IR tree or understand the context of a statement.\n\n"
      "Example usage:\n"
      "    analysis = ir.ParentStmtAnalysis()\n"
      "    analysis.build_map(function)\n"
      "    parent = analysis.get_parent(some_stmt)\n"
      "    if parent:\n"
      "        # Use parent statement\n\n"
      "Note: The analysis becomes invalid after IR transformations. Call build_map again\n"
      "if the IR tree is modified.");

  parent_stmt_analysis_class.def(nb::init<>(), "Create a ParentStmtAnalysis instance");

  parent_stmt_analysis_class.def(
      "build_map", &ParentStmtAnalysis::BuildMap, nb::arg("func"),
      "Build the parent mapping from a function's body.\n\n"
      "Traverses the function's statement tree and records parent-child relationships.\n"
      "This method clears any existing mapping before building the new one.\n\n"
      "Args:\n"
      "    func: The function to analyze (can be None, resulting in empty map)\n\n"
      "Parent relationships established:\n"
      "- For SeqStmts/OpStmts: Each child statement's parent is the SeqStmts/OpStmts\n"
      "- For IfStmt: then_body and else_body (if present) have IfStmt as parent\n"
      "- For ForStmt: body has ForStmt as parent\n"
      "- Root statement (function.body) has no parent");

  parent_stmt_analysis_class.def("get_parent", &ParentStmtAnalysis::GetParent, nb::arg("stmt"),
                                 "Get the parent statement of a given statement.\n\n"
                                 "Args:\n"
                                 "    stmt: The statement to query\n\n"
                                 "Returns:\n"
                                 "    Parent statement, or None if:\n"
                                 "    - stmt is the root statement (function body)\n"
                                 "    - stmt is not found in the analyzed tree\n"
                                 "    - stmt is None");

  parent_stmt_analysis_class.def("has_parent", &ParentStmtAnalysis::HasParent, nb::arg("stmt"),
                                 "Check if a statement has a recorded parent.\n\n"
                                 "Args:\n"
                                 "    stmt: The statement to check\n\n"
                                 "Returns:\n"
                                 "    True if stmt has a parent in the map, False otherwise");

  parent_stmt_analysis_class.def("clear", &ParentStmtAnalysis::Clear,
                                 "Clear the parent mapping.\n\n"
                                 "Removes all recorded parent-child relationships. Useful for reusing\n"
                                 "the same ParentStmtAnalysis instance with different functions.");
}

}  // namespace python
}  // namespace pypto
