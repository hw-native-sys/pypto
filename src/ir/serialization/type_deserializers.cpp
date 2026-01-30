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

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/serialization/type_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace serialization {

// Use alias for cleaner code
using DeserializerContext = serialization::detail::DeserializerContext;

// Helper macros for deserializing fields
#define GET_FIELD(Type, name) ctx.GetField<Type>(fields_obj, name)
#define GET_FIELD_OBJ(name) ctx.GetFieldObj(fields_obj, name)

// Helper function to get optional field (returns nullopt if field doesn't exist or is null)
static std::optional<msgpack::object> GetOptionalFieldObj(const msgpack::object& fields_obj,
                                                          const std::string& field_name,
                                                          DeserializerContext& ctx) {
  if (fields_obj.type != msgpack::type::MAP) {
    return std::nullopt;
  }
  msgpack::object_kv* p = fields_obj.via.map.ptr;
  msgpack::object_kv* const pend = fields_obj.via.map.ptr + fields_obj.via.map.size;
  for (; p < pend; ++p) {
    std::string key;
    p->key.convert(key);
    if (key == field_name) {
      auto obj = p->val;
      // Check if it's null or empty
      if (obj.type == msgpack::type::NIL) {
        return std::nullopt;
      }
      return obj;
    }
  }
  return std::nullopt;
}

DataType DeserializeDataType(const msgpack::object& fields_obj, const std::string& field_name) {
  msgpack::object_kv* map_p = fields_obj.via.map.ptr;
  msgpack::object_kv* const map_pend = fields_obj.via.map.ptr + fields_obj.via.map.size;
  std::string type_name;
  bool is_dtype = false;
  uint8_t dtype_code = 0;

  for (; map_p < map_pend; ++map_p) {
    std::string field_name;
    map_p->key.convert(field_name);
    if (field_name == "type") {
      map_p->val.convert(type_name);
      is_dtype = (type_name == "DataType");
    } else if (field_name == "code") {
      dtype_code = map_p->val.as<uint8_t>();
    }
  }

  if (is_dtype) {
    return DataType(dtype_code);
  } else {
    throw TypeError("Invalid kwarg MAP type for key: " + field_name);
  }
}

std::vector<std::pair<std::string, std::any>> DeserializeKwargs(const msgpack::object& kwargs_obj,
                                                                const std::string& field_name) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  if (kwargs_obj.type != msgpack::type::ARRAY) {
    throw TypeError("Invalid kwargs type for field: " + field_name);
  }

  for (uint32_t i = 0; i < kwargs_obj.via.array.size; ++i) {
    const msgpack::object& pair_obj = kwargs_obj.via.array.ptr[i];
    if (pair_obj.type != msgpack::type::MAP) {
      throw TypeError("Invalid kwarg pair type for field: " + field_name);
    }

    std::string key;
    msgpack::object value_obj;
    bool has_key = false;
    bool has_value = false;
    msgpack::object_kv* map_p = pair_obj.via.map.ptr;
    msgpack::object_kv* const map_pend = pair_obj.via.map.ptr + pair_obj.via.map.size;
    for (; map_p < map_pend; ++map_p) {
      std::string map_key;
      map_p->key.convert(map_key);
      if (map_key == "key") {
        map_p->val.convert(key);
        has_key = true;
      } else if (map_key == "value") {
        value_obj = map_p->val;
        has_value = true;
      }
    }

    if (!has_key || !has_value) {
      throw TypeError("Invalid kwarg pair for field: " + field_name);
    }

    // Deserialize value based on type
    if (value_obj.type == msgpack::type::BOOLEAN) {
      kwargs.emplace_back(key, value_obj.as<bool>());
    } else if (value_obj.type == msgpack::type::POSITIVE_INTEGER ||
               value_obj.type == msgpack::type::NEGATIVE_INTEGER) {
      kwargs.emplace_back(key, value_obj.as<int>());
    } else if (value_obj.type == msgpack::type::FLOAT32) {
      kwargs.emplace_back(key, value_obj.as<float>());
    } else if (value_obj.type == msgpack::type::FLOAT64) {
      kwargs.emplace_back(key, value_obj.as<double>());
    } else if (value_obj.type == msgpack::type::STR) {
      kwargs.emplace_back(key, value_obj.as<std::string>());
    } else if (value_obj.type == msgpack::type::MAP) {
      // Try to deserialize as DataType
      try {
        kwargs.emplace_back(key, DeserializeDataType(value_obj, key));
      } catch (const TypeError&) {
        throw TypeError("Invalid kwarg type for key: " + key);
      }
    } else {
      throw TypeError("Invalid kwarg type for key: " + key);
    }
  }

  return kwargs;
}

// Deserialize Var
static IRNodePtr DeserializeVar(const msgpack::object& fields_obj, msgpack::zone& zone,
                                DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  std::string name = GET_FIELD(std::string, "name");
  return std::make_shared<Var>(name, type, span);
}

// Deserialize IterArg
static IRNodePtr DeserializeIterArg(const msgpack::object& fields_obj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  std::string name = GET_FIELD(std::string, "name");
  auto initValue =
      std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("initValue"), zone));
  return std::make_shared<IterArg>(name, type, initValue, span);
}

// Deserialize ConstInt
static IRNodePtr DeserializeConstInt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  int64_t value = GET_FIELD(int64_t, "value");
  auto scalar_type = As<ScalarType>(type);
  INTERNAL_CHECK(scalar_type) << "ConstInt is expected to have ScalarType type, but got " + type->TypeName();
  return std::make_shared<ConstInt>(value, scalar_type->dtype_, span);
}

// Deserialize ConstFloat
static IRNodePtr DeserializeConstFloat(const msgpack::object& fields_obj, msgpack::zone& zone,
                                       DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  double value = GET_FIELD(double, "value");
  auto scalar_type = As<ScalarType>(type);
  INTERNAL_CHECK(scalar_type) << "ConstFloat is expected to have ScalarType type, but got " +
                                     type->TypeName();
  return std::make_shared<ConstFloat>(value, scalar_type->dtype_, span);
}

// Deserialize ConstBool
static IRNodePtr DeserializeConstBool(const msgpack::object& fields_obj, msgpack::zone& zone,
                                      DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  bool value = GET_FIELD(bool, "value");
  return std::make_shared<ConstBool>(value, span);
}

// Deserialize Call
static IRNodePtr DeserializeCall(const msgpack::object& fields_obj, msgpack::zone& zone,
                                 DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto op = ctx.DeserializeOp(GET_FIELD_OBJ("op"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);

  std::vector<ExprPtr> args;
  auto args_obj = GET_FIELD_OBJ("args");
  if (args_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < args_obj.via.array.size; ++i) {
      args.push_back(
          std::static_pointer_cast<const Expr>(ctx.DeserializeNode(args_obj.via.array.ptr[i], zone)));
    }
  }

  // Deserialize kwargs (preserve order using vector)
  auto kwargs_obj = GET_FIELD_OBJ("kwargs");
  std::vector<std::pair<std::string, std::any>> kwargs = DeserializeKwargs(kwargs_obj, "kwargs");

  return std::make_shared<Call>(op, args, kwargs, type, span);
}

// Macro for binary expressions
#define DESERIALIZE_BINARY_EXPR(ClassName)                                                                \
  static IRNodePtr Deserialize##ClassName(const msgpack::object& fields_obj, msgpack::zone& zone,         \
                                          DeserializerContext& ctx) {                                     \
    auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));                                               \
    auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);                                         \
    auto scalar_type = As<ScalarType>(type);                                                              \
    INTERNAL_CHECK(scalar_type) << #ClassName " is expected to have ScalarType type, but got " +          \
                                       type->TypeName();                                                  \
    auto left = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("left"), zone));   \
    auto right = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("right"), zone)); \
    return std::make_shared<ClassName>(left, right, scalar_type->dtype_, span);                           \
  }

DESERIALIZE_BINARY_EXPR(Add)
DESERIALIZE_BINARY_EXPR(Sub)
DESERIALIZE_BINARY_EXPR(Mul)
DESERIALIZE_BINARY_EXPR(FloorDiv)
DESERIALIZE_BINARY_EXPR(FloorMod)
DESERIALIZE_BINARY_EXPR(FloatDiv)
DESERIALIZE_BINARY_EXPR(Min)
DESERIALIZE_BINARY_EXPR(Max)
DESERIALIZE_BINARY_EXPR(Pow)
DESERIALIZE_BINARY_EXPR(Eq)
DESERIALIZE_BINARY_EXPR(Ne)
DESERIALIZE_BINARY_EXPR(Lt)
DESERIALIZE_BINARY_EXPR(Le)
DESERIALIZE_BINARY_EXPR(Gt)
DESERIALIZE_BINARY_EXPR(Ge)
DESERIALIZE_BINARY_EXPR(And)
DESERIALIZE_BINARY_EXPR(Or)
DESERIALIZE_BINARY_EXPR(Xor)
DESERIALIZE_BINARY_EXPR(BitAnd)
DESERIALIZE_BINARY_EXPR(BitOr)
DESERIALIZE_BINARY_EXPR(BitXor)
DESERIALIZE_BINARY_EXPR(BitShiftLeft)
DESERIALIZE_BINARY_EXPR(BitShiftRight)

// Macro for unary expressions
#define DESERIALIZE_UNARY_EXPR(ClassName)                                                          \
  static IRNodePtr Deserialize##ClassName(const msgpack::object& fields_obj, msgpack::zone& zone,  \
                                          DeserializerContext& ctx) {                              \
    auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));                                        \
    auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);                                  \
    auto scalar_type = As<ScalarType>(type);                                                       \
    INTERNAL_CHECK(scalar_type) << #ClassName " is expected to have ScalarType type, but got " +   \
                                       type->TypeName();                                           \
    auto operand =                                                                                 \
        std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("operand"), zone)); \
    return std::make_shared<ClassName>(operand, scalar_type->dtype_, span);                        \
  }

DESERIALIZE_UNARY_EXPR(Abs)
DESERIALIZE_UNARY_EXPR(Neg)
DESERIALIZE_UNARY_EXPR(Not)
DESERIALIZE_UNARY_EXPR(BitNot)
DESERIALIZE_UNARY_EXPR(Cast)

// Deserialize AssignStmt
static IRNodePtr DeserializeAssignStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                       DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto var = std::static_pointer_cast<const Var>(ctx.DeserializeNode(GET_FIELD_OBJ("var"), zone));
  auto value = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("value"), zone));
  return std::make_shared<AssignStmt>(var, value, span);
}

// Deserialize IfStmt
static IRNodePtr DeserializeIfStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                   DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto condition =
      std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("condition"), zone));

  // Deserialize then_body as single StmtPtr
  auto then_body =
      std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("then_body"), zone));

  // Deserialize else_body as optional StmtPtr
  std::optional<StmtPtr> else_body;
  auto else_obj_opt = GetOptionalFieldObj(fields_obj, "else_body", ctx);
  if (else_obj_opt.has_value()) {
    else_body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(*else_obj_opt, zone));
  }

  std::vector<VarPtr> return_vars;
  auto return_vars_obj = GET_FIELD_OBJ("return_vars");
  if (return_vars_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < return_vars_obj.via.array.size; ++i) {
      return_vars.push_back(
          std::static_pointer_cast<const Var>(ctx.DeserializeNode(return_vars_obj.via.array.ptr[i], zone)));
    }
  }

  return std::make_shared<IfStmt>(condition, then_body, else_body, return_vars, span);
}

// Deserialize YieldStmt
static IRNodePtr DeserializeYieldStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                      DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  std::vector<ExprPtr> value;
  auto value_obj = GET_FIELD_OBJ("value");
  if (value_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < value_obj.via.array.size; ++i) {
      value.push_back(
          std::static_pointer_cast<const Expr>(ctx.DeserializeNode(value_obj.via.array.ptr[i], zone)));
    }
  }

  return std::make_shared<YieldStmt>(value, span);
}

// Deserialize ReturnStmt
static IRNodePtr DeserializeReturnStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                       DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  std::vector<ExprPtr> value;
  auto value_obj = GET_FIELD_OBJ("value");
  if (value_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < value_obj.via.array.size; ++i) {
      value.push_back(
          std::static_pointer_cast<const Expr>(ctx.DeserializeNode(value_obj.via.array.ptr[i], zone)));
    }
  }

  return std::make_shared<ReturnStmt>(value, span);
}

// Deserialize ForStmt
static IRNodePtr DeserializeForStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto loop_var = std::static_pointer_cast<const Var>(ctx.DeserializeNode(GET_FIELD_OBJ("loop_var"), zone));
  auto start = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("start"), zone));
  auto stop = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("stop"), zone));
  auto step = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("step"), zone));

  std::vector<IterArgPtr> iter_args;
  auto iter_args_obj = GET_FIELD_OBJ("iter_args");
  if (iter_args_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < iter_args_obj.via.array.size; ++i) {
      iter_args.push_back(
          std::static_pointer_cast<const IterArg>(ctx.DeserializeNode(iter_args_obj.via.array.ptr[i], zone)));
    }
  }

  // Deserialize body as single StmtPtr
  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));

  std::vector<VarPtr> return_vars;
  auto return_vars_obj = GET_FIELD_OBJ("return_vars");
  if (return_vars_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < return_vars_obj.via.array.size; ++i) {
      return_vars.push_back(
          std::static_pointer_cast<const Var>(ctx.DeserializeNode(return_vars_obj.via.array.ptr[i], zone)));
    }
  }

  return std::make_shared<ForStmt>(loop_var, start, stop, step, iter_args, body, return_vars, span);
}

// Deserialize SeqStmts
static IRNodePtr DeserializeSeqStmts(const msgpack::object& fields_obj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  std::vector<StmtPtr> stmts;
  auto stmts_obj = GET_FIELD_OBJ("stmts");
  if (stmts_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < stmts_obj.via.array.size; ++i) {
      stmts.push_back(
          std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(stmts_obj.via.array.ptr[i], zone)));
    }
  }

  return std::make_shared<SeqStmts>(stmts, span);
}

// Deserialize OpStmts
static IRNodePtr DeserializeOpStmts(const msgpack::object& fields_obj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  std::vector<AssignStmtPtr> stmts;
  auto stmts_obj = GET_FIELD_OBJ("stmts");
  if (stmts_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < stmts_obj.via.array.size; ++i) {
      stmts.push_back(
          std::static_pointer_cast<const AssignStmt>(ctx.DeserializeNode(stmts_obj.via.array.ptr[i], zone)));
    }
  }

  return std::make_shared<OpStmts>(stmts, span);
}

// Deserialize EvalStmt
static IRNodePtr DeserializeEvalStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto expr = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("expr"), zone));
  return std::make_shared<EvalStmt>(expr, span);
}

// Deserialize Function
static IRNodePtr DeserializeFunction(const msgpack::object& fields_obj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  std::string name = GET_FIELD(std::string, "name");

  // Deserialize func_type field (default to Opaque for backward compatibility)
  FunctionType func_type = FunctionType::Opaque;
  try {
    uint8_t type_code = GET_FIELD(uint8_t, "func_type");
    func_type = static_cast<FunctionType>(type_code);
  } catch (...) {
    // Field doesn't exist in old serialized data, use default
    func_type = FunctionType::Opaque;
  }

  std::vector<VarPtr> params;
  auto params_obj = GET_FIELD_OBJ("params");
  if (params_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < params_obj.via.array.size; ++i) {
      params.push_back(
          std::static_pointer_cast<const Var>(ctx.DeserializeNode(params_obj.via.array.ptr[i], zone)));
    }
  }

  std::vector<TypePtr> return_types;
  auto return_types_obj = GET_FIELD_OBJ("return_types");
  if (return_types_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < return_types_obj.via.array.size; ++i) {
      return_types.push_back(ctx.DeserializeType(return_types_obj.via.array.ptr[i], zone));
    }
  }

  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));

  return std::make_shared<Function>(name, params, return_types, body, span, func_type);
}

// Deserialize Program
static IRNodePtr DeserializeProgram(const msgpack::object& fields_obj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  std::string name = GET_FIELD(std::string, "name");

  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions;
  auto functions_obj = GET_FIELD_OBJ("functions");
  if (functions_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < functions_obj.via.array.size; ++i) {
      auto entry_obj = functions_obj.via.array.ptr[i];
      if (entry_obj.type == msgpack::type::MAP) {
        msgpack::object key_obj, value_obj;
        bool has_key = false, has_value = false;

        msgpack::object_kv* p = entry_obj.via.map.ptr;
        msgpack::object_kv* const pend = entry_obj.via.map.ptr + entry_obj.via.map.size;
        for (; p < pend; ++p) {
          std::string key;
          p->key.convert(key);
          if (key == "key") {
            key_obj = p->val;
            has_key = true;
          } else if (key == "value") {
            value_obj = p->val;
            has_value = true;
          }
        }

        if (has_key && has_value) {
          auto global_var = std::static_pointer_cast<const GlobalVar>(ctx.DeserializeOp(key_obj));
          auto function = std::static_pointer_cast<const Function>(ctx.DeserializeNode(value_obj, zone));
          functions[global_var] = function;
        }
      }
    }
  }

  return std::make_shared<Program>(functions, name, span);
}

// Deserialize TupleGetItemExpr
static IRNodePtr DeserializeTupleGetItemExpr(const msgpack::object& fields_obj, msgpack::zone& zone,
                                             DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto tuple = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("tuple"), zone));
  int index = GET_FIELD(int, "index");
  return std::make_shared<TupleGetItemExpr>(tuple, index, span);
}

// Register all types with the registry
static TypeRegistrar _var_registrar("Var", DeserializeVar);
static TypeRegistrar _iter_arg_registrar("IterArg", DeserializeIterArg);
static TypeRegistrar _const_int_registrar("ConstInt", DeserializeConstInt);
static TypeRegistrar _const_float_registrar("ConstFloat", DeserializeConstFloat);
static TypeRegistrar _const_bool_registrar("ConstBool", DeserializeConstBool);
static TypeRegistrar _call_registrar("Call", DeserializeCall);

static TypeRegistrar _add_registrar("Add", DeserializeAdd);
static TypeRegistrar _sub_registrar("Sub", DeserializeSub);
static TypeRegistrar _mul_registrar("Mul", DeserializeMul);
static TypeRegistrar _floor_div_registrar("FloorDiv", DeserializeFloorDiv);
static TypeRegistrar _floor_mod_registrar("FloorMod", DeserializeFloorMod);
static TypeRegistrar _float_div_registrar("FloatDiv", DeserializeFloatDiv);
static TypeRegistrar _min_registrar("Min", DeserializeMin);
static TypeRegistrar _max_registrar("Max", DeserializeMax);
static TypeRegistrar _pow_registrar("Pow", DeserializePow);
static TypeRegistrar _eq_registrar("Eq", DeserializeEq);
static TypeRegistrar _ne_registrar("Ne", DeserializeNe);
static TypeRegistrar _lt_registrar("Lt", DeserializeLt);
static TypeRegistrar _le_registrar("Le", DeserializeLe);
static TypeRegistrar _gt_registrar("Gt", DeserializeGt);
static TypeRegistrar _ge_registrar("Ge", DeserializeGe);
static TypeRegistrar _and_registrar("And", DeserializeAnd);
static TypeRegistrar _or_registrar("Or", DeserializeOr);
static TypeRegistrar _xor_registrar("Xor", DeserializeXor);
static TypeRegistrar _bit_and_registrar("BitAnd", DeserializeBitAnd);
static TypeRegistrar _bit_or_registrar("BitOr", DeserializeBitOr);
static TypeRegistrar _bit_xor_registrar("BitXor", DeserializeBitXor);
static TypeRegistrar _bit_shift_left_registrar("BitShiftLeft", DeserializeBitShiftLeft);
static TypeRegistrar _bit_shift_right_registrar("BitShiftRight", DeserializeBitShiftRight);

static TypeRegistrar _abs_registrar("Abs", DeserializeAbs);
static TypeRegistrar _neg_registrar("Neg", DeserializeNeg);
static TypeRegistrar _not_registrar("Not", DeserializeNot);
static TypeRegistrar _bit_not_registrar("BitNot", DeserializeBitNot);
static TypeRegistrar _cast_registrar("Cast", DeserializeCast);

static TypeRegistrar _assign_stmt_registrar("AssignStmt", DeserializeAssignStmt);
static TypeRegistrar _if_stmt_registrar("IfStmt", DeserializeIfStmt);
static TypeRegistrar _yield_stmt_registrar("YieldStmt", DeserializeYieldStmt);
static TypeRegistrar _return_stmt_registrar("ReturnStmt", DeserializeReturnStmt);
static TypeRegistrar _for_stmt_registrar("ForStmt", DeserializeForStmt);
static TypeRegistrar _seq_stmts_registrar("SeqStmts", DeserializeSeqStmts);
static TypeRegistrar _op_stmts_registrar("OpStmts", DeserializeOpStmts);
static TypeRegistrar _eval_stmt_registrar("EvalStmt", DeserializeEvalStmt);

static TypeRegistrar _function_registrar("Function", DeserializeFunction);
static TypeRegistrar _program_registrar("Program", DeserializeProgram);

static TypeRegistrar _tuple_get_item_expr_registrar("TupleGetItemExpr", DeserializeTupleGetItemExpr);

}  // namespace serialization
}  // namespace ir
}  // namespace pypto
