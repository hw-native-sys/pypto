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

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
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

// Extract a stmt's "leading_comments" field (absent ⇒ empty vector). Symmetric with
// DeserializeSpan — each Stmt deserializer passes the result as the last ctor arg so
// leading_comments is initialized at construction time, not attached after the fact.
//
// A missing field silently defaults to empty (backward compat with older .pto blobs).
// A present field with an unexpected type raises — silently treating malformed data
// as "no comments" would hide serializer/deserializer mismatches.
static std::vector<std::string> DeserializeLeadingComments(const msgpack::object& fields_obj) {
  std::vector<std::string> comments;
  if (fields_obj.type != msgpack::type::MAP) return comments;
  msgpack::object_kv* p = fields_obj.via.map.ptr;
  msgpack::object_kv* const pend = fields_obj.via.map.ptr + fields_obj.via.map.size;
  for (; p < pend; ++p) {
    std::string key;
    p->key.convert(key);
    if (key != "leading_comments") continue;
    CHECK(p->val.type == msgpack::type::ARRAY)
        << "Deserializer: 'leading_comments' must be a string array, got msgpack type "
        << static_cast<int>(p->val.type);
    comments.reserve(p->val.via.array.size);
    for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
      CHECK(p->val.via.array.ptr[i].type == msgpack::type::STR)
          << "Deserializer: 'leading_comments[" << i << "]' must be a string, got msgpack type "
          << static_cast<int>(p->val.via.array.ptr[i].type);
      std::string text;
      p->val.via.array.ptr[i].convert(text);
      comments.push_back(std::move(text));
    }
    break;
  }
  return comments;
}

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

static std::vector<std::pair<std::string, std::any>> DeserializeKwargs(const msgpack::object& kwargs_obj,
                                                                       const std::string& field_name,
                                                                       DeserializerContext& ctx,
                                                                       msgpack::zone& zone) {
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
      // Try to deserialize as DataType or TensorLayout
      std::string type_name;
      std::string value_str;
      msgpack::object value_obj_inner;
      bool has_value_obj = false;
      msgpack::object_kv* const map_p = value_obj.via.map.ptr;
      msgpack::object_kv* const map_pend = value_obj.via.map.ptr + value_obj.via.map.size;
      for (auto* it = map_p; it < map_pend; ++it) {
        std::string field_key;
        it->key.convert(field_key);
        if (field_key == "type") {
          it->val.convert(type_name);
        } else if (field_key == "value") {
          value_obj_inner = it->val;
          has_value_obj = true;
          if (it->val.type == msgpack::type::STR) {
            it->val.convert(value_str);
          }
        }
      }
      if (type_name == "ArgDirectionVector") {
        if (!has_value_obj || value_obj_inner.type != msgpack::type::ARRAY) {
          throw TypeError("ArgDirectionVector kwarg '" + key + "' must have ARRAY value");
        }
        std::vector<ArgDirection> dirs;
        dirs.reserve(value_obj_inner.via.array.size);
        for (uint32_t j = 0; j < value_obj_inner.via.array.size; ++j) {
          uint8_t code = value_obj_inner.via.array.ptr[j].as<uint8_t>();
          if (code > static_cast<uint8_t>(ArgDirection::Scalar)) {
            throw TypeError("Invalid ArgDirection value " + std::to_string(static_cast<int>(code)) +
                            " for kwarg: " + key);
          }
          dirs.push_back(static_cast<ArgDirection>(code));
        }
        kwargs.emplace_back(key, std::move(dirs));
      } else if (type_name == "TensorLayout") {
        if (value_str.empty()) {
          throw TypeError("Missing 'value' field for TensorLayout kwarg: " + key);
        }
        kwargs.emplace_back(key, StringToTensorLayout(value_str));
      } else if (type_name == "TileLayout") {
        if (value_str.empty()) {
          throw TypeError("Missing 'value' field for TileLayout kwarg: " + key);
        }
        kwargs.emplace_back(key, StringToTileLayout(value_str));
      } else if (type_name == "MemorySpace") {
        if (value_str.empty()) {
          throw TypeError("Missing 'value' field for MemorySpace kwarg: " + key);
        }
        kwargs.emplace_back(key, StringToMemorySpace(value_str));
      } else if (type_name == "PadValue") {
        if (value_str.empty()) {
          throw TypeError("Missing 'value' field for PadValue kwarg: " + key);
        }
        if (value_str == "null") {
          kwargs.emplace_back(key, PadValue::null);
        } else if (value_str == "zero") {
          kwargs.emplace_back(key, PadValue::zero);
        } else if (value_str == "max") {
          kwargs.emplace_back(key, PadValue::max);
        } else if (value_str == "min") {
          kwargs.emplace_back(key, PadValue::min);
        } else {
          throw TypeError("Unknown PadValue: " + value_str + " for kwarg: " + key);
        }
      } else if (type_name == "Var") {
        // Reserved Var-valued attr (kAttrTaskIdVar). Resolved through the node
        // table so it round-trips by identity. A missing 'value' field is a
        // malformed envelope (fail fast, like VarList/Int32Vector); an explicit
        // nil value is the null placeholder (the attr held a null VarPtr).
        if (!has_value_obj) {
          throw TypeError("Var kwarg '" + key + "' must have a 'value' field");
        }
        if (value_obj_inner.type == msgpack::type::NIL) {
          kwargs.emplace_back(key, VarPtr(nullptr));
        } else {
          kwargs.emplace_back(
              key, std::static_pointer_cast<const Var>(ctx.DeserializeNode(value_obj_inner, zone)));
        }
      } else if (type_name == "VarList") {
        // Reserved Var-list attrs (kAttrManualDepEdges / kAttrArgDirOverrideVars /
        // kAttrDumpVars). Each entry resolves through the node table; a nil entry
        // reconstructs a null VarPtr so list positions are preserved.
        if (!has_value_obj || value_obj_inner.type != msgpack::type::ARRAY) {
          throw TypeError("VarList kwarg '" + key + "' must have ARRAY value");
        }
        std::vector<VarPtr> vars;
        vars.reserve(value_obj_inner.via.array.size);
        for (uint32_t j = 0; j < value_obj_inner.via.array.size; ++j) {
          const msgpack::object& elem = value_obj_inner.via.array.ptr[j];
          if (elem.type == msgpack::type::NIL) {
            vars.emplace_back(nullptr);
          } else {
            vars.push_back(std::static_pointer_cast<const Var>(ctx.DeserializeNode(elem, zone)));
          }
        }
        kwargs.emplace_back(key, std::move(vars));
      } else if (type_name == "Int32Vector") {
        // kAttrArgDirectionOverrides — no_dep argument indices (std::vector<int32_t>).
        if (!has_value_obj || value_obj_inner.type != msgpack::type::ARRAY) {
          throw TypeError("Int32Vector kwarg '" + key + "' must have ARRAY value");
        }
        std::vector<int32_t> idxs;
        idxs.reserve(value_obj_inner.via.array.size);
        for (uint32_t j = 0; j < value_obj_inner.via.array.size; ++j) {
          idxs.push_back(value_obj_inner.via.array.ptr[j].as<int32_t>());
        }
        kwargs.emplace_back(key, std::move(idxs));
      } else if (type_name == "Expr") {
        // Reserved Expr-valued attr (kAttrDevice). Resolved through the node table.
        // A missing 'value' field is a malformed envelope (fail fast); an
        // explicit nil value is the null placeholder.
        if (!has_value_obj) {
          throw TypeError("Expr kwarg '" + key + "' must have a 'value' field");
        }
        if (value_obj_inner.type == msgpack::type::NIL) {
          kwargs.emplace_back(key, ExprPtr(nullptr));
        } else {
          kwargs.emplace_back(
              key, std::static_pointer_cast<const Expr>(ctx.DeserializeNode(value_obj_inner, zone)));
        }
      } else {
        // Try to deserialize as DataType
        try {
          kwargs.emplace_back(key, DeserializeDataType(value_obj, key));
        } catch (const TypeError&) {
          throw TypeError("Invalid kwarg type for key: " + key);
        }
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
  std::string name_hint = GET_FIELD(std::string, "name_hint");
  return std::make_shared<Var>(name_hint, type, span);
}

// Deserialize IterArg
static IRNodePtr DeserializeIterArg(const msgpack::object& fields_obj, msgpack::zone& zone,
                                    DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  std::string name_hint = GET_FIELD(std::string, "name_hint");
  auto initValue =
      std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("initValue"), zone));
  return std::make_shared<IterArg>(name_hint, type, initValue, span);
}

// Deserialize MemRef
static IRNodePtr DeserializeMemRef(const msgpack::object& fields_obj, msgpack::zone& zone,
                                   DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  std::string name_hint = GET_FIELD(std::string, "name_hint");
  auto byte_offset =
      std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("byte_offset"), zone));
  uint64_t size = GET_FIELD(uint64_t, "size");
  // base_ is a VarPtr, serialized as a full IRNode
  auto base = std::static_pointer_cast<const Var>(ctx.DeserializeNode(GET_FIELD_OBJ("base"), zone));
  INTERNAL_CHECK_SPAN(base, span) << "MemRef base deserialized to null";
  return std::make_shared<MemRef>(name_hint, base, byte_offset, size, span);
}

// Deserialize ConstInt
static IRNodePtr DeserializeConstInt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  int64_t value = GET_FIELD(int64_t, "value");
  auto scalar_type = As<ScalarType>(type);
  INTERNAL_CHECK_SPAN(scalar_type, span)
      << "ConstInt is expected to have ScalarType type, but got " + type->TypeName();
  return std::make_shared<ConstInt>(value, scalar_type->dtype_, span);
}

// Deserialize ConstFloat
static IRNodePtr DeserializeConstFloat(const msgpack::object& fields_obj, msgpack::zone& zone,
                                       DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);
  double value = GET_FIELD(double, "value");
  auto scalar_type = As<ScalarType>(type);
  INTERNAL_CHECK_SPAN(scalar_type, span)
      << "ConstFloat is expected to have ScalarType type, but got " + type->TypeName();
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

  // Deserialize generic attrs map (preserve order using vector). Optional in payloads
  // produced by older serializers that only carried a top-level `arg_directions` field.
  std::vector<std::pair<std::string, std::any>> attrs;
  auto attrs_opt = GetOptionalFieldObj(fields_obj, "attrs", ctx);
  if (attrs_opt.has_value() && attrs_opt->type != msgpack::type::NIL) {
    attrs = DeserializeKwargs(*attrs_opt, "attrs", ctx, zone);
  }

  // Backward compatibility: legacy .pir payloads stored arg_directions as a top-level
  // ARRAY field on Call. Lift it into attrs_["arg_directions"] so consumers continue
  // to work via the new GetArgDirections() API. Absent or NIL means "legacy /
  // not yet derived"; any other msgpack type indicates a malformed payload.
  auto arg_dirs_opt = GetOptionalFieldObj(fields_obj, "arg_directions", ctx);
  if (arg_dirs_opt.has_value() && arg_dirs_opt->type != msgpack::type::NIL) {
    CHECK_SPAN(arg_dirs_opt->type == msgpack::type::ARRAY, span)
        << "Invalid arg_directions field for Call: expected ARRAY, got msgpack type "
        << static_cast<int>(arg_dirs_opt->type);
    std::vector<ArgDirection> arg_directions;
    arg_directions.reserve(arg_dirs_opt->via.array.size);
    for (uint32_t i = 0; i < arg_dirs_opt->via.array.size; ++i) {
      uint8_t code = arg_dirs_opt->via.array.ptr[i].as<uint8_t>();
      CHECK_SPAN(code <= static_cast<uint8_t>(ArgDirection::Scalar), span)
          << "Invalid ArgDirection value: " << static_cast<int>(code);
      arg_directions.push_back(static_cast<ArgDirection>(code));
    }
    if (!arg_directions.empty()) {
      CHECK_SPAN(arg_directions.size() == args.size(), span)
          << "Call arg_directions size (" << arg_directions.size() << ") must match args size ("
          << args.size() << ")";
      attrs = WithArgDirectionsAttr(std::move(attrs), std::move(arg_directions));
    }
  }

  // Deserialize kwargs (preserve order using vector)
  auto kwargs_obj = GET_FIELD_OBJ("kwargs");
  std::vector<std::pair<std::string, std::any>> kwargs = DeserializeKwargs(kwargs_obj, "kwargs", ctx, zone);

  return std::make_shared<Call>(op, args, std::move(kwargs), std::move(attrs), type, span);
}

// Deserialize Submit (pl.submit / pl.spmd_submit task launch). Mirrors
// DeserializeCall plus the first-class deps_ field and the SPMD launch spec
// (core_num / sync_start). Serialization is reflection-driven, so this is the
// only hand-written half (see serializer.cpp SERIALIZE_FIELDS(Submit)).
static IRNodePtr DeserializeSubmit(const msgpack::object& fields_obj, msgpack::zone& zone,
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

  // deps_ is a first-class field (Scalar[TASK_ID] / Array[N, TASK_ID] Vars),
  // serialized as an array of nodes.
  std::vector<ExprPtr> deps;
  auto deps_obj = GET_FIELD_OBJ("deps");
  if (deps_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < deps_obj.via.array.size; ++i) {
      deps.push_back(
          std::static_pointer_cast<const Expr>(ctx.DeserializeNode(deps_obj.via.array.ptr[i], zone)));
    }
  }

  // SPMD launch spec (pl.spmd_submit). core_num is an optional Expr node —
  // NIL / absent means a plain submit (single block). sync_start is a bool
  // (default false), only meaningful when core_num is present.
  std::optional<ExprPtr> core_num;
  auto core_num_opt = GetOptionalFieldObj(fields_obj, "core_num", ctx);
  // GetOptionalFieldObj already maps a NIL payload to nullopt, but guard
  // explicitly so a present-but-NIL field never deserializes to a null ExprPtr
  // (which would trip the Submit ctor's launch-spec validation).
  if (core_num_opt.has_value() && core_num_opt->type != msgpack::type::NIL) {
    core_num = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(*core_num_opt, zone));
  }
  bool sync_start = false;
  auto sync_start_obj = GetOptionalFieldObj(fields_obj, "sync_start", ctx);
  if (sync_start_obj.has_value() && sync_start_obj->type != msgpack::type::NIL) {
    CHECK_SPAN(sync_start_obj->type == msgpack::type::BOOLEAN, span)
        << "Submit sync_start must be a bool, got msgpack type " << static_cast<int>(sync_start_obj->type);
    sync_start = sync_start_obj->as<bool>();
  }
  bool allow_early_resolve = false;
  auto allow_early_resolve_obj = GetOptionalFieldObj(fields_obj, "allow_early_resolve", ctx);
  if (allow_early_resolve_obj.has_value() && allow_early_resolve_obj->type != msgpack::type::NIL) {
    CHECK_SPAN(allow_early_resolve_obj->type == msgpack::type::BOOLEAN, span)
        << "Submit allow_early_resolve must be a bool, got msgpack type "
        << static_cast<int>(allow_early_resolve_obj->type);
    allow_early_resolve = allow_early_resolve_obj->as<bool>();
  }

  // Generic attrs map. Submit never stores manual_dep_edges in attrs (deps_ is
  // the source of truth), so no legacy top-level arg_directions lifting is
  // needed — arg_directions, when present, round-trips through the attrs map.
  std::vector<std::pair<std::string, std::any>> attrs;
  auto attrs_opt = GetOptionalFieldObj(fields_obj, "attrs", ctx);
  if (attrs_opt.has_value() && attrs_opt->type != msgpack::type::NIL) {
    attrs = DeserializeKwargs(*attrs_opt, "attrs", ctx, zone);
  }

  auto kwargs_obj = GET_FIELD_OBJ("kwargs");
  std::vector<std::pair<std::string, std::any>> kwargs = DeserializeKwargs(kwargs_obj, "kwargs", ctx, zone);

  return std::make_shared<Submit>(op, args, deps, std::move(kwargs), std::move(attrs), type, span,
                                  std::move(core_num), sync_start, allow_early_resolve);
}

// Macro for binary expressions
#define DESERIALIZE_BINARY_EXPR(ClassName)                                                                \
  static IRNodePtr Deserialize##ClassName(const msgpack::object& fields_obj, msgpack::zone& zone,         \
                                          DeserializerContext& ctx) {                                     \
    auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));                                               \
    auto type = ctx.DeserializeType(GET_FIELD_OBJ("type"), zone);                                         \
    auto scalar_type = As<ScalarType>(type);                                                              \
    INTERNAL_CHECK_SPAN(scalar_type, span)                                                                \
        << #ClassName " is expected to have ScalarType type, but got " + type->TypeName();                \
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
    INTERNAL_CHECK_SPAN(scalar_type, span)                                                         \
        << #ClassName " is expected to have ScalarType type, but got " + type->TypeName();         \
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
  return std::make_shared<AssignStmt>(var, value, span, DeserializeLeadingComments(fields_obj));
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

  return std::make_shared<IfStmt>(condition, then_body, else_body, return_vars, span,
                                  DeserializeLeadingComments(fields_obj));
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

  return std::make_shared<YieldStmt>(value, span, DeserializeLeadingComments(fields_obj));
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

  return std::make_shared<ReturnStmt>(value, span, DeserializeLeadingComments(fields_obj));
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

  // Deserialize kind with backward compatibility (defaults to Sequential)
  ForKind kind = ForKind::Sequential;
  auto kind_obj = GetOptionalFieldObj(fields_obj, "kind", ctx);
  if (kind_obj.has_value()) {
    kind = static_cast<ForKind>(kind_obj->via.u64);
  }

  // Deserialize attrs. The obsolete legacy loop-origin field is ignored.
  std::vector<std::pair<std::string, std::any>> attrs;
  auto attrs_obj = GetOptionalFieldObj(fields_obj, "attrs", ctx);
  if (attrs_obj.has_value() && attrs_obj->type != msgpack::type::NIL) {
    attrs = DeserializeKwargs(*attrs_obj, "attrs", ctx, zone);
  }

  // Reject a legacy chunked ForStmt: the chunk / auto_chunk loop-chunking feature was removed,
  // so a serialized "chunk_config" field can no longer be honored. Unlike the cosmetic
  // loop-origin tag, chunk_config is semantic — silently dropping it would change the loop's
  // meaning — so fail loudly with a migration hint instead.
  CHECK(!GetOptionalFieldObj(fields_obj, "chunk_config", ctx).has_value())
      << "Cannot deserialize a ForStmt carrying a legacy 'chunk_config' field: the chunk / "
      << "auto_chunk loop-chunking feature was removed. Re-export this IR from a current PyPTO "
      << "version (chunked loops can be expressed with manual tiling).";

  return std::make_shared<ForStmt>(loop_var, start, stop, step, iter_args, body, return_vars, span, kind,
                                   std::move(attrs), DeserializeLeadingComments(fields_obj));
}

// Deserialize WhileStmt
static IRNodePtr DeserializeWhileStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                      DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto condition =
      std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("condition"), zone));

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

  return std::make_shared<WhileStmt>(condition, iter_args, body, return_vars, span,
                                     DeserializeLeadingComments(fields_obj));
}

// Helpers shared across the per-kind ScopeStmt deserializers.
static std::string DeserializeScopeNameHint(const msgpack::object& fields_obj, DeserializerContext& ctx) {
  std::string name_hint;
  auto name_hint_obj = GetOptionalFieldObj(fields_obj, "name_hint", ctx);
  if (name_hint_obj.has_value() && name_hint_obj->type == msgpack::type::STR) {
    name_hint = name_hint_obj->as<std::string>();
  }
  return name_hint;
}

static std::optional<SplitMode> DeserializeScopeSplit(const msgpack::object& fields_obj,
                                                      DeserializerContext& ctx) {
  std::optional<SplitMode> split = std::nullopt;
  auto split_obj = GetOptionalFieldObj(fields_obj, "split", ctx);
  if (split_obj.has_value() && split_obj->type != msgpack::type::NIL) {
    CHECK(split_obj->type == msgpack::type::POSITIVE_INTEGER ||
          split_obj->type == msgpack::type::NEGATIVE_INTEGER)
        << "ScopeStmt split must be an integer SplitMode code, got msgpack type "
        << static_cast<int>(split_obj->type);
    split = static_cast<SplitMode>(split_obj->as<uint64_t>());
  }
  return split;
}

// Deserialize a ScopeStmt's generic attrs_ map (preserve order; absent ⇒ empty).
// ScopeStmt serializes attrs_ via reflection, so every scope kind must read it
// back to survive a round-trip — captured scopes (``with pl.at(...) as tid:`` /
// ``with pl.spmd(...) as tid:`` / ``deps=[...]``) carry the reserved Var-valued
// kAttrTaskIdVar / kAttrManualDepEdges here, which DeserializeKwargs resolves
// through its "Var" / "VarList" branches.
static std::vector<std::pair<std::string, std::any>> DeserializeScopeAttrs(const msgpack::object& fields_obj,
                                                                           DeserializerContext& ctx,
                                                                           msgpack::zone& zone) {
  std::vector<std::pair<std::string, std::any>> attrs;
  auto attrs_opt = GetOptionalFieldObj(fields_obj, "attrs", ctx);
  if (attrs_opt.has_value() && attrs_opt->type != msgpack::type::NIL) {
    attrs = DeserializeKwargs(*attrs_opt, "attrs", ctx, zone);
  }
  return attrs;
}

// Deserialize InCoreScopeStmt
static IRNodePtr DeserializeInCoreScopeStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                            DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto split = DeserializeScopeSplit(fields_obj, ctx);
  auto name_hint = DeserializeScopeNameHint(fields_obj, ctx);
  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));
  return std::make_shared<InCoreScopeStmt>(split, std::move(name_hint), body, span,
                                           DeserializeLeadingComments(fields_obj),
                                           DeserializeScopeAttrs(fields_obj, ctx, zone));
}

// Deserialize ClusterScopeStmt
static IRNodePtr DeserializeClusterScopeStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                             DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto name_hint = DeserializeScopeNameHint(fields_obj, ctx);
  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));
  return std::make_shared<ClusterScopeStmt>(std::move(name_hint), body, span,
                                            DeserializeLeadingComments(fields_obj),
                                            DeserializeScopeAttrs(fields_obj, ctx, zone));
}

// Deserialize HierarchyScopeStmt
static IRNodePtr DeserializeHierarchyScopeStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                               DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  // level is required
  auto level_obj = GET_FIELD_OBJ("level");
  CHECK_SPAN(level_obj.type != msgpack::type::NIL, span) << "HierarchyScopeStmt requires a level";
  Level level = static_cast<Level>(level_obj.via.u64);

  // role is optional
  std::optional<Role> role = std::nullopt;
  auto role_obj = GetOptionalFieldObj(fields_obj, "role", ctx);
  if (role_obj.has_value() && role_obj->type != msgpack::type::NIL) {
    role = static_cast<Role>(role_obj->via.u64);
  }

  auto name_hint = DeserializeScopeNameHint(fields_obj, ctx);
  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));
  return std::make_shared<HierarchyScopeStmt>(level, role, std::move(name_hint), body, span,
                                              DeserializeLeadingComments(fields_obj),
                                              DeserializeScopeAttrs(fields_obj, ctx, zone));
}

// Deserialize SpmdScopeStmt
static IRNodePtr DeserializeSpmdScopeStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                          DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  // core_num is stored as a full Expr node.
  auto core_num = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("core_num"), zone));

  // sync_start is required (bool, defaults to false if missing)
  bool sync_start = false;
  auto sync_start_obj = GetOptionalFieldObj(fields_obj, "sync_start", ctx);
  if (sync_start_obj.has_value() && sync_start_obj->type != msgpack::type::NIL) {
    CHECK_SPAN(sync_start_obj->type == msgpack::type::BOOLEAN, span)
        << "SpmdScopeStmt sync_start must be a bool, got msgpack type "
        << static_cast<int>(sync_start_obj->type);
    sync_start = sync_start_obj->as<bool>();
  }

  auto name_hint = DeserializeScopeNameHint(fields_obj, ctx);
  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));
  return std::make_shared<SpmdScopeStmt>(core_num, sync_start, std::move(name_hint), body, span,
                                         DeserializeLeadingComments(fields_obj),
                                         DeserializeScopeAttrs(fields_obj, ctx, zone));
}

// Deserialize RuntimeScopeStmt (pl.manual_scope MANUAL wrapper / AUTO PTO2_SCOPE
// wrapper added by MaterializeRuntimeScopes). Submit nodes live inside this, so
// serializing a manual_scope program requires it.
static IRNodePtr DeserializeRuntimeScopeStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                             DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  // manual is required (bool, defaults to false = AUTO if missing).
  bool manual = false;
  auto manual_obj = GetOptionalFieldObj(fields_obj, "manual", ctx);
  if (manual_obj.has_value() && manual_obj->type != msgpack::type::NIL) {
    CHECK_SPAN(manual_obj->type == msgpack::type::BOOLEAN, span)
        << "RuntimeScopeStmt manual must be a bool, got msgpack type " << static_cast<int>(manual_obj->type);
    manual = manual_obj->as<bool>();
  }

  auto name_hint = DeserializeScopeNameHint(fields_obj, ctx);
  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));

  return std::make_shared<RuntimeScopeStmt>(manual, std::move(name_hint), body, span,
                                            DeserializeLeadingComments(fields_obj),
                                            DeserializeScopeAttrs(fields_obj, ctx, zone));
}

// Deserialize CommDomainScopeStmt — synthesized by MaterializeCommDomainScopes,
// wraps host_orch use sites of a comm domain's WindowBuffer slots.
static IRNodePtr DeserializeCommDomainScopeStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                                DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));

  std::vector<int64_t> devices;
  auto devices_obj = GetOptionalFieldObj(fields_obj, "devices", ctx);
  if (devices_obj.has_value() && devices_obj->type == msgpack::type::ARRAY) {
    devices.reserve(devices_obj->via.array.size);
    for (uint32_t i = 0; i < devices_obj->via.array.size; ++i) {
      int64_t v = 0;
      devices_obj->via.array.ptr[i].convert(v);
      devices.push_back(v);
    }
  }

  std::vector<WindowBufferPtr> slots;
  auto slots_obj = GetOptionalFieldObj(fields_obj, "slots", ctx);
  if (slots_obj.has_value() && slots_obj->type == msgpack::type::ARRAY) {
    slots.reserve(slots_obj->via.array.size);
    for (uint32_t i = 0; i < slots_obj->via.array.size; ++i) {
      slots.push_back(std::static_pointer_cast<const WindowBuffer>(
          ctx.DeserializeNode(slots_obj->via.array.ptr[i], zone)));
    }
  }

  auto name_hint = DeserializeScopeNameHint(fields_obj, ctx);
  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));

  std::vector<std::pair<std::string, std::any>> attrs;
  auto attrs_opt = GetOptionalFieldObj(fields_obj, "attrs", ctx);
  if (attrs_opt.has_value() && attrs_opt->type != msgpack::type::NIL) {
    attrs = DeserializeKwargs(*attrs_opt, "attrs", ctx, zone);
  }

  return std::make_shared<CommDomainScopeStmt>(std::move(devices), std::move(slots), std::move(name_hint),
                                               body, span, DeserializeLeadingComments(fields_obj),
                                               std::move(attrs));
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

// Deserialize EvalStmt
static IRNodePtr DeserializeEvalStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                     DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto expr = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("expr"), zone));
  return std::make_shared<EvalStmt>(expr, span, DeserializeLeadingComments(fields_obj));
}

// Deserialize BreakStmt
static IRNodePtr DeserializeBreakStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                      DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  return std::make_shared<BreakStmt>(span, DeserializeLeadingComments(fields_obj));
}

// Deserialize ContinueStmt
static IRNodePtr DeserializeContinueStmt(const msgpack::object& fields_obj, msgpack::zone& zone,
                                         DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  return std::make_shared<ContinueStmt>(span, DeserializeLeadingComments(fields_obj));
}

// Deserialize InlineStmt
static IRNodePtr DeserializeInlineStmt(const msgpack::object& fields_obj, msgpack::zone& /*zone*/,
                                       DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  std::string body = GET_FIELD_OBJ("body").as<std::string>();
  auto language = static_cast<InlineLanguage>(GET_FIELD_OBJ("language").as<uint64_t>());
  return std::make_shared<InlineStmt>(std::move(body), language, span,
                                      DeserializeLeadingComments(fields_obj));
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
      auto var = std::static_pointer_cast<const Var>(ctx.DeserializeNode(params_obj.via.array.ptr[i], zone));
      params.push_back(var);
    }
  }

  // Read param_directions; default all to In only when field is absent (backward compatibility)
  std::vector<ParamDirection> param_directions(params.size(), ParamDirection::In);
  auto dirs_opt = GetOptionalFieldObj(fields_obj, "param_directions", ctx);
  if (dirs_opt.has_value()) {
    CHECK_SPAN(dirs_opt->type == msgpack::type::ARRAY, span)
        << "Invalid param_directions type for Function: expected ARRAY";
    CHECK_SPAN(dirs_opt->via.array.size == params.size(), span)
        << "Invalid param_directions size for Function: expected " << params.size() << ", got "
        << dirs_opt->via.array.size;
    for (uint32_t i = 0; i < dirs_opt->via.array.size; ++i) {
      uint8_t code = dirs_opt->via.array.ptr[i].as<uint8_t>();
      CHECK_SPAN(code <= static_cast<uint8_t>(ParamDirection::InOut), span)
          << "Invalid ParamDirection value: " << static_cast<int>(code);
      param_directions[i] = static_cast<ParamDirection>(code);
    }
  }

  std::vector<TypePtr> return_types;
  auto return_types_obj = GET_FIELD_OBJ("return_types");
  if (return_types_obj.type == msgpack::type::ARRAY) {
    for (uint32_t i = 0; i < return_types_obj.via.array.size; ++i) {
      return_types.push_back(ctx.DeserializeType(return_types_obj.via.array.ptr[i], zone));
    }
  }

  // Deserialize optional level
  std::optional<Level> level = std::nullopt;
  auto level_obj = GetOptionalFieldObj(fields_obj, "level", ctx);
  if (level_obj.has_value() && level_obj->type != msgpack::type::NIL) {
    level = static_cast<Level>(level_obj->via.u64);
  }

  // Deserialize optional role
  std::optional<Role> role = std::nullopt;
  auto role_obj = GetOptionalFieldObj(fields_obj, "role", ctx);
  if (role_obj.has_value() && role_obj->type != msgpack::type::NIL) {
    role = static_cast<Role>(role_obj->via.u64);
  }

  // Deserialize function attrs (new format), with backward compat for old "split" field
  std::vector<std::pair<std::string, std::any>> attrs;
  auto attrs_obj = GetOptionalFieldObj(fields_obj, "attrs", ctx);
  if (attrs_obj.has_value() && attrs_obj->type != msgpack::type::NIL) {
    attrs = DeserializeKwargs(*attrs_obj, "attrs", ctx, zone);
  } else {
    // Legacy backward compat: convert old "split" field to attrs
    auto split_obj = GetOptionalFieldObj(fields_obj, "split", ctx);
    if (split_obj.has_value() && split_obj->type != msgpack::type::NIL) {
      int split_val = static_cast<int>(split_obj->via.u64);
      if (split_val != 0) {
        attrs.emplace_back("split", split_val);
      }
    }
  }

  // Deserialize requires_runtime_binding (default false for backward compat with
  // blobs written before abstract SubWorkers existed).
  bool requires_runtime_binding = false;
  auto rrb_obj = GetOptionalFieldObj(fields_obj, "requires_runtime_binding", ctx);
  if (rrb_obj.has_value() && rrb_obj->type != msgpack::type::NIL) {
    requires_runtime_binding = rrb_obj->via.boolean;
  }

  auto body = std::static_pointer_cast<const Stmt>(ctx.DeserializeNode(GET_FIELD_OBJ("body"), zone));

  return std::make_shared<Function>(name, params, param_directions, return_types, body, span, func_type,
                                    level, role, std::move(attrs), requires_runtime_binding);
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

  // Older serialized programs may still carry a top-level ``comm_groups``
  // field; the materialised IR now embeds comm-domain scopes inside each
  // function body, so the field is silently dropped (MaterializeCommDomainScopes will
  // re-derive the equivalent scope chain on the next pass run).

  return std::make_shared<Program>(std::move(functions), name, span);
}

// Deserialize WindowBuffer
static IRNodePtr DeserializeWindowBuffer(const msgpack::object& fields_obj, msgpack::zone& zone,
                                         DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  // ``name_hint`` is the inherited Var field carrying the runtime-unique
  // identifier (mirrors MemRef). The base Ptr Var that this WindowBuffer
  // wraps is reconstructed from the ``base`` field below.
  auto base = std::static_pointer_cast<const Var>(ctx.DeserializeNode(GET_FIELD_OBJ("base"), zone));
  auto size = std::static_pointer_cast<const Expr>(ctx.DeserializeNode(GET_FIELD_OBJ("size"), zone));
  bool load_from_host = GET_FIELD(bool, "load_from_host");
  bool store_to_host = GET_FIELD(bool, "store_to_host");
  return std::make_shared<WindowBuffer>(base, size, load_from_host, store_to_host, span);
}

// Deserialize MakeTuple
static IRNodePtr DeserializeMakeTuple(const msgpack::object& fields_obj, msgpack::zone& zone,
                                      DeserializerContext& ctx) {
  auto span = ctx.DeserializeSpan(GET_FIELD_OBJ("span"));
  auto elements_obj = GET_FIELD_OBJ("elements");
  auto elements_vec = elements_obj.as<std::vector<msgpack::object>>();
  std::vector<ExprPtr> elements;
  elements.reserve(elements_vec.size());
  for (const auto& elem_obj : elements_vec) {
    elements.push_back(std::static_pointer_cast<const Expr>(ctx.DeserializeNode(elem_obj, zone)));
  }
  return std::make_shared<MakeTuple>(std::move(elements), span);
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
static TypeRegistrar _memref_registrar("MemRef", DeserializeMemRef);
static TypeRegistrar _var_registrar("Var", DeserializeVar);
static TypeRegistrar _iter_arg_registrar("IterArg", DeserializeIterArg);
static TypeRegistrar _const_int_registrar("ConstInt", DeserializeConstInt);
static TypeRegistrar _const_float_registrar("ConstFloat", DeserializeConstFloat);
static TypeRegistrar _const_bool_registrar("ConstBool", DeserializeConstBool);
static TypeRegistrar _call_registrar("Call", DeserializeCall);
static TypeRegistrar _submit_registrar("Submit", DeserializeSubmit);

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
static TypeRegistrar _while_stmt_registrar("WhileStmt", DeserializeWhileStmt);
static TypeRegistrar _in_core_scope_stmt_registrar("InCoreScopeStmt", DeserializeInCoreScopeStmt);
static TypeRegistrar _cluster_scope_stmt_registrar("ClusterScopeStmt", DeserializeClusterScopeStmt);
static TypeRegistrar _hierarchy_scope_stmt_registrar("HierarchyScopeStmt", DeserializeHierarchyScopeStmt);
static TypeRegistrar _spmd_scope_stmt_registrar("SpmdScopeStmt", DeserializeSpmdScopeStmt);
static TypeRegistrar _runtime_scope_stmt_registrar("RuntimeScopeStmt", DeserializeRuntimeScopeStmt);
static TypeRegistrar _comm_domain_scope_stmt_registrar("CommDomainScopeStmt", DeserializeCommDomainScopeStmt);
static TypeRegistrar _seq_stmts_registrar("SeqStmts", DeserializeSeqStmts);
static TypeRegistrar _eval_stmt_registrar("EvalStmt", DeserializeEvalStmt);
static TypeRegistrar _break_stmt_registrar("BreakStmt", DeserializeBreakStmt);
static TypeRegistrar _continue_stmt_registrar("ContinueStmt", DeserializeContinueStmt);
static TypeRegistrar _inline_stmt_registrar("InlineStmt", DeserializeInlineStmt);

static TypeRegistrar _function_registrar("Function", DeserializeFunction);
static TypeRegistrar _program_registrar("Program", DeserializeProgram);
static TypeRegistrar _window_buffer_registrar("WindowBuffer", DeserializeWindowBuffer);

static TypeRegistrar _make_tuple_registrar("MakeTuple", DeserializeMakeTuple);
static TypeRegistrar _tuple_get_item_expr_registrar("TupleGetItemExpr", DeserializeTupleGetItemExpr);

}  // namespace serialization
}  // namespace ir
}  // namespace pypto
