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
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
using Kwargs = std::vector<std::pair<std::string, std::any>>;

TypePtr MathType(const std::vector<ExprPtr>& args, const Kwargs& kwargs, const std::string& name,
                 size_t arity, DataType input, DataType output, bool rounding, bool signedness) {
  // Type deduction precedes registry arity checking, including low-level IR calls.
  CHECK(args.size() == arity) << name << " requires exactly " << arity << " arguments, got " << args.size();
  for (const auto& arg : args) {
    auto scalar = As<ScalarType>(arg->GetType());
    CHECK_SPAN(scalar && scalar->dtype_ == input, arg->span_)
        << name << " requires " << input.ToString() << " scalar operands";
  }
  for (const auto& [key, value] : kwargs) {
    if (key == "rounding" && rounding) {
      const auto mode = AnyCast<std::string>(value, key);
      CHECK(mode == "r" || mode == "a" || mode == "f" || mode == "c" || mode == "z")
          << name << " rounding must be r/a/f/c/z, got " << mode;
    } else if (key == "unsigned" && signedness) {
      // INT32 is a bit carrier here; this flag never enables generic unsigned arithmetic.
      AnyCast<bool>(value, key);
    } else {
      CHECK(false) << name << " does not accept attribute " << key;
    }
  }
  return std::make_shared<ScalarType>(output);
}
}  // namespace

REGISTER_OP("simt.abs")
    .set_description("Absolute value; clears the FP32 sign bit.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.abs", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.min")
    .set_description("FP32 minimum with minNum semantics; one NaN yields the numeric operand.")
    .set_op_category("SimtOp")
    .add_argument("lhs", "FP32 scalar")
    .add_argument("rhs", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.min", 2, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.max")
    .set_description("FP32 maximum with maxNum semantics; one NaN yields the numeric operand.")
    .set_op_category("SimtOp")
    .add_argument("lhs", "FP32 scalar")
    .add_argument("rhs", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.max", 2, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.exp")
    .set_description("Compute the FP32 natural exponential using target math.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.exp", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.log")
    .set_description("Compute the FP32 natural logarithm using target math.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.log", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.sqrt")
    .set_description("Compute the FP32 square root using target math.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.sqrt", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.sin")
    .set_description("Compute the FP32 sine in radians using target math.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.sin", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.cos")
    .set_description("Compute the FP32 cosine in radians using target math.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.cos", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.ceil")
    .set_description("Round FP32 toward positive infinity, retaining FP32.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.ceil", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.floor")
    .set_description("Round FP32 toward negative infinity, retaining FP32.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.floor", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.rint")
    .set_description("Round FP32 to nearest integer, ties to even, retaining FP32.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.rint", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.round")
    .set_description("Round FP32 to nearest integer, ties away from zero, retaining FP32.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.round", 1, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.pow")
    .set_description("Compute FP32 power using target math.")
    .set_op_category("SimtOp")
    .add_argument("lhs", "FP32 scalar")
    .add_argument("rhs", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.pow", 2, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.fma")
    .set_description("Compute FP32 lhs * rhs + acc with one final rounding.")
    .set_op_category("SimtOp")
    .add_argument("lhs", "FP32 scalar")
    .add_argument("rhs", "FP32 scalar")
    .add_argument("acc", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.fma", 3, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.prmt")
    .set_description("Permute bytes from two INT32 bit carriers with an INT32 selector.")
    .set_op_category("SimtOp")
    .add_argument("lhs", "INT32 scalar")
    .add_argument("rhs", "INT32 scalar")
    .add_argument("selector", "INT32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.prmt", 3, DataType::INT32, DataType::INT32, false, false);
    });
REGISTER_OP("simt.mulhi")
    .set_description("Return the high 32 bits of the signed or unsigned 64-bit product.")
    .set_op_category("SimtOp")
    .add_argument("lhs", "INT32 scalar")
    .add_argument("rhs", "INT32 scalar")
    .set_attr<bool>("unsigned")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.mulhi", 2, DataType::INT32, DataType::INT32, false, true);
    });
REGISTER_OP("simt.ftoi")
    .set_description("Convert FP32 to a saturated INT32 bit carrier with explicit rounding.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP32 scalar")
    .set_attr<std::string>("rounding")
    .set_attr<bool>("unsigned")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.ftoi", 1, DataType::FP32, DataType::INT32, true, true);
    });
REGISTER_OP("simt.itof")
    .set_description("Convert an INT32 bit carrier to FP32 with explicit rounding.")
    .set_op_category("SimtOp")
    .add_argument("value", "INT32 scalar")
    .set_attr<std::string>("rounding")
    .set_attr<bool>("unsigned")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.itof", 1, DataType::INT32, DataType::FP32, true, true);
    });
}  // namespace pypto::ir
