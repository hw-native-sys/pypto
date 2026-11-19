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

bool ConversionFloat(DataType dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

TypePtr ConversionType(const std::vector<ExprPtr>& args, const Kwargs& kwargs, const std::string& name,
                       bool input_float, bool output_float) {
  CHECK(args.size() == 1) << name << " requires exactly one argument";
  const auto input = As<ScalarType>(args[0]->GetType());
  CHECK(input) << name << " requires a scalar operand";
  const auto source = input->dtype_;
  auto destination = output_float ? DataType::FP32 : DataType::INT32;
  std::string rounding = output_float ? "r" : "z";
  bool has_dtype = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "dtype") {
      destination = AnyCast<DataType>(value, key);
      has_dtype = true;
    } else if (key == "rounding") {
      rounding = AnyCast<std::string>(value, key);
    } else if ((key == "unsigned" && input_float != output_float) ||
               (key == "saturate" && input_float && output_float)) {
      AnyCast<bool>(value, key);
    } else {
      CHECK(false) << name << " does not accept attribute " << key;
    }
  }
  auto integer = [](DataType dtype) {
    return dtype != DataType::INDEX && dtype.IsInt() && (dtype.GetBit() == 32 || dtype.GetBit() == 64);
  };
  CHECK(input_float ? ConversionFloat(source) : integer(source))
      << name << " requires " << (input_float ? "FP16/BF16/FP32" : "32/64-bit integer") << " input";
  CHECK(output_float ? ConversionFloat(destination) : integer(destination))
      << name << " has an unsupported destination dtype";
  if (input_float && output_float) {
    CHECK(has_dtype && source != destination) << name << " requires a different destination floating format";
  } else {
    CHECK((source.GetBit() != 64 || destination == DataType::FP32) &&
          (destination.GetBit() != 64 || source == DataType::FP32))
        << name << " supports 64-bit integer conversions only with FP32";
  }
  // PTOAS v0.74 accepts round-to-odd only for FP32 -> FP16; no exposed
  // non-packed conversion accepts the hybrid selector. Do not generalize the
  // enum's full set to every source/destination pair.
  const bool ordinary =
      rounding == "r" || rounding == "a" || rounding == "f" || rounding == "c" || rounding == "z";
  CHECK(ordinary || (rounding == "o" && source == DataType::FP32 && destination == DataType::FP16))
      << name << " rounding must be r/a/f/c/z; o is supported only for FP32 to FP16";
  return std::make_shared<ScalarType>(destination);
}
}  // namespace

REGISTER_OP("simt.div")
    .set_description("FP32 division with explicit target precision")
    .set_op_category("SimtOp")
    .add_argument("lhs", "FP32 scalar")
    .add_argument("rhs", "FP32 scalar")
    .set_attr<std::string>("precision")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      for (const auto& [key, attr] : kwargs) {
        CHECK(key == "precision") << "simt.div only accepts precision";
        const auto value = AnyCast<std::string>(attr, key);
        CHECK(value == "intrinsic" || value == "precision_0ulp_ftz_true")
            << "simt.div precision must be intrinsic or precision_0ulp_ftz_true";
      }
      return MathType(args, {}, "simt.div", 2, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.rem")
    .set_description("FP32 remainder with a quotient truncated toward zero")
    .set_op_category("SimtOp")
    .add_argument("lhs", "FP32 scalar")
    .add_argument("rhs", "FP32 scalar")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.rem", 2, DataType::FP32, DataType::FP32, false, false);
    });
REGISTER_OP("simt.mul_i32toi64")
    .set_description("Widened signed or unsigned INT32 product in an INT64 bit carrier")
    .set_op_category("SimtOp")
    .add_argument("lhs", "INT32 scalar")
    .add_argument("rhs", "INT32 scalar")
    .set_attr<bool>("unsigned")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return MathType(args, kwargs, "simt.mul_i32toi64", 2, DataType::INT32, DataType::INT64, false, true);
    });

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
    .set_description("Convert FP16/BF16/FP32 to a saturated 32/64-bit integer with explicit rounding.")
    .set_op_category("SimtOp")
    .add_argument("value", "FP16/BF16/FP32 scalar")
    .set_attr<std::string>("rounding")
    .set_attr<bool>("unsigned")
    .set_attr<DataType>("dtype")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return ConversionType(args, kwargs, "simt.ftoi", true, false);
    });
REGISTER_OP("simt.itof")
    .set_description("Convert a 32/64-bit integer to FP16/BF16/FP32 with explicit rounding.")
    .set_op_category("SimtOp")
    .add_argument("value", "32/64-bit integer scalar")
    .set_attr<std::string>("rounding")
    .set_attr<bool>("unsigned")
    .set_attr<DataType>("dtype")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return ConversionType(args, kwargs, "simt.itof", false, true);
    });
REGISTER_OP("simt.ftof")
    .set_description("Convert FP16/BF16/FP32 with explicit rounding and saturation")
    .set_op_category("SimtOp")
    .add_argument("value", "Floating scalar")
    .set_attr<DataType>("dtype")
    .set_attr<std::string>("rounding")
    .set_attr<bool>("saturate")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const Kwargs& kwargs) {
      return ConversionType(args, kwargs, "simt.ftof", true, true);
    });
}  // namespace pypto::ir
