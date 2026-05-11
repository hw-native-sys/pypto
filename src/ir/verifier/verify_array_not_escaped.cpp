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

/**
 * @file verify_array_not_escaped.cpp
 * @brief Verifier that ArrayType never crosses function boundaries.
 *
 * ArrayType represents an on-core scalar register file / C-stack array. Its
 * storage is owned by the enclosing function — passing it across a function
 * boundary would leak a stack pointer (or require copy semantics PyPTO does
 * not yet implement). v1 therefore forbids ArrayType from appearing as any
 * function parameter or return type. Local create / get_element /
 * update_element inside a function/region remains fine.
 */

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

class ArrayNotEscapedVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "ArrayNotEscaped"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;
      CheckFunction(func, diagnostics);
    }
  }

 private:
  static void CheckFunction(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
    // Parameters
    for (size_t i = 0; i < func->params_.size(); ++i) {
      const auto& param = func->params_[i];
      if (!param) continue;
      if (TypeContainsArray(param->GetType())) {
        std::ostringstream msg;
        msg << "Function '" << func->name_ << "' parameter #" << i << " ('" << param->name_hint_
            << "') has type containing ArrayType. ArrayType lives on the on-core scalar "
               "register file / C stack and cannot cross function boundaries — create it "
               "inside the function body and use it locally instead.";
        diagnostics.emplace_back(DiagnosticSeverity::Error, "ArrayNotEscaped",
                                 /*error_code=*/0, msg.str(), param->span_);
      }
    }

    // Return types
    for (size_t i = 0; i < func->return_types_.size(); ++i) {
      if (TypeContainsArray(func->return_types_[i])) {
        std::ostringstream msg;
        msg << "Function '" << func->name_ << "' return type #" << i
            << " contains ArrayType. ArrayType lives on the on-core scalar register file / "
               "C stack and cannot cross function boundaries — restructure to return a "
               "Tensor / Scalar instead, or copy the contents into an output tensor before "
               "returning.";
        diagnostics.emplace_back(DiagnosticSeverity::Error, "ArrayNotEscaped",
                                 /*error_code=*/0, msg.str(), func->span_);
      }
    }
  }

  /// Returns true if the type is ArrayType, or a TupleType transitively containing ArrayType.
  /// We don't have to look inside Tensor / Tile / ScalarType because those can't carry an
  /// ArrayType element.
  static bool TypeContainsArray(const TypePtr& type) {
    if (!type) return false;
    if (IsA<ArrayType>(type)) return true;
    if (auto tup = As<TupleType>(type)) {
      for (const auto& t : tup->types_) {
        if (TypeContainsArray(t)) return true;
      }
    }
    return false;
  }
};

}  // namespace

PropertyVerifierPtr CreateArrayNotEscapedPropertyVerifier() {
  return std::make_shared<ArrayNotEscapedVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
