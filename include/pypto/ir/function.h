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

#ifndef PYPTO_IR_FUNCTION_H_
#define PYPTO_IR_FUNCTION_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Function type classification
 *
 * Categorizes functions by their execution context and purpose:
 * - Opaque: Unspecified (default)
 * - Orchestration: Runs on host/AICPU for control flow and dependency analysis
 * - InCore: Sub-graph on specific AICore
 */
enum class FunctionType : uint8_t {
  Opaque = 0,         ///< Default: unspecified function type
  Orchestration = 1,  ///< Host/AICPU control and coordination
  InCore = 2          ///< AICore sub-graph execution
};

/**
 * @brief Convert FunctionType to string
 * @param type The function type
 * @return String representation ("Opaque", "Orchestration", or "InCore")
 */
inline std::string FunctionTypeToString(FunctionType type) {
  switch (type) {
    case FunctionType::Opaque:
      return "Opaque";
    case FunctionType::Orchestration:
      return "Orchestration";
    case FunctionType::InCore:
      return "InCore";
  }
  throw pypto::TypeError("Unknown FunctionType");
}

/**
 * @brief Convert string to FunctionType
 * @param str String representation
 * @return FunctionType enum value
 * @throws std::invalid_argument if string is not recognized
 */
inline FunctionType StringToFunctionType(const std::string& str) {
  if (str == "Opaque") {
    return FunctionType::Opaque;
  } else if (str == "Orchestration") {
    return FunctionType::Orchestration;
  } else if (str == "InCore") {
    return FunctionType::InCore;
  } else {
    throw std::invalid_argument("Unknown FunctionType: " + str);
  }
}

/**
 * @brief Function definition
 *
 * Represents a complete function definition with name, parameters, return types, and body.
 * Functions are immutable IR nodes.
 */
class Function : public IRNode {
 public:
  /**
   * @brief Create a function definition
   *
   * @param name Function name
   * @param params Parameter variables
   * @param return_types Return types
   * @param body Function body statement (use SeqStmts for multiple statements)
   * @param span Source location
   * @param type Function type (default: Opaque)
   */
  Function(std::string name, std::vector<VarPtr> params, std::vector<TypePtr> return_types, StmtPtr body,
           Span span, FunctionType type = FunctionType::Opaque)
      : IRNode(std::move(span)),
        name_(std::move(name)),
        params_(std::move(params)),
        return_types_(std::move(return_types)),
        body_(std::move(body)),
        func_type_(type) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Function; }
  [[nodiscard]] std::string TypeName() const override { return "Function"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (params as DEF field, func_type, return_types and body as USUAL
   * fields, name as an IGNORE field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::DefField(&Function::params_, "params"),
                                          reflection::UsualField(&Function::func_type_, "func_type"),
                                          reflection::UsualField(&Function::return_types_, "return_types"),
                                          reflection::UsualField(&Function::body_, "body"),
                                          reflection::IgnoreField(&Function::name_, "name")));
  }

 public:
  std::string name_;                   // Function name
  FunctionType func_type_;             // Function type (orchestration, incore, or opaque)
  std::vector<VarPtr> params_;         // Parameter variables
  std::vector<TypePtr> return_types_;  // Return types
  StmtPtr body_;                       // Function body statement
};

using FunctionPtr = std::shared_ptr<const Function>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_FUNCTION_H_
