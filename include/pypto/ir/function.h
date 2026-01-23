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
   */
  Function(std::string name, std::vector<VarPtr> params, std::vector<TypePtr> return_types, StmtPtr body,
           Span span)
      : IRNode(std::move(span)),
        name_(std::move(name)),
        params_(std::move(params)),
        return_types_(std::move(return_types)),
        body_(std::move(body)) {}

  [[nodiscard]] IRNodeKind GetKind() const override { return IRNodeKind::Function; }
  [[nodiscard]] std::string TypeName() const override { return "Function"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (params as DEF field, return_types and body as USUAL fields, name as
   * IGNORE field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::IgnoreField(&Function::name_, "name"),
                                          reflection::DefField(&Function::params_, "params"),
                                          reflection::UsualField(&Function::return_types_, "return_types"),
                                          reflection::UsualField(&Function::body_, "body")));
  }

 public:
  std::string name_;                   // Function name
  std::vector<VarPtr> params_;         // Parameter variables
  std::vector<TypePtr> return_types_;  // Return types
  StmtPtr body_;                       // Function body statement
};

using FunctionPtr = std::shared_ptr<const Function>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_FUNCTION_H_
