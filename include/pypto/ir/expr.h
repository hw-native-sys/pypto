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

#ifndef PYPTO_IR_EXPR_H_
#define PYPTO_IR_EXPR_H_

#include <memory>
#include <utility>

#include "pypto/ir/core.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for all expressions in the IR
 *
 * This is the root base class for all expression types (scalar, tensor, etc).
 * Expressions represent computations that produce values.
 * All expressions are immutable.
 */
class Expr : public IRNode {
 public:
  /**
   * @brief Create an expression
   *
   * @param span Source location
   */
  explicit Expr(Span s) : IRNode(std::move(s)) {}
  ~Expr() override = default;

  /**
   * @brief Get the type name of this expression
   *
   * @return Human-readable type name (e.g., "ScalarExpr", "TensorExpr")
   */
  [[nodiscard]] virtual const char* type_name() const { return "Expr"; }

  static constexpr auto GetFieldDescriptors() { return IRNode::GetFieldDescriptors(); }
};

using ExprPtr = std::shared_ptr<const Expr>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_EXPR_H_
