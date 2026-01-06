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

#ifndef PYPTO_IR_STMT_H_
#define PYPTO_IR_STMT_H_

#include <memory>
#include <string>
#include <utility>

#include "pypto/ir/core.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for all statements in the IR
 *
 * Statements represent operations that perform side effects or control flow.
 * All statements are immutable.
 */
class Stmt : public IRNode {
 public:
  /**
   * @brief Create a statement
   *
   * @param span Source location
   */
  explicit Stmt(Span s) : IRNode(std::move(s)) {}
  ~Stmt() override = default;

  /**
   * @brief Get the type name of this statement
   *
   * @return Human-readable type name (e.g., "Stmt", "Assign", "Return")
   */
  [[nodiscard]] std::string TypeName() const override { return "Stmt"; }

  static constexpr auto GetFieldDescriptors() { return IRNode::GetFieldDescriptors(); }
};

using StmtPtr = std::shared_ptr<const Stmt>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_STMT_H_
