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

#ifndef PYPTO_IR_TENSOR_EXPR_H_
#define PYPTO_IR_TENSOR_EXPR_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for tensor expressions in the IR
 *
 * Tensor expressions represent computations that produce tensor values.
 * All expressions are immutable.
 */
class TensorExpr : public Expr {
 public:
  DataType dtype_;                    // Element data type
  std::vector<ScalarExprPtr> shape_;  // Shape dimensions (symbolic or constant)

  /**
   * @brief Create a tensor expression
   *
   * @param span Source location
   * @param dtype Element data type
   * @param shape Shape dimensions
   */
  TensorExpr(Span s, DataType dtype, std::vector<ScalarExprPtr> shape)
      : Expr(std::move(s)), dtype_(dtype), shape_(std::move(shape)) {}
  ~TensorExpr() override = default;

  /**
   * @brief Get the type name of this expression
   *
   * @return Human-readable type name (e.g., "TensorAdd", "TensorVar")
   */
  [[nodiscard]] const char* type_name() const override { return "TensorExpr"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TensorExpr::dtype_, "dtype"),
                                          reflection::UsualField(&TensorExpr::shape_, "shape")));
  }
};

using TensorExprPtr = std::shared_ptr<const TensorExpr>;

/**
 * @brief Tensor variable reference expression
 *
 * Represents a reference to a named tensor variable.
 */
class TensorVar : public TensorExpr {
 public:
  std::string name_;

  /**
   * @brief Create a tensor variable reference
   *
   * @param name Variable name
   * @param dtype Element data type
   * @param shape Shape dimensions
   * @param span Source location
   * @return Shared pointer to const TensorVar expression
   */
  TensorVar(std::string name, DataType dtype, std::vector<ScalarExprPtr> shape, Span span)
      : TensorExpr(std::move(span), dtype, std::move(shape)), name_(std::move(name)) {}

  [[nodiscard]] const char* type_name() const override { return "TensorVar"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(TensorExpr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TensorVar::name_, "name")));
  }
};

using TensorVarPtr = std::shared_ptr<const TensorVar>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TENSOR_EXPR_H_
