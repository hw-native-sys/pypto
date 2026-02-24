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

#ifndef PYPTO_IR_TRANSFORMS_OP_CONVERSION_REGISTRY_H_
#define PYPTO_IR_TRANSFORMS_OP_CONVERSION_REGISTRY_H_

#include <any>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/common.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

/**
 * @brief Result of an op conversion rule
 *
 * A conversion may produce:
 * - Simple: Just one block op call (empty prologue, result expr only)
 * - Complex: Multiple prologue statements + a final result expression
 */
struct ConversionResult {
  std::vector<StmtPtr> prologue;  ///< Statements to insert before the assignment
  ExprPtr result;                 ///< The result expression

  /// Convenience: construct from Expr only (simple case)
  explicit ConversionResult(ExprPtr expr) : prologue{}, result{std::move(expr)} {}

  /// Full constructor (complex case)
  ConversionResult(std::vector<StmtPtr> stmts, ExprPtr expr)
      : prologue{std::move(stmts)}, result{std::move(expr)} {}
};

/**
 * @brief Signature for custom conversion functions
 *
 * @param args Positional arguments (already substituted to tile types)
 * @param kwargs Keyword arguments from the original call
 * @param span Source location of the original call
 * @return ConversionResult with optional prologue and result expression
 */
using ConversionFunc = std::function<ConversionResult(
    const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
    const Span& span)>;

/**
 * @brief Registry mapping tensor op names to block op conversion rules
 *
 * Supports two registration styles:
 * - Simple name mapping: tensor.add -> block.add (auto-creates conversion)
 * - Custom converter: full ConversionFunc for complex conversions
 *
 * Re-registering the same op name replaces the previous rule (override semantics).
 */
class OpConversionRegistry {
 public:
  OpConversionRegistry(const OpConversionRegistry&) = delete;
  OpConversionRegistry& operator=(const OpConversionRegistry&) = delete;

  /**
   * @brief Get the singleton instance
   */
  static OpConversionRegistry& GetInstance();

  /**
   * @brief Register a simple name mapping (tensor op -> block op)
   *
   * Creates a ConversionFunc that calls OpRegistry::Create with the target name.
   * Re-registering the same from_op replaces the previous rule.
   *
   * @param from_op Source op name (e.g., "tensor.add")
   * @param to_op Target op name (e.g., "block.add")
   */
  void RegisterSimple(const std::string& from_op, const std::string& to_op);

  /**
   * @brief Register a custom conversion function
   *
   * Re-registering the same from_op replaces the previous rule.
   *
   * @param from_op Source op name (e.g., "tensor.matmul")
   * @param func Custom conversion function
   */
  void RegisterCustom(const std::string& from_op, ConversionFunc func);

  /**
   * @brief Look up a conversion rule for an op
   *
   * @param op_name The operator name to look up
   * @return Pointer to the ConversionFunc, or nullptr if not registered
   */
  [[nodiscard]] const ConversionFunc* Lookup(const std::string& op_name) const;

  /**
   * @brief Check if a conversion rule exists for an op
   */
  [[nodiscard]] bool HasConversion(const std::string& op_name) const;

 private:
  OpConversionRegistry();

  std::unordered_map<std::string, ConversionFunc> conversions_;
};

/**
 * @brief Helper macro for simple op conversion registration
 */
#define REGISTER_OP_CONVERSION(FromOp, ToOp)                \
  static bool PYPTO_STR_CONCAT(op_conv_reg_, __COUNTER__) = \
      (::pypto::ir::OpConversionRegistry::GetInstance().RegisterSimple(FromOp, ToOp), true)

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_OP_CONVERSION_REGISTRY_H_
