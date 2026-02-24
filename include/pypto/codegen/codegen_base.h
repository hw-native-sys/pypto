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

#ifndef PYPTO_CODEGEN_CODEGEN_BASE_H_
#define PYPTO_CODEGEN_CODEGEN_BASE_H_

#include <cstdint>
#include <string>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

/**
 * @brief Base class for platform code generators (CCE, PTO, etc.)
 *
 * Provides a common API used by operator codegen callbacks (f_codegen_cce, f_codegen_pto).
 * Subclasses implement platform-specific code generation while sharing this contract.
 * Does not define Generate() as each platform has different signature (e.g. map vs string).
 */
class CodegenBase : public ir::IRVisitor {
 public:
  ~CodegenBase() override = default;

  // ---------------------------------------------------------------------------
  // Common API for operator codegen callbacks (unified names and semantics)
  // ---------------------------------------------------------------------------

  /**
   * @brief Get the current result target for the active Call
   *
   * Where the result of the current operation should be written (C++ variable name
   * or MLIR SSA buffer name).
   *
   * @return Current result target name
   */
  [[nodiscard]] virtual std::string GetCurrentResultTarget() const = 0;

  /**
   * @brief Emit one line of platform code (C++ or MLIR)
   *
   * @param line Line of code to emit
   */
  virtual void Emit(const std::string& line) = 0;

  /**
   * @brief Get platform code for an expression
   *
   * Converts an IR expression to platform-usable code (C++ fragment or MLIR SSA name).
   *
   * @param expr Expression to convert
   * @return Platform code string for the expression
   */
  virtual std::string GetExprAsCode(const ir::ExprPtr& expr) = 0;

  /**
   * @brief Convert DataType to platform type string
   *
   * @param dtype Data type (e.g. FP32, INT32)
   * @return Platform type string (e.g. "float"/"f32", "int32_t"/"i32")
   */
  [[nodiscard]] virtual std::string GetTypeString(const DataType& dtype) const = 0;

  /**
   * @brief Extract constant integer value from expression
   *
   * @param expr Expression (must be ConstInt)
   * @return Integer value
   */
  virtual int64_t GetConstIntValue(const ir::ExprPtr& expr) = 0;

  /**
   * @brief Get platform name for a Var
   *
   * @param var The IR Var
   * @return Platform variable name (C++ name or MLIR SSA name)
   */
  virtual std::string GetVarName(const ir::VarPtr& var) = 0;

  /**
   * @brief Get the data pointer expression for a tensor variable
   *
   * Returns the C++ expression to access a tensor's raw data pointer.
   * Orchestration codegen returns "arg_<name>_ptr" for external tensors or "<name>.data" for local tensors.
   *
   * @param tensor_name Tensor variable name
   * @return C++ expression for the data pointer (e.g., "arg_x_ptr", "x.data")
   */
  [[nodiscard]] virtual std::string GetTensorDataPtr(const std::string& tensor_name) const {
    throw ValueError("GetTensorDataPtr not implemented for this codegen");
  }

  /**
   * @brief Get the external tensor name for a variable
   *
   * Returns the name used to reference a tensor in generated code.
   * For external tensors (params/returns), returns "ext_<name>".
   * For local tensors, returns the name as-is.
   *
   * @param name Tensor variable name
   * @return External tensor name (e.g., "ext_x" or "x")
   */
  [[nodiscard]] virtual std::string GetExternalTensorName(const std::string& name) const { return name; }

  /**
   * @brief Get the runtime DataType enum string for generated code
   *
   * Returns the fully qualified DataType enum name as used by the runtime
   * (e.g., "DataType::FLOAT32", "DataType::INT64"). Subclasses can override
   * to match their target runtime's DataType enum naming.
   *
   * @param dtype The data type to convert
   * @return Runtime DataType string (e.g., "DataType::FLOAT32")
   */
  [[nodiscard]] virtual std::string GetRuntimeDataTypeString(const DataType& dtype) const;

  /**
   * @brief Try to extract variable name from expression
   *
   * Supports Var and IterArg expressions. Returns empty string if not a variable.
   * Subclasses can override to transform variable names (e.g., strip SSA suffixes).
   *
   * @param expr Expression to extract name from
   * @return Variable name or empty string
   */
  [[nodiscard]] virtual std::string TryGetVarName(const ir::ExprPtr& expr) const;

  /**
   * @brief Generate C++ code string for an IR expression
   *
   * Converts IR expressions to C++ code strings for inline operations.
   * Supports variables, constants, binary operations, and tuple access.
   * Calls TryGetVarName() for variable name extraction, enabling subclass
   * name resolution via virtual dispatch.
   *
   * @param expr Expression to convert
   * @return C++ code string
   */
  [[nodiscard]] std::string GenerateExprString(const ir::ExprPtr& expr) const;

 protected:
  /**
   * @brief Throw when no codegen is registered for a Call
   *
   * Subclasses call this from VisitExpr_(Call) when the op has no platform codegen.
   *
   * @param op_name IR operation name (e.g., "block.load")
   */
  [[noreturn]] void ThrowNoCodegenForCall(const std::string& op_name) const {
    throw ValueError("No codegen registered for operation: " + op_name);
  }

  /**
   * @brief Default VisitExpr_(Call): throws (subclasses must override)
   *
   * All actual codegen paths go through subclass overrides. This default ensures
   * any Call that reaches the base implementation results in a compile-time error.
   */
  void VisitExpr_(const ir::CallPtr& op) override { ThrowNoCodegenForCall(op->op_->name_); }
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_CODEGEN_BASE_H_
