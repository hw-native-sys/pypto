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

#ifndef PYPTO_CODEGEN_CODE_GENERATOR_H_
#define PYPTO_CODEGEN_CODE_GENERATOR_H_

#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/type.h"
#include "pypto/ir/transform/base/visitor.h"
#include "pypto/codegen/code_context.h"
#include "pypto/codegen/code_emitter.h"
#include "pypto/codegen/isa_mapper.h"
#include "pypto/codegen/type_converter.h"

namespace pypto {
namespace codegen {

/**
 * @brief Main code generator for converting PyPTO IR to pto-isa C++ code
 *
 * CodeGenerator traverses the IR using the visitor pattern and generates
 * compilable C++ code using pto-isa instructions. It handles:
 * - Function prologue (signature, argument unpacking, type definitions)
 * - Function body (block operations, sync operations, control flow)
 * - Type conversions and memory management
 */
class CodeGenerator : public ir::IRVisitor {
 public:
  CodeGenerator();

  /**
   * @brief Generate C++ code from a PyPTO IR function
   *
   * @param func The IR function to generate code for
   * @return Generated C++ code as a string
   */
  [[nodiscard]] std::string Generate(const ir::FunctionPtr& func);

 protected:
  // Override visitor methods for code generation
  void VisitStmt_(const ir::AssignStmtPtr& op) override;
  void VisitStmt_(const ir::EvalStmtPtr& op) override;
  void VisitStmt_(const ir::ReturnStmtPtr& op) override;

 private:
  /**
   * @brief Generate function prologue
   *
   * Emits function signature, argument unpacking, GlobalTensor declarations,
   * and Tile declarations with TASSIGN.
   *
   * @param func The function to generate prologue for
   */
  void GeneratePrologue(const ir::FunctionPtr& func);

  /**
   * @brief Generate function body
   *
   * Visits the function body statement to generate the main code.
   *
   * @param func The function to generate body for
   */
  void GenerateBody(const ir::FunctionPtr& func);

  /**
   * @brief Get C++ variable name for an expression
   *
   * For Var expressions, returns the variable name.
   * For ConstInt/ConstFloat/ConstBool, returns the literal.
   * For BinaryExpr/UnaryExpr, recursively generates C++ expression.
   *
   * @param expr The expression
   * @return The C++ variable name or expression
   */
  std::string GetExprName(const ir::ExprPtr& expr);

  /**
   * @brief Get C++ operator string for binary expression
   *
   * Maps IR binary expression types to C++ operators:
   * Add -> "+", Sub -> "-", Mul -> "*", etc.
   *
   * @param expr The binary expression
   * @return C++ operator string
   */
  std::string GetBinaryOperator(const ir::ExprPtr& expr);

  /**
   * @brief Get C++ operator string for unary expression
   *
   * Maps IR unary expression types to C++ operators:
   * Neg -> "-", Not -> "!", BitNot -> "~"
   *
   * @param expr The unary expression
   * @return C++ operator string
   */
  std::string GetUnaryOperator(const ir::ExprPtr& expr);

  /**
   * @brief Extract constant integer value from expression
   *
   * @param expr The expression (must be ConstInt)
   * @return The integer value
   */
  int64_t ExtractConstInt(const ir::ExprPtr& expr);

  /**
   * @brief Collect all TileType variables from function body
   *
   * Recursively traverses the statement tree to find all variables
   * with TileType that need Tile declarations in the prologue.
   *
   * @param stmt The statement to scan (typically func->body_)
   * @return Vector of (Var, TileType) pairs
   */
  std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CollectTileVariables(const ir::StmtPtr& stmt);

  /**
   * @brief Extract shape dimensions from shape expressions
   *
   * Converts a vector of shape expressions (assumed to be ConstInt)
   * into a vector of integer dimensions.
   *
   * @param shape_exprs Vector of shape expressions (ConstInt)
   * @return Vector of integer dimensions
   */
  std::vector<int64_t> ExtractShapeDimensions(const std::vector<ir::ExprPtr>& shape_exprs);

  /**
   * @brief Format address as hexadecimal string
   *
   * Converts an integer address to hex format for TASSIGN instructions.
   *
   * @param addr Address value
   * @return Hex string (e.g., "0x0", "0x10000")
   */
  std::string FormatAddressHex(int64_t addr);

  CodeEmitter emitter_;          ///< Code emitter for structured output
  CodeContext context_;          ///< Context for variable tracking
  TypeConverter type_converter_; ///< Type converter
  ISAMapper isa_mapper_;         ///< Operation → ISA mapping
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_CODE_GENERATOR_H_
