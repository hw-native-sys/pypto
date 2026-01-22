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

#ifndef PYPTO_CODEGEN_PTO_CODEGEN_H_
#define PYPTO_CODEGEN_PTO_CODEGEN_H_

#include <map>
#include <string>
#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/transform/base/mutator.h"

namespace pypto {
namespace ir {

/**
 * @brief Tile declaration information
 */
struct TileInfo {
  std::string name;
  int rows;
  int cols;
  DataType dtype;
};

/**
 * @brief PTO Assembly Code Generator
 *
 * Generates PTO assembly (.pto files) from PyPTO IR.
 * Traverses IR tree and emits PTO ISA instructions in SSA form.
 *
 * This class transforms PyPTO IR operations and control flow into
 * PTO assembly instructions, supporting:
 * - Tile operations (binary, unary, scalar) -> PTO instructions (tmul, tadd, etc.)
 * - Control flow (for loops, if statements) -> FOR/ENDFOR, IF/ENDIF
 * - SSA-style variable naming with % prefix
 * - Proper type annotations (!pto.tile<...>, !pto.memref<...>)
 */
class PTOCodegen : public IRMutator {
 public:
  PTOCodegen() = default;
  ~PTOCodegen() override = default;

  // Disable copying and moving
  PTOCodegen(const PTOCodegen&) = delete;
  PTOCodegen& operator=(const PTOCodegen&) = delete;
  PTOCodegen(PTOCodegen&&) = delete;
  PTOCodegen& operator=(PTOCodegen&&) = delete;

  /**
   * @brief Generate PTO assembly from PyPTO IR Program
   *
   * Transforms the entire program into PTO assembly (.pto format).
   *
   * @param program Input PyPTO IR Program
   * @return PTO assembly string
   */
  std::string Generate(const ProgramPtr& program);

 protected:
  // Override specific visit methods to emit Python code
  // These methods generate DSL method calls for different IR nodes

  /**
   * @brief Transform Call nodes (operations) to PTO instructions
   *
   * Maps PyPTO IR operations to PTO assembly instructions.
   * For example, block.mul(a, b) becomes %result = tmul %a, %b : !pto.tile<...>
   *
   * @param op Call expression to transform
   * @return Transformed Call expression (passthrough for visitor pattern)
   */
  ExprPtr VisitExpr_(const CallPtr& op) override;

  /**
   * @brief Transform AssignStmt nodes to PTO instructions
   *
   * Generates PTO assembly for variable assignments.
   * For example, x = block.mul(a, b) becomes %x = tmul %a, %b : !pto.tile<...>
   *
   * @param op AssignStmt to transform
   * @return Transformed AssignStmt (passthrough for visitor pattern)
   */
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override;

  /**
   * @brief Transform ForStmt nodes to FOR/ENDFOR instructions
   *
   * Generates PTO assembly for for loops with proper indentation.
   *
   * @param op ForStmt to transform
   * @return Transformed ForStmt (passthrough for visitor pattern)
   */
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;

  /**
   * @brief Transform IfStmt nodes to IF/ENDIF instructions
   *
   * Generates PTO assembly for if statements with optional else branches.
   *
   * @param op IfStmt to transform
   * @return Transformed IfStmt (passthrough for visitor pattern)
   */
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;

  /**
   * @brief Transform SeqStmts nodes by visiting each statement
   *
   * Recursively processes statement sequences.
   *
   * @param op SeqStmts to transform
   * @return Transformed SeqStmts (passthrough for visitor pattern)
   */
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;

 private:
  /**
   * @brief Generate PTO assembly for a single function
   *
   * Emits function signature, tile/scalar declarations, and instructions.
   *
   * @param func Function to generate code for
   * @return PTO assembly string for this function
   */
  std::string GenerateFunction(const FunctionPtr& func);

  /**
   * @brief Emit a line of PTO assembly with proper indentation
   *
   * Adds indentation based on indent_level_ and appends to code_lines_.
   *
   * @param line Code line to emit (without indentation)
   */
  void EmitLine(const std::string& line);

  /**
   * @brief Map IR operation to PTO assembly instruction
   *
   * Converts PyPTO IR Call node to PTO instruction syntax.
   * For example, block.mul with args [a, b] becomes "%result = tmul %a, %b : !pto.tile<...>"
   *
   * @param op Call node representing the operation
   * @param result_var Name of the result variable
   * @return PTO instruction string
   */
  std::string OpToPTOInstruction(const CallPtr& op, const std::string& result_var);

  /**
   * @brief Extract variable name or constant value from expression
   *
   * Handles Var nodes (returns name) and Constant nodes (returns string representation).
   *
   * @param expr Expression to extract from
   * @return Variable name or constant value as string
   */
  std::string ExtractVarName(const ExprPtr& expr);

  /**
   * @brief Convert DataType to PTO type string
   *
   * Maps PyPTO DataType enum to PTO type notation.
   * For example, DataType::FP32 becomes "f32", DataType::INT32 becomes "i32"
   *
   * @param dtype DataType to convert
   * @return PTO type string
   */
  std::string DataTypeToPTOType(DataType dtype);

  /**
   * @brief Convert IR Type to PTO type string
   *
   * Converts complete IR type to PTO type notation with full syntax.
   * For example:
   *   TileType([32, 128], FP32) -> "!pto.tile<32x128xf32>"
   *   ScalarType(INT32) -> "i32"
   *   MemRefType(...) -> "!pto.memref<gm,...,f32>"
   *
   * @param type Type to convert
   * @return PTO type string
   */
  std::string TypeToPTOType(const TypePtr& type);

  /**
   * @brief Emit a CMP instruction for scalar comparison
   *
   * Generates PTO CMP instruction for scalar comparisons.
   * For example: CMP %result:u1, %left:i32, %right:i32, GT
   *
   * @param result_var Result variable to store comparison result
   * @param left Left operand expression
   * @param right Right operand expression
   * @param cmp_op Comparison operation (GT, GE, LT, LE, EQ, NE)
   */
  void EmitComparisonInstruction(const VarPtr& result_var, const ExprPtr& left, const ExprPtr& right,
                                 const std::string& cmp_op);

  /**
   * @brief Format float value with proper decimal notation
   *
   * Ensures float values are formatted with at least one decimal place.
   * For example: 2 -> "2.0", 3.14 -> "3.14"
   *
   * @param value Float value to format
   * @return Formatted string representation
   */
  std::string FormatFloat(double value);

  // State variables for code generation
  std::vector<std::string> code_lines_;                         // Accumulated code lines
  std::map<std::string, TileInfo> tile_decls_;                  // Tile declarations (name -> info)
  std::vector<std::pair<std::string, DataType>> scalar_decls_;  // Scalar declarations (ordered)
  int indent_level_ = 0;                                        // Current indentation level
  std::string current_function_name_;                           // Current function being generated
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_CODEGEN_H_
