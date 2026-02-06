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

#ifndef PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_
#define PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/cce/code_context.h"
#include "pypto/codegen/cce/code_emitter.h"
#include "pypto/codegen/cce/type_converter.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/type.h"

namespace pypto {

namespace codegen {

/**
 * @brief CCE code generator for converting PyPTO IR to pto-isa C++ code
 *
 * CCECodegen traverses the IR using the visitor pattern and generates
 * compilable C++ code using pto-isa instructions. It handles:
 * - Function prologue (signature, argument unpacking, type definitions)
 * - Function body (block operations, sync operations, control flow)
 * - Type conversions and memory management
 */
class CCECodegen : public CodegenBase {
 public:
  /** @brief Default constructor (backend is always CCE) */
  CCECodegen();

  /**
   * @brief Generate C++ code from a PyPTO IR Program
   *
   * Classifies functions into kernel and orchestration, then generates:
   * - Kernel functions -> kernels/<func_name>.cpp (CCE kernel C++ code)
   * - Orchestration function -> orchestration/<func_name>.cpp (orchestration C++ code)
   *
   * @param program The IR Program to generate code for
   * @return Map from file path to generated C++ code content
   */
  [[nodiscard]] std::map<std::string, std::string> Generate(const ir::ProgramPtr& program);

  // CodegenBase interface (unified API for operator codegen callbacks)
  [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_target_var_; }
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override;
  int64_t GetConstIntValue(const ir::ExprPtr& expr) override;
  std::string GetVarName(const ir::VarPtr& var) override;

  /**
   * @brief Get pointer name for a variable (CCE-specific)
   */
  std::string GetPointer(const std::string& var_name);

  /**
   * @brief Register pointer mapping for block.store result (CCE-specific)
   *
   * Associates the assignment target variable with the output tensor variable
   * for pointer lookup. Used when block.store returns a tensor reference.
   *
   * @param output_var_name Assignment target variable name
   * @param tensor_var_name Output tensor variable name (e.g., from GlobalTensor)
   */
  void RegisterOutputPointer(const std::string& output_var_name, const std::string& tensor_var_name);

 protected:
  // Override visitor methods for code generation - Statements
  void VisitStmt_(const ir::AssignStmtPtr& op) override;
  void VisitStmt_(const ir::EvalStmtPtr& op) override;
  void VisitStmt_(const ir::ReturnStmtPtr& op) override;
  void VisitStmt_(const ir::ForStmtPtr& op) override;
  void VisitStmt_(const ir::IfStmtPtr& op) override;
  void VisitStmt_(const ir::YieldStmtPtr& op) override;

  // Override visitor methods for code generation - Expressions
  // Leaf nodes
  void VisitExpr_(const ir::VarPtr& op) override;
  void VisitExpr_(const ir::IterArgPtr& op) override;
  void VisitExpr_(const ir::ConstIntPtr& op) override;
  void VisitExpr_(const ir::ConstFloatPtr& op) override;
  void VisitExpr_(const ir::ConstBoolPtr& op) override;
  void VisitExpr_(const ir::CallPtr& op) override;
  void VisitExpr_(const ir::TupleGetItemExprPtr& op) override;

  // Binary operations
  void VisitExpr_(const ir::AddPtr& op) override;
  void VisitExpr_(const ir::SubPtr& op) override;
  void VisitExpr_(const ir::MulPtr& op) override;
  void VisitExpr_(const ir::FloorDivPtr& op) override;
  void VisitExpr_(const ir::FloorModPtr& op) override;
  void VisitExpr_(const ir::FloatDivPtr& op) override;
  void VisitExpr_(const ir::MinPtr& op) override;
  void VisitExpr_(const ir::MaxPtr& op) override;
  void VisitExpr_(const ir::PowPtr& op) override;
  void VisitExpr_(const ir::EqPtr& op) override;
  void VisitExpr_(const ir::NePtr& op) override;
  void VisitExpr_(const ir::LtPtr& op) override;
  void VisitExpr_(const ir::LePtr& op) override;
  void VisitExpr_(const ir::GtPtr& op) override;
  void VisitExpr_(const ir::GePtr& op) override;
  void VisitExpr_(const ir::AndPtr& op) override;
  void VisitExpr_(const ir::OrPtr& op) override;
  void VisitExpr_(const ir::XorPtr& op) override;
  void VisitExpr_(const ir::BitAndPtr& op) override;
  void VisitExpr_(const ir::BitOrPtr& op) override;
  void VisitExpr_(const ir::BitXorPtr& op) override;
  void VisitExpr_(const ir::BitShiftLeftPtr& op) override;
  void VisitExpr_(const ir::BitShiftRightPtr& op) override;

  // Unary operations
  void VisitExpr_(const ir::AbsPtr& op) override;
  void VisitExpr_(const ir::NegPtr& op) override;
  void VisitExpr_(const ir::NotPtr& op) override;
  void VisitExpr_(const ir::BitNotPtr& op) override;
  void VisitExpr_(const ir::CastPtr& op) override;

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

  /**
   * @brief Generate CCE kernel C++ code for a single function
   *
   * Emits function prologue (signature, argument unpacking, type declarations)
   * and body (block operations, control flow) for kernel (InCore) functions.
   *
   * @param func The kernel function to generate code for
   * @return Generated C++ code as a string
   */
  std::string GenerateFunction(const ir::FunctionPtr& func);

  /**
   * @brief Generate config file for orchestration and kernels
   *
   * @param orch_func_name Orchestration function name
   * @param func_name_to_id Kernel function name -> func id mapping
   * @param func_name_to_core_type Kernel function name -> core type mapping
   * @return Generated config file as a string
   */
  std::string GenerateConfigFile(const std::string& orch_func_name,
                                 const std::map<std::string, int>& func_name_to_id,
                                 const std::map<std::string, ir::CoreType>& func_name_to_core_type);

  /**
   * @brief Generate Tile type declaration and instance
   *
   * Emits type alias and instance declaration for a Tile variable.
   * Automatically extracts memref address from tile_type if present and emits TASSIGN.
   *
   * @param var_name Variable name for the tile
   * @param tile_type The TileType to generate declaration for (memref extracted automatically)
   */
  void GenerateTileTypeDeclaration(const std::string& var_name, const ir::TileTypePtr& tile_type);

  /**
   * @brief Generate GlobalTensor type declaration and instance
   *
   * Emits shape type alias, stride type alias, GlobalTensor type alias,
   * and instance declaration for a GlobalTensor variable.
   *
   * @param var_name Variable name for the global tensor
   * @param tensor_type The TensorType to generate declaration for
   * @param base_pointer Optional base pointer name for initialization
   */
  void GenerateGlobalTensorTypeDeclaration(const std::string& var_name, const ir::TensorTypePtr& tensor_type,
                                           const std::optional<std::string>& base_pointer = std::nullopt);

  // Dual-mode context for expression visitor pattern
  std::string current_target_var_;         ///< INPUT: Assignment target variable name (for Call expressions)
  std::string current_expr_value_;         ///< OUTPUT: Inline C++ value for scalar expressions
  std::vector<std::string> yield_buffer_;  ///< Temporary storage for yielded values from loops

  CodeEmitter emitter_;              ///< Code emitter for structured output
  CodeContext context_;              ///< Context for variable tracking
  TypeConverter type_converter_;     ///< Type converter
  const backend::Backend* backend_;  ///< CCE backend instance (for op info, core type, orchestration)
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_
