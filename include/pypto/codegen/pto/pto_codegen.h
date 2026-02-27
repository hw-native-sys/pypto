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

#ifndef PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_
#define PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {

namespace codegen {

/**
 * @brief PTO MLIR code generator
 *
 * Generates PTO-ISA MLIR format code from PyPTO IR Program.
 * Traverses the IR using the visitor pattern (aligned with CCECodegen).
 * Automatically generates make_tensor_view, partition_view, and alloc_tile instructions.
 */
class PTOCodegen : public CodegenBase {
 public:
  /** @brief Default constructor (backend is always PTO) */
  PTOCodegen();

  /**
   * @brief Construct PTO codegen with backend pointer (for internal use)
   */
  explicit PTOCodegen(const backend::Backend* backend);

  ~PTOCodegen() override = default;

  /**
   * @brief Generate PTO-ISA MLIR format code from IR Program
   *
   * @param program Input PyPTO IR Program
   * @return MLIR code as string
   */
  std::string Generate(const ir::ProgramPtr& program);

  // CodegenBase interface (unified API for operator codegen callbacks)
  [[nodiscard]] std::string GetCurrentResultTarget() const override;
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override;
  int64_t GetConstIntValue(const ir::ExprPtr& expr) override;
  std::string GetVarName(const ir::VarPtr& var) override;

  // PTO-specific helper methods for operator codegen functions

  /**
   * @brief Create a new temporary SSA variable
   *
   * @return New SSA variable name (e.g., "%1", "%2")
   */
  std::string NewTemp();

  /**
   * @brief Get or create tensor view for a variable
   *
   * @param tensor Tensor variable
   * @return Tensor view name
   */
  std::string GetOrCreateTensorView(const ir::VarPtr& tensor);

  /**
   * @brief Get or emit index constant
   *
   * @param val Constant value
   * @return Index constant string
   */
  std::string GetIndexConstant(int64_t val);

  /**
   * @brief Get or emit float constant (emits to constants section, returns SSA name)
   *
   * @param value Constant value
   * @param mlir_type MLIR type string (e.g., "f32", "i32")
   * @return SSA variable name for the constant
   */
  std::string GetOrEmitFloatConstant(double value, const std::string& mlir_type = "f32");

  /**
   * @brief Get tensor_view type string for a TensorType (e.g., "!pto.tensor_view<?x?xf32>")
   */
  std::string GetTensorViewTypeString(const ir::TensorType* tensor_type) const;

  /**
   * @brief Get tile_buf type string for a MemRef (e.g., "!pto.tile_buf<loc=vec, dtype=f32, ...>")
   */
  std::string GetTileBufTypeString(const ir::MemRef* memref) const;

  /**
   * @brief Get type annotation for an expression (for ins/outs clauses)
   */
  std::string GetExprTypeAnnotation(const ir::ExprPtr& expr);

  /**
   * @brief Get tile_buf type string for the current assignment result target
   */
  std::string GetCurrentResultTileBufTypeString() const;

 protected:
  // Override visitor methods for code generation - Statements
  void VisitStmt_(const ir::AssignStmtPtr& op) override;
  void VisitStmt_(const ir::ForStmtPtr& op) override;
  void VisitStmt_(const ir::IfStmtPtr& op) override;
  void VisitStmt_(const ir::YieldStmtPtr& op) override;
  void VisitStmt_(const ir::EvalStmtPtr& op) override;

  // Override visitor methods for code generation - Expressions
  void VisitExpr_(const ir::CallPtr& op) override;
  void VisitExpr_(const ir::VarPtr& op) override;
  void VisitExpr_(const ir::IterArgPtr& op) override;
  void VisitExpr_(const ir::ConstIntPtr& op) override;
  void VisitExpr_(const ir::ConstFloatPtr& op) override;
  void VisitExpr_(const ir::ConstBoolPtr& op) override;
  void VisitExpr_(const ir::AddPtr& op) override;
  void VisitExpr_(const ir::SubPtr& op) override;
  void VisitExpr_(const ir::MulPtr& op) override;
  void VisitExpr_(const ir::FloorDivPtr& op) override;
  void VisitExpr_(const ir::FloorModPtr& op) override;
  void VisitExpr_(const ir::EqPtr& op) override;
  void VisitExpr_(const ir::NePtr& op) override;
  void VisitExpr_(const ir::LtPtr& op) override;
  void VisitExpr_(const ir::LePtr& op) override;
  void VisitExpr_(const ir::GtPtr& op) override;
  void VisitExpr_(const ir::GePtr& op) override;

 private:
  /**
   * @brief Generate PTO-ISA MLIR for a single function
   */
  void GenerateFunction(const ir::FunctionPtr& func);

  /**
   * @brief Build variable name to MemRef mapping from function body
   */
  void BuildVarToMemRefMapping(const ir::FunctionPtr& func);

  /**
   * @brief Emit make_tensor_view for all tensor parameters
   */
  void EmitMakeTensorViews(const ir::FunctionPtr& func);

  /**
   * @brief Emit alloc_tile for all MemRefs
   */
  void EmitAllocTiles(const ir::FunctionPtr& func, const std::vector<ir::MemRefPtr>& memrefs);

  /**
   * @brief Get indent string for current level
   */
  std::string GetIndent() const;

  /**
   * @brief Get or emit index constant (internal; writes to constants section)
   */
  std::string GetOrEmitIndexConstant(int64_t value);

  /**
   * @brief Get tile_buf name for a MemRef
   */
  std::string GetTileBufForMemRef(const ir::MemRefPtr& memref);

  // Output streams
  std::ostringstream stream_;
  std::ostringstream constants_section_;
  std::ostringstream body_section_;
  int indent_level_ = 0;

  // Variable mappings
  std::map<std::string, std::string> var_to_mlir_;
  std::map<std::string, std::string> tensor_to_view_;
  std::map<const ir::MemRef*, std::string> memref_to_mlir_;
  std::map<std::string, const ir::MemRef*> var_to_memref_;
  std::map<const ir::MemRef*, std::shared_ptr<const ir::TileType>> memref_to_tile_type_;
  std::set<int64_t> emitted_constants_;
  std::set<double> emitted_float_constants_;
  std::map<double, std::string> float_const_names_;

  int temp_counter_ = 0;

  // Current function context
  ir::FunctionPtr current_function_;
  std::string current_result_buf_;
  std::shared_ptr<const ir::TileType> current_result_tile_type_;

  const backend::Backend* backend_;  ///< Backend instance for querying op info

  // Control flow expression result communication
  std::string current_expr_value_;         ///< SSA name from expression visitors
  std::vector<std::string> yield_buffer_;  ///< Temporary storage for yielded values

  /// Emit an arith binary op, return SSA result name
  std::string EmitArithBinaryOp(const std::string& mlir_op, const std::string& lhs, const std::string& rhs,
                                const std::string& result_type);

  /// Emit an arith.cmpi comparison, return SSA result name (i1)
  std::string EmitArithCmpi(const std::string& predicate, const std::string& lhs, const std::string& rhs,
                            const std::string& operand_type);

  /// Helper for binary expression visitors
  void VisitBinaryArithExpr(const ir::BinaryExprPtr& op, const std::string& int_op,
                            const std::string& float_op);

  /// Helper for comparison expression visitors
  void VisitCmpExpr(const ir::BinaryExprPtr& op, const std::string& predicate);
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_
