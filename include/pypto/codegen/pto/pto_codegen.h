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
#include <utility>
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
   * @brief Create a named SSA variable using an IR variable name
   *
   * If the name is non-empty and not already used, returns "%<name>".
   * Otherwise falls back to NewTemp() for a numeric name.
   *
   * @param name IR variable name (e.g., "sq_sum_0_tile")
   * @return Named SSA variable (e.g., "%sq_sum_0_tile") or numeric fallback
   */
  std::string NewNamedTemp(const std::string& name);

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
   * @brief Get or emit i32 constant (for cross-core consumer buffer addresses)
   *
   * @param value Constant value
   * @return SSA variable name for the constant (e.g., "%c0_i32")
   */
  std::string GetOrEmitI32Constant(int32_t value);

  /**
   * @brief Register a variable to an MLIR SSA name
   *
   * @param var IR variable
   * @param mlir_name MLIR SSA name (e.g., "%arg3")
   */
  void RegisterVarToMlir(const ir::VarPtr& var, const std::string& mlir_name);

  /**
   * @brief Register a tensor variable to its tensor view SSA name
   *
   * Used when block.store assigns a tensor result that inherits the input tensor's view.
   *
   * @param var IR variable
   * @param tensor_view_name MLIR tensor view SSA name
   */
  void RegisterTensorView(const ir::VarPtr& var, const std::string& tensor_view_name);

  /**
   * @brief Get the IR variable currently being assigned
   */
  [[nodiscard]] ir::VarPtr GetCurrentResultVar() const;

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
   *
   * Uses the memref-based lookup (same as alloc_tile) to ensure the emitted
   * type is consistent with the SSA value's definition.
   */
  std::string GetCurrentResultTileBufTypeString() const;

  /**
   * @brief Get tile_buf type string from the current result's own TileType
   *
   * Unlike GetCurrentResultTileBufTypeString(), this bypasses the memref lookup
   * and uses current_result_tile_type_ directly. Needed for operations like
   * reshape where the output shape differs from the memref's alloc_tile shape.
   */
  std::string GetCurrentResultTileBufTypeStringFromTileType() const;

  /**
   * @brief Get tile_buf type string directly from a TileType
   *
   * Unlike GetTileBufTypeString(memref), this uses the shape/layout from the
   * provided TileType directly, bypassing the memref_to_tile_type_ lookup.
   * Needed when multiple variables with different shapes share the same MemRef
   * (e.g., reshape input/output).
   */
  std::string GetTileBufTypeStringFromTileType(const std::shared_ptr<const ir::TileType>& tile_type) const;

  /**
   * @brief Allocate a new tile buffer for codegen (emitted at function scope)
   *
   * Used when an operation needs a distinct output buffer (e.g., reshape where
   * input and output would otherwise share the same buffer).
   *
   * @param tile_buf_type_string The tile_buf type string for the alloc_tile instruction
   * @return New SSA variable name for the allocated buffer
   */
  std::string AllocNewTileBuf(const std::string& tile_buf_type_string, const std::string& name_hint = "");

  /**
   * @brief Override the current result buffer name
   *
   * Allows codegen lambdas to redirect the result to a newly allocated buffer.
   * VisitStmt_ detects the change and updates variable-to-MLIR mappings accordingly.
   *
   * @param buf New result buffer SSA name
   */
  void SetCurrentResultBuf(const std::string& buf);
  void RegisterTileBufType(const std::string& ssa_name, const std::string& type_string);
  std::string GetSSATileBufType(const std::string& ssa_name) const;

  /**
   * @brief Record the SSA name produced by reserve_buffer for cross-core pipe setup
   */
  void RecordReserveBufferSSA(const std::string& ssa);

  /**
   * @brief Get the recorded reserve_buffer SSA name (empty if none)
   */
  [[nodiscard]] std::string GetReserveBufferSSA() const;

  /**
   * @brief Record the SSA name produced by import_reserved_buffer for cross-core pipe setup
   */
  void RecordImportBufferSSA(const std::string& ssa);

  /**
   * @brief Get the recorded import_reserved_buffer SSA name (empty if none)
   */
  [[nodiscard]] std::string GetImportBufferSSA() const;

  /**
   * @brief Get the split value for a tile var produced by a tpop operation
   * @param var Raw pointer to the tile variable
   * @return Split value from the originating tpop (0 if not found)
   */
  [[nodiscard]] int GetTpopSplit(const ir::Var* var) const;

  /**
   * @brief Check if the current function is an AIC (Cube) function
   */
  [[nodiscard]] bool IsAICFunction() const;

  /**
   * @brief Check if the current function is an AIV (Vector) function
   */
  [[nodiscard]] bool IsAIVFunction() const;

 protected:
  // Override visitor methods for code generation - Statements
  void VisitStmt_(const ir::AssignStmtPtr& op) override;
  void VisitStmt_(const ir::ForStmtPtr& op) override;
  void VisitStmt_(const ir::IfStmtPtr& op) override;
  void VisitStmt_(const ir::WhileStmtPtr& op) override;
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
  void VisitExpr_(const ir::CastPtr& op) override;
  // Logical
  void VisitExpr_(const ir::AndPtr& op) override;
  void VisitExpr_(const ir::OrPtr& op) override;
  void VisitExpr_(const ir::XorPtr& op) override;
  // Bitwise
  void VisitExpr_(const ir::BitAndPtr& op) override;
  void VisitExpr_(const ir::BitOrPtr& op) override;
  void VisitExpr_(const ir::BitXorPtr& op) override;
  void VisitExpr_(const ir::BitShiftLeftPtr& op) override;
  void VisitExpr_(const ir::BitShiftRightPtr& op) override;
  // Other binary
  void VisitExpr_(const ir::FloatDivPtr& op) override;
  void VisitExpr_(const ir::MinPtr& op) override;
  void VisitExpr_(const ir::MaxPtr& op) override;
  // Unary
  void VisitExpr_(const ir::NotPtr& op) override;
  void VisitExpr_(const ir::NegPtr& op) override;
  void VisitExpr_(const ir::AbsPtr& op) override;
  void VisitExpr_(const ir::BitNotPtr& op) override;

 private:
  /**
   * @brief Generate PTO-ISA MLIR for a single function
   */
  void GenerateFunction(const ir::FunctionPtr& func);

  /**
   * @brief Reorder top-level statements so each tpop chain follows pop-use-free order
   *
   * Hardware requires: tpop(tile) → use(tile) → tfree(tile) before the next tpop.
   * Groups tpop assignment, its direct users, and its tfree into sequential chains.
   */
  std::vector<ir::StmtPtr> ReorderTpopChains(const std::vector<ir::StmtPtr>& stmts) const;

  /**
   * @brief Build variable identity to MemRef mapping from function body
   */
  void BuildVarToMemRefMapping(const ir::FunctionPtr& func);

  /**
   * @brief Get the pointer-identity key for a variable
   */
  [[nodiscard]] const ir::Var* GetVarKey(const ir::VarPtr& var) const;
  void BindVarToMlir(const ir::VarPtr& var, const std::string& mlir_name);
  void BindTensorView(const ir::VarPtr& var, const std::string& tensor_view_name);
  void BindVarToMemRef(const ir::VarPtr& var, const ir::MemRef* memref);

  /**
   * @brief Emit make_tensor_view for all tensor parameters
   */
  void EmitMakeTensorViews(const ir::FunctionPtr& func);

  /**
   * @brief Emit alloc_tile for a tile variable before its first use
   */
  void EmitAllocTileForVar(const ir::VarPtr& tile_var, const std::shared_ptr<const ir::TileType>& tile_type);

  /**
   * @brief Emit alloc_tile for dynamically allocated tile buffers (e.g., reshape outputs)
   */
  void EmitExtraAllocTiles();

  /**
   * @brief Get indent string for current level
   */
  std::string GetIndent() const;

  /**
   * @brief Get or emit index constant (internal; writes to constants section)
   */
  std::string GetOrEmitIndexConstant(int64_t value);

  /**
   * @brief Get or emit i64 constant (for tile buffer addresses)
   */
  std::string GetOrEmitI64Constant(int64_t value);

  /**
   * @brief Get tile_buf name for a MemRef
   */
  std::string GetTileBufForMemRef(const ir::MemRefPtr& memref);

  // Output streams
  std::ostringstream stream_;
  std::ostringstream constants_section_;
  std::ostringstream body_section_;
  int indent_level_ = 0;

  // Variable mappings keyed by Var identity in the final IR snapshot.
  std::map<const ir::Var*, std::string> var_to_mlir_;
  std::map<const ir::Var*, std::string> tensor_to_view_;
  std::map<const ir::MemRef*, std::string> memref_to_mlir_;
  std::map<const ir::Var*, const ir::MemRef*> var_to_memref_;
  /// Root alloc TileType per MemRef (first writer's type, used for pto.alloc_tile)
  std::map<const ir::MemRef*, std::shared_ptr<const ir::TileType>> memref_to_tile_type_;
  std::map<int64_t, std::string> emitted_constants_;
  std::map<int64_t, std::string> emitted_i64_constants_;
  std::map<int32_t, std::string> emitted_i32_constants_;
  std::set<double> emitted_float_constants_;
  std::map<double, std::string> float_const_names_;

  /// Dynamically allocated tile buffers (SSA name, type string) emitted at function scope
  std::vector<std::pair<std::string, std::string>> extra_alloc_tiles_;
  /// Unified SSA → tile_buf type mapping.  Every typed tile SSA value
  /// (root alloc, reshape result, fillpad result, etc.) has an entry here.
  /// GetExprTypeAnnotation uses this as the primary lookup.
  std::map<std::string, std::string> ssa_to_tile_buf_type_;

  int temp_counter_ = 0;
  std::set<std::string> used_ssa_names_;

  /// Maps each unique MemRef to the first IR variable name assigned to it (program order)
  std::map<const ir::MemRef*, std::string> memref_to_var_name_;

  /// Ordered tile variable allocations: (VarPtr, TileType) pairs in program order.
  /// This is the single source of truth for per-variable alloc_tile emission.
  std::vector<std::pair<ir::VarPtr, std::shared_ptr<const ir::TileType>>> tile_var_allocs_;
  std::set<const ir::Var*> emitted_tile_alloc_vars_;
  std::map<const ir::Var*, int> tpop_result_vars_;  ///< Tile vars from tpop: var -> split value

  // Current function context
  ir::FunctionPtr current_function_;
  ir::VarPtr current_result_var_;
  std::string current_result_buf_;
  std::shared_ptr<const ir::TileType> current_result_tile_type_;

  const backend::Backend* backend_;  ///< Backend instance for querying op info

  // Cross-core buffer SSA tracking (per function)
  // NOTE: These are singletons because the cross-core protocol guarantees at most
  // one reserve_buffer and one import_peer_buffer per function.  If the protocol
  // evolves to support multiple buffers per direction, these should be replaced
  // with a map keyed by buffer name or direction.
  std::string reserve_buf_ssa_;  ///< SSA name from reserve_buffer
  std::string import_buf_ssa_;   ///< SSA name from import_reserved_buffer

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

  /// Get MLIR type string for a scalar iter_arg/return_var (e.g., "index", "i1", "f32")
  std::string GetScalarIterArgTypeString(const std::shared_ptr<const ir::ScalarType>& scalar_type) const;
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_PTO_CODEGEN_H_
