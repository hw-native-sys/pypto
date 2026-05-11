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
#include <optional>
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

struct TpopResultInfo {
  int split = 0;
  std::string op_name;
  std::optional<int> pipe_id;
};

/**
 * @brief PTO MLIR code generator
 *
 * Generates PTO-ISA MLIR format code from PyPTO IR Program.
 * Traverses the IR using the visitor pattern.
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
  int64_t GetConstIntValue(const ir::ExprPtr& expr) const override;
  std::string GetVarName(const ir::VarPtr& var) const override;

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
   * @brief Get or emit a numeric constant of any dtype (int, index, or float).
   *
   * Both overloads write the constant to the constants section on first use and
   * return the SSA name. Subsequent calls for the same (value, dtype) pair
   * return the cached name without emitting again.
   *
   * @param value Integer or index value
   * @param dt    Data type (e.g., DataType::INDEX, DataType::INT32, DataType::INT64)
   * @return SSA variable name for the constant
   */
  std::string GetOrEmitConstant(int64_t value, DataType dt);

  /**
   * @brief Get or emit a floating-point constant of any float dtype.
   *
   * @param value Floating-point value
   * @param dt    Data type (e.g., DataType::FP32, DataType::BF16, DataType::FP16)
   * @return SSA variable name for the constant
   */
  std::string GetOrEmitConstant(double value, DataType dt);

  /**
   * @brief Emit arith.index_cast if var is not already index type
   *
   * Valid_shape vars may be INT64/INT32 (from pl.min(...)), but pto.alloc_tile
   * and pto.set_validshape need index type operands.
   *
   * @param var IR variable to cast
   * @param mlir_name Current MLIR SSA name for the variable
   * @return SSA name of the index-typed value (original if already index)
   */
  std::string EmitCastToIndex(const ir::VarPtr& var, const std::string& mlir_name);

  /**
   * @brief Emit arith.index_cast if expression is not already index type
   *
   * Shape/stride expressions in PTO codegen may be constants, variables, or
   * general scalar expressions. PTO ops that consume dimensions require index
   * operands, so dynamic integer expressions must be cast on demand.
   *
   * @param expr IR expression whose type determines the cast
   * @param mlir_name Current MLIR SSA name for the expression value
   * @return SSA name of the index-typed value (original if already index)
   */
  std::string EmitCastToIndex(const ir::ExprPtr& expr, const std::string& mlir_name);

  /**
   * @brief Emit arith.index_cast if expression is not already i32 type
   *
   * PTO ISA instructions like pto.tmrgsort require i32 operands. When the
   * operand is a runtime variable (e.g., loop induction variable typed as
   * index), this emits the necessary cast.
   *
   * @param expr IR expression whose type determines the cast
   * @param mlir_name Current MLIR SSA name for the expression value
   * @return SSA name of the i32-typed value (original if already i32)
   */
  std::string EmitCastToI32(const ir::ExprPtr& expr, const std::string& mlir_name);

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
   * @brief Get tensor_view type string for a TensorType (e.g., "!pto.tensor_view<?x?xf32>")
   */
  std::string GetTensorViewTypeString(const ir::TensorType* tensor_type) const;

  /**
   * @brief Get tile_buf type string for a MemRef (e.g., "!pto.tile_buf<loc=vec, dtype=f32, ...>")
   */
  std::string GetTileBufTypeString(const ir::Var* base_ptr) const;

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
   * @brief Get tpop result valid_shape operands as index-typed SSA values.
   *
   * PTOAS frontend tpop accepts optional `(valid_row, valid_col)` operands only
   * when the result tile type carries dynamic valid shape (`v_row=?, v_col=?`).
   * Returns empty strings when the current result does not require dynamic
   * valid_shape operands.
   */
  std::pair<std::string, std::string> GetCurrentResultTpopValidShapeOperands();

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
   * @param name_hint Preferred SSA name seed
   * @param addr_ssa Optional SSA value for the alloc_tile addr operand
   * @param valid_row_ssa Optional SSA value for the alloc_tile valid_row operand
   * @param valid_col_ssa Optional SSA value for the alloc_tile valid_col operand
   * @return New SSA variable name for the allocated buffer
   */
  std::string AllocNewTileBuf(const std::string& tile_buf_type_string, const std::string& name_hint = "",
                              const std::string& addr_ssa = "", const std::string& valid_row_ssa = "",
                              const std::string& valid_col_ssa = "");

  /**
   * @brief Emit alloc_tile for a tile variable before its first use
   *
   * Idempotent: a Var is only allocated once per function (tracked via
   * `fs_.emitted_tile_alloc_vars`). Multi-output op codegen (e.g. tile.gather_compare)
   * uses this to eagerly allocate DPS dst/cdst tiles bound by downstream
   * `dst = tuple_var[i]` AssignStmts before they are visited.
   */
  void EmitAllocTileForVar(const ir::VarPtr& tile_var, const std::shared_ptr<const ir::TileType>& tile_type);

  /**
   * @brief Resolve the DPS element vars of a tuple-returning op call
   *
   * Multi-output ops (e.g. tile.gather_compare) return a TupleType. The parser
   * desugars `a, b = call(...)` into:
   *   _tuple_tmp = call(...)
   *   a = _tuple_tmp[0]
   *   b = _tuple_tmp[1]
   *
   * Since the dst element Vars do not appear in the Call's args, codegen must
   * scan the current function body for these `<var> = tuple_var[i]` AssignStmts
   * to recover the SSA names of the DPS outputs.
   *
   * @param tuple_var The tuple-result Var (typically GetCurrentResultVar()).
   * @param arity     Number of expected tuple elements.
   * @return Vector of length `arity`; entry i is the Var bound to
   *         `tuple_var[i]`, or nullptr if no such consumer exists.
   */
  [[nodiscard]] std::vector<ir::VarPtr> ResolveTupleResultElements(const ir::VarPtr& tuple_var,
                                                                   size_t arity) const;

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
  struct SubviewMaterializationInfo {
    std::string source_ssa;
    std::string source_type;
    std::string row_off_ssa;
    std::string col_off_ssa;
    std::string materialize_target_ssa;
    std::string materialize_target_type;
    bool emitted = false;
  };
  void RegisterSubviewMaterialization(const std::string& subview_ssa, const SubviewMaterializationInfo& info);
  SubviewMaterializationInfo* GetSubviewMaterialization(const std::string& subview_ssa);
  const SubviewMaterializationInfo* GetSubviewMaterialization(const std::string& subview_ssa) const;

  /**
   * @brief Record the SSA name of the __gm_pipe_buffer function parameter
   *
   * On Ascend910B (a2a3), the GM slot buffer is a function parameter used as
   * intermediary for cross-core pipe communication. The codegen emits it as
   * a gm_slot_buffer operand in initialize_pipe instructions.
   */
  void RecordGMSlotBufferSSA(const std::string& ssa, const DataType& dtype);

  /**
   * @brief Get the recorded GM slot buffer SSA name (empty if none)
   */
  [[nodiscard]] std::string GetGMSlotBufferSSA() const;

  /// Increase/decrease the current indentation level (used by op codegen helpers that emit scf.for blocks)
  void IncreaseIndent() { indent_level_++; }
  void DecreaseIndent() { indent_level_--; }

  /**
   * @brief Return the GM slot buffer SSA region for one frontend pipe.
   *
   * Ascend910B uses a single function parameter as the backing GM FIFO
   * workspace. Multiple frontend pipe ids in one function must point at
   * disjoint byte ranges within that parameter.
   */
  [[nodiscard]] std::string GetGMSlotBufferSSAForPipe(int pipe_id, int dir_mask);

  /**
   * @brief Get metadata for a tile var produced by a matching tpop operation
   * @param var Raw pointer to the tile variable
   * @param expected_tpop_op_name Expected originating tpop op name
   * @param tfree_op_name Name of the consuming tfree op for diagnostics
   * @return Metadata from the originating tpop
   */
  [[nodiscard]] const TpopResultInfo& GetValidatedTpopInfo(const ir::Var* var,
                                                           const std::string& expected_tpop_op_name,
                                                           const std::string& tfree_op_name) const;

  /**
   * @brief Get the split value for a tile var produced by a matching tpop operation
   */
  [[nodiscard]] int GetValidatedTpopSplit(const ir::Var* var, const std::string& expected_tpop_op_name,
                                          const std::string& tfree_op_name) const;

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
   * @brief Collect deterministic GM slot buffer byte offsets for frontend pipe ids in a module.
   */
  void PrepareGMSlotBufferLayout(const ir::ProgramPtr& program);

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
  void BindVarToMemRef(const ir::VarPtr& var, const ir::Var* base_ptr);

  /**
   * @brief Emit make_tensor_view for all tensor parameters
   */
  void EmitMakeTensorViews(const ir::FunctionPtr& func);

  /**
   * @brief Bundle of fields needed to emit a `pto.alloc_tile` op.
   *
   * `pto.alloc_tile` is always emitted in dynamic form: the type string carries
   * `v_row=?, v_col=?`, and `valid_row` / `valid_col` operands carry the
   * actual extent (constant SSA when the IR-level extent is a constant,
   * runtime SSA otherwise).
   *
   * Returned by ComputeAllocTileFields and consumed by EmitAllocTileForVar
   * (single-statement allocs) and the IfStmt return-tile path
   * (deferred allocs via AllocNewTileBuf).
   */
  struct AllocTileFields {
    std::string type_str;       ///< pto.tile_buf<...> type string
    std::string addr_ssa;       ///< Optional addr operand SSA value
    std::string valid_row_ssa;  ///< valid_row operand SSA value (always emitted)
    std::string valid_col_ssa;  ///< valid_col operand SSA value (always emitted)
  };

  /**
   * @brief Compute the type string and (addr, valid_row, valid_col) operands
   *        for a `pto.alloc_tile` op.
   *
   * The result is always dynamic (`v_row=?, v_col=?`) and carries explicit
   * `valid_row` / `valid_col` operands lowered from `tile_type->tile_view_.valid_shape`
   * when present, falling back to `tile_type->shape_` otherwise.
   *
   * @param tile_type Tile type carrying shape/tile_view/memref metadata.
   */
  AllocTileFields ComputeAllocTileFields(const std::shared_ptr<const ir::TileType>& tile_type);

  /**
   * @brief Emit alloc_tile for dynamically allocated tile buffers (e.g., reshape outputs)
   */
  void EmitExtraAllocTiles();

  /**
   * @brief Get indent string for current level
   */
  std::string GetIndent() const;

  /**
   * @brief Get tile_buf name for a MemRef
   */
  std::string GetTileBufForMemRef(const ir::MemRefPtr& memref) const;

  /// Per-function mutable state that is reset at the start of each GenerateFunction call.
  struct FunctionState {
    std::ostringstream constants_section;
    std::ostringstream body_section;
    std::string constants_indent;  ///< Fixed indent for constants_section (set once per function)

    std::map<const ir::Var*, std::string> var_to_mlir;
    std::map<const ir::Var*, std::string> tensor_to_view;
    std::map<const ir::Var*, std::string> memref_to_mlir;    ///< keyed by base_ Ptr
    std::map<const ir::Var*, const ir::Var*> var_to_memref;  ///< maps tile var → base_ Ptr
    std::map<const ir::Var*, std::shared_ptr<const ir::TileType>>
        memref_to_tile_type;  ///< keyed by base_ Ptr

    std::map<std::pair<int64_t, uint8_t>, std::string> emitted_numeric_constants;

    struct ExtraAllocTile {
      std::string name;
      std::string type_string;
      std::string addr_ssa;
      std::string valid_row_ssa;
      std::string valid_col_ssa;
    };
    std::vector<ExtraAllocTile> extra_alloc_tiles;
    std::map<std::string, std::string> ssa_to_tile_buf_type;
    std::map<std::string, SubviewMaterializationInfo> subview_materializations;

    int temp_counter = 0;
    std::set<std::string> used_ssa_names;

    std::map<const ir::Var*, std::string> memref_to_var_name;  ///< keyed by base_ Ptr
    std::vector<std::pair<ir::VarPtr, std::shared_ptr<const ir::TileType>>> tile_var_allocs;
    std::set<const ir::Var*> emitted_tile_alloc_vars;
    std::map<const ir::Var*, TpopResultInfo> tpop_result_vars;

    ir::FunctionPtr current_function;
    ir::VarPtr current_result_var;
    std::string current_result_buf;
    std::shared_ptr<const ir::TileType> current_result_tile_type;

    std::string gm_slot_buffer_ssa;
    DataType gm_slot_buffer_dtype = DataType::FP32;
    std::map<std::pair<int, int>, std::string> gm_slot_buffer_region_by_pipe;

    std::string current_expr_value;
    std::vector<std::string> yield_buffer;

    void Reset() {
      constants_section.str("");
      constants_section.clear();
      body_section.str("");
      body_section.clear();
      constants_indent.clear();

      var_to_mlir.clear();
      tensor_to_view.clear();
      memref_to_mlir.clear();
      var_to_memref.clear();
      memref_to_tile_type.clear();

      emitted_numeric_constants.clear();

      extra_alloc_tiles.clear();
      ssa_to_tile_buf_type.clear();
      subview_materializations.clear();

      temp_counter = 0;
      used_ssa_names.clear();

      memref_to_var_name.clear();
      tile_var_allocs.clear();
      emitted_tile_alloc_vars.clear();
      tpop_result_vars.clear();

      current_function.reset();
      current_result_var.reset();
      current_result_buf.clear();
      current_result_tile_type = nullptr;

      gm_slot_buffer_ssa.clear();
      gm_slot_buffer_dtype = DataType::FP32;
      gm_slot_buffer_region_by_pipe.clear();

      current_expr_value.clear();
      yield_buffer.clear();
    }
  };

  /// Function-level mutable state, reset per GenerateFunction call.
  FunctionState fs_;

  // Module-level output stream (persists across functions)
  std::ostringstream stream_;
  int indent_level_ = 0;
  std::map<std::pair<int, int>, int64_t> gm_slot_buffer_offsets_;

  const backend::Backend* backend_;  ///< Backend instance for querying op info

  /// Emit an arith binary op, return SSA result name
  std::string EmitArithBinaryOp(const std::string& mlir_op, const std::string& lhs, const std::string& rhs,
                                const std::string& result_type);

  /// Emit an arith.cmpi comparison, return SSA result name (i1)
  std::string EmitArithCmpi(const std::string& predicate, const std::string& lhs, const std::string& rhs,
                            const std::string& operand_type);

  /// Emit @p expr as an SSA suitable for arith.*i with result/operand type @p wanted_mlir_type
  /// (e.g. "index", "i64"): integer literals use typed constants; index↔int uses arith.index_cast.
  std::string EmitArithOperand(const ir::ExprPtr& expr, const std::string& wanted_mlir_type);

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
