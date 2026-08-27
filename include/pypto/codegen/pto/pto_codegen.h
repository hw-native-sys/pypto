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

#include <cstddef>
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
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {

// Forward declaration for PTOCodegen::GetBackendHandler()'s return type. The full
// definition lives in pypto/backend/common/backend_handler.h and is included by
// the translation units that call the handler's methods (e.g. op-emit callbacks).
namespace backend {
class BackendHandler;
}  // namespace backend

namespace codegen {

/**
 * @brief Collect Vars referenced by a tensor-shape expression, in first-seen DFS order.
 *
 * Used by:
 *   - PTOCodegen, to emit trailing `%argN: index` params on `func.func` signatures
 *     (see `CollectTensorShapeDynVars` in pto_codegen.cpp).
 *   - The Python kernel-wrapper codegen, to recover dynamic dims from
 *     `tensor->shapes[]` and forward them to the inner call in matching positional
 *     order. This is the single source of truth shared by both paths: the wrapper
 *     and the compiled function signature stay in lockstep by construction.
 *
 * Supported node kinds: Var / BinaryExpr / UnaryExpr / Call / TupleGetItemExpr /
 * ConstInt / ConstFloat / ConstBool. Any other expression kind triggers an
 * INTERNAL_CHECK failure. Adding a new shape-expressible `Expr` subclass requires
 * updating only this function.
 *
 * Dedup key: raw `Var*` (sound because the IR holds the canonical shared_ptr
 * graph, so each Var has exactly one address). The dedup scope is this single
 * call; cross-expression dedup is the caller's responsibility.
 *
 * @param expr Tensor-shape expression (a dim from `TensorType::shape_`).
 * @return Vars in first-seen DFS order, deduped within this single call.
 */
std::vector<ir::VarPtr> CollectVarsFromShapeExpr(const ir::ExprPtr& expr);

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
   * @brief Backend handler for backend-specific codegen decisions.
   *
   * Never null: the constructor requires a backend that exposes a handler.
   * Used by op-emit callbacks that must gate behaviour on the target backend
   * (e.g. rejecting a bf16 atomic-add store on Ascend950).
   */
  [[nodiscard]] const backend::BackendHandler* GetBackendHandler() const;

  /**
   * @brief Generate PTO-ISA MLIR format code from IR Program
   *
   * @param program Input PyPTO IR Program
   * @param emit_tile_addr When true (default), emit the physical `addr` operand
   *        on `pto.alloc_tile` from the MemRef byte offset (ptoas
   *        --pto-level=level3). When false, omit `addr` so the ptoas PlanMemory
   *        pass allocates instead (--pto-level=level2).
   * @param emit_source_loc When true (default), suffix every emitted operation
   *        with an MLIR `loc("file":line:col)` derived from the IR Span, so
   *        ptoas diagnostics name the user's source instead of a line in the
   *        generated `.pto`. Three kinds of line are excluded by design: the
   *        structural region braces and block labels emitted by
   *        EmitStructural(), which MLIR forbids a trailing location on; the
   *        `arith.constant` operations GetOrEmitConstant() writes to the
   *        constants section, which are deduplicated across every use so no
   *        single span fits them; and any operation whose span is unknown or
   *        carries no filename. When false, emit no locations at all.
   * @return MLIR code as string
   */
  std::string Generate(const ir::ProgramPtr& program, bool emit_tile_addr = true,
                       bool emit_source_loc = true);

  // CodegenBase interface (unified API for operator codegen callbacks)
  [[nodiscard]] std::string GetCurrentResultTarget() const override;
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override;
  int64_t GetConstIntValue(const ir::ExprPtr& expr) const override;
  std::string GetVarName(const ir::VarPtr& var) const override;

  /**
   * @brief Emit one structural (non-operation) MLIR line
   *
   * MLIR's trailing `loc(...)` is only legal at the end of a complete
   * operation. Region openers (`scf.for ... {`), separators (`} else {`,
   * `} do {`), closers (`}`) and block labels (`^bb0(...):`) are not
   * operations, so they must bypass the location suffix that Emit() appends.
   *
   * @param line Line of MLIR to emit verbatim (indented, no location)
   */
  void EmitStructural(const std::string& line);

  /**
   * @brief Resolve @p var to its MLIR SSA name, or "" when nothing binds it.
   *
   * The lenient counterpart to GetVarName, for the rare caller that can
   * genuinely proceed without a binding. GetVarName is the default: an
   * unresolvable symbol there is a user error, and emitting an empty operand
   * would produce MLIR that only fails much later inside ptoas.
   */
  [[nodiscard]] std::string LookupVarName(const ir::VarPtr& var) const;

  /**
   * @brief Explain why @p var has no SSA binding, for the GetVarName failure.
   *
   * Names the parameter whose valid_shape introduced the symbol when the origin
   * is known, so the diagnostic points at editable DSL source.
   */
  [[nodiscard]] std::string DescribeUnbindableSymbol(const ir::VarPtr& var) const;

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
   * @brief Look up the tensor view for a variable without creating/failing.
   *
   * Like GetOrCreateTensorView but returns an empty string when no view is
   * registered (and none is reachable via an IterArg init chain), instead of
   * raising. Callers that have a valid fallback (e.g. yielding a tensor that
   * has no make_tensor_view, or propagating a plain tensor alias) use this to
   * avoid a hard failure.
   *
   * @param tensor Tensor variable
   * @return Tensor view SSA name, or "" if none is registered
   */
  [[nodiscard]] std::string TryGetTensorView(const ir::VarPtr& tensor) const;

  /**
   * @brief Record that a tensor's `CachePolicy.BYPASS` request has been reported.
   *
   * A declaration is made once per tensor but read by every load of it, and a
   * load inside an unrolled loop is emitted many times over. Diagnose the
   * declaration, not the emission: this returns true only the FIRST time the
   * current function sees @p tensor, so the caller warns once per tensor per
   * kernel. State lives on the per-function frame, so the next kernel warns
   * about its own tensors again.
   *
   * @param tensor Tensor variable key (`VarPtr::get()`, like `tensor_to_view`)
   * @return true if this is the first report for @p tensor in this function
   */
  bool NoteCacheBypassWarned(const ir::Var* tensor);

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

  /// Record the base pointer SSA for a tensor var (keyed by Var, like tensor_to_view).
  void RegisterBasePtr(const ir::VarPtr& var, const std::string& ptr_name);

  /// Base pointer SSA for a tensor var; lets element-wise pl.read/pl.write recover
  /// the underlying !pto.ptr even after a slice-assign rebound the var to a view,
  /// so mixing both access styles cannot bind one SSA to two types (issue #1493).
  std::string GetTensorBasePtr(const ir::VarPtr& tensor) const;

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
   * @brief Get the TileType of the current assignment result, if any.
   *
   * Backend emitters use this alongside the result buffer/type helpers when an
   * operation's transport shape differs temporarily from its logical shape.
   */
  std::shared_ptr<const ir::TileType> GetCurrentResultTileType() const {
    return fs_.current_result_tile_type;
  }

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
   * @brief tile_buf type string for a VIEW result (`pto.treshape`).
   *
   * Same as GetTileBufTypeStringFromTileType but renders STATIC valid dims when
   * they are statically known. A view op takes no `valid_row` / `valid_col`
   * operands, so ptoas builds its destination tile from the result type alone; a
   * `v_row=?, v_col=?` result would leave the tile's valid extent at zero.
   */
  std::string GetViewTileBufTypeStringFromTileType(
      const std::shared_ptr<const ir::TileType>& tile_type) const;

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
  /// Record `ssa_name` as a tile *view* — the result of a `pto.subview` or a
  /// `pto.treshape`, which reinterprets another handle's bytes.
  void RegisterTileViewName(const std::string& ssa_name);
  /// Whether `ssa_name` was emitted as a tile view.
  ///
  /// A view carries its valid extent in its own type and has no `valid_row` /
  /// `valid_col` operands, so `pto.set_validshape` cannot mutate one and ptoas
  /// rejects the attempt. Every other tile handle — an alloc, an `scf.if` result,
  /// a cross-core pop slot — does accept it.
  ///
  /// View-ness has to be tracked, not inferred from the rendered valid dims: a
  /// `tile.slice` given a runtime `valid_shape` renders `v_row=?, v_col=?`, the
  /// same way an alloc-backed handle does.
  bool IsTileViewName(const std::string& ssa_name) const;
  struct SubviewMaterializationInfo {
    std::string source_ssa;
    std::string source_type;
    std::string row_off_ssa;
    std::string col_off_ssa;
    std::string materialize_target_ssa;
    std::string materialize_target_type;
    std::optional<ir::MemorySpace> source_memory_space;
    /// Column count of the tile the subview is taken of, and the subview's own
    /// shape. The materialize target inherits the source's buffer, so the lazy
    /// pto.textract writes into its own input: it is only safe when the window is
    /// contiguous (view_rows == 1 or view_cols == source_cols) and the repack is
    /// therefore an identity copy. See MaterializeSubviewOperandIfNeeded (#2010).
    int64_t source_cols = 0;
    int64_t view_rows = 0;
    int64_t view_cols = 0;
    /// Both slice offset components are ConstInt. A dynamic offset cannot be
    /// folded into the inherited buffer's address, which then falls back to the
    /// bare source base — so even a contiguous window would be extracted onto the
    /// source's row 0. See MaterializeSubviewOperandIfNeeded (#1640).
    bool const_offset = false;
    /// Byte-address residue modulo 32 when it is statically known for every
    /// runtime value of dynamic offsets; -1 means unknown. Keeping the residue
    /// (rather than only an aligned flag) lets nested static subviews cancel a
    /// parent's non-zero residue.
    int64_t byte_offset_mod_32 = -1;
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

  /**
   * @brief SSA name of the synthetic SDMA workspace pointer parameter.
   *
   * When the current function uses prefetch.make_context, PTOCodegen appends
   * one ``!pto.ptr<i8>`` parameter after the user-derived arguments and before
   * the synthetic SPMD identity parameters. The kernel wrapper resolves the
   * pointer from the runtime DMA workspace and forwards it as a hidden call
   * argument. Returns empty when the function does not use prefetch.
   */
  [[nodiscard]] std::string GetSdmaWorkspaceArgSSA() const { return fs_.sdma_workspace_arg_ssa; }

  /**
   * @brief SSA name of the synthetic raw dispatch-args pointer parameter.
   *
   * Functions containing ``pld.system.defer_wait`` receive one hidden
   * ``!pto.ptr<i64>`` parameter after dynamic dimensions and before other
   * runtime-owned parameters. The kernel wrapper forwards its ``args`` pointer
   * through this slot so deferred-completion lowering can reach the runtime's
   * per-task AsyncCtx. Returns empty for functions without deferred waits.
   */
  [[nodiscard]] std::string GetDeferredCompletionRawArgsSSA() const {
    return fs_.deferred_completion_raw_args_ssa;
  }

  /**
   * @brief Register the module-level deferred counter-completion adapter.
   *
   * ``pld.system.defer_wait`` lowering calls this when it emits the adapter
   * call. The declaration is emitted once at module scope and implemented by
   * the generated C++ kernel wrapper.
   *
   * @return Stable adapter symbol name.
   */
  std::string RegisterDeferredCompletionAdapter();

  /**
   * @brief SSA name of the synthetic SPMD block_idx param.
   *
   * When the current function uses tile.get_block_idx / tile.get_block_num,
   * PTOCodegen appends two i32 params to the end of the emitted func.func
   * signature. The kernel wrapper resolves the runtime values via
   * intrinsic.h::get_block_idx(args) / get_block_num(args) and forwards them
   * as trailing call args. Returns empty when the function does not use
   * SPMD block ops.
   */
  [[nodiscard]] std::string GetSpmdBlockIdxArgSSA() const { return fs_.spmd_block_idx_arg; }

  /**
   * @brief SSA name of the synthetic SPMD block_num param. See
   * GetSpmdBlockIdxArgSSA() for the surrounding mechanism.
   */
  [[nodiscard]] std::string GetSpmdBlockNumArgSSA() const { return fs_.spmd_block_num_arg; }

  /**
   * @brief SSA name of the synthetic SPMD subblock_idx (AIV lane) param.
   *
   * Mirrors GetSpmdBlockIdxArgSSA(): when the function uses
   * tile.get_subblock_idx, PTOCodegen appends one i32 param to the func.func
   * signature and the kernel wrapper resolves it from
   * intrinsic.h::get_sub_block_id(args) (the runtime's per-core lane id),
   * rather than reading the ccec get_subblockid() register. Returns empty when
   * the function does not use the op.
   */
  [[nodiscard]] std::string GetSpmdSubblockIdxArgSSA() const { return fs_.spmd_subblock_idx_arg; }

  /**
   * @brief SSA name of the materialized CommContext pointer arg for a
   * DistributedTensor parameter.
   *
   * MaterializeDistTensorCtx adds one explicit ``CommCtxType``
   * parameter per DistributedTensor parameter. PTOCodegen lowers those
   * params as ``!pto.ptr<i64>`` scalar arguments and records the
   * ``dist_tensor_var -> ctx_ssa`` mapping so pld.system.get_comm_ctx /
   * pld.tile.remote_load / pld.tensor.put / pld.system.notify /
   * pld.system.wait codegen can recover the matching context pointer.
   *
   * @param dist_var DistributedTensor parameter variable.
   * @return SSA name (e.g. ``%arg7``), or empty string if @p dist_var is
   *         not a DistributedTensor param of the current function.
   */
  [[nodiscard]] std::string GetCommCtxSSAFor(const ir::Var* dist_var) const;

  /**
   * @brief Alias a DistributedTensor LHS Var to an existing CommContext SSA.
   *
   * Mirrors the ``RegisterBasePtr`` alias mechanism that ``tile.store`` /
   * ``tensor.write`` etc. use to thread a parameter's base pointer through
   * an SSA-rebound write (``data = pl.store(local, [0, 0], data)``). The
   * CommContext binding follows the same path: an op codegen that propagates
   * the base ptr from its source DistributedTensor arg should also propagate
   * the CommContext, so subsequent cross-rank ops on the rebound Var
   * (``pld.tile.remote_load`` etc.) resolve to the same ctx pointer.
   *
   * @param dist_var SSA-rebound DistributedTensor Var (a Call's LHS).
   * @param ctx_ssa  CommContext SSA name from ``GetCommCtxSSAFor(source)``.
   *                 No-op if @p dist_var is null or @p ctx_ssa is empty.
   */
  void RegisterCommCtxFor(const ir::VarPtr& dist_var, const std::string& ctx_ssa);

  /**
   * @brief Set the current expression's SSA result.
   *
   * Op codegen lambdas that produce a value without emitting an MLIR line
   * (e.g. ``pld.system.get_comm_ctx`` aliases the existing ctx-ptr arg) call
   * this to publish their result; the surrounding ``VisitStmt_(AssignStmt)``
   * then binds the LHS Var to the same SSA. Lambdas that emit an MLIR line
   * AND want the emitted LHS to be the result must also call this — Emit()
   * alone does not update ``current_expr_value``.
   */
  void SetCurrentExprValue(std::string value) { fs_.current_expr_value = std::move(value); }

  /**
   * @brief Emit the peer-vs-local element offset arithmetic **inline**, into
   *        the function currently being generated.
   *
   * Distributed remote ops that need cross-rank peer addressing read the
   * runtime CommContext to compute the **element offset** (``index``)
   * between the local rank's window slice and the peer rank's slice. The
   * caller then does ``pto.addptr %local_ptr, %delems`` followed by
   * ``pto.make_tensor_view`` — keeping ``addptr`` and ``make_tensor_view``
   * co-located in the user kernel's ``func.func``, which is what PTOAS's
   * per-func lowering check (``addptr must feed make_tensor_view /
   * initialize_l2g2l_pipe(gm_addr) / load|store_scalar``) requires.
   *
   * The arithmetic is emitted inline rather than shared through a
   * module-level ``func.func`` helper. A mixed cube+vector kernel group is
   * ONE MLIR module holding both the AIC and the AIV function, and PTOAS
   * compiles that module into a single ``.cpp`` that is built once per
   * core. A module-level helper carries no ``pto.kernel_kind``, so PTOAS's
   * section wrapping leaves its value-returning ``return`` outside the
   * ``__DAV_VEC__`` guard and the cube compile fails on undeclared
   * identifiers. Emitting into the caller's body instead puts every line
   * inside that function's own correctly guarded section.
   *
   * All emitted values get unique SSA names (``NewTemp`` / the shared
   * constants section), so a function may contain arbitrarily many remote
   * ops.
   *
   * @param ctx_ssa  SSA name of the ``!pto.ptr<i64>`` CommContext pointer.
   * @param peer_ssa SSA name of the peer rank, already ``index``-typed.
   * @param dtype    Element dtype of the DistributedTensor (e.g. ``FP16``,
   *                 ``INT32``); only the element-size divisor depends on it.
   * @return SSA name holding the element offset (``index``).
   */
  std::string EmitCommRemoteOffsetInline(const std::string& ctx_ssa, const std::string& peer_ssa,
                                         const DataType& dtype);

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
   * @brief Whether physical addresses are baked into the emitted PTO.
   *
   * False under `memory_planner=PtoAS` (--pto-level=level2), where ptoas
   * PlanMemory owns local-memory placement: `pto.alloc_tile` omits `addr` and
   * `pto.reserve_buffer` is emitted as `auto = true` with no `base`.
   */
  [[nodiscard]] bool EmitTileAddr() const { return emit_tile_addr_; }

  /**
   * @brief Check if the current function is an AIC (Cube) function
   */
  [[nodiscard]] bool IsAICFunction() const;

  /**
   * @brief Check if the current function is an AIV (Vector) function
   */
  [[nodiscard]] bool IsAIVFunction() const;

  /**
   * @brief Check if the current function carries the `dual_aiv_dispatch`
   * attribute (910B no-split dual-AIV dispatch). In that mode the single cube
   * consumer reads the FULL slot while two AIV subblocks share it, so the
   * cross-core tpush transport widens only the COLUMN axis to the producer's
   * box (carrying its fillpad'd columns) while PRESERVING the row
   * `valid_shape[0]`: subblock 0's real push stays full and subblock 1's
   * 0-row replay stays a no-op. Genuine `split==1/2` paths widen both axes --
   * see `EmitTpushTransportValidShape`.
   */
  [[nodiscard]] bool IsDualAivDispatchFunction() const;

 protected:
  // Statement-entry dispatch guard: rejects any SplitAivScopeStmt that survived
  // to PTO codegen (it must be lowered and erased by LowerAutoVectorSplit,
  // pass 21). The base visitor would otherwise silently unwrap it.
  void VisitStmt(const ir::StmtPtr& stmt) override;

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

  /// Emit the external declaration implemented by the generated kernel wrapper.
  void EmitDeferredCompletionAdapterDeclaration();

  /**
   * @brief Build variable identity to MemRef mapping from function body
   */
  void BuildVarToMemRefMapping(const ir::FunctionPtr& func);

  /**
   * @brief Get the pointer-identity key for a variable
   */
  [[nodiscard]] const ir::Var* GetVarKey(const ir::VarPtr& var) const;
  void CheckExprVarsBound(const std::vector<ir::ExprPtr>& exprs, const ir::Span& span,
                          const std::string& context) const;
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

  /// One author-declared multi-slot allocation (`pl.MemRef(slots=N)`), lowered to
  /// a ptoas `pto.alloc_multi_tile` region whose slots are selected per use by
  /// `pto.multi_tile_get`. PTOAS-planner mode only — see PlanMultiBufferRegions.
  struct MultiBufferRegion {
    std::string region_ssa;     ///< The `%mb` handle the slots are taken from
    std::string mtb_type_str;   ///< `!pto.multi_tile_buf<<slot>, count=N>`
    std::string slot_type_str;  ///< The single-slot `!pto.tile_buf<...>` type
    std::string valid_row_ssa;  ///< valid_row operand (shared by every slot)
    std::string valid_col_ssa;  ///< valid_col operand (shared by every slot)
    uint64_t count = 1;         ///< Slot count, in [2, 16] (the ptoas bound)
  };

  /**
   * @brief Compute the type string and (addr, valid_row, valid_col) operands
   *        for a `pto.alloc_tile` op.
   *
   * The result is always dynamic (`v_row=?, v_col=?`) and carries explicit
   * `valid_row` / `valid_col` operands lowered from `tile_type->tile_view_.valid_shape`
   * when present, falling back to `tile_type->shape_` otherwise. Head-declared
   * control-flow buffers may request the physical shape so their declaration
   * does not reference a body-local valid-shape SSA value; codegen restores the
   * logical valid shape at the control-flow site before the buffer is used.
   *
   * @param tile_type Tile type carrying shape/tile_view/memref metadata.
   * @param use_physical_valid_shape Use `shape_`, ignoring an explicit logical
   *        `tile_view_.valid_shape`, for the alloc operands.
   */
  AllocTileFields ComputeAllocTileFields(const std::shared_ptr<const ir::TileType>& tile_type,
                                         bool use_physical_valid_shape = false);

  /**
   * @brief The tile_buf handle already bound to the buffer `memref` denotes.
   *
   * Only meaningful under the PTOAS memory planner (`emit_tile_addr_ == false`),
   * where variables denoting the same buffer must share one handle because
   * there is no baked `addr` to alias through. Returns "" when addresses are
   * baked, when `memref` is null, or when no handle is bound yet.
   */
  [[nodiscard]] std::string TryGetSharedTileBufHandle(const ir::MemRefPtr& memref) const;

  /**
   * @brief Declare `ssa_name`'s `pto.alloc_tile` in the function head.
   *
   * The head prologue is rendered after the body and prepended, so a handle
   * declared here dominates every use — including uses inside `scf.if` branches
   * and reads after the region. Returns false (and declares nothing) when the
   * handle already has an `alloc_tile`.
   */
  bool DeclareTileBufAtHead(const std::string& ssa_name, const AllocTileFields& fields);

  /**
   * @brief Emit alloc_tile for dynamically allocated tile buffers (e.g., reshape outputs)
   */
  void EmitExtraAllocTiles();

  /**
   * @brief Decide which author-declared multi-slot allocations become ptoas
   *        multi-buffer regions, and reserve their `%mb` handles.
   *
   * `pl.MemRef(slots=N)` says "one allocation, N uniform slots, this use takes
   * slot k" — exactly what ptoas `pto.alloc_multi_tile` + `pto.multi_tile_get`
   * describe, and describing it that way is what lets ptoas plan the slots as one
   * region and derive per-slot (dynamic event id) synchronization from the slot
   * expression. Emitting N unrelated `alloc_tile`s instead throws that away.
   *
   * Runs only under the PTOAS memory planner (`emit_tile_addr_ == false`). Under
   * the PyPTO planner, ptoas runs at `--pto-level=level3`, where the fan-out of an
   * explicit base address is not constant-folded, so its slot narrowing degrades
   * to conservative aliasing — the multi-buffer form is measurably *worse* there
   * than the baked-address `alloc_tile` path (an extra false WAR pair between two
   * constant slots). See hw-native-sys/PTOAS#1106.
   *
   * A region is eligible when every tile bound to that allocation selects a slot,
   * the slots share one tile_buf type and one static valid extent, at most one of
   * them is live per loop iteration, the memory space is a local one ptoas supports
   * for multi_tile_buf (vec / mat / acc), and the count is within ptoas's `[2, 16]`.
   *
   * The one-slot-per-iteration condition is a ptoas synchronization limit, not a
   * typing one — see CoLiveSlotCollector.
   *
   * Anything else is a `ValueError` naming the shape, *not* a fallback: under this
   * planner per-slot `alloc_tile`s would leave ptoas free to plan the slots on top
   * of each other, which is the one thing the declaration exists to prevent. The
   * ordinary `alloc_tile` path is reached only when no region is planned at all —
   * under the PyPTO planner, or for an allocation that declares no slots.
   *
   * @param func The function being generated (scanned for tile phis, which take a
   *             head-declared handle a per-use slot cannot provide)
   */
  void PlanMultiBufferRegions(const ir::FunctionPtr& func);

  /**
   * @brief The multi-buffer region `memref` takes a slot of, or null.
   */
  [[nodiscard]] const MultiBufferRegion* GetMultiBufferRegion(const ir::MemRefPtr& memref) const;

  /**
   * @brief Emit `%slot = pto.multi_tile_get %mb[%k]` for a slot of a region.
   *
   * Emitted where the ordinary `alloc_tile` would be — at the tile's definition —
   * so a runtime slot index (`l0c[i % 2]`) is read inside the loop that names it.
   *
   * @return false when no region was planned for `memref`'s allocation — it
   *         declares no slots, or the PyPTO planner is in use. An allocation that
   *         declares slots this planner cannot describe never reaches here:
   *         PlanMultiBufferRegions has already raised.
   */
  bool TryEmitMultiTileGet(const ir::MemRefPtr& memref, const std::string& tile_buf, const ir::Span& span);

  /**
   * @brief Emit the `pto.alloc_multi_tile` declarations in the function head.
   */
  void EmitMultiBufferRegionAllocs();

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
    /// Symbols that appear ONLY in a tensor parameter's valid_shape, mapped to
    /// that parameter's name. Such a symbol is bound at the call site, so a
    /// precompiled kernel never receives it — read on the GetVarName failure
    /// path to name the parameter the unbindable symbol came from.
    std::map<const ir::Var*, std::string> valid_shape_symbol_origin;
    std::map<const ir::Var*, std::string> tensor_to_view;
    std::map<const ir::Var*, std::string> tensor_to_base_ptr;  ///< tensor var → base ptr SSA
    std::map<std::string, std::string>
        view_ssa_to_base_ptr;  ///< tensor_view SSA → base ptr SSA (for rebinding IfStmt phi return_vars)
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
    /// SSA names emitted as tile views (`pto.subview` / `pto.treshape`).
    std::set<std::string> tile_view_names;

    /// Tensor vars whose `CachePolicy.BYPASS` request has already been reported
    /// (pypto #2534). Keeps the diagnostic one-per-tensor instead of
    /// one-per-emitted-load. See NoteCacheBypassWarned.
    std::set<const ir::Var*> cache_bypass_warned;

    /// Eligible multi-buffer regions, keyed by the allocation's base Ptr.
    std::map<const ir::Var*, MultiBufferRegion> multi_buffer_regions;
    /// The same regions in discovery order — the map is keyed by pointer, which
    /// is not a stable order to emit declarations in.
    std::vector<const ir::Var*> multi_buffer_region_order;

    int temp_counter = 0;
    std::set<std::string> used_ssa_names;

    std::map<const ir::Var*, std::string> memref_to_var_name;  ///< keyed by base_ Ptr
    std::vector<std::pair<ir::VarPtr, std::shared_ptr<const ir::TileType>>> tile_var_allocs;
    std::set<const ir::Var*> emitted_tile_alloc_vars;
    /// PTOAS memory-planner mode only (no addr baked): full-MemRef-identity key
    /// (base+offset+size) -> canonical tile_buf SSA. Variables that resolve to
    /// the same buffer (e.g. a loop-carried accumulator coalesced by
    /// MemoryReuse) share one handle so the op writes in place and ptoas
    /// PlanMemory keeps them one buffer. Views (same base, different
    /// offset/size) get distinct keys and are never merged.
    std::map<std::string, std::string> memref_identity_to_mlir;
    /// MemRef-identity key -> the tile_buf type of the first var bound to it.
    std::map<std::string, std::string> memref_identity_type;
    /// MemRef-identity keys whose vars do NOT all share one tile_buf type — e.g.
    /// a `[1, N]` row-major op result and its `[N, 1]` col-major reshape view,
    /// which occupy the same bytes. Their shared handle carries exactly one type
    /// (differently-typed reads become `pto.treshape` views of it), so it must
    /// never be re-typed to suit another var. `TryGetSharedTileBufHandle` refuses
    /// these identities.
    std::set<std::string> memref_identity_mixed_types;
    /// alloc_tile SSA handles already emitted — dedups the alloc when several
    /// vars share one handle (PTOAS in-place aliasing).
    std::set<std::string> emitted_tile_alloc_names;

    ir::FunctionPtr current_function;
    ir::VarPtr current_result_var;
    std::string current_result_buf;
    std::shared_ptr<const ir::TileType> current_result_tile_type;

    std::string gm_slot_buffer_ssa;
    DataType gm_slot_buffer_dtype = DataType::FP32;
    std::map<std::pair<int, int>, std::string> gm_slot_buffer_region_by_pipe;
    std::set<const ir::Var*> ffts_workspace_vars;

    /// SSA name of the synthetic runtime-owned SDMA workspace pointer param.
    /// Empty when the current function does not use prefetch.make_context.
    std::string sdma_workspace_arg_ssa;

    /// Raw runtime dispatch-args pointer used by deferred completion adapters.
    std::string deferred_completion_raw_args_ssa;

    /// SSA names of the synthetic SPMD block_idx/block_num params, appended at
    /// the func.func signature tail. Empty when the current function does not
    /// use tile.get_block_idx / tile.get_block_num.
    std::string spmd_block_idx_arg;
    std::string spmd_block_num_arg;

    /// SSA name of the synthetic SPMD subblock_idx (AIV lane) param.
    /// Empty when the current function does not use tile.get_subblock_idx.
    std::string spmd_subblock_idx_arg;

    /// Mapping from DistributedTensor parameter Var → CommContext pointer
    /// arg SSA name. Populated in GenerateFunction when appending the
    /// trailing ``!pto.ptr<i64>`` ctx params. Consumed by
    /// pld.system.get_comm_ctx / pld.tile.remote_load / pld.tensor.put /
    /// pld.system.notify / pld.system.wait codegen
    /// to recover the per-tensor CommContext pointer.
    std::map<const ir::Var*, std::string> dist_tensor_to_ctx;

    std::string current_expr_value;
    std::vector<std::string> yield_buffer;

    void Reset() {
      constants_section.str("");
      constants_section.clear();
      body_section.str("");
      body_section.clear();
      constants_indent.clear();

      var_to_mlir.clear();
      valid_shape_symbol_origin.clear();
      tensor_to_view.clear();
      tensor_to_base_ptr.clear();
      view_ssa_to_base_ptr.clear();
      memref_to_mlir.clear();
      var_to_memref.clear();
      memref_to_tile_type.clear();

      emitted_numeric_constants.clear();

      extra_alloc_tiles.clear();
      ssa_to_tile_buf_type.clear();
      subview_materializations.clear();
      tile_view_names.clear();
      cache_bypass_warned.clear();

      temp_counter = 0;
      used_ssa_names.clear();

      memref_to_var_name.clear();
      tile_var_allocs.clear();
      emitted_tile_alloc_vars.clear();
      memref_identity_to_mlir.clear();
      memref_identity_type.clear();
      memref_identity_mixed_types.clear();
      emitted_tile_alloc_names.clear();
      multi_buffer_regions.clear();
      multi_buffer_region_order.clear();

      current_function.reset();
      current_result_var.reset();
      current_result_buf.clear();
      current_result_tile_type = nullptr;

      gm_slot_buffer_ssa.clear();
      gm_slot_buffer_dtype = DataType::FP32;
      gm_slot_buffer_region_by_pipe.clear();
      ffts_workspace_vars.clear();

      sdma_workspace_arg_ssa.clear();
      deferred_completion_raw_args_ssa.clear();
      spmd_block_idx_arg.clear();
      spmd_block_num_arg.clear();
      spmd_subblock_idx_arg.clear();
      dist_tensor_to_ctx.clear();

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

  /// True when the module needs the wrapper-defined counter-completion adapter.
  bool needs_deferred_completion_adapter_ = false;

  const backend::Backend* backend_;  ///< Backend instance for querying op info

  /// When false, `pto.alloc_tile` omits the physical `addr` operand so the
  /// ptoas PlanMemory pass owns allocation (--pto-level=level2). Set by Generate.
  bool emit_tile_addr_ = true;

  /// When false, no operation carries a trailing `loc(...)`. Set by Generate.
  bool emit_source_loc_ = true;

  /// Source span attached to the operations being emitted right now; null means
  /// "no location", which makes Emit() behave exactly as it did before locations
  /// existed. Points into IR-owned storage — see SpanScope.
  const ir::Span* current_span_ = nullptr;

  /**
   * @brief RAII guard binding the source location of every op emitted while alive
   *
   * Set at two levels: once per statement (PTOCodegen::VisitStmt) and once more
   * per Call when the Call's own span is a genuine refinement
   * (PTOCodegen::VisitExpr_(CallPtr)). Restores the previous span on scope exit,
   * so nesting composes.
   *
   * A null @p span leaves the enclosing scope's location in place, which lets
   * callers express "refine only if the span is trustworthy" without an
   * `std::optional<SpanScope>`.
   */
  class SpanScope {
   public:
    SpanScope(PTOCodegen* codegen, const ir::Span* span) : codegen_(codegen), saved_(codegen->current_span_) {
      if (span != nullptr) codegen_->current_span_ = span;
    }
    ~SpanScope() { codegen_->current_span_ = saved_; }

    /// `current_span_` is a non-owning pointer, so a temporary would dangle.
    SpanScope(PTOCodegen* codegen, ir::Span&& span) = delete;
    SpanScope(const SpanScope&) = delete;
    SpanScope& operator=(const SpanScope&) = delete;

   private:
    PTOCodegen* codegen_;
    const ir::Span* saved_;
  };

  /**
   * @brief Trailing MLIR location for the currently bound span
   *
   * @return `" loc(\"file\":line:col)"`, or an empty string when locations are
   *         disabled or no usable span is bound.
   */
  [[nodiscard]] std::string LocSuffix() const;

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
