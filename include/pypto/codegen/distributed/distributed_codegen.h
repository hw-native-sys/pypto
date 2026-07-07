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

#ifndef PYPTO_CODEGEN_DISTRIBUTED_DISTRIBUTED_CODEGEN_H_
#define PYPTO_CODEGEN_DISTRIBUTED_DISTRIBUTED_CODEGEN_H_

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/codegen/code_emitter.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

struct BuiltinNextLevelSpec {
  std::string op_name;
  std::string variant;
  std::string entry_symbol;
  std::string template_dir;
  std::map<std::string, std::string> template_vars;
};

/**
 * @brief Distributed code generator for simpler runtime Python orchestration
 *
 * Generates Python source code that uses the simpler distributed runtime API
 * (orch.submit_next_level, orch.submit_sub) from PyPTO IR programs
 * that have been lowered through OutlineHierarchyScopes.
 *
 * Call-site lowering: infers Python dispatch pattern from callee function metadata:
 * - CHIP-level Worker functions -> orch.submit_next_level(callable, task_args, config)
 * - HOST-level Worker functions -> orch.submit_sub(callable_id, task_args)
 * - Orchestrator functions -> nested orchestrator call
 */
class DistributedCodegen : public CodegenBase {
 public:
  DistributedCodegen() = default;

  /**
   * @brief Generate distributed Python code from a Program
   *
   * @param program IR Program (after OutlineHierarchyScopes)
   * @return Complete Python source code as a string
   */
  [[nodiscard]] std::string Generate(const ir::ProgramPtr& program);

  // CodegenBase interface
  [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_target_var_; }
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override;
  int64_t GetConstIntValue(const ir::ExprPtr& expr) const override;
  std::string GetVarName(const ir::VarPtr& var) const override;

  /// Public hook for op codegen functions registered via
  /// :c:macro:`REGISTER_DISTRIBUTED_OP`. Marks @p var_name as emitted so the
  /// surrounding ``AssignStmt`` visitor does not re-emit ``var = ...`` on
  /// the fall-through path. Mirrors the manual ``declared_vars_.insert(...)``
  /// performed by in-class emitters such as :func:`EmitTensorCreate`.
  void MarkDeclared(const std::string& var_name) { declared_vars_.insert(var_name); }

  /// Return builtin chip-callable variants requested while emitting the HOST
  /// orchestrator. Python's backend materializes these specs under
  /// ``next_levels/<variant>/`` so runtime assembly can keep scanning the
  /// existing chip-callable layout.
  [[nodiscard]] const std::vector<BuiltinNextLevelSpec>& GetBuiltinNextLevelSpecs() const {
    return builtin_next_level_specs_;
  }

  /// Mark a builtin variant as seen in this codegen run. Returns true only for
  /// the first sighting, letting registered op handlers de-duplicate
  /// ``next_levels/<variant>/`` materialization.
  [[nodiscard]] bool MarkBuiltinEmitted(const std::string& variant);

  /// Record a materializable builtin next-level callable variant.
  void RecordBuiltinNextLevel(const ir::CallPtr& call, const std::string& variant,
                              std::map<std::string, std::string> template_vars);

  /// Return a fresh TaskArgs variable name for registered distributed op
  /// codegen functions that emit their own submit path.
  [[nodiscard]] std::string NextTaskArgsVar();

  /// Resolve a dispatch call's ``device=`` attr to a Python rank expression.
  [[nodiscard]] std::string ResolveRankExpr(const ir::CallPtr& call) const;

  /// Lower a ``CommCtxType`` call argument. get_comm_ctx-derived locals map to
  /// the runtime device_ctx expression for the current dispatch rank; explicit
  /// CommCtx params are forwarded as scalar names.
  [[nodiscard]] std::string ResolveCommCtxArg(const ir::ExprPtr& arg, const std::string& rank_expr,
                                              const ir::Span& span) const;

  /// Return the emitted Python handle variable for the comm domain that owns
  /// ``wb``.
  [[nodiscard]] std::string GetCommDomainHandleVar(const ir::WindowBufferPtr& wb) const;

  /// Format a DistributedTensor's shape as a Python tuple literal.
  [[nodiscard]] std::string FormatShapeTuple(const std::vector<ir::ExprPtr>& shape);

  /// Return a Python-safe identifier derived from an IR/runtime name hint.
  [[nodiscard]] std::string SanitizeName(const std::string& name) const;

  /// Map a PyPTO :class:`DataType` to the matching ``simpler.task_interface.DataType``
  /// enum name (e.g. FP32 -> "FLOAT32", INT32 -> "INT32").
  [[nodiscard]] static std::string DataTypeToSimplerEnum(const DataType& dtype);

 protected:
  // Statement visitors
  void VisitStmt_(const ir::AssignStmtPtr& op) override;
  void VisitStmt_(const ir::EvalStmtPtr& op) override;
  void VisitStmt_(const ir::ReturnStmtPtr& op) override;
  void VisitStmt_(const ir::ForStmtPtr& op) override;
  void VisitStmt_(const ir::IfStmtPtr& op) override;
  void VisitStmt_(const ir::SeqStmtsPtr& op) override;
  void VisitStmt_(const ir::CommDomainScopeStmtPtr& op) override;

  // Expression visitors
  void VisitExpr_(const ir::CallPtr& op) override;
  void VisitExpr_(const ir::VarPtr& op) override;
  void VisitExpr_(const ir::ConstIntPtr& op) override;
  void VisitExpr_(const ir::ConstFloatPtr& op) override;
  void VisitExpr_(const ir::ConstBoolPtr& op) override;

  // Binary arithmetic expression visitors (src/codegen/distributed/distributed_scalar_expr_codegen.cpp)
  void VisitExpr_(const ir::AddPtr& op) override;
  void VisitExpr_(const ir::SubPtr& op) override;
  void VisitExpr_(const ir::MulPtr& op) override;
  void VisitExpr_(const ir::FloorDivPtr& op) override;
  void VisitExpr_(const ir::FloorModPtr& op) override;
  void VisitExpr_(const ir::FloatDivPtr& op) override;
  void VisitExpr_(const ir::PowPtr& op) override;
  void VisitExpr_(const ir::MinPtr& op) override;
  void VisitExpr_(const ir::MaxPtr& op) override;

  // Comparison expression visitors
  void VisitExpr_(const ir::EqPtr& op) override;
  void VisitExpr_(const ir::NePtr& op) override;
  void VisitExpr_(const ir::LtPtr& op) override;
  void VisitExpr_(const ir::LePtr& op) override;
  void VisitExpr_(const ir::GtPtr& op) override;
  void VisitExpr_(const ir::GePtr& op) override;

  // Logical expression visitors
  void VisitExpr_(const ir::AndPtr& op) override;
  void VisitExpr_(const ir::OrPtr& op) override;
  void VisitExpr_(const ir::XorPtr& op) override;

  // Bitwise expression visitors
  void VisitExpr_(const ir::BitAndPtr& op) override;
  void VisitExpr_(const ir::BitOrPtr& op) override;
  void VisitExpr_(const ir::BitXorPtr& op) override;
  void VisitExpr_(const ir::BitShiftLeftPtr& op) override;
  void VisitExpr_(const ir::BitShiftRightPtr& op) override;

  // Unary expression visitors
  void VisitExpr_(const ir::NegPtr& op) override;
  void VisitExpr_(const ir::NotPtr& op) override;
  void VisitExpr_(const ir::BitNotPtr& op) override;
  void VisitExpr_(const ir::AbsPtr& op) override;
  void VisitExpr_(const ir::CastPtr& op) override;

 private:
  // Code structure emission
  void EmitImports();
  void EmitFunction(const ir::FunctionPtr& func);
  void EmitEntryFunction();
  // Tag the emitted orchestrator/entry with a sentinel attribute so the runtime
  // resolves it by marker, not by function name (issue #1678). Keep the
  // attribute name in sync with `_ENTRY_MARKER` in distributed_runner.py.
  void EmitEntryMarker(const std::string& func_name);

  // Call-site lowering
  void EmitCallToWorker(const ir::CallPtr& call, const ir::FunctionPtr& callee);
  /**
   * @brief Emit a same-level worker / next-level orchestrator call if @p expr
   *        is one. Returns true if it emitted; false if @p expr is not a
   *        hierarchy call (caller should fall back to standard lowering).
   *        Triggers UNREACHABLE if the call targets an invalid level/role.
   */
  bool TryEmitHierarchyCall(const ir::ExprPtr& expr);
  void EmitDistIntrinsic(const ir::CallPtr& call);
  void EmitTreeReduce(const ir::CallPtr& call);
  void EmitTensorCreate(const ir::CallPtr& call);

  // Pre-init allocation hoisting for HOST orchestrator. tensor.create
  // statements at the top level of the HOST orchestrator body are emitted
  // into a separate `_alloc_intermediates(tensors)` Python function so the
  // simpler runtime can populate shared-memory tensors *before* w.init()
  // forks subworker / chip-worker child processes. Allocations made after
  // fork are not visible to inherited children.
  void CollectHostOrchHoistableAllocs(const ir::FunctionPtr& host_orch);
  void EmitAllocIntermediatesFunction(const ir::FunctionPtr& host_orch);

  /// Collect AssignStmt defs from a HOST orchestrator body so comm-slot size
  /// lowering can unwrap CSE/SSA temps (e.g. ``t = pld.system.world_size()``)
  /// before the body walk reaches ``VisitStmt_(CommDomainScopeStmtPtr)``, which
  /// emits the ``with orch.allocate_domain(...)`` line ahead of any inner
  /// AssignStmt — so referenced temps are not yet bound in the emitted Python
  /// when the scope's ``window_size`` / ``CommBufferSpec`` lines are written.
  void CollectHostOrchVarDefs(const ir::FunctionPtr& func);

  /// Emit ``<dim> = tensors["<param>"].shape[<i>]`` (or an inverted affine
  /// form) at the top of a HOST-orchestrator body for every ``pl.dynamic()``
  /// shape dim carried by a tensor parameter. Without this, a per-rank host
  /// slice that references a dynamic dim (e.g. ``tensors["x"][r, 0:M, 0:N]``)
  /// emits the bare symbol ``M`` with no binding, raising ``NameError`` at
  /// runtime (#1873). Mirrors the device-side ``_append_dynamic_dim_unpacking``,
  /// sharing ``CollectVarsFromShapeExpr`` as the single source of truth so host
  /// and device recover dims in lockstep.
  void EmitHostOrchDynamicDimBindings(const ir::FunctionPtr& func);

  /// Return a Python expression that recovers ``target_var`` from
  /// ``shape_access`` (e.g. ``tensors["x"].shape[1]``), or an empty string if
  /// the shape dim is not invertible for that var. Supports ``var`` and the
  /// single-var affine forms ``var +/- c``, ``c - var``, ``var * c`` / ``c *
  /// var`` (-> ``shape // c``) and ``var // c`` (-> ``shape * c``). Python
  /// mirror of the device-side ``_invert_shape_dim_for_var`` (integer ``//``
  /// keeps slice bounds int-typed).
  [[nodiscard]] std::string InvertShapeDimForVar(const ir::ExprPtr& dim_expr, const ir::VarPtr& target_var,
                                                 const std::string& shape_access) const;

  /// Lower a comm-domain slot ``size_`` expression for ``window_size`` /
  /// ``CommBufferSpec`` emission, unwrapping hoisted scalar temps via
  /// ``host_orch_var_defs_``.
  [[nodiscard]] std::string GetCommSlotSizeAsCode(const ir::ExprPtr& size_expr);

  // Scalar-expression Python-emission helpers (see distributed_scalar_expr_codegen.cpp).
  // Each writes the rendered Python expression into ``current_expr_value_``.
  void EmitInfixBinaryOp(const ir::BinaryExprPtr& op, const char* symbol);
  void EmitCallStyleBinaryOp(const ir::BinaryExprPtr& op, const char* func_name);
  void EmitUnaryPrefixOp(const ir::UnaryExprPtr& op, const char* prefix);
  void EmitUnaryCallOp(const ir::UnaryExprPtr& op, const char* func_name);

  // Helpers
  void RegisterParamsAndEmitScalarBindings(const ir::FunctionPtr& func);
  [[nodiscard]] std::string ParamDirectionToTensorArgType(ir::ParamDirection dir) const;
  [[nodiscard]] std::vector<ir::FunctionPtr> SortFunctionsByRoleAndLevel() const;
  void ClassifyFunctions();
  std::string FormatArgs(const std::vector<ir::ExprPtr>& args);
  [[nodiscard]] bool IsSubWorker(const ir::FunctionPtr& func) const;
  [[nodiscard]] static std::string DataTypeToPythonDType(const DataType& dtype);

  /// Look up the innermost open CommDomainScopeStmt whose ``slots_`` contain
  /// ``wb`` (by shared_ptr identity — ``MaterializeCommDomainScopes`` shares
  /// the same ``shared_ptr<const WindowBuffer>`` between the scope's slots
  /// and every consuming ``DistributedTensorType::window_buffer_``).
  /// Triggers INTERNAL_CHECK if ``wb`` is not a slot of any open scope —
  /// that indicates either the pass did not run or codegen visited a
  /// dispatch outside its enclosing comm-domain scope.
  [[nodiscard]] ir::CommDomainScopeStmtPtr ScopeForWindowBuffer(const ir::WindowBufferPtr& wb) const;

  ir::ProgramPtr program_;
  CodeEmitter emitter_;

  // Stack of currently-open ``CommDomainScopeStmt``s, pushed on entry to
  // ``VisitStmt_(CommDomainScopeStmtPtr)`` and popped on exit. Used by
  // ``EmitCallToWorker`` to route each ``DistributedTensor`` arg to the
  // matching ``__<name_hint>`` handle var: scan inner-to-outer for the
  // first scope whose ``slots_`` contain the arg's
  // ``DistributedTensorType::window_buffer_``. Nesting depth equals the
  // number of comm groups in the host_orch (small, bounded), so the
  // linear scan stays O(depth) per query — well within the O(N log N)
  // pass-complexity bound.
  std::vector<ir::CommDomainScopeStmtPtr> comm_domain_stack_;

  // Function classification
  std::map<std::string, ir::FunctionPtr> workers_;
  std::map<std::string, ir::FunctionPtr> orchestrators_;
  ir::FunctionPtr entry_func_;
  std::map<std::string, ir::FunctionPtr> all_funcs_;
  std::set<int> used_levels_;

  // Per-function state
  ir::FunctionPtr current_func_;
  std::string current_target_var_;
  std::string current_expr_value_;
  std::set<std::string> declared_vars_;
  bool is_worker_context_{false};
  int task_args_counter_{0};  // Counter for generating unique TaskArgs variable names

  // HOST orchestrator alloc-hoisting state. Populated by
  // CollectHostOrchHoistableAllocs() before EmitFunction() runs on the HOST
  // orchestrator; consulted by VisitStmt_(AssignStmt) to skip tensor.create
  // assignments that have already been emitted in _alloc_intermediates.
  std::unordered_set<const ir::AssignStmt*> hoisted_allocs_;
  bool host_orch_body_after_hoist_{false};

  // Tuple-return support: maps (tuple_tmp_var_name, element_index) to the
  // actual Out/InOut parameter tensor name in tensors[...].  Populated by
  // EmitCallToWorker when the callee has a TupleType return; consumed by
  // VisitStmt_(AssignStmt) when it encounters TupleGetItemExpr unpacking.
  std::map<std::pair<std::string, int>, std::string> tuple_element_tensors_;

  // HOST orchestrator AssignStmt defs, populated before comm-domain emission.
  std::unordered_map<const ir::Var*, ir::ExprPtr> host_orch_var_defs_;
  bool unwrap_hoisted_var_refs_{false};

  std::unordered_set<std::string> emitted_builtin_variants_;
  std::vector<BuiltinNextLevelSpec> builtin_next_level_specs_;
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_DISTRIBUTED_DISTRIBUTED_CODEGEN_H_
