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

#ifndef SRC_CODEGEN_ORCHESTRATION_ORCHESTRATION_STMT_CODEGEN_H_
#define SRC_CODEGEN_ORCHESTRATION_ORCHESTRATION_STMT_CODEGEN_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "pypto/codegen/code_emitter.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/codegen/orchestration/orchestration_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using GraphConfigScalarSlots = std::map<std::string, std::unordered_set<size_t>>;

std::string GraphFunctionSymbol(const std::string& func_name);

// Shared by the entry and Graph fragments: IDs describe the final tensor ABI,
// while code_name identifies the binary compiled once for all registrations.
class KernelRegistrations {
 public:
  int GetOrCreate(const std::string& name, ir::CoreType core_type, std::vector<std::string> signature);

  std::vector<KernelRegistration> entries;

 private:
  std::map<std::tuple<std::string, ir::CoreType, std::vector<std::string>>, int> ids_;
};

// Private statement visitor shared by entry and Graph emission. Implementations
// are grouped by control flow, dependencies, dispatch, and tensor lifetime.
// All groups operate on this same instance: scope snapshots, emit names, and
// TaskId bindings must stay synchronized across recursive statement visits.
class OrchestrationStmtCodegen : public CodegenBase {
 public:
  using ManualTaskIdBinding = std::variant<int, std::string, std::vector<std::string>>;

  /// One task dependency edge plus its provenance. ``user_written`` marks an
  /// edge the user spelled as ``deps=[...]`` (``kAttrManualDepEdges``, or the
  /// typed ``Submit::deps_`` projected through ``SubmitToCallView``) as opposed
  /// to a compiler-derived hazard patch (``kAttrCompilerManualDepEdges``).
  /// Only the former is an error when it fails to resolve — see
  /// ``ResolveDepEdgeBinding``.
  struct DepEdge {
    ir::VarPtr var;
    bool user_written;
  };

  /// Backing storage of one TaskId array carry — see ``array_carry_vars_``.
  struct ArrayCarryEntry {
    std::string array_name;
    int64_t size;
  };

  explicit OrchestrationStmtCodegen(const ir::ProgramPtr& prog, KernelRegistrations* registrations,
                                    std::unordered_map<const ir::Var*, std::string> param_to_emit_name,
                                    std::set<std::string> param_name_set,
                                    std::map<std::string, int> param_name_to_orch_index,
                                    std::unordered_map<std::string, std::string> dist_param_to_ctx_param);

  void SetCallTupleElements(const std::map<std::string, std::vector<TupleElement>>& elements);

  void SetTupleVarToKey(std::map<const ir::Var*, std::string> mapping);

  void SetInitialIndent(int indent);
  void SetTaskVarPrefix(std::string prefix);

  void SetGraphScalarParams(const std::vector<ir::VarPtr>& params);

  const std::unordered_set<size_t>& GetGraphScalarValueReads() const;

  void SetGraphConfigScalarSlots(GraphConfigScalarSlots slots);

  void ReserveDeclaredNames(const std::set<std::string>& names);

  std::string ReserveSyntheticEmitName(const std::string& base_name);

  void SetEffectiveUses(std::unordered_set<const ir::Var*> uses);
  [[nodiscard]] bool NeedsVectorInclude() const;

  void PrepareCrossScopeTaskIdHoists(const ir::StmtPtr& body);

  std::string GetGeneratedCode() const;
  [[nodiscard]] std::string GetCurrentResultTarget() const override;
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override;
  int64_t GetConstIntValue(const ir::ExprPtr& expr) const override;
  std::string GetVarName(const ir::VarPtr& var) const override;
  [[nodiscard]] std::string TryGetVarName(const ir::ExprPtr& expr) const override;
  [[nodiscard]] std::string GenerateExprString(const ir::ExprPtr& expr) const override;
  [[nodiscard]] std::string GetTensorShapeDim(const std::string& name, int64_t axis) const override;

  [[nodiscard]] std::string GetTensorCreateSizeExpr(const std::string& result_var,
                                                    const std::string& default_dim_expr) const override;

  std::vector<ir::VarPtr> GetCompilerDependencyEdges(const ir::CallPtr& call) const;

  std::vector<const ir::Var*> CollectCompilerDepArrayBarrierEdges(const ir::ForStmtPtr& for_stmt) const;

  std::string EmitCompilerDepArrayBarrier(const std::vector<std::string>& names);

  struct DynamicTaskIdCollection {
    std::string data_name;
    std::string count_name;
  };

  std::string EmitDynamicCompilerDepBarrier(const DynamicTaskIdCollection& collection);

  void VisitStmt_(const ir::ForStmtPtr& for_stmt) override;

  void VisitStmt_(const ir::RuntimeScopeStmtPtr& scope) override;

  void MarkArrayReturnVars(const ir::IfStmtPtr& if_stmt);

  void RecordArrayPhiYield(const ir::VarPtr& rv, const ir::ExprPtr& yield_value, const ir::Span& span);

  void InstallArrayPhiBindings(const ir::IfStmtPtr& if_stmt);

  void VisitStmt_(const ir::IfStmtPtr& if_stmt) override;

  void VisitStmt_(const ir::AssignStmtPtr& assign) override;

  void VisitStmt_(const ir::ReturnStmtPtr& ret) override;

  void VisitStmt_(const ir::SeqStmtsPtr& seq) override;

  void VisitStmt_(const ir::YieldStmtPtr& yield_stmt) override;

  void VisitStmt_(const ir::EvalStmtPtr& eval) override;

 private:
  class IndentGuard {
   public:
    explicit IndentGuard(CodeEmitter& emitter) : emitter_(emitter) { emitter_.IncreaseIndent(); }
    ~IndentGuard() { emitter_.DecreaseIndent(); }

   private:
    CodeEmitter& emitter_;
  };

  CodeEmitter& Active();
  const CodeEmitter& Active() const;

  void EmitIndentedLine(const std::string& line);

  void EmitBlankLine();

  void EmitForLoopHeader(const std::string& loop_var, const std::string& start_expr,
                         const std::string& stop_expr, const std::string& step_expr, const ir::ExprPtr& step);

  void EmitArrayCopyLoop(int64_t extent, const std::string& dst_array, const std::string& src_array,
                         const std::string& index_var = "__i");

  void EmitArrayFillLoop(int64_t extent, const std::string& dst_array, const std::string& value_expr,
                         const std::string& index_var = "__i");

  std::string IndentAtLevel(int level) const;

  std::string GetCppType(const ir::TypePtr& type);

  std::string EncodeScalarVar(const ir::ExprPtr& expr, const std::string& var_name,
                              const std::string& cpp_type) const;

  static std::string EncodeScalarConst(const std::string& value, const std::string& cpp_type);

  [[nodiscard]] std::string GetExternalTensorName(const std::string& name) const override;

  static const char* ArgDirectionToMethodName(ir::ArgDirection dir);

  struct ParamEntry {
    ir::ArgDirection direction;
    std::string value;
    /// True when this arg's Var is listed in the Call's ``kAttrDumpVars`` set
    /// (selective dump from ``pl.dump_tag`` / ``dumps=``). Set by
    /// VarPtr identity in the param builders — no name comparison.
    bool dump = false;
  };

  static std::vector<ParamEntry> ReorderTensorsBeforeScalars(std::vector<ParamEntry> params);

  ParamEntry BuildOneArgParam(const ir::CallPtr& call, const std::string& callee_name,
                              const std::vector<ir::ArgDirection>& call_arg_directions, size_t arg_idx);

  std::set<const ir::Var*> CollectDumpVarSet(const ir::CallPtr& call);

  size_t CountTrailingCommCtxParams(const ir::FunctionPtr& callee_func) const;

  std::vector<std::string> CollectSelectiveDumpValues(const std::vector<ParamEntry>& params);

  void EmitSelectiveDumpCall(const std::string& task_var, const std::vector<ParamEntry>& params);

  ParamEntry EmitSubmitSynthOutputEntry(const ir::FunctionPtr& callee_func, size_t param_idx);

  static const char* ArgDirectionToRuntimeName(ir::ArgDirection dir);

  int RegisterKernel(const std::string& name, ir::CoreType core_type, const std::vector<ParamEntry>& params);

  std::vector<ParamEntry> BuildTaskParams(const ir::CallPtr& call, const ir::FunctionPtr& callee_func);

  void GenerateTensorOpCode(const ir::CallPtr& call, const std::string& result_var,
                            const ir::VarPtr& assign_var);

  struct GroupCalleeInfo {
    std::string aic_name;
    std::string aiv_name;
    ir::CallPtr inner_call;        // The call to the InCore kernel inside the Group body
    ir::FunctionPtr inner_callee;  // The InCore function being called
  };

  struct WrapperCallInfo {
    ir::CallPtr inner_call;
    ir::FunctionPtr inner_callee;
  };

  /// Bridges the two-level nesting that arises when a ``Group`` is dispatched
  /// *through* a ``Spmd`` wrapper (``GenerateSpmdCallCode`` ->
  /// ``GenerateGroupCallCode``). In that case the function ``outer_call``
  /// actually invokes is the Spmd wrapper, NOT the Group that
  /// ``BuildWrapperReorderedParams`` receives as ``wrapper_func``. The two can
  /// have different param counts: the Spmd outliner deduplicates an aliased
  /// arg (the same buffer passed as both an input and the output, e.g.
  /// ``self.kernel(out, b, bias, out)``) into a single wrapper param, while the
  /// Group keeps every kernel param. Without the bridge, codegen would index
  /// ``outer_call->args_[<group_param_idx>]`` out of bounds.
  ///
  /// ``bridge_call`` is the Group call inside the Spmd-wrapper body
  /// (``FindFirstInnerCall(spmd_func).inner_call``); its args are positionally
  /// 1:1 with the Group's params and reference the Spmd-wrapper params,
  /// complete-parameter SSA aliases, or constants. ``bridge_func`` is the Spmd wrapper, whose params are
  /// positionally 1:1 with ``outer_call->args_``. An empty ``bridge_func``
  /// means "no bridge" (plain-Spmd / direct-Group): ``wrapper_func`` IS the
  /// function ``outer_call`` invokes, and the legacy 1-hop lookup is correct.
  struct WrapperBridge {
    ir::CallPtr bridge_call;
    ir::FunctionPtr bridge_func;
  };

  WrapperCallInfo FindWrapperInnerCall(const ir::FunctionPtr& wrapper_func);

  GroupCalleeInfo FindGroupCallees(const ir::FunctionPtr& group_func);

  std::vector<ParamEntry> BuildWrapperReorderedParams(const ir::CallPtr& outer_call,
                                                      const ir::FunctionPtr& wrapper_func,
                                                      const ir::CallPtr& inner_call,
                                                      const WrapperBridge& bridge = {});

  [[nodiscard]] static std::string RenderAvailableCoreCount(const ir::CallPtr& call);

  [[nodiscard]] std::string RenderLaunchCoreNum(const ir::ExprPtr& expr) const;

  [[nodiscard]] std::pair<ir::ExprPtr, bool> EffectiveLaunchSpec(const ir::CallPtr& call,
                                                                 const ir::FunctionPtr& launch_func) const;

  void EmitLaunchSpec(const std::string& task_var, const ir::ExprPtr& core_num_expr, bool sync_start);

  void EmitEarlyResolveHint(const std::string& task_var, const ir::CallPtr& call);

  // Selective device timing (pl.submit(..., timing_slot=N)). The runtime folds
  // every task tagged with a slot into one device-domain dispatch-to-finish span.
  // The parser validates the user-facing literal; retain this codegen guard for
  // hand-built or deserialized IR that bypasses the parser.
  // Mirrors the runtime's ``NUM_TASK_TIMING_SLOTS`` in ``device_phase.h``.
  static constexpr int kRuntimeNumTaskTimingSlots = 16;

  void EmitTaskTimingSlot(const std::string& task_var, const ir::CallPtr& call);

  // Mirrors the runtime's ``MAX_TENSOR_DIMS`` — the capacity of the fixed
  // ``uint32_t indices[]`` array in ``CorePredicateOperand``.
  static constexpr size_t kRuntimeMaxTensorDims = 5;

  static const char* RenderPredicateOp(ir::ObjectKind kind, bool flipped);

  [[nodiscard]] static ir::ExprPtr StripCast(const ir::ExprPtr& e);

  void EmitPredicateHint(const std::string& task_var, const ir::CallPtr& call);

  void EmitTaskSubmitAndBind(const std::string& submit_expr, bool capture_outputs);

  std::vector<DepEdge> GetDependencyEdges(const ir::CallPtr& call) const;

  const ManualTaskIdBinding* ResolveDepEdgeBinding(const DepEdge& edge, const ir::CallPtr& call) const;

  void CollectCompilerDepTaskIds(const ir::ProgramPtr& program);

  static bool ShouldCaptureTaskOutputs(const ir::CallPtr& call, bool capture_plain_task_id);

  bool NeedsCompilerDepTaskId(const ir::Var* var) const;

  std::vector<const ir::Var*> CompilerDepOutputArgs(const ir::CallPtr& call) const;

  size_t CountManualDeps(const std::vector<DepEdge>& edges, const ir::CallPtr& call) const;

  void EmitTaskParamsDecl(const std::string& task_var, const std::string& args_type = "CoreTaskArgs");

  struct TaskDispatchPlan {
    std::string comment;
    std::string task_var;
    std::vector<std::string> pre_lines;
    std::vector<ParamEntry> params;
    ir::CallPtr call;
    ir::ExprPtr launch_core_num;
    bool launch_sync_start{false};
    std::string submit_expr;
    bool capture_outputs{false};
    /// Direct kernel calls emit ``set_dependencies`` *before* the launch spec;
    /// wrapper paths (Spmd/Group/Mixed) emit it *after*. This mirrors the
    /// historical per-path emission order so generated orchestration C++ is
    /// byte-identical across the refactor. See ``BuildDirectCallDispatchPlan``.
    bool deps_before_launch{false};
    /// Runtime operand checks emitted ahead of the task (see
    /// ``BuildContiguityGuardLines``); empty for every kernel without
    /// ``kAttrRequireContiguousOperands``, so other dispatches are unchanged.
    std::vector<std::string> guard_lines;

    /// Emit the full task-dispatch sequence using the supplied codegen's emit surface.
    /// Consumes ``*this`` (rvalue-reference) — the plan is moved-from after emission.
    void Emit(OrchestrationStmtCodegen& cg) &&;
  };

  std::string CurrentTaskVarName() const;

  TaskDispatchPlan BuildTaskDispatchPlan(const std::string& comment, const ir::CallPtr& call,
                                         std::vector<ParamEntry>&& params, const ir::ExprPtr& launch_core_num,
                                         bool launch_sync_start, const std::string& submit_expr,
                                         bool capture_outputs, std::vector<std::string>&& pre_lines = {});

  /// A submit expression plus the lines that must precede it (emitted by
  /// ``TaskDispatchPlan::Emit`` between the ``add_*`` params and the launch spec).
  struct SubmitEmission {
    std::string expr;
    std::vector<std::string> pre_lines;
  };

  SubmitEmission BuildSingleKernelSubmit(ir::CoreType core_type, const ir::FunctionPtr& kernel_func,
                                         int func_id, const std::string& task_var);

  TaskDispatchPlan BuildDirectCallDispatchPlan(const ir::CallPtr& call, const ir::FunctionPtr& callee_func,
                                               const std::string& callee_name, ir::CoreType core_type,
                                               int func_id, std::vector<ParamEntry>&& params,
                                               bool capture_plain_task_id);

  TaskDispatchPlan BuildSpmdCallDispatchPlan(const ir::CallPtr& call, const ir::FunctionPtr& spmd_func,
                                             const ir::FunctionPtr& kernel_func,
                                             const std::string& callee_name, ir::CoreType core_type,
                                             int func_id, std::vector<ParamEntry>&& params,
                                             bool capture_plain_task_id);

  TaskDispatchPlan BuildAivOnlyGroupDispatchPlan(const ir::CallPtr& call, const ir::FunctionPtr& launch_func,
                                                 const std::string& group_name,
                                                 const ir::FunctionPtr& aiv_func, int aiv_id,
                                                 std::vector<ParamEntry>&& params,
                                                 bool capture_plain_task_id);

  TaskDispatchPlan BuildMixedGroupDispatchPlan(const ir::CallPtr& call, const ir::FunctionPtr& launch_func,
                                               const std::string& group_name, int aic_id, int aiv_id,
                                               const ir::FunctionPtr& aiv_func,
                                               std::vector<ParamEntry>&& params, bool capture_plain_task_id);

  std::vector<std::string> BuildContiguityGuardLines(const ir::CallPtr& call,
                                                     const ir::FunctionPtr& callee_func);
  void CheckWrapperNeedsNoContiguityGuard(const ir::FunctionPtr& kernel_func, const ir::CallPtr& call);

  void EmitDepArrayInsert(const std::string& name, const std::string& deps_arr, const std::string& deps_cnt);
  void EmitManualDeps(const ir::CallPtr& call, const std::string& task_var);

  void EmitDummyTask(const ir::CallPtr& call, const std::string& tid_name);

  static constexpr size_t kMaxAllocTensorsArgs = ir::alloc_batching::kAllocTensorsArgs;

  static bool IsInjectedGMPipeCreateVar(const ir::VarPtr& var);

  int64_t GetGMPipeWorkspaceElements(const ir::FunctionPtr& root_func);

  struct GMPipeCreateUse {
    ir::FunctionPtr callee;
    std::string core_num_expr;
    ir::ExprPtr core_num_node;
  };

  [[nodiscard]] std::optional<GMPipeCreateUse> ResolveGMPipeCreateUse(const std::vector<ir::StmtPtr>& stmts,
                                                                      size_t create_stmt_idx,
                                                                      const ir::VarPtr& create_var) const;

  static bool ExprRefsAnyOf(const ir::ExprPtr& expr, const std::unordered_set<const ir::Var*>& vars);

  bool ShapeDependsOnLocalVars(const ir::CallPtr& call,
                               const std::unordered_set<const ir::Var*>& locally_defined) const;

  void EmitAllocBatch(const std::vector<std::string>& emit_names);

  void EmitBatchedAllocTensors(const std::vector<ir::StmtPtr>& stmts);

  void GenerateFunctionCallCode(const ir::CallPtr& call, const std::string& result_var,
                                bool capture_plain_task_id = false);

  void GenerateGraphCallCode(const ir::CallPtr& call, const ir::FunctionPtr& callee_func,
                             bool capture_plain_task_id);

  void GenerateSpmdCallCode(const ir::CallPtr& call, const ir::FunctionPtr& spmd_func,
                            bool capture_plain_task_id);

  void GenerateGroupCallCode(const ir::CallPtr& call, const ir::FunctionPtr& group_func,
                             const ir::FunctionPtr& launch_func, bool capture_plain_task_id,
                             const WrapperBridge& bridge = {});

  // --- Alias generation helpers ---

  std::vector<size_t> CollectOutIndices(const ir::FunctionPtr& callee);

  const std::vector<std::optional<size_t>>& GetReturnedParamIndices(const ir::FunctionPtr& callee);

  static bool IsReturnedParamMapPrecise(const std::vector<std::optional<size_t>>& ret_map,
                                        const ir::CallPtr& call);

  void EmitTensorAlias(const ir::Var* result_var, const std::string& alias_name, const ir::CallPtr& call,
                       size_t arg_idx);

  void ValidateValueUse(const ir::ExprPtr& value, const ir::Span& use_span, bool read_storage = true) const;

  void ValidateTensorStorage(size_t value, const ir::VarPtr& var, const ir::Span& use_span) const;

  bool IsEnclosingScopeValid(const std::string& name) const;

  bool IsAtRuntimeScopeBodyIndent() const;

  void RegisterMutableTensorName(const std::string& cpp_type, const std::string& emit_name);

  bool IsMutableTensorName(const std::string& name) const;

  void EmitMutableTensorCarryDecl(const std::string& name, const std::string& init_expr);

  void RecordTensorAllocation(const std::string& name);

  size_t NewTensorValue();

  size_t GetTensorValue(const std::string& name);

  void PropagateTensorFlag(size_t value, std::unordered_set<size_t>* values);

  void RecordTensorSource(size_t value, size_t source);

  void RecordTensorSource(const std::string& name, const std::string& source);

  bool IsMutableTensorNameInCurrentScope(const std::string& emit_name) const;

  void PushCppScope();

  void PopCppScope();

  void DeclareHoistedTaskIdsForScope(const ir::RuntimeScopeStmt* scope);

  void DeclareCompilerDepTaskIdsAtFirstScope();

  void DeclareHoistedTaskId(const ir::Var* var);

  static uint64_t TaskIdHoistKey(const ir::Var* var);

  static std::string CompilerDepAliasKey(const ir::Var* var);

  const ManualTaskIdBinding* ResolveManualTaskIdBinding(const ir::Var* var) const;

  std::string BindSyntheticTaskId(const ir::Var* var, const std::string& base_name,
                                  const std::string& value_expr);

  std::string BindVarTaskId(const ir::Var* var, const std::string& value_expr);

  void GenerateSingleReturnAlias(const ir::Var* result_var, const ir::CallPtr& call,
                                 const std::string& var_name);

  void GenerateTupleReturnAliases(const ir::CallPtr& call, const ir::Var* result_var);

  static bool IsSubmitCall(const ir::CallPtr& call);

  void GenerateSubmitReturnAliases(const ir::CallPtr& call, int task_idx, const ir::Var* result_var);

  void VisitScopedBranchBody(const ir::StmtPtr& body, const std::vector<ir::VarPtr>& return_vars);

  void HandleTensorAssembleAssign(const ir::AssignStmtPtr& assign, const ir::CallPtr& call);

  ir::VarPtr FindClosedScopeTaskId(const ir::ExprPtr& value) const;

  void CheckTaskIdSlotValueInScope(const ir::CallPtr& call, const std::string& array_name);

  void HandleArrayUpdateElementAssign(const ir::AssignStmtPtr& assign, const ir::CallPtr& call);

  void PropagateMakeTupleAssign(const ir::AssignStmtPtr& assign, const ir::MakeTuplePtr& make_tuple);

  std::string ReserveVarEmitName(const ir::Var* var);

  /// Keep the array carries a just-closed scope minted over storage that
  /// outlives it, when restoring the entry snapshot.
  ///
  /// ``array.update_element`` is SSA-functional: it aliases its LHS onto the
  /// input array's storage (``HandleArrayUpdateElementAssign``) and emits the
  /// C++ slot write in place, at the statement's own position. The *carry* it
  /// registers is read later though — by the enclosing loop's yield, emitted
  /// after this block's closing brace. Restoring the snapshot wholesale drops
  /// that carry, and the yield then misreads an Array value as a scalar TaskId
  /// (issue #1811 for a manual scope; the same for an AUTO ``pl.scope()``).
  ///
  /// Only carries whose backing storage outlives the block may be kept, so each
  /// scope kind supplies its own ``is_scope_local`` test over the array's emit
  /// name — a manual scope hoists its allocations and knows its local names
  /// outright; an AUTO scope hoists nothing, so storage is enclosing exactly
  /// when a pre-entry carry already named it.
  template <typename IsScopeLocal>
  void PreserveEnclosingArrayCarries(std::unordered_map<const ir::Var*, ArrayCarryEntry>* saved_array_carry,
                                     std::unordered_map<const ir::Var*, ManualTaskIdBinding>* saved_map,
                                     std::unordered_map<uint64_t, ManualTaskIdBinding>* saved_map_by_key,
                                     const IsScopeLocal& is_scope_local) {
    for (const auto& [var, entry] : array_carry_vars_) {
      if (saved_array_carry->count(var)) continue;     // outer entry — keep outer value
      if (is_scope_local(entry.array_name)) continue;  // scope-local storage — drop
      (*saved_array_carry)[var] = entry;               // enclosing-valid carry — preserve
      auto tid_it = manual_task_id_map_.find(var);     // keep its per-slot dep names in sync
      if (tid_it != manual_task_id_map_.end()) {
        (*saved_map)[var] = tid_it->second;
        (*saved_map_by_key)[TaskIdHoistKey(var)] = tid_it->second;
      }
    }
  }

  void RebuildLiveManualTaskIdEmitNames();

  void NoteLiveManualTaskIdEmitName(const std::string& name);

  void RegisterFunctionTaskIdParams();

  void RegisterArrayCarry(const ir::Var* var, const std::string& array_name, int64_t size);

  const ir::ProgramPtr& program_;
  KernelRegistrations* registrations_;
  std::unordered_map<const ir::Var*, std::string> emit_name_map_;
  // Only Graph boundary scalars are wrappers; body-local scalars hold values.
  std::unordered_map<const ir::Var*, size_t> graph_scalar_params_;
  // Value reads freeze host computation into a Graph definition. Its caller
  // must declare those slots static so changed values invalidate replay.
  mutable std::unordered_set<size_t> graph_scalar_value_reads_;
  GraphConfigScalarSlots graph_config_scalar_slots_;
  std::set<std::string> declared_var_names_;
  std::set<std::string> param_name_set_;
  std::map<std::string, int> param_name_to_orch_index_;
  CodeEmitter emitter_;
  CodeEmitter* active_emitter_ = &emitter_;
  std::string current_result_var_;
  std::vector<ir::VarPtr> current_return_vars_;
  int task_counter_ = 0;
  /// Disambiguates task-arg variables between the entry and each Graph helper,
  /// whose counters restart at 0. Empty for the entry (preserving its names).
  std::string task_var_prefix_;
  int phase_fence_barrier_counter_ = 0;
  int alloc_counter_ = 0;
  /// Depth of nested ``RuntimeScopeStmt(manual=true)``. While > 0, the codegen
  /// suppresses the implicit ``SIMPLER_SCOPE()`` wrapper around ForStmt/IfStmt
  /// bodies (the runtime forbids nesting AUTO scope inside MANUAL).
  int in_manual_scope_depth_ = 0;
  /// Map from a producer ``Var`` to the task identity to use when it appears
  /// as a ``manual_dep_edge``. Three cases:
  ///   * ``int`` value: kernel-Call LHS, the int is the task counter assigned
  ///     by ``EmitTaskSubmitAndBind``. Invariant-only: parser-built dep edges
  ///     are always ``Scalar[TASK_ID]`` Vars and resolve through the string
  ///     branch; the int branch is reserved as a tripwire for hand-built IR.
  ///   * ``std::string`` value: a ``ForStmt`` TaskId iter_arg's emit name, a
  ///     ``pl.submit`` producer-TaskId tuple element's emit name, or a
  ///     ``system.task_invalid`` (``None``) LHS emit name — a single
  ///     ``TaskId`` variable.
  ///   * ``std::vector<std::string>`` value: a ``ForStmt`` TaskId iter_arg that
  ///     carries an *array* of task ids (Parallel ForStmt with const trip
  ///     count, or Sequential ForStmt propagating an inner Parallel array
  ///     across iterations). Each element of the vector is the C++ expression
  ///     naming one slot of the array (e.g. ``out_arr[0]``, ``out_arr[1]``).
  ///     EmitManualDeps fills each valid slot into the ``<task>_deps`` array.
  /// On entry to a manual scope, snapshotted (by copy) and restored on exit
  /// so the outer entries remain visible *inside* the inner scope while the
  /// inner-scope adds — which reference C++ identifiers that die with the
  /// inner block — are discarded once the manual scope exits.
  std::unordered_map<const ir::Var*, ManualTaskIdBinding> manual_task_id_map_;
  std::unordered_map<uint64_t, ManualTaskIdBinding> manual_task_id_map_by_key_;
  /// Reverse index of scalar ``TaskId`` emit names currently present in
  /// ``manual_task_id_map_`` / ``manual_task_id_map_by_key_``. Kept in sync on
  /// registration and rebuilt after every scope restore so
  /// ``CheckTaskIdSlotValueInScope`` can test emit-name aliases in O(1).
  std::unordered_set<std::string> live_manual_task_id_emit_names_;
  /// Records the C++ array allocation backing a TaskId carry that holds an
  /// array of task ids (not a scalar). Used by ``YieldStmt`` to decide how
  /// to write into the carry:
  ///   * Parallel ForStmt rv: yield writes one slot ``arr[<loop_var>] = value``
  ///   * Sequential ForStmt rv whose body yields an array: yield copies
  ///     slot-by-slot from the inner array
  /// The key is either a ``ForStmt`` iter_arg (when used as a deps source) or
  /// a ``ForStmt`` return_var (when used as a yield target). For each key the
  /// recorded ``array_name`` is the C++ identifier of the underlying
  /// ``TaskId[N]`` array and ``size`` is the slot count ``N``.
  std::unordered_map<const ir::Var*, ArrayCarryEntry> array_carry_vars_;
  /// In-flight ArrayType ``IfStmt`` return_vars (phis). An ArrayType SSA value
  /// names one backing C-stack array rather than a copyable value, so a phi is
  /// aliased onto that array instead of being declared (see
  /// ``MarkArrayReturnVars``). An entry is added before the branches are
  /// emitted, filled in by each branch's yield, and erased once bound.
  struct ArrayPhiBinding {
    bool resolved = false;
    std::string array_name;
    std::optional<ArrayCarryEntry> carry;
    std::optional<ManualTaskIdBinding> task_id;
  };
  std::unordered_map<const ir::Var*, ArrayPhiBinding> pending_array_phis_;
  std::unordered_map<const ir::Var*, DynamicTaskIdCollection> dynamic_task_id_collections_;
  bool needs_vector_include_ = false;
  /// Names of mutable Tensor values declared in each generated C++ block.
  /// Tuple-output alias emission must avoid redeclaring names already declared
  /// in the same block, but must not treat outer-block declarations as aliases:
  /// C++ shadowing is valid and sometimes required to avoid rebinding an outer
  /// loop-carried Tensor too early.
  std::vector<std::unordered_set<std::string>> mutable_tensor_name_scopes_{{}};
  /// Buffered runtime-scope emission. Tensor handles may be hoisted in either
  /// mode; buffer allocations move only out of MANUAL scopes. Local-name sets
  /// distinguish storage in this block from storage that outlives it, including
  /// declarations hoisted out of a nested scope into this scope's body.
  struct TensorHoistFrame {
    const ir::RuntimeScopeStmt* scope;
    std::vector<std::string>* sink;
    std::set<std::string>* local_names;
    std::set<std::string>* enclosing_local_names;
    int parent_indent;
    size_t parent_cpp_scope;
    std::vector<size_t> allocations;
  };
  std::vector<TensorHoistFrame> tensor_hoist_frames_;
  std::unordered_map<std::string, size_t> tensor_values_;
  std::unordered_map<const ir::Var*, size_t> tensor_yield_values_;
  std::vector<std::unordered_set<size_t>> tensor_dependents_;
  // Operand rendering is const; remember validated reads for loop-backedge checks.
  mutable std::unordered_set<size_t> tensor_read_values_;
  std::unordered_set<size_t> invalid_tensor_storage_;
  // Validate storage behind descriptors hoisted out of AUTO scopes and their
  // aliases; retain the existing MANUAL-scope carry behavior.
  std::unordered_set<size_t> auto_scope_carry_values_;
  std::vector<std::string>* scope_hoist_sink_ = nullptr;
  bool scope_hoist_allocations_ = false;
  int scope_hoist_indent_level_ = 0;
  // Hoisted names are removed from each local set before its AUTO scope closes.
  std::unordered_set<std::string> closed_auto_scope_names_;
  std::set<std::string>* scope_local_names_ = nullptr;
  std::set<std::string>* enclosing_scope_local_names_ = nullptr;
  /// Original body indent for each hoisted carry/phi. Copies in nested loop
  /// bodies must remain snapshots even if entering another runtime scope changes
  /// the current hoist sink. Names are globally unique across the function.
  std::unordered_map<std::string, int> hoisted_carry_body_indents_;
  /// Stack of 0-based slot expressions for the enclosing ForStmts. Pushed
  /// when entering a ForStmt body and popped on exit. Used by ``YieldStmt``
  /// to emit ``arr[<slot>] = value`` for Parallel inner array writes. The
  /// expression is ``(loop_var - start) / step``, peephole-simplified to
  /// just ``loop_var`` when start=0 and step=1.
  std::vector<std::string> current_loop_slot_exprs_;
  std::map<std::string, std::vector<TupleElement>> tuple_var_to_elements_;
  std::map<const ir::Var*, std::string> tuple_var_to_key_;
  std::unordered_set<const ir::Var*> compiler_dep_task_id_vars_;
  std::unordered_set<std::string> compiler_dep_task_id_keys_;
  std::unordered_set<std::string> compiler_dep_loop_return_task_id_keys_;
  std::unordered_map<const ir::RuntimeScopeStmt*, std::vector<const ir::Var*>> hoisted_task_id_vars_by_scope_;
  std::unordered_map<uint64_t, std::string> hoisted_task_id_emit_names_;
  std::unordered_map<uint64_t, std::string> hoisted_task_id_emit_names_by_key_;
  /// Emit names of fresh direct-producer TaskId locals
  /// (``TaskId <name> = task_<n>_outs.task_id();``) — statically always
  /// valid, so their dependency-array insert skips the runtime is_valid()
  /// guard (see EmitDepArrayInsert). Populated only by the fresh-declaration
  /// branch of BindSyntheticTaskId / BindVarTaskId; hoisted / loop-carried /
  /// None-seed / array-slot / dummy / barrier tids are never recorded and keep
  /// the guard. Monotonic: emit names are globally unique, so entries are never
  /// cleared (a stale name cannot match a different, possibly-invalid var).
  std::unordered_set<std::string> guaranteed_valid_task_ids_;
  bool declared_compiler_dep_task_ids_ = false;
  std::unordered_set<const ir::Var*> declared_var_ptrs_;
  std::unordered_set<const ir::Stmt*> batched_create_stmts_;
  std::unordered_set<const ir::Var*> effective_uses_;
  std::unordered_map<std::string, int64_t> gm_pipe_workspace_elements_by_callee_;
  std::unordered_map<std::string, std::string> tensor_create_size_expr_by_emit_name_;
  std::unordered_map<std::string, std::string> dist_param_to_ctx_param_;
  /// Memoizes ``ExplicitReturnedParamIndices`` per callee Function. Tuple/submit
  /// alias generation runs once per call site, but distinct call sites may
  /// share a callee; caching the per-callee return→param map keeps the codegen
  /// from re-walking the same callee body and stays within the O(N log N) pass
  /// budget.
  std::unordered_map<const ir::Function*, std::vector<std::optional<size_t>>> returned_param_indices_cache_;
  /// The program and each wrapper's first inner call are immutable during
  /// emission. Key by the owning function: Submit lookups create transient Call
  /// views. Repeated dispatches then share one batch lineage/body index.
  std::unordered_map<const ir::Function*, std::vector<std::optional<size_t>>>
      wrapper_bridge_param_indices_cache_;
};
}  // namespace codegen
}  // namespace pypto

#endif  // SRC_CODEGEN_ORCHESTRATION_ORCHESTRATION_STMT_CODEGEN_H_
