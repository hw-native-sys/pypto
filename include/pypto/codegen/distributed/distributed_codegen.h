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

 protected:
  // Statement visitors
  void VisitStmt_(const ir::AssignStmtPtr& op) override;
  void VisitStmt_(const ir::EvalStmtPtr& op) override;
  void VisitStmt_(const ir::ReturnStmtPtr& op) override;
  void VisitStmt_(const ir::ForStmtPtr& op) override;
  void VisitStmt_(const ir::IfStmtPtr& op) override;
  void VisitStmt_(const ir::SeqStmtsPtr& op) override;

  // Expression visitors
  void VisitExpr_(const ir::CallPtr& op) override;
  void VisitExpr_(const ir::VarPtr& op) override;
  void VisitExpr_(const ir::ConstIntPtr& op) override;
  void VisitExpr_(const ir::ConstFloatPtr& op) override;
  void VisitExpr_(const ir::ConstBoolPtr& op) override;

 private:
  // Code structure emission
  void EmitImports();
  void EmitFunction(const ir::FunctionPtr& func);
  void EmitEntryFunction();

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

  // Helpers
  [[nodiscard]] std::string ParamDirectionToTensorArgType(ir::ParamDirection dir) const;
  [[nodiscard]] std::vector<ir::FunctionPtr> SortFunctionsByRoleAndLevel() const;
  void ClassifyFunctions();
  [[nodiscard]] std::string SanitizeName(const std::string& name) const;
  std::string FormatArgs(const std::vector<ir::ExprPtr>& args);
  [[nodiscard]] bool IsSubWorker(const ir::FunctionPtr& func) const;
  [[nodiscard]] static std::string DataTypeToPythonDType(const DataType& dtype);

  ir::ProgramPtr program_;
  CodeEmitter emitter_;

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
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_DISTRIBUTED_DISTRIBUTED_CODEGEN_H_
