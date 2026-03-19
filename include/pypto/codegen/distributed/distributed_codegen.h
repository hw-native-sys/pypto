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
#include <vector>

#include "pypto/codegen/cce/code_emitter.h"
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
 * @brief Distributed code generator for Linqu hierarchy runtime C++ code
 *
 * Generates complete C++ source files that use the Linqu LevelRuntime API
 * (submit_worker, submit_orchestrator, tree_reduce) from PyPTO IR programs
 * that have been lowered through OutlineHierarchyScopes.
 *
 * Call-site lowering: infers C++ dispatch pattern from callee function metadata:
 * - Worker functions -> rt_lN.submit_worker(...)
 * - Orchestrator functions -> rt_lN.submit_orchestrator(...)
 * - Plain functions -> regular C++ function call
 */
class DistributedCodegen : public CodegenBase {
 public:
  DistributedCodegen() = default;

  /**
   * @brief Generate distributed C++ code from a Program
   *
   * @param program IR Program (after OutlineHierarchyScopes)
   * @return Complete C++ source code as a string
   */
  [[nodiscard]] std::string Generate(const ir::ProgramPtr& program);

  // CodegenBase interface
  [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_target_var_; }
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override;
  int64_t GetConstIntValue(const ir::ExprPtr& expr) override;
  std::string GetVarName(const ir::VarPtr& var) override;

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
  void EmitIncludes();
  void EmitUsingDeclarations();
  void EmitTopologyConstants();
  void EmitTraceHelpers();
  void EmitFunction(const ir::FunctionPtr& func);
  void EmitMain();

  // Call-site lowering
  void EmitCallToWorker(const ir::CallPtr& call, const ir::FunctionPtr& callee);
  void EmitCallToOrchestrator(const ir::CallPtr& call, const ir::FunctionPtr& callee);
  void EmitDistIntrinsic(const ir::CallPtr& call);
  void EmitTreeReduce(const ir::CallPtr& call);

  // Helpers
  [[nodiscard]] std::string RuntimeVarForLevel(ir::Level level) const;
  [[nodiscard]] std::string CppTypeForIRType(const ir::TypePtr& type) const;
  [[nodiscard]] std::vector<ir::FunctionPtr> TopoSortFunctions() const;
  void ClassifyFunctions();
  void CollectNeededRuntimes(const ir::FunctionPtr& func, std::set<int>& needed) const;
  [[nodiscard]] std::string SanitizeName(const std::string& name) const;
  std::string FormatArgs(const std::vector<ir::ExprPtr>& args);

  ir::ProgramPtr program_;
  CodeEmitter emitter_;

  // Function classification
  std::map<std::string, ir::FunctionPtr> workers_;
  std::map<std::string, ir::FunctionPtr> orchestrators_;
  ir::FunctionPtr entry_func_;
  std::map<std::string, ir::FunctionPtr> all_funcs_;
  std::set<int> used_levels_;

  // Per-function state
  std::string current_target_var_;
  std::string current_expr_value_;
  std::set<std::string> declared_vars_;
  bool is_worker_context_{false};
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_DISTRIBUTED_DISTRIBUTED_CODEGEN_H_
