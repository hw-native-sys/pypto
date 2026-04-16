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

#ifndef PYPTO_CODEGEN_ORCHESTRATION_ORCHESTRATION_ANALYSIS_H_
#define PYPTO_CODEGEN_ORCHESTRATION_ORCHESTRATION_ANALYSIS_H_

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

// ---------------------------------------------------------------------------
// Utility functions shared between orchestration analysis and codegen
// ---------------------------------------------------------------------------

std::string GetSSABaseName(const std::string& name);
bool IsBuiltinOp(const std::string& op_name);
bool IsTensorOp(const std::string& op_name);
std::string FormatConstIntValue(const ir::ConstIntPtr& c, const std::string& cpp_type);
std::string FormatConstFloatValue(const ir::ConstFloatPtr& c, const std::string& cpp_type);
void ValidateOrchestrationReferences(const ir::ProgramPtr& program, const ir::FunctionPtr& func);
int GetOrCreateFuncId(const std::string& func_name, std::map<std::string, int>* func_name_to_id,
                      int* next_func_id);

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

struct TupleElement {
  int index;
  const ir::Var* var;
};

// ---------------------------------------------------------------------------
// IR analysis visitors for orchestration codegen
// ---------------------------------------------------------------------------

/**
 * @brief Collect tuple metadata from IR for orchestration codegen
 *
 * Tracks tuple-returning function calls and their TupleGetItemExpr consumers,
 * building a map from unique call keys to extracted tuple elements.
 */
class OrchestrationInfoCollector : public ir::IRVisitor {
 public:
  std::map<std::string, std::vector<TupleElement>> call_tuple_elements;
  std::map<const ir::Call*, std::string> call_to_tuple_key;

 protected:
  void VisitStmt_(const ir::AssignStmtPtr& assign) override;

 private:
  int tuple_call_counter_ = 0;
  std::map<std::string, std::string> current_tuple_key_;
};

/**
 * @brief Determine the canonical buffer root for every Var in the function body
 *
 * Walks the IR and maps each Var* to the Var* that owns its underlying buffer.
 * Propagates root identity through assignments, loops, and function calls.
 * This is a pure structural analysis with no optimization logic.
 */
class BufferRootCollector : public ir::IRVisitor {
 public:
  explicit BufferRootCollector(ir::ProgramPtr program);

  void Initialize(const std::vector<ir::VarPtr>& params);

  std::unordered_map<const ir::Var*, const ir::Var*> buffer_roots;

 protected:
  void VisitStmt_(const ir::ForStmtPtr& for_stmt) override;
  void VisitStmt_(const ir::WhileStmtPtr& while_stmt) override;
  void VisitStmt_(const ir::AssignStmtPtr& assign) override;

 private:
  [[nodiscard]] const ir::Var* ResolveVar(const ir::Var* var) const;
  [[nodiscard]] const ir::Var* ResolveExpr(const ir::ExprPtr& expr) const;
  [[nodiscard]] std::vector<const ir::Var*> CollectCallOutputRoots(const ir::CallPtr& call) const;

  ir::ProgramPtr program_;
  std::unordered_map<const ir::Var*, std::vector<const ir::Var*>> tuple_output_roots_;
};

/**
 * @brief Trace variable lineage from body vars back to function parameters
 *
 * Walks the function body and builds a mapping from every body Var* (including
 * IterArgs, which extend Var) back to its originating function parameter Var*.
 * This enables VarPtr-based identity checks instead of fragile string matching.
 */
class VarLineageCollector : public ir::IRVisitor {
 public:
  explicit VarLineageCollector(ir::ProgramPtr program);

  std::unordered_map<const ir::Var*, const ir::Var*> var_to_param;

  void Initialize(const std::vector<ir::VarPtr>& params);

 protected:
  void VisitStmt_(const ir::ForStmtPtr& for_stmt) override;
  void VisitStmt_(const ir::WhileStmtPtr& while_stmt) override;

  // IfStmt lineage is not tracked: orchestration IfStmt return_vars are rare
  // and their lineage requires analyzing yield values across branches.

  void VisitStmt_(const ir::AssignStmtPtr& assign) override;

 private:
  [[nodiscard]] const ir::Var* ResolveVar(const ir::Var* var) const;
  [[nodiscard]] const ir::Var* ResolveExpr(const ir::ExprPtr& expr) const;

  ir::ProgramPtr program_;
};

/**
 * @brief Compute effective call-site parameter directions for orchestration calls
 *
 * Resolves the semantic mismatch between function-level ParamDirection (a property
 * of the callee: "this function reads/writes this parameter") and orchestration
 * call-site direction (a property of the task submission: "establish dependencies
 * and manage memory for this buffer").
 *
 * Key rule: when a locally allocated tensor (from tensor.create / alloc_tensors)
 * is passed as Out to an InCore call, the call-site direction must be InOut so
 * that the runtime establishes WAW dependencies between tasks sharing the buffer.
 *
 * Uses BufferRootCollector results to trace each argument back to its buffer root
 * and determine whether it originates from a function parameter (external) or a
 * local tensor.create (pre-allocated).
 */
class CallSiteDirectionResolver : public ir::IRVisitor {
 public:
  CallSiteDirectionResolver(ir::ProgramPtr program,
                            const std::unordered_map<const ir::Var*, const ir::Var*>& buffer_roots,
                            const std::vector<ir::VarPtr>& params);

  /// Per-call effective directions. Only populated for calls that need overrides
  /// (i.e., at least one Out parameter targets a locally allocated buffer).
  std::unordered_map<const ir::Call*, std::vector<ir::ParamDirection>> call_site_directions;

 protected:
  void VisitExpr_(const ir::CallPtr& call) override;

 private:
  [[nodiscard]] bool IsLocallyAllocated(const ir::Var* var) const;

  ir::ProgramPtr program_;
  const std::unordered_map<const ir::Var*, const ir::Var*>& buffer_roots_;
  std::unordered_set<const ir::Var*> param_vars_;
};

/// Compute effective param directions for a Group function.
///
/// Group functions produced by the scope outliner have their parameters sorted
/// alphabetically and all directions set to In. To recover the true
/// Out/InOut direction, walk the Group body to find its inner kernel call and
/// map the inner callee's directions back to the Group's parameter positions
/// via pointer identity of the Var passed as the inner call argument.
std::vector<ir::ParamDirection> ComputeGroupEffectiveDirections(const ir::FunctionPtr& group_func,
                                                                const ir::ProgramPtr& program);

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_ORCHESTRATION_ORCHESTRATION_ANALYSIS_H_
