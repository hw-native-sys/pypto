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

#include <cstddef>
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

struct AssembleViewInfo {
  ir::ExprPtr target_expr;
  ir::MakeTuplePtr offset_tuple;
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
 * @brief Detect tensor.assemble patterns eligible for view optimization
 *
 * Walks the IR in a second pass over a completed buffer_roots map.
 * Records (source_root → target, offsets) pairs for assembles that can be
 * lowered to a single .view() call, and marks roots with multiple assembles
 * as non-optimizable.
 */
class AssembleViewOptimizer : public ir::IRVisitor {
 public:
  explicit AssembleViewOptimizer(const std::unordered_map<const ir::Var*, const ir::Var*>& buffer_roots);

  std::unordered_map<const ir::Var*, AssembleViewInfo> assemble_view_infos;
  std::unordered_set<const ir::Var*> non_optimizable_roots;

 protected:
  void VisitStmt_(const ir::AssignStmtPtr& assign) override;

 private:
  void RecordAssembleViewInfo(const ir::Var* source_root, const ir::ExprPtr& target_expr,
                              const ir::MakeTuplePtr& offset_tuple);
  [[nodiscard]] const ir::Var* ResolveVar(const ir::Var* var) const;
  [[nodiscard]] const ir::Var* ResolveExpr(const ir::ExprPtr& expr) const;
  [[nodiscard]] ir::MakeTuplePtr ResolveTupleExpr(const ir::ExprPtr& expr) const;

  const std::unordered_map<const ir::Var*, const ir::Var*>& buffer_roots_;
  std::unordered_map<const ir::Var*, ir::MakeTuplePtr> tuple_values_;
};

/**
 * @brief Detect ForStmt return variables that are referenced after the loop
 *
 * Identifies loop-carried variables whose values escape the loop scope and are
 * used in subsequent statements, enabling codegen to emit tracking state tensors.
 */
class LoopEscapeInfoCollector : public ir::IRVisitor {
 public:
  std::unordered_set<const ir::Var*> escaping_loop_returns;

 protected:
  void VisitStmt_(const ir::SeqStmtsPtr& seq) override;

 private:
  static bool IsVarReferencedLater(const ir::SeqStmtsPtr& seq, size_t start_index, const ir::Var* target);
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
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_ORCHESTRATION_ORCHESTRATION_ANALYSIS_H_
