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
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

// ---------------------------------------------------------------------------
// Utility functions shared between orchestration analysis and codegen
// ---------------------------------------------------------------------------

std::string GetSSABaseName(const std::string& name);
bool IsBuiltinOp(const std::string& op_name);
bool IsTensorOp(const std::string& op_name);

/// True for ops starting with ``array.`` (create / get_element / update_element).
/// ArrayType ops are handled by the orchestration codegen similarly to tensor ops,
/// but emit C-stack array operations instead of GM/DDR pointer operations.
bool IsArrayOp(const std::string& op_name);

/// Returns a Call-shaped view of ``expr`` when it is a Call or a Submit, else
/// null. Thin forward to ``ir::transform_utils::AsCallOrSubmitView``, kept so
/// codegen call sites read unqualified.
inline ir::CallPtr AsCallOrSubmitView(const ir::ExprPtr& expr) {
  return ir::transform_utils::AsCallOrSubmitView(expr);
}
std::string FormatConstIntValue(const ir::ConstIntPtr& c, const std::string& cpp_type);
std::string FormatConstFloatValue(const ir::ConstFloatPtr& c, const std::string& cpp_type);
int GetOrCreateFuncId(const std::string& func_name, std::map<std::string, int>* func_name_to_id,
                      int* next_func_id);

/// Constant-evaluate ``expr`` if it is a ``ConstInt``; returns ``nullopt``
/// otherwise. Thin forward to ``ir::transform_utils::EvalConstInt``.
inline std::optional<int64_t> EvalConstInt(const ir::ExprPtr& expr) {
  return ir::transform_utils::EvalConstInt(expr);
}
/// Return the const trip count of ``for_stmt`` if start/stop/step are all
/// ``ConstInt`` and step is positive; 0 otherwise. Thin forward to
/// ``ir::transform_utils::EvalConstTripCount``.
inline int64_t EvalConstTripCount(const ir::ForStmtPtr& for_stmt) {
  return ir::transform_utils::EvalConstTripCount(for_stmt);
}

/// Compute total GM-pipe workspace elements required by a root orchestration
/// function by walking reachable statements/callees and summing
/// ``initialize_pipe`` slot_count * slot_size.
int64_t ComputeGMPipeWorkspaceElements(const ir::ProgramPtr& program, const ir::FunctionPtr& root_func);

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
  // Tuple-returning call/submit results are keyed on the *binding Var* (stable),
  // not the call pointer: a Submit is viewed as a Call via a transient
  // SubmitToCallView, so a call-pointer key would not survive to codegen.
  std::map<const ir::Var*, std::string> tuple_var_to_key;

 protected:
  void VisitStmt_(const ir::AssignStmtPtr& assign) override;

 private:
  int tuple_call_counter_ = 0;
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

/// Find which Out/InOut parameter index a callee function actually returns.
///
/// Walks the callee's body to locate the topmost ReturnStmt, then traces the
/// returned value (the first element of ``ret.value_``) back through SSA
/// rebinds (``var = pl.assemble(var, ...)`` chains, ForStmt iter args) to a
/// source Param. The returned index points into ``callee->params_`` of the
/// Param that the return ultimately came from.
///
/// Returns ``std::nullopt`` when:
///   - callee is null or has no body
///   - no ReturnStmt is reachable in the top-level body
///   - the return value cannot be resolved to a single Param (e.g., a fresh
///     value not derived from any parameter)
///
/// Use case: orchestration alias generation for kernels that declare multiple
/// ``pl.Out[...]`` parameters (typically the real output + GM scratch
/// passed-through) but expose only one return value at the Python call
/// site. Without this trace, the codegen falls back to the *first* Out
/// parameter, which silently aliases the kernel's result SSA to a scratch
/// tensor.
std::optional<size_t> FindReturnedParamIndex(const ir::FunctionPtr& callee, const ir::ProgramPtr& program);

/// Per-position generalization of ``FindReturnedParamIndex`` for multi-output
/// (tuple-returning) kernels.
///
/// Walks the callee's topmost ``ReturnStmt`` and traces *each* returned value
/// back to its source ``callee->params_`` index. The returned vector is indexed
/// by return-tuple position; entry ``j`` is the param index that tuple element
/// ``j`` writes back, or ``std::nullopt`` when that position is not a param
/// writeback (e.g. an auxiliary scalar such as an SPMD loop iv).
///
/// Returns an **empty vector** when the callee has no traceable top-level
/// ``ReturnStmt`` (null/bodyless callee, or a Group/Spmd wrapper that ends in
/// the inner kernel call). Callers should fall back to a direction-based
/// heuristic in that case.
///
/// Use case: orchestration tuple/submit alias generation. The naive
/// "tail-align return elements onto the trailing Out/InOut params" heuristic
/// silently mis-maps when a kernel takes an ``InOut`` param that is written
/// in place but *not* returned (issue #1573) — the unreturned param shifts the
/// alignment and every carry binds to the wrong source tensor. This precise
/// map removes that ambiguity by consulting the ReturnStmt directly.
std::vector<std::optional<size_t>> FindReturnedParamIndices(const ir::FunctionPtr& callee,
                                                            const ir::ProgramPtr& program);

/// Peek through a leading AUTO ``RuntimeScopeStmt`` so structural analyses
/// reach the original statements. Thin forward to
/// ``ir::transform_utils::UnwrapAutoScope``.
inline ir::StmtPtr UnwrapAutoScope(const ir::StmtPtr& stmt) {
  return ir::transform_utils::UnwrapAutoScope(stmt);
}

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_ORCHESTRATION_ORCHESTRATION_ANALYSIS_H_
