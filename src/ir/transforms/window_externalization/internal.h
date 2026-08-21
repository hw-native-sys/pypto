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

#ifndef SRC_IR_TRANSFORMS_WINDOW_EXTERNALIZATION_INTERNAL_H_
#define SRC_IR_TRANSFORMS_WINDOW_EXTERNALIZATION_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace window_externalization {

/// Shared vocabulary for the window-externalization stages.
///
/// The stages form a DAG: `pass.cpp` drives `analysis.cpp`, `callee_rewrite.cpp`
/// and `orch_rewriter.cpp`; `callee_rewrite.cpp` drives `localizers.cpp`; every
/// stage sits on the helpers declared here and implemented in `common.cpp`.

enum class RewriteKind {
  FinalStore,
  AggregateWindowLoop,
};

struct DenseRegionPiece {
  std::vector<ExprPtr> window_shape;
  std::vector<ExprPtr> callsite_offsets;
  std::vector<ExprPtr> local_offsets;
};

struct AccessRegion {
  // Internal proof result. Today every region lowers to one or more dense
  // tensor.slice views; unsupported access sets stay baseline.
  std::vector<DenseRegionPiece> dense_pieces;
};

struct WindowRewriteContext {
  std::string NextScalarTempName(const std::string& prefix) {
    return prefix + "__expr_tmp_" + std::to_string(next_scalar_temp_id++);
  }

  size_t next_scalar_temp_id = 0;
};

struct OutputRewriteInfo {
  size_t out_param_index;
  size_t return_index;
  std::vector<ExprPtr> parent_shape;
  std::vector<ExprPtr> window_shape;
  std::vector<ExprPtr> callsite_offsets;
  std::vector<ExprPtr> local_store_offsets;
  AccessRegion region;
  std::vector<size_t> piece_return_indices;
  size_t iter_arg_index = SIZE_MAX;
  /// True when the dense aggregate path proved a window covering the whole
  /// parent at offset zero. Windowing that narrows nothing, so `Analyze` drops
  /// it from the rewrite plan -- while the pure-input-window verdict still
  /// counts it. The static-piece fallback deliberately never sets this: a
  /// full-parent *piece* there is one of several, and the multi-piece rewrite
  /// is real.
  bool dense_window_covers_full_parent = false;
};

struct InputRewriteInfo {
  size_t in_param_index;
  std::vector<ExprPtr> parent_shape;
  std::vector<ExprPtr> window_shape;
  std::vector<ExprPtr> callsite_offsets;
  std::vector<ExprPtr> local_read_offsets;
  AccessRegion region;
};

struct CalleeRewriteAnalysis {
  RewriteKind kind = RewriteKind::FinalStore;
  std::vector<OutputRewriteInfo> outputs;
  std::vector<InputRewriteInfo> inputs;
};

using AnalysisMap = std::unordered_map<std::string, CalleeRewriteAnalysis>;

struct AffineForm {
  int64_t coeff = 0;
  ExprPtr base;
};

struct LinearIndexExpr {
  std::unordered_map<const Var*, int64_t> coeffs;
  int64_t constant = 0;
};

// ---------------------------------------------------------------------------
// Shared helpers -- implemented in common.cpp.
// ---------------------------------------------------------------------------

/// True when `func` opts into windowization via the `windowize` attribute.
bool IsWindowizeEnabled(const FunctionPtr& func);

/// Index every function of `program` by name.
std::unordered_map<std::string, FunctionPtr> BuildFunctionLookup(const ProgramPtr& program);

size_t CountVarRefsInStmt(const StmtPtr& stmt, const Var* target);
size_t CountVarRefsInExpr(const ExprPtr& expr, const Var* target);
bool ExprReferencesOnlyVarsIn(const ExprPtr& expr, const std::unordered_set<const Var*>& allowed);

/// Hoist non-trivial generated scalar sub-expressions into local temporaries.
ExprPtr FlattenGeneratedScalarExprWithLocalTemps(const ExprPtr& expr, const std::string& name_prefix,
                                                 const Span& span, std::vector<StmtPtr>* stmts,
                                                 WindowRewriteContext& rewrite_context);

AccessRegion MakeDenseRegion(std::vector<DenseRegionPiece> pieces);
const std::vector<DenseRegionPiece>& DensePieces(const OutputRewriteInfo& info);
const std::vector<DenseRegionPiece>& DensePieces(const InputRewriteInfo& info);

std::optional<TensorView> MakeWindowTensorView(const std::shared_ptr<const TensorType>& tensor_type,
                                               const std::vector<ExprPtr>& parent_shape,
                                               const std::vector<ExprPtr>& window_shape);
TypePtr MakeWindowTensorType(const std::shared_ptr<const TensorType>& tensor_type,
                             const std::vector<ExprPtr>& parent_shape,
                             const std::vector<ExprPtr>& window_shape);

std::vector<ExprPtr> SubstituteExprVector(const std::vector<ExprPtr>& exprs,
                                          const std::unordered_map<const Var*, ExprPtr>& subst);
bool ExprVectorsPointerEqual(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs);
TypePtr SubstituteTypeExprs(const TypePtr& type, const std::unordered_map<const Var*, ExprPtr>& subst);

std::optional<int64_t> CheckedAdd(int64_t lhs, int64_t rhs);
std::optional<int64_t> CheckedSub(int64_t lhs, int64_t rhs);
std::optional<int64_t> CheckedMul(int64_t lhs, int64_t rhs);
/// `|value|`, or nullopt for INT64_MIN, whose magnitude is not representable.
std::optional<int64_t> CheckedAbs(int64_t value);
bool AddLinearCoeff(LinearIndexExpr* expr, const Var* var, int64_t coeff);
std::optional<LinearIndexExpr> ParseLinearIndexExpr(const ExprPtr& expr);
std::optional<int64_t> ConstantDiffIfSameLinearBase(const ExprPtr& lhs, const ExprPtr& rhs);
std::optional<AffineForm> ParseAffineInLoop(const ExprPtr& expr, const Var* loop_var);

/// Trip count of @p loop when all three bounds are compile-time integers.
///
/// Deliberately NOT ``transform_utils::EvalConstTripCount``: this pass needs a
/// stricter contract than that helper offers. Here ``nullopt`` means "cannot
/// prove anything about this loop, do not window it", so the two cases where
/// the shared helper answers with a *value* must answer ``nullopt`` instead:
///
///  - A zero step. ``ComputeStaticTripCount`` reports 0 trips; that is a claim
///    the loop is provably empty, and callers here would rewrite on it.
///  - Bounds whose span or step magnitude overflows ``int64_t``.
///    ``ComputeStaticTripCount`` saturates to ``INT64_MAX`` — safe for the
///    size/threshold callers it was written for, but this pass feeds trip counts
///    into buffer sizing (see #2477).
///
/// The const-int reading itself is shared: both use ``EvalConstInt``.
std::optional<int64_t> GetStaticTripCount(const ForStmtPtr& loop);
std::optional<int64_t> GetKnownPositiveTripCount(const ForStmtPtr& loop);
std::optional<ExprPtr> SimplifyWithLoopBound(const ExprPtr& expr, const VarPtr& loop_var, int64_t value);
std::optional<ExprPtr> SimplifyWithLoopValue(const ExprPtr& expr, const VarPtr& loop_var,
                                             const ExprPtr& value);
std::optional<ExprPtr> GetLoopValueAtTrip(const ForStmtPtr& loop, int64_t trip_index);

// ---------------------------------------------------------------------------
// Stage entry points.
// ---------------------------------------------------------------------------

/// Decide which callees are windowable, then apply the type/ABI safety policy.
/// Implemented in analysis.cpp.
AnalysisMap AnalyzeProgram(const ProgramPtr& program);

/// Clone `func` into its windowed form, or nullptr when the rewrite cannot be
/// completed. Implemented in callee_rewrite.cpp.
FunctionPtr RewriteCallee(const ProgramPtr& program, const FunctionPtr& func,
                          const CalleeRewriteAnalysis& analysis, const std::string& clone_suffix,
                          WindowRewriteContext& rewrite_context);

/// Rewrite window writes / reads inside a cloned callee body to the local
/// window shape. Implemented in localizers.cpp.
StmtPtr LocalizeWindowWrites(const StmtPtr& body,
                             const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                             const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                             WindowRewriteContext& rewrite_context);
StmtPtr LocalizeWindowReads(const StmtPtr& body,
                            const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var,
                            WindowRewriteContext& rewrite_context);

/// Result of rewriting one Orchestration function body.
struct OrchRewriteResult {
  StmtPtr body;                                      ///< Unchanged pointer when nothing was rewritten.
  std::unordered_set<std::string> used_clone_names;  ///< Clones the rewrite actually called.
};

/// Retarget windowable call sites in one Orchestration body to the windowed
/// clones. Implemented in orch_rewriter.cpp.
OrchRewriteResult RewriteOrchestrationBody(
    const ProgramPtr& program, const AnalysisMap& analyses,
    const std::unordered_map<std::string, FunctionPtr>& cloned_funcs,
    const std::unordered_map<std::string, FunctionPtr>& function_lookup,
    WindowRewriteContext& rewrite_context, const StmtPtr& body);

}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_WINDOW_EXTERNALIZATION_INTERNAL_H_
