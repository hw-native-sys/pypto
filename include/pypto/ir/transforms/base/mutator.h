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

#ifndef PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_
#define PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_

#include <unordered_map>
#include <unordered_set>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/functor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief IR mutator for immutable transformations
 *
 * Provides default implementations that recursively transform the IR tree.
 * Returns new ExprPtr or StmtPtr for transformed IR nodes, respecting immutability.
 * Uses copy-on-write: if children are unchanged, returns the original shared_ptr.
 */
class IRMutator : public ExprFunctor<ExprPtr>, public StmtFunctor<StmtPtr> {
 public:
  ~IRMutator() override = default;

  /// Top-level entry points for mutating a full program or function.
  virtual ProgramPtr VisitProgram(const ProgramPtr& program);
  virtual FunctionPtr VisitFunction(const FunctionPtr& func);

  // Override base class methods
  ExprPtr VisitExpr(const ExprPtr& expr) override;
  StmtPtr VisitStmt(const StmtPtr& stmt) override;

 protected:
  // Leaf nodes - return as-is by default
  ExprPtr VisitExpr_(const VarPtr& op) override;
  ExprPtr VisitExpr_(const IterArgPtr& op) override;
  ExprPtr VisitExpr_(const MemRefPtr& op) override;
  ExprPtr VisitExpr_(const ConstIntPtr& op) override;
  ExprPtr VisitExpr_(const ConstFloatPtr& op) override;
  ExprPtr VisitExpr_(const ConstBoolPtr& op) override;
  ExprPtr VisitExpr_(const CallPtr& op) override;
  ExprPtr VisitExpr_(const MakeTuplePtr& op) override;
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override;

  // Binary operations - reconstruct with mutated children
  ExprPtr VisitExpr_(const AddPtr& op) override;
  ExprPtr VisitExpr_(const SubPtr& op) override;
  ExprPtr VisitExpr_(const MulPtr& op) override;
  ExprPtr VisitExpr_(const FloorDivPtr& op) override;
  ExprPtr VisitExpr_(const FloorModPtr& op) override;
  ExprPtr VisitExpr_(const FloatDivPtr& op) override;
  ExprPtr VisitExpr_(const MinPtr& op) override;
  ExprPtr VisitExpr_(const MaxPtr& op) override;
  ExprPtr VisitExpr_(const PowPtr& op) override;
  ExprPtr VisitExpr_(const EqPtr& op) override;
  ExprPtr VisitExpr_(const NePtr& op) override;
  ExprPtr VisitExpr_(const LtPtr& op) override;
  ExprPtr VisitExpr_(const LePtr& op) override;
  ExprPtr VisitExpr_(const GtPtr& op) override;
  ExprPtr VisitExpr_(const GePtr& op) override;
  ExprPtr VisitExpr_(const AndPtr& op) override;
  ExprPtr VisitExpr_(const OrPtr& op) override;
  ExprPtr VisitExpr_(const XorPtr& op) override;
  ExprPtr VisitExpr_(const BitAndPtr& op) override;
  ExprPtr VisitExpr_(const BitOrPtr& op) override;
  ExprPtr VisitExpr_(const BitXorPtr& op) override;
  ExprPtr VisitExpr_(const BitShiftLeftPtr& op) override;
  ExprPtr VisitExpr_(const BitShiftRightPtr& op) override;

  // Unary operations - reconstruct with mutated operand
  ExprPtr VisitExpr_(const AbsPtr& op) override;
  ExprPtr VisitExpr_(const NegPtr& op) override;
  ExprPtr VisitExpr_(const NotPtr& op) override;
  ExprPtr VisitExpr_(const BitNotPtr& op) override;
  ExprPtr VisitExpr_(const CastPtr& op) override;

  // Statement types
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override;
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;
  StmtPtr VisitStmt_(const YieldStmtPtr& op) override;
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override;
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override;
  StmtPtr VisitStmt_(const InCoreScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const AutoInCoreScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const ClusterScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const HierarchyScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const SpmdScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override;
  StmtPtr VisitStmt_(const BreakStmtPtr& op) override;
  StmtPtr VisitStmt_(const ContinueStmtPtr& op) override;
  StmtPtr VisitStmt_(const InlineStmtPtr& op) override;
  StmtPtr VisitStmt_(const StmtPtr& op) override;

  /// Override to handle ALL binary expressions (Add, Sub, Mul, ...) in one method.
  /// Default: visits children, reconstructs if changed (copy-on-write).
  virtual ExprPtr VisitBinaryExpr_(const BinaryExprPtr& op);

  /// Override to handle ALL unary expressions (Abs, Neg, Not, ...) in one method.
  /// Default: visits operand, reconstructs if changed (copy-on-write).
  virtual ExprPtr VisitUnaryExpr_(const UnaryExprPtr& op);

  /// Walk the embedded expressions inside a TypePtr — shape dims,
  /// TileView/TensorView fields, and any embedded MemRef's base/offset —
  /// dispatching each through ExprFunctor::VisitExpr so the active substitution
  /// (var_remap_ or subclass overrides) reaches Var refs that live inside types.
  /// Copy-on-write inside CloneTypeWithMemRefAndRemapExprs returns the original
  /// TypePtr when nothing inside changes.
  TypePtr RemapTypeViaVisitor(const TypePtr& type);

  /// Resolve a var_remap_ hit transitively — the seeded value may itself need
  /// further substitution (its type embeds a Var that's also in var_remap_).
  /// Memoizes the resolved value back into var_remap_ so subsequent lookups
  /// skip the chain. Direct self-references and indirect cycles (A→B, B→A) are
  /// detected via remap_resolving_ and short-circuit by returning the unresolved
  /// value rather than recursing.
  ExprPtr ResolveVarRemapHit(const Expr* key, ExprPtr remapped);

  /// Pointer remapping for Vars whose definitions changed during mutation.
  /// Two sources populate this map: (1) subclass-seeded substitutions (e.g.
  /// SubstituteMutator copies the user's var_map at construction); (2) fresh
  /// Vars minted by VisitExpr_(VarPtr/IterArgPtr/MemRefPtr) when an old Var's
  /// type embeds a substituted Var and gets remapped. Both forms preserve
  /// def-use closure: every visit of the old Var resolves to the same
  /// replacement.
  std::unordered_map<const Expr*, ExprPtr> var_remap_;

  /// Keys currently being resolved by ResolveVarRemapHit. Detects cycles in
  /// caller-supplied substitution maps (e.g. {A→B, B→A}) and stops recursion
  /// before a stack overflow.
  std::unordered_set<const Expr*> remap_resolving_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_
