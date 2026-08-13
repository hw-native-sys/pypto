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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// The dyn-dim symbols this signature declares.
///
/// Mirrors ``ASTParser._param_dim_symbols`` (``python/pypto/language/parser/
/// ast_parser.py``): a bare Var in a tensor param's shape names that argument's
/// runtime extent, and Orchestration codegen defines exactly those symbols from
/// the task-arg descriptors. ``AsVarLike`` rather than ``As<Var>`` so this
/// matches the Python ``isinstance(extent, ir.Var)``, which also accepts the
/// ``IterArg`` subclass.
std::unordered_set<const Var*> CollectParamDimSymbols(const std::vector<VarPtr>& params) {
  std::unordered_set<const Var*> symbols;
  for (const auto& param : params) {
    auto tensor_type = As<TensorType>(param->GetType());
    if (!tensor_type) continue;
    for (const auto& extent : tensor_type->shape_) {
      if (auto sym = AsVarLike(extent)) symbols.insert(sym.get());
    }
  }
  return symbols;
}

/// The extent ``assign``'s ``tensor.dim`` read already names, or null when the
/// read is a genuine runtime read that must stay.
///
/// Mirrors ``ASTParser._fold_tensor_dim``: keep the two in step, since a shape
/// this folds but the parser does not (or vice versa) is exactly the print/parse
/// divergence the fold exists to prevent.
ExprPtr FoldedDimExtent(const AssignStmtPtr& assign, const std::unordered_set<const Var*>& symbols) {
  auto call = As<Call>(assign->value_);
  if (!call || !IsOp(call, "tensor.dim") || call->args_.size() != 2) return nullptr;
  auto axis_const = As<ConstInt>(call->args_[1]);
  if (!axis_const) return nullptr;  // Runtime axis — not foldable.
  auto tensor_type = As<TensorType>(call->args_[0]->GetType());
  if (!tensor_type) return nullptr;
  const auto rank = static_cast<int64_t>(tensor_type->shape_.size());
  int64_t axis = axis_const->value_;
  if (axis < 0) axis += rank;
  // An out-of-range axis is the op's error to raise, not ours to fold.
  if (axis < 0 || axis >= rank) return nullptr;
  const auto& extent = tensor_type->shape_[axis];
  auto sym = AsVarLike(extent);
  // Only a symbol this signature declares: a local scalar in the shape may since
  // have been reassigned, so it is left alone.
  return (sym && symbols.count(sym.get()) > 0) ? extent : nullptr;
}

/// Drops each foldable ``tensor.dim`` binding, rewriting its uses to the extent.
///
/// Folding is transitive: substituting one folded read into a local tensor's
/// shape is what exposes a *later* read of that tensor, as in
/// ``m = dim(a, 0); t = create([m, 128]); n = dim(t, 0)`` — once ``t`` is typed
/// ``[M, 128]``, ``n`` names ``M`` too, and the parser folds it on reparse. A
/// collect-then-substitute pair would miss ``n`` and leave exactly the roundtrip
/// mismatch this fold exists to remove.
///
/// One forward traversal reaches that fixed point because the body is in SSA
/// form, so every definition precedes its uses: the base mutator rewrites this
/// statement's operands (and the Var refs inside their types) from the folds
/// recorded so far, and the rewritten statement is what gets tested.
class DimReadFolder : public IRMutator {
 public:
  explicit DimReadFolder(std::unordered_set<const Var*> symbols) : symbols_(std::move(symbols)) {}

  [[nodiscard]] bool folded_any() const { return folded_any_; }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto rewritten = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(rewritten);
    if (!assign) return rewritten;
    if (auto extent = FoldedDimExtent(assign, symbols_)) {
      // Overrides any old->fresh entry the base visit recorded for this LHS:
      // the binding is going away, so its uses must reach the extent itself.
      var_remap_[op->var_.get()] = extent;
      folded_any_ = true;
      // Empty SeqStmts is the statement-deletion idiom: the parent's
      // SeqStmts::Flatten splices it out of the surrounding list.
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }
    return rewritten;
  }

 private:
  std::unordered_set<const Var*> symbols_;
  bool folded_any_ = false;
};

/// Whether this body still carries an InCore scope — the same condition under
/// which the pass below promotes an Opaque parent to Orchestration.
class HasInCoreScope : public IRVisitor {
 public:
  bool found_ = false;

 protected:
  void VisitStmt_(const InCoreScopeStmtPtr& op) override {
    found_ = true;
    IRVisitor::VisitStmt_(op);
  }
};

/// Establish the Orchestration normal form: a param's declared extent *is* its
/// runtime extent, so a ``tensor.dim`` read of it mints a second IR name for one
/// quantity, and shapes built from the copy no longer compare equal to shapes
/// built from the symbol. The DSL parser folds such a read away, but only in an
/// Orchestration body — a body parsed as Opaque and promoted by this pass would
/// otherwise reach Orchestration still carrying the read, and would no longer
/// parse back to itself.
StmtPtr FoldParamDimReads(const std::vector<VarPtr>& params, const StmtPtr& body) {
  auto symbols = CollectParamDimSymbols(params);
  if (symbols.empty()) return body;
  DimReadFolder folder(std::move(symbols));
  auto folded = folder.VisitStmt(body);
  return folder.folded_any() ? folded : body;
}

}  // namespace

namespace pass {

/**
 * @brief Pass to outline InCore scopes into separate functions
 *
 * This pass transforms ScopeStmt(InCore) nodes into separate Function(InCore) definitions
 * and replaces the scope with a Call to the outlined function.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Processes Opaque and Orchestration functions. Orchestration functions can
 *   carry InCore scopes when the parser desugars high-level constructs
 *   (e.g. ``for i in pl.spmd(...)``) into SpmdScopeStmt(InCoreScopeStmt(...)).
 *
 * Transformation:
 * 1. For each ScopeStmt(InCore) in an Opaque/Orchestration function:
 *    - Analyze body to determine external variable references (inputs)
 *    - Analyze subsequent statements to determine which definitions are outputs
 *    - Extract body into new Function(InCore) with appropriate params/returns
 *    - Replace scope with Call to the outlined function + output assignments
 *    - EvalStmt(store) calls on output tensors are converted to AssignStmt
 * 2. Recursively handles nested InCore scopes
 * 3. Add outlined functions to the program
 * 4. Promote Opaque parents to Orchestration when at least one InCore scope is
 *    outlined. Orchestration parents stay Orchestration.
 */
Pass OutlineIncoreScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

    // Program-wide set of outlined function names, seeded with the existing
    // function names. Shared across each function's ScopeOutliner so that two
    // functions outlining InCore scopes with the same `name_hint` (e.g. a
    // shared `@pl.jit.inline` helper reused across child kernels) get
    // suffix-disambiguated instead of colliding at Program construction (#1711).
    auto reserved_func_names = std::make_shared<std::unordered_set<std::string>>();
    for (const auto& [gvar, func] : program->functions_) {
      reserved_func_names->insert(func->name_);
    }

    for (const auto& [gvar, func] : program->functions_) {
      // Process Opaque and Orchestration functions; other function types
      // (InCore/Group/Spmd) are already outlined or not expected to carry
      // InCore scopes.
      if (func->func_type_ != FunctionType::Opaque && func->func_type_ != FunctionType::Orchestration) {
        new_functions.push_back(func);
        continue;
      }

      // An Opaque body that carries an InCore scope is about to be promoted to
      // Orchestration (see below). Fold its param dyn-dim reads first, so the
      // outliner sees the same body the parser hands an already-Orchestration
      // function — one runtime extent, one IR name, on both paths.
      //
      // The probe is not speculative: ScopeOutliner::VisitScopeKind outlines
      // every InCoreScopeStmt it reaches unconditionally, so "body has an InCore
      // scope" and the promotion condition below (`!outlined.empty()`) are the
      // same predicate. An Opaque function that stays Opaque is never folded —
      // it may be a callee (OutlineHierarchyScopes mints Opaque callees), whose
      // symbol placeholder is not the caller's and may be reached with a
      // statically-shaped actual.
      StmtPtr source_body = func->body_;
      if (func->func_type_ == FunctionType::Opaque) {
        HasInCoreScope probe;
        probe.VisitStmt(source_body);
        if (probe.found_) source_body = FoldParamDimReads(func->params_, source_body);
      }

      // Build symbol table for this function
      outline_utils::VarCollector type_collector;
      for (const auto& var : func->params_) {
        type_collector.var_types[var.get()] = var->GetType();
        type_collector.var_objects[var.get()] = var;
        type_collector.known_names.insert(var->name_hint_);
      }
      type_collector.VisitStmt(source_body);

      // Outline InCore scopes in this function
      outline_utils::ScopeOutliner outliner(
          func->name_, type_collector.var_types, type_collector.var_objects, type_collector.known_names,
          ScopeKind::InCore, FunctionType::InCore, "_incore_", /*program=*/nullptr, reserved_func_names);
      auto new_body = outliner.VisitStmt(source_body);

      // Create new function with transformed body.
      // If any InCore scopes were outlined, promote Opaque -> Orchestration.
      const auto& outlined = outliner.GetOutlinedFunctions();
      FunctionType new_func_type = outlined.empty() ? func->func_type_ : FunctionType::Orchestration;
      auto new_func = MutableCopy(func);
      new_func->body_ = new_body;
      new_func->func_type_ = new_func_type;
      if (new_func_type == FunctionType::Orchestration) {
        new_func->level_ = FunctionTypeToLevel(new_func_type);
        new_func->role_ = Role::Orchestrator;
      }
      new_functions.push_back(new_func);

      // Collect outlined functions (prepend before parent so inner functions come first)
      all_outlined_functions.insert(all_outlined_functions.end(), outlined.begin(), outlined.end());
    }

    // Add all outlined functions before the originals
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());

    // Create new program with all functions
    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineIncoreScopes", kOutlineIncoreScopesProperties);
}

}  // namespace pass

// ============================================================================
// SplitIncoreOrch property verifier
// ============================================================================

namespace {

/**
 * @brief Checks no InCore ScopeStmts remain in Opaque or Orchestration functions.
 */
using SplitIncoreOrchVerifier = outline_utils::ScopeKindAbsenceVerifier<ScopeKind::InCore>;

static bool IsComputeTensorOp(const OpPtr& op) { return transform_utils::IsComputeTensorOp(op); }

/// Checks Orchestration functions for compute tensor ops that should be in InCore.
class OrchComputeTensorOpVerifier : public IRVisitor {
 public:
  explicit OrchComputeTensorOpVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitExpr_(const CallPtr& op) override {
    if (op && op->op_ && IsComputeTensorOp(op->op_)) {
      diagnostics_.emplace_back(DiagnosticSeverity::Warning, "SplitIncoreOrch", 0,
                                "Compute tensor op '" + op->op_->name_ +
                                    "' found in Orchestration function (should be inside InCore)",
                                op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class SplitIncoreOrchPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SplitIncoreOrch"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Check Opaque and Orchestration functions — InCore functions are expected to have InCore content
      if (func->func_type_ == FunctionType::InCore) continue;
      SplitIncoreOrchVerifier verifier(
          diagnostics, "SplitIncoreOrch",
          "InCore ScopeStmt found in non-InCore function (should have been outlined)");
      verifier.VisitStmt(func->body_);
      // Also check Orchestration functions for leaked compute tensor ops
      if (func->func_type_ == FunctionType::Orchestration) {
        OrchComputeTensorOpVerifier compute_verifier(diagnostics);
        compute_verifier.VisitStmt(func->body_);
      }
    }
  }
};

PropertyVerifierPtr CreateSplitIncoreOrchPropertyVerifier() {
  return std::make_shared<SplitIncoreOrchPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
