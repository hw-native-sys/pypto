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

#include "pypto/ir/transforms/utils/signature_symbols.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
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
    auto tensor_type = AsTensorTypeLike(param->GetType());
    if (!tensor_type) continue;
    for (const auto& extent : tensor_type->shape_) {
      if (auto sym = AsVarLike(extent)) symbols.insert(sym.get());
    }
  }
  return symbols;
}

namespace {

// Iterative and DAG-aware: signature preparation must not expand shared shapes
// or overflow before the structural comparator can enforce its depth limit.
template <typename Visit>
void VisitShapeVars(const ExprPtr& root, std::unordered_set<const Expr*>& seen, Visit visit) {
  std::vector<ExprPtr> pending{root};
  while (!pending.empty()) {
    auto expr = pending.back();
    pending.pop_back();
    if (!expr || !seen.insert(expr.get()).second) continue;
    if (auto var = AsVarLike(expr)) {
      visit(var);
    } else if (auto binary = As<BinaryExpr>(expr)) {
      pending.push_back(binary->right_);
      pending.push_back(binary->left_);
    } else if (auto unary = As<UnaryExpr>(expr)) {
      pending.push_back(unary->operand_);
    } else if (auto call = As<Call>(expr)) {
      pending.insert(pending.end(), call->args_.rbegin(), call->args_.rend());
    } else if (auto tuple = As<MakeTuple>(expr)) {
      pending.insert(pending.end(), tuple->elements_.rbegin(), tuple->elements_.rend());
    } else if (auto item = As<TupleGetItemExpr>(expr)) {
      pending.push_back(item->tuple_);
    }
  }
}

/// The extent ``assign``'s ``tensor.dim`` read already names, or null when the
/// read is a genuine runtime read that must stay.
///
/// Mirrors ``ASTParser._fold_tensor_dim``: keep the two in step, since a shape
/// this folds but the parser does not (or vice versa) is exactly the print/parse
/// divergence the fold exists to prevent.
ExprPtr FoldedDimExtent(const AssignStmtPtr& assign, const std::unordered_set<const Var*>& symbols,
                        bool fold_constants) {
  auto call = As<Call>(assign->value_);
  if (!call || !IsOp(call, "tensor.dim") || call->args_.size() != 2) return nullptr;
  auto axis_const = As<ConstInt>(call->args_[1]);
  if (!axis_const) return nullptr;  // Runtime axis — not foldable.
  auto tensor_type = AsTensorTypeLike(call->args_[0]->GetType());
  if (!tensor_type) return nullptr;
  const auto rank = static_cast<int64_t>(tensor_type->shape_.size());
  int64_t axis = axis_const->value_;
  if (axis < 0) axis += rank;
  // An out-of-range axis is the op's error to raise, not ours to fold.
  if (axis < 0 || axis >= rank) return nullptr;
  const auto& extent = tensor_type->shape_[axis];
  // Substitution can turn a helper's dynamic placeholder into a static actual
  // extent. This value is equally safe to use in the cloned body's shapes.
  if (fold_constants && As<ConstInt>(extent)) return extent;
  auto sym = AsVarLike(extent);
  // Only a symbol this signature declares: a local scalar in the shape may since
  // have been reassigned, so it is left alone.
  return (sym && symbols.count(sym.get()) > 0) ? extent : nullptr;
}

class DimDefinitionCounts : public IRVisitor {
 public:
  explicit DimDefinitionCounts(const std::unordered_map<std::string, FunctionPtr>* functions)
      : functions_(functions) {}

  std::unordered_map<const Var*, size_t> counts;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    ++counts[op->var_.get()];
    IRVisitor::VisitStmt_(op);
  }

  // Submit launches pass scalars by value, so they cannot write caller scalar
  // locals. Device scalar returns are rejected by NoScalarKernelReturn. The
  // inherited Submit visitor still walks its arguments and dependency values.
  void VisitExpr_(const CallPtr& op) override {
    // Before inlining, an Out/InOut scalar argument is an implicit write to
    // its caller's local. Count it before substituting away any dim binding;
    // otherwise a later inline expansion could write into a signature symbol.
    if (auto callee = As<GlobalVar>(op->op_); callee && functions_) {
      auto found = functions_->find(callee->name_);
      if (found != functions_->end()) {
        const auto& directions = found->second->param_directions_;
        for (size_t i = 0; i < op->args_.size() && i < directions.size(); ++i) {
          if (directions[i] == ParamDirection::In || !As<ScalarType>(op->args_[i]->GetType())) continue;
          if (auto var = AsVarLike(op->args_[i])) ++counts[var.get()];
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  const std::unordered_map<std::string, FunctionPtr>* functions_;
};

/// Drops each foldable ``tensor.dim`` binding, rewriting its uses to the extent.
///
/// Folding is transitive: substituting one folded read into a local tensor's
/// shape is what exposes a *later* read of that tensor, as in
/// ``m = dim(a, 0); t = create([m, 128]); n = dim(t, 0)`` — once ``t`` is typed
/// ``[M, 128]``, ``n`` names ``M`` too, and the parser folds it on reparse. A
/// collect-then-substitute pair would miss ``n`` and leave exactly the roundtrip
/// mismatch this fold exists to remove.
///
/// Definition counts also admit pre-SSA bodies without rewriting mutable names.
/// The forward traversal rewrites operands and their type references using
/// earlier folds, so transitive reads need no fixed-point rescan.
class DimReadFolder : public IRMutator {
 public:
  DimReadFolder(const std::unordered_set<const Var*>& symbols,
                const std::unordered_set<const Var*>& preserved,
                std::unordered_map<const Var*, size_t> definitions, bool fold_constants)
      : symbols_(symbols),
        preserved_(preserved),
        definitions_(std::move(definitions)),
        fold_constants_(fold_constants) {}

  [[nodiscard]] bool folded_any() const { return folded_any_; }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto rewritten = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(rewritten);
    if (!assign) return rewritten;
    // Inline expansion runs before SSA. Replacing a rebound local with an
    // immutable signature symbol would redirect its later assignments into
    // that symbol. Keep those bindings, and parameter reassignments, intact.
    if (preserved_.count(op->var_.get()) || definitions_.at(op->var_.get()) != 1) return rewritten;
    if (auto extent = FoldedDimExtent(assign, symbols_, fold_constants_)) {
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
  const std::unordered_set<const Var*>& symbols_;
  const std::unordered_set<const Var*>& preserved_;
  std::unordered_map<const Var*, size_t> definitions_;
  bool fold_constants_;
  bool folded_any_ = false;
};

}  // namespace

/// Establish the Orchestration normal form: a param's declared extent *is* its
/// runtime extent, so a ``tensor.dim`` read of it mints a second IR name for one
/// quantity, and shapes built from the copy no longer compare equal to shapes
/// built from the symbol. The DSL parser folds such a read away, but only in an
/// Orchestration body. Outlining can promote an Opaque body, and inlining can
/// introduce reads from a helper. Both must establish this normal form before
/// comparing shapes or printing the resulting Orchestration body.
StmtPtr FoldParamDimReads(const std::vector<VarPtr>& params, const StmtPtr& body,
                          const std::unordered_map<std::string, FunctionPtr>* functions) {
  auto symbols = CollectParamDimSymbols(params);
  std::unordered_set<const Var*> preserved;
  for (const auto& param : params) preserved.insert(param.get());
  return FoldSignatureDimReads(body, symbols, preserved, false, functions);
}

StmtPtr FoldSignatureDimReads(const StmtPtr& body, const std::unordered_set<const Var*>& symbols,
                              const std::unordered_set<const Var*>& preserved, bool fold_constants,
                              const std::unordered_map<std::string, FunctionPtr>* functions) {
  if (symbols.empty() && !fold_constants) return body;
  DimDefinitionCounts definitions(functions);
  definitions.VisitStmt(body);
  DimReadFolder folder(symbols, preserved, std::move(definitions.counts), fold_constants);
  auto folded = folder.VisitStmt(body);
  return folder.folded_any() ? folded : body;
}

SignatureSymbols ClassifySignatureSymbols(const FunctionPtr& function) {
  SignatureSymbols result;
  std::unordered_set<const Var*> bound;
  auto bind = [&](const VarPtr& var) {
    if (bound.insert(var.get()).second) result.bound.push_back(var);
  };
  std::unordered_set<const Expr*> physical;
  for (const auto& param : function->params_) {
    if (auto shaped = As<ShapedType>(param->GetType())) {
      for (const auto& dim : shaped->shape_) VisitShapeVars(dim, physical, bind);
    } else {
      bind(param);
    }
  }
  std::unordered_map<const Var*, size_t> slots;
  for (size_t i = 0; i < function->params_.size(); ++i) {
    auto tensor = AsTensorTypeLike(function->params_[i]->GetType());
    if (!tensor || !tensor->tensor_view_) continue;
    const auto& shape = tensor->tensor_view_->valid_shape;
    for (size_t d = 0; d < shape.size(); ++d) {
      auto var = AsVarLike(shape[d]);
      if (!var || bound.count(var.get())) continue;
      auto [entry, inserted] = slots.emplace(var.get(), result.valid_shape.size());
      if (inserted) result.valid_shape.push_back({var, {}});
      result.valid_shape[entry->second].slots.emplace_back(i, d);
    }
  }
  std::unordered_set<const Expr*> valid;
  for (size_t i = 0; i < function->params_.size(); ++i) {
    auto tensor = AsTensorTypeLike(function->params_[i]->GetType());
    if (!tensor || !tensor->tensor_view_) continue;
    for (const auto& dim : tensor->tensor_view_->valid_shape) {
      VisitShapeVars(dim, valid, [&](const VarPtr& var) {
        if (!bound.count(var.get()) && !slots.count(var.get())) result.unbound.push_back({i, dim, var});
      });
    }
  }
  return result;
}

}  // namespace pypto::ir
