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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

// Sentinel meaning "no matching parameter found".
static constexpr int kNoParam = -1;

using ReturnPermutationMap = std::unordered_map<std::string, std::vector<int>>;

const std::vector<int>* FindReturnPermutation(const ReturnPermutationMap& permutations,
                                              const OpPtr& callee_op) {
  auto global_var = As<GlobalVar>(callee_op);
  if (!global_var) return nullptr;
  auto it = permutations.find(global_var->name_);
  if (it == permutations.end() || it->second.empty()) return nullptr;
  return &it->second;
}

// Build a mapping from each return value index to the parameter index it
// corresponds to.  This replicates the analysis that was previously inlined
// in orchestration codegen (BuildReturnToParamMapping).
//
// Traverses the function body (excluding the final ReturnStmt) and builds
// var_to_out_param, a map from Var* to the parameter index of the Out/InOut
// parameter it ultimately originates from.  Handles:
//   - tile.store assignments:    lhs -> param index of store's output arg
//   - Var-to-Var assignments:    lhs -> lookup(rhs) if rhs is already mapped
//   - ForStmt iter_args:         return_var[i] -> lookup(initValue) or find_param
//
// For each ReturnStmt value, the var is looked up in var_to_out_param first,
// then falls back to direct param-identity matching.
std::vector<int> BuildReturnToParamMapping(const FunctionPtr& func) {
  std::vector<int> mapping;
  if (!func || !func->body_) return mapping;

  auto seq = As<SeqStmts>(func->body_);
  if (!seq || seq->stmts_.empty()) return mapping;
  auto return_stmt = As<ReturnStmt>(seq->stmts_.back());
  if (!return_stmt) return mapping;

  auto find_param_index = [&](const Var* v) -> int {
    for (int pi = 0; pi < static_cast<int>(func->params_.size()); ++pi) {
      if (func->params_[pi].get() == v) return pi;
    }
    return kNoParam;
  };

  // Look up v in var_to_out_param; if not found, fall back to direct param match.
  auto lookup = [&](const std::unordered_map<const Var*, int>& m, const Var* v) -> int {
    auto it = m.find(v);
    return it != m.end() ? it->second : find_param_index(v);
  };

  std::unordered_map<const Var*, int> var_to_out_param;
  for (int si = 0; si + 1 < static_cast<int>(seq->stmts_.size()); ++si) {
    if (auto assign = As<AssignStmt>(seq->stmts_[si])) {
      if (!assign->var_) continue;
      if (auto call = As<Call>(assign->value_)) {
        // tile.store(tile, offsets, out_param, ...) → lhs tracks out_param
        if (IsOp(call, "tile.store") && call->args_.size() >= 3) {
          if (auto out_param = As<Var>(call->args_[2])) {
            var_to_out_param[assign->var_.get()] = find_param_index(out_param.get());
          }
        }
      } else if (auto src_var = As<Var>(assign->value_)) {
        // Var-to-var assignment: propagate existing mapping
        int idx = lookup(var_to_out_param, src_var.get());
        if (idx != kNoParam) {
          var_to_out_param[assign->var_.get()] = idx;
        }
      }
    } else if (auto for_stmt = As<ForStmt>(seq->stmts_[si])) {
      int n = std::min(static_cast<int>(for_stmt->return_vars_.size()),
                       static_cast<int>(for_stmt->iter_args_.size()));
      for (int ri = 0; ri < n; ++ri) {
        const auto& iter_arg = for_stmt->iter_args_[ri];
        if (!iter_arg || !iter_arg->initValue_ || !for_stmt->return_vars_[ri]) continue;
        if (auto init_var = As<Var>(iter_arg->initValue_)) {
          int idx = lookup(var_to_out_param, init_var.get());
          if (idx != kNoParam) {
            var_to_out_param[for_stmt->return_vars_[ri].get()] = idx;
          }
        }
      }
    }
  }

  for (const auto& ret_expr : return_stmt->value_) {
    auto var = As<Var>(ret_expr);
    if (!var) {
      mapping.push_back(kNoParam);
      continue;
    }
    mapping.push_back(lookup(var_to_out_param, var.get()));
  }
  return mapping;
}

// Rewrite tensor return values to the param Var each one writes through
// (pointer identity). Orchestration codegen aliases call results to their
// source args via this mapping; explicit param returns make it a lookup
// instead of a heuristic re-derivation that can mis-bind multi-output
// kernels (#1702, #1573, #1693). Returns nullptr when nothing changes.
FunctionPtr CanonicalizeReturnValues(const FunctionPtr& func, const ProgramPtr& program) {
  if (!func || !func->body_) return nullptr;

  auto ret_to_param = return_lineage::ReturnedParamIndices(func, program);
  if (ret_to_param.empty()) return nullptr;

  // The topmost ReturnStmt is not always the trailing statement of a
  // top-level SeqStmts (e.g. AIV kernels keep theirs inside the split body),
  // so rewrite it wherever it sits.
  class ReturnRewriter : public IRMutator {
   public:
    ReturnRewriter(const FunctionPtr& func, const std::vector<std::optional<size_t>>& ret_to_param)
        : func_(func), ret_to_param_(ret_to_param) {}
    bool changed = false;

   protected:
    StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
      if (done_ || op->value_.empty()) return op;

      // A Group/Spmd wrapper around a multi-result kernel forwards the inner
      // call's whole tuple as ONE return value, while declaring N return
      // positions:  ``result = self.inner(...); return result``.  Expand that
      // into the N params the inner call writes back, so the return is
      // positionally explicit like every other kernel's.
      //
      // Left un-expanded it is invisible to every consumer of the
      // return-position -> param map: the map comes back one-short, callers
      // deem it imprecise and fall back to the legacy tail-alignment
      // heuristic, which shifts each returned element onto the wrong Out/InOut
      // param as soon as the callee writes an Out/InOut param it does not
      // return (an accumulator, ``__gm_pipe_buffer``, an in-place KV cache).
      // That silently binds a consumer task to the wrong tensor operand --
      // #1573 resurfacing on exactly the wrappers its fix left on the
      // heuristic.
      if (op->value_.size() == 1 && ret_to_param_.size() > 1) {
        // Only a function that declares N flat return positions may be given an
        // N-value return. A single ``pl.Tuple[T1, ..., TN]`` return declares ONE
        // (its ``return_types_`` is one TupleType) and stays one value.
        if (func_->return_types_.size() != ret_to_param_.size()) return op;
        std::vector<ExprPtr> expanded;
        expanded.reserve(ret_to_param_.size());
        for (const auto& idx : ret_to_param_) {
          // Every position must be nameable as a param for the expansion to be
          // well-formed; otherwise leave the return alone (status quo ante).
          if (!idx) return op;
          const auto& param = func_->params_[idx.value()];  // NOLINT(bugprone-unchecked-optional-access)
          if (!AsTensorTypeLike(param->GetType())) return op;
          expanded.push_back(param);
        }
        done_ = true;
        changed = true;
        return std::make_shared<ReturnStmt>(expanded, op->span_);
      }

      if (ret_to_param_.size() != op->value_.size()) return op;
      done_ = true;
      std::vector<ExprPtr> new_values = op->value_;
      for (size_t i = 0; i < new_values.size(); ++i) {
        if (!ret_to_param_[i]) continue;
        const auto& param =
            func_->params_[ret_to_param_[i].value()];  // NOLINT(bugprone-unchecked-optional-access)
        if (!AsTensorTypeLike(param->GetType())) continue;
        if (new_values[i].get() == param.get()) continue;
        new_values[i] = param;
        changed = true;
      }
      if (!changed) return op;
      return std::make_shared<ReturnStmt>(new_values, op->span_);
    }

   private:
    const FunctionPtr& func_;
    const std::vector<std::optional<size_t>>& ret_to_param_;
    bool done_ = false;
  };

  ReturnRewriter rewriter(func, ret_to_param);
  auto new_body = rewriter.VisitStmt(func->body_);
  if (!rewriter.changed) return nullptr;

  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

std::vector<int> CollectOutIndices(const FunctionPtr& func) {
  std::vector<int> out_indices;
  for (int i = 0; i < static_cast<int>(func->param_directions_.size()); ++i) {
    if (func->param_directions_[i] == ParamDirection::Out ||
        func->param_directions_[i] == ParamDirection::InOut) {
      out_indices.push_back(i);
    }
  }
  return out_indices;
}

// Compute the permutation that reorders returns so that return[k] corresponds
// to out_indices[k].  Returns an empty vector when no reordering is needed.
//
// permutation[old_index] = new_index
std::vector<int> ComputeReturnPermutation(const FunctionPtr& func) {
  auto ret_to_param = BuildReturnToParamMapping(func);
  if (ret_to_param.empty()) return {};

  auto out_indices = CollectOutIndices(func);
  if (out_indices.empty()) return {};

  // If there are more Out params than return values the mapping is incomplete
  // (e.g. some outputs are not yet covered by the IR analysis).  Skip reorder
  // to avoid constructing an out-of-bounds permutation.
  if (static_cast<int>(out_indices.size()) > static_cast<int>(ret_to_param.size())) return {};

  // Map param_index -> position in out_indices
  std::unordered_map<int, int> param_to_out_pos;
  for (int k = 0; k < static_cast<int>(out_indices.size()); ++k) {
    param_to_out_pos[out_indices[k]] = k;
  }

  std::vector<int> permutation(ret_to_param.size(), kNoParam);
  bool needs_reorder = false;

  for (int i = 0; i < static_cast<int>(ret_to_param.size()); ++i) {
    if (ret_to_param[i] == kNoParam) {
      permutation[i] = i;
      continue;
    }
    auto it = param_to_out_pos.find(ret_to_param[i]);
    if (it == param_to_out_pos.end()) {
      permutation[i] = i;
      continue;
    }
    permutation[i] = it->second;
    if (permutation[i] != i) needs_reorder = true;
  }

  // Guard against non-bijective permutations (duplicate targets or holes).
  // A malformed permutation can later create ReturnStmt entries with null values.
  std::vector<bool> seen(permutation.size(), false);
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    int target = permutation[i];
    if (target < 0 || target >= static_cast<int>(permutation.size())) {
      return {};
    }
    if (seen[target]) {
      // Collision: two old indices map to the same new index.
      return {};
    }
    seen[target] = true;
  }
  for (bool v : seen) {
    if (!v) {
      // Hole: at least one new index has no source.
      return {};
    }
  }

  if (!needs_reorder) return {};
  return permutation;
}

// Reorder return values and return types of an InCore function according to
// the given permutation.  Returns a new Function with the reordered return.
FunctionPtr ReorderReturns(const FunctionPtr& func, const std::vector<int>& permutation) {
  auto seq = As<SeqStmts>(func->body_);
  INTERNAL_CHECK_SPAN(seq && !seq->stmts_.empty(), func->span_)
      << "NormalizeReturnOrder: function body has no statements";
  auto return_stmt = As<ReturnStmt>(seq->stmts_.back());
  INTERNAL_CHECK_SPAN(return_stmt, seq->span_) << "NormalizeReturnOrder: function body has no ReturnStmt";
  INTERNAL_CHECK_SPAN(permutation.size() == return_stmt->value_.size(), return_stmt->span_)
      << "NormalizeReturnOrder: permutation size mismatch";

  std::vector<ExprPtr> new_values(return_stmt->value_.size());
  std::vector<TypePtr> new_return_types(func->return_types_.size());

  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    INTERNAL_CHECK_SPAN(permutation[i] >= 0 && permutation[i] < static_cast<int>(new_values.size()),
                        return_stmt->span_)
        << "NormalizeReturnOrder: permutation index out of range";
    new_values[permutation[i]] = return_stmt->value_[i];
    if (i < static_cast<int>(func->return_types_.size()) &&
        permutation[i] < static_cast<int>(new_return_types.size())) {
      new_return_types[permutation[i]] = func->return_types_[i];
    }
  }

  auto new_return = std::make_shared<ReturnStmt>(new_values, return_stmt->span_);
  std::vector<StmtPtr> new_stmts(seq->stmts_.begin(), seq->stmts_.end() - 1);
  new_stmts.push_back(new_return);
  auto new_body = std::make_shared<SeqStmts>(new_stmts, seq->span_);

  auto new_func = MutableCopy(func);
  new_func->return_types_ = new_return_types;
  new_func->body_ = new_body;
  return new_func;
}

// Step B can preserve a caller's logical tuple contract only when every use of
// a reordered call result is an element projection.  In that form we can
// permute the physical result type and rewrite each TupleGetItem index in
// lockstep.  A whole-tuple escape (alias, yield/carry, return, call argument,
// etc.) would require materializing an inverse-permutation adapter; silently
// changing the tuple's type/order instead is incorrect, especially when two
// elements have the same type.  Reject that unsupported shape explicitly
// instead of emitting type-inconsistent IR or silently cross-wiring outputs.
class CallResultBindingCollector : public IRVisitor {
 public:
  explicit CallResultBindingCollector(const ReturnPermutationMap& permutations)
      : permutations_(permutations) {}

  std::unordered_map<const Var*, std::unordered_set<std::string>> bindings;
  std::unordered_set<const Expr*> directly_bound_calls;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_) {
      if (auto call = As<Call>(op->value_)) {
        RecordBinding(op->var_.get(), call->op_, call.get());
      } else if (auto submit = As<Submit>(op->value_)) {
        RecordBinding(op->var_.get(), submit->op_, submit.get());
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void RecordBinding(const Var* var, const OpPtr& callee_op, const Expr* call_expr) {
    auto global_var = As<GlobalVar>(callee_op);
    if (!global_var || !FindReturnPermutation(permutations_, callee_op)) return;
    bindings[var].insert(global_var->name_);
    directly_bound_calls.insert(call_expr);
  }

  const ReturnPermutationMap& permutations_;
};

struct UnsafeCallResultUse {
  UnsafeCallResultUse(std::string caller_in, std::string callee_in, std::string reason_in,
                      std::string suggestion_in, Span span_in)
      : caller(std::move(caller_in)),
        callee(std::move(callee_in)),
        reason(std::move(reason_in)),
        suggestion(std::move(suggestion_in)),
        span(std::move(span_in)) {}

  std::string caller;
  std::string callee;
  std::string reason;
  std::string suggestion;
  Span span;
};

class UnsupportedCallResultUseCollector : public IRVisitor {
 public:
  UnsupportedCallResultUseCollector(
      const ReturnPermutationMap& permutations,
      const std::unordered_map<const Var*, std::unordered_set<std::string>>& bindings,
      const std::unordered_set<const Expr*>& directly_bound_calls, bool allow_direct_projections,
      bool allow_wrapper_tuple_forward, std::string caller_name)
      : permutations_(permutations),
        bindings_(bindings),
        directly_bound_calls_(directly_bound_calls),
        allow_direct_projections_(allow_direct_projections),
        allow_wrapper_tuple_forward_(allow_wrapper_tuple_forward),
        caller_name_(std::move(caller_name)) {}

  void VisitExpr(const ExprPtr& expr) override {
    const Span* previous_use_span = current_use_span_;
    // A Var node carries its binding/definition span. Keep the enclosing
    // expression or statement span so diagnostics identify this occurrence.
    if (expr && !AsVarLike(expr)) current_use_span_ = &expr->span_;
    IRVisitor::VisitExpr(expr);
    current_use_span_ = previous_use_span;
  }

  void VisitStmt(const StmtPtr& stmt) override {
    const Span* previous_use_span = current_use_span_;
    if (stmt) current_use_span_ = &stmt->span_;
    IRVisitor::VisitStmt(stmt);
    current_use_span_ = previous_use_span;
  }

  std::unordered_set<std::string> unsafe_callees;
  std::optional<UnsafeCallResultUse> first_unsafe_use;

 protected:
  // The LHS is a definition, not a use.  Visiting it through the base visitor
  // would incorrectly classify the direct call binding itself as an escape.
  void VisitStmt_(const AssignStmtPtr& op) override { VisitExpr(op->value_); }

  void VisitStmt_(const EvalStmtPtr& op) override {
    // A top-level EvalStmt intentionally discards the result. There is no
    // tuple contract to preserve, but nested reordered calls in its arguments
    // must still go through the ordinary safety checks.
    const Expr* previous_discarded_call = discarded_result_call_;
    if (allow_direct_projections_ && (As<Call>(op->expr_) || As<Submit>(op->expr_))) {
      discarded_result_call_ = op->expr_.get();
    }
    VisitExpr(op->expr_);
    discarded_result_call_ = previous_discarded_call;
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    for (const auto& value : op->value_) {
      if (allow_wrapper_tuple_forward_) {
        if (auto var = AsVarLike(value)) {
          auto it = bindings_.find(var.get());
          if (it != bindings_.end() && it->second.size() == 1) {
            // Step B materializes an inverse-permutation tuple adapter for this
            // directly forwarded component, preserving the wrapper contract.
            continue;
          }
        }
      }
      VisitExpr(value);
    }
  }

  void VisitExpr_(const TupleGetItemExprPtr& op) override {
    if (allow_direct_projections_) {
      if (auto tuple_var = AsVarLike(op->tuple_); tuple_var && bindings_.count(tuple_var.get())) {
        // This is precisely the supported use: Step B rewrites its index.
        return;
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitVarLike_(const VarPtr& op) override {
    auto it = bindings_.find(op.get());
    if (it != bindings_.end()) {
      std::vector<std::string> callees(it->second.begin(), it->second.end());
      std::sort(callees.begin(), callees.end());
      for (const auto& callee : callees) {
        const Span& use_span =
            current_use_span_ && current_use_span_->is_valid() ? *current_use_span_ : op->span_;
        RecordUnsafe(callee, use_span, "result binding '" + op->name_hint_ + "' is used as a whole tuple",
                     "Destructure the result directly and use only its projected elements; do not alias, "
                     "yield/carry, return, or pass the whole tuple as an argument");
      }
    }
    IRVisitor::VisitVarLike_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    RecordUnsupportedCall(op->op_, op.get());
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    RecordUnsupportedCall(op->op_, op.get());
    IRVisitor::VisitExpr_(op);
  }

 private:
  void RecordUnsupportedCall(const OpPtr& callee_op, const Expr* call_expr) {
    auto global_var = As<GlobalVar>(callee_op);
    if (!global_var || !FindReturnPermutation(permutations_, callee_op)) return;
    if (!allow_direct_projections_) {
      RecordUnsafe(global_var->name_, call_expr->span_, "the reordered callee is called from an InCore body",
                   "Move the call to a non-InCore caller, or make the callee return Out/InOut tensors in "
                   "parameter order");
    } else if (!directly_bound_calls_.count(call_expr) && call_expr != discarded_result_call_) {
      RecordUnsafe(
          global_var->name_, call_expr->span_, "the call result is not directly assigned to a tuple binding",
          "Assign the call result directly to a tuple binding, then use only TupleGetItem projections "
          "from that binding");
    }
  }

  void RecordUnsafe(const std::string& callee, const Span& span, std::string reason, std::string suggestion) {
    unsafe_callees.insert(callee);
    if (!first_unsafe_use) {
      first_unsafe_use.emplace(caller_name_, callee, std::move(reason), std::move(suggestion), span);
    }
  }

  const ReturnPermutationMap& permutations_;
  const std::unordered_map<const Var*, std::unordered_set<std::string>>& bindings_;
  const std::unordered_set<const Expr*>& directly_bound_calls_;
  bool allow_direct_projections_;
  bool allow_wrapper_tuple_forward_;
  std::string caller_name_;
  const Span* current_use_span_ = nullptr;
  const Expr* discarded_result_call_ = nullptr;
};

struct ReturnPermutationSafetyReport {
  std::unordered_set<std::string> unsafe_callees;
  std::optional<UnsafeCallResultUse> first_unsafe_use;
};

ReturnPermutationSafetyReport FindUnsafeReturnPermutations(const std::vector<FunctionPtr>& functions,
                                                           const ReturnPermutationMap& permutations) {
  ReturnPermutationSafetyReport report;
  for (const auto& func : functions) {
    if (!func || !func->body_) continue;

    CallResultBindingCollector binding_collector(permutations);
    binding_collector.VisitStmt(func->body_);

    // Step B intentionally skips InCore bodies.  Therefore even a direct
    // projection there is unsupported and rejects the callee permutation.
    const bool allow_direct_projections = !IsInCoreType(func->func_type_);
    UnsupportedCallResultUseCollector use_collector(
        permutations, binding_collector.bindings, binding_collector.directly_bound_calls,
        allow_direct_projections, IsWrapperType(func->func_type_), func->name_);
    use_collector.VisitStmt(func->body_);
    report.unsafe_callees.insert(use_collector.unsafe_callees.begin(), use_collector.unsafe_callees.end());
    if (!report.first_unsafe_use && use_collector.first_unsafe_use) {
      report.first_unsafe_use.emplace(std::move(*use_collector.first_unsafe_use));
    }
  }
  return report;
}

// Permute a reordered callee's result tuple at its call site. Submit carries one
// extra TASK_ID element at the tail; it is not a callee return and stays fixed.
TypePtr PermuteCallLikeResultType(const TypePtr& type, const std::vector<int>& permutation, bool is_submit,
                                  const Span& span) {
  auto tuple_type = As<TupleType>(type);
  INTERNAL_CHECK_SPAN(tuple_type, span)
      << "Internal error: NormalizeReturnOrder call to a reordered multi-return function has non-tuple "
         "result type";

  const size_t tail_size = is_submit ? 1 : 0;
  INTERNAL_CHECK_SPAN(tuple_type->types_.size() == permutation.size() + tail_size, span)
      << "Internal error: NormalizeReturnOrder call result arity does not match return permutation";
  if (is_submit) {
    auto task_id_type = As<ScalarType>(tuple_type->types_.back());
    INTERNAL_CHECK_SPAN(task_id_type && task_id_type->dtype_ == DataType::TASK_ID, span)
        << "Internal error: NormalizeReturnOrder Submit result must end with Scalar[TASK_ID]";
  }

  std::vector<TypePtr> new_types = tuple_type->types_;
  for (size_t old_index = 0; old_index < permutation.size(); ++old_index) {
    const int new_index = permutation[old_index];
    INTERNAL_CHECK_SPAN(new_index >= 0 && new_index < static_cast<int>(permutation.size()), span)
        << "Internal error: NormalizeReturnOrder call result permutation index out of range";
    new_types[static_cast<size_t>(new_index)] = tuple_type->types_[old_index];
  }
  return std::make_shared<TupleType>(std::move(new_types));
}

// Mutator that applies return-order permutations to call/submit result types,
// their binding Vars, and TupleGetItemExpr indices in orchestration / opaque
// functions that call reordered InCore functions.
class TupleIndexPermutationMutator : public IRMutator {
 public:
  TupleIndexPermutationMutator(const ReturnPermutationMap& permutations, bool adapt_wrapper_tuple_return)
      : permutations_(permutations), adapt_wrapper_tuple_return_(adapt_wrapper_tuple_return) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    INTERNAL_CHECK_SPAN(call, op->span_)
        << "Internal error: NormalizeReturnOrder Call mutated to a non-Call expression";
    const auto* permutation = FindReturnPermutation(permutations_, call->op_);
    if (!permutation) return call;

    auto new_type =
        PermuteCallLikeResultType(call->GetType(), *permutation, /*is_submit=*/false, call->span_);
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_, std::move(new_type),
                                  call->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto submit = As<Submit>(base);
    INTERNAL_CHECK_SPAN(submit, op->span_)
        << "Internal error: NormalizeReturnOrder Submit mutated to a non-Submit expression";
    const auto* permutation = FindReturnPermutation(permutations_, submit->op_);
    if (!permutation) return submit;

    auto new_type =
        PermuteCallLikeResultType(submit->GetType(), *permutation, /*is_submit=*/true, submit->span_);
    return std::make_shared<Submit>(submit->op_, submit->args_, submit->deps_, submit->kwargs_,
                                    submit->attrs_, std::move(new_type), submit->span_, submit->core_num_,
                                    submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = VisitExpr(op->value_);

    if (op->var_) {
      const std::vector<int>* permutation = nullptr;
      if (auto call = As<Call>(new_value)) {
        permutation = FindReturnPermutation(permutations_, call->op_);
      } else if (auto submit = As<Submit>(new_value)) {
        permutation = FindReturnPermutation(permutations_, submit->op_);
      }

      // Visit the RHS before clearing the old binding so a reassignment such
      // as `result = consume(result)` still sees the previous tuple Var. The
      // new definition below then replaces (or removes) that identity mapping
      // for all subsequent uses.
      ForgetBinding(op->var_);
      if (permutation) {
        auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_value->GetType(), op->var_->span_);
        var_remap_[op->var_.get()] = new_var;
        reordered_tuple_vars_[new_var.get()] = permutation;
        return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
      }
    }

    if (new_value.get() != op->value_.get()) {
      return std::make_shared<AssignStmt>(op->var_, new_value, op->span_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    new_values.reserve(op->value_.size());
    bool modified = false;
    for (const auto& value : op->value_) {
      auto new_value = VisitExpr(value);
      if (adapt_wrapper_tuple_return_) {
        if (auto var = As<Var>(new_value)) {
          auto it = reordered_tuple_vars_.find(var.get());
          if (it != reordered_tuple_vars_.end()) {
            const auto& perm = *it->second;
            auto tuple_type = As<TupleType>(new_value->GetType());
            INTERNAL_CHECK_SPAN(tuple_type, op->span_)
                << "Internal error: NormalizeReturnOrder wrapper forwards a non-tuple reordered result";
            INTERNAL_CHECK_SPAN(tuple_type->types_.size() >= perm.size(), op->span_)
                << "Internal error: NormalizeReturnOrder wrapper result arity is smaller than its "
                   "permutation";

            std::vector<ExprPtr> elements;
            elements.reserve(tuple_type->types_.size());
            for (size_t old_index = 0; old_index < tuple_type->types_.size(); ++old_index) {
              int new_index = static_cast<int>(old_index);
              if (old_index < perm.size() && perm[old_index] != kNoParam) {
                new_index = perm[old_index];
              }
              elements.push_back(std::make_shared<TupleGetItemExpr>(new_value, new_index, op->span_));
            }
            new_value = std::make_shared<MakeTuple>(std::move(elements), op->span_);
          }
        }
      }
      modified = modified || new_value.get() != value.get();
      new_values.push_back(std::move(new_value));
    }
    if (modified) return std::make_shared<ReturnStmt>(std::move(new_values), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    auto new_tuple = IRMutator::VisitExpr(op->tuple_);

    int new_index = op->index_;
    // Only consider the transformed tuple node (new_tuple).  If VisitExpr
    // replaced it, any identity-based lookup on op->tuple_ would be stale.
    if (auto var = As<Var>(new_tuple)) {
      auto it = reordered_tuple_vars_.find(var.get());
      if (it != reordered_tuple_vars_.end()) {
        const auto& perm = *it->second;
        if (op->index_ >= 0 && op->index_ < static_cast<int>(perm.size()) && perm[op->index_] != kNoParam) {
          new_index = perm[op->index_];
        }
      }
    }

    if (new_tuple.get() != op->tuple_.get() || new_index != op->index_) {
      return std::make_shared<TupleGetItemExpr>(new_tuple, new_index, op->span_);
    }
    return op;
  }

 private:
  void ForgetBinding(const VarPtr& var) {
    reordered_tuple_vars_.erase(var.get());
    auto remap_it = var_remap_.find(var.get());
    if (remap_it == var_remap_.end()) return;
    if (auto remapped_var = AsVarLike(remap_it->second)) {
      reordered_tuple_vars_.erase(remapped_var.get());
    }
    var_remap_.erase(remap_it);
  }

  const ReturnPermutationMap& permutations_;
  bool adapt_wrapper_tuple_return_;
  std::unordered_map<const Var*, const std::vector<int>*> reordered_tuple_vars_;
};

}  // namespace

namespace pass {

Pass NormalizeReturnOrder() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Step A0: canonicalize tensor returns, then compute candidate InCore
    // permutations without applying them yet.  The intervening safety scan
    // verifies that every program-local caller can be remapped element-wise.
    ReturnPermutationMap candidate_permutations;
    std::vector<FunctionPtr> canonical_functions;
    bool modified = false;

    for (const auto& [gvar, func] : program->functions_) {
      const bool is_wrapper = IsWrapperType(func->func_type_);
      if (IsInCoreType(func->func_type_) || is_wrapper) {
        // Step A0: make every tensor return an explicit param reference.
        FunctionPtr current = func;
        if (auto canonical = CanonicalizeReturnValues(current, program)) {
          current = canonical;
          modified = true;
        }
        std::vector<int> perm;
        if (IsInCoreType(current->func_type_)) perm = ComputeReturnPermutation(current);
        if (!perm.empty()) {
          candidate_permutations[current->name_] = std::move(perm);
        }
        canonical_functions.push_back(current);
      } else {
        canonical_functions.push_back(func);
      }
    }

    auto safety_report = FindUnsafeReturnPermutations(canonical_functions, candidate_permutations);
    if (!safety_report.unsafe_callees.empty()) {
      const auto* first = safety_report.first_unsafe_use ? &*safety_report.first_unsafe_use : nullptr;
      INTERNAL_CHECK_SPAN(first, program->span_)
          << "Internal error: unsafe return permutations were found without an offending use";
      CHECK_SPAN(false, first->span) << "NormalizeReturnOrder cannot safely reorder return values of callee '"
                                     << first->callee << "' in caller '" << first->caller
                                     << "': " << first->reason << ". " << first->suggestion;
    }
    ReturnPermutationMap permutations = std::move(candidate_permutations);

    // Every surviving call result is observed element-wise, so Step B can
    // update its tuple type and projection indices without changing the
    // caller's logical output identities.
    std::vector<FunctionPtr> functions;
    functions.reserve(canonical_functions.size());
    for (const auto& func : canonical_functions) {
      auto perm_it = permutations.find(func->name_);
      if (perm_it != permutations.end()) {
        functions.push_back(ReorderReturns(func, perm_it->second));
        modified = true;
      } else {
        functions.push_back(func);
      }
    }

    if (!modified) return program;

    // Step B: Update TupleGetItemExpr indices in non-InCore functions.
    std::vector<FunctionPtr> final_functions;
    for (const auto& func : functions) {
      if (!IsInCoreType(func->func_type_)) {
        TupleIndexPermutationMutator mutator(permutations, IsWrapperType(func->func_type_));
        auto new_body = mutator.VisitStmt(func->body_);
        if (new_body.get() != func->body_.get()) {
          final_functions.push_back(std::make_shared<Function>(
              func->name_, func->params_, func->param_directions_, func->return_types_, new_body, func->span_,
              func->func_type_, func->level_, func->role_, func->attrs_));
        } else {
          final_functions.push_back(func);
        }
      } else {
        final_functions.push_back(func);
      }
    }

    return std::make_shared<Program>(final_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "NormalizeReturnOrder", kNormalizeReturnOrderProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
