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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Extract a compile-time integer value from a ConstInt or Neg(ConstInt) expression.
 */
static int64_t GetConstIntValue(const ExprPtr& expr, const std::string& what) {
  auto ci = std::dynamic_pointer_cast<const ConstInt>(expr);
  if (ci) {
    return ci->value_;
  }
  auto neg = std::dynamic_pointer_cast<const Neg>(expr);
  if (neg) {
    auto inner = std::dynamic_pointer_cast<const ConstInt>(neg->operand_);
    if (inner) {
      return -inner->value_;
    }
  }
  throw pypto::ValueError("Chunked loop " + what + " must be a compile-time integer constant, got " +
                          expr->TypeName());
}

/**
 * @brief Create a ConstInt expression with INDEX dtype.
 */
static ExprPtr MakeConstIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

/**
 * @brief Collect all AssignStmt var_ (DEF sites) from a statement tree.
 *
 * When the body is visited multiple times (inner + remainder), the same
 * VarPtr would appear as a DEF in both, violating SSA. This function
 * collects all such DEF vars so we can create fresh copies before the
 * second visit.
 */
static void CollectDefVars(const StmtPtr& stmt, std::vector<VarPtr>& result) {
  if (!stmt) return;

  auto kind = stmt->GetKind();
  switch (kind) {
    case ObjectKind::AssignStmt: {
      auto assign = std::static_pointer_cast<const AssignStmt>(stmt);
      result.push_back(assign->var_);
      break;
    }
    case ObjectKind::SeqStmts: {
      auto seq = std::static_pointer_cast<const SeqStmts>(stmt);
      for (const auto& s : seq->stmts_) {
        CollectDefVars(s, result);
      }
      break;
    }
    case ObjectKind::ForStmt: {
      auto for_stmt = std::static_pointer_cast<const ForStmt>(stmt);
      CollectDefVars(for_stmt->body_, result);
      break;
    }
    case ObjectKind::IfStmt: {
      auto if_stmt = std::static_pointer_cast<const IfStmt>(stmt);
      CollectDefVars(if_stmt->then_body_, result);
      if (if_stmt->else_body_.has_value()) {
        CollectDefVars(*if_stmt->else_body_, result);
      }
      break;
    }
    case ObjectKind::ScopeStmt: {
      auto scope = std::static_pointer_cast<const ScopeStmt>(stmt);
      CollectDefVars(scope->body_, result);
      break;
    }
    case ObjectKind::OpStmts: {
      auto ops = std::static_pointer_cast<const OpStmts>(stmt);
      for (const auto& s : ops->stmts_) {
        CollectDefVars(s, result);
      }
      break;
    }
    default:
      // YieldStmt, ReturnStmt, EvalStmt, BreakStmt, ContinueStmt — no DEFs
      break;
  }
}

/**
 * @brief Convert a vector of statements into a single StmtPtr.
 *
 * Returns an empty SeqStmts for empty input, the single statement for
 * size==1, or a SeqStmts wrapping multiple statements.
 */
static StmtPtr MakeResultStmt(const std::vector<StmtPtr>& stmts, const Span& span) {
  if (stmts.empty()) {
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  }
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return std::make_shared<SeqStmts>(stmts, span);
}

/**
 * @brief Mutator that splits ForStmt nodes with chunk_size_ into nested loops.
 *
 * Runs after SSA conversion. Propagates iter_args through generated loops.
 *
 * Transforms (SSA form):
 *   for i, (x_iter=x_0,) in range(start, stop, step, chunk=C) -> (x_rv,):
 *     x_1 = add(x_iter, 1.0)
 *     yield(x_1)
 *
 * Into:
 *   for i_out, (x_outer=x_0,) in range(0, num_full_chunks) -> (x_outer_rv,):
 *     for i_in, (x_inner=x_outer,) in range(0, C) -> (x_inner_rv,):
 *       x_1 = add(x_inner, 1.0)
 *       yield(x_1)
 *     yield(x_inner_rv)
 *   # optional remainder
 *   for i_rem, (x_rem=x_outer_rv,) in range(0, remainder) -> (x_rem_rv,):
 *     x_1_f = add(x_rem, 1.0)   (fresh DEF variable)
 *     yield(x_1_f)
 *   return uses x_rem_rv (or x_outer_rv if no remainder)
 */
class ChunkedLoopSplitter : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ScopeStmtPtr& op) override {
    if (op->scope_kind_ == ScopeKind::AutoInCore) {
      bool prev = inside_auto_incore_;
      inside_auto_incore_ = true;
      auto new_body = VisitStmt(op->body_);
      inside_auto_incore_ = prev;
      if (new_body.get() == op->body_.get()) {
        return op;
      }
      return std::make_shared<ScopeStmt>(op->scope_kind_, new_body, op->span_);
    }
    return IRMutator::VisitStmt_(op);
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto sub_it = substitution_map_.find(op.get());
    if (sub_it != substitution_map_.end()) {
      return sub_it->second;
    }
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto sub_it = substitution_map_.find(op.get());
    if (sub_it != substitution_map_.end()) {
      return sub_it->second;
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (!op->chunk_size_.has_value() || !inside_auto_incore_) {
      return IRMutator::VisitStmt_(op);
    }

    // Extract compile-time constants
    int64_t start = GetConstIntValue(op->start_, "start");
    int64_t stop = GetConstIntValue(op->stop_, "stop");
    int64_t step = GetConstIntValue(op->step_, "step");
    int64_t chunk_size = GetConstIntValue(*op->chunk_size_, "chunk_size");
    CHECK(step != 0) << "Chunked loop step cannot be zero";
    CHECK(chunk_size > 0) << "Chunk size must be positive, got " << chunk_size;

    // Compute trip count
    int64_t trip_count = 0;
    if (step > 0 && start < stop) {
      trip_count = (stop - start + step - 1) / step;
    } else if (step < 0 && start > stop) {
      trip_count = (start - stop + (-step) - 1) / (-step);
    }

    int64_t num_full_chunks = trip_count / chunk_size;
    int64_t remainder = trip_count % chunk_size;

    const Var* loop_var_key = op->loop_var_.get();
    std::string base_name = op->loop_var_->name_;

    // Save previous substitutions for loop var and iter_args
    auto prev_loop_sub = SaveSubstitution(loop_var_key);

    std::vector<SavedSubstitution> prev_ia_subs;
    for (const auto& ia : op->iter_args_) {
      prev_ia_subs.push_back(SaveSubstitution(ia.get()));
    }

    auto start_expr = MakeConstIndex(start, op->span_);
    auto step_expr = MakeConstIndex(step, op->span_);
    auto chunk_const = MakeConstIndex(chunk_size, op->span_);

    bool has_iter_args = !op->iter_args_.empty();

    if (!has_iter_args) {
      // Simple path: no iter_args to propagate
      return SplitSimple(op, start, step, chunk_size, num_full_chunks, remainder, loop_var_key, base_name,
                         start_expr, step_expr, chunk_const, prev_loop_sub);
    }

    // Zero-trip loop: return vars resolve to iter_arg init values
    if (trip_count == 0) {
      INTERNAL_CHECK(op->return_vars_.size() == op->iter_args_.size())
          << "ForStmt return_vars/iter_args size mismatch in zero-trip chunk split";
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        substitution_map_[op->return_vars_[i].get()] = op->iter_args_[i]->initValue_;
      }
      RestoreSubstitution(prev_loop_sub);
      RestoreSubstitutions(prev_ia_subs);
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }

    // SSA path: propagate iter_args through outer/inner/remainder loops
    std::vector<StmtPtr> result_stmts;

    // Track final return vars (from outer or remainder) to remap original return_vars
    std::vector<VarPtr> final_return_vars;

    // Main nested loops (if there are full chunks)
    if (num_full_chunks > 0) {
      auto out_var =
          std::make_shared<Var>(base_name + "_out", std::make_shared<ScalarType>(DataType::INDEX), op->span_);
      auto in_var =
          std::make_shared<Var>(base_name + "_in", std::make_shared<ScalarType>(DataType::INDEX), op->span_);

      // Create outer and inner iter_args/return_vars
      std::vector<IterArgPtr> outer_iter_args;
      std::vector<VarPtr> outer_return_vars;
      std::vector<IterArgPtr> inner_iter_args;
      std::vector<VarPtr> inner_return_vars;

      for (const auto& ia : op->iter_args_) {
        auto outer_ia =
            std::make_shared<IterArg>(ia->name_ + "_outer", ia->GetType(), ia->initValue_, ia->span_);
        auto outer_rv = std::make_shared<Var>(ia->name_ + "_outer_rv", ia->GetType(), ia->span_);
        outer_iter_args.push_back(outer_ia);
        outer_return_vars.push_back(outer_rv);

        ExprPtr inner_init = outer_ia;
        auto inner_ia = std::make_shared<IterArg>(ia->name_ + "_inner", ia->GetType(), inner_init, ia->span_);
        auto inner_rv = std::make_shared<Var>(ia->name_ + "_inner_rv", ia->GetType(), ia->span_);
        inner_iter_args.push_back(inner_ia);
        inner_return_vars.push_back(inner_rv);

        // Remap original iter_arg references to inner iter_arg
        substitution_map_[ia.get()] = inner_ia;
      }

      // Loop var substitution: i = start + (i_out * C + i_in) * step
      ExprPtr substitution =
          MakeAdd(start_expr, MakeMul(MakeAdd(MakeMul(out_var, chunk_const), in_var), step_expr));
      substitution_map_[loop_var_key] = substitution;

      // Visit body -> inner_body
      auto inner_body = VisitStmt(op->body_);

      // Inner loop
      auto inner_for = std::make_shared<ForStmt>(
          in_var, MakeConstIndex(0, op->span_), MakeConstIndex(chunk_size, op->span_),
          MakeConstIndex(1, op->span_), inner_iter_args, inner_body, inner_return_vars, op->span_, op->kind_,
          std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkInner);

      // Outer body = [inner_for, yield(inner_return_vars)]
      auto outer_yield = std::make_shared<YieldStmt>(
          std::vector<ExprPtr>(inner_return_vars.begin(), inner_return_vars.end()), op->span_);
      auto outer_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{inner_for, outer_yield}, op->span_);

      // Outer loop
      auto outer_for = std::make_shared<ForStmt>(
          out_var, MakeConstIndex(0, op->span_), MakeConstIndex(num_full_chunks, op->span_),
          MakeConstIndex(1, op->span_), outer_iter_args, outer_body, outer_return_vars, op->span_, op->kind_,
          std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkOuter);

      result_stmts.push_back(outer_for);
      final_return_vars = outer_return_vars;
    }

    // Remainder loop
    if (remainder > 0) {
      auto rem_var =
          std::make_shared<Var>(base_name + "_rem", std::make_shared<ScalarType>(DataType::INDEX), op->span_);

      int64_t rem_start = start + num_full_chunks * chunk_size * step;
      auto rem_start_expr = MakeConstIndex(rem_start, op->span_);

      // Remainder iter_args
      std::vector<IterArgPtr> rem_iter_args;
      std::vector<VarPtr> rem_return_vars;

      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        const auto& ia = op->iter_args_[i];
        ExprPtr rem_init = (num_full_chunks > 0) ? final_return_vars[i] : ia->initValue_;
        auto rem_ia = std::make_shared<IterArg>(ia->name_ + "_rem", ia->GetType(), rem_init, ia->span_);
        auto rem_rv = std::make_shared<Var>(ia->name_ + "_rem_rv", ia->GetType(), ia->span_);
        rem_iter_args.push_back(rem_ia);
        rem_return_vars.push_back(rem_rv);

        substitution_map_[ia.get()] = rem_ia;
      }

      // Loop var substitution for remainder
      ExprPtr rem_substitution = MakeAdd(rem_start_expr, MakeMul(rem_var, step_expr));
      substitution_map_[loop_var_key] = rem_substitution;

      // When both inner and remainder loops exist, the body is visited twice.
      // DEF variables (AssignStmt targets) in the body would be shared, violating SSA.
      // Create fresh copies of all DEF vars for the remainder body.
      std::vector<SavedSubstitution> prev_def_subs;
      if (num_full_chunks > 0) {
        std::vector<VarPtr> body_def_vars;
        CollectDefVars(op->body_, body_def_vars);
        for (const auto& var : body_def_vars) {
          prev_def_subs.push_back(SaveSubstitution(var.get()));
          auto fresh = std::make_shared<Var>(var->name_, var->GetType(), var->span_);
          substitution_map_[var.get()] = fresh;
        }
      }

      auto rem_body = VisitStmt(op->body_);

      // Restore DEF var substitutions
      RestoreSubstitutions(prev_def_subs);

      auto rem_for = std::make_shared<ForStmt>(
          rem_var, MakeConstIndex(0, op->span_), MakeConstIndex(remainder, op->span_),
          MakeConstIndex(1, op->span_), rem_iter_args, rem_body, rem_return_vars, op->span_, op->kind_,
          std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkRemainder);

      result_stmts.push_back(rem_for);
      final_return_vars = rem_return_vars;
    }

    // Remap original return_vars to final return vars
    INTERNAL_CHECK(op->return_vars_.size() == final_return_vars.size())
        << "SplitChunkedLoops produced mismatched return vars";
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      substitution_map_[op->return_vars_[i].get()] = final_return_vars[i];
    }

    // Restore substitutions
    RestoreSubstitution(prev_loop_sub);
    RestoreSubstitutions(prev_ia_subs);

    return MakeResultStmt(result_stmts, op->span_);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;

    for (const auto& stmt : op->stmts_) {
      auto new_stmt = VisitStmt(stmt);
      if (new_stmt.get() != stmt.get()) {
        changed = true;
      }
      // Flatten nested SeqStmts
      auto seq = std::dynamic_pointer_cast<const SeqStmts>(new_stmt);
      if (seq) {
        for (const auto& inner : seq->stmts_) {
          new_stmts.push_back(inner);
        }
      } else {
        new_stmts.push_back(new_stmt);
      }
    }

    if (!changed) {
      return op;
    }
    return std::make_shared<SeqStmts>(new_stmts, op->span_);
  }

 private:
  bool inside_auto_incore_ = false;
  std::unordered_map<const Var*, ExprPtr> substitution_map_;

  using SavedSubstitution = std::pair<const Var*, ExprPtr>;

  /**
   * @brief Save the current substitution for a key (returns nullptr if none).
   */
  SavedSubstitution SaveSubstitution(const Var* key) {
    auto it = substitution_map_.find(key);
    return {key, (it != substitution_map_.end()) ? it->second : nullptr};
  }

  /**
   * @brief Restore a previously saved substitution.
   */
  void RestoreSubstitution(const SavedSubstitution& saved) {
    if (saved.second) {
      substitution_map_[saved.first] = saved.second;
    } else {
      substitution_map_.erase(saved.first);
    }
  }

  /**
   * @brief Restore a batch of saved substitutions.
   */
  void RestoreSubstitutions(const std::vector<SavedSubstitution>& saved) {
    for (const auto& entry : saved) {
      RestoreSubstitution(entry);
    }
  }

  /**
   * @brief Simple split path for loops without iter_args.
   */
  StmtPtr SplitSimple(const ForStmtPtr& op, int64_t start, int64_t step, int64_t chunk_size,
                      int64_t num_full_chunks, int64_t remainder, const Var* loop_var_key,
                      const std::string& base_name, const ExprPtr& start_expr, const ExprPtr& step_expr,
                      const ExprPtr& chunk_const, const SavedSubstitution& prev_loop_sub) {
    std::vector<StmtPtr> result_stmts;

    if (num_full_chunks > 0) {
      auto out_var =
          std::make_shared<Var>(base_name + "_out", std::make_shared<ScalarType>(DataType::INDEX), op->span_);
      auto in_var =
          std::make_shared<Var>(base_name + "_in", std::make_shared<ScalarType>(DataType::INDEX), op->span_);

      ExprPtr substitution =
          MakeAdd(start_expr, MakeMul(MakeAdd(MakeMul(out_var, chunk_const), in_var), step_expr));

      substitution_map_[loop_var_key] = substitution;
      auto inner_body = VisitStmt(op->body_);

      auto inner_for = std::make_shared<ForStmt>(
          in_var, MakeConstIndex(0, op->span_), MakeConstIndex(chunk_size, op->span_),
          MakeConstIndex(1, op->span_), std::vector<IterArgPtr>{}, inner_body, std::vector<VarPtr>{},
          op->span_, op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkInner);

      auto outer_for = std::make_shared<ForStmt>(
          out_var, MakeConstIndex(0, op->span_), MakeConstIndex(num_full_chunks, op->span_),
          MakeConstIndex(1, op->span_), std::vector<IterArgPtr>{}, inner_for, std::vector<VarPtr>{},
          op->span_, op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkOuter);

      result_stmts.push_back(outer_for);
    }

    if (remainder > 0) {
      auto rem_var =
          std::make_shared<Var>(base_name + "_rem", std::make_shared<ScalarType>(DataType::INDEX), op->span_);

      int64_t rem_start = start + num_full_chunks * chunk_size * step;
      auto rem_start_expr = MakeConstIndex(rem_start, op->span_);

      ExprPtr rem_substitution = MakeAdd(rem_start_expr, MakeMul(rem_var, step_expr));

      substitution_map_[loop_var_key] = rem_substitution;

      // When both full chunks and remainder exist, the body is visited twice.
      // Freshen DEF vars for the remainder to preserve SSA uniqueness.
      std::vector<SavedSubstitution> prev_def_subs;
      if (num_full_chunks > 0) {
        std::vector<VarPtr> body_def_vars;
        CollectDefVars(op->body_, body_def_vars);
        for (const auto& var : body_def_vars) {
          prev_def_subs.push_back(SaveSubstitution(var.get()));
          auto fresh = std::make_shared<Var>(var->name_, var->GetType(), var->span_);
          substitution_map_[var.get()] = fresh;
        }
      }
      auto rem_body = VisitStmt(op->body_);
      RestoreSubstitutions(prev_def_subs);

      auto rem_for = std::make_shared<ForStmt>(
          rem_var, MakeConstIndex(0, op->span_), MakeConstIndex(remainder, op->span_),
          MakeConstIndex(1, op->span_), std::vector<IterArgPtr>{}, rem_body, std::vector<VarPtr>{}, op->span_,
          op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkRemainder);

      result_stmts.push_back(rem_for);
    }

    // Restore loop var substitution
    RestoreSubstitution(prev_loop_sub);

    return MakeResultStmt(result_stmts, op->span_);
  }
};

/**
 * @brief Transform a function by splitting chunked loops.
 */
FunctionPtr TransformSplitChunkedLoops(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "SplitChunkedLoops cannot run on null function";

  ChunkedLoopSplitter splitter;
  auto new_body = splitter.VisitStmt(func->body_);

  if (new_body.get() == func->body_.get()) {
    return func;
  }

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_);
}

}  // namespace

// Factory function
namespace pass {
Pass SplitChunkedLoops() {
  return CreateFunctionPass(TransformSplitChunkedLoops, "SplitChunkedLoops", kSplitChunkedLoopsProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
