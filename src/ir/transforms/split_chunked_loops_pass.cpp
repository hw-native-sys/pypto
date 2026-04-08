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
#include <unordered_set>
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
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using transform_utils::CollectDefVars;

namespace {

/**
 * @brief Try to extract a compile-time integer from a ConstInt or Neg(ConstInt).
 * @return The integer value, or std::nullopt if the expression is not a compile-time constant.
 */
static std::optional<int64_t> TryGetConstInt(const ExprPtr& expr) {
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
  return std::nullopt;
}

/**
 * @brief Extract a compile-time integer value from a ConstInt or Neg(ConstInt) expression.
 */
static int64_t GetConstIntValue(const ExprPtr& expr, const std::string& what) {
  auto val = TryGetConstInt(expr);
  if (val.has_value()) {
    return *val;
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

static void CollectDeclaredNames(const StmtPtr& stmt, std::unordered_set<std::string>& result) {
  if (!stmt) return;

  auto kind = stmt->GetKind();
  switch (kind) {
    case ObjectKind::AssignStmt: {
      auto assign = std::static_pointer_cast<const AssignStmt>(stmt);
      result.insert(assign->var_->name_hint_);
      break;
    }
    case ObjectKind::ForStmt: {
      auto for_stmt = std::static_pointer_cast<const ForStmt>(stmt);
      result.insert(for_stmt->loop_var_->name_hint_);
      for (const auto& ia : for_stmt->iter_args_) result.insert(ia->name_hint_);
      for (const auto& rv : for_stmt->return_vars_) result.insert(rv->name_hint_);
      CollectDeclaredNames(for_stmt->body_, result);
      break;
    }
    case ObjectKind::WhileStmt: {
      auto while_stmt = std::static_pointer_cast<const WhileStmt>(stmt);
      for (const auto& ia : while_stmt->iter_args_) result.insert(ia->name_hint_);
      for (const auto& rv : while_stmt->return_vars_) result.insert(rv->name_hint_);
      CollectDeclaredNames(while_stmt->body_, result);
      break;
    }
    case ObjectKind::IfStmt: {
      auto if_stmt = std::static_pointer_cast<const IfStmt>(stmt);
      for (const auto& rv : if_stmt->return_vars_) result.insert(rv->name_hint_);
      CollectDeclaredNames(if_stmt->then_body_, result);
      if (if_stmt->else_body_.has_value()) {
        CollectDeclaredNames(*if_stmt->else_body_, result);
      }
      break;
    }
    case ObjectKind::SeqStmts: {
      auto seq = std::static_pointer_cast<const SeqStmts>(stmt);
      for (const auto& s : seq->stmts_) {
        CollectDeclaredNames(s, result);
      }
      break;
    }
    case ObjectKind::ScopeStmt: {
      auto scope = std::static_pointer_cast<const ScopeStmt>(stmt);
      CollectDeclaredNames(scope->body_, result);
      break;
    }
    default:
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
  return SeqStmts::Flatten(std::vector<StmtPtr>(stmts), span);
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
  void SeedUsedNames(const FunctionPtr& func) {
    function_used_names_.clear();
    for (const auto& param : func->params_) {
      if (param) {
        function_used_names_.insert(param->name_hint_);
      }
    }
    CollectDeclaredNames(func->body_, function_used_names_);
  }

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

    // chunk_size and step must always be compile-time constants.
    int64_t chunk_size = GetConstIntValue(*op->chunk_size_, "chunk_size");
    int64_t step = GetConstIntValue(op->step_, "step");
    CHECK(step != 0) << "Chunked loop step cannot be zero";
    CHECK(chunk_size > 0) << "Chunk size must be positive, got " << chunk_size;

    auto start_const = TryGetConstInt(op->start_);
    auto stop_const = TryGetConstInt(op->stop_);

    if (start_const && stop_const) {
      // All-static path: original behavior, enables zero-trip elimination at compile time.
      return SplitAllStatic(op, *start_const, *stop_const, step, chunk_size);
    }

    // Dynamic path: start and/or stop are runtime expressions.
    // We emit IR arithmetic (FloorDiv / FloorMod) so the trip count,
    // num_full_chunks, and remainder are computed at runtime.
    return SplitDynamic(op, step, chunk_size);
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
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

 private:
  bool inside_auto_incore_ = false;
  std::unordered_set<std::string> function_used_names_;
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
   * @brief Freshen all DEF vars in the body to preserve SSA uniqueness.
   *
   * Used when the body is visited more than once (e.g. full-chunk + remainder).
   * Returns saved substitutions that must be restored after visiting the body.
   */
  std::vector<SavedSubstitution> FreshenBodyDefVars(const StmtPtr& body) {
    std::vector<SavedSubstitution> prev_def_subs;
    std::vector<VarPtr> body_def_vars;
    CollectDefVars(body, body_def_vars);
    std::unordered_set<std::string> used_names = function_used_names_;
    for (const auto& var : body_def_vars) {
      used_names.insert(var->name_hint_);
    }
    for (const auto& var : body_def_vars) {
      prev_def_subs.push_back(SaveSubstitution(var.get()));
      auto fresh_name = auto_name::GenerateFreshNameLike(var->name_hint_, used_names);
      used_names.insert(fresh_name);
      function_used_names_.insert(fresh_name);
      auto fresh = std::make_shared<Var>(fresh_name, var->GetType(), var->span_);
      substitution_map_[var.get()] = fresh;
    }
    return prev_def_subs;
  }

  /**
   * @brief All-static split: start, stop, step, chunk are compile-time constants.
   *
   * This is the original code path — zero-trip elimination, exact remainder
   * computation, and dead-code pruning all happen at compile time.
   */
  StmtPtr SplitAllStatic(const ForStmtPtr& op, int64_t start, int64_t stop, int64_t step,
                         int64_t chunk_size) {
    int64_t trip_count = 0;
    if (step > 0 && start < stop) {
      trip_count = (stop - start + step - 1) / step;
    } else if (step < 0 && start > stop) {
      trip_count = (start - stop + (-step) - 1) / (-step);
    }

    int64_t num_full_chunks = trip_count / chunk_size;
    int64_t remainder = trip_count % chunk_size;

    const Var* loop_var_key = op->loop_var_.get();
    auto loop_name = auto_name::Parse(op->loop_var_->name_hint_);
    std::string base_name = loop_name.base_name;

    auto prev_loop_sub = SaveSubstitution(loop_var_key);
    std::vector<SavedSubstitution> prev_ia_subs;
    for (const auto& ia : op->iter_args_) {
      prev_ia_subs.push_back(SaveSubstitution(ia.get()));
    }

    auto start_expr = MakeConstIndex(start, op->span_);
    auto step_expr = MakeConstIndex(step, op->span_);
    auto chunk_const = MakeConstIndex(chunk_size, op->span_);
    auto zero = MakeConstIndex(0, op->span_);
    auto one = MakeConstIndex(1, op->span_);
    Span sp = op->span_;

    bool has_iter_args = !op->iter_args_.empty();

    if (!has_iter_args) {
      // Simple path: no iter_args to propagate
      std::vector<StmtPtr> result_stmts;

      if (num_full_chunks > 0) {
        auto out_var = std::make_shared<Var>(
            auto_name::BuildName(base_name, auto_name::ChunkOuterQualifier(), "idx", loop_name.version),
            std::make_shared<ScalarType>(DataType::INDEX), sp);
        auto in_var = std::make_shared<Var>(
            auto_name::BuildName(base_name, auto_name::ChunkInnerQualifier(), "idx", loop_name.version),
            std::make_shared<ScalarType>(DataType::INDEX), sp);

        substitution_map_[loop_var_key] =
            MakeAdd(start_expr, MakeMul(MakeAdd(MakeMul(out_var, chunk_const), in_var), step_expr));
        auto inner_body = VisitStmt(op->body_);

        auto inner_for = std::make_shared<ForStmt>(
            in_var, zero, MakeConstIndex(chunk_size, sp), one, std::vector<IterArgPtr>{}, inner_body,
            std::vector<VarPtr>{}, sp, op->kind_, std::nullopt, ChunkPolicy::LeadingFull,
            LoopOrigin::ChunkInner);
        auto outer_for = std::make_shared<ForStmt>(
            out_var, zero, MakeConstIndex(num_full_chunks, sp), one, std::vector<IterArgPtr>{}, inner_for,
            std::vector<VarPtr>{}, sp, op->kind_, std::nullopt, ChunkPolicy::LeadingFull,
            LoopOrigin::ChunkOuter);
        result_stmts.push_back(outer_for);
      }

      if (remainder > 0) {
        auto rem_var = std::make_shared<Var>(
            auto_name::BuildName(base_name, auto_name::ChunkRemainderQualifier(), "idx", loop_name.version),
            std::make_shared<ScalarType>(DataType::INDEX), sp);
        int64_t rem_start = start + num_full_chunks * chunk_size * step;
        substitution_map_[loop_var_key] = MakeAdd(MakeConstIndex(rem_start, sp), MakeMul(rem_var, step_expr));

        std::vector<SavedSubstitution> prev_def_subs;
        if (num_full_chunks > 0) {
          prev_def_subs = FreshenBodyDefVars(op->body_);
        }
        auto rem_body = VisitStmt(op->body_);
        RestoreSubstitutions(prev_def_subs);

        auto rem_for = std::make_shared<ForStmt>(
            rem_var, zero, MakeConstIndex(remainder, sp), one, std::vector<IterArgPtr>{}, rem_body,
            std::vector<VarPtr>{}, sp, op->kind_, std::nullopt, ChunkPolicy::LeadingFull,
            LoopOrigin::ChunkRemainder);
        result_stmts.push_back(rem_for);
      }

      RestoreSubstitution(prev_loop_sub);
      return MakeResultStmt(result_stmts, sp);
    }

    // --- iter_args path ---

    if (trip_count == 0) {
      INTERNAL_CHECK(op->return_vars_.size() == op->iter_args_.size())
          << "ForStmt return_vars/iter_args size mismatch in zero-trip chunk split";
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        substitution_map_[op->return_vars_[i].get()] = VisitExpr(op->iter_args_[i]->initValue_);
      }
      RestoreSubstitution(prev_loop_sub);
      RestoreSubstitutions(prev_ia_subs);
      return SeqStmts::Flatten(std::vector<StmtPtr>{}, sp);
    }

    std::vector<StmtPtr> result_stmts;
    std::vector<VarPtr> final_return_vars;

    if (num_full_chunks > 0) {
      auto out_var = std::make_shared<Var>(
          auto_name::BuildName(base_name, auto_name::ChunkOuterQualifier(), "idx", loop_name.version),
          std::make_shared<ScalarType>(DataType::INDEX), sp);
      auto in_var = std::make_shared<Var>(
          auto_name::BuildName(base_name, auto_name::ChunkInnerQualifier(), "idx", loop_name.version),
          std::make_shared<ScalarType>(DataType::INDEX), sp);

      std::vector<IterArgPtr> outer_iter_args;
      std::vector<VarPtr> outer_return_vars;
      std::vector<IterArgPtr> inner_iter_args;
      std::vector<VarPtr> inner_return_vars;

      for (const auto& ia : op->iter_args_) {
        auto visited_init = VisitExpr(ia->initValue_);
        auto ia_name = auto_name::Parse(ia->name_hint_);
        auto outer_ia = std::make_shared<IterArg>(
            auto_name::BuildName(ia_name.base_name, auto_name::ChunkOuterQualifier(), "iter",
                                 ia_name.version),
            ia->GetType(), visited_init, ia->span_);
        auto outer_rv = std::make_shared<Var>(
            auto_name::BuildName(ia_name.base_name, auto_name::ChunkOuterQualifier(), "rv", ia_name.version),
            ia->GetType(), ia->span_);
        outer_iter_args.push_back(outer_ia);
        outer_return_vars.push_back(outer_rv);

        auto inner_ia = std::make_shared<IterArg>(
            auto_name::BuildName(ia_name.base_name, auto_name::ChunkInnerQualifier(), "iter",
                                 ia_name.version),
            ia->GetType(), ExprPtr(outer_ia), ia->span_);
        auto inner_rv = std::make_shared<Var>(
            auto_name::BuildName(ia_name.base_name, auto_name::ChunkInnerQualifier(), "rv", ia_name.version),
            ia->GetType(), ia->span_);
        inner_iter_args.push_back(inner_ia);
        inner_return_vars.push_back(inner_rv);

        substitution_map_[ia.get()] = inner_ia;
      }

      substitution_map_[loop_var_key] =
          MakeAdd(start_expr, MakeMul(MakeAdd(MakeMul(out_var, chunk_const), in_var), step_expr));
      auto inner_body = VisitStmt(op->body_);

      auto inner_for = std::make_shared<ForStmt>(
          in_var, zero, MakeConstIndex(chunk_size, sp), one, inner_iter_args, inner_body, inner_return_vars,
          sp, op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkInner);
      auto outer_yield = std::make_shared<YieldStmt>(
          std::vector<ExprPtr>(inner_return_vars.begin(), inner_return_vars.end()), sp);
      auto outer_body = SeqStmts::Flatten(std::vector<StmtPtr>{inner_for, outer_yield}, sp);

      auto outer_for = std::make_shared<ForStmt>(
          out_var, zero, MakeConstIndex(num_full_chunks, sp), one, outer_iter_args, outer_body,
          outer_return_vars, sp, op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkOuter);

      result_stmts.push_back(outer_for);
      final_return_vars = outer_return_vars;
    }

    if (remainder > 0) {
      auto rem_var = std::make_shared<Var>(
          auto_name::BuildName(base_name, auto_name::ChunkRemainderQualifier(), "idx", loop_name.version),
          std::make_shared<ScalarType>(DataType::INDEX), sp);
      int64_t rem_start = start + num_full_chunks * chunk_size * step;

      std::vector<IterArgPtr> rem_iter_args;
      std::vector<VarPtr> rem_return_vars;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        const auto& ia = op->iter_args_[i];
        ExprPtr rem_init = (num_full_chunks > 0) ? final_return_vars[i] : VisitExpr(ia->initValue_);
        auto ia_name = auto_name::Parse(ia->name_hint_);
        auto rem_ia = std::make_shared<IterArg>(
            auto_name::BuildName(ia_name.base_name, auto_name::ChunkRemainderQualifier(), "iter",
                                 ia_name.version),
            ia->GetType(), rem_init, ia->span_);
        auto rem_rv = std::make_shared<Var>(
            auto_name::BuildName(ia_name.base_name, auto_name::ChunkRemainderQualifier(), "rv",
                                 ia_name.version),
            ia->GetType(), ia->span_);
        rem_iter_args.push_back(rem_ia);
        rem_return_vars.push_back(rem_rv);
        substitution_map_[ia.get()] = rem_ia;
      }

      substitution_map_[loop_var_key] =
          MakeAdd(MakeConstIndex(rem_start, sp), MakeMul(rem_var, step_expr));

      std::vector<SavedSubstitution> prev_def_subs;
      if (num_full_chunks > 0) {
        prev_def_subs = FreshenBodyDefVars(op->body_);
      }
      auto rem_body = VisitStmt(op->body_);
      RestoreSubstitutions(prev_def_subs);

      auto rem_for = std::make_shared<ForStmt>(
          rem_var, zero, MakeConstIndex(remainder, sp), one, rem_iter_args, rem_body, rem_return_vars, sp,
          op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkRemainder);
      result_stmts.push_back(rem_for);
      final_return_vars = rem_return_vars;
    }

    INTERNAL_CHECK(op->return_vars_.size() == final_return_vars.size())
        << "SplitChunkedLoops produced mismatched return vars";
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      substitution_map_[op->return_vars_[i].get()] = final_return_vars[i];
    }
    RestoreSubstitution(prev_loop_sub);
    RestoreSubstitutions(prev_ia_subs);
    return MakeResultStmt(result_stmts, sp);
  }

  /**
   * @brief Dynamic split: start and/or stop are runtime expressions.
   *
   * Generates IR expressions for trip_count, num_full_chunks, and remainder
   * using FloorDiv / FloorMod so all arithmetic is deferred to runtime.
   * The inner loop stop is still the compile-time chunk_size constant.
   *
   * Transforms:
   *   for i in range(start, stop, step, chunk=C):
   *     body(i)
   * Into:
   *   trip = (stop - start + (step-1)) // step    (assumes step > 0)
   *   n_full = trip // C
   *   n_rem  = trip %  C
   *   for i_out in range(0, n_full):
   *     for i_in in range(0, C):
   *       i = start + (i_out * C + i_in) * step
   *       body(i)
   *   for i_rem in range(0, n_rem):
   *     i = start + (n_full * C + i_rem) * step
   *     body(i)
   */
  StmtPtr SplitDynamic(const ForStmtPtr& op, int64_t step, int64_t chunk_size) {
    Span sp = op->span_;
    auto zero = MakeConstIndex(0, sp);
    auto one = MakeConstIndex(1, sp);
    auto step_expr = MakeConstIndex(step, sp);
    auto chunk_expr = MakeConstIndex(chunk_size, sp);

    ExprPtr start_expr = VisitExpr(op->start_);
    ExprPtr stop_expr = VisitExpr(op->stop_);

    // trip_count = ceildiv(stop - start, step)
    ExprPtr range_size = MakeSub(stop_expr, start_expr, sp);
    ExprPtr trip_count;
    if (step == 1) {
      trip_count = range_size;
    } else {
      trip_count = MakeFloorDiv(MakeAdd(range_size, MakeConstIndex(step - 1, sp), sp), step_expr, sp);
    }

    // Clamp trip_count to non-negative: max(trip_count, 0)
    trip_count = MakeMax(trip_count, zero, sp);

    ExprPtr n_full = MakeFloorDiv(trip_count, chunk_expr, sp);
    ExprPtr n_rem = MakeFloorMod(trip_count, chunk_expr, sp);

    const Var* loop_var_key = op->loop_var_.get();
    auto loop_name = auto_name::Parse(op->loop_var_->name_hint_);
    std::string base_name = loop_name.base_name;

    auto prev_loop_sub = SaveSubstitution(loop_var_key);
    std::vector<SavedSubstitution> prev_ia_subs;
    for (const auto& ia : op->iter_args_) {
      prev_ia_subs.push_back(SaveSubstitution(ia.get()));
    }

    bool has_iter_args = !op->iter_args_.empty();
    std::vector<StmtPtr> result_stmts;

    if (!has_iter_args) {
      // --- Simple (no iter_args) ---
      auto out_var = std::make_shared<Var>(
          auto_name::BuildName(base_name, auto_name::ChunkOuterQualifier(), "idx", loop_name.version),
          std::make_shared<ScalarType>(DataType::INDEX), sp);
      auto in_var = std::make_shared<Var>(
          auto_name::BuildName(base_name, auto_name::ChunkInnerQualifier(), "idx", loop_name.version),
          std::make_shared<ScalarType>(DataType::INDEX), sp);

      // i = start + (i_out * C + i_in) * step
      substitution_map_[loop_var_key] =
          MakeAdd(start_expr, MakeMul(MakeAdd(MakeMul(out_var, chunk_expr), in_var), step_expr));
      auto inner_body = VisitStmt(op->body_);

      auto inner_for = std::make_shared<ForStmt>(
          in_var, zero, MakeConstIndex(chunk_size, sp), one, std::vector<IterArgPtr>{}, inner_body,
          std::vector<VarPtr>{}, sp, op->kind_, std::nullopt, ChunkPolicy::LeadingFull,
          LoopOrigin::ChunkInner);
      auto outer_for = std::make_shared<ForStmt>(
          out_var, zero, n_full, one, std::vector<IterArgPtr>{}, inner_for, std::vector<VarPtr>{}, sp,
          op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkOuter);
      result_stmts.push_back(outer_for);

      // Remainder
      auto rem_var = std::make_shared<Var>(
          auto_name::BuildName(base_name, auto_name::ChunkRemainderQualifier(), "idx", loop_name.version),
          std::make_shared<ScalarType>(DataType::INDEX), sp);
      // i = start + (n_full * C + i_rem) * step
      substitution_map_[loop_var_key] =
          MakeAdd(start_expr, MakeMul(MakeAdd(MakeMul(n_full, chunk_expr), rem_var), step_expr));

      auto prev_def_subs = FreshenBodyDefVars(op->body_);
      auto rem_body = VisitStmt(op->body_);
      RestoreSubstitutions(prev_def_subs);

      auto rem_for = std::make_shared<ForStmt>(
          rem_var, zero, n_rem, one, std::vector<IterArgPtr>{}, rem_body, std::vector<VarPtr>{}, sp,
          op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkRemainder);
      result_stmts.push_back(rem_for);

      RestoreSubstitution(prev_loop_sub);
      return MakeResultStmt(result_stmts, sp);
    }

    // --- iter_args path ---
    std::vector<VarPtr> final_return_vars;

    // Outer × Inner
    auto out_var = std::make_shared<Var>(
        auto_name::BuildName(base_name, auto_name::ChunkOuterQualifier(), "idx", loop_name.version),
        std::make_shared<ScalarType>(DataType::INDEX), sp);
    auto in_var = std::make_shared<Var>(
        auto_name::BuildName(base_name, auto_name::ChunkInnerQualifier(), "idx", loop_name.version),
        std::make_shared<ScalarType>(DataType::INDEX), sp);

    std::vector<IterArgPtr> outer_iter_args;
    std::vector<VarPtr> outer_return_vars;
    std::vector<IterArgPtr> inner_iter_args;
    std::vector<VarPtr> inner_return_vars;

    for (const auto& ia : op->iter_args_) {
      auto visited_init = VisitExpr(ia->initValue_);
      auto ia_name = auto_name::Parse(ia->name_hint_);
      auto outer_ia = std::make_shared<IterArg>(
          auto_name::BuildName(ia_name.base_name, auto_name::ChunkOuterQualifier(), "iter", ia_name.version),
          ia->GetType(), visited_init, ia->span_);
      auto outer_rv = std::make_shared<Var>(
          auto_name::BuildName(ia_name.base_name, auto_name::ChunkOuterQualifier(), "rv", ia_name.version),
          ia->GetType(), ia->span_);
      outer_iter_args.push_back(outer_ia);
      outer_return_vars.push_back(outer_rv);

      auto inner_ia = std::make_shared<IterArg>(
          auto_name::BuildName(ia_name.base_name, auto_name::ChunkInnerQualifier(), "iter", ia_name.version),
          ia->GetType(), ExprPtr(outer_ia), ia->span_);
      auto inner_rv = std::make_shared<Var>(
          auto_name::BuildName(ia_name.base_name, auto_name::ChunkInnerQualifier(), "rv", ia_name.version),
          ia->GetType(), ia->span_);
      inner_iter_args.push_back(inner_ia);
      inner_return_vars.push_back(inner_rv);

      substitution_map_[ia.get()] = inner_ia;
    }

    substitution_map_[loop_var_key] =
        MakeAdd(start_expr, MakeMul(MakeAdd(MakeMul(out_var, chunk_expr), in_var), step_expr));
    auto inner_body = VisitStmt(op->body_);

    auto inner_for = std::make_shared<ForStmt>(
        in_var, zero, MakeConstIndex(chunk_size, sp), one, inner_iter_args, inner_body, inner_return_vars, sp,
        op->kind_, std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkInner);
    auto outer_yield = std::make_shared<YieldStmt>(
        std::vector<ExprPtr>(inner_return_vars.begin(), inner_return_vars.end()), sp);
    auto outer_body = SeqStmts::Flatten(std::vector<StmtPtr>{inner_for, outer_yield}, sp);

    auto outer_for = std::make_shared<ForStmt>(
        out_var, zero, n_full, one, outer_iter_args, outer_body, outer_return_vars, sp, op->kind_,
        std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkOuter);
    result_stmts.push_back(outer_for);
    final_return_vars = outer_return_vars;

    // Remainder
    auto rem_var = std::make_shared<Var>(
        auto_name::BuildName(base_name, auto_name::ChunkRemainderQualifier(), "idx", loop_name.version),
        std::make_shared<ScalarType>(DataType::INDEX), sp);

    std::vector<IterArgPtr> rem_iter_args;
    std::vector<VarPtr> rem_return_vars;
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      const auto& ia = op->iter_args_[i];
      auto ia_name = auto_name::Parse(ia->name_hint_);
      auto rem_ia = std::make_shared<IterArg>(
          auto_name::BuildName(ia_name.base_name, auto_name::ChunkRemainderQualifier(), "iter",
                               ia_name.version),
          ia->GetType(), ExprPtr(final_return_vars[i]), ia->span_);
      auto rem_rv = std::make_shared<Var>(
          auto_name::BuildName(ia_name.base_name, auto_name::ChunkRemainderQualifier(), "rv",
                               ia_name.version),
          ia->GetType(), ia->span_);
      rem_iter_args.push_back(rem_ia);
      rem_return_vars.push_back(rem_rv);
      substitution_map_[ia.get()] = rem_ia;
    }

    // i = start + (n_full * C + i_rem) * step
    substitution_map_[loop_var_key] =
        MakeAdd(start_expr, MakeMul(MakeAdd(MakeMul(n_full, chunk_expr), rem_var), step_expr));

    auto prev_def_subs = FreshenBodyDefVars(op->body_);
    auto rem_body = VisitStmt(op->body_);
    RestoreSubstitutions(prev_def_subs);

    auto rem_for = std::make_shared<ForStmt>(
        rem_var, zero, n_rem, one, rem_iter_args, rem_body, rem_return_vars, sp, op->kind_, std::nullopt,
        ChunkPolicy::LeadingFull, LoopOrigin::ChunkRemainder);
    result_stmts.push_back(rem_for);
    final_return_vars = rem_return_vars;

    INTERNAL_CHECK(op->return_vars_.size() == final_return_vars.size())
        << "SplitChunkedLoops produced mismatched return vars";
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      substitution_map_[op->return_vars_[i].get()] = final_return_vars[i];
    }
    RestoreSubstitution(prev_loop_sub);
    RestoreSubstitutions(prev_ia_subs);
    return MakeResultStmt(result_stmts, sp);
  }
};

/**
 * @brief Transform a function by splitting chunked loops.
 */
FunctionPtr TransformSplitChunkedLoops(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "SplitChunkedLoops cannot run on null function";

  ChunkedLoopSplitter splitter;
  splitter.SeedUsedNames(func);
  auto new_body = splitter.VisitStmt(func->body_);

  if (new_body.get() == func->body_.get()) {
    return func;
  }

  auto result =
      std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                 new_body, func->span_, func->func_type_, func->level_, func->role_);
  return result;
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
