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
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {

using transform_utils::SubstituteStmt;

namespace {

/// Maximum number of iterations allowed for compile-time unrolling.
/// Prevents excessive memory/CPU usage from large trip counts.
constexpr int64_t kMaxUnrollIterations = 1024;

/**
 * @brief Extract a compile-time integer value from a ConstInt or Neg(ConstInt) expression.
 *
 * Handles both positive constants (ConstInt) and negative literals (Neg wrapping ConstInt),
 * since the Python parser represents `-1` as `ir.neg(ir.ConstInt(1))`.
 *
 * @param expr Expression to extract from
 * @param what Description for error messages (e.g., "start", "stop", "step")
 * @return int64_t The constant value
 * @throws pypto::ValueError if expression is not a compile-time constant integer
 */
static int64_t GetConstIntValue(const ExprPtr& expr, const std::string& what) {
  auto ci = std::dynamic_pointer_cast<const ConstInt>(expr);
  if (ci) {
    return ci->value_;
  }
  // Handle Neg(ConstInt) for negative literals
  auto neg = std::dynamic_pointer_cast<const Neg>(expr);
  if (neg) {
    auto inner = std::dynamic_pointer_cast<const ConstInt>(neg->operand_);
    if (inner) {
      return -inner->value_;
    }
  }
  throw pypto::ValueError("Unroll loop " + what + " must be a compile-time integer constant, got " +
                          expr->TypeName());
}

/**
 * @brief Mutator that expands ForStmt nodes with ForKind::Unroll into
 * a SeqStmts of deep-cloned bodies, substituting the loop variable with each
 * iteration's constant value, and chaining iteration outputs as inputs to the next.
 *
 * Uses DeepClone to create fresh Var objects at definition sites for each
 * iteration, ensuring structural equality works correctly and no Var identity
 * is shared across iterations.
 */
class LoopUnrollMutator : public IRMutator {
 public:
  /// Unroll a single ForStmt with ForKind::Unroll.
  /// Returns the unrolled SeqStmts and populates final_carry with the
  /// substitution map to apply to statements following this loop.
  StmtPtr UnrollForStmt(const ForStmtPtr& op, std::map<const Var*, VarPtr>& final_carry) {
    // Validate: no iter_args for unroll loops
    CHECK(op->iter_args_.empty()) << "Unroll loops cannot have iter_args (init_values)";

    // Extract compile-time constants for start/stop/step
    int64_t start = GetConstIntValue(op->start_, "start");
    int64_t stop = GetConstIntValue(op->stop_, "stop");
    int64_t step = GetConstIntValue(op->step_, "step");
    if (step == 0) {
      throw pypto::ValueError("Unroll loop step cannot be zero");
    }

    // Compute trip count and enforce max unroll limit
    int64_t trip_count = 0;
    if (step > 0 && start < stop) {
      trip_count = (stop - start + step - 1) / step;
    } else if (step < 0 && start > stop) {
      trip_count = (start - stop + (-step) - 1) / (-step);
    }
    if (trip_count > kMaxUnrollIterations) {
      throw pypto::ValueError("Unroll loop trip count " + std::to_string(trip_count) +
                              " exceeds maximum allowed (" + std::to_string(kMaxUnrollIterations) +
                              "). Reduce the loop range or use pl.range() instead");
    }

    // Collect body def vars that shadow outer-scope vars by name.
    // shadow_map: body_def_var* → outer_var VarPtr (the var it shadows)
    std::map<const Var*, VarPtr> shadow_map;
    {
      std::unordered_map<std::string, const Var*> def_by_name;
      std::function<void(const StmtPtr&)> collect_defs = [&](const StmtPtr& s) {
        if (!s) return;
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
          if (assign->var_) def_by_name[assign->var_->name_hint_] = assign->var_.get();
        } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(s)) {
          for (const auto& child : seq->stmts_) collect_defs(child);
        }
      };
      collect_defs(op->body_);

      std::function<void(const ExprPtr&)> collect_uses = [&](const ExprPtr& e) {
        if (!e) return;
        if (auto var = std::dynamic_pointer_cast<const Var>(e)) {
          auto it = def_by_name.find(var->name_hint_);
          if (it != def_by_name.end() && var.get() != it->second) {
            shadow_map[it->second] = std::const_pointer_cast<Var>(var);
          }
        } else if (auto call = std::dynamic_pointer_cast<const Call>(e)) {
          for (const auto& arg : call->args_) collect_uses(arg);
        }
      };
      std::function<void(const StmtPtr&)> collect_rhs = [&](const StmtPtr& s) {
        if (!s) return;
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
          collect_uses(assign->value_);
        } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(s)) {
          for (const auto& child : seq->stmts_) collect_rhs(child);
        }
      };
      collect_rhs(op->body_);
    }

    // Generate unrolled bodies. carry_map maps outer_var* → fresh output var (as ExprPtr)
    // for substitution into each subsequent iteration's clone.
    std::vector<StmtPtr> unrolled;
    std::map<const Var*, ExprPtr> carry_map;

    auto emit_iteration = [&](int64_t i) {
      auto const_expr = std::make_shared<ConstInt>(i, DataType::INDEX, op->loop_var_->span_);
      std::unordered_map<const Var*, ExprPtr> sub_map(carry_map.begin(), carry_map.end());
      sub_map[op->loop_var_.get()] = const_expr;
      auto [cloned_body, def_var_map] = DeepClone(op->body_, sub_map);
      // Update carry: outer_var → this iteration's fresh output var
      for (const auto& [body_def, outer_var] : shadow_map) {
        auto it = def_var_map.find(body_def);
        if (it != def_var_map.end()) {
          carry_map[outer_var.get()] = it->second;
        }
      }
      unrolled.push_back(VisitStmt(cloned_body));
    };

    if (step > 0) {
      for (int64_t i = start; i < stop; i += step) emit_iteration(i);
    } else {
      for (int64_t i = start; i > stop; i += step) emit_iteration(i);
    }

    // Build final_carry for statements following the loop.
    // Maps body_def_var → last fresh output (or outer_var for zero-trip).
    for (const auto& [body_def, outer_var] : shadow_map) {
      auto it = carry_map.find(outer_var.get());
      if (it != carry_map.end()) {
        auto fresh_var = std::dynamic_pointer_cast<const Var>(it->second);
        if (fresh_var) {
          final_carry[body_def] = std::const_pointer_cast<Var>(fresh_var);
          final_carry[outer_var.get()] = std::const_pointer_cast<Var>(fresh_var);
        }
      } else {
        // Zero-trip: body def var maps back to the outer var it shadows (identity)
        final_carry[body_def] = outer_var;
      }
    }

    if (unrolled.empty()) {
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }
    return std::make_shared<SeqStmts>(unrolled, op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ != ForKind::Unroll || op->chunk_size_.has_value()) {
      return IRMutator::VisitStmt_(op);
    }
    std::map<const Var*, VarPtr> unused_carry;
    return UnrollForStmt(op, unused_carry);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;
    std::unordered_map<const Var*, VarPtr> pending_subst;

    for (const auto& stmt : op->stmts_) {
      StmtPtr cur = stmt;
      if (!pending_subst.empty()) {
        cur = SubstituteStmt(cur, pending_subst);
        changed = true;
      }

      auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(cur);
      if (for_stmt && for_stmt->kind_ == ForKind::Unroll && !for_stmt->chunk_size_.has_value()) {
        std::map<const Var*, VarPtr> carry;
        auto new_stmt = UnrollForStmt(for_stmt, carry);
        new_stmts.push_back(new_stmt);
        for (const auto& [k, v] : carry) pending_subst[k] = v;
        changed = true;
      } else {
        // Collect original last-def vars by name before processing
        std::unordered_map<std::string, VarPtr> orig_defs;
        CollectLastDefsByName(cur, orig_defs);

        auto new_stmt = VisitStmt(cur);
        if (new_stmt.get() != cur.get()) {
          changed = true;
          // Build substitution: orig_def_var → new_def_var for each name that changed
          std::unordered_map<std::string, VarPtr> new_defs;
          CollectLastDefsByName(new_stmt, new_defs);
          for (const auto& [name, orig_var] : orig_defs) {
            auto it = new_defs.find(name);
            if (it != new_defs.end() && it->second.get() != orig_var.get()) {
              pending_subst[orig_var.get()] = it->second;
            }
          }
        }
        new_stmts.push_back(new_stmt);
      }
    }

    if (!changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

 private:
  /// Collect the last AssignStmt VarPtr for each name_hint in a statement tree.
  /// Recurses into SeqStmts and bodies of structured control-flow statements.
  static void CollectLastDefsByName(const StmtPtr& s, std::unordered_map<std::string, VarPtr>& defs) {
    if (!s) return;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
      if (assign->var_) defs[assign->var_->name_hint_] = assign->var_;
    } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(s)) {
      for (const auto& child : seq->stmts_) CollectLastDefsByName(child, defs);
    } else if (auto for_s = std::dynamic_pointer_cast<const ForStmt>(s)) {
      if (for_s->body_) CollectLastDefsByName(for_s->body_, defs);
    } else if (auto if_s = std::dynamic_pointer_cast<const IfStmt>(s)) {
      if (if_s->then_body_) CollectLastDefsByName(if_s->then_body_, defs);
      if (if_s->else_body_.has_value() && *if_s->else_body_) {
        CollectLastDefsByName(*if_s->else_body_, defs);
      }
    } else if (auto while_s = std::dynamic_pointer_cast<const WhileStmt>(s)) {
      if (while_s->body_) CollectLastDefsByName(while_s->body_, defs);
    } else if (auto scope_s = std::dynamic_pointer_cast<const ScopeStmt>(s)) {
      if (scope_s->body_) CollectLastDefsByName(scope_s->body_, defs);
    }
  }
};

/**
 * @brief Transform a function by unrolling ForKind::Unroll loops.
 */
FunctionPtr TransformUnrollLoops(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "UnrollLoops cannot run on null function";

  LoopUnrollMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);

  if (new_body.get() == func->body_.get()) {
    return func;  // No changes
  }

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_);
}

}  // namespace

// Factory function
namespace pass {
Pass UnrollLoops() { return CreateFunctionPass(TransformUnrollLoops, "UnrollLoops", kUnrollLoopsProperties); }
}  // namespace pass

}  // namespace ir
}  // namespace pypto
