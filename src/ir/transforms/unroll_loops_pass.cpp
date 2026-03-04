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

namespace pypto {
namespace ir {

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
 * a SeqStmts of cloned bodies, substituting the loop variable with each
 * iteration's constant value.
 *
 * Each unrolled iteration gets fresh Var objects for definition sites
 * (AssignStmt::var_) to ensure structural equality works correctly.
 * A per-iteration clone map tracks original Var -> fresh Var mappings
 * so that def-use chains within one iteration are consistent.
 */
class LoopUnrollMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    // Check substitution map (loop variable pointer -> constant replacement)
    auto sub_it = substitution_map_.find(op.get());
    if (sub_it != substitution_map_.end()) {
      return sub_it->second;
    }
    // Check clone map (per-iteration fresh copies for def-use tracking)
    if (in_unroll_) {
      auto clone_it = var_clone_map_.find(op.get());
      if (clone_it != var_clone_map_.end()) {
        return clone_it->second;
      }
    }
    return op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (!in_unroll_) {
      return IRMutator::VisitStmt_(op);
    }
    // In unrolled regions, only rewrite the RHS. Preserve the original LHS
    // expression node to avoid changing its kind (e.g., IterArg, MemRef).
    // SSA conversion and other passes handle versioning of assignment targets.
    auto new_value = VisitExpr(op->value_);
    return std::make_shared<AssignStmt>(op->var_, new_value, op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ != ForKind::Unroll) {
      // Non-unroll loops: just recurse normally
      return IRMutator::VisitStmt_(op);
    }

    if (op->chunk_size_.has_value()) {
      // Chunked unroll loops: skip, let SplitChunkedLoops handle first
      return IRMutator::VisitStmt_(op);
    }

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

    const Var* loop_var_key = op->loop_var_.get();

    // Save/restore substitution for this variable (handles nested unrolls with same var)
    bool prev_in_unroll = in_unroll_;
    auto prev_clone_map = var_clone_map_;
    auto prev_sub_it = substitution_map_.find(loop_var_key);
    ExprPtr prev_sub_value = (prev_sub_it != substitution_map_.end()) ? prev_sub_it->second : nullptr;

    // Generate unrolled bodies
    std::vector<StmtPtr> unrolled;
    auto emit_iteration = [&](int64_t i) {
      var_clone_map_ = prev_clone_map;
      in_unroll_ = true;
      auto const_expr = std::make_shared<ConstInt>(i, DataType::INDEX, op->loop_var_->span_);
      substitution_map_[loop_var_key] = const_expr;
      unrolled.push_back(VisitStmt(op->body_));
    };

    if (step > 0) {
      for (int64_t i = start; i < stop; i += step) {
        emit_iteration(i);
      }
    } else {
      for (int64_t i = start; i > stop; i += step) {
        emit_iteration(i);
      }
    }

    // Restore substitution state (stack semantics for nested unrolls)
    if (prev_sub_value) {
      substitution_map_[loop_var_key] = prev_sub_value;
    } else {
      substitution_map_.erase(loop_var_key);
    }
    var_clone_map_ = prev_clone_map;
    in_unroll_ = prev_in_unroll;

    if (unrolled.empty()) {
      // Zero-trip loop: return empty SeqStmts
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }

    return std::make_shared<SeqStmts>(unrolled, op->span_);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;

    for (const auto& stmt : op->stmts_) {
      auto new_stmt = VisitStmt(stmt);
      if (new_stmt.get() != stmt.get()) {
        changed = true;
      }
      // Flatten nested SeqStmts produced by unrolling
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
  bool in_unroll_ = false;
  std::unordered_map<const Var*, ExprPtr> substitution_map_;
  std::unordered_map<const Expr*, ExprPtr> var_clone_map_;
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
                                    new_body, func->span_, func->func_type_);
}

}  // namespace

// Factory function
namespace pass {
Pass UnrollLoops() { return CreateFunctionPass(TransformUnrollLoops, "UnrollLoops", kUnrollLoopsProperties); }
}  // namespace pass

}  // namespace ir
}  // namespace pypto
