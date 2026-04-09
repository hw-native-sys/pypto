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
#include "pypto/ir/transforms/utils/deep_clone_utils.h"

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
 * a SeqStmts of deep-cloned bodies, substituting the loop variable with each
 * iteration's constant value.
 *
 * With shared Var pointers (the parser reuses the same Var object for
 * reassignments of the same source-level variable), we use DeepClone with
 * clone_def_vars=false so the original Var pointers are preserved across
 * unrolled iterations. The SSA pass (which always runs after unrolling)
 * handles versioning naturally.
 */
class LoopUnrollMutator : public IRMutator {
 public:
  /// Unroll a single ForStmt with ForKind::Unroll.
  StmtPtr UnrollForStmt(const ForStmtPtr& op) {
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

    // Generate unrolled bodies. With shared Var pointers (parser reuses the
    // same Var object for reassignments), we pass clone_def_vars=false so
    // DeepClone keeps the original Var pointers. The SSA pass (which always
    // runs after unrolling) handles versioning naturally.
    std::vector<StmtPtr> unrolled;

    auto emit_iteration = [&](int64_t i) {
      auto const_expr = std::make_shared<ConstInt>(i, DataType::INDEX, op->loop_var_->span_);
      std::unordered_map<const Var*, ExprPtr> sub_map;
      sub_map[op->loop_var_.get()] = const_expr;
      auto [cloned_body, def_var_map] = DeepClone(op->body_, sub_map, /*clone_def_vars=*/false);
      unrolled.push_back(VisitStmt(cloned_body));
    };

    if (step > 0) {
      for (int64_t i = start; i < stop; i += step) emit_iteration(i);
    } else {
      for (int64_t i = start; i > stop; i += step) emit_iteration(i);
    }

    if (unrolled.empty()) {
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }
    return std::make_shared<SeqStmts>(unrolled, op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->kind_ != ForKind::Unroll || op->chunk_config_.has_value()) {
      return IRMutator::VisitStmt_(op);
    }
    return UnrollForStmt(op);
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
                                    new_body, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_);
}

}  // namespace

// Factory function
namespace pass {
Pass UnrollLoops() { return CreateFunctionPass(TransformUnrollLoops, "UnrollLoops", kUnrollLoopsProperties); }
}  // namespace pass

}  // namespace ir
}  // namespace pypto
