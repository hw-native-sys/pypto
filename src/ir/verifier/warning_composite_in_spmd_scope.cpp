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

/**
 * @file warning_composite_in_spmd_scope.cpp
 * @brief Warn when an InCore composite collective sits inside a pl.spmd scope.
 *
 * A composite collective is a *rank-level* operation — one logical collective
 * per rank. `pl.spmd(N)` is a *core-level* scope — N blocks within one rank.
 * Nesting them is a category error unless the collective defines its own core
 * decomposition, and the InCore rail has no such parameter (the HOST rail's
 * `core_num` is the only multi-core knob that exists).
 *
 * LowerCompositeOps never reads the block index, so the emitted body is not a
 * function of the enclosing spmd width: the push loop's bounds are `nranks` and
 * its put offsets are `my_rank`. Every block therefore issues the *same*
 * transfers to the *same* peers — the traffic is duplicated N times, not
 * divided N ways. The barrier is affected too: its expected credit is the
 * compile-time constant 1 while N blocks each notify +1, so it releases once a
 * peer's *first* block has notified rather than its last.
 *
 * None of that fails a test today. Every block writes byte-identical content,
 * so an early reader still observes correct values, and the epilogue subtracts
 * -1 per block so the signal still returns to zero. The cost is silent: an
 * N-fold traffic multiplier with no diagnostic. This check makes it visible.
 *
 * Runs `PrePipeline` — the composite Call must still exist, and
 * LowerCompositeOps replaces it during the pipeline.
 *
 * A warning rather than an error, deliberately: a caller may guard the call so
 * only one block executes it (`if block_idx == 0: ...`), which is legitimate
 * and which this check does not try to prove. Erroring would forbid a valid
 * pattern to catch an invalid one.
 */

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Warning error code (1000+ range for warnings; see warning_unused_var.cpp).
constexpr int kCompositeInSpmdScopeCode = 1005;

/// The InCore composite collectives. Each is lowered by LowerCompositeOps into
/// a push loop plus a notify/wait barrier, none of which reads the block index.
bool IsCompositeCollective(const CallPtr& call) {
  return IsOp(call, "pld.tensor.allreduce") || IsOp(call, "pld.tensor.allgather") ||
         IsOp(call, "pld.tensor.reduce_scatter") || IsOp(call, "pld.tensor.broadcast") ||
         IsOp(call, "pld.tensor.barrier") || IsOp(call, "pld.tensor.all_to_all") ||
         IsOp(call, "pld.tensor.all_to_all_v");
}

/// Render the enclosing width when it is a compile-time constant, so the
/// message can name the actual multiplier rather than a generic "N".
std::string DescribeWidth(const ExprPtr& core_num) {
  if (auto width = As<ConstInt>(core_num)) {
    return std::to_string(width->value_);
  }
  return "N";
}

class CompositeInSpmdScopeChecker : public IRVisitor {
 public:
  explicit CompositeInSpmdScopeChecker(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

 protected:
  void VisitStmt_(const SpmdScopeStmtPtr& op) override {
    enclosing_widths_.push_back(op->core_num_);
    IRVisitor::VisitStmt_(op);
    enclosing_widths_.pop_back();
  }

  void VisitExpr_(const CallPtr& op) override {
    if (!enclosing_widths_.empty() && IsCompositeCollective(op)) {
      // A statically-known width of 1 executes exactly once — one block, one
      // transfer, one barrier notifier — so none of the duplication this check
      // exists to surface applies. Emitting here would be a false positive on
      // the very single-block pattern the message recommends.
      if (auto width_const = As<ConstInt>(enclosing_widths_.back());
          width_const && width_const->value_ == 1) {
        IRVisitor::VisitExpr_(op);
        return;
      }
      // Innermost enclosing scope decides how many blocks run this call.
      const std::string width = DescribeWidth(enclosing_widths_.back());
      std::ostringstream msg;
      msg << op->op_->name_ << " is inside a pl.spmd scope, which does NOT parallelise it. "
          << "The lowering never reads the block index, so all " << width
          << " blocks run the whole peer loop: the transfer is issued " << width
          << " times to the same peers rather than split between them, and the barrier — whose "
          << "expected credit is a compile-time 1 while " << width
          << " blocks each notify +1 — releases after a peer's first block rather than its last. "
          << "Issue the collective from a single-block scope, or guard it so one block executes "
          << "it. Multi-core collectives are available on the HOST rail via core_num.";
      diagnostics_.emplace_back(DiagnosticSeverity::Warning, "CompositeInSpmdScope",
                                kCompositeInSpmdScopeCode, msg.str(), op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::vector<ExprPtr> enclosing_widths_;
};

class CompositeInSpmdScopeWarningVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "CompositeInSpmdScope"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    CompositeInSpmdScopeChecker checker(diagnostics);
    checker.VisitProgram(program);
  }
};

}  // namespace

PropertyVerifierPtr CreateCompositeInSpmdScopeWarningVerifier() {
  return std::make_shared<CompositeInSpmdScopeWarningVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
