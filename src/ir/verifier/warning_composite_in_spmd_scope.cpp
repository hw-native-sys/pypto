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
 * decomposition, and the InCore rail has no such parameter for most of them:
 * only `pld.tensor.allreduce` has a working multi-core knob (HOST-rail
 * `core_num`); `pld.tensor.all_to_all_v` accepts `core_num` too, but only on
 * the managed CHIP/L2 rail, and only `core_num=1` is implemented there.
 *
 * LowerCompositeOps never reads the block index, so the emitted body is not a
 * function of the enclosing spmd width: the push loop's bounds are `nranks` and
 * its put offsets are `my_rank`. Every block therefore issues the *same*
 * transfers to the *same* peers — the traffic is duplicated N times, not
 * divided N ways. The barrier is affected too: its expected credit is the
 * compile-time constant 1 while N blocks each notify +1, so it releases once a
 * peer's *first* block has notified rather than its last. Nested `pl.spmd`
 * scopes compound: the effective multiplier is the product of every enclosing
 * width, not just the innermost one.
 *
 * None of that fails a test today. Every block writes byte-identical content,
 * so an early reader still observes correct values, and the epilogue subtracts
 * -1 per block so the signal still returns to zero. The cost is silent: an
 * N-fold traffic multiplier with no diagnostic. This check makes it visible.
 *
 * Runs `PrePipeline` — the composite Call must still exist, and
 * LowerCompositeOps replaces it during the pipeline. That is also before
 * InlineFunctions (pass 01) splices an Inline function's body into its call
 * sites, so a composite reached only through such a call — `pl.spmd(N):
 * self.helper()` where `helper` is Inline and calls a composite — would sit in
 * `helper`'s own function body, visited independently with no enclosing
 * `pl.spmd` in sight. This check descends into an Inline callee's body from
 * the call site itself to keep the enclosing-width context intact.
 *
 * A warning rather than an error, deliberately: a caller may guard the call so
 * only one block executes it (`if block_idx == 0: ...`), which is legitimate
 * and which this check does not try to prove. Erroring would forbid a valid
 * pattern to catch an invalid one.
 */

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
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

/// A width is "known one" only when it is a compile-time constant equal to 1.
bool IsKnownOne(const ExprPtr& width) {
  auto width_const = As<ConstInt>(width);
  return width_const && width_const->value_ == 1;
}

/// Every enclosing pl.spmd nested around the call multiplies how many blocks
/// run it, so only the innermost width being 1 is not enough to suppress the
/// warning — `pl.spmd(8): pl.spmd(1): allgather(...)` still runs 8 times.
bool AllEnclosingWidthsKnownOne(const std::vector<ExprPtr>& widths) {
  for (const auto& width : widths) {
    if (!IsKnownOne(width)) return false;
  }
  return true;
}

/// Render the total enclosing multiplier when every nested width is a
/// compile-time constant (their product), else fall back to a generic "N".
std::string DescribeTotalWidth(const std::vector<ExprPtr>& widths) {
  int64_t product = 1;
  for (const auto& width : widths) {
    auto width_const = As<ConstInt>(width);
    if (!width_const) return "N";
    product *= width_const->value_;
  }
  return std::to_string(product);
}

/// Only `pld.tensor.allreduce` has a working multi-core knob today
/// (HOST-rail `core_num`). `pld.tensor.all_to_all_v` accepts `core_num` too,
/// but only on the managed CHIP/L2 rail, and only `core_num=1` is implemented
/// there — `core_num > 1` is not yet functional. The remaining five composites
/// (allgather, reduce_scatter, broadcast, barrier, all_to_all) have no
/// multi-core alternative on any rail. Naming one that does not exist would
/// send the caller looking for a kwarg that is not there.
std::string DescribeMultiCoreAlternative(const CallPtr& op) {
  if (IsOp(op, "pld.tensor.allreduce")) {
    return "Multi-core collectives are available on the HOST rail via core_num.";
  }
  if (IsOp(op, "pld.tensor.all_to_all_v")) {
    return "A managed CHIP/L2 orchestration rail accepts core_num, but core_num > 1 "
           "is not yet implemented there.";
  }
  return "No multi-core alternative exists for this collective yet.";
}

class CompositeInSpmdScopeChecker : public IRVisitor {
 public:
  CompositeInSpmdScopeChecker(std::vector<Diagnostic>& diagnostics, ProgramPtr program)
      : diagnostics_(diagnostics), program_(std::move(program)) {}

 protected:
  void VisitStmt_(const SpmdScopeStmtPtr& op) override {
    enclosing_widths_.push_back(op->core_num_);
    IRVisitor::VisitStmt_(op);
    enclosing_widths_.pop_back();
  }

  void VisitExpr_(const CallPtr& op) override {
    if (!enclosing_widths_.empty() && IsCompositeCollective(op)) {
      // Every enclosing width statically known to be 1 executes exactly
      // once — one block, one transfer, one barrier notifier — so none of
      // the duplication this check exists to surface applies. Emitting here
      // would be a false positive on the single-block pattern the message
      // recommends.
      if (!AllEnclosingWidthsKnownOne(enclosing_widths_)) {
        const std::string width = DescribeTotalWidth(enclosing_widths_);
        std::ostringstream msg;
        msg << op->op_->name_ << " is inside a pl.spmd scope, which does NOT parallelise it. "
            << "The lowering never reads the block index, so all " << width
            << " blocks run the whole peer loop: the transfer is issued " << width
            << " times to the same peers rather than split between them, and the barrier — whose "
            << "expected credit is a compile-time 1 while " << width
            << " blocks each notify +1 — releases after a peer's first block rather than its last. "
            << "Issue the collective from a single-block scope, or guard it so one block executes "
            << "it. " << DescribeMultiCoreAlternative(op);
        diagnostics_.emplace_back(DiagnosticSeverity::Warning, "CompositeInSpmdScope",
                                  kCompositeInSpmdScopeCode, msg.str(), op->span_);
      }
    }
    DescendIntoInlineCallee(op);
    IRVisitor::VisitExpr_(op);
  }

 private:
  // This check runs PrePipeline, before InlineFunctions (pass 01) splices an
  // Inline function's body into its call sites. A composite collective
  // reached only through such a call — `for _ in pl.spmd(8): self.helper()`
  // where `helper` calls `pld.tensor.allreduce` — sits in `helper`'s own
  // function body, which VisitProgram visits on its own with an empty
  // enclosing_widths_ stack; IRVisitor's default Call handling only visits
  // args/attrs, never a callee's body. Without this, the duplication that
  // check exists to catch goes undiagnosed for this pattern. Descending here,
  // under the caller's live width stack, closes that gap.
  void DescendIntoInlineCallee(const CallPtr& op) {
    if (!program_ || !op || !op->op_) return;
    auto callee_var = As<GlobalVar>(op->op_);
    if (!callee_var) return;
    const std::string& name = callee_var->name_;
    // Guards a call cycle the pipeline has not yet rejected (this check runs
    // before InlineFunctions' own cycle detection) from recursing forever.
    if (!in_progress_inline_calls_.insert(name).second) return;
    if (FunctionPtr callee = program_->GetFunction(name);
        callee && callee->func_type_ == FunctionType::Inline && callee->body_) {
      VisitStmt(callee->body_);
    }
    in_progress_inline_calls_.erase(name);
  }

  std::vector<Diagnostic>& diagnostics_;
  ProgramPtr program_;
  std::vector<ExprPtr> enclosing_widths_;
  std::unordered_set<std::string> in_progress_inline_calls_;
};

class CompositeInSpmdScopeWarningVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "CompositeInSpmdScope"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    CompositeInSpmdScopeChecker checker(diagnostics, program);
    checker.VisitProgram(program);
  }
};

}  // namespace

PropertyVerifierPtr CreateCompositeInSpmdScopeWarningVerifier() {
  return std::make_shared<CompositeInSpmdScopeWarningVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
