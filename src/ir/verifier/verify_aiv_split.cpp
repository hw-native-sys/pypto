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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Flags any vector reduce op that collapses the split axis of a split-mode
// AIV/AIC function. Reduce ops are plain Calls with a non-null op_; Submits
// (GlobalVar callee, no op_) are naturally skipped by IsReduceOnSplitAxis, so
// no SubmitPtr override is needed (see pass-submit-awareness.md). One walk per
// function body, O(1) per Call → O(N) per function.
class ReduceOnSplitAxisVerifier : public IRVisitor {
 public:
  ReduceOnSplitAxisVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name, int split_dim)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), split_dim_(split_dim) {}

  void VisitExpr_(const CallPtr& op) override {
    if (op && split_axis::IsReduceOnSplitAxis(op, split_dim_)) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "AivSplitValid", 0,
          "Function '" + func_name_ + "': reduce op '" + op->op_->name_ +
              "' reduces over the split axis (dim " + std::to_string(split_dim_) +
              "); each AIV lane holds only half of the tile, so this produces a partial "
              "reduction. Reduce over the non-split axis, or gather the lanes back to a full "
              "tile (tile.aic_gather) before reducing.",
          op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  int split_dim_;
};

}  // namespace

// Verifies IRProperty::AivSplitValid: a split-mode AIV/AIC function must not
// contain a vector reduce over its split axis. The AUTO SplitVectorKernel path
// catches this inline (IsReduceOnSplitAxis throws during per-op halving), but
// the EXPLICIT split_aiv path bypasses that rewrite, so this verifier closes
// the gap. Gating on the split_aiv marker keeps the check scoped to exactly the
// bypassed subset; the AUTO path is already guaranteed clean.
class AivSplitValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AivSplitValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::AIC && func->func_type_ != FunctionType::AIV) continue;
      if (!func->HasAttr("split_aiv") || !func->GetAttr<bool>("split_aiv", false)) continue;
      auto mode = func->GetSplitMode();
      if (!mode.has_value() || mode.value() == SplitMode::None) continue;

      ReduceOnSplitAxisVerifier verifier(diagnostics, func->name_, split_axis::SplitDimension(mode.value()));
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateAivSplitValidPropertyVerifier() {
  return std::make_shared<AivSplitValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
