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
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Structural verifier for the first-class SplitAivScopeStmt region (live between
// OutlineIncoreScopes and LowerAutoVectorSplit). It keys every check on the node
// itself — tracking region nesting via VisitStmt_(SplitAivScopeStmtPtr) — rather
// than on a function-level split_aiv attr, so multi-mode / nested / sub-region
// functions are checked region by region.
//
// Checks performed:
//   (a) No cube compute inside a region — each AIV lane only holds half the
//       tile, so cube ops (matmul/mmad family) cannot be vector-split.
//   (b) No AIV reduce that collapses the split axis inside a region — that
//       produces a partial per-lane reduction (a miscompile).
//   (c) tile.aiv_shard / tile.aic_gather (the AIV-split boundary) must appear
//       inside a region, never at top level.
//
// Check (d) ("no bare vector compute outside a region") is DELIBERATELY OMITTED:
// full-width vector compute outside a region is now legal (multi-mode goal).
//
// The checked ops (matmul, reduces, aiv_shard/aic_gather) are always plain Calls
// with a non-null op_; Submits carry a GlobalVar callee and no op_, so no
// SubmitPtr override is needed (see pass-submit-awareness.md).
class SplitAivStructuralVerifier : public IRVisitor {
 public:
  explicit SplitAivStructuralVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const SplitAivScopeStmtPtr& op) override {
    INTERNAL_CHECK_SPAN(op->body_, op->span_) << "Internal error: SplitAivScopeStmt has null body";
    int prev_split_dim = cur_split_dim_;
    ++depth_;
    // A task-parallel (None) region has NO split axis — both lanes run the full
    // body, dispatched via aiv_id. Mark cur_split_dim_ = -1 so the split-axis
    // rules (a)/(b) are skipped for it; only the boundary-op rejection applies.
    cur_split_dim_ = (op->split_ == SplitMode::None) ? -1 : split_axis::SplitDimension(op->split_);
    IRVisitor::VisitStmt(op->body_);
    --depth_;
    cur_split_dim_ = prev_split_dim;
  }

  void VisitExpr_(const CallPtr& op) override {
    if (op && op->op_) {
      // The AIV-split boundary appears in two forms in this window: the tile-level
      // tile.aiv_shard / tile.aic_gather (AUTO split_aiv path, and the outlined
      // low-level form) and the author-facing tensor.aiv_shard / tensor.aic_gather
      // (pl.aiv_shard(tensor) inside a pl.split_aiv region, still tensor.* until
      // ConvertTensorToTileOps lowers them 1:1). Both must be region-scoped.
      const bool boundary = IsOp(op, "tile.aiv_shard") || IsOp(op, "tile.aic_gather") ||
                            IsOp(op, "tensor.aiv_shard") || IsOp(op, "tensor.aic_gather");
      const bool in_split_region = depth_ > 0 && cur_split_dim_ != -1;  // data-parallel (UpDown/LeftRight)
      const bool in_none_region = depth_ > 0 && cur_split_dim_ == -1;   // task-parallel (None)
      if (in_split_region) {
        // (a) Cube compute cannot live inside a vector-split region.
        if (!boundary && core_affinity::ClassifyCallAffinity(op) == core_affinity::CoreAffinity::CUBE) {
          Err(op->span_, "cube op '" + op->op_->name_ +
                             "' inside a pl.split_aiv region cannot be vector-split (each AIV lane "
                             "holds only half the tile); move it outside the region, or gather the "
                             "lanes back to a full tile (tile.aic_gather) first.");
        }
        // (b) A reduce over the split axis yields a partial per-lane result.
        if (split_axis::IsReduceOnSplitAxis(op, cur_split_dim_)) {
          Err(op->span_, "reduce op '" + op->op_->name_ + "' reduces over the split axis (dim " +
                             std::to_string(cur_split_dim_) +
                             ") inside a pl.split_aiv region, producing a partial reduction; reduce "
                             "the non-split axis, or gather the lanes back (tile.aic_gather) before "
                             "reducing.");
        }
      } else if (in_none_region) {
        // (c') A boundary op needs a split axis to mark — none exists in a
        // task-parallel (mode=NONE) region. Cube / reduce / full-width vector
        // ops are all fine here (both lanes run the full body).
        if (boundary) {
          Err(op->span_, "'" + op->op_->name_ +
                             "' cannot appear inside a task-parallel pl.split_aiv region "
                             "(mode=pl.SplitMode.NONE): there is no split axis to shard / gather. Use "
                             "mode=pl.SplitMode.UP_DOWN / LEFT_RIGHT for data-parallel halving instead.");
        }
      } else if (boundary) {
        // (c) The AIV-split boundary op escaped its region.
        Err(op->span_, "'" + op->op_->name_ +
                           "' must appear inside a pl.split_aiv region (it marks the AIV-split "
                           "boundary and is only meaningful there).");
      }
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  void Err(const Span& span, const std::string& message) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "AivSplitValid", 0, message, span);
  }

  std::vector<Diagnostic>& diagnostics_;
  int depth_ = 0;           ///< Region nesting depth (>0 means inside a split_aiv region).
  int cur_split_dim_ = -1;  ///< Split axis of the innermost enclosing region (-1 outside any region).
};

}  // namespace

// Verifies IRProperty::AivSplitValid as a structural property of the first-class
// SplitAivScopeStmt region. The node is live only between OutlineIncoreScopes
// (which produces the property) and LowerAutoVectorSplit (which consumes/erases
// the node and invalidates the property), so the verifier walks every function
// body in that window and applies the region-scoped checks above. No
// function-attr / split-mode gate — the node itself is the source of truth.
class AivSplitValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AivSplitValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      SplitAivStructuralVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateAivSplitValidPropertyVerifier() {
  return std::make_shared<AivSplitValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
