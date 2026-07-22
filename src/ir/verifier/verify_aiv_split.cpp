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
#include <optional>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Everything that differs between the two split-axis boundary ops: the memory
/// space each side must carry, plus the direction-specific diagnostic wording.
/// Keeping it in one table row per op means a reader (and an editor) sees the
/// whole contract for an op in one place, and the check itself stays uniform.
///
/// `result` mirrors the op's set_output_memory declaration (cross_core.cpp),
/// restated here as the checkable form of the same contract. `operand` is
/// deliberately NOT declared as a set_input_memory constraint over there — a
/// violated input constraint makes InferTileMemorySpace insert a physically
/// impossible move instead of reporting the authoring error — so this verifier
/// is the only place it is stated at all.
struct BoundaryMemoryContract {
  MemorySpace operand;       ///< space on the PRODUCING lane (the op's input)
  MemorySpace result;        ///< space on the CONSUMING lane (the op's output)
  const char* producer;      ///< lane that must have produced the operand
  const char* delivery;      ///< what the result hands to the consuming lane
  const char* operand_hint;  ///< authoring fix appended to the operand diagnostic
};

/// The contract for a tile-level boundary op, or nullopt for anything else.
/// The tensor.* forms are deliberately excluded: a TensorType has no memory
/// space, so there is nothing to check until ConvertTensorToTileOps lowers them.
std::optional<BoundaryMemoryContract> GetBoundaryMemoryContract(const CallPtr& op) {
  if (IsOp(op, "tile.aiv_shard")) {
    // The hint stays space-neutral: the check rejects any non-Acc operand, and
    // only Vec means "vector-produced". A Mat operand is a different mistake
    // (L1 is not a supported cross-core producer pipe), so naming pl.load /
    // pl.full unconditionally would misdescribe it.
    return BoundaryMemoryContract{
        MemorySpace::Acc, MemorySpace::Vec, "cube", "half to the vector",
        " Only a cube-produced value reaches the vector lane through this boundary: Acc is the "
        "matmul result the c2v pipe can push. A Vec operand is vector-produced (pl.load / "
        "pl.full) and already lives on the AIV lane — drop the pl.aiv_shard and let the implicit "
        "affinity-gated split halve it. A Mat operand is not a supported producer pipe; move it "
        "through a pl.matmul, or load it on the vector lane instead."};
  }
  if (IsOp(op, "tile.aic_gather")) {
    return BoundaryMemoryContract{
        MemorySpace::Vec, MemorySpace::Mat, "vector", "full tile to the cube",
        " Gather the value only after it has been computed by vector ops on the AIV lane."};
  }
  return std::nullopt;
}

/// Memory space of `expr` when it is a tile whose space is already resolved.
/// nullopt means "not a tile" or "space not yet inferred" — both are skip
/// conditions, since the verifier also runs before InferTileMemorySpace.
std::optional<MemorySpace> ResolvedTileMemory(const ExprPtr& expr) {
  if (!expr) return std::nullopt;
  auto tile_type = As<TileType>(expr->GetType());
  if (!tile_type) return std::nullopt;
  return tile_type->memory_space_;
}

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
//       inside a region, never at top level. (c') additionally rejects them in a
//       task-parallel (SplitMode::None) region: both lanes run the full body
//       there, so there is no split axis to shard / gather along.
//   (d) The boundary memory contract: tile.aiv_shard is Acc -> Vec and
//       tile.aic_gather is Vec -> Mat (see cross_core.cpp). Both ops ARE the
//       cross-core transfer, so the operand must live on the PRODUCING lane and
//       the result on the CONSUMING one. Each side is skipped until its memory
//       space is resolved, so the check is inert at the OutlineIncoreScopes
//       verification point (where the boundary is still the space-less
//       `tensor.*` form) and live from ConvertTensorToTileOps onwards — which is
//       why both that pass and InferTileMemorySpace re-produce this property
//       (see pass_properties.h); without that, (d) would never run.
//
// Check (e) ("no bare vector compute outside a region") is DELIBERATELY OMITTED:
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
      // ConvertTensorToTileOps lowers them 1:1).
      //
      // The two flags have different domains: region-scoping (c)/(c') applies to
      // BOTH forms, while the memory contract (d) is meaningful only once the
      // operand and result are tiles — a TensorType carries no memory space.
      // Keeping them apart lets (d) skip the tensor.* forms without re-deriving
      // the op identity inside CheckBoundaryMemory.
      const bool tile_boundary = IsOp(op, "tile.aiv_shard") || IsOp(op, "tile.aic_gather");
      const bool boundary = tile_boundary || IsOp(op, "tensor.aiv_shard") || IsOp(op, "tensor.aic_gather");
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
        // (d) The boundary memory contract — tile forms only (see above).
        if (tile_boundary) CheckBoundaryMemory(op);
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
  /// (d) Enforce the producing-lane / consuming-lane memory spaces of a
  /// split-axis boundary op. Both sides are skipped until their space is
  /// resolved, so the same verifier is safe to run across the whole
  /// OutlineIncoreScopes .. LowerAutoVectorSplit window.
  void CheckBoundaryMemory(const CallPtr& op) {
    auto contract = GetBoundaryMemoryContract(op);
    if (!contract) return;

    // Operand side: the value must still sit on the lane that produced it. The
    // dominant failure is a shard of a vector-produced value: the use migrates to
    // the AIV half while the producer stays behind, leaving the cube half
    // referencing a value it never defines (which surfaces much later as an orphan
    // Mem.Vec allocation and an internal codegen error).
    if (!op->args_.empty()) {
      if (auto operand_ms = ResolvedTileMemory(op->args_[0]);
          operand_ms.has_value() && *operand_ms != contract->operand) {
        Err(op->span_, "'" + op->op_->name_ + "' operand is in " + MemorySpaceToString(*operand_ms) +
                           ", but it transfers a " + contract->producer +
                           "-produced value across the cross-core boundary and requires " +
                           MemorySpaceToString(contract->operand) + "." + contract->operand_hint);
      }
    }

    // Result side: the declared type describes the CONSUMING lane, i.e. the space
    // ExpandMixedKernel materializes the boundary tpop in.
    if (auto result_ms = ResolvedTileMemory(op); result_ms.has_value() && *result_ms != contract->result) {
      Err(op->span_, "'" + op->op_->name_ + "' result is in " + MemorySpaceToString(*result_ms) +
                         ", but it delivers its " + contract->delivery + " lane and must be " +
                         MemorySpaceToString(contract->result) +
                         " (the memory ExpandMixedKernel pops it into).");
    }
  }

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
