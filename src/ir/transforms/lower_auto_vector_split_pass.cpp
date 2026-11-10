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

// Preserve explicit AIV regions and localize ragged boundary valid extents.
// The public pass name is retained for compatibility. It no longer synthesizes
// shard/gather operations or rewrites the author's computation and addresses.

#include <algorithm>
#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {

namespace {

using core_affinity::ClassifyCallAffinity;
using core_affinity::CombineAffinity;
using core_affinity::CoreAffinity;
using split_axis::SplitDimension;

CallPtr AsCall(const ExprPtr& expr) { return std::dynamic_pointer_cast<const Call>(expr); }

void ValidateTransposeSplitHazard(const std::vector<StmtPtr>& stmts, int split_dim, const Span& region_span) {
  auto body = std::make_shared<SeqStmts>(stmts, region_span);
  auto hazard = split_axis::FindTransposeSplitHazard(body, split_dim);
  if (hazard.call) {
    const char* mode_name = (split_dim == 0) ? "UP_DOWN" : "LEFT_RIGHT";
    std::string where = hazard.result_name.empty() ? std::string() : " (result '" + hazard.result_name + "')";
    CHECK_SPAN(false, hazard.call->span_)
        << "LowerAutoVectorSplit: a pl.split_aiv(" << mode_name << ") region contains a tile.transpose"
        << where << " that swaps the split axis (dim " << split_dim
        << "). The transpose moves the per-lane split data to the other dimension, so the region cannot "
           "be split correctly. Fix it one of two ways: (1) remove the transpose, e.g. replace a "
           "transpose-then-row-index with a direct column slice such as pre[:, h:h+1]; or (2) move the "
           "transpose outside the pl.split_aiv region.";
  }
}

void ValidateFunctionTransposeSplitHazard(const FunctionPtr& func, SplitMode mode) {
  const int split_dim = SplitDimension(mode);
  auto hazard = split_axis::FindTransposeSplitHazard(func->body_, split_dim);
  if (!hazard.call) return;
  const char* mode_name = (split_dim == 0) ? "UP_DOWN" : "LEFT_RIGHT";
  std::string where = hazard.result_name.empty() ? std::string() : " (result '" + hazard.result_name + "')";
  CHECK_SPAN(false, hazard.call->span_)
      << "LowerAutoVectorSplit: kernel '" << func->name_ << "' requests pl.split(" << mode_name
      << ") but contains a tile.transpose" << where << " that swaps the split axis (dim " << split_dim
      << "). The split halves that axis while the transpose moves the data to the other dimension, so "
         "the split cannot be applied correctly. Fix it one of two ways: (1) drop the split — remove "
         "the pl.split optimization (or set attrs={\"split\": pl.SplitMode.NONE}); or (2) eliminate "
         "this transpose, e.g. replace a transpose-then-row-index with a direct column slice such as "
         "pre[:, h:h+1].";
}

std::vector<StmtPtr> LowerExplicitRegions(const std::vector<StmtPtr>& stmts) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    if (auto region = As<SplitAivScopeStmt>(stmt)) {
      auto body = transform_utils::FlattenToStmts(region->body_);
      if (body.empty()) continue;
      if (region->split_ != SplitMode::None) {
        const int split_dim = SplitDimension(region->split_);
        ValidateTransposeSplitHazard(body, split_dim, region->span_);
        // Only explicit boundary metadata depends on the lane. Computation
        // shapes and memory offsets are authored per lane, even when the region
        // has no crossing. Never infer sharding from the absence of a boundary.
        body = split_axis::LocalizeExplicitBoundaryValid(body, split_dim, region->span_);
      }
      auto lowered = MutableCopy(region);
      lowered->body_ = loop_repair::MakeBody(LowerExplicitRegions(body), region->span_);
      result.push_back(lowered);
      continue;
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto new_body = LowerExplicitRegions(transform_utils::FlattenToStmts(for_stmt->body_));
      auto new_for = MutableCopy(for_stmt);
      new_for->body_ = loop_repair::MakeBody(new_body, for_stmt->span_);
      result.push_back(new_for);
      continue;
    }
    // A SplitAivScopeStmt may also nest inside a while body; mirror the ForStmt
    // arm so the region body is lowered before ExpandMixedKernel consumes it.
    // A region must never reach the codegen
    // guard (which rejects any live SplitAivScopeStmt).
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto new_body = LowerExplicitRegions(transform_utils::FlattenToStmts(while_stmt->body_));
      auto new_while = MutableCopy(while_stmt);
      new_while->body_ = loop_repair::MakeBody(new_body, while_stmt->span_);
      result.push_back(new_while);
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_then = LowerExplicitRegions(transform_utils::FlattenToStmts(if_stmt->then_body_));
      std::optional<StmtPtr> new_else;
      if (if_stmt->else_body_.has_value()) {
        auto new_else_stmts = LowerExplicitRegions(transform_utils::FlattenToStmts(*if_stmt->else_body_));
        new_else = loop_repair::MakeBody(new_else_stmts, if_stmt->span_);
      }
      auto new_if = MutableCopy(if_stmt);
      new_if->then_body_ = loop_repair::MakeBody(new_then, if_stmt->span_);
      new_if->else_body_ = new_else;
      result.push_back(new_if);
      continue;
    }
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      result.push_back(std::make_shared<SeqStmts>(LowerExplicitRegions(seq->stmts_), seq->span_));
      continue;
    }
    // Out-of-region statement: full width, unchanged.
    result.push_back(stmt);
  }
  return result;
}

// Find the first live ``SplitAivScopeStmt`` in a body (O(N) single walk), so a
// caller can point a diagnostic at the author's ``for aiv_id in
// pl.split_aiv(...)`` line via the returned node's span. Unlike the hand-rolled
// walks above this is an IRVisitor, so it descends into ScopeStmt bodies too —
// that asymmetry is exactly what the post-lowering guard below exists to catch.
class SplitAivScopeFinder : public IRVisitor {
 public:
  SplitAivScopeStmtPtr found_;

 protected:
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override {
    if (!found_) found_ = op;
  }
};

SplitAivScopeStmtPtr FindFirstSplitAivScope(const StmtPtr& body) {
  if (!body) return nullptr;
  SplitAivScopeFinder finder;
  finder.VisitStmt(body);
  return finder.found_;
}

bool BodyContainsSplitAivScope(const StmtPtr& body) { return FindFirstSplitAivScope(body) != nullptr; }

// Find scopes that should have been outlined before explicit-region lowering.
class AnyScopeFinder : public IRVisitor {
 public:
  ScopeStmtPtr found_;

 protected:
  void VisitStmt_(const InCoreScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override { IRVisitor::VisitStmt(op->body_); }
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override { Record(op); }

 private:
  template <typename T>
  void Record(const std::shared_ptr<const T>& op) {
    if (!found_) found_ = op;
    IRVisitor::VisitStmt(op->body_);
  }
};

ScopeStmtPtr FindFirstScope(const StmtPtr& body) {
  if (!body) return nullptr;
  AnyScopeFinder finder;
  finder.VisitStmt(body);
  return finder.found_;
}

// Lower each explicit region independently and retain its wrapper. Non-split
// scopes must already have been outlined; lowering cannot cross their boundary.
FunctionPtr LowerExplicitRegionFunction(const FunctionPtr& func) {
  auto stmts = transform_utils::FlattenToStmts(func->body_);
  auto new_stmts = LowerExplicitRegions(stmts);
  StmtPtr new_body =
      (new_stmts.size() == 1) ? new_stmts[0] : std::make_shared<SeqStmts>(new_stmts, func->span_);

  // A region behind a non-split scope was unreachable by the lowering walk.
  auto enclosing_scope = FindFirstScope(new_body);
  auto survivor = enclosing_scope ? FindFirstSplitAivScope(enclosing_scope->body_) : nullptr;
  CHECK_SPAN(!survivor, survivor->span_)
      << "LowerAutoVectorSplit: this pl.split_aiv region is nested inside a scope and cannot be "
         "lowered — region lowering does not cross a scope boundary. The scope may be one you "
         "wrote ('with pl.at(...)') or the InCore scope the parser adds around a top-level "
         "pl.split_aiv. OutlineIncoreScopes only outlines scopes out of Opaque / Orchestration "
         "functions, so a scope inside a function declared pl.FunctionType.InCore reaches this "
         "pass intact. Fix it one of two ways: (1) declare the enclosing function with plain "
         "@pl.function / @pl.jit so the scope is outlined before this pass runs; or (2) move the "
         "pl.split_aiv region out of the enclosing scope.";

  // Also reject a non-split scope nested inside a retained region.
  CHECK_SPAN(!enclosing_scope, enclosing_scope->span_)
      << "LowerAutoVectorSplit: a scope survives inside a pl.split_aiv region body. Region "
         "lowering does not cross a scope boundary, so its execution context would be ambiguous. "
         "Every scope must already be outlined by the time this pass runs — declare the enclosing function "
         "with "
         "plain @pl.function / @pl.jit (Opaque) so OutlineIncoreScopes lifts it, or drop the "
         "scope from inside the region.";

  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  auto attrs = func->attrs_;
  attrs.erase(
      std::remove_if(attrs.begin(), attrs.end(), [](const auto& kv) { return kv.first == kAttrSplitAiv; }),
      attrs.end());
  attrs.emplace_back(kAttrSplitAiv, true);

  auto new_func = MutableCopy(func);
  new_func->body_ = cloned_body;
  new_func->attrs_ = std::move(attrs);
  return new_func;
}

CoreAffinity RollupAffinity(const std::vector<StmtPtr>& stmts) {
  CoreAffinity combined = CoreAffinity::SHARED;
  for (const auto& stmt : stmts) {
    CoreAffinity result = CoreAffinity::SHARED;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto call = AsCall(assign->value_)) result = ClassifyCallAffinity(call);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      if (auto call = AsCall(eval->expr_)) result = ClassifyCallAffinity(call);
    } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      result = RollupAffinity(transform_utils::FlattenToStmts(for_stmt->body_));
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      result = RollupAffinity(transform_utils::FlattenToStmts(if_stmt->then_body_));
      if (if_stmt->else_body_.has_value()) {
        result =
            CombineAffinity(result, RollupAffinity(transform_utils::FlattenToStmts(*if_stmt->else_body_)));
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      result = RollupAffinity(transform_utils::FlattenToStmts(while_stmt->body_));
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      // Include scopes so an unoutlined mixed AUTO body cannot evade the diagnostic.
      result = RollupAffinity(transform_utils::FlattenToStmts(scope->body_));
    } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      result = RollupAffinity(seq->stmts_);
    }
    combined = CombineAffinity(combined, result);
  }
  return combined;
}

// Classify only InCore AUTO candidates; standalone AIV/AIC kernels retain their own split path.
bool IsMixedCubeVector(const FunctionPtr& func) {
  if (!func->body_) return false;
  return RollupAffinity(transform_utils::FlattenToStmts(func->body_)) == CoreAffinity::MIXED;
}

}  // namespace

namespace pass {

Pass LowerAutoVectorSplit() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> functions;
    functions.reserve(program->functions_.size());
    bool changed = false;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ != FunctionType::InCore) {
        functions.push_back(func);
        continue;
      }
      if (BodyContainsSplitAivScope(func->body_)) {
        functions.push_back(LowerExplicitRegionFunction(func));
        changed = true;
        continue;
      }
      const auto mode = func->GetSplitMode();
      const bool explicit_split = func->GetAttr<bool>(kAttrSplitAiv, false);
      // Explicit regions and printed flat explicit IR carry derived split
      // metadata too. Only reject the old whole-function AUTO authoring path.
      CHECK_SPAN(!(mode && *mode != SplitMode::None && !explicit_split && IsMixedCubeVector(func)),
                 func->span_)
          << "Implicit AIV splitting is no longer supported in mixed kernel '" << func->name_
          << "'. Remove pl.split(...), use pl.split_aiv(2, mode=...), and write pl.aiv_shard / "
             "pl.aic_gather at cross-core boundaries with explicit per-lane shapes and offsets.";
      if (mode && *mode != SplitMode::None) ValidateFunctionTransposeSplitHazard(func, *mode);
      functions.push_back(func);
    }
    if (!changed) return program;
    return std::make_shared<Program>(functions, program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "LowerAutoVectorSplit", kLowerAutoVectorSplitProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
