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
 * Fuse a native FP4 cast and its sole NZ move at the CAST statement. Analysis
 * is immutable and linear: one syntactic read/definition/position walk, one
 * indexed candidate walk, then one rewrite. No statement interval is scanned.
 * Producer-site replacement preserves reads across intervening source writes
 * and preserves zero-trip/branch execution counts by requiring one sequence.
 */
#include "src/ir/transforms/fp4_cast_fusion.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/syntactic_use_analysis.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace {

bool IsFreshTile(const TypePtr& type) {
  auto tile = As<TileType>(type);
  return tile && !tile->memref_.has_value();
}

class NZFusionRewriter : public IRMutator {
 public:
  explicit NZFusionRewriter(const SyntacticUseAnalysis& analysis) {
    auto& registry = OpRegistry::GetInstance();
    for (const auto& [var, assignment] : analysis.definitions) {
      auto move = As<Call>(assignment->value_);
      if (!IsOp(move, "tile.move") || move->args_.size() != 1 || !move->attrs_.empty()) continue;
      bool supported_metadata = true;
      for (const auto& [key, value] : move->kwargs_) {
        if (key != "target_memory" && key != "blayout" && key != "slayout") supported_metadata = false;
      }
      if (!supported_metadata || !IsFreshTile(var->GetType()) || !IsFreshTile(move->GetType())) continue;
      auto cast_var = AsVarLike(move->args_[0]);
      if (!cast_var || !IsFreshTile(cast_var->GetType())) continue;
      const auto reads = analysis.read_counts.find(cast_var.get());
      const auto definition = analysis.definitions.find(cast_var.get());
      if (reads == analysis.read_counts.end() || reads->second != 1 ||
          definition == analysis.definitions.end())
        continue;
      const auto& cast_assignment = definition->second;
      auto cast = As<Call>(cast_assignment->value_);
      if (!IsOp(cast, "tile.cast") || cast->args_.size() != 1 || !IsFreshTile(cast->GetType())) continue;
      auto source = As<TileType>(cast->args_[0]->GetType());
      auto target = As<TileType>(cast->GetType());
      if (!source || !target || source->dtype_ != DataType::FP4E2M1X2 ||
          target->dtype_ != DataType::FP8E4M3FN)
        continue;
      const auto producer = analysis.positions.find(cast_assignment.get());
      const auto consumer = analysis.positions.find(assignment.get());
      if (producer == analysis.positions.end() || consumer == analysis.positions.end() ||
          producer->second.sequence != consumer->second.sequence ||
          producer->second.index >= consumer->second.index)
        continue;
      CallPtr fused;
      try {
        fused = registry.Create("tile.cast_to_nz", cast->args_, cast->kwargs_, cast->span_);
      } catch (const ValueError&) {
        continue;  // Ordinary ND conversion remains legal for unfusable geometry.
      } catch (const TypeError&) {
        continue;
      }
      // Complete equality prevents newly requiring a move-site expression at
      // the producer and keeps valid/pad/compact/addressing semantics exact.
      if (!structural_equal(fused->GetType(), assignment->var_->GetType()) ||
          !structural_equal(fused->GetType(), move->GetType()))
        continue;
      fused = transform_utils::PreserveCallAttrs(cast, fused);
      auto replacement = MutableCopy(cast_assignment);
      replacement->var_ = assignment->var_;
      replacement->value_ = fused;
      replacements_.emplace(cast_assignment.get(), replacement);
      removed_.insert(assignment.get());
    }
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto replacement = replacements_.find(op.get());
    return replacement == replacements_.end() ? op : replacement->second;
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> stmts;
    bool changed = false;
    for (const auto& stmt : op->stmts_) {
      if (removed_.count(stmt.get())) {
        changed = true;
        continue;
      }
      auto rewritten = VisitStmt(stmt);
      changed |= rewritten != stmt;
      stmts.push_back(std::move(rewritten));
    }
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->stmts_ = std::move(stmts);
    return result;
  }

 private:
  std::unordered_map<const Stmt*, StmtPtr> replacements_;
  std::unordered_set<const Stmt*> removed_;
};

// A padded output is selected only when its sole read is the standard V->C
// consumer, which strips the extra NZ row. Store/arithmetic/view consumers and
// loop-carried values conservatively keep ordinary NZ. SyntacticUseAnalysis
// includes Submit args/deps; the push collector intentionally recognizes plain
// intrinsics only, never turns an asynchronous Submit into a Call.
class PushSourceCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> sources;

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (IsOp(op, "tile.tpush_to_aic") && !op->args_.empty()) {
      if (auto source = AsVarLike(op->args_[0])) sources.insert(source.get());
    }
    IRVisitor::VisitExpr_(op);
  }
};

class PaddedNZRewriter : public IRMutator {
 public:
  explicit PaddedNZRewriter(const FunctionPtr& function) {
    SyntacticUseAnalysis analysis;
    analysis.VisitStmt(function->body_);
    PushSourceCollector pushes;
    pushes.VisitStmt(function->body_);
    auto& registry = OpRegistry::GetInstance();
    for (const auto& [var, assignment] : analysis.definitions) {
      auto cast = As<Call>(assignment->value_);
      auto reads = analysis.read_counts.find(var);
      if (!IsOp(cast, "tile.cast_to_nz") || !pushes.sources.count(var) ||
          reads == analysis.read_counts.end() || reads->second != 1 || !IsFreshTile(var->GetType()))
        continue;
      auto tile = As<TileType>(cast->GetType());
      if (!tile || tile_view_semantics::GetEffectiveTileView(*tile).compact != CompactMode::null) continue;
      auto kwargs = cast->kwargs_;
      kwargs.emplace_back("compact", static_cast<int>(CompactMode::row_plus_one));
      CallPtr padded;
      try {
        // The registered contract checks full 128x256 physical AND valid
        // geometry. No runtime-tail assertion or silent shape widening.
        padded = registry.Create("tile.cast_to_nz", cast->args_, kwargs, cast->span_);
      } catch (const ValueError&) {
        continue;
      } catch (const TypeError&) {
        continue;
      }
      calls_.emplace(cast.get(), transform_utils::PreserveCallAttrs(cast, padded));
      var_remap_.emplace(var, std::make_shared<Var>(assignment->var_->name_hint_, padded->GetType(),
                                                    assignment->var_->span_));
    }
  }

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto replacement = calls_.find(op.get());
    return replacement == calls_.end() ? IRMutator::VisitExpr_(op) : replacement->second;
  }

 private:
  std::unordered_map<const Call*, CallPtr> calls_;
};

}  // namespace

FunctionPtr FuseFp4CastToNZ(const FunctionPtr& func) {
  if (!func || !IsInCoreType(func->func_type_) || !backend::BackendConfig::IsConfigured()) return func;
  const auto* ctx = PassContext::Current();
  const auto* handler = ctx ? ctx->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
  if (!handler || !handler->SupportsFp4CastToNZ()) return func;
  SyntacticUseAnalysis analysis;
  analysis.VisitStmt(func->body_);
  NZFusionRewriter rewriter(analysis);
  auto fused = rewriter.VisitFunction(func);
  if (!ctx || !ctx->GetEnableFp4NZRowPlusOne()) return fused;
  PaddedNZRewriter padded(fused);
  return padded.VisitFunction(fused);
}

}  // namespace ir
}  // namespace pypto
