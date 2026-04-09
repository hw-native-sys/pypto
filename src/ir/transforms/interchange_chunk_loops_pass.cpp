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

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using Attrs = std::vector<std::pair<std::string, std::any>>;

namespace {

/// Build attrs for a generated loop: copy original attrs (excluding loop_origin) and set the new origin.
Attrs MakeLoopAttrs(const Attrs& original_attrs, LoopOrigin origin) {
  Attrs result;
  for (const auto& [key, value] : original_attrs) {
    if (key != "loop_origin") result.emplace_back(key, value);
  }
  result.emplace_back("loop_origin", origin);
  return result;
}

/**
 * @brief A single entry in a chunk-loop chain.
 */
struct ChainEntry {
  ForStmtPtr for_stmt;
  LoopOrigin origin;
};

/**
 * @brief Check if a statement body contains a ScopeStmt(InCore).
 */
static bool ContainsInCoreScope(const StmtPtr& stmt) {
  if (!stmt) return false;

  auto kind = stmt->GetKind();
  switch (kind) {
    case ObjectKind::ScopeStmt: {
      auto scope = std::static_pointer_cast<const ScopeStmt>(stmt);
      if (scope->scope_kind_ == ScopeKind::InCore) return true;
      return ContainsInCoreScope(scope->body_);
    }
    case ObjectKind::SeqStmts: {
      auto seq = std::static_pointer_cast<const SeqStmts>(stmt);
      for (const auto& s : seq->stmts_) {
        if (ContainsInCoreScope(s)) return true;
      }
      return false;
    }
    case ObjectKind::ForStmt: {
      auto for_stmt = std::static_pointer_cast<const ForStmt>(stmt);
      return ContainsInCoreScope(for_stmt->body_);
    }
    default:
      return false;
  }
}

static bool IsComputeTensorOp(const std::string& op_name) {
  return transform_utils::IsComputeTensorOp(op_name);
}

class ComputeTensorOpDetector : public IRVisitor {
 public:
  [[nodiscard]] bool Found() const { return found_; }

  void VisitExpr_(const CallPtr& op) override {
    if (!op || found_) return;
    if (op->op_ && IsComputeTensorOp(op->op_->name_)) {
      found_ = true;
      return;
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  bool found_ = false;
};

static bool ContainsComputeTensorOp(const StmtPtr& stmt) {
  if (!stmt) return false;
  ComputeTensorOpDetector detector;
  detector.VisitStmt(stmt);
  return detector.Found();
}

/// Detects whether an expression tree contains any sub-expression with TensorType or TileType.
class TensorOrTileTypedExprDetector : public IRVisitor {
 public:
  [[nodiscard]] bool Found() const { return found_; }

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr || found_) return;
    auto type = expr->GetType();
    if (type) {
      auto kind = type->GetKind();
      if (kind == ObjectKind::TensorType || kind == ObjectKind::TileType) {
        found_ = true;
        return;
      }
    }
    IRVisitor::VisitExpr(expr);
  }

 private:
  bool found_ = false;
};

/// Returns true if stmt is an AssignStmt with a scalar-typed target variable
/// and a value expression that involves no tensor/tile data.
static bool IsPureScalarAssignment(const StmtPtr& stmt) {
  if (!stmt) return false;

  auto kind = stmt->GetKind();
  if (kind == ObjectKind::AssignStmt) {
    auto assign = std::static_pointer_cast<const AssignStmt>(stmt);
    auto var_type = assign->var_->GetType();
    if (!var_type || var_type->GetKind() != ObjectKind::ScalarType) return false;
    TensorOrTileTypedExprDetector detector;
    detector.VisitExpr(assign->value_);
    return !detector.Found();
  }

  return false;
}

static bool ContainsChunkLoop(const StmtPtr& stmt) {
  if (!stmt) return false;

  auto kind = stmt->GetKind();
  switch (kind) {
    case ObjectKind::ForStmt: {
      auto for_stmt = std::static_pointer_cast<const ForStmt>(stmt);
      return for_stmt->GetAttr<LoopOrigin>("loop_origin") != LoopOrigin::Original ||
             ContainsChunkLoop(for_stmt->body_);
    }
    case ObjectKind::SeqStmts: {
      auto seq = std::static_pointer_cast<const SeqStmts>(stmt);
      for (const auto& s : seq->stmts_) {
        if (ContainsChunkLoop(s)) return true;
      }
      return false;
    }
    case ObjectKind::ScopeStmt: {
      auto scope = std::static_pointer_cast<const ScopeStmt>(stmt);
      return ContainsChunkLoop(scope->body_);
    }
    default:
      return false;
  }
}

/**
 * @brief Check whether a statement needs an InCore wrapper after auto_incore is consumed.
 *
 * We only wrap statements that still need outlining:
 * - compute tensor ops
 * - chunk loops that failed interchange or remain sequential
 *
 * The following stay in orchestration (not wrapped):
 * - Pure host-side groups (tensor.assemble/create/slice)
 * - Pure scalar assignments (e.g., index arithmetic like `offset = ob * 32`)
 *   whose value expression contains no tensor/tile-typed sub-expressions
 */
static bool NeedsInCoreWrapping(const StmtPtr& stmt) {
  if (!stmt) return false;

  auto kind = stmt->GetKind();
  if (kind == ObjectKind::YieldStmt || kind == ObjectKind::ReturnStmt) return false;
  if (ContainsInCoreScope(stmt)) return false;
  if (IsPureScalarAssignment(stmt)) return false;

  return ContainsChunkLoop(stmt) || ContainsComputeTensorOp(stmt);
}

/**
 * @brief Wrap statements that lack InCore coverage in ScopeStmt(InCore).
 *
 * After InterchangeChunkLoops processes the auto_incore body, some statements
 * (standalone tensor ops, non-chunked loops, failed-interchange chains) may
 * lack InCore wrapping. This function groups consecutive such statements and
 * wraps each group in ScopeStmt(InCore).
 *
 * Control flow statements (YieldStmt, ReturnStmt) are never wrapped.
 */
static StmtPtr WrapNonIncoreStatementsInInCore(const StmtPtr& body, const Span& span,
                                               std::optional<SplitMode> split = std::nullopt) {
  // When a ForStmt contains InCore scopes in its body (e.g. a pl.range loop
  // wrapping interchanged parallel chunks), recurse into it so that non-InCore
  // statements *inside* the loop body also get wrapped.
  auto maybe_recurse_into_compound = [&](const StmtPtr& s) -> StmtPtr {
    auto fs = std::dynamic_pointer_cast<const ForStmt>(s);
    if (fs && ContainsInCoreScope(fs->body_)) {
      auto new_body = WrapNonIncoreStatementsInInCore(fs->body_, span, split);
      if (new_body.get() != fs->body_.get()) {
        return std::make_shared<ForStmt>(fs->loop_var_, fs->start_, fs->stop_, fs->step_, fs->iter_args_,
                                         new_body, fs->return_vars_, fs->span_, fs->kind_, fs->chunk_config_,
                                         fs->attrs_);
      }
    }
    return s;
  };

  auto seq = std::dynamic_pointer_cast<const SeqStmts>(body);
  if (!seq) {
    if (NeedsInCoreWrapping(body)) {
      return std::make_shared<ScopeStmt>(ScopeKind::InCore, body, span, std::nullopt, std::nullopt, split);
    }
    return maybe_recurse_into_compound(body);
  }

  // Check if any wrapping or recursion is needed (fast path)
  bool has_work = false;
  for (const auto& s : seq->stmts_) {
    if (NeedsInCoreWrapping(s)) {
      has_work = true;
      break;
    }
    auto fs = std::dynamic_pointer_cast<const ForStmt>(s);
    if (fs && ContainsInCoreScope(fs->body_)) {
      has_work = true;
      break;
    }
  }
  if (!has_work) return body;

  // Group consecutive wrappable statements and wrap each group in InCore
  std::vector<StmtPtr> result;
  std::vector<StmtPtr> pending;

  auto flush = [&]() {
    if (pending.empty()) return;
    StmtPtr content = SeqStmts::Flatten(std::vector<StmtPtr>(pending), span);
    result.push_back(
        std::make_shared<ScopeStmt>(ScopeKind::InCore, content, span, std::nullopt, std::nullopt, split));
    pending.clear();
  };

  for (const auto& s : seq->stmts_) {
    if (NeedsInCoreWrapping(s)) {
      pending.push_back(s);
    } else {
      flush();
      result.push_back(maybe_recurse_into_compound(s));
    }
  }
  flush();

  return SeqStmts::Flatten(std::move(result), span);
}

/**
 * @brief Mutator that interchanges ChunkOuter/ChunkInner loops and inserts InCore scopes.
 *
 * After SplitChunkedLoops produces nested ChunkOuter → ChunkInner pairs,
 * this pass reorders them so all outers are on top, wraps inners + body
 * in ScopeStmt(InCore).
 *
 * Only interchanges when ALL ChunkInner loops in the chain have ForKind::Parallel.
 */
class InterchangeChunkLoopsMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = substitution_map_.find(op.get());
    if (it != substitution_map_.end()) {
      return it->second;
    }
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = substitution_map_.find(op.get());
    if (it != substitution_map_.end()) {
      return it->second;
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const ScopeStmtPtr& op) override {
    if (op->scope_kind_ == ScopeKind::AutoInCore) {
      bool prev = inside_auto_incore_;
      auto prev_split = current_split_;
      inside_auto_incore_ = true;
      current_split_ = op->split_;
      auto new_body = VisitStmt(op->body_);
      inside_auto_incore_ = prev;
      current_split_ = prev_split;
      // Consume the AutoInCore wrapper — return body directly.
      // Wrap any statements that lack InCore coverage, propagating split.
      new_body = WrapNonIncoreStatementsInInCore(new_body, op->span_, op->split_);
      return new_body;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (!inside_auto_incore_) {
      return IRMutator::VisitStmt_(op);
    }

    auto loop_origin = op->GetAttr<LoopOrigin>("loop_origin");
    if (loop_origin == LoopOrigin::ChunkOuter) {
      return HandleChunkOuter(op);
    }

    if (loop_origin == LoopOrigin::ChunkRemainder) {
      return HandleChunkRemainder(op);
    }

    // Non-chunk loop: recurse normally
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;

    for (const auto& stmt : op->stmts_) {
      auto new_stmt = VisitStmt(stmt);
      if (new_stmt.get() != stmt.get()) {
        changed = true;
      }
      // Flatten nested SeqStmts
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
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

 private:
  bool inside_auto_incore_ = false;
  bool inside_incore_context_ = false;
  std::optional<SplitMode> current_split_;
  std::unordered_map<const Var*, ExprPtr> substitution_map_;

  /**
   * @brief Visit a body that will be placed inside an InCore scope.
   *
   * Sets inside_incore_context_ so nested chains skip their own InCore wrapping.
   * Returns whether a parent chain already provides InCore context (prev value).
   */
  std::pair<StmtPtr, bool> VisitBodyInIncoreContext(const StmtPtr& body) {
    bool prev_incore = inside_incore_context_;
    inside_incore_context_ = true;
    auto result = VisitStmt(body);
    inside_incore_context_ = prev_incore;
    return {result, prev_incore};
  }

  /**
   * @brief Collect a chain of chunk loops starting from a ChunkOuter.
   *
   * Walk into nested ForStmt bodies, collecting (ForStmt, LoopOrigin) entries.
   * Stop at non-ForStmt or Original loop.
   */
  static std::vector<ChainEntry> CollectChunkChain(const ForStmtPtr& start) {
    std::vector<ChainEntry> chain;
    chain.push_back({start, start->GetAttr<LoopOrigin>("loop_origin")});

    StmtPtr body = start->body_;

    // Walk through SeqStmts to find the actual ForStmt body
    // (body can be SeqStmts with [for_loop, yield])
    while (true) {
      ForStmtPtr next_for;
      auto seq = std::dynamic_pointer_cast<const SeqStmts>(body);
      if (seq) {
        // Verify body is exactly {ForStmt} or {ForStmt, YieldStmt}
        // to ensure no side-effect statements are dropped during rebuild
        size_t for_count = 0;
        size_t yield_count = 0;
        for (const auto& s : seq->stmts_) {
          auto f = std::dynamic_pointer_cast<const ForStmt>(s);
          if (f) {
            next_for = f;
            ++for_count;
          } else if (s->GetKind() == ObjectKind::YieldStmt) {
            ++yield_count;
          } else {
            // Non-loop, non-yield statement found — not safe to interchange
            return chain;
          }
        }
        if (for_count != 1 || yield_count > 1) {
          return chain;
        }
      } else {
        next_for = std::dynamic_pointer_cast<const ForStmt>(body);
      }

      if (!next_for) break;
      auto next_origin = next_for->GetAttr<LoopOrigin>("loop_origin");
      if (next_origin == LoopOrigin::Original) break;

      chain.push_back({next_for, next_origin});
      body = next_for->body_;
    }

    return chain;
  }

  /**
   * @brief Handle a ChunkOuter loop: collect chain, check guards, interchange if applicable.
   */
  StmtPtr HandleChunkOuter(const ForStmtPtr& op) {
    auto chain = CollectChunkChain(op);

    // Separate into outers and inners
    std::vector<ForStmtPtr> outers;
    std::vector<ForStmtPtr> inners;
    for (const auto& entry : chain) {
      if (entry.origin == LoopOrigin::ChunkOuter) {
        outers.push_back(entry.for_stmt);
      } else if (entry.origin == LoopOrigin::ChunkInner) {
        inners.push_back(entry.for_stmt);
      }
    }

    // Guard: need at least 1 outer and 1 inner
    if (outers.empty() || inners.empty()) {
      return IRMutator::VisitStmt_(op);
    }

    // Guard: all loops in the chain must have compatible iter_arg arity
    const size_t ref_iter_args_size = chain.front().for_stmt->iter_args_.size();
    for (const auto& entry : chain) {
      if (entry.for_stmt->iter_args_.size() != ref_iter_args_size) {
        return IRMutator::VisitStmt_(op);
      }
    }

    // Guard: all ChunkInner loops must be Parallel
    for (const auto& inner : inners) {
      if (inner->kind_ != ForKind::Parallel) {
        return IRMutator::VisitStmt_(op);
      }
    }

    // Guard: no existing InCore scope in innermost body
    const auto& innermost = chain.back().for_stmt;
    if (ContainsInCoreScope(innermost->body_)) {
      return IRMutator::VisitStmt_(op);
    }

    // Warn if this interchange is nested inside a parent chain's InCore context
    if (inside_incore_context_) {
      LOG_WARN << op->span_.filename_ << ":" << op->span_.begin_line_ << " — "
               << "Nested chunked parallel loop found with intervening statements between it and its parent "
               << "chunked parallel — the inner chunk will share the parent's InCore scope instead of "
               << "getting its own. Consider removing the intervening statements or restructuring the loop "
               << "nest so the chunked parallels are directly nested.";
    }

    // Perform the interchange
    return RebuildInterchanged(outers, inners, chain, op->span_);
  }

  /**
   * @brief Handle a ChunkRemainder loop: recurse into body and wrap sub-remainder loops in InCore.
   *
   * After recursion handles nested chunk chains (via HandleChunkOuter), scan the visited body
   * for standalone parallel ChunkRemainder sub-loops and wrap each in InCore.
   */
  StmtPtr HandleChunkRemainder(const ForStmtPtr& op) {
    // Create new iter_args BEFORE visiting the body, and register old->new
    // IterArg mappings in substitution_map_ so body references get rewritten.
    std::vector<IterArgPtr> new_iter_args;
    bool iter_args_changed = false;
    new_iter_args.reserve(op->iter_args_.size());
    for (const auto& ia : op->iter_args_) {
      auto new_init = VisitExpr(ia->initValue_);
      if (new_init.get() != ia->initValue_.get()) {
        auto new_ia = std::make_shared<IterArg>(ia->name_hint_, ia->GetType(), new_init, ia->span_);
        new_iter_args.push_back(new_ia);
        // Register old -> new mapping so body references get rewritten
        substitution_map_[ia.get()] = new_ia;
        iter_args_changed = true;
      } else {
        new_iter_args.push_back(ia);
      }
    }

    // Recurse into the remainder body to handle nested chunk chains
    auto new_body = VisitStmt(op->body_);

    // Wrap standalone parallel ChunkRemainder sub-loops in InCore
    new_body = WrapSubRemainderLoopsInInCore(new_body, op->span_, current_split_);

    if (new_body.get() == op->body_.get() && !iter_args_changed) {
      return op;
    }

    return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, new_iter_args, new_body,
                                     op->return_vars_, op->span_, op->kind_, op->chunk_config_, op->attrs_);
  }

  /**
   * @brief Wrap standalone parallel ChunkRemainder ForStmts in InCore scopes.
   *
   * Scans top-level statements in body and wraps each ChunkRemainder loop that is
   * Parallel and whose body doesn't already contain InCore.
   */
  static StmtPtr WrapSubRemainderLoopsInInCore(const StmtPtr& body, const Span& span,
                                               std::optional<SplitMode> split = std::nullopt) {
    auto should_wrap = [](const StmtPtr& s) -> bool {
      auto fs = std::dynamic_pointer_cast<const ForStmt>(s);
      return fs && fs->GetAttr<LoopOrigin>("loop_origin") == LoopOrigin::ChunkRemainder &&
             fs->kind_ == ForKind::Parallel && !ContainsInCoreScope(fs->body_);
    };

    auto seq = std::dynamic_pointer_cast<const SeqStmts>(body);
    if (seq) {
      std::vector<StmtPtr> new_stmts;
      bool changed = false;
      for (const auto& s : seq->stmts_) {
        if (should_wrap(s)) {
          new_stmts.push_back(
              std::make_shared<ScopeStmt>(ScopeKind::InCore, s, span, std::nullopt, std::nullopt, split));
          changed = true;
        } else {
          new_stmts.push_back(s);
        }
      }
      if (!changed) return body;
      return SeqStmts::Flatten(std::move(new_stmts), span);
    }

    // Single statement
    if (should_wrap(body)) {
      return std::make_shared<ScopeStmt>(ScopeKind::InCore, body, span, std::nullopt, std::nullopt, split);
    }
    return body;
  }

  /**
   * @brief Rebuild the interchanged loop nest: outers on top, InCore { inners → body }.
   *
   * Original chain: O1 → I1 → O2 → I2 → body
   * Result:         O1 → O2 → InCore{ I1 → I2 → body }
   *
   * Iter_args are reconnected so the linear data flow is maintained:
   * O1.init(original) → O2.init(from O1 iter_arg) → I1.init(from O2 iter_arg)
   * → I2.init(from I1 iter_arg) → body
   * Yields reverse the data flow back out.
   */
  StmtPtr RebuildInterchanged(const std::vector<ForStmtPtr>& outers, const std::vector<ForStmtPtr>& inners,
                              const std::vector<ChainEntry>& chain, const Span& span) {
    bool has_iter_args = !chain[0].for_stmt->iter_args_.empty();

    if (!has_iter_args) {
      return RebuildSimple(outers, inners, chain, span);
    }

    return RebuildWithIterArgs(outers, inners, chain, span);
  }

  /**
   * @brief Simple rebuild without iter_args.
   */
  StmtPtr RebuildSimple(const std::vector<ForStmtPtr>& outers, const std::vector<ForStmtPtr>& inners,
                        const std::vector<ChainEntry>& chain, const Span& span) {
    // Get the body from the last loop in inners (not chain.back(), which may be a remainder)
    const auto& innermost = inners.back();

    auto [body, prev_incore] = VisitBodyInIncoreContext(innermost->body_);

    // Build inners inside-out
    StmtPtr current = body;
    for (int i = static_cast<int>(inners.size()) - 1; i >= 0; --i) {
      const auto& inner = inners[i];
      current = std::make_shared<ForStmt>(inner->loop_var_, inner->start_, inner->stop_, inner->step_,
                                          std::vector<IterArgPtr>{}, current, std::vector<VarPtr>{},
                                          inner->span_, inner->kind_, std::nullopt,
                                          MakeLoopAttrs(inner->attrs_, LoopOrigin::ChunkInner));
    }

    // Wrap in InCore — skip if a parent chain already provides InCore context
    if (!prev_incore) {
      current = std::make_shared<ScopeStmt>(ScopeKind::InCore, current, span, std::nullopt, std::nullopt,
                                            current_split_);
    }

    // Build outers inside-out, preserving the original ForKind.
    for (int i = static_cast<int>(outers.size()) - 1; i >= 0; --i) {
      const auto& outer = outers[i];
      current = std::make_shared<ForStmt>(outer->loop_var_, outer->start_, outer->stop_, outer->step_,
                                          std::vector<IterArgPtr>{}, current, std::vector<VarPtr>{},
                                          outer->span_, outer->kind_, std::nullopt,
                                          MakeLoopAttrs(outer->attrs_, LoopOrigin::ChunkOuter));
    }

    return current;
  }

  /**
   * @brief Rebuild with iter_args, reconnecting the SSA data flow.
   *
   * Original chain passes iter_args linearly through nested loops:
   *   O1.init(x_0) → I1.init(from O1_ia) → O2.init(from I1_ia) → I2.init(from O2_ia) → body
   *
   * After interchange: O1 → O2 → InCore{ I1 → I2 → body }
   * New data flow:
   *   O1.init(x_0) → O2.init(from O1_ia) → I1.init(from O2_ia) → I2.init(from I1_ia) → body
   */
  StmtPtr RebuildWithIterArgs(const std::vector<ForStmtPtr>& outers, const std::vector<ForStmtPtr>& inners,
                              const std::vector<ChainEntry>& chain, const Span& span) {
    // Reorder the chain entries: outers first, then inners
    std::vector<ForStmtPtr> reordered;
    reordered.reserve(outers.size() + inners.size());
    for (const auto& o : outers) reordered.push_back(o);
    for (const auto& i : inners) reordered.push_back(i);

    size_t num_iter_args = chain[0].for_stmt->iter_args_.size();
    size_t total_loops = reordered.size();

    // Create fresh iter_args and return_vars for each loop in the reordered chain
    std::vector<std::vector<IterArgPtr>> new_iter_args(total_loops);
    std::vector<std::vector<VarPtr>> new_return_vars(total_loops);

    // The outermost loop gets the original init values from the first chain entry
    const auto& first_orig = chain[0].for_stmt;

    for (size_t loop_idx = 0; loop_idx < total_loops; ++loop_idx) {
      const auto& orig_loop = reordered[loop_idx];
      for (size_t ia_idx = 0; ia_idx < num_iter_args; ++ia_idx) {
        const auto& orig_ia = first_orig->iter_args_[ia_idx];
        auto parsed_name = auto_name::Parse(orig_ia->name_hint_);
        std::string loop_qualifier = auto_name::LoopLevelQualifier(static_cast<int>(loop_idx));
        std::string combined_qualifier =
            parsed_name.qualifier.empty() ? loop_qualifier : parsed_name.qualifier + "_" + loop_qualifier;
        std::string ia_name =
            auto_name::BuildName(parsed_name.base_name, combined_qualifier, "iter", parsed_name.version);
        std::string rv_name =
            auto_name::BuildName(parsed_name.base_name, combined_qualifier, "rv", parsed_name.version);

        ExprPtr init_value;
        if (loop_idx == 0) {
          // Outermost: use original init values (apply substitutions for nested chains)
          init_value = VisitExpr(orig_ia->initValue_);
        } else {
          // Chain from previous loop's iter_arg
          init_value = new_iter_args[loop_idx - 1][ia_idx];
        }

        auto new_ia = std::make_shared<IterArg>(ia_name, orig_ia->GetType(), init_value, orig_ia->span_);
        auto new_rv = std::make_shared<Var>(rv_name, orig_ia->GetType(), orig_ia->span_);

        new_iter_args[loop_idx].push_back(new_ia);
        new_return_vars[loop_idx].push_back(new_rv);
      }
    }

    // Now set up substitutions for the body:
    // The last loop in reordered (last inner) passes its iter_args to the body.
    // We remap its original iter_args to the new innermost iter_args.
    // Note: chain.back() may be a ChunkRemainder that is NOT in reordered,
    // so we must use reordered.back() to get the actual innermost interchange loop.
    const auto& orig_innermost = reordered.back();
    size_t innermost_reordered_idx = total_loops - 1;

    for (size_t ia_idx = 0; ia_idx < num_iter_args; ++ia_idx) {
      substitution_map_[orig_innermost->iter_args_[ia_idx].get()] =
          new_iter_args[innermost_reordered_idx][ia_idx];
    }

    // Visit the innermost body with substitutions
    auto [body, prev_incore] = VisitBodyInIncoreContext(orig_innermost->body_);

    // Build the loop nest inside-out, starting from the innermost (last in reordered)
    StmtPtr current = body;

    for (int i = static_cast<int>(total_loops) - 1; i >= 0; --i) {
      const auto& orig_loop = reordered[i];
      bool is_inner = (orig_loop->GetAttr<LoopOrigin>("loop_origin") == LoopOrigin::ChunkInner);

      // Build yield for this loop from the inner loop's return_vars
      // (or body's yield values for the innermost)
      if (!new_return_vars[i].empty()) {
        std::vector<ExprPtr> yield_values;
        if (i < static_cast<int>(total_loops) - 1) {
          // Yield the return vars of the next inner loop
          for (const auto& rv : new_return_vars[i + 1]) {
            yield_values.push_back(rv);
          }
        } else {
          // Innermost: body already contains yield, current already has it
          // Don't add extra yield
        }

        if (!yield_values.empty()) {
          auto yield_stmt = std::make_shared<YieldStmt>(yield_values, span);
          current = SeqStmts::Flatten(std::vector<StmtPtr>{current, yield_stmt}, span);
        }
      }

      current = std::make_shared<ForStmt>(
          orig_loop->loop_var_, orig_loop->start_, orig_loop->stop_, orig_loop->step_, new_iter_args[i],
          current, new_return_vars[i], orig_loop->span_, orig_loop->kind_, std::nullopt,
          MakeLoopAttrs(orig_loop->attrs_, is_inner ? LoopOrigin::ChunkInner : LoopOrigin::ChunkOuter));

      // Insert InCore scope right after building all inners (at the boundary).
      // Skip if a parent chain already provides InCore context.
      if (!prev_incore && !is_inner && i + 1 < static_cast<int>(total_loops) &&
          reordered[i + 1]->GetAttr<LoopOrigin>("loop_origin") == LoopOrigin::ChunkInner) {
        // The current ForStmt body already contains the inner loops.
        // We need to wrap the inner loop nest (current's body) in InCore.
        // But current IS the outermost outer that contains inners already.
        // Actually, we need to insert InCore between the last outer and first inner.
        // Let's restructure: wrap the body of this outer in InCore.
        auto outer_for = std::static_pointer_cast<const ForStmt>(current);

        // Extract the body (which is inners + yield)
        auto incore_body = outer_for->body_;
        // Separate the yield at the end from the body content
        auto body_seq = std::dynamic_pointer_cast<const SeqStmts>(incore_body);
        if (body_seq && body_seq->stmts_.size() >= 2) {
          // Last stmt should be yield, rest goes into InCore
          std::vector<StmtPtr> incore_stmts;
          incore_stmts.reserve(body_seq->stmts_.size() - 1);
          for (size_t si = 0; si < body_seq->stmts_.size() - 1; ++si) {
            incore_stmts.push_back(body_seq->stmts_[si]);
          }
          auto last_stmt = body_seq->stmts_.back();

          StmtPtr incore_content;
          if (incore_stmts.size() == 1) {
            incore_content = incore_stmts[0];
          } else {
            incore_content = SeqStmts::Flatten(std::move(incore_stmts), span);
          }

          auto incore_scope = std::make_shared<ScopeStmt>(ScopeKind::InCore, incore_content, span,
                                                          std::nullopt, std::nullopt, current_split_);
          auto new_body = SeqStmts::Flatten(std::vector<StmtPtr>{incore_scope, last_stmt}, span);

          current = std::make_shared<ForStmt>(
              outer_for->loop_var_, outer_for->start_, outer_for->stop_, outer_for->step_,
              outer_for->iter_args_, new_body, outer_for->return_vars_, outer_for->span_, outer_for->kind_,
              std::nullopt, MakeLoopAttrs(outer_for->attrs_, LoopOrigin::ChunkOuter));
        } else {
          // No yield, wrap entire body
          auto incore_scope = std::make_shared<ScopeStmt>(ScopeKind::InCore, incore_body, span, std::nullopt,
                                                          std::nullopt, current_split_);
          current = std::make_shared<ForStmt>(
              outer_for->loop_var_, outer_for->start_, outer_for->stop_, outer_for->step_,
              outer_for->iter_args_, incore_scope, outer_for->return_vars_, outer_for->span_,
              outer_for->kind_, std::nullopt, MakeLoopAttrs(outer_for->attrs_, LoopOrigin::ChunkOuter));
        }
      }
    }

    // Remap original outer return_vars to new outermost return_vars
    for (size_t ia_idx = 0; ia_idx < num_iter_args; ++ia_idx) {
      substitution_map_[first_orig->return_vars_[ia_idx].get()] = new_return_vars[0][ia_idx];
    }

    return current;
  }
};

/**
 * @brief Transform a function by interchanging chunk loops and inserting InCore scopes.
 */
FunctionPtr TransformInterchangeChunkLoops(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "InterchangeChunkLoops cannot run on null function";

  InterchangeChunkLoopsMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);

  if (new_body.get() == func->body_.get()) {
    return func;
  }

  auto result = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                           func->return_types_, new_body, func->span_, func->func_type_,
                                           func->level_, func->role_, func->attrs_);
  return result;
}

}  // namespace

// Factory function
namespace pass {
Pass InterchangeChunkLoops() {
  return CreateFunctionPass(TransformInterchangeChunkLoops, "InterchangeChunkLoops",
                            kInterchangeChunkLoopsProperties);
}
}  // namespace pass

// ============================================================================
// NoNestedInCore structural property verifier
// ============================================================================

namespace {

constexpr int kNestedIncoreCode = 501;

/// Detects nested ScopeStmt(InCore) scopes in an IR tree.
class NestedInCoreScopeDetector : public IRVisitor {
 public:
  explicit NestedInCoreScopeDetector(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (!op) return;
    if (op->scope_kind_ == ScopeKind::InCore) {
      if (inside_incore_) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "NoNestedInCore", kNestedIncoreCode,
                                  "Nested InCore scope detected — InCore scopes must not contain other "
                                  "InCore scopes",
                                  op->span_);
      }
      bool prev = inside_incore_;
      inside_incore_ = true;
      IRVisitor::VisitStmt_(op);
      inside_incore_ = prev;
    } else {
      IRVisitor::VisitStmt_(op);
    }
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  bool inside_incore_ = false;
};

}  // namespace

class NoNestedIncorePropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "NoNestedInCore"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      NestedInCoreScopeDetector detector(diagnostics);
      detector.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateNoNestedIncorePropertyVerifier() {
  return std::make_shared<NoNestedIncorePropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
