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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {

namespace {

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
      inside_auto_incore_ = true;
      auto new_body = VisitStmt(op->body_);
      inside_auto_incore_ = prev;
      // Consume the AutoInCore wrapper — return body directly
      return new_body;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (!inside_auto_incore_) {
      return IRMutator::VisitStmt_(op);
    }

    if (op->loop_origin_ == LoopOrigin::ChunkOuter) {
      return HandleChunkOuter(op);
    }

    if (op->loop_origin_ == LoopOrigin::ChunkRemainder) {
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
    return std::make_shared<SeqStmts>(new_stmts, op->span_);
  }

 private:
  bool inside_auto_incore_ = false;
  std::unordered_map<const Var*, ExprPtr> substitution_map_;

  /**
   * @brief Collect a chain of chunk loops starting from a ChunkOuter.
   *
   * Walk into nested ForStmt bodies, collecting (ForStmt, LoopOrigin) entries.
   * Stop at non-ForStmt or Original loop.
   */
  static std::vector<ChainEntry> CollectChunkChain(const ForStmtPtr& start) {
    std::vector<ChainEntry> chain;
    chain.push_back({start, start->loop_origin_});

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
      if (next_for->loop_origin_ == LoopOrigin::Original) break;

      chain.push_back({next_for, next_for->loop_origin_});
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
    // Recurse into the remainder body to handle nested chunk chains
    auto new_body = VisitStmt(op->body_);

    // Wrap standalone parallel ChunkRemainder sub-loops in InCore
    new_body = WrapSubRemainderLoopsInInCore(new_body, op->span_);

    if (new_body.get() == op->body_.get()) {
      return op;
    }

    return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                     new_body, op->return_vars_, op->span_, op->kind_, op->chunk_size_,
                                     op->chunk_policy_, op->loop_origin_);
  }

  /**
   * @brief Wrap standalone parallel ChunkRemainder ForStmts in InCore scopes.
   *
   * Scans top-level statements in body and wraps each ChunkRemainder loop that is
   * Parallel and whose body doesn't already contain InCore.
   */
  static StmtPtr WrapSubRemainderLoopsInInCore(const StmtPtr& body, const Span& span) {
    auto should_wrap = [](const StmtPtr& s) -> bool {
      auto fs = std::dynamic_pointer_cast<const ForStmt>(s);
      return fs && fs->loop_origin_ == LoopOrigin::ChunkRemainder && fs->kind_ == ForKind::Parallel &&
             !ContainsInCoreScope(fs->body_);
    };

    auto seq = std::dynamic_pointer_cast<const SeqStmts>(body);
    if (seq) {
      std::vector<StmtPtr> new_stmts;
      bool changed = false;
      for (const auto& s : seq->stmts_) {
        if (should_wrap(s)) {
          new_stmts.push_back(std::make_shared<ScopeStmt>(ScopeKind::InCore, s, span));
          changed = true;
        } else {
          new_stmts.push_back(s);
        }
      }
      if (!changed) return body;
      return std::make_shared<SeqStmts>(new_stmts, span);
    }

    // Single statement
    if (should_wrap(body)) {
      return std::make_shared<ScopeStmt>(ScopeKind::InCore, body, span);
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
    // Get the innermost body
    const auto& innermost = chain.back().for_stmt;
    auto body = VisitStmt(innermost->body_);

    // Build inners inside-out
    StmtPtr current = body;
    for (int i = static_cast<int>(inners.size()) - 1; i >= 0; --i) {
      const auto& inner = inners[i];
      current = std::make_shared<ForStmt>(inner->loop_var_, inner->start_, inner->stop_, inner->step_,
                                          std::vector<IterArgPtr>{}, current, std::vector<VarPtr>{},
                                          inner->span_, inner->kind_, std::nullopt, ChunkPolicy::LeadingFull,
                                          LoopOrigin::ChunkInner);
    }

    // Wrap in InCore
    current = std::make_shared<ScopeStmt>(ScopeKind::InCore, current, span);

    // Build outers inside-out
    for (int i = static_cast<int>(outers.size()) - 1; i >= 0; --i) {
      const auto& outer = outers[i];
      current = std::make_shared<ForStmt>(outer->loop_var_, outer->start_, outer->stop_, outer->step_,
                                          std::vector<IterArgPtr>{}, current, std::vector<VarPtr>{},
                                          outer->span_, ForKind::Sequential, std::nullopt,
                                          ChunkPolicy::LeadingFull, LoopOrigin::ChunkOuter);
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
        std::string ia_name = orig_ia->name_ + "_l" + std::to_string(loop_idx);
        std::string rv_name = orig_ia->name_ + "_l" + std::to_string(loop_idx) + "_rv";

        ExprPtr init_value;
        if (loop_idx == 0) {
          // Outermost: use original init values
          init_value = orig_ia->initValue_;
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

    // Set up substitutions: map original iter_args of the innermost loop to the
    // innermost new iter_args
    // We need to find the original chain order and map each loop's iter_args
    // to the corresponding new iter_args

    // Map each original loop's iter_args to the new iter_args at their new position
    // Original chain: O1, I1, O2, I2 (positions 0,1,2,3)
    // Reordered:      O1, O2, I1, I2 (positions 0,1,2,3)
    // So O1's iter_args (used by I1's body) should map to reordered position 0's iter_args
    // I1's iter_args (used by O2's body/init) should map to reordered position 2's iter_args
    // etc.

    // Build a mapping: for each original chain entry, find its position in reordered
    std::unordered_map<const ForStmt*, size_t> reordered_pos;
    for (size_t i = 0; i < total_loops; ++i) {
      reordered_pos[reordered[i].get()] = i;
    }

    // Now set up substitutions for the body:
    // The innermost loop in the reordered chain (last inner) passes its iter_args to the body.
    // We need to remap the original innermost loop's iter_args to the new innermost iter_args.
    const auto& orig_innermost = chain.back().for_stmt;
    size_t innermost_reordered_idx = reordered_pos[orig_innermost.get()];

    for (size_t ia_idx = 0; ia_idx < num_iter_args; ++ia_idx) {
      substitution_map_[orig_innermost->iter_args_[ia_idx].get()] =
          new_iter_args[innermost_reordered_idx][ia_idx];
    }

    // Visit the innermost body with substitutions
    auto body = VisitStmt(orig_innermost->body_);

    // Build the loop nest inside-out, starting from the innermost (last in reordered)
    StmtPtr current = body;

    for (int i = static_cast<int>(total_loops) - 1; i >= 0; --i) {
      const auto& orig_loop = reordered[i];
      bool is_inner = (orig_loop->loop_origin_ == LoopOrigin::ChunkInner);

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
          current = std::make_shared<SeqStmts>(std::vector<StmtPtr>{current, yield_stmt}, span);
        }
      }

      current = std::make_shared<ForStmt>(
          orig_loop->loop_var_, orig_loop->start_, orig_loop->stop_, orig_loop->step_, new_iter_args[i],
          current, new_return_vars[i], orig_loop->span_, is_inner ? orig_loop->kind_ : ForKind::Sequential,
          std::nullopt, ChunkPolicy::LeadingFull, is_inner ? LoopOrigin::ChunkInner : LoopOrigin::ChunkOuter);

      // Insert InCore scope right after building all inners (at the boundary)
      if (!is_inner && i + 1 < static_cast<int>(total_loops) &&
          reordered[i + 1]->loop_origin_ == LoopOrigin::ChunkInner) {
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
            incore_content = std::make_shared<SeqStmts>(incore_stmts, span);
          }

          auto incore_scope = std::make_shared<ScopeStmt>(ScopeKind::InCore, incore_content, span);
          auto new_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{incore_scope, last_stmt}, span);

          current = std::make_shared<ForStmt>(outer_for->loop_var_, outer_for->start_, outer_for->stop_,
                                              outer_for->step_, outer_for->iter_args_, new_body,
                                              outer_for->return_vars_, outer_for->span_, outer_for->kind_,
                                              std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkOuter);
        } else {
          // No yield, wrap entire body
          auto incore_scope = std::make_shared<ScopeStmt>(ScopeKind::InCore, incore_body, span);
          current = std::make_shared<ForStmt>(outer_for->loop_var_, outer_for->start_, outer_for->stop_,
                                              outer_for->step_, outer_for->iter_args_, incore_scope,
                                              outer_for->return_vars_, outer_for->span_, outer_for->kind_,
                                              std::nullopt, ChunkPolicy::LeadingFull, LoopOrigin::ChunkOuter);
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

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_);
}

}  // namespace

// Factory function
namespace pass {
Pass InterchangeChunkLoops() {
  return CreateFunctionPass(TransformInterchangeChunkLoops, "InterchangeChunkLoops",
                            kInterchangeChunkLoopsProperties);
}
}  // namespace pass

}  // namespace ir
}  // namespace pypto
