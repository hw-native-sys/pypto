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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/flatten_tile_nd_to_2d/internal.h"
#include "src/ir/transforms/flatten_tile_nd_to_2d/rewrite_internal.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;
using transform_utils::Substitute;

namespace flatten_tile_nd_to_2d {
namespace rewrite_internal {

/**
 * @brief Recursively transform statements, flattening >2D tile ops to 2D.
 */
std::vector<StmtPtr> TransformBody(const std::vector<StmtPtr>& stmts, FlattenContext& ctx,
                                   const OpRegistry& op_registry, const Span& span) {
  std::vector<StmtPtr> result;

  // Pre-scan: identify operand chains (tile.load -> tile.transpose_view/reshape)
  // consumed EXCLUSIVELY by tile.batch_matmul. Used for two things: (a) the
  // tile.transpose / tile.reshape rewrite-loop skips below (peeled by the lowering),
  // and (b) the !fit capacity drop — when a matmul's whole operands do not fit Mat,
  // ExtractBatchPage loads them per batch and the dead whole chain is dropped.
  //
  // Safety: we count ALL Var references across every statement type (Return, Yield,
  // If conditions, For/While bounds, etc.), not just Call arguments. A Var used
  // anywhere outside a tile.batch_matmul Call prevents it from being treated as a
  // batch_matmul-only chain.
  std::unordered_set<const Var*> batch_matmul_only_vars;
  // Operand-chain vars (whole load + transpose_view) feeding EXCLUSIVELY !fit
  // batch_matmuls — their whole tile would overflow Mat, so ExtractBatchPage loads
  // them per batch and the dead whole chain is dropped during rewriting below.
  std::unordered_set<const Var*> not_fit_drop_vars;
  {
    std::unordered_map<const Var*, int> use_count;
    std::vector<const Var*> batch_matmul_operands;  // ordered to avoid nondeterministic iteration

    // Helper: recursively count all Var references within an expression.
    std::function<void(const ExprPtr&)> CountVarRefs = [&](const ExprPtr& expr) {
      if (!expr) return;
      if (auto v = As<Var>(expr)) {
        use_count[v.get()]++;
        return;
      }
      if (auto tup = As<MakeTuple>(expr)) {
        for (const auto& e : tup->elements_) CountVarRefs(e);
        return;
      }
      if (auto call = As<Call>(expr)) {
        for (const auto& a : call->args_) CountVarRefs(a);
        return;
      }
    };

    for (const auto& s : stmts) {
      // AssignStmt: count call args; mark batch_matmul[_acc] operands separately.
      // For tile.batch_matmul_acc, only lhs (arg 1) and rhs (arg 2) are Mat
      // operand chains — the acc operand (arg 0) is in Acc memory and never goes
      // through tile.load(target_memory=Mat).
      if (auto a = As<AssignStmt>(s)) {
        if (auto c = As<Call>(a->value_)) {
          const std::string& cname = c->op_->name_;
          bool is_batch_mm = (cname == "tile.batch_matmul");
          bool is_batch_mm_acc = (cname == "tile.batch_matmul_acc");
          for (size_t arg_i = 0; arg_i < c->args_.size(); ++arg_i) {
            const auto& arg = c->args_[arg_i];
            if (auto v = As<Var>(arg)) {
              use_count[v.get()]++;
              const bool eligible = is_batch_mm || (is_batch_mm_acc && (arg_i == 1 || arg_i == 2));
              if (eligible) {
                batch_matmul_operands.push_back(v.get());
              }
            }
          }
        } else {
          // Non-call assignment (e.g. plain Var alias): count all Var refs.
          CountVarRefs(a->value_);
        }
        continue;
      }
      // ReturnStmt / YieldStmt: count all returned/yielded Var refs.
      if (auto ret = As<ReturnStmt>(s)) {
        for (const auto& v : ret->value_) CountVarRefs(v);
        continue;
      }
      if (auto yield = As<YieldStmt>(s)) {
        for (const auto& v : yield->value_) CountVarRefs(v);
        continue;
      }
      // EvalStmt: count Var refs in the expression.
      if (auto eval = As<EvalStmt>(s)) {
        CountVarRefs(eval->expr_);
        continue;
      }
      // IfStmt: count condition Var refs.
      if (auto if_stmt = As<IfStmt>(s)) {
        CountVarRefs(if_stmt->condition_);
        continue;
      }
      // ForStmt: count start/stop/step and iter_arg init Var refs.
      if (auto for_stmt = As<ForStmt>(s)) {
        CountVarRefs(for_stmt->start_);
        CountVarRefs(for_stmt->stop_);
        CountVarRefs(for_stmt->step_);
        for (const auto& ia : for_stmt->iter_args_) CountVarRefs(ia->initValue_);
        continue;
      }
      // WhileStmt: count condition and iter_arg init Var refs.
      if (auto while_stmt = As<WhileStmt>(s)) {
        CountVarRefs(while_stmt->condition_);
        for (const auto& ia : while_stmt->iter_args_) CountVarRefs(ia->initValue_);
        continue;
      }
    }
    // The per-statement scan above counts only TOP-LEVEL uses (for an
    // IfStmt/ForStmt/WhileStmt it counts the condition / loop bounds / iter_arg
    // inits, but NOT uses inside the nested body). Separately count EVERY use of
    // each Var, recursing into nested IfStmt/ForStmt/WhileStmt/ScopeStmt bodies.
    // A batch_matmul operand load that is also consumed inside a nested block
    // must NOT be skipped: dropping its definition would leave the nested use
    // referencing an undefined Var after the nested block is rewritten.
    std::unordered_map<const Var*, int> total_counts;
    std::function<void(const ExprPtr&)> CountTotalExprRefs = [&](const ExprPtr& expr) {
      if (!expr) return;
      if (auto v = As<Var>(expr)) {
        total_counts[v.get()]++;
        return;
      }
      if (auto tup = As<MakeTuple>(expr)) {
        for (const auto& e : tup->elements_) CountTotalExprRefs(e);
        return;
      }
      if (auto call = As<Call>(expr)) {
        for (const auto& a : call->args_) CountTotalExprRefs(a);
        return;
      }
    };
    std::function<void(const StmtPtr&)> CountTotalStmtRefs = [&](const StmtPtr& s) {
      if (!s) return;
      if (auto seq = As<SeqStmts>(s)) {
        for (const auto& inner : seq->stmts_) CountTotalStmtRefs(inner);
      } else if (auto scope = As<ScopeStmt>(s)) {
        CountTotalStmtRefs(scope->body_);
      } else if (auto if_stmt = As<IfStmt>(s)) {
        CountTotalExprRefs(if_stmt->condition_);
        CountTotalStmtRefs(if_stmt->then_body_);
        if (if_stmt->else_body_.has_value()) CountTotalStmtRefs(*if_stmt->else_body_);
      } else if (auto for_stmt = As<ForStmt>(s)) {
        CountTotalExprRefs(for_stmt->start_);
        CountTotalExprRefs(for_stmt->stop_);
        CountTotalExprRefs(for_stmt->step_);
        for (const auto& ia : for_stmt->iter_args_) CountTotalExprRefs(ia->initValue_);
        CountTotalStmtRefs(for_stmt->body_);
      } else if (auto while_stmt = As<WhileStmt>(s)) {
        CountTotalExprRefs(while_stmt->condition_);
        for (const auto& ia : while_stmt->iter_args_) CountTotalExprRefs(ia->initValue_);
        CountTotalStmtRefs(while_stmt->body_);
      } else if (auto a = As<AssignStmt>(s)) {
        CountTotalExprRefs(a->value_);
      } else if (auto ret = As<ReturnStmt>(s)) {
        for (const auto& v : ret->value_) CountTotalExprRefs(v);
      } else if (auto yield = As<YieldStmt>(s)) {
        for (const auto& v : yield->value_) CountTotalExprRefs(v);
      } else if (auto eval = As<EvalStmt>(s)) {
        CountTotalExprRefs(eval->expr_);
      }
    };
    for (const auto& s : stmts) CountTotalStmtRefs(s);

    // Collect operands whose EVERY use is a (top-level) batch_matmul operand.
    // Two conditions: all top-level uses are batch_matmul operands
    // (batch_matmul_operand_uses == use_count) AND the Var has no nested uses
    // (use_count == total_counts). Comparing against the total use count —
    // rather than requiring use_count == 1 — covers the shared-operand case:
    // e.g. a SwiGLU FFN where the activation X is the common LHS of both the
    // gate (X@W1) and up (X@W3) matmuls (use_count > 1). Treating a shared
    // operand as batch_matmul-only is safe: in the fit path it is sliced (not
    // dropped); in the !fit path it is dropped only when EVERY consuming matmul
    // is !fit (see the capacity gate below).
    std::unordered_map<const Var*, int> batch_matmul_operand_uses;
    for (const auto* v : batch_matmul_operands) batch_matmul_operand_uses[v]++;
    std::unordered_set<const Var*> seen;
    for (const auto* v : batch_matmul_operands) {
      if (seen.insert(v).second && batch_matmul_operand_uses[v] == use_count[v] &&
          use_count[v] == total_counts[v]) {
        batch_matmul_only_vars.insert(v);
      }
    }

    // Walk back through safe peelable `tile.reshape` chains so the upstream
    // dead `tile.reshape` (and any further-upstream `tile.load`) are also
    // skipped during rewriting. Without this, the orphan reshape would emit a
    // rank>2 tile that violates the post-pass `TileOps2D` property. Require the
    // input to have exactly one use across the WHOLE block (total_counts) so a
    // load also consumed inside a nested body is never peeled away.
    // Also walk back through a single-use tile.transpose_view so its underlying
    // whole tile.load joins the chain (needed so the !fit path can drop the dead
    // whole load, not just the view).
    auto stmt_def_map = BuildAssignDefMap(stmts);
    std::vector<const Var*> reshape_worklist(batch_matmul_only_vars.begin(), batch_matmul_only_vars.end());
    while (!reshape_worklist.empty()) {
      const Var* current = reshape_worklist.back();
      reshape_worklist.pop_back();
      auto def_it = stmt_def_map.find(current);
      if (def_it == stmt_def_map.end()) continue;
      auto chain_call = As<Call>(def_it->second->value_);
      const bool is_view = IsOp(chain_call, "tile.transpose_view");
      if (!is_view && !IsSafePeelableBatchMatmulReshape(chain_call)) continue;
      auto input_var = As<Var>(chain_call->args_[0]);
      if (!input_var) continue;
      if (total_counts[input_var.get()] != 1) continue;
      if (batch_matmul_only_vars.insert(input_var.get()).second) {
        reshape_worklist.push_back(input_var.get());
      }
    }

    // Per-batch_matmul capacity gate: when the operands' whole tiles do not fit
    // Mat together, their batch_matmul_only operand chains are loaded per batch
    // (ExtractBatchPage !fit path) — drop the dead whole chain here. A chain
    // shared with any FIT matmul stays whole (drop only chains feeding
    // exclusively !fit matmuls).
    std::unordered_set<const Var*> any_fit, any_notfit;
    std::function<void(const Var*, bool)> MarkChain = [&](const Var* v, bool fits) {
      if (!v) return;
      (fits ? any_fit : any_notfit).insert(v);
      auto it = stmt_def_map.find(v);
      if (it == stmt_def_map.end()) return;
      auto c = As<Call>(it->second->value_);
      if (!c || !c->op_ || c->args_.empty()) return;
      const std::string& n = c->op_->name_;
      if (n == "tile.transpose_view" || n == "tile.reshape") {
        if (auto in = As<Var>(c->args_[0])) MarkChain(in.get(), fits);
      }
    };
    for (const auto& s : stmts) {
      auto a = As<AssignStmt>(s);
      auto c = a ? As<Call>(a->value_) : nullptr;
      if (!c || !c->op_) continue;
      const std::string& n = c->op_->name_;
      const bool is_bmm = (n == "tile.batch_matmul");
      const bool is_bmm_acc = (n == "tile.batch_matmul_acc");
      if (!is_bmm && !is_bmm_acc) continue;
      const size_t lhs_i = is_bmm ? 0 : 1;
      const size_t rhs_i = is_bmm ? 1 : 2;
      if (rhs_i >= c->args_.size()) continue;
      // Mirror LowerBatchMatmul's per-operand routing so a non-contiguous whole
      // load (which goes per-batch) is also recognized as !fit here and its dead
      // whole chain is dropped (otherwise it would survive to codegen and trip the
      // ND2NZ contiguity guard).
      const bool capacity_fits = BatchOperandsWholeFit(As<TileType>(c->args_[lhs_i]->GetType()),
                                                       As<TileType>(c->args_[rhs_i]->GetType()));
      const bool lhs_fits =
          KeepOperandWhole(capacity_fits, TraceOperandBaseLoad(c->args_[lhs_i], stmt_def_map));
      const bool rhs_fits =
          KeepOperandWhole(capacity_fits, TraceOperandBaseLoad(c->args_[rhs_i], stmt_def_map));
      if (auto lv = As<Var>(c->args_[lhs_i])) MarkChain(lv.get(), lhs_fits);
      if (auto rv = As<Var>(c->args_[rhs_i])) MarkChain(rv.get(), rhs_fits);
    }
    // Visit order does not escape: the loop only tests membership and inserts
    // into a set that is itself consumed by lookup, so the result is identical
    // for any traversal order.
    // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
    for (const auto* v : batch_matmul_only_vars) {
      if (any_notfit.count(v) != 0 && any_fit.count(v) == 0) not_fit_drop_vars.insert(v);
    }
  }

  for (size_t stmt_index = 0; stmt_index < stmts.size(); ++stmt_index) {
    const auto& stmt = stmts[stmt_index];
    // ReturnStmt: substitute return values
    if (auto ret = As<ReturnStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      new_values.reserve(ret->value_.size());
      for (const auto& v : ret->value_) {
        new_values.push_back(Substitute(v, ctx.var_map));
      }
      result.push_back(std::make_shared<ReturnStmt>(new_values, ret->span_));
      continue;
    }

    // YieldStmt: substitute variables
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<ExprPtr> new_values;
      new_values.reserve(yield->value_.size());
      for (const auto& v : yield->value_) {
        new_values.push_back(Substitute(v, ctx.var_map));
      }
      result.push_back(std::make_shared<YieldStmt>(new_values, yield->span_));
      continue;
    }

    // SeqStmts: recurse
    if (auto seq = As<SeqStmts>(stmt)) {
      auto inner = TransformBody(seq->stmts_, ctx, op_registry, span);
      result.insert(result.end(), inner.begin(), inner.end());
      continue;
    }

    // ScopeStmt: recurse into body — dispatch on the concrete derived class
    // since ScopeStmt is abstract and MutableCopy needs a concrete type.
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto inner = TransformBody(body_stmts, ctx, op_registry, span);
      auto new_body = SeqStmts::Flatten(std::move(inner), scope->body_->span_);
      auto rewrite = [&](auto&& concrete) -> StmtPtr {
        auto new_scope = MutableCopy(concrete);
        new_scope->body_ = new_body;
        return new_scope;
      };
      if (auto in_core = As<InCoreScopeStmt>(stmt)) {
        result.push_back(rewrite(in_core));
      } else if (auto cluster = As<ClusterScopeStmt>(stmt)) {
        result.push_back(rewrite(cluster));
      } else if (auto hier = As<HierarchyScopeStmt>(stmt)) {
        result.push_back(rewrite(hier));
      } else if (auto spmd = As<SpmdScopeStmt>(stmt)) {
        result.push_back(rewrite(spmd));
      } else if (auto split_aiv = As<SplitAivScopeStmt>(stmt)) {
        result.push_back(rewrite(split_aiv));
      } else if (auto runtime_scope = As<RuntimeScopeStmt>(stmt)) {
        result.push_back(rewrite(runtime_scope));
      } else {
        INTERNAL_UNREACHABLE_SPAN(scope->span_) << "Unknown ScopeStmt subclass: " << scope->TypeName();
      }
      continue;
    }

    // IfStmt: recurse into branches, substitute return_vars
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto new_cond = Substitute(if_stmt->condition_, ctx.var_map);

      auto then_ctx = ctx;
      auto then_stmts = FlattenToStmts(if_stmt->then_body_);
      auto new_then = TransformBody(then_stmts, then_ctx, op_registry, span);
      // Extract yield types before moving the vector
      auto yield_types = FindYieldTypes(new_then);
      auto new_then_body = SeqStmts::Flatten(std::move(new_then), if_stmt->then_body_->span_);

      FlattenContext else_ctx = ctx;
      std::optional<StmtPtr> new_else_body;
      if (if_stmt->else_body_.has_value()) {
        auto else_stmts = FlattenToStmts(*if_stmt->else_body_);
        auto new_else = TransformBody(else_stmts, else_ctx, op_registry, span);
        new_else_body = SeqStmts::Flatten(std::move(new_else), (*if_stmt->else_body_)->span_);
      }

      // Update return_vars types based on yield types (positional matching)
      if (yield_types.empty() && new_else_body.has_value()) {
        yield_types = FindYieldTypes(FlattenToStmts(*new_else_body));
      }
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(if_stmt->return_vars_.size());
      for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
        const auto& rv = if_stmt->return_vars_[i];
        if (i < yield_types.size() && yield_types[i] != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_types[i], rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      auto new_if = MutableCopy(if_stmt);
      new_if->condition_ = new_cond;
      new_if->then_body_ = new_then_body;
      new_if->else_body_ = new_else_body;
      new_if->return_vars_ = new_return_vars;
      result.push_back(new_if);
      continue;
    }

    // ForStmt: recurse into body, substitute return_vars
    if (auto for_stmt = As<ForStmt>(stmt)) {
      auto new_start = Substitute(for_stmt->start_, ctx.var_map);
      auto new_stop = Substitute(for_stmt->stop_, ctx.var_map);
      auto new_step = Substitute(for_stmt->step_, ctx.var_map);

      auto body_ctx = ctx;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(for_stmt->iter_args_.size());
      for (const auto& ia : for_stmt->iter_args_) {
        auto new_init = Substitute(ia->initValue_, ctx.var_map);
        auto new_ia = ia;
        if (new_init != ia->initValue_) {
          new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
          body_ctx.Insert(ia, new_ia);
        } else {
          body_ctx.Erase(ia);
        }
        new_iter_args.push_back(new_ia);
      }

      auto body_stmts = FlattenToStmts(for_stmt->body_);
      auto new_body_stmts = TransformBody(body_stmts, body_ctx, op_registry, span);
      auto new_body = SeqStmts::Flatten(std::move(new_body_stmts), for_stmt->body_->span_);

      // Update return_vars types to match iter_arg types (positional matching)
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(for_stmt->return_vars_.size());
      for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
        const auto& rv = for_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      auto new_for = MutableCopy(for_stmt);
      new_for->start_ = new_start;
      new_for->stop_ = new_stop;
      new_for->step_ = new_step;
      new_for->iter_args_ = new_iter_args;
      new_for->body_ = new_body;
      new_for->return_vars_ = new_return_vars;
      result.push_back(new_for);
      continue;
    }

    // WhileStmt: recurse into body, substitute return_vars
    if (auto while_stmt = As<WhileStmt>(stmt)) {
      auto body_ctx = ctx;
      std::vector<IterArgPtr> new_iter_args;
      new_iter_args.reserve(while_stmt->iter_args_.size());
      for (const auto& ia : while_stmt->iter_args_) {
        auto new_init = Substitute(ia->initValue_, ctx.var_map);
        auto new_ia = ia;
        if (new_init != ia->initValue_) {
          new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
          body_ctx.Insert(ia, new_ia);
        } else {
          body_ctx.Erase(ia);
        }
        new_iter_args.push_back(new_ia);
      }

      auto new_cond = Substitute(while_stmt->condition_, body_ctx.var_map);
      auto body_stmts = FlattenToStmts(while_stmt->body_);
      auto new_body_stmts = TransformBody(body_stmts, body_ctx, op_registry, span);
      auto new_body = SeqStmts::Flatten(std::move(new_body_stmts), while_stmt->body_->span_);

      // Update return_vars types to match iter_arg types (positional matching)
      std::vector<VarPtr> new_return_vars;
      new_return_vars.reserve(while_stmt->return_vars_.size());
      for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
        const auto& rv = while_stmt->return_vars_[i];
        if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
          auto new_rv = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
          new_return_vars.push_back(new_rv);
          ctx.Insert(rv, new_rv);
        } else {
          new_return_vars.push_back(rv);
        }
      }

      auto new_while = MutableCopy(while_stmt);
      new_while->condition_ = new_cond;
      new_while->iter_args_ = new_iter_args;
      new_while->body_ = new_body;
      new_while->return_vars_ = new_return_vars;
      result.push_back(new_while);
      continue;
    }

    // EvalStmt: substitute variables in the expression
    if (auto eval = As<EvalStmt>(stmt)) {
      auto new_expr = Substitute(eval->expr_, ctx.var_map);
      if (new_expr != eval->expr_) {
        // Re-create tile ops via OpRegistry for proper type deduction
        if (auto call = As<Call>(new_expr)) {
          if (call->op_ && call->op_->name_.substr(0, 5) == "tile.") {
            auto new_call = op_registry.Create(call->op_->name_, call->args_, call->kwargs_, span);
            result.push_back(std::make_shared<EvalStmt>(new_call, eval->span_));
            continue;
          }
        }
        result.push_back(std::make_shared<EvalStmt>(new_expr, eval->span_));
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    // AssignStmt: the main transformation logic
    auto assign = As<AssignStmt>(stmt);
    if (!assign) {
      result.push_back(stmt);
      continue;
    }

    // Drop the dead whole load/transpose_view chain of a !fit batch_matmul: it is
    // loaded per batch at the matmul (ExtractBatchPage !fit path), so the whole
    // tile is never referenced and must not occupy L1.
    if (not_fit_drop_vars.count(assign->var_.get()) != 0) {
      ctx.Insert(assign->var_, assign->var_);  // identity mapping so any lookup resolves
      continue;
    }

    auto call = As<Call>(assign->value_);
    auto global_var = call ? As<GlobalVar>(call->op_) : nullptr;

    // Non-call assignment or function call (GlobalVar): substitute and pass through
    if (!call || global_var) {
      auto new_value = Substitute(assign->value_, ctx.var_map);
      if (new_value != assign->value_) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
        ctx.Insert(assign->var_, new_var);
      } else {
        result.push_back(stmt);
      }
      continue;
    }

    const auto& op_name = call->op_->name_;

    // ---- tile.load on >2D tile: flatten the result tile to 2D (hardware tiles
    //      are always 2D), keeping the tensor-rank source window for codegen —
    //      except a natural Mat load with a real batch (>1) collapses its window
    //      to 2D as well (the ND2NZ path rejects rank>2 GlobalTensors). ----
    if (IsOp(call, "tile.load")) {
      // A batch_matmul operand load is KEPT here and sliced per batch by
      // ExtractBatchPage (the fit path) — both operands, transposed or not, are
      // handled identically (whole tile in L1 + per-batch slice). Only a !fit
      // operand load is dropped (above), to be re-emitted per batch.

      // Substitute args via ctx.var_map so all operand Vars reference the latest SSA values.
      std::vector<ExprPtr> sub_args;
      sub_args.reserve(call->args_.size());
      for (const auto& arg : call->args_) {
        sub_args.push_back(Substitute(arg, ctx.var_map));
      }

      auto result_tile = As<TileType>(call->GetType());
      if (result_tile && result_tile->shape_.size() > 2) {
        // Rank>2 tile.load: keep the original tensor-rank offsets/shapes, but
        // construct a 2D TileType for the result. DeduceTileLoadType produces a
        // rank>2 TileType from those shapes, but hardware tiles are always 2D.
        // The pass manually overrides the result type to 2D.
        auto [merged, last] = ComputeMergedShape(result_tile->shape_, "tile.load result");

        auto flat_shape_exprs = Make2DShapeExprs(merged, last, span);
        // Preserve any TileView (blayout/slayout/fractal/pad) the source tile
        // already carried — e.g. LowerCompositeOps tags a transposed-load Mat
        // rhs with TileView(blayout=row_major, slayout=col_major) so the
        // downstream TLOAD matches the DN2ZN pattern. The implicit Mat default
        // (col/row = ND) would otherwise emit an unsupported DN2ND TLOAD (#1540).
        std::optional<TileView> flat_tile_view;
        if (result_tile->tile_view_.has_value()) {
          const auto& orig_tv = *result_tile->tile_view_;
          // Carry the original valid_shape through the flatten. When it is a
          // proper per-dim valid_shape (e.g. a dynamic min(CHUNK, D-c) tail from
          // the dynamic-tile strip-mine), merge it the same way as the physical
          // shape so the runtime tail extent survives; otherwise the flattened
          // tile is fully valid (valid_shape == physical 2D shape).
          std::vector<ExprPtr> flat_valid = flat_shape_exprs;
          if (orig_tv.valid_shape.size() == result_tile->shape_.size()) {
            flat_valid = ComputeMergedValidShape(orig_tv.valid_shape, span);
          }
          flat_tile_view = TileView(flat_valid, /*stride=*/{}, /*start_offset=*/nullptr, orig_tv.blayout,
                                    orig_tv.slayout, orig_tv.fractal, orig_tv.pad);
        } else {
          flat_tile_view =
              tile_view_semantics::GetImplicitTileView(flat_shape_exprs, result_tile->memory_space_);
        }
        auto flat_tile_type = std::make_shared<TileType>(flat_shape_exprs, result_tile->dtype_, std::nullopt,
                                                         flat_tile_view, result_tile->memory_space_);

        // The rank>2 source window is preserved as-is for codegen. A natural Mat
        // load lowers to ND2NZ, which requires a 2-dim GlobalTensor — that
        // collapse is owned by the tile.load codegen (it triggers on the NZ result
        // tile), so flatten only needs to flatten the RESULT tile to 2D here.
        auto flat_call =
            std::make_shared<Call>(call->op_, sub_args, call->kwargs_, flat_tile_type, call->span_);
        auto flat_var = std::make_shared<Var>(assign->var_->name_hint_, flat_tile_type, assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, flat_call, assign->span_));
        ctx.Insert(assign->var_, flat_var);
        continue;
      }
      // ≤2D tile.load: honor any pending var_map substitutions
      auto new_call = op_registry.Create(op_name, sub_args, call->kwargs_, span);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- tile.store: inject original tensor-rank partition shape for rank>2 tensors ----
    // tile.store semantics: (2D) tile -> rank>2 tensor. Original tensor-rank
    // offsets are preserved; codegen uses the tensor view plus a partition_view
    // over the original tensor-rank window to produce the 2D result.
    // Signature: (tile, offsets, output_tensor[, shapes])
    if (IsOp(call, "tile.store")) {
      auto orig_tile_type = As<TileType>(call->args_[0]->GetType());

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size() + 1);
      // Push all original args (tile, offsets, output_tensor) with substitution
      for (const auto& arg : call->args_) {
        new_args.push_back(Substitute(arg, ctx.var_map));
      }

      // If the (substituted) tile operand is still >2D — e.g. a user-written
      // ``pl.reshape(tile_2d, [B, 1, D])`` to feed ``pl.assemble`` into a
      // rank>2 tensor view — insert a ``tile.reshape`` to flatten it to 2D.
      // Codegen for ``tile.store`` requires a 2D tile; the original N-rank
      // shape still flows through as the ``shapes`` partition operand built
      // below from ``orig_tile_type``.
      auto tile_arg_type = As<TileType>(new_args[0]->GetType());
      if (tile_arg_type && tile_arg_type->shape_.size() > 2) {
        auto [merged, last] = ComputeMergedShape(tile_arg_type->shape_, "tile.store tile operand");
        auto reshape_shape = MakeShapeTupleFromInts({merged, last}, span);
        auto reshape_call = op_registry.Create("tile.reshape", {new_args[0], reshape_shape}, span);
        auto flat_var = std::make_shared<Var>("flat_tile", reshape_call->GetType(), span);
        result.push_back(std::make_shared<AssignStmt>(flat_var, reshape_call, span));
        new_args[0] = flat_var;
      }

      auto out_tensor_type = As<TensorType>(new_args[2]->GetType());
      if (orig_tile_type && out_tensor_type && out_tensor_type->shape_.size() > 2) {
        // Inject the original tensor-rank partition shape tuple as the 4th argument.
        // The partition shape has the same rank as the tensor, with 1s for
        // batch dims that are not covered by the tile, followed by the tile dims.
        const size_t tensor_rank = out_tensor_type->shape_.size();
        const size_t tile_rank = orig_tile_type->shape_.size();
        std::vector<ExprPtr> partition_shape;
        partition_shape.reserve(tensor_rank);
        for (size_t i = tile_rank; i < tensor_rank; ++i) {
          partition_shape.push_back(std::make_shared<ConstInt>(1, DataType::INDEX, span));
        }
        for (const auto& dim : orig_tile_type->shape_) {
          partition_shape.push_back(dim);
        }
        new_args.push_back(std::make_shared<MakeTuple>(partition_shape, span));
      }

      // Construct call directly: store result type = output tensor type (args[2])
      auto out_type = new_args[2]->GetType();
      auto new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, out_type, call->span_);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
      ctx.Insert(assign->var_, new_var);
      continue;
    }

    // ---- tile.create / tile.full with >2D shape: flatten shape directly ----
    if (IsOp(call, "tile.create") || IsOp(call, "tile.full")) {
      auto result_tile = As<TileType>(call->GetType());
      if (result_tile && result_tile->shape_.size() > 2) {
        auto [merged, last] = ComputeMergedShape(result_tile->shape_, op_name);

        // Rebuild the call with 2D shape
        auto new_shape_tuple = MakeShapeTupleFromInts({merged, last}, span);
        std::vector<ExprPtr> new_args;
        // First arg is the shape tuple
        new_args.push_back(new_shape_tuple);
        // Remaining args (e.g., fill value for tile.full)
        for (size_t i = 1; i < call->args_.size(); ++i) {
          new_args.push_back(Substitute(call->args_[i], ctx.var_map));
        }

        auto new_call = op_registry.Create(op_name, new_args, call->kwargs_, span);
        auto flat_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(flat_var, new_call, assign->span_));
        ctx.Insert(assign->var_, flat_var);
        continue;
      }
      // ≤2D: pass through
      result.push_back(stmt);
      continue;
    }

    // ---- tile.sum/tile.max/tile.min: remap axis to 1 (last axis of 2D) ----
    if (IsOp(call, "tile.sum") || IsOp(call, "tile.max") || IsOp(call, "tile.min")) {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        if (IsNdTile(input_tile)) {
          // Substitute args
          std::vector<ExprPtr> new_args;
          new_args.reserve(call->args_.size());
          for (const auto& arg : call->args_) {
            new_args.push_back(Substitute(arg, ctx.var_map));
          }

          // Update axis kwarg to 1 (last axis of 2D tile)
          std::vector<std::pair<std::string, std::any>> new_kwargs;
          for (const auto& [key, val] : call->kwargs_) {
            if (key == "axis") {
              new_kwargs.emplace_back("axis", 1);
            } else {
              new_kwargs.emplace_back(key, val);
            }
          }

          auto new_call = op_registry.Create(op_name, new_args, new_kwargs, span);
          auto new_var =
              std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
          result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
          ctx.Insert(assign->var_, new_var);
          continue;
        }
      }
    }

    // ---- tile.batch_matmul: delegate to LowerBatchMatmul ----
    if (IsOp(call, "tile.batch_matmul")) {
      auto lowering = LowerBatchMatmul(assign, call, stmts, stmt_index, ctx, op_registry, span);
      result.insert(result.end(), lowering.stmts.begin(), lowering.stmts.end());
      if (lowering.fused_store) {
        ctx.Insert(lowering.store_orig_var, lowering.store_result_var);
        ++stmt_index;  // Skip the next tile.store; it has been fused above.
      } else {
        ctx.Insert(assign->var_, lowering.output_var);
      }
      continue;
    }

    // ---- tile.batch_matmul_acc: delegate to LowerBatchMatmulAcc ----
    if (IsOp(call, "tile.batch_matmul_acc")) {
      auto lowering = LowerBatchMatmulAcc(assign, call, stmts, ctx, op_registry, span);
      result.insert(result.end(), lowering.stmts.begin(), lowering.stmts.end());
      ctx.Insert(assign->var_, lowering.output_var);
      continue;
    }

    // ---- tile.transpose feeding only tile.batch_matmul[_acc]: skip and let lowering peel it ----
    if (IsOp(call, "tile.transpose") && batch_matmul_only_vars.count(assign->var_.get()) != 0) {
      ctx.Insert(assign->var_, assign->var_);  // identity mapping for safety
      continue;
    }

    // ---- standalone tile.transpose: this pass solely owns scratch materialization ----
    // High-level transposes arrive in the 3-arg form (input, axis1, axis2); the
    // pto.ttrans scratch is emitted here as the codegen-ready 4-arg form.
    //   >2D  → LowerNdTranspose: per-page 2D transposes, each with sliced scratch.
    //   2D   → emit one scratch tile.create + a 4-arg tile.transpose.
    // An already-4-arg 2D transpose (e.g. hand-built IR) falls through to the generic
    // re-create path unchanged.
    if (IsOp(call, "tile.transpose") && batch_matmul_only_vars.count(assign->var_.get()) == 0) {
      if (IsNdTile(As<TileType>(call->args_[0]->GetType()))) {
        auto lowering = LowerNdTranspose(assign, call, ctx, op_registry, span);
        result.insert(result.end(), lowering.stmts.begin(), lowering.stmts.end());
        ctx.Insert(assign->var_, lowering.output_var);
        continue;
      }
      if (call->args_.size() == 3) {
        auto in = Substitute(call->args_[0], ctx.var_map);
        auto in_type = As<TileType>(in->GetType());
        INTERNAL_CHECK_SPAN(in_type, span)
            << "Internal error: tile.transpose input must be TileType in FlattenTileNdTo2D";
        // pto.ttrans reuses the SOURCE type for both ins operands, so scratch shape ==
        // input shape (NOT the transposed output shape), in the input's memory space.
        MemorySpace scratch_mem =
            in_type->memory_space_.has_value() ? *in_type->memory_space_ : MemorySpace::Vec;
        auto scratch_shape = std::make_shared<MakeTuple>(in_type->shape_, span);
        std::vector<std::pair<std::string, std::any>> scratch_kw = {
            {"dtype", in_type->dtype_},
            {"target_memory", scratch_mem},
        };
        auto scratch_create = op_registry.Create("tile.create", {scratch_shape}, scratch_kw, span);
        auto scratch_var = std::make_shared<Var>("transpose_tmp", scratch_create->GetType(), span);
        result.push_back(std::make_shared<AssignStmt>(scratch_var, scratch_create, span));

        auto t_call =
            op_registry.Create("tile.transpose", {in, call->args_[1], call->args_[2], scratch_var}, span);
        auto t_var = std::make_shared<Var>(assign->var_->name_hint_, t_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(t_var, t_call, assign->span_));
        ctx.Insert(assign->var_, t_var);
        continue;
      }
      // 4-arg 2D transpose: fall through to the generic re-create path below.
    }

    // ---- tile.reshape feeding only tile.batch_matmul: skip (identity) when it is
    //      a safe batch-only reshape that `NormalizeBatchMatmulOperand` peels, so
    //      no orphan rank>2 reshape survives. The underlying tile.load is reused by
    //      the lowering (fit path) or re-emitted per batch (!fit path). ----
    if (IsOp(call, "tile.reshape") && batch_matmul_only_vars.count(assign->var_.get()) != 0 &&
        IsSafePeelableBatchMatmulReshape(call)) {
      ctx.Insert(assign->var_, assign->var_);  // identity mapping for safety
      continue;
    }

    // ---- All other tile ops (including tile.reshape) and non-tile ops: substitute args ----
    {
      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size());
      bool changed = false;
      for (const auto& arg : call->args_) {
        auto new_arg = Substitute(arg, ctx.var_map);
        new_args.push_back(new_arg);
        if (new_arg != arg) changed = true;
      }

      if (!changed) {
        result.push_back(stmt);
      } else {
        // Re-create tile ops via OpRegistry for proper type deduction with 2D args;
        // non-tile ops keep the original type.
        auto new_call =
            (op_name.substr(0, 5) == "tile.")
                ? op_registry.Create(op_name, new_args, call->kwargs_, span)
                : std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->GetType(), call->span_);

        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
        result.push_back(std::make_shared<AssignStmt>(new_var, new_call, assign->span_));
        ctx.Insert(assign->var_, new_var);
      }
    }
  }

  return result;
}

}  // namespace rewrite_internal

FunctionPtr Rewrite(const FunctionPtr& func) {
  const auto& span = func->span_;
  auto& op_registry = OpRegistry::GetInstance();

  rewrite_internal::FlattenContext ctx;

  auto body_stmts = FlattenToStmts(func->body_);
  auto new_stmts = rewrite_internal::TransformBody(body_stmts, ctx, op_registry, span);
  auto new_body = SeqStmts::Flatten(std::move(new_stmts), span);

  // return_types_ are unchanged: InCore functions return tensors (not tiles),
  // and this pass only flattens tile ops. Tensor types are never modified.
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto
