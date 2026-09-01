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
 * @file verify_acc_store_phase.cpp
 * @brief Verify the final accumulator-producer / final-store unit-flag protocol.
 *
 * On A2/A3, ``acc_phase=pl.AccPhase.Final`` lowers to a producer-side
 * check-and-set. The matching ``st_phase=pl.STPhase.Final`` store performs the
 * consumer-side check-and-clear. Omitting either side is not a benign metadata
 * mismatch: the device can wait forever on an unset flag, or leave a set flag
 * behind and stall a later accumulator producer.
 *
 * A function-level count is insufficient. Two unrelated producers and stores
 * may have equal counts while clearing the wrong value, and a store that exists
 * only on one branch does not post-dominate a producer before that branch. This
 * verifier therefore tracks the exact SSA value created by each final producer.
 * A plain SSA alias may carry the obligation, but a view/copy/call may not: the
 * verifier cannot prove that such an operation preserves the hardware tile and
 * must reject rather than silently accept a possible deadlock.
 *
 * Control-flow policy is deliberately conservative. A final producer and its
 * final store must be in the same straight-line region. A balanced pair inside
 * a branch or loop body is legal; an obligation may not cross an if/loop/scope
 * boundary. This makes zero-iteration loops, one-sided branches, and later
 * outlining safe without a path-exponential analysis.
 *
 * ``AccStorePhaseValid`` runs immediately after InlineFunctions, so a final
 * producer returned by an Inline helper is checked in the same function and
 * region as its consuming store. The visitor handles both the resulting
 * pre-SSA form (including rebindings and an immediately nested
 * ``store(gemv(..., final), ..., final)``) and the normalized three-address form
 * seen by later verification instrumentation.
 * Complexity is O(N log P), where N is the IR size and P the number of live
 * final producers (normally zero or one).
 */

#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/phase.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

using TokenId = size_t;

struct FinalProducer {
  std::string op_name;
  Span span;
};

bool IsPhasedAccumulatorProducer(const CallPtr& call) {
  return IsOp(call, "tile.gemv") || IsOp(call, "tile.gemv_acc") || IsOp(call, "tile.gemv_bias");
}

bool IsFinalAccumulatorProducer(const CallPtr& call) {
  return IsPhasedAccumulatorProducer(call) &&
         call->GetKwarg<int>("acc_phase", static_cast<int>(AccPhase::kUnspecified)) ==
             static_cast<int>(AccPhase::kFinal);
}

/**
 * @brief Straight-line typestate checker for one function.
 *
 * ``pending_`` contains producer obligations that have been set but not yet
 * cleared. ``var_tokens_`` carries an obligation through a bare SSA alias.
 * Consuming a token removes it from ``pending_`` but deliberately leaves the
 * Var mapping in place: a second final store of the same value can then be
 * diagnosed as a double-consume instead of as an unrelated store.
 */
class AccStorePhaseVisitor : public IRVisitor {
 public:
  AccStorePhaseVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void Run(const StmtPtr& body) {
    if (body) VisitStmt(body);
    FlushPending("before leaving function '" + func_name_ + "'");
  }

 protected:
  void VisitExpr_(const CallPtr& op) override { static_cast<void>(ProcessCall(op)); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op || !op->var_) return;
    auto token = ProcessValue(op->value_);
    if (token.has_value()) {
      var_tokens_[op->var_.get()] = *token;
    } else {
      // The pre-SSA form may reuse one Var object for a later assignment. Do
      // not let an old final-producer obligation leak through that rebind.
      var_tokens_.erase(op->var_.get());
    }
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (op && op->expr_) static_cast<void>(ProcessValue(op->expr_));
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    if (!op) return;
    // SeqStmts is a transparent container, so nested sequences do not create a
    // control-flow boundary and an obligation may cross them.
    for (const auto& stmt : op->stmts_) {
      if (stmt) VisitStmt(stmt);
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    ScanExpr(op->condition_);
    BreakRegion("before entering an if statement");
    AnalyzeIsolatedRegion(op->then_body_, "before leaving an if branch");
    if (op->else_body_.has_value()) {
      AnalyzeIsolatedRegion(*op->else_body_, "before leaving an else branch");
    }
    for (const auto& var : op->return_vars_) {
      if (var) var_tokens_.erase(var.get());
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    ScanExpr(op->start_);
    ScanExpr(op->stop_);
    ScanExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) ScanExpr(iter_arg->initValue_);
    }
    BreakRegion("before entering a loop");
    AnalyzeIsolatedRegion(op->body_, "before completing a loop iteration");
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) var_tokens_.erase(iter_arg.get());
    }
    for (const auto& var : op->return_vars_) {
      if (var) var_tokens_.erase(var.get());
    }
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    ScanExpr(op->condition_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) ScanExpr(iter_arg->initValue_);
    }
    BreakRegion("before entering a while loop");
    AnalyzeIsolatedRegion(op->body_, "before completing a while-loop iteration");
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) var_tokens_.erase(iter_arg.get());
    }
    for (const auto& var : op->return_vars_) {
      if (var) var_tokens_.erase(var.get());
    }
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override {
    if (op) ScanExpr(op->core_num_);
    AnalyzeScope(op);
  }
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override { AnalyzeScope(op); }

  void VisitStmt_(const YieldStmtPtr& op) override {
    if (op) {
      for (const auto& value : op->value_) ScanExpr(value);
    }
    FlushPending("before yielding from a control-flow region");
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    if (op) {
      for (const auto& value : op->value_) ScanExpr(value);
    }
    FlushPending("before returning from function '" + func_name_ + "'");
  }

  void VisitStmt_(const BreakStmtPtr& /*op*/) override { FlushPending("before breaking out of a loop"); }

  void VisitStmt_(const ContinueStmtPtr& /*op*/) override { FlushPending("before continuing a loop"); }

 private:
  std::optional<TokenId> ProcessValue(const ExprPtr& expr) {
    if (!expr) return std::nullopt;
    if (auto var = AsVarLike(expr)) {
      auto it = var_tokens_.find(var.get());
      return it == var_tokens_.end() ? std::nullopt : std::make_optional(it->second);
    }
    if (auto call = As<Call>(expr)) return ProcessCall(call);
    // Any final producer buried in another expression is still recorded and
    // will be reported as unmatched. Its containing expression is not a plain
    // alias, so it cannot carry the token to a store.
    VisitExpr(expr);
    return std::nullopt;
  }

  std::optional<TokenId> ProcessCall(const CallPtr& call) {
    if (!call) return std::nullopt;

    if (IsOp(call, "tile.store")) {
      std::optional<TokenId> source_token;
      if (!call->args_.empty()) source_token = ProcessValue(call->args_[0]);
      for (size_t i = 1; i < call->args_.size(); ++i) ScanExpr(call->args_[i]);
      CheckStore(call, source_token);
      return std::nullopt;
    }

    for (const auto& arg : call->args_) ScanExpr(arg);
    if (!IsFinalAccumulatorProducer(call)) return std::nullopt;

    const TokenId token = producers_.size();
    producers_.push_back(FinalProducer{call->op_->name_, call->span_});
    pending_.insert(token);
    return token;
  }

  void ScanExpr(const ExprPtr& expr) {
    if (expr) static_cast<void>(ProcessValue(expr));
  }

  void CheckStore(const CallPtr& store, const std::optional<TokenId>& source_token) {
    const int st_phase = store->GetKwarg<int>("st_phase", static_cast<int>(STPhase::kUnspecified));
    if (!IsValidSTPhase(st_phase)) {
      if (source_token.has_value() && pending_.erase(*source_token) != 0) {
        reported_.insert(*source_token);
      }
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/5,
          "tile.store st_phase in function '" + func_name_ +
              "' must encode pl.STPhase.Unspecified (0) or pl.STPhase.Final (3), but got " +
              std::to_string(st_phase) +
              ". A final accumulator producer must be consumed with pl.STPhase.Final so its unit flag is "
              "cleared.",
          store->span_);
      return;
    }
    if (st_phase == static_cast<int>(STPhase::kFinal)) {
      if (!source_token.has_value()) {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/1,
            "tile.store(..., st_phase=pl.STPhase.Final) in function '" + func_name_ +
                "' must consume the exact SSA value produced by tile.gemv, tile.gemv_acc, or "
                "tile.gemv_bias with acc_phase=pl.AccPhase.Final in the same straight-line region. A final "
                "store "
                "without a matching producer waits on a unit flag that was never set and can stall device "
                "execution.",
            store->span_);
        return;
      }
      if (pending_.erase(*source_token) == 0) {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/2,
            "tile.store(..., st_phase=pl.STPhase.Final) in function '" + func_name_ +
                "' does not have a live matching final accumulator producer in this straight-line region. "
                "The producer may already have been consumed, or the value crossed a control-flow boundary; "
                "either case makes the unit-flag clear ambiguous and can stall device execution.",
            store->span_);
      }
      return;
    }

    if (source_token.has_value() && pending_.erase(*source_token) != 0) {
      const auto& producer = producers_[*source_token];
      reported_.insert(*source_token);
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/3,
          "final accumulator producer '" + producer.op_name + "' in function '" + func_name_ +
              "' is consumed by tile.store with st_phase=pl.STPhase." +
              STPhaseToString(static_cast<STPhase>(st_phase)) +
              ". Use st_phase=pl.STPhase.Final so the store performs check-and-clear; otherwise the "
              "producer's "
              "unit flag remains set and a later accumulator operation can stall.",
          store->span_);
    }
  }

  void AnalyzeScope(const ScopeStmtPtr& scope) {
    if (!scope) return;
    BreakRegion("before entering scope '" + scope->name_hint_ + "'");
    AnalyzeIsolatedRegion(scope->body_, "before leaving scope '" + scope->name_hint_ + "'");
  }

  void AnalyzeIsolatedRegion(const StmtPtr& body, const std::string& exit_description) {
    auto outer_pending = std::move(pending_);
    pending_.clear();
    if (body) VisitStmt(body);
    FlushPending(exit_description);
    pending_ = std::move(outer_pending);
  }

  void BreakRegion(const std::string& description) { FlushPending(description); }

  void FlushPending(const std::string& description) {
    for (TokenId token : pending_) {
      if (!reported_.insert(token).second) continue;
      const auto& producer = producers_[token];
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/4,
          "final accumulator producer '" + producer.op_name + "' in function '" + func_name_ +
              "' must be consumed exactly once by tile.store(..., st_phase=pl.STPhase.Final) using that SSA "
              "value " +
              description +
              ". acc_phase=pl.AccPhase.Final performs check-and-set; leaving it uncleared can stall a later "
              "device "
              "operation.",
          producer.span);
    }
    pending_.clear();
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  std::vector<FinalProducer> producers_;
  std::set<TokenId> pending_;
  std::set<TokenId> reported_;
  std::unordered_map<const Var*, TokenId> var_tokens_;
};

}  // namespace

class AccStorePhaseValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AccStorePhaseValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, func] : program->functions_) {
      static_cast<void>(global_var);
      if (!func || !func->body_) continue;
      AccStorePhaseVisitor visitor(diagnostics, func->name_);
      visitor.Run(func->body_);
    }
  }
};

PropertyVerifierPtr CreateAccStorePhaseValidPropertyVerifier() {
  return std::make_shared<AccStorePhaseValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
