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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/verifier/property_verifier_registry.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir {
namespace {

bool IsDestinationForm(const CallPtr& call) {
  return IsOp(call, "tile.load_into") || IsOp(call, "tile.extract_into") || IsOp(call, "tile.move_into") ||
         IsOp(call, "tile.matmul_into") || IsOp(call, "tile.matmul_acc_into");
}

class TileBufferLifetimeVerifier : public IRVisitor {
 public:
  std::vector<Diagnostic> TakeDiagnostics() { return std::move(diagnostics_); }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (IsOp(call, "tile.release")) {
      HandleRelease(call);
      return;
    }

    CheckUses(op->value_);
    if (IsOp(call, "tile.buffer_slot")) {
      lease_by_var_[op->var_.get()] = next_lease_++;
      return;
    }
    if (IsDestinationForm(call) && !call->args_.empty()) {
      if (auto destination = AsVarLike(call->args_.back())) {
        auto it = lease_by_var_.find(destination.get());
        if (it != lease_by_var_.end()) lease_by_var_[op->var_.get()] = it->second;
      }
      return;
    }
    if (auto alias = AsVarLike(op->value_)) {
      auto it = lease_by_var_.find(alias.get());
      if (it != lease_by_var_.end()) lease_by_var_[op->var_.get()] = it->second;
    }
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_); IsOp(call, "tile.release")) {
      HandleRelease(call);
      return;
    }
    CheckUses(op->expr_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    CheckUses(op->condition_);
    const auto leases_before = lease_by_var_;
    const auto released_before = released_leases_;

    VisitStmt(op->then_body_);
    const auto released_after_then = released_leases_;

    lease_by_var_ = leases_before;
    released_leases_ = released_before;
    if (op->else_body_.has_value()) VisitStmt(*op->else_body_);

    // A lease released on either reachable branch is conservatively dead
    // after the merge. Branch-local selection vars do not escape the branch.
    released_leases_.insert(released_after_then.begin(), released_after_then.end());
    lease_by_var_ = leases_before;
  }

 private:
  std::map<const Var*, int> lease_by_var_;
  std::set<int> released_leases_;
  std::vector<Diagnostic> diagnostics_;
  int next_lease_ = 0;

  void CheckUses(const ExprPtr& expr) {
    var_collectors::VarDefUseCollector collector;
    collector.VisitExpr(expr);
    for (const Var* var : collector.var_uses) {
      auto lease_it = lease_by_var_.find(var);
      if (lease_it == lease_by_var_.end() || released_leases_.count(lease_it->second) == 0) continue;
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "TileBufferLifetime", 0,
          "variable '" + var->name_hint_ + "' uses an explicit tile slot after it was released", var->span_);
    }
  }

  void HandleRelease(const CallPtr& call) {
    if (!call || call->args_.size() != 1) return;
    auto slot = AsVarLike(call->args_[0]);
    auto lease_it = slot ? lease_by_var_.find(slot.get()) : lease_by_var_.end();
    if (!slot || lease_it == lease_by_var_.end()) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileBufferLifetime", 1,
                                "tile.release requires a value derived from a selected slot", call->span_);
      return;
    }
    if (!released_leases_.insert(lease_it->second).second) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileBufferLifetime", 2,
                                "explicit tile slot was released more than once", call->span_);
    }
  }
};

void CollectTileBufferLifetimeDiagnostics(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) {
  if (!program) return;
  for (const auto& [global, function] : program->functions_) {
    (void)global;
    if (!function || !function->body_) continue;
    TileBufferLifetimeVerifier verifier;
    verifier.VisitStmt(function->body_);
    auto function_diagnostics = verifier.TakeDiagnostics();
    for (const auto& diagnostic : function_diagnostics) {
      diagnostics.emplace_back(diagnostic.severity, diagnostic.rule_name, diagnostic.error_code,
                               diagnostic.message, diagnostic.span);
    }
  }
}

}  // namespace

class TileBufferLifetimePropertyVerifier : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TileBufferLifetimeValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    CollectTileBufferLifetimeDiagnostics(program, diagnostics);
  }
};

PropertyVerifierPtr CreateTileBufferLifetimePropertyVerifier() {
  return std::make_shared<TileBufferLifetimePropertyVerifier>();
}

void VerifyTileBufferLifetime(const ProgramPtr& program) {
  std::vector<Diagnostic> diagnostics;
  CollectTileBufferLifetimeDiagnostics(program, diagnostics);
  if (diagnostics.empty()) return;
  throw VerificationError(PropertyVerifierRegistry::GenerateReport(diagnostics), std::move(diagnostics));
}

}  // namespace pypto::ir
