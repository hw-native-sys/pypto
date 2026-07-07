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

// AllocTileDominatesUses verifier (issue #1956).
//
// After MaterializeAllocTiles, every PTO tile buffer is declared by exactly one
// explicit `alloc_tile` op placed at a scope that dominates every use of the
// buffer. This verifier enforces that dominance structurally: walking each
// function body it tracks the set of buffer identities (MemRef base + byte_offset
// + size) materialized by an `alloc_tile` in the current scope, and flags any use
// of a TileType-with-MemRef variable whose buffer has not been materialized in a
// dominating scope. Branch/loop-local materializations do not leak to enclosing
// scopes (SSA scoping), so an `alloc_tile` declared inside one branch and read
// from another — the pre-#1956 miscompile under memory_planner=PTOAS — is caught.
//
// Keyed by MemRef identity only (not the PyPTO signature split): it suffices that
// *some* alloc_tile for the byte-slot dominates the use, which holds under both
// memory planners.

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

// Buffer identity (base Ptr + byte_offset + size) comes from the shared
// `MemRefIdentityKey` helper — the same string PTO codegen resolves handles by
// and the MaterializeAllocTiles pass groups buffers by — so this dominance check
// is keyed byte-for-byte against what those two produce.

class AllocTileDominatesChecker : public IRVisitor {
 public:
  AllocTileDominatesChecker(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (!op) return;
    if (auto tile_type = GetTileTypeWithMemRef(op->GetType())) {
      auto memref = GetDefinedMemRef(tile_type);
      if (memref && !materialized_.count(MemRefIdentityKey(memref))) {
        std::ostringstream msg;
        msg << "Tile variable '" << op->name_hint_ << "' in function '" << func_name_
            << "' uses a buffer with no dominating alloc_tile handle. MaterializeAllocTiles must "
               "emit an alloc_tile for every tile buffer at a scope that dominates all its uses "
               "(a handle declared inside a branch/loop does not dominate uses outside it).";
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "AllocTileDominatesUses", 0, msg.str(),
                                  op->span_);
      }
    }
    IRVisitor::VisitVarLike_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    // An `alloc_tile` AssignStmt is the buffer's declaration: materialize its
    // identity for the current scope. It carries no tile-var operand to check
    // (its args are a base Ptr, an offset, and a shape tuple).
    if (auto call = As<Call>(op->value_); call && IsOp(call, "alloc_tile")) {
      if (auto tile_type = GetTileTypeWithMemRef(op->var_->GetType())) {
        if (auto memref = GetDefinedMemRef(tile_type)) materialized_.insert(MemRefIdentityKey(memref));
      }
      return;
    }
    if (op->value_) VisitExpr(op->value_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    auto saved = materialized_;
    if (op->body_) VisitStmt(op->body_);
    materialized_ = std::move(saved);  // body-local handles do not leak past the loop
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    auto saved = materialized_;
    if (op->condition_) VisitExpr(op->condition_);
    if (op->body_) VisitStmt(op->body_);
    materialized_ = std::move(saved);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    if (op->condition_) VisitExpr(op->condition_);
    auto saved = materialized_;
    if (op->then_body_) VisitStmt(op->then_body_);
    materialized_ = saved;
    if (op->else_body_.has_value() && *op->else_body_) VisitStmt(*op->else_body_);
    materialized_ = std::move(saved);  // branch-local handles do not leak past the if
  }

 private:
  std::unordered_set<std::string> materialized_;
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

class AllocTileDominatesPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AllocTileDominatesUses"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, func] : program->functions_) {
      if (!func || !func->body_) continue;
      AllocTileDominatesChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateAllocTileDominatesPropertyVerifier() {
  return std::make_shared<AllocTileDominatesPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
