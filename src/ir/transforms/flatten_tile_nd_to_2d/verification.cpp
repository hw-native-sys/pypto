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
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

class TileOps2DVerifier : public IRVisitor {
 public:
  explicit TileOps2DVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckCall(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckCall(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckCall(const CallPtr& call, const Span& stmt_span) {
    if (!call || !call->op_ || As<GlobalVar>(call->op_)) return;

    const auto& name = call->op_->name_;
    if (name.substr(0, 5) != "tile.") return;

    if (IsOp(call, "tile.load") || IsOp(call, "tile.store") || IsOp(call, "tile.reshape") ||
        IsOp(call, "tile.reinterpret_view")) {
      return;
    }

    auto result_tile = As<TileType>(call->GetType());
    if (result_tile && result_tile->shape_.size() > 2) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                "Tile op '" + name + "' in InCore function '" + func_name_ +
                                    "' produces >2D tile (should have been flattened to 2D)",
                                stmt_span);
    }

    if (IsOp(call, "tile.transpose") && call->args_.size() != 4) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                "tile.transpose in InCore function '" + func_name_ + "' has " +
                                    std::to_string(call->args_.size()) +
                                    " arguments (expected 4: input, axis1, axis2, scratch after "
                                    "FlattenTileNdTo2D)",
                                stmt_span);
    }

    for (const auto& arg : call->args_) {
      auto arg_tile = As<TileType>(arg->GetType());
      if (arg_tile && arg_tile->shape_.size() > 2) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "TileOps2D", 0,
                                  "Tile op '" + name + "' in InCore function '" + func_name_ +
                                      "' has >2D tile argument (should have been flattened to 2D)",
                                  stmt_span);
        break;
      }
    }
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

class TileOps2DPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TileOps2D"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_ || !IsInCoreType(func->func_type_)) continue;
      TileOps2DVerifier verifier(diagnostics, func->name_);
      verifier.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateTileOps2DPropertyVerifier() {
  return std::make_shared<TileOps2DPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
