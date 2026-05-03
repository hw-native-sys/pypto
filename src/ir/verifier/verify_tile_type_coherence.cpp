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
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Walks every Var/IterArg/Call/MemRef/Function-param/return type reachable
/// from a program and asserts the TileType canonical-encoding invariant on
/// each TileType (including TileTypes nested in TupleTypes).
class TileTypeCoherenceVisitor : public IRVisitor {
 public:
  TileTypeCoherenceVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void CheckType(const TypePtr& type, const Span& span) {
    if (!type) return;
    if (auto tile_type = As<TileType>(type)) {
      CheckTileType(tile_type, span);
    } else if (auto tuple_type = As<TupleType>(type)) {
      for (const auto& sub : tuple_type->types_) {
        CheckType(sub, span);
      }
    }
  }

  void CheckFunction(const FunctionPtr& func) {
    for (const auto& param : func->params_) {
      if (param) CheckType(param->GetType(), param->span_);
    }
    for (const auto& rt : func->return_types_) {
      CheckType(rt, func->span_);
    }
    if (func->body_) {
      VisitStmt(func->body_);
    }
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) CheckType(op->GetType(), op->span_);
    IRVisitor::VisitVarLike_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    if (op) CheckType(op->GetType(), op->span_);
    IRVisitor::VisitExpr_(op);
  }

 private:
  void CheckTileType(const TileTypePtr& tile_type, const Span& span) {
    if (!tile_type || !tile_type->tile_view_.has_value()) return;
    if (tile_view_semantics::IsImplicitPrintedTileView(*tile_type->tile_view_, tile_type->shape_,
                                                       tile_type->memory_space_)) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "TileTypeCoherence", /*error_code=*/1,
          "TileType has implicit-but-present tile_view (canonical encoding requires nullopt) in "
          "function '" +
              func_name_ + "'",
          span);
    }
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

}  // namespace

class TileTypeCoherencePropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "TileTypeCoherence"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;
      TileTypeCoherenceVisitor visitor(diagnostics, func->name_);
      visitor.CheckFunction(func);
    }
  }
};

PropertyVerifierPtr CreateTileTypeCoherencePropertyVerifier() {
  return std::make_shared<TileTypeCoherencePropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
