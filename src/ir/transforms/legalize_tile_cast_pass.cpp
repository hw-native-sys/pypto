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
 * @file legalize_tile_cast_pass.cpp
 * @brief Expand hardware-unsupported tile.cast pairs into native cast chains.
 *
 * Converts (src, dst) pairs that the active pto.tcvt profile cannot emit as a
 * single instruction into a shortest sequence of native casts. Path search is
 * BFS over the native-conversion table the active BackendHandler supplies via
 * GetTcvtAdjacency(), so this pass holds no per-architecture knowledge of its
 * own. Typical outcome for A5 INT32→FP16 is INT32→FP32→FP16 — same byte-width
 * to float, then resize — which adds no precision loss beyond the final narrow.
 */

#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/tcvt_path.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

// Round modes for tile.cast (None=0, RINT=1, ROUND=2, ...).
constexpr int kCastModeRound = 2;

ExprPtr MakeCast(const ExprPtr& x, DataType to, int mode, const Span& span) {
  std::vector<std::pair<std::string, std::any>> kw = {{"target_type", to}, {"mode", mode}};
  return OpRegistry::GetInstance().Create("tile.cast", {x}, kw, span);
}

class LegalizeTileCastMutator : public IRMutator {
 public:
  LegalizeTileCastMutator(const backend::TcvtAdjacency& table, std::string arch_name)
      : table_(table), arch_name_(std::move(arch_name)) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call || !IsOp(call, "tile.cast")) {
      return IRMutator::VisitStmt_(op);
    }
    if (call->args_.empty()) {
      return IRMutator::VisitStmt_(op);
    }

    auto src_tile = As<TileType>(call->args_[0]->GetType());
    INTERNAL_CHECK_SPAN(src_tile, op->span_) << "tile.cast input must be TileType";
    DataType src = src_tile->dtype_;
    DataType dst = call->GetKwarg<DataType>("target_type");
    const int mode = call->GetKwarg<int>("mode", kCastModeRound);

    if (backend::IsNativeTcvt(table_, src, dst)) {
      return IRMutator::VisitStmt_(op);
    }

    std::vector<DataType> chain = backend::FindTcvtPath(table_, src, dst);
    CHECK_SPAN(!chain.empty(), op->span_)
        << "LegalizeTileCast: no native cast path from " << src.ToString() << " to " << dst.ToString()
        << " for arch " << arch_name_ << "; pto.tcvt does not support this conversion";

    // Intermediate hops use the original mode (matches model-side INT32→FP32→FP16
    // chains where the narrow step carries mode="round"). Final hop also keeps it.
    ExprPtr cur = VisitExpr(call->args_[0]);
    std::vector<StmtPtr> stmts;
    stmts.reserve(chain.size());

    for (size_t i = 0; i + 1 < chain.size(); ++i) {
      ExprPtr cast_expr = MakeCast(cur, chain[i], mode, op->span_);
      const std::string name =
          auto_name::BuildName(auto_name::GetBaseName(op->var_->name_hint_), "cast_" + chain[i].ToString(),
                               "tmp", static_cast<int>(temp_counter_++));
      auto mid_var = std::make_shared<Var>(name, cast_expr->GetType(), op->span_);
      stmts.push_back(std::make_shared<AssignStmt>(mid_var, cast_expr, op->span_));
      cur = mid_var;
    }

    auto final_assign = MutableCopy(op);
    final_assign->value_ = MakeCast(cur, chain.back(), mode, op->span_);
    stmts.push_back(std::move(final_assign));

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

 private:
  const backend::TcvtAdjacency& table_;
  std::string arch_name_;
  std::size_t temp_counter_ = 0;
};

FunctionPtr TransformLegalizeTileCast(const FunctionPtr& func) {
  if (!func) return func;
  // Tile casts only live in InCore (and AIC/AIV after expansion). Skip host orch.
  if (func->level_.has_value() && *func->level_ == Level::HOST) {
    return func;
  }
  // The native-cast table is a backend fact, so without a configured backend
  // there is nothing to legalize against -- leave the IR untouched rather than
  // guess a profile (several codegen tests drive passes with no backend set).
  // Both lookups below CHECK-fail when unconfigured, so probe first.
  if (!backend::BackendConfig::IsConfigured()) {
    return func;
  }
  const auto* ctx = PassContext::Current();
  const backend::BackendHandler* handler =
      ctx != nullptr ? ctx->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
  if (handler == nullptr) {
    return func;
  }
  LegalizeTileCastMutator mutator(handler->GetTcvtAdjacency(), handler->GetPtoTargetArch());
  return mutator.VisitFunction(func);
}

}  // namespace

namespace pass {

Pass LegalizeTileCast() {
  return CreateFunctionPass(TransformLegalizeTileCast, "LegalizeTileCast", kLegalizeTileCastProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
