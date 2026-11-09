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

#include "pypto/ir/simt_analysis.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
class AccessCollector : public IRVisitor {
 public:
  explicit AccessCollector(const FunctionPtr& function) : summary(function->params_.size()) {
    for (size_t i = 0; i < function->params_.size(); ++i) parameters_.emplace(function->params_[i].get(), i);
  }

  SimtAccessSummary summary;

 protected:
  void VisitExpr_(const CallPtr& call) override {
    const auto* entry = LookupOpEntry(call->op_);
    if (entry && entry->GetOpCategory() == "SimtOp") {
      for (size_t i = 0; i < call->args_.size(); ++i) {
        if (auto it = parameters_.find(call->args_[i].get()); it != parameters_.end()) {
          // Scalar parameters have no physical memory effect. Use registered
          // effects for storage so atomics also drive outlining and scheduling.
          const auto& type = call->args_[i]->GetType();
          if (!As<TileType>(type) && !AsTensorTypeLike(type)) continue;
          const auto effect = entry->GetArgEffect(i, call->kwargs_);
          summary[it->second].reads |= ArgEffectReads(effect);
          summary[it->second].writes |= ArgEffectWrites(effect);
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  std::unordered_map<const Expr*, size_t> parameters_;
};
}  // namespace

SimtAccessSummary AnalyzeSimtAccess(const FunctionPtr& function) {
  CHECK(function && function->func_type_ == FunctionType::SIMT) << "Expected a SIMT helper";
  AccessCollector collector(function);
  collector.VisitStmt(function->body_);
  return collector.summary;
}

SimtResources InferSimtResources(const SimtLaunchStmtPtr& launch) {
  CHECK(launch) << "Expected a SimtLaunchStmt";
  auto* context = PassContext::Current();
  const auto* handler =
      context ? context->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
  CHECK(handler) << "SIMT resource inference requires a configured target";
  const auto limits = handler->GetSimtLimits();
  CHECK(limits) << "SIMT launch is unsupported by this target";
  int64_t threads = 1;
  for (size_t i = 0; i < launch->block_dims_.size(); ++i) {
    const auto value = As<ConstInt>(launch->block_dims_[i]);
    CHECK(value && value->value_ > 0 && value->value_ <= limits->max_block_dims[i])
        << "SIMT block dimension exceeds target limit";
    CHECK(value->value_ <= limits->max_threads / threads) << "SIMT thread count exceeds target limit";
    threads *= value->value_;
  }
  for (const auto& [bound, regs] : limits->register_tiers) {
    if (threads <= bound) return {static_cast<int>(threads), regs};
  }
  throw ValueError("SIMT target has no register tier for this launch");
}
}  // namespace pypto::ir
