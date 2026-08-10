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

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Flags every `tile.store` that drains an Acc-resident tile into a GM tensor
/// whose dtype the cube fix-pipe cannot narrow into.
///
/// The cube fix-pipe writes an accumulator to global memory through a fixed
/// destination set (INT32/FP32/FP16[/BF16], see
/// `BackendHandler::SupportsAccToGmDtype`). An INT8/INT16 destination is not in
/// it, and ptoas rejects the resulting `pto.tstore` -- but only after codegen,
/// pointing at a line in a generated `.pto` the user never wrote. One such
/// program also tends to trip a *second*, unrelated-looking op downstream (an
/// int8 zero-init lowers to a `pto.texpands` with the same illegal dtype), so
/// the late diagnostic names two symptoms and never the cause.
///
/// Checking here -- the first point at which memory spaces are resolved -- lets
/// the error carry the original `Span` and name the one decision behind both.
class AccToGmStoreVisitor : public IRVisitor {
 public:
  AccToGmStoreVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name,
                      const backend::BackendHandler* handler)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), handler_(handler) {}

  void VisitExpr_(const CallPtr& op) override {
    CheckStore(op);
    IRVisitor::VisitExpr_(op);
  }

  // `tile.store` is an operator, and a Submit launches a Function, so a Submit
  // cannot carry one today. Funnelling it through the Call view anyway keeps
  // this verifier correct if that ever changes, and costs one null-safe IsOp.
  void VisitExpr_(const SubmitPtr& op) override {
    if (op) CheckStore(SubmitToCallView(op));
    IRVisitor::VisitExpr_(op);
  }

 private:
  /// `tile.store(tile, offsets, output_tensor, ...)` -- args[0] is the source
  /// tile, args[2] the GM destination.
  static constexpr size_t kSourceTileArg = 0;
  static constexpr size_t kOutputTensorArg = 2;

  void CheckStore(const CallPtr& call) {
    if (!IsOp(call, "tile.store")) return;

    const auto& args = call->args_;
    if (args.size() <= kOutputTensorArg) return;

    auto source_tile = args[kSourceTileArg];
    auto dest_tensor = args[kOutputTensorArg];
    if (!source_tile || !dest_tensor) return;

    auto tile_type = As<TileType>(source_tile->GetType());
    // memory_space_ is nullopt before InferTileMemorySpace resolves it; this
    // verifier only runs after, but a tile with an unresolved space is not
    // something we can classify, so leave it to the downstream check.
    if (!tile_type || !tile_type->memory_space_.has_value()) return;
    if (*tile_type->memory_space_ != MemorySpace::Acc) return;

    auto tensor_type = AsTensorTypeLike(dest_tensor->GetType());
    if (!tensor_type) return;

    const auto& dtype = tensor_type->dtype_;
    if (handler_->SupportsAccToGmDtype(dtype)) return;

    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "AccToGmStoreValid", /*error_code=*/1,
        "cube accumulator cannot be stored directly into a '" + dtype.ToString() +
            "' global tensor on the '" + handler_->GetPtoTargetArch() + "' backend (function '" + func_name_ +
            "'). The fix-pipe narrows an Acc tile only into INT32/FP32/FP16" +
            (handler_->SupportsAccToGmDtype(DataType::BF16) ? "/BF16" : "") +
            ". Narrow through the vector unit first -- cast the matmul result to '" + dtype.ToString() +
            "' (e.g. pl.cast(result, ...)) and store that -- or accumulate "
            "into an INT32/FP32 tensor and convert afterwards.",
        call->span_);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  const backend::BackendHandler* handler_;
};

}  // namespace

class AccToGmStoreValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AccToGmStoreValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    // The destination whitelist is a backend fact. Several codegen tests drive
    // passes with no backend configured; there is nothing to verify against
    // then, and guessing a profile would reject programs the real target
    // accepts. Both lookups below CHECK-fail when unconfigured, so probe first.
    if (!backend::BackendConfig::IsConfigured()) return;
    const auto* ctx = PassContext::Current();
    const backend::BackendHandler* handler =
        ctx != nullptr ? ctx->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
    if (handler == nullptr) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;
      AccToGmStoreVisitor visitor(diagnostics, func->name_, handler);
      visitor.VisitFunction(func);
    }
  }
};

PropertyVerifierPtr CreateAccToGmStoreValidPropertyVerifier() {
  return std::make_shared<AccToGmStoreValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
