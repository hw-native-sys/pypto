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
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Flags every atomic-add write into a global tensor whose dtype the target
/// backend's store pipe cannot combine.
///
/// A bf16 atomic-add lowers to pto-isa `SetAtomicAdd<bfloat16_t>` ->
/// `set_atomic_bf16`, honoured on the A2/A3 store path and not on A5
/// (`BackendHandler::SupportsBf16AtomicAdd`). The distributed remote-put path
/// is the *same* mechanism rather than a parallel one: pto-isa's comm `TPut`
/// streams the transfer through its VEC staging tile and lands each chunk with
/// `TSTORE_IMPL<..., AtomicAdd>`, and `remote_store` emits a `pto.tstore`
/// directly, so one predicate governs every atomic site. ptoas carries no
/// atomic dtype rule of its own (`TPutOp::verify` checks element-type agreement
/// and shapes only), so without this check the program reaches a pto-isa
/// `static_assert` in generated code the user never wrote.
///
/// Unlike `AccToGmStoreValid`, nothing here depends on lowering: the atomic
/// kwarg and the destination dtype are present in the user's own IR, so this
/// runs at pipeline input and the error carries the original `Span`.
class AtomicAddDtypeVisitor : public IRVisitor {
 public:
  AtomicAddDtypeVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name,
                        const backend::BackendHandler* handler)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), handler_(handler) {}

  void VisitExpr_(const CallPtr& op) override {
    CheckAtomicWrite(op);
    IRVisitor::VisitExpr_(op);
  }

  // The atomic ops below are all operators, and a Submit launches a Function,
  // so a Submit cannot carry one today. Funnelling it through the Call view
  // anyway keeps this verifier correct if that ever changes.
  void VisitExpr_(const SubmitPtr& op) override {
    if (op) CheckAtomicWrite(SubmitToCallView(op));
    IRVisitor::VisitExpr_(op);
  }

 private:
  /// The GM destination operand index, per atomic-capable op:
  ///   tile.store(tile, offsets, output_tensor, *, atomic)      -> args[2]
  ///   tensor.assemble(target, source, offsets, *, atomic)      -> args[0]
  ///   pld.tensor.put(dst, peer, src, ..., *, atomic)           -> args[0]
  ///   pld.tile.put(dst, peer, src, stage, ..., *, atomic)      -> args[0]
  ///   pld.tensor.remote_store(src, target, peer, offsets, *, atomic) -> args[1]
  ///   pld.tile.remote_store(src, target, peer, offsets, *, atomic)   -> args[1]
  /// Returns false when the call is not an atomic-capable op.
  static bool GetDestArgIndex(const CallPtr& call, size_t* dest_index) {
    if (IsOp(call, "tile.store")) {
      *dest_index = 2;
      return true;
    }
    if (IsOp(call, "tensor.assemble") || IsOp(call, "pld.tensor.put") || IsOp(call, "pld.tile.put")) {
      *dest_index = 0;
      return true;
    }
    if (IsOp(call, "pld.tensor.remote_store") || IsOp(call, "pld.tile.remote_store")) {
      *dest_index = 1;
      return true;
    }
    return false;
  }

  void CheckAtomicWrite(const CallPtr& call) {
    size_t dest_index = 0;
    if (!GetDestArgIndex(call, &dest_index)) return;
    if (call->args_.size() <= dest_index) return;
    if (call->GetKwarg<int>("atomic", 0) != static_cast<int>(AtomicType::kAdd)) return;

    const auto& dest = call->args_[dest_index];
    if (!dest) return;
    // AsTensorTypeLike covers the plain TensorType destinations of the local
    // store/assemble path and the DistributedTensorType destination of a put.
    auto tensor_type = AsTensorTypeLike(dest->GetType());
    if (!tensor_type) return;

    // Only bf16 varies by backend; the remaining hardware atomic-add dtypes are
    // accepted everywhere and the backend-neutral set is gated in the op
    // deducers (tile_ops/memory.cpp, tensor_ops/memory.cpp, and the shared
    // comm_op::ValidateAtomicAddDtype the distributed ops call).
    if (tensor_type->dtype_ != DataType::BF16) return;
    if (handler_->SupportsBf16AtomicAdd()) return;

    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "AtomicAddDtypeValid", /*error_code=*/1,
        "atomic-add into a bf16 global tensor is not supported on the '" + handler_->GetPtoTargetArch() +
            "' backend (function '" + func_name_ +
            "'). bf16 atomic-add requires the Ascend910B (A2/A3) profile, whose store pipe honours "
            "set_atomic_bf16. Accumulate into an fp32 tensor and cast to bf16 after the reduction "
            "instead.",
        call->span_);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  const backend::BackendHandler* handler_;
};

}  // namespace

class AtomicAddDtypeValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AtomicAddDtypeValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    // Which atomic dtypes the store pipe combines is a backend fact. Several
    // codegen tests drive passes with no backend configured; there is nothing
    // to verify against then, and guessing a profile would reject programs the
    // real target accepts. Both lookups below CHECK-fail when unconfigured, so
    // probe first.
    if (!backend::BackendConfig::IsConfigured()) return;
    const auto* ctx = PassContext::Current();
    const backend::BackendHandler* handler =
        ctx != nullptr ? ctx->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
    if (handler == nullptr) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;
      AtomicAddDtypeVisitor visitor(diagnostics, func->name_, handler);
      visitor.VisitFunction(func);
    }
  }
};

PropertyVerifierPtr CreateAtomicAddDtypeValidPropertyVerifier() {
  return std::make_shared<AtomicAddDtypeValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
