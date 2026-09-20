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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Flags every `tile.assemble` whose FIXPIPE epilogue (`pre_quant` / `pre_relu`)
/// the backend cannot actually perform.
///
/// The epilogue only exists on the cube's Acc->Mat writeback (`pto.tinsert`,
/// pto-isa `mte_l0c_l1`), so two things have to hold: the assemble must *be*
/// that writeback, and the backend's fix-pipe must have a scale-bearing mode for
/// this (source, target) dtype pair.
///
/// **Why this is an error and not a perf hint.** For the Acc->GM half
/// (`AccToGmStoreValid`) an illegal pair is at least caught downstream: ptoas
/// verifies the a2a3 `pto.tstore` dtype pair explicitly. Here there is no such
/// backstop on every target — ptoas verifies the quantized `pto.tinsert` pair on
/// a2a3 only, and pto-isa answers an unsupported pair from
/// `GetScalarPreQuantMode` with `QuantMode_t::NoQuant`, which *silently drops the
/// scale*. A pair this verifier lets through on a5 would compile, run, and
/// return unscaled numbers. Rejecting it here, where the span still points at
/// the user's own line, is the only place the mistake is visible.
///
/// Memory spaces are resolved by `InferTileMemorySpace`, so this runs after it;
/// an assemble whose spaces are still unknown is left to that pass.
class FixpipeEpilogueVisitor : public IRVisitor {
 public:
  FixpipeEpilogueVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name,
                         const backend::BackendHandler* handler)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), handler_(handler) {}

  void VisitExpr_(const CallPtr& op) override {
    CheckAssemble(op);
    IRVisitor::VisitExpr_(op);
  }

  // `tile.assemble` is an operator and a Submit launches a Function, so a Submit
  // cannot carry one today. Routing it through the Call view keeps this correct
  // if that ever changes, at the cost of one null-safe IsOp.
  void VisitExpr_(const SubmitPtr& op) override {
    if (op) CheckAssemble(SubmitToCallView(op));
    IRVisitor::VisitExpr_(op);
  }

 private:
  /// `tile.assemble(target, source, offset)` -- args[0] is the Mat target,
  /// args[1] the Acc source being drained into it.
  static constexpr size_t kTargetArg = 0;
  static constexpr size_t kSourceArg = 1;

  void CheckAssemble(const CallPtr& call) {
    if (!IsOp(call, "tile.assemble")) return;

    const auto pre_quant = GetOptionalDoubleKwarg(call->kwargs_, "pre_quant");
    const bool pre_relu = call->GetKwarg<bool>("pre_relu", false);
    if (!pre_quant.has_value() && !pre_relu) return;

    const auto& args = call->args_;
    if (args.size() <= kSourceArg) return;
    auto target_type = args[kTargetArg] ? As<TileType>(args[kTargetArg]->GetType()) : nullptr;
    auto source_type = args[kSourceArg] ? As<TileType>(args[kSourceArg]->GetType()) : nullptr;
    if (!target_type || !source_type) return;
    // Unresolved spaces are not classifiable; InferTileMemorySpace owns that.
    if (!target_type->memory_space_.has_value() || !source_type->memory_space_.has_value()) return;

    const bool acc_to_mat =
        *source_type->memory_space_ == MemorySpace::Acc && *target_type->memory_space_ == MemorySpace::Mat;
    if (!acc_to_mat) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "FixpipeEpilogueValid", /*error_code=*/1,
          "tile.assemble carries a FIXPIPE epilogue (pre_quant / pre_relu) but moves " +
              MemorySpaceToString(*source_type->memory_space_) + " -> " +
              MemorySpaceToString(*target_type->memory_space_) + " (function '" + func_name_ +
              "'). The epilogue is part of the cube's Acc-to-Mat writeback and has no meaning on any "
              "other move; scale and activate in the vector unit instead -- "
              "pl.maximum(pl.mul(tile, scale), 0.0).",
          call->span_);
      return;
    }

    const auto& src_dtype = source_type->dtype_;
    const auto& dst_dtype = target_type->dtype_;

    // Acc -> Mat has two lowerings and only one of them is the fix-pipe. A
    // same-dtype assemble is an MTE1 `pto.subview` + `pto.tmov` with no fix-pipe
    // in the path, so it can carry neither pre-op -- unlike the Acc -> GM store,
    // where every write is a fix-pipe drain and `pre_relu` alone is fine.
    if (!CubeMatWritebackUsesFixpipe(src_dtype, dst_dtype, pre_quant.has_value())) {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "FixpipeEpilogueValid", /*error_code=*/1,
          "tile.assemble carries pre_relu but writes a '" + dst_dtype.ToString() + "' Mat tile from a '" +
              src_dtype.ToString() + "' accumulator (function '" + func_name_ +
              "'), which is a plain on-chip move, not a fix-pipe writeback -- there is no fix-pipe to "
              "apply the activation. Narrow on the way out so the writeback converts (assemble into an "
              "FP16 or BF16 Mat tile), add a pre_quant scale, or apply the activation in the vector "
              "unit -- pl.maximum(result, 0.0).",
          call->span_);
      return;
    }

    if (!pre_quant.has_value()) return;  // pre_relu alone rides the unscaled narrowing.

    constexpr auto kMat = backend::BackendHandler::FixpipeDest::kMat;
    if (handler_->SupportsFixpipePreQuant(src_dtype, dst_dtype, kMat)) return;

    // A handler that names no target at all for *any* accumulator withholds the
    // whole destination rather than this one dtype pair -- today every backend
    // does, because ptoas mis-emits the scale on `pto.tinsert` (PTOAS#1570; see
    // the comment on `Ascend910BHandler::SupportsFixpipePreQuant`). Saying "supported
    // targets: none" there would blame the dtypes for a toolchain defect.
    if (DescribeFixpipePreQuantTargets(*handler_, DataType::INT32, kMat) == "none" &&
        DescribeFixpipePreQuantTargets(*handler_, DataType::FP32, kMat) == "none") {
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "FixpipeEpilogueValid", /*error_code=*/1,
          "tile.assemble carries a pre_quant scale on the Acc -> Mat writeback (function '" + func_name_ +
              "'), which the '" + handler_->GetPtoTargetArch() +
              "' backend withholds: ptoas assembles the scale but emits a pto-isa call that silently "
              "drops it (PTOAS#1570). The Acc -> GM form is unaffected -- "
              "pl.tile.store(acc, ..., pre_quant=s) -- "
              "and pre_relu alone is still available here on a converting assemble. Otherwise scale in "
              "the vector unit: pl.mul(pl.cast(result, ...), scale).",
          call->span_);
      return;
    }
    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "FixpipeEpilogueValid", /*error_code=*/1,
        "a '" + src_dtype.ToString() + "' cube accumulator cannot be assembled into a '" +
            dst_dtype.ToString() + "' Mat tile with a pre_quant scale on the '" +
            handler_->GetPtoTargetArch() + "' backend (function '" + func_name_ +
            "'): the fix-pipe has no scale-bearing " + src_dtype.ToString() + " -> " + dst_dtype.ToString() +
            " mode. Supported Mat targets from a '" + src_dtype.ToString() + "' accumulator here are " +
            DescribeFixpipePreQuantTargets(*handler_, src_dtype, backend::BackendHandler::FixpipeDest::kMat) +
            ". Scale in the vector unit instead -- pl.mul(pl.cast(result, ...), scale) -- and move "
            "that to Mat.",
        call->span_);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  const backend::BackendHandler* handler_;
};

}  // namespace

std::string DescribeFixpipePreQuantTargets(const backend::BackendHandler& handler, DataType src_dtype,
                                           backend::BackendHandler::FixpipeDest dest) {
  // Every destination any backend's table can name, probed rather than
  // enumerated per backend, so a new entry in a handler reaches the diagnostic
  // without a second edit here.
  static const DataType kCandidates[] = {DataType::INT8, DataType::UINT8, DataType::FP16,     DataType::BF16,
                                         DataType::FP32, DataType::HF8,   DataType::FP8E4M3FN};
  std::string listed;
  for (const auto& candidate : kCandidates) {
    if (!handler.SupportsFixpipePreQuant(src_dtype, candidate, dest)) continue;
    if (!listed.empty()) listed += "/";
    listed += candidate.ToString();
  }
  return listed.empty() ? "none" : listed;
}

class FixpipeEpilogueValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "FixpipeEpilogueValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    // The supported dtype pairs are a backend fact. Several codegen tests drive
    // passes with no backend configured; there is nothing to verify against then,
    // and guessing a profile would reject programs the real target accepts.
    if (!backend::BackendConfig::IsConfigured()) return;
    const auto* ctx = PassContext::Current();
    const backend::BackendHandler* handler =
        ctx != nullptr ? ctx->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
    if (handler == nullptr) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;
      FixpipeEpilogueVisitor visitor(diagnostics, func->name_, handler);
      visitor.VisitFunction(func);
    }
  }
};

PropertyVerifierPtr CreateFixpipeEpilogueValidPropertyVerifier() {
  return std::make_shared<FixpipeEpilogueValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
