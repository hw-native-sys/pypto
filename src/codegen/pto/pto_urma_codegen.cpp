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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto::codegen {
namespace {

constexpr char kSession[] = "!emitc.opaque<\"simpler::urma::UrmaSession\">";
constexpr char kSessionPtr[] = R"(!emitc.ptr<!emitc.opaque<"simpler::urma::UrmaSession">>)";

}  // namespace

void PTOCodegen::EmitUrmaMakeSession(const ir::CallPtr& op) {
  CHECK_SPAN(GetBackendHandler()->GetPtoTargetArch() == "a5", op->span_)
      << "pld.system.make_session requires the A5 backend";
  const auto storage = NewTemp();
  Emit(storage + R"( = "emitc.variable"() {value = #emitc.opaque<"">} : () -> )" + kSession);
  auto session = GetCurrentResultVar() ? GetCurrentResultTarget() : NewTemp();
  Emit(session + " = emitc.apply \"&\"(" + storage + ") : (" + kSession + ") -> " + kSessionPtr);
  const std::string signature = std::string("(!pto.ptr<i64>, !pto.ptr<i64>, ") + kSessionPtr + ")";
  urma_adapter_signatures_["pypto_urma_make_session"] = signature;
  Emit("func.call @pypto_urma_make_session(" + GetDeferredCompletionRawArgsSSA() + ", " +
       GetExprAsCode(op->args_[0]) + ", " + session + ") : " + signature + " -> ()");
  SetCurrentExprValue(session);
}

// Shared by both directions. pld.tile.get_async and pld.tile.put_async have the
// same op shape - result arity 2, tuple element 0 aliasing args_[0] and element 1
// carrying the event, session at args_[3] - so only two things differ: which
// endpoint is window-bound and supplies the CommContext, and which adapter the
// emitted call names.
void PTOCodegen::EmitUrmaTileTransfer(const ir::CallPtr& op, bool is_put) {
  const auto dst = ir::AsVarLike(op->args_[0]);
  const auto src = ir::AsVarLike(op->args_[2]);
  const auto* kind = is_put ? "Gather PUT" : "Scatter GET";
  const auto* op_name = is_put ? "pld.tile.put_async" : "pld.tile.get_async";
  CHECK_SPAN(GetBackendHandler()->GetPtoTargetArch() == "a5", op->span_)
      << op_name << " requires the A5 backend";
  INTERNAL_CHECK_SPAN(dst && src, op->span_) << kind << " endpoints must be materialized tensor variables";
  // The first tuple element borrows the submitted destination's view and storage.
  // The result var must be bound: an unbound transfer lowers to an EvalStmt, where
  // GetCurrentResultVar() is null, and a null key here is matched by any later
  // TupleGetItemExpr over a non-Var tuple, silently rebinding that result to this
  // destination.
  const auto result = GetCurrentResultVar();
  INTERNAL_CHECK_SPAN(result, op->span_)
      << kind << " must bind its result tuple before the destination alias is recorded";
  fs_.urma_transfer_destinations[result.get()] = dst;
  const auto type = ir::AsTensorTypeLike(dst->GetType());
  const auto bits = type->dtype_.GetBit();
  CHECK_SPAN(bits >= 8 && bits % 8 == 0, op->span_) << op_name << " requires byte-sized elements";
  constexpr int64_t kMaxBytes = int64_t{256} << 20;
  int64_t fixed_bytes = static_cast<int64_t>(bits / 8);
  std::vector<ir::ExprPtr> dynamic_dims;
  for (const auto& dim : type->shape_) {
    if (auto extent = ir::As<ir::ConstInt>(dim)) {
      CHECK_SPAN(extent->value_ > 0 && extent->value_ <= kMaxBytes / fixed_bytes, op->span_)
          << "URMA " << kind << " requires 1 byte through 256 MiB per request";
      fixed_bytes *= extent->value_;
    } else {
      dynamic_dims.push_back(dim);
    }
  }
  std::string bytes = GetOrEmitConstant(fixed_bytes, DataType::INT64);
  for (const auto& dim : dynamic_dims) {
    // Saturate each factor before multiplying. An oversized dynamic request
    // must reach the helper's range diagnostic even when the product would wrap.
    auto limit = GetOrEmitConstant(kMaxBytes + 1, DataType::INT64);
    auto extent = EmitCastToI64(dim, GetExprAsCode(dim));
    auto factor = NewTemp(), partial = NewTemp(), product = NewTemp();
    Emit(factor + " = arith.minui " + extent + ", " + limit + " : i64");
    Emit(partial + " = arith.minui " + bytes + ", " + limit + " : i64");
    Emit(product + " = arith.muli " + partial + ", " + factor + " : i64");
    bytes = product;
  }

  const auto peer = EmitCastToI64(op->args_[1], GetExprAsCode(op->args_[1]));
  const auto element = GetTypeString(type->dtype_);
  const std::string ptr = "!pto.ptr<" + element + ">";
  const auto suffix = element.rfind("!pto.", 0) == 0 ? element.substr(5) : element;
  const auto* adapter = is_put ? "pypto_urma_gather_put_" : "pypto_urma_scatter_get_";
  const auto name = adapter + suffix;
  const auto signature = std::string("(!pto.ptr<i64>, !pto.ptr<i64>, ") + kSessionPtr + ", !pto.ptr<ui8>, " +
                         ptr + ", " + ptr + ", i64, i64)";
  urma_adapter_signatures_[name] = signature;
  const auto& params = fs_.current_function->params_;
  const auto scratch = std::find_if(params.begin(), params.end(), [](const auto& param) {
    return param->name_hint_ == backend::kUrmaScatterScratchParam;
  });
  // InjectGMPipeBuffer establishes both of these, not the user, so a failure here
  // is a compiler bug rather than bad input - notably a third transfer op left out
  // of that pass's matches lambda.
  INTERNAL_CHECK_SPAN(scratch != params.end(), op->span_)
      << "URMA " << kind << " requires materialized GM descriptor scratch";
  const auto buffer = ir::AsTensorTypeLike((*scratch)->GetType());
  const auto extent =
      buffer && buffer->shape_.size() == 1 ? ir::As<ir::ConstInt>(buffer->shape_[0]) : nullptr;
  INTERNAL_CHECK_SPAN(extent && buffer->dtype_ == DataType::UINT8 &&
                          extent->value_ >= GetBackendHandler()->GetUrmaScatterScratchBytes(),
                      op->span_)
      << "URMA descriptor scratch must cover every physical sender";
  // PUT writes into the remote window through dst; GET reads the remote window
  // through src. The CommContext comes from whichever endpoint is window-bound.
  const auto* remote = is_put ? dst.get() : src.get();
  const auto comm_ctx = GetCommCtxSSAFor(remote);
  // Interpolating an empty context would emit a call with a missing operand and
  // fail downstream in PTOAS as an unrelated parse error.
  INTERNAL_CHECK_SPAN(!comm_ctx.empty(), op->span_)
      << kind << " has no CommContext bound to its window endpoint";
  Emit("func.call @" + name + "(" + GetDeferredCompletionRawArgsSSA() + ", " + comm_ctx + ", " +
       GetExprAsCode(op->args_[3]) + ", " + GetTensorBasePtr(*scratch) + ", " + GetTensorBasePtr(dst) + ", " +
       GetTensorBasePtr(src) + ", " + bytes + ", " + peer + ") : " + signature + " -> ()");
}

void PTOCodegen::EmitUrmaScatterGet(const ir::CallPtr& op) { EmitUrmaTileTransfer(op, /*is_put=*/false); }

void PTOCodegen::EmitUrmaGatherPut(const ir::CallPtr& op) { EmitUrmaTileTransfer(op, /*is_put=*/true); }
}  // namespace pypto::codegen
