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
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"

namespace pypto::codegen {
namespace {

constexpr char kSession[] = "!emitc.opaque<\"simpler::urma::UrmaSession\">";
constexpr char kSessionPtr[] = R"(!emitc.ptr<!emitc.opaque<"simpler::urma::UrmaSession">>)";

}  // namespace

void PTOCodegen::EmitUrmaMakeSession(const ir::CallPtr& op) {
  CHECK_SPAN(GetBackendHandler()->GetPtoTargetArch() == "a5", op->span_)
      << "pld.system.make_session requires the A5 backend";
  const auto storage = NewTemp();
  Emit(storage + " = \"emitc.variable\"() {value = #emitc.opaque<\"\">} : () -> " + kSession);
  auto session = GetCurrentResultVar() ? GetCurrentResultTarget() : NewTemp();
  Emit(session + " = emitc.apply \"&\"(" + storage + ") : (" + kSession + ") -> " + kSessionPtr);
  const std::string signature = std::string("(!pto.ptr<i64>, !pto.ptr<i64>, ") + kSessionPtr + ")";
  urma_adapter_signatures_["pypto_urma_make_session"] = signature;
  Emit("func.call @pypto_urma_make_session(" + GetDeferredCompletionRawArgsSSA() + ", " +
       GetExprAsCode(op->args_[0]) + ", " + session + ") : " + signature + " -> ()");
  SetCurrentExprValue(session);
}

void PTOCodegen::EmitUrmaScatterGet(const ir::CallPtr& op) {
  const auto dst = ir::AsVarLike(op->args_[0]);
  const auto src = ir::AsVarLike(op->args_[2]);
  INTERNAL_CHECK_SPAN(dst && src, op->span_) << "Scatter GET endpoints must be materialized tensor variables";
  // The first tuple element borrows the submitted destination's view and storage.
  fs_.urma_get_destinations[GetCurrentResultVar().get()] = dst;
  const auto type = ir::AsTensorTypeLike(dst->GetType());
  const auto bits = type->dtype_.GetBit();
  CHECK_SPAN(bits >= 8 && bits % 8 == 0, op->span_) << "pld.tile.get_async requires byte-sized elements";
  constexpr int64_t kMaxBytes = 256 * 1024 * 1024;
  int64_t fixed_bytes = bits / 8;
  std::vector<ir::ExprPtr> dynamic_dims;
  for (const auto& dim : type->shape_) {
    if (auto extent = ir::As<ir::ConstInt>(dim)) {
      CHECK_SPAN(extent->value_ > 0 && extent->value_ <= kMaxBytes / fixed_bytes, op->span_)
          << "URMA Scatter GET requires 1 byte through 256 MiB per request";
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
  const auto name = "pypto_urma_scatter_get_" + suffix;
  const auto signature = std::string("(!pto.ptr<i64>, !pto.ptr<i64>, ") + kSessionPtr + ", !pto.ptr<ui8>, " +
                         ptr + ", " + ptr + ", i64, i64)";
  urma_adapter_signatures_[name] = signature;
  const auto& params = fs_.current_function->params_;
  const auto scratch = std::find_if(params.begin(), params.end(), [](const auto& param) {
    return param->name_hint_ == backend::kUrmaScatterScratchParam;
  });
  CHECK_SPAN(scratch != params.end(), op->span_)
      << "URMA Scatter GET requires materialized GM descriptor scratch";
  const auto buffer = ir::AsTensorTypeLike((*scratch)->GetType());
  const auto extent =
      buffer && buffer->shape_.size() == 1 ? ir::As<ir::ConstInt>(buffer->shape_[0]) : nullptr;
  CHECK_SPAN(extent && buffer->dtype_ == DataType::UINT8 &&
                 extent->value_ >= GetBackendHandler()->GetUrmaScatterScratchBytes(),
             op->span_)
      << "URMA Scatter descriptor scratch must cover every physical sender";
  Emit("func.call @" + name + "(" + GetDeferredCompletionRawArgsSSA() + ", " + GetCommCtxSSAFor(src.get()) +
       ", " + GetExprAsCode(op->args_[3]) + ", " + GetTensorBasePtr(*scratch) + ", " + GetTensorBasePtr(dst) +
       ", " + GetTensorBasePtr(src) + ", " + bytes + ", " + peer + ") : " + signature + " -> ()");
}
}  // namespace pypto::codegen
