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
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto::codegen {
namespace {

using ir::As;
using ir::AssignStmtPtr;
using ir::AsVarLike;
using ir::AsyncEventType;
using ir::AsyncSessionType;
using ir::BreakStmtPtr;
using ir::Call;
using ir::CallPtr;
using ir::ContinueStmtPtr;
using ir::ExprPtr;
using ir::ForStmtPtr;
using ir::IfStmtPtr;
using ir::IRVisitor;
using ir::IsA;
using ir::ReturnStmtPtr;
using ir::StmtPtr;
using ir::TupleGetItemExpr;
using ir::TupleType;
using ir::Var;
using ir::VarPtr;
using ir::WhileStmtPtr;

constexpr char kSession[] = "!emitc.opaque<\"simpler::urma::UrmaSession\">";
constexpr char kSessionPtr[] = R"(!emitc.ptr<!emitc.opaque<"simpler::urma::UrmaSession">>)";

// Track the identity of each submission through SSA copies and tuple projections.
// The combined Simpler request transfers ownership at the GET, so every execution
// of that GET must also reach its unique completion marker.
class UrmaEventPlanner : public IRVisitor {
 public:
  explicit UrmaEventPlanner(std::map<const Var*, VarPtr>& destinations) : destinations_(destinations) {}

  void Check(const StmtPtr& body) { VisitRegion(body); }

 private:
  void VisitRegion(const StmtPtr& body) {
    auto enclosing = std::move(pending_);
    pending_.clear();
    VisitStmt(body);
    CHECK_SPAN(pending_.empty(), body->span_)
        << "pld.tile.get_async requires a matching defer_wait_event in the same branch and loop iteration";
    pending_ = std::move(enclosing);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    VisitRegion(op->then_body_);
    if (op->else_body_) VisitRegion(*op->else_body_);
  }
  void VisitLoop(const StmtPtr& body) {
    const auto enclosing = pending_in_loop_;
    pending_in_loop_ = 0;
    VisitRegion(body);
    pending_in_loop_ = enclosing;
  }
  void VisitStmt_(const ForStmtPtr& op) override { VisitLoop(op->body_); }
  void VisitStmt_(const WhileStmtPtr& op) override {
    VisitExpr(op->condition_);
    VisitLoop(op->body_);
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    CHECK_SPAN(pending_total_ == 0, op->span_)
        << "tile.get_async requires a matching defer_wait_event before return";
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BreakStmtPtr& op) override {
    CHECK_SPAN(pending_in_loop_ == 0, op->span_)
        << "break can bypass the defer_wait_event for tile.get_async";
  }
  void VisitStmt_(const ContinueStmtPtr& op) override {
    CHECK_SPAN(pending_in_loop_ == 0, op->span_)
        << "continue can bypass the defer_wait_event for tile.get_async";
  }

  struct Definition {
    const Var* binding;
    ExprPtr value;
  };

  Definition Resolve(const ExprPtr& expr) const {
    auto var = AsVarLike(expr);
    if (var) {
      auto it = definitions_.find(var.get());
      if (it != definitions_.end()) return it->second;
    }
    return {var.get(), expr};
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    auto definition = Resolve(op->value_);
    if (IsA<TupleType>(op->var_->GetType()) || IsA<AsyncEventType>(op->var_->GetType()) ||
        IsA<AsyncSessionType>(op->var_->GetType())) {
      if (!definition.binding) definition.binding = op->var_.get();
      definitions_[op->var_.get()] = definition;
    }
    if (IsOp(As<Call>(op->value_), "pld.tile.get_async")) {
      pending_.insert(op->var_.get());
      ++pending_total_;
      ++pending_in_loop_;
      return;
    }
    if (auto item = As<TupleGetItemExpr>(definition.value); item && item->index_ == 0) {
      auto get = As<Call>(Resolve(item->tuple_).value);
      if (IsOp(get, "pld.tile.get_async")) destinations_[op->var_.get()] = AsVarLike(get->args_[0]);
    }
    VisitExpr(op->value_);
  }

  void VisitExpr_(const CallPtr& op) override {
    CHECK_SPAN(!IsOp(op, "pld.tile.get_async"), op->span_)
        << "tile.get_async requires a matching defer_wait_event on its event result";
    CHECK_SPAN(!IsOp(op, "pld.system.wait_event"), op->span_)
        << "pld.system.wait_event is not supported by URMA codegen";
    if (IsOp(op, "pld.system.defer_wait_event")) {
      auto item = As<TupleGetItemExpr>(Resolve(op->args_[0]).value);
      auto tuple = item && item->index_ == 1 ? Resolve(item->tuple_) : Definition{};
      CHECK_SPAN(pending_.erase(tuple.binding) == 1, op->span_)
          << "defer_wait_event requires one preceding tile.get_async in the same branch and loop iteration";
      --pending_total_;
      --pending_in_loop_;
      if (op->args_.size() == 2) {
        auto get = As<Call>(tuple.value);
        INTERNAL_CHECK_SPAN(Resolve(op->args_[1]).binding == Resolve(get->args_[3]).binding, op->span_)
            << "defer_wait_event requires the same session that submitted the GET";
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  std::map<const Var*, Definition> definitions_;
  std::set<const Var*> pending_;
  size_t pending_total_ = 0;
  size_t pending_in_loop_ = 0;
  std::map<const Var*, VarPtr>& destinations_;
};

}  // namespace

void PTOCodegen::PlanUrmaEventCompletions(const ir::StmtPtr& body) {
  UrmaEventPlanner(fs_.urma_destination_aliases).Check(body);
}

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
