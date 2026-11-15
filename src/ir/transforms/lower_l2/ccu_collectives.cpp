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

#include "src/ir/transforms/lower_l2/ccu_collectives.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/ccu_collective_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"

namespace pypto::ir {
bool IsCcuAllGather(const CallPtr& call) {
  return IsOp(call, "pld.tensor.allgather") && call->GetKwarg<std::string>("engine", "default") == "ccu";
}
bool IsCcuCollective(const CallPtr& call) { return IsCcuAllGather(call); }

namespace {

// SimplerCcuRegistrationBinding::channels has SIMPLER_CCU_MAX_RANKS entries.
constexpr int64_t kMaxCcuRanks = 64;

bool IsOrchAt(const FunctionPtr& func, Level level) {
  return (IsOrchestrationLike(func->func_type_) || func->role_ == Role::Orchestrator) &&
         func->level_ == level;
}

ExprPtr Index(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

TypePtr OrderType(const Span& span) {
  return std::make_shared<TensorType>(std::vector<ExprPtr>{Index(1, span)}, DataType::UINT8);
}

class Names : public IRVisitor {
 public:
  std::set<std::string> used;
  std::string Fresh(const std::string& prefix) {
    while (true) {
      auto name = prefix + std::to_string(next_++);
      if (used.insert(name).second) return name;
    }
  }

 protected:
  void VisitVarLike_(const VarPtr& var) override {
    used.insert(var->name_hint_);
    used.insert(auto_name::GetCompatibleBaseName(var->name_hint_));
  }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "pld.tensor.alloc_window_buffer")) used.insert(call->GetKwarg<std::string>("name"));
    IRVisitor::VisitExpr_(call);
  }

 private:
  size_t next_ = 0;
};

// Discovery also rejects control flow before rewriting it. A branch may make
// participants execute different CCU sequences; repeated device execution is
// not qualified yet. Host rank-dispatch loops are handled separately below.
class CcuFinder : public IRVisitor {
 public:
  explicit CcuFinder(FunctionPtr func) : func_(std::move(func)) {}
  bool found = false;

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (IsCcuCollective(call)) {
      CHECK_SPAN(IsOrchAt(func_, Level::CHIP), call->span_)
          << "CCU collective is supported only in CHIP orchestration, not HOST/InCore";
      CHECK_SPAN(depth_ == 0, call->span_) << "CHIP CCU collective requires a straight-line sequence; loops "
                                              "and conditional calls are unsupported";
      found = true;
    }
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    CHECK_SPAN(
        !IsOp(submit, "pld.tensor.allgather") || submit->GetKwarg<std::string>("engine", "default") != "ccu",
        submit->span_)
        << "CCU collective direct-op Submit is unsupported; submit a CHIP callable instead";
    IRVisitor::VisitExpr_(submit);
  }
  void VisitStmt_(const ForStmtPtr& op) override { Nested(op); }
  void VisitStmt_(const WhileStmtPtr& op) override { Nested(op); }
  void VisitStmt_(const IfStmtPtr& op) override { Nested(op); }

 private:
  template <typename T>
  void Nested(const T& op) {
    ++depth_;
    IRVisitor::VisitStmt_(op);
    --depth_;
  }
  FunctionPtr func_;
  int depth_ = 0;
};

// CCU allgather requires static positive packed ND views with no partial
// valid region. Dtype and rank restrictions are checked by its lowering.
std::string ValidateView(const DistributedTensorTypePtr& type, const Span& span) {
  CHECK_SPAN(type && !type->shape_.empty(), span) << "CCU collective requires a nonempty static shape";
  std::string key;
  int64_t stride = 1;
  const auto& view = type->tensor_view_;
  CHECK_SPAN(!view || view->layout == TensorLayout::ND, span)
      << "CCU collective requires contiguous ND views";
  CHECK_SPAN(!view || view->stride.empty() || view->stride.size() == type->shape_.size(), span)
      << "CCU collective requires full contiguous strides";
  CHECK_SPAN(!view || view->valid_shape.empty() || view->valid_shape.size() == type->shape_.size(), span)
      << "CCU collective requires a full valid view";
  for (size_t i = type->shape_.size(); i-- > 0;) {
    auto extent = As<ConstInt>(type->shape_[i]);
    CHECK_SPAN(extent && extent->value_ > 0, span) << "CCU collective requires static positive dimensions";
    if (view && !view->stride.empty()) {
      auto actual = As<ConstInt>(view->stride[i]);
      CHECK_SPAN(actual && actual->value_ == stride, span) << "CCU collective requires contiguous ND strides";
    }
    if (view && !view->valid_shape.empty()) {
      auto valid = As<ConstInt>(view->valid_shape[i]);
      CHECK_SPAN(valid && valid->value_ == extent->value_, span)
          << "CCU collective requires a full valid view";
    }
    CHECK_SPAN(extent->value_ <= std::numeric_limits<int64_t>::max() / stride, span)
        << "CCU collective element count exceeds INT64_MAX";
    stride *= extent->value_;
  }
  for (const auto& dim : type->shape_) key += "_" + std::to_string(As<ConstInt>(dim)->value_);
  return key;
}

using Chips = std::set<std::string>;

class CcuLowerer : public IRMutator {
 public:
  CcuLowerer(Names* names, std::map<std::string, FunctionPtr>* kernels) : names_(names), kernels_(kernels) {}

  StmtPtr Lower(const StmtPtr& body) {
    auto lowered = VisitStmt(body);
    allocations_.push_back(lowered);
    return SeqStmts::Flatten(std::move(allocations_), body->span_);
  }

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto call = As<Call>(IRMutator::VisitExpr_(op));
    return IsCcuAllGather(call) ? LowerAllGather(call) : call;
  }

 private:
  VarPtr Order(const std::string& key, const Span& span) {
    auto [token, inserted] = tokens_.try_emplace(key);
    if (inserted) {
      auto shape = std::make_shared<MakeTuple>(std::vector<ExprPtr>{Index(1, span)}, span);
      auto alloc = OpRegistry::GetInstance().Create(
          "tensor.create", {shape}, {{"dtype", DataType::UINT8}, {"layout", TensorLayout::ND}}, span);
      token->second = std::make_shared<Var>(names_->Fresh("_ccu_order_"), alloc->GetType(), span);
      allocations_.push_back(std::make_shared<AssignStmt>(token->second, alloc, span));
    }
    return token->second;
  }

  ExprPtr LowerAllGather(const CallPtr& call) {
    const auto& span = call->span_;
    auto input = As<DistributedTensorType>(call->args_[0]->GetType());
    auto target = As<DistributedTensorType>(call->args_[1]->GetType());
    auto signal = As<DistributedTensorType>(call->args_[2]->GetType());
    CHECK_SPAN(input, span) << "CCU allgather input must be a DistributedTensor window";
    ValidateView(input, span);
    auto shape_key = ValidateView(target, span);
    ValidateView(signal, span);
    CHECK_SPAN(input->dtype_ == DataType::BF16, span) << "CCU allgather supports only BF16";
    const int64_t nranks = As<ConstInt>(target->shape_[0])->value_;
    CHECK_SPAN(nranks >= 2 && nranks <= kMaxCcuRanks, span)
        << "CCU allgather requires 2..64 ranks, matching the simpler registration ABI";
    CHECK_SPAN(signal->shape_.size() == 2 && As<ConstInt>(signal->shape_[0])->value_ == nranks &&
                   As<ConstInt>(signal->shape_[1])->value_ == 1,
               span)
        << "CCU allgather requires signal shape [NR, 1] matching target [NR, N]";
    CHECK_SPAN(As<ConstInt>(input->shape_[1])->value_ <= std::numeric_limits<int64_t>::max() / (2 * nranks),
               span)
        << "CCU allgather target byte size exceeds INT64_MAX";
    const auto* handler = PassContext::Current()->GetBackendHandler();
    CHECK_SPAN(handler && handler->SupportCcu(), span)
        << "CCU allgather is unsupported by this backend; compiler lowering requires Ascend950";
    const std::string entry = "dtype_cpp=bfloat16_t,nranks_cpp=" + std::to_string(nranks);
    auto token = Order("chip", span);
    auto name = "__builtin_allgather_ccu__bf16_p" + std::to_string(nranks) + shape_key;
    if (!kernels_->count(name)) {
      std::vector<VarPtr> params{
          std::make_shared<Var>("input",
                                std::make_shared<DistributedTensorType>(input->shape_, input->dtype_), span),
          std::make_shared<Var>(
              "target", std::make_shared<DistributedTensorType>(target->shape_, target->dtype_), span),
          std::make_shared<Var>(
              "signal", std::make_shared<DistributedTensorType>(signal->shape_, signal->dtype_), span),
          std::make_shared<Var>("order", OrderType(span), span)};
      const auto definition_name = "__ccu_allgather_bf16_p" + std::to_string(nranks) + shape_key;
      if (!kernels_->count(definition_name)) {
        // Shapes select the wrapper; the native implementation accepts a runtime count.
        // Keep the declaration's DT signature for provenance and ctx forwarding.
        std::vector<VarPtr> native_params{std::make_shared<Var>("input", params[0]->GetType(), span),
                                          std::make_shared<Var>("target", params[1]->GetType(), span)};
        auto body = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{native_params[1]}, span);
        (*kernels_)[definition_name] =
            std::make_shared<Function>(definition_name, native_params,
                                       std::vector<ParamDirection>{ParamDirection::In, ParamDirection::InOut},
                                       std::vector<TypePtr>{native_params[1]->GetType()}, body, span,
                                       FunctionType::CCU, std::nullopt, std::nullopt,
                                       std::vector<std::pair<std::string, std::any>>{
                                           {kAttrBuiltinTemplateDir, std::string(kCcuAllGatherTemplate)},
                                           {kAttrBuiltinTemplateVars, entry}});
      }
      auto result = std::make_shared<Var>("result", params[1]->GetType(), span);
      auto invoke = std::make_shared<Call>(
          std::make_shared<GlobalVar>(definition_name), std::vector<ExprPtr>{params[0], params[1]},
          std::vector<std::pair<std::string, std::any>>{}, std::vector<std::pair<std::string, std::any>>{},
          result->GetType(), span);
      // Preserve the native request as an effectful statement; the returned
      // target alias is already represented by the wrapper signature.
      auto body = SeqStmts::Flatten({std::make_shared<EvalStmt>(invoke, span),
                                     std::make_shared<ReturnStmt>(std::vector<ExprPtr>{params[1]}, span)},
                                    span);
      (*kernels_)[name] = std::make_shared<Function>(
          name, params,
          std::vector<ParamDirection>{ParamDirection::In, ParamDirection::InOut, ParamDirection::InOut,
                                      ParamDirection::InOut},
          std::vector<TypePtr>{params[1]->GetType()}, body, span, FunctionType::AIV, std::nullopt,
          std::nullopt,
          std::vector<std::pair<std::string, std::any>>{
              {kAttrBuiltinTemplateDir, std::string(kCcuAllGatherTemplate)},
              {kAttrRequireContiguousOperands, true},
              {kAttrBufferIrSelected, false}});
    }
    return std::make_shared<Call>(
        std::make_shared<GlobalVar>(name),
        std::vector<ExprPtr>{call->args_[0], call->args_[1], call->args_[2], token},
        std::vector<std::pair<std::string, std::any>>{},
        std::vector<std::pair<std::string, std::any>>{{kAttrCoreNum, Index(1, span)}}, call->GetType(), span);
  }

  // Each Lowerer instance owns one CHIP invocation's scheduling token.
  // A shared InOut tensor conservatively orders every CCU request in that CHIP.
  // No remote access, Host parameter, communication window or ctx is needed.
  Names* names_;
  std::map<std::string, VarPtr> tokens_;
  std::vector<StmtPtr> allocations_;
  std::map<std::string, FunctionPtr>* kernels_;
};

}  // namespace

ProgramPtr LowerCcuCollectives(const ProgramPtr& program) {
  Chips chips;
  Names names;
  names.VisitProgram(program);
  for (const auto& [gvar, func] : program->functions_) {
    CcuFinder finder(func);
    finder.VisitStmt(func->body_);
    if (!finder.found) continue;
    // Re-running after ctx materialization would require remapping the suffix;
    // normal reruns are a no-op because public CCU ops have already disappeared.
    CHECK_SPAN(std::none_of(func->params_.begin(), func->params_.end(),
                            [](const auto& param) { return As<CommCtxType>(param->GetType()) != nullptr; }),
               func->span_)
        << "LowerL2TensorCollectives must run before MaterializeDistTensorCtx for CCU allgather";
    chips.insert(func->name_);
  }
  if (chips.empty()) return program;

  std::map<std::string, FunctionPtr> kernels;
  auto result = MutableCopy(program);
  for (const auto& [gvar, func] : program->functions_) {
    auto updated = MutableCopy(func);
    if (chips.count(func->name_)) {
      CcuLowerer lowerer(&names, &kernels);
      updated->body_ = lowerer.Lower(func->body_);
    }
    result->functions_[gvar] = updated;
  }
  for (const auto& [name, kernel] : kernels) {
    auto gvar = std::make_shared<GlobalVar>(name);
    CHECK_SPAN(!result->functions_.count(gvar), kernel->span_)
        << "CCU builtin name collides with existing function: " << name;
    result->functions_.emplace(gvar, kernel);
  }
  return result;
}

}  // namespace pypto::ir
