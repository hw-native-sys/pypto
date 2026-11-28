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

#include <any>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/inline_functions.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"

namespace pypto::ir {
namespace {

class GraphLowerer : public IRMutator {
 public:
  explicit GraphLowerer(const ProgramPtr& program) {
    for (const auto& [global, function] : program->functions_) {
      if (function->func_type_ == FunctionType::Graph) graph_names_.insert(global->name_);
      if (function->func_type_ == FunctionType::Inline) inline_names_.insert(global->name_);
    }
  }

  StmtPtr LowerBody(const FunctionPtr& function) {
    inline_caller_ = function->func_type_ == FunctionType::Inline ? &function->name_ : nullptr;
    return function->body_ ? VisitStmt(function->body_) : function->body_;
  }

  void CheckManualCallSites() {
    if (manual_inline_calls_.empty()) return;
    // Existing Inline helpers can contain a Graph marker. Propagate that fact
    // backwards through Inline calls so we see the final caller's scope mode.
    while (!pending_graph_helpers_.empty()) {
      auto callers = inline_callers_.find(pending_graph_helpers_.front());
      pending_graph_helpers_.pop_front();
      if (callers == inline_callers_.end()) continue;
      for (const auto& caller : callers->second) MarkGraphHelper(&caller);
    }
    for (const auto& call : manual_inline_calls_) {
      CheckEnclosingManualScope(inline_graph_helpers_.count(call->op_->name_) != 0, call->span_);
    }
  }

 protected:
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    const bool was_manual = in_manual_scope_;
    in_manual_scope_ |= op->manual_;
    auto body = IRMutator::VisitStmt_(op);
    in_manual_scope_ = was_manual;
    return body;
  }

  StmtPtr VisitStmt_(const GraphScopeStmtPtr& op) override {
    CheckEnclosingManualScope(in_manual_scope_, op->span_);
    MarkGraphHelper(inline_caller_);
    for (const auto& [key, value] : op->attrs_) {
      CHECK_SPAN(key == kAttrDumpVars || key == kAttrCalendarHash, op->span_)
          << "TMR cannot lower Graph scope '" << op->name_hint_ << "' with boundary attribute '" << key
          << "'. Put dependencies and completion captures on the individual tasks, or target "
             "host_build_graph.";
    }
    // Keep the original Var identities. A lexical Graph scope can define
    // tensors that the surrounding orchestration uses after the region.
    auto body = VisitStmt(op->body_);
    return transform_utils::TransferDispatchDumpVars(body, op->GetAttr<std::vector<VarPtr>>(kAttrDumpVars));
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    if (graph_names_.empty() && inline_names_.empty()) return op;
    bool strip_calendar_hash = false;
    if (auto callee = GraphCallee(op->op_)) {
      CheckEnclosingManualScope(in_manual_scope_, op->span_);
      MarkGraphHelper(inline_caller_);
      // InlineFunctions transfers dump selections to the kernel dispatches;
      // calendar_hash is an A5 Calendar PGO hint and is dropped when TMR inlines
      // the Graph body. Other call-site metadata would disappear when spliced.
      for (const auto& [key, value] : op->attrs_) {
        CHECK_SPAN(key == kAttrDumpVars || key == kAttrCalendarHash, op->span_)
            << "TMR cannot lower Graph call to '" << callee->name_ << "' with boundary attribute '" << key
            << "'. Apply dispatch options to the individual tasks, or target host_build_graph.";
        if (key == kAttrCalendarHash) strip_calendar_hash = true;
      }
      CHECK_SPAN(op->kwargs_.empty(), op->span_)
          << "TMR cannot lower Graph call to '" << callee->name_ << "' with keyword metadata.";
    } else if (auto target = As<GlobalVar>(op->op_); target && inline_names_.count(target->name_)) {
      if (inline_caller_) inline_callers_[target->name_].push_back(*inline_caller_);
      if (in_manual_scope_) manual_inline_calls_.push_back(op);
    }
    auto visited = IRMutator::VisitExpr_(op);
    if (!strip_calendar_hash) return visited;
    auto call = As<Call>(visited);
    if (!call) return visited;
    std::vector<std::pair<std::string, std::any>> attrs;
    attrs.reserve(call->attrs_.size());
    for (const auto& kv : call->attrs_) {
      if (kv.first != kAttrCalendarHash) attrs.push_back(kv);
    }
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(attrs), call->GetType(),
                                  call->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    if (graph_names_.empty() && inline_names_.empty()) return op;
    // A Submit has one completion TaskId, even if the caller ignores it.
    // Submitting the constituent tasks separately cannot retain that identity,
    // or apply a launch predicate/explicit dependencies to the region as a unit.
    if (auto callee = GraphCallee(op->op_)) {
      CHECK_SPAN(false, op->span_)
          << "TMR cannot lower Graph submit to '" << callee->name_
          << "'. Graph-level dependencies, predicates, and completion TaskIds need a Graph task. "
             "Call the Graph function directly and submit its individual tasks with any required "
             "dependencies, or target host_build_graph.";
    }
    return IRMutator::VisitExpr_(op);
  }

 private:
  static void CheckEnclosingManualScope(bool is_manual, const Span& span) {
    // HBG records the body with an independent scope stack. Inlining into a
    // TMR manual scope would turn off auto dependency inference for that body,
    // and TMR does not permit an AUTO scope inside a MANUAL scope.
    CHECK_SPAN(!is_manual, span)
        << "TMR cannot lower Graph inside an enclosing manual scope. Move the Graph outside the "
           "manual scope, or target host_build_graph.";
  }

  void MarkGraphHelper(const std::string* name) {
    if (name && inline_graph_helpers_.insert(*name).second) pending_graph_helpers_.push_back(*name);
  }

  GlobalVarPtr GraphCallee(const OpPtr& op) const {
    auto callee = As<GlobalVar>(op);
    return callee && graph_names_.count(callee->name_) ? callee : nullptr;
  }

  std::unordered_set<std::string> graph_names_;
  std::unordered_set<std::string> inline_names_;
  std::unordered_map<std::string, std::vector<std::string>> inline_callers_;
  std::unordered_set<std::string> inline_graph_helpers_;
  std::deque<std::string> pending_graph_helpers_;
  std::vector<CallPtr> manual_inline_calls_;
  const std::string* inline_caller_ = nullptr;
  bool in_manual_scope_ = false;
};

ProgramPtr LowerGraphProgram(const ProgramPtr& program) {
  const auto* ctx = PassContext::Current();
  const auto runtime = ctx ? ctx->GetRuntime() : kDefaultRuntimeKind;
  if (runtime != RuntimeKind::TensorMapAndRingBuffer) return program;

  GraphLowerer lowerer(program);
  std::shared_ptr<Program> result;
  for (const auto& [global, function] : program->functions_) {
    auto body = lowerer.LowerBody(function);
    const bool is_graph = function->func_type_ == FunctionType::Graph;
    if (!is_graph && body == function->body_) continue;

    auto replacement = MutableCopy(function);
    replacement->body_ = std::move(body);
    if (is_graph) {
      // These describe the Graph dispatch boundary. The inlined body inherits
      // the placement of its caller, while its kernel calls keep their own placement.
      replacement->func_type_ = FunctionType::Inline;
      replacement->level_.reset();
      replacement->role_.reset();
    }
    if (!result) result = MutableCopy(program);
    result->functions_.at(global) = std::move(replacement);
  }
  lowerer.CheckManualCallSites();
  return result ? ProgramPtr(result) : program;
}

}  // namespace

namespace pass {

Pass LowerGraphForRuntime() {
  return CreateProgramPass(LowerGraphProgram, "LowerGraphForRuntime", kLowerGraphForRuntimeProperties);
}

}  // namespace pass
}  // namespace pypto::ir
