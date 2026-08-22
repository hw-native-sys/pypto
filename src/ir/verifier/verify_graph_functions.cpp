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
 * @file verify_graph_functions.cpp
 * @brief Verifies the GraphBoundaryLegalized property.
 *
 * `LegalizeGraphBoundary` rejects illegal graphs as it rewrites them. This
 * verifier re-states the resulting invariants over the whole program, so a later
 * pass that reintroduces a violation is caught rather than silently producing a
 * program the runtime declines to record.
 *
 * That matters more here than for a typical property. Almost every
 * host_build_graph constraint degrades to a *silent* non-graph fallback in a
 * release build: the program stays numerically correct and simply loses the
 * speedup, which no correctness test can detect. This verifier is the automated
 * detector.
 */

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

constexpr size_t kMaxBoundaryTensors = 32;

/// Walks one function body and reports every illegal reference to a Graph.
///
/// Covers both call-like kinds. `Submit` is not a subclass of `Call` and
/// `IRVisitor` dispatches them through separate handlers, so a checker that
/// overrides only the `Call` path silently skips every `pl.submit(...)` — and a
/// Graph is launched from a manual scope precisely as a submit.
class GraphReferenceChecker : public IRVisitor {
 public:
  GraphReferenceChecker(ProgramPtr program, FunctionPtr caller, std::vector<Diagnostic>& diagnostics)
      : program_(std::move(program)), caller_(std::move(caller)), diagnostics_(diagnostics) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    auto callee = LookupGraph(op->op_);
    if (!callee) return;
    CheckCaller(callee, op->span_);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    IRVisitor::VisitExpr_(op);
    auto callee = LookupGraph(op->op_);
    if (!callee) return;
    CheckCaller(callee, op->span_);

    if (!op->deps_.empty()) {
      Report(callee, op->span_,
             "is submitted with explicit dependencies. An explicit dependency edge makes the launch "
             "uncacheable, so the region would silently run as ordinary tasks with no graph replay. "
             "Order the graph against its producers through its boundary tensors instead.");
    }
    if (op->predicate_ != nullptr) {
      Report(callee, op->span_,
             "is submitted with a dispatch predicate. The runtime neither honours nor rejects a "
             "predicate on a graph launch — it silently zeroes it — so the region would run "
             "unconditionally.");
    }
  }

 private:
  [[nodiscard]] FunctionPtr LookupGraph(const OpPtr& callee_op) const {
    auto gvar = As<GlobalVar>(callee_op);
    if (!gvar || !program_) return nullptr;
    auto callee = program_->GetFunction(gvar->name_);
    if (!callee || callee->func_type_ != FunctionType::Graph) return nullptr;
    return callee;
  }

  void CheckCaller(const FunctionPtr& callee, const Span& span) const {
    // Opaque is accepted alongside Orchestration: an entry function is Opaque
    // until OutlineIncoreScopes promotes it, and this verifier also runs on IR
    // that has not reached that point.
    if (caller_->func_type_ == FunctionType::Orchestration || caller_->func_type_ == FunctionType::Opaque) {
      return;
    }
    if (caller_->func_type_ == FunctionType::Graph) {
      Report(callee, span,
             "is called from another Graph function. The runtime cannot record a graph from inside "
             "one it is already recording, so the whole region becomes uncacheable.");
      return;
    }
    Report(callee, span,
           "is called from a '" + FunctionTypeToString(caller_->func_type_) +
               "' function. A graph is a task launch, so only an Orchestration entry may call it.");
  }

  void Report(const FunctionPtr& callee, const Span& span, const std::string& what) const {
    std::ostringstream oss;
    oss << "Graph function '" << callee->name_ << "', referenced from '" << caller_->name_ << "', " << what;
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "GraphBoundaryLegalized", 0, oss.str(), span);
  }

  ProgramPtr program_;
  FunctionPtr caller_;
  std::vector<Diagnostic>& diagnostics_;
};

/// Re-states the signature half of the boundary contract.
void VerifySignature(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
  auto report = [&](const std::string& message, const Span& span) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "GraphBoundaryLegalized", 0,
                             "Graph function '" + func->name_ + "' " + message, span);
  };

  size_t tensor_params = 0;
  for (size_t i = 0; i < func->params_.size(); ++i) {
    const auto& param = func->params_[i];
    const auto dir = func->param_directions_[i];
    if (As<ScalarType>(param->GetType()) != nullptr) {
      if (dir != ParamDirection::In) {
        report("declares scalar parameter '" + param->name_hint_ +
                   "' as an output. A boundary scalar is passed by value and replayed from the call "
                   "site, so it can only be an input.",
               param->span_);
      }
      continue;
    }
    ++tensor_params;
    if (dir == ParamDirection::Out) {
      report("declares tensor parameter '" + param->name_hint_ +
                 "' as Out, meaning the runtime allocates it. A recorded graph's boundary tensors "
                 "must already exist so replay can patch their addresses.",
             param->span_);
    }
  }

  if (tensor_params == 0) {
    report(
        "takes no tensor parameters. A graph with an empty boundary has nothing to patch on replay "
        "and the runtime refuses to cache it.",
        func->span_);
  }
  if (tensor_params > kMaxBoundaryTensors) {
    report("takes " + std::to_string(tensor_params) +
               " tensor parameters, over the runtime's boundary "
               "limit of " +
               std::to_string(kMaxBoundaryTensors) + ".",
           func->span_);
  }
}

class GraphBoundaryLegalizedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "GraphBoundaryLegalized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func) continue;
      if (func->func_type_ == FunctionType::Graph) VerifySignature(func, diagnostics);
      if (!func->body_) continue;
      GraphReferenceChecker checker(program, func, diagnostics);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateGraphBoundaryLegalizedPropertyVerifier() {
  return std::make_shared<GraphBoundaryLegalizedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
