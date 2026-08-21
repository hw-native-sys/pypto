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
 * @file verify_param_directions_sound.cpp
 * @brief Reject a parameter declared `In` that its own function body writes.
 *
 * This is the safety net direction inference never had. Every pass that derives
 * directions builds a write set, and until an operator declares what it does to
 * its arguments, an operator missing from that set reads as a pure consumer:
 * the write disappears, the parameter stays `In`, no RAW edge is emitted
 * against it, and the failure surfaces on device as a race or a scheduler
 * deadlock rather than at compile time. `pld.system.notify` reached production
 * that way, and `tile.mscatter` was still in that state when this check was
 * written.
 *
 * It runs on the **finished program**, registered as a `PostPipeline` warning
 * (`DiagnosticCheck::ParamDirectionsUnsound`) rather than after any one pass: a
 * Group/Spmd wrapper's signature legitimately reads `In` for a parameter its
 * inner kernel writes until `DeriveCallDirections` phase 0 materialises the
 * effective directions back into the IR. The registry stamps the registered
 * severity onto whatever this emits, so the same verifier is a hard error when
 * a caller requests `IRProperty::ParamDirectionsSound` directly — which is the
 * promotion path once the residual report is empty.
 *
 * It never invents a write. Both sources of evidence are declarations the IR
 * already carries — an operator's registry effects for a builtin, and the
 * callee's own `param_directions_` for a cross-function call — and a variable
 * whose owning buffer control flow leaves ambiguous is skipped rather than
 * blamed. A false positive here would block a compilation that is fine; a
 * false negative only restores today's behaviour. For the same reason a write
 * reaching its parameter through a `tensor.slice` is not reported:
 * `BufferRootCollector` treats a slice as a fresh root, and overriding that
 * here would mix two alias models rather than unify them.
 */

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/buffer_root_collector.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::ir::buffer_root::AmbiguousRootPolicy;
using ::pypto::ir::buffer_root::BufferRootCollector;
using ::pypto::ir::outline_utils::CallWriteTargets;

/// Report every write that reaches a parameter its function declares `In`.
class InParamWriteFinder : public IRVisitor {
 public:
  InParamWriteFinder(const ProgramPtr& program, const FunctionPtr& func,
                     const std::unordered_map<const Var*, const Var*>& buffer_roots,
                     const std::unordered_set<const Var*>& ambiguous, std::vector<Diagnostic>& diagnostics)
      : program_(program),
        func_(func),
        buffer_roots_(buffer_roots),
        ambiguous_(ambiguous),
        diagnostics_(diagnostics) {
    for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i) {
      if (func->param_directions_[i] != ParamDirection::In) continue;
      // Only a buffer can be written through. A scalar parameter is passed by
      // value and its direction carries no aliasing claim.
      if (!AsTensorTypeLike(func->params_[i]->GetType())) continue;
      in_params_.emplace(func->params_[i].get(), func->params_[i]);
    }
  }

  [[nodiscard]] bool HasInParams() const { return !in_params_.empty(); }

 protected:
  void VisitExpr_(const CallPtr& call) override {
    CheckCallLike(call, call->span_);
    IRVisitor::VisitExpr_(call);
  }

  /// A task launch writes its callee's Out/InOut parameters just as a call
  /// does; the base visitor does not forward Submit to the Call handler.
  void VisitExpr_(const SubmitPtr& submit) override {
    CheckCallLike(SubmitToCallView(submit), submit->span_);
    IRVisitor::VisitExpr_(submit);
  }

 private:
  void CheckCallLike(const CallPtr& call, const Span& span) {
    // Builtin operator: the registry says which arguments it writes.
    for (const auto& target : CallWriteTargets(call)) {
      Report(target.var, call->op_->name_, span);
    }

    // Cross-function call: the callee's signature says which of its parameters
    // it writes, and an argument in such a slot is written by this body too.
    auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!gvar || !program_) return;
    auto callee = program_->GetFunction(gvar->name_);
    if (!callee) return;
    // Submit carries a positional prefix of the callee's parameters; both kinds
    // map args_[i] to params_[i] identically over the args they do carry.
    for (size_t i = 0; i < call->args_.size() && i < callee->param_directions_.size(); ++i) {
      if (callee->param_directions_[i] == ParamDirection::In) continue;
      if (auto var = AsVarLike(call->args_[i])) {
        Report(var, gvar->name_, span);
      }
    }
  }

  void Report(const VarPtr& written, const std::string& writer, const Span& span) {
    // Resolve the written variable to the buffer it owns, so a write through a
    // slice or a loop-carried alias is attributed to the parameter behind it.
    if (ambiguous_.count(written.get()) > 0) return;
    auto root_it = buffer_roots_.find(written.get());
    const Var* root = root_it == buffer_roots_.end() ? written.get() : root_it->second;

    auto param_it = in_params_.find(root);
    if (param_it == in_params_.end()) return;
    if (!reported_.insert(root).second) return;

    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "ParamDirectionsSound", 0,
        "parameter '" + param_it->second->name_hint_ + "' of function '" + func_->name_ +
            "' is declared In but is written by '" + writer +
            "'. A written parameter read as an input drops the dependency edge its writer needs, "
            "so the program races or deadlocks on device instead of failing here. If the operator "
            "really does write it, declare that write with .set_arg_effect(...) in its REGISTER_OP "
            "block so direction inference can see it; if the parameter really is an output, declare "
            "it pl.Out or pl.InOut",
        span);
  }

  const ProgramPtr& program_;
  const FunctionPtr& func_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  const std::unordered_set<const Var*>& ambiguous_;
  std::vector<Diagnostic>& diagnostics_;
  std::unordered_map<const Var*, VarPtr> in_params_;
  /// One diagnostic per parameter: a loop writing it every iteration is one bug.
  std::unordered_set<const Var*> reported_;
};

}  // namespace

class ParamDirectionsSoundPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "ParamDirectionsSound"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Only signatures the compiler derived. An Orchestration function is the
      // program's entry: its directions are the user's declaration and its
      // parameters are the host ABI, so flipping one is a migration the user
      // makes, not an inference the compiler completes. Under-declaring there
      // is worth a warning (a written buffer the host is never told to copy
      // back), but it is a different diagnostic with a different audience — and
      // making it an error here would reject working programs.
      if (func->func_type_ == FunctionType::Orchestration) continue;

      BufferRootCollector roots(program, AmbiguousRootPolicy::kSkip);
      roots.Initialize(func->params_);
      roots.VisitStmt(func->body_);

      InParamWriteFinder finder(program, func, roots.buffer_roots, roots.ambiguous_buffer_vars, diagnostics);
      if (!finder.HasInParams()) continue;
      finder.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateParamDirectionsSoundPropertyVerifier() {
  return std::make_shared<ParamDirectionsSoundPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
