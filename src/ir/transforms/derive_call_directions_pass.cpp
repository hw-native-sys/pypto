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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::codegen::BufferRootCollector;
using ::pypto::codegen::ComputeGroupEffectiveDirections;
using ::pypto::codegen::IsBuiltinOp;

/// Decide whether an argument expression refers to a tensor (not a scalar/index).
bool IsTensorTypedArg(const ExprPtr& arg) {
  TypePtr ty = arg ? arg->GetType() : TypePtr{};
  if (!ty) return false;
  if (As<TensorType>(ty)) return true;
  if (As<TupleType>(ty)) return true;
  return false;
}

/// IRMutator that rewrites every non-builtin Call in a function body and writes
/// the per-argument ArgDirection vector based on callee param directions and the
/// pre-computed buffer-root map for that function.
class CallDirectionMutator : public IRMutator {
 public:
  CallDirectionMutator(ProgramPtr program, const std::unordered_map<const Var*, const Var*>& buffer_roots,
                       const std::unordered_set<const Var*>& param_vars)
      : program_(std::move(program)), buffer_roots_(buffer_roots), param_vars_(param_vars) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    // First descend so nested Calls also get arg_directions assigned.
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    if (!call) return base;

    if (IsBuiltinOp(call->op_->name_)) {
      return call;
    }

    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) {
      // Unknown op (e.g. Opaque function not in program). Leave directions empty.
      return call;
    }

    std::vector<ParamDirection> effective = callee->param_directions_;
    if (callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd) {
      effective = ComputeGroupEffectiveDirections(callee, program_);
    }

    if (effective.size() != call->args_.size()) {
      // Safety: if the length disagrees we can't produce a sound mapping.
      // Leave directions empty so the verify pass surfaces a clear error.
      return call;
    }

    std::vector<ArgDirection> dirs;
    dirs.reserve(call->args_.size());
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto& arg = call->args_[i];
      bool is_tensor = IsTensorTypedArg(arg);
      if (!is_tensor) {
        dirs.push_back(ArgDirection::Scalar);
        continue;
      }

      ParamDirection cd = effective[i];
      if (cd == ParamDirection::In) {
        dirs.push_back(ArgDirection::Input);
      } else if (cd == ParamDirection::InOut) {
        dirs.push_back(ArgDirection::InOut);
      } else {
        // ParamDirection::Out
        if (auto arg_var = AsVarLike(arg)) {
          if (IsLocallyAllocated(arg_var.get())) {
            // WAW promotion: the runtime needs InOut to chain dependencies on
            // a pre-allocated local buffer being reused across tasks.
            dirs.push_back(ArgDirection::InOut);
          } else {
            // External (param-rooted) buffer: treat as write-only into an existing tensor.
            dirs.push_back(ArgDirection::OutputExisting);
          }
        } else {
          // Non-var Out argument is unusual; fall back to OutputExisting which is
          // the conservative choice (no allocation done by the runtime).
          dirs.push_back(ArgDirection::OutputExisting);
        }
      }
    }

    // Skip rewriting if directions are unchanged.
    if (call->GetArgDirections() == dirs) {
      return call;
    }

    auto new_attrs = WithArgDirectionsAttr(call->attrs_, std::move(dirs));
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

 private:
  bool IsLocallyAllocated(const Var* var) const {
    auto it = buffer_roots_.find(var);
    if (it == buffer_roots_.end()) return false;
    const Var* root = it->second;
    return param_vars_.count(root) == 0;
  }

  ProgramPtr program_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  const std::unordered_set<const Var*>& param_vars_;
};

}  // namespace

namespace pass {

Pass DeriveCallDirections() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;

    // We need a non-const handle to rewrite functions with new bodies.
    auto new_functions = program->functions_;

    for (auto& [gvar, func] : new_functions) {
      if (!func || !func->body_) continue;

      BufferRootCollector br_collector(program);
      br_collector.Initialize(func->params_);
      br_collector.VisitStmt(func->body_);

      std::unordered_set<const Var*> param_vars;
      param_vars.reserve(func->params_.size());
      for (const auto& p : func->params_) {
        param_vars.insert(p.get());
      }

      CallDirectionMutator mutator(program, br_collector.buffer_roots, param_vars);
      auto new_body = mutator.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;

      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }

    if (new_functions == program->functions_) {
      return program;
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "DeriveCallDirections", kDeriveCallDirectionsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
