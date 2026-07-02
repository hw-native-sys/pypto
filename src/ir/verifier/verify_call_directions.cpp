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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

using ::pypto::codegen::ComputeGroupEffectiveDirections;
using ::pypto::codegen::IsBuiltinOp;

bool IsTensorTypedArg(const ExprPtr& arg) {
  TypePtr ty = arg ? arg->GetType() : TypePtr{};
  if (!ty) return false;
  if (AsTensorTypeLike(ty)) return true;
  if (As<TupleType>(ty)) return true;
  return false;
}

/// Walks every non-builtin Call in a function body and validates the integrity
/// of ``Call::GetArgDirections()`` (stored in ``attrs_["arg_directions"]``)
/// against the callee's ``param_directions_``.
class CallDirectionChecker : public IRVisitor {
 public:
  CallDirectionChecker(ProgramPtr program, std::vector<Diagnostic>& diagnostics, std::string func_name)
      : program_(std::move(program)), diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

 protected:
  void VisitExpr_(const CallPtr& call) override {
    IRVisitor::VisitExpr_(call);
    CheckCallLike(call);
  }

  // A Submit (task launch) carries arg_directions in the same attr as a Call.
  // Validate it through the Call-shaped view so CallDirectionsResolved covers
  // submits too (pass-submit-awareness.md rule 6).
  void VisitExpr_(const SubmitPtr& submit) override {
    IRVisitor::VisitExpr_(submit);
    CheckCallLike(SubmitToCallView(submit));
  }

 private:
  void CheckCallLike(const CallPtr& call) {
    if (IsBuiltinOp(call->op_->name_)) return;

    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) return;  // Opaque / not in program — skip.

    // A 0-arg call/submit has no per-arg directions to verify. DeriveCallDirections
    // derives an empty direction vector for it, which is correct — not "missing"
    // or "empty as a failure". (A bare ``pl.submit(self.kernel)`` whose callee
    // takes no positional tensor args is legal.)
    if (call->args_.empty()) return;

    if (!call->HasArgDirections()) {
      Fail(call, "Call attrs['arg_directions'] is missing after DeriveCallDirections");
      return;
    }
    auto arg_dirs = call->GetArgDirections();
    if (arg_dirs.empty()) {
      Fail(call, "Call attrs['arg_directions'] is empty after DeriveCallDirections");
      return;
    }
    if (arg_dirs.size() != call->args_.size()) {
      std::ostringstream oss;
      oss << "Call attrs['arg_directions'] size (" << arg_dirs.size() << ") != args_ size ("
          << call->args_.size() << ")";
      Fail(call, oss.str());
      return;
    }

    std::vector<ParamDirection> effective = callee->param_directions_;
    if (callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd) {
      effective = ComputeGroupEffectiveDirections(callee, program_);
    }

    for (size_t i = 0; i < call->args_.size(); ++i) {
      ArgDirection d = arg_dirs[i];
      bool is_tensor = IsTensorTypedArg(call->args_[i]);

      // 1) Scalar / tensor consistency
      if (is_tensor && d == ArgDirection::Scalar) {
        std::ostringstream oss;
        oss << "tensor argument at index " << i << " has ArgDirection::Scalar";
        Fail(call, oss.str());
        return;
      }
      if (!is_tensor && d != ArgDirection::Scalar) {
        std::ostringstream oss;
        oss << "non-tensor argument at index " << i << " has " << ArgDirectionToString(d)
            << " (expected Scalar)";
        Fail(call, oss.str());
        return;
      }

      // NOTE: We deliberately do NOT enforce a "tensors precede scalars" order
      // here. The IR Call preserves the user's parameter order (e.g.
      // ``kernel(t_in, 1.0, t_out)`` is a legitimate signature). The runtime
      // requirement that ``add_input/add_output`` come before ``add_scalar`` is
      // satisfied by orchestration codegen via ``std::stable_partition`` over
      // ``ParamEntry`` (see orchestration_codegen.cpp), so the call site itself
      // is free to interleave tensors and scalars.

      if (i >= effective.size()) continue;
      ParamDirection cd = effective[i];
      // ``NoDep`` is a user opt-out of OverlapMap tracking — the runtime
      // performs no producer lookup *and* no producer insert for the slot.
      // It is therefore legal regardless of whether the callee declares the
      // param as read-only (In) or as a writer (Out / InOut): when the user
      // can prove (out-of-band) that the writes are disjoint across siblings
      // — e.g. the paged-attention pattern of writing slot ``slot_mapping[b]``
      // per batch, where the offset is data-dependent and the compiler can
      // therefore not prove disjointness — they may opt the slot out of
      // auto-dep tracking via ``pl.no_dep(t)`` or
      // ``pl.at(no_dep_args=[t])``. The verifier accepts NoDep on every
      // tensor slot and leaves correctness to the user.
      if (cd == ParamDirection::In && is_tensor) {
        // Allow Input (default) or NoDep (caller-site override via pl.no_dep).
        if (d != ArgDirection::Input && d != ArgDirection::NoDep) {
          std::ostringstream oss;
          oss << "tensor argument at index " << i << " has " << ArgDirectionToString(d)
              << " but callee param direction is In";
          Fail(call, oss.str());
          return;
        }
      } else if (cd == ParamDirection::InOut && is_tensor) {
        // Allowed: InOut (default), OutputExisting (auto-deps rewrite when
        // explicit deps cover the read-side ordering), or NoDep
        // (caller-site override).
        if (d != ArgDirection::InOut && d != ArgDirection::OutputExisting && d != ArgDirection::NoDep) {
          std::ostringstream oss;
          oss << "tensor argument at index " << i << " has " << ArgDirectionToString(d)
              << " but callee param direction is InOut";
          Fail(call, oss.str());
          return;
        }
      } else if (cd == ParamDirection::Out && is_tensor) {
        // Allowed: Output / OutputExisting / InOut (WAW promotion) / NoDep
        // (caller-site override).
        if (d != ArgDirection::Output && d != ArgDirection::OutputExisting && d != ArgDirection::InOut &&
            d != ArgDirection::NoDep) {
          std::ostringstream oss;
          oss << "tensor argument at index " << i << " has " << ArgDirectionToString(d)
              << " but callee param direction is Out";
          Fail(call, oss.str());
          return;
        }
      }
    }
  }

  void Fail(const CallPtr& call, const std::string& msg) {
    std::ostringstream oss;
    oss << "in function '" << func_name_ << "', call to '" << call->op_->name_ << "': " << msg;
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "CallDirectionsResolved", 0, oss.str(), call->span_);
  }

  ProgramPtr program_;
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

class CallDirectionsResolvedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "CallDirectionsResolved"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      CallDirectionChecker checker(program, diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateCallDirectionsResolvedPropertyVerifier() {
  return std::make_shared<CallDirectionsResolvedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
