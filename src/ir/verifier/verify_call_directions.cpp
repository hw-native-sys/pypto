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
  if (As<TensorType>(ty)) return true;
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
    if (IsBuiltinOp(call->op_->name_)) return;

    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) return;  // Opaque / not in program — skip.

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
      if (cd == ParamDirection::In && is_tensor) {
        // Allow Input (default) or NoDep (caller-site override via pl.no_dep).
        // Both are read-only at the runtime level; NoDep additionally skips
        // OverlapMap dep tracking. NoDep is forbidden on Out/InOut params
        // because it would suppress producer registration for a writer.
        if (d != ArgDirection::Input && d != ArgDirection::NoDep) {
          std::ostringstream oss;
          oss << "tensor argument at index " << i << " has " << ArgDirectionToString(d)
              << " but callee param direction is In";
          Fail(call, oss.str());
          return;
        }
      } else if (cd == ParamDirection::InOut && is_tensor) {
        if (d != ArgDirection::InOut) {
          std::ostringstream oss;
          oss << "tensor argument at index " << i << " has " << ArgDirectionToString(d)
              << " but callee param direction is InOut";
          Fail(call, oss.str());
          return;
        }
      } else if (cd == ParamDirection::Out && is_tensor) {
        // Allowed: Output / OutputExisting / InOut (WAW promotion).
        if (d != ArgDirection::Output && d != ArgDirection::OutputExisting && d != ArgDirection::InOut) {
          std::ostringstream oss;
          oss << "tensor argument at index " << i << " has " << ArgDirectionToString(d)
              << " but callee param direction is Out";
          Fail(call, oss.str());
          return;
        }
      }
    }
  }

 private:
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
