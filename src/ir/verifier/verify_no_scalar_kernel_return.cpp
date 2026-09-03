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
 * @file verify_no_scalar_kernel_return.cpp
 * @brief Verifier that a device kernel never returns a scalar value.
 *
 * The runtime has two disjoint task-argument channels: ``Arg::add_scalar``
 * passes a scalar *in* by value, and ``TaskOutputTensors`` hands back only
 * ChipTensors. There is no scalar output channel, so a ``ScalarType`` in a
 * device function's ``return_types_`` is unrepresentable — orchestration
 * codegen has nothing to bind it to. Such a return used to reach the generated
 * C++ as an undefined identifier, and later as a silently wrong ``= 0``
 * (issue #631).
 *
 * The check is on the *function*, not on its call sites, because
 * ``FunctionType::InCore`` / ``AIC`` / ``AIV`` / ``Group`` / ``Spmd`` *means*
 * "a dispatchable task" in this IR: there is no device-side helper function
 * kind. A helper meant to run inside a kernel is written ``FunctionType::Inline``
 * and spliced by ``InlineFunctions`` long before codegen — a non-Inline callee
 * inside a scope body has no lowering at all, independent of what it returns.
 * Checking the declaration also catches a launch its caller does not yet
 * advertise: an ``Opaque`` parent is only promoted to ``Orchestration`` at
 * ``OutlineIncoreScopes``, so a call-site rule would miss every launch written
 * before that pass.
 *
 * ``Scalar[TASK_ID]`` is rejected like any other scalar, because a declaration
 * is not where a TaskId ever legitimately appears. The handle is attached to
 * the *call*: ``pl.submit`` and the outliner append the trailing element to the
 * Submit's tuple type and bind it at the call site from
 * ``task_<n>_outs.task_id()``, leaving the callee's own ``return_types_``
 * untouched (see ``ScopeOutliner``, which builds the outlined Function from
 * ``return_types`` and only the call from ``effective_return_types``). A device
 * function that nonetheless declares one is not merely uncarried but actively
 * misread: orchestration codegen's ``IsSubmitCall`` keys on "TupleType whose
 * last element is ``Scalar[TASK_ID]``", so an ordinary call to such a callee is
 * lowered as a task launch.
 *
 * The property is decidable on the user's own IR, so it sits in
 * ``GetStructuralProperties()`` and is verified at every pass boundary: it
 * rejects a user-written signature at pipeline input, and catches any pass that
 * synthesises one. ``OutlineIncoreScopes`` upholds it by hoisting
 * caller-computable scalars out of the scope body and rejecting the rest.
 */

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Function types whose body runs on an AICore and is dispatched as a runtime
/// task: the InCore variants plus the Group / Spmd scope wrappers.
/// Orchestration / Graph / Inline / Opaque functions are ordinary callables and
/// may return scalars.
bool IsDeviceFunctionType(FunctionType type) { return IsInCoreType(type) || IsWrapperType(type); }

/// Finds the first unsupported ScalarType reachable from ``type``, writing the
/// path that reached it (e.g. ``"#0 element #1"``) to ``found_path``.
///
/// A ``-> pl.Tuple[T1, ..., TN]`` annotation is ONE ``return_types_`` entry
/// holding a TupleType, so a nested Scalar needs the same rejection as a
/// top-level one -- the runtime has no more of a carrier for a tuple element
/// than for a bare return. Same recursion as ArrayNotEscaped's
/// TypeContainsArray; Tensor / Tile elements cannot carry a ScalarType, so
/// TupleType is the only container worth descending into.
///
/// Every ScalarType is unsupported at every depth, ``TASK_ID`` included -- a
/// TaskId rides the call node, never the callee's declaration; see the file
/// comment. ``is_task_id`` reports which of the two it was, so the diagnostic
/// can name the right fix.
bool FindUnsupportedScalar(const TypePtr& type, const std::string& path, std::string* found_path,
                           bool* is_task_id) {
  if (!type) return false;
  if (auto scalar_type = As<ScalarType>(type)) {
    *found_path = path;
    *is_task_id = scalar_type->dtype_ == DataType::TASK_ID;
    return true;
  }
  if (auto tuple_type = As<TupleType>(type)) {
    for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
      if (FindUnsupportedScalar(tuple_type->types_[i], path + " element #" + std::to_string(i), found_path,
                                is_task_id)) {
        return true;
      }
    }
  }
  return false;
}

class NoScalarKernelReturnVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "NoScalarKernelReturn"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func || !IsDeviceFunctionType(func->func_type_)) continue;
      for (size_t i = 0; i < func->return_types_.size(); ++i) {
        std::string path;
        bool is_task_id = false;
        if (!FindUnsupportedScalar(func->return_types_[i], "#" + std::to_string(i), &path, &is_task_id)) {
          continue;
        }
        std::ostringstream msg;
        msg << "Device function '" << func->name_ << "' return type " << path << " is a Scalar"
            << (is_task_id ? "[TASK_ID]" : "") << ". A task cannot return a scalar: the runtime "
            << "passes scalars in by value and returns only tensors, so there is no carrier for "
            << "this value. ";
        if (is_task_id) {
          msg << "A TaskId is produced by the launch, not by the callee: capture it at the call "
                 "site with 'out, tid = pl.submit(kernel, ...)' or 'with pl.at(...) as tid', "
                 "which appends the handle to the call's tuple and leaves this signature alone. "
                 "Declaring one here also makes orchestration codegen read an ordinary call to "
                 "this function as a task launch.";
        } else {
          msg << "Write it into a [1] tensor output inside the kernel and read it back after the "
                 "launch with pl.tensor.read(t, [0]).";
        }
        diagnostics.emplace_back(DiagnosticSeverity::Error, "NoScalarKernelReturn",
                                 /*error_code=*/0, msg.str(), func->span_);
      }
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateNoScalarKernelReturnPropertyVerifier() {
  return std::make_shared<NoScalarKernelReturnVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
