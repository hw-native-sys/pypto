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
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace out_param {
std::string ErrorTypeToString(ErrorType type) {
  switch (type) {
    case ErrorType::OUT_PARAM_REASSIGNED:
      return "OUT_PARAM_REASSIGNED";
    default:
      return "UNKNOWN";
  }
}
}  // namespace out_param

namespace {

/// Local builtin-op classifier (string-prefix). Verifiers live in the IR layer
/// and must not depend on `pypto::codegen::IsBuiltinOp`; this mirrors the copy in
/// verify_orchestration_references.cpp. If a fourth reader appears, promote it to
/// a shared IR utility.
bool IsBuiltinOpName(const std::string& name) {
  return name.rfind("tile.", 0) == 0 || name.rfind("tensor.", 0) == 0 || name.rfind("system.", 0) == 0 ||
         name.rfind("array.", 0) == 0;
}

/// True when `param` appears anywhere in the subtree of argument `idx`.
bool ArgSubtreeReferences(const CallPtr& call, size_t idx, const Var* param) {
  if (idx >= call->args_.size() || !call->args_[idx]) return false;
  var_collectors::VarDefUseCollector uses;
  uses.VisitExpr(call->args_[idx]);
  return uses.var_uses.count(param) > 0;
}

/// True when the reassignment `param = call(...)` keeps `param` aliased to its
/// external buffer — i.e. the call's *result* is `param`, not a fresh value.
///
/// This mirrors `TraceReturnedToParam` in the orchestration codegen so the
/// verifier flags exactly the reassignments codegen would fail to bind back to
/// the Out buffer (and allows exactly those it binds):
///   - Builtin output-side ops alias their **target** argument only:
///       tensor.assemble(target, tile, off)      -> args[0]
///       tile.store(value, indices, target)      -> args[2]
///       tensor.set_validshape(target, rows, col)-> args[0]
///   - Any other builtin op (tensor.add, tensor.matmul, ...) produces a fresh
///     value: the param is never written even when it appears as an input
///     operand (e.g. `out = pl.add(out, x)` — issue #1525, second form).
///   - A user-function call may write the param through an Out/InOut argument,
///     so the param appearing anywhere in the args counts as threading
///     (keeps `out = self.kernel(a, out)` valid).
bool CallThreadsParam(const CallPtr& call, const Var* param) {
  const std::string& op = call->op_->name_;
  if (op == "tensor.assemble" || op == "tensor.set_validshape") return ArgSubtreeReferences(call, 0, param);
  if (op == "tile.store") return ArgSubtreeReferences(call, 2, param);
  if (IsBuiltinOpName(op)) return false;

  var_collectors::VarDefUseCollector uses;
  uses.VisitExpr(call);
  return uses.var_uses.count(param) > 0;
}

/**
 * @brief Visitor that detects Out/InOut parameters reassigned to a detached value
 *
 * For each function, collects Out/InOut parameter Var pointers. Then during
 * traversal, flags any AssignStmt that reassigns one of those Var pointers with
 * a call whose *result* is a fresh value rather than the parameter itself (see
 * CallThreadsParam). Such a reassignment rebinds the name to a value
 * disconnected from the external output buffer — `out = pl.matmul(...)`,
 * `out = tensor.create(...)`, or even `out = pl.add(out, x)` (which reads the
 * param but returns a new tensor) — so the buffer is never written and the
 * kernel silently produces its init value (all-zero). See issue #1525.
 *
 * Legitimate reassignments are allowed: builtin output-side ops threading the
 * param through their target arg (`out = tensor.assemble(out, tile, offsets)`,
 * `out = tile.store(value, idx, out)`) and user-function calls that may write the
 * param through an Out/InOut argument (`out = self.kernel(a, out)`).
 */
class OutParamShadowVerifier : public IRVisitor {
 public:
  OutParamShadowVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name, FunctionPtr func)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), func_(std::move(func)) {}

  void RegisterOutParams(const std::vector<VarPtr>& params, const std::vector<ParamDirection>& directions) {
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i] && (directions[i] == ParamDirection::Out || directions[i] == ParamDirection::InOut)) {
        out_params_[params[i].get()] = directions[i];
      }
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op || !op->var_ || !op->value_) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    auto it = out_params_.find(op->var_.get());
    if (it != out_params_.end()) {
      if (auto call = As<Call>(op->value_)) {
        if (call->op_ && !CallThreadsParam(call, op->var_.get())) {
          RecordError(op->var_->name_hint_, it->second, call->op_->name_, op->span_);
        }
      }
    }

    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  FunctionPtr func_;
  std::string cached_func_str_;
  std::unordered_map<const Var*, ParamDirection> out_params_;

  void RecordError(const std::string& var_name, ParamDirection direction, const std::string& op_name,
                   const Span& span) {
    std::ostringstream msg;
    msg << "'" << var_name << "' is an " << ParamDirectionToString(direction)
        << " parameter but is reassigned to the result of '" << op_name
        << "', which does not write into it. This rebinds the name to a detached value, "
        << "so the external output buffer is never written (the kernel silently produces "
        << "its init value, e.g. all-zero — see issue #1525). Write into the parameter "
        << "instead — e.g. `" << var_name << " = pl.assemble(" << var_name << ", value, offset)`, "
        << "`pl.store(value, offset, " << var_name << ")`, or an indexed write `" << var_name
        << "[...] = value`."
        << "\n  In function '" << func_name_ << "'";
    if (func_) {
      if (cached_func_str_.empty()) {
        cached_func_str_ = PythonPrint(func_);
      }
      msg << ":\n" << cached_func_str_;
    }
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "OutParamNotShadowed",
                              static_cast<int>(out_param::ErrorType::OUT_PARAM_REASSIGNED), msg.str(), span);
  }
};

}  // namespace

/**
 * @brief OutParamNotShadowed property verifier
 */
class OutParamNotShadowedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "OutParamNotShadowed"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) {
      return;
    }

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) {
        continue;
      }

      OutParamShadowVerifier verifier(diagnostics, func->name_, func);
      verifier.RegisterOutParams(func->params_, func->param_directions_);

      if (func->body_) {
        verifier.VisitStmt(func->body_);
      }
    }
  }
};

PropertyVerifierPtr CreateOutParamNotShadowedPropertyVerifier() {
  return std::make_shared<OutParamNotShadowedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
