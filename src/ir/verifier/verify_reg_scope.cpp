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
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/hardware_loop_analysis.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/reg_scope.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir {
namespace {
// A malformed builder/deserialized literal must not prove an ancestor empty:
// codegen narrows literals to their declared dtype before executing the loop.
bool IsRepresentableLoopLiteral(const ConstIntPtr& literal) {
  if (!literal) return false;
  const auto type = As<ScalarType>(literal->GetType());
  if (!type || !(type->dtype_.IsInt() || type->dtype_ == DataType::INDEX)) return false;
  const auto bits = type->dtype_.GetBit();
  const bool is_unsigned = type->dtype_.IsUnsignedInt();
  if (is_unsigned && literal->value_ < 0) return false;
  if (bits == 0) return false;
  if (bits >= 64) return true;
  if (is_unsigned) return static_cast<uint64_t>(literal->value_) < (uint64_t{1} << bits);
  const auto limit = int64_t{1} << (bits - 1);
  return literal->value_ >= -limit && literal->value_ < limit;
}

/// Two linear walks separate ownership from uses, including uses preceding an
/// illegal definition. Maps use Var identity, never spelling: rebinding a name
/// to a Tile result must not accidentally authorize an earlier Reg escape.
class RegionChecker : public IRVisitor {
 public:
  RegionChecker(FunctionPtr function, std::vector<Diagnostic>& diagnostics, bool final)
      : function_(std::move(function)),
        diagnostics_(diagnostics),
        final_(final),
        incore_(IsInCoreType(function_->func_type_) && function_->func_type_ != FunctionType::AIC) {}

  void Run() {
    for (const auto& p : function_->params_) {
      owners_[p.get()] = nullptr;
      if (ContainsRegionValue(p->GetType(), false)) Error("Reg cannot cross a function boundary", p->span_);
    }
    for (const auto& t : function_->return_types_) {
      if (ContainsRegionValue(t, false)) {
        Error("Reg cannot cross a function return boundary", function_->span_);
      }
    }
    collecting_ = true;
    VisitStmt(function_->body_);
    collecting_ = false;
    VisitStmt(function_->body_);
  }

 protected:
  void VisitStmt_(const RegScopeStmtPtr& op) override {
    if (collecting_) {
      if (region_) Error("RegScope cannot nest", op->span_);
      if (!incore_ || blocked_ || function_->func_type_ == FunctionType::AIC) {
        Error("RegScope requires an InCore/AIV context", op->span_);
      }
      if (final_ && function_->func_type_ != FunctionType::AIV) {
        Error("RegScope final placement requires AIV after ExpandMixedKernel", op->span_);
      }
      if (backend::BackendConfig::IsConfigured() &&
          backend::BackendConfig::GetBackendType() != backend::BackendType::Ascend950) {
        Error("RegScope requires an A5 (Ascend950) target", op->span_);
      }
    }
    auto previous = region_;
    region_ = op.get();
    IRVisitor::VisitStmt_(op);
    region_ = previous;
  }
  void VisitStmt_(const InCoreScopeStmtPtr& op) override {
    bool prev = incore_;
    bool previous_blocked = blocked_;
    // Wrapper Group/Spmd functions cannot contain RegScope directly. An
    // explicit InCore scope still establishes a legal pre-outlining context.
    // Match the existing outliner's orchestration-like owners: this is a
    // device boundary even while its containing function still orchestrates.
    incore_ = IsOrchestrationLike(function_->func_type_) || function_->func_type_ == FunctionType::Opaque ||
              function_->func_type_ == FunctionType::Inline ||
              function_->func_type_ == FunctionType::InCore || function_->func_type_ == FunctionType::AIV;
    // An orchestration ancestor is outside this device region. Scopes opened
    // inside it still set blocked_ through Block(); restore the outer state
    // afterward so sibling orchestration code cannot inherit device access.
    blocked_ = false;
    IRVisitor::VisitStmt_(op);
    blocked_ = previous_blocked;
    incore_ = prev;
  }
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override { Block(op); }
  void VisitStmt_(const GraphScopeStmtPtr& op) override { Block(op); }
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override { Block(op); }
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (collecting_) {
      auto type = op->var_->GetType();
      if (ContainsRegionValue(type, true)) {
        auto [it, inserted] = owners_.emplace(op->var_.get(), region_);
        if (!inserted && it->second != region_) {
          Error("Reg/local Scalar assignment escapes RegScope ownership", op->span_);
        }
      }
    }
    if (!collecting_ &&
        (ContainsRegionValue(op->var_->GetType(), false) ||
         ContainsRegionValue(op->value_->GetType(), false)) &&
        !structural_equal(op->var_->GetType(), op->value_->GetType())) {
      Error("Register assignment dtype/kind/lane mismatch", op->span_);
    }
    if (collecting_ && region_ && As<TileType>(op->var_->GetType())) {
      auto call = As<Call>(op->value_);
      if (!IsOp(call, "reg.store") && !IsOp(call, "reg.scatter")) {
        Error(
            "RegScope Tile inputs must originate outside the region; only reg.store/reg.scatter return Tile",
            op->span_);
      }
    }
    // LHS is a definition, not a use. Only inspect the RHS here.
    VisitExpr(op->value_);
  }
  void VisitVarLike_(const VarPtr& op) override {
    if (collecting_) return;
    auto found = owners_.find(op.get());
    if (found != owners_.end() && found->second && found->second != region_) {
      Error("Reg/local Scalar cannot escape its RegScope", op->span_);
    }
    if (storing_container_ && found != owners_.end() && found->second && found->second == region_) {
      Error("Reg/local Scalar cannot escape through an external container", op->span_);
    }
    if (ContainsRegionValue(op->GetType(), false) &&
        (found == owners_.end() || !region_ || found->second != region_)) {
      Error("Reg must be defined and used within the same RegScope", op->span_);
    }
  }
  void VisitExpr_(const TupleGetItemExprPtr& op) override {
    // Codegen resolves projections from the SSA results of an assigned pair.
    // A nested call has no assignment to emit or bind those results first.
    if (collecting_ && As<RegType>(op->GetType()) && !AsVarLike(op->tuple_)) {
      Error("Register tuple projections require a bound tuple variable; bind the paired call before indexing",
            op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const CallPtr& op) override {
    if (!collecting_ && op->op_->name_.rfind("reg.", 0) == 0) {
      if (!region_) Error("reg operation requires RegScope", op->span_);
      // Raw Call construction can bypass type deduction. Revalidate the actual
      // operands and compare the supplied result rather than trusting the type.
      try {
        auto inferred =
            OpRegistry::GetInstance().Create(op->op_->name_, op->args_, op->kwargs_, op->span_)->GetType();
        if (!structural_equal(inferred, op->GetType())) {
          Error("Register call result type mismatch", op->span_);
        }
      } catch (const pypto::Error& error) {
        Error(error.what(), op->span_);
      }
    }
    bool previous = storing_container_;
    if (region_ && IsOp(op, "array.update_element")) storing_container_ = true;
    IRVisitor::VisitExpr_(op);
    storing_container_ = previous;
  }
  void VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_); call && call->op_->name_.rfind("reg.", 0) == 0) {
      Error("Register calls require AssignStmt; bare store is not allowed", op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    const auto previous = yield_region_;
    yield_region_ = region_;
    if (collecting_) {
      for (const auto& var : op->return_vars_) RecordControlFlowValue(var);
    }
    IRVisitor::VisitStmt_(op);
    yield_region_ = previous;
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    // Require is VF-local intent, including for empty loops. Check containment
    // before visiting the body: a RegScope inside a loop cannot authorize it.
    if (collecting_ && !region_ && op->hardware_loop_policy_ == HardwareLoopPolicy::Require) {
      Error("HardwareLoopPolicy.Require requires RegScope", op->span_);
    }
    if (collecting_ && op->hardware_loop_policy_ == HardwareLoopPolicy::Require &&
        !HasCanonicalHardwareLoopBounds(op)) {
      Error("HardwareLoopPolicy.Require requires constant start=0 and step=1 with INDEX dtype", op->span_);
    }
    if (collecting_ && op->hardware_loop_policy_ == HardwareLoopPolicy::Require && !op->attrs_.empty()) {
      Error("HardwareLoopPolicy.Require does not support loop attrs", op->span_);
    }
    const auto induction_type = As<ScalarType>(op->loop_var_->GetType());
    if (collecting_ && op->hardware_loop_policy_ == HardwareLoopPolicy::Require &&
        (!induction_type || induction_type->dtype_ != DataType::INDEX)) {
      Error("Hardware loop induction variable must have INDEX dtype", op->span_);
    }
    // Enforce target-independent hardware-loop limits during the existing IR
    // walk. No target profile or repeated per-loop body analysis is needed.
    const bool hardware_loop = op->hardware_loop_policy_ == HardwareLoopPolicy::Require;
    if (hardware_loop) ++hardware_loop_depth_;
    const bool previous_empty = empty_hardware_loop_;
    const auto stop = As<ConstInt>(op->stop_);
    const bool invalid_stop = stop && !IsRepresentableLoopLiteral(stop);
    const auto start = As<ConstInt>(op->start_);
    const auto step = As<ConstInt>(op->step_);
    // Empty sequential ancestors suppress hardware realization requirements
    // even when the ancestor uses Auto policy rather than Require.
    if (IsRepresentableLoopLiteral(start) && stop && !invalid_stop && IsRepresentableLoopLiteral(step) &&
        ((step->value_ > 0 && start->value_ >= stop->value_) ||
         (step->value_ < 0 && start->value_ <= stop->value_))) {
      empty_hardware_loop_ = true;
    }
    if (collecting_ && hardware_loop && !previous_empty) {
      if (invalid_stop) {
        Error("Hardware loop stop must be representable in its dtype", op->span_);
      }
      if (!empty_hardware_loop_) {
        if (hardware_loop_depth_ > 4) {
          Error("HWL_DEPTH: hardware loops can nest at most four levels", op->span_);
        }
        if (stop && stop->value_ > 65535) {
          Error("Hardware loop stop must not exceed 65535", op->span_);
        }
      }
    }
    const auto previous = yield_region_;
    yield_region_ = region_;
    if (collecting_) {
      owners_.emplace(op->loop_var_.get(), region_);
      for (const auto& arg : op->iter_args_) RecordControlFlowValue(arg);
      for (const auto& var : op->return_vars_) RecordControlFlowValue(var);
    }
    IRVisitor::VisitStmt_(op);
    if (hardware_loop) --hardware_loop_depth_;
    empty_hardware_loop_ = previous_empty;
    yield_region_ = previous;
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    // Check lexical containment, including while nested under if/for. The
    // two ownership walks must report this unsupported statement only once.
    if (collecting_ && region_) Error("WhileStmt is not supported inside RegScope", op->span_);
    const auto previous = yield_region_;
    yield_region_ = region_;
    if (collecting_) {
      for (const auto& arg : op->iter_args_) RecordControlFlowValue(arg);
      for (const auto& var : op->return_vars_) RecordControlFlowValue(var);
    }
    IRVisitor::VisitStmt_(op);
    yield_region_ = previous;
  }
  void VisitStmt_(const YieldStmtPtr& op) override {
    auto previous = region_;
    region_ = yield_region_;
    IRVisitor::VisitStmt_(op);
    region_ = previous;
  }
  void VisitStmt_(const BreakStmtPtr& op) override {
    if (collecting_ && hardware_loop_depth_ > 0) {
      Error("BreakStmt is not supported inside hardware_loop", op->span_);
    }
  }
  void VisitStmt_(const ContinueStmtPtr& op) override {
    if (collecting_ && hardware_loop_depth_ > 0) {
      Error("ContinueStmt is not supported inside hardware_loop", op->span_);
    }
  }
  void VisitStmt_(const ReturnStmtPtr& op) override {
    if (collecting_ && region_) {
      Error("ReturnStmt is not supported inside RegScope", op->span_);
    }
    auto previous = region_;
    region_ = nullptr;
    IRVisitor::VisitStmt_(op);
    region_ = previous;
  }

 private:
  bool ContainsRegionValue(const TypePtr& type, bool include_scalar) {
    const auto key = std::make_pair(type.get(), include_scalar);
    if (auto found = region_types_.find(key); found != region_types_.end()) {
      return found->second;
    }
    bool result = As<RegType>(type) || (include_scalar && As<ScalarType>(type));
    if (auto tuple = As<TupleType>(type); tuple && !result) {
      for (const auto& element : tuple->types_) {
        if (ContainsRegionValue(element, include_scalar)) {
          result = true;
          break;
        }
      }
    }
    // Shared tuple types must not be rescanned at every use in a large body.
    region_types_.emplace(key, result);
    return result;
  }

  void RecordControlFlowValue(const VarPtr& var) {
    // Tuple aliases are only lowered within a straight-line block. SCF results
    // and carries must expose individual registers, including nested tuples.
    if (region_ && As<TupleType>(var->GetType()) && ContainsRegionValue(var->GetType(), false)) {
      Error("Register tuples cannot cross control-flow joins or loop carries; unpack the tuple first",
            var->span_);
    }
    if (ContainsRegionValue(var->GetType(), true)) owners_.emplace(var.get(), region_);
  }
  template <typename T>
  void Block(const std::shared_ptr<const T>& op) {
    bool prev = blocked_;
    if (collecting_ && region_) Error("Orchestration scopes cannot appear inside RegScope", op->span_);
    blocked_ = true;
    IRVisitor::VisitStmt_(op);
    blocked_ = prev;
  }
  void Error(const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "RegScopeValid", 0, message, span);
  }
  FunctionPtr function_;
  std::vector<Diagnostic>& diagnostics_;
  bool final_;
  int hardware_loop_depth_ = 0;
  bool empty_hardware_loop_ = false;
  bool incore_;
  bool collecting_ = true;
  bool blocked_ = false;
  bool storing_container_ = false;
  const RegScopeStmt* region_ = nullptr;
  const RegScopeStmt* yield_region_ = nullptr;
  std::map<const Var*, const RegScopeStmt*> owners_;
  std::map<std::pair<const Type*, bool>, bool> region_types_;
};
}  // namespace

void VerifyRegScopes(const FunctionPtr& function, std::vector<Diagnostic>& diagnostics,
                     bool final_placement) {
  if (function && function->body_) RegionChecker(function, diagnostics, final_placement).Run();
}
void CheckRegScopes(const FunctionPtr& function) {
  std::vector<Diagnostic> diagnostics;
  VerifyRegScopes(function, diagnostics);
  if (!diagnostics.empty()) {
    CHECK_SPAN(false, diagnostics.front().span) << diagnostics.front().message;
  }
}
class RegScopePropertyVerifier : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "RegScopeValid"; }
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, function] : program->functions_) VerifyRegScopes(function, diagnostics);
  }
};
PropertyVerifierPtr CreateRegScopePropertyVerifier() { return std::make_shared<RegScopePropertyVerifier>(); }
}  // namespace pypto::ir
