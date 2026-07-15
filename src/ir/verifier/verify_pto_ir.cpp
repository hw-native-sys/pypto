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
 * @file verify_pto_ir.cpp
 * @brief Verifier for explicit destination-passing PTO target IR.
 */

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

bool IsPTONamespaceOp(const OpPtr& op) { return op && op->name_.rfind("pto.", 0) == 0; }

bool TypeContainsTile(const TypePtr& type) {
  if (IsA<TileType>(type)) return true;
  auto tuple = As<TupleType>(type);
  if (!tuple) return false;
  for (const auto& element : tuple->types_) {
    if (TypeContainsTile(element)) return true;
  }
  return false;
}

bool CallTouchesTile(const CallPtr& call) {
  if (!call) return false;
  if (TypeContainsTile(call->GetType())) return true;
  for (const auto& arg : call->args_) {
    if (arg && TypeContainsTile(arg->GetType())) return true;
  }
  return false;
}

bool IsSupportedHandlePlanOp(const CallPtr& call) {
  return IsOp(call, "tile.load") || IsOp(call, "tile.sqrt") || IsOp(call, "tile.add") ||
         IsOp(call, "tile.mul") || IsOp(call, "tile.store");
}

const std::any* FindAttr(const CallPtr& call, const std::string& key) {
  for (const auto& [attr_key, value] : call->attrs_) {
    if (attr_key == key) return &value;
  }
  return nullptr;
}

const char* ConstraintName(PTOOperandTypeConstraint constraint) {
  switch (constraint) {
    case PTOOperandTypeConstraint::Any:
      return "Any";
    case PTOOperandTypeConstraint::Scalar:
      return "ScalarType";
    case PTOOperandTypeConstraint::TileBuffer:
      return "PTOTileBufType";
  }
  return "Unknown";
}

bool MatchesConstraint(const TypePtr& type, PTOOperandTypeConstraint constraint) {
  if (!type) return false;
  switch (constraint) {
    case PTOOperandTypeConstraint::Any:
      return true;
    case PTOOperandTypeConstraint::Scalar:
      return IsA<ScalarType>(type);
    case PTOOperandTypeConstraint::TileBuffer:
      return IsA<PTOTileBufType>(type);
  }
  return false;
}

class PTOCallVerifier final : public IRVisitor {
 public:
  PTOCallVerifier(FunctionType function_type, std::vector<Diagnostic>& diagnostics)
      : function_type_(function_type), diagnostics_(diagnostics) {}

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    WithDirectCall(As<Call>(op->value_), op->var_, [&]() { IRVisitor::VisitStmt_(op); });
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    WithDirectCall(As<Call>(op->expr_), std::nullopt, [&]() { IRVisitor::VisitStmt_(op); });
  }

  void VisitExpr_(const CallPtr& call) override {
    if (IsPTONamespaceOp(call->op_)) VerifyPTOCall(call);
    IRVisitor::VisitExpr_(call);
  }

 private:
  template <typename F>
  void WithDirectCall(CallPtr direct_call, std::optional<VarPtr> assigned_var, F&& visit) {
    auto old_call = direct_call_;
    auto old_var = assigned_var_;
    direct_call_ = std::move(direct_call);
    assigned_var_ = std::move(assigned_var);
    std::forward<F>(visit)();
    direct_call_ = std::move(old_call);
    assigned_var_ = std::move(old_var);
  }

  void Error(const CallPtr& call, const std::string& message) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "PTOBufferized", 0, message, call->span_);
  }

  void VerifyPTOCall(const CallPtr& call) {
    const std::string& op_name = call->op_->name_;
    auto& registry = OpRegistry::GetInstance();
    if (!registry.IsRegistered(op_name)) {
      Error(call, "PTO target operation '" + op_name + "' is not registered");
      return;
    }

    const auto& spec_opt = registry.GetEntry(op_name).GetPTOOpSpec();
    if (!spec_opt.has_value()) {
      Error(call, "PTO target operation '" + op_name + "' has no operand/effect schema");
      return;
    }
    const auto& spec = *spec_opt;

    if (function_type_ == FunctionType::Orchestration) {
      Error(call, "PTO target operation '" + op_name + "' cannot appear in an Orchestration function");
    }
    if (spec.IsPure()) {
      Error(call, "PTO target operation '" + op_name + "' is incorrectly marked pure");
    }

    auto segments = spec.ResolveOperandSegments(call->args_.size());
    if (!segments.has_value()) {
      std::ostringstream msg;
      msg << "PTO target operation '" << op_name << "' has invalid operand count " << call->args_.size();
      Error(call, msg.str());
      return;
    }

    size_t arg_index = 0;
    for (size_t group_index = 0; group_index < spec.operand_groups.size(); ++group_index) {
      const auto& group = spec.operand_groups[group_index];
      for (size_t i = 0; i < (*segments)[group_index]; ++i, ++arg_index) {
        const auto& arg = call->args_[arg_index];
        if (!arg || !MatchesConstraint(arg->GetType(), group.type_constraint)) {
          std::ostringstream msg;
          msg << "PTO target operation '" << op_name << "' operand #" << arg_index << " must have "
              << ConstraintName(group.type_constraint);
          Error(call, msg.str());
        }
      }
    }

    const bool is_direct_statement_value = direct_call_ && direct_call_.get() == call.get();
    if (spec.result_kind == PTOResultKind::TileBuffer) {
      if (!is_direct_statement_value || !assigned_var_.has_value()) {
        Error(call, "PTO allocating operation '" + op_name + "' must be the value of an AssignStmt");
      }
      if (!IsA<PTOTileBufType>(call->GetType())) {
        Error(call, "PTO allocating operation '" + op_name + "' result must have PTOTileBufType");
      }
      if (assigned_var_.has_value()) {
        const auto& var = *assigned_var_;
        if (!IsA<PTOTileBufType>(var->GetType())) {
          Error(call, "PTO allocation destination must have PTOTileBufType");
        } else if (!structural_equal(var->GetType(), call->GetType())) {
          Error(call, "PTO allocation destination type must equal the allocation result type");
        }
      }
    } else {
      if (!is_direct_statement_value || assigned_var_.has_value()) {
        Error(call, "Destination-passing PTO operation '" + op_name + "' must appear in an EvalStmt");
      }
      if (!IsA<UnknownType>(call->GetType())) {
        Error(call, "Destination-passing PTO operation '" + op_name + "' must not return a value");
      }
    }
  }

  FunctionType function_type_;
  std::vector<Diagnostic>& diagnostics_;
  CallPtr direct_call_;
  std::optional<VarPtr> assigned_var_;
};

class PTOBufferizedVerifier final : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "PTOBufferized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, function] : program->functions_) {
      if (!function || !function->body_) continue;
      PTOCallVerifier visitor(function->func_type_, diagnostics);
      visitor.VisitStmt(function->body_);
    }
  }
};

class PTOHandlesMaterializedVerifier final : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "PTOHandlesMaterialized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, function] : program->functions_) {
      if (!function || !function->body_ || !IsInCoreType(function->func_type_)) continue;
      VerifyFunction(function, diagnostics);
    }
  }

 private:
  static void Error(std::vector<Diagnostic>& diagnostics, const Span& span, const std::string& message) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "PTOHandlesMaterialized", 0, message, span);
  }

  static void VerifyFunction(const FunctionPtr& function, std::vector<Diagnostic>& diagnostics) {
    std::unordered_set<const Var*> allocated;
    std::unordered_set<const Var*> claimed;
    std::unordered_map<const Var*, VarPtr> logical_to_handle;

    for (const auto& param : function->params_) {
      if (TypeContainsTile(param->GetType())) {
        Error(diagnostics, param->span_,
              "Tile-typed function parameters are not supported by the Step-3 plan");
      }
    }

    const auto stmts = transform_utils::FlattenToStmts(function->body_);
    for (const auto& stmt : stmts) {
      if (auto assign = As<AssignStmt>(stmt)) {
        auto call = As<Call>(assign->value_);
        if (call && IsOp(call, "pto.alloc_tile")) {
          if (!IsA<PTOTileBufType>(assign->var_->GetType()) || !IsA<PTOTileBufType>(call->GetType())) {
            Error(diagnostics, assign->span_,
                  "pto.alloc_tile definition and result must have PTOTileBufType");
          } else if (!structural_equal(assign->var_->GetType(), call->GetType())) {
            Error(diagnostics, assign->span_, "pto.alloc_tile definition and result types must match");
          }
          if (call->args_.size() < 2 || call->args_.size() > 3) {
            Error(diagnostics, call->span_, "pto.alloc_tile must have two or three metadata operands");
          }
          for (const auto& arg : call->args_) {
            if (!arg || !IsA<ScalarType>(arg->GetType())) {
              Error(diagnostics, call->span_, "pto.alloc_tile metadata operands must have ScalarType");
              break;
            }
          }
          if (!allocated.insert(assign->var_.get()).second) {
            Error(diagnostics, assign->span_, "PTO handle is allocated more than once");
          }
          continue;
        }
        VerifyCall(call, assign->var_, stmt->span_, allocated, claimed, logical_to_handle, diagnostics);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        VerifyCall(As<Call>(eval->expr_), std::nullopt, stmt->span_, allocated, claimed, logical_to_handle,
                   diagnostics);
      } else if (auto ret = As<ReturnStmt>(stmt)) {
        for (const auto& value : ret->value_) {
          if (TypeContainsTile(value->GetType())) {
            Error(diagnostics, ret->span_, "Returning Tile values is not supported by the Step-3 plan");
          }
        }
      } else {
        Error(diagnostics, stmt->span_,
              "PTO handle plan requires straight-line AssignStmt/EvalStmt/ReturnStmt structure");
      }
    }

    if (allocated.size() != claimed.size()) {
      Error(diagnostics, function->span_,
            "Every pto.alloc_tile handle must be claimed by exactly one Tile producer");
    }
  }

  static void VerifyCall(const CallPtr& call, const std::optional<VarPtr>& result_var, const Span& span,
                         const std::unordered_set<const Var*>& allocated,
                         std::unordered_set<const Var*>& claimed,
                         std::unordered_map<const Var*, VarPtr>& logical_to_handle,
                         std::vector<Diagnostic>& diagnostics) {
    if (!call) {
      if (result_var.has_value() && TypeContainsTile((*result_var)->GetType())) {
        Error(diagnostics, span, "Tile definitions in the Step-3 plan must be direct Calls");
      }
      return;
    }
    if (IsPTONamespaceOp(call->op_)) {
      Error(diagnostics, call->span_,
            "Only pto.alloc_tile may appear before the PTO target-op rewrite stage");
      return;
    }
    if (!IsSupportedHandlePlanOp(call)) {
      if (CallTouchesTile(call)) {
        Error(diagnostics, call->span_,
              "Unsupported Tile operation in PTO handle plan: '" + call->op_->name_ + "'");
      }
      return;
    }

    std::vector<VarPtr> expected_inputs;
    for (const auto& arg : call->args_) {
      if (!IsA<TileType>(arg->GetType())) continue;
      auto logical_var = AsVarLike(arg);
      if (!logical_var) {
        Error(diagnostics, call->span_, "Tile operands in the PTO handle plan must be flattened Vars");
        continue;
      }
      auto it = logical_to_handle.find(logical_var.get());
      if (it == logical_to_handle.end()) {
        Error(diagnostics, call->span_, "Tile operand has no dominating PTO handle");
        continue;
      }
      expected_inputs.push_back(it->second);
    }

    const std::any* input_attr = FindAttr(call, kAttrPTOInputHandles);
    const auto* actual_inputs = input_attr ? std::any_cast<std::vector<VarPtr>>(input_attr) : nullptr;
    if (!actual_inputs) {
      Error(diagnostics, call->span_, "Supported Tile call is missing vector<Var> pto_input_handles attr");
    } else if (actual_inputs->size() != expected_inputs.size()) {
      Error(diagnostics, call->span_, "pto_input_handles count does not match Tile operand count");
    } else {
      for (size_t i = 0; i < actual_inputs->size(); ++i) {
        if ((*actual_inputs)[i].get() != expected_inputs[i].get()) {
          Error(diagnostics, call->span_,
                "pto_input_handles does not match the dominating value-to-handle plan");
          break;
        }
      }
    }

    const bool tile_result = result_var.has_value() && IsA<TileType>((*result_var)->GetType());
    const std::any* output_attr = FindAttr(call, kAttrPTOOutputHandle);
    const auto* output_handle = output_attr ? std::any_cast<VarPtr>(output_attr) : nullptr;
    if (tile_result) {
      if (!output_handle || !*output_handle) {
        Error(diagnostics, call->span_, "Tile producer is missing Var pto_output_handle attr");
        return;
      }
      if (allocated.count(output_handle->get()) == 0) {
        Error(diagnostics, call->span_, "pto_output_handle does not name a dominating pto.alloc_tile");
      }
      if (!claimed.insert(output_handle->get()).second) {
        Error(diagnostics, call->span_, "PTO handle is claimed by more than one logical Tile producer");
      }
      logical_to_handle[result_var->get()] = *output_handle;
    } else if (output_attr) {
      Error(diagnostics, call->span_, "Non-Tile call must not carry pto_output_handle");
    }
  }
};

}  // namespace

PropertyVerifierPtr CreatePTOBufferizedPropertyVerifier() {
  return std::make_shared<PTOBufferizedVerifier>();
}

PropertyVerifierPtr CreatePTOHandlesMaterializedPropertyVerifier() {
  return std::make_shared<PTOHandlesMaterializedVerifier>();
}

}  // namespace ir
}  // namespace pypto
