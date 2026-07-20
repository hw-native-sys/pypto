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
#include <functional>
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
#include "pypto/ir/pto_target_lowering.h"
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
    if (TypeContainsTile(op->var_->GetType())) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "PTOBufferized", 0,
                                "Logical Tile assignment remains in PTO target IR", op->span_);
    }
    WithDirectCall(As<Call>(op->value_), op->var_, [&]() { IRVisitor::VisitStmt_(op); });
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    WithDirectCall(As<Call>(op->expr_), std::nullopt, [&]() { IRVisitor::VisitStmt_(op); });
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    for (const auto& value : op->value_) {
      if (TypeContainsTile(value->GetType())) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "PTOBufferized", 0,
                                  "Logical Tile return remains in PTO target IR", op->span_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallPtr& call) override {
    if (IsPTONamespaceOp(call->op_)) {
      VerifyPTOCall(call);
    } else if (CallTouchesTile(call)) {
      Error(call, "Logical Tile operation '" + call->op_->name_ + "' remains in PTO target IR");
    }
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

    if (!IsInCoreType(function_type_)) {
      Error(call, "PTO target operation '" + op_name + "' can appear only in an InCore function");
    }
    auto segments = spec.ResolveOperandSegments(call->args_.size());
    if (!segments.has_value()) {
      std::ostringstream msg;
      msg << "PTO target operation '" << op_name << "' has invalid operand count " << call->args_.size();
      Error(call, msg.str());
      return;
    }

    if (IsOp(call, "pto.tload") || IsOp(call, "pto.tstore")) {
      VerifyPartitionTransfer(call);
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
        Error(call, "PTO handle-defining operation '" + op_name + "' must be the value of an AssignStmt");
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

  void VerifyPartitionTransfer(const CallPtr& call) {
    if (call->args_.size() != 4) return;  // Generic schema diagnostic already reports the count.
    const size_t tensor_index = IsOp(call, "pto.tload") ? 0 : 3;
    auto tensor_type = AsTensorTypeLike(call->args_[tensor_index]->GetType());
    if (!tensor_type) {
      Error(call, "PTO partition transfer tensor operand must have TensorType");
    }
    std::optional<size_t> transfer_rank;
    for (size_t tuple_index : {size_t{1}, size_t{2}}) {
      auto tuple = As<MakeTuple>(call->args_[tuple_index]);
      if (!tuple || tuple->elements_.empty()) {
        Error(call, "PTO partition transfer offsets/extents must be non-empty MakeTuple operands");
        continue;
      }
      if (!transfer_rank.has_value()) {
        transfer_rank = tuple->elements_.size();
      } else if (*transfer_rank != tuple->elements_.size()) {
        Error(call, "PTO partition transfer offsets and extents must have the same rank");
      }
      for (const auto& element : tuple->elements_) {
        if (!element || !IsA<ScalarType>(element->GetType())) {
          Error(call, "PTO partition transfer offsets/extents must contain ScalarType values");
          break;
        }
      }
    }
    if (tensor_type && transfer_rank.has_value() && *transfer_rank != tensor_type->shape_.size()) {
      Error(call, "PTO partition transfer rank must match the tensor operand rank");
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
      if (function->GetAttr<bool>(kAttrPTOTargetLoweringDeferred, false)) continue;
      PTOCallVerifier visitor(function->func_type_, diagnostics);
      if (!IsInCoreType(function->func_type_)) {
        // Orchestration remains outside the Step-4 rewrite, but still reject
        // target PTO calls that escaped into the wrong function layer.
        visitor.VisitStmt(function->body_);
        continue;
      }
      for (const auto& param : function->params_) {
        if (TypeContainsTile(param->GetType())) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "PTOBufferized", 0,
                                   "Logical Tile parameter remains in PTO target IR", param->span_);
        }
      }
      VerifyHandleDominance(function, diagnostics);
      visitor.VisitStmt(function->body_);
    }
  }

 private:
  static void VerifyHandleDominance(const FunctionPtr& function, std::vector<Diagnostic>& diagnostics) {
    std::unordered_set<const Var*> parameter_handles;
    for (const auto& param : function->params_) {
      if (IsA<PTOTileBufType>(param->GetType())) parameter_handles.insert(param.get());
    }

    const auto verify_operands = [&](const CallPtr& call,
                                     const std::unordered_set<const Var*>& available_handles) {
      if (!call || !IsPTONamespaceOp(call->op_)) return;
      for (const auto& arg : call->args_) {
        if (!arg || !IsA<PTOTileBufType>(arg->GetType())) continue;
        auto handle = AsVarLike(arg);
        if (!handle || available_handles.count(handle.get()) == 0) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "PTOBufferized", 0,
                                   "PTO tile-buffer operand has no dominating allocation or parameter",
                                   call->span_);
        }
      }
    };

    std::function<void(const StmtPtr&, std::unordered_set<const Var*>)> verify_body;
    verify_body = [&](const StmtPtr& body, std::unordered_set<const Var*> available_handles) {
      for (const auto& stmt : transform_utils::FlattenToStmts(body)) {
        if (auto assign = As<AssignStmt>(stmt)) {
          auto call = As<Call>(assign->value_);
          verify_operands(call, available_handles);
          if (IsA<PTOTileBufType>(assign->var_->GetType())) {
            const auto* spec = call && IsPTONamespaceOp(call->op_)
                                   ? &OpRegistry::GetInstance().GetEntry(call->op_->name_).GetPTOOpSpec()
                                   : nullptr;
            const bool defines_handle =
                spec && spec->has_value() && (*spec)->result_kind == PTOResultKind::TileBuffer;
            if (!defines_handle) {
              diagnostics.emplace_back(
                  DiagnosticSeverity::Error, "PTOBufferized", 0,
                  "PTO tile-buffer handles must be defined by a registered target handle op", assign->span_);
            } else if (!available_handles.insert(assign->var_.get()).second) {
              diagnostics.emplace_back(DiagnosticSeverity::Error, "PTOBufferized", 0,
                                       "PTO tile-buffer handle is allocated more than once", assign->span_);
            }
          }
        } else if (auto eval = As<EvalStmt>(stmt)) {
          verify_operands(As<Call>(eval->expr_), available_handles);
        } else if (auto loop = As<ForStmt>(stmt)) {
          verify_body(loop->body_, available_handles);
        } else if (auto branch = As<IfStmt>(stmt)) {
          verify_body(branch->then_body_, available_handles);
          if (branch->else_body_) verify_body(*branch->else_body_, available_handles);
        } else if (!As<ReturnStmt>(stmt) && !As<YieldStmt>(stmt)) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "PTOBufferized", 0,
                                   "Step-4 PTO target IR contains an unsupported structured statement",
                                   stmt->span_);
        }
      }
    };
    verify_body(function->body_, std::move(parameter_handles));
  }
};

class PTOHandlesMaterializedVerifier final : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "PTOHandlesMaterialized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, function] : program->functions_) {
      if (!function || !function->body_ || !IsInCoreType(function->func_type_)) continue;
      if (function->GetAttr<bool>(kAttrPTOTargetLoweringDeferred, false)) continue;
      VerifyFunction(function, diagnostics);
    }
  }

 private:
  static void Error(std::vector<Diagnostic>& diagnostics, const Span& span, const std::string& message) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "PTOHandlesMaterialized", 0, message, span);
  }

  static void VerifyFunction(const FunctionPtr& function, std::vector<Diagnostic>& diagnostics) {
    std::unordered_set<const Var*> allocated;
    std::vector<const Var*> allocated_order;
    std::unordered_set<const Var*> claimed;
    std::unordered_map<const Var*, VarPtr> logical_to_handle;
    std::unordered_set<const Var*> control_handles;
    std::vector<const Var*> control_handle_order;

    if (function->HasAttr(kAttrPTOControlFlowHandles)) {
      const auto aliases =
          function->GetAttr<std::vector<std::pair<VarPtr, VarPtr>>>(kAttrPTOControlFlowHandles, {});
      for (const auto& [logical, handle] : aliases) {
        if (!logical || !handle) {
          Error(diagnostics, function->span_, "Control-flow PTO handle plan contains a null value");
          continue;
        }
        logical_to_handle[logical.get()] = handle;
        if (control_handles.insert(handle.get()).second) control_handle_order.push_back(handle.get());
      }
    }

    for (const auto& param : function->params_) {
      if (TypeContainsTile(param->GetType())) {
        Error(diagnostics, param->span_,
              "Tile-typed function parameters are not supported by the Step-3 plan");
      }
    }

    std::function<void(const StmtPtr&, std::unordered_set<const Var*>)> verify_body;
    verify_body = [&](const StmtPtr& body, std::unordered_set<const Var*> available) {
      for (const auto& stmt : transform_utils::FlattenToStmts(body)) {
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
            } else {
              allocated_order.push_back(assign->var_.get());
            }
            available.insert(assign->var_.get());
            continue;
          }
          if (call && IsOp(call, "pto.subview")) {
            if (!IsA<PTOTileBufType>(assign->var_->GetType()) || !IsA<PTOTileBufType>(call->GetType()) ||
                !structural_equal(assign->var_->GetType(), call->GetType())) {
              Error(diagnostics, assign->span_,
                    "pto.subview definition and result must have matching PTOTileBufType");
            }
            if (call->args_.size() != 4) {
              Error(diagnostics, call->span_,
                    "pto.subview must have source, shape, offset, and valid-shape operands");
            } else {
              auto source = AsVarLike(call->args_[0]);
              if (!source || available.count(source.get()) == 0) {
                Error(diagnostics, call->span_, "pto.subview source must name a dominating PTO handle");
              }
              for (size_t i = 1; i < call->args_.size(); ++i) {
                auto tuple = As<MakeTuple>(call->args_[i]);
                if (!tuple || tuple->elements_.size() != 2) {
                  Error(diagnostics, call->span_,
                        "pto.subview shape, offset, and valid-shape operands must be rank-2 tuples");
                }
              }
            }
            if (!allocated.insert(assign->var_.get()).second) {
              Error(diagnostics, assign->span_, "PTO handle is defined more than once");
            } else {
              allocated_order.push_back(assign->var_.get());
            }
            available.insert(assign->var_.get());
            continue;
          }
          VerifyCall(call, assign->var_, stmt->span_, available, claimed, control_handles, logical_to_handle,
                     diagnostics);
        } else if (auto eval = As<EvalStmt>(stmt)) {
          auto call = As<Call>(eval->expr_);
          if (call && IsPTONamespaceOp(call->op_)) {
            for (const auto& arg : call->args_) {
              if (!IsA<PTOTileBufType>(arg->GetType())) continue;
              auto handle = AsVarLike(arg);
              if (!handle || available.count(handle.get()) == 0) {
                Error(diagnostics, call->span_,
                      "Structured PTO target operation uses a non-dominating tile-buffer handle");
              }
            }
          } else {
            VerifyCall(call, std::nullopt, stmt->span_, available, claimed, control_handles,
                       logical_to_handle, diagnostics);
          }
        } else if (auto loop = As<ForStmt>(stmt)) {
          verify_body(loop->body_, available);
        } else if (auto branch = As<IfStmt>(stmt)) {
          verify_body(branch->then_body_, available);
          if (branch->else_body_) verify_body(*branch->else_body_, available);
        } else if (auto yield = As<YieldStmt>(stmt)) {
          for (const auto& value : yield->value_) {
            if (!IsA<TileType>(value->GetType())) continue;
            auto logical = AsVarLike(value);
            if (!logical || logical_to_handle.count(logical.get()) == 0) {
              Error(diagnostics, yield->span_, "Yielded Tile value has no explicit PTO handle plan");
            }
          }
        } else if (auto ret = As<ReturnStmt>(stmt)) {
          for (const auto& value : ret->value_) {
            if (TypeContainsTile(value->GetType())) {
              Error(diagnostics, ret->span_, "Returning Tile values is not supported by the Step-3 plan");
            }
          }
        } else {
          Error(diagnostics, stmt->span_, "PTO handle plan contains an unsupported structured statement");
        }
      }
    };
    verify_body(function->body_, {});

    for (const auto* handle : allocated_order) {
      if (claimed.count(handle) == 0 && control_handles.count(handle) == 0) {
        Error(diagnostics, function->span_,
              "Every PTO handle definition must be claimed by a Tile producer or control-flow result");
        break;
      }
    }
    for (const auto* handle : control_handle_order) {
      if (allocated.count(handle) == 0) {
        Error(diagnostics, function->span_,
              "Control-flow PTO handle plan references a handle not defined in the function");
        break;
      }
    }
  }

  static void VerifyCall(const CallPtr& call, const std::optional<VarPtr>& result_var, const Span& span,
                         const std::unordered_set<const Var*>& allocated,
                         std::unordered_set<const Var*>& claimed,
                         const std::unordered_set<const Var*>& control_handles,
                         std::unordered_map<const Var*, VarPtr>& logical_to_handle,
                         std::vector<Diagnostic>& diagnostics) {
    if (!call) {
      if (result_var.has_value() && TypeContainsTile((*result_var)->GetType())) {
        if (logical_to_handle.count(result_var->get()) == 0) {
          Error(diagnostics, span,
                "Tile aliases in the Step-3 plan must have an explicit control-flow handle mapping");
        }
      }
      return;
    }
    if (IsPTONamespaceOp(call->op_)) {
      Error(diagnostics, call->span_,
            "Only PTO handle definitions may appear before the target-op rewrite stage");
      return;
    }
    if (!IsPTOHandlePlanOp(call->op_->name_)) {
      if (CallTouchesTile(call)) {
        Error(diagnostics, call->span_,
              "Unsupported Tile operation in PTO handle plan: '" + call->op_->name_ + "'");
      }
      return;
    }

    std::vector<VarPtr> expected_inputs;
    for (const auto& arg : call->args_) {
      if (!arg || !IsA<TileType>(arg->GetType())) continue;
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
        Error(diagnostics, call->span_, "pto_output_handle does not name a dominating PTO handle");
      }
      if (!claimed.insert(output_handle->get()).second && control_handles.count(output_handle->get()) == 0) {
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
