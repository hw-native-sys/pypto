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

#include <algorithm>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/tpop_tfree_finalizer.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using core_affinity::ClassifyCallAffinity;
using core_affinity::CoreAffinity;
using core_affinity::kDirMaskC2V;
using core_affinity::kDirMaskV2C;
using cross_core_pipe::CollectCrossCorePipeMetadata;
using cross_core_pipe::CollectDominatingPipeSetupMetadata;
using cross_core_pipe::CrossCorePipeMetadata;
using tpop_tfree::IsExpectedTpopAssignStmt;
using tpop_tfree::IsTfreeStmt;

namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

bool IsAutoPlaceholderBufferOperand(const ExprPtr& e) {
  auto c = As<ConstInt>(e);
  return c != nullptr && c->value_ == 0;
}

void VerifySingleInitializePipeCall(const CallPtr& call, const std::string& func_name,
                                    const std::string& op_name, std::vector<Diagnostic>& diagnostics) {
  if (call->args_.size() != 2) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             "Function '" + func_name + "': '" + op_name +
                                 "' requires exactly 2 operands (c2v_consumer_buf, v2c_consumer_buf), got " +
                                 std::to_string(call->args_.size()),
                             call->span_);
    return;
  }

  const int dir_mask = call->GetKwarg<int>("dir_mask", -1);
  const int slot_size = call->GetKwarg<int>("slot_size", -1);
  const int valid_dir_mask = kDirMaskC2V | kDirMaskV2C;
  if (dir_mask < 0 || (dir_mask & ~valid_dir_mask) != 0 || (dir_mask & valid_dir_mask) == 0) {
    diagnostics.emplace_back(
        DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
        "Function '" + func_name + "': '" + op_name + "' requires valid 'dir_mask' attribute", call->span_);
  }
  if (slot_size <= 0) {
    diagnostics.emplace_back(
        DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
        "Function '" + func_name + "': '" + op_name + "' requires positive 'slot_size' attribute",
        call->span_);
  }
  if (dir_mask < 0 || (dir_mask & ~valid_dir_mask) != 0 || (dir_mask & valid_dir_mask) == 0 ||
      slot_size <= 0) {
    return;
  }

  const bool c2v_active = (dir_mask & kDirMaskC2V) != 0;
  const bool v2c_active = (dir_mask & kDirMaskV2C) != 0;

  auto check_operand = [&](const ExprPtr& arg, bool active, const char* role) {
    const bool placeholder = IsAutoPlaceholderBufferOperand(arg);
    if (active) {
      if (placeholder) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                 "Function '" + func_name + "': '" + op_name + "' enables " + role +
                                     " for this dir_mask but operand is ConstInt(0/-1) placeholder; use a "
                                     "concrete i32 SSA (Var or "
                                     "reserve/import Call)",
                                 call->span_);
      }
    } else {
      if (!placeholder) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                 "Function '" + func_name + "': '" + op_name + "' does not use " + role +
                                     " for this dir_mask; operand must be ConstInt(0) placeholder",
                                 call->span_);
      }
    }
  };

  check_operand(call->args_[0], c2v_active, "C2V (c2v_consumer_buf)");
  check_operand(call->args_[1], v2c_active, "V2C (v2c_consumer_buf)");
}

void TryVerifyInitializePipeFromStmt(const StmtPtr& stmt, const std::string& func_name,
                                     std::vector<Diagnostic>& diagnostics) {
  CallPtr call;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  if (!call || !call->op_) return;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return;
  if (op->name_ != "system.aic_initialize_pipe" && op->name_ != "system.aiv_initialize_pipe") return;
  VerifySingleInitializePipeCall(call, func_name, op->name_, diagnostics);
}

void WalkStmtsVerifyInitializePipe(const std::vector<StmtPtr>& stmts, const std::string& func_name,
                                   std::vector<Diagnostic>& diagnostics) {
  for (const auto& stmt : stmts) {
    TryVerifyInitializePipeFromStmt(stmt, func_name, diagnostics);
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      WalkStmtsVerifyInitializePipe(FlattenBody(for_stmt->body_), func_name, diagnostics);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      WalkStmtsVerifyInitializePipe(FlattenBody(if_stmt->then_body_), func_name, diagnostics);
      if (const auto& else_body = if_stmt->else_body_) {
        WalkStmtsVerifyInitializePipe(FlattenBody(*else_body), func_name, diagnostics);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      WalkStmtsVerifyInitializePipe(FlattenBody(while_stmt->body_), func_name, diagnostics);
    }
  }
}

void VerifyInitializePipeOperands(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
  if (!func->body_) return;
  WalkStmtsVerifyInitializePipe(FlattenBody(func->body_), func->name_, diagnostics);
}

void TryCountReserveImportFromStmt(const StmtPtr& stmt, int& reserve_count, int& import_count) {
  CallPtr call;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  if (!call || !call->op_) return;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return;
  if (op->name_ == "system.reserve_buffer") {
    ++reserve_count;
  } else if (op->name_ == "system.import_peer_buffer") {
    ++import_count;
  }
}

void AccumulateRequiredReserveImportFromInitializePipe(const CallPtr& call, FunctionType func_type,
                                                       int& required_reserve_count,
                                                       int& required_import_count) {
  if (!call || !call->op_) return;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op || (op->name_ != "system.aic_initialize_pipe" && op->name_ != "system.aiv_initialize_pipe")) {
    return;
  }
  const int dir_mask = call->GetKwarg<int>("dir_mask", 0);
  const bool c2v_active = (dir_mask & kDirMaskC2V) != 0;
  const bool v2c_active = (dir_mask & kDirMaskV2C) != 0;
  if (func_type == FunctionType::AIC) {
    if (c2v_active) ++required_import_count;
    if (v2c_active) ++required_reserve_count;
  } else if (func_type == FunctionType::AIV) {
    if (c2v_active) ++required_reserve_count;
    if (v2c_active) ++required_import_count;
  }
}

void WalkStmtsCountReserveImport(const std::vector<StmtPtr>& stmts, int& reserve_count, int& import_count,
                                 int& required_reserve_count, int& required_import_count,
                                 FunctionType func_type) {
  for (const auto& stmt : stmts) {
    TryCountReserveImportFromStmt(stmt, reserve_count, import_count);
    AccumulateRequiredReserveImportFromInitializePipe(transform_utils::GetCallFromStmt(stmt), func_type,
                                                      required_reserve_count, required_import_count);
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      WalkStmtsCountReserveImport(FlattenBody(for_stmt->body_), reserve_count, import_count,
                                  required_reserve_count, required_import_count, func_type);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      WalkStmtsCountReserveImport(FlattenBody(if_stmt->then_body_), reserve_count, import_count,
                                  required_reserve_count, required_import_count, func_type);
      if (const auto& else_body = if_stmt->else_body_) {
        WalkStmtsCountReserveImport(FlattenBody(*else_body), reserve_count, import_count,
                                    required_reserve_count, required_import_count, func_type);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      WalkStmtsCountReserveImport(FlattenBody(while_stmt->body_), reserve_count, import_count,
                                  required_reserve_count, required_import_count, func_type);
    }
  }
}

void VerifyReserveAndImportPeerBufferCoverage(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
  if (!func->body_) return;
  int reserve_count = 0;
  int import_count = 0;
  int required_reserve_count = 0;
  int required_import_count = 0;
  WalkStmtsCountReserveImport(FlattenBody(func->body_), reserve_count, import_count, required_reserve_count,
                              required_import_count, func->func_type_);
  if (reserve_count < required_reserve_count) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             "Function '" + func->name_ +
                                 "' has fewer 'system.reserve_buffer' calls than initialized pipe "
                                 "directions require",
                             func->span_);
  }
  if (reserve_count > required_reserve_count) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             "Function '" + func->name_ +
                                 "' has more 'system.reserve_buffer' calls than initialized pipe "
                                 "directions require",
                             func->span_);
  }
  if (import_count < required_import_count) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             "Function '" + func->name_ +
                                 "' has fewer 'system.import_peer_buffer' calls than initialized pipe "
                                 "directions require",
                             func->span_);
  }
  if (import_count > required_import_count) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             "Function '" + func->name_ +
                                 "' has more 'system.import_peer_buffer' calls than initialized pipe "
                                 "directions require",
                             func->span_);
  }
}

class MixedKernelExpandedVerifier : public IRVisitor {
 public:
  explicit MixedKernelExpandedVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void VisitExpr_(const CallPtr& op) override {
    if (!op || !op->op_) {
      IRVisitor::VisitExpr_(op);
      return;
    }
    auto affinity = ClassifyCallAffinity(op);
    if (affinity == CoreAffinity::CUBE) {
      has_cube_ = true;
    } else if (affinity == CoreAffinity::VECTOR) {
      has_vector_ = true;
    } else if (affinity == CoreAffinity::MIXED) {
      // Leaf-call MIXED can only come from a C/V-crossing tile.move — contributes
      // to both sides for the purposes of this check.
      has_cube_ = true;
      has_vector_ = true;
    }
    IRVisitor::VisitExpr_(op);
  }

  void CheckResult() {
    if (has_cube_ && has_vector_) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                "InCore function '" + func_name_ +
                                    "' contains both Cube and Vector tile ops (should have been expanded)",
                                Span::unknown());
    }
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  bool has_cube_ = false;
  bool has_vector_ = false;
};

class TpopMemoryVerifier : public IRVisitor {
 public:
  TpopMemoryVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name, FunctionType func_type)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), func_type_(func_type) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    auto call = std::dynamic_pointer_cast<const Call>(op->value_);
    auto ir_op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    if (!ir_op) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    std::optional<MemorySpace> expected_memory;
    if (func_type_ == FunctionType::AIC && ir_op->name_ == "tile.tpop_from_aiv") {
      expected_memory = MemorySpace::Mat;
    } else if (func_type_ == FunctionType::AIV && ir_op->name_ == "tile.tpop_from_aic") {
      expected_memory = MemorySpace::Vec;
    }

    if (expected_memory.has_value()) {
      auto tile_type = std::dynamic_pointer_cast<const TileType>(op->var_->GetType());
      bool valid =
          tile_type && tile_type->memory_space_.has_value() && tile_type->memory_space_ == expected_memory;
      if (!valid) {
        std::string func_kind = (func_type_ == FunctionType::AIC) ? "AIC" : "AIV";
        std::string actual_memory = "unset";
        if (tile_type) {
          if (const auto& memory_space = tile_type->memory_space_) {
            actual_memory = MemorySpaceToString(*memory_space);
          }
        }
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                  func_kind + " function '" + func_name_ + "' requires " + ir_op->name_ +
                                      " result in MemorySpace::" + MemorySpaceToString(*expected_memory) +
                                      ", got MemorySpace::" + actual_memory,
                                  op->span_);
      }
    }

    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  FunctionType func_type_;
};

void VerifyCrossCorePipeSetup(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
  CrossCorePipeMetadata metadata;
  CollectCrossCorePipeMetadata(FlattenBody(func->body_), metadata);
  if (!metadata.HasCrossCoreOps()) return;
  CrossCorePipeMetadata dominating_setup = CollectDominatingPipeSetupMetadata(FlattenBody(func->body_));

  if ((metadata.c2v.has_ops && !metadata.c2v.slot_size_bytes.has_value()) ||
      (metadata.v2c.has_ops && !metadata.v2c.slot_size_bytes.has_value())) {
    diagnostics.emplace_back(
        DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
        "Function '" + func->name_ +
            "' uses cross-core tile ops with non-static tile size; auto pipe setup requires "
            "statically known tile shapes",
        func->span_);
  }
  if (func->func_type_ == FunctionType::AIC) {
    if (!dominating_setup.has_aic_initialize_pipe) {
      diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                               "AIC function '" + func->name_ +
                                   "' uses cross-core tile ops but has no 'system.aic_initialize_pipe' call",
                               func->span_);
    }
    if (metadata.v2c.has_ops && !dominating_setup.has_reserve_buffer) {
      diagnostics.emplace_back(
          DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
          "AIC function '" + func->name_ + "' uses V2C cross-core ops but has no 'system.reserve_buffer'",
          func->span_);
    }
    if (metadata.c2v.has_ops && !dominating_setup.has_import_peer_buffer) {
      diagnostics.emplace_back(
          DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
          "AIC function '" + func->name_ + "' uses C2V cross-core ops but has no 'system.import_peer_buffer'",
          func->span_);
    }
  } else if (func->func_type_ == FunctionType::AIV) {
    if (!dominating_setup.has_aiv_initialize_pipe) {
      diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                               "AIV function '" + func->name_ +
                                   "' uses cross-core tile ops but has no 'system.aiv_initialize_pipe' call",
                               func->span_);
    }
    if (metadata.c2v.has_ops && !dominating_setup.has_reserve_buffer) {
      diagnostics.emplace_back(
          DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
          "AIV function '" + func->name_ + "' uses C2V cross-core ops but has no 'system.reserve_buffer'",
          func->span_);
    }
    if (metadata.v2c.has_ops && !dominating_setup.has_import_peer_buffer) {
      diagnostics.emplace_back(
          DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
          "AIV function '" + func->name_ + "' uses V2C cross-core ops but has no 'system.import_peer_buffer'",
          func->span_);
    }
  }
}

struct OutstandingTpop {
  VarPtr var;
  std::string op_name;
  Span span;
};

void VerifyTpopTfreeMatchingInBlock(const std::vector<StmtPtr>& stmts, const FunctionPtr& func,
                                    std::vector<Diagnostic>& diagnostics);

void VerifyNestedTpopTfreeMatching(const StmtPtr& stmt, const FunctionPtr& func,
                                   std::vector<Diagnostic>& diagnostics) {
  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    VerifyTpopTfreeMatchingInBlock(FlattenBody(for_stmt->body_), func, diagnostics);
  } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    VerifyTpopTfreeMatchingInBlock(FlattenBody(if_stmt->then_body_), func, diagnostics);
    if (const auto& else_body = if_stmt->else_body_) {
      VerifyTpopTfreeMatchingInBlock(FlattenBody(*else_body), func, diagnostics);
    }
  } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    VerifyTpopTfreeMatchingInBlock(FlattenBody(while_stmt->body_), func, diagnostics);
  }
}

void VerifyTpopTfreeMatchingInBlock(const std::vector<StmtPtr>& stmts, const FunctionPtr& func,
                                    std::vector<Diagnostic>& diagnostics) {
  const std::string expected_tfree =
      (func->func_type_ == FunctionType::AIC) ? "system.tfree_to_aiv" : "system.tfree_to_aic";
  std::list<OutstandingTpop> outstanding_tpops;

  for (const auto& stmt : stmts) {
    VerifyNestedTpopTfreeMatching(stmt, func, diagnostics);

    VarPtr tpop_var;
    if (IsExpectedTpopAssignStmt(stmt, func->func_type_, &tpop_var)) {
      outstanding_tpops.push_back({tpop_var, dce::GetStmtOpName(stmt), stmt->span_});
      continue;
    }

    VarPtr tfree_var;
    std::string tfree_op_name;
    if (IsTfreeStmt(stmt, &tfree_var, &tfree_op_name)) {
      if (!tfree_var) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                 ((func->func_type_ == FunctionType::AIC) ? "AIC" : "AIV") +
                                     std::string(" function '") + func->name_ + "' has '" + tfree_op_name +
                                     "' without a tile value operand",
                                 stmt->span_);
        continue;
      }
      auto it = std::find_if(
          outstanding_tpops.begin(), outstanding_tpops.end(),
          [&](const OutstandingTpop& outstanding) { return outstanding.var.get() == tfree_var.get(); });
      if (it == outstanding_tpops.end()) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                 ((func->func_type_ == FunctionType::AIC) ? "AIC" : "AIV") +
                                     std::string(" function '") + func->name_ + "' has '" + tfree_op_name +
                                     "' without a matching outstanding tpop on the same tile value",
                                 stmt->span_);
        continue;
      }
      if (tfree_op_name != expected_tfree) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                 ((func->func_type_ == FunctionType::AIC) ? "AIC" : "AIV") +
                                     std::string(" function '") + func->name_ + "' must match " +
                                     it->op_name + " with '" + expected_tfree + "' on the same tile value",
                                 stmt->span_);
      } else {
        outstanding_tpops.erase(it);
      }
    }
  }

  for (const auto& outstanding : outstanding_tpops) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             ((func->func_type_ == FunctionType::AIC) ? "AIC" : "AIV") +
                                 std::string(" function '") + func->name_ + "' uses " + outstanding.op_name +
                                 " but has no matching '" + expected_tfree + "' call",
                             outstanding.span);
  }
}

void VerifyTpopTfreeMatching(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
  VerifyTpopTfreeMatchingInBlock(FlattenBody(func->body_), func, diagnostics);
}

}  // namespace

class MixedKernelExpandedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "MixedKernelExpanded"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ == FunctionType::InCore) {
        MixedKernelExpandedVerifier verifier(diagnostics, func->name_);
        verifier.VisitStmt(func->body_);
        verifier.CheckResult();
        continue;
      }
      if (func->func_type_ == FunctionType::AIC || func->func_type_ == FunctionType::AIV) {
        TpopMemoryVerifier verifier(diagnostics, func->name_, func->func_type_);
        verifier.VisitStmt(func->body_);
        VerifyReserveAndImportPeerBufferCoverage(func, diagnostics);
        VerifyCrossCorePipeSetup(func, diagnostics);
        VerifyInitializePipeOperands(func, diagnostics);
        VerifyTpopTfreeMatching(func, diagnostics);
      }
    }
  }
};

PropertyVerifierPtr CreateMixedKernelExpandedPropertyVerifier() {
  return std::make_shared<MixedKernelExpandedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
