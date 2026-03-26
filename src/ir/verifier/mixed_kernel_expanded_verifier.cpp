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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/tpop_chain_normalizer.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using core_affinity::ClassifyCallAffinity;
using core_affinity::CoreAffinity;
using cross_core_pipe::CollectCrossCorePipeMetadata;
using cross_core_pipe::CollectDominatingPipeSetupMetadata;
using cross_core_pipe::CrossCorePipeMetadata;
using cross_core_pipe::FormatObservedSlotSizes;
using cross_core_pipe::PipeDirectionMetadata;
using tpop_chain::IsExpectedTpopAssignStmt;
using tpop_chain::IsTfreeStmt;
using tpop_chain::StmtReferencesVar;

namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

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
    } else if (affinity == CoreAffinity::BOUNDARY) {
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
      bool valid = tile_type && tile_type->memory_space_.has_value() &&
                   tile_type->memory_space_.value() == expected_memory.value();
      if (!valid) {
        std::string func_kind = (func_type_ == FunctionType::AIC) ? "AIC" : "AIV";
        std::string actual_memory = (tile_type && tile_type->memory_space_.has_value())
                                        ? MemorySpaceToString(tile_type->memory_space_.value())
                                        : "unset";
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
            func_kind + " function '" + func_name_ + "' requires " + ir_op->name_ +
                " result in MemorySpace::" + MemorySpaceToString(expected_memory.value()) +
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

  auto report_slot_issue = [&](const std::string& issue, const PipeDirectionMetadata& direction,
                               const std::string& direction_name) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             "Function '" + func->name_ + "' uses " + direction_name + " cross-core tiles " +
                                 issue + ": " + FormatObservedSlotSizes(direction.observed_slot_sizes),
                             func->span_);
  };

  if (metadata.c2v.has_inconsistent_slot_size) {
    report_slot_issue("with inconsistent slot sizes", metadata.c2v, "C2V");
  }
  if (metadata.v2c.has_inconsistent_slot_size) {
    report_slot_issue("with inconsistent slot sizes", metadata.v2c, "V2C");
  }
  if ((metadata.c2v.has_ops && !metadata.c2v.slot_size_bytes.has_value()) ||
      (metadata.v2c.has_ops && !metadata.v2c.slot_size_bytes.has_value())) {
    diagnostics.emplace_back(
        DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
        "Function '" + func->name_ +
            "' uses cross-core tile ops with non-static tile size; auto pipe setup requires "
            "statically known tile shapes",
        func->span_);
  }
  if (metadata.c2v.has_ops && metadata.v2c.has_ops && metadata.c2v.slot_size_bytes.has_value() &&
      metadata.v2c.slot_size_bytes.has_value() &&
      metadata.c2v.slot_size_bytes.value() != metadata.v2c.slot_size_bytes.value()) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             "Function '" + func->name_ +
                                 "' uses bidirectional cross-core tiles with different "
                                 "slot sizes (C2V=" +
                                 std::to_string(metadata.c2v.slot_size_bytes.value()) +
                                 ", V2C=" + std::to_string(metadata.v2c.slot_size_bytes.value()) +
                                 "); single initialize_pipe slot_size is unsupported",
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

void VerifyTpopTfreeOrderInBlock(const std::vector<StmtPtr>& stmts, const FunctionPtr& func,
                                 std::vector<Diagnostic>& diagnostics);

void VerifyNestedTpopTfreeOrder(const StmtPtr& stmt, const FunctionPtr& func,
                                std::vector<Diagnostic>& diagnostics) {
  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    VerifyTpopTfreeOrderInBlock(FlattenBody(for_stmt->body_), func, diagnostics);
  } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    VerifyTpopTfreeOrderInBlock(FlattenBody(if_stmt->then_body_), func, diagnostics);
    if (if_stmt->else_body_.has_value()) {
      VerifyTpopTfreeOrderInBlock(FlattenBody(if_stmt->else_body_.value()), func, diagnostics);
    }
  } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    VerifyTpopTfreeOrderInBlock(FlattenBody(while_stmt->body_), func, diagnostics);
  }
}

void VerifyTpopTfreeOrderInBlock(const std::vector<StmtPtr>& stmts, const FunctionPtr& func,
                                 std::vector<Diagnostic>& diagnostics) {
  const std::string expected_tfree =
      (func->func_type_ == FunctionType::AIC) ? "system.tfree_to_aiv" : "system.tfree_to_aic";
  VarPtr open_tpop_var;
  std::string open_tpop_op_name;
  const Span* open_tpop_span = &func->span_;

  for (const auto& stmt : stmts) {
    VerifyNestedTpopTfreeOrder(stmt, func, diagnostics);

    VarPtr tpop_var;
    if (IsExpectedTpopAssignStmt(stmt, func->func_type_, &tpop_var)) {
      if (open_tpop_var) {
        diagnostics.emplace_back(
            DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
            "Function '" + func->name_ +
                "' must order cross-core tpop chains as 'tpop -> use -> tfree -> next tpop'",
            stmt->span_);
      }
      open_tpop_var = tpop_var;
      open_tpop_span = &stmt->span_;
      open_tpop_op_name = dce::GetStmtOpName(stmt);
      continue;
    }

    VarPtr tfree_var;
    std::string tfree_op_name;
    if (IsTfreeStmt(stmt, &tfree_var, &tfree_op_name)) {
      if (!open_tpop_var) {
        continue;
      }
      if (tfree_op_name != expected_tfree || !tfree_var || tfree_var.get() != open_tpop_var.get()) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                 ((func->func_type_ == FunctionType::AIC) ? "AIC" : "AIV") +
                                     std::string(" function '") + func->name_ + "' must match " +
                                     open_tpop_op_name + " with '" + expected_tfree +
                                     "' on the same tile value",
                                 stmt->span_);
      } else {
        open_tpop_var.reset();
      }
      continue;
    }

    if (open_tpop_var && !StmtReferencesVar(stmt, open_tpop_var.get())) {
      diagnostics.emplace_back(
          DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
          "Function '" + func->name_ +
              "' must order cross-core tpop chains as 'tpop -> use -> tfree -> next tpop'",
          stmt->span_);
      open_tpop_var.reset();
    }
  }

  if (open_tpop_var) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                             ((func->func_type_ == FunctionType::AIC) ? "AIC" : "AIV") +
                                 std::string(" function '") + func->name_ + "' uses " + open_tpop_op_name +
                                 " but has no matching '" + expected_tfree + "' call",
                             *open_tpop_span);
  }
}

void VerifyTpopTfreeOrder(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) {
  VerifyTpopTfreeOrderInBlock(FlattenBody(func->body_), func, diagnostics);
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
        VerifyCrossCorePipeSetup(func, diagnostics);
        VerifyTpopTfreeOrder(func, diagnostics);
      }
    }
  }
};

PropertyVerifierPtr CreateMixedKernelExpandedPropertyVerifier() {
  return std::make_shared<MixedKernelExpandedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
