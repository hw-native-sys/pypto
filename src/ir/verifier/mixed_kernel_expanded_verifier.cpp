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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/acc_push_epilogue.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/tpop_tfree_finalizer.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/reg_scope.h"
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
  if (call->HasKwarg("slot_num") && call->GetKwarg<int>("slot_num", 0) <= 0) {
    diagnostics.emplace_back(
        DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
        "Function '" + func_name + "': '" + op_name + "' 'slot_num' attribute must be positive when set",
        call->span_);
  }
  if (call->HasKwarg("local_slot_num")) {
    const int local_slot_num = call->GetKwarg<int>("local_slot_num", 0);
    if (local_slot_num <= 0) {
      diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                               "Function '" + func_name + "': '" + op_name +
                                   "' 'local_slot_num' attribute must be positive when set",
                               call->span_);
    } else {
      // When slot_num is omitted PTOAS derives it from dir_mask (8 unidirectional,
      // 4 bidirectional); validate local_slot_num against that effective value.
      const int effective_slot_num = call->HasKwarg("slot_num")
                                         ? call->GetKwarg<int>("slot_num", 0)
                                         : cross_core_pipe::GetPtoasImplicitSlotNum(dir_mask);
      if (local_slot_num > effective_slot_num) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                 "Function '" + func_name + "': '" + op_name + "' 'local_slot_num' (" +
                                     std::to_string(local_slot_num) + ") must be <= effective slot_num (" +
                                     std::to_string(effective_slot_num) + ")",
                                 call->span_);
      }
    }
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
  if (!IsOp(op, "system.aic_initialize_pipe") && !IsOp(op, "system.aiv_initialize_pipe")) return;
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
  if (IsOp(op, "system.reserve_buffer")) {
    ++reserve_count;
  } else if (IsOp(op, "system.import_peer_buffer")) {
    ++import_count;
  }
}

void AccumulateRequiredReserveImportFromInitializePipe(const CallPtr& call, FunctionType func_type,
                                                       int& required_reserve_count,
                                                       int& required_import_count) {
  if (!call || !call->op_) return;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op || (!IsOp(op, "system.aic_initialize_pipe") && !IsOp(op, "system.aiv_initialize_pipe"))) {
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
    if (func_type_ == FunctionType::AIC && IsOp(ir_op, "tile.tpop_from_aiv")) {
      expected_memory = MemorySpace::Mat;
    } else if (func_type_ == FunctionType::AIV && IsOp(ir_op, "tile.tpop_from_aic")) {
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

constexpr int kGuardFalse = 1, kGuardTrue = 2, kEitherGuard = kGuardFalse | kGuardTrue;

struct FixpipeEndpointCounts {
  std::size_t seen = 0, when_true = 0, when_false = 0;
  std::optional<std::size_t> guard_index;
};

struct FixpipeFunctionInfo : IRVisitor {
  bool has_fixpipe = false;
  std::map<int, FixpipeEndpointCounts> counts;
  std::map<int, CallPtr> inits;
  std::set<int> duplicate_ids;
  std::map<const Var*, CallPtr> definitions;
  std::map<int, std::vector<CallPtr>> pushes;
  std::map<int, std::vector<CallPtr>> pops;
  [[nodiscard]] CallPtr FindOrigin(const ExprPtr& operand) const {
    const auto def = definitions.find(AsVarLike(operand).get());
    return def == definitions.end() ? nullptr : def->second;
  }
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (auto call = As<Call>(stmt->value_)) {
      definitions[stmt->var_.get()] = call;
    } else if (auto origin = FindOrigin(stmt->value_)) {
      definitions[stmt->var_.get()] = origin;
    }
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "system.aic_initialize_pipe") || IsOp(call, "system.aiv_initialize_pipe")) {
      has_fixpipe |= call->HasKwarg("acc_push_epilogue");
      const int id = call->GetKwarg<int>("id", 0);
      auto [it, inserted] = inits.emplace(id, call);
      if (!inserted) {
        duplicate_ids.insert(id);
        if (!it->second->HasKwarg("acc_push_epilogue")) it->second = call;
      }
    } else if (IsOp(call, "tile.tpush_to_aiv")) {
      pushes[call->GetKwarg<int>("id", 0)].push_back(call);
    } else if (IsOp(call, "tile.tpop_from_aic")) {
      pops[call->GetKwarg<int>("id", 0)].push_back(call);
    }
    IRVisitor::VisitExpr_(call);
  }
};

void CountFixpipeEndpoints(const FunctionPtr& func, FixpipeFunctionInfo& local) {
  const std::string op = func->func_type_ == FunctionType::AIC ? "tile.tpush_to_aiv" : "tile.tpop_from_aic";
  std::map<const Var*, std::size_t> bool_params;
  for (std::size_t i = 0; i < func->params_.size(); ++i) {
    const auto param = func->params_[i];
    auto type = As<ScalarType>(param->GetType());
    if (type && type->dtype_ == DataType::BOOL) bool_params.emplace(param.get(), i);
  }
  auto count = [&](const StmtPtr& stmt, std::optional<std::size_t> guard, int mask) {
    auto call = transform_utils::GetCallFromStmt(stmt);
    if (!IsOp(call, op)) return;
    auto& flow = local.counts[call->GetKwarg<int>("id", 0)];
    if (guard && flow.guard_index && guard != flow.guard_index) return;
    if (guard) flow.guard_index = guard;
    ++flow.seen;
    if (mask & kGuardTrue) ++flow.when_true;
    if (mask & kGuardFalse) ++flow.when_false;
  };
  for (const auto& stmt : FlattenBody(func->body_)) {
    if (auto branch = As<IfStmt>(stmt)) {
      const auto literal = As<ConstBool>(branch->condition_);
      const auto input = bool_params.find(AsVarLike(branch->condition_).get());
      if (!literal && input == bool_params.end()) continue;
      const auto guard = literal ? std::nullopt : std::optional<std::size_t>(input->second);
      const int then_mask = literal ? (literal->value_ ? kEitherGuard : 0) : kGuardTrue;
      for (const auto& child : FlattenBody(branch->then_body_)) count(child, guard, then_mask);
      if (branch->else_body_) {
        const int else_mask = kEitherGuard ^ then_mask;
        for (const auto& child : FlattenBody(*branch->else_body_)) count(child, guard, else_mask);
      }
    } else {
      count(stmt, std::nullopt, kEitherGuard);
    }
  }
}

bool SameGroupArguments(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), [](const ExprPtr& a, const ExprPtr& b) {
    return a == b ||
           ((IsA<ConstBool>(a) || IsA<ConstInt>(a) || IsA<ConstFloat>(a)) && structural_equal(a, b));
  });
}

void VerifyFixpipeContracts(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) {
  const auto* context = PassContext::Current();
  const auto* handler =
      backend::BackendConfig::IsConfigured()
          ? (context ? context->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler())
          : nullptr;
  std::map<std::string, FixpipeFunctionInfo> info;
  using KernelPair = std::pair<std::string, std::string>;
  using LiteralTrace = std::pair<std::size_t, std::size_t>;
  std::map<LiteralTrace, std::size_t> trace_nodes;
  // Intern both polarities of each launch column. IDs 1 and 2 denote constants.
  auto extend = [&](std::size_t prefix, std::size_t next) {
    return trace_nodes.try_emplace(LiteralTrace{prefix, next}, trace_nodes.size() + 3).first->second;
  };
  auto append_literal = [&](std::optional<LiteralTrace>& trace, LiteralTrace value) {
    trace = trace && value.first ? std::make_optional(LiteralTrace{extend(trace->first, value.first),
                                                                   extend(trace->second, value.second)})
                                 : std::nullopt;
  };
  std::vector<std::pair<KernelPair, CallPtr>> groups;
  std::map<std::string, std::set<std::string>> group_peers;
  struct GuardArgument {
    const Var* origin = nullptr;
    LiteralTrace trace{};
  };
  std::map<const Var*, GuardArgument> group_bindings;
  std::map<const Var*, std::vector<std::pair<const Var*, std::size_t>>> launch_inputs;
  for (const auto& [gv, func] : program->functions_) {
    if (!func || !func->body_) continue;
    auto& local = info[func->name_];
    local.VisitStmt(func->body_);
    if (!local.pushes.empty() || !local.pops.empty()) CountFixpipeEndpoints(func, local);
    if (!IsWrapperType(func->func_type_) && !IsOrchestrationLike(func->func_type_)) continue;
    const auto members = CollectInnerCalls(func, program);
    if (func->func_type_ == FunctionType::Group) {
      const auto aiv = std::find_if(members.begin(), members.end(), [](const auto& member) {
        return member.inner_callee->func_type_ == FunctionType::AIV;
      });
      for (const auto& member : members) {
        if (member.inner_callee->func_type_ == FunctionType::AIC) {
          group_peers[member.inner_callee->name_].insert(aiv == members.end() ? ""
                                                                              : aiv->inner_callee->name_);
        }
      }
    }
    if (func->func_type_ == FunctionType::Group && members.size() == 2 &&
        SameGroupArguments(members[0].inner_call->args_, members[1].inner_call->args_)) {
      groups.emplace_back(std::minmax(members[0].inner_callee->name_, members[1].inner_callee->name_),
                          members[0].inner_call);
    }
    const std::set<VarPtr> parameters(func->params_.begin(), func->params_.end());
    for (const auto& member : members) {
      if (!IsWrapperType(member.inner_callee->func_type_)) continue;
      const auto& args = member.inner_call->args_;
      const auto& params = member.inner_callee->params_;
      for (std::size_t i = 0; i < std::min(args.size(), params.size()); ++i) {
        const auto* target = params[i].get();
        const auto type = As<ScalarType>(target->GetType());
        if (!type || type->dtype_ != DataType::BOOL) continue;
        const auto source = AsVarLike(args[i]);
        auto [it, inserted] = group_bindings.try_emplace(target, GuardArgument{source.get()});
        if (!inserted && it->second.origin != source.get()) it->second.origin = nullptr;
        const auto literal = As<ConstBool>(args[i]);
        const bool forwarded = IsWrapperType(func->func_type_) && parameters.count(source);
        launch_inputs[target].emplace_back(forwarded ? source.get() : nullptr,
                                           literal ? (literal->value_ ? kGuardTrue : kGuardFalse) : 0);
      }
    }
  }
  // Memoize each input column in call-site order. A forwarded input references its caller's column.
  std::function<LiteralTrace(const Var*)> resolve = [&](const Var* param) -> LiteralTrace {
    auto& trace = group_bindings[param].trace;
    auto inputs = launch_inputs.extract(param);  // A cycle or an unbound argument yields an empty trace.
    if (inputs.empty()) return trace;
    std::size_t mask = 0;
    std::optional<LiteralTrace> column = LiteralTrace{};
    for (const auto& [source, bit] : inputs.mapped()) {
      const auto value = source ? resolve(source) : LiteralTrace{bit, kEitherGuard ^ bit};
      if (!value.first) return {};
      mask |= value.first < kEitherGuard ? value.first : kEitherGuard;
      append_literal(column, value);
    }
    trace = mask == kEitherGuard ? *column : LiteralTrace{mask, kEitherGuard ^ mask};
    return trace;
  };
  while (!launch_inputs.empty()) resolve(launch_inputs.begin()->first);
  struct PeerArgument {
    int mask = 0;
    std::size_t representative = 0;
    std::optional<LiteralTrace> trace = LiteralTrace{};
  };
  // Intersect equal argument positions across Groups, each of which may use a different SSA value.
  std::map<KernelPair, std::vector<PeerArgument>> peer_bindings;
  for (const auto& [peers, member] : groups) {
    const auto& args = member->args_;
    auto& allowed = peer_bindings[peers];
    allowed.resize(std::max(allowed.size(), args.size()));
    std::map<std::size_t, std::map<const Var*, std::size_t>> classes;
    for (std::size_t i = 0; i < args.size(); ++i) {
      const auto value = As<ConstBool>(args[i]);
      const auto* var = AsVarLike(args[i]).get();
      const auto found = group_bindings.find(var);
      const auto known = found != group_bindings.end() ? found->second : GuardArgument{};
      const auto* identity = known.origin ? known.origin : var;
      auto& argument = allowed[i];
      argument.representative =
          identity ? classes[argument.representative].try_emplace(identity, i).first->second : i;
      const std::size_t bit = value && value->value_ ? kGuardTrue : kGuardFalse;
      const auto trace = value ? LiteralTrace{bit, kEitherGuard ^ bit} : known.trace;
      argument.mask |=
          trace.first && trace.first < kEitherGuard ? static_cast<int>(trace.first) : kEitherGuard;
      append_literal(argument.trace, trace);
    }
  }
  std::map<std::tuple<std::string, std::string, int>, int> incoming;
  for (const auto& [name, local] : info) {
    for (const auto& [id, init] : local.inits) {
      if (init->args_.empty()) continue;
      if (auto def = local.FindOrigin(init->args_[0]); IsOp(def, "system.import_peer_buffer")) {
        ++incoming[{def->GetKwarg<std::string>("peer_func", ""), def->GetKwarg<std::string>("name", ""), id}];
      }
    }
  }
  auto error = [&](const CallPtr& call, const std::string& message) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0, message, call->span_);
  };
  for (const auto& [gv, func] : program->functions_) {
    if (!func || !func->body_) continue;
    const auto& local = info.at(func->name_);
    std::map<const Call*, bool> reserve_users;
    for (const auto& [id, init] : local.inits) {
      const bool has_epilogue = init->HasKwarg("acc_push_epilogue");
      const auto e = init->GetKwarg<AccPushEpilogue>("acc_push_epilogue");
      if (has_epilogue) {
        if (init->HasKwarg("c2v_dtype") || init->GetKwarg<bool>("pre_relu", false)) {
          error(init, "acc_push_epilogue cannot be combined with c2v_dtype/pre_relu");
        }
        if (init->HasKwarg("local_slot_num")) {
          error(init, "local_slot_num is only supported on A2/A3; omit it on A5 Fixpipe pipes");
        }
        if (handler && !handler->SupportsC2VFixpipeEpilogue()) {
          error(init, "Cross-core Fixpipe epilogues currently require Ascend950 (A5); A2/A3 is not enabled");
        }
        const auto schema_error = ValidateAccPushEpilogue(e);
        if (!schema_error.empty()) {
          error(init, schema_error);
          continue;
        }
        if (local.duplicate_ids.count(id)) {
          error(init, "Fixpipe pipe id requires exactly one initialize_pipe");
        }
        if (e.at("layout") == "nz2nz" && e.at("quant") != "no_convert") {
          error(init, "NZ2NZ currently supports no_convert only; use NZ2ND/NZ2DN for quantization");
        }
        if (init->GetKwarg<int>("dir_mask", 0) != 1 || !init->GetKwarg<bool>("nosplit", false)) {
          error(init, "acc_push_epilogue requires C2V-only dir_mask=1 and nosplit=true");
        }
      }
      if (init->args_.empty()) continue;
      const auto def = local.FindOrigin(init->args_[0]);
      const bool reserve = IsOp(def, "system.reserve_buffer"),
                 imported = IsOp(def, "system.import_peer_buffer");
      if (has_epilogue && !(func->func_type_ == FunctionType::AIV ? reserve : imported)) {
        error(init,
              "Fixpipe c2v_consumer_buf must trace to reserve_buffer/import_peer_buffer on AIV/AIC "
              "respectively");
        continue;
      }
      if (!reserve && !imported) continue;
      if (reserve && (init->GetKwarg<int>("dir_mask", 0) & kDirMaskC2V) && local.pops.count(id)) {
        auto [usage, inserted] = reserve_users.emplace(def.get(), has_epilogue);
        if (!inserted && (usage->second || has_epilogue)) {
          error(init, "Fixpipe C2V pipe ids require distinct reserve_buffer allocations");
        }
        usage->second |= has_epilogue;
      }
      if (has_epilogue && reserve) {
        const auto name = def->GetKwarg<std::string>("name", "");
        if (incoming[{func->name_, name, id}] != 1) {
          error(init, "Fixpipe consumer requires exactly one peer producer importing its buffer and pipe id");
        }
        const int slots = init->GetKwarg<int>("slot_num", cross_core_pipe::GetPtoasImplicitSlotNum(1));
        const int local_slots = handler ? handler->GetC2VConsumerRingSlotCount(slots, slots) : slots;
        const int64_t required = static_cast<int64_t>(init->GetKwarg<int>("slot_size", 0)) * local_slots;
        if (required > def->GetKwarg<int>("size", 0)) {
          error(init, "Fixpipe reserve_buffer requires at least " + std::to_string(required) +
                          " bytes for its local ring");
        }
      }
      if (!IsOp(def, "system.import_peer_buffer")) continue;
      const auto peer_name = def->GetKwarg<std::string>("peer_func", "");
      const auto members = group_peers.find(func->name_);
      if (has_epilogue && members != group_peers.end() &&
          (members->second.size() != 1 || !members->second.count(peer_name))) {
        error(def, "Fixpipe peer_func must belong to every Group dispatching its producer");
      }
      auto peer = info.find(peer_name);
      if (peer == info.end() || !peer->second.inits.count(id)) {
        if (has_epilogue) {
          error(init, "Fixpipe requires a matching peer initialize_pipe with the same pipe id");
        }
        continue;
      }
      const auto peer_init = peer->second.inits.at(id);
      if (!has_epilogue && !peer_init->HasKwarg("acc_push_epilogue")) continue;
      if (!has_epilogue || !peer_init->HasKwarg("acc_push_epilogue") ||
          peer_init->GetKwarg<AccPushEpilogue>("acc_push_epilogue") != e ||
          peer_init->GetKwarg<bool>("nosplit", false) != init->GetKwarg<bool>("nosplit", false) ||
          peer_init->GetKwarg<int>("slot_size", 0) != init->GetKwarg<int>("slot_size", 0) ||
          peer_init->GetKwarg<int>("slot_num", cross_core_pipe::GetPtoasImplicitSlotNum(1)) !=
              init->GetKwarg<int>("slot_num", cross_core_pipe::GetPtoasImplicitSlotNum(1))) {
        error(init, "Fixpipe peer epilogue/nosplit/slot_size contract mismatch");
        continue;
      }
      const auto peer_def = peer_init->args_.empty() ? nullptr : peer->second.FindOrigin(peer_init->args_[0]);
      if (!IsOp(peer_def, "system.reserve_buffer") ||
          peer_def->GetKwarg<std::string>("name", "") != def->GetKwarg<std::string>("name", "")) {
        error(init, "Fixpipe peer init must use the imported named reserve_buffer");
      }
      auto pushes = local.pushes.find(id);
      auto pops = peer->second.pops.find(id);
      if ((pushes == local.pushes.end()) != (pops == peer->second.pops.end())) {
        error(init, "Fixpipe requires matching TPUSH and TPOP endpoints for the same pipe id");
      }
      if (pushes != local.pushes.end() && pops != peer->second.pops.end()) {
        const auto push_count = local.counts.find(id);
        const auto pop_count = peer->second.counts.find(id);
        if (push_count != local.counts.end() && pop_count != peer->second.counts.end()) {
          const auto& push = push_count->second;
          const auto& pop = pop_count->second;
          const auto pair = peer_bindings.find(std::minmax(func->name_, peer_name));
          auto argument = [&](const FixpipeEndpointCounts& flow) -> const PeerArgument* {
            return pair != peer_bindings.end() && flow.guard_index && *flow.guard_index < pair->second.size()
                       ? &pair->second[*flow.guard_index]
                       : nullptr;
          };
          const auto [push_arg, pop_arg] = std::make_pair(argument(push), argument(pop));
          auto counts = [](const FixpipeEndpointCounts& flow, const PeerArgument* binding) {
            const auto mask = binding ? binding->mask : kEitherGuard;
            return std::make_pair(mask == kGuardFalse ? flow.when_false : flow.when_true,
                                  mask == kGuardTrue ? flow.when_true : flow.when_false);
          };
          const auto p = counts(push, push_arg), q = counts(pop, pop_arg);
          const bool shared = push_arg && pop_arg && push_arg->representative == pop_arg->representative;
          const bool comparable = p.first == p.second || q.first == q.second || shared;
          const auto p_range = std::minmax(p.first, p.second), q_range = std::minmax(q.first, q.second);
          const bool literals = push_arg && pop_arg && push_arg->trace && pop_arg->trace;
          const bool literal_match =
              literals && ((p == q && push_arg->trace->first == pop_arg->trace->first) ||
                           (p.first == q.second && p.second == q.first &&
                            push_arg->trace->first == pop_arg->trace->second));
          const bool mismatch = comparable ? p != q
                                           : (literals && !literal_match) || p_range.second < q_range.first ||
                                                 q_range.second < p_range.first;
          if (push.seen == pushes->second.size() && pop.seen == pops->second.size() && mismatch) {
            const bool excess_pop =
                comparable ? p.first < q.first || p.second < q.second : p_range.second < q_range.first;
            error(excess_pop ? pops->second.front() : pushes->second.front(),
                  "Fixpipe TPUSH and TPOP endpoint counts differ across control-flow paths");
          }
        }
        std::set<std::vector<int64_t>> source_signatures;
        auto signature = [](const TileTypePtr& tile, DataType dtype) {
          std::vector<int64_t> result{dtype.Code()};
          for (const auto& dim : tile->shape_) {
            const auto value = cross_core_pipe::TryGetNonNegativeConstInt(dim);
            result.push_back(value.value_or(-1));
          }
          return result;
        };
        for (const auto& push : pushes->second) {
          auto src = push->args_.empty() ? nullptr : As<TileType>(push->args_[0]->GetType());
          if (src) source_signatures.insert(signature(src, AccPushResultDtype(e.at("quant"), src->dtype_)));
        }
        if (source_signatures.size() > 1) {
          error(init,
                "Current PTOAS requires one physical payload shape/converted dtype per Fixpipe pipe id; "
                "use separate explicit pipe ids for different payloads");
        }
        for (const auto& pop : pops->second) {
          auto dst = As<TileType>(pop->GetType());
          if (dst &&
              (source_signatures.size() != 1 || !source_signatures.count(signature(dst, dst->dtype_)))) {
            error(pop, "Fixpipe producer/consumer shape or converted dtype mismatch");
          }
        }
      }
    }
    // After peer checks, validate state before Buffer IR lowering. Each block
    // inherits and restores inits and starts fresh quant state. A nested statement
    // invalidates its parent's setters. Retained slots stay shared across blocks,
    // and block-local TPOP/TFREE matching requires each block to free its own pops.
    using InitMap = std::map<int, CallPtr>;
    std::map<int, std::deque<const Var*>> retained;
    std::function<void(const std::vector<StmtPtr>&, InitMap&)> verify_block;
    verify_block = [&](const std::vector<StmtPtr>& stmts, InitMap& visible) {
      std::vector<std::pair<int, std::optional<CallPtr>>> saved_inits;
      std::optional<int> quant_id;
      int64_t vector_width = 0;  // A scalar configuration has no vector controls.
      for (const auto& stmt : stmts) {
        auto call = transform_utils::GetCallFromStmt(stmt);
        if (!call) {
          if (auto loop = As<ForStmt>(stmt)) {
            verify_block(FlattenBody(loop->body_), visible);
          } else if (auto loop = As<WhileStmt>(stmt)) {
            verify_block(FlattenBody(loop->body_), visible);
          } else if (auto branch = As<IfStmt>(stmt)) {
            verify_block(FlattenBody(branch->then_body_), visible);
            if (branch->else_body_) verify_block(FlattenBody(*branch->else_body_), visible);
          } else {
            continue;
          }
          quant_id.reset();
          continue;
        }
        if ((IsOp(call, "tile.store") || IsOp(call, "tile.assemble")) && call->HasKwarg("pre_quant")) {
          quant_id.reset();
        }
        if (auto result = As<TileType>(call->GetType());
            result && result->memory_space_ == MemorySpace::Scaling && vector_width > 0) {
          quant_id.reset();
        }
        const int id = call->GetKwarg<int>("id", 0);
        if (IsOp(call, "system.aic_initialize_pipe") || IsOp(call, "system.aiv_initialize_pipe")) {
          auto previous = visible.find(id);
          saved_inits.emplace_back(
              id, previous == visible.end() ? std::nullopt : std::optional<CallPtr>(previous->second));
          visible[id] = call;
          if (quant_id == id) quant_id.reset();
          continue;
        }
        if (IsOp(call, "system.tfree_to_aic") && !call->args_.empty()) {
          const auto origin = local.FindOrigin(call->args_[0]);
          if (!IsOp(origin, "tile.tpop_from_aic")) continue;
          auto& fifo = retained[origin->GetKwarg<int>("id", 0)];
          if (fifo.empty()) continue;
          if (fifo.front() != AsVarLike(call->args_[0]).get()) {
            error(call, "Fixpipe TFREE must release the oldest outstanding TPOP of its pipe");
          } else {
            fifo.pop_front();
          }
          continue;
        }
        const bool scalar = IsOp(call, "system.set_quant_scalar"),
                   vector = IsOp(call, "system.set_quant_vector");
        const bool push = IsOp(call, "tile.tpush_to_aiv"), pop = IsOp(call, "tile.tpop_from_aic");
        if (!scalar && !vector && !push && !pop) continue;
        const auto found = visible.find(id);
        if (found == visible.end()) {
          if (scalar || vector || local.has_fixpipe) {
            error(call, "Fixpipe operation requires a dominating initialize_pipe for its pipe id");
          }
          continue;
        }
        const auto init = found->second;
        if (!init->HasKwarg("acc_push_epilogue")) {
          if (scalar || vector) {
            error(call, "set_quant requires an acc_push_epilogue on the matching pipe id");
          }
          continue;
        }
        const auto e = init->GetKwarg<AccPushEpilogue>("acc_push_epilogue");
        if (!ValidateAccPushEpilogue(e).empty()) continue;
        const auto quant = e.at("quant");
        if (scalar || vector) {
          quant_id.reset();
          if ((scalar && !IsScalarAccPushQuant(quant)) || (vector && !IsVectorAccPushQuant(quant))) {
            error(call, "set_quant kind does not match acc_push_epilogue.quant");
            continue;
          }
          try {
            OpRegistry::GetInstance().GetEntry(call->op_->name_).GetDeduceType()(call->args_, call->kwargs_);
            quant_id = id;
            vector_width =
                vector ? As<ConstInt>(As<TileType>(call->args_[0]->GetType())->shape_[1])->value_ : 0;
          } catch (const Error& exception) {
            error(call, exception.what());
          }
          continue;
        }
        if (call->GetKwarg<int>("split", 0) != 0) error(call, "Fixpipe transfers require split=0");
        if (pop && !call->args_.empty()) error(call, "Fixpipe TPOP takes no operands");
        if (auto assign = As<AssignStmt>(stmt); pop && assign) {
          auto& fifo = retained[id];
          fifo.push_back(assign->var_.get());
          const int slots = init->GetKwarg<int>("slot_num", cross_core_pipe::GetPtoasImplicitSlotNum(1));
          if (fifo.size() > static_cast<std::size_t>(std::max(0, slots))) {
            error(call, "Fixpipe outstanding TPOP count exceeds slot_num");
          }
        }
        if (push && (IsScalarAccPushQuant(quant) || IsVectorAccPushQuant(quant)) && quant_id != id) {
          error(call, "Fixpipe TPUSH requires preceding set_quant with the same pipe id in the same block");
        }
        auto tile = push ? (call->args_.size() == 1 ? As<TileType>(call->args_[0]->GetType()) : nullptr)
                         : As<TileType>(call->GetType());
        if (!tile) {
          if (push) error(call, "Fixpipe TPUSH requires exactly one tile operand");
          continue;
        }
        if (push && IsVectorAccPushQuant(quant) && quant_id == id) {
          auto columns = tile->shape_.size() == 2
                             ? cross_core_pipe::TryGetNonNegativeConstInt(tile->shape_[1])
                             : std::nullopt;
          if (!columns || vector_width < *columns) {
            error(call, "Fixpipe vector controls do not cover the producer columns");
          }
        }
        const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
        if (push) {
          if (tile->memory_space_ != MemorySpace::Acc) {
            error(call, "Fixpipe TPUSH requires an Acc source");
          }
          if (e.at("layout") == "nz2nz" && tile->dtype_ != DataType::FP32) {
            error(call, "NZ2NZ no_convert currently requires an FP32 source");
          }
          if ((quant == "no_convert" && tile->dtype_ != DataType::FP32 && tile->dtype_ != DataType::INT32) ||
              (quant != "no_convert" && tile->dtype_ != AccPushSourceDtype(quant))) {
            error(call, "Fixpipe producer dtype does not match quant mode");
          }
          if (view.blayout != TileLayout::col_major || view.slayout != TileLayout::row_major ||
              view.fractal != tile_view_semantics::kAccFractal) {
            error(call, "Fixpipe producer requires NZ layout with fractal=1024");
          }
        } else {
          const auto layout = e.at("layout");
          const auto block = layout == "nz2nd" ? TileLayout::row_major : TileLayout::col_major;
          const auto scatter = layout == "nz2nz" ? TileLayout::row_major : TileLayout::none_box;
          if (view.blayout != block || view.slayout != scatter) {
            error(call, "Fixpipe consumer layout does not match acc_push_epilogue.layout");
          }
          if (quant != "no_convert" && tile->dtype_ != AccPushResultDtype(quant, DataType::FP32)) {
            error(call, "Fixpipe consumer dtype does not match quant mode");
          }
        }
        auto bytes = cross_core_pipe::TryGetTileSlotSizeBytes(tile);
        if (bytes && push && tile->dtype_.GetBit() == 32) {
          const auto result_bytes =
              static_cast<int64_t>(AccPushResultDtype(quant, tile->dtype_).GetBit() / 8);
          *bytes = (*bytes / 4) * result_bytes;
        }
        if (!bytes || *bytes > init->GetKwarg<int>("slot_size", 0)) {
          error(call, "Fixpipe converted tile exceeds slot_size or has a non-static footprint");
        }
      }
      for (auto saved = saved_inits.rbegin(); saved != saved_inits.rend(); ++saved) {
        if (saved->second) {
          visible[saved->first] = *saved->second;
        } else {
          visible.erase(saved->first);
        }
      }
    };
    InitMap visible;
    verify_block(FlattenBody(func->body_), visible);
  }
}

}  // namespace

class MixedKernelExpandedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "MixedKernelExpanded"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    VerifyFixpipeContracts(program, diagnostics);
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      VerifyRegScopes(func, diagnostics, true);
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
