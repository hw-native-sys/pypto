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
#include <cstdint>
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
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/tpop_tfree_finalizer.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
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

// Fixpipe contracts are checked before Buffer IR lowering, while both peers and
// lexical quantization-state dominance are still explicit in the tile IR.
struct FixpipeFunctionInfo : IRVisitor {
  std::map<int, CallPtr> inits;
  std::set<int> duplicate_ids;
  std::map<const Var*, CallPtr> definitions;
  std::map<int, std::vector<CallPtr>> pushes;
  std::map<int, std::vector<CallPtr>> pops;
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (auto call = As<Call>(stmt->value_)) {
      definitions[stmt->var_.get()] = call;
    } else if (auto alias = AsVarLike(stmt->value_)) {
      // SSA definitions precede their uses. Cache the terminal call at each
      // identity alias so peer lookups remain O(log N), even for shared chains.
      auto origin = definitions.find(alias.get());
      if (origin != definitions.end()) definitions[stmt->var_.get()] = origin->second;
    }
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "system.aic_initialize_pipe") || IsOp(call, "system.aiv_initialize_pipe")) {
      const int id = call->GetKwarg<int>("id", 0);
      auto [it, inserted] = inits.emplace(id, call);
      if (!inserted) {
        duplicate_ids.insert(id);
        // Keep an epilogue representative: a later legacy init must not hide
        // the duplicate-id diagnostic or peer contract for an earlier Fixpipe.
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

// Prove count mismatches for straight-line peers and top-level branches on one
// boolean input. Loops and other guards can have correlated execution counts.
std::optional<std::pair<size_t, size_t>> FixpipeEndpointCountRange(const FunctionPtr& func,
                                                                   const std::vector<CallPtr>& sites) {
  const std::set<CallPtr> endpoints(sites.begin(), sites.end());
  auto count = [&](const StmtPtr& body) {
    const auto stmts = FlattenBody(body);
    return std::count_if(stmts.begin(), stmts.end(), [&](const auto& stmt) {
      return endpoints.count(transform_utils::GetCallFromStmt(stmt));
    });
  };
  size_t seen = static_cast<size_t>(count(func->body_));
  size_t on_true = seen;
  size_t on_false = seen;
  VarPtr guard;
  for (const auto& stmt : FlattenBody(func->body_)) {
    auto branch = As<IfStmt>(stmt);
    if (!branch) continue;
    const auto then_count = static_cast<size_t>(count(branch->then_body_));
    const auto else_count = branch->else_body_ ? static_cast<size_t>(count(*branch->else_body_)) : 0;
    seen += then_count + else_count;
    if (then_count != else_count) {
      auto param = AsVarLike(branch->condition_);
      auto type = param ? As<ScalarType>(param->GetType()) : nullptr;
      if (!type || type->dtype_ != DataType::BOOL || (guard && param != guard) ||
          std::find(func->params_.begin(), func->params_.end(), param) == func->params_.end()) {
        return std::nullopt;
      }
      guard = param;
    }
    on_true += then_count;
    on_false += else_count;
  }
  if (seen != sites.size()) return std::nullopt;
  return std::minmax(on_true, on_false);
}

void VerifyFixpipeContracts(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) {
  const auto* context = PassContext::Current();
  const auto* handler =
      backend::BackendConfig::IsConfigured()
          ? (context ? context->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler())
          : nullptr;
  std::map<std::string, FixpipeFunctionInfo> info;
  for (const auto& [gv, func] : program->functions_) {
    if (func && func->body_) info[func->name_].VisitStmt(func->body_);
  }
  std::map<std::tuple<std::string, std::string, int>, int> incoming;
  for (const auto& [name, local] : info) {
    for (const auto& [id, init] : local.inits) {
      if (init->args_.empty()) continue;
      auto buffer = AsVarLike(init->args_[0]);
      auto def = buffer ? local.definitions.find(buffer.get()) : local.definitions.end();
      if (def != local.definitions.end() && IsOp(def->second, "system.import_peer_buffer")) {
        ++incoming[{def->second->GetKwarg<std::string>("peer_func", ""),
                    def->second->GetKwarg<std::string>("name", ""), id}];
      }
    }
  }
  auto error = [&](const CallPtr& call, const std::string& message) {
    diagnostics.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0, message, call->span_);
  };
  for (const auto& [gv, func] : program->functions_) {
    if (!func || !func->body_) continue;
    const auto& local = info.at(func->name_);
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
      }
      if (has_epilogue &&
          (init->GetKwarg<int>("dir_mask", 0) != 1 || !init->GetKwarg<bool>("nosplit", false))) {
        error(init, "acc_push_epilogue requires C2V-only dir_mask=1 and nosplit=true");
      }
      if (init->args_.empty()) continue;
      auto buffer = AsVarLike(init->args_[0]);
      auto def = buffer ? local.definitions.find(buffer.get()) : local.definitions.end();
      if (def == local.definitions.end() ||
          (!IsOp(def->second, "system.reserve_buffer") && !IsOp(def->second, "system.import_peer_buffer"))) {
        if (has_epilogue) {
          error(init, "Fixpipe c2v_consumer_buf must trace to reserve_buffer/import_peer_buffer");
        }
        continue;
      }
      if (has_epilogue && IsOp(def->second, "system.reserve_buffer")) {
        const auto name = def->second->GetKwarg<std::string>("name", "");
        if (incoming[{func->name_, name, id}] != 1) {
          error(init, "Fixpipe consumer requires exactly one peer producer importing its buffer and pipe id");
        }
        const int slots = init->GetKwarg<int>("slot_num", cross_core_pipe::GetPtoasImplicitSlotNum(1));
        const int requested_local_slots = init->GetKwarg<int>("local_slot_num", slots);
        const int local_slots = handler ? handler->GetC2VConsumerRingSlotCount(slots, requested_local_slots)
                                        : requested_local_slots;
        const int64_t required = static_cast<int64_t>(init->GetKwarg<int>("slot_size", 0)) * local_slots;
        if (required > def->second->GetKwarg<int>("size", 0)) {
          error(init, "Fixpipe reserve_buffer requires at least " + std::to_string(required) +
                          " bytes for its local ring");
        }
      }
      if (!IsOp(def->second, "system.import_peer_buffer")) continue;
      const auto peer_name = def->second->GetKwarg<std::string>("peer_func", "");
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
      auto peer_buf = peer_init->args_.empty() ? nullptr : AsVarLike(peer_init->args_[0]);
      auto peer_def =
          peer_buf ? peer->second.definitions.find(peer_buf.get()) : peer->second.definitions.end();
      if (peer_def == peer->second.definitions.end() || !IsOp(peer_def->second, "system.reserve_buffer") ||
          peer_def->second->GetKwarg<std::string>("name", "") !=
              def->second->GetKwarg<std::string>("name", "")) {
        error(init, "Fixpipe peer init must use the imported named reserve_buffer");
      }
      auto pushes = local.pushes.find(id);
      auto pops = peer->second.pops.find(id);
      if ((pushes == local.pushes.end()) != (pops == peer->second.pops.end())) {
        error(init, "Fixpipe requires matching TPUSH and TPOP endpoints for the same pipe id");
      }
      if (pushes != local.pushes.end() && pops != peer->second.pops.end()) {
        auto push_range = FixpipeEndpointCountRange(func, pushes->second);
        auto pop_range = FixpipeEndpointCountRange(program->GetFunction(peer_name), pops->second);
        if (push_range && pop_range && *push_range != *pop_range) {
          const bool excess_pop =
              push_range->first < pop_range->first || push_range->second < pop_range->second;
          error(excess_pop ? pops->second.front() : pushes->second.front(),
                "Fixpipe TPUSH and TPOP endpoint counts differ across control-flow paths");
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
    using InitMap = std::map<int, CallPtr>;
    std::function<void(const std::vector<StmtPtr>&, InitMap&)> verify_block;
    verify_block = [&](const std::vector<StmtPtr>& stmts, InitMap& visible) {
      std::vector<std::pair<int, std::optional<CallPtr>>> saved_inits;
      // PTOAS quant state must precede TPUSH in the same block, with the same id.
      std::set<int> scalar_state;
      std::map<int, int64_t> vector_state;
      for (const auto& stmt : stmts) {
        auto assign = As<AssignStmt>(stmt);
        auto eval = As<EvalStmt>(stmt);
        auto call = assign ? As<Call>(assign->value_) : (eval ? As<Call>(eval->expr_) : nullptr);
        if (call) {
          if ((IsOp(call, "tile.store") || IsOp(call, "tile.assemble")) && call->HasKwarg("pre_quant")) {
            scalar_state.clear();
            vector_state.clear();
          }
          if (auto result = As<TileType>(call->GetType());
              result && result->memory_space_ == MemorySpace::Scaling) {
            vector_state.clear();
          }
          const int id = call->GetKwarg<int>("id", 0);
          if (IsOp(call, "system.aic_initialize_pipe") || IsOp(call, "system.aiv_initialize_pipe")) {
            auto previous = visible.find(id);
            saved_inits.emplace_back(
                id, previous == visible.end() ? std::nullopt : std::optional<CallPtr>(previous->second));
            visible[id] = call;
            scalar_state.erase(id);
            vector_state.erase(id);
          } else {
            const bool scalar = IsOp(call, "system.set_quant_scalar");
            const bool vector = IsOp(call, "system.set_quant_vector");
            const bool push = IsOp(call, "tile.tpush_to_aiv");
            const bool pop = IsOp(call, "tile.tpop_from_aic");
            const bool free = IsOp(call, "system.tfree_to_aic");
            if (scalar || vector || push || pop || free) {
              auto it = visible.find(id);
              if (it == visible.end()) {
                if (scalar || vector ||
                    (local.inits.count(id) && local.inits.at(id)->HasKwarg("acc_push_epilogue"))) {
                  error(call, "Fixpipe operation requires a dominating initialize_pipe for its pipe id");
                }
              } else if (!it->second->HasKwarg("acc_push_epilogue")) {
                if (scalar || vector) {
                  error(call, "set_quant requires an acc_push_epilogue on the matching pipe id");
                }
              } else {
                const auto init = it->second;
                const auto e = init->GetKwarg<AccPushEpilogue>("acc_push_epilogue");
                if (!ValidateAccPushEpilogue(e).empty()) continue;
                const auto quant = e.at("quant");
                if (scalar || vector) {
                  scalar_state.clear();
                  vector_state.clear();
                  if ((scalar && !IsScalarAccPushQuant(quant)) || (vector && !IsVectorAccPushQuant(quant))) {
                    error(call, "set_quant kind does not match acc_push_epilogue.quant");
                  } else {
                    try {
                      // Raw calls can bypass the operator builder's operand validation.
                      OpRegistry::GetInstance()
                          .GetEntry(call->op_->name_)
                          .GetDeduceType()(call->args_, call->kwargs_);
                      if (scalar) {
                        scalar_state.insert(id);
                      } else {
                        auto payload = As<TileType>(call->args_[0]->GetType());
                        vector_state[id] = As<ConstInt>(payload->shape_[1])->value_;
                      }
                    } catch (const Error& exception) {
                      error(call, exception.what());
                    }
                  }
                } else {
                  if (call->GetKwarg<int>("split", 0) != 0) error(call, "Fixpipe transfers require split=0");
                  if (push && ((IsScalarAccPushQuant(quant) && !scalar_state.count(id)) ||
                               (IsVectorAccPushQuant(quant) && !vector_state.count(id)))) {
                    error(
                        call,
                        "Fixpipe TPUSH requires preceding set_quant with the same pipe id in the same block");
                  }
                  auto tile = push ? (call->args_.empty() ? nullptr : As<TileType>(call->args_[0]->GetType()))
                                   : (pop ? As<TileType>(call->GetType()) : nullptr);
                  if (tile && (push || pop)) {
                    if (push && IsVectorAccPushQuant(quant) && vector_state.count(id)) {
                      auto columns = tile->shape_.size() == 2
                                         ? cross_core_pipe::TryGetNonNegativeConstInt(tile->shape_[1])
                                         : std::nullopt;
                      if (!columns || vector_state.at(id) < *columns) {
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
                      if ((quant == "no_convert" && tile->dtype_ != DataType::FP32 &&
                           tile->dtype_ != DataType::INT32) ||
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
                      if (quant != "no_convert" &&
                          tile->dtype_ != AccPushResultDtype(quant, DataType::FP32)) {
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
                }
              }
            }
          }
        }
        if (auto loop = As<ForStmt>(stmt)) {
          verify_block(FlattenBody(loop->body_), visible);
        } else if (auto loop = As<WhileStmt>(stmt)) {
          verify_block(FlattenBody(loop->body_), visible);
        } else if (auto branch = As<IfStmt>(stmt)) {
          verify_block(FlattenBody(branch->then_body_), visible);
          if (branch->else_body_) verify_block(FlattenBody(*branch->else_body_), visible);
        }
        // A nested block may change the hardware quant registers on either path.
        // Require an explicit reconfiguration before a subsequent enclosing push.
        if (IsA<ForStmt>(stmt) || IsA<WhileStmt>(stmt) || IsA<IfStmt>(stmt)) {
          scalar_state.clear();
          vector_state.clear();
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
