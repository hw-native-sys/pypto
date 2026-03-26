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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_LOOP_STATE_REPAIR_H_
#define PYPTO_IR_TRANSFORMS_UTILS_LOOP_STATE_REPAIR_H_

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace loop_repair {

// --- Statement construction helpers ---

StmtPtr MakeBody(const std::vector<StmtPtr>& stmts, const Span& span);

// --- Compound statement rebuild helpers ---

StmtPtr RebuildForStmt(const std::shared_ptr<const ForStmt>& f, const StmtPtr& new_body);

StmtPtr RebuildForStmt(const std::shared_ptr<const ForStmt>& f, const std::vector<IterArgPtr>& iter_args,
                       const StmtPtr& new_body, const std::vector<VarPtr>& return_vars);

StmtPtr RebuildWhileStmt(const std::shared_ptr<const WhileStmt>& w, const StmtPtr& new_body);

StmtPtr RebuildWhileStmt(const std::shared_ptr<const WhileStmt>& w, const std::vector<IterArgPtr>& iter_args,
                         const StmtPtr& new_body, const std::vector<VarPtr>& return_vars);

StmtPtr RebuildIfStmt(const std::shared_ptr<const IfStmt>& s, const std::vector<StmtPtr>& new_then,
                      const std::optional<std::vector<StmtPtr>>& new_else_stmts);

StmtPtr RebuildLoop(const std::shared_ptr<const ForStmt>& for_stmt,
                    const std::shared_ptr<const WhileStmt>& while_stmt,
                    const std::vector<IterArgPtr>& iter_args, const StmtPtr& new_body,
                    const std::vector<VarPtr>& return_vars);

// --- IfStmt else-branch processing template ---

template <typename Fn>
std::optional<std::vector<StmtPtr>> ProcessElseBranch(const std::shared_ptr<const IfStmt>& if_stmt,
                                                      Fn&& transform_fn);

// --- Tail-statement transformation template ---

template <typename Fn>
StmtPtr TransformLastStmt(const StmtPtr& stmt, Fn&& transform_fn);

// --- Loop state repair functions ---

std::vector<StmtPtr> StripDeadIterArgs(const std::vector<StmtPtr>& stmts);

void BuildDefMap(const std::vector<StmtPtr>& stmts, std::unordered_map<const Var*, StmtPtr>& def_map);

std::vector<StmtPtr> FixupIterArgInitValues(const std::vector<StmtPtr>& stmts,
                                            const std::unordered_map<const Var*, StmtPtr>& original_def_map);

std::vector<StmtPtr> FixupDanglingYieldValues(const std::vector<StmtPtr>& stmts);

std::vector<StmtPtr> FinalizeSplitCoreBody(const std::vector<StmtPtr>& stmts,
                                           const std::unordered_map<const Var*, StmtPtr>& original_def_map);

// ============================================================================
// Template implementations (must be in header)
// ============================================================================

template <typename Fn>
std::optional<std::vector<StmtPtr>> ProcessElseBranch(const std::shared_ptr<const IfStmt>& if_stmt,
                                                      Fn&& transform_fn) {
  if (!if_stmt->else_body_.has_value()) return std::nullopt;
  return transform_fn(transform_utils::FlattenToStmts(if_stmt->else_body_.value()));
}

template <typename Fn>
StmtPtr TransformLastStmt(const StmtPtr& stmt, Fn&& transform_fn) {
  auto apply_to_back = [&](std::vector<StmtPtr>& stmts) {
    if (stmts.empty()) return;
    auto result = TransformLastStmt(stmts.back(), transform_fn);
    if (result) {
      stmts.back() = result;
    } else {
      stmts.pop_back();
    }
  };

  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto then_stmts = transform_utils::FlattenToStmts(if_stmt->then_body_);
    apply_to_back(then_stmts);
    auto else_stmts = ProcessElseBranch(if_stmt, [&](std::vector<StmtPtr> es) {
      apply_to_back(es);
      return es;
    });
    return RebuildIfStmt(if_stmt, then_stmts, else_stmts);
  }

  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    auto seq_stmts = seq->stmts_;
    apply_to_back(seq_stmts);
    return MakeBody(seq_stmts, seq->span_);
  }

  return transform_fn(stmt);
}

}  // namespace loop_repair
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LOOP_STATE_REPAIR_H_
