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

#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"

#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

void ParentStmtAnalysis::BuildMap(const FunctionPtr& func) {
  // Clear any existing mapping
  Clear();

  // Handle nullptr function
  if (!func) {
    return;
  }

  // Initialize with no parent for the root
  current_parent_ = nullptr;

  // Start traversal from function body
  if (func->body_) {
    VisitStmt(func->body_);
  }
}

StmtPtr ParentStmtAnalysis::GetParent(const StmtPtr& stmt) const {
  if (!stmt) {
    return nullptr;
  }

  auto it = parent_map_.find(stmt);
  if (it != parent_map_.end()) {
    return it->second;
  }

  return nullptr;
}

bool ParentStmtAnalysis::HasParent(const StmtPtr& stmt) const {
  if (!stmt) {
    return false;
  }

  return parent_map_.find(stmt) != parent_map_.end();
}

void ParentStmtAnalysis::Clear() {
  parent_map_.clear();
  current_parent_ = nullptr;
}

void ParentStmtAnalysis::VisitStmt(const StmtPtr& stmt) {
  if (!stmt) {
    return;
  }

  // Save the previous parent
  auto prev_parent = current_parent_;

  // Record parent-child relationship (if current_parent_ is set)
  if (current_parent_) {
    parent_map_[stmt] = current_parent_;
  }

  // Update current parent for children
  current_parent_ = stmt;

  // Visit children using the base visitor
  IRVisitor::VisitStmt(stmt);

  // Restore previous parent
  current_parent_ = prev_parent;
}

}  // namespace ir
}  // namespace pypto
