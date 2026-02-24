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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_PARENT_STMT_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_PARENT_STMT_ANALYSIS_H_

#include <unordered_map>

#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto {
namespace ir {

/**
 * @brief Utility class for analyzing parent-child relationships in statement trees
 *
 * This class builds a mapping from each statement to its parent statement within
 * a function's body. It is useful for passes that need to traverse upward in the
 * IR tree or understand the context of a statement.
 *
 * Example usage:
 * @code
 *   ParentStmtAnalysis analysis;
 *   analysis.BuildMap(function);
 *
 *   StmtPtr parent = analysis.GetParent(some_stmt);
 *   if (parent) {
 *     // Use parent statement
 *   }
 * @endcode
 *
 * Key features:
 * - One-time analysis: Build the map once, query multiple times
 * - Root statements have no parent (GetParent returns nullptr)
 * - Handles all statement types: SeqStmts, OpStmts, IfStmt, ForStmt, etc.
 * - Thread-safe for read operations after BuildMap completes
 *
 * Note: The analysis becomes invalid after IR transformations. Call BuildMap again
 * if the IR tree is modified.
 */
class ParentStmtAnalysis : public IRVisitor {
 public:
  /**
   * @brief Build the parent mapping from a function's body
   *
   * Traverses the function's statement tree and records parent-child relationships.
   * This method clears any existing mapping before building the new one.
   *
   * @param func The function to analyze (can be nullptr, resulting in empty map)
   *
   * Parent relationships established:
   * - For SeqStmts/OpStmts: Each child statement's parent is the SeqStmts/OpStmts
   * - For IfStmt: then_body and else_body (if present) have IfStmt as parent
   * - For ForStmt: body has ForStmt as parent
   * - Root statement (function->body_) has no parent
   */
  void BuildMap(const FunctionPtr& func);

  /**
   * @brief Get the parent statement of a given statement
   *
   * @param stmt The statement to query
   * @return Parent statement, or nullptr if:
   *         - stmt is the root statement (function body)
   *         - stmt is not found in the analyzed tree
   *         - stmt is nullptr
   */
  [[nodiscard]] StmtPtr GetParent(const StmtPtr& stmt) const;

  /**
   * @brief Check if a statement has a recorded parent
   *
   * @param stmt The statement to check
   * @return true if stmt has a parent in the map, false otherwise
   */
  [[nodiscard]] bool HasParent(const StmtPtr& stmt) const;

  /**
   * @brief Clear the parent mapping
   *
   * Removes all recorded parent-child relationships. Useful for reusing
   * the same ParentStmtAnalysis instance with different functions.
   */
  void Clear();

 protected:
  /**
   * @brief Override VisitStmt to record parent-child relationships
   *
   * This method is called during tree traversal to establish mappings.
   * It records the current statement's parent before recursively visiting children.
   *
   * @param stmt The statement being visited
   */
  void VisitStmt(const StmtPtr& stmt) override;

 private:
  // Map from statement to its parent statement
  std::unordered_map<StmtPtr, StmtPtr> parent_map_;

  // Current parent during traversal (nullptr for root)
  StmtPtr current_parent_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_PARENT_STMT_ANALYSIS_H_
