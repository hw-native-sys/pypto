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

#ifndef PYPTO_IR_TRANSFORM_VERIFY_SSA_PASS_H_
#define PYPTO_IR_TRANSFORM_VERIFY_SSA_PASS_H_

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/transform/base/pass.h"
#include "pypto/ir/transform/base/visitor.h"

namespace pypto {
namespace ir {

/**
 * @brief Error types for SSA verification
 */
enum class SSAErrorType {
  MULTIPLE_ASSIGNMENT,        // Variable assigned more than once
  NAME_SHADOWING,             // Variable name shadows outer scope variable
  MISSING_YIELD,              // ForStmt or IfStmt missing required YieldStmt
  CONTROL_FLOW_TYPE_MISMATCH  // Type mismatch in control flow (ForStmt or IfStmt)
};

/**
 * @brief SSA verification error information
 */
struct SSAError {
  SSAErrorType type;
  std::string message;
  Span span;  // Source location
};

/**
 * @brief Helper visitor class for SSA verification
 *
 * Traverses the IR tree and collects SSA violations
 */
class SSAVerifier : public IRVisitor {
 public:
  explicit SSAVerifier(std::vector<SSAError>& errors) : errors_(errors) {}

  void VisitStmt_(const AssignStmtPtr& op) override;
  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;

  [[nodiscard]] const std::vector<SSAError>& GetErrors() const { return errors_; }

  /**
   * @brief Enter a new scope
   */
  void EnterScope();

  /**
   * @brief Exit the current scope
   */
  void ExitScope();

  /**
   * @brief Declare a variable in the current scope
   */
  void DeclareVariable(const VarPtr& var);

 private:
  std::vector<SSAError>& errors_;
  std::unordered_map<std::string, int> var_assignment_count_;
  std::vector<std::unordered_set<std::string>> scope_stack_;  // Track variable names in each scope

  /**
   * @brief Check if a variable has been assigned multiple times
   */
  void CheckVariableAssignment(const VarPtr& var);

  /**
   * @brief Check if a variable name shadows an outer scope variable
   */
  void CheckNameShadowing(const VarPtr& var);

  /**
   * @brief Record an error
   */
  void RecordError(SSAErrorType type, const std::string& message, const Span& span);

  /**
   * @brief Get the last statement in a statement block (recursive for SeqStmts)
   */
  StmtPtr GetLastStmt(const StmtPtr& stmt);

  /**
   * @brief Verify ForStmt specific constraints
   */
  void VerifyForStmt(const ForStmtPtr& for_stmt);

  /**
   * @brief Verify IfStmt specific constraints
   */
  void VerifyIfStmt(const IfStmtPtr& if_stmt);

  /**
   * @brief Check type equality including shape for TensorType and TileType
   */
  void CheckTypeEquality(const TypePtr& type1, const TypePtr& type2, const std::string& context,
                         const std::string& desc1, const std::string& desc2, const Span& span);

  /**
   * @brief Check if two ExprPtr represent the same constant value
   */
  bool IsSameConstant(const ExprPtr& expr1, const ExprPtr& expr2) const;
};

/**
 * @brief Pass that verifies SSA form of IR
 *
 * This pass checks the following SSA properties:
 * 1. Each variable is assigned only once (MULTIPLE_ASSIGNMENT)
 * 2. No variable name shadowing across scopes (NAME_SHADOWING)
 * 3. ForStmt with iter_args must have YieldStmt as last statement (MISSING_YIELD)
 * 4. Type consistency in ForStmt: iter_args initValue, yield values, return_vars (CONTROL_FLOW_TYPE_MISMATCH)
 * 5. IfStmt with return_vars must have YieldStmt in both then and else branches (MISSING_YIELD)
 * 6. Type consistency in IfStmt: then yield and else yield (CONTROL_FLOW_TYPE_MISMATCH)
 *
 * The pass collects all errors and generates a verification report instead of
 * throwing exceptions, allowing detection of all issues in a single run.
 */
class VerifySSAPass : public Pass {
 public:
  VerifySSAPass() = default;

  /**
   * @brief Execute the SSA verification pass on a function
   *
   * @param func Input function to verify
   * @return The same function (verification pass doesn't modify IR)
   */
  FunctionPtr Run(const FunctionPtr& func) override;

  /**
   * @brief Get the list of errors from the last verification run
   *
   * @return Vector of SSA errors
   */
  [[nodiscard]] const std::vector<SSAError>& GetErrors() const { return errors_; }

  /**
   * @brief Check if any errors were found during verification
   *
   * @return true if errors exist, false otherwise
   */
  [[nodiscard]] bool HasErrors() const { return !errors_.empty(); }

  /**
   * @brief Get a formatted verification report
   *
   * @return String containing the formatted report
   */
  [[nodiscard]] std::string GetReport() const;

 private:
  std::vector<SSAError> errors_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_VERIFY_SSA_PASS_H_
