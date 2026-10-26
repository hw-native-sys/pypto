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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SYNTACTIC_USE_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SYNTACTIC_USE_ANALYSIS_H_

#include <cstddef>
#include <unordered_map>

#include "pypto/ir/transforms/base/visitor.h"

namespace pypto {
namespace ir {

/// Syntactic reads, excluding definitions and counting loop initializers once.
/// IterArg uses must not revisit their initializer. Submit args/dependencies
/// and metadata reads retain the base visitor's traversal semantics.
/// Build once against an immutable body; discard after rewriting that body.
class SyntacticUseAnalysis : public IRVisitor {
 public:
  struct Position {
    const SeqStmts* sequence;
    size_t index;
  };
  std::unordered_map<const Var*, size_t> read_counts;
  std::unordered_map<const Var*, AssignStmtPtr> definitions;
  std::unordered_map<const Stmt*, Position> positions;

 protected:
  void VisitVarLike_(const VarPtr& op) override;
  void VisitExpr_(const IterArgPtr& op) override;
  void VisitStmt_(const AssignStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;
  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const WhileStmtPtr& op) override;
  void VisitStmt_(const SeqStmtsPtr& op) override;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SYNTACTIC_USE_ANALYSIS_H_
