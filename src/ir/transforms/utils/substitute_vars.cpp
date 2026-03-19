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

#include "pypto/ir/transforms/utils/substitute_vars.h"

#include <unordered_map>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"

namespace pypto {
namespace ir {

namespace {

/// Mutator that substitutes Var references by pointer identity.
///
/// Only overrides VisitExpr_(VarPtr), not VisitExpr_(IterArgPtr).  IterArg
/// nodes are visited by the base IRMutator which handles initValue_ traversal
/// and preserves the IterArg type required by ForStmt/WhileStmt iter_arg slots.
class SubstituteVarsMutator : public IRMutator {
 public:
  explicit SubstituteVarsMutator(const std::unordered_map<const Var*, VarPtr>& var_map) : var_map_(var_map) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) {
      return it->second;
    }
    return op;
  }

 private:
  const std::unordered_map<const Var*, VarPtr>& var_map_;
};

}  // namespace

StmtPtr SubstituteVars(const StmtPtr& body, const std::unordered_map<const Var*, VarPtr>& var_map) {
  SubstituteVarsMutator mutator(var_map);
  return mutator.VisitStmt(body);
}

}  // namespace ir
}  // namespace pypto
