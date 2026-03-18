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

/// Mutator that substitutes Var/IterArg references by pointer identity.
///
/// Overrides both VisitExpr_(VarPtr) and VisitExpr_(IterArgPtr) to ensure
/// all variable references are substituted, including IterArgs used as
/// expression operands. For IterArg, initValue_ is also visited recursively.
class SubstituteVarsMutator : public IRMutator {
 public:
  explicit SubstituteVarsMutator(const std::unordered_map<const Var*, VarPtr>& var_map) : var_map_(var_map) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    // Check our explicit substitution map first
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) {
      return it->second;
    }
    // Fall back to base class which checks var_remap_ (populated when
    // ForStmt/WhileStmt iter_args are recreated with new initValues)
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // Check our explicit substitution map first
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) {
      return it->second;
    }
    // Delegate to base: visits initValue_ and may create a new IterArg.
    // When a new IterArg is created, the base's ForStmt handler registers
    // old→new in var_remap_, so body references get rewritten via VisitExpr_(VarPtr).
    return IRMutator::VisitExpr_(op);
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
