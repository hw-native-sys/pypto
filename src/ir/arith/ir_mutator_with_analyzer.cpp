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

/*
 * The arithmetic simplification module takes reference from:
 * - Apache TVM (https://github.com/apache/tvm), Apache License 2.0
 * - MLC-Python (https://github.com/mlc-ai/mlc-python), Apache License 2.0
 */

#include "pypto/ir/arith/ir_mutator_with_analyzer.h"

#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace arith {

StmtPtr IRMutatorWithAnalyzer::VisitStmt_(const ForStmtPtr& op) {
  // Bind loop variable to its iteration range [start, stop) when both
  // bounds are known constants. This enables range-aware simplification
  // of expressions inside the loop body (e.g., i // 8 == 0 when i in [0, 8)).
  auto start_ci = As<ConstInt>(op->start_);
  auto stop_ci = As<ConstInt>(op->stop_);

  bool bound = start_ci && stop_ci && stop_ci->value_ > start_ci->value_;
  if (bound) {
    analyzer_->Bind(op->loop_var_, start_ci->value_, stop_ci->value_);
  }

  // Delegate to base IRMutator for standard child visiting and reconstruction.
  auto result = IRMutator::VisitStmt_(op);

  // Unbind loop variable so the binding doesn't leak past the loop.
  if (bound) {
    analyzer_->Unbind(op->loop_var_);
  }

  return result;
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto
