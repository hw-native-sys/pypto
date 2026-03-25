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

#ifndef PYPTO_IR_ARITH_IR_MUTATOR_WITH_ANALYZER_H_
#define PYPTO_IR_ARITH_IR_MUTATOR_WITH_ANALYZER_H_

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"

namespace pypto {
namespace ir {
namespace arith {

/// Base class for IR mutators that need arithmetic analysis.
///
/// Automatically binds ForStmt loop variables to their iteration ranges
/// so that sub-analyzers can use range information during expression
/// simplification (e.g., proving i // 8 == 0 when i is in [0, 8)).
///
/// Subclasses should override VisitStmt_ methods as needed. Subclasses
/// that simplify expressions will typically override VisitStmt_(ForStmt)
/// to simplify bounds before binding, since the base class binds on
/// the original (unsimplified) bounds.
///
/// The Analyzer pointer is non-owning — the caller must ensure the
/// Analyzer outlives the mutator.
class IRMutatorWithAnalyzer : public IRMutator {
 protected:
  Analyzer* analyzer_;

  explicit IRMutatorWithAnalyzer(Analyzer* analyzer) : analyzer_(analyzer) {}

  /// Override ForStmt to bind loop variable range before visiting body.
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;
};

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_ARITH_IR_MUTATOR_WITH_ANALYZER_H_
