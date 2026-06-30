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

#ifndef PYPTO_CODEGEN_ORCHESTRATION_ITER_ARG_CARRY_ANALYZER_H_
#define PYPTO_CODEGEN_ORCHESTRATION_ITER_ARG_CARRY_ANALYZER_H_

#include <cstdint>
#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace codegen {

/// Per-iter_arg carry lowering plan computed before visiting the loop body.
struct IterArgCarryPlan {
  /// True when the yield value is not in the iter_arg's alias class (or TaskId).
  bool is_rebind = false;
  /// TaskId manual-scope array-carry extent; 0 means scalar/tensor/ArrayType path.
  int64_t array_size = 0;
  /// True when this iter_arg collects compiler-derived task dependencies
  /// (NeedsCompilerDepTaskId). Set by the caller post-analysis; the analyzer
  /// always returns false here. The carry is initialised with
  /// PTO2TaskId::invalid() and filled by yielded producer TaskIds.
  bool compiler_dep_collection = false;
};

/// Classifies ForStmt iter_args (trivial vs rebind) and sizes TaskId array carries.
class IterArgCarryAnalyzer {
 public:
  IterArgCarryAnalyzer(ir::ProgramPtr program, int manual_scope_depth);

  /// Analyze ``for_stmt`` iter_args. Runs the parallel TaskId const-trip CHECK when
  /// applicable. Must be called before visiting the loop body.
  std::vector<IterArgCarryPlan> Analyze(const ir::ForStmtPtr& for_stmt);

 private:
  int64_t ResolveArrayCarrySize(const ir::ForStmtPtr& for_stmt, size_t idx) const;

  ir::ProgramPtr program_;
  int manual_scope_depth_;
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_ORCHESTRATION_ITER_ARG_CARRY_ANALYZER_H_
