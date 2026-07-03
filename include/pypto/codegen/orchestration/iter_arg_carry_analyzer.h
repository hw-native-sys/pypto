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
  /// True when compiler-dep collection needs a dynamic (vector) backing store
  /// because the ForStmt trip count is not a compile-time constant. Set by the
  /// caller post-analysis alongside compiler_dep_collection.
  bool dynamic_compiler_dep_collection = false;
};

/// Classifies ForStmt iter_args and sizes TaskId array carries.
///
/// Each iter_arg is classified as **trivial** or **rebind** by examining the
/// loop body's trailing ``YieldStmt``:
///
///   - **trivial**: the yield value is the iter_arg itself (or an alias).
///     No materialised carry variable is needed — the iter_arg and return_var
///     share the init value's emit name. The runtime dependency tracker keys
///     off ``Tensor*`` identity, and ``OUTPUT_EXISTING`` / ``INOUT`` params
///     record the address of the ``Tensor`` lvalue passed in. Materialising a
///     fresh ``Tensor`` for the carry would break dep chains because kernel
///     reads/writes would see a different ``&tensor`` than the producer.
///
///   - **rebind**: the yield value is a different variable (e.g. a freshly
///     created tensor inside the body). A mutable carry variable is declared
///     and ``YieldStmt`` assigns back to it. Without this, a Python rebind
///     like ``current = next`` would never propagate to the next iteration
///     or to code following the loop. See issue #1286.
///
/// The classification must happen **before** visiting the loop body so the
/// carry declarations are emitted ahead of the body code. The body may be
/// wrapped in an AUTO ``RuntimeScopeStmt`` by ``MaterializeRuntimeScopes``;
/// ``UnwrapAutoScope`` peeks through it so the trailing yield is found.
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
