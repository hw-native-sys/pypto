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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ROOT_COLLECTOR_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ROOT_COLLECTOR_H_

#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {

/**
 * @brief Determine the canonical buffer root for every Var in a function body.
 *
 * Walks the IR and maps each Var* to the Var* that owns its underlying buffer,
 * propagating root identity through assignments, loop carries, and function
 * calls (Call and Submit alike). Pure structural analysis — no optimization
 * logic.
 *
 * For a call's single (non-tuple) return value the owning root is selected by
 * matching the return type's shape+dtype against the callee's Out/InOut args
 * (see SelectReturnRoot), so a differently-typed InOut scratch is never
 * mistaken for the real output buffer (issue #1564 / #1580). When the match is
 * ambiguous no root is recorded — skipping an alias is always safe, whereas
 * guessing one can corrupt results.
 */
class BufferRootCollector : public IRVisitor {
 public:
  explicit BufferRootCollector(ProgramPtr program);

  void Initialize(const std::vector<VarPtr>& params);

  /// Var* -> owning buffer-root Var*. Populated by Initialize + VisitStmt.
  std::unordered_map<const Var*, const Var*> buffer_roots;

 protected:
  void VisitStmt_(const ForStmtPtr& for_stmt) override;
  void VisitStmt_(const WhileStmtPtr& while_stmt) override;
  void VisitStmt_(const AssignStmtPtr& assign) override;

 private:
  // A candidate output buffer: the resolved root of an Out/InOut arg, paired
  // with that arg's type so a single return value can be matched to the param
  // it actually aliases (see SelectReturnRoot).
  struct OutputRoot {
    const Var* root;
    TypePtr type;
  };

  [[nodiscard]] const Var* ResolveVar(const Var* var) const;
  [[nodiscard]] const Var* ResolveExpr(const ExprPtr& expr) const;
  [[nodiscard]] std::vector<OutputRoot> CollectCallOutputRoots(const CallPtr& call) const;

  // Pick the buffer root for a call's single (non-tuple) return value. A
  // SubWorker group may take an InOut scratch (e.g. a matmul's kv_final)
  // *before* its real Out param, so the first Out/InOut in param order is not
  // necessarily the one the return aliases. Match on the return type instead.
  // Issue #1564: without this, the FP32 scratch was fused onto the BF16 output,
  // making tensor.create -> tensor.slice(output) alias and corrupt the result.
  [[nodiscard]] const Var* SelectReturnRoot(const std::vector<OutputRoot>& out_roots,
                                            const TypePtr& return_type) const;

  // Structural shape + dtype equality, ignoring memref / tensor_view: a return
  // value aliases its source buffer with the same logical shape and dtype.
  [[nodiscard]] static bool TypesMatchShapeDtype(const TypePtr& a, const TypePtr& b);

  ProgramPtr program_;
  std::unordered_map<const Var*, std::vector<const Var*>> tuple_output_roots_;
};

}  // namespace buffer_root
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ROOT_COLLECTOR_H_
