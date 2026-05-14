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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_LOWER_MANUAL_DEPS_TO_TASK_ID_H_
#define PYPTO_IR_TRANSFORMS_UTILS_LOWER_MANUAL_DEPS_TO_TASK_ID_H_

#include "pypto/ir/stmt.h"

namespace pypto::ir {

/**
 * @brief Lower ``RuntimeScopeStmt(manual=true)`` regions into runtime TaskId
 * infrastructure consumed by orchestration codegen.
 *
 * Resolves user-supplied ``deps=[...]`` from ``Call.attrs[user_manual_dep_edges]``
 * into ``Call.attrs[manual_dep_edges]`` and synthesises ``__tid`` TaskId
 * companions for the closure of involved Vars. The 16-deps-per-submit cap is
 * enforced later at orchestration codegen, not here.
 *
 * Returns the same StmtPtr when no manual scope is present (no-op).
 *
 * @param body The function body StmtPtr to transform
 * @return The transformed body (or the same StmtPtr if unchanged)
 */
StmtPtr LowerManualDepsToTaskId(const StmtPtr& body);

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_LOWER_MANUAL_DEPS_TO_TASK_ID_H_
