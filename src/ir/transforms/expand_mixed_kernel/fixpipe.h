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

#ifndef SRC_IR_TRANSFORMS_EXPAND_MIXED_KERNEL_FIXPIPE_H_
#define SRC_IR_TRANSFORMS_EXPAND_MIXED_KERNEL_FIXPIPE_H_

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/core_affinity.h"

namespace pypto::ir::mixed_kernel {

struct C2VFixpipeFold {
  bool enabled = false;
  std::unordered_set<const Stmt*> erased;
  std::unordered_map<const Stmt*, StmtPtr> replacements;
};

// Plan a uniform epilogue for the automatic C2V pipe. The original IR stays
// intact: only the boundary plan and the split bodies change.
C2VFixpipeFold PlanC2VFixpipeFold(
    const FunctionPtr& function, const std::vector<StmtPtr>& statements,
    std::map<const Stmt*, core_affinity::CVBoundaryMove>& boundaries,
    const std::unordered_map<const Stmt*, core_affinity::CoreAffinity>& affinities);

std::vector<StmtPtr> RouteFixpipeV2C(const std::vector<StmtPtr>& statements);

}  // namespace pypto::ir::mixed_kernel

#endif  // SRC_IR_TRANSFORMS_EXPAND_MIXED_KERNEL_FIXPIPE_H_
