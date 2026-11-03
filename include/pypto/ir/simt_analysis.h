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

#ifndef PYPTO_IR_SIMT_ANALYSIS_H_
#define PYPTO_IR_SIMT_ANALYSIS_H_

#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir {
/// Physical accesses made through one borrowed helper parameter. A write is
/// logically InOut at launch boundaries because untouched elements survive.
struct SimtParamAccess {
  bool reads = false;
  bool writes = false;
};
using SimtAccessSummary = std::vector<SimtParamAccess>;
SimtAccessSummary AnalyzeSimtAccess(const FunctionPtr& function);

struct SimtResources {
  int max_threads;
  int max_regs;
};
/// Derive resource requirements through the current PassContext target handler.
SimtResources InferSimtResources(const SimtLaunchStmtPtr& launch);
}  // namespace pypto::ir
#endif  // PYPTO_IR_SIMT_ANALYSIS_H_
