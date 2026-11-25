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
#ifndef PYPTO_IR_TRANSFORMS_UTILS_EXTERNAL_WINDOW_H_
#define PYPTO_IR_TRANSFORMS_UTILS_EXTERNAL_WINDOW_H_

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"

namespace pypto::ir {

struct ExternalWindowUse {
  bool all_workers = false;
  std::vector<int64_t> workers;
  bool uses_urma = false;
  bool uses_mapped_peer = false;
};

struct ExternalWindowAnalysis {
  std::unordered_map<const Var*, size_t> parameter_indices;
  std::unordered_map<size_t, ExternalWindowUse> uses;
};

ExternalWindowAnalysis AnalyzeExternalWindows(const ProgramPtr& program, const FunctionPtr& host);

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_EXTERNAL_WINDOW_H_
