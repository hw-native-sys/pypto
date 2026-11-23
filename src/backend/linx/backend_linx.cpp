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

#include "pypto/backend/linx/backend_linx.h"

#include <cstdint>
#include <vector>

#include "pypto/backend/common/soc.h"
#include "pypto/core/error.h"
#include "pypto/ir/pipe.h"

namespace pypto::backend {

const SoC& CreateLinxSoC() {
  constexpr uint64_t kCellBytes = 256 * 1024;
  constexpr uint64_t kSharedBytes = 256 * 1024;
  static const SoC soc = [] {
    PE pe(kCellBytes, {"scalar", "cube", "fixpipe", "v-int", "v-fma", "v-lnexp"});
    Core core(ir::CoreType::SuperScalar, {}, std::vector<PE>(4, pe), kSharedBytes);
    return SoC(Die(core, 36), 4);
  }();
  return soc;
}

BackendLinx::BackendLinx() : Backend(CreateLinxSoC()) {}

const BackendLinx& BackendLinx::Instance() {
  static const BackendLinx instance;
  return instance;
}

const BackendHandler* BackendLinx::GetHandler() const {
  throw ValueError("Linx supports IR lowering only; PTO code generation is unavailable");
}

}  // namespace pypto::backend
