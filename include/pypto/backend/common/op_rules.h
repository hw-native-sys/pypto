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

#ifndef PYPTO_BACKEND_COMMON_OP_RULES_H_
#define PYPTO_BACKEND_COMMON_OP_RULES_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "pypto/core/dtype.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
class Call;
}

namespace pypto::backend {

/// Geometry of a compiler-created Vec scratch tile, in elements, not bytes.
/// This is a synthesis recipe, not a restriction on caller-owned placeholders.
struct TileScratchSpec {
  DataType dtype;
  int64_t rows;
  int64_t cols;
};

/// Optional inputs to registered scratch rules. Missing information is distinct
/// from zero/false. Select uses source_dtype/source_cols; cast uses target_dtype,
/// the effective saturation mode, and immutable source_tile geometry. A supplied
/// source_dtype must agree with source_tile for cast. No scratch query mutates IR.
struct OpScratchRequest {
  std::optional<DataType> source_dtype;
  std::optional<int64_t> source_cols;
  std::optional<DataType> target_dtype = std::nullopt;
  std::optional<bool> saturating = std::nullopt;
  ir::TileTypePtr source_tile = nullptr;
};

enum class OpScratchStatus { Required, NotRequired, Unknown, Unsupported };

/// Required carries a spec; all other outcomes carry no spec. Unknown means
/// more input information is needed, not that the op is unsupported. A missing
/// callback is represented separately by Backend::PlanOpScratch's nullopt.
struct OpScratchPlan {
  OpScratchStatus status;
  std::optional<TileScratchSpec> spec;
  std::string reason;
};

using BackendScratchPlanFunc = std::function<OpScratchPlan(const OpScratchRequest&)>;

/// Final target checks before Tile PTO emission. The result type comes from
/// the assignment target, whose MemRef need not be present on the Call type.
/// Callers must supply the post-allocation IR. These callbacks diagnose errors
/// with the Call span; they neither rewrite IR nor allocate/replace scratch.
using BackendOpValidateFunc = std::function<void(const std::shared_ptr<const ir::Call>&, const ir::TypePtr&)>;

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_OP_RULES_H_
