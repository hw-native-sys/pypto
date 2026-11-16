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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_TRANSPOSE_RECIPE_H_
#define PYPTO_BACKEND_COMMON_BUFFER_TRANSPOSE_RECIPE_H_

#include <vector>

#include "pypto/ir/expr.h"

namespace pypto::backend {

/// Dense full-valid FP32 transpose; scratch is materialized by FlattenTileNdTo2D.
/// Native operands are source, scratch and destination, never the logical axes.
bool SupportsLogicalBufferTranspose(const ir::CallPtr& call);
void ValidateBufferTransposeOperands(const std::vector<ir::ExprPtr>& args);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_TRANSPOSE_RECIPE_H_
