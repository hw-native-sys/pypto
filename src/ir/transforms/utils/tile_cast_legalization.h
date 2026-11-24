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

#ifndef SRC_IR_TRANSFORMS_UTILS_TILE_CAST_LEGALIZATION_H_
#define SRC_IR_TRANSFORMS_UTILS_TILE_CAST_LEGALIZATION_H_

#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"

namespace pypto::backend {
struct TcvtAdjacency;
}  // namespace pypto::backend

namespace pypto::ir::utils {

// List each native cast's destination dtype, including the final destination.
// An empty chain means the conversion has no path.
std::vector<DataType> PlanTileCastChain(const backend::TcvtAdjacency& table, DataType from, DataType to,
                                        bool allow_direct_fp4_to_fp8);

CallPtr MakeIntermediateTileCast(const ExprPtr& source, DataType to, int mode, const Span& span);

}  // namespace pypto::ir::utils

#endif  // SRC_IR_TRANSFORMS_UTILS_TILE_CAST_LEGALIZATION_H_
