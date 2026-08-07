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

#ifndef SRC_IR_TRANSFORMS_AUTO_TILE_VECTOR_COST_910B_H_
#define SRC_IR_TRANSFORMS_AUTO_TILE_VECTOR_COST_910B_H_

#include <cstdint>

#include "src/ir/transforms/auto_tile/vector_graph.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {

/** One reduction estimate and whether it used the legacy unsupported-dtype fallback. */
struct VectorReductionCost {
  double cycles = 0.0;
  bool used_fallback = false;
};

/** Price one emitted pointwise primitive for one valid tile frame. */
[[nodiscard]] double PointwiseCycles910B(VectorPrimitive primitive, VectorGeometry geometry,
                                         const DataType& output_dtype, int64_t rows, int64_t cols,
                                         bool stream_start, bool row_expand_composite,
                                         int64_t vector_register_bytes);

/** Price one row/column reduction for one valid tile frame. */
[[nodiscard]] VectorReductionCost ReductionCycles910B(VectorOpKind kind, const DataType& dtype, int64_t rows,
                                                      int64_t cols, int64_t vector_register_bytes);

/** Price the generated thin add/max after each non-initial reduction chunk. */
[[nodiscard]] double GeneratedReductionMergeCycles910B(int reduced_axis, int64_t free_tile,
                                                       int64_t iterations, const DataType& dtype,
                                                       int64_t work_units, int64_t vector_register_bytes);

/** Price one generated online-softmax statistics phase across all work units. */
[[nodiscard]] double GeneratedSoftmaxCycles910B(bool update, int64_t free_tile, int64_t chunk_extent,
                                                int64_t iterations, const DataType& dtype, int64_t work_units,
                                                int64_t vector_register_bytes, bool* used_reduction_fallback);

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_AUTO_TILE_VECTOR_COST_910B_H_
