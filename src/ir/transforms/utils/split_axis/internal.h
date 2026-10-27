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

#ifndef SRC_IR_TRANSFORMS_UTILS_SPLIT_AXIS_INTERNAL_H_
#define SRC_IR_TRANSFORMS_UTILS_SPLIT_AXIS_INTERNAL_H_

#include <memory>
#include <optional>

#include "pypto/ir/expr.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace split_axis {

namespace detail {

/// Static parity check shared by shape halving and boundary localization.
bool IsOddSplitExtent(const ExprPtr& dim_size);

/// Recover an existing split axis from an original and halved tile type.
std::optional<TileInfo> SplitInfoFromHalvedType(const TypePtr& before_type, const TypePtr& after_type);

/// Replace one valid extent without changing the physical shape or allocation metadata.
TypePtr WithValidDim(const std::shared_ptr<const TileType>& tt, int dim, const ExprPtr& valid_dim);

}  // namespace detail
}  // namespace split_axis
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_UTILS_SPLIT_AXIS_INTERNAL_H_
