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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_CORE_AFFINITY_H_
#define PYPTO_IR_TRANSFORMS_UTILS_CORE_AFFINITY_H_

#include <optional>
#include <string>

#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace core_affinity {

enum class CoreAffinity { CUBE, VECTOR, SHARED, MIXED, BOUNDARY };

enum class PipeDirection { C2V = 1, V2C = 2 };

enum class CoreSide { AIC, AIV };

enum class CVDirection { NONE, CUBE_TO_VECTOR, VECTOR_TO_CUBE };

constexpr int kDirMaskC2V = static_cast<int>(PipeDirection::C2V);
constexpr int kDirMaskV2C = static_cast<int>(PipeDirection::V2C);

CoreAffinity CombineAffinity(CoreAffinity a, CoreAffinity b);

bool IsCubeOp(const std::string& name);

bool IsCubeMemorySpace(MemorySpace ms);

std::optional<MemorySpace> GetFirstTileArgMemory(const CallPtr& call);

CVDirection ClassifyMoveDirection(const CallPtr& call);

CoreAffinity ClassifyCallAffinity(const CallPtr& call);

struct CVBoundaryMove {
  CVDirection direction;
  VarPtr dest_var;
  ExprPtr source_tile;
  TypePtr result_type;
  CallPtr source_tpop_call;
};

}  // namespace core_affinity
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_CORE_AFFINITY_H_
