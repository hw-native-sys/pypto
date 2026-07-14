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

#include "internal.h"
#include "pypto/ir/function.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

  flatten_tile_nd_to_2d::Analyze(func);
  return flatten_tile_nd_to_2d::Rewrite(func);
}

}  // namespace

Pass FlattenTileNdTo2D() {
  return CreateFunctionPass(TransformFunction, "FlattenTileNdTo2D", kFlattenTileNdTo2DProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
