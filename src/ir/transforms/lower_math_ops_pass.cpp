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

#include "pypto/ir/function.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {

namespace {

/// Skeleton mutator for the LowerMathOps pass.
///
/// In a future change this will decompose ``tile.sin`` / ``tile.cos`` into
/// primitive arithmetic tile ops (Cody-Waite range reduction + degree-9
/// Horner polynomial). Until that lands, the mutator inherits the base
/// IRMutator's copy-on-write traversal unchanged: every node is returned
/// as-is, so the pass is a structural no-op.
class LowerMathOpsMutator : public IRMutator {};

FunctionPtr TransformLowerMathOps(const FunctionPtr& func) {
  LowerMathOpsMutator mutator;
  return mutator.VisitFunction(func);
}

}  // namespace

namespace pass {

Pass LowerMathOps() {
  return CreateFunctionPass(TransformLowerMathOps, "LowerMathOps", kLowerMathOpsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
