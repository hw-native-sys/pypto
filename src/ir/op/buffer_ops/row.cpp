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

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_row_recipes.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// Explicit Buffer effects describe scratch and destination writes. The legacy
// Functional execution-access classification models Tile SSA results instead,
// so it intentionally remains Unknown for these destination-passing operations.
[[maybe_unused]] const bool kRowRecipesRegistered = [] {
  for (const auto& recipe : backend::GetBufferRowRecipes()) {
    auto& entry = OpRegistry::GetInstance().Register(recipe.buffer_op);
    entry.set_description(std::string("Explicit FP32 row operation for ") + recipe.logical_op)
        .set_op_category("BufferOp")
        .set_ir_stage(OpIRStage::Buffer)
        .set_internal_only()
        .add_argument("source", "Source matrix with nonempty static valid extents")
        .add_argument("auxiliary", recipe.reduction ? "Full-valid workspace" : "Column vector")
        .add_argument("destination", "Explicit destination")
        .set_output_arity(0)
        .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
        .set_buffer_arg_effect(1, recipe.reduction ? BufferAccess::Write : BufferAccess::Read,
                               BufferAccess::Read)
        .set_buffer_arg_effect(2, BufferAccess::Write, BufferAccess::Read)
        .set_buffer_result_behavior(BufferResultBehavior::None)
        .f_deduce_type(
            [recipe](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>&) {
              backend::ValidateBufferRowOperands(recipe, args);
              return GetVoidType();
            });
    if (recipe.reduction) entry.set_workspace_arg(1);
  }
  return true;
}();

}  // namespace pypto::ir
