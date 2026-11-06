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
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// These destination-passing ops declare Buffer effects directly; the legacy
// Functional execution-access model describes Tile SSA results instead.
[[maybe_unused]] const bool kIndexedRecipesRegistered = [] {
  for (const auto& recipe : backend::GetBufferIndexedRecipes()) {
    auto& entry = OpRegistry::GetInstance().Register(recipe.buffer_op);
    entry.set_description(std::string("Explicit destination indexed operation for ") + recipe.logical_op)
        .set_op_category("BufferOp")
        .set_ir_stage(OpIRStage::Buffer)
        .set_internal_only()
        .set_output_arity(0)
        .set_buffer_result_behavior(BufferResultBehavior::None);
    for (size_t i = 0; i < recipe.input_count; ++i) {
      entry.add_argument("input" + std::to_string(i), "Explicit indexed operand")
          .set_buffer_arg_effect(i, !recipe.scatter && i == 2 ? BufferAccess::Write : BufferAccess::Read,
                                 BufferAccess::Read);
    }
    entry.add_argument("destination", "Explicit destination buffer")
        .set_buffer_arg_effect(recipe.input_count, BufferAccess::Write, BufferAccess::Read);
    if (!recipe.scatter) entry.set_workspace_arg(2);
    entry.f_deduce_type([recipe](const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
      backend::ValidateBufferIndexedOperands(recipe, args, kwargs);
      return GetVoidType();
    });
  }
  return true;
}();

}  // namespace pypto::ir
