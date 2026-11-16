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

#include "pypto/backend/common/buffer_transpose_recipe.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// Planning has already materialized scratch and a distinct destination. State
// the scratch write explicitly so later lifetime analysis cannot reuse it while
// TTRANS is running; it is not an initialized semantic input.
REGISTER_OP("buffer.transpose")
    .set_description("Transpose a dense FP32 buffer into an explicit destination")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("source", "Full-valid source matrix")
    .add_argument("scratch", "Full-valid workspace with the source descriptor")
    .add_argument("destination", "Full-valid transposed destination")
    .set_output_arity(0)
    .set_workspace_arg(1)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(1, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_arg_effect(2, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      backend::ValidateBufferTransposeOperands(args);
      return GetVoidType();
    });

}  // namespace pypto::ir
