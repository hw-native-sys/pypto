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

#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// View construction establishes an ordinary SSA alias without reading data or
// initializing the result. Native valid metadata is fixed by the result type.
REGISTER_OP("buffer.subview")
    .set_description("Alias a strided window of a buffer at static or runtime offsets")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("source", "Dense row-major Vec source buffer")
    .add_argument("offsets", "Row and column offsets (static or runtime integer/INDEX scalars)")
    .add_argument("valid_extents", "Optional tuple of result valid extents; required for dynamic dimensions")
    .set_output_arity(1)
    .set_buffer_arg_effect(0, BufferAccess::None, BufferAccess::Read)
    .set_buffer_non_memory_arg(1)
    .set_buffer_non_memory_arg(2)
    .set_buffer_result_behavior(BufferResultBehavior::Alias, 0)
    .f_validate_explicit_type([](const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>&,
                                 const TypePtr& result) {
      CHECK((args.size() == 2 || args.size() == 3) && args[0] && args[1])
          << "buffer.subview requires a source, an offsets tuple and optional valid extents";
      backend::ValidateBufferSubview(args, result);
    });

REGISTER_OP("buffer.reshape")
    .set_description("Alias the same physical bytes through an explicit static descriptor")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("source", "Source buffer whose physical byte count is preserved")
    .set_output_arity(1)
    .set_buffer_arg_effect(0, BufferAccess::None, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::Alias, 0)
    .f_validate_explicit_type([](const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>&,
                                 const TypePtr& result) {
      CHECK(args.size() == 1 && args[0]) << "buffer.reshape requires exactly one source";
      backend::ValidateBufferReshape(args, result);
    });

}  // namespace pypto::ir
