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

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceVecBufferWrite(const std::vector<ExprPtr>& args, size_t argument_count,
                             const std::string& op_name) {
  CHECK(args.size() == argument_count)
      << op_name << " requires " << argument_count << " buffer operands, got " << args.size();
  BufferTypePtr descriptor;
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << op_name << " argument " << i << " must not be null";
    auto buffer = As<BufferType>(args[i]->GetType());
    CHECK(buffer) << op_name << " argument " << i << " must have BufferType";
    CHECK(buffer->memory_space_ == MemorySpace::Vec)
        << op_name << " argument " << i << " must be in Vec memory";
    if (!descriptor) {
      descriptor = buffer;
    } else {
      CHECK(structural_equal(descriptor, buffer))
          << op_name << " requires identical physical descriptors, including valid shape, layout and padding";
    }
  }
  return GetVoidType();
}

}  // namespace

// These initial internal operators require equal physical descriptors. An
// exact input/destination alias is legal; partially overlapping views must be
// legalized before these calls are constructed. Write concerns the active data
// region, not whole-allocation initialization. Runtime valid extents are read
// from each handle's metadata, and must agree when the descriptor is dynamic.
// ExecutionMemoryAccessEvidence remains Unknown: its Functional classification
// describes Tile SSA results and cannot represent destination-passing writes.
REGISTER_OP("buffer.copy")
    .set_description("Copy active buffer data into an explicit destination with the same descriptor")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("src", "Source buffer")
    .add_argument("dst", "Destination buffer")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(1, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceVecBufferWrite(args, 2, "buffer.copy");
    });

REGISTER_OP("buffer.mul")
    .set_description("Multiply active buffer data into an explicit destination with the same descriptor")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("lhs", "Left input buffer")
    .add_argument("rhs", "Right input buffer")
    .add_argument("dst", "Destination buffer")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(1, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(2, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceVecBufferWrite(args, 3, "buffer.mul");
    });

REGISTER_OP("buffer.add")
    .set_description("Add active buffer data into an explicit destination with the same descriptor")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("lhs", "Left input buffer")
    .add_argument("rhs", "Right input buffer")
    .add_argument("dst", "Destination buffer")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(1, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(2, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceVecBufferWrite(args, 3, "buffer.add");
    });

}  // namespace ir
}  // namespace pypto
