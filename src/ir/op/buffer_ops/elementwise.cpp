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

#include "pypto/backend/common/buffer_elementwise_recipes.h"
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

// A Vec copy duplicates one descriptor. A Mat -> Left/Right copy is MTE1's
// relayout into a cube operand buffer: the element type and the physical and
// valid extents are kept, while the destination descriptor supplies the target
// fractal layout. Other cross-space moves need their own recipes.
TypePtr DeduceBufferCopy(const std::vector<ExprPtr>& args) {
  CHECK(args.size() == 2) << "buffer.copy requires 2 buffer operands, got " << args.size();
  std::vector<BufferTypePtr> buffers;
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "buffer.copy argument " << i << " must not be null";
    auto buffer = As<BufferType>(args[i]->GetType());
    CHECK(buffer) << "buffer.copy argument " << i << " must have BufferType";
    buffers.push_back(buffer);
  }
  const auto& source = buffers[0];
  const auto& destination = buffers[1];
  if (source->memory_space_ == MemorySpace::Vec && destination->memory_space_ == MemorySpace::Vec) {
    CHECK(structural_equal(source, destination))
        << "buffer.copy requires identical physical descriptors, including valid shape, layout and padding";
    return GetVoidType();
  }
  CHECK(source->memory_space_ == MemorySpace::Mat &&
        (destination->memory_space_ == MemorySpace::Left || destination->memory_space_ == MemorySpace::Right))
      << "buffer.copy operands must be in Vec memory, except for a Mat -> Left/Right cube operand copy; got "
      << MemorySpaceToString(source->memory_space_) << " -> "
      << MemorySpaceToString(destination->memory_space_);
  CHECK(source->dtype_ == destination->dtype_ && source->shape_ == destination->shape_ &&
        source->valid_shape_ == destination->valid_shape_)
      << "buffer.copy from Mat requires the destination to keep the element type and the physical and "
         "valid extents";
  return GetVoidType();
}

}  // namespace

// These internal operators require equal physical descriptors, except for the
// Mat -> Left/Right copy above. An exact input/destination alias is legal
// within one space; partially overlapping views must be
// legalized before these calls are constructed. Write concerns the active data
// region, not whole-allocation initialization. Runtime valid extents are read
// from each handle's metadata, and must agree when the descriptor is dynamic.
// ExecutionMemoryAccessEvidence remains Unknown: its Functional classification
// describes Tile SSA results and cannot represent destination-passing writes.
REGISTER_OP("buffer.copy")
    .set_description("Copy active buffer data into an explicit Vec or cube-operand destination")
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
      return DeduceBufferCopy(args);
    });

// The actual conversion and emitter table also defines explicit operand arity,
// precision kwargs, physical restrictions, and destination-alias requirements.
[[maybe_unused]] const bool kElementwiseRecipesRegistered = [] {
  for (const auto& recipe : backend::GetBufferElementwiseRecipes()) {
    auto& entry = OpRegistry::GetInstance().Register(recipe.buffer_op);
    entry.set_description(std::string("Explicit destination recipe for ") + recipe.logical_op)
        .set_op_category("BufferOp")
        .set_ir_stage(OpIRStage::Buffer)
        .set_internal_only()
        .set_output_arity(0)
        .set_buffer_result_behavior(BufferResultBehavior::None);
    for (size_t i = 0; i < recipe.inputs.size(); ++i) {
      const bool buffer = recipe.inputs[i].kind == backend::BufferElementwiseOperandKind::Buffer;
      entry.add_argument("src" + std::to_string(i), buffer ? "Source buffer" : "Resolved element scalar");
      if (buffer) {
        entry.set_buffer_arg_effect(i, BufferAccess::Read, BufferAccess::Read);
      } else {
        entry.set_buffer_non_memory_arg(i);
      }
    }
    entry.add_argument("dst", "Destination buffer")
        .set_buffer_arg_effect(recipe.inputs.size(), BufferAccess::Write, BufferAccess::Read);
    if (recipe.precision != backend::BufferPrecisionKind::None) entry.set_attr<bool>("high_precision");
    entry.f_deduce_type(
        [recipe](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>&) {
          backend::ValidateBufferElementwiseOperands(recipe, args);
          return GetVoidType();
        });
  }
  return true;
}();

}  // namespace ir
}  // namespace pypto
