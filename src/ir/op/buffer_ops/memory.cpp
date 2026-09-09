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

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

void CheckIntegerOperand(const ExprPtr& value, const std::string& context) {
  CHECK(value) << context << " must not be null";
  auto scalar = As<ScalarType>(value->GetType());
  CHECK_SPAN(scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX), value->span_)
      << context << " must be an integer or INDEX scalar";
}

void CheckValidExtent(const ExprPtr& extent, int64_t physical_extent, size_t axis,
                      const std::string& op_name) {
  CheckIntegerOperand(extent, op_name + " runtime valid extent");
  if (auto constant = As<ConstInt>(extent)) {
    CHECK_SPAN(constant->value_ >= 0 && constant->value_ <= physical_extent, extent->span_)
        << op_name << " valid extent for dimension " << axis << " must be between 0 and " << physical_extent
        << ", got " << constant->value_;
  }
}

void ValidateBufferAlloc(const std::vector<ExprPtr>& args, const TypePtr& result_type) {
  CHECK(args.size() == 1 || args.size() == 2)
      << "buffer.alloc requires a runtime-valid tuple and an optional effective address";
  auto buffer = As<BufferType>(result_type);
  CHECK(buffer) << "buffer.alloc result must have BufferType";
  CHECK(args[0]) << "buffer.alloc runtime valid extents must not be null";
  auto valid = As<MakeTuple>(args[0]);
  CHECK_SPAN(valid, args[0]->span_) << "buffer.alloc runtime valid extents must be a MakeTuple";
  const auto dynamic_count =
      static_cast<size_t>(std::count(buffer->valid_shape_.begin(), buffer->valid_shape_.end(), int64_t{-1}));
  CHECK_SPAN(valid->elements_.size() == dynamic_count, valid->span_)
      << "buffer.alloc requires " << dynamic_count
      << " runtime valid extent(s), one per dynamic descriptor dimension, got " << valid->elements_.size();

  size_t operand_index = 0;
  for (size_t axis = 0; axis < buffer->valid_shape_.size(); ++axis) {
    if (buffer->valid_shape_[axis] != -1) continue;
    const auto& extent = valid->elements_[operand_index++];
    CheckValidExtent(extent, buffer->shape_[axis], axis, "buffer.alloc");
  }

  if (args.size() == 2) {
    CheckIntegerOperand(args[1], "buffer.alloc effective address");
    if (auto address = As<ConstInt>(args[1])) {
      CHECK_SPAN(address->value_ >= 0, address->span_)
          << "buffer.alloc effective address must be nonnegative; omit the operand for addressless "
             "allocation";
    }
  }
}

TypePtr DeduceSetValidShape(const std::vector<ExprPtr>& args) {
  CHECK(args.size() == 2) << "buffer.set_validshape requires a buffer and a tuple of all valid extents";
  CHECK(args[0] && args[1]) << "buffer.set_validshape operands must not be null";
  auto buffer = As<BufferType>(args[0]->GetType());
  CHECK_SPAN(buffer, args[0]->span_) << "buffer.set_validshape requires a BufferType operand";
  auto valid = As<MakeTuple>(args[1]);
  CHECK_SPAN(valid, args[1]->span_) << "buffer.set_validshape valid extents must be a MakeTuple";
  CHECK_SPAN(valid->elements_.size() == buffer->shape_.size(), valid->span_)
      << "buffer.set_validshape requires one valid extent per physical dimension";
  for (size_t axis = 0; axis < buffer->shape_.size(); ++axis) {
    const auto& extent = valid->elements_[axis];
    CheckValidExtent(extent, buffer->shape_[axis], axis, "buffer.set_validshape");
    if (buffer->valid_shape_[axis] != -1) {
      auto constant = As<ConstInt>(extent);
      CHECK_SPAN(constant && constant->value_ == buffer->valid_shape_[axis], extent->span_)
          << "buffer.set_validshape cannot change static valid dimension " << axis << " ("
          << buffer->valid_shape_[axis] << "); its descriptor must already mark changing dimensions dynamic";
    }
  }
  return GetVoidType();
}

TypePtr DeduceBufferTransfer(const std::vector<ExprPtr>& args, bool load) {
  const std::string name = load ? "buffer.load" : "buffer.store";
  CHECK(args.size() == 4) << name << " requires tensor/buffer, offsets, valid extents, and destination";
  for (const auto& arg : args) CHECK(arg) << name << " operands must not be null";
  auto tensor = As<TensorType>(args[load ? 0 : 3]->GetType());
  auto buffer = As<BufferType>(args[load ? 3 : 0]->GetType());
  CHECK(tensor && buffer) << name << " requires an ordinary GM TensorType and a BufferType";
  CHECK(tensor->shape_.size() == 2 && buffer->shape_.size() == 2)
      << name << " requires rank-2 tensor and buffer operands";
  CHECK(tensor->dtype_ == DataType::FP32 && buffer->dtype_ == DataType::FP32 &&
        buffer->memory_space_ == MemorySpace::Vec)
      << name << " currently requires FP32 GM tensors and Vec buffers";
  auto offsets = As<MakeTuple>(args[1]);
  auto valid = As<MakeTuple>(args[2]);
  CHECK(offsets && offsets->elements_.size() == 2 && valid && valid->elements_.size() == 2)
      << name << " requires rank-2 MakeTuple offsets and valid extents";
  for (size_t axis = 0; axis < 2; ++axis) {
    const auto& offset = offsets->elements_[axis];
    const auto& extent = valid->elements_[axis];
    CheckIntegerOperand(offset, name + " offset");
    CheckValidExtent(extent, buffer->shape_[axis], axis, name);
    auto constant_offset = As<ConstInt>(offset);
    auto constant_extent = As<ConstInt>(extent);
    if (constant_offset) CHECK(constant_offset->value_ >= 0) << name << " offsets must be nonnegative";
    if (buffer->valid_shape_[axis] != -1) {
      CHECK(constant_extent && constant_extent->value_ == buffer->valid_shape_[axis])
          << name << " valid extent must equal the buffer's static valid dimension " << axis;
    }
    if (auto dimension = As<ConstInt>(tensor->shape_[axis]);
        dimension && constant_offset && constant_extent) {
      CHECK(constant_offset->value_ <= dimension->value_ &&
            constant_extent->value_ <= dimension->value_ - constant_offset->value_)
          << name << " transfer window exceeds GM tensor dimension " << axis;
    }
  }
  return GetVoidType();
}

}  // namespace

// The immutable physical descriptor is Call::type_, never a duplicate kwarg.
// Only dynamic valid extents occur in arg 0, ordered by their descriptor axes.
// Arg 1, when present, is the final effective byte address: zero is valid and
// no base/offset is added here. Addressless roots request fresh storage;
// addressed roots can overlap. Allocate does not imply initialized data or
// disjoint storage. Runtime extent bounds and address nonnegativity remain
// preconditions when they cannot be checked statically.
REGISTER_OP("buffer.alloc")
    .set_description("Declare a buffer with explicit runtime valid extents and an optional effective address")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("valid_extents", "Tuple of runtime valid extents for dynamic descriptor dimensions")
    .add_argument("address", "Optional final effective byte address; omission requests fresh storage")
    .set_output_arity(1)
    .set_buffer_non_memory_arg(0)
    .set_buffer_non_memory_arg(1)
    .set_buffer_result_behavior(BufferResultBehavior::Allocate)
    .f_validate_explicit_type([](const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>&,
                                 const TypePtr& result_type) { ValidateBufferAlloc(args, result_type); });

// Runtime valid state belongs to the handle. Updating it does not create a
// second handle or write data, and must not change immutable descriptor fields.
REGISTER_OP("buffer.set_validshape")
    .set_description("Update runtime valid extents without changing the buffer descriptor or data")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("buffer", "Buffer whose runtime valid metadata is updated")
    .add_argument("valid_extents", "Tuple of all valid extents; static descriptor dimensions must agree")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::None, BufferAccess::Write)
    .set_buffer_non_memory_arg(1)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceSetValidShape(args);
    });

// Transfers expose their complete GM window and do not change valid metadata.
// Dynamic extents must match the buffer's current metadata at execution time;
// bounds not statically provable remain lowering/runtime preconditions. Like
// other destination-passing operations, execution access evidence stays Unknown.
REGISTER_OP("buffer.load")
    .set_description("Load an explicit GM tensor window into a buffer destination")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("tensor", "Source GM tensor")
    .add_argument("offsets", "Tuple of GM element offsets")
    .add_argument("valid_extents", "Tuple of transfer extents matching current buffer valid metadata")
    .add_argument("dst", "Destination buffer")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_non_memory_arg(1)
    .set_buffer_non_memory_arg(2)
    .set_buffer_arg_effect(3, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceBufferTransfer(args, true);
    });

REGISTER_OP("buffer.store")
    .set_description("Store a buffer's active data into an explicit GM tensor window")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("src", "Source buffer")
    .add_argument("offsets", "Tuple of GM element offsets")
    .add_argument("valid_extents", "Tuple of transfer extents matching current buffer valid metadata")
    .add_argument("tensor", "Destination GM tensor")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_non_memory_arg(1)
    .set_buffer_non_memory_arg(2)
    .set_buffer_arg_effect(3, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceBufferTransfer(args, false);
    });

}  // namespace ir
}  // namespace pypto
