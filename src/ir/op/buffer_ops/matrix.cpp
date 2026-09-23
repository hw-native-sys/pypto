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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

BufferTypePtr MatrixOperand(const std::vector<ExprPtr>& args, size_t index, MemorySpace space,
                            const std::string& op_name) {
  CHECK(args[index]) << op_name << " argument " << index << " must not be null";
  auto buffer = As<BufferType>(args[index]->GetType());
  CHECK(buffer && buffer->memory_space_ == space && buffer->shape_.size() == 2)
      << op_name << " argument " << index << " must be a rank-2 " << MemorySpaceToString(space) << " buffer";
  CHECK(backend::IsMatrixBufferDtype(space, buffer->dtype_))
      << op_name << " does not support " << buffer->dtype_.ToString() << " " << MemorySpaceToString(space)
      << " buffers";
  return buffer;
}

// A static valid extent is compared; a dynamic (-1) one is a runtime precondition.
bool StaticLessEqual(int64_t lhs, int64_t rhs) { return lhs < 0 || rhs < 0 || lhs <= rhs; }

// The cube reads lhs[M, K] from Left and rhs[K, N] from Right and writes the
// Acc window [M, N]. Physical K must agree because L0 boxes are indexed
// directly; rhs valid K must cover lhs valid K. An accumulating write may
// target a wider valid rectangle than the new product, which it contains.
TypePtr DeduceBufferMatmul(const std::vector<ExprPtr>& args, bool accumulate) {
  const std::string op_name = accumulate ? "buffer.matmul_acc" : "buffer.matmul";
  CHECK(args.size() == 3) << op_name << " requires lhs, rhs and destination buffers, got " << args.size()
                          << " operands";
  auto lhs = MatrixOperand(args, 0, MemorySpace::Left, op_name);
  auto rhs = MatrixOperand(args, 1, MemorySpace::Right, op_name);
  auto destination = MatrixOperand(args, 2, MemorySpace::Acc, op_name);
  CHECK(lhs->dtype_ == rhs->dtype_) << op_name << " requires identical lhs and rhs element types, got "
                                    << lhs->dtype_.ToString() << " and " << rhs->dtype_.ToString();
  const auto accumulator = MatmulAccumulatorDataType(lhs->dtype_, rhs->dtype_);
  CHECK(destination->dtype_ == accumulator)
      << op_name << " requires a " << accumulator.ToString() << " accumulator for " << lhs->dtype_.ToString()
      << " operands, got " << destination->dtype_.ToString();
  CHECK(lhs->shape_[1] == rhs->shape_[0] && destination->shape_[0] == lhs->shape_[0] &&
        destination->shape_[1] == rhs->shape_[1])
      << op_name << " requires physical shapes [M, K] x [K, N] -> [M, N], got [" << lhs->shape_[0] << ", "
      << lhs->shape_[1] << "] x [" << rhs->shape_[0] << ", " << rhs->shape_[1] << "] -> ["
      << destination->shape_[0] << ", " << destination->shape_[1] << "]";
  CHECK(StaticLessEqual(lhs->valid_shape_[1], rhs->valid_shape_[0]))
      << op_name << " requires rhs valid K to cover lhs valid K";
  for (size_t axis = 0; axis < 2; ++axis) {
    const auto product = axis == 0 ? lhs->valid_shape_[0] : rhs->valid_shape_[1];
    const auto written = destination->valid_shape_[axis];
    CHECK(accumulate ? StaticLessEqual(product, written) : (product < 0 || written < 0 || product == written))
        << op_name << " destination valid dimension " << axis << " (" << written
        << (accumulate ? ") must contain" : ") must equal") << " the product valid extent " << product;
  }
  return GetVoidType();
}

void CheckExtractOffset(const ExprPtr& offset, int64_t source, int64_t window, const std::string& axis) {
  CHECK(offset) << "buffer.extract " << axis << " offset must not be null";
  auto scalar = As<ScalarType>(offset->GetType());
  CHECK_SPAN(scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX), offset->span_)
      << "buffer.extract " << axis << " offset must be an integer or INDEX scalar";
  if (auto constant = As<ConstInt>(offset)) {
    CHECK_SPAN(constant->value_ >= 0 && window <= source && constant->value_ <= source - window,
               offset->span_)
        << "buffer.extract " << axis << " window [" << constant->value_ << ", " << constant->value_ + window
        << ") exceeds the source extent " << source;
  }
}

// Extraction copies a static-shape window at runtime offsets. From Mat it is
// MTE1's fractal read into a cube operand; within Vec it is a vector copy.
TypePtr DeduceBufferExtract(const std::vector<ExprPtr>& args) {
  CHECK(args.size() == 4) << "buffer.extract requires source, row offset, column offset and destination, got "
                          << args.size() << " operands";
  CHECK(args[0] && args[3]) << "buffer.extract buffer operands must not be null";
  auto source = As<BufferType>(args[0]->GetType());
  auto destination = As<BufferType>(args[3]->GetType());
  CHECK(source && destination && source->shape_.size() == 2 && destination->shape_.size() == 2)
      << "buffer.extract requires rank-2 source and destination buffers";
  const auto from = source->memory_space_;
  const auto to = destination->memory_space_;
  CHECK((from == MemorySpace::Mat && (to == MemorySpace::Left || to == MemorySpace::Right)) ||
        (from == MemorySpace::Vec && to == MemorySpace::Vec))
      << "buffer.extract supports Mat -> Left/Right and Vec -> Vec, got " << MemorySpaceToString(from)
      << " -> " << MemorySpaceToString(to);
  CHECK(source->dtype_ == destination->dtype_)
      << "buffer.extract requires matching element types, got " << source->dtype_.ToString() << " and "
      << destination->dtype_.ToString();
  CheckExtractOffset(args[1], source->shape_[0], destination->shape_[0], "row");
  CheckExtractOffset(args[2], source->shape_[1], destination->shape_[1], "column");
  return GetVoidType();
}

}  // namespace

// Matrix products write the whole Acc window named by their destination. The
// initializing form never reads the destination; the accumulating form adds
// the product to its current data (native `ins(acc, ...) outs(acc)`).
REGISTER_OP("buffer.matmul")
    .set_description("Write lhs @ rhs from Left/Right operands into an explicit Acc destination")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("lhs", "Left operand buffer [M, K]")
    .add_argument("rhs", "Right operand buffer [K, N]")
    .add_argument("dst", "Acc destination buffer [M, N]")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(1, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(2, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceBufferMatmul(args, false);
    });

REGISTER_OP("buffer.matmul_acc")
    .set_description("Accumulate lhs @ rhs from Left/Right operands into an explicit Acc destination")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("lhs", "Left operand buffer [M, K]")
    .add_argument("rhs", "Right operand buffer [K, N]")
    .add_argument("dst", "Acc buffer read as the running sum and overwritten with the new sum")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(1, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_arg_effect(2, BufferAccess::ReadWrite, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceBufferMatmul(args, true);
    });

REGISTER_OP("buffer.extract")
    .set_description("Copy a static-shape window at runtime offsets into an explicit destination")
    .set_op_category("BufferOp")
    .set_ir_stage(OpIRStage::Buffer)
    .set_internal_only()
    .add_argument("src", "Source buffer")
    .add_argument("row", "Row offset of the window in source elements")
    .add_argument("col", "Column offset of the window in source elements")
    .add_argument("dst", "Destination buffer whose physical shape is the window shape")
    .set_output_arity(0)
    .set_buffer_arg_effect(0, BufferAccess::Read, BufferAccess::Read)
    .set_buffer_non_memory_arg(1)
    .set_buffer_non_memory_arg(2)
    .set_buffer_arg_effect(3, BufferAccess::Write, BufferAccess::Read)
    .set_buffer_result_behavior(BufferResultBehavior::None)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceBufferExtract(args);
    });

}  // namespace ir
}  // namespace pypto
