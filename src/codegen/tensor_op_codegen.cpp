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
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// Helper function to calculate tensor size expression
static std::string CalculateTensorSizeExpr(const TensorTypePtr& tensor_type, CodegenBase& codegen) {
  std::ostringstream oss;

  // Calculate total number of elements by multiplying all dimensions
  bool first = true;
  for (const auto& dim : tensor_type->shape_) {
    if (first) {
      oss << codegen.GenerateExprString(dim);
      first = false;
    } else {
      oss << " * " << codegen.GenerateExprString(dim);
    }
  }

  // If shape is empty, it's a scalar (1 element)
  if (first) {
    oss << "1";
  }

  // Multiply by element size in bytes
  size_t element_bits = tensor_type->dtype_.GetBit();
  size_t element_bytes = (element_bits + 7) / 8;  // Round up to nearest byte
  oss << " * " << element_bytes;

  return oss.str();
}

static std::string EmitAsUint32(const ExprPtr& expr, CodegenBase& codegen) {
  std::string str = codegen.GenerateExprString(expr);
  if (As<ConstInt>(expr)) {
    return str;
  }
  return "static_cast<uint32_t>(" + str + ")";
}

REGISTER_ORCHESTRATION_OP(tensor_create, ("tensor.create")) {
  // tensor.create emits TensorCreateInfo for runtime memory allocation via alloc_tensors().
  // The batched alloc_tensors call and const Tensor& binding are emitted by
  // EmitBatchedAllocTensors at scope entry (SeqStmts).
  auto result_type = As<TensorType>(op->GetType());
  CHECK(result_type) << "tensor.create must return TensorType";

  std::string result_var = codegen.GetCurrentResultTarget();
  size_t ndim = result_type->shape_.size();

  std::ostringstream oss;
  oss << "uint32_t " << result_var << "_ci_shapes[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    std::string dim_expr = EmitAsUint32(result_type->shape_[i], codegen);
    // Backends may override tensor.create shape without mutating IR.
    if (ndim == 1 && i == 0) {
      dim_expr = codegen.GetTensorCreateSizeExpr(result_var, dim_expr);
    }
    oss << dim_expr;
  }
  oss << "};\n";

  std::string dtype_str = codegen.GetRuntimeDataTypeString(result_type->dtype_);
  bool manual_dep = op->GetKwarg<bool>("manual_dep", false);
  oss << "TensorCreateInfo " << result_var << "_ci(" << result_var << "_ci_shapes, " << ndim << ", "
      << dtype_str;
  if (manual_dep) {
    oss << ", /*manual_dep=*/true";
  }
  oss << ");";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_read, ("tensor.read")) {
  // tensor.read(tensor, indices_tuple) -> scalar value
  //
  // Emit a call to the runtime's get_tensor_data<T>(tensor, ndims, indices).
  // The runtime spin-waits on the producer task in TensorMap (for
  // internally-allocated tensors) before reading, and reads immediately for
  // external tensors with no producer entry. Using the API uniformly avoids
  // both the missing producer-sync bug and the type-unsafe raw deref via
  // buffer.addr that a direct static_cast<T*>(ptr)[idx] would imply.
  CHECK(op->args_.size() == 2) << "tensor.read requires 2 arguments";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.read input must be a variable";

  auto input_type = As<TensorType>(op->args_[0]->GetType());
  CHECK(input_type) << "tensor.read input must be TensorType";

  auto result_type = As<ScalarType>(op->GetType());
  CHECK(result_type) << "tensor.read must return ScalarType";
  std::string cpp_type = result_type->dtype_.ToCTypeString();

  std::string result_var = codegen.GetCurrentResultTarget();
  std::string tensor_ref = codegen.GetExternalTensorName(input_name);

  // Extract indices from MakeTuple
  auto indices_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(indices_tuple) << "tensor.read indices must be MakeTuple";
  const auto& indices = indices_tuple->elements_;
  size_t ndims = indices.size();

  std::ostringstream oss;
  if (ndims == 0) {
    // Rank-0 tensor: pass ndims=0 and a null index pointer.
    oss << cpp_type << " " << result_var << " = get_tensor_data<" << cpp_type << ">(" << tensor_ref
        << ", 0, nullptr);";
  } else {
    std::string indices_var = "indices_" + result_var;
    oss << "uint32_t " << indices_var << "[" << ndims << "] = {";
    for (size_t i = 0; i < ndims; ++i) {
      if (i > 0) oss << ", ";
      oss << EmitAsUint32(indices[i], codegen);
    }
    oss << "};\n";
    oss << cpp_type << " " << result_var << " = get_tensor_data<" << cpp_type << ">(" << tensor_ref << ", "
        << ndims << ", " << indices_var << ");";
  }

  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_write, ("tensor.write")) {
  // tensor.write(tensor, indices_tuple, value) -> write scalar value to tensor at indices
  //
  // Emit a call to the runtime's set_tensor_data<T>(tensor, ndims, indices, value).
  // The runtime spin-waits on the producer task in TensorMap (and any
  // tracked INOUT consumers) before writing, so cross-thread WAW/WAR
  // hazards stay contained — same rationale as tensor.read using
  // get_tensor_data<T>(). For external tensors with no TensorMap entry
  // the write happens immediately (matches the previous raw store).
  CHECK(op->args_.size() == 3) << "tensor.write requires 3 arguments";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.write input must be a variable";

  auto input_type = As<TensorType>(op->args_[0]->GetType());
  CHECK(input_type) << "tensor.write input must be TensorType";

  std::string tensor_ref = codegen.GetExternalTensorName(input_name);

  auto indices_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(indices_tuple) << "tensor.write indices must be MakeTuple";

  std::string value_expr = codegen.GenerateExprString(op->args_[2]);
  auto value_type = As<ScalarType>(op->args_[2]->GetType());
  CHECK(value_type) << "tensor.write value must be ScalarType";
  std::string cpp_type = value_type->dtype_.ToCTypeString();

  const auto& indices = indices_tuple->elements_;
  size_t ndims = indices.size();

  std::ostringstream oss;
  if (ndims == 0) {
    // Rank-0 tensor: pass ndims=0 and a null index pointer.
    oss << "set_tensor_data<" << cpp_type << ">(" << tensor_ref << ", 0, nullptr, " << value_expr << ");";
  } else {
    // Wrap in a local block so the indices_<name>[N] temp lives in its own
    // scope — multiple writes to the same tensor in the same outer scope
    // (or across loop iterations) won't collide on the array name.
    oss << "{\n";
    oss << "  uint32_t indices_" << input_name << "[" << ndims << "] = {";
    for (size_t i = 0; i < ndims; ++i) {
      if (i > 0) oss << ", ";
      oss << EmitAsUint32(indices[i], codegen);
    }
    oss << "};\n";
    oss << "  set_tensor_data<" << cpp_type << ">(" << tensor_ref << ", " << ndims << ", indices_"
        << input_name << ", " << value_expr << ");\n";
    oss << "}";
  }

  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_slice, ("tensor.slice")) {
  // tensor.slice(input, shape_tuple, offset_tuple[, valid_shape_tuple[, drop_dims_tuple]])
  // -> Generate array variables and call .view(), then drop rank-reduced axes
  //    (numpy-style ``C[i]`` / ``C[i, j]``) via a follow-up .reshape().
  CHECK(op->args_.size() >= 3 && op->args_.size() <= 5)
      << "tensor.slice requires 3-5 arguments (input, shape, offset[, valid_shape[, drop_dims]])";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.slice input must be a variable";

  std::string ext_input_name = codegen.GetExternalTensorName(input_name);
  std::string result_var = codegen.GetCurrentResultTarget();

  // Extract shape elements from MakeTuple
  auto shape_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(shape_tuple) << "tensor.slice shape must be MakeTuple";

  // Extract offset elements from MakeTuple
  auto offset_tuple = As<MakeTuple>(op->args_[2]);
  CHECK(offset_tuple) << "tensor.slice offset must be MakeTuple";

  size_t ndim = shape_tuple->elements_.size();
  std::ostringstream oss;

  // Generate offset array (emitted first so the shape clamp below can read it).
  oss << "uint32_t " << result_var << "_offsets[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    oss << EmitAsUint32(offset_tuple->elements_[i], codegen);
  }
  oss << "};\n";

  // Generate shape array, clamped to stay within the source tensor's extent.
  // The #808 strided-Tensor runtime enforces ``offset[i] + shape[i] <= parent
  // shapes[i]`` in ``Tensor::view`` and derives the dependency/extent footprint
  // from start_offset + strides; an over-extent view corrupts host-side
  // dependency tracking (scheduler timeout / device fault) rather than being a
  // benign no-op as it was under the old (raw_shapes, offsets) model. The clamp
  // is a no-op for in-bounds slices and only trims declared over-extent (e.g.
  // a fixed unroll-width block_table slice near the buffer end); the trimmed
  // tail is never the addressed region.
  oss << "uint32_t " << result_var << "_shapes[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    // Saturate the remaining extent to 0 before std::min: ``shapes[i] - offsets[i]`` is
    // unsigned, so an offset past the source extent would underflow and let std::min return
    // the original (over-extent) size — defeating the clamp. The ternary keeps it at 0.
    oss << "(" << result_var << "_offsets[" << i << "] >= " << ext_input_name << ".shapes[" << i
        << "] ? 0u : std::min<uint32_t>(" << EmitAsUint32(shape_tuple->elements_[i], codegen) << ", "
        << ext_input_name << ".shapes[" << i << "] - " << result_var << "_offsets[" << i << "]))";
  }
  oss << "};\n";

  if (op->args_.size() >= 4) {
    // valid_shape (4th arg) may be an empty MakeTuple — the "no valid_shape"
    // sentinel that lets callers pass drop_dims (5th arg) without a custom
    // valid_shape (see DeduceTensorSliceType / tile.slice). Only enforce the
    // rank match when it carries elements.
    auto valid_shape_tuple = As<MakeTuple>(op->args_[3]);
    CHECK(valid_shape_tuple) << "tensor.slice valid_shape must be MakeTuple";
    CHECK(valid_shape_tuple->elements_.empty() || valid_shape_tuple->elements_.size() == ndim)
        << "tensor.slice valid_shape must have same rank as shape";
  }

  // Runtime tensor views use shape+offset; valid_shape only affects IR metadata.
  // The view is always full-rank: Tensor::view loops over the parent's ndims and
  // requires shape/offset arrays at the parent rank.
  oss << "Tensor " << result_var << " = " << ext_input_name << ".view(" << result_var << "_shapes, "
      << result_var << "_offsets);";

  // drop_dims (5th arg): numpy-style rank reduction (``C[i]`` drops axis 0,
  // ``C[i, j]`` drops axes 0 and 1). Each dropped axis is a unit dim of the
  // slice shape, so the addressed data is unchanged — only the result Tensor's
  // rank shrinks. The full-rank ``.view`` above already folded every offset
  // (including the dropped axes) into ``start_offset``; we then reinterpret the
  // unit-extent view at the reduced rank declared by the result Tensor type.
  // We emit the reduced shape from the result type (already rank-reduced and, for
  // sub-2D tensor results, never clamped — tiles clamp, tensors do not) rather
  // than recomputing ApplyDropDims here.
  if (op->args_.size() == 5) {
    auto drop_dims_tuple = As<MakeTuple>(op->args_[4]);
    CHECK(drop_dims_tuple) << "tensor.slice drop_dims must be MakeTuple";
    if (!drop_dims_tuple->elements_.empty()) {
      auto result_type = As<TensorType>(op->GetType());
      CHECK(result_type) << "tensor.slice with drop_dims must return TensorType";
      const size_t reduced_ndim = result_type->shape_.size();
      CHECK(reduced_ndim + drop_dims_tuple->elements_.size() == ndim)
          << "tensor.slice result rank " << reduced_ndim << " plus drop_dims count "
          << drop_dims_tuple->elements_.size() << " must equal slice rank " << ndim;
      // The view().reshape() lowering only stays contiguous when the dropped axes
      // are a leading prefix {0, 1, ..., k-1}: slicing leading unit axes of a
      // row-major DDR tensor leaves a contiguous trailing block, so reshape's
      // runtime is_contiguous assert holds. Dropping a non-leading axis (e.g.
      // ``C[:, i, :]`` -> drop_dims=[1]) yields a strided view that reshape would
      // reject at runtime, so reject it here with an actionable message instead of
      // emitting wrong-rank/asserting code. The per-axis index lowering needed to
      // support arbitrary dropped axes lives in the distributed tensor.slice
      // handler (src/codegen/distributed/distributed_ops_codegen.cpp) and is a
      // follow-up for orchestration.
      std::vector<int64_t> dropped;
      dropped.reserve(drop_dims_tuple->elements_.size());
      for (const auto& e : drop_dims_tuple->elements_) {
        auto e_const = As<ConstInt>(e);
        CHECK(e_const) << "tensor.slice drop_dims elements must be ConstInt";
        dropped.push_back(e_const->value_);
      }
      std::sort(dropped.begin(), dropped.end());
      for (size_t i = 0; i < dropped.size(); ++i) {
        CHECK(dropped[i] == static_cast<int64_t>(i))
            << "Orchestration tensor.slice only supports dropping a leading prefix "
               "of axes (drop_dims must be {0, 1, ..., k-1}); got a non-leading "
               "dropped axis "
            << dropped[i] << " at position " << i
            << ". Dropping non-leading axes (e.g. C[:, i, :]) needs per-axis index "
               "lowering not yet wired for orchestration (see #1349 / the "
               "distributed tensor.slice handler).";
      }
      if (reduced_ndim == 0) {
        // Dropping every axis (e.g. ``C[i]`` on a 1D tensor) yields a 0D scalar
        // result. A zero-sized array (``uint32_t shapes[0]``) is ill-formed in
        // standard C++, so emit a nullptr/0 reshape instead of the array form.
        oss << "\n" << result_var << " = " << result_var << ".reshape(nullptr, 0);";
      } else {
        oss << "\nuint32_t " << result_var << "_reduced_shapes[" << reduced_ndim << "] = {";
        for (size_t i = 0; i < reduced_ndim; ++i) {
          if (i > 0) oss << ", ";
          oss << EmitAsUint32(result_type->shape_[i], codegen);
        }
        oss << "};\n";
        oss << result_var << " = " << result_var << ".reshape(" << result_var << "_reduced_shapes, "
            << reduced_ndim << ");";
      }
    }
  }
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_reshape, ("tensor.reshape")) {
  // tensor.reshape(input, shape_tuple[, valid_shape_tuple]) -> Generate shape array variable and call
  // .reshape() on the runtime Tensor (see runtime/.../tensor.h: Tensor::reshape).
  CHECK(op->args_.size() == 2 || op->args_.size() == 3)
      << "tensor.reshape requires 2 or 3 arguments (input, shape[, valid_shape])";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.reshape input must be a variable";

  std::string ext_input_name = codegen.GetExternalTensorName(input_name);
  std::string result_var = codegen.GetCurrentResultTarget();

  // Extract shape elements from MakeTuple
  auto shape_tuple = As<MakeTuple>(op->args_[1]);
  CHECK(shape_tuple) << "tensor.reshape shape must be MakeTuple";
  size_t ndim = shape_tuple->elements_.size();

  std::ostringstream oss;

  // Generate shape array
  oss << "uint32_t " << result_var << "_shapes[" << ndim << "] = {";
  for (size_t i = 0; i < ndim; ++i) {
    if (i > 0) oss << ", ";
    oss << EmitAsUint32(shape_tuple->elements_[i], codegen);
  }
  oss << "};\n";

  if (op->args_.size() == 3) {
    auto valid_shape_tuple = As<MakeTuple>(op->args_[2]);
    CHECK(valid_shape_tuple) << "tensor.reshape valid_shape must be MakeTuple";
    CHECK(valid_shape_tuple->elements_.size() == ndim)
        << "tensor.reshape valid_shape must have same rank as shape";
  }

  // Runtime Tensor::reshape requires the source to be contiguous; valid_shape only affects IR metadata.
  oss << "Tensor " << result_var << " = " << ext_input_name << ".reshape(" << result_var << "_shapes, "
      << ndim << ");";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_transpose, ("tensor.transpose")) {
  // tensor.transpose(input, axis1, axis2) -> Tensor view with two axes swapped.
  // Lowered to runtime Tensor::transpose(x, y), a zero-copy metadata swap of the
  // two axes' shapes and strides (start_offset preserved; see runtime tensor.h:
  // Tensor::transpose under the #808 strided model).
  // The optional 4th `valid_shape` argument from the IR op is intentionally
  // ignored at the orchestration layer (it only affects IR metadata, mirroring
  // how tensor.reshape handles valid_shape here).
  CHECK(op->args_.size() == 3 || op->args_.size() == 4)
      << "tensor.transpose requires 3 or 4 arguments (input, axis1, axis2[, valid_shape])";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.transpose input must be a variable";

  auto input_type = As<TensorType>(op->args_[0]->GetType());
  CHECK(input_type) << "tensor.transpose input must be TensorType";

  auto axis1_const = As<ConstInt>(op->args_[1]);
  CHECK(axis1_const) << "tensor.transpose requires second argument (axis1) to be a ConstInt";
  auto axis2_const = As<ConstInt>(op->args_[2]);
  CHECK(axis2_const) << "tensor.transpose requires third argument (axis2) to be a ConstInt";

  // Normalize negative axes against the input rank (matches DeduceTensorTransposeType).
  int64_t ndim = static_cast<int64_t>(input_type->shape_.size());
  int64_t axis1 = axis1_const->value_;
  int64_t axis2 = axis2_const->value_;
  if (axis1 < 0) axis1 += ndim;
  if (axis2 < 0) axis2 += ndim;
  CHECK(axis1 >= 0 && axis1 < ndim) << "tensor.transpose axis1 out of range: " << axis1_const->value_
                                    << " for " << ndim << "D tensor";
  CHECK(axis2 >= 0 && axis2 < ndim) << "tensor.transpose axis2 out of range: " << axis2_const->value_
                                    << " for " << ndim << "D tensor";
  CHECK(axis1 != axis2) << "tensor.transpose axis1 and axis2 must be different, got " << axis1;

  // If the optional valid_shape operand is present, validate its structure even though it is
  // intentionally not emitted at the orchestration layer (mirrors tensor.reshape / tensor.slice).
  if (op->args_.size() == 4) {
    auto valid_shape_tuple = As<MakeTuple>(op->args_[3]);
    CHECK(valid_shape_tuple) << "tensor.transpose valid_shape must be MakeTuple";
    CHECK(static_cast<int64_t>(valid_shape_tuple->elements_.size()) == ndim)
        << "tensor.transpose valid_shape must have same rank as input shape";
  }

  std::string ext_input_name = codegen.GetExternalTensorName(input_name);
  std::string result_var = codegen.GetCurrentResultTarget();

  std::ostringstream oss;
  oss << "Tensor " << result_var << " = " << ext_input_name << ".transpose(" << axis1 << ", " << axis2
      << ");";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_as_layout, ("tensor.as_layout")) {
  // tensor.as_layout(input, layout=...) — metadata reinterpret over the same
  // physical buffer (RFC #1300 §3.3). The op is internal-only: passes inject
  // it at orch ↔ InCore bridge sites so the downstream callee's IR-declared
  // param type carries the new layout / canonical shape.
  //
  // **Lowering** (runtime strided-Tensor model, #808 — addressing is
  // ``buffer.addr + (start_offset + Σ coords[i]·strides[i]) · elem_size``):
  //
  // - **Identity flip** (target layout == source layout): emit a plain
  //   ``Tensor result = input;`` alias. ``DeduceTensorAsLayoutType`` also
  //   keeps the shape unchanged in this case.
  // - **Cross-layout flip** (ND ↔ DN, §4.2 canonical pair): lower to
  //   ``input.transpose(ndim-2, ndim-1)``. The runtime ``Tensor::transpose``
  //   swaps the trailing-pair ``shapes`` **and** ``strides`` together while
  //   leaving ``start_offset`` untouched — exactly the post-flip view that
  //   ``DeduceTensorAsLayoutType`` deduces (it trailing-pair-swaps both shapes
  //   and strides). Because ``start_offset`` is preserved, strided/paged
  //   sub-views (e.g. paged-attention's ``[block_offset, 0]`` slice into
  //   ``key_cache``) keep pointing at the original physical region.
  //
  // (This reverses the pre-#808 lowering, which avoided ``transpose`` because the
  // old helper swapped ``raw_shapes``/``offsets`` and shifted ``start_offset``;
  // those fields are gone, so ``transpose`` is now the correct lowering.)
  CHECK(op->args_.size() == 1) << "tensor.as_layout requires 1 arg (input) plus a 'layout' kwarg";

  std::string input_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.as_layout input must be a variable";

  auto input_type = As<TensorType>(op->args_[0]->GetType());
  CHECK(input_type) << "tensor.as_layout input must be TensorType";

  TensorLayout src_layout =
      input_type->tensor_view_.has_value() ? input_type->tensor_view_->layout : TensorLayout::ND;
  TensorLayout target_layout = src_layout;
  for (const auto& [k, v] : op->kwargs_) {
    if (k == "layout") {
      target_layout = AnyCast<TensorLayout>(v, "layout");
      break;
    }
  }

  std::string ext_input_name = codegen.GetExternalTensorName(input_name);
  std::string result_var = codegen.GetCurrentResultTarget();

  std::ostringstream oss;
  if (target_layout == src_layout) {
    // Identity flip: pure alias over the same physical buffer.
    oss << "Tensor " << result_var << " = " << ext_input_name << ";";
  } else {
    // Cross-layout flip (ND ↔ DN, §4.2 canonical pair): swap the trailing pair
    // via the runtime strided-Tensor transpose (shapes + strides swapped,
    // start_offset preserved).
    int64_t ndim = static_cast<int64_t>(input_type->shape_.size());
    INTERNAL_CHECK_SPAN(ndim >= 2, op->span_)
        << "Internal error: tensor.as_layout cross-layout flip reached codegen with rank=" << ndim
        << "; DeduceTensorAsLayoutType is supposed to reject cross-layout flips below rank 2";
    oss << "Tensor " << result_var << " = " << ext_input_name << ".transpose(" << (ndim - 2) << ", "
        << (ndim - 1) << ");";
  }

  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_dim, ("tensor.dim")) {
  // tensor.dim(tensor, axis) -> extract shape dimension as scalar
  // Validation already performed by DeduceTensorDimType during type deduction.
  INTERNAL_CHECK_SPAN(op->args_.size() == 2, op->span_) << "Internal error: tensor.dim expected 2 arguments";

  auto tensor_type = As<TensorType>(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tensor_type, op->span_) << "Internal error: tensor.dim input must be TensorType";

  auto axis_const = As<ConstInt>(op->args_[1]);
  INTERNAL_CHECK_SPAN(axis_const, op->span_) << "Internal error: tensor.dim axis must be ConstInt";

  int64_t axis = axis_const->value_;
  int64_t rank = static_cast<int64_t>(tensor_type->shape_.size());
  if (axis < 0) {
    axis += rank;
  }
  INTERNAL_CHECK_SPAN(axis >= 0 && axis < rank, op->span_) << "Internal error: tensor.dim axis out of range";

  std::string result_var = codegen.GetCurrentResultTarget();

  // For a compile-time constant dim, emit the literal directly.
  // For a dynamic dim (e.g. pl.dynamic("M")), GenerateExprString returns the
  // dynamic var name (e.g. "M"), which is not a valid C++ identifier in the
  // orchestration scope.  Read the runtime shape from the TaskArg instead.
  std::string dim_expr;
  if (As<ConstInt>(tensor_type->shape_[axis])) {
    dim_expr = codegen.GenerateExprString(tensor_type->shape_[axis]);
  } else {
    std::string tensor_name = codegen.TryGetVarName(op->args_[0]);
    CHECK(!tensor_name.empty()) << "tensor.dim: cannot resolve tensor name from first argument";
    dim_expr = codegen.GetTensorShapeDim(tensor_name, axis);
    CHECK(!dim_expr.empty()) << "tensor.dim: GetTensorShapeDim not supported for '" << tensor_name
                             << "' in this codegen context";
  }

  std::ostringstream oss;
  oss << "int64_t " << result_var << " = " << dim_expr << ";";
  return oss.str();
}

}  // namespace codegen
}  // namespace pypto
