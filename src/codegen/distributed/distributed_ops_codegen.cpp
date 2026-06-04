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

#include <cstddef>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/distributed/distributed_codegen.h"
#include "pypto/codegen/distributed/distributed_op_registry.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::CallPtr;
using ir::ConstInt;
using ir::ExprPtr;
using ir::MakeTuple;

// ============================================================================
// pld.tensor.alloc_window_buffer — host-side marker, no runtime emission.
//
// The alloc op is consumed at compile time by ``MaterializeCommDomainScopes``; the
// resulting ``WindowBuffer`` metadata is emitted by ``EmitCommDomainAllocations``
// as part of the ``orch.allocate_domain(buffers=[CommBufferSpec(...), ...])``
// spec list wrapping the host_orch body. The host_orch.py module never needs
// to reach for the IR-level alloc op again — chip dispatch reads the device
// pointer from ``__comm_d0[r].buffer_ptrs["<name>"]`` instead. Returning
// empty signals the surrounding ``AssignStmt`` visitor to drop the line.
// ============================================================================
REGISTER_DISTRIBUTED_OP(pld_tensor_alloc_window_buffer, "pld.tensor.alloc_window_buffer") {
  (void)op;
  (void)codegen;
  return "";
}

// ============================================================================
// pld.tensor.window — host-side marker, no runtime emission.
//
// ``pld.tensor.window`` materialises a window-bound view at IR construction
// time; ``MaterializeCommDomainScopes`` rewires every dispatch site so the per-rank
// device pointer is read from ``__comm_d0[r].buffer_ptrs["<name>"]`` at
// chip-arg emission time. The host_orch.py module never calls back into
// the IR window op.
// ============================================================================
REGISTER_DISTRIBUTED_OP(pld_tensor_window, "pld.tensor.window") {
  (void)op;
  (void)codegen;
  return "";
}

// ============================================================================
// tensor.slice — emit Python tensor indexing into ``tensors[...]``.
//
// IR form:
//   t = tensor.slice(input, shape, offset, valid_shape, drop_dims)
//
// where shape / offset / valid_shape / drop_dims are MakeTuples. valid_shape
// is purely IR metadata and is ignored at the host layer. For each axis:
//   * axis ∈ drop_dims → scalar Python index ``offset[axis]`` (rank
//     reduction, must be a unit dim)
//   * otherwise        → slice ``offset[axis] : offset[axis] + shape[axis]``
//
// The result is registered into the ``tensors`` dict so downstream
// dispatch sites can ``chip_args.add_tensor(make_tensor_arg(tensors["t"]), ...)``
// without an extra binding step.
// ============================================================================
REGISTER_DISTRIBUTED_OP(tensor_slice, "tensor.slice") {
  auto& dist_codegen = dynamic_cast<DistributedCodegen&>(codegen);

  CHECK(op->args_.size() == 3 || op->args_.size() == 4 || op->args_.size() == 5)
      << "tensor.slice host_orch codegen expects 3-5 args (input, shape, offset[, valid_shape[, "
         "drop_dims]]), "
         "got "
      << op->args_.size();

  const std::string input_name = codegen.GetExprAsCode(op->args_[0]);
  CHECK(!input_name.empty()) << "tensor.slice input must resolve to a non-empty Python name";

  const std::string lhs = codegen.GetCurrentResultTarget();
  CHECK(!lhs.empty()) << "tensor.slice in host_orch must have an assignment target";

  auto shape_tuple = As<MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(shape_tuple, op->span_) << "tensor.slice shape must be MakeTuple";
  auto offset_tuple = As<MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offset_tuple, op->span_) << "tensor.slice offset must be MakeTuple";
  CHECK(offset_tuple->elements_.size() == shape_tuple->elements_.size())
      << "tensor.slice offset/shape rank mismatch";

  std::set<int64_t> drop_dims;
  if (op->args_.size() == 5) {
    auto dd_tuple = As<MakeTuple>(op->args_[4]);
    INTERNAL_CHECK_SPAN(dd_tuple, op->span_) << "tensor.slice drop_dims must be MakeTuple";
    for (const auto& e : dd_tuple->elements_) {
      auto ci = As<ConstInt>(e);
      CHECK(ci) << "tensor.slice drop_dims entries must be ConstInt";
      drop_dims.insert(ci->value_);
    }
  }

  std::ostringstream indices;
  for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
    if (i > 0) indices << ", ";
    const std::string offset_i = codegen.GetExprAsCode(offset_tuple->elements_[i]);
    if (drop_dims.count(static_cast<int64_t>(i)) > 0) {
      // Drop this axis via scalar indexing — torch reduces the rank by 1.
      indices << offset_i;
    } else {
      const std::string shape_i = codegen.GetExprAsCode(shape_tuple->elements_[i]);
      // Constant-fold ``0 + shape_i`` for readability when offset is 0.
      auto offset_const = As<ConstInt>(offset_tuple->elements_[i]);
      if (offset_const && offset_const->value_ == 0) {
        indices << "0:" << shape_i;
      } else {
        indices << offset_i << ":" << offset_i << " + " << shape_i;
      }
    }
  }

  std::ostringstream line;
  line << "tensors[\"" << lhs << "\"] = tensors[\"" << input_name << "\"][" << indices.str() << "]";
  codegen.Emit(line.str());
  dist_codegen.MarkDeclared(lhs);
  return "";
}

}  // namespace codegen
}  // namespace pypto
