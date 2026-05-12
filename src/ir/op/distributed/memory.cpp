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

/**
 * @file memory.cpp
 * @brief Distributed-memory ops — CommGroup window-buffer allocation & materialization.
 *
 * Mirrors :file:`src/ir/op/tile_ops/memory.cpp` in structure: this translation
 * unit owns every op that touches CommGroup window-buffer address-space life
 * cycle. Two ops are registered:
 *
 * * ``pld.alloc_window_buffer(size, *, name)`` — pure address-space allocation.
 *   Takes a per-rank scalar ``size`` (in **bytes**, matching ``tile.alloc``)
 *   plus a ``name`` kwarg, and returns the singleton :class:`PtrType` (the
 *   same allocation-identity token ``tile.alloc`` produces). The parser binds
 *   the result to a plain ``Var(PtrType, name_hint=name)``; the comm-collection
 *   pass later wraps the Ptr in an :class:`ir.WindowBuffer` Var subclass and
 *   threads it through ``DistributedTensorType.window_buffer_``.
 *
 * * ``pld.window(buf, shape, *, dtype)`` — materialises a Ptr handle as a
 *   :class:`DistributedTensorType` view with the supplied shape and dtype.
 *   The result type's ``window_buffer_`` field is left ``nullopt`` at parse
 *   time; the comm-collection pass populates it from the def-use chain back
 *   to the alloc. Shape & dtype enter the type system only here.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::string& op_name) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  throw ValueError("Missing kwarg '" + key + "' on " + op_name);
}

TypePtr DeduceAllocWindowBufferType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1) << "pld.alloc_window_buffer requires exactly 1 positional argument "
                             "(size: ScalarType expression in bytes), but got "
                          << args.size();
  CHECK(args[0]) << "pld.alloc_window_buffer size argument must not be null";

  auto name = GetKwarg<std::string>(kwargs, "name", "pld.alloc_window_buffer");
  CHECK(!name.empty()) << "pld.alloc_window_buffer requires a non-empty 'name' kwarg";

  // The op produces a Ptr — exact mirror of tile.alloc / tensor.alloc.
  return GetPtrType();
}

TypePtr DeduceWindowType(const std::vector<ExprPtr>& args,
                         const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "pld.window requires 2 positional args (buf, shape), but got " << args.size();
  CHECK(args[0]) << "pld.window 'buf' argument must not be null";

  // First arg is the allocation-identity token from pld.alloc_window_buffer
  // (or, in principle, any Ptr-typed Var the parser routes here). The
  // back-reference to the actual ``WindowBuffer`` is filled in later by the
  // comm-collection pass; until then ``window_buffer_`` is ``std::nullopt``.
  CHECK(IsA<PtrType>(args[0]->GetType()))
      << "pld.window 'buf' must have type Ptr (output of pld.alloc_window_buffer), got "
      << args[0]->GetType()->TypeName();

  auto shape_tuple = As<MakeTuple>(args[1]);
  CHECK(shape_tuple) << "pld.window second argument must be a shape tuple (MakeTuple of ints / Exprs), got "
                     << args[1]->TypeName();

  auto dtype = GetKwarg<DataType>(kwargs, "dtype", "pld.window");

  // Shape & dtype enter the type system here. window_buffer_ stays nullopt
  // until CollectCommGroups (N4) wires it to the originating WindowBuffer.
  return std::make_shared<DistributedTensorType>(shape_tuple->elements_, dtype);
}

}  // namespace

// ============================================================================
// pld.alloc_window_buffer — per-rank CommGroup window-buffer allocation
// ============================================================================

REGISTER_OP("pld.alloc_window_buffer")
    .set_description(
        "Declare a per-rank CommGroup window-buffer slot of `size` bytes. Returns a Ptr "
        "(allocation-identity token, exactly like tile.alloc / tensor.alloc). The "
        "comm-collection pass later wraps the Ptr into an ir.WindowBuffer Var subclass and "
        "registers it on the program's CommGroup metadata.")
    .set_op_category("DistributedOp")
    .add_argument("size", "Per-rank allocation size in bytes (ScalarType expression)")
    .set_attr<std::string>("name")
    .no_memory_spec()
    .f_deduce_type(DeduceAllocWindowBufferType);

// ============================================================================
// pld.window — materialise a window-buffer Ptr as a DistributedTensor view
// ============================================================================

REGISTER_OP("pld.window")
    .set_description(
        "Materialise a window-buffer Ptr as a DistributedTensor view with the given shape "
        "and dtype. The result type's window_buffer back-reference is left unset at parse "
        "time; the comm-collection pass populates it from the def-use chain back to the "
        "alloc op.")
    .set_op_category("DistributedOp")
    .add_argument("buf", "Ptr handle (output of pld.alloc_window_buffer)")
    .add_argument("shape", "Per-rank shape (MakeTuple of ExprPtr / ConstInt)")
    .set_attr<DataType>("dtype")
    .no_memory_spec()
    .f_deduce_type(DeduceWindowType);

}  // namespace ir
}  // namespace pypto
