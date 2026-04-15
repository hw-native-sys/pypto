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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_MUTABLE_COPY_H_
#define PYPTO_IR_TRANSFORMS_UTILS_MUTABLE_COPY_H_

#include <memory>
#include <type_traits>

namespace pypto {
namespace ir {

class Var;

/// Create a mutable copy of an immutable IR node.
///
/// Usage:
///   auto new_op = MutableCopy(op);
///   new_op->field_ = new_value;
///   return new_op;  // implicitly converts to shared_ptr<const T>
///
/// The mutable window is intentionally small — modify fields,
/// then return as const. This preserves the IR's immutability contract
/// while eliminating verbose constructor calls in passes.
template <typename T>
std::shared_ptr<T> MutableCopy(const std::shared_ptr<const T>& node) {
  static_assert(!std::is_base_of_v<Var, T>,
                "MutableCopy must not be used on identity-bearing nodes "
                "(Var, IterArg, MemRef) — their unique_id_ must stay unique");
  return std::make_shared<T>(*node);
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_MUTABLE_COPY_H_
