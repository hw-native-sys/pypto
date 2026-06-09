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

#ifndef PYPTO_IR_COMM_H_
#define PYPTO_IR_COMM_H_

#include <string>

#include "pypto/core/error.h"

namespace pypto {
namespace ir {

// Cross-rank synchronisation primitives consumed by the pld.system.* ops.
//
// NotifyOp selects the device-side semantics of pld.system.notify (TNOTIFY):
// - kAtomicAdd : atomically add `value` to the peer rank's signal slot.
// - kSet       : non-atomic store of `value` to the peer rank's signal slot.
//
// WaitCmp selects the wait predicate of pld.system.wait (TWAIT):
// - kEq : block until *signal_slot == expected.
// - kGe : block until *signal_slot >= expected.
//
// AtomicType selects the device-side combine mode of pld.tensor.put (TPUT):
// - kNone : plain remote store — overwrite the peer's destination slice.
// - kAdd  : atomically add the source data into the peer's destination slice.
//
// ReduceOp selects the reduction operator of pld.tensor.allreduce (and
// future collective reductions):
// - kSum  : element-wise sum across ranks.
// - kMax / kMin / kProd : reserved placeholders; lowering is currently
//   kSum-only and rejects other variants at the deducer.
//
// Underlying integer values are part of the IR ABI: they are stored as the
// `int` kwarg payload of the corresponding ops (`op` for notify, `cmp` for
// wait, `atomic` for put, `op` for allreduce) and cast back to the enum at
// codegen time. Insert new variants only at the end so existing IR / cached
// programs keep their meaning.
enum class NotifyOp : int {
  kAtomicAdd = 0,
  kSet = 1,
};

enum class WaitCmp : int {
  kEq = 0,
  kGe = 1,
};

enum class AtomicType : int {
  kNone = 0,
  kAdd = 1,
};

enum class ReduceOp : int {
  kSum = 0,
  kMax = 1,
  kMin = 2,
  kProd = 3,
};

// Convert AtomicType to the matching Python enum member name. The Python
// member is `None_` (trailing underscore) because `None` is a reserved word —
// keep this in sync with the `nb::enum_<AtomicType>` binding in
// `python/bindings/modules/ir.cpp`.
inline std::string AtomicTypeToString(AtomicType atomic) {
  switch (atomic) {
    case AtomicType::kNone:
      return "None_";
    case AtomicType::kAdd:
      return "Add";
    default:
      throw pypto::TypeError("Unknown AtomicType: " + std::to_string(static_cast<int>(atomic)));
  }
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_COMM_H_
