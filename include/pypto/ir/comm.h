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
// Underlying integer values are part of the IR ABI: they are stored as the
// `int` kwarg payload of the corresponding ops (`op` for notify, `cmp` for
// wait) and cast back to the enum at codegen time. Insert new variants only
// at the end so existing IR / cached programs keep their meaning.
enum class NotifyOp : int {
  kAtomicAdd = 0,
  kSet = 1,
};

enum class WaitCmp : int {
  kEq = 0,
  kGe = 1,
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_COMM_H_
