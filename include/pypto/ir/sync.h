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

#ifndef PYPTO_IR_SYNC_H_
#define PYPTO_IR_SYNC_H_

#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {

/// Logical completion operations, distinct from existing physical flag ops.
///
/// Record reads position-dependent execution state: it cannot be moved or CSE'd
/// merely because its attributes match another record. An entirely unused record
/// can be deleted, because capture itself does not wait or change execution.
/// Wait and VFBarrier establish ordering and must survive ordinary dead-code
/// elimination. A same-pipeline wait is not generally redundant: its token may
/// describe a prefix that requires a completion boundary before subsequent work.
/// Completed, Join and MemoryOverlaps are pure values; no logical token denotes
/// a physical flag or consumes a signal when used.
enum class LogicalSyncKind { None, Record, Completed, Join, Wait, MemoryOverlaps, VFBarrier };

/// Classify logical sync by registered name, including deserialized Op objects.
/// Returns None for a null or non-logical operator. This is the shared identity
/// query for verification, optimization and the unlowered-codegen boundary.
[[nodiscard]] LogicalSyncKind GetLogicalSyncKind(const OpPtr& op);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_SYNC_H_
