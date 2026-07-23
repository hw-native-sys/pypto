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
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Helper to deduce UnknownType (for ops with no return value)
TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// Registration Function for Sync Operations
// ============================================================================

// Register system.sync_src (Set Flag)
// Attributes: set_pipe, wait_pipe, event_id
REGISTER_OP("system.sync_src")
    .set_description("Send a synchronization signal (Set Flag)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .f_deduce_type(DeduceUnknownType);

// Register system.sync_dst (Wait Flag)
// Attributes: set_pipe, wait_pipe, event_id
REGISTER_OP("system.sync_dst")
    .set_description("Wait for a synchronization signal (Wait Flag)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_v (Vector Barrier)
// Attributes: None
REGISTER_OP("system.bar_v")
    .set_description("Vector unit barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_m (Matrix Barrier)
// Attributes: None
REGISTER_OP("system.bar_m")
    .set_description("Matrix unit barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_all (Global Barrier)
// Attributes: None
REGISTER_OP("system.bar_all")
    .set_description("Global barrier synchronization")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.fence (Memory Barrier)
// Attributes: None
REGISTER_OP("system.fence")
    .set_description("Memory barrier over global memory")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.cacheinvalid (Cache Maintenance Operation).
// Two forms selected by arg count:
//   - No args: invalidate the whole GM address space
//     (`pto.cmo.cacheinvalid all #pto.address_space<gm>`). Used as the coarse
//     data-before-signal release marker before a bare barrier notify, and on the
//     consume side after a wait before the next cacheable GM read.
//   - (tensor, shapes, offsets): invalidate one tensor sub-region; codegen
//     dispatches on shapes: all-1 -> scalar-write ptr form (pto.addptr);
//     otherwise -> tile-store partition-view form (pto.partition_view).
// Variadic arity (0 or 3), like system.syncall below: the three arguments
// below describe ONLY the region form; omitting all of them selects the
// whole-GM form. The registry does not enforce argument count.
REGISTER_OP("system.cacheinvalid")
    .set_description(
        "Invalidate cache lines: whole GM when called with no args, else a tensor sub-region "
        "(ptr form when the region is a single element, partition-view form otherwise)")
    .set_op_category("SyncOp")
    .add_argument("tensor", "Region form: target tensor whose sub-region is invalidated")
    .add_argument("shapes",
                  "Region form: per-dimension region sizes (N-D tuple; all 1 selects the scalar/ptr form)")
    .add_argument("offsets", "Region form: per-dimension start offsets (N-D tuple matching tensor rank)")
    .f_deduce_type(DeduceUnknownType);

// Register system.syncall (Cross-core all-participant barrier). Models
// pto::SYNCALL with two modes selected by the `mode` attribute:
//   - "hard" (default): FFTS barrier, no operands. Codegen emits
//     `pto.syncall() mode = <hard>`. Requires full-core occupancy.
//   - "soft": GM-polling barrier with operands. Codegen emits
//     `pto.syncall(%gm, %scratch[, %l1], %used : ...) mode = <soft>`.
//     Operand order (positional, count not enforced by the registry):
//       aiv_only / aic_only: [gm_workspace, scratch_tile, used_cores]
//       mix:                 [gm_workspace, ub_scratch, l1_scratch, used_cores]
//     where gm_workspace is a shared GM int32 buffer (used_cores*8 slots,
//     zero-initialized), scratch tiles are local int32 staging (UB on AIV,
//     L1 on AIC), and used_cores is an i32 participant count (0 = auto).
// Attributes: core_type ("aiv_only"|"aic_only"|"mix"), mode ("hard"|"soft").
REGISTER_OP("system.syncall")
    .set_description("Cross-core all-participant barrier (pto::SYNCALL)")
    .set_op_category("SyncOp")
    .add_argument("gm_workspace", "Soft form: shared GM int32 workspace (used_cores*8 slots, zero-init)")
    .add_argument("scratch", "Soft form: local int32 staging tile (UB on AIV, L1 on AIC)")
    .add_argument("used_cores", "Soft form: participant core count (i32; 0 = auto)")
    .set_attr<std::string>("core_type")
    .set_attr<std::string>("mode")
    .f_deduce_type(DeduceUnknownType);

}  // namespace ir
}  // namespace pypto
