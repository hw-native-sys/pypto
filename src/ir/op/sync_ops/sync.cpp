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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Helper to deduce UnknownType (for ops with no return value)
TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

// PTOAS's PTO_EventEnum (see PTOAS's PTOAttrs.td) only defines
// EVENT_ID0..EVENT_ID7 — 8 values — for the static event_id attribute
// consumed by pto.set_flag/pto.wait_flag. Unrelated to the cross-core FFTS
// event range in sync_ops/cross_core.cpp (kMaxUserCrossCoreEventId = 13),
// which is a different hardware event space.
constexpr int kMaxSyncFlagEventId = 7;

// Shared type-deduction / validation for system.sync_src and system.sync_dst.
// event_id is either a static compile-time attribute (`event_id`) or a
// dynamic runtime operand (`event_id_dyn`, ScalarType(INDEX)) — never both,
// mirroring DeduceCrossCoreSyncType in sync_ops/cross_core.cpp. Static lowers
// to pto.set_flag/pto.wait_flag; dynamic lowers to pto.set_flag_dyn/
// pto.wait_flag_dyn (see MakeSyncFlagCodegenPTO in
// src/backend/common/pto_ops_memory.cpp).
TypePtr DeduceSyncFlagType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                          const std::string& op_name) {
  CHECK(args.size() <= 1) << op_name << " accepts at most one dynamic event-id operand, got " << args.size();

  bool has_static_event_id = false;
  bool has_set_pipe = false;
  bool has_wait_pipe = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "event_id") {
      const int event_id = AnyCast<int>(value, "kwarg key: event_id");
      CHECK(event_id >= 0 && event_id <= kMaxSyncFlagEventId)
          << op_name << " event_id must be in the user-available range [0, " << kMaxSyncFlagEventId
          << "], got " << event_id;
      has_static_event_id = true;
    } else if (key == "set_pipe") {
      const int pipe = AnyCast<int>(value, "kwarg key: set_pipe");
      CHECK(pipe >= static_cast<int>(PipeType::MTE1) && pipe <= static_cast<int>(PipeType::ALL))
          << op_name << " set_pipe is invalid: " << pipe;
      has_set_pipe = true;
    } else if (key == "wait_pipe") {
      const int pipe = AnyCast<int>(value, "kwarg key: wait_pipe");
      CHECK(pipe >= static_cast<int>(PipeType::MTE1) && pipe <= static_cast<int>(PipeType::ALL))
          << op_name << " wait_pipe is invalid: " << pipe;
      has_wait_pipe = true;
    }
  }

  CHECK(has_set_pipe && has_wait_pipe) << op_name << " requires set_pipe and wait_pipe attributes";
  const bool has_dynamic_event_id = args.size() == 1;
  CHECK(has_static_event_id != has_dynamic_event_id)
      << op_name << " requires exactly one static event_id attribute or dynamic event-id operand";
  if (has_dynamic_event_id) {
    auto event_type = std::dynamic_pointer_cast<const ScalarType>(args[0]->GetType());
    CHECK(event_type && event_type->dtype_ == DataType::INDEX)
        << op_name << " dynamic event id must have ScalarType(INDEX), got " << args[0]->GetType()->TypeName();
  }
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// Registration Function for Sync Operations
// ============================================================================

// Register system.sync_src (Set Flag)
// Attributes: set_pipe, wait_pipe, and either a static event_id attribute or
// a dynamic event_id_dyn operand (ScalarType(INDEX)) — never both.
REGISTER_OP("system.sync_src")
    .set_description("Send a synchronization signal (Set Flag)")
    .set_op_category("SyncOp")
    .add_argument("event_id_dyn", "Optional dynamic event id (ScalarType(INDEX)); omit when event_id is static")
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSyncFlagType(args, kwargs, "system.sync_src");
    });

// Register system.sync_dst (Wait Flag)
// Attributes: set_pipe, wait_pipe, and either a static event_id attribute or
// a dynamic event_id_dyn operand (ScalarType(INDEX)) — never both.
REGISTER_OP("system.sync_dst")
    .set_description("Wait for a synchronization signal (Wait Flag)")
    .set_op_category("SyncOp")
    .add_argument("event_id_dyn", "Optional dynamic event id (ScalarType(INDEX)); omit when event_id is static")
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceSyncFlagType(args, kwargs, "system.sync_dst");
    });

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
//   - (tensor, shapes, offsets): invalidate one tensor sub-region. Codegen emits
//     one shape-independent form — `pto.partition_view` + `pto.cmo.cacheinvalid
//     %view single_cache_line` — for every region size, a single element
//     included. A raw `!pto.ptr` operand is rejected by ptoas outright, so there
//     is no scalar/ptr variant to dispatch to.
// Variadic arity (0 or 3), like system.syncall below: the three arguments
// below describe ONLY the region form; omitting all of them selects the
// whole-GM form. The registry does not enforce argument count.
REGISTER_OP("system.cacheinvalid")
    .set_description(
        "Invalidate cache lines: whole GM when called with no args, else a tensor sub-region "
        "(always lowered through a partition view, a single-element region included)")
    .set_op_category("SyncOp")
    .add_argument("tensor", "Region form: target tensor whose sub-region is invalidated")
    .add_argument("shapes", "Region form: per-dimension region sizes (N-D tuple matching tensor rank)")
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
