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

#include "pypto/ir/sync.h"

#include <any>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr int kPipelineEventSlots = 8;

// Helper to deduce UnknownType (for ops with no return value)
TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

TypePtr DeducePipelineFlagType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const char* name) {
  CHECK(args.size() <= 1) << name << " accepts at most one dynamic event-id operand";
  bool has_src = false;
  bool has_dst = false;
  bool has_static_id = false;
  int src = -1;
  int dst = -1;
  for (const auto& [key, value] : kwargs) {
    if (key == "src_pipe" || key == "dst_pipe") {
      int pipe = AnyCast<int>(value, key);
      CHECK(pipe >= static_cast<int>(PipeType::MTE1) && pipe < static_cast<int>(PipeType::ALL))
          << name << " " << key << " must be a concrete PipeType, got " << pipe;
      if (key == "src_pipe") {
        CHECK(!has_src) << name << " has duplicate src_pipe";
        has_src = true;
        src = pipe;
      } else {
        CHECK(!has_dst) << name << " has duplicate dst_pipe";
        has_dst = true;
        dst = pipe;
      }
    } else if (key == "event_id") {
      CHECK(!has_static_id) << name << " has duplicate event_id";
      int id = AnyCast<int>(value, key);
      CHECK(id >= 0 && id < kPipelineEventSlots) << name << " event_id must be in [0, 7], got " << id;
      has_static_id = true;
    } else if (key == "core_type") {
      const auto core = AnyCast<std::string>(value, key);
      CHECK(core == "aic" || core == "aiv") << name << " core_type must be 'aic' or 'aiv', got " << core;
    } else {
      CHECK(false) << name << " has unknown attribute '" << key << "'";
    }
  }
  CHECK(has_src && has_dst) << name << " requires src_pipe and dst_pipe";
  CHECK(src != dst) << name << " requires different source and destination pipes";
  CHECK(has_static_id != (args.size() == 1))
      << name << " requires exactly one static event_id attribute or dynamic event-id operand";
  if (!args.empty()) {
    CHECK(args[0]) << name << " dynamic event id is null";
    auto type = As<ScalarType>(args[0]->GetType());
    CHECK(type && type->dtype_ == DataType::INDEX) << name << " dynamic event id must be a ScalarType(INDEX)";
  }
  return GetUnknownType();
}

/// Construct a pipe-local completion type from the complete low-level schema.
TypePtr DeduceSyncTokenType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs, const char* name) {
  CHECK(args.empty()) << name << " takes no operands";
  // Execution-instance ownership comes from the enclosing device function,
  // rather than a duplicated core attribute with a potentially wrong default.
  CHECK(kwargs.size() == 1 && kwargs[0].first == "pipe") << name << " requires only the pipe attribute";
  const int pipe = AnyCast<int>(kwargs[0].second, "pipe");
  CHECK(pipe >= static_cast<int>(PipeType::MTE1) && pipe < static_cast<int>(PipeType::ALL))
      << name << " pipe must be a concrete PipeType, got " << pipe;
  return std::make_shared<SyncTokenType>(static_cast<PipeType>(pipe));
}

/// Resolve only storage types whose values carry a runtime data address.
/// PtrType and MemRef identify allocations; neither promises such a pointer.
std::optional<MemorySpace> StorageAddressSpace(const TypePtr& type) {
  if (AsTensorTypeLike(type)) return MemorySpace::DDR;
  if (const auto tile = As<TileType>(type)) return tile->GetMemorySpace();
  if (const auto buffer = As<BufferType>(type)) return buffer->memory_space_;
  return std::nullopt;
}

/// Validate INDEX-valued interval bounds and reject statically reversed ranges.
/// Dynamic callers retain the precondition begin <= end; this predicate is not
/// a recovery mechanism for overflow in expressions that compute the bounds.
void ValidateByteInterval(const ExprPtr& begin, const ExprPtr& end) {
  for (const auto& bound : {begin, end}) {
    CHECK(bound) << "sync.memory_overlaps interval bounds must not be null";
    const auto scalar = As<ScalarType>(bound->GetType());
    CHECK_SPAN(scalar && scalar->dtype_ == DataType::INDEX, bound->span_)
        << "sync.memory_overlaps interval bounds must have ScalarType(INDEX)";
  }
  const auto begin_constant = As<ConstInt>(begin);
  const auto end_constant = As<ConstInt>(end);
  CHECK_SPAN(!begin_constant || !end_constant || begin_constant->value_ <= end_constant->value_, begin->span_)
      << "sync.memory_overlaps requires half-open byte intervals with begin <= end";
}

}  // namespace

LogicalSyncKind GetLogicalSyncKind(const OpPtr& op) {
  if (!op) return LogicalSyncKind::None;

  // Validate the names once through the registry, then compare canonical names:
  // deserialization may construct a different Op object for the same operator.
  static const std::array<std::pair<OpPtr, LogicalSyncKind>, 6> logical_ops = {{
      {OpRegistry::GetInstance().GetOp("sync.record"), LogicalSyncKind::Record},
      {OpRegistry::GetInstance().GetOp("sync.completed"), LogicalSyncKind::Completed},
      {OpRegistry::GetInstance().GetOp("sync.join"), LogicalSyncKind::Join},
      {OpRegistry::GetInstance().GetOp("sync.wait"), LogicalSyncKind::Wait},
      {OpRegistry::GetInstance().GetOp("sync.memory_overlaps"), LogicalSyncKind::MemoryOverlaps},
      {OpRegistry::GetInstance().GetOp("sync.vf_barrier"), LogicalSyncKind::VFBarrier},
  }};
  for (const auto& [registered_op, kind] : logical_ops) {
    if (op->name_ == registered_op->name_) return kind;
  }
  return LogicalSyncKind::None;
}

// Capture the entire source command prefix, including earlier imported waits,
// without blocking or allocating a flag. Its position is semantically relevant:
// capture cannot be CSE'd or moved merely because another record has this pipe.
REGISTER_OP("sync.record")
    .set_description("Capture completion of a source-pipeline command prefix, including prior waits")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("pipe")
    .no_execution_memory_access()
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeduceSyncTokenType(args, kwargs, "sync.record");
    });

REGISTER_OP("sync.completed")
    .set_description("Create an already-completed token for an initial value or empty path")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("pipe")
    .no_execution_memory_access()
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeduceSyncTokenType(args, kwargs, "sync.completed");
    });

// A pure conjunction keeps merged dependency frontiers compact. Keeping joins
// within one source pipeline preserves a precise type for later flag lowering.
REGISTER_OP("sync.join")
    .set_description("Complete after both same-pipeline completion tokens have completed")
    .set_op_category("SyncOp")
    .add_argument("first", "First immutable completion token")
    .add_argument("second", "Second immutable completion token with the same source pipe")
    .no_execution_memory_access()
    .f_deduce_type([](const auto& args, const auto& kwargs) -> TypePtr {
      CHECK(args.size() == 2) << "sync.join requires exactly two completion tokens";
      CHECK(kwargs.empty()) << "sync.join takes no attributes";
      CHECK(args[0] && args[1]) << "sync.join tokens must not be null";
      auto first = As<SyncTokenType>(args[0]->GetType());
      auto second = As<SyncTokenType>(args[1]->GetType());
      CHECK(first && second) << "sync.join requires SyncTokenType operands";
      CHECK(first->pipe_ == second->pipe_) << "sync.join requires tokens with the same source pipe";
      return first;
    });

// The operands name actual storage handles, not the compiler's MemRef metadata
// objects. Bounds are signed INDEX byte offsets from each runtime data address.
// Empty intervals are disjoint. Lowering must compare mathematical addresses
// without wrapping, and must not turn this pure predicate into a payload read,
// allocation-bounds check, or implicit wait.
REGISTER_OP("sync.memory_overlaps")
    .set_description("Test whether two runtime storage byte intervals overlap")
    .set_op_category("SyncOp")
    .add_argument("first", "First Tensor, Tile, or Buffer runtime storage handle")
    .add_argument("second", "Second Tensor, Tile, or Buffer in the same known address space")
    .add_argument("first_begin", "INDEX byte offset of the first interval's inclusive beginning")
    .add_argument("first_end", "INDEX byte offset of the first interval's exclusive end")
    .add_argument("second_begin", "INDEX byte offset of the second interval's inclusive beginning")
    .add_argument("second_end", "INDEX byte offset of the second interval's exclusive end")
    .no_arg_writes()
    .no_execution_memory_access()
    .f_deduce_type([](const auto& args, const auto& kwargs) -> TypePtr {
      CHECK(args.size() == 6) << "sync.memory_overlaps requires two storage handles and four INDEX bounds";
      CHECK(kwargs.empty()) << "sync.memory_overlaps takes interval bounds as operands, not attributes";
      for (size_t i = 0; i < 2; ++i) {
        const auto& value = args[i];
        CHECK(value) << "sync.memory_overlaps storage handles must not be null";
        const auto& type = value->GetType();
        CHECK_SPAN(AsTensorTypeLike(type) || As<TileType>(type) || As<BufferType>(type), value->span_)
            << "sync.memory_overlaps requires Tensor, Tile, or Buffer storage handles; "
               "Ptr and MemRef allocation identities are not runtime data pointers";
      }
      const auto first_space = StorageAddressSpace(args[0]->GetType());
      const auto second_space = StorageAddressSpace(args[1]->GetType());
      CHECK(first_space && second_space && first_space == second_space)
          << "sync.memory_overlaps requires storage handles in the same known memory space";
      ValidateByteInterval(args[2], args[3]);
      ValidateByteInterval(args[4], args[5]);
      return std::make_shared<ScalarType>(DataType::BOOL);
    });

// This is a non-consuming order constraint. A same-source/destination wait can
// still require an intra-pipeline completion boundary and is not inherently dead.
REGISTER_OP("sync.wait")
    .set_description("Order subsequent destination-pipeline work after an SSA completion token")
    .set_op_category("SyncOp")
    .set_output_arity(0)
    .add_argument("token", "Immutable completion token; it may have multiple wait users")
    .set_attr<int>("dst_pipe")
    .no_execution_memory_access()
    .f_deduce_type([](const auto& args, const auto& kwargs) -> TypePtr {
      CHECK(args.size() == 1) << "sync.wait requires exactly one completion token";
      CHECK(args[0]) << "sync.wait token must not be null";
      CHECK(As<SyncTokenType>(args[0]->GetType())) << "sync.wait requires a SyncTokenType token";
      CHECK(kwargs.size() == 1 && kwargs[0].first == "dst_pipe") << "sync.wait requires dst_pipe";
      int dst = AnyCast<int>(kwargs[0].second, "dst_pipe");
      CHECK(dst >= PipeType::MTE1 && dst < PipeType::ALL) << "sync.wait dst_pipe must be a concrete PipeType";
      return GetVoidType();
    });

// VF executes a register/VMEM instruction stream inside one vector-function
// region. Its internal completion boundary is distinct from an outer pipe flag.
REGISTER_OP("sync.vf_barrier")
    .set_description("Complete preceding VF work before subsequent VF instructions")
    .set_op_category("SyncOp")
    .set_output_arity(0)
    .no_argument()
    .no_execution_memory_access()
    .f_deduce_type([](const auto& args, const auto& kwargs) -> TypePtr {
      CHECK(args.empty() && kwargs.empty()) << "sync.vf_barrier takes no operands or attributes";
      return GetVoidType();
    });

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

REGISTER_OP("system.set_flag")
    .set_description("Signal an intra-core hardware pipeline event")
    .set_op_category("SyncOp")
    .add_argument("event_id_dyn", "Optional dynamic INDEX event id")
    .set_attr<int>("src_pipe")
    .set_attr<int>("dst_pipe")
    .set_attr<int>("event_id")
    .set_attr<std::string>("core_type")
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeducePipelineFlagType(args, kwargs, "system.set_flag");
    });

REGISTER_OP("system.wait_flag")
    .set_description("Wait on an intra-core hardware pipeline event")
    .set_op_category("SyncOp")
    .add_argument("event_id_dyn", "Optional dynamic INDEX event id")
    .set_attr<int>("src_pipe")
    .set_attr<int>("dst_pipe")
    .set_attr<int>("event_id")
    .set_attr<std::string>("core_type")
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return DeducePipelineFlagType(args, kwargs, "system.wait_flag");
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
//   - (tensor, shapes, offsets): locate a tensor sub-region, then invalidate only
//     the cache line containing that view's base address. Codegen emits one
//     shape-independent form — `pto.partition_view` + `pto.cmo.cacheinvalid
//     %view single_cache_line` — for every region size, a single element
//     included. `shapes` does not make the operation walk the region. A raw
//     `!pto.ptr` operand is rejected by ptoas outright, so there is no scalar/ptr
//     variant to dispatch to.
// Variadic arity (0 or 3): the three arguments below describe ONLY the region
// form; omitting all of them selects the whole-GM form. The registry does not
// enforce argument count.
REGISTER_OP("system.cacheinvalid")
    .set_description(
        "Invalidate whole GM when called with no args, else the cache line containing a tensor-view base "
        "(always lowered through a partition view; shapes do not cause a range walk)")
    .set_op_category("SyncOp")
    .add_argument("tensor", "Region form: target tensor whose view base addresses the cache line")
    .add_argument("shapes", "Region form: per-dimension view sizes (does not request a cache-line range)")
    .add_argument("offsets", "Region form: per-dimension start offsets (N-D tuple matching tensor rank)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
      CHECK(kwargs.empty()) << "system.cacheinvalid takes no kwargs";
      CHECK(args.empty() || args.size() == 3)
          << "system.cacheinvalid requires either no arguments or (tensor, shapes, offsets), but got "
          << args.size();
      if (!args.empty()) {
        auto tensor_type = AsTensorTypeLike(args[0]->GetType());
        CHECK(tensor_type) << "system.cacheinvalid region form requires a tensor as its first argument";
        CHECK_SPAN(!tensor_type->tensor_view_ || !IsMxTensorLayout(tensor_type->tensor_view_->layout),
                   args[0]->span_)
            << "system.cacheinvalid does not support MX-layout tensors";
      }
      return GetUnknownType();
    });

// Register system.syncall (Cross-core all-participant barrier). Models
// pto::SYNCALL with two modes selected by the `mode` attribute:
//   - "hard" (default): FFTS barrier, no operands. Codegen emits
//     `pto.syncall() mode = <hard>`. Requires full-core occupancy.
//   - "soft": GM-polling barrier with operands. Codegen emits
//     `pto.syncall(%gm[, %used] : ...) mode = <soft>` for every core_type.
//     gm_workspace is a shared, zero-initialized GM int32 buffer with at least
//     16 elements (one exclusive 64-byte cache line). used_cores is an optional
//     i32 participant count; omitting it asks PTO-ISA to derive the count from
//     the launch configuration.
// Attributes: core_type ("aiv_only"|"aic_only"|"mix"), mode ("hard"|"soft").
REGISTER_OP("system.syncall")
    .set_description("Cross-core all-participant barrier (pto::SYNCALL)")
    .set_op_category("SyncOp")
    .add_argument("gm_workspace", "Soft form: shared GM int32 workspace (at least 16 elements, zero-init)")
    .add_argument("used_cores", "Soft form: optional participant core count (i32; omitted = auto)")
    .set_attr<std::string>("core_type")
    .set_attr<std::string>("mode")
    // Soft form: every core writes its arrival counter into `gm_workspace` and
    // polls it, so the workspace is read and written. The hard form is an FFTS
    // barrier that touches no workspace at all.
    .set_arg_effect(0,
                    [](const std::vector<std::pair<std::string, std::any>>& kwargs) {
                      return GetStringKwarg(kwargs, "mode", "hard") == "soft" ? ArgEffect::ReadWrite
                                                                              : ArgEffect::Read;
                    })
    .f_deduce_type(DeduceUnknownType);

}  // namespace ir
}  // namespace pypto
