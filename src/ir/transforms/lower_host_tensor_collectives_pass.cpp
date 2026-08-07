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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/window_alias_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

static constexpr size_t kMaxSupportedRanks = 16;

[[nodiscard]] std::string GetAllReduceMode(const CallPtr& call) {
  if (call->HasKwarg("mode")) {
    return call->GetKwarg<std::string>("mode");
  }
  if (call->HasAttr("mode")) {
    return call->GetAttr<std::string>("mode");
  }
  return "mesh";
}

[[nodiscard]] bool ShouldLowerAllReduceAsRing(const CallPtr& call) {
  // An absent mode kwarg means the default mesh schedule. Never infer the
  // schedule from the signal shape.
  const std::string mode = GetAllReduceMode(call);
  CHECK_SPAN(mode == "ring" || mode == "mesh", call->span_)
      << R"(pld.tensor.allreduce mode must be "ring" or "mesh", got ")" << mode << "\"";
  return mode == "ring";
}

[[nodiscard]] bool IsHostOrch(const FunctionPtr& func) {
  if (!func || !func->level_.has_value() || *func->level_ != Level::HOST) return false;
  return func->func_type_ == FunctionType::Orchestration ||
         (func->role_.has_value() && *func->role_ == Role::Orchestrator);
}

[[nodiscard]] WindowBufferPtr GetWindowBuffer(const ExprPtr& expr, const char* context) {
  auto dist_type = As<DistributedTensorType>(expr->GetType());
  INTERNAL_CHECK_SPAN(dist_type, expr->span_)
      << "LowerHostTensorCollectives: " << context << " must be DistributedTensorType";
  INTERNAL_CHECK_SPAN(dist_type->window_buffer_.has_value(), expr->span_)
      << "LowerHostTensorCollectives: " << context << " must have materialized WindowBuffer back-references";
  return *dist_type->window_buffer_;
}

[[nodiscard]] VarPtr MintDistributedResultVar(const VarPtr& old_var, const ExprPtr& src) {
  auto lhs_type = As<DistributedTensorType>(old_var->GetType());
  INTERNAL_CHECK_SPAN(lhs_type, old_var->span_)
      << "LowerHostTensorCollectives: collective result Var should have DistributedTensorType";
  auto src_type = As<DistributedTensorType>(src->GetType());
  INTERNAL_CHECK_SPAN(src_type && src_type->window_buffer_.has_value(), old_var->span_)
      << "LowerHostTensorCollectives: collective alias source must carry a materialized WindowBuffer";
  auto new_type =
      std::make_shared<const DistributedTensorType>(lhs_type->shape_, lhs_type->dtype_, lhs_type->memref_,
                                                    lhs_type->tensor_view_, src_type->window_buffer_);
  return std::make_shared<Var>(old_var->name_hint_, new_type, old_var->span_);
}

[[nodiscard]] bool ScopeContainsSlot(const CommDomainScopeStmtPtr& scope, const WindowBufferPtr& wb) {
  for (const auto& slot : scope->slots_) {
    if (slot.get() == wb.get()) return true;
  }
  return false;
}

[[nodiscard]] CommDomainScopeStmtPtr FindScopeForBuffers(
    const std::vector<CommDomainScopeStmtPtr>& scope_stack, const std::vector<NamedWindowBuffer>& buffers) {
  INTERNAL_CHECK(!buffers.empty()) << "LowerHostTensorCollectives: scope lookup needs at least one buffer";
  for (auto it = scope_stack.rbegin(); it != scope_stack.rend(); ++it) {
    const auto& scope = *it;
    bool all_present = true;
    for (const auto& named : buffers) {
      if (!ScopeContainsSlot(scope, named.buffer)) {
        all_present = false;
        break;
      }
    }
    if (all_present) return scope;
  }
  return nullptr;
}

// broadcast's `root` kwarg selects a specific rank; the type deducer only
// validates `root >= 0` (the participating device count isn't known there).
// On an explicit static device subset, an out-of-range root would remote-load
// from a rank outside the subset — check the upper bound here, where the
// subset size is known. The fully-dynamic "all device" domain is left
// unchecked (no compile-time device count to check against), same documented
// limitation as CheckStaticSignalCapacity's dynamic-domain gap.
void CheckBroadcastRootInRange(const CallPtr& call, size_t participating_devices) {
  auto root_value = call->GetKwarg<int>("root");
  CHECK_SPAN(root_value >= 0 && static_cast<size_t>(root_value) < participating_devices, call->span_)
      << "pld.tensor.broadcast root (" << root_value << ") must be a valid rank in [0, "
      << participating_devices << ") for this explicit device subset";
}

/// Validate a collective signal's shape against the participating device count.
///
/// ``required_lanes`` is the number of per-peer signal lanes the collective
/// needs — one for every collective except the mesh AllReduce, which needs one
/// lane per launched SPMD block. ``allow_wider_lanes`` lets an explicitly
/// supplied signal carry spare lanes beyond the required count.
///
/// The builtins index the signal as a flat row-major array. That is always
/// sound here because HOST collective signals originate from
/// ``pld.tensor.window``, whose type deducer builds a plain
/// ``DistributedTensorType(shape, dtype)`` with no ``tensor_view_`` — a
/// window-bound signal is packed by construction, so there is no strided or
/// partial-view case to reject.
void CheckStaticSignalCapacity(const CallPtr& call, const ExprPtr& signal_expr, size_t required_slots,
                               int64_t required_lanes = 1, bool allow_wider_lanes = false) {
  auto signal_type = As<DistributedTensorType>(signal_expr->GetType());
  INTERNAL_CHECK_SPAN(signal_type, call->span_)
      << "LowerHostTensorCollectives: collective signal must be DistributedTensorType";
  CHECK_SPAN(signal_type->shape_.size() == 1 || signal_type->shape_.size() == 2, call->span_)
      << "LowerHostTensorCollectives: collective signal must be rank-1 [world_size] "
         "or rank-2 [world_size, signal_stride]";
  if (signal_type->shape_.empty()) return;
  CHECK_SPAN(signal_type->shape_.size() == 2 || required_lanes == 1, call->span_)
      << "LowerHostTensorCollectives: rank-1 signal is valid only when one signal lane is required";
  if (signal_type->shape_.size() == 2) {
    auto second_extent = As<ConstInt>(signal_type->shape_[1]);
    CHECK_SPAN(second_extent, call->span_)
        << "LowerHostTensorCollectives: collective rank-2 signal shape[1] must be constant";
    if (allow_wider_lanes) {
      CHECK_SPAN(second_extent->value_ >= required_lanes, call->span_)
          << "LowerHostTensorCollectives: collective rank-2 signal shape[1] (" << second_extent->value_
          << ") must be at least the required lane count (" << required_lanes << ")";
    } else {
      CHECK_SPAN(second_extent->value_ == required_lanes, call->span_)
          << "LowerHostTensorCollectives: collective rank-2 signal shape[1] (" << second_extent->value_
          << ") must equal the required lane count (" << required_lanes << ")";
    }
  }
  auto extent = As<ConstInt>(signal_type->shape_[0]);
  if (!extent) return;
  CHECK_SPAN(extent->value_ >= static_cast<int64_t>(required_slots), call->span_)
      << "LowerHostTensorCollectives: collective signal shape[0] (" << extent->value_
      << ") must be at least the participating device count (" << required_slots << ")";
}

void CheckRingChunkConstraints(const CallPtr& call, const ExprPtr& src_expr, size_t world_size) {
  auto src_type = As<DistributedTensorType>(src_expr->GetType());
  INTERNAL_CHECK_SPAN(src_type, call->span_)
      << "LowerHostTensorCollectives: ring allreduce src must be DistributedTensorType";
  if (src_type->shape_.empty()) return;

  // The ring kernel partitions src into NR contiguous compile-time chunks
  // (chunk_elems = numel / NR), so the host-ring path requires a
  // statically-known src shape.  A dynamic extent would let a runtime numel
  // that is not divisible by NR reach the kernel, which silently returns
  // unreduced data instead of failing — reject dynamic host-ring extents here.
  int64_t src_numel = 1;
  for (const auto& dim : src_type->shape_) {
    auto extent = As<ConstInt>(dim);
    CHECK_SPAN(extent, call->span_)
        << "LowerHostTensorCollectives: ring allreduce requires a statically-known "
           "src shape (dynamic host-ring extents are not supported; the ring "
           "schedule partitions src into NR compile-time chunks)";
    src_numel *= extent->value_;
  }

  int64_t nr = static_cast<int64_t>(world_size);

  // Divisibility: the host builtin ring schedule partitions data into NR
  // contiguous chunks of chunk_elems = numel // NR.  A non-divisible numel
  // would produce a trailing partial chunk that the kernel cannot handle.
  CHECK_SPAN(src_numel % nr == 0, call->span_)
      << "LowerHostTensorCollectives: ring allreduce requires the per-rank data size (product of src shape = "
      << src_numel << ") to be an exact multiple of the rank count (" << nr << "); got a remainder of "
      << (src_numel % nr);
}

void CheckRingSignalCapacity(const CallPtr& call, const ExprPtr& signal_expr, size_t world_size) {
  auto signal_type = As<DistributedTensorType>(signal_expr->GetType());
  INTERNAL_CHECK_SPAN(signal_type, call->span_)
      << "LowerHostTensorCollectives: ring allreduce signal must be DistributedTensorType";
  CHECK_SPAN(signal_type->dtype_ == DataType::INT32, call->span_)
      << "LowerHostTensorCollectives: ring allreduce signal dtype must be INT32, got "
      << signal_type->dtype_.ToString();
  CHECK_SPAN(signal_type->shape_.size() == 2, call->span_)
      << "LowerHostTensorCollectives: ring allreduce signal must be rank-2 [2*(NR-1) + 1, NR], got rank "
      << signal_type->shape_.size();

  const auto required_rounds = static_cast<int64_t>(2 * (world_size - 1) + 1);
  auto sig_rounds = As<ConstInt>(signal_type->shape_[0]);
  if (sig_rounds) {
    CHECK_SPAN(sig_rounds->value_ >= required_rounds, call->span_)
        << "LowerHostTensorCollectives: ring allreduce signal shape[0] (" << sig_rounds->value_
        << ") must be at least 2*(NR-1) + 1 = " << required_rounds << " for NR = " << world_size;
  }
  auto sig_nr = As<ConstInt>(signal_type->shape_[1]);
  if (sig_nr) {
    CHECK_SPAN(sig_nr->value_ == static_cast<int64_t>(world_size), call->span_)
        << "LowerHostTensorCollectives: ring allreduce signal shape[1] (" << sig_nr->value_
        << ") must equal the participating device count (" << world_size << ")";
  }
  if (sig_rounds && sig_nr && sig_nr->value_ > 0) {
    // Exact match mirrors builtin.tensor.allreduce_ring's type deducer so an
    // internally-inconsistent signal is rejected here with a clear message
    // instead of failing later at builtin construction.
    const auto expected_rounds = 2 * (sig_nr->value_ - 1) + 1;
    CHECK_SPAN(sig_rounds->value_ == expected_rounds, call->span_)
        << "LowerHostTensorCollectives: ring allreduce signal shape[0] (" << sig_rounds->value_
        << ") must equal 2*(NR-1) + 1 = " << expected_rounds << " for NR = " << sig_nr->value_;
  }
}

/// Reject a `core_num` that the configured backend could never admit.
///
/// The mesh AllReduce builtin is submitted through `rt_submit_aiv_task`, so it
/// is a standalone AIV kernel: one logical block maps to one AIV core, and the
/// bound is the vector-core count rather than the cube-core count (the same
/// mapping VerifyHardSyncAllOccupancy applies to standalone AIV kernels).
///
/// This bound is a correctness requirement, not an optimisation. The generated
/// launch sets `require_sync_start`, which admits all blocks atomically, so a
/// request above the physical core count can never be admitted — the device
/// would hang rather than report an error. Reject it at compile time instead.
///
/// Pure-IR unit tests run without a configured backend; there is nothing to
/// bound in that case.
void CheckAllReduceCoreCapacity(const CallPtr& call, int64_t core_num) {
  if (!backend::BackendConfig::IsConfigured()) return;
  const auto* be = backend::GetBackend();
  const auto max_blocks = static_cast<int64_t>(be->GetCoreCount(CoreType::VECTOR));
  CHECK_SPAN(core_num <= max_blocks, call->span_)
      << "pld.tensor.allreduce core_num (" << core_num << ") exceeds the backend AIV core count ("
      << max_blocks << ")";
}

/// Validate a HOST AllReduce call.
///
/// ``world_size_known`` is false when the collective lowers to a loop over a
/// dynamic ``pld.system.world_size``. The schedule/``core_num`` compatibility and
/// signal-lane checks are world-size independent and always run; only the ring
/// layout and rank-count checks are skipped when the device set is dynamic.
void CheckAllReduceSignalCapacity(const CallPtr& call, const ExprPtr& signal_expr, size_t world_size,
                                  bool world_size_known) {
  const auto core_num = static_cast<int64_t>(call->GetKwarg<int>("core_num"));
  if (ShouldLowerAllReduceAsRing(call)) {
    // The ring builtin runs a single block per rank; multicore is mesh-only.
    CHECK_SPAN(core_num == 1, call->span_)
        << R"(HOST pld.tensor.allreduce mode="ring" does not support core_num > 1, got core_num=)" << core_num
        << R"(; use mode="mesh" for a multi-core AllReduce)";
    if (!world_size_known) return;
    CheckRingSignalCapacity(call, signal_expr, world_size);
    CHECK_SPAN(world_size <= kMaxSupportedRanks, call->span_)
        << "LowerHostTensorCollectives: ring allreduce requires " << static_cast<int>(kMaxSupportedRanks)
        << " or fewer participating devices, got " << world_size;
    INTERNAL_CHECK_SPAN(call->args_.size() >= 1, call->span_)
        << "LowerHostTensorCollectives: ring allreduce requires a src arg";
    CheckRingChunkConstraints(call, call->args_[0], world_size);
    return;
  }
  CheckAllReduceCoreCapacity(call, core_num);
  // A ``world_size`` of 0 makes the shape[0] bound vacuous, which is exactly the
  // right behaviour when the participating device count is only known at runtime.
  CheckStaticSignalCapacity(call, signal_expr, world_size, core_num, /*allow_wider_lanes=*/true);
}

[[nodiscard]] CallPtr MakeBuiltinCallWithAttrs(const std::string& builtin_name, const CallPtr& call,
                                               const std::vector<ExprPtr>& args,
                                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                                               const ExprPtr& device,
                                               std::vector<std::pair<std::string, std::any>> attrs,
                                               std::vector<ArgDirection> arg_directions) {
  auto builtin = OpRegistry::GetInstance().CreateInternal(builtin_name, args, kwargs, call->span_);
  attrs.emplace_back(kAttrDevice, device);
  attrs = WithArgDirectionsAttr(std::move(attrs), std::move(arg_directions));
  return std::make_shared<Call>(builtin->op_, builtin->args_, builtin->kwargs_, std::move(attrs),
                                builtin->GetType(), builtin->span_);
}

[[nodiscard]] CallPtr MakeBuiltinAllReduce(const CallPtr& call, const ExprPtr& device) {
  auto src_type = As<DistributedTensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(src_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.allreduce src must be DistributedTensorType";
  auto op_value = call->GetKwarg<int>("op");
  const bool as_ring = ShouldLowerAllReduceAsRing(call);
  std::vector<std::pair<std::string, std::any>> kwargs = {
      {"op", op_value},
      {"dtype", src_type->dtype_},
  };
  std::vector<std::pair<std::string, std::any>> attrs = {
      {"op", op_value},
      {"dtype", src_type->dtype_},
  };
  // Only the mesh builtin launches an SPMD grid, so only it declares `core_num`.
  // The ring builtin runs a single block per rank.
  if (!as_ring) {
    const auto core_num = call->GetKwarg<int>("core_num");
    kwargs.emplace_back("core_num", core_num);
    attrs.emplace_back("core_num", core_num);
  }
  INTERNAL_CHECK_SPAN(call->args_.size() >= 2, call->span_)
      << "LowerHostTensorCollectives: expected pld.tensor.allreduce to have an explicit signal by the time "
         "this pass runs";
  const char* builtin_name = as_ring ? "builtin.tensor.allreduce_ring" : "builtin.tensor.allreduce";
  return MakeBuiltinCallWithAttrs(builtin_name, call, call->args_, kwargs, device, std::move(attrs),
                                  {ArgDirection::InOut, ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinBarrier(const CallPtr& call, const ExprPtr& device) {
  return MakeBuiltinCallWithAttrs("builtin.tensor.barrier", call, call->args_, {}, device, {},
                                  {ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinBroadcast(const CallPtr& call, const ExprPtr& device) {
  auto target_type = As<DistributedTensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(target_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.broadcast target must be DistributedTensorType";
  auto root_value = call->GetKwarg<int>("root");
  std::vector<std::pair<std::string, std::any>> kwargs = {{"root", root_value},
                                                          {"dtype", target_type->dtype_}};
  std::vector<std::pair<std::string, std::any>> attrs = {
      {"root", root_value},
      {"dtype", target_type->dtype_},
  };
  return MakeBuiltinCallWithAttrs("builtin.tensor.broadcast", call, call->args_, kwargs, device,
                                  std::move(attrs), {ArgDirection::InOut, ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinReduceScatter(const CallPtr& call, const ExprPtr& device) {
  auto target_type = As<DistributedTensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(target_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.reduce_scatter target must be DistributedTensorType";
  auto op_value = call->GetKwarg<int>("op");
  std::vector<std::pair<std::string, std::any>> kwargs = {
      {"op", op_value},
      {"dtype", target_type->dtype_},
  };
  std::vector<std::pair<std::string, std::any>> attrs = {
      {"op", op_value},
      {"dtype", target_type->dtype_},
  };
  return MakeBuiltinCallWithAttrs("builtin.tensor.reduce_scatter", call, call->args_, kwargs, device,
                                  std::move(attrs), {ArgDirection::InOut, ArgDirection::InOut});
}

// pld.tensor.all_to_all_v's public deducer accepts a plain Tensor for `input`
// and `send_counts` (AsTensorTypeLike) — legitimate on the InCore composite
// path, where LowerCompositeOps consumes them directly. The HOST builtin path
// requires both to be window-bound (EmitBuiltinWindowCollectiveDispatch has no
// dispatch arm for a plain Tensor), so a plain-Tensor value reaching here is
// user-reachable HOST-specific input, not a compiler invariant violation.
// A user-declared pld.DistributedTensor parameter is equally user-reachable
// (window_buffer_ == nullopt until pld.tensor.window binds it) and must be
// rejected the same way — reject both with a CHECK_SPAN (not GetWindowBuffer's
// INTERNAL_CHECK_SPAN) before any window-buffer lookup is attempted.
void CheckHostWindowBoundArg(const ExprPtr& expr, const char* op_name, const char* role) {
  auto dist_type = As<DistributedTensorType>(expr->GetType());
  CHECK_SPAN(dist_type != nullptr && dist_type->window_buffer_.has_value(), expr->span_)
      << op_name << " " << role
      << " must be a window-bound DistributedTensor (a view of an alloc_window_buffer) when "
         "called from a HOST orchestrator; a plain Tensor or an unbound DistributedTensor "
         "parameter is only supported on the InCore composite path";
}
[[nodiscard]] CallPtr MakeBuiltinAllGather(const CallPtr& call, const ExprPtr& device) {
  // Emit namesake builtin: in-kernel TPUT push (this rank's chunk from the
  // `input` staging window into every peer's `target` window) + barrier
  // (TNOTIFY / TWAIT), all in a single AIV kernel. All window operands must
  // be pairwise distinct (checked generically in LowerCollective via
  // scope_buffers). All chips must run concurrently — the host orchestrator
  // submits asynchronously.
  auto target_type = As<DistributedTensorType>(call->args_[1]->GetType());
  return MakeBuiltinCallWithAttrs(
      "builtin.tensor.allgather", call,
      {call->args_[0], call->args_[1], call->args_[2]},  // (input, target, signal)
      {{"dtype", target_type->dtype_}}, device, {{"dtype", target_type->dtype_}},
      {ArgDirection::Input, ArgDirection::InOut, ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinAllToAll(const CallPtr& call, const ExprPtr& device) {
  // Emit namesake builtin: in-kernel TPUT push (this rank's chunks from the
  // `input` staging window into every peer's `target` window) + barrier
  // (TNOTIFY / TWAIT), all in a single AIV kernel. All window operands must
  // be pairwise distinct (checked generically in LowerCollective via
  // scope_buffers). All chips must run concurrently — the host orchestrator
  // submits asynchronously.
  auto target_type = As<DistributedTensorType>(call->args_[1]->GetType());
  return MakeBuiltinCallWithAttrs(
      "builtin.tensor.all_to_all", call,
      {call->args_[0], call->args_[1], call->args_[2]},  // (input, target, signal)
      {{"dtype", target_type->dtype_}}, device, {{"dtype", target_type->dtype_}},
      {ArgDirection::Input, ArgDirection::InOut, ArgDirection::InOut});
}

[[nodiscard]] CallPtr MakeBuiltinAllToAllV(const CallPtr& call, const ExprPtr& device) {
  // Emit namesake builtin: in-kernel TPUT push of the full per-destination
  // MAX_RECV block (this rank's chunks from `input` into every peer's
  // `target` window) + an inline cross-rank publish of the runtime-clamped
  // send_counts[dest] into peer `recv_counts[my_rank, 0]` + one barrier
  // (TNOTIFY / TWAIT), all in a single AIV kernel. All five operands (input,
  // target, signal, send_counts, recv_counts) must be pairwise-distinct
  // windows (checked generically in LowerCollective via scope_buffers). All
  // chips must run concurrently — the host orchestrator submits asynchronously.
  auto target_type = As<DistributedTensorType>(call->args_[1]->GetType());
  INTERNAL_CHECK_SPAN(target_type, call->span_)
      << "LowerHostTensorCollectives: pld.tensor.all_to_all_v target must be DistributedTensorType";

  return MakeBuiltinCallWithAttrs(
      "builtin.tensor.all_to_all_v", call,
      {call->args_[0], call->args_[1], call->args_[2], call->args_[3], call->args_[4]},
      {{"dtype", target_type->dtype_}}, device, {{"dtype", target_type->dtype_}},
      {ArgDirection::Input, ArgDirection::InOut, ArgDirection::InOut, ArgDirection::Input,
       ArgDirection::InOut});
}

struct HostCollectiveRule {
  const char* pld_name;
  using MakeBuiltinFn = std::function<CallPtr(const CallPtr&, const ExprPtr&)>;
  using ScopeBuffersFn = std::function<std::vector<NamedWindowBuffer>(const CallPtr&)>;
  using SignalExprFn = std::function<ExprPtr(const CallPtr&)>;
  using AliasSourceFn = std::function<std::optional<ExprPtr>(const CallPtr&)>;
  // Optional extra static-domain check beyond the generic signal-capacity
  // bound (CheckStaticSignalCapacity) — e.g. broadcast's root-in-range check.
  // Empty for rules that don't need one.
  using ExtraStaticCheckFn = std::function<void(const CallPtr&, size_t)>;
  MakeBuiltinFn make_builtin;
  ScopeBuffersFn scope_buffers;
  SignalExprFn signal_expr;
  AliasSourceFn alias_source;
  ExtraStaticCheckFn extra_static_check;
  // Args whose public deducer accepts a plain Tensor (the InCore composite
  // path) but which the HOST builtin requires window-bound. Checked with
  // CHECK_SPAN before GetWindowBuffer, so a plain Tensor reaching HOST
  // lowering is a user error, not an INTERNAL_CHECK. (arg index, role)
  std::vector<std::pair<size_t, const char*>> host_bound_args;
};

[[nodiscard]] const HostCollectiveRule* LookupHostCollectiveRule(const std::string& op_name) {
  static const HostCollectiveRule kRules[] = {
      {
          "pld.tensor.allreduce",
          &MakeBuiltinAllReduce,
          [](const CallPtr& call) {
            return std::vector<NamedWindowBuffer>{
                {GetWindowBuffer(call->args_[0], "allreduce src"), "src"},
                {GetWindowBuffer(call->args_[1], "allreduce signal"), "signal"},
            };
          },
          [](const CallPtr& call) { return call->args_[1]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
      },
      {
          "pld.tensor.barrier",
          &MakeBuiltinBarrier,
          [](const CallPtr& call) {
            return std::vector<NamedWindowBuffer>{
                {GetWindowBuffer(call->args_[0], "barrier signal"), "signal"},
            };
          },
          [](const CallPtr& call) { return call->args_[0]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
      },
      {
          "pld.tensor.broadcast",
          &MakeBuiltinBroadcast,
          [](const CallPtr& call) {
            return std::vector<NamedWindowBuffer>{
                {GetWindowBuffer(call->args_[0], "broadcast target"), "target"},
                {GetWindowBuffer(call->args_[1], "broadcast signal"), "signal"},
            };
          },
          [](const CallPtr& call) { return call->args_[1]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
          &CheckBroadcastRootInRange,
      },
      {
          "pld.tensor.reduce_scatter",
          &MakeBuiltinReduceScatter,
          [](const CallPtr& call) {
            return std::vector<NamedWindowBuffer>{
                {GetWindowBuffer(call->args_[0], "reduce_scatter target"), "target"},
                {GetWindowBuffer(call->args_[1], "reduce_scatter signal"), "signal"},
            };
          },
          [](const CallPtr& call) { return call->args_[1]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[0]; },
      },
      {
          "pld.tensor.allgather",
          &MakeBuiltinAllGather,
          [](const CallPtr& call) {
            return std::vector<NamedWindowBuffer>{
                {GetWindowBuffer(call->args_[0], "allgather input"), "input"},
                {GetWindowBuffer(call->args_[1], "allgather target"), "target"},
                {GetWindowBuffer(call->args_[2], "allgather signal"), "signal"},
            };
          },
          [](const CallPtr& call) { return call->args_[2]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[1]; },
          {},              // extra_static_check: none
          {{0, "input"}},  // host_bound_args: local_data must be window-bound on HOST
      },
      {
          "pld.tensor.all_to_all",
          &MakeBuiltinAllToAll,
          [](const CallPtr& call) {
            return std::vector<NamedWindowBuffer>{
                {GetWindowBuffer(call->args_[0], "all_to_all input"), "input"},
                {GetWindowBuffer(call->args_[1], "all_to_all target"), "target"},
                {GetWindowBuffer(call->args_[2], "all_to_all signal"), "signal"},
            };
          },
          [](const CallPtr& call) { return call->args_[2]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[1]; },
          {},              // extra_static_check: none
          {{0, "input"}},  // host_bound_args: input must be window-bound on HOST
      },
      {
          "pld.tensor.all_to_all_v",
          &MakeBuiltinAllToAllV,
          [](const CallPtr& call) {
            return std::vector<NamedWindowBuffer>{
                {GetWindowBuffer(call->args_[0], "all_to_all_v input"), "input"},
                {GetWindowBuffer(call->args_[1], "all_to_all_v target"), "target"},
                {GetWindowBuffer(call->args_[2], "all_to_all_v signal"), "signal"},
                {GetWindowBuffer(call->args_[3], "all_to_all_v send_counts"), "send_counts"},
                {GetWindowBuffer(call->args_[4], "all_to_all_v recv_counts"), "recv_counts"},
            };
          },
          [](const CallPtr& call) { return call->args_[2]; },
          [](const CallPtr& call) -> std::optional<ExprPtr> { return call->args_[1]; },
          {},                                  // extra_static_check: none
          {{0, "input"}, {3, "send_counts"}},  // host_bound_args
      },
  };
  for (const auto& rule : kRules) {
    if (op_name == rule.pld_name) return &rule;
  }
  return nullptr;
}

[[nodiscard]] bool IsHostTensorCollective(const CallPtr& call) {
  return call && call->op_ && LookupHostCollectiveRule(call->op_->name_) != nullptr;
}

StmtPtr EmitPerDeviceBuiltinCalls(const CallPtr& call, const HostCollectiveRule& rule,
                                  const CommDomainScopeStmtPtr& scope, const Span& span,
                                  const std::vector<std::string>& leading_comments) {
  if (!scope->devices_.empty()) {
    if (IsOp(call, "pld.tensor.allreduce")) {
      CheckAllReduceSignalCapacity(call, rule.signal_expr(call), scope->devices_.size(),
                                   /*world_size_known=*/true);
    } else {
      CheckStaticSignalCapacity(call, rule.signal_expr(call), scope->devices_.size());
    }
    if (rule.extra_static_check) rule.extra_static_check(call, scope->devices_.size());
    std::vector<StmtPtr> stmts;
    stmts.reserve(scope->devices_.size());
    for (auto device : scope->devices_) {
      auto device_expr = std::make_shared<ConstInt>(device, DataType::INT64, call->span_);
      stmts.push_back(std::make_shared<EvalStmt>(rule.make_builtin(call, device_expr), call->span_));
    }
    return std::make_shared<SeqStmts>(std::move(stmts), span, leading_comments);
  }

  // NOTE: for the fully-dynamic all-device domain (devices_ empty), signal's
  // compile-time shape[0] is trusted to equal the runtime world_size — PyPTO
  // has no runtime-assert IR primitive today to check this at compile time
  // (see KNOWN_ISSUES.md).
  auto loop_var = std::make_shared<Var>("r", std::make_shared<ScalarType>(DataType::INT64), call->span_);
  auto zero = std::make_shared<ConstInt>(0, DataType::INT64, call->span_);
  auto one = std::make_shared<ConstInt>(1, DataType::INT64, call->span_);
  auto stop = OpRegistry::GetInstance().Create("pld.system.world_size", {}, call->span_);
  // The device set is dynamic here, so world-size-dependent capacity cannot be
  // checked; pass 0 so only the world-size-independent constraints apply.
  if (IsOp(call, "pld.tensor.allreduce")) {
    CheckAllReduceSignalCapacity(call, rule.signal_expr(call), 0, /*world_size_known=*/false);
  } else {
    CheckStaticSignalCapacity(call, rule.signal_expr(call), 0);
  }
  auto body = std::make_shared<EvalStmt>(rule.make_builtin(call, loop_var), call->span_);
  return std::make_shared<ForStmt>(loop_var, zero, stop, one, std::vector<IterArgPtr>{}, body,
                                   std::vector<VarPtr>{}, span, ForKind::Sequential,
                                   std::vector<std::pair<std::string, std::any>>{}, leading_comments);
}

class LowerHostTensorCollectivesMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    scope_stack_.push_back(op);
    auto new_body = VisitStmt(op->body_);
    scope_stack_.pop_back();
    if (new_body.get() == op->body_.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = new_body;
    return result;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (IsHostTensorCollective(call)) {
      auto visited_call = As<Call>(VisitExpr(op->expr_));
      INTERNAL_CHECK_SPAN(IsHostTensorCollective(visited_call), op->span_)
          << "LowerHostTensorCollectives: collective EvalStmt rewrote to a non-collective expression";
      return LowerCollective(visited_call, op->span_, op->leading_comments_);
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!IsHostTensorCollective(call)) {
      return IRMutator::VisitStmt_(op);
    }
    auto visited_call = As<Call>(VisitExpr(op->value_));
    INTERNAL_CHECK_SPAN(IsHostTensorCollective(visited_call), op->span_)
        << "LowerHostTensorCollectives: collective AssignStmt rewrote to a non-collective expression";
    std::vector<StmtPtr> stmts;
    stmts.push_back(LowerCollective(visited_call, op->span_, op->leading_comments_));
    const auto* rule = LookupHostCollectiveRule(visited_call->op_->name_);
    INTERNAL_CHECK(rule) << "LowerHostTensorCollectives: missing rule for " << visited_call->op_->name_;
    if (auto alias_src = rule->alias_source(visited_call)) {
      auto result_var = MintDistributedResultVar(op->var_, *alias_src);
      var_remap_[op->var_.get()] = result_var;
      stmts.push_back(std::make_shared<AssignStmt>(result_var, *alias_src, op->span_));
    }
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

 private:
  StmtPtr LowerCollective(const CallPtr& call, const Span& span,
                          const std::vector<std::string>& leading_comments) {
    const auto* rule = LookupHostCollectiveRule(call->op_->name_);
    INTERNAL_CHECK(rule) << "LowerHostTensorCollectives: missing rule for " << call->op_->name_;
    INTERNAL_CHECK_SPAN(!scope_stack_.empty(), call->span_)
        << "LowerHostTensorCollectives: " << call->op_->name_ << " must appear inside a CommDomainScopeStmt";
    for (const auto& [arg_idx, role] : rule->host_bound_args) {
      CheckHostWindowBoundArg(call->args_[arg_idx], call->op_->name_.c_str(), role);
    }
    auto buffers = rule->scope_buffers(call);
    CheckPairwiseDistinctWindows(buffers, call->span_, call->op_->name_.c_str());
    auto scope = FindScopeForBuffers(scope_stack_, buffers);
    INTERNAL_CHECK_SPAN(scope, call->span_) << "LowerHostTensorCollectives: " << call->op_->name_
                                            << " window buffers must resolve to the same comm-domain scope";
    return EmitPerDeviceBuiltinCalls(call, *rule, scope, span, leading_comments);
  }

  std::vector<CommDomainScopeStmtPtr> scope_stack_;
};

FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (!IsHostOrch(func)) return func;
  LowerHostTensorCollectivesMutator mutator;
  return mutator.VisitFunction(func);
}

ProgramPtr TransformProgram(const ProgramPtr& program) {
  bool modified = false;
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
  for (const auto& [gvar, func] : program->functions_) {
    auto new_func = TransformFunction(func);
    new_functions[gvar] = new_func;
    if (new_func.get() != func.get()) modified = true;
  }
  if (!modified) return program;
  return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
}

}  // namespace

namespace pass {

Pass LowerHostTensorCollectives() {
  return CreateProgramPass(TransformProgram, "LowerHostTensorCollectives",
                           kLowerHostTensorCollectivesProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
