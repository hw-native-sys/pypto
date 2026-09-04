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

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_allocator_policy.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/dsa/dsa_reuse_penalty_solver.h"
#include "pypto/ir/transforms/dsa/memref_dsa_adapter.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memory_footprint.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/reserve_buffer_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

using MemRefWithSpace = std::pair<MemRefPtr, MemorySpace>;
// ReserveBufferBaseMap / ReservedEndBySpace / ResolveReserveBufferBases now live in the shared
// reserve_buffer_utils.h so AllocateMemoryAddr and MemoryReuse resolve the reserved region identically.

/// Whether `resolution` holds one of the automatic cross-core pipe rings, i.e.
/// whether `pl.cross_core_slot(slot_num=N)` can actually move these bytes.
///
/// Matched EXACTLY, not by suffix. BuildAutomaticPipeSetup mints the ring's name as
/// BuildPipeBufferName(<mixed kernel name>, dir) while ExpandMixedKernel names the
/// halves "<mixed kernel name>_aic" / "_aiv", so the expected names are
/// reconstructible from this function's own name. That precision matters because
/// `pl.reserve_buffer` takes an arbitrary name: a hand-authored
/// "scratch_v2c_slot_buffer" would pass a suffix test and then be pointed at a knob
/// that cannot resize it.
bool HasCrossCorePipeRing(const ReserveBufferResolution& resolution, const std::string& func_name) {
  auto strip_suffix = [&func_name](const std::string& suffix) -> std::string {
    if (func_name.size() <= suffix.size()) return "";
    if (func_name.compare(func_name.size() - suffix.size(), suffix.size(), suffix) != 0) return "";
    return func_name.substr(0, func_name.size() - suffix.size());
  };
  std::vector<std::string> expected;
  for (const auto& kernel : {strip_suffix("_aic"), strip_suffix("_aiv")}) {
    if (kernel.empty()) continue;
    expected.push_back(cross_core_pipe::BuildPipeBufferName(kernel, core_affinity::PipeDirection::C2V));
    expected.push_back(cross_core_pipe::BuildPipeBufferName(kernel, core_affinity::PipeDirection::V2C));
  }
  if (expected.empty()) return false;
  for (const auto& [call, base] : resolution.resolved_bases) {
    if (!call) continue;
    const auto name = call->GetKwarg<std::string>("name", "");
    if (std::find(expected.begin(), expected.end(), name) != expected.end()) return true;
  }
  return false;
}

/// The "the first N bytes ... are reserved" clause appended to a capacity-overflow
/// message, or empty when `space` pays for no reserve buffer.
///
/// A reserve_buffer is NOT a MemRef: every tile is allocated *above* it, so it is
/// counted in the footprint yet is invisible in the per-tile accounting an author can
/// inspect. On a cube/vector boundary carrying a large tile the automatic pipe ring is
/// routinely most of the overflow, so name those bytes — and the knob for them, but
/// only when the buffer really is a pipe ring.
///
/// The figure is `reserved_end_by_space`: the aligned max-END that tiles are placed
/// above, which is exactly the quantity charged to this overflow. It is deliberately
/// NOT worded as "bytes the buffers occupy" — an explicitly based buffer
/// (`base=0x1000, size=4096`) or an alignment gap makes the floor exceed the summed
/// buffer sizes, so the wording states the floor.
///
/// Shared by the in-pass capacity CHECK and the AllocatedMemoryAddr verifier so the two
/// explain the same overflow identically. Which of them a given compile hits depends on
/// configuration (the pass CHECK is skipped under memory_planner=PTOAS, which skips the
/// pass), so the wording must not live in only one of them.
std::string ReservedBytesNote(const ReserveBufferResolution& resolution, MemorySpace space,
                              const std::string& func_name) {
  const auto& ends = resolution.reserved_end_by_space;
  auto it = ends.find(space);
  if (it == ends.end() || it->second == 0) return "";
  std::string note = ". The first " + std::to_string(it->second) +
                     " bytes of that space are reserved by system.reserve_buffer, so tiles are "
                     "allocated above them";
  if (HasCrossCorePipeRing(resolution, func_name)) {
    note +=
        " — this is the cross-core pipe ring. Lower its depth with "
        "optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the "
        "tile that crosses the cube/vector boundary";
  }
  return note;
}

class StripPipelineMembershipMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    if (!call || !call->HasAttr(kPipelineMembershipAttr)) return visited;
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_,
                                  StripAttr(call->attrs_, kPipelineMembershipAttr), call->GetType(),
                                  call->span_);
  }
};

bool IsUnusedAllocStmt(const StmtPtr& stmt, const std::set<const Var*>& used_bases) {
  const auto assign = As<AssignStmt>(stmt);
  const auto call = assign ? As<Call>(assign->value_) : nullptr;
  return call && (IsOp(call, "tile.alloc") || IsOp(call, "tensor.alloc")) &&
         used_bases.count(assign->var_.get()) == 0;
}

StmtPtr RemoveUnusedAllocStatements(const StmtPtr& body) {
  const auto used_bases = memref_collectors::CollectUsedBasePtrs(body);
  const auto sequence = As<SeqStmts>(body);
  if (!sequence) return body;

  std::vector<StmtPtr> retained;
  retained.reserve(sequence->stmts_.size());
  for (const StmtPtr& statement : sequence->stmts_) {
    if (!IsUnusedAllocStmt(statement, used_bases)) retained.push_back(statement);
  }
  return retained.size() == sequence->stmts_.size() ? body
                                                    : SeqStmts::Flatten(std::move(retained), body->span_);
}

// Mutator to update MemRef addresses in IR (both variable types and alloc statements)
class MemRefUpdateMutator : public IRMutator {
 public:
  explicit MemRefUpdateMutator(const std::vector<std::pair<const MemRef*, MemRefPtr>>& memref_pairs,
                               ReserveBufferBaseMap reserve_buffer_bases)
      : reserve_buffer_bases_(std::move(reserve_buffer_bases)) {
    for (const auto& [old_ptr, new_memref] : memref_pairs) {
      memref_map_[old_ptr] = new_memref;
    }
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    // Check if already remapped (same old pointer seen again).
    auto it = var_remap_.find(op.get());
    if (it != var_remap_.end()) {
      return it->second;
    }
    TypePtr new_type = UpdateTypeMemRef(op->GetType());
    if (new_type != op->GetType()) {
      auto new_var = std::make_shared<Var>(op->name_hint_, new_type, op->span_);
      var_remap_[op.get()] = new_var;
      return new_var;
    }
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // Check if already remapped.
    auto it = var_remap_.find(op.get());
    if (it != var_remap_.end()) {
      return it->second;
    }
    auto new_init = VisitExpr(op->initValue_);
    TypePtr new_type = UpdateTypeMemRef(op->GetType());

    if (new_init != op->initValue_ || new_type != op->GetType()) {
      auto new_iter_arg = std::make_shared<IterArg>(op->name_hint_, new_type, new_init, op->span_);
      var_remap_[op.get()] = new_iter_arg;
      return new_iter_arg;
    }
    return op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // tile.alloc statements now have Ptr LHS (not MemRef), so no special handling needed.
    // Just fall through to the default mutator which updates types via UpdateTypeMemRef.
    return IRMutator::VisitStmt_(op);
  }

 private:
  std::unordered_map<const MemRef*, MemRefPtr> memref_map_;
  std::unordered_map<const Expr*, ExprPtr> var_remap_;
  ReserveBufferBaseMap reserve_buffer_bases_;

  ExprPtr VisitExpr_(const CallPtr& op) override {
    std::vector<ExprPtr> new_args;
    bool args_changed = false;
    new_args.reserve(op->args_.size());

    for (const auto& arg : op->args_) {
      INTERNAL_CHECK_SPAN(arg, op->span_) << "Call has null argument during AllocateMemoryAddr mutation";
      auto new_arg = IRMutator::VisitExpr(arg);
      INTERNAL_CHECK_SPAN(new_arg, op->span_) << "Call argument mutated to null during AllocateMemoryAddr";
      args_changed = args_changed || new_arg.get() != arg.get();
      new_args.push_back(new_arg);
    }

    std::vector<std::pair<std::string, std::any>> new_kwargs = op->kwargs_;
    bool kwargs_changed = false;
    auto base_it = reserve_buffer_bases_.find(op.get());
    if (base_it != reserve_buffer_bases_.end()) {
      const int resolved_base = static_cast<int>(base_it->second);
      bool found_base = false;
      for (auto& [key, value] : new_kwargs) {
        if (key != "base") continue;
        found_base = true;
        if (AnyCast<int>(value, "kwarg key: base") != resolved_base) {
          value = resolved_base;
          kwargs_changed = true;
        }
        break;
      }
      if (!found_base) {
        new_kwargs.emplace_back("base", resolved_base);
        kwargs_changed = true;
      }
    }

    if (args_changed || kwargs_changed) {
      return std::make_shared<Call>(op->op_, std::move(new_args), std::move(new_kwargs), op->GetType(),
                                    op->span_);
    }
    return op;
  }

  TypePtr UpdateTypeMemRef(const TypePtr& type) {
    auto memref = GetTypeMemRef(type);
    auto new_memref = memref;
    if (memref.has_value()) {
      auto it = memref_map_.find(memref.value().get());
      if (it != memref_map_.end()) {
        new_memref = it->second;
      }
    }
    return CloneTypeWithMemRefAndRemapExprs(type, new_memref,
                                            [this](const ExprPtr& expr) { return VisitExpr(expr); });
  }
};

/// Author-declared (`pinned=True`) allocations in a body: base Ptr -> reserved bytes.
///
/// InitMemRef clears `is_pinned_` on the MemRefs it resolves — the declaration is
/// resolved, and the flag is what confines MemRef rebuilds to the window before
/// this pass. It keeps `slot_count_` / `slot_index_`, but those describe one slot,
/// not the declaration: no resolved MemRef states the reserved size. So by this
/// pass the only record of the allocation *as a whole* is its alloc statement.
/// Two things here depend on it:
///
///  * a declared allocation is the one place a *dynamic* address is meaningful;
///  * it is the one place the allocation is deliberately **larger than any single
///    MemRef in it**. A multi-slot declaration reserves `slots x slot_size` while
///    each slot MemRef is sized to its own slot, so sizing the buffer from the
///    largest member — as every other base can be — would reserve one slot and let
///    the next allocation land on top of slot 1.
///
/// The size therefore has to come from the alloc statement, which InitMemRef
/// already sized to the whole slot set.
class PinnedAllocCollector : public IRVisitor {
 public:
  std::map<const Var*, uint64_t> alloc_sizes;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto base = GetPinnedAllocBase(op)) {
      // CreateAllocStatement builds `alloc(memory_space, size)`, so the size is
      // args_[1]. Absent or non-constant, fall back to the members (0 = no floor).
      uint64_t size = 0;
      if (auto call = As<Call>(op->value_); call && call->args_.size() >= 2) {
        if (auto const_size = As<ConstInt>(call->args_[1]); const_size && const_size->value_ > 0) {
          size = static_cast<uint64_t>(const_size->value_);
        }
      }
      alloc_sizes.emplace(base.get(), size);
    }
    IRVisitor::VisitStmt_(op);
  }
};

/**
 * @brief Allocate memory addresses using the given allocation policy
 *
 * MemRefs sharing the same ``base_`` Ptr are co-located in a single bumped
 * slot sized by the largest member.size_ in the group.  This handles view
 * MemRefs (e.g. produced by ``tile.slice``) — every view physically aliases
 * its parent allocation, so they should share one address slot rather than
 * each consuming size_ bytes of fresh L1.
 *
 * ``pinned_alloc_sizes`` maps each author-declared allocation's base to the bytes
 * its alloc statement reserves. Those are the only bases that may receive a
 * dynamic (expression) address, and the only ones whose buffer can be larger than
 * their largest member (a multi-slot declaration).
 */
std::vector<std::pair<const MemRef*, MemRefPtr>> AllocateMemoryAddresses(
    const std::vector<MemRefWithSpace>& memrefs, const ReserveBufferResolution& reserve_resolution,
    const MemoryAllocatorPolicy& policy, const std::map<const Var*, uint64_t>& pinned_alloc_sizes,
    const std::string& func_name) {
  const ReservedEndBySpace& reserved_end_by_space = reserve_resolution.reserved_end_by_space;
  // Group MemRefs by memory space
  std::unordered_map<MemorySpace, std::vector<MemRefPtr>> space_to_memrefs;
  for (const auto& [memref, memory_space] : memrefs) {
    space_to_memrefs[memory_space].push_back(memref);
  }

  // Create new MemRefs with allocated addresses for each memory space
  std::vector<std::pair<const MemRef*, MemRefPtr>> memref_pairs;

  for (auto& [space, refs] : space_to_memrefs) {
    if (!policy.ShouldAllocate(space)) {
      continue;
    }

    policy.OrderMemRefs(refs);

    // Group MemRefs by base_ Ptr identity.  base_order preserves the policy's
    // sort order via the first MemRef that introduces each base.
    std::map<const Var*, std::vector<MemRefPtr>> base_groups;
    std::vector<const Var*> base_order;
    for (const auto& ref : refs) {
      const Var* base_key = ref->base_.get();
      auto inserted = base_groups.try_emplace(base_key);
      if (inserted.second) base_order.push_back(base_key);
      inserted.first->second.push_back(ref);
    }

    // The ordering + alignment bump walk lives in SpaceFootprint, shared with MemoryReuse's
    // capacity fit check so the two footprints are identical by construction (#1475).
    uint64_t reserved_start = 0;
    auto reserved_it = reserved_end_by_space.find(space);
    if (reserved_it != reserved_end_by_space.end()) {
      reserved_start = reserved_it->second;
    }
    SpaceFootprint footprint(space, policy, reserved_start);
    for (const Var* base_key : base_order) {
      const auto& group = base_groups.at(base_key);

      // Slot size = max member.size_.  The root MemRef (byte_offset == 0) is
      // sized to the full alloc; views are sub-regions and never exceed it.
      uint64_t slot_size = 0;
      for (const auto& ref : group) {
        INTERNAL_CHECK_SPAN(ref->size_ > 0, ref->span_)
            << "AllocateMemoryAddr encountered zero-sized MemRef '" << ref->name_hint_
            << "'. InitMemRef should reject dynamic or invalid allocation shapes before address assignment.";
        slot_size = std::max(slot_size, static_cast<uint64_t>(ref->size_));
      }
      // A declared allocation reserves what its alloc statement says, which for a
      // multi-slot declaration exceeds any single member. Everything else is sized
      // by its largest member, as before.
      uint64_t buffer_size = slot_size;
      auto declared_size = pinned_alloc_sizes.find(base_key);
      if (declared_size != pinned_alloc_sizes.end()) {
        buffer_size = std::max(buffer_size, declared_size->second);
      }
      // Reserve this base-group's physical buffer; base_addr is where its members land.
      const uint64_t base_addr = footprint.OpenBuffer(buffer_size);

      // Bump the whole group to `current_addr`, then preserve each member's
      // own offset within the slot: new byte_offset = base_addr + old offset.
      //
      // InitMemRef already records each view's relative offset (parent offset +
      // the view op's byte offset, see ShareMemRefFrom).  The root MemRef has
      // offset 0, so it lands on base_addr; a ``tile.slice`` view at row k lands
      // on base_addr + k*row_stride.  Codegen reads ``pto.alloc_tile`` addr 1:1
      // from this ConstInt, so a reshape-of-slice chain — whose result inherits
      // the slice's offset but does NOT go through ``pto.subview`` — gets the
      // correct per-view address instead of collapsing onto the parent base
      // (issue #1510).  Pure ``tile.slice`` codegen is unaffected: it still
      // derives the offset from the slice op's own operands off the root base.
      for (const auto& old_memref : group) {
        // Fold a const relative offset into a single ConstInt: base + offset.
        // A reshape-of-slice chain inherits the slice's offset but does NOT go
        // through `pto.subview`, so its address must come from this MemRef
        // (issue #1510).
        //
        // A *dynamic* offset cannot fold. For a declared allocation the address
        // becomes the expression `base_addr + offset` and codegen lowers it into
        // the tile's runtime address assignment — that is how a runtime slot index
        // (`l0c[i % 2]`, scaled to a byte offset by InitMemRef) reaches the
        // hardware. Dropping it would silently address slot 0 every iteration.
        //
        // Every OTHER dynamic offset keeps the old behaviour: fall back to the
        // bare base. Those are `tile.slice` views, which reach codegen through
        // `pto.subview` and re-derive their offset from the slice op's own
        // operands, so this address is unused — and emitting the expression anyway
        // is actively harmful: it is not renderable at the `pto.alloc_tile addr`
        // position and PTOAS rejects the module with "expected SSA operand".
        //
        // INT64 dtype is required by the PTOAS dialect's `pto.alloc_tile` addr
        // operand; PTO codegen reads this dtype from the ConstInt 1:1.
        ExprPtr member_addr_expr;
        auto old_offset = std::dynamic_pointer_cast<const ConstInt>(old_memref->byte_offset_);
        if (!old_offset && pinned_alloc_sizes.count(old_memref->base_.get()) == 0) {
          // Not a declared allocation: bare base, exactly as before slots existed.
          old_offset = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());
        }
        if (old_offset) {
          member_addr_expr = std::make_shared<ConstInt>(static_cast<int64_t>(base_addr) + old_offset->value_,
                                                        DataType::INT64, Span::unknown());
        } else {
          auto base_expr =
              std::make_shared<ConstInt>(static_cast<int64_t>(base_addr), DataType::INDEX, Span::unknown());
          member_addr_expr =
              std::make_shared<Add>(base_expr, old_memref->byte_offset_, DataType::INDEX, Span::unknown());
        }
        // NOTE: MemRef is identity-bearing — each result must get a fresh
        // unique_id_, so build it via the explicit constructor (MutableCopy is
        // static_assert-forbidden for Var/MemRef).
        auto new_memref = std::make_shared<MemRef>(
            old_memref->name_hint_, old_memref->base_, member_addr_expr, old_memref->size_, old_memref->span_,
            old_memref->is_pinned_, old_memref->slot_count_, old_memref->slot_index_);
        memref_pairs.emplace_back(old_memref.get(), new_memref);
      }
    }

    // Capacity is checked HERE because this is the only place the number is exact.
    // The footprint is built from each buffer'strue reserved size, so it counts a
    // declared allocation's unbound slots and does not depend on any tile's address
    // being constant. Reconstructing it downstream from the addressed tiles cannot
    // do either: a slot nobody binds leaves no MemRef to see, and a runtime slot
    // index leaves no constant to add up.
    if (backend::BackendConfig::IsConfigured()) {
      const uint64_t limit = backend::GetBackend()->GetMemSize(space);
      const uint64_t used = footprint.HighWater();
      CHECK(limit == 0 || used <= limit)
          << MemorySpaceToString(space) << " buffer usage (" << used << " bytes) exceeds platform limit ("
          << limit << " bytes)" << ReservedBytesNote(reserve_resolution, space, func_name);
    }
  }

  // Sort by byte_offset (ascending order) so alloc statements are in address order.
  //
  // Comparing by offset only when *both* sides are constant, and by name
  // otherwise, is not a strict weak ordering once dynamic offsets are reachable:
  // with A=(name "z", offset 0), B=(name "a", offset 1) and C=(name "m", dynamic),
  // A < B by offset and B < C by name, yet A < C is false — and a non-transitive
  // comparator makes std::sort undefined behaviour. A declared allocation's
  // runtime slot index made that case reachable.
  //
  // Ordering on one total key instead: constants first in address order, then
  // dynamic addresses by name. `name_hint_` breaks ties among equal offsets so
  // the result is deterministic (two MemRefs of one base group can share an
  // offset, e.g. a view over its parent).
  std::sort(memref_pairs.begin(), memref_pairs.end(),
            [](const std::pair<const MemRef*, MemRefPtr>& a, const std::pair<const MemRef*, MemRefPtr>& b) {
              auto off_a = std::dynamic_pointer_cast<const ConstInt>(a.second->byte_offset_);
              auto off_b = std::dynamic_pointer_cast<const ConstInt>(b.second->byte_offset_);
              if (static_cast<bool>(off_a) != static_cast<bool>(off_b)) {
                return static_cast<bool>(off_a);  // constants before dynamic
              }
              if (off_a && off_b && off_a->value_ != off_b->value_) {
                return off_a->value_ < off_b->value_;
              }
              return a.second->name_hint_ < b.second->name_hint_;
            });

  return memref_pairs;
}

std::string ObviousDsaCapacityOverflow(const dsa::DsaProblem& problem,
                                       const ReserveBufferResolution& reserve_resolution,
                                       const std::string& func_name) {
  for (const dsa::Pool& pool : problem.pools) {
    uint64_t reserved_end = 0;
    for (const dsa::AddressRange& range : pool.reserved_ranges) {
      reserved_end = std::max(reserved_end, range.end);
    }

    uint64_t minimum_end = reserved_end;
    for (const dsa::Buffer& buffer : problem.buffers) {
      if (buffer.pool != pool.id) continue;
      const uint64_t alignment = std::max<uint64_t>(1, buffer.alignment);
      const uint64_t remainder = reserved_end % alignment;
      const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
      if (reserved_end > std::numeric_limits<uint64_t>::max() - padding ||
          reserved_end + padding > std::numeric_limits<uint64_t>::max() - buffer.size) {
        minimum_end = std::numeric_limits<uint64_t>::max();
        break;
      }
      minimum_end = std::max(minimum_end, reserved_end + padding + buffer.size);
    }
    if (minimum_end <= pool.capacity) continue;

    const auto space = static_cast<MemorySpace>(pool.id);
    std::ostringstream message;
    message << MemorySpaceToString(space) << " buffer usage (" << minimum_end
            << " bytes) exceeds platform limit (" << pool.capacity << " bytes)"
            << ReservedBytesNote(reserve_resolution, space, func_name);
    return message.str();
  }
  return "";
}

std::vector<std::pair<const MemRef*, MemRefPtr>> PlanWithDsaRP(
    const FunctionPtr& func, const MemoryAllocatorPolicy& policy,
    const ReserveBufferResolution& reserve_resolution, const std::vector<MemRefWithSpace>& memrefs) {
  const dsa_adapter::AllocationPlan allocation_plan = dsa_adapter::BuildDsaAllocationPlan(func);
  if (allocation_plan.intervals.empty()) return {};

  std::unordered_map<MemorySpace, uint64_t> pool_caps;
  const backend::Backend* active_backend =
      backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;
  if (active_backend != nullptr) {
    for (const LifetimeInterval& lifetime : allocation_plan.intervals) {
      if (pool_caps.count(lifetime.memory_space) != 0) continue;
      const uint64_t capacity = active_backend->GetMemSize(lifetime.memory_space);
      if (capacity > 0) pool_caps[lifetime.memory_space] = capacity;
    }
  }

  const dsa_adapter::PreparedProblem prepared = dsa_adapter::BuildProblem(
      func, allocation_plan, policy, reserve_resolution.reserved_end_by_space, pool_caps, active_backend);
  if (prepared.strict_problem.buffers.empty()) return {};

  const dsa::CanonicalGreedySolver solver;
  dsa::DsaProblem solved_problem = prepared.strict_problem;
  dsa::DsaResult result = solver.Solve(solved_problem);
  if (result.status == dsa::SolveStatus::kNoFit && !prepared.pipeline_pairs.empty()) {
    solved_problem = dsa_adapter::RelaxPipelineIntent(prepared);
    result = solver.Solve(solved_problem);
  }

  INTERNAL_CHECK_SPAN(result.status != dsa::SolveStatus::kInvalidProblem, func->span_)
      << "DSA-RP constructed or produced invalid state for '" << func->name_ << "'"
      << (result.diagnostics.empty() ? std::string() : ": " + result.diagnostics.front());
  const std::string obvious_overflow =
      result.status == dsa::SolveStatus::kNoFit
          ? ObviousDsaCapacityOverflow(solved_problem, reserve_resolution, func->name_)
          : std::string();
  CHECK_SPAN(result.status == dsa::SolveStatus::kFeasible, func->span_)
      << "DSA-RP could not find a placement for '" << func->name_ << "' within the on-chip memory capacities"
      << (obvious_overflow.empty() ? std::string() : ": " + obvious_overflow)
      << (result.diagnostics.empty() ? std::string() : ": " + result.diagnostics.front());
  INTERNAL_CHECK_SPAN(result.solution.has_value(), func->span_)
      << "DSA-RP reported a feasible result without a placement";

  // Revalidate at the compiler boundary even though the in-tree solver
  // validates its own incumbent. This keeps writeback independent of solver
  // bookkeeping and catches future solver implementations that violate the
  // shared solution contract.
  const std::vector<std::string> validation = dsa::ValidateSolution(solved_problem, *result.solution);
  INTERNAL_CHECK_SPAN(validation.empty(), func->span_)
      << "Independent validation rejected DSA-RP placement for '" << func->name_
      << "': " << validation.front();

  const size_t active_pipeline_pairs =
      dsa::CountOverlappingPairs(prepared.strict_problem, *result.solution, prepared.pipeline_pairs);
  if (active_pipeline_pairs != 0) {
    std::ostringstream message;
    message << "DSA-RP's bounded strict search did not retain all software-pipeline buffer "
               "separations for '"
            << func->name_ << "'; " << active_pipeline_pairs << " of " << prepared.pipeline_pairs.size()
            << " relaxed pair(s) reuse physical storage. Pipeline overlap may be reduced.";
    EmitDiagnostics({Diagnostic(DiagnosticSeverity::PerfHint, "AllocateMemoryAddr", 0, "PH-DSA-001",
                                message.str(), func->span_)},
                    "AllocateMemoryAddr");
  }

  return dsa_adapter::BuildMemRefReplacements(prepared, *result.solution, memrefs, policy);
}

/**
 * @brief Allocate real memory addresses for existing alloc operations
 *
 * Alloc statements already exist (created by InitMemRef with addr=-1).
 * This pass assigns real addresses and updates both variable MemRef references
 * and the alloc statement arguments in place.
 */
FunctionPtr TransformAllocateMemoryAddr(const FunctionPtr& func) {
  // Only InCore-variant functions use reserve_buffer / tile memory allocation.
  // Spmd, Group, Orchestration, and Opaque functions do not have on-chip tile buffers.
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

  // Obtain the allocation policy from the backend (or fall back to the default).
  auto policy = backend::BackendConfig::IsConfigured() ? backend::GetBackend()->CreateMemoryAllocatorPolicy()
                                                       : std::make_unique<DefaultMemoryAllocatorPolicy>();
  INTERNAL_CHECK_SPAN(policy, func->span_) << "Backend::CreateMemoryAllocatorPolicy() returned null";

  // Step 1: Resolve reserve_buffer bases before assigning tile addresses.
  auto reserve_resolution = ResolveReserveBufferBases(func, *policy);

  // Step 2: Collect all unique MemRef objects from TileType variables
  auto memrefs = memref_collectors::CollectMemRefsWithSpace(func->body_);

  const PassContext* context = PassContext::Current();
  const MemoryPlanner planner = context == nullptr ? kDefaultMemoryPlanner : context->GetMemoryPlanner();

  // Step 3: use the selected in-tree allocator. PTOAS never reaches this pass.
  std::vector<std::pair<const MemRef*, MemRefPtr>> memref_pairs;
  if (planner == MemoryPlanner::DsaRP) {
    memref_pairs = PlanWithDsaRP(func, *policy, reserve_resolution, memrefs);
  } else {
    // Declared allocations are the only ones that may take a dynamic address
    // (a runtime slot index).
    PinnedAllocCollector pinned_collector;
    pinned_collector.VisitStmt(func->body_);
    memref_pairs = AllocateMemoryAddresses(memrefs, reserve_resolution, *policy, pinned_collector.alloc_sizes,
                                           func->name_);
  }

  if (memref_pairs.empty() && reserve_resolution.resolved_bases.empty()) {
    return func;
  }

  // Step 4: Update all MemRef references, alloc statements, and reserve_buffer bases in the IR.
  MemRefUpdateMutator mutator(memref_pairs, std::move(reserve_resolution.resolved_bases));

  std::vector<VarPtr> new_params;
  for (const auto& param : func->params_) {
    auto new_param_expr = mutator.VisitExpr(param);
    auto new_param = std::dynamic_pointer_cast<const Var>(new_param_expr);
    INTERNAL_CHECK_SPAN(new_param, param->span_) << "Failed to cast mutated param to Var";
    new_params.push_back(new_param);
  }

  auto new_body = mutator.VisitStmt(func->body_);
  if (planner == MemoryPlanner::DsaRP) {
    // DSA-RP consumed the transient stage provenance while constructing its
    // strict pipeline separations. The legacy planner strips the same
    // attribute at the end of MemoryReuse.
    new_body = StripPipelineMembershipMutator().VisitStmt(new_body);
    // MaterializeSemanticAliases can make the producer's original allocation
    // unreachable. MemoryReuse normally removes those declarations, but
    // DSA-RP deliberately skips that pass.
    new_body = RemoveUnusedAllocStatements(new_body);
  }

  auto new_func = MutableCopy(func);
  new_func->params_ = new_params;
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

// Factory function
namespace pass {
Pass AllocateMemoryAddr() {
  return CreateFunctionPass(TransformAllocateMemoryAddr, "AllocateMemoryAddr", kAllocateMemoryAddrProperties);
}
}  // namespace pass

// ============================================================================
// AllocatedMemoryAddr property verifier
// ============================================================================

namespace {

/**
 * @brief Collects non-DDR MemRefs and checks address validity.
 *
 * Records diagnostics for MemRefs whose address is still -1 (unallocated).
 * Also tracks the high-water mark (addr + size) per memory space so the
 * caller can compare against platform buffer limits.
 */
class AllocatedMemoryAddrVerifier : public IRVisitor {
 public:
  explicit AllocatedMemoryAddrVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitVarLike_(const VarPtr& op) override {
    if (!op || !op->GetType()) return;
    auto tile_type = As<TileType>(op->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      auto memory_space = tile_type->GetMemorySpace();
      INTERNAL_CHECK_SPAN(memory_space.has_value(), op->span_)
          << "TileType with MemRef must have memory_space for address verification";
      CheckMemRefAddr(tile_type->memref_.value(), *memory_space, op->name_hint_, op->span_);
    }
  }

  [[nodiscard]] const std::unordered_map<MemorySpace, uint64_t>& GetHighWaterMarks() const {
    return high_water_;
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::set<const MemRef*> seen_;
  std::unordered_map<MemorySpace, uint64_t> high_water_;

  void CheckMemRefAddr(const MemRefPtr& memref, MemorySpace memory_space, const std::string& var_name,
                       const Span& span) {
    if (memory_space == MemorySpace::DDR) return;
    if (!seen_.insert(memref.get()).second) return;

    // An address may legitimately be an expression: a declared allocation's runtime
    // slot index becomes `base_addr + index * slot_size`, which codegen lowers into
    // the tile's address assignment. What the property requires is that an address
    // was *assigned* — a null offset, or a negative constant, means it was not.
    if (!memref->byte_offset_) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "AllocatedMemoryAddr", 0,
                                "MemRef for variable '" + var_name + "' in " +
                                    MemorySpaceToString(memory_space) + " has no address allocated",
                                span);
      return;
    }
    auto const_offset = std::dynamic_pointer_cast<const ConstInt>(memref->byte_offset_);
    if (const_offset && const_offset->value_ < 0) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "AllocatedMemoryAddr", 0,
                                "MemRef for variable '" + var_name + "' in " +
                                    MemorySpaceToString(memory_space) + " has a negative address (" +
                                    std::to_string(const_offset->value_) + ")",
                                span);
      return;
    }
    // High-water tracking needs a concrete address. A dynamic one is bounded by
    // the whole declared allocation, which its own root MemRef already accounts
    // for, so skipping it here cannot under-report the space's footprint.
    if (!const_offset) return;

    uint64_t end = static_cast<uint64_t>(const_offset->value_) + memref->size_;
    auto& hw = high_water_[memory_space];
    if (end > hw) hw = end;
  }
};

}  // namespace

class AllocatedMemoryAddrPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AllocatedMemoryAddr"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    const backend::Backend* be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;

    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;

      AllocatedMemoryAddrVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);

      if (!be) continue;

      // Resolved lazily and reused across spaces: only an overflow reads it, and that
      // path aborts the compile — so on every green build this costs nothing. Shares
      // ResolveReserveBufferBases with the transform half of this pass (see
      // reserve_buffer_utils.h, "the SINGLE source of truth"), so the attributed
      // bytes ARE the floor tiles were placed above, not an independent re-sum.
      std::optional<ReserveBufferResolution> reserved;

      for (const auto& [space, used] : verifier.GetHighWaterMarks()) {
        uint64_t limit = be->GetMemSize(space);
        if (limit == 0 || used <= limit) continue;
        std::string message = "Function '" + func->name_ + "': " + MemorySpaceToString(space) +
                              " buffer usage (" + std::to_string(used) + " bytes) exceeds platform limit (" +
                              std::to_string(limit) + " bytes)";
        if (!reserved.has_value()) {
          reserved.emplace();
          // Only InCore-variant functions carry reserve_buffer; the space resolution
          // inside is unreachable for the others (Group / Orchestration / Opaque).
          if (IsInCoreType(func->func_type_)) {
            auto policy = be->CreateMemoryAllocatorPolicy();
            INTERNAL_CHECK_SPAN(policy, func->span_)
                << "Internal error: Backend::CreateMemoryAllocatorPolicy() returned null";
            *reserved = ResolveReserveBufferBases(func, *policy);
          }
        }
        message += ReservedBytesNote(*reserved, space, func->name_);
        diagnostics.emplace_back(DiagnosticSeverity::Error, "AllocatedMemoryAddr", 1, message, func->span_);
      }
    }
  }
};

PropertyVerifierPtr CreateAllocatedMemoryAddrPropertyVerifier() {
  return std::make_shared<AllocatedMemoryAddrPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
