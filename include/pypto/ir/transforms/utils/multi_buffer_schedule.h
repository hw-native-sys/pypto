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
#ifndef PYPTO_IR_TRANSFORMS_UTILS_MULTI_BUFFER_SCHEDULE_H_
#define PYPTO_IR_TRANSFORMS_UTILS_MULTI_BUFFER_SCHEDULE_H_

#include <map>
#include <set>
#include <utility>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// Both Tile codegen and Tile-to-Buffer lowering must validate the original
// schedule before slot handles are hoisted or converted. O(N log N) for N IR
// nodes; each loop owns a separately indexed set of allocation/slot uses.
inline constexpr const char* kUnsafePtoasMultiBufferLoopReason =
    "two of its slots are live at once inside a loop, and ptoas mis-synchronizes one form of "
    "that — a slot filled an iteration ahead of its read — into wrong data or a device hang "
    "(hw-native-sys/PTOAS#1519: fixed in 0.63, broken again since 0.64). Take one slot per "
    "iteration, which is the shape the region form accelerates";

// Allocations with two or more slots selected inside one loop body.
//
// ptoas synchronizes such a body from the slot expressions, and has gotten two
// forms of it wrong:
//
// - Two slots filled and read in the same iteration. ptoas <= 0.55 guarded only
//   the FIRST `multi_tile_get` of an iteration, leaving the other load unguarded
//   against the next iteration's write — measured silently wrong on device with
//   ptoas 0.54 (hw-native-sys/PTOAS#1118). Fixed in 0.56, which guards the body
//   with one static event for the whole region: correct, but with none of the
//   per-slot overlap the region form exists for.
// - The prefetch: slot 0 filled before the loop, then each iteration fills slot
//   (i+1)%2 while reading slot i%2. ptoas <= 0.62 primes and drains both slots'
//   events as for a one-slot rotation, which is off by one here — wrong data for
//   an even trip count, a device hang for an odd one (hw-native-sys/PTOAS#1519).
//   0.63 fixed it; 0.64 and 0.65 prime both slots again and fail on device the
//   same two ways.
//
// Constant indices alone do not distinguish these forms: an unrolled prefetch
// can also use literals. Only compiler-proven fresh Acc dbC stages receive an
// exemption; LowerPipelineLoops emits those as compute/drain pairs with constant
// slots. Unrelated multi-slot loops retain the conservative rejection, including
// static-index prefetch. The author is pointed at the PyPTO planner, whose
// baked-address alloc_tile path runs the same-iteration form correctly.
// The ping-pong the region form exists for takes ONE slot per iteration and is
// guarded correctly. Straight-line code is untouched: with no loop there is no
// cross-iteration reuse to guard.
class PtoasMultiBufferLoopSafety : public ir::IRVisitor {
 public:
  std::set<const ir::Var*> bases;  ///< Allocations with >= 2 slots live in one loop body

  void VisitStmt_(const ir::ForStmtPtr& op) override { VisitLoop(op); }
  void VisitStmt_(const ir::WhileStmtPtr& op) override { VisitLoop(op); }

  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    if (loop_depth_ > 0) {
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        const auto memref = ir::GetDefinedMemRef(tile_type);
        if (memref->slot_count_ > 1 && memref->slot_index_.has_value() && *memref->slot_index_) {
          // Repeated uses of the SAME slot are one mutable tile handle, not two
          // co-live slots. This is how an accumulator is threaded through a
          // sequence of in-place MADs. Reject only when one loop body selects
          // structurally different slots of the same region.
          const auto call = As<ir::Call>(op->value_);
          const auto slot = As<ir::ConstInt>(*memref->slot_index_);
          const bool compiler_dbc = call && call->GetAttr<bool>(ir::kCompilerPtoasDbCStageAttr, false) &&
                                    tile_type->GetMemorySpace() == ir::MemorySpace::Acc &&
                                    memref->slot_count_ == 2 && slot &&
                                    (slot->value_ == 0 || slot->value_ == 1) &&
                                    (ir::IsOp(call, "tile.matmul") || ir::IsOp(call, "tile.matmul_bias"));
          auto [it, fresh] = per_loop_first_slot_.try_emplace(
              memref->base_.get(), SlotUse{*memref->slot_index_, compiler_dbc, false});
          auto& use = it->second;
          use.compiler_dbc_only = use.compiler_dbc_only && compiler_dbc;
          use.distinct_slots =
              use.distinct_slots || (!fresh && !ir::structural_equal(use.first_slot, *memref->slot_index_));
          if (use.distinct_slots && !use.compiler_dbc_only) {
            bases.insert(memref->base_.get());
          }
        }
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

 private:
  template <typename LoopPtr>
  void VisitLoop(const LoopPtr& op) {
    // Each loop body counts on its own: two slots in *sibling* loops are never
    // live together, and a nested loop's own body is the iteration that matters.
    auto saved = std::move(per_loop_first_slot_);
    per_loop_first_slot_.clear();
    ++loop_depth_;
    ir::IRVisitor::VisitStmt_(op);
    --loop_depth_;
    per_loop_first_slot_ = std::move(saved);
  }

  int loop_depth_ = 0;
  struct SlotUse {
    ir::ExprPtr first_slot;
    bool compiler_dbc_only;
    bool distinct_slots;
  };
  std::map<const ir::Var*, SlotUse> per_loop_first_slot_;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_MULTI_BUFFER_SCHEDULE_H_
