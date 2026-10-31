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
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/allocation_constraint_analysis.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_ir_selection.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/memory_reuse/internal.h"
#include "src/ir/transforms/memory_reuse/storage.h"

namespace pypto {
namespace ir {
namespace {
using memory_reuse::ForbidAliasMap;
using memory_reuse::HazardInputs;
using memory_reuse::IsA5Prelu;
using memory_reuse::IsA5Target;
using memory_reuse::TileMemRefBase;

// ---------------------------------------------------------------------------
// Ascend910 split-AIV  load + tpop_from_aic  in-place hazard
// ---------------------------------------------------------------------------
//
// On Ascend910 AIV functions with a non-None SplitMode, an op that consumes
// BOTH a tile.load result (or a legal-view descendant of one) AND a
// tile.tpop_from_aic value must not write its output into the same physical
// buffer as the load result it reads.  Letting the writer's output reuse that
// buffer (an in-place touch) yields silently wrong results on this hardware.
//
// MemoryReuse owns every buffer-coalescing decision, so the cleanest fix is to
// never create the hazardous sharing in the first place.  This used to be
// undone after the fact by a dedicated LegalizePTOBufferReuse split pass; the
// guard below folds that responsibility into the reuse decision.
//
// `HazardInputs` is collected by walking the function body (see
// `HazardInputCollector::Run` for why that walk runs twice):
//   - load_derived: vars produced by tile.load or by a legal view op chained
//     from a load (these alias the load's physical buffer).
//   - reads_tpop:   vars whose defining op consumes a tile.tpop_from_aic value.
// Both sets are keyed on Var identity and checked against the reuse-decision
// representatives (a sharing group's earliest-defined member, which for a
// writer is the writer's own output and for a load is the load itself).
//
// Reading an *operand* takes one extra step, because a value can reach the op
// through a loop carry.  An IterArg is never an AssignStmt def, so it is never a
// member of either set; but the carry chain it belongs to (init value, IterArg,
// yield value) has already been fused onto one MemRef base by
// MaterializeSemanticAliases, so an IterArg operand is classified by that base
// instead — see `HazardInputCollector::IsTainted`.  A carry also breaks program
// order, since the producer that taints the base may stand after the use it
// taints; `HazardInputCollector::Run` covers that with a second traversal.

/// Ops through which a `tile.load` result's taint reaches the writer below.
///
/// This is deliberately NOT one of the registry alias predicates, and the
/// distinction is the whole point.  Those answer a *semantic* question — does
/// this operator declare that its result names an argument's buffer.  The taint
/// closure needs a *physical* one — can this result end up occupying the load's
/// buffer — and after opportunistic reuse the two differ.  `tile.fillpad`
/// declares no aliasing at all, yet it is in-place safe, so `RetargetAssign`
/// may coalesce its output straight onto its input's buffer; that is asserted
/// by `test_fillpad_output_can_reuse_input`.  Deriving this set from the
/// registry drops `tile.fillpad` and reopens
/// `load -> fillpad -> writer(result, tpop_from_aic)`.
///
/// The list is therefore kept as-is, and it is an approximation with two known
/// gaps, both pre-existing:
///   * it omits `tile.assemble` / `tile.set_validshape` / `tile.tget_scale_addr`,
///     which do declare aliasing, so a load reaching the writer through one of
///     them is not tainted; and
///   * it omits every other in-place-safe op (`tile.add`, ...) whose output
///     `RetargetAssign` may likewise coalesce onto a load buffer.
/// Closing either needs the taint to follow post-reuse physical bases rather
/// than op names, which is a change to the reuse decision itself.
static bool IsLegalTileViewOp(const OpPtr& op) {
  return IsOp(op, "tile.reshape") || IsOp(op, "tile.extract") || IsOp(op, "tile.slice") ||
         IsOp(op, "tile.fillpad") || IsOp(op, "tile.fillpad_inplace") || IsOp(op, "tile.transpose_view") ||
         IsOp(op, "tile.reinterpret_view") || IsOp(op, "tensor.slice");
}

class HazardInputCollector : public IRVisitor {
 public:
  /// Classify `body` in TWO traversals, then hand the result to `Take()`.
  ///
  /// One forward walk is not enough, because a value can reach a use through a
  /// loop back edge — from a producer that stands *after* that use in program
  /// order:
  ///
  /// ```
  /// for (carry,) in range(..., init=[non_tpop]):
  ///   w = tile.add(l, carry)        // from iteration 1 on, `carry` is a tpop value
  ///   p = tile.tpop_from_aic()      // ... but the producer is only seen here
  ///   yield p                       // p must-aliases carry's base
  /// ```
  ///
  /// A single walk classifies `w` before `p` taints the carry's base, so `w`
  /// escapes `reads_tpop` and its output may be coalesced onto `l`'s buffer.
  ///
  /// Two traversals suffice, and a third could never add anything: the base
  /// sets are complete after the first pass. Direct producer defs are
  /// order-independent; propagated views and aliases share their source's base,
  /// so marking one inserts a base that is already present (or no base for a
  /// buffer-less value). Only the *Var* sets
  /// grow in the second pass, and those propagate in program order within it.
  /// Both traversals are O(N), so the collector stays linear.
  void Run(const StmtPtr& body) {
    VisitStmt(body);
    VisitStmt(body);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // Producer provenance is independent of general-pool ownership. InitMemRef
    // deliberately leaves TPOP results and their zero-copy views without a
    // MemRef: their storage belongs to the cross-core pipe instead.
    if (As<TileType>(op->var_->GetType())) {
      if (auto call = As<Call>(op->value_)) {
        // AsVarLike, not As<Var>: an operand may be an enclosing loop's IterArg
        // — the loop-carry shape of `tile.add(x, c_iter)` / `tile.matmul_acc(
        // c_iter, ...)`.  IterArg has its own ObjectKind, so As<Var> returns
        // null there and would drop the operand, and with it the load / tpop
        // taint the carry brings into the body.
        std::vector<VarPtr> input_vars;
        for (const auto& arg : call->args_) {
          if (auto v = AsVarLike(arg)) input_vars.push_back(v);
        }
        // load_derived closure: within one traversal defs precede uses, so a
        // view's source is already classified by the time we reach the view;
        // a source that only becomes tainted across a loop back edge is picked
        // up by the second traversal (see `Run`).
        if (IsOp(call, "tile.load")) {
          MarkLoadDerived(op->var_);
        } else if (IsLegalTileViewOp(call->op_)) {
          for (const VarPtr& in : input_vars) {
            if (IsLoadDerived(in)) {
              MarkLoadDerived(op->var_);
              break;
            }
          }
        }
        if (IsOp(call, "tile.tpop_from_aic")) {
          MarkTpop(op->var_);
        } else if (call->op_ && op_predicates::IsBufferAliasingViewOp(call->op_->name_) &&
                   !call->args_.empty()) {
          if (const auto source = AsVarLike(call->args_[0]); source && IsTpop(source)) {
            MarkTpop(op->var_);
          }
        }
        for (const VarPtr& in : input_vars) {
          if (IsTpop(in)) {
            inputs_.reads_tpop.insert(op->var_.get());
            break;
          }
        }
      } else if (const auto source = AsVarLike(op->value_)) {
        // A bare SSA rename preserves the same physical value, including when
        // neither side owns a MemRef. Do not confuse this with a data-producing
        // operation which may independently choose a different allocation.
        if (IsLoadDerived(source)) MarkLoadDerived(op->var_);
        if (IsTpop(source)) MarkTpop(op->var_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  HazardInputs Take() { return std::move(inputs_); }

 private:
  // Taint a var and the physical buffer it occupies.  The buffer mirror exists
  // for loop carries: an IterArg is never itself an AssignStmt def, so it can
  // never be a member of the Var-keyed sets no matter how it is spelled at the
  // use site.  MaterializeSemanticAliases (which runs immediately before this
  // pass) has already fused each carry — init value, IterArg, yield value —
  // onto ONE MemRef base, so a carry's taint is exactly the taint of the base it
  // must-aliases.
  void MarkLoadDerived(const VarPtr& v) {
    inputs_.load_derived.insert(v.get());
    if (const Var* base = TileMemRefBase(v)) load_derived_bases_.insert(base);
  }

  void MarkTpop(const VarPtr& v) {
    tpop_vars_.insert(v.get());
    if (const Var* base = TileMemRefBase(v)) tpop_bases_.insert(base);
  }

  [[nodiscard]] bool IsLoadDerived(const VarPtr& v) const {
    return IsTainted(inputs_.load_derived, load_derived_bases_, v);
  }

  [[nodiscard]] bool IsTpop(const VarPtr& v) const { return IsTainted(tpop_vars_, tpop_bases_, v); }

  // Var identity stays the exact test for an ordinary operand — two tiles that
  // merely share a base (a view and its source) are still classified by the view
  // allowlist above, not by the buffer.  Only a carry, which identity provably
  // cannot answer for, falls back to its must-aliased base.
  [[nodiscard]] static bool IsTainted(const std::unordered_set<const Var*>& vars,
                                      const std::unordered_set<const Var*>& bases, const VarPtr& v) {
    if (vars.count(v.get()) != 0) return true;
    if (!As<IterArg>(v)) return false;
    const Var* base = TileMemRefBase(v);
    return base != nullptr && bases.count(base) != 0;
  }

  HazardInputs inputs_;
  std::unordered_set<const Var*> tpop_vars_;
  std::unordered_set<const Var*> load_derived_bases_;  ///< MemRef bases holding a load_derived value
  std::unordered_set<const Var*> tpop_bases_;          ///< MemRef bases holding a tpop_from_aic value
};

// ---------------------------------------------------------------------------
// "Output must not alias this input" map
// ---------------------------------------------------------------------------
//
// An op may forbid its output from sharing a buffer with one or more of its
// input operands, because the hardware reads those inputs while writing the
// output (aliasing them corrupts results).  Two registry declarations feed the
// same per-output forbidden-input set, unified here:
//
//   * not_inplace_safe()        -> the op cannot run src == dst at all, so the
//                                  output must not alias ANY of its inputs
//                                  (e.g. tile.recip, tile.mrgsort_format1).
//   * forbid_output_alias(i)    -> the op is in-place-safe w.r.t. its value
//                                  operands but reads a *specific* operand
//                                  while writing dst (e.g. tile.sel's predicate
//                                  mask / tmp scratch), so dst must not alias
//                                  that one operand's buffer.
//
// Because every read input of `op->var_`'s defining call appears in `args_`,
// "forbid all inputs" is exactly "forbid every args_ entry" — so this single
// walk records, per output tile var, the input tile vars its op forbids the
// output from aliasing.  MemoryReuse then refuses to place the output on any of
// those inputs' buffers.  Distinct from the load+tpop hazard (backend-gated) —
// this is op-semantic and always collected.
// Maps each output tile Var to the input operand Vars its defining op forbids
// the output from sharing a buffer with.  Enforcement resolves each operand to
// the *physical buffer* it ends up on (following both reuse-map reassignment and
// VIEW inheritance) and blocks the output from landing there — see the use site.
class ForbidAliasCollector : public IRVisitor {
 public:
  // `sharing_groups` (from AnalyzeAllocationLifetimes) maps every var that shares a MemRef
  // to its group. IdentifyReuseOpportunities keys lifetimes on each group's
  // *representative* (`sharing_group[0]`), so the forbidden set must be keyed on
  // the representative of the defining op's output too — otherwise a forbidden
  // op whose output is a non-representative group member (e.g. its result is
  // also viewed/reshaped elsewhere) would never be looked up. Forbidden input
  // *values* need no such mapping: enforcement compares physical MemRef bases,
  // which every group member already shares.
  ForbidAliasCollector(const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
                       const BufferViewAnalysis* views)
      : views_(views) {
    for (const auto& [var, group] : sharing_groups) {
      if (!group.empty()) member_to_rep_[var.get()] = group[0].get();
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // Multi-result calls are assigned to a tuple temporary and unpacked by
    // subsequent TupleGetItem assignments. Carry the call's semantic
    // no-alias inputs onto every physical tile result; the tuple temporary
    // itself has no allocation interval to constrain.
    if (const auto get_item = As<TupleGetItemExpr>(op->value_)) {
      if (const VarPtr tuple = AsVarLike(get_item->tuple_)) {
        const auto pending = tuple_forbidden_.find(tuple.get());
        if (pending != tuple_forbidden_.end()) RecordForOutput(op->var_, pending->second);
      }
    }

    if (auto call = As<Call>(op->value_); call && call->op_) {
      const auto& reg = OpRegistry::GetInstance();
      if (reg.IsRegistered(call->op_->name_)) {
        const auto& entry = reg.GetEntry(call->op_->name_);
        std::vector<VarPtr> forbidden_inputs;
        auto forbid_arg = [&](size_t i) {
          if (i < call->args_.size()) {
            if (auto v = AsVarLike(call->args_[i])) forbidden_inputs.push_back(v);
          }
        };
        if (!entry.IsInplaceSafe()) {
          // Non-in-place ops forbid output aliasing active inputs. A5 TPRELU
          // retains tmp (arg 2) in the ABI but does not read it, so only src
          // and slope remain active.
          const size_t forbidden_arg_count = IsA5Prelu(call) ? 2 : call->args_.size();
          for (size_t i = 0; i < forbidden_arg_count; ++i) forbid_arg(i);
        } else {
          for (size_t i : entry.ForbidOutputAliasArgs()) forbid_arg(i);
        }
        // Narrowing is also unsafe on A2/A3: TCVT writes every row's head
        // before reading the tails, clobbering earlier rows' unread source.
        if (op_predicates::CastRequiresDisjointSource(call, !IsA5Target())) {
          forbid_arg(0);
        }
        // In-place safety permits exact aliases, not a packed output placed
        // inside a strided input window. Constrain the input's whole storage
        // family before either planner chooses addresses.
        if (views_ && !op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) {
          for (size_t i = 0; i < call->args_.size(); ++i) {
            if (views_->IsWindow(call->args_[i])) forbid_arg(i);
          }
        }
        if (views_ && IsOp(call, "tile.assemble")) {
          const auto target = GetTypeMemRef(call->args_[0]->GetType()).value_or(nullptr);
          const auto source = GetTypeMemRef(call->args_[1]->GetType()).value_or(nullptr);
          if (target && source && target->base_ != source->base_) forbid_arg(1);
        }
        RecordForOutput(op->var_, forbidden_inputs);
        // tile.transpose is registered not_inplace_safe(), so its output is
        // already forbidden from aliasing any input above (pto.ttrans writes
        // dst directly from src on the scalar path — dst == src corrupts).
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  ForbidAliasMap Take() { return std::move(forbidden_); }

 private:
  void RecordForOutput(const VarPtr& output, const std::vector<VarPtr>& forbidden_inputs) {
    if (!output || forbidden_inputs.empty()) return;
    if (As<TupleType>(output->GetType())) {
      tuple_forbidden_[output.get()] = forbidden_inputs;
      return;
    }
    if (!As<TileType>(output->GetType())) return;
    const auto rep_it = member_to_rep_.find(output.get());
    const Var* out_key = rep_it != member_to_rep_.end() ? rep_it->second : output.get();
    auto& recorded = forbidden_[out_key];
    recorded.insert(recorded.end(), forbidden_inputs.begin(), forbidden_inputs.end());
  }

  ForbidAliasMap forbidden_;
  const BufferViewAnalysis* views_;
  std::map<const Var*, std::vector<VarPtr>> tuple_forbidden_;
  std::map<const Var*, const Var*> member_to_rep_;  ///< sharing-group member -> representative
};

/// Collect every tile operand an operator declares as a WRITTEN WORKSPACE.
///
/// `set_workspace_arg(i)` means "compiler-supplied scratch: the hardware writes
/// it, but it carries no result the caller reads", and `set_arg_effect(i, ...)`
/// says whether that write happens. Such a buffer is written by the intrinsic
/// itself, on whatever pipe its implementation chooses -- A2/A3's TSELS stages
/// its scalar into `tmp` from the SCALAR pipe while every tile operand is read
/// on the vector pipe. Program order does not order a write against a read
/// still in flight on another pipe, and the write never reaches the generated
/// `.pto` as a write (operands are emitted positionally), so no downstream sync
/// insertion can see the WAR. The buffer therefore must not be inherited from a
/// value that dies before the workspace is defined.
///
/// Keyed on the registry rather than on operator names or a backend flag: any
/// operator declaring a written workspace is exposed to the same hazard, and
/// PyPTO has no way to know which pipe a given intrinsic writes from.
class WrittenWorkspaceCollector : public IRVisitor {
 public:
  void VisitExpr_(const CallPtr& op) override {
    Record(op->op_, op->args_);
    IRVisitor::VisitExpr_(op);
  }

  // The registry keys both call-like kinds the same way; classifying only Call
  // would silently skip an operator launched through pl.submit.
  void VisitExpr_(const SubmitPtr& op) override {
    Record(op->op_, op->args_);
    IRVisitor::VisitExpr_(op);
  }

  // Only these two repeat a body once MemoryReuse runs: `pl.pipeline` is
  // lowered to a loop by LowerPipelineToSlots / LowerPipelineLoops (passes
  // 30-31), well before this pass.
  void VisitStmt_(const ForStmtPtr& op) override {
    const bool software_pipeline = op->HasAttr(kSoftwarePipelineSlotsAttr);
    if (software_pipeline) ++software_pipeline_depth_;
    ++loop_depth_;
    IRVisitor::VisitStmt_(op);
    --loop_depth_;
    if (software_pipeline) --software_pipeline_depth_;
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    ++loop_depth_;
    IRVisitor::VisitStmt_(op);
    --loop_depth_;
  }

  std::unordered_set<const Var*> TakeVars() { return std::move(workspaces_); }
  std::unordered_set<const Var*> TakeBases() { return std::move(bases_); }
  std::unordered_set<const Var*> TakeLoopingVars() { return std::move(looping_workspaces_); }
  std::unordered_set<const Var*> TakeLoopingBases() { return std::move(looping_bases_); }

 private:
  void Record(const OpPtr& op, const std::vector<ExprPtr>& args) {
    // LookupOpEntry, not a name lookup: a Submit's callee is a GlobalVar naming
    // a *function*, and resolving that against the operator registry would hand
    // a user function whose name collides with an operator that operator's
    // workspace contract.
    const auto* entry = LookupOpEntry(op);
    if (entry == nullptr) return;
    for (size_t index = 0; index < args.size(); ++index) {
      // Legacy reductions expose hidden writes through the scratch-role
      // contract. Recognize them only in an actually transformed pipeline;
      // ordinary loops retain their existing allocation policy.
      const bool pipeline_scratch =
          software_pipeline_depth_ > 0 && entry->GetLaneInvariantArgKind(index) == LaneInvariantArg::Scratch;
      const bool explicit_workspace = entry->IsWorkspaceArg(index) && entry->MayWriteArg(index);
      if (!explicit_workspace && !pipeline_scratch) continue;
      // AsVarLike, not As<Var>: a workspace can reach the op as a loop IterArg
      // when the call sits inside a loop body.
      auto workspace = AsVarLike(args[index]);
      if (!workspace) continue;
      workspaces_.insert(workspace.get());
      // Record the allocation base as well -- see `written_workspace_bases`.
      const Var* base = TileMemRefBase(workspace);
      if (base != nullptr) bases_.insert(base);
      if (loop_depth_ > 0) {
        looping_workspaces_.insert(workspace.get());
        if (base != nullptr) looping_bases_.insert(base);
      }
    }
  }

  int loop_depth_ = 0;
  int software_pipeline_depth_ = 0;
  std::unordered_set<const Var*> workspaces_;
  std::unordered_set<const Var*> bases_;
  std::unordered_set<const Var*> looping_workspaces_;
  std::unordered_set<const Var*> looping_bases_;
};

/// True only for Ascend910 AIV split-mode functions, which need the load +
/// tpop_from_aic in-place hazard guard.  All other backends / function kinds
/// reuse buffers freely.  Defensive against unit-test contexts that run
/// MemoryReuse without a configured backend.
bool NeedsLoadTpopHazardGuard(const FunctionPtr& func) {
  if (func->func_type_ != FunctionType::AIV) return false;
  if (!backend::BackendConfig::IsConfigured()) return false;
  const auto* ctx = PassContext::Current();
  // AllocationConstraintAnalysis is also used directly outside a pass scope.
  // PassContext's accessor delegates to this same configured backend; absence
  // of a context must not silently turn off a target correctness constraint.
  const auto* handler = ctx ? ctx->GetBackendHandler() : backend::GetBackend()->GetHandler();
  if (!handler->RequiresSplitLoadTpopWorkaround()) return false;
  // A split AIV function needs the guard. The split may be signalled either as a
  // function-level mode (single-mode / standalone kernels carry ``split``) or
  // ONLY as the mode-agnostic ``split_aiv`` marker — multi-mode explicit
  // pl.split_aiv regions have their per-region modes lowered + erased by
  // LowerAutoVectorSplit, so no single function-level mode survives. Key on the
  // marker too so the guard still fires for them.
  if (func->HasAttr(kAttrSplitAiv) && func->GetAttr<bool>(kAttrSplitAiv, false)) return true;
  const auto split_mode = func->GetSplitMode();
  return split_mode.has_value() && *split_mode != SplitMode::None;
}

/// One tile independently bound to a declared allocation, with its lifetime.
struct BoundMember {
  VarPtr var;
  int def_point;
  int last_use_point;
};

/// Group the tiles a user *independently bound* to each pinned buffer.
///
/// A tile whose defining op inherits its source's buffer (a view, an in-place
/// op, or a bare SSA alias) lands on the same base without the author binding it
/// there — it is the same data, so overlapping with its source is expected and
/// legal. Only independently produced tiles compete for the buffer's bytes, and
/// only those are checked for lifetime overlap.
class IndependentlyBoundCollector : public IRVisitor {
 public:
  IndependentlyBoundCollector(const std::set<const Var*>& pinned_bases,
                              const std::map<const Var*, std::pair<int, int>>& var_liveness)
      : pinned_bases_(pinned_bases), var_liveness_(var_liveness) {}

  /// One slot of a declared allocation: its base, and the byte offset that tells
  /// slots of the same allocation apart. `InitMemRef` has already consumed the
  /// slot index into that offset, so a null offset (a dynamic index) collapses to
  /// "some slot" — see the note on `by_slot`.
  struct SlotKey {
    const Var* base;
    int64_t byte_offset;

    bool operator<(const SlotKey& other) const {
      if (base != other.base) return base < other.base;
      return byte_offset < other.byte_offset;
    }
  };

  /// Slot -> the tiles bound to it, with their lifetimes.
  ///
  /// Keyed per *slot*, not per allocation: two tiles on different slots of one
  /// declared allocation are supposed to be live together — that is what a
  /// ping-pong is — and only tiles landing on the *same* slot can corrupt each
  /// other. Tiles whose offset is a runtime expression are skipped entirely,
  /// since which slot they occupy is not a static fact.
  std::map<SlotKey, std::vector<BoundMember>> by_slot;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) Record(op);
    IRVisitor::VisitStmt_(op);
  }

 private:
  void Record(const AssignStmtPtr& op) {
    auto memref = GetTypeMemRef(op->var_->GetType());
    if (!memref.has_value() || !(*memref)->base_) return;
    const Var* base = (*memref)->base_.get();
    if (pinned_bases_.count(base) == 0) return;
    if (AsVarLike(op->value_)) return;  // bare SSA alias — same data
    if (auto call = As<Call>(op->value_);
        call && call->op_ && op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) {
      return;
    }
    auto liveness = var_liveness_.find(op->var_.get());
    if (liveness == var_liveness_.end()) return;
    // A runtime slot index leaves no static slot to attribute the tile to, so
    // there is nothing to compare it against. The author owns the rotation in
    // that case; isolation from other allocations still holds.
    auto const_offset = As<ConstInt>((*memref)->byte_offset_);
    if (!const_offset) return;
    by_slot[{base, const_offset->value_}].push_back(
        {op->var_, liveness->second.first, liveness->second.second});
  }

  const std::set<const Var*>& pinned_bases_;
  const std::map<const Var*, std::pair<int, int>>& var_liveness_;
};

/// Reject a declared allocation whose independently bound tiles are co-live.
///
/// Binding two co-live tiles to one buffer is not a reuse decision, it is data
/// corruption: the later write lands on bytes the earlier tile still needs. The
/// packer refuses such a pairing among its own candidates; a hand-written
/// binding bypasses that path, so it is re-checked here. Touching is allowed —
/// one tile's last read may be the statement that produces the next.
///
/// Sorted sweep rather than a pairwise scan: a *valid* program is exactly one
/// where every pair is disjoint, so a pairwise check never exits early and would
/// be quadratic in the member count — which grows with the IR once a binding
/// inside a loop body is unrolled.
void ValidateDeclaredAllocs(const StmtPtr& body, const std::set<const Var*>& pinned_bases,
                            const std::map<const Var*, std::pair<int, int>>& var_liveness) {
  if (pinned_bases.empty()) return;
  IndependentlyBoundCollector collector(pinned_bases, var_liveness);
  collector.VisitStmt(body);

  for (auto& [slot, members] : collector.by_slot) {
    const Var* base = slot.base;
    std::sort(members.begin(), members.end(),
              [](const BoundMember& a, const BoundMember& b) { return a.def_point < b.def_point; });
    for (size_t i = 1; i < members.size(); ++i) {
      const BoundMember& prev = members[i - 1];
      const BoundMember& cur = members[i];
      CHECK_SPAN(prev.last_use_point <= cur.def_point, cur.var->span_)
          << "Tiles '" << prev.var->name_hint_ << "' and '" << cur.var->name_hint_
          << "' are both bound to the same slot of the declared allocation '" << base->name_hint_
          << "' but are live at the same time (" << prev.var->name_hint_ << ": [" << prev.def_point << ", "
          << prev.last_use_point << "], " << cur.var->name_hint_ << ": [" << cur.def_point << ", "
          << cur.last_use_point
          << "]). Sharing one slot would overwrite data that is still needed — put them on different "
             "slots, declare separate allocations, or bind only tiles whose lifetimes do not overlap.";
    }
  }
}

/// Collect the full byte extent of allocations declared by the author.
///
/// InitMemRef hoists every alloc to the function body's top level. When an
/// allocation exists, prepending it forms a SeqStmts; a function with no
/// allocation may legitimately retain a singleton body such as ReturnStmt.
/// A multi-slot declaration is larger than any one member MemRef, so DSA-RP
/// must take the extent from the alloc statement rather than reconstruct it
/// from members.
std::map<const Var*, uint64_t> CollectPinnedAllocSizes(const StmtPtr& body, const Span& span,
                                                       const std::string& consumer) {
  INTERNAL_CHECK_SPAN(body, span) << consumer << " expected a non-null function body";

  std::map<const Var*, uint64_t> pinned_allocations;
  auto collect = [&](const StmtPtr& stmt) {
    auto base = GetPinnedAllocBase(stmt);
    if (!base) return;
    auto assign = As<AssignStmt>(stmt);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    auto size = call && call->args_.size() >= 2 ? As<ConstInt>(call->args_[1]) : nullptr;
    INTERNAL_CHECK_SPAN(size && size->value_ > 0, stmt->span_)
        << consumer << " expected declared allocation '" << base->name_hint_
        << "' to have a positive constant byte extent after InitMemRef";
    pinned_allocations.emplace(base.get(), static_cast<uint64_t>(size->value_));
  };

  if (auto top_level = As<SeqStmts>(body)) {
    for (const auto& stmt : top_level->stmts_) collect(stmt);
  } else {
    collect(body);
  }
  return pinned_allocations;
}

}  // namespace

AllocationConstraintAnalysis AnalyzeAllocationConstraints(const FunctionPtr& func,
                                                          const LifetimeAnalysisResult& lifetimes,
                                                          const char* consumer) {
  AllocationConstraintAnalysis result;
  result.declared_allocation_sizes = CollectPinnedAllocSizes(func->body_, func->span_, consumer);
  for (const auto& [base, size] : result.declared_allocation_sizes) {
    static_cast<void>(size);
    result.declared_allocation_bases.insert(base);
  }
  ValidateDeclaredAllocs(func->body_, result.declared_allocation_bases, lifetimes.var_liveness);

  result.needs_load_tpop_hazard_guard = NeedsLoadTpopHazardGuard(func);
  if (result.needs_load_tpop_hazard_guard) {
    HazardInputCollector collector;
    collector.Run(func->body_);
    result.target_hazard_inputs = collector.Take();
  }

  WrittenWorkspaceCollector workspace_collector;
  workspace_collector.VisitStmt(func->body_);
  result.target_hazard_inputs.written_workspaces = workspace_collector.TakeVars();
  result.target_hazard_inputs.written_workspace_bases = workspace_collector.TakeBases();
  result.target_hazard_inputs.looping_written_workspaces = workspace_collector.TakeLoopingVars();
  result.target_hazard_inputs.looping_written_workspace_bases = workspace_collector.TakeLoopingBases();

  const auto* context = PassContext::Current();
  std::optional<BufferViewAnalysis> views;
  if (context && context->GetEnableBufferIR() && IsBufferIRSelected(func)) views.emplace(func->body_);
  ForbidAliasCollector forbid_collector(lifetimes.var_sharing_groups, views ? &*views : nullptr);
  forbid_collector.VisitStmt(func->body_);
  result.forbid_alias = forbid_collector.Take();
  return result;
}

}  // namespace ir
}  // namespace pypto
