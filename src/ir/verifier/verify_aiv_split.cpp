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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Function attr recording that ``OutlineIncoreScopes`` minted this InCore
/// function by outlining a CORE_GROUP scope that held ``pl.split_aiv`` regions.
/// Written in ``scope_outline_utils.cpp`` and re-stamped by ``LowerAutoVectorSplit``;
/// read here as the provenance signal for check (h). Kept as a literal for the
/// same reason both producers do — it is a plain attr key, not an operator name.
constexpr const char* kSplitAivAttr = "split_aiv";

/// Everything that differs between the two split-axis boundary ops: the memory
/// space each side must carry, plus the direction-specific diagnostic wording.
/// Keeping it in one table row per op means a reader (and an editor) sees the
/// whole contract for an op in one place, and the check itself stays uniform.
///
/// `result` mirrors the op's set_output_memory declaration (cross_core.cpp),
/// restated here as the checkable form of the same contract. `operand` is
/// deliberately NOT declared as a set_input_memory constraint over there — a
/// violated input constraint makes InferTileMemorySpace insert a physically
/// impossible move instead of reporting the authoring error — so this verifier
/// is the only place it is stated at all.
struct BoundaryMemoryContract {
  MemorySpace operand;       ///< space on the PRODUCING lane (the op's input)
  MemorySpace result;        ///< space on the CONSUMING lane (the op's output)
  const char* producer;      ///< lane that must have produced the operand
  const char* delivery;      ///< what the result hands to the consuming lane
  const char* operand_hint;  ///< authoring fix appended to the operand diagnostic
};

/// The contract for a tile-level boundary op, or nullopt for anything else.
/// The tensor.* forms are deliberately excluded: a TensorType has no memory
/// space, so there is nothing to check until ConvertTensorToTileOps lowers them.
std::optional<BoundaryMemoryContract> GetBoundaryMemoryContract(const CallPtr& op) {
  if (IsOp(op, "tile.aiv_shard")) {
    // The hint stays space-neutral: the check rejects any non-Acc operand, and
    // only Vec means "vector-produced". A Mat operand is a different mistake
    // (L1 is not a supported cross-core producer pipe), so naming pl.load /
    // pl.full unconditionally would misdescribe it.
    return BoundaryMemoryContract{
        MemorySpace::Acc, MemorySpace::Vec, "cube", "half to the vector",
        " Only a cube-produced value reaches the vector lane through this boundary: Acc is the "
        "matmul result the c2v pipe can push. A Vec operand is vector-produced (pl.load / "
        "pl.full) and already lives on the AIV lane — drop the pl.aiv_shard and let the implicit "
        "affinity-gated split halve it. A Mat operand is not a supported producer pipe; move it "
        "through a pl.matmul, or load it on the vector lane instead."};
  }
  if (IsOp(op, "tile.aic_gather")) {
    return BoundaryMemoryContract{
        MemorySpace::Vec, MemorySpace::Mat, "vector", "full tile to the cube",
        " Gather the value only after it has been computed by vector ops on the AIV lane."};
  }
  return std::nullopt;
}

/// Memory space of `expr` when it is a tile whose space is already resolved.
/// nullopt means "not a tile" or "space not yet inferred" — both are skip
/// conditions, since the verifier also runs before InferTileMemorySpace.
std::optional<MemorySpace> ResolvedTileMemory(const ExprPtr& expr) {
  if (!expr) return std::nullopt;
  auto tile_type = As<TileType>(expr->GetType());
  if (!tile_type) return std::nullopt;
  return tile_type->memory_space_;
}

/// True for the two ops ConvertTensorToTileOps materializes OUTSIDE a region
/// even when the compute they feed was written inside one — see the carve-out
/// note on check (e) and on the crossing checks (f)/(g). Not an authoring escape
/// hatch: without it, both would fire on the compiler's own output.
bool IsHoistedMemoryOp(const CallPtr& op) { return IsOp(op, "tile.load") || IsOp(op, "tile.store"); }

/// True for the AIV-split boundary ops in either vocabulary. The tile.* forms
/// are what survives ConvertTensorToTileOps; the tensor.* forms are the
/// author-facing `pl.aiv_shard(tensor)` / `pl.aic_gather(tensor)` still live in
/// the OutlineIncoreScopes .. ConvertTensorToTileOps window.
bool IsBoundaryOp(const CallPtr& op) {
  return IsOp(op, "tile.aiv_shard") || IsOp(op, "tile.aic_gather") || IsOp(op, "tensor.aiv_shard") ||
         IsOp(op, "tensor.aic_gather");
}

bool IsGatherOp(const CallPtr& op) { return IsOp(op, "tile.aic_gather") || IsOp(op, "tensor.aic_gather"); }

/// The lane a call reads its OPERANDS on — what the crossing checks (f)/(g) need
/// from a consumer.
///
/// For most ops that is simply their affinity. A cross-C/V `tile.move` is the
/// exception: it classifies MIXED (it *is* a transfer, running as a tpush on one
/// side and a tpop on the other), yet it consumes its source on exactly one
/// lane — the one it moves AWAY from is where the operand is read, so the lane
/// that matters for "did this value cross the region edge" is the destination
/// side. InferTileMemorySpace inserts exactly such a move for an implicit
/// crossing, so without this the checks would go blind at the verification point
/// that runs after it.
core_affinity::CoreAffinity ConsumerLane(const CallPtr& op) {
  switch (core_affinity::ClassifyMoveDirection(op)) {
    case core_affinity::CVDirection::CUBE_TO_VECTOR:
      return core_affinity::CoreAffinity::VECTOR;
    case core_affinity::CVDirection::VECTOR_TO_CUBE:
      return core_affinity::CoreAffinity::CUBE;
    default:
      return core_affinity::ClassifyCallAffinity(op);
  }
}

/// Where a Var was bound, for the region-edge crossing checks.
struct ValueDef {
  CallPtr call;    ///< the defining call (null when the binding is not a Call)
  bool in_region;  ///< bound at region depth > 0
};

/// Function-level facts the per-node checks need but cannot read off a single
/// node.
struct FunctionSplitFacts {
  /// The function holds at least one SplitAivScopeStmt, i.e. it opted into
  /// MANUAL MODE: the region is authoritative for vector placement, so vector
  /// compute outside every region has no defined home (check (e)), and every
  /// tile value crossing a region edge must say so (checks (f)/(g)).
  bool has_region = false;
  /// Var -> where it was bound. Only AssignStmt-bound Calls are recorded: a
  /// param, an IterArg or a loop return var has no defining call, and a check
  /// that cannot see how a value was produced must stay silent about it.
  std::unordered_map<const Var*, ValueDef> defs;
};

/// Single pre-pass over one function body collecting `FunctionSplitFacts`.
/// Kept separate from the checking walk because both facts are properties of the
/// WHOLE body: a region appearing AFTER the offending statement still decides
/// check (e)'s gate, and check (f) reads the definition of a value the use may
/// precede in a loop body — neither can be settled in one pass. Two O(N) walks
/// with O(1) hash lookups keep the verifier at O(N).
class FunctionSplitFactScanner : public IRVisitor {
 public:
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override {
    facts_.has_region = true;
    ++depth_;
    IRVisitor::VisitStmt_(op);
    --depth_;
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_) {
      facts_.defs[op->var_.get()] = ValueDef{std::dynamic_pointer_cast<const Call>(op->value_), depth_ > 0};
    }
    IRVisitor::VisitStmt_(op);
  }

  [[nodiscard]] const FunctionSplitFacts& facts() const { return facts_; }

 private:
  FunctionSplitFacts facts_;
  int depth_ = 0;
};

// Structural verifier for the first-class SplitAivScopeStmt region (live between
// OutlineIncoreScopes and LowerAutoVectorSplit). It keys every check on the node
// itself — tracking region nesting via VisitStmt_(SplitAivScopeStmtPtr) — rather
// than on a function-level split_aiv attr, so multi-mode / nested / sub-region
// functions are checked region by region.
//
// MANUAL MODE. A function holding AT LEAST ONE region opts in, and there the
// region is authoritative for placement:
//
//   | op / value                 | inside a region | outside a region             |
//   | vector compute             | AIV             | ERROR — check (e)            |
//   | tile.load / tile.store     | AIV             | allowed (compiler-inserted)  |
//   | cube compute               | ERROR — check(a)| AIC                          |
//   | aiv_shard / aic_gather     | the boundary    | ERROR — check (c)            |
//
// and every tile value that crosses a region edge must say so with a boundary
// op — checks (f) (V->C) and (g) (C->V). That is what manual mode buys: the
// author, not the compiler, owns the AIC/AIV boundary, so the boundary is
// written down.
//
// A function with NO region keeps the pre-manual-mode behaviour untouched.
//
// Checks performed:
//   (a) No cube compute inside a region — for two independent reasons, so the
//       check fires in EVERY region regardless of mode. In a data-parallel
//       region each AIV lane holds only half the tile, so a cube op cannot be
//       vector-split; and in ANY region (task-parallel included) the region IS
//       the AIV lane's body, so cube work does not belong in it at all.
//   (b) No AIV reduce that collapses the split axis inside a region — that
//       produces a partial per-lane reduction (a miscompile). Unlike (a), this
//       reasoning really is split-axis-specific, so (b) stays gated on the
//       data-parallel modes (a NONE region has no split axis to collapse).
//   (c) tile.aiv_shard / tile.aic_gather (the AIV-split boundary) must appear
//       inside a region, never at top level. They are ACCEPTED in a
//       task-parallel (SplitMode::None) region: there they mean "this value
//       crosses the AIC/AIV boundary" without splitting it (the deducer is
//       shape-preserving at split=0 — see cross_core.cpp), which is exactly what
//       checks (f)/(g) require an author to write there.
//   (d) The boundary memory contract: tile.aiv_shard is Acc -> Vec and
//       tile.aic_gather is Vec -> Mat (see cross_core.cpp). Both ops ARE the
//       cross-core transfer, so the operand must live on the PRODUCING lane and
//       the result on the CONSUMING one. Mode-independent — a task-parallel
//       crossing spans the same two lanes as a data-parallel one — so it runs in
//       every region. Each side is skipped until its memory space is resolved,
//       so the check is inert at the OutlineIncoreScopes verification point
//       (where the boundary is still the space-less `tensor.*` form) and live
//       from ConvertTensorToTileOps onwards — which is why both that pass and
//       InferTileMemorySpace re-produce this property (see pass_properties.h);
//       without that, (d) would never run.
//   (e) No VECTOR-affine compute outside every region, in a function that holds
//       at least one region. With the region authoritative for placement, such
//       an op has no defined home: it is neither pinned to the AIV lane by a
//       region nor cube work. tile.load / tile.store are carved out — they are
//       NOT an authoring escape hatch but the COMPILER's own output:
//       ConvertTensorToTileOps materializes the load/store pair for a
//       tensor-level op OUTSIDE the region even when the compute it feeds was
//       written inside one (a `pl.exp(gm_tensor)` in a region becomes a hoisted
//       tile.load plus an in-region tile.exp). Without the carve-out the check
//       fires on IR the compiler itself produced.
//   (f) V->C: a value DEFINED inside a region and consumed by a CUBE-lane op
//       OUTSIDE it must be the result of a tile.aic_gather.
//   (g) C->V: a CUBE-produced value defined OUTSIDE every region and consumed by
//       a VECTOR-lane op INSIDE one must reach it through a tile.aiv_shard.
//
// (f) and (g) are the point of manual mode. BOTH crossings already lower without
// them: the compiler happily emits tpush/tpop(split=0) for an implicit crossing
// in either direction, and did so before these checks existed. What it cannot do
// is decide, on the author's behalf, WHERE the AIC/AIV boundary belongs — and a
// boundary the author did not write is a boundary the author did not think
// about. In manual mode the crossing is therefore spelled out or it is an error.
// The `tile.load` / `tile.store` carve-out from (e) applies to both roles here
// (definer and consumer) for the same reason and with the same weight: the
// compiler hoists them out of the region itself, so without the carve-out (f)
// and (g) fire on the compiler's own output rather than on anything an author
// wrote.
//
// Both are keyed on operand/result MEMORY SPACES (via core_affinity), so like
// (d) they are inert until those are resolved: at the OutlineIncoreScopes
// verification point a `tensor.matmul` classifies SHARED and neither check has
// anything to say. They come alive at the ConvertTensorToTileOps and
// InferTileMemorySpace verification points, which is early enough to reject the
// program long before the boundary is folded into tpush/tpop.
//
// NOT CHECKED, DELIBERATELY: "a once-only side effect (pld.system.notify) sits
// outside every region in a mixed kernel". Such a check was written and then
// removed, because it cannot deliver the guarantee its diagnostic would
// promise. Pinning the op to the AIV lane (which the pass 20 placement stamp
// does, and which is kept) removes the CUBE-lane copy — but the AIV function
// carries `dual_aiv_dispatch`, so its body still runs on BOTH AIV sub-lanes.
// An un-sharded notify therefore lands twice whether or not it is in a region,
// and a region cannot be made to mean "once". Worse, the correct authoring form
// — sharding the work by `aiv_id` so each lane notifies DIFFERENT peers — and
// the incorrect one — both lanes notifying the SAME peer — compile to the very
// same single statement in the AIV body; they differ only in whether `aiv_id`
// reaches the call's arguments, which no robust structural check can tell apart
// (any dataflow approximation false-negatives the moment the index goes through
// arithmetic). The rule is therefore stated for authors in
// docs/en/user/language/04-scopes.md and the `pl.split_aiv` docstring, and left
// unenforced. See the placement stamp (kCorePlacementAttr) for what the
// compiler DOES guarantee: comm ops written in a region stay off the cube lane.
//
// The checked ops (matmul, reduces, aiv_shard/aic_gather, vector compute) are
// always plain Calls with a non-null op_; Submits carry a GlobalVar callee and
// no op_, so no SubmitPtr override is needed (see pass-submit-awareness.md).
class SplitAivStructuralVerifier : public IRVisitor {
 public:
  SplitAivStructuralVerifier(std::vector<Diagnostic>& diagnostics, const FunctionSplitFacts& facts,
                             bool func_is_incore, bool func_from_outlining)
      : diagnostics_(diagnostics),
        facts_(facts),
        func_is_incore_(func_is_incore),
        func_from_outlining_(func_from_outlining) {}

  // (h) PLACEMENT — a CORE_GROUP-level region must not be authored inside a
  // function that is already a core function.
  //
  // A region inside a `FunctionType::InCore` function is legal in exactly one
  // case: that function is the one OutlineIncoreScopes MADE by outlining a scope
  // that held the region. The IR records that provenance directly — the outliner
  // stamps `split_aiv` on every function it mints from a region-bearing scope
  // (`scope_outline_utils.cpp`, and LowerAutoVectorSplit re-stamps it) — so the
  // check is "InCore function carrying a region, but not one the outliner
  // produced".
  //
  // Provenance rather than shape is what makes this survive the parser emitting
  // the region BARE in an InCore function (which it must, so that printing an
  // outlined function and reparsing it rebuilds the same IR). A shape-based test
  // — "region nested in a surviving InCore scope" — worked only while the parser
  // wrapped every top-level region, and would silently stop rejecting anything
  // once it stopped.
  //
  // Reporting here rather than at LowerAutoVectorSplit (pass 20) puts the
  // diagnostic 12 passes closer to the source; that pass keeps its own guard as
  // the backstop for a region behind a scope the lowering walks cannot enter.
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override {
    INTERNAL_CHECK_SPAN(op->body_, op->span_) << "Internal error: SplitAivScopeStmt has null body";
    if (func_is_incore_ && !func_from_outlining_) {
      Err(op->span_,
          "'pl.split_aiv' opens a CORE_GROUP-level region, so it cannot be authored inside a "
          "function declared 'pl.FunctionType.InCore' — that function is already a core function. "
          "A region reaches an InCore function only when OutlineIncoreScopes lifted the enclosing "
          "CORE_GROUP scope into it, and that pass processes only Opaque / Orchestration functions. "
          "Declare the enclosing function with plain @pl.function / @pl.jit (Opaque), or open the "
          "region inside a 'with pl.at(level=pl.Level.CORE_GROUP):' scope in such a function, and "
          "let pass 8 outline it.");
    }
    int prev_split_dim = cur_split_dim_;
    ++depth_;
    // A task-parallel (None) region has NO split axis — both lanes run the full
    // body, dispatched via aiv_id. Mark cur_split_dim_ = -1 so the one genuinely
    // split-axis-specific rule, (b) (partial reduction), is skipped for it. Every
    // other check is mode-independent and runs here as in any region.
    cur_split_dim_ = (op->split_ == SplitMode::None) ? -1 : split_axis::SplitDimension(op->split_);
    IRVisitor::VisitStmt(op->body_);
    --depth_;
    cur_split_dim_ = prev_split_dim;
  }

  void VisitExpr_(const CallPtr& op) override {
    if (op && op->op_) {
      // The AIV-split boundary appears in two forms in this window: the tile-level
      // tile.aiv_shard / tile.aic_gather (AUTO split_aiv path, and the outlined
      // low-level form) and the author-facing tensor.aiv_shard / tensor.aic_gather
      // (pl.aiv_shard(tensor) inside a pl.split_aiv region, still tensor.* until
      // ConvertTensorToTileOps lowers them 1:1).
      //
      // The two flags have different domains: region-scoping (c) applies to
      // BOTH forms, while the memory contract (d) is meaningful only once the
      // operand and result are tiles — a TensorType carries no memory space.
      // Keeping them apart lets (d) skip the tensor.* forms without re-deriving
      // the op identity inside CheckBoundaryMemory.
      const bool tile_boundary = IsOp(op, "tile.aiv_shard") || IsOp(op, "tile.aic_gather");
      const bool boundary = IsBoundaryOp(op);
      const bool in_region = depth_ > 0;
      const bool data_parallel = in_region && cur_split_dim_ != -1;  // UpDown / LeftRight
      CheckCrossingUses(op, in_region);
      if (in_region) {
        // (a) Cube compute cannot live inside ANY region — see the header block:
        // a data-parallel region cannot vector-split it, and every region (NONE
        // included) is the AIV lane's body, where cube work does not belong.
        if (!boundary && core_affinity::ClassifyCallAffinity(op) == core_affinity::CoreAffinity::CUBE) {
          Err(op->span_,
              "cube op '" + op->op_->name_ +
                  "' inside a pl.split_aiv region: the region body is AIV work, so cube ops do not "
                  "belong in it" +
                  (data_parallel ? ", and each AIV lane holds only half the tile so it cannot be "
                                   "vector-split either"
                                 : "") +
                  "; move it outside the region" +
                  (data_parallel ? ", or gather the lanes back to a full tile (tile.aic_gather) first."
                                 : "."));
        }
        if (data_parallel) {
          // (b) A reduce over the split axis yields a partial per-lane result.
          // Gated on the data-parallel modes: a NONE region has no split axis,
          // so the same reduce is a full (not partial) reduction on both lanes.
          if (split_axis::IsReduceOnSplitAxis(op, cur_split_dim_)) {
            Err(op->span_, "reduce op '" + op->op_->name_ + "' reduces over the split axis (dim " +
                               std::to_string(cur_split_dim_) +
                               ") inside a pl.split_aiv region, producing a partial reduction; reduce "
                               "the non-split axis, or gather the lanes back (tile.aic_gather) before "
                               "reducing.");
          }
        }
        // (d) The boundary memory contract — tile forms only (see above). Runs in
        // EVERY region: which lane produces the value and which consumes it does
        // not depend on the split mode, only the shape does.
        if (tile_boundary) CheckBoundaryMemory(op);
      } else {
        // Outside every region.
        if (boundary) {
          // (c) The AIV-split boundary op escaped its region.
          Err(op->span_, "'" + op->op_->name_ +
                             "' must appear inside a pl.split_aiv region (it marks the AIV-split "
                             "boundary and is only meaningful there).");
        } else {
          CheckOutOfRegionOp(op);
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  /// (f) and (g): every tile value crossing a region edge must name the crossing.
  ///
  /// One walk over the consumer's operands, one hash lookup per operand against
  /// the pre-pass def map — O(1) per operand, so the whole check is O(N).
  ///
  /// The two directions are deliberately keyed on opposite ends of the boundary
  /// op, because that is where each one sits: an `aic_gather` is written INSIDE
  /// the region and is the DEFINITION of the value the cube reads, while an
  /// `aiv_shard` is written INSIDE the region and is the CONSUMER of the value
  /// the cube produced. Both spellings put the boundary op in the region, which
  /// is what check (c) independently requires.
  void CheckCrossingUses(const CallPtr& op, bool in_region) {
    if (!facts_.has_region) return;
    // The boundary ops ARE the crossing; the hoisted memory ops are the
    // compiler's own out-of-region output (see the header block). Neither is an
    // implicit crossing, so neither is judged as a consumer.
    if (IsBoundaryOp(op) || IsHoistedMemoryOp(op)) return;

    const auto lane = ConsumerLane(op);
    if (in_region ? (lane != core_affinity::CoreAffinity::VECTOR)
                  : (lane != core_affinity::CoreAffinity::CUBE)) {
      return;
    }

    // A call may name the same value twice (`tile.add(x, x)`); one diagnostic per
    // offending value is the report, two is noise. Args are few, so a linear scan
    // of this list is cheaper than a hash set and keeps the check O(1) per operand.
    std::vector<const Var*> reported;
    for (const auto& arg : op->args_) {
      auto var = AsVarLike(arg);
      if (!var) continue;
      if (std::find(reported.begin(), reported.end(), var.get()) != reported.end()) continue;
      auto it = facts_.defs.find(var.get());
      if (it == facts_.defs.end()) continue;
      const ValueDef& def = it->second;
      // A definition the compiler hoisted out of the region says nothing about
      // where the author put the value — same carve-out, same reason.
      if (def.call && IsHoistedMemoryOp(def.call)) continue;

      if (!in_region && def.in_region) {
        // (f) V->C: defined inside a region, read on the cube lane outside it.
        if (def.call && IsGatherOp(def.call)) continue;
        Err(op->span_,
            "'" + var->name_hint_ + "' is defined inside a pl.split_aiv region but '" + op->op_->name_ +
                "' reads it on the CUBE lane outside that region, so the value crosses the AIV -> AIC "
                "boundary implicitly. In manual mode (this function opens a region) the crossing must "
                "be written down: wrap it in pl.aic_gather(...) inside the region — "
                "'" +
                var->name_hint_ + " = pl.aic_gather(" + var->name_hint_ +
                ")' — and read the "
                "gathered value here.");
        reported.push_back(var.get());
      } else if (in_region && !def.in_region && def.call &&
                 core_affinity::ClassifyCallAffinity(def.call) == core_affinity::CoreAffinity::CUBE) {
        // (g) C->V: cube-produced outside every region, read on the vector lane
        // inside one. Gated on the DEFINER being cube work rather than on the
        // value's memory space, so a Vec value the compiler hoisted (or a
        // lane-neutral scalar) is never mistaken for a crossing.
        Err(op->span_,
            "'" + var->name_hint_ + "' is produced on the CUBE lane by '" + def.call->op_->name_ +
                "' outside every pl.split_aiv region, but '" + op->op_->name_ +
                "' reads it on the VECTOR lane inside one, so the value crosses the AIC -> AIV boundary "
                "implicitly. In manual mode (this function opens a region) the crossing must be written "
                "down: bring it in with pl.aiv_shard(...) at the top of the region — "
                "'" +
                var->name_hint_ + " = pl.aiv_shard(" + var->name_hint_ +
                ")' — and read the "
                "sharded value here.");
        reported.push_back(var.get());
      }
    }
  }

  /// (e): the rule that applies to a call sitting outside every region. It is
  /// function-gated (see FunctionSplitFacts) rather than node-local, which is
  /// why it lives here and not in the region arm above.
  void CheckOutOfRegionOp(const CallPtr& op) {
    // (e) Manual mode: with the region authoritative for vector placement,
    // vector compute at this level has no defined home.
    //
    // tile.load / tile.store are carved out because they are the COMPILER's own
    // out-of-region output, not an authoring choice — ConvertTensorToTileOps
    // hoists the load/store pair for a tensor-level op out of the region that
    // holds the compute it feeds. Without the carve-out this check fires on
    // legal programs whose only out-of-region vector ops the compiler inserted.
    //
    // An op that DECLARES its lane via set_core_affinity is carved out too, and
    // for a stronger reason: manual mode exists so the compiler stops INFERRING
    // AIV placement outside a region. An op whose lane the registry states
    // outright was never inferred, so a region cannot make it any less
    // ambiguous. Without this, the check rejects programs that were legal and
    // unambiguous before — a `system.syncall(core_type="aiv_only")` barrier, a
    // `system.sync_set(core_type="aiv")` event, or a `pld.tile.put` — with a
    // message about placement that is simply untrue of them.
    if (facts_.has_region && !IsHoistedMemoryOp(op) && !core_affinity::HasStatedLane(op) &&
        core_affinity::ClassifyCallAffinity(op) == core_affinity::CoreAffinity::VECTOR) {
      Err(op->span_,
          "vector op '" + op->op_->name_ +
              "' sits outside every pl.split_aiv region, but this function already opens one, so "
              "the regions are authoritative for where vector work runs and this op has no lane to "
              "run on. Wrap this phase in its own region — "
              "'for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):' runs the full body on both AIV "
              "lanes and is the task-parallel form for full-width compute — or move the op into an "
              "existing region.");
    }
  }

  /// (d) Enforce the producing-lane / consuming-lane memory spaces of a
  /// split-axis boundary op. Both sides are skipped until their space is
  /// resolved, so the same verifier is safe to run across the whole
  /// OutlineIncoreScopes .. LowerAutoVectorSplit window.
  void CheckBoundaryMemory(const CallPtr& op) {
    auto contract = GetBoundaryMemoryContract(op);
    if (!contract) return;

    // Operand side: the value must still sit on the lane that produced it. The
    // dominant failure is a shard of a vector-produced value: the use migrates to
    // the AIV half while the producer stays behind, leaving the cube half
    // referencing a value it never defines (which surfaces much later as an orphan
    // Mem.Vec allocation and an internal codegen error).
    if (!op->args_.empty()) {
      if (auto operand_ms = ResolvedTileMemory(op->args_[0]);
          operand_ms.has_value() && *operand_ms != contract->operand) {
        Err(op->span_, "'" + op->op_->name_ + "' operand is in " + MemorySpaceToString(*operand_ms) +
                           ", but it transfers a " + contract->producer +
                           "-produced value across the cross-core boundary and requires " +
                           MemorySpaceToString(contract->operand) + "." + contract->operand_hint);
      }
    }

    // Result side: the declared type describes the CONSUMING lane, i.e. the space
    // ExpandMixedKernel materializes the boundary tpop in.
    if (auto result_ms = ResolvedTileMemory(op); result_ms.has_value() && *result_ms != contract->result) {
      Err(op->span_, "'" + op->op_->name_ + "' result is in " + MemorySpaceToString(*result_ms) +
                         ", but it delivers its " + contract->delivery + " lane and must be " +
                         MemorySpaceToString(contract->result) +
                         " (the memory ExpandMixedKernel pops it into).");
    }
  }

  void Err(const Span& span, const std::string& message) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "AivSplitValid", 0, message, span);
  }

  std::vector<Diagnostic>& diagnostics_;
  /// Whole-function facts for checks (e), (f) and (g). Held by reference — the
  /// scanner that owns them is declared before this verifier in `Verify` and
  /// outlives it, and `defs` is one entry per binding, too big to copy per
  /// function for no gain.
  const FunctionSplitFacts& facts_;
  int depth_ = 0;                ///< Region nesting depth (>0 means inside a split_aiv region).
  int cur_split_dim_ = -1;       ///< Split axis of the innermost enclosing region (-1 outside any region).
  bool func_is_incore_ = false;  ///< Enclosing function is FunctionType::InCore, for check (h).
  bool func_from_outlining_ = false;  ///< It carries the outliner's ``split_aiv`` stamp, for check (h).
};

}  // namespace

// Verifies IRProperty::AivSplitValid as a structural property of the first-class
// SplitAivScopeStmt region. The node is live only between OutlineIncoreScopes
// (which produces the property) and LowerAutoVectorSplit (which consumes/erases
// the node and invalidates the property), so the verifier walks every function
// body in that window and applies the region-scoped checks above. No
// function-attr / split-mode gate — the node itself is the source of truth,
// including for the manual-mode gate ((e) region presence), which the
// per-function pre-pass derives by walking the body.
class AivSplitValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AivSplitValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Pre-pass first: check (e) is gated on a whole-function fact (does this
      // function open a region at all), which a statement appearing later in
      // the body can still establish.
      FunctionSplitFactScanner scanner;
      scanner.VisitStmt(func->body_);
      SplitAivStructuralVerifier verifier(
          diagnostics, scanner.facts(), func->func_type_ == FunctionType::InCore,
          func->HasAttr(kSplitAivAttr) && func->GetAttr<bool>(kSplitAivAttr, false));
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateAivSplitValidPropertyVerifier() {
  return std::make_shared<AivSplitValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
