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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Is `space` one of the fractal-organized on-chip spaces, i.e. one where a
/// tile has an N-fractal pitch that a compact mode can change?
bool IsFractalSpace(MemorySpace space) {
  return space == MemorySpace::Left || space == MemorySpace::Right || space == MemorySpace::Acc;
}

/// Guards the two halves of the Acc compact-mode contract.
///
/// **(1) An accumulator `mad` writes at the valid-row pitch must say so.**
/// `mad` takes M from the L0A operand's *valid* rows and lays the product out in
/// L0C with an N-fractal stride of `ceil(M/16)*16` (pto-isa `TMatmul.hpp`:
/// `uint16_t m = aMatrix.GetValidRow()`). Every L0C reader instead derives that
/// stride from the tile's compile-time physical `Rows` *unless* the tile is
/// compact, in which case it recomputes `ceil(validRow/16)*16` -- exactly the
/// pitch `mad` wrote at (`tstore_common.hpp`, `TStoreAccNz2nd` and siblings). A
/// narrowed accumulator that stays non-compact is therefore read back at a pitch
/// it was never written at, scrambling every N-fractal above the first: issue
/// #2470 through `tile.store`, issue #2510 through the Cube->Vector
/// `tile.tpush_to_aiv`. Both shipped as wrong numbers on device with no
/// diagnostic at any layer.
///
/// The check sits on `tile.matmul_acc` / `tile.matmul_mx_acc`, which is where
/// both halves of the comparison are in hand: the lhs whose valid rows `mad`
/// takes M from, and the accumulator buffer the result aliases in place. That is
/// also the one op that *inherits* compact rather than deriving it
/// (`matmul.cpp`), so it is exactly where a chain seeded by a non-compact buffer
/// loses the mode. Checking the readers instead would be wrong: a `tile.store`
/// cannot tell an accumulator `mad` wrote from an Acc tile some `tile.load`
/// filled at the physical pitch.
///
/// **(2) A compact accumulator's packed pitch survives its aliases.**
/// `tile.set_validshape` is metadata-only and deliberately keeps `compact`
/// (`transform.cpp`), because the pitch its readers must use is the one the
/// producing `mad` already laid the bytes out at. Nothing enforced that: a
/// compact accumulator re-narrowed across a fractal boundary -- `mad` writes 17
/// valid rows at pitch 32, the alias then declares 16 -- keeps the flag while
/// changing the number every compact reader derives the pitch from, so TPUSH and
/// TSTORE walk L0C at 16 over bytes packed at 32. The alias is rejected here
/// rather than silently re-interpreted; narrow the result after it leaves L0C
/// instead. A *fresh* full-box source is exempt: `AutoTileMatmulL0` declares its
/// accumulator seed as `tile.create(compact=True)` and narrows it immediately,
/// and a buffer nothing has written yet re-interprets nothing.
///
/// **(3) Only a fractal space carries a compact mode at all.** Compact is a
/// property of a fractal pitch. A UB (`Vec`) tile has none, and no pto-isa Vec
/// path reads `TileData::Compact` -- stamping it there is inert at best and, on
/// the ops that *do* consult it (`TMov`, `TFillPad`), a silent layout change.
/// Marking the Vec side of a C2V pop compact is a tempting non-fix for #2510;
/// this rejects it up front.
class AccCompactVisitor : public IRVisitor {
 public:
  AccCompactVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void CheckFunction(const FunctionPtr& func) {
    for (const auto& param : func->params_) {
      if (param) CheckSpaceOfType(param->GetType(), param->span_);
    }
    if (func->body_) VisitStmt(func->body_);
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) CheckSpaceOfType(op->GetType(), op->span_);
    IRVisitor::VisitVarLike_(op);
  }

  // The space rule is checked on Var-like definitions and params only, not on
  // call result types: `AssignStmt` binds the same type to both sides
  // (`AssignTypeSymmetry`), so checking the call as well would report one tile
  // twice, and a call result nothing binds is a tile nothing reads.
  void VisitExpr_(const CallPtr& op) override {
    if (op) {
      CheckAccumulate(op);
      CheckValidShapeAlias(op);
    }
    IRVisitor::VisitExpr_(op);
  }

  // A Submit launches a Function, so it cannot carry an operator today; routing
  // it through the Call view keeps the check correct if that ever changes (see
  // `pass-submit-awareness.md`).
  void VisitExpr_(const SubmitPtr& op) override {
    if (op) {
      const CallPtr view = SubmitToCallView(op);
      CheckAccumulate(view);
      CheckValidShapeAlias(view);
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  /// `args = (accumulator, lhs, rhs)`. The lhs is the L0A operand whose valid
  /// rows `mad` takes M from; the accumulator is the L0C buffer it writes.
  void CheckAccumulate(const CallPtr& call) {
    if (!IsOp(call, "tile.matmul_acc") && !IsOp(call, "tile.matmul_mx_acc")) return;
    if (call->args_.size() < 2 || !call->args_[0] || !call->args_[1]) return;

    auto acc_type = As<TileType>(call->args_[0]->GetType());
    auto lhs_type = As<TileType>(call->args_[1]->GetType());
    if (!acc_type || !lhs_type || acc_type->shape_.empty()) return;

    const TileView acc_view = tile_view_semantics::GetEffectiveTileView(*acc_type);
    if (acc_view.compact == CompactMode::normal) return;

    const TileView lhs_view = tile_view_semantics::GetEffectiveTileView(*lhs_type);
    if (lhs_view.valid_shape.empty()) return;
    if (AccPitchesCoincide(lhs_view.valid_shape[0], acc_type->shape_[0])) return;

    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "AccCompactValid", /*error_code=*/1,
        "'" + call->op_->name_ + "' accumulates " + PythonPrint(lhs_view.valid_shape[0]) +
            " valid rows into an accumulator that is not compact (function '" + func_name_ +
            "'). mad lays the product out at a pitch of ceil(validRow/16)*16, but a non-compact "
            "accumulator is read back at its physical row count " +
            PythonPrint(acc_type->shape_[0]) +
            ", which skews every N-fractal above the first. The accumulator this op writes into "
            "never carried CompactMode::normal -- a seed built by tile.create declares it with "
            "compact=True, and tile.matmul derives it via StampCompactForNarrowedAccRows.",
        call->span_);
  }

  /// `tile.set_validshape(src, rows, cols)` -- args[1] is the new valid row count.
  void CheckValidShapeAlias(const CallPtr& call) {
    if (!IsOp(call, "tile.set_validshape")) return;
    if (call->args_.size() < 2 || !call->args_[0] || !call->args_[1]) return;

    auto src_type = As<TileType>(call->args_[0]->GetType());
    if (!src_type || !src_type->memory_space_.has_value()) return;
    if (*src_type->memory_space_ != MemorySpace::Acc || src_type->shape_.empty()) return;

    const TileView src_view = tile_view_semantics::GetEffectiveTileView(*src_type);
    if (src_view.compact != CompactMode::normal || src_view.valid_shape.empty()) return;
    // A full-box source is a declaration, not a re-interpretation: nothing has
    // been written into it yet (the AutoTileMatmulL0 accumulator seed).
    if (ProveValidExtentEqual(src_view.valid_shape[0], src_type->shape_[0]) == ProofResult::kTrue) {
      return;
    }
    if (!PackedPitchDiffers(src_view.valid_shape[0], call->args_[1])) return;

    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "AccCompactValid", /*error_code=*/2,
        "'tile.set_validshape' re-narrows a compact accumulator from " +
            PythonPrint(src_view.valid_shape[0]) + " to " + PythonPrint(call->args_[1]) +
            " valid rows, which changes the pitch its readers derive (function '" + func_name_ +
            "'). The op is metadata-only, so the bytes stay packed at ceil(" +
            PythonPrint(src_view.valid_shape[0]) +
            "/16)*16 while every compact reader would now walk them at the new extent's pitch. "
            "Narrow the result after it leaves L0C -- cast or move it to Vec first, then "
            "pl.set_validshape the vector tile.",
        call->span_);
  }

  /// Do the two extents provably pack to different N-fractal pitches? Only a
  /// decidable difference is reported; an undecidable pair is left alone rather
  /// than rejecting IR that may well be consistent.
  static bool PackedPitchDiffers(const ExprPtr& old_rows, const ExprPtr& new_rows) {
    if (ProveValidExtentEqual(old_rows, new_rows) == ProofResult::kTrue) return false;
    auto old_const = As<ConstInt>(old_rows);
    auto new_const = As<ConstInt>(new_rows);
    if (!old_const || !new_const) return false;
    const auto packed = [](int64_t v) {
      return (v + kAccFractalRows - 1) / kAccFractalRows * kAccFractalRows;
    };
    return packed(old_const->value_) != packed(new_const->value_);
  }

  void CheckSpaceOfType(const TypePtr& type, const Span& span) {
    if (!type) return;
    if (auto tuple_type = As<TupleType>(type)) {
      for (const auto& sub : tuple_type->types_) {
        CheckSpaceOfType(sub, span);
      }
      return;
    }
    auto tile_type = As<TileType>(type);
    if (!tile_type || !tile_type->tile_view_.has_value() || !tile_type->memory_space_.has_value()) return;
    if (tile_type->tile_view_->compact == CompactMode::null) return;
    if (IsFractalSpace(*tile_type->memory_space_)) return;

    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "AccCompactValid", /*error_code=*/3,
        "tile in memory space '" + MemorySpaceToString(*tile_type->memory_space_) +
            "' carries a compact mode (function '" + func_name_ +
            "'). Compact describes an N-fractal pitch, so only Left/Right/Acc tiles may carry it; "
            "no pto-isa path reads it for any other space.",
        span);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

}  // namespace

class AccCompactValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AccCompactValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;
      AccCompactVisitor visitor(diagnostics, func->name_);
      visitor.CheckFunction(func);
    }
  }
};

PropertyVerifierPtr CreateAccCompactValidPropertyVerifier() {
  return std::make_shared<AccCompactValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
