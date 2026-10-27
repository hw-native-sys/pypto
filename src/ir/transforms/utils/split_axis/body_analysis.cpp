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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/utils/split_axis/internal.h"

namespace pypto {
namespace ir {
namespace split_axis {

using detail::SplitInfoFromHalvedType;

namespace {

// Mirrors the (formerly file-local) hazard finder in ExpandMixedKernel: records
// the first tile.transpose whose source carries the split axis and whose
// transpose actually swaps it. Shared so the explicit per-region check in pass 23
// and the AUTO whole-function check in pass 24 use one detector.
class TransposeSplitHazardFinder : public IRVisitor {
 public:
  explicit TransposeSplitHazardFinder(int split_dim) : split_dim_(split_dim) {}
  [[nodiscard]] CallPtr Offending() const { return offending_; }
  [[nodiscard]] const std::string& ResultName() const { return result_name_; }

 protected:
  // Each child region is checked separately against its own mode.
  void VisitStmt_(const SplitAivScopeStmtPtr&) override {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    Consider(As<Call>(op->value_), op->var_ ? op->var_->name_hint_ : "");
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const EvalStmtPtr& op) override {
    Consider(As<Call>(op->expr_), "");
    IRVisitor::VisitStmt_(op);
  }

 private:
  // Whether the transpose actually swaps the split axis (so the split data
  // migrates). tile.transpose carries the two axis indices as args[1]/args[2].
  [[nodiscard]] bool SwapsSplitAxis(const CallPtr& call) const {
    if (call->args_.size() < 3) return true;  // conservative if the axes are absent
    auto a0 = std::dynamic_pointer_cast<const ConstInt>(call->args_[1]);
    auto a1 = std::dynamic_pointer_cast<const ConstInt>(call->args_[2]);
    if (!a0 || !a1) return true;
    return static_cast<int>(a0->value_) == split_dim_ || static_cast<int>(a1->value_) == split_dim_;
  }

  void Consider(const CallPtr& call, const std::string& result_name) {
    if (offending_ || !call || !call->op_ || !IsOp(call, "tile.transpose") || call->args_.empty()) {
      return;
    }
    auto tt = std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
    if (!tt || split_dim_ < 0 || split_dim_ >= static_cast<int>(tt->shape_.size())) return;
    if (!SwapsSplitAxis(call)) return;  // split axis not transposed -> stays put, typed correctly
    // The split axis carries real data unless it is statically 1. A dynamic
    // (non-ConstInt) extent is treated as non-singleton: it cannot be proven
    // safe, so flag it conservatively.
    auto dim = std::dynamic_pointer_cast<const ConstInt>(tt->shape_[split_dim_]);
    if (!dim || dim->value_ != 1) {
      offending_ = call;
      result_name_ = result_name;
    }
  }

  int split_dim_;
  CallPtr offending_;
  std::string result_name_;
};

}  // namespace

TransposeSplitHazard FindTransposeSplitHazard(const StmtPtr& body, int split_dim) {
  if (!body) return {};
  TransposeSplitHazardFinder finder(split_dim);
  finder.VisitStmt(body);
  return {finder.Offending(), finder.ResultName()};
}

namespace {

// True iff any Var / IterArg referenced anywhere in ``exprs`` is lane-derived.
// One collector over all of them: the walk covers each whole expression tree, so
// a lane reference nested inside a MakeTuple offset list or an arithmetic
// sub-expression is found. VarDefUseCollector overrides both the Var and the
// IterArg handler (see var_collectors.h) — a loop-carried lane offset is a
// reference too.
bool ReferencesLaneIndex(const std::vector<ExprPtr>& exprs,
                         const std::unordered_set<const Var*>& lane_scalars) {
  if (lane_scalars.empty()) return false;
  var_collectors::VarDefUseCollector collector;
  for (const auto& e : exprs) {
    if (e) collector.VisitExpr(e);
  }
  for (const auto* v : collector.var_uses) {
    if (lane_scalars.count(v) != 0) return true;
  }
  return false;
}

// The positional args that carry an op's ADDRESS — the base offset selecting
// which window of the source this call reads. Empty for anything that is not an
// addressing op.
//
// Only these args are consulted for a lane reference. A lane-derived scalar
// anywhere ELSE in an addressing op — a shape, a valid_shape — does not localize
// the window: ``tile.load(data, [0, 0], [64, 128], valid_shape=[aiv_id + 1,
// 128])`` mentions aiv_id, yet both lanes still read the same base rows. Scanning
// every arg would admit it and then trust its consumers as half-width.
std::vector<ExprPtr> AddressArgs(const CallPtr& call) {
  const auto& args = call->args_;
  auto at = [&args](size_t i) -> ExprPtr { return i < args.size() ? args[i] : nullptr; };
  if (IsOp(call, "tile.load")) return {at(1)};            // (tensor, offsets, shapes, valid_shape)
  if (IsOp(call, "tile.slice")) return {at(2)};           // (input, shape, offset, valid_shape, drop_dims)
  if (IsOp(call, "tile.extract")) return {at(1), at(2)};  // (src, index_row, index_col, shape)
  // (dst, src, dst_offset, src_offset, shapes[, valid_shape]) — DPS: the op reads
  // a GM row window at ``src_offset`` and writes it into its own accumulator at
  // ``dst_offset``. Only ``src_offset`` is the READ window, so only it localizes:
  // a lane-derived src_offset means the two lanes gather DIFFERENT rows (the
  // per-lane scattered gather). A lane-derived dst_offset with a lane-invariant
  // src is the opposite shape — both lanes fetch the same rows into different
  // slots of a FULL-width accumulator — which is exactly what this scan must
  // still reject.
  if (IsOp(call, "tile.gather_row")) return {at(3)};
  return {};
}

// Replicated singleton arithmetic is neutral, just like a broadcast load. Only
// read-only tile/scalar inputs qualify; a singleton result cannot excuse writes.
bool IsBroadcastComputation(const CallPtr& call, int split_dim) {
  auto result = As<TileType>(call->GetType());
  if (!result || !IsSingletonSplitAxis(*result, split_dim) || core_affinity::IsNoDuplicateCall(call)) {
    return false;
  }
  const auto* entry = LookupOpEntry(call->op_);
  if (!entry) return false;
  bool has_tile = false;
  for (size_t i = 0; i < call->args_.size(); ++i) {
    if (entry->GetArgEffect(i, call->kwargs_) != ArgEffect::Read) return false;
    if (auto tile = As<TileType>(call->args_[i]->GetType())) {
      has_tile = true;
      if (!IsBroadcastOnSplitAxis(*tile, split_dim, static_cast<int>(result->shape_.size()))) return false;
    } else if (!As<ScalarType>(call->args_[i]->GetType())) {
      return false;
    }
  }
  return has_tile;
}

using TupleHalfFacts = std::unordered_map<const Var*, std::shared_ptr<const std::vector<bool>>>;

bool IsHalfExpr(const ExprPtr& expr, const SplitBodyAnalysis& scan, const TupleHalfFacts& tuples) {
  if (auto var = AsVarLike(expr)) return scan.half_tiles.count(var.get()) != 0;
  if (auto item = As<TupleGetItemExpr>(expr)) {
    if (auto tuple = AsVarLike(item->tuple_)) {
      auto it = tuples.find(tuple.get());
      return it != tuples.end() && item->index_ < it->second->size() && (*it->second)[item->index_];
    }
    if (auto tuple = As<MakeTuple>(item->tuple_)) {
      return item->index_ < tuple->elements_.size() &&
             IsHalfExpr(tuple->elements_[item->index_], scan, tuples);
    }
  }
  return false;
}

std::shared_ptr<const std::vector<bool>> TupleSplitFacts(const ExprPtr& value, const SplitBodyAnalysis& scan,
                                                         const TupleHalfFacts& tuples) {
  if (auto var = AsVarLike(value)) {
    if (auto it = tuples.find(var.get()); it != tuples.end()) {
      return it->second;
    }
  } else if (auto tuple = As<MakeTuple>(value)) {
    std::vector<bool> facts;
    facts.reserve(tuple->elements_.size());
    for (const auto& element : tuple->elements_) facts.push_back(IsHalfExpr(element, scan, tuples));
    return std::make_shared<const std::vector<bool>>(std::move(facts));
  }
  return nullptr;
}

void CopySplitFact(const VarPtr& target, const ExprPtr& value, SplitBodyAnalysis& scan,
                   TupleHalfFacts& tuples) {
  if (IsHalfExpr(value, scan, tuples)) scan.half_tiles.insert(target.get());
  if (auto facts = TupleSplitFacts(value, scan, tuples)) tuples[target.get()] = std::move(facts);
}

// A merge is lane-local only on every incoming edge. For a loop, the backedge
// must preserve any entry fact used to admit the body. A neutral entry may gain
// a lane-local value, but the carry and exit stay neutral for zero iterations.
void MergeSplitFacts(const VarPtr& target, const ExprPtr& lhs, const ExprPtr& rhs, SplitBodyAnalysis& scan,
                     TupleHalfFacts& tuples, const VarPtr& loop_carry = nullptr) {
  bool mismatch = false;
  if (auto tuple = As<TupleType>(target->GetType())) {
    auto left = TupleSplitFacts(lhs, scan, tuples);
    auto right = TupleSplitFacts(rhs, scan, tuples);
    std::vector<bool> merged;
    merged.reserve(tuple->types_.size());
    for (size_t i = 0; i < tuple->types_.size(); ++i) {
      const bool a = left && i < left->size() && (*left)[i];
      const bool b = right && i < right->size() && (*right)[i];
      mismatch |= a && !b;
      merged.push_back(a && b);
    }
    tuples[target.get()] = std::make_shared<const std::vector<bool>>(std::move(merged));
  } else {
    const bool a = IsHalfExpr(lhs, scan, tuples);
    const bool b = IsHalfExpr(rhs, scan, tuples);
    mismatch = a && !b;
    if (a && b) scan.half_tiles.insert(target.get());
  }
  if (loop_carry && mismatch) {
    scan.carry_mismatches.push_back(loop_carry->name_hint_);
  }
}

// Track tiles that are part of the half-width boundary dataflow: results of
// tile.aiv_shard, plus results of VECTOR-affine ops that consume such a half
// tile. Any VECTOR-affine op consuming NONE of them operates on full-width data
// — exactly what the implicit affinity gate would have halved. Records the names
// of such full-width vector ops in a single ordered walk.
void ScanSplitBody(const std::vector<StmtPtr>& stmts, int split_dim, SplitBodyAnalysis& scan,
                   TupleHalfFacts& tuples) {
  for (const auto& stmt : stmts) {
    CallPtr leaf;
    VarPtr def_var;
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    if (assign) {
      leaf = std::dynamic_pointer_cast<const Call>(assign->value_);
      def_var = assign->var_;
      CopySplitFact(def_var, assign->value_, scan, tuples);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      leaf = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }

    // --- Lane-index dataflow (SCALARS) --------------------------------------
    // SEED: the region's own lane index, bound by ``aiv_id =
    // tile.get_subblock_idx()`` (the parser emits it as the region body's first
    // statement — ast_parser.py::_emit_split_aiv_region). Matched by OP, not by
    // position, so a re-injected or reordered binding is still found. The parser
    // emits it unconditionally, so the set is normally non-empty from statement
    // one; it stays empty only if the binding is absent (e.g. DCE stripped an
    // unused aiv_id, or hand-built IR omits it), in which case nothing is
    // admitted below and the guard behaves exactly as before.
    // PROPAGATE: any scalar bound from an expression referencing a lane-derived
    // scalar (``kv_lane0 = kv0 + aiv_id * 64``). Program order + SSA put every
    // def before its uses, so one forward pass reaches the fixpoint. A lane value
    // carried through a loop iter_arg is NOT propagated (the walk does not visit
    // loop phi defs) — conservative: it rejects rather than wrongly admits.
    if (def_var && As<ScalarType>(def_var->GetType()) &&
        (IsOp(leaf, "tile.get_subblock_idx") ||
         // ``assign &&`` is defensive: def_var is currently only set in the
         // AssignStmt arm above, so def_var non-null already implies it.
         (assign && ReferencesLaneIndex({assign->value_}, scan.lane_scalars)))) {
      scan.lane_scalars.insert(def_var.get());
    }

    if (leaf && leaf->op_) {
      // aiv_shard produces a HALF tile (the C->V boundary); seed the dataflow.
      if (IsOp(leaf, "tile.aiv_shard")) {
        if (def_var) scan.half_tiles.insert(def_var.get());
        continue;
      }
      // aic_gather doubles HALF -> FULL (the V->C boundary back to cube); its
      // result leaves the half-width dataflow, so it is not tracked.
      if (IsOp(leaf, "tile.aic_gather")) {
        continue;
      }
      // A singleton broadcast read is replicated, not partitioned. Do not
      // seed half_tiles: accepting this producer does not prove its users half-width.
      if (auto tile = As<TileType>(leaf->GetType());
          IsOp(leaf, "tile.load") && tile &&
          (tile->shape_.size() < 2 || split_axis::IsSingletonSplitAxis(*tile, split_dim))) {
        continue;
      }
      if (IsBroadcastComputation(leaf, split_dim)) continue;
      // Pure generators are lane-invariant ROOTS: the result is a function of the
      // op's ATTRIBUTES ONLY -- they read no tile and no memory, so there is no
      // "which half do I read?" question for them to get wrong, and replicating
      // one on both lanes is correct by construction at whatever extent the
      // author wrote.
      //
      // Deliberately NOT inserted into half_tiles: being lane-invariant makes a
      // generator SAFE, it does not make it HALF-WIDTH. Admitting it would let a
      // full-width generator vouch for its consumers and silently suppress a real
      // rejection -- e.g. `z = tile.full([128,128]); y = tile.add(z, z);
      // tile.store(y, [0,0], out, atomic)`, where both lanes would atomically add
      // the same full tile (double-counted result). Leaving it NEUTRAL keeps each
      // consumer judged on its own merits.
      //
      // (tile.create is listed for category completeness; it also classifies
      // SHARED, so the VECTOR arm below would never report it anyway. For
      // tile.random, replication means both lanes draw the SAME stream from the
      // same key/counter -- identical to what the implicit path produces, since
      // halving shrinks the shape, not the stream. Per-lane independence is the
      // author's job: vary the counter with aiv_id.)
      if (IsOp(leaf, "tile.full") || IsOp(leaf, "tile.create") || IsOp(leaf, "tile.ci") ||
          IsOp(leaf, "tile.random")) {
        continue;
      }
      // Dataflow propagation over tile-producing ops. An op that consumes a half
      // tile STAYS in the half-width dataflow regardless of its affinity
      // classification -- crucially this includes a Vec->Vec tile.move between the
      // shard and the compute, which classifies MIXED/SHARED (not VECTOR). Only a
      // VECTOR-affine tile op that consumes NONE of the half tiles is genuinely
      // full-width: that is exactly what the implicit affinity gate would halve,
      // and what the explicit-passthrough path would leave un-localized (both AIV
      // lanes computing the full tile). A scalar-producing VECTOR op (e.g.
      // tile.get_subblock_idx) is not a tile op, so it never flags.
      if (As<TileType>(leaf->GetType()) || As<TupleType>(leaf->GetType())) {
        bool consumes_half = def_var && scan.half_tiles.count(def_var.get()) != 0;
        for (const auto& arg : leaf->args_) consumes_half |= IsHalfExpr(arg, scan, tuples);
        // Stays in the half-width dataflow when it either consumes a half tile,
        // or is AUTHOR-LOCALIZED: an op whose ADDRESS args reference the region's
        // lane index, so the author explicitly wrote it per-lane — e.g. a GM load
        // at ``[kv0 + aiv_id * 64, 0]``. That is the same trust the pass already
        // extends to tile.store, whose lane-dependent offset it never checks (a
        // store returns a TensorType, so it never reaches this branch).
        //
        // Localization is trusted only via AddressArgs, and so only on addressing
        // ops. On any other op a lane-derived scalar is just an operand and proves
        // nothing about width; without that restriction
        // ``tile.set_validshape(full_width_tile, 1, valid_aiv)`` would launder a
        // full-width tile into the half-width dataflow and silence every
        // downstream check.
        if (consumes_half ||
            (!scan.lane_scalars.empty() && ReferencesLaneIndex(AddressArgs(leaf), scan.lane_scalars))) {
          if (def_var) {
            if (auto tuple = As<TupleType>(leaf->GetType())) {
              std::vector<bool> facts;
              for (const auto& type : tuple->types_) facts.push_back(As<TileType>(type) != nullptr);
              tuples[def_var.get()] = std::make_shared<const std::vector<bool>>(std::move(facts));
            } else {
              scan.half_tiles.insert(def_var.get());
            }
          }
        } else if (IsOp(leaf, "tile.reshape") || IsOp(leaf, "tile.reinterpret_view")) {
          // A full view may stage a later lane-local slice. It remains neutral;
          // a full-width arithmetic consumer still has to establish its own split.
          continue;
        } else if (core_affinity::ClassifyCallAffinity(leaf) == core_affinity::CoreAffinity::VECTOR) {
          scan.full_width_vec_ops.push_back(leaf->op_->name_);
        }
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      for (const auto& arg : for_stmt->iter_args_) CopySplitFact(arg, arg->initValue_, scan, tuples);
      ScanSplitBody(transform_utils::FlattenToStmts(for_stmt->body_), split_dim, scan, tuples);
      auto yield = transform_utils::GetLastYieldStmt(for_stmt->body_);
      if (yield) {
        for (size_t i = 0;
             i < for_stmt->return_vars_.size() && i < for_stmt->iter_args_.size() && i < yield->value_.size();
             ++i) {
          MergeSplitFacts(for_stmt->return_vars_[i], for_stmt->iter_args_[i]->initValue_, yield->value_[i],
                          scan, tuples, for_stmt->iter_args_[i]);
        }
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      ScanSplitBody(transform_utils::FlattenToStmts(if_stmt->then_body_), split_dim, scan, tuples);
      if (if_stmt->else_body_.has_value()) {
        ScanSplitBody(transform_utils::FlattenToStmts(*if_stmt->else_body_), split_dim, scan, tuples);
        auto lhs = transform_utils::GetLastYieldStmt(if_stmt->then_body_);
        auto rhs = transform_utils::GetLastYieldStmt(*if_stmt->else_body_);
        if (lhs && rhs) {
          for (size_t i = 0;
               i < if_stmt->return_vars_.size() && i < lhs->value_.size() && i < rhs->value_.size(); ++i) {
            MergeSplitFacts(if_stmt->return_vars_[i], lhs->value_[i], rhs->value_[i], scan, tuples);
          }
        }
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      for (const auto& arg : while_stmt->iter_args_) CopySplitFact(arg, arg->initValue_, scan, tuples);
      ScanSplitBody(transform_utils::FlattenToStmts(while_stmt->body_), split_dim, scan, tuples);
      auto yield = transform_utils::GetLastYieldStmt(while_stmt->body_);
      if (yield) {
        for (size_t i = 0; i < while_stmt->return_vars_.size() && i < while_stmt->iter_args_.size() &&
                           i < yield->value_.size();
             ++i) {
          MergeSplitFacts(while_stmt->return_vars_[i], while_stmt->iter_args_[i]->initValue_,
                          yield->value_[i], scan, tuples, while_stmt->iter_args_[i]);
        }
      }
    } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      ScanSplitBody(seq->stmts_, split_dim, scan, tuples);
    }
  }
}

}  // namespace

SplitBodyAnalysis AnalyzeSplitBody(const std::vector<StmtPtr>& stmts, int split_dim,
                                   const std::unordered_map<const Var*, TileInfo>& known_tiles,
                                   const std::unordered_map<const Var*, VarPtr>& replacements) {
  SplitBodyAnalysis analysis;
  for (const auto& [var, info] : known_tiles) analysis.half_tiles.insert(var);
  TupleHalfFacts tuples;
  // Loop repair can already reference replacement Vars while other uses still
  // name the originals. Both identities describe the same transformed value;
  // retain per-element tuple facts before the final Substitute/DeepClone.
  for (const auto& [before, after] : replacements) {
    if (analysis.half_tiles.count(before)) analysis.half_tiles.insert(after.get());
    auto before_tuple = As<TupleType>(before->GetType());
    auto after_tuple = As<TupleType>(after->GetType());
    if (!before_tuple || !after_tuple || before_tuple->types_.size() != after_tuple->types_.size()) {
      continue;
    }
    std::vector<bool> facts;
    facts.reserve(before_tuple->types_.size());
    for (size_t i = 0; i < before_tuple->types_.size(); ++i) {
      facts.push_back(SplitInfoFromHalvedType(before_tuple->types_[i], after_tuple->types_[i]).has_value());
    }
    auto shared = std::make_shared<const std::vector<bool>>(std::move(facts));
    tuples[before] = shared;
    tuples[after.get()] = std::move(shared);
  }
  ScanSplitBody(stmts, split_dim, analysis, tuples);
  return analysis;
}

}  // namespace split_axis
}  // namespace ir
}  // namespace pypto
