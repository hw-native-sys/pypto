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

/**
 * @file legalize_tile_cast_pass.cpp
 * @brief Expand hardware-unsupported tile.cast pairs into native cast chains.
 *
 * Converts (src, dst) pairs that the active pto.tcvt profile cannot emit as a
 * single instruction into a shortest sequence of native casts. Path search is
 * BFS over the native-conversion table the active BackendHandler supplies via
 * GetTcvtAdjacency(), so this pass holds no per-architecture knowledge of its
 * own for the cast graph. Typical outcome for A5 INT32→FP16 is INT32→FP32→FP16
 * — same byte-width to float, then resize — which adds no precision loss beyond
 * the final narrow.
 *
 * Scratch required by the final native hops is materialized immediately before
 * MemRef initialization, after this pass has finished constructing the chain.
 */

#include <algorithm>
#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

// Round modes for tile.cast (None=0, RINT=1, ROUND=2, ...).
constexpr int kCastModeRound = 2;

using AdjList = std::unordered_map<uint8_t, std::vector<DataType>>;

void AddEdge(AdjList& adj, DataType from, DataType to) {
  if (from == to) return;
  adj[from.Code()].push_back(to);
}

// Build the BFS graph from the backend's native `pto.tcvt` pair list. The table
// itself lives on the BackendHandler (see pass-context-config.md: passes never
// branch on the backend), so a new architecture ships its own table and this
// pass needs no change.
AdjList BuildAdj(const backend::TcvtAdjacency& table) {
  AdjList adj;
  for (const auto& edge : table.edges) {
    AddEdge(adj, edge.source, edge.destination);
  }
  return adj;
}

// Preferred same-width float bridge used when preferring "convert kind without
// changing width, then change width" paths among equal-length BFS results.
std::optional<DataType> SameWidthFloat(DataType dt) {
  if (dt.IsFloat()) return std::nullopt;
  switch (dt.GetBit()) {
    case 32:
      return DataType::FP32;
    case 16:
      return DataType::FP16;
    default:
      return std::nullopt;
  }
}

// Significand bits (including the implicit leading bit) and exponent bits for
// the float formats the cast tables use. Unknown floats return nullopt, which
// makes the narrowing check below conservative (it only rejects what it can
// prove).
std::optional<std::pair<int, int>> FloatFormat(DataType dt) {
  if (dt == DataType::FP32) return std::make_pair(24, 8);
  if (dt == DataType::FP16) return std::make_pair(11, 5);
  if (dt == DataType::BF16) return std::make_pair(8, 8);
  if (dt == DataType::FP8E4M3FN) return std::make_pair(4, 4);
  if (dt == DataType::FP8E5M2) return std::make_pair(3, 5);
  if (dt == DataType::FP4E2M1X2) return std::make_pair(2, 2);
  if (dt == DataType::FP4E1M2X2) return std::make_pair(3, 1);
  return std::nullopt;
}

// Value bits an integer type can hold (excluding the sign bit).
size_t IntValueBits(DataType dt) { return dt.IsSignedInt() ? dt.GetBit() - 1 : dt.GetBit(); }

// True when routing through `mid` on the way to `dst` provably discards values
// that a direct `src -> dst` conversion would have kept.
//
// Without this, the shortest-path search happily picks a chain that is shorter
// but lossy: on A5 there is no native UINT32 -> FP32, and BFS would otherwise
// route it through INT16, so an input of 40000 -- exactly representable in
// FP32 -- would come back as garbage. Only provable narrowing is rejected;
// anything this cannot reason about is left admissible so unfamiliar dtypes do
// not turn a working lowering into a hard failure.
bool NarrowsRelativeTo(DataType mid, DataType dst) {
  if (mid == dst) return false;
  // An integer bridge cannot carry a float destination's fractional values.
  if (mid.IsInt() && dst.IsFloat()) return true;
  if (mid.IsFloat() && dst.IsFloat()) {
    const auto m = FloatFormat(mid);
    const auto d = FloatFormat(dst);
    if (!m || !d) return false;
    return m->first < d->first || m->second < d->second;
  }
  if (mid.IsFloat() && dst.IsInt()) {
    const auto m = FloatFormat(mid);
    if (!m) return false;
    return static_cast<size_t>(m->first) < IntValueBits(dst);
  }
  if (mid.IsInt() && dst.IsInt()) {
    // An unsigned bridge drops a signed destination's negatives.
    if (mid.IsUnsignedInt() && dst.IsSignedInt()) return true;
    return IntValueBits(mid) < IntValueBits(dst);
  }
  return false;
}

// Cost for ranking equal-length BFS paths: lower is better. Favours edges that
// convert int→same-width float first, then float width changes.
int EdgePreferenceCost(DataType from, DataType to) {
  if (!from.IsFloat() && to.IsFloat() && from.GetBit() == to.GetBit()) {
    return 0;  // same-byte → float
  }
  if (from.IsFloat() && to.IsFloat()) {
    return 1;  // adjust byte width in float domain
  }
  return 2;
}

// All of these source formats fit exactly in BF16, which is the only native
// FP4 encoding source. Larger floats/integers can double-round through BF16;
// they must make that intermediate conversion explicit in the source program.
bool HasLosslessFp4EncodingBridge(DataType src) {
  return src.IsPackedFp4() || src == DataType::BF16 || src == DataType::FP8E4M3FN ||
         src == DataType::FP8E5M2 || src == DataType::HF8 || (src.IsInt() && src.GetBit() <= 8);
}

bool IsFp4FloatBridge(DataType dtype) {
  return dtype == DataType::BF16 || dtype == DataType::FP16 || dtype == DataType::FP32;
}

ExprPtr MakeCast(const ExprPtr& x, DataType to, int mode, const Span& span);

// BFS over typed candidate hops. Empty means no admissible native chain.
std::vector<DataType> FindCastChain(const AdjList& adj, const backend::BackendHandler& handler,
                                    const ExprPtr& source, DataType to, int mode, const Span& span) {
  const auto source_type = As<TileType>(source->GetType());
  INTERNAL_CHECK_SPAN(source_type, span) << "cast search requires a TileType source";
  const DataType from = source_type->dtype_;
  if (from == to) return {};
  if (to.IsPackedFp4() && !HasLosslessFp4EncodingBridge(from)) return {};

  // State: dtype code → (parent code, edge-to dtype, path_len, path_pref_cost)
  struct NodeInfo {
    uint8_t parent = 0;
    DataType via = DataType::BOOL;  // dtype of this node
    int dist = -1;
    int pref = 0;
    ExprPtr value;  // Typed candidate hop; search never emits these calls.
  };
  std::array<NodeInfo, 256> info{};
  std::queue<uint8_t> q;

  info[from.Code()] = NodeInfo{from.Code(), from, 0, 0, source};
  q.push(from.Code());

  while (!q.empty()) {
    uint8_t cur = q.front();
    q.pop();
    const NodeInfo& cur_info = info[cur];
    auto it = adj.find(cur);
    if (it == adj.end()) continue;

    // Prefer same-width float neighbor first when expanding (stable among
    // equal BFS depths via preference cost).
    std::vector<DataType> neigh = it->second;
    if (auto sw = SameWidthFloat(cur_info.via)) {
      auto sw_it = std::find(neigh.begin(), neigh.end(), *sw);
      if (sw_it != neigh.end()) {
        std::iter_swap(neigh.begin(), sw_it);
      }
    }

    for (const DataType& nxt : neigh) {
      if (to.IsPackedFp4() && nxt != to && !IsFp4FloatBridge(nxt)) continue;
      // Intermediates must preserve everything the destination can represent;
      // the destination itself is always admissible.
      // FP4 has a much smaller value set than wider destinations: its native
      // BF16 decode and FP16/FP32/FP8 bridges preserve every source value
      // exactly. Conversely, BF16 -> FP4 -> FP8 is a lossy shortcut even though
      // both hops can individually be native, so FloatFormat models FP4/FP8. Rejecting BF16 merely because
      // FP32 has more precision would make an otherwise lossless FP4 -> BF16 -> FP32 cast unreachable.
      const bool exact_fp4_decode =
          from.IsPackedFp4() &&
          (IsFp4FloatBridge(nxt) || nxt == DataType::FP8E4M3FN || nxt == DataType::FP8E5M2);
      if (nxt != to && !exact_fp4_decode && NarrowsRelativeTo(nxt, to)) continue;
      // Geometry changes with packed/unpacked units. Validate every concrete
      // candidate hop against the backend rule, not just the original cast.
      // All cast paths preserve layout and logical value geometry, so a dtype
      // uniquely determines its carrier geometry for this search.
      ExprPtr candidate;
      try {
        candidate = MakeCast(cur_info.value, nxt, mode, span);
      } catch (const ValueError&) {
        continue;  // This candidate's dtype/packing contract is not admissible.
      } catch (const TypeError&) {
        continue;
      }
      if (!handler.IsDirectCast(As<Call>(candidate))) continue;
      const int edge_cost = EdgePreferenceCost(cur_info.via, nxt);
      const int new_dist = cur_info.dist + 1;
      const int new_pref = cur_info.pref + edge_cost;
      NodeInfo& nxt_info = info[nxt.Code()];
      if (nxt_info.dist < 0) {
        nxt_info = NodeInfo{cur, nxt, new_dist, new_pref, candidate};
        q.push(nxt.Code());
      } else if (nxt_info.dist == new_dist && new_pref < nxt_info.pref) {
        nxt_info.parent = cur;
        nxt_info.via = nxt;
        nxt_info.pref = new_pref;
        nxt_info.value = candidate;
      }
    }
  }

  const NodeInfo& goal = info[to.Code()];
  if (goal.dist < 0) {
    return {};
  }

  std::vector<DataType> rev;
  for (uint8_t c = to.Code(); c != from.Code(); c = info[c].parent) {
    rev.push_back(info[c].via);
  }
  std::reverse(rev.begin(), rev.end());
  return rev;
}

ExprPtr MakeCast(const ExprPtr& x, DataType to, int mode, const Span& span) {
  std::vector<std::pair<std::string, std::any>> kw = {{"target_type", to}, {"mode", mode}};
  return OpRegistry::GetInstance().Create("tile.cast", {x}, kw, span);
}

bool IsFp8DataCastTarget(DataType dst) { return dst == DataType::FP8E4M3FN || dst == DataType::FP8E5M2; }

/// Report only fallback chains; a supported native FP4 -> FP8 cast is silent.
void MaybeWarnPackedFp4ToFp8(DataType src, DataType dst, const Span& span, std::vector<Diagnostic>* diags) {
  if (!diags || !src.IsPackedFp4() || !IsFp8DataCastTarget(dst)) return;
  diags->emplace_back(DiagnosticSeverity::Warning, "LegalizeTileCast", 0,
                      "cast from " + src.ToString() + " to " + dst.ToString() +
                          " expands to FP4→BF16→FP32→FP8, which is slow and memory-heavy; prefer a LUT "
                          "gather for better performance and lower memory use",
                      span);
}

class LegalizeTileCastMutator : public IRMutator {
 public:
  LegalizeTileCastMutator(const backend::BackendHandler& handler, std::vector<Diagnostic>* diags)
      : handler_(handler),
        arch_name_(handler.GetPtoTargetArch()),
        adj_(BuildAdj(handler.GetTcvtAdjacency())),
        diags_(diags) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call || !IsOp(call, "tile.cast")) {
      return IRMutator::VisitStmt_(op);
    }
    if (call->args_.empty()) {
      return IRMutator::VisitStmt_(op);
    }

    auto src_tile = As<TileType>(call->args_[0]->GetType());
    INTERNAL_CHECK_SPAN(src_tile, op->span_) << "tile.cast input must be TileType";
    const auto result_tile = As<TileType>(call->GetType());
    INTERNAL_CHECK_SPAN(result_tile, op->span_) << "tile.cast result must be TileType";
    DataType src = src_tile->dtype_;
    DataType dst = call->GetKwarg<DataType>("target_type");
    const int mode = call->GetKwarg<int>("mode", kCastModeRound);

    if (handler_.IsDirectCast(call)) {
      return IRMutator::VisitStmt_(op);
    }

    MaybeWarnPackedFp4ToFp8(src, dst, op->span_, diags_);
    std::vector<DataType> chain = FindCastChain(adj_, handler_, call->args_[0], dst, mode, op->span_);
    CHECK_SPAN(!chain.empty() || !dst.IsPackedFp4() || HasLosslessFp4EncodingBridge(src), op->span_)
        << "LegalizeTileCast: no lossless cast path from " << src.ToString() << " to " << dst.ToString()
        << "; the required BF16 intermediate can double-round. Cast to BF16 explicitly before FP4 "
           "to request that two-step conversion";
    CHECK_SPAN(!chain.empty(), op->span_)
        << "LegalizeTileCast: no native cast path from " << src.ToString() << " to " << dst.ToString()
        << " for arch " << arch_name_ << "; pto.tcvt does not support this conversion";

    // Intermediate hops use the original mode (matches model-side INT32→FP32→FP16
    // chains where the narrow step carries mode="round"). Final hop also keeps it.
    //
    // Everything else the author wrote — the optional destination
    // `saturation_mode`, an explicit `tmp` scratch operand — rides on the final
    // hop only, because the final hop is the one that actually reaches the
    // destination dtype. Saturation names the *destination* range, so stamping
    // it on an intermediate would clamp to a range the author never asked
    // about. Deferring is also safe: BFS already refuses any intermediate that
    // narrows relative to the destination (NarrowsRelativeTo above), so every
    // value the destination *can* represent reaches the final hop exactly. A
    // value it cannot is out of range at both ends of the chain -- an
    // intermediate float may overflow it to an infinity, but with its sign
    // intact, so it clamps to the same endpoint either way.
    ExprPtr cur = VisitExpr(call->args_[0]);
    std::vector<StmtPtr> stmts;
    stmts.reserve(chain.size());

    for (size_t i = 0; i + 1 < chain.size(); ++i) {
      ExprPtr cast_expr = MakeCast(cur, chain[i], mode, op->span_);
      if (result_tile->memory_space_) {
        // The late run follows memory inference and vector splitting. Fresh
        // intermediate storage inherits the resolved placement, not the old
        // result's MemRef. Early legalization leaves placement unresolved.
        const auto hop = As<Call>(cast_expr);
        const auto hop_type = As<TileType>(cast_expr->GetType());
        INTERNAL_CHECK_SPAN(hop && hop_type, op->span_)
            << "Internal error: native cast bridge must produce a tile Call";
        const auto placed_type = std::make_shared<TileType>(hop_type->shape_, hop_type->dtype_, std::nullopt,
                                                            hop_type->tile_view_, result_tile->memory_space_);
        cast_expr =
            std::make_shared<Call>(hop->op_, hop->args_, hop->kwargs_, hop->attrs_, placed_type, hop->span_);
      }
      const std::string name =
          auto_name::BuildName(auto_name::GetBaseName(op->var_->name_hint_), "cast_" + chain[i].ToString(),
                               "tmp", static_cast<int>(temp_counter_++));
      auto mid_var = std::make_shared<Var>(name, cast_expr->GetType(), op->span_);
      stmts.push_back(std::make_shared<AssignStmt>(mid_var, cast_expr, op->span_));
      cur = mid_var;
    }

    // The final hop *is* the caller's cast, retargeted onto the last chain dtype
    // (identical to it when the cast was already single-hop), so it carries the
    // original kwargs and any operand past the source.
    auto final_kwargs = call->kwargs_;
    for (auto& [key, value] : final_kwargs) {
      if (key == "target_type") value = chain.back();
    }
    std::vector<ExprPtr> final_args{cur};
    for (size_t i = 1; i < call->args_.size(); ++i) {
      final_args.push_back(VisitExpr(call->args_[i]));
    }
    auto final_assign = MutableCopy(op);
    // Keep final storage/layout and call-site attrs (including the original
    // tmp's argument directions). Intermediate calls carry their own schema.
    final_assign->value_ = std::make_shared<Call>(call->op_, std::move(final_args), std::move(final_kwargs),
                                                  call->attrs_, call->GetType(), call->span_);
    stmts.push_back(std::move(final_assign));

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

 private:
  const backend::BackendHandler& handler_;
  std::string arch_name_;
  AdjList adj_;
  std::size_t temp_counter_ = 0;
  std::vector<Diagnostic>* diags_;
};

FunctionPtr TransformLegalizeTileCast(const FunctionPtr& func) {
  if (!func) return func;
  // Tile casts only live in InCore (and AIC/AIV after expansion). Skip host orch.
  if (func->level_.has_value() && *func->level_ == Level::HOST) {
    return func;
  }
  // The native-cast table is a backend fact, so without a configured backend
  // there is nothing to legalize against -- leave the IR untouched rather than
  // guess a profile (several codegen tests drive passes with no backend set).
  // Both lookups below CHECK-fail when unconfigured, so probe first.
  if (!backend::BackendConfig::IsConfigured()) {
    return func;
  }
  const auto* ctx = PassContext::Current();
  const backend::BackendHandler* handler =
      ctx != nullptr ? ctx->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
  if (handler == nullptr) {
    return func;
  }
  std::vector<Diagnostic> diags;
  LegalizeTileCastMutator mutator(*handler, &diags);
  auto result = mutator.VisitFunction(func);
  EmitDiagnostics(diags, "LegalizeTileCast");
  return result;
}

}  // namespace

namespace pass {

Pass LegalizeTileCast() {
  return CreateFunctionPass(TransformLegalizeTileCast, "LegalizeTileCast", kLegalizeTileCastProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
