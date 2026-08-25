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
  for (const auto& [from, to] : table.edges) {
    AddEdge(adj, from, to);
  }
  return adj;
}

bool IsNativeCast(const AdjList& adj, DataType from, DataType to) {
  if (from == to) return false;
  auto it = adj.find(from.Code());
  if (it == adj.end()) return false;
  for (const DataType& d : it->second) {
    if (d == to) return true;
  }
  return false;
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

// BFS shortest path; returns the sequence of intermediate/final target types
// (excluding `from`). Empty vector means already native? No — caller checks
// native first. Empty here means unreachable.
std::vector<DataType> FindCastChain(const AdjList& adj, DataType from, DataType to) {
  if (from == to) return {};
  if (IsNativeCast(adj, from, to)) {
    return {to};
  }

  // State: dtype code → (parent code, edge-to dtype, path_len, path_pref_cost)
  struct NodeInfo {
    uint8_t parent = 0;
    DataType via = DataType::BOOL;  // dtype of this node
    int dist = -1;
    int pref = 0;
  };
  std::array<NodeInfo, 256> info{};
  std::queue<uint8_t> q;

  info[from.Code()] = NodeInfo{from.Code(), from, 0, 0};
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
      // Intermediates must preserve everything the destination can represent;
      // the destination itself is always admissible.
      if (nxt != to && NarrowsRelativeTo(nxt, to)) continue;
      const int edge_cost = EdgePreferenceCost(cur_info.via, nxt);
      const int new_dist = cur_info.dist + 1;
      const int new_pref = cur_info.pref + edge_cost;
      NodeInfo& nxt_info = info[nxt.Code()];
      if (nxt_info.dist < 0) {
        nxt_info = NodeInfo{cur, nxt, new_dist, new_pref};
        q.push(nxt.Code());
      } else if (nxt_info.dist == new_dist && new_pref < nxt_info.pref) {
        nxt_info.parent = cur;
        nxt_info.via = nxt;
        nxt_info.pref = new_pref;
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

class LegalizeTileCastMutator : public IRMutator {
 public:
  LegalizeTileCastMutator(const backend::TcvtAdjacency& table, std::string arch_name)
      : arch_name_(std::move(arch_name)), adj_(BuildAdj(table)) {}

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
    DataType src = src_tile->dtype_;
    DataType dst = call->GetKwarg<DataType>("target_type");
    const int mode = call->GetKwarg<int>("mode", kCastModeRound);

    if (IsNativeCast(adj_, src, dst)) {
      return IRMutator::VisitStmt_(op);
    }

    std::vector<DataType> chain = FindCastChain(adj_, src, dst);
    CHECK_SPAN(!chain.empty(), op->span_)
        << "LegalizeTileCast: no native cast path from " << src.ToString() << " to " << dst.ToString()
        << " for arch " << arch_name_ << "; pto.tcvt does not support this conversion";

    // Intermediate hops use the original mode (matches model-side INT32→FP32→FP16
    // chains where the narrow step carries mode="round"). Final hop also keeps it.
    ExprPtr cur = VisitExpr(call->args_[0]);
    std::vector<StmtPtr> stmts;
    stmts.reserve(chain.size());

    for (size_t i = 0; i + 1 < chain.size(); ++i) {
      ExprPtr cast_expr = MakeCast(cur, chain[i], mode, op->span_);
      const std::string name =
          auto_name::BuildName(auto_name::GetBaseName(op->var_->name_hint_), "cast_" + chain[i].ToString(),
                               "tmp", static_cast<int>(temp_counter_++));
      auto mid_var = std::make_shared<Var>(name, cast_expr->GetType(), op->span_);
      stmts.push_back(std::make_shared<AssignStmt>(mid_var, cast_expr, op->span_));
      cur = mid_var;
    }

    auto final_assign = MutableCopy(op);
    final_assign->value_ = MakeCast(cur, chain.back(), mode, op->span_);
    stmts.push_back(std::move(final_assign));

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

 private:
  std::string arch_name_;
  AdjList adj_;
  std::size_t temp_counter_ = 0;
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
  LegalizeTileCastMutator mutator(handler->GetTcvtAdjacency(), handler->GetPtoTargetArch());
  return mutator.VisitFunction(func);
}

}  // namespace

namespace pass {

Pass LegalizeTileCast() {
  return CreateFunctionPass(TransformLegalizeTileCast, "LegalizeTileCast", kLegalizeTileCastProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
