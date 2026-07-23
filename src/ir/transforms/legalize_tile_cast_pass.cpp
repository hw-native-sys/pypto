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
 * BFS over the ISA-supported adjacency table (A5 / A2A3). Typical outcome for
 * A5 INT32→FP16 is INT32→FP32→FP16 — same byte-width to float, then resize —
 * which does not introduce extra precision loss beyond the final narrow.
 */

#include <algorithm>
#include <any>
#include <array>
#include <cstddef>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

enum class CastArch { A2A3, A5 };

CastArch ResolveCastArch() {
  const auto* ctx = PassContext::Current();
  INTERNAL_CHECK(ctx) << "LegalizeTileCast requires an active PassContext";
  const auto* handler = ctx->GetBackendHandler();
  INTERNAL_CHECK(handler) << "LegalizeTileCast requires a configured BackendHandler";
  const std::string arch = handler->GetPtoTargetArch();
  if (arch == "a5") {
    return CastArch::A5;
  }
  // Default "a2a3" (and any unknown arch treated as a2a3 until a new profile
  // ships its own adjacency table).
  return CastArch::A2A3;
}

using AdjList = std::unordered_map<uint8_t, std::vector<DataType>>;

void AddEdge(AdjList& adj, DataType from, DataType to) {
  if (from == to) return;
  adj[from.Code()].push_back(to);
}

// ISA Supported Conversions from pto-isa tcvt docs (A2A3 / A5 columns).
AdjList BuildAdj(CastArch arch) {
  AdjList adj;

  // FP32
  AddEdge(adj, DataType::FP32, DataType::FP16);
  AddEdge(adj, DataType::FP32, DataType::BF16);
  AddEdge(adj, DataType::FP32, DataType::INT16);
  AddEdge(adj, DataType::FP32, DataType::INT32);
  AddEdge(adj, DataType::FP32, DataType::INT64);
  if (arch == CastArch::A5) {
    AddEdge(adj, DataType::FP32, DataType::FP8E4M3FN);
    AddEdge(adj, DataType::FP32, DataType::FP8E5M2);
    AddEdge(adj, DataType::FP32, DataType::HF8);
  }

  // FP16
  AddEdge(adj, DataType::FP16, DataType::FP32);
  AddEdge(adj, DataType::FP16, DataType::INT32);
  AddEdge(adj, DataType::FP16, DataType::INT16);
  AddEdge(adj, DataType::FP16, DataType::INT8);
  AddEdge(adj, DataType::FP16, DataType::UINT8);
  if (arch == CastArch::A2A3) {
    AddEdge(adj, DataType::FP16, DataType::INT4);
  } else {
    AddEdge(adj, DataType::FP16, DataType::HF8);
  }

  // BF16
  AddEdge(adj, DataType::BF16, DataType::FP32);
  AddEdge(adj, DataType::BF16, DataType::INT32);
  if (arch == CastArch::A5) {
    AddEdge(adj, DataType::BF16, DataType::FP16);
    AddEdge(adj, DataType::BF16, DataType::FP4);
  }

  // I16
  AddEdge(adj, DataType::INT16, DataType::FP16);
  AddEdge(adj, DataType::INT16, DataType::FP32);
  if (arch == CastArch::A5) {
    AddEdge(adj, DataType::INT16, DataType::UINT8);
    AddEdge(adj, DataType::INT16, DataType::UINT32);
    AddEdge(adj, DataType::INT16, DataType::INT32);
  }

  // I32
  AddEdge(adj, DataType::INT32, DataType::FP32);
  AddEdge(adj, DataType::INT32, DataType::INT16);
  AddEdge(adj, DataType::INT32, DataType::INT64);
  if (arch == CastArch::A2A3) {
    AddEdge(adj, DataType::INT32, DataType::FP16);  // deq path
  } else {
    AddEdge(adj, DataType::INT32, DataType::UINT16);
    AddEdge(adj, DataType::INT32, DataType::UINT8);
  }

  // I64
  AddEdge(adj, DataType::INT64, DataType::FP32);
  AddEdge(adj, DataType::INT64, DataType::INT32);

  // U8
  AddEdge(adj, DataType::UINT8, DataType::FP16);
  if (arch == CastArch::A5) {
    AddEdge(adj, DataType::UINT8, DataType::UINT16);
  }

  // I8
  AddEdge(adj, DataType::INT8, DataType::FP16);
  if (arch == CastArch::A5) {
    AddEdge(adj, DataType::INT8, DataType::INT16);
    AddEdge(adj, DataType::INT8, DataType::INT32);
  }

  // S4 (A2A3 only)
  if (arch == CastArch::A2A3) {
    AddEdge(adj, DataType::INT4, DataType::FP16);
  }

  // A5-only sources
  if (arch == CastArch::A5) {
    AddEdge(adj, DataType::UINT32, DataType::UINT8);
    AddEdge(adj, DataType::UINT32, DataType::UINT16);
    AddEdge(adj, DataType::UINT32, DataType::INT16);
    AddEdge(adj, DataType::FP8E4M3FN, DataType::FP32);
    AddEdge(adj, DataType::FP8E5M2, DataType::FP32);
    AddEdge(adj, DataType::HF8, DataType::FP32);
    AddEdge(adj, DataType::FP4, DataType::BF16);
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
  explicit LegalizeTileCastMutator(CastArch arch) : arch_(arch), adj_(BuildAdj(arch)) {}

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
        << " for arch " << (arch_ == CastArch::A5 ? "a5" : "a2a3")
        << "; pto.tcvt does not support this conversion";

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
  CastArch arch_;
  AdjList adj_;
  std::size_t temp_counter_ = 0;
};

FunctionPtr TransformLegalizeTileCast(const FunctionPtr& func) {
  if (!func) return func;
  // Tile casts only live in InCore (and AIC/AIV after expansion). Skip host orch.
  if (func->level_.has_value() && *func->level_ == Level::HOST) {
    return func;
  }
  LegalizeTileCastMutator mutator(ResolveCastArch());
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
