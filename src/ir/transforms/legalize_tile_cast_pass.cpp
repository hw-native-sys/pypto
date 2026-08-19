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
 * After each hop is native, A2/A3 also materializes the optional pto.tcvt tmp
 * required by PTOAS when PlanMemory is skipped (non-saturating narrowing:
 * FP32→i16, FP16→i16/i8). This runs here rather than in FlattenTileNdTo2D so
 * legalized multi-hop casts (e.g. FP32→INT8 → FP32→FP16→INT8) still get a
 * scratch on the hop that needs it.
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
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

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

ExprPtr MakeCast(const ExprPtr& x, DataType to, int mode, const Span& span, const ExprPtr& tmp = nullptr) {
  std::vector<std::pair<std::string, std::any>> kw = {{"target_type", to}, {"mode", mode}};
  std::vector<ExprPtr> args = {x};
  if (tmp) args.push_back(tmp);
  return OpRegistry::GetInstance().Create("tile.cast", args, kw, span);
}

// PTOAS v0.58 TCvtOp: non-saturating (sat_mode=OFF, the default) narrowing on
// A2/A3 needs an explicit tmp when PlanMemory is skipped. Matches
// tcvtNeedsTmp / makeTCvtTmpType in PTOMaterializeImplicitTmp.cpp.
bool TcvtNeedsA2A3Scratch(DataType src, DataType dst) {
  if (src == DataType::FP32 && dst == DataType::INT16) return true;
  // PTOAS uses MLIR Type::isInteger(8/16), which matches signed and unsigned.
  if (src == DataType::FP16 && (dst == DataType::INT16 || dst == DataType::INT8 || dst == DataType::UINT8)) {
    return true;
  }
  return false;
}

int64_t CeilDivI64(int64_t num, int64_t den) { return (num + den - 1) / den; }

int64_t TcvtScratchCapacityBytes(const TileTypePtr& src_tile, DataType dst) {
  const auto src_shape = src_tile->shape_;
  const auto dst_valid = GetValidShape(src_tile);
  INTERNAL_CHECK(src_shape.size() == 2 && dst_valid.size() == 2)
      << "LegalizeTileCast: tcvt scratch requires a 2D tile";
  auto rows_ci = As<ConstInt>(dst_valid[0]);
  auto cols_ci = As<ConstInt>(dst_valid[1]);
  auto src_cols_ci = As<ConstInt>(src_shape[1]);
  CHECK(rows_ci && cols_ci && src_cols_ci)
      << "LegalizeTileCast: A2/A3 non-saturating narrowing tcvt requires a static "
         "src shape and dst valid_shape to size the PTOAS scratch tile";
  const int64_t rows = rows_ci->value_;
  const int64_t cols = cols_ci->value_;
  const int64_t src_cols = src_cols_ci->value_;
  int64_t bytes = 0;
  if (src_tile->dtype_ == DataType::FP32) {
    if (rows > 0 && cols > 0) {
      const int64_t head = int64_t{4} * 64 * std::min<int64_t>(cols / 64, 255);
      const int64_t remainder = cols % 64;
      const int64_t tail = remainder == 0
                               ? 0
                               : int64_t{32} * ((std::min<int64_t>(rows, 255) - 1) * (src_cols / 8) +
                                                CeilDivI64(remainder, 8));
      bytes = std::max(head, tail);
    }
  } else if (src_tile->dtype_ == DataType::FP16 && cols > 0) {
    const int64_t width = std::min<int64_t>(cols, 64);
    const int64_t half_to_i16 = 32 * CeilDivI64(width, 8);
    const int64_t half_to_i8 = std::max(half_to_i16, 128 + 32 * CeilDivI64(width, 16));
    bytes = (dst == DataType::INT8 || dst == DataType::UINT8) ? half_to_i8 : half_to_i16;
  }
  return std::max<int64_t>(32, CeilDivI64(bytes, 32) * 32);
}

ExprPtr MakeScratchShapeTuple(int64_t bytes, const Span& span) {
  std::vector<ExprPtr> elems = {std::make_shared<ConstInt>(1, DataType::INDEX, span),
                                std::make_shared<ConstInt>(bytes, DataType::INDEX, span)};
  return std::make_shared<MakeTuple>(elems, span);
}

class LegalizeTileCastMutator : public IRMutator {
 public:
  LegalizeTileCastMutator(const backend::TcvtAdjacency& table, std::string arch_name,
                          bool materialize_scratch)
      : arch_name_(std::move(arch_name)), adj_(BuildAdj(table)), materialize_scratch_(materialize_scratch) {}

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
      if (call->args_.size() == 1 && materialize_scratch_ && TcvtNeedsA2A3Scratch(src, dst)) {
        return MaterializeCastWithScratch(op, call, dst, mode);
      }
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
    stmts.reserve(chain.size() * 2);

    for (size_t i = 0; i + 1 < chain.size(); ++i) {
      AppendCastHop(stmts, cur, chain[i], mode, op, /*final_assign=*/nullptr);
    }
    AppendCastHop(stmts, cur, chain.back(), mode, op, op);

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

 private:
  void AppendCastHop(std::vector<StmtPtr>& stmts, ExprPtr& cur, DataType hop_dst, int mode,
                     const AssignStmtPtr& origin, const AssignStmtPtr& final_assign) {
    auto src_tile = As<TileType>(cur->GetType());
    INTERNAL_CHECK_SPAN(src_tile, origin->span_) << "tile.cast hop input must be TileType";
    ExprPtr tmp;
    if (materialize_scratch_ && TcvtNeedsA2A3Scratch(src_tile->dtype_, hop_dst)) {
      tmp = CreateScratch(stmts, src_tile, hop_dst, origin);
    }
    ExprPtr cast_expr = MakeCast(cur, hop_dst, mode, origin->span_, tmp);
    if (final_assign) {
      auto assign = MutableCopy(final_assign);
      assign->value_ = cast_expr;
      stmts.push_back(std::move(assign));
      return;
    }
    const std::string name =
        auto_name::BuildName(auto_name::GetBaseName(origin->var_->name_hint_), "cast_" + hop_dst.ToString(),
                             "tmp", static_cast<int>(temp_counter_++));
    auto mid_var = std::make_shared<Var>(name, cast_expr->GetType(), origin->span_);
    stmts.push_back(std::make_shared<AssignStmt>(mid_var, cast_expr, origin->span_));
    cur = mid_var;
  }

  ExprPtr CreateScratch(std::vector<StmtPtr>& stmts, const TileTypePtr& src_tile, DataType dst,
                        const AssignStmtPtr& origin) {
    const int64_t bytes = TcvtScratchCapacityBytes(src_tile, dst);
    const Span& span = origin->span_;
    std::vector<std::pair<std::string, std::any>> tmp_kwargs = {
        {"dtype", DataType::INT8},
        {"target_memory", MemorySpace::Vec},
    };
    auto tmp_create = OpRegistry::GetInstance().Create("tile.create", {MakeScratchShapeTuple(bytes, span)},
                                                       tmp_kwargs, span);
    const std::string tmp_name = auto_name::BuildName(auto_name::GetBaseName(origin->var_->name_hint_),
                                                      "tcvt", "tmp", static_cast<int>(scratch_counter_++));
    auto tmp_var = std::make_shared<Var>(tmp_name, tmp_create->GetType(), span);
    stmts.push_back(std::make_shared<AssignStmt>(tmp_var, tmp_create, span));
    return tmp_var;
  }

  StmtPtr MaterializeCastWithScratch(const AssignStmtPtr& op, const CallPtr& call, DataType dst, int mode) {
    std::vector<StmtPtr> stmts;
    ExprPtr src = VisitExpr(call->args_[0]);
    auto src_after = As<TileType>(src->GetType());
    INTERNAL_CHECK_SPAN(src_after, op->span_) << "tile.cast input must be TileType";
    ExprPtr tmp = CreateScratch(stmts, src_after, dst, op);
    auto assign = MutableCopy(op);
    assign->value_ = MakeCast(src, dst, mode, op->span_, tmp);
    stmts.push_back(std::move(assign));
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

  std::string arch_name_;
  AdjList adj_;
  bool materialize_scratch_ = false;
  std::size_t temp_counter_ = 0;
  std::size_t scratch_counter_ = 0;
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
  const bool materialize_scratch = handler->GetPtoTargetArch() == "a2a3";
  LegalizeTileCastMutator mutator(handler->GetTcvtAdjacency(), handler->GetPtoTargetArch(),
                                  materialize_scratch);
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
