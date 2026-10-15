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
#include <cstdlib>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
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

bool IsFp8DataCastTarget(DataType dst) { return dst == DataType::FP8E4M3FN || dst == DataType::FP8E5M2; }

/// FP4 / FP4E2M1X2 → FP8* is allowed but discouraged (prefer LUT gather or host precast).
void MaybeWarnFp4FamilyToFp8(DataType src, DataType dst, const Span& span, std::vector<Diagnostic>* diags) {
  if (!diags || !src.IsFp4Family() || !IsFp8DataCastTarget(dst)) return;
  diags->emplace_back(DiagnosticSeverity::Warning, "LegalizeTileCast", 0,
                      "cast from " + src.ToString() + " to " + dst.ToString() +
                          " expands to FP4→BF16→FP32→FP8, which is slow and memory-heavy; prefer a LUT "
                          "gather for better performance and lower memory use",
                      span);
}

// ============================================================================
// FP4 -> FP8 identity bit-op chain (single pass, no FP32 intermediate)
// ============================================================================
//
// E2M1 embeds exactly into E4M3: the exponent field shifts by +6 (bias 1 ->
// bias 7) and the mantissa by +2 (1 -> 3 bits), except the two subnormal
// E2M1 codes, which normalize: code 0 (+-0) maps to 0x00/0x80 and code 1
// (+-0.5, E=0 M=1) maps to 0x30/0xB0. Per nibble c (s = c&8, c7 = c&7):
//
//   f8(c) = s << 7 | (48 + 8*E + 4*M - 4M*[E==0] - 48*[E==0 && M==0])
//
// with E = c7 >> 1, M = c7 & 1 -- verified against all 16 codes
// (0x00,0x30,0x38,0x3C,0x40,0x44,0x48,0x4C and their 0x80-sign mirrors).
// Applied to a UINT16 view of the packed carrier byte, both nibbles' results
// combine into the little-endian FP8 pair, so no nibble expansion is needed:
//
//   u8   = reinterpret_view(fp4, UINT8)        # [R, C/2] carriers
//   b    = cast(u8, UINT16)                    # zero-extended packed byte
//   out  = f8(b & 15) | f8(b >> 4) << 8        # elementwise, ~30 vec ops
//   res  = reinterpret_view(out, FP8E4M3FN)    # [R, C] bytes
//
// All products in the formula are gated by 0/1 bits, so they lower to masked
// ANDs (mask = 0 - bit in UINT16 wraparound) and the chain needs no multiply.
namespace {

/// t12: PYPTO_FP4_CAST_NZ_DIRECT gates the NZ direct-write cast emission.
/// Read once per process (the pass runs per function).
bool Fp4CastNzDirectEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("PYPTO_FP4_CAST_NZ_DIRECT");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

}  // namespace

ExprPtr MakeUintConst(int64_t value, DataType dtype, const Span& span) {
  return std::make_shared<ConstInt>(value, dtype, span);
}

/// One `var = op(args)` with an auto-named temp; returns the var as an expr.
ExprPtr EmitTileOp(const std::string& op, const std::vector<ExprPtr>& args,
                   const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& var_name,
                   const Span& span, std::vector<StmtPtr>& stmts) {
  auto call = OpRegistry::GetInstance().Create(op, args, kwargs, span);
  auto var = std::make_shared<Var>(var_name, call->GetType(), span);
  stmts.push_back(std::make_shared<AssignStmt>(var, call, span));
  return var;
}

class LegalizeTileCastMutator : public IRMutator {
 public:
  LegalizeTileCastMutator(const backend::TcvtAdjacency& table, std::string arch_name,
                          std::vector<Diagnostic>* diags, bool fp4_to_fp8_bitops,
                          std::unordered_set<const Var*> kn_rhs_consumers)
      : arch_name_(std::move(arch_name)),
        adj_(BuildAdj(table)),
        diags_(diags),
        fp4_to_fp8_bitops_(fp4_to_fp8_bitops),
        kn_rhs_consumers_(std::move(kn_rhs_consumers)) {}

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

    // Single-pass FP4 -> FP8 via kernel-resident LUT + tgather (t2 route A1,
    // DeepGEMM vf_build_lut precedent): build a 256-entry u16 table once per
    // function, then each cast site is 3 real ops (TCVT u8->u16 zero-extend,
    // dummy VAND for the A5-unused tmp, TGATHER) instead of the three TCVT
    // hops through BF16/FP32. UB traffic ~4C bytes vs the 3-hop chain's
    // ~13.5C (t12 measured the bit-op variant at 67B/elem; this is ~4B/elem).
    // Codex P1 #4 (PR#2987 round 2): the gather route's zero-copy carrier
    // reinterprets (FP4->U8 head, U16->FP8 pair tail) only materialize as
    // no-op aliases when PyPTO bakes tile addresses. Under
    // MemoryPlanner.PtoAS (emit_tile_addr=false) every reinterpret falls to
    // pto.treshape, which PTOAS rejects for the FP4 source dtype — the
    // whole route fails to assemble. Fall back to the bridged native-hop
    // chain there (pure tcvt ops, planner-agnostic).
    const bool ptoas_planner = [] {
      const auto* ctx = PassContext::Current();
      return ctx != nullptr && ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS;
    }();
    if (fp4_to_fp8_bitops_ && !ptoas_planner && src.IsFp4Family() && dst == DataType::FP8E4M3FN) {
      return RewriteFp4ToFp8Gather(op, call, mode);
    }

    MaybeWarnFp4FamilyToFp8(src, dst, op->span_, diags_);

    if (IsNativeCast(adj_, src, dst)) {
      return IRMutator::VisitStmt_(op);
    }

    std::vector<DataType> chain = FindCastChain(adj_, src, dst);
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
    final_assign->value_ = OpRegistry::GetInstance().Create("tile.cast", final_args, final_kwargs, op->span_);
    stmts.push_back(std::move(final_assign));

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

 private:
  /// Build the 256-entry u16 LUT once per function: LUT[i] = f8(i&15) |
  /// f8(i>>4)<<8. The construction fills `lut_prelude_` (NOT the caller's
  /// stmts) so that VisitStmt_(InCoreScopeStmt) can hoist it to the CORE
  /// scope's body top — emitting it inline at the first cast site would
  /// rebuild the table on every pipeline iteration for non-peeled
  /// structures (t23 finding: cast directly inside pl.pipeline without
  /// chunk0 peeling puts tile.ci at loop depth 1).
  void BuildLutTable(const Span& span) {
    std::vector<StmtPtr>& stmts = lut_prelude_;
    // Codex P1 #3 (PR#2987 round 2): do NOT clear a pending prelude here.
    // A second build happens after a branch guard restored lut_table_ to
    // null (loop { if { first cast } ... cast after the if }); the pending
    // definitions from the first build are still referenced by the casts
    // inside the closed branch and are hoisted (consumed) only at the
    // enclosing scope/loop exit — clearing here would delete still-live
    // table definitions. Each build appends its own definitions; every
    // hoist site moves and clears the whole accumulated prelude.
    if (lut_prelude_.empty() && prelude_pending_depth_ < 0) {
      prelude_pending_depth_ = loop_depth_;
    }
    auto name = [&](const char* tag) {
      return auto_name::BuildName("fp4_lut", tag, "tbl", static_cast<int>(temp_counter_++));
    };
    // Scalar dtype: bitwise ops on u16 tiles want the signless same-width
    // encoding (INT16); shift/add scalars take any integer (INT32).
    auto u16_scalar = [&](int64_t v) { return MakeUintConst(v, DataType::INT16, span); };
    auto i32_scalar = [&](int64_t v) { return MakeUintConst(v, DataType::INT32, span); };
    auto unop = [&](const char* op, const ExprPtr& x, int64_t s, bool bitwise, const char* tag) {
      return EmitTileOp(op, {x, bitwise ? u16_scalar(s) : i32_scalar(s)}, {}, name(tag), span, stmts);
    };
    auto binop = [&](const char* op, const ExprPtr& lhs, const ExprPtr& rhs, const char* tag) {
      return EmitTileOp(op, {lhs, rhs}, {}, name(tag), span, stmts);
    };

    // ci(0, [1, 256], dtype=UINT16): the byte-value ramp 0..255 in u16 lanes.
    auto start = std::make_shared<ConstInt>(0, DataType::UINT16, span);
    auto shape = std::make_shared<MakeTuple>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span),
                             std::make_shared<ConstInt>(256, DataType::INDEX, span)},
        span);
    ExprPtr ramp = EmitTileOp("tile.ci", {start, shape}, {{"dtype", DataType::UINT16}, {"descending", false}},
                              name("ramp"), span, stmts);

    // Constant tiles for the indicator arithmetic.
    ExprPtr zero = unop("tile.ands", ramp, 0, true, "zero");
    ExprPtr ones = unop("tile.ors", zero, 1, true, "ones");

    // f8 of one nibble on u16 lanes (same formula, u16-appropriate scalars).
    auto nibble_f8 = [&](const ExprPtr& lanes, const char* tag) {
      ExprPtr c7 = unop("tile.ands", lanes, 7, true, (std::string(tag) + "_c7").c_str());
      ExprPtr e = unop("tile.shrs", c7, 1, false, (std::string(tag) + "_e").c_str());
      ExprPtr m = unop("tile.ands", c7, 1, true, (std::string(tag) + "_m").c_str());
      ExprPtr e8 = unop("tile.shls", e, 3, false, (std::string(tag) + "_e8").c_str());
      ExprPtr m4 = unop("tile.shls", m, 2, false, (std::string(tag) + "_m4").c_str());
      ExprPtr et = binop("tile.add", e8, m4, (std::string(tag) + "_et").c_str());
      ExprPtr base = EmitTileOp("tile.adds", {et, i32_scalar(48)}, {},
                                name((std::string(tag) + "_base").c_str()), span, stmts);
      ExprPtr ep3 = EmitTileOp("tile.adds", {e, i32_scalar(3)}, {}, name((std::string(tag) + "_ep3").c_str()),
                               span, stmts);
      ExprPtr nz0 = unop("tile.shrs", ep3, 2, false, (std::string(tag) + "_nz0").c_str());
      ExprPtr z0 = binop("tile.sub", ones, nz0, (std::string(tag) + "_z0").c_str());
      ExprPtr z0m = binop("tile.and", z0, m, (std::string(tag) + "_z0m").c_str());
      ExprPtr t1 = unop("tile.shls", z0m, 2, false, (std::string(tag) + "_t1").c_str());
      ExprPtr notm = binop("tile.sub", ones, m, (std::string(tag) + "_notm").c_str());
      ExprPtr z00 = binop("tile.and", z0, notm, (std::string(tag) + "_z00").c_str());
      ExprPtr z00h = unop("tile.shls", z00, 4, false, (std::string(tag) + "_zh").c_str());
      ExprPtr z00t = unop("tile.shls", z00, 5, false, (std::string(tag) + "_zt").c_str());
      ExprPtr t2 = binop("tile.or", z00h, z00t, (std::string(tag) + "_t2").c_str());
      ExprPtr corr = binop("tile.add", t1, t2, (std::string(tag) + "_cr").c_str());
      ExprPtr g = binop("tile.sub", base, corr, (std::string(tag) + "_g").c_str());
      ExprPtr sbit = unop("tile.ands", lanes, 8, true, (std::string(tag) + "_sb").c_str());
      ExprPtr sign = unop("tile.shls", sbit, 4, false, (std::string(tag) + "_sn").c_str());
      return binop("tile.or", g, sign, (std::string(tag) + "_f8").c_str());
    };

    // Extract nibbles from the ramp and compute their f8 embeddings.
    ExprPtr lo_n = unop("tile.ands", ramp, 15, true, "lo_n");
    ExprPtr hi_n = unop("tile.shrs", ramp, 4, false, "hi_n");
    ExprPtr lo_f8 = nibble_f8(lo_n, "lo");
    ExprPtr hi_f8 = nibble_f8(hi_n, "hi");

    // LUT[i] = f8(i&15) | f8(i>>4)<<8 — one u16 lane = two adjacent FP8 bytes.
    ExprPtr hi_s = unop("tile.shls", hi_f8, 8, false, "hi_s");
    lut_table_ = binop("tile.or", lo_f8, hi_s, "table");

    // tgather tmp stand-in for A5 (see member comment). Built once beside the
    // LUT so it is function-scoped and loop-invariant like the table itself.
    auto tstart = std::make_shared<ConstInt>(0, DataType::UINT16, span);
    auto tshape = std::make_shared<MakeTuple>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span),
                             std::make_shared<ConstInt>(16, DataType::INDEX, span)},
        span);
    lut_tmp_ = EmitTileOp("tile.ci", {tstart, tshape}, {{"dtype", DataType::UINT16}, {"descending", false}},
                          name("gtmp"), span, stmts);
  }

  /// Lower one `var = tile.cast(fp4_tile, FP8E4M3FN)` to the gather chain:
  /// reinterpret(U8) → cast(U16) → gather(lut, idx, tmp) → reinterpret(FP8).
  /// 3 real ops per cast site (TCVT + VAND + TGATHER); the LUT is built once.
  StmtPtr RewriteFp4ToFp8Gather(const AssignStmtPtr& op, const CallPtr& call, int mode) {
    const Span& span = op->span_;
    const std::string base = auto_name::GetBaseName(op->var_->name_hint_);
    std::vector<StmtPtr> stmts;
    auto name = [&](const std::string& tag) {
      return auto_name::BuildName(base, tag, "tmp", static_cast<int>(temp_counter_++));
    };

    // Build the LUT on first use in this function — into the prelude
    // (hoisted to the CORE scope body top by VisitStmt_(InCoreScopeStmt),
    // or to just above the enclosing loop by VisitStmt_(ForStmt /
    // WhileStmt) so a non-peeled cast inside pl.pipeline does not rebuild
    // the table on every iteration). If nothing can hoist it here (no CORE
    // scope, no enclosing loop: depth 0 at function level), the prelude
    // falls back to inline emission at this cast site.
    if (!lut_table_) {
      BuildLutTable(span);
      if (in_core_scope_depth_ == 0 && loop_depth_ == 0) {
        stmts.insert(stmts.begin(), lut_prelude_.begin(), lut_prelude_.end());
        lut_prelude_.clear();
        prelude_pending_depth_ = -1;
      }
    }

    ExprPtr src = VisitExpr(call->args_[0]);

    // u8 = reinterpret(fp4, UINT8) — zero-copy carrier byte view.
    ExprPtr u8 =
        EmitTileOp("tile.reinterpret_view", {src}, {{"dtype", DataType::UINT8}}, name("g_u8"), span, stmts);
    // idx = cast(u8, UINT16) — zero-extend each byte (1 TCVT).
    ExprPtr idx = EmitTileOp("tile.cast", {u8}, {{"target_type", DataType::UINT16}, {"mode", mode}},
                             name("g_idx"), span, stmts);
    // tgather's tmp operand. A5 (Ascend950) hardware never reads tmp
    // (TGATHER_IMPL drops the argument), so the 32-byte function-scoped
    // constant built beside the LUT satisfies the op contract with zero
    // per-site work. A2/A3's PTOAS verifier instead requires tmp to match
    // the indices' element type and valid shape, so keep the original
    // full-width dummy VAND there (ands(idx, 0) — written, never read).
    ExprPtr tmp;
    if (arch_name_ == "a5") {
      tmp = lut_tmp_;
    } else {
      tmp = EmitTileOp("tile.ands", {idx, MakeUintConst(0, DataType::INT16, span)}, {}, name("g_tmp"), span,
                       stmts);
    }
    // gathered = gather(lut, idx, tmp) — 1 TGATHER: each u16 = two FP8 bytes.
    // t12 (KN cast NZ direct-write): PYPTO_FP4_CAST_NZ_DIRECT=1 relabels the
    // gather output as NZ (col_major + row-major fractal). Contract: the
    // pto-isa a5 TGather NZ variant (shadow) stores the lanes directly at
    // NZ-fractal addresses, so the tile is truly NZ-arranged and the V2C
    // boundary adapt move (TMOV) becomes an identity to elide/skip. Only for
    // KN-form consumers (the cast output feeds matmul_mx as the rhs); the NK
    // chain transposes the cast output and must NOT set this.
    std::vector<std::pair<std::string, std::any>> gather_kwargs;
    // Codex P2 #7 (PR#2987 round 2): the NZ relabel is only valid when this
    // cast's result provably feeds a matmul(_mx) as the RHS (KN contract);
    // every other consumer keeps the default row-major gather.
    if (Fp4CastNzDirectEnabled() && kn_rhs_consumers_.count(op->var_.get()) > 0) {
      gather_kwargs.emplace_back("dst_nz", true);
    }
    ExprPtr gathered =
        EmitTileOp("tile.gather", {lut_table_, idx, tmp}, gather_kwargs, name("g_out"), span, stmts);
    // f8 = reinterpret(gathered, FP8E4M3FN) — zero-copy pair view.
    ExprPtr f8 = EmitTileOp("tile.reinterpret_view", {gathered}, {{"dtype", DataType::FP8E4M3FN}},
                            name("g_f8"), span, stmts);

    auto final_assign = MutableCopy(op);
    final_assign->value_ = f8;
    stmts.push_back(std::move(final_assign));

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

  std::string arch_name_;
  AdjList adj_;
  std::size_t temp_counter_ = 0;
  std::vector<Diagnostic>* diags_;
  bool fp4_to_fp8_bitops_;
  std::unordered_set<const Var*> kn_rhs_consumers_;  // vars read as a matmul(_mx) rhs (Codex P2 #7)

  /// LUT cache region scoping (PR#2987 review, Major 1). lut_table_ /
  /// lut_tmp_ reference Vars whose defining statements are emitted INSIDE
  /// the region where the LUT was first built: inline at the first cast
  /// site (function-level fallback) or hoisted into the enclosing InCore
  /// scope body top. A Var defined inside a nested region (scope, loop or
  /// branch body) does not dominate code AFTER that region, so the cache
  /// must be restored to its pre-region state when the region closes; a
  /// cast after the region then rebuilds its own table instead of
  /// referencing out-of-scope definitions (PTOAS rejects those as
  /// "use of undeclared SSA value name"). Restore-only — no clear on
  /// entry — keeps the legal reuse direction: a table built at an outer
  /// level dominates the nested region and is reused inside it.
  class LutCacheRegionGuard {
   public:
    explicit LutCacheRegionGuard(LegalizeTileCastMutator& m)
        : mutator_(m), table_(m.lut_table_), tmp_(m.lut_tmp_) {}
    ~LutCacheRegionGuard() {
      mutator_.lut_table_ = table_;
      mutator_.lut_tmp_ = tmp_;
    }
    LutCacheRegionGuard(const LutCacheRegionGuard&) = delete;
    LutCacheRegionGuard& operator=(const LutCacheRegionGuard&) = delete;

   private:
    LegalizeTileCastMutator& mutator_;
    ExprPtr table_;
    ExprPtr tmp_;
  };
  friend class LutCacheRegionGuard;

  ExprPtr lut_table_;  // 256-entry u16 LUT, built once per function (gather route)
  // [1,16] u16 constant tile standing in for tgather's tmp on A5: the A5
  // TGATHER_IMPL(dst, src0, src1, tmp) never touches tmp (verified in
  // pto/npu/a5/TGather.hpp), so a 32-byte constant satisfies the 3-arg
  // op contract without the per-site dummy full-width VAND that cost
  // 2x tile bytes of UB traffic per cast site (t2: 64 KiB per
  // [256,128]-nibble tile = 27% of the chain's vec traffic).
  ExprPtr lut_tmp_;
  std::vector<StmtPtr> lut_prelude_;  // LUT construction, hoisted to CORE scope top
  int in_core_scope_depth_ = 0;       // >0 inside pl.at(CORE_GROUP); 0 = function level
  int loop_depth_ = 0;                // >0 inside a For/While body (any nesting)
  // Codex P2 (PR#2987 round 4): loop-body depth the pending prelude was
  // created at (-1 = nothing pending). A nested inner loop must not hoist a
  // prelude owned by an outer loop body — hoisting above the inner loop can
  // place the construction after the gather that references it.
  int prelude_pending_depth_ = -1;

  /// Prepend the gather LUT construction to the InCore scope's body top.
  /// For non-peeled structures (cast directly inside pl.pipeline, no chunk0
  /// peeling) the inline emission at the first cast site would rebuild the
  /// 256-entry table on every pipeline iteration. Hoisting to the scope
  /// body top keeps the build at once-per-scope-invocation while staying
  /// inside the correct core-side scope — NOT the function body top, which
  /// is outside pl.at(CORE_GROUP) and unsafe before mixed-kernel expansion.
  StmtPtr VisitStmt_(const InCoreScopeStmtPtr& op) override {
    LutCacheRegionGuard guard(*this);
    in_core_scope_depth_++;
    auto body = VisitStmt(op->body_);
    in_core_scope_depth_--;
    if (!lut_prelude_.empty()) {
      auto prelude = std::move(lut_prelude_);
      lut_prelude_.clear();
      prelude_pending_depth_ = -1;
      std::vector<StmtPtr> stmts = std::move(prelude);
      if (auto seq = As<SeqStmts>(body)) {
        stmts.insert(stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
      } else {
        stmts.push_back(body);
      }
      return std::make_shared<InCoreScopeStmt>(op->split_, op->name_hint_,
                                               std::make_shared<SeqStmts>(std::move(stmts), body->span_),
                                               op->span_, op->leading_comments_, op->attrs_);
    }
    if (body != op->body_) {
      return std::make_shared<InCoreScopeStmt>(op->split_, op->name_hint_, body, op->span_,
                                               op->leading_comments_, op->attrs_);
    }
    return op;
  }

  /// Region-boundary LUT cache invalidation (see LutCacheRegionGuard): a
  /// cast inside a loop or branch body must not leak its table Var to casts
  /// after the region. The guard restores the cache when the region closes,
  /// on every return path (RAII); the base handlers rebuild the statement
  /// with mutated children. Loops additionally HOIST a pending function-
  /// level LUT prelude to just above the loop (same dominance region as
  /// the casts that consume it), so a non-peeled cast inside pl.pipeline
  /// does not rebuild the 256-entry table on every iteration.
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    LutCacheRegionGuard guard(*this);
    ++loop_depth_;
    const int body_depth = loop_depth_;
    StmtPtr visited = IRMutator::VisitStmt_(op);
    --loop_depth_;
    return HoistPendingPreludeAboveLoop(std::move(visited), op->span_, body_depth);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    LutCacheRegionGuard guard(*this);
    ++loop_depth_;
    const int body_depth = loop_depth_;
    StmtPtr visited = IRMutator::VisitStmt_(op);
    --loop_depth_;
    return HoistPendingPreludeAboveLoop(std::move(visited), op->span_, body_depth);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // Codex P1 #2 (PR#2987 round 2): the LUT cache must be scoped per
    // BRANCH, not around the whole if — a table built in the then-branch
    // does not dominate the else-branch (the else would reuse a
    // non-dominating SSA var). lut_prelude_ is deliberately NOT restored:
    // pending definitions may still be referenced by casts inside a closed
    // branch (see BuildLutTable) and are hoisted at the enclosing
    // scope/loop exit. Mirrors IRMutator::VisitStmt_(IfStmt) with a
    // LutCacheRegionGuard around each branch body.
    INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "IfStmt has null condition";
    auto new_condition = VisitExpr(op->condition_);
    INTERNAL_CHECK_SPAN(new_condition, op->span_) << "IfStmt condition mutated to null";

    INTERNAL_CHECK_SPAN(op->then_body_, op->span_) << "IfStmt has null then_body";
    StmtPtr new_then_body;
    {
      LutCacheRegionGuard guard(*this);
      new_then_body = VisitStmt(op->then_body_);
    }
    INTERNAL_CHECK_SPAN(new_then_body, op->span_) << "IfStmt then_body mutated to null";
    bool then_changed = (new_then_body.get() != op->then_body_.get());

    std::optional<StmtPtr> new_else_body;
    bool else_changed = false;
    if (op->else_body_.has_value()) {
      INTERNAL_CHECK_SPAN(*op->else_body_, op->span_) << "IfStmt has null else_body";
      StmtPtr new_stmt;
      {
        LutCacheRegionGuard guard(*this);
        new_stmt = VisitStmt(*op->else_body_);
      }
      INTERNAL_CHECK_SPAN(new_stmt, op->span_) << "IfStmt else_body mutated to null";
      new_else_body = new_stmt;
      if (new_stmt.get() != op->else_body_->get()) {
        else_changed = true;
      }
    }

    std::vector<VarPtr> new_return_vars;
    bool return_vars_changed = false;
    new_return_vars.reserve(op->return_vars_.size());
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      INTERNAL_CHECK_SPAN(op->return_vars_[i], op->span_) << "IfStmt has null return_vars at index " << i;
      auto new_var_expr = VisitExpr(op->return_vars_[i]);
      INTERNAL_CHECK_SPAN(new_var_expr, op->span_)
          << "IfStmt return_vars at index " << i << " mutated to null";
      auto new_var = As<Var>(new_var_expr);
      INTERNAL_CHECK_SPAN(new_var, op->span_)
          << "IfStmt return_vars at index " << i << " is not a Var after mutation";
      new_return_vars.push_back(new_var);
      if (new_var.get() != op->return_vars_[i].get()) {
        return_vars_changed = true;
      }
    }

    if (new_condition.get() != op->condition_.get() || then_changed || else_changed || return_vars_changed) {
      auto result = MutableCopy(op);
      result->condition_ = std::move(new_condition);
      result->then_body_ = std::move(new_then_body);
      result->else_body_ = new_else_body;
      result->return_vars_ = std::move(new_return_vars);
      return result;
    }
    return op;
  }

  /// Move a function-level pending LUT prelude to just above *visited* (the
  /// loop that contains — directly or in a nested region — the cast sites
  /// that will consume it). Inside a CORE scope the scope-exit hoist owns
  /// the prelude, so this is a no-op there.
  StmtPtr HoistPendingPreludeAboveLoop(StmtPtr visited, const Span& span, int body_depth) {
    if (lut_prelude_.empty() || in_core_scope_depth_ > 0) return visited;
    // Codex P2 (PR#2987 round 4): hoist only at the loop that OWNS the
    // pending prelude. A prelude created in an outer loop body must not be
    // hoisted above an inner loop — the referencing gather may precede the
    // inner loop in the outer body, and the hoisted construction would then
    // not dominate it.
    if (prelude_pending_depth_ >= 0 && prelude_pending_depth_ != body_depth) return visited;
    // Codex P2 (PR#2987 round 6): a prelude created inside NESTED loops must
    // not be materialized above the innermost loop — the outer loop would
    // then rebuild the ~58-op table on every iteration. The construction is a
    // pure constant table (no dependence on loop state), so when an enclosing
    // loop exists, re-own the pending prelude to it and let it ride up to the
    // outermost loop's entry, which still dominates every use. loop_depth_ is
    // already decremented past this loop, so it equals the enclosing loop's
    // body depth (0 = function level: flush here).
    if (loop_depth_ > 0) {
      prelude_pending_depth_ = loop_depth_;
      return visited;
    }
    std::vector<StmtPtr> stmts = std::move(lut_prelude_);
    lut_prelude_.clear();
    prelude_pending_depth_ = -1;
    stmts.push_back(std::move(visited));
    return std::make_shared<SeqStmts>(std::move(stmts), span);
  }
};

/// Codex P2 (PR#2987 round 2): PYPTO_FP4_CAST_NZ_DIRECT may only relabel a
/// gather NZ when its cast result provably feeds a matmul_mx/matmul as the
/// RHS operand (the KN contract — the consumer reads the tile in NZ/ZN dual
/// form). Collect the SSA vars read as the rhs argument (index 2 for
/// matmul_mx, index 1 for matmul) anywhere in the function, so casts with
/// any other consumer (store, elementwise, LHS feed, NK chains that
/// transpose the output) keep the default row-major emission.
std::unordered_set<const Var*> CollectKnRhsConsumerVars(const StmtPtr& stmt);

class KnRhsConsumerCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> rhs_vars;

  void VisitExpr_(const CallPtr& op) override {
    if (op->op_ && (op->op_->name_ == "tile.matmul_mx" || op->op_->name_ == "tile.matmul")) {
      const size_t rhs_index = op->op_->name_ == "tile.matmul_mx" ? 2 : 1;
      if (op->args_.size() > rhs_index) {
        if (auto v = std::dynamic_pointer_cast<const Var>(op->args_[rhs_index])) {
          rhs_vars.insert(v.get());
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }
};

std::unordered_set<const Var*> CollectKnRhsConsumerVars(const StmtPtr& stmt) {
  KnRhsConsumerCollector collector;
  collector.VisitStmt(stmt);
  return std::move(collector.rhs_vars);
}

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
  LegalizeTileCastMutator mutator(handler->GetTcvtAdjacency(), handler->GetPtoTargetArch(), &diags,
                                  handler->LowersFp4ToFp8InKernel(), CollectKnRhsConsumerVars(func->body_));
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
