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

/// AutoTileMatmulL0
/// ----------------
/// For each ``tile.matmul`` or ``tile.matmul_acc`` with static 2D operands,
/// picks an L0 tile shape ``(m, n, k)`` from the active ``BackendHandler``'s
/// L0 capacities (via ``utils::ChooseL0Tile``) and rewrites the call into a
/// K-loop.  The right (B) operand must be ``Mat``-resident; the left (A)
/// operand may be ``Mat`` (the QK pattern) or ``Vec`` (the fused-attention
/// ``score·V`` / PV pattern, where the softmax output crosses the cube↔vector
/// boundary resident in ``Vec``).  Tiling the Vec-fed PV matmul symmetrically
/// with QK makes its L0B right buffer a reusable sub-tile so ``MemoryReuse``
/// can alias it onto QK's freed L0B (peak L0B = ``max(QK, PV)`` instead of the
/// sum).  The K-loop has the shape:
///
///   * ``tile.matmul`` — the loop body branches on the iteration index
///     (``ko == 0``) so the first iteration uses ``tile.matmul`` (fresh
///     accumulator) and subsequent iterations use ``tile.matmul_acc``
///     (accumulating into the iter-arg).  The iter-arg init is an Acc-
///     resident ``tile.create`` placeholder so the iter-arg / yield /
///     return_var chain is Acc-typed end-to-end.
///   * ``tile.matmul_acc`` — every iteration is ``tile.matmul_acc``; the
///     iter-arg init is the caller-provided accumulator directly, so the
///     chain is uniform and no if-else is needed.
///
/// The K-loop is marked ``ForKind::Pipeline`` with ``pipeline_stages=2`` so
/// the downstream ``LowerPipelineLoops`` pass produces a 2-deep ping-pong
/// on the per-iter Mat→Left/Right extracts.
///
/// Operand extraction uses ``tile.extract(src, idx_row, idx_col, shape,
/// target_memory=Left|Right)`` directly — the SSA-form fusion of the older
/// ``tile.slice`` (Mat-resident result) + ``tile.mov`` (Mat→Left/Right) pair.
/// This (a) eliminates the intermediate Mat-resident slice tiles and their
/// MemRef allocations, and (b) lowers to ``pto.textract`` rather than
/// ``pto.subview``, sidestepping the latter's ``valid_row`` codegen
/// mismatch.
///
/// Layout for ``tile.matmul``:
///   c_init = tile.create([m, n], dtype, target_memory=Acc)  // placeholder
///   for ko in pl.pipeline(0, K, k, init_values=(c_init,), stage=2):
///     sa = tile.extract(x_mat, 0, ko, [m, k], target_memory=Left)
///     sb = tile.extract(y_mat, ko, 0, [k, n], target_memory=Right)
///     if ko == 0:
///       c1 = tile.matmul(sa, sb)             // fresh Acc
///       c_phi = pl.yield_(c1)                // if's return_var
///     else:
///       c2 = tile.matmul_acc(c_iter, sa, sb) // accumulate
///       c_phi = pl.yield_(c2)
///     yield c_phi
///
/// Layout for ``tile.matmul_acc`` (acc_init is the caller's accumulator):
///   for ko in pl.pipeline(0, K, k, init_values=(acc_init,), stage=2):
///     sa = tile.extract(x_mat, 0, ko, [m, k], target_memory=Left)
///     sb = tile.extract(y_mat, ko, 0, [k, n], target_memory=Right)
///     c_new = tile.matmul_acc(c_iter, sa, sb)
///     yield c_new
///
/// A fresh return_var typed identically to the iter-arg replaces the original
/// matmul's Var; uses of the original Var in the enclosing SeqStmts are
/// substituted by the mutator.
///
/// M/N tiling (output exceeds L0c)
/// -------------------------------
/// When ``ChooseL0Tile`` returns ``m < M`` or ``n < N`` the ``[M, N]`` output
/// Acc overflows L0c.  The operands are already Mat-resident, so only the
/// output overflows: for a plain ``tile.matmul`` whose result is consumed by a
/// single 2D ``tile.store(c, base, out)`` (the direct-store path), the pass
/// unrolls the output into a ``ceil(M/m) x ceil(N/n)`` grid and emits, per
/// sub-tile origin ``(mi, ni)``, the ``[m_eff, n_eff]`` (partial on the
/// boundary) sub-tile compute followed by
/// ``tile.store(c_sub, [base_r + mi, base_c + ni], out_prev)``.  Each sub-tile
/// uses the pipelined K-loop above when K spans >= 2 L0 blocks, or — when
/// ``k == K`` (the full K fits L0a/L0b at once) — a single straight-line
/// ``tile.matmul``, emitted as **nested pipelined loops** over the divisible
/// interior so ``LowerPipelineLoops`` double-buffers the operand extracts (see
/// ``BuildFullKPipelined``; the partial boundary is peeled into a straight-line
/// tail, and the stationary axis is auto-picked by panel cost).  The stores
/// chain the output tensor in SSA form; the final store's result replaces the
/// original store downstream.  Boundary sub-tiles use static partial extents, so
/// ``m`` / ``n`` need not divide ``M`` / ``N``.
///
/// Supported today:
///   * ``tile.matmul`` and ``tile.matmul_acc``.  ``tile.matmul_bias`` is
///     deferred — bias add only after the final iteration needs extra
///     rewriting that is not yet implemented.
///   * K tiling (``m == M and n == N``) for ``tile.matmul`` and
///     ``tile.matmul_acc``; M/N tiling for plain ``tile.matmul`` with a single
///     2D ``tile.store`` consumer (the direct-store path), with either a
///     pipelined K-loop or a straight-line single-K-block (``k == K``) per
///     sub-tile.  M/N tiling of ``tile.matmul_acc`` (needs per-sub-tile
///     accumulator slicing), of a Vec left operand, or of a result consumed
///     on-chip (not by a single 2D store) is deferred — those emit a
///     ``PerfHint`` and skip.
///   * ``K % k == 0``.  K-boundary handling (slice valid_shape on the last
///     iteration) is not yet implemented; mismatched cases emit a
///     ``PerfHint`` and skip.
///
/// Already-L0-sized matmuls (chooser returns ``(M, N, K)``) are left
/// untouched.

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
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
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kPassName = "AutoTileMatmulL0";

ExprPtr MakeIndex(int64_t v, const Span& span) {
  return std::make_shared<ConstInt>(v, DataType::INDEX, span);
}

ExprPtr MakeIndexTuple(const std::vector<int64_t>& values, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(values.size());
  for (auto v : values) elements.push_back(MakeIndex(v, span));
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

/// True if `tile`'s 2D shape is static and its memory space is one of
/// `allowed`.  Operand-source residency check for the L0 tiling rewrite:
///
///   * The right (B) operand must be ``Mat`` — it is loaded from DDR into L1
///     and fed into L0B.
///   * The left (A) operand may be ``Mat`` (the QK pattern) *or* ``Vec`` (the
///     fused-attention ``score·V`` / PV pattern, where the softmax/``exp``
///     output crosses the cube↔vector boundary resident in ``Vec`` rather
///     than ``Mat``).
///
/// This is purely a residency/static-shape check.  A ``Vec`` left operand is
/// not extracted directly: ``BuildKLoopRewrite`` stages it into ``Mat`` first
/// via ``BuildMoveToMat``, so the per-iter ``tile.extract`` always slices from
/// a ``Mat`` source regardless of the original operand space.
bool IsStatic2DInSpaces(const TileTypePtr& tile, std::initializer_list<MemorySpace> allowed, int64_t& out_d0,
                        int64_t& out_d1) {
  if (!tile || tile->shape_.size() != 2) return false;
  auto mem = tile->GetMemorySpace();
  if (!mem.has_value()) return false;
  bool space_ok = false;
  for (auto space : allowed) {
    if (*mem == space) {
      space_ok = true;
      break;
    }
  }
  if (!space_ok) return false;
  auto a = As<ConstInt>(tile->shape_[0]);
  auto b = As<ConstInt>(tile->shape_[1]);
  if (!a || !b) return false;
  out_d0 = a->value_;
  out_d1 = b->value_;
  return true;
}

/// Element width in bytes for a tile dtype.  Returns 0 for sub-byte types
/// (INT4, FP4 et al.) which the cube path does not support; the caller emits
/// a ``PerfHint`` and skips in that case.
uint32_t DTypeBytes(const DataType& dt) {
  size_t bits = dt.GetBit();
  if (bits % 8 != 0) return 0;
  return static_cast<uint32_t>(bits / 8);
}

/// Build a ``tile.extract(source, idx_row, idx_col, [shape],
/// target_memory=target)`` AssignStmt — the Mat→Left/Right SSA-form
/// extract used inside the K-loop.  Offsets are passed as separate scalar
/// exprs (typically a ConstInt 0 for the static axis and the loop var
/// ``ko`` for the K axis).  The result tile is already in the destination
/// memory space, so no follow-up ``tile.mov`` is needed.  The source is
/// always Mat-resident — a Vec-fed left operand is first staged into Mat by
/// ``BuildMoveToMat`` (see ``BuildKLoopRewrite``).
AssignStmtPtr BuildExtract(const VarPtr& source, const std::vector<int64_t>& shape, const ExprPtr& index_row,
                           const ExprPtr& index_col, MemorySpace target, const std::string& name_hint,
                           const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<ExprPtr> args = {source, index_row, index_col, MakeIndexTuple(shape, span)};
  std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", target}};
  auto call = reg.Create("tile.extract", args, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Build a ``tile.move(source, target_memory=Mat)`` AssignStmt that stages a
/// Vec-resident left operand into Mat (L1) *before* the K-loop, so the per-iter
/// ``tile.extract`` slices from Mat exactly like the QK (Mat-fed) path.
///
/// This matters for fused cube+vector roots (fused-attention PV / ``score·V``):
/// the softmax/``exp`` output reaches the matmul resident in ``Vec`` at the
/// cube↔vector boundary.  Keeping the boundary crossing a ``tile.move`` lets
/// ``ExpandMixedKernel`` recognise it (``CollectCVBoundaryMoves`` only matches
/// ``tile.move``) and lower it to the cross-core ``tpop_from_aiv`` handshake
/// (which lands the data in Mat — ``GetBoundaryTpopMemory(AIC) == Mat``).
/// Extracting straight from the Vec tile instead would leave the operand a
/// dangling cross-boundary free variable on the cube side.
AssignStmtPtr BuildMoveToMat(const VarPtr& source, const std::string& name_hint, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", MemorySpace::Mat}};
  auto call = reg.Create("tile.move", {source}, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Build the ``tile.create([m, n], dtype, target_memory=Acc)`` placeholder
/// that initializes the iter-arg.  Acc keeps the iter-arg / yield / return_var
/// chain structurally consistent with the per-iter ``tile.matmul[_acc]``
/// outputs, so subsequent matmul_acc consumers (and any nested for-loops
/// initialised from this return_var) still see an Acc-typed accumulator and
/// can be tiled in turn.  ``tile.create``'s deduce_type honors ``Acc`` and
/// emits the Nz TileView ``(col_major, row_major, fractal=1024)`` that
/// matches matmul output, so iter_arg/yield TileViews line up.
AssignStmtPtr BuildAccInit(int64_t m, int64_t n, const DataType& dtype, const std::string& name_hint,
                           const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", dtype},
                                                          {"target_memory", MemorySpace::Acc}};
  auto call = reg.Create("tile.create", {MakeIndexTuple({m, n}, span)}, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

struct KLoopRewrite {
  AssignStmtPtr original;
  VarPtr lhs_src;                 ///< [M, K] left operand — Mat- or Vec-resident
  VarPtr rhs_src;                 ///< [K, N] right operand — Mat-resident
  bool stage_lhs_to_mat = false;  ///< lhs is Vec-resident: stage Vec→Mat before the K-loop
  VarPtr acc_init = nullptr;      ///< Caller-provided accumulator for matmul_acc;
                                  ///< nullptr for plain matmul (Vec placeholder is built instead).
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  /// Output sub-tile origin (row, col) within the [M, N] product. The per-iter
  /// extracts slice ``lhs[mi : mi + m, ko : ko + k]`` and ``rhs[ko : ko + k,
  /// ni : ni + n]``. Null means 0 — the K-only path (m == M, n == N) leaves
  /// these null so the emitted IR is identical to the un-tiled output case.
  ExprPtr mi = nullptr;
  ExprPtr ni = nullptr;
  /// Var-name prefix for the loop's locals. Empty means use the original
  /// matmul's name hint. M/N tiling sets a per-sub-tile prefix so unrolled
  /// sub-tiles get distinct names (the print/parse round-trip needs unique
  /// names within a scope).
  std::string name_base;
};

struct RewriteResult {
  std::vector<StmtPtr> stmts;  ///< [Optional init,] ForStmt replacing the original AssignStmt.
  VarPtr return_var;           ///< ForStmt's return_var; substituted into downstream uses.
};

/// Body of the K-loop for plain ``tile.matmul``: branches on ``ko == 0``
/// between ``tile.matmul`` (fresh Acc) and ``tile.matmul_acc`` (accumulating).
/// The ``IfStmt`` materializes a phi return_var that the outer yield carries
/// back to the iter-arg.
StmtPtr BuildMatmulBody(const VarPtr& ko_var, const IterArgPtr& c_iter, const AssignStmtPtr& sa,
                        const AssignStmtPtr& sb, const std::string& base, const Span& sp) {
  auto& reg = OpRegistry::GetInstance();

  // Then-branch: fresh Acc tile from tile.matmul.
  auto c_then_call = reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
  auto c_then_var = std::make_shared<Var>(base + "_l0_c_first", c_then_call->GetType(), sp);
  auto c_then_assign = std::make_shared<AssignStmt>(c_then_var, c_then_call, sp);
  auto then_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_then_var}, sp);
  StmtPtr then_body = SeqStmts::Flatten(std::vector<StmtPtr>{c_then_assign, then_yield}, sp);

  // Else-branch: accumulate into the iter-arg.
  auto c_else_call = reg.Create("tile.matmul_acc", {ExprPtr(c_iter), sa->var_, sb->var_}, sp);
  auto c_else_var = std::make_shared<Var>(base + "_l0_c_acc", c_else_call->GetType(), sp);
  auto c_else_assign = std::make_shared<AssignStmt>(c_else_var, c_else_call, sp);
  auto else_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_else_var}, sp);
  StmtPtr else_body = SeqStmts::Flatten(std::vector<StmtPtr>{c_else_assign, else_yield}, sp);

  auto c_phi = std::make_shared<Var>(base + "_l0_c_phi", c_then_call->GetType(), sp);
  auto cond = MakeEq(ko_var, MakeIndex(0, sp), sp);
  auto if_stmt = std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>(else_body),
                                          std::vector<VarPtr>{c_phi}, sp);
  auto outer_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_phi}, sp);
  return SeqStmts::Flatten(std::vector<StmtPtr>{sa, sb, if_stmt, outer_yield}, sp);
}

/// Body of the K-loop for ``tile.matmul_acc``: every iteration accumulates
/// into ``c_iter`` via ``tile.matmul_acc``.  The first iteration's ``c_iter``
/// is the caller-supplied ``acc_init`` (threaded through ``init_values``), so
/// no if-else is needed — the accumulator chain is uniform.
StmtPtr BuildMatmulAccBody(const IterArgPtr& c_iter, const AssignStmtPtr& sa, const AssignStmtPtr& sb,
                           const std::string& base, const Span& sp) {
  auto& reg = OpRegistry::GetInstance();
  auto c_call = reg.Create("tile.matmul_acc", {ExprPtr(c_iter), sa->var_, sb->var_}, sp);
  auto c_var = std::make_shared<Var>(base + "_l0_c_acc", c_call->GetType(), sp);
  auto c_assign = std::make_shared<AssignStmt>(c_var, c_call, sp);
  auto outer_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_var}, sp);
  return SeqStmts::Flatten(std::vector<StmtPtr>{sa, sb, c_assign, outer_yield}, sp);
}

/// Build the replacement statements for one Mat-resident matmul or matmul_acc.
/// See the file-level comment for the emitted shape.
RewriteResult BuildKLoopRewrite(const KLoopRewrite& r) {
  const Span sp = r.original->span_;
  const std::string base = r.name_base.empty() ? r.original->var_->name_hint_ : r.name_base;
  const bool is_acc = r.acc_init != nullptr;

  std::vector<StmtPtr> out;
  out.reserve(2);

  // Iter-arg init.  For matmul_acc, use the caller's accumulator directly —
  // its type already matches the per-iter matmul_acc output (Acc with Nz
  // TileView), so iter_arg / yield types are structurally consistent.  For
  // plain matmul, build an Acc-resident ``tile.create`` placeholder; the
  // real accumulator buffer is materialized by the first iteration's
  // ``tile.matmul``, and the Nz TileView from ``tile.create`` matches the
  // matmul output so iter_arg / yield / return_var stay Acc-typed.
  ExprPtr init_value;
  TypePtr iter_type;
  if (is_acc) {
    init_value = r.acc_init;
    iter_type = r.acc_init->GetType();
  } else {
    auto acc_dtype = As<TileType>(r.original->var_->GetType())->dtype_;
    auto c_init = BuildAccInit(r.m, r.n, acc_dtype, base + "_l0_init", sp);
    out.push_back(c_init);
    init_value = c_init->var_;
    iter_type = c_init->var_->GetType();
  }

  auto ko_var = std::make_shared<Var>(base + "_l0_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);
  auto c_iter = std::make_shared<IterArg>(base + "_l0_c", iter_type, init_value, sp);

  // A Vec-resident left operand (fused-attention PV / ``score·V``) is staged
  // into Mat once, before the K-loop, so the per-iter extract slices from Mat
  // exactly like the QK path — and so ``ExpandMixedKernel`` can lower the
  // Vec→Mat boundary crossing via its ``tile.move``-based handshake (see
  // ``BuildMoveToMat``).  Mat-resident left operands extract directly.
  VarPtr lhs_extract_src = r.lhs_src;
  if (r.stage_lhs_to_mat) {
    auto lhs_mat = BuildMoveToMat(r.lhs_src, base + "_l0_lmat", sp);
    out.push_back(lhs_mat);
    lhs_extract_src = lhs_mat->var_;
  }

  // Per-iter operand extracts: lhs is sliced over rows [mi, mi + m) and along K
  // and lands in Left; rhs is sliced along K and over cols [ni, ni + n) and
  // lands in Right.  No intermediate Mat-resident tile and no follow-up
  // tile.mov is needed.  The K-only path passes mi == ni == null (== 0) with
  // m == M, n == N, so the extracts are identical to the un-tiled case.
  ExprPtr mi_off = r.mi ? r.mi : MakeIndex(0, sp);
  ExprPtr ni_off = r.ni ? r.ni : MakeIndex(0, sp);
  auto sa = BuildExtract(lhs_extract_src, {r.m, r.k}, mi_off, ko_var, MemorySpace::Left, base + "_l0_a", sp);
  auto sb = BuildExtract(r.rhs_src, {r.k, r.n}, ko_var, ni_off, MemorySpace::Right, base + "_l0_b", sp);

  StmtPtr body = is_acc ? BuildMatmulAccBody(c_iter, sa, sb, base, sp)
                        : BuildMatmulBody(ko_var, c_iter, sa, sb, base, sp);

  // The caller filters K/k < 2 cases (already-L0-sized when K == k); the loop
  // here always runs at least twice, so pipelining is always meaningful.
  std::vector<std::pair<std::string, std::any>> attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2}};

  // Build a fresh return_var typed identically to the iter-arg.  For
  // matmul_acc the type matches the original Var's type, but we still create
  // a fresh Var so the rewrite is uniform with the matmul case (downstream
  // substitution treats both identically).
  auto rv = std::make_shared<Var>(base, iter_type, r.original->var_->span_);

  auto for_stmt = std::make_shared<ForStmt>(ko_var, MakeIndex(0, sp), MakeIndex(r.K, sp), MakeIndex(r.k, sp),
                                            std::vector<IterArgPtr>{c_iter}, body, std::vector<VarPtr>{rv},
                                            sp, ForKind::Pipeline,
                                            /*chunk_config=*/std::nullopt, std::move(attrs));
  out.push_back(for_stmt);
  return RewriteResult{std::move(out), rv};
}

/// Operands + chosen L0 tile shape for a tileable matmul.  Produced by
/// ``AnalyzeMatmul``; the caller dispatches on ``needs_mn_tiling()`` to build
/// either the whole-output K-loop or the unrolled M/N grid of sub-tiles.
struct MatmulTiling {
  AssignStmtPtr assign;
  VarPtr lhs;       ///< [M, K] left operand — Mat (or Vec for the PV pattern; see stage_lhs_to_mat)
  VarPtr rhs;       ///< [K, N] right operand — Mat
  VarPtr acc_init;  ///< caller-provided accumulator for matmul_acc; null for plain matmul
  bool stage_lhs_to_mat = false;
  int64_t M = 0, N = 0, K = 0;
  int64_t m = 0, n = 0, k = 0;
  [[nodiscard]] bool is_acc() const { return acc_init != nullptr; }
  /// True when the chosen L0 tile is smaller than the [M, N] output on either
  /// axis — the output Acc would overflow L0c, so the output must be tiled.
  [[nodiscard]] bool needs_mn_tiling() const { return m != M || n != N; }
};

/// Build the K-loop descriptor for one output sub-tile ``[mi : mi + m_eff,
/// ni : ni + n_eff]``.  Passing ``mi == ni == nullptr`` with ``m_eff == M`` and
/// ``n_eff == N`` yields the whole-output (K-only) case unchanged.
KLoopRewrite MakeKLoop(const MatmulTiling& t, ExprPtr mi, ExprPtr ni, int64_t m_eff, int64_t n_eff,
                       std::string name_base) {
  KLoopRewrite r;
  r.original = t.assign;
  r.lhs_src = t.lhs;
  r.rhs_src = t.rhs;
  r.stage_lhs_to_mat = t.stage_lhs_to_mat;
  r.acc_init = t.acc_init;
  r.M = t.M;
  r.N = t.N;
  r.K = t.K;
  r.m = m_eff;
  r.n = n_eff;
  r.k = t.k;
  r.mi = std::move(mi);
  r.ni = std::move(ni);
  r.name_base = std::move(name_base);
  return r;
}

/// Decide whether `assign` is a Mat-resident matmul we know how to tile, and if
/// so which L0 tile shape to use.  Returns the tiling plan on success;
/// otherwise nullopt and (when useful) appends a PerfHint.  The caller
/// dispatches K-only vs M/N tiling on ``MatmulTiling::needs_mn_tiling()``.
std::optional<MatmulTiling> AnalyzeMatmul(const AssignStmtPtr& assign, std::vector<Diagnostic>& hints) {
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_) return std::nullopt;

  // ``tile.matmul`` and ``tile.matmul_acc`` are rewritten by this pass.
  // ``tile.matmul_bias`` is deferred — bias add inside a tiled K-loop needs
  // bias-add only after the final iteration, which is extra rewriting.
  const bool is_matmul = IsOp(call, "tile.matmul");
  const bool is_matmul_acc = IsOp(call, "tile.matmul_acc");
  if (!is_matmul && !is_matmul_acc) return std::nullopt;

  // Operand layout: (lhs, rhs) for matmul; (acc, lhs, rhs) for matmul_acc.
  // Use ``AsVarLike`` for the operands so IterArg (Var subclass) is accepted —
  // this is the common case for the accumulator inside a pipelined K-loop.
  const size_t expected_arity = is_matmul ? 2u : 3u;
  if (call->args_.size() != expected_arity) return std::nullopt;
  const size_t lhs_idx = is_matmul ? 0u : 1u;
  auto lhs = AsVarLike(call->args_[lhs_idx]);
  auto rhs = AsVarLike(call->args_[lhs_idx + 1u]);
  if (!lhs || !rhs) return std::nullopt;
  auto lhs_tile = As<TileType>(lhs->GetType());
  auto rhs_tile = As<TileType>(rhs->GetType());
  if (!lhs_tile || !rhs_tile) return std::nullopt;

  // For matmul_acc, ensure the caller's accumulator is a Var/IterArg with a
  // 2D TileType.  We accept both Acc- and Vec-typed accumulators: Vec is
  // common when the user pre-allocated the running accumulator with
  // ``pl.create_tensor`` / ``tile.create(target=Vec)`` and lets downstream
  // passes (``InferTileMemorySpace``) bridge to Acc.  We thread the
  // accumulator through the inner K-loop's iter-arg in either case.
  VarPtr acc_var;
  if (is_matmul_acc) {
    acc_var = AsVarLike(call->args_[0]);
    if (!acc_var) return std::nullopt;
    auto acc_tile = As<TileType>(acc_var->GetType());
    if (!acc_tile || acc_tile->shape_.size() != 2) return std::nullopt;
  }

  // Operand source residency, with static 2D shapes.  The right (B) operand
  // must be Mat — it is loaded from DDR into L1 and fed into L0B.  The left (A)
  // operand may be Mat (the QK pattern) or Vec (the fused-attention PV /
  // ``score·V`` pattern, where the softmax/``exp`` output crosses the
  // cube↔vector boundary resident in Vec).  Other cases (Acc operands, a Vec
  // right operand, dynamic shapes) are out of scope; return silently.
  int64_t M = 0, K_lhs = 0, K_rhs = 0, N = 0;
  if (!IsStatic2DInSpaces(lhs_tile, {MemorySpace::Mat, MemorySpace::Vec}, M, K_lhs) ||
      !IsStatic2DInSpaces(rhs_tile, {MemorySpace::Mat}, K_rhs, N)) {
    return std::nullopt;
  }
  // K mismatch is an ill-typed matmul — the op verifier should have caught it
  // upstream.  Treat as an internal invariant.
  INTERNAL_CHECK(K_lhs == K_rhs) << "tile.matmul: K dimensions don't match (lhs K=" << K_lhs
                                 << ", rhs K=" << K_rhs << ")";
  const int64_t K = K_lhs;

  uint32_t bytes_a = DTypeBytes(lhs_tile->dtype_);
  uint32_t bytes_b = DTypeBytes(rhs_tile->dtype_);
  // Output dtype is set by the matmul op's deduction (FP32 / INT32 today, but
  // future cube paths may add half-precision accumulation).  Read from the
  // call's result type rather than hardcoding so the chooser sees the actual
  // accumulator footprint.
  auto out_tile = As<TileType>(call->GetType());
  INTERNAL_CHECK(out_tile) << "Internal error: tile.matmul result is not a TileType";
  uint32_t bytes_c = DTypeBytes(out_tile->dtype_);
  if (bytes_a == 0 || bytes_b == 0 || bytes_c == 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-003",
                       "tile.matmul: unsupported sub-byte dtype on operand or accumulator — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  // Prefer the active PassContext's BackendHandler (the production path runs
  // under PassPipeline::Run, which establishes a context).  Fall back to the
  // global default backend so direct callers — e.g. tests that call
  // PassManager strategies' run_passes() without wrapping in a PassContext —
  // still work; this mirrors the env-var fallback documented in
  // .claude/rules/pass-context-config.md.
  const auto* ctx = PassContext::Current();
  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  INTERNAL_CHECK(handler) << "Internal error: BackendHandler is null";

  utils::L0TileConfig cfg;
  cfg.M = static_cast<int>(M);
  cfg.N = static_cast<int>(N);
  cfg.K = static_cast<int>(K);
  cfg.l0a_bytes = handler->GetL0aCapacityBytes();
  cfg.l0b_bytes = handler->GetL0bCapacityBytes();
  cfg.l0c_bytes = handler->GetL0cCapacityBytes();
  cfg.bytes_a = bytes_a;
  cfg.bytes_b = bytes_b;
  cfg.bytes_c = bytes_c;
  cfg.align_m = handler->GetL0FractalAlignment();
  cfg.align_n = handler->GetL0FractalAlignment();
  cfg.align_k = handler->GetL0FractalAlignment();
  cfg.min_m = handler->GetMinL0TileDim();
  cfg.min_n = handler->GetMinL0TileDim();
  cfg.min_k = handler->GetMinL0TileDim();
  cfg.double_buffer_a = true;
  cfg.double_buffer_b = true;
  cfg.double_buffer_c = false;
  // tile.matmul_acc threads the caller's accumulator into the K-loop's
  // iter-arg, so each invocation reads C from L1 at start and writes back at
  // end (gamma_c = 2 in the chooser's traffic model).  Plain tile.matmul
  // starts from a fresh Acc placeholder so C is write-only (gamma_c = 1).
  cfg.c_read = is_matmul_acc;
  cfg.allow_padding = false;

  utils::L0TileResult res;
  try {
    res = utils::ChooseL0Tile(cfg);
  } catch (const pypto::ValueError& e) {
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-005",
        std::string("tile.matmul: ChooseL0Tile rejected configuration — left untouched. ") + e.what(),
        assign->span_);
    return std::nullopt;
  }

  // Already L0-sized — nothing to do.
  if (res.m == M && res.n == N && res.k == K) return std::nullopt;

  // Require K divisible by the chosen k (applies to both K-only and M/N
  // tiling).  K-boundary handling (slice valid_shape on the last K iteration)
  // is not yet implemented.
  if (K % res.k != 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-007",
                       "tile.matmul: chooser picked k=" + std::to_string(res.k) + " not dividing K=" +
                           std::to_string(K) + "; K-boundary handling not yet supported — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  if (!res.perf_hint.empty()) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-008",
                       "tile.matmul: ChooseL0Tile fallback. " + res.perf_hint, assign->span_);
  }

  MatmulTiling t;
  t.assign = assign;
  t.lhs = lhs;
  t.rhs = rhs;
  // A Vec-resident left operand is staged into Mat before the K-loop (see
  // BuildMoveToMat); Mat-resident left operands extract directly.  The right
  // operand is always Mat (checked above), so it never needs staging.
  t.stage_lhs_to_mat = lhs_tile->GetMemorySpace() == MemorySpace::Vec;
  t.acc_init = acc_var;  // null for tile.matmul, set for tile.matmul_acc
  t.M = M;
  t.N = N;
  t.K = K;
  t.m = res.m;
  t.n = res.n;
  t.k = res.k;
  return t;
}

/// Per-output-sub-tile origin offset ``base + delta``.  Folds the common
/// constant-``base`` case (almost always ``0``) to a single ConstInt so the
/// emitted store offsets stay literal and round-trip cleanly.
ExprPtr OffsetPlus(const ExprPtr& base, int64_t delta, const Span& sp) {
  if (auto ci = As<ConstInt>(base)) return MakeIndex(ci->value_ + delta, sp);
  if (delta == 0) return base;
  return MakeAdd(base, MakeIndex(delta, sp), sp);
}

/// ``base + delta`` where ``delta`` is an Expr — a static ConstInt in the
/// unrolled grid (folded like ``OffsetPlus``) or a loop variable in the
/// pipelined emitter (left as a ``MakeAdd``).
ExprPtr AddOffset(const ExprPtr& base, const ExprPtr& delta, const Span& sp) {
  if (auto cd = As<ConstInt>(delta)) {
    if (cd->value_ == 0) return base;  // base + 0 = base (folds even for a dynamic base)
    if (auto cb = As<ConstInt>(base)) return MakeIndex(cb->value_ + cd->value_, sp);
  }
  if (auto cb = As<ConstInt>(base)) {
    if (cb->value_ == 0) return delta;  // 0 + delta = delta
  }
  return MakeAdd(base, delta, sp);
}

/// Counts reads (uses) of every Var/IterArg across a statement list, excluding
/// AssignStmt LHS defs.  Built once per SeqStmts (see ``BuildSiblingIndex``) so
/// the M/N foldability check — "is the matmul result used exactly once?" — is
/// an O(1) lookup, keeping the pass O(N) overall rather than rescanning the
/// siblings for every oversized matmul (.claude/rules/pass-complexity.md).
/// ``VisitVarLike_`` covers both Var and IterArg (.claude/rules/ir-kind-traits.md).
class SiblingUseCounter : public IRVisitor {
 public:
  std::unordered_map<const Var*, int> counts;  ///< all reads

 protected:
  void VisitVarLike_(const VarPtr& op) override { ++counts[op.get()]; }
  // Skip the LHS (a def); count only reads in the RHS value.
  void VisitStmt_(const AssignStmtPtr& op) override { VisitExpr(op->value_); }
};

/// One-shot index over a SeqStmts' children, built lazily on the first
/// oversized matmul and reused for the rest so M/N folding stays O(N):
///   * ``use_counts[v]`` — number of reads of ``v`` (excluding defs).
///   * ``store_of[v]`` — the top-level 2D ``tile.store`` whose source operand
///     is ``v`` (its sole direct consumer when ``use_counts[v] == 1``).
/// Counts/sites reflect the original (pre-rewrite) siblings, which is what the
/// foldability check needs (a matmul result is freshly defined; its uses do
/// not change until we rewrite it).
struct SiblingIndex {
  std::unordered_map<const Var*, int> use_counts;
  std::unordered_map<const Var*, const AssignStmt*> store_of;
};

SiblingIndex BuildSiblingIndex(const std::vector<StmtPtr>& stmts) {
  SiblingIndex idx;
  SiblingUseCounter counter;
  for (const auto& s : stmts) {
    counter.VisitStmt(s);
    auto as = std::dynamic_pointer_cast<const AssignStmt>(s);
    if (!as) continue;
    auto call = As<Call>(as->value_);
    // Record top-level 2D ``tile.store(src, offsets, out)`` by source operand.
    if (!call || !call->op_ || !IsOp(call, "tile.store") || call->args_.size() != 3) continue;
    if (auto src = AsVarLike(call->args_[0])) idx.store_of.emplace(src.get(), as.get());
  }
  idx.use_counts = std::move(counter.counts);
  return idx;
}

/// One folded M/N rewrite: the unrolled per-sub-tile K-loops + stores that
/// replace ``c = tile.matmul(...)`` together with its consumer store
/// ``out = tile.store(c, base, out)``.  The grid emits at the consumer-store
/// position; the store's LHS is remapped to ``return_var`` and the store
/// dropped.
struct MNFold {
  std::vector<StmtPtr> stmts;         ///< pipelined interior + tail / K-loops + per-sub-tile placement
  VarPtr return_var;                  ///< final output tensor value (replaces the store's result downstream)
  VarPtr store_result_var;            ///< the consumer store's LHS (remapped to return_var)
  const AssignStmt* store = nullptr;  ///< consumer store to drop from the SeqStmts
};

/// Where each computed ``[m_eff, n_eff]`` Acc sub-tile is placed.  The M/N grid
/// builders (``BuildFullKPipelined`` interior+tail, ``BuildSplitKGrid`` K-loop
/// grid) are placement-agnostic: they compute each sub-tile's Acc result and
/// hand it to a ``SubtilePlacer``, which stores it to a DDR output tensor
/// (``DirectGmPlacer`` — the only placement supported today).  The placer
/// threads its chained output Var in traversal order and yields the final Var
/// via ``PlaceAt``.  The abstraction is kept (rather than inlining the single
/// placer) so additional placements can be added without touching the grid
/// builders.
class SubtilePlacer {
 public:
  virtual ~SubtilePlacer() = default;
  /// Emit any prologue and return the initial chained Var: the raw output
  /// tensor for DirectGM.  The grid threads this Var through each ``PlaceAt``
  /// and returns the final one.
  [[nodiscard]] virtual VarPtr Init(std::vector<StmtPtr>& stmts) = 0;
  /// Place ``sub`` (an ``[m, n]`` Acc result) into ``chain_in`` at output offsets
  /// ``(row_off, col_off)`` — both Exprs (static ConstInt in the unrolled grid,
  /// loop variables in the pipelined emitter).  Append the placement stmt and
  /// return the new chained Var.  Stateless so it works inside a loop body where
  /// the chain is a loop iter-arg.
  [[nodiscard]] virtual VarPtr PlaceAt(std::vector<StmtPtr>& stmts, const VarPtr& sub, const ExprPtr& row_off,
                                       const ExprPtr& col_off, const VarPtr& chain_in, int step) = 0;
};

/// Direct-store placement: ``out = tile.store(sub, [base_r + mi, base_c + ni],
/// out_prev)`` per sub-tile, chaining the DDR output tensor in SSA form.
class DirectGmPlacer : public SubtilePlacer {
 public:
  DirectGmPlacer(ExprPtr base_r, ExprPtr base_c, VarPtr out_in,
                 std::vector<std::pair<std::string, std::any>> store_kwargs, Span span)
      : base_r_(std::move(base_r)),
        base_c_(std::move(base_c)),
        out_in_(std::move(out_in)),
        out_base_(out_in_->name_hint_),
        kwargs_(std::move(store_kwargs)),
        sp_(std::move(span)) {}

  [[nodiscard]] VarPtr Init(std::vector<StmtPtr>& /*stmts*/) override { return out_in_; }

  [[nodiscard]] VarPtr PlaceAt(std::vector<StmtPtr>& stmts, const VarPtr& sub, const ExprPtr& row_off,
                               const ExprPtr& col_off, const VarPtr& chain_in, int step) override {
    auto& reg = OpRegistry::GetInstance();
    auto offs = std::make_shared<MakeTuple>(
        std::vector<ExprPtr>{AddOffset(base_r_, row_off, sp_), AddOffset(base_c_, col_off, sp_)}, sp_);
    auto scall = reg.Create("tile.store", {sub, offs, chain_in}, kwargs_, sp_);
    auto sv = std::make_shared<Var>(out_base_ + "_t" + std::to_string(step), scall->GetType(), sp_);
    stmts.push_back(std::make_shared<AssignStmt>(sv, scall, sp_));
    return sv;
  }

 private:
  ExprPtr base_r_, base_c_;
  VarPtr out_in_;
  std::string out_base_;
  std::vector<std::pair<std::string, std::any>> kwargs_;
  Span sp_;
};

/// Emit one straight-line full-K sub-tile (no K-loop): extract the ``[m_eff, K]``
/// left and ``[K, n_eff]`` right panels, ``tile.matmul``, and hand the
/// ``[m_eff, n_eff]`` result to ``placer``.  ``m_eff`` / ``n_eff`` may be a
/// partial (< m / < n) remainder — this is the boundary-tile emitter for the
/// full-K grid's L-shaped tail (the divisible interior is pipelined instead).
/// Returns the chain after placement.
VarPtr EmitFullKTile(std::vector<StmtPtr>& stmts, const MatmulTiling& t, SubtilePlacer& placer,
                     const VarPtr& chain, int64_t mi, int64_t ni, int64_t m_eff, int64_t n_eff,
                     const std::string& base, int step) {
  const Span sp = t.assign->span_;
  auto& reg = OpRegistry::GetInstance();
  auto sa = BuildExtract(t.lhs, {m_eff, t.K}, MakeIndex(mi, sp), MakeIndex(0, sp), MemorySpace::Left,
                         base + "_ta" + std::to_string(step), sp);
  auto sb = BuildExtract(t.rhs, {t.K, n_eff}, MakeIndex(0, sp), MakeIndex(ni, sp), MemorySpace::Right,
                         base + "_tb" + std::to_string(step), sp);
  stmts.push_back(sa);
  stmts.push_back(sb);
  auto c_call = reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
  auto c_var = std::make_shared<Var>(base + "_tc" + std::to_string(step), c_call->GetType(), sp);
  stmts.push_back(std::make_shared<AssignStmt>(c_var, c_call, sp));
  return placer.PlaceAt(stmts, c_var, MakeIndex(mi, sp), MakeIndex(ni, sp), chain, step);
}

/// Build the full-K (``k == K``) M/N grid as a pipelined **interior** plus a
/// straight-line **tail**, so the downstream ``LowerPipelineLoops`` double-buffers
/// both operand extracts (the latency win the pto-isa A2A3 cost model predicts:
/// hiding the per-sub-tile L1→L0 extract behind the cube keeps it fed).
///
///   # interior: the [0, full_m) x [0, full_n) region tiled by FULL m x n blocks
///   out = for mi in pipeline(0, full_m, m):       # outer (stationary) axis
///       A = extract(lhs, mi, 0, [m, K], Left)
///       out = for ni in pipeline(0, full_n, n):   # inner (moving) axis → B double-buffered
///           B = extract(rhs, 0, ni, [K, n], Right)
///           out = place(matmul(A, B), mi, ni, out)
///   # tail: straight-line partial tiles over the L-shaped boundary
///   for ni in 0..N:           out = place(matmul(A[M-full_m,K], B[K, n_eff]), full_m, ni, out)
///   for mi in 0..full_m:      out = place(matmul(A[m, K], B[K, N-full_n]), mi, full_n, out)
///
/// The interior pipelines with **exact trip counts** (``full_m`` / ``full_n`` are
/// multiples of ``m`` / ``n`` by construction), so no divisibility constraint on
/// ``M`` / ``N`` is needed — any aligned tile the chooser picks works, and the
/// partial boundary is peeled rather than forcing a tiny exact-divisor tile.
/// The outer-loop (stationary) axis is chosen to minimise total operand-extract
/// traffic over the interior grid: A-stationary (rows outer) costs
/// ``P*A + P*Q*B``, B-stationary (cols outer) ``P*Q*A + Q*B`` (``P`` / ``Q`` =
/// interior row/col blocks, ``A`` / ``B`` = the per-panel extract bytes), so the
/// stationary operand is re-extracted once per outer step.  Drives the same
/// ``SubtilePlacer`` as the split-K grid, so the direct-store placement comes
/// out double-buffered.
std::pair<std::vector<StmtPtr>, VarPtr> BuildFullKPipelined(const MatmulTiling& t, SubtilePlacer& placer) {
  const Span sp = t.assign->span_;
  auto& reg = OpRegistry::GetInstance();
  const std::string base = t.assign->var_->name_hint_;
  // A tile never exceeds the problem dims (you cannot tile M into blocks larger
  // than M); the chooser guarantees this, so the interior always has >= 1 block.
  INTERNAL_CHECK_SPAN(t.m <= t.M && t.n <= t.N, sp)
      << "Internal error: full-K tile must not exceed the problem dims; got M=" << t.M << " m=" << t.m
      << " N=" << t.N << " n=" << t.n;

  // Choose the OUTER (stationary) panel by total interior extract traffic (see
  // the row/column cost below) — the cheaper traversal's panel is re-extracted
  // once per outer step.  AnalyzeMatmul guarantees both operands are TileType.
  auto lhs_tile = As<TileType>(t.lhs->GetType());
  auto rhs_tile = As<TileType>(t.rhs->GetType());
  INTERNAL_CHECK_SPAN(lhs_tile && rhs_tile, sp)
      << "Internal error: full-K pipelined operands must be TileType (guaranteed by AnalyzeMatmul)";
  const int64_t a_panel = t.m * t.K * static_cast<int64_t>(DTypeBytes(lhs_tile->dtype_));
  const int64_t b_panel = t.K * t.n * static_cast<int64_t>(DTypeBytes(rhs_tile->dtype_));
  // Pick the traversal that minimises total operand-extract traffic over the
  // interior grid (``p_blocks`` rows x ``q_blocks`` cols).  A-stationary (rows
  // outer) extracts A once per row + B every step: T_row = P*A + P*Q*B.
  // B-stationary (cols outer): T_col = P*Q*A + Q*B.  Comparing the totals is
  // exact for rectangular grids — unlike the ``A >= B`` panel-size heuristic,
  // which mis-picks when one axis has far more blocks (e.g. P=100, Q=2,
  // A=1.5B: T_row=350B > T_col=302B, so column traversal wins despite A > B).
  const int64_t p_blocks = t.M / t.m;  // interior row blocks (>= 1)
  const int64_t q_blocks = t.N / t.n;  // interior col blocks (>= 1)
  const int64_t t_row = p_blocks * a_panel + p_blocks * q_blocks * b_panel;
  const int64_t t_col = p_blocks * q_blocks * a_panel + q_blocks * b_panel;
  const bool row_outer = t_row <= t_col;  // A-stationary (rows outer) when row traversal is cheaper

  // Interior = the region tiled by FULL m x n blocks; the L-shaped partial
  // boundary beyond it is peeled into straight-line tiles below.
  const int64_t full_m = (t.M / t.m) * t.m;
  const int64_t full_n = (t.N / t.n) * t.n;

  std::vector<StmtPtr> stmts;
  VarPtr chain = placer.Init(stmts);

  // --- Interior: nested pipelined loops over [0, full_m) x [0, full_n) ---
  {
    const int64_t outer_extent = row_outer ? full_m : full_n;
    const int64_t outer_step = row_outer ? t.m : t.n;
    const int64_t inner_extent = row_outer ? full_n : full_m;
    const int64_t inner_step = row_outer ? t.n : t.m;
    const TypePtr out_type = chain->GetType();
    auto idx_type = std::make_shared<ScalarType>(DataType::INDEX);
    auto outer_var = std::make_shared<Var>(base + "_o", idx_type, sp);
    auto inner_var = std::make_shared<Var>(base + "_i", idx_type, sp);
    ExprPtr mi = row_outer ? ExprPtr(outer_var) : ExprPtr(inner_var);
    ExprPtr ni = row_outer ? ExprPtr(inner_var) : ExprPtr(outer_var);
    // The output/scratch chain threads through both loops as an iter-arg; the
    // inner iter-arg is initialised from the outer iter-arg.
    auto out_outer = std::make_shared<IterArg>(base + "_oc", out_type, chain, sp);
    auto out_inner = std::make_shared<IterArg>(base + "_ic", out_type, out_outer, sp);
    auto sa = BuildExtract(t.lhs, {t.m, t.K}, mi, MakeIndex(0, sp), MemorySpace::Left, base + "_a", sp);
    auto sb = BuildExtract(t.rhs, {t.K, t.n}, MakeIndex(0, sp), ni, MemorySpace::Right, base + "_b", sp);
    const AssignStmtPtr& outer_extract = row_outer ? sa : sb;  // stationary panel
    const AssignStmtPtr& inner_extract = row_outer ? sb : sa;  // moving panel
    auto c_call = reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
    auto c_var = std::make_shared<Var>(base + "_c", c_call->GetType(), sp);
    std::vector<StmtPtr> inner_body{inner_extract, std::make_shared<AssignStmt>(c_var, c_call, sp)};
    VarPtr inner_chain = placer.PlaceAt(inner_body, c_var, mi, ni, out_inner, /*step=*/0);
    inner_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{inner_chain}, sp));
    // overlap_stores=false: keep one L0C accumulator.  Letting CanonicalizeIOOrder
    // float the stores below the next matmul would keep two [m, n] results co-live
    // (2·m·n·bytes_c) while the chooser budgets one L0C buffer (double_buffer_c=
    // false) → L0C overflow.  The one-accumulator schedule still double-buffers
    // the moving-operand extract.
    std::vector<std::pair<std::string, std::any>> inner_attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2},
                                                                 {kPipelineOverlapStoresAttr, false}};
    auto inner_rv = std::make_shared<Var>(base + "_irv", out_type, sp);
    auto inner_for = std::make_shared<ForStmt>(inner_var, MakeIndex(0, sp), MakeIndex(inner_extent, sp),
                                               MakeIndex(inner_step, sp), std::vector<IterArgPtr>{out_inner},
                                               SeqStmts::Flatten(std::move(inner_body), sp),
                                               std::vector<VarPtr>{inner_rv}, sp, ForKind::Pipeline,
                                               /*chunk_config=*/std::nullopt, std::move(inner_attrs));
    std::vector<StmtPtr> outer_body{outer_extract, inner_for,
                                    std::make_shared<YieldStmt>(std::vector<ExprPtr>{inner_rv}, sp)};
    std::vector<std::pair<std::string, std::any>> outer_attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2},
                                                                 {kPipelineOverlapStoresAttr, false}};
    auto outer_rv = std::make_shared<Var>(base + "_orv", out_type, sp);
    auto outer_for = std::make_shared<ForStmt>(outer_var, MakeIndex(0, sp), MakeIndex(outer_extent, sp),
                                               MakeIndex(outer_step, sp), std::vector<IterArgPtr>{out_outer},
                                               SeqStmts::Flatten(std::move(outer_body), sp),
                                               std::vector<VarPtr>{outer_rv}, sp, ForKind::Pipeline,
                                               /*chunk_config=*/std::nullopt, std::move(outer_attrs));
    stmts.push_back(outer_for);
    chain = outer_rv;
  }

  // --- Tail: straight-line partial tiles for the L-shaped boundary ---
  // Bottom strip [full_m, M) x [0, N) (covers the corner), then right strip
  // [0, full_m) x [full_n, N).  Either is empty when its dim divides evenly.
  // step 0 is used by the interior placement; the tail continues from step 1.
  int tail_step = 1;
  for (int64_t ni = 0; full_m < t.M && ni < t.N; ni += t.n) {
    chain = EmitFullKTile(stmts, t, placer, chain, full_m, ni, t.M - full_m, std::min<int64_t>(t.n, t.N - ni),
                          base, tail_step++);
  }
  for (int64_t mi = 0; full_n < t.N && mi < full_m; mi += t.m) {
    chain = EmitFullKTile(stmts, t, placer, chain, mi, full_n, t.m, t.N - full_n, base, tail_step++);
  }
  return {std::move(stmts), chain};
}

/// Build the split-K M/N grid: ``ceil(M/m) x ceil(N/n)`` sub-tiles, each a
/// pipelined K-loop (``BuildKLoopRewrite``) over the ``[m_eff, n_eff]`` output,
/// handed to ``placer`` for placement.  Used when K spans >= 2 L0 blocks, so the
/// operand panel does not fit L0 and cannot stay resident across sub-tiles
/// (unlike the full-K pipelined path).  N-major traversal preserves the
/// historical sub-tile ordering / naming.
std::pair<std::vector<StmtPtr>, VarPtr> BuildSplitKGrid(const MatmulTiling& t, SubtilePlacer& placer) {
  const Span sp = t.assign->span_;
  const std::string base = t.assign->var_->name_hint_;
  const int64_t num_m = (t.M + t.m - 1) / t.m;
  const int64_t num_n = (t.N + t.n - 1) / t.n;

  std::vector<StmtPtr> stmts;
  VarPtr chain = placer.Init(stmts);
  int step = 0;
  for (int64_t nj = 0; nj < num_n; ++nj) {
    const int64_t ni = nj * t.n;
    const int64_t n_eff = std::min<int64_t>(t.n, t.N - ni);
    for (int64_t mj = 0; mj < num_m; ++mj) {
      const int64_t mi = mj * t.m;
      const int64_t m_eff = std::min<int64_t>(t.m, t.M - mi);
      const std::string tbase = base + "_t" + std::to_string(step);

      auto inner = BuildKLoopRewrite(MakeKLoop(t, MakeIndex(mi, sp), MakeIndex(ni, sp), m_eff, n_eff, tbase));
      for (auto& s : inner.stmts) stmts.push_back(std::move(s));
      chain = placer.PlaceAt(stmts, inner.return_var, MakeIndex(mi, sp), MakeIndex(ni, sp), chain, step);
      ++step;
    }
  }
  return {std::move(stmts), chain};
}

/// Try to fold a Mat-resident plain ``tile.matmul`` whose [M, N] output exceeds
/// L0c into a ``ceil(M/m) x ceil(N/n)`` grid of sub-tile matmuls, each computing
/// an ``[m, n]`` (partial on the boundary) Acc result.  Operands are already
/// Mat-resident, so only the output Acc overflows; sub-tiling keeps every Acc
/// tile within L0c.  Only the direct-store consumer is supported today:
///
///   * **Direct-store** — the sole consumer is a 2D ``tile.store(c, base, out)``:
///     each sub-tile stores straight to ``out[mi:, ni:]`` (the DDR-output case
///     our solver kernels need).  The store is folded in and emitted at the
///     store site.
///
/// ``result_uses`` / ``store_stmt`` come from the precomputed SiblingIndex.
/// Returns nullopt (with a PerfHint) when the consumer is not a single 2D
/// store — ``matmul_acc`` (caller-supplied [M, N] accumulator), a Vec left
/// operand, and a result consumed on-chip (not by a single 2D store) are all
/// deferred.
std::optional<MNFold> TryFoldMNTiling(const MatmulTiling& t, int result_uses, const AssignStmt* store_stmt,
                                      std::vector<Diagnostic>& hints) {
  const Span sp = t.assign->span_;
  auto skip = [&](const std::string& msg) -> std::optional<MNFold> {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006", msg, sp);
    return std::nullopt;
  };

  if (t.is_acc()) {
    return skip(
        "tile.matmul_acc with an oversized [M, N] output needs M/N tiling — the matmul_acc path is "
        "deferred (needs per-sub-tile accumulator slicing); left untouched");
  }
  if (t.stage_lhs_to_mat) {
    return skip(
        "tile.matmul with a Vec left operand needs M/N tiling — the PV path is deferred; left untouched");
  }
  // K spans >= 2 L0 blocks → pipelined K-loop per sub-tile (BuildSplitKGrid);
  // k == K (full K fits L0a/L0b) → pipelined interior + straight-line partial
  // tail (BuildFullKPipelined).  Either grid drives the chosen SubtilePlacer;
  // neither requires m | M or n | N (the full-K tail peels the partial boundary).
  const bool full_k = t.K / t.k < 2;

  // Direct-store: the sole consumer is a 2D tile.store.  The grid is emitted
  // later at the store site, where the caller re-applies the then-current remap
  // — so a prior fold that redefined this output is rewritten correctly (a
  // stale-output SSA guard); resolving it here would miss folds emitted before
  // this one.
  if (store_stmt && result_uses == 1) {
    auto store_call = As<Call>(store_stmt->value_);
    INTERNAL_CHECK_SPAN(store_call, store_stmt->span_)
        << "Internal error: SiblingIndex store_of mapped a non-Call AssignStmt";
    auto offs = As<MakeTuple>(store_call->args_[1]);
    if (!offs || offs->elements_.size() != 2) {
      return skip("tile.store offsets are not a 2D tuple — M/N fold not applicable; left untouched");
    }
    auto out_in = AsVarLike(store_call->args_[2]);
    if (!out_in) {
      return skip(
          "tile.store target is not a simple tensor variable — M/N fold not applicable; left untouched");
    }
    DirectGmPlacer placer(offs->elements_[0], offs->elements_[1], out_in, store_call->kwargs_, sp);
    auto [stmts, last_out] = full_k ? BuildFullKPipelined(t, placer) : BuildSplitKGrid(t, placer);
    return MNFold{std::move(stmts), last_out, store_stmt->var_, store_stmt};
  }

  return skip(
      "tile.matmul output exceeds L0c but its result is not consumed by a single 2D tile.store "
      "(direct-store) — a result consumed on-chip (chained matmul / elementwise), stored-and-reused, "
      "or fed to a non-store consumer is deferred; left untouched");
}

class AutoTileMutator : public IRMutator {
 public:
  std::vector<Diagnostic> hints;

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Per-SeqStmts substitution map: when we rewrite ``c = tile.matmul(...)``
    // into a ForStmt with a fresh return_var, subsequent statements in the
    // same SeqStmts that referenced ``c`` need to be redirected to that
    // return_var.  Scoped to this SeqStmts so substitutions don't leak into
    // sibling regions.
    std::unordered_map<const Var*, VarPtr> remap;
    // M/N tiling folds the matmul's consumer store into the per-sub-tile
    // rewrite.  We drop the matmul at its own position and emit the sub-tile
    // stmts where the store was (preserving the order of any statements between
    // them), keyed by the store statement's identity.
    std::unordered_map<const Stmt*, MNFold> pending_folds;
    // Use counts + store-consumer sites across this SeqStmts, built lazily on
    // the first oversized matmul and reused — O(N) total, no rescan per matmul.
    std::optional<SiblingIndex> sibling_index;
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      const StmtPtr& child = op->stmts_[i];

      // A consumer store folded into a prior M/N rewrite: emit the sub-tile
      // stmts in the store's original position and drop the store itself.
      // Apply the now-current remap so the folded stores' output-tensor chain
      // start (and any other operands) reflect rewrites installed between the
      // matmul and this store — in particular a prior fold that redefined the
      // output this store fed from.  Without this, a fold built before that
      // remap existed (e.g. when the matmuls are defined in the reverse order
      // of their stores) would keep a stale, now-undefined output Var.
      //
      // Exclude this fold's *own* store-result var: its rewrite targets only
      // downstream uses, never the fold's internal chain start.  For an
      // output-param store the input tensor and the store result are the same
      // SSA var, so applying that one entry would rewrite the chain start onto
      // the fold's final output — a self-referential use-before-def.
      if (auto it = pending_folds.find(child.get()); it != pending_folds.end()) {
        auto self = remap.extract(it->second.store_result_var.get());
        for (auto& s : it->second.stmts) {
          out.push_back(remap.empty() ? s : transform_utils::Substitute(s, remap));
        }
        if (!self.empty()) remap.insert(std::move(self));  // restore for downstream uses
        changed = true;
        continue;
      }

      // Apply the running remap to redirect prior rewrites' downstream uses.
      StmtPtr current = remap.empty() ? child : transform_utils::Substitute(child, remap);

      // Check if this is a matmul we rewrite *at this SeqStmts level*.  We
      // try this before recursive visitation so the rewrite — which produces
      // a sequence of stmts — lands in this enclosing SeqStmts.  Recursive
      // visitation happens after rewrite-rejection so nested matmuls inside
      // ForStmt bodies still get rewritten by the recursive visit.
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(current)) {
        if (auto tiling = AnalyzeMatmul(assign, hints)) {
          if (!tiling->needs_mn_tiling()) {
            // Whole output fits L0c — tile K only (existing behaviour).
            INTERNAL_CHECK_SPAN(tiling->K / tiling->k >= 2, tiling->assign->span_)
                << "Internal error: K-only tiling expects K / k >= 2 (K=" << tiling->K << ", k=" << tiling->k
                << ")";
            auto rewrite = BuildKLoopRewrite(
                MakeKLoop(*tiling, /*mi=*/nullptr, /*ni=*/nullptr, tiling->m, tiling->n, /*name_base=*/""));
            remap[assign->var_.get()] = rewrite.return_var;
            for (auto& s : rewrite.stmts) out.push_back(std::move(s));
            changed = true;
            continue;
          }
          // Output exceeds L0c — tile M/N by folding the consumer store, found
          // via the raw (un-substituted) SiblingIndex: the matmul's result is
          // freshly defined here, so its use count / store site are never
          // affected by the running remap.
          if (!sibling_index) sibling_index = BuildSiblingIndex(op->stmts_);
          const Var* result = assign->var_.get();
          auto uc_it = sibling_index->use_counts.find(result);
          const int result_uses = uc_it == sibling_index->use_counts.end() ? 0 : uc_it->second;
          auto store_it = sibling_index->store_of.find(result);
          const AssignStmt* store_stmt =
              store_it == sibling_index->store_of.end() ? nullptr : store_it->second;
          if (auto fold = TryFoldMNTiling(*tiling, result_uses, store_stmt, hints)) {
            remap[fold->store_result_var.get()] = fold->return_var;
            pending_folds.emplace(static_cast<const Stmt*>(fold->store), std::move(*fold));
            changed = true;
            continue;  // drop the matmul; sub-tile stmts emit at the store site
          }
          // M/N tiling not applicable — fall through and leave it untouched.
        }
      }
      auto visited = VisitStmt(current);
      if (visited.get() != child.get()) changed = true;
      out.push_back(visited);
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }
};

FunctionPtr TransformFunction(const FunctionPtr& func, std::vector<Diagnostic>& hints) {
  if (!func || !func->body_) return func;
  if (!IsInCoreType(func->func_type_)) return func;
  AutoTileMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  for (auto& d : mutator.hints) hints.push_back(std::move(d));
  if (new_body == func->body_) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass AutoTileMatmulL0() {
  auto run = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool any_change = false;
    std::vector<Diagnostic> hints;
    for (const auto& [gvar, func] : program->functions_) {
      auto new_func = TransformFunction(func, hints);
      if (new_func != func) any_change = true;
      new_functions.emplace(gvar, new_func);
    }
    if (!hints.empty()) EmitDiagnostics(hints, kPassName);
    if (!any_change) return program;
    auto new_program = MutableCopy(program);
    new_program->functions_ = std::move(new_functions);
    return new_program;
  };
  return CreateProgramPass(run, kPassName, kAutoTileMatmulL0Properties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
