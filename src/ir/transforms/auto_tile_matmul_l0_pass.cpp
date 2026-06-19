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
/// single 2D ``tile.store(c, base, out)``, the pass unrolls the output into a
/// ``ceil(M/m) x ceil(N/n)`` grid and emits, per sub-tile origin ``(mi, ni)``,
/// the K-loop above for the ``[m_eff, n_eff]`` (partial on the boundary) sub-
/// tile followed by ``tile.store(c_sub, [base_r + mi, base_c + ni], out_prev)``.
/// The stores chain the output tensor in SSA form; the final store's result
/// replaces the original store downstream.  Boundary sub-tiles use static
/// partial extents, so ``m`` / ``n`` need not divide ``M`` / ``N``.
///
/// Supported today:
///   * ``tile.matmul`` and ``tile.matmul_acc``.  ``tile.matmul_bias`` is
///     deferred — bias add only after the final iteration needs extra
///     rewriting that is not yet implemented.
///   * K tiling (``m == M and n == N``) for ``tile.matmul`` and
///     ``tile.matmul_acc``; M/N tiling for plain ``tile.matmul`` with a single
///     2D ``tile.store`` consumer.  M/N tiling of ``tile.matmul_acc`` (needs
///     per-sub-tile accumulator slicing), of a Vec left operand, of a single
///     K block (``K / k < 2``), or of a non-store consumer (needs the
///     Mat-scratch / ``tile.assemble`` path) is deferred — those emit a
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
  auto sa = BuildExtract(lhs_extract_src, {r.m, r.k}, mi_off, ko_var, MemorySpace::Left,
                         base + "_l0_a", sp);
  auto sb =
      BuildExtract(r.rhs_src, {r.k, r.n}, ko_var, ni_off, MemorySpace::Right, base + "_l0_b", sp);

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
  const std::string& op_name = call->op_->name_;
  const bool is_matmul = op_name == "tile.matmul";
  const bool is_matmul_acc = op_name == "tile.matmul_acc";
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

/// Counts reads (uses) of every Var/IterArg across a statement list, excluding
/// AssignStmt LHS defs.  Built once per SeqStmts (see ``CountSiblingUses``) so
/// the M/N foldability check — "is the matmul result used exactly once?" — is
/// an O(1) lookup, keeping the pass O(N) overall rather than rescanning the
/// siblings for every oversized matmul (.claude/rules/pass-complexity.md).
/// ``VisitVarLike_`` covers both Var and IterArg (.claude/rules/ir-kind-traits.md).
class SiblingUseCounter : public IRVisitor {
 public:
  std::unordered_map<const Var*, int> counts;

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
    if (!call || !call->op_ || call->op_->name_ != "tile.store" || call->args_.size() != 3) continue;
    if (auto src = AsVarLike(call->args_[0])) idx.store_of.emplace(src.get(), as.get());
  }
  idx.use_counts = std::move(counter.counts);
  return idx;
}

/// One folded M/N rewrite: the unrolled per-sub-tile K-loops + stores that
/// replace ``c = tile.matmul(...)`` together with its consumer store
/// ``out = tile.store(c, base, out)``.
struct MNFold {
  std::vector<StmtPtr> stmts;          ///< inner K-loops + per-sub-tile stores (emitted at the store site)
  VarPtr return_var;                   ///< final output-tensor SSA value
  VarPtr store_result_var;             ///< the consumer store's LHS (remapped to return_var)
  const AssignStmt* store = nullptr;   ///< consumer store to drop from the SeqStmts
};

/// Try to fold a Mat-resident plain ``tile.matmul`` whose [M, N] output exceeds
/// L0c into an unrolled grid of ``ceil(M/m) x ceil(N/n)`` sub-tile matmuls,
/// each computing an ``[m, n]`` (partial on the boundary) Acc result and
/// storing it straight to the output tensor at ``out[mi:, ni:]``.  Operands are
/// already Mat-resident, so only the output Acc overflows; sub-tiling the
/// output keeps every Acc tile within L0c.
///
/// Requires the matmul's result to be consumed by exactly one 2D
/// ``tile.store(c, base, out)`` (``result_uses`` and ``store_stmt`` come from
/// the precomputed SiblingIndex); that store is folded into the per-sub-tile
/// rewrite (the DDR-output case our solver kernels need).  Returns nullopt
/// (with a PerfHint) when the pattern does not match — the matmul is then left
/// untouched.  ``matmul_acc`` (caller-supplied [M, N] accumulator) and a Vec
/// left operand both need the Mat-scratch / per-iter assemble path that is
/// still deferred.
std::optional<MNFold> TryFoldMNTiling(const MatmulTiling& t, int result_uses, const AssignStmt* store_stmt,
                                      const std::unordered_map<const Var*, VarPtr>& remap,
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
  // M/N tiling reuses the whole-output K-loop builder per sub-tile, which
  // assumes a real (>= 2-trip) pipelined K-loop.  K == k (single block) would
  // need a straight-line per-sub-tile matmul instead — deferred.
  if (t.K / t.k < 2) {
    return skip("tile.matmul output exceeds L0c but K fits in a single L0 block (K / k < 2); the "
                "single-K-block M/N form is not yet supported — left untouched");
  }

  // The result must be consumed exactly once, by the 2D store we fold in.
  if (result_uses != 1 || !store_stmt) {
    return skip("tile.matmul output exceeds L0c but its result is not consumed by exactly one 2D "
                "tile.store — M/N direct-store fold needs `out = tile.store(c, ...)`; left untouched");
  }
  auto store_call = As<Call>(store_stmt->value_);
  INTERNAL_CHECK_SPAN(store_call, store_stmt->span_)
      << "Internal error: SiblingIndex store_of mapped a non-Call AssignStmt";

  auto offs = As<MakeTuple>(store_call->args_[1]);
  if (!offs || offs->elements_.size() != 2) {
    return skip("tile.store offsets are not a 2D tuple — M/N fold not applicable; left untouched");
  }
  const ExprPtr base_r = offs->elements_[0];
  const ExprPtr base_c = offs->elements_[1];
  auto out_in = AsVarLike(store_call->args_[2]);
  if (!out_in) {
    return skip("tile.store target is not a simple tensor variable — M/N fold not applicable; left "
                "untouched");
  }

  // Chain the per-sub-tile stores from the (possibly already-remapped) output
  // tensor so a sequence of folded matmul→store chains writing the same output
  // composes correctly.
  ExprPtr out_value = out_in;
  if (auto it = remap.find(out_in.get()); it != remap.end()) out_value = it->second;

  auto& reg = OpRegistry::GetInstance();
  const std::string base = t.assign->var_->name_hint_;
  const std::string out_base = out_in->name_hint_;
  const int64_t num_m = (t.M + t.m - 1) / t.m;
  const int64_t num_n = (t.N + t.n - 1) / t.n;

  std::vector<StmtPtr> stmts;
  VarPtr last_out = out_in;
  int tile_idx = 0;
  for (int64_t nj = 0; nj < num_n; ++nj) {
    const int64_t ni = nj * t.n;
    const int64_t n_eff = std::min<int64_t>(t.n, t.N - ni);
    for (int64_t mj = 0; mj < num_m; ++mj) {
      const int64_t mi = mj * t.m;
      const int64_t m_eff = std::min<int64_t>(t.m, t.M - mi);
      const std::string tbase = base + "_t" + std::to_string(tile_idx);

      // Inner K-loop: accumulate lhs[mi:mi+m_eff, :] @ rhs[:, ni:ni+n_eff] over
      // K into an [m_eff, n_eff] Acc result.
      auto inner =
          BuildKLoopRewrite(MakeKLoop(t, MakeIndex(mi, sp), MakeIndex(ni, sp), m_eff, n_eff, tbase));
      for (auto& s : inner.stmts) stmts.push_back(std::move(s));

      // Store the sub-tile straight to out[base_r + mi :, base_c + ni :].
      auto store_offs = std::make_shared<MakeTuple>(
          std::vector<ExprPtr>{OffsetPlus(base_r, mi, sp), OffsetPlus(base_c, ni, sp)}, sp);
      auto scall = reg.Create("tile.store", {inner.return_var, store_offs, out_value},
                              store_call->kwargs_, sp);
      auto sv = std::make_shared<Var>(out_base + "_t" + std::to_string(tile_idx), scall->GetType(), sp);
      stmts.push_back(std::make_shared<AssignStmt>(sv, scall, sp));
      out_value = sv;
      last_out = sv;
      ++tile_idx;
    }
  }

  return MNFold{std::move(stmts), last_out, store_stmt->var_, store_stmt};
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
      if (auto it = pending_folds.find(child.get()); it != pending_folds.end()) {
        for (auto& s : it->second.stmts) out.push_back(std::move(s));
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
                << "Internal error: K-only tiling expects K / k >= 2 (K=" << tiling->K
                << ", k=" << tiling->k << ")";
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
          if (auto fold = TryFoldMNTiling(*tiling, result_uses, store_stmt, remap, hints)) {
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
