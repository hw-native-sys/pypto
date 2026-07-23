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
///   * Any 16-aligned K.  When the chosen ``k`` does not divide ``K``
///     (``allow_k_boundary``) the K-loop peels a partial last block of width
///     ``K - (K/k)*k``; with K and k both 16-aligned the tail is itself 16-aligned
///     -- an ordinary matmul_acc block (ptoas requires 16-aligned tile cols).
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
  auto& reg = OpRegistry::GetInstance();

  INTERNAL_CHECK_SPAN(r.k < r.K, sp) << "Internal error: BuildKLoopRewrite expects a tiled K (k < K), got k="
                                     << r.k << ", K=" << r.K;

  // K-block decomposition.  With allow_k_boundary the chosen k need not divide
  // K: the reduction is `num_full` full blocks of width k plus a final partial
  // block of width `k_eff = K - num_full*k` (k_eff == 0 when k divides K — the
  // legacy uniform loop).  K and k are 16-aligned, so k_eff is itself 16-aligned —
  // an ordinary matmul_acc block (ptoas requires 16-aligned tile cols).
  const int64_t num_full = r.K / r.k;
  const int64_t k_full = num_full * r.k;
  const int64_t k_eff = r.K - k_full;
  const bool has_tail = k_eff > 0;

  std::vector<StmtPtr> out;

  // Iter-arg init for the pipelined K-loop (only when there are >= 2 full
  // blocks).  Emitted FIRST — before the optional Vec->Mat staging below — so a
  // k that divides K yields byte-identical output to the pre-peel emitter.  For
  // matmul_acc the caller's accumulator is the init directly; for plain matmul a
  // fresh Acc-resident ``tile.create`` placeholder that the first iteration's
  // ``tile.matmul`` overwrites (the Nz TileView matches, so iter_arg / yield /
  // return_var stay Acc-typed).
  ExprPtr loop_init;
  TypePtr loop_iter_type;
  if (num_full >= 2) {
    if (is_acc) {
      loop_init = r.acc_init;
      loop_iter_type = r.acc_init->GetType();
    } else {
      auto acc_dtype = As<TileType>(r.original->var_->GetType())->dtype_;
      auto c_init = BuildAccInit(r.m, r.n, acc_dtype, base + "_l0_init", sp);
      out.push_back(c_init);
      loop_init = c_init->var_;
      loop_iter_type = c_init->var_->GetType();
    }
  }

  // A Vec-resident left operand (fused-attention PV / ``score·V``) is staged into
  // Mat once, before any K block, so each extract slices from Mat exactly like
  // the QK path — and so ``ExpandMixedKernel`` can lower the Vec→Mat crossing via
  // its ``tile.move`` handshake (``CollectCVBoundaryMoves`` only matches
  // ``tile.move``).  Mat-resident left operands extract directly.
  VarPtr lhs_extract_src = r.lhs_src;
  if (r.stage_lhs_to_mat) {
    auto lhs_mat = BuildMoveToMat(r.lhs_src, base + "_l0_lmat", sp);
    out.push_back(lhs_mat);
    lhs_extract_src = lhs_mat->var_;
  }

  ExprPtr mi_off = r.mi ? r.mi : MakeIndex(0, sp);
  ExprPtr ni_off = r.ni ? r.ni : MakeIndex(0, sp);

  // Emit one straight-line K block ``[m, kb] x [kb, n]`` at static K-offset
  // ``ko``, accumulating into ``acc_in`` (``tile.matmul_acc``) or starting fresh
  // (``tile.matmul`` when ``acc_in`` is null).  Used for the single-full-block
  // (num_full == 1) and partial-tail cases; the multi-block case pipelines below.
  auto emit_block = [&](int64_t ko, int64_t kb, const ExprPtr& acc_in, const std::string& tag) -> VarPtr {
    auto sa = BuildExtract(lhs_extract_src, {r.m, kb}, mi_off, MakeIndex(ko, sp), MemorySpace::Left,
                           base + "_l0_a" + tag, sp);
    auto sb = BuildExtract(r.rhs_src, {kb, r.n}, MakeIndex(ko, sp), ni_off, MemorySpace::Right,
                           base + "_l0_b" + tag, sp);
    ExprPtr call = acc_in ? reg.Create("tile.matmul_acc", {acc_in, sa->var_, sb->var_}, sp)
                          : reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
    auto cvar = std::make_shared<Var>(base + "_l0_c" + tag, call->GetType(), sp);
    out.push_back(sa);
    out.push_back(sb);
    out.push_back(std::make_shared<AssignStmt>(cvar, call, sp));
    return cvar;
  };

  // --- Full blocks: a pipelined K-loop when there are >= 2 of them, else a
  //     single straight-line block (a 1-trip pipeline loop would be degenerate).
  VarPtr main_var;
  if (num_full >= 2) {
    auto ko_var = std::make_shared<Var>(base + "_l0_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);
    auto c_iter = std::make_shared<IterArg>(base + "_l0_c", loop_iter_type, loop_init, sp);
    auto sa =
        BuildExtract(lhs_extract_src, {r.m, r.k}, mi_off, ko_var, MemorySpace::Left, base + "_l0_a", sp);
    auto sb = BuildExtract(r.rhs_src, {r.k, r.n}, ko_var, ni_off, MemorySpace::Right, base + "_l0_b", sp);
    StmtPtr body = is_acc ? BuildMatmulAccBody(c_iter, sa, sb, base, sp)
                          : BuildMatmulBody(ko_var, c_iter, sa, sb, base, sp);
    std::vector<std::pair<std::string, std::any>> attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2}};
    // Loop return var: an intermediate when a partial tail follows (named
    // distinctly so round-trip names stay unique), else the final result.
    auto rv =
        std::make_shared<Var>(has_tail ? base + "_l0_kmain" : base, loop_iter_type, r.original->var_->span_);
    auto for_stmt = std::make_shared<ForStmt>(
        ko_var, MakeIndex(0, sp), MakeIndex(k_full, sp), MakeIndex(r.k, sp), std::vector<IterArgPtr>{c_iter},
        body, std::vector<VarPtr>{rv}, sp, ForKind::Pipeline, std::move(attrs));
    out.push_back(for_stmt);
    main_var = rv;
  } else {
    // num_full == 1: a single straight-line full block (k < K checked above, so a
    // partial tail always follows).  This is a correctness guard, not a hot path:
    // the roofline chooser never selects k in (K/2, K) -- a near-full k is wall-
    // dominated (2x the MAD ceil-step of a divisor, and it loses the min-padding
    // tie-break), so num_full is >= 2 in practice.  Kept so the emitter stays
    // correct (no degenerate 1-trip pipeline) if the cost model ever changes.
    main_var = emit_block(/*ko=*/0, /*kb=*/r.k, is_acc ? ExprPtr(r.acc_init) : nullptr, "0");
  }

  // --- Partial tail: matmul_acc the [m, k_eff] x [k_eff, n] block onto the
  //     full-blocks accumulator (no-op when k divides K). ---
  VarPtr result_var = has_tail ? emit_block(k_full, k_eff, ExprPtr(main_var), "t") : main_var;
  return RewriteResult{std::move(out), result_var};
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
  /// Chosen L0 stationarity (which operand is pinned single-buffered across the
  /// moving grid). Output-stationary unless the chooser picked A/B-stationary
  /// (k == K); BuildFullKPipelined sets the loop order + single-buffers the
  /// stationary (outer) operand accordingly.
  utils::Stationarity stationarity = utils::Stationarity::kOutputStationary;
  /// For full-K output-stationary (BuildFullKPipelined at k == K): which operand
  /// the chooser hoists to the outer loop — true = hold A (rows outer), false =
  /// hold B (cols outer). Set from L0TileResult::os_holds_a so the emitted hoist
  /// matches the bandwidth-weighted hoist the wall cost was scored under. Ignored
  /// for A/B-stationary (loop order comes from `stationarity`) and split-K.
  bool os_holds_a = true;
  /// True when the chooser picked dbC=2 (double-buffered L0C): the accumulator is
  /// budgeted at L0C/2 so two co-live [m, n] Acc tiles fit, and BuildFullKPipelined
  /// tags the moving loop with kPipelineDoubleBufferCAttr so CanonicalizeIOOrder
  /// floats the drains past the next matmul, keeping the two tiles co-live.  With
  /// MemoryReuse skipped (ptoas planner) InitMemRef keeps the two overlapping-live-
  /// range buffers distinct and ptoas places them, hiding tile i's drain behind tile
  /// i+1's MAD.  Set from L0TileResult::double_buffer_c; only true under the ptoas
  /// memory planner, and only for full-K tiles (see the assert in AnalyzeMatmul).
  bool double_buffer_c = false;
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
std::optional<MatmulTiling> AnalyzeMatmul(const AssignStmtPtr& assign, std::vector<Diagnostic>& hints,
                                          bool force_output_stationary = false) {
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
  const auto cost_model = handler->GetL0CostModel();
  cfg.bw_a = cost_model.bw_l0a;
  cfg.bw_b = cost_model.bw_l0b;
  cfg.bw_drain = cost_model.bw_drain;
  cfg.drain_fixed_cycles = cost_model.drain_fixed_cycles;
  cfg.drain_row_cycles = cost_model.drain_row_cycles;
  cfg.drain_penalty_cycles = cost_model.drain_penalty_cycles;
  cfg.drain_c0_bytes = cost_model.drain_c0_bytes;
  cfg.mad_head = cost_model.mad_head_cycles;
  cfg.mad_k_fractal_bytes = cost_model.mad_k_fractal_bytes;
  cfg.mad_fp32_passes = cost_model.mad_fp32_passes;
  cfg.bytes_a = bytes_a;
  cfg.bytes_b = bytes_b;
  cfg.bytes_c = bytes_c;
  cfg.align_m = handler->GetL0FractalAlignment();
  cfg.align_n = handler->GetL0FractalAlignment();
  cfg.align_k = handler->GetL0FractalAlignment();
  cfg.min_m = handler->GetMinL0TileDim();
  cfg.min_n = handler->GetMinL0TileDim();
  cfg.min_k = handler->GetMinL0TileDim();
  // Operand-stationary (the chooser's allow_a/b_stationary; dbA/dbB derived):
  // pin the chosen operand in a SINGLE-buffered (full) L0 buffer across the moving
  // grid (requires k == K), streaming the other operand double-buffered.
  // BuildFullKPipelined realizes it by making the stationary (outer) loop
  // ForKind::Sequential — the held operand then uses the full L0 buffer the chooser
  // budgeted (no /2) and is loaded once per outer step (no re-stream across the
  // moving axis). The chooser adopts A/B-stationary only on a strictly lower wall,
  // so it stays output-stationary for compute-bound shapes.
  // Chained Mat-scratch producers pass force_output_stationary=true to turn these
  // off (see the #1908 guard at the fold site): the Mat-scratch offset-packing path
  // cannot yet pack a single-buffered A/B-stationary producer against the consumer
  // matmul's double-buffered operands, so those producers must stay output-stationary.
  cfg.allow_a_stationary = !force_output_stationary;
  cfg.allow_b_stationary = !force_output_stationary;
  // L0C double-buffering (dbC=2): the chooser budgets the accumulator at L0C/2 and
  // scores the drain-hidden wall (max(C_load, C_mad, C_drain)) so tile i's FIXPIPE
  // drain overlaps tile i+1's MAD.  This needs two *co-live* L0C accumulators, and
  // the co-live pair only survives under the ptoas memory planner: PTOAS skips
  // MemoryReuse, whose opportunistic reuse over-coalesces the two overlapping-live-
  // range accumulators back into one buffer.  With MemoryReuse skipped, InitMemRef
  // keeps the two buffers distinct and ptoas assigns their physical offsets (any
  // correct allocator must separate overlapping live ranges).
  //
  // Under the PyPTO planner dbC=2 is an experimental opt-in (PassContext flag,
  // default OFF). It now works there because the pipeline-membership tagger gives
  // the dbC accumulator a *flat depth-2* membership — only the moving (dbC) loop
  // tags it; enclosing loops skip it since the cube serializes MADs — so
  // MemoryReuse's capacity gate (#1475) allocates exactly the two co-live L0C
  // buffers instead of over-coalescing the pair (its former behaviour, which
  // shrank the tile to L0C/2 with no second buffer). Kept default-off pending
  // device validation of the numerics + drain-hidden win. The co-live emit is
  // gated on the chooser's `double_buffer_c` result below, which tags the moving
  // loop with kPipelineDoubleBufferCAttr.
  //
  // KNOWN-FRAGILE (experimental): this reads the memory planner from the *mutable*
  // mid-pipeline PassContext, so dbC=2 behaviour depends on run-time context state
  // rather than an immutable IR property. A nested PassContext once silently reset
  // it (fixed by propagating memory_planner + PassManager's fail-loud planner check),
  // but the underlying smell remains. The durable design is a first-class co-live /
  // no-coalesce Acc-buffer-pair IR property set once at emit and honoured by BOTH
  // planners. Tracked as a follow-up.
  const bool ptoas_planner = ctx && ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS;
  const bool pypto_dbc =
      ctx && ctx->GetMemoryPlanner() == MemoryPlanner::PyPTO && ctx->GetEnablePyptoL0cDoubleBuffer();
  cfg.allow_double_buffer_c = ptoas_planner || pypto_dbc;
  // tile.matmul_acc threads the caller's accumulator into the K-loop's
  // iter-arg, so each invocation reads C from L1 at start and writes back at
  // end (gamma_c = 2 in the chooser's traffic model).  Plain tile.matmul
  // starts from a fresh Acc placeholder so C is write-only (gamma_c = 1).
  cfg.c_read = is_matmul_acc;
  cfg.allow_padding = false;
  // Permit a non-divisor final K block: the chooser may return a k that does not
  // divide K, and BuildKLoopRewrite peels the partial last K iteration.  The peel
  // is only valid when K is 16-aligned — then the tail K - floor(K/k)*k is itself
  // 16-aligned (ptoas requires 16-aligned tile cols).  A non-16-aligned K has no
  // valid K-tiling (any tail or whole-K block has non-fractal cols), so skip it
  // here with a perf hint rather than emit invalid extracts (the pre-roofline path
  // also bailed on unsupported K).
  cfg.allow_k_boundary = true;
  if (K % cfg.align_k != 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-007",
                       "tile.matmul: K=" + std::to_string(K) + " is not a multiple of the cube fractal " +
                           std::to_string(cfg.align_k) +
                           " — non-16-aligned K is unsupported; left untouched.",
                       assign->span_);
    return std::nullopt;
  }

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

  // Already L0-sized.  Mat-fed matmuls need nothing.  A Vec-resident LHS still
  // needs an explicit Vec→Mat ``tile.move`` so ExpandMixedKernel sees a
  // ``CollectCVBoundaryMoves`` handshake (same as the K-tiled PV pattern).
  // Without it InferTileMemorySpace inserts Vec→Left directly; on A5 the V→C
  // ND→NZ adapt can then share the cast buffer and silently mis-transfer
  // (prefill_indexer Hadamard / fused cast→matmul Left).
  if (res.m == M && res.n == N && res.k == K) {
    if (lhs_tile->GetMemorySpace() != MemorySpace::Vec) return std::nullopt;
    MatmulTiling t;
    t.assign = assign;
    t.lhs = lhs;
    t.rhs = rhs;
    t.stage_lhs_to_mat = true;
    t.acc_init = acc_var;
    t.M = M;
    t.N = N;
    t.K = K;
    t.m = M;
    t.n = N;
    t.k = K;
    return t;
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
  t.stationarity = res.stationarity;
  t.os_holds_a = res.os_holds_a;
  // dbC=2 is realized only by the full-K emitter: BuildFullKPipelined attaches
  // kPipelineDoubleBufferCAttr, BuildSplitKGrid never does.  The chooser already
  // guarantees dbC ⇒ k == K (l0_tile_chooser's require_full_k), and it applied the
  // L0C/2 accumulator budget on that promise.  Assert the invariant here rather
  // than silently clamping (`&& k == K`): a clamp would drop the attr but keep the
  // L0C/2 budget, shipping a shrunk single-buffer tile — the exact regression this
  // feature exists to avoid.  A future chooser change that sets dbC on a split-K
  // tile must fail loudly instead.
  INTERNAL_CHECK_SPAN(!res.double_buffer_c || res.k == K, assign->span_)
      << "Internal error: chooser set double_buffer_c on a split-K tile (k=" << res.k << ", K=" << K
      << "); dbC=2 requires the full-K emitter";
  t.double_buffer_c = res.double_buffer_c;
  return t;
}

/// Stage-only rewrite for an already-L0-sized matmul whose LHS is Vec: insert
/// ``tile.move``→Mat and rewrite the matmul to consume the Mat tile.  No K/M/N
/// tiling — BuildKLoopRewrite requires ``k < K``.
std::pair<std::vector<StmtPtr>, VarPtr> BuildStageOnlyVecLhs(const MatmulTiling& t) {
  INTERNAL_CHECK_SPAN(t.stage_lhs_to_mat && t.m == t.M && t.n == t.N && t.k == t.K, t.assign->span_)
      << "Internal error: BuildStageOnlyVecLhs expects full-size Vec→Mat staging";
  const Span& sp = t.assign->span_;
  const std::string& base = t.assign->var_->name_hint_;
  std::vector<StmtPtr> out;
  auto lhs_mat = BuildMoveToMat(t.lhs, base + "_l0_lmat", sp);
  out.push_back(lhs_mat);
  auto& reg = OpRegistry::GetInstance();
  ExprPtr call = t.is_acc() ? reg.Create("tile.matmul_acc", {t.acc_init, lhs_mat->var_, t.rhs}, sp)
                            : reg.Create("tile.matmul", {lhs_mat->var_, t.rhs}, sp);
  auto cvar = std::make_shared<Var>(base, call->GetType(), sp);
  out.push_back(std::make_shared<AssignStmt>(cvar, call, sp));
  return {std::move(out), cvar};
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
  std::unordered_map<const Var*, int> counts;               ///< all reads
  std::unordered_map<const Var*, int> matmul_operand_uses;  ///< reads at a matmul-operand position

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    ++counts[op.get()];
    if (in_matmul_operand_) ++matmul_operand_uses[op.get()];
  }
  // Skip the LHS (a def); count only reads in the RHS value.
  void VisitStmt_(const AssignStmtPtr& op) override { VisitExpr(op->value_); }
  // A *direct* Var at a matmul OPERAND position (``tile.matmul`` args {0,1};
  // ``tile.matmul_acc`` args {1,2} — arg 0 is the Acc accumulator, NOT a matrix
  // operand) is a Mat-safe consumer use: the consumer K-tiles that operand, so an
  // L1/Mat scratch produced upstream is legal there.  Classifying by operand index
  // is essential — a scratch fed to ``matmul_acc`` arg 0 would be an illegal
  // Mat-for-Acc substitution and must stay deferred.
  void VisitExpr_(const CallPtr& op) override {
    const std::string& name = op->op_ ? op->op_->name_ : std::string();
    const bool is_mm = name == "tile.matmul";
    const bool is_acc = name == "tile.matmul_acc";
    for (size_t i = 0; i < op->args_.size(); ++i) {
      const bool operand_pos = (is_mm && (i == 0 || i == 1)) || (is_acc && (i == 1 || i == 2));
      const bool prev = in_matmul_operand_;
      in_matmul_operand_ = operand_pos && (AsVarLike(op->args_[i]) != nullptr);
      VisitExpr(op->args_[i]);
      in_matmul_operand_ = prev;
    }
  }

 private:
  bool in_matmul_operand_ = false;
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
  std::unordered_map<const Var*, int> matmul_operand_uses;  ///< reads at a matmul-operand position
  std::unordered_map<const Var*, const AssignStmt*> store_of;
  std::unordered_map<const Var*, const AssignStmt*> cast_of;  ///< ``cb = tile.cast(v, dtype)`` by source
};

SiblingIndex BuildSiblingIndex(const std::vector<StmtPtr>& stmts) {
  SiblingIndex idx;
  SiblingUseCounter counter;
  for (const auto& s : stmts) {
    counter.VisitStmt(s);
    auto as = std::dynamic_pointer_cast<const AssignStmt>(s);
    if (!as) continue;
    auto call = As<Call>(as->value_);
    if (!call || !call->op_) continue;
    // Record top-level 2D ``tile.store(src, offsets, out)`` by source operand.
    if (IsOp(call, "tile.store") && call->args_.size() == 3) {
      if (auto src = AsVarLike(call->args_[0])) idx.store_of.emplace(src.get(), as.get());
    } else if (IsOp(call, "tile.cast") && !call->args_.empty()) {
      // Record ``cb = tile.cast(src, dtype)`` by source: a chained matmul whose
      // result is downcast (Acc f32 -> bf16/f16) before the consumer matmul is
      // foldable into a low-precision Mat scratch (the cast = FIXPIPE writeback).
      if (auto src = AsVarLike(call->args_[0])) idx.cast_of.emplace(src.get(), as.get());
    }
  }
  idx.use_counts = std::move(counter.counts);
  idx.matmul_operand_uses = std::move(counter.matmul_operand_uses);
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

/// Mat-scratch placement (on-chip matmul consumers): keep the whole ``[M, N]``
/// result in an L1/Mat scratch instead of storing it to a DDR tensor, so a
/// matmul-operand consumer reads it on-chip.  ``Init`` creates the scratch (mirrors
/// ``BuildAccInit`` but in ``Mat``); each ``PlaceAt`` assembles a sub-tile in place:
/// ``scratch_{k+1} = tile.assemble(scratch_k, sub, [row_off, col_off])`` — Acc→Mat,
/// lowering to ``pto.subview`` + ``pto.tmov`` (the codegen landed in PR #1860).
/// ``tile.assemble`` is ``set_output_memory_inherit_input()``, so the chain shares
/// one Mat base before MemoryReuse runs (no full-scratch copy per insert).
///
/// ``tile.assemble``'s offset is a literal ``MakeTuple`` whose *elements* may be
/// loop variables (``ValidateIndexTupleElements`` only requires index-typed
/// elements, not constants), so this placer drives both the constant-offset
/// unrolled grid (``BuildSplitKGrid``, K-split) and the loop-variable pipelined
/// emitter (``BuildFullKPipelined``, full-K).
class MatScratchPlacer : public SubtilePlacer {
 public:
  MatScratchPlacer(int64_t big_m, int64_t big_n, DataType dtype, std::string base, Span span)
      : m_(big_m), n_(big_n), dtype_(dtype), base_(std::move(base)), sp_(std::move(span)) {}

  [[nodiscard]] VarPtr Init(std::vector<StmtPtr>& stmts) override {
    auto& reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", dtype_},
                                                            {"target_memory", MemorySpace::Mat}};
    auto call = reg.Create("tile.create", {MakeIndexTuple({m_, n_}, sp_)}, kwargs, sp_);
    auto scratch = std::make_shared<Var>(base_, call->GetType(), sp_);
    stmts.push_back(std::make_shared<AssignStmt>(scratch, call, sp_));
    return scratch;
  }

  [[nodiscard]] VarPtr PlaceAt(std::vector<StmtPtr>& stmts, const VarPtr& sub, const ExprPtr& row_off,
                               const ExprPtr& col_off, const VarPtr& chain_in, int step) override {
    auto& reg = OpRegistry::GetInstance();
    auto offs = std::make_shared<MakeTuple>(std::vector<ExprPtr>{row_off, col_off}, sp_);
    auto call = reg.Create("tile.assemble", {chain_in, sub, offs}, sp_);
    auto sv = std::make_shared<Var>(base_ + "_t" + std::to_string(step), call->GetType(), sp_);
    stmts.push_back(std::make_shared<AssignStmt>(sv, call, sp_));
    return sv;
  }

 private:
  int64_t m_, n_;
  DataType dtype_;
  std::string base_;
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

  // Loop order + buffering. When the chooser picked an operand-stationary point
  // (k == K), the stationary operand is the OUTER (held) panel and is SINGLE-
  // buffered — the chooser budgeted it the full L0 buffer (no /2). A-stationary
  // -> A held -> rows outer; B-stationary -> B held -> cols outer. Output-
  // stationary double-buffers both and hoists one operand to the outer loop; the
  // chooser already made that bandwidth-weighted choice while scoring the wall
  // (LoadCycles' min-hoist) and recorded it in `os_holds_a`, so we obey it here
  // rather than re-derive from raw bytes — the two objectives disagree under the
  // ~200:132 L0A:L0B bandwidth ratio, and diverging would emit a different loop
  // order than the wall was scored under.
  const bool a_stationary = t.stationarity == utils::Stationarity::kAStationary;
  const bool b_stationary = t.stationarity == utils::Stationarity::kBStationary;
  const bool stationary_single_buffered = a_stationary || b_stationary;
  const bool row_outer = stationary_single_buffered ? a_stationary : t.os_holds_a;

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
    // overlap_stores stays false: the one-accumulator schedule
    // (matmul_i, store_i, matmul_{i+1}, store_{i+1}) drains each L0C result before
    // the next matmul overwrites it.  dbC=2 (double_buffer_c) instead sets the
    // stronger double_buffer_c attr, which floats *both* stores below *both*
    // matmuls (matmul c, matmul c₁, store c, store c₁) so the two [m, n] Acc tiles
    // stay co-live; the chooser budgeted them at L0C/2 so both fit.  With MemoryReuse
    // skipped (ptoas planner) InitMemRef keeps the two overlapping-live-range buffers
    // distinct and ptoas places them, so tile i's FIXPIPE drain overlaps tile i+1's
    // MAD.  Absent (⇒ false) under the pypto planner, where MemoryReuse coalesces the
    // pair to one accumulator.  The moving-operand extract is double-buffered (Load
    // tier, hoisted) in both schedules.
    std::vector<std::pair<std::string, std::any>> inner_attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2},
                                                                 {kPipelineOverlapStoresAttr, false}};
    // Only dbC=2 loops carry the attr (absent ⇒ false), so non-dbC=2 emit is
    // unchanged.  It lifts the moving-loop stores above both matmuls in
    // CanonicalizeIOOrder to keep the two L0C accumulators co-live.
    if (t.double_buffer_c) inner_attrs.emplace_back(kPipelineDoubleBufferCAttr, true);
    auto inner_rv = std::make_shared<Var>(base + "_irv", out_type, sp);
    auto inner_for = std::make_shared<ForStmt>(
        inner_var, MakeIndex(0, sp), MakeIndex(inner_extent, sp), MakeIndex(inner_step, sp),
        std::vector<IterArgPtr>{out_inner}, SeqStmts::Flatten(std::move(inner_body), sp),
        std::vector<VarPtr>{inner_rv}, sp, ForKind::Pipeline, std::move(inner_attrs));
    std::vector<StmtPtr> outer_body{outer_extract, inner_for,
                                    std::make_shared<YieldStmt>(std::vector<ExprPtr>{inner_rv}, sp)};
    // Operand-stationary: the outer loop carries the SINGLE-buffered stationary
    // panel, so it is Sequential — a Pipeline stage=2 outer would double-buffer the
    // held operand (2x its full-L0 budget -> overflow). The inner (moving) loop
    // stays pipelined. Output-stationary double-buffers both -> outer Pipeline.
    const ForKind outer_kind = stationary_single_buffered ? ForKind::Sequential : ForKind::Pipeline;
    std::vector<std::pair<std::string, std::any>> outer_attrs;
    if (!stationary_single_buffered) {
      outer_attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2}, {kPipelineOverlapStoresAttr, false}};
    }
    auto outer_rv = std::make_shared<Var>(base + "_orv", out_type, sp);
    auto outer_for = std::make_shared<ForStmt>(
        outer_var, MakeIndex(0, sp), MakeIndex(outer_extent, sp), MakeIndex(outer_step, sp),
        std::vector<IterArgPtr>{out_outer}, SeqStmts::Flatten(std::move(outer_body), sp),
        std::vector<VarPtr>{outer_rv}, sp, outer_kind, std::move(outer_attrs));
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
  // The chooser returns k == K exactly when the full K reduction fits one L0
  // block (possibly an unaligned k < align under allow_k_boundary) — only then
  // is there no K-loop.  A non-divisor k with k < K < 2k still needs the
  // K-loop+peel, so test k == K rather than the integer-division proxy K/k < 2.
  const bool full_k = t.k == t.K;

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

/// True when a ``tile.cast`` may be folded into a cube FIXPIPE Acc->Mat writeback
/// (``pto.tinsert``) instead of a standalone Vector ``pto.tcvt``.  FIXPIPE narrows
/// only ``f32 -> bf16`` / ``f32 -> f16`` (the ``F322BF16`` / ``F322F16`` writeback
/// modes; an ``int32`` source would need a *scaled dequant*, not a plain cast) and
/// applies a single fixed tie rule: **round-to-nearest-even** — the pto-isa CPU
/// reference narrows via ``std::bfloat16_t`` and ``pto.tinsert`` carries no
/// ``rmode``.  So fold only an ``f32`` source cast to ``bf16``/``f16`` whose round
/// mode is ``RINT`` (round-half-to-even).  ``ROUND`` (round-half-*away*, the
/// frontend default) and the directional/truncating modes round ties differently,
/// so they must keep the Vector cast — it lowers to ``pto.tcvt``, the only path
/// that honors the requested ``rmode``.
bool CastFoldableToFixpipeMat(const CallPtr& cast, const TileTypePtr& src_ty, DataType dst_dtype) {
  if (!cast || !src_ty) return false;
  if (src_ty->dtype_ != DataType::FP32) return false;
  if (dst_dtype != DataType::BF16 && dst_dtype != DataType::FP16) return false;
  // ``tile.cast`` "mode" (see src/ir/op/tile_ops/unary.cpp): NONE(0), RINT(1),
  // ROUND(2), FLOOR(3), CEIL(4), TRUNC(5), ODD(6).  FIXPIPE's fixed narrowing is
  // round-half-to-even == RINT; only that mode matches.  A missing "mode" defaults
  // to the frontend's ROUND (ties away) — not foldable.
  constexpr int kRoundRint = 1;
  constexpr int kRoundRound = 2;
  const int mode = cast->GetKwarg<int>("mode", kRoundRound);
  return mode == kRoundRint;
}

/// Try to fold a Mat-resident plain ``tile.matmul`` whose [M, N] output exceeds
/// L0c into a Mat-scratch grid when the result is consumed *entirely* at
/// matmul-operand positions (a chained matmul reads it on-chip).  Each sub-tile is
/// assembled into an L1/Mat scratch (``MatScratchPlacer``) instead of stored to a
/// DDR tensor, keeping the whole result on-chip; the caller remaps the matmul
/// result Var to the returned scratch Var.  Returns the grid stmts + scratch Var.
///
/// Both K-split (unrolled, constant offsets) and full-K (pipelined, loop-variable
/// offsets) are supported: ``tile.assemble`` only needs a literal ``MakeTuple``
/// offset whose *elements* may be loop variables (`ValidateIndexTupleElements`
/// requires index-typed elements, not constants).  ``matmul_acc`` and Vec-left
/// stay deferred.
std::optional<std::pair<std::vector<StmtPtr>, VarPtr>> TryFoldMatScratch(const MatmulTiling& t,
                                                                         int result_uses, int operand_uses,
                                                                         DataType scratch_dtype,
                                                                         std::vector<Diagnostic>& hints) {
  const Span sp = t.assign->span_;
  // matmul_acc / Vec-left are deferred (the direct-store path already hinted these).
  if (t.is_acc() || t.stage_lhs_to_mat) return std::nullopt;
  // Every use must be a matmul operand: a non-operand use (store, elementwise,
  // matmul_acc accumulator) means substituting an upstream Mat scratch is illegal.
  if (result_uses < 1 || operand_uses != result_uses) return std::nullopt;
  // On backends whose only offset Acc->Mat path is the FIXPIPE writeback
  // (`pto.tinsert`), that path downcasts f32 -> bf16/f16 and cannot keep f32 — a
  // same-dtype f32 Acc->Mat assemble lowers to subview+tmov, which the assembler
  // rejects for a partial window. So an oversized chained-matmul scratch must be
  // bf16/f16 there (the dtype comes from a `tile.cast(result, bf16/f16)` fused
  // into the assemble, the cube's native operand precision); without it, defer
  // (left whole) rather than emit an unassemblable f32 Mat scratch. A5's tinsert
  // accepts dst=f32, so its handler returns false and an f32 scratch is kept.
  const auto* ctx = PassContext::Current();
  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  const bool requires_low_precision = handler && handler->RequiresLowPrecisionMatScratch();
  if (requires_low_precision && scratch_dtype != DataType::BF16 && scratch_dtype != DataType::FP16) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-009",
                       "chained-matmul [" + std::to_string(t.M) + ", " + std::to_string(t.N) +
                           "] intermediate is " + scratch_dtype.ToString() +
                           "; this backend's oversized on-chip Mat scratch needs a bf16/f16 "
                           "intermediate (cast the matmul result to bf16 before the consumer "
                           "matmul, the cube's native operand precision) — left on the deferred path",
                       sp);
    return std::nullopt;
  }
  auto result_ty = As<TileType>(t.assign->var_->GetType());
  INTERNAL_CHECK_SPAN(result_ty, sp) << "Internal error: matmul result is not a TileType";
  // Necessary capacity gate: the Mat scratch alone must fit. The allocator still
  // performs the full live-range/packing check later.
  const uint64_t mat_capacity = handler ? handler->GetMatCapacityBytes() : 0;
  const uint64_t scratch_bytes =
      static_cast<uint64_t>(t.M) * static_cast<uint64_t>(t.N) * DTypeBytes(scratch_dtype);
  if (mat_capacity > 0 && scratch_bytes > mat_capacity) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
                       "chained-matmul [" + std::to_string(t.M) + ", " + std::to_string(t.N) +
                           "] Mat scratch (" + std::to_string(scratch_bytes) +
                           " bytes) exceeds Mat capacity (" + std::to_string(mat_capacity) +
                           " bytes); left on the deferred path",
                       sp);
    return std::nullopt;
  }
  const std::string base = t.assign->var_->name_hint_ + "_mat";
  MatScratchPlacer placer(t.M, t.N, scratch_dtype, base, sp);
  // K-split (K spans >= 2 L0 blocks) → unrolled per-sub-tile K-loop grid; full-K →
  // the pipelined interior + straight-line tail.  Both drive MatScratchPlacer,
  // which assembles each sub-tile into the L1/Mat scratch (tile.assemble accepts
  // constant or loop-variable offsets).  A non-divisor k with k < K < 2k still needs
  // the K-loop + peel (BuildSplitKGrid), so dispatch on k == K rather than the
  // integer-division proxy K/k < 2 — which would mis-route a split-K tile to the
  // full-K [m,K]/[K,n] emitter and blow the L0A/L0B budget.
  // dbC=2 works on the Mat-scratch path too: the Acc->Mat drain is `tile.assemble`,
  // which CanonicalizeIOOrder floats above the compute tier under the dbC attr (same
  // as tile.store for the direct-store path), keeping the two accumulators co-live.
  // BuildFullKPipelined attaches the attr when t.double_buffer_c; the split-K grid
  // never carries it.  (The Acc->Mat drain is cheaper than Acc->GM, so the hiding
  // upside is smaller here, but the mechanism is the same.)
  const bool full_k = t.k == t.K;
  auto [stmts, scratch] = full_k ? BuildFullKPipelined(t, placer) : BuildSplitKGrid(t, placer);
  return std::make_pair(std::move(stmts), scratch);
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
    // Defs to drop: a downcast ``cb = tile.cast(c, bf16)`` whose matmul+cast was
    // folded into a low-precision Mat scratch — the scratch already holds the
    // bf16 intermediate, so the now same-dtype cast is a dead no-op to remove.
    std::unordered_set<const Var*> dropped;
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

      // A downcast whose matmul+cast was folded into a Mat scratch (its def is in
      // ``dropped``): skip it — the scratch already holds the bf16 intermediate and
      // re-emitting ``cb = tile.cast(scratch_bf16, bf16)`` is an invalid no-op cast.
      if (auto as = std::dynamic_pointer_cast<const AssignStmt>(child); as && dropped.count(as->var_.get())) {
        changed = true;
        continue;
      }

      // Apply the running remap to redirect prior rewrites' downstream uses.
      StmtPtr current = remap.empty() ? child : transform_utils::Substitute(child, remap);

      // Fits-L0c chained-matmul cast-fold: rewrite ``cb = tile.cast(src_acc,
      // bf16/f16)`` into a full-window Acc->Mat scratch (``tile.create`` +
      // ``tile.assemble``) when every use of ``cb`` is a matmul operand.  This
      // routes the f32->bf16 downcast through the cube FIXPIPE (``pto.tinsert``)
      // instead of the Vector (``pto.tcvt``) — the fits-L0c analogue of the
      // oversized per-sub-tile Mat-scratch fold (``TryFoldMatScratch``).  The
      // matmul producing ``src`` is K-tiled (or left untouched) by the dispatch
      // below; here we only redirect the cast's result onto a Mat scratch and
      // drop the now-dead cast.  Oversized chains never reach here — their cast
      // is dropped via ``dropped`` at the matmul site (see ``TryFoldMatScratch``
      // remap above), so this only fires for results that fit L0c.
      if (auto cast_as = std::dynamic_pointer_cast<const AssignStmt>(current)) {
        auto cast = As<Call>(cast_as->value_);
        auto cb_ty = As<TileType>(cast_as->var_->GetType());
        if (cast && IsOp(cast, "tile.cast") && !cast->args_.empty() && cb_ty) {
          auto src = AsVarLike(cast->args_[0]);
          auto src_ty = src ? As<TileType>(src->GetType()) : nullptr;
          const bool src_acc = src_ty && src_ty->memory_space_ == MemorySpace::Acc;
          // Only fold what FIXPIPE can reproduce: f32 -> bf16/f16, round-to-nearest.
          const bool fixpipe_castable = CastFoldableToFixpipeMat(cast, src_ty, cb_ty->dtype_);
          // Static [M, N] — a fits-L0c chained matmul result is always static here.
          auto m_ci = cb_ty->shape_.size() == 2 ? As<ConstInt>(cb_ty->shape_[0]) : nullptr;
          auto n_ci = cb_ty->shape_.size() == 2 ? As<ConstInt>(cb_ty->shape_[1]) : nullptr;
          // Only fold when the Acc result fits L0c.  An oversized result is
          // Case 2's domain: ``TryFoldMatScratch`` folds the cast into per-sub-tile
          // assembles (and drops it), or defers it when the scratch exceeds Mat
          // capacity — in which case the cast must stay (we must not collapse an
          // oversized [M, N] into one impossible full-window assemble here).
          const auto* bh_ctx = PassContext::Current();
          const auto* bh = bh_ctx ? bh_ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
          const uint64_t l0c_bytes = bh ? bh->GetL0cCapacityBytes() : 0;
          const uint64_t acc_bytes = src_acc && m_ci && n_ci ? static_cast<uint64_t>(m_ci->value_) *
                                                                   static_cast<uint64_t>(n_ci->value_) *
                                                                   DTypeBytes(src_ty->dtype_)
                                                             : 0;
          if (src_acc && m_ci && n_ci && l0c_bytes && acc_bytes <= l0c_bytes) {
            if (!sibling_index) sibling_index = BuildSiblingIndex(op->stmts_);
            const Var* cb = cast_as->var_.get();
            auto uc = sibling_index->use_counts.find(cb);
            auto mo = sibling_index->matmul_operand_uses.find(cb);
            const int cb_uses = uc == sibling_index->use_counts.end() ? 0 : uc->second;
            const int cb_mm = mo == sibling_index->matmul_operand_uses.end() ? 0 : mo->second;
            // A chained cast: every use of ``cb`` is a matmul operand, so the bf16
            // value can live entirely in Mat (a store / elementwise consumer could
            // not read it there, so those keep the Vector cast path regardless).
            if (cb_uses >= 1 && cb_uses == cb_mm) {
              if (fixpipe_castable) {
                MatScratchPlacer placer(m_ci->value_, n_ci->value_, cb_ty->dtype_, src->name_hint_ + "_mat",
                                        cast_as->span_);
                VarPtr scratch = placer.Init(out);
                VarPtr cmat = placer.PlaceAt(out, src, MakeIndex(0, cast_as->span_),
                                             MakeIndex(0, cast_as->span_), scratch, /*step=*/0);
                remap[cb] = cmat;  // the consumer matmul now reads the Mat scratch
                changed = true;
                continue;  // drop the dead tile.cast
              }
              // Chained cast that fits L0c but FIXPIPE cannot reproduce (a non-f32
              // accumulator, or a round mode other than round-half-to-even): keep
              // the standalone Vector cast and warn — it stays a cube->vector->cube
              // round-trip that overflows the Vec buffer at large [M, N].
              hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-010",
                                 "chained-matmul [" + std::to_string(m_ci->value_) + ", " +
                                     std::to_string(n_ci->value_) + "] cast to " + cb_ty->dtype_.ToString() +
                                     " cannot fold onto the cube FIXPIPE (which narrows f32->bf16/f16 with "
                                     "round-half-to-even only); kept on the Vector path (pto.tcvt), a "
                                     "cube->vector->cube round-trip that may overflow the Vec buffer at this "
                                     "[M, N] — cast an f32 result with mode=\"rint\" to keep it on the cube",
                                 cast_as->span_);
              // fall through: the tile.cast stays (Vector pto.tcvt)
            }
          }
        }
      }

      // Check if this is a matmul we rewrite *at this SeqStmts level*.  We
      // try this before recursive visitation so the rewrite — which produces
      // a sequence of stmts — lands in this enclosing SeqStmts.  Recursive
      // visitation happens after rewrite-rejection so nested matmuls inside
      // ForStmt bodies still get rewritten by the recursive visit.
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(current)) {
        if (auto tiling = AnalyzeMatmul(assign, hints)) {
          if (!tiling->needs_mn_tiling()) {
            // Whole output fits L0c.  k == K is the already-L0 Vec-LHS stage-only
            // path (Mat LHS returns nullopt from AnalyzeMatmul); k < K tiles K
            // (chooser may return a non-divisor k that BuildKLoopRewrite peels).
            if (tiling->k == tiling->K) {
              INTERNAL_CHECK_SPAN(tiling->stage_lhs_to_mat, tiling->assign->span_)
                  << "Internal error: full-K L0-sized rewrite expects Vec→Mat staging";
              auto [stmts, ret] = BuildStageOnlyVecLhs(*tiling);
              remap[assign->var_.get()] = ret;
              for (auto& s : stmts) out.push_back(std::move(s));
              changed = true;
              continue;
            }
            INTERNAL_CHECK_SPAN(tiling->k < tiling->K, tiling->assign->span_)
                << "Internal error: K-only tiling expects k < K (K=" << tiling->K << ", k=" << tiling->k
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
          int result_uses = uc_it == sibling_index->use_counts.end() ? 0 : uc_it->second;
          auto mo_it = sibling_index->matmul_operand_uses.find(result);
          int operand_uses = mo_it == sibling_index->matmul_operand_uses.end() ? 0 : mo_it->second;
          // Mat-scratch dtype + remap target. Default: the matmul result itself at
          // its own dtype. Chained-matmul-with-downcast — `c -> tile.cast(c,
          // bf16/f16) -> matmul` — fuses the cast (the cube FIXPIPE writeback,
          // pto.tinsert) into the scratch: the scratch holds the bf16/f16
          // intermediate, the per-sub-tile assemble downcasts Acc f32 -> Mat bf16,
          // and the cast result is remapped to the scratch so the consumer matmul
          // reads it on-chip (the cast op then goes dead).
          auto result_tile_ty = As<TileType>(result->GetType());
          DataType scratch_dtype = result_tile_ty->dtype_;
          const Var* remap_target = result;
          const Var* extra_remap = nullptr;
          if (auto cast_it = sibling_index->cast_of.find(result); cast_it != sibling_index->cast_of.end()) {
            const Var* cb = cast_it->second->var_.get();
            auto cb_ty = As<TileType>(cb->GetType());
            auto cast_call = As<Call>(cast_it->second->value_);
            // Fold the downcast into the FIXPIPE Acc->Mat writeback only when FIXPIPE
            // can reproduce it: f32 (the matmul Acc) -> bf16/f16, round-to-nearest.
            // Otherwise keep the standalone Vector cast (e.g. an int accumulator, or a
            // directional/truncating round mode FIXPIPE has no `rmode` for).
            if (cb_ty && CastFoldableToFixpipeMat(cast_call, result_tile_ty, cb_ty->dtype_)) {
              auto cb_uc = sibling_index->use_counts.find(cb);
              auto cb_mo = sibling_index->matmul_operand_uses.find(cb);
              const int cb_uses = cb_uc == sibling_index->use_counts.end() ? 0 : cb_uc->second;
              const int cb_mm = cb_mo == sibling_index->matmul_operand_uses.end() ? 0 : cb_mo->second;
              // `c`'s sole use is the cast, whose result is consumed entirely as
              // matmul operands — fold the matmul+cast into a low-precision scratch.
              if (result_uses == 1 && cb_uses >= 1 && cb_uses == cb_mm) {
                scratch_dtype = cb_ty->dtype_;
                remap_target = cb;     // consumer matmul reads the scratch
                extra_remap = result;  // c -> scratch too; the cast op goes dead
                result_uses = cb_uses;
                operand_uses = cb_mm;
              }
            }
          }
          // Mat-scratch: result consumed entirely on-chip at matmul-operand
          // positions — assemble the sub-tiles into an L1/Mat scratch and remap the
          // matmul result (or its downcast) to it.  Emitted at the matmul site
          // (like the K-only rewrite), with no store to defer.  Checked before the
          // direct-store fold so its hints stay clean.
          // #1908 guard: a chained Mat-scratch producer must stay output-stationary.
          // The Mat-scratch offset-packing path cannot yet pack an A/B-stationary
          // producer (its held operand is a monolithic single-buffered [m,K]/[K,n]
          // panel in the full L0 buffer) against the consumer matmul's double-buffered
          // operands, so an A/B-stationary chained producer overflows at
          // AllocateMemoryAddr. OS is always a legal tile and the oversized producer
          // must be tiled (deferring would overflow L0c), so re-choose OS-only for the
          // fold rather than defer. Remove once offset packing lands.
          const MatmulTiling* fold_tiling = &*tiling;
          std::optional<MatmulTiling> os_tiling;
          if (tiling->stationarity != utils::Stationarity::kOutputStationary) {
            std::vector<Diagnostic> discard;  // the first AnalyzeMatmul already emitted the hints
            os_tiling = AnalyzeMatmul(assign, discard, /*force_output_stationary=*/true);
            if (os_tiling) fold_tiling = &*os_tiling;
          }
          if (auto ms = TryFoldMatScratch(*fold_tiling, result_uses, operand_uses, scratch_dtype, hints)) {
            for (auto& s : ms->first) out.push_back(std::move(s));
            remap[remap_target] = ms->second;
            if (extra_remap) {
              remap[extra_remap] = ms->second;  // c -> scratch (cast reads the scratch)
              dropped.insert(remap_target);     // ... and drop the now-dead cast def
            }
            changed = true;
            continue;
          }
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
