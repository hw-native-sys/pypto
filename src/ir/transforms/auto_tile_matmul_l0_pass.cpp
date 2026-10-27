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
/// For each ``tile.matmul``, ``tile.matmul_acc``, or ``tile.matmul_bias`` with
/// static 2D operands, picks an L0 tile shape ``(m, n, k)`` from the active ``BackendHandler``'s
/// L0 capacities (via ``utils::ChooseL0Tile``) and rewrites the call into a
/// K-loop.  The right (B) operand must be ``Mat``-resident; the left (A)
/// operand may be ``Mat`` (the QK pattern) or ``Vec`` (the fused-attention
/// ``score·V`` / PV pattern, where the softmax output crosses the cube↔vector
/// boundary resident in ``Vec``).  Tiling the Vec-fed PV matmul symmetrically
/// with QK makes its L0B right buffer a reusable sub-tile so ``MemoryReuse``
/// can alias it onto QK's freed L0B (peak L0B = ``max(QK, PV)`` instead of the
/// sum).  The K-loop has the shape:
///
///   * ``tile.matmul`` — the loop body is a single predicated
///     ``tile.matmul_acc(c_iter, sa, sb, ko == 0)``.  The iter-arg init is an
///     Acc-resident ``tile.create`` placeholder that the predicate overwrites
///     on the first L0 block (and accumulates into afterwards), so the
///     iter-arg / yield / return_var chain is Acc-typed end-to-end and,
///     because ``tile.matmul_acc`` declares ``set_output_reuses_input(0)``,
///     lands on one L0C buffer by construction.
///   * ``tile.matmul_acc`` — every iteration is ``tile.matmul_acc``; the
///     iter-arg init is the caller-provided accumulator directly, so the
///     chain is uniform and no if-else is needed.  A caller-supplied
///     ``init_cond`` (the 4-operand spelling) is threaded through: the loop
///     body ANDs it with the generated ``ko == 0``, a lone straight-line full
///     block passes it through unchanged, and the peeled partial tail drops it
///     (the tail is never the first K block, so the predicate is dead there).
///   * ``tile.matmul_bias`` — the first iteration uses ``tile.matmul_bias``;
///     subsequent iterations use ``tile.matmul_acc``, so bias is applied once.
///     This is the only remaining generator of the ``ko == 0`` branch:
///     ``tile.matmul_bias`` carries no ``init_cond`` operand, so the first K
///     block has to be peeled to apply the bias exactly once.
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
///   c_init = tile.create([m, n], dtype, target_memory=Acc)  // accumulator seed
///   for ko in pl.pipeline(0, K, k, init_values=(c_init,), stage=2):
///     sa = tile.extract(x_mat, 0, ko, [m, k], target_memory=Left)
///     sb = tile.extract(y_mat, ko, 0, [k, n], target_memory=Right)
///     c = tile.matmul_acc(c_iter, sa, sb, ko == 0)  // overwrite on block 0
///     yield c
///
/// Layout for ``tile.matmul_acc`` (acc_init is the caller's accumulator):
///   for ko in pl.pipeline(0, K, k, init_values=(acc_init,), stage=2):
///     sa = tile.extract(x_mat, 0, ko, [m, k], target_memory=Left)
///     sb = tile.extract(y_mat, ko, 0, [k, n], target_memory=Right)
///     c_new = tile.matmul_acc(c_iter, sa, sb)
///     yield c_new
///
/// Layout for a *predicated* ``tile.matmul_acc(acc_init, x, y, user_cond)``.
/// ``user_cond`` means "this is the first K step of the user's reduction";
/// ``ko == 0`` means "the first L0 block of that step".  The accumulator is
/// overwritten only where both hold:
///   for ko in pl.pipeline(0, k_full, k, init_values=(acc_init,), stage=2):
///     sa = tile.extract(x_mat, 0, ko, [m, k], target_memory=Left)
///     sb = tile.extract(y_mat, ko, 0, [k, n], target_memory=Right)
///     c_new = tile.matmul_acc(c_iter, sa, sb, user_cond and ko == 0)
///     yield c_new
///   // peeled partial tail (k does not divide K): 3-operand, unpredicated —
///   // it runs at K offset k_full > 0, so it is never the first block.
///   ct = tile.matmul_acc(c_kmain, sat, sbt)
///
/// Layout for ``tile.matmul_bias`` — the one shape that still branches, since
/// the bias has to be applied on the first K block only:
///   c_init = tile.create([m, n], dtype, target_memory=Acc)  // placeholder
///   for ko in pl.pipeline(0, K, k, init_values=(c_init,), stage=2):
///     sa = tile.extract(x_mat, 0, ko, [m, k], target_memory=Left)
///     sb = tile.extract(y_mat, ko, 0, [k, n], target_memory=Right)
///     if ko == 0:
///       c1 = tile.matmul_bias(sa, sb, bias)  // fresh Acc + bias
///       c_phi = pl.yield_(c1)                // if's return_var
///     else:
///       c2 = tile.matmul_acc(c_iter, sa, sb) // accumulate
///       c_phi = pl.yield_(c2)
///     yield c_phi
///
/// A fresh return_var typed identically to the iter-arg replaces the original
/// matmul's Var; uses of the original Var in the enclosing SeqStmts are
/// substituted by the mutator.
///
/// M/N tiling (output exceeds L0c)
/// -------------------------------
/// When ``ChooseL0Tile`` returns ``m < M`` or ``n < N`` the ``[M, N]`` output
/// Acc's physical footprint overflows L0c.  Capacity uses the backend's
/// accumulator-row alignment, which may be stricter than the logical M shape
/// (Ascend910 INT32 M=16 occupies 32 physical rows).  The operands are already
/// Mat-resident, so only the output overflows. For fresh ``tile.matmul`` or
/// ``tile.matmul_bias``, the pass emits a
/// ``ceil(M/m) x ceil(N/n)`` grid and hands each ``[m_eff, n_eff]`` Acc result to
/// a placement strategy: either direct-store to the sole 2D
/// ``tile.store(c, base, out)`` consumer, or assembly into an on-chip Mat scratch
/// when every use is a later matmul operand. Each sub-tile uses the pipelined
/// K-loop above when K spans >= 2 L0 blocks, or — when ``k == K`` (the full K
/// fits L0a/L0b at once) — a single straight-line matmul emitted inside
/// **nested pipelined loops** over the divisible interior so
/// ``LowerPipelineLoops`` double-buffers the operand extracts (see
/// ``BuildFullKPipelined``). The partial boundary is peeled into a
/// straight-line tail, so ``m`` / ``n`` need not divide ``M`` / ``N``.
///
/// Supported today:
///   * ``tile.matmul``, ``tile.matmul_acc``, and ``tile.matmul_bias``. For a
///     biased matmul, each bias window is reloaded along N and applied exactly
///     once on
///     the first K block of each output sub-tile.
///   * K tiling (``m == M and n == N``) for all three calls; M/N tiling for
///     fresh ``tile.matmul`` / ``tile.matmul_bias`` with either a direct-store
///     or Mat-scratch placement, with a pipelined K-loop or a straight-line
///     single-K-block (``k == K``) per sub-tile.
///     The canonical frontend split-K create/pipeline/store form is M/N-tiled
///     outside its K loop: each output sub-tile completes the whole source K
///     reduction before the next one starts. Both source spellings of that
///     form are recognized and tiled identically -- the peeled
///     ``if ko == 0 { tile.matmul } else { tile.matmul_acc }`` and the
///     predicated single ``tile.matmul_acc(acc, lhs, rhs, ko == 0)``. The
///     retiled output keeps whichever spelling the source used; only the two
///     statements at the tail of the K-loop body differ between them.
///     Arbitrary standalone ``tile.matmul_acc`` (which would need slices of a
///     caller-owned accumulator), a Vec left operand, or a mixed/non-matmul
///     on-chip consumer is deferred with a ``PerfHint``.
///   * A compatible f32-to-bf16/f16 ``tile.cast(mode="rint")`` feeding only
///     matmul operands is folded into the Acc-to-Mat FIXPIPE writeback.
///   * Any 16-aligned K.  When the chosen ``k`` does not divide ``K``
///     (``allow_k_boundary``) the K-loop peels a partial last block of width
///     ``K - (K/k)*k``; with K and k both 16-aligned the tail is itself 16-aligned
///     -- an ordinary matmul_acc block (ptoas requires 16-aligned tile cols).
///
/// When the chooser returns ``(M, N, K)``, the matmul itself needs no tiling
/// rewrite.  Its result may still participate in the fits-L0c chained
/// cast-fold, which remaps the compatible downcast to a full-window Mat
/// scratch.

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
#include "pypto/ir/cast_saturation.h"
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
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/acc_init_builder.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/auto_tile_matmul_l0/internal.h"

namespace pypto {
namespace ir {

namespace {

using transform_utils::PreserveCallAttrs;

using auto_tile_matmul_l0::AlignStaticExtent;
using auto_tile_matmul_l0::AnalyzeMatmul;
using auto_tile_matmul_l0::BuildPipelineDbCPlan;
using auto_tile_matmul_l0::DirectDefMap;
using auto_tile_matmul_l0::DTypeBytes;
using auto_tile_matmul_l0::IsStatic2DInSpaces;
using auto_tile_matmul_l0::IsStatic2DTuple;
using auto_tile_matmul_l0::kPassName;
using auto_tile_matmul_l0::MatmulKind;
using auto_tile_matmul_l0::MatmulTiling;
using auto_tile_matmul_l0::SiblingUseCounter;

ExprPtr MakeIndex(int64_t v, const Span& span) {
  return std::make_shared<ConstInt>(v, DataType::INDEX, span);
}

ExprPtr MakeIndexTuple(const std::vector<int64_t>& values, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(values.size());
  for (auto v : values) elements.push_back(MakeIndex(v, span));
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

/// Build a ``tile.extract(source, idx_row, idx_col, [shape],
/// target_memory=target)`` AssignStmt — the Mat→Left/Right SSA-form
/// extract used inside the K-loop.  Offsets are passed as separate scalar
/// exprs (typically a ConstInt 0 for the static axis and the loop var
/// ``ko`` for the K axis).  The result tile is already in the destination
/// memory space, so no follow-up ``tile.mov`` is needed.  The source is
/// always Mat-resident — a Vec-fed left operand is first staged into Mat by
/// ``BuildMove(..., Mat)`` (see ``BuildKLoopRewrite``).
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

/// Build a ``tile.move(source, target_memory=target)`` AssignStmt.
///
/// Moving to Mat matters for fused cube+vector roots (fused-attention PV /
/// ``score·V``):
/// the softmax/``exp`` output reaches the matmul resident in ``Vec`` at the
/// cube↔vector boundary.  Keeping the boundary crossing a ``tile.move`` lets
/// ``ExpandMixedKernel`` recognise it (``CollectCVBoundaryMoves`` only matches
/// ``tile.move``) and lower it to the cross-core ``tpop_from_aiv`` handshake
/// (which lands the data in Mat — ``GetBoundaryTpopMemory(AIC) == Mat``).
/// Extracting straight from the Vec tile instead would leave the operand a
/// dangling cross-boundary free variable on the cube side.
AssignStmtPtr BuildMove(const VarPtr& source, MemorySpace target, const std::string& name_hint,
                        const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {{"target_memory", target}};
  auto call = reg.Create("tile.move", {source}, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Build the ``tile.create([m, n], dtype, target_memory=Acc)`` placeholder
/// that initializes the iter-arg.  Acc keeps the iter-arg / yield / return_var
/// chain structurally consistent with the per-iter ``tile.matmul[_acc]``
/// outputs, so subsequent matmul_acc consumers (and any nested for-loops
/// initialised from this return_var) still see an Acc-typed accumulator and
/// can be tiled in turn.
AssignStmtPtr BuildAccInit(int64_t m, int64_t n, const DataType& dtype, const std::string& name_hint,
                           const Span& span, bool compact = false) {
  return acc_init::BuildAccStorage({MakeIndex(m, span), MakeIndex(n, span)}, dtype, name_hint, span, compact);
}

using AccInitValue = acc_init::AccInit;

/// Build an Acc placeholder whose allocation may be box-padded while its valid
/// rectangle remains the logical output tile.  Delegates to
/// ``acc_init::BuildNarrowedAccInit``, which ``narrow_loop_carry`` also uses to
/// re-declare a user-written accumulator seed once a loop proves its carry is
/// narrower, so the two cannot drift on the compact rule.
AccInitValue BuildAccInitWithValidShape(int64_t physical_m, int64_t physical_n, ExprPtr valid_m,
                                        ExprPtr valid_n, const DataType& dtype, const std::string& name_hint,
                                        const Span& span) {
  INTERNAL_CHECK_SPAN(valid_m && valid_n, span)
      << "Internal error: accumulator initializer requires two valid extents";
  return acc_init::BuildNarrowedAccInit({MakeIndex(physical_m, span), MakeIndex(physical_n, span)},
                                        {std::move(valid_m), std::move(valid_n)}, dtype, name_hint, span);
}

struct KLoopRewrite {
  AssignStmtPtr original;
  MatmulKind kind = MatmulKind::kFresh;
  VarPtr lhs_src;                 ///< [M, K] left operand — Mat- or Vec-resident
  VarPtr rhs_src;                 ///< [K, N] right operand — Mat-resident
  VarPtr bias_src;                ///< optional [1, N] bias — Mat- or Bias-resident
  CallPtr bias_load;              ///< defining Mat tile.load, required when bias is N-tiled
  bool stage_lhs_to_mat = false;  ///< lhs is Vec-resident: stage Vec→Mat before the K-loop
  VarPtr acc_init = nullptr;      ///< Caller-provided accumulator for matmul_acc;
                                  ///< nullptr for plain matmul (Vec placeholder is built instead).
  /// Caller's ``init_cond`` predicate from a 4-operand ``tile.matmul_acc``;
  /// nullptr for the 3-operand spelling and for fresh/bias matmuls.  Composed
  /// with the generated ``ko == 0`` in the pipelined body, passed through
  /// unmodified on a lone straight-line full block, and dropped on the tail.
  ExprPtr init_cond = nullptr;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  /// Logical output window carried by the loop accumulator. These may be
  /// smaller than m/n when a boxed boundary operand is physically padded.
  ExprPtr valid_m = nullptr;
  ExprPtr valid_n = nullptr;
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

/// Materialize the bias operand for one output-column window. A full vector
/// already resident in the architectural Bias buffer is reusable as-is. A
/// full Mat source moves directly. For an N window, reconstruct a fresh Mat
/// tile from the defining tensor load and move that tile into Bias. A boxed
/// one-row Mat ``tile.slice`` is deliberately not used: it survives as
/// ``pto.subview`` and PTOAS rejects that boxed subview geometry.
VarPtr BuildBiasOperand(std::vector<StmtPtr>& stmts, const VarPtr& bias_src, int64_t n, int64_t N,
                        const ExprPtr& ni, const CallPtr& bias_load, const std::string& name,
                        const Span& sp) {
  if (!bias_src) return nullptr;
  auto bias_ty = As<TileType>(bias_src->GetType());
  INTERNAL_CHECK_SPAN(bias_ty, sp) << "Internal error: matmul bias is not a TileType";
  if (bias_ty->GetMemorySpace() == MemorySpace::Bias) {
    INTERNAL_CHECK_SPAN(n == N, sp)
        << "Internal error: an already-Bias-resident source cannot form an N sub-window";
    return bias_src;
  }
  VarPtr move_source = bias_src;
  if (n != N) {
    INTERNAL_CHECK_SPAN(bias_load && bias_load->args_.size() == 4, sp)
        << "Internal error: N-tiled bias requires a four-argument defining tile.load";
    auto base_offsets = As<MakeTuple>(bias_load->args_[1]);
    INTERNAL_CHECK_SPAN(base_offsets && base_offsets->elements_.size() == 2, sp)
        << "Internal error: N-tiled bias load requires two source offsets";
    auto add_offset = [&](const ExprPtr& base, const ExprPtr& delta) -> ExprPtr {
      if (auto d = As<ConstInt>(delta); d && d->value_ == 0) {
        return base;
      }
      if (auto b = As<ConstInt>(base)) {
        if (auto d = As<ConstInt>(delta)) {
          return MakeIndex(b->value_ + d->value_, sp);
        }
        if (b->value_ == 0) {
          return delta;
        }
      }
      return MakeAdd(base, delta, sp);
    };
    auto offsets = std::make_shared<MakeTuple>(
        std::vector<ExprPtr>{base_offsets->elements_[0], add_offset(base_offsets->elements_[1], ni)}, sp);
    auto window = MakeIndexTuple({1, n}, sp);
    auto load_call = OpRegistry::GetInstance().Create(
        "tile.load", {bias_load->args_[0], offsets, window, window}, bias_load->kwargs_, sp);
    auto load_var = std::make_shared<Var>(name + "_mat", load_call->GetType(), sp);
    auto load = std::make_shared<AssignStmt>(load_var, load_call, sp);
    stmts.push_back(load);
    move_source = load_var;
  }
  auto bias = BuildMove(move_source, MemorySpace::Bias, name, sp);
  stmts.push_back(bias);
  return bias->var_;
}

/// Body of the pipelined L0 K-loop for a fresh ``tile.matmul``: one predicated
/// ``tile.matmul_acc(c_iter, sa, sb, ko == 0)``.
///
/// The predicate is the MAD's ``cmatrixInit`` bit — the accumulator is
/// overwritten on the first L0 block and accumulated into afterwards — and
/// ``set_output_reuses_input(0)`` keeps the whole chain on one L0C buffer by
/// construction.  The ``if ko == 0`` shape this replaces minted a second Acc
/// buffer in the then-arm and merged the two at a phi: unrealizable, since
/// nothing reads L0C but the FIXPIPE drain, so no Acc->Acc copy exists to
/// reconcile the arms.
///
/// ``tile.matmul_bias`` carries no ``init_cond`` operand, so it cannot use this
/// body; ``BuildKLoopRewrite`` head-peels its first K block instead, which
/// reaches the same one-buffer chain without a predicate.  No caller therefore
/// passes a bias operand here.
StmtPtr BuildMatmulBody(const VarPtr& ko_var, const IterArgPtr& c_iter, const AssignStmtPtr& sa,
                        const AssignStmtPtr& sb, const std::string& base, const Span& sp) {
  auto& reg = OpRegistry::GetInstance();
  auto init_cond = MakeEq(ko_var, MakeIndex(0, sp), sp);
  auto c_call = reg.Create("tile.matmul_acc", {ExprPtr(c_iter), sa->var_, sb->var_, init_cond}, sp);
  auto c_var = std::make_shared<Var>(base + "_l0_c_acc", c_call->GetType(), sp);
  auto c_assign = std::make_shared<AssignStmt>(c_var, c_call, sp);
  auto body_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_var}, sp);
  return SeqStmts::Flatten(std::vector<StmtPtr>{sa, sb, c_assign, body_yield}, sp);
}

/// Body of the K-loop for ``tile.matmul_acc``: every iteration accumulates
/// into ``c_iter`` via ``tile.matmul_acc``.  The first iteration's ``c_iter``
/// is the caller-supplied ``acc_init`` (threaded through ``init_values``), so
/// no if-else is needed — the accumulator chain stays uniform.  It optionally
/// carries a predicate: when the caller wrote the 4-operand
/// ``tile.matmul_acc(acc, a, b, user_cond)``, @p user_init_cond is that
/// operand and the emitted call ANDs it with the generated ``ko == 0``.
StmtPtr BuildMatmulAccBody(const IterArgPtr& c_iter, const AssignStmtPtr& sa, const AssignStmtPtr& sb,
                           const std::string& base, const Span& sp, const ExprPtr& user_init_cond,
                           const VarPtr& ko_var) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<ExprPtr> args{ExprPtr(c_iter), sa->var_, sb->var_};
  if (user_init_cond) {
    INTERNAL_CHECK_SPAN(ko_var, sp)
        << "Internal error: a predicated matmul_acc K-loop body needs its loop variable";
    // The caller's predicate means "first K step of the user's reduction"; the
    // generated ``ko == 0`` means "first L0 block of that step".  The
    // accumulator is overwritten only where both hold.  There is no
    // pre-existing ``ko == 0`` on this path to compose with — the term is
    // introduced here.
    args.push_back(MakeAnd(user_init_cond, MakeEq(ko_var, MakeIndex(0, sp), sp), sp));
  }
  auto c_call = reg.Create("tile.matmul_acc", args, sp);
  auto c_var = std::make_shared<Var>(base + "_l0_c_acc", c_call->GetType(), sp);
  auto c_assign = std::make_shared<AssignStmt>(c_var, c_call, sp);
  auto outer_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_var}, sp);
  return SeqStmts::Flatten(std::vector<StmtPtr>{sa, sb, c_assign, outer_yield}, sp);
}

/// Build the replacement statements for one supported Mat-resident matmul-family call.
/// See the file-level comment for the emitted shape.
RewriteResult BuildKLoopRewrite(const KLoopRewrite& r) {
  const Span sp = r.original->span_;
  const std::string base = r.name_base.empty() ? r.original->var_->name_hint_ : r.name_base;
  const bool is_acc = r.kind == MatmulKind::kAccumulate;
  const bool is_bias = r.kind == MatmulKind::kBias;
  auto& reg = OpRegistry::GetInstance();

  INTERNAL_CHECK_SPAN(is_acc == (r.acc_init != nullptr), sp)
      << "Internal error: matmul kind and accumulator initializer disagree";
  INTERNAL_CHECK_SPAN(is_acc || r.init_cond == nullptr, sp)
      << "Internal error: only tile.matmul_acc carries an init_cond predicate";
  INTERNAL_CHECK_SPAN((r.kind == MatmulKind::kBias) == (r.bias_src != nullptr), sp)
      << "Internal error: matmul kind and bias operand disagree";

  INTERNAL_CHECK_SPAN(r.k > 0 && r.k <= r.K, sp)
      << "Internal error: BuildKLoopRewrite expects 0 < k <= K, got k=" << r.k << ", K=" << r.K;

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
  //
  // ``tile.matmul_bias`` is the exception: it has no ``init_cond`` operand, so
  // its first K block is head-peeled below and *is* the seed — a placeholder
  // here would be dead.
  ExprPtr loop_init;
  TypePtr loop_iter_type;
  if (num_full >= 2 && !is_bias) {
    if (is_acc) {
      loop_init = r.acc_init;
      loop_iter_type = r.acc_init->GetType();
    } else {
      auto acc_dtype = As<TileType>(r.original->var_->GetType())->dtype_;
      auto c_init =
          BuildAccInitWithValidShape(r.m, r.n, r.valid_m, r.valid_n, acc_dtype, base + "_l0_init", sp);
      for (auto& stmt : c_init.stmts) out.push_back(std::move(stmt));
      loop_init = c_init.value;
      loop_iter_type = c_init.value->GetType();
    }
  }

  // A Vec-resident left operand (fused-attention PV / ``score·V``) is staged into
  // Mat once, before any K block, so each extract slices from Mat exactly like
  // the QK path — and so ``ExpandMixedKernel`` can lower the Vec→Mat crossing via
  // its ``tile.move`` handshake (``CollectCVBoundaryMoves`` only matches
  // ``tile.move``).  Mat-resident left operands extract directly.
  VarPtr lhs_extract_src = r.lhs_src;
  if (r.stage_lhs_to_mat) {
    auto lhs_mat = BuildMove(r.lhs_src, MemorySpace::Mat, base + "_l0_lmat", sp);
    out.push_back(lhs_mat);
    lhs_extract_src = lhs_mat->var_;
  }

  ExprPtr mi_off = r.mi ? r.mi : MakeIndex(0, sp);
  ExprPtr ni_off = r.ni ? r.ni : MakeIndex(0, sp);
  VarPtr bias_operand =
      BuildBiasOperand(out, r.bias_src, r.n, r.N, ni_off, r.bias_load, base + "_l0_bias", sp);

  // Emit one straight-line K block ``[m, kb] x [kb, n]`` at static K-offset
  // ``ko``, accumulating into ``acc_in`` (``tile.matmul_acc``) or starting fresh
  // (``tile.matmul`` when ``acc_in`` is null).  Used for the single-full-block
  // (num_full == 1) and partial-tail cases; the multi-block case pipelines below.
  // The optional ``init_cond`` predicates the accumulator overwrite and is only
  // legal alongside ``acc_in`` — it is the caller's predicate, emitted verbatim
  // (a straight-line block has no loop variable to compose ``ko == 0`` from).
  auto emit_block = [&](int64_t ko, int64_t kb, const ExprPtr& acc_in, const ExprPtr& init_cond,
                        const std::string& tag) -> VarPtr {
    auto sa = BuildExtract(lhs_extract_src, {r.m, kb}, mi_off, MakeIndex(ko, sp), MemorySpace::Left,
                           base + "_l0_a" + tag, sp);
    auto sb = BuildExtract(r.rhs_src, {kb, r.n}, MakeIndex(ko, sp), ni_off, MemorySpace::Right,
                           base + "_l0_b" + tag, sp);
    ExprPtr call;
    INTERNAL_CHECK_SPAN(acc_in || !init_cond, sp)
        << "Internal error: init_cond on a straight-line block with no accumulator";
    if (acc_in) {
      std::vector<ExprPtr> mm_args{acc_in, sa->var_, sb->var_};
      if (init_cond) mm_args.push_back(init_cond);
      call = reg.Create("tile.matmul_acc", mm_args, sp);
    } else if (bias_operand) {
      call = reg.Create("tile.matmul_bias", {sa->var_, sb->var_, bias_operand}, sp);
    } else {
      call = reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
    }
    auto cvar = std::make_shared<Var>(base + "_l0_c" + tag, call->GetType(), sp);
    out.push_back(sa);
    out.push_back(sb);
    out.push_back(std::make_shared<AssignStmt>(cvar, call, sp));
    return cvar;
  };

  // Emit the pipelined K-loop over the full blocks in ``[k_lo, k_full)``,
  // accumulating into ``seed_init``.  Shared by the ordinary path (all
  // ``num_full`` blocks, seeded by a placeholder or the caller's accumulator)
  // and the bias path (the blocks after the peeled first one).
  auto emit_k_loop = [&](int64_t k_lo, const ExprPtr& seed_init, const ExprPtr& user_cond) -> VarPtr {
    auto ko_var = std::make_shared<Var>(base + "_l0_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);
    auto iter_type = seed_init->GetType();
    auto c_iter = std::make_shared<IterArg>(base + "_l0_c", iter_type, seed_init, sp);
    auto sa =
        BuildExtract(lhs_extract_src, {r.m, r.k}, mi_off, ko_var, MemorySpace::Left, base + "_l0_a", sp);
    auto sb = BuildExtract(r.rhs_src, {r.k, r.n}, ko_var, ni_off, MemorySpace::Right, base + "_l0_b", sp);
    // A fresh ``tile.matmul`` must overwrite its placeholder on the first block;
    // an accumulate seeded by a real accumulator must not, unless the caller
    // predicated it.  The bias path passes neither: its seed already holds the
    // first block's product, so every block here accumulates.
    StmtPtr body = (k_lo == 0 && r.kind == MatmulKind::kFresh)
                       ? BuildMatmulBody(ko_var, c_iter, sa, sb, base, sp)
                       : BuildMatmulAccBody(c_iter, sa, sb, base, sp, user_cond, ko_var);
    std::vector<std::pair<std::string, std::any>> attrs = {{kPipelineStagesAttr, /*pipeline_stages=*/2}};
    // Loop return var: an intermediate when a partial tail follows (named
    // distinctly so round-trip names stay unique), else the final result.
    auto rv = std::make_shared<Var>(has_tail ? base + "_l0_kmain" : base, iter_type, r.original->var_->span_);
    auto for_stmt =
        std::make_shared<ForStmt>(ko_var, MakeIndex(k_lo, sp), MakeIndex(k_full, sp), MakeIndex(r.k, sp),
                                  std::vector<IterArgPtr>{c_iter}, body, std::vector<VarPtr>{rv}, sp,
                                  ForKind::Pipeline, std::move(attrs));
    out.push_back(for_stmt);
    return rv;
  };

  // --- Full blocks: a pipelined K-loop when there are >= 2 of them, else a
  //     single straight-line block (a 1-trip pipeline loop would be degenerate).
  VarPtr main_var;
  if (is_bias && num_full >= 2) {
    // Head-peel the first K block.  ``tile.matmul_bias`` applies the bias
    // exactly once *and* mints the accumulator, so the remaining blocks
    // accumulate into it uniformly and the whole chain stays on one L0C buffer.
    // The alternative — an ``if ko == 0`` phi between ``tile.matmul_bias`` and
    // ``tile.matmul_acc`` — gives one logical value two producers on two Acc
    // buffers, which no target can realize (there is no Acc->Acc copy), and
    // ``tile.matmul_bias`` has no ``init_cond`` operand to predicate instead.
    auto seed = emit_block(/*ko=*/0, /*kb=*/r.k, /*acc_in=*/nullptr, /*init_cond=*/nullptr, "0");
    // A 1-trip pipeline loop would be degenerate, so two full blocks means the
    // second one is straight-line too.
    main_var = (num_full - 1 >= 2)
                   ? emit_k_loop(/*k_lo=*/r.k, ExprPtr(seed), /*user_cond=*/nullptr)
                   : emit_block(/*ko=*/r.k, /*kb=*/r.k, ExprPtr(seed), /*init_cond=*/nullptr, "1");
  } else if (num_full >= 2) {
    main_var = emit_k_loop(/*k_lo=*/0, loop_init, is_acc ? r.init_cond : nullptr);
  } else {
    // num_full == 1: emit the single full block directly, without a degenerate
    // one-trip pipeline.  Canonical full-K plans use this path with k == K;
    // plans with K/2 < k < K also use it before emitting their partial tail.
    // The caller's predicate passes through unmodified: this block *is* the
    // first (and only) full K block, so a generated ``ko == 0`` term would be
    // statically true and would only add a foldable node.
    main_var = emit_block(/*ko=*/0, /*kb=*/r.k, is_acc ? ExprPtr(r.acc_init) : nullptr,
                          /*init_cond=*/is_acc ? r.init_cond : nullptr, "0");
  }

  // --- Partial tail: matmul_acc the [m, k_eff] x [k_eff, n] block onto the
  //     full-blocks accumulator (no-op when k divides K).  Never predicated:
  //     the tail runs at K offset k_full >= r.k > 0, so it is never the first
  //     block and the caller's predicate is dead by construction there —
  //     emit the plain 3-operand accumulating call instead of `cond && false`. ---
  VarPtr result_var =
      has_tail ? emit_block(k_full, k_eff, ExprPtr(main_var), /*init_cond=*/nullptr, "t") : main_var;
  return RewriteResult{std::move(out), result_var};
}

/// Build the K-loop descriptor for one output sub-tile ``[mi : mi + m_eff,
/// ni : ni + n_eff]``.  Passing ``mi == ni == nullptr`` with ``m_eff == M`` and
/// ``n_eff == N`` yields the whole-output (K-only) case unchanged.
KLoopRewrite MakeKLoop(const MatmulTiling& t, ExprPtr mi, ExprPtr ni, int64_t m_eff, int64_t n_eff,
                       std::string name_base) {
  KLoopRewrite r;
  r.original = t.assign;
  r.kind = t.kind;
  r.lhs_src = t.lhs;
  r.rhs_src = t.rhs;
  r.bias_src = t.bias;
  r.bias_load = t.bias_load;
  r.stage_lhs_to_mat = t.stage_lhs_to_mat;
  r.acc_init = t.acc_init;
  r.init_cond = t.init_cond;
  r.M = t.M;
  r.N = t.N;
  r.K = t.K;
  r.m = m_eff;
  r.n = n_eff;
  r.k = t.k;
  auto local_valid_extent = [&](const ExprPtr& output_valid, const ExprPtr& offset,
                                int64_t physical_extent) -> ExprPtr {
    INTERNAL_CHECK_SPAN(output_valid, t.assign->span_)
        << "Internal error: matmul output requires a logical valid extent";
    if (!offset) return output_valid;

    // Constant output grids are by far the common case. Fold them here so the
    // generated set_validshape operands stay simple constants.
    if (auto valid_const = As<ConstInt>(output_valid)) {
      if (auto offset_const = As<ConstInt>(offset)) {
        return MakeIndex(std::clamp<int64_t>(valid_const->value_ - offset_const->value_, 0, physical_extent),
                         t.assign->span_);
      }
    }

    // Keep the zero-offset form canonical with InferWindowReadValidShape. This
    // also avoids asking the verifier to prove `max(valid, 0) - 0 == valid`.
    if (auto offset_const = As<ConstInt>(offset); offset_const && offset_const->value_ == 0) {
      return MakeMin(output_valid, MakeIndex(physical_extent, t.assign->span_), t.assign->span_);
    }

    // The logical window of a physical sub-tile is
    // clamp(output_valid - offset, 0, physical_extent). Spell the lower clamp
    // as max(output_valid, offset) - offset so UINT64 valid extents cannot
    // underflow when the sub-tile starts beyond the logical boundary.
    auto clamped_valid = MakeMax(output_valid, offset, t.assign->span_);
    auto remaining = MakeSub(clamped_valid, offset, t.assign->span_);
    return MakeMin(remaining, MakeIndex(physical_extent, t.assign->span_), t.assign->span_);
  };
  r.valid_m = local_valid_extent(t.output_valid_m, mi, m_eff);
  r.valid_n = local_valid_extent(t.output_valid_n, ni, n_eff);
  r.mi = std::move(mi);
  r.ni = std::move(ni);
  r.name_base = std::move(name_base);
  return r;
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

/// The direct-store forms whose output windows the grid can reconstruct.
/// A nonzero full_m denotes NZ: its row axis is indivisible, while its GM
/// coordinates may already be blocked by BlockNzTensorViews.
struct DirectGmStoreInfo {
  CallPtr call;
  bool blocked_nz = false;
  int64_t full_m = 0;

  [[nodiscard]] bool is_nz() const { return full_m != 0; }
};

std::optional<DirectGmStoreInfo> AnalyzeDirectGmStore(const CallPtr& call) {
  if (!IsOp(call, "tile.store") || (call->args_.size() != 3 && call->args_.size() != 4)) {
    return std::nullopt;
  }
  auto offsets = As<MakeTuple>(call->args_[1]);
  auto output = AsVarLike(call->args_[2]);
  auto tensor = output ? AsTensorTypeLike(output->GetType()) : nullptr;
  if (!offsets || !tensor) return std::nullopt;
  const bool is_nz = tensor->tensor_view_ && tensor->tensor_view_->layout == TensorLayout::NZ;
  if (!is_nz) {
    if (call->args_.size() != 3 || offsets->elements_.size() != 2) return std::nullopt;
    return DirectGmStoreInfo{call};
  }

  auto source = As<TileType>(call->args_[0]->GetType());
  if (!source || source->shape_.size() != 2 || source->dtype_ != DataType::FP32 ||
      source->GetMemorySpace() != MemorySpace::Acc || tensor->dtype_ != DataType::FP32) {
    return std::nullopt;
  }
  const auto view = tile_view_semantics::GetEffectiveTileView(*source);
  if (view.valid_shape.size() != 2) return std::nullopt;
  auto rows = As<ConstInt>(source->shape_[0]);
  auto cols = As<ConstInt>(source->shape_[1]);
  auto valid_rows = As<ConstInt>(view.valid_shape[0]);
  auto valid_cols = As<ConstInt>(view.valid_shape[1]);
  if (!rows || !cols || !valid_rows || !valid_cols || rows->value_ <= 0 || cols->value_ <= 0 ||
      rows->value_ % 16 != 0 || cols->value_ % 16 != 0 || rows->value_ != valid_rows->value_ ||
      cols->value_ != valid_cols->value_) {
    return std::nullopt;
  }

  const bool blocked = tensor_view_semantics::IsBlockedNzShape(tensor->shape_, tensor->dtype_);
  const size_t rank = tensor->shape_.size();
  if (rank < 2 || offsets->elements_.size() != rank) return std::nullopt;
  const size_t row_axis = blocked ? 2 : rank - 2;
  auto parent_rows = As<ConstInt>(tensor->shape_[row_axis]);
  auto row_offset = As<ConstInt>(offsets->elements_[row_axis]);
  const int64_t expected_parent_rows = blocked ? rows->value_ / 16 : rows->value_;
  if (!parent_rows || parent_rows->value_ != expected_parent_rows || !row_offset || row_offset->value_ != 0) {
    return std::nullopt;
  }

  // Blocked NZ always carries its physical partition explicitly. Supporting
  // the logical spelling too lets isolated-pass callers keep their natural
  // 2-D coordinates; the production pipeline reaches the blocked arm.
  if (blocked && call->args_.size() != 4) return std::nullopt;
  if (call->args_.size() == 4) {
    auto sizes = As<MakeTuple>(call->args_[3]);
    if (!sizes || sizes->elements_.size() != rank) return std::nullopt;
    std::vector<int64_t> expected(rank, 1);
    if (blocked) {
      expected = {1, cols->value_ / 8, rows->value_ / 16, 16, 8};
    } else {
      expected[rank - 2] = rows->value_;
      expected[rank - 1] = cols->value_;
    }
    for (size_t axis = 0; axis < rank; ++axis) {
      auto size = As<ConstInt>(sizes->elements_[axis]);
      if (!size || size->value_ != expected[axis]) return std::nullopt;
    }
  }
  return DirectGmStoreInfo{call, blocked, rows->value_};
}

/// One-shot index over a SeqStmts' children, built lazily on the first
/// oversized matmul and reused for the rest so M/N folding stays O(N):
///   * ``use_counts[v]`` — number of reads of ``v`` (excluding defs).
///   * ``store_of[v]`` — the top-level direct ``tile.store`` whose source operand
///     is ``v`` (its sole direct consumer when ``use_counts[v] == 1``).
///     Includes ordinary 2-D stores and blocked rank-5 NZ partitions.
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
    // Keep existing three-argument stores indexed, and additionally recognize
    // the explicit rank-5 partition carried by an NZ direct store.
    if (IsOp(call, "tile.store") && (call->args_.size() == 3 || AnalyzeDirectGmStore(call).has_value())) {
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
/// hand it to a ``SubtilePlacer``. ``DirectGmPlacer`` stores to a DDR output;
/// ``MatScratchPlacer`` assembles into an on-chip Mat scratch for a chained
/// matmul consumer. Each placer threads its chained output Var in traversal
/// order and yields the final Var via ``PlaceAt``.
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

/// Direct-store placement, chaining the DDR output tensor in SSA form. ND
/// adds the logical row/column origins; NZ keeps the complete parent rows and
/// maps only the N origin into the original logical or blocked GM coordinates.
class DirectGmPlacer : public SubtilePlacer {
 public:
  DirectGmPlacer(DirectGmStoreInfo store, Span span)
      : store_(std::move(store)),
        out_in_(AsVarLike(store_.call->args_[2])),
        out_base_(out_in_->name_hint_),
        sp_(std::move(span)) {}

  [[nodiscard]] VarPtr Init(std::vector<StmtPtr>& /*stmts*/) override { return out_in_; }

  [[nodiscard]] VarPtr PlaceAt(std::vector<StmtPtr>& stmts, const VarPtr& sub, const ExprPtr& row_off,
                               const ExprPtr& col_off, const VarPtr& chain_in, int step) override {
    auto base_offsets = As<MakeTuple>(store_.call->args_[1]);
    INTERNAL_CHECK_SPAN(base_offsets, sp_) << "Internal error: direct-store offsets must remain a tuple";
    std::vector<ExprPtr> offsets = base_offsets->elements_;
    ExprPtr partition;
    if (!store_.is_nz()) {
      offsets[0] = AddOffset(offsets[0], row_off, sp_);
      offsets[1] = AddOffset(offsets[1], col_off, sp_);
    } else {
      auto tile = As<TileType>(sub->GetType());
      auto rows = tile && tile->shape_.size() == 2 ? As<ConstInt>(tile->shape_[0]) : nullptr;
      auto cols = tile && tile->shape_.size() == 2 ? As<ConstInt>(tile->shape_[1]) : nullptr;
      INTERNAL_CHECK_SPAN(rows && cols && rows->value_ == store_.full_m && cols->value_ % 16 == 0, sp_)
          << "Internal error: NZ output tiling must emit full-height, 16-column-aligned Acc tiles";
      // The fixed-M chooser admits only one row origin, zero. The full-K
      // emitter may still represent that one origin by a one-trip loop Var;
      // retain the literal blocked row offset instead of manufacturing an
      // unprovable row expression from that loop.
      ExprPtr column_delta = col_off;
      if (store_.blocked_nz) {
        // Only this generated N-grid offset is divided. The user's original
        // offset remains in its already-proven blocked coordinates, so its
        // arithmetic is never reassociated or changed in width.
        if (auto constant = As<ConstInt>(col_off)) {
          INTERNAL_CHECK_SPAN(constant->value_ % 8 == 0, sp_)
              << "Internal error: NZ output column offset must be C0-aligned";
          column_delta = MakeIndex(constant->value_ / 8, sp_);
        } else {
          column_delta = MakeFloorDiv(col_off, MakeIndex(8, sp_), sp_);
        }
        offsets[1] = AddOffset(offsets[1], column_delta, sp_);
      } else {
        offsets.back() = AddOffset(offsets.back(), column_delta, sp_);
      }
      if (store_.blocked_nz) {
        partition = MakeIndexTuple({1, cols->value_ / 8, rows->value_ / 16, 16, 8}, sp_);
      } else if (store_.call->args_.size() == 4) {
        std::vector<int64_t> window(base_offsets->elements_.size(), 1);
        window[window.size() - 2] = rows->value_;
        window.back() = cols->value_;
        partition = MakeIndexTuple(window, sp_);
      }
    }
    std::vector<ExprPtr> args{sub, std::make_shared<MakeTuple>(std::move(offsets), sp_), chain_in};
    if (partition) args.push_back(partition);
    const auto original_source = AsVarLike(store_.call->args_[0]);
    INTERNAL_CHECK_SPAN(original_source, sp_) << "Internal error: direct-store source must be a variable";
    // These generated values already have their final types and definitions.
    // Keep replacements terminal: walking chain_in's IterArg initializer back
    // to out_in would otherwise substitute the new loop carry into its own seed.
    const std::unordered_map<const Var*, VarPtr> replacements = {{original_source.get(), sub},
                                                                 {out_in_.get(), chain_in},
                                                                 {sub.get(), sub},
                                                                 {chain_in.get(), chain_in}};
    // Both layouts share metadata remapping and the store's type rules. An
    // explicit partition preserves the destination type without turning the
    // 2-D Acc into a rank-5 tile; ordinary ND stores retain valid-region union.
    auto scall = transform_utils::RebuildRegisteredCall(store_.call, args, [&](const ExprPtr& expr) {
      return transform_utils::Substitute(expr, replacements);
    });
    auto sv = std::make_shared<Var>(out_base_ + "_t" + std::to_string(step), scall->GetType(), sp_);
    stmts.push_back(std::make_shared<AssignStmt>(sv, scall, sp_));
    return sv;
  }

 private:
  DirectGmStoreInfo store_;
  VarPtr out_in_;
  std::string out_base_;
  Span sp_;
};

/// Mat-scratch placement (on-chip matmul consumers): keep the whole ``[M, N]``
/// result in an L1/Mat scratch instead of storing it to a DDR tensor, so a
/// matmul-operand consumer reads it on-chip.  ``Init`` creates the scratch (mirrors
/// ``BuildAccInit`` but in ``Mat``); each ``PlaceAt`` assembles a sub-tile in place:
/// ``scratch_{k+1} = tile.assemble(scratch_k, sub, [row_off, col_off])`` — Acc→Mat.
/// A low-precision bf16/f16 scratch lowers to FIXPIPE ``pto.tinsert``; a
/// supported same-dtype full-window assemble uses ``pto.subview`` + ``pto.tmov``.
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

/// Count every syntactic Var/IterArg read in one function traversal while
/// excluding SSA definition sites. The canonical split-K pre-phase uses this
/// index to prove that its create and loop result have no consumers outside the
/// matched create/loop/store chain.
class CanonicalReadCounter : public IRVisitor {
 public:
  std::unordered_map<const Var*, size_t> counts;

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) ++counts[op.get()];
  }

  // The generic visitor follows an IterArg read through ``initValue_``.  That
  // is useful for recursive dependency walks, but this index counts direct SSA
  // edges: the initializer is visited once at the enclosing loop and an
  // IterArg use in the body must not count it again.
  void VisitExpr_(const IterArgPtr& op) override { VisitVarLike_(op); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->value_) VisitExpr(op->value_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (op->condition_) VisitExpr(op->condition_);
    if (op->then_body_) VisitStmt(op->then_body_);
    if (op->else_body_.has_value() && *op->else_body_) VisitStmt(*op->else_body_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    if (op->body_) VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (op->condition_) VisitExpr(op->condition_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
    }
    if (op->body_) VisitStmt(op->body_);
  }
};

/// Canonical frontend split-K reduction matched for loop-level M/N tiling:
///
///   init = tile.create([M, N])
///   for kb, (acc,) in pipeline(..., init_values=(init,)):
///     lhs = tile.load(A, [mi0, k0], [M, Kb], ...)
///     rhs = tile.load(B, [k0, ni0], [Kb, N], ...)
///     if first:
///       next = tile.matmul(lhs, rhs)
///     else:
///       next = tile.matmul_acc(acc, lhs, rhs)
///     yield next
///   out = tile.store(loop_result, [base_m, base_n], out)
///
/// or, predicated -- one MAD, one buffer, ``cmatrixInit`` carried as
/// ``init_cond`` instead of a branch:
///
///   init = tile.create([M, N])
///   for kb, (acc,) in pipeline(..., init_values=(init,)):
///     lhs = tile.load(A, [mi0, k0], [M, Kb], ...)
///     rhs = tile.load(B, [k0, ni0], [Kb, N], ...)
///     next = tile.matmul_acc(acc, lhs, rhs, kb == 0)
///     yield next
///   out = tile.store(loop_result, [base_m, base_n], out)
///
/// The surrounding triplet (single-use ``tile.create`` init, single-use loop
/// result feeding one 2D ``tile.store``) and every operand / shape requirement
/// are identical for both spellings; only the two statements at the tail of the
/// loop body differ.
///
/// The whole loop must move outside the output-tile grid: every [m, n]
/// accumulator completes all split-K iterations before the next output tile is
/// started. Merely slicing the matmul_acc call in place would leave the
/// impossible full [M, N] loop-carried accumulator live in L0C.
struct CanonicalSplitKAccMatch {
  AssignStmtPtr init;
  ForStmtPtr loop;
  AssignStmtPtr store;
  AssignStmtPtr lhs_load;
  AssignStmtPtr rhs_load;
  /// Peeled form only: the fresh ``tile.matmul`` that seeds the first K block.
  /// **Null in the predicated form**, where the single 4-operand
  /// ``tile.matmul_acc`` carries the seed as its ``init_cond`` operand.
  AssignStmtPtr matmul;
  /// The accumulating call. Peeled: the else-arm 3-operand call. Predicated:
  /// the loop body's only MAD, and therefore the whole reduction.
  AssignStmtPtr matmul_acc;
  /// Peeled form only: the ``IfStmt``'s phi ``return_vars_[0]``, which the outer
  /// yield carries back to the iter-arg. **Null in the predicated form** --
  /// there is no branch, so there is no merge and no phi definition; the outer
  /// yield reads ``matmul_acc->var_`` directly.
  VarPtr phi;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;  ///< K width of one source split-K iteration
  /// The statement whose operands and result type describe the whole reduction:
  /// the fresh seed when it exists, otherwise the predicated accumulate. Both
  /// carry the same [M, K] x [K, N] -> [M, N] geometry and result dtype.
  /// ``AnalyzeMatmul`` supplies the same source-reduction schedule for both,
  /// including write-only C traffic: the first source iteration overwrites
  /// the placeholder, so neither spelling reads an initial accumulator.
  [[nodiscard]] const AssignStmtPtr& shape_source() const { return matmul ? matmul : matmul_acc; }
};

struct CanonicalOutputWindow {
  int64_t valid_m = 0;
  int64_t valid_n = 0;
  int64_t physical_m = 0;
  int64_t physical_n = 0;
};

std::optional<std::pair<AssignStmtPtr, YieldStmtPtr>> MatchSingleAssignYield(const StmtPtr& body,
                                                                             const char* op_name) {
  auto seq = As<SeqStmts>(body);
  if (!seq || seq->stmts_.size() != 2) return std::nullopt;
  auto assign = As<AssignStmt>(seq->stmts_[0]);
  auto yield = As<YieldStmt>(seq->stmts_[1]);
  auto call = assign ? As<Call>(assign->value_) : nullptr;
  if (!assign || !yield || !call || !IsOp(call, op_name) || yield->value_.size() != 1 ||
      yield->value_[0].get() != assign->var_.get()) {
    return std::nullopt;
  }
  return std::make_pair(assign, yield);
}

/// Recognize the split-K seed predicate of a *predicated* canonical loop body:
/// ``init_cond`` must test this loop's own induction variable against zero.
///
/// Matching by pointer identity on ``loop_var`` (rather than by name or by
/// structure) is what rejects a predicate over some other loop's variable or a
/// caller-supplied flag: those carry no evidence that the accumulator is
/// overwritten on the first K block.
///
/// Two spellings reach here from the DSL. ``init_cond=(kb == 0)`` names the
/// induction variable directly. ``init_cond=(k0 == 0)`` -- the spelling every
/// peeled kernel already uses for its ``if`` condition, and therefore the one a
/// migration produces -- names a body-local scalar ``k0 = kb * <tile width>``,
/// which is zero exactly where ``kb`` is. Resolving that single multiply
/// through ``direct_defs`` is O(1) and keeps the natural migration spelling on
/// the tiled path.
[[nodiscard]] bool IsSplitKSeedPredicate(const ExprPtr& init_cond, const VarPtr& loop_var,
                                         const std::unordered_map<const Var*, AssignStmtPtr>& direct_defs) {
  auto eq = As<Eq>(init_cond);
  if (!eq) return false;
  // ``x == 0`` and ``0 == x`` are the same test; accept either operand order.
  ExprPtr candidate;
  if (auto right_zero = As<ConstInt>(eq->right_); right_zero && right_zero->value_ == 0) {
    candidate = eq->left_;
  } else if (auto left_zero = As<ConstInt>(eq->left_); left_zero && left_zero->value_ == 0) {
    candidate = eq->right_;
  } else {
    return false;
  }
  auto tested = AsVarLike(candidate);
  if (!tested) return false;
  if (tested.get() == loop_var.get()) return true;

  auto def = direct_defs.find(tested.get());
  if (def == direct_defs.end()) return false;
  auto mul = As<Mul>(def->second->value_);
  if (!mul) return false;
  // A nonzero factor makes ``loop_var * factor`` zero exactly where the
  // induction variable is, in either operand order.
  auto scaled = AsVarLike(mul->left_);
  auto factor = As<ConstInt>(mul->right_);
  if (!scaled || !factor) {
    scaled = AsVarLike(mul->right_);
    factor = As<ConstInt>(mul->left_);
  }
  return scaled && factor && factor->value_ != 0 && scaled.get() == loop_var.get();
}

/// ``hints`` is optional: when supplied, an otherwise-canonical *predicated*
/// triplet whose ``init_cond`` is not a split-K seed test reports a PerfHint
/// instead of silently losing loop-level M/N tiling.
std::optional<CanonicalSplitKAccMatch> MatchCanonicalSplitKAcc(
    const AssignStmtPtr& init, const ForStmtPtr& loop, const AssignStmtPtr& store,
    const std::unordered_map<const Var*, size_t>& use_counts, std::vector<Diagnostic>* hints = nullptr) {
  if (!init || !loop || !store || loop->kind_ != ForKind::Pipeline || loop->iter_args_.size() != 1 ||
      loop->return_vars_.size() != 1) {
    return std::nullopt;
  }

  auto init_call = As<Call>(init->value_);
  auto store_call = As<Call>(store->value_);
  if (!init_call || !IsOp(init_call, "tile.create") || !AnalyzeDirectGmStore(store_call) ||
      loop->iter_args_[0]->initValue_.get() != init->var_.get() ||
      store_call->args_[0].get() != loop->return_vars_[0].get()) {
    return std::nullopt;
  }
  auto init_use = use_counts.find(init->var_.get());
  auto result_use = use_counts.find(loop->return_vars_[0].get());
  // True read counts exclude definitions: the create feeds exactly one IterArg
  // initializer and the loop result feeds exactly one store. Any other read
  // would be invalidated when the triplet is replaced.
  if (init_use == use_counts.end() || init_use->second != 1 || result_use == use_counts.end() ||
      result_use->second != 1) {
    return std::nullopt;
  }

  auto body = As<SeqStmts>(loop->body_);
  if (!body || body->stmts_.size() < 2) return std::nullopt;
  // The reduction tail is either an IfStmt (peeled) or a single AssignStmt
  // (predicated); the outer yield follows it in both spellings.
  const size_t tail_pos = body->stmts_.size() - 2;
  auto outer_yield = As<YieldStmt>(body->stmts_.back());
  if (!outer_yield || outer_yield->value_.size() != 1) return std::nullopt;

  // Index the body's leading definitions once. The predicated arm resolves its
  // predicate through this map, and the legality sweep below reuses it.
  std::unordered_map<const Var*, AssignStmtPtr> direct_defs;
  for (size_t i = 0; i < tail_pos; ++i) {
    auto assign = As<AssignStmt>(body->stmts_[i]);
    if (!assign) return std::nullopt;
    direct_defs.emplace(assign->var_.get(), assign);
  }

  AssignStmtPtr matmul;      // null in the predicated form
  AssignStmtPtr matmul_acc;  // the whole reduction in the predicated form
  VarPtr phi;                // null in the predicated form
  VarPtr lhs;
  VarPtr rhs;
  if (auto if_stmt = As<IfStmt>(body->stmts_[tail_pos])) {
    // Peeled form: a fresh tile.matmul seeds the first K block and a 3-operand
    // tile.matmul_acc accumulates the rest, merged at the IfStmt's phi.
    if (!if_stmt->else_body_.has_value() || if_stmt->return_vars_.size() != 1 ||
        outer_yield->value_[0].get() != if_stmt->return_vars_[0].get()) {
      return std::nullopt;
    }
    auto then_mm = MatchSingleAssignYield(if_stmt->then_body_, "tile.matmul");
    auto else_mm = MatchSingleAssignYield(*if_stmt->else_body_, "tile.matmul_acc");
    auto then_acc = MatchSingleAssignYield(if_stmt->then_body_, "tile.matmul_acc");
    auto else_plain = MatchSingleAssignYield(*if_stmt->else_body_, "tile.matmul");
    if (then_mm && else_mm) {
      matmul = then_mm->first;
      matmul_acc = else_mm->first;
    } else if (then_acc && else_plain) {
      matmul = else_plain->first;
      matmul_acc = then_acc->first;
    } else {
      return std::nullopt;
    }

    auto mm_call = As<Call>(matmul->value_);
    auto acc_call = As<Call>(matmul_acc->value_);
    if (!mm_call || !acc_call || mm_call->args_.size() != 2 || acc_call->args_.size() != 3 ||
        acc_call->args_[0].get() != loop->iter_args_[0].get() ||
        mm_call->args_[0].get() != acc_call->args_[1].get() ||
        mm_call->args_[1].get() != acc_call->args_[2].get()) {
      return std::nullopt;
    }
    phi = if_stmt->return_vars_[0];
    lhs = AsVarLike(mm_call->args_[0]);
    rhs = AsVarLike(mm_call->args_[1]);
  } else {
    // Predicated form: the body's last two statements are the single 4-operand
    // tile.matmul_acc accumulating into the loop's only iter-arg, and the outer
    // yield of its result. There is no branch, so there is no phi.
    auto assign = As<AssignStmt>(body->stmts_[tail_pos]);
    auto acc_call = assign ? As<Call>(assign->value_) : nullptr;
    if (!assign || !acc_call || !IsOp(acc_call, "tile.matmul_acc") || acc_call->args_.size() != 4 ||
        outer_yield->value_[0].get() != assign->var_.get() ||
        acc_call->args_[0].get() != loop->iter_args_[0].get()) {
      return std::nullopt;
    }
    if (!IsSplitKSeedPredicate(acc_call->args_[3], loop->loop_var_, direct_defs)) {
      // Everything except the predicate matched, so this is an M/N-tiling cliff
      // rather than an unrelated loop: report it instead of silently declining.
      if (hints) {
        hints->emplace_back(
            DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
            "predicated split-K tile.matmul_acc: init_cond must test this loop's induction variable "
            "against 0 to qualify for loop-level M/N tiling; got a different predicate, so the "
            "reduction is left untouched. That only costs you if its full [M, N] accumulator does "
            "not fit L0C — the ordinary per-call K-tiling still applies either way.",
            acc_call->span_);
      }
      return std::nullopt;
    }
    matmul_acc = assign;
    lhs = AsVarLike(acc_call->args_[1]);
    rhs = AsVarLike(acc_call->args_[2]);
  }
  // One source load cannot simultaneously be narrowed as [m, K] and [K, n].
  // Reject the square self-matmul corner rather than mutate it ambiguously.
  if (!lhs || !rhs || lhs.get() == rhs.get()) return std::nullopt;

  for (size_t i = 0; i < tail_pos; ++i) {
    auto assign = As<AssignStmt>(body->stmts_[i]);
    // Duplicating a source split-K loop per output tile is legal only when all
    // definitions other than the two operand loads are scalar computations.
    // This excludes hidden tile side effects or unrelated loads instead of
    // silently multiplying them.
    if (assign->var_.get() != lhs.get() && assign->var_.get() != rhs.get() &&
        (As<Call>(assign->value_) || !As<ScalarType>(assign->var_->GetType()))) {
      return std::nullopt;
    }
  }
  auto lhs_it = direct_defs.find(lhs.get());
  auto rhs_it = direct_defs.find(rhs.get());
  if (lhs_it == direct_defs.end() || rhs_it == direct_defs.end()) return std::nullopt;
  auto lhs_load_call = As<Call>(lhs_it->second->value_);
  auto rhs_load_call = As<Call>(rhs_it->second->value_);
  if (!lhs_load_call || !rhs_load_call || !IsOp(lhs_load_call, "tile.load") ||
      !IsOp(rhs_load_call, "tile.load") || lhs_load_call->args_.size() != 4 ||
      rhs_load_call->args_.size() != 4) {
    return std::nullopt;
  }
  int64_t M = 0;
  int64_t K_lhs = 0;
  int64_t K_rhs = 0;
  int64_t N = 0;
  auto lhs_ty = As<TileType>(lhs->GetType());
  auto rhs_ty = As<TileType>(rhs->GetType());
  if (!IsStatic2DInSpaces(lhs_ty, {MemorySpace::Mat}, M, K_lhs) ||
      !IsStatic2DInSpaces(rhs_ty, {MemorySpace::Mat}, K_rhs, N) || K_lhs != K_rhs) {
    return std::nullopt;
  }
  // Replacing shape and valid_shape with the narrowed output window is sound
  // only for full rectangular loads. A padded or dynamic valid_shape requires
  // its own intersection arithmetic and stays on the conservative deferred
  // path for now.
  if (!As<MakeTuple>(lhs_load_call->args_[1]) || !As<MakeTuple>(rhs_load_call->args_[1]) ||
      !IsStatic2DTuple(lhs_load_call->args_[2], M, K_lhs) ||
      !IsStatic2DTuple(lhs_load_call->args_[3], M, K_lhs) ||
      !IsStatic2DTuple(rhs_load_call->args_[2], K_rhs, N) ||
      !IsStatic2DTuple(rhs_load_call->args_[3], K_rhs, N)) {
    return std::nullopt;
  }
  auto init_ty = As<TileType>(init->var_->GetType());
  // Declared accumulator/operand storage and explicit buffering loops are
  // manual decisions. Rebuilding even an unchanged Mat load from its deduced
  // type would discard its declared allocation identity and slot selection.
  // Leave these loads to ordinary K tiling, which retains their original Vars.
  if ((init_ty && init_ty->memref_.has_value()) || lhs_ty->memref_.has_value() ||
      rhs_ty->memref_.has_value() || loop->GetAttr<bool>(kPipelineDoubleBufferCAttr, false)) {
    return std::nullopt;
  }
  auto init_m = init_ty && init_ty->shape_.size() == 2 ? As<ConstInt>(init_ty->shape_[0]) : nullptr;
  auto init_n = init_ty && init_ty->shape_.size() == 2 ? As<ConstInt>(init_ty->shape_[1]) : nullptr;
  if (!init_m || !init_n || init_m->value_ != M || init_n->value_ != N) return std::nullopt;

  return CanonicalSplitKAccMatch{init, loop, store, lhs_it->second, rhs_it->second, matmul, matmul_acc,
                                 phi,  M,    N,     K_lhs};
}

std::optional<tile_view_semantics::BoxedTileAlignment> GetCanonicalOutputBoxAlignment(
    const CanonicalSplitKAccMatch& match) {
  auto lhs_type = As<TileType>(match.lhs_load->var_->GetType());
  auto rhs_type = As<TileType>(match.rhs_load->var_->GetType());
  INTERNAL_CHECK_SPAN(lhs_type && rhs_type, match.loop->span_)
      << "Internal error: canonical split-K operand loads lost their TileTypes";

  const auto lhs_alignment = tile_view_semantics::GetBoxedTileAlignment(*lhs_type);
  const auto rhs_alignment = tile_view_semantics::GetBoxedTileAlignment(*rhs_type);
  if (!lhs_alignment || !rhs_alignment) return std::nullopt;

  return tile_view_semantics::BoxedTileAlignment{/*rows=*/lhs_alignment->rows,
                                                 /*cols=*/rhs_alignment->cols};
}

CanonicalOutputWindow BuildCanonicalOutputWindow(
    const CanonicalSplitKAccMatch& match, int64_t valid_m, int64_t valid_n,
    const tile_view_semantics::BoxedTileAlignment& output_box_alignment) {
  return CanonicalOutputWindow{
      /*valid_m=*/valid_m,
      /*valid_n=*/valid_n,
      /*physical_m=*/AlignStaticExtent(valid_m, output_box_alignment.rows, match.loop->span_),
      /*physical_n=*/AlignStaticExtent(valid_n, output_box_alignment.cols, match.loop->span_),
  };
}

/// Retile one DeepClone of the canonical source K loop. Definitions are fresh
/// already; this mutator narrows the two GM->Mat loads and re-deduces both MAD
/// calls plus the if/loop phi types for one [m, n] output tile.
class CanonicalSplitKRetiler : public IRMutator {
 public:
  CanonicalSplitKRetiler(const CanonicalSplitKAccMatch& match,
                         const std::unordered_map<const Var*, VarPtr>& clone_map, const VarPtr& init,
                         int64_t mi, int64_t ni, CanonicalOutputWindow window,
                         std::optional<int64_t> selected_k, std::string suffix)
      : mi_(mi), ni_(ni), window_(window), k_(match.K), selected_k_(selected_k), suffix_(std::move(suffix)) {
    // Expr-valued kwargs can still reference the original reduction's Vars.
    // Keep original-to-clone links as well as the clone-to-narrowed mappings
    // below, including the externally seeded initializer omitted by clone_map.
    for (const auto& [original, clone] : clone_map) var_remap_[original] = clone;
    var_remap_[match.init->var_.get()] = init;
    auto cloned = [&](const VarPtr& original) -> VarPtr {
      auto it = clone_map.find(original.get());
      INTERNAL_CHECK_SPAN(it != clone_map.end(), original->span_)
          << "Internal error: canonical split-K clone omitted definition " << original->name_hint_;
      return it->second;
    };

    lhs_load_ = cloned(match.lhs_load->var_).get();
    rhs_load_ = cloned(match.rhs_load->var_).get();
    // Null in the predicated form: there is no fresh seed statement to rebuild.
    // An AssignStmt's var_ is never null, so the `key != matmul_` test in
    // VisitStmt_ can never accidentally match a null matmul_.
    matmul_ = match.matmul ? cloned(match.matmul->var_).get() : nullptr;
    matmul_acc_ = cloned(match.matmul_acc->var_).get();

    auto old_iter_var = cloned(match.loop->iter_args_[0]);
    auto old_iter = std::dynamic_pointer_cast<const IterArg>(old_iter_var);
    INTERNAL_CHECK_SPAN(old_iter, match.loop->span_)
        << "Internal error: canonical split-K IterArg cloned as a plain Var";
    INTERNAL_CHECK_SPAN(init, match.loop->span_)
        << "Internal error: canonical split-K output tile has no accumulator initializer";
    auto acc_ty = init->GetType();
    auto new_iter = std::make_shared<IterArg>(old_iter->name_hint_ + suffix_, acc_ty, init, old_iter->span_);
    var_remap_[old_iter.get()] = new_iter;

    // Peeled form only. In the predicated form the value the outer yield reads
    // is the matmul_acc result itself, and VisitStmt_ already re-creates that
    // Var with the narrowed call's re-deduced type and registers it in
    // var_remap_ before the yield is visited (SeqStmts order, pinned by the
    // matcher), so pre-registering anything here would be dead.
    if (match.phi) {
      auto old_phi = cloned(match.phi);
      auto new_phi = std::make_shared<Var>(old_phi->name_hint_ + suffix_, acc_ty, old_phi->span_);
      var_remap_[old_phi.get()] = new_phi;
    }

    auto old_return = cloned(match.loop->return_vars_[0]);
    auto new_return = std::make_shared<Var>(old_return->name_hint_ + suffix_, acc_ty, old_return->span_);
    var_remap_[old_return.get()] = new_return;
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    const Var* key = op->var_.get();
    if (key != lhs_load_ && key != rhs_load_ && key != matmul_ && key != matmul_acc_) {
      return IRMutator::VisitStmt_(op);
    }
    auto call = As<Call>(op->value_);
    INTERNAL_CHECK_SPAN(call, op->span_) << "Internal error: canonical split-K definition is not a Call";
    CallPtr rebuilt;
    if (key == lhs_load_ || key == rhs_load_) {
      rebuilt = RebuildLoad(call, /*lhs=*/key == lhs_load_);
    } else {
      std::vector<ExprPtr> args;
      args.reserve(call->args_.size());
      for (const auto& arg : call->args_) args.push_back(VisitExpr(arg));
      rebuilt = transform_utils::RebuildRegisteredCall(
          call, args, [this](const ExprPtr& expr) { return VisitExpr(expr); });
    }
    auto var = std::make_shared<Var>(op->var_->name_hint_ + suffix_, rebuilt->GetType(), op->var_->span_);
    if (selected_k_ && (key == matmul_ || key == matmul_acc_)) {
      // Emit the outer decision directly, including boxed M/N boundaries.
      // Re-running the chooser on this narrowed call would discard selected_k.
      const bool is_acc = key == matmul_acc_;
      const size_t lhs_index = is_acc ? 1 : 0;
      KLoopRewrite plan;
      plan.original = std::make_shared<AssignStmt>(var, rebuilt, op->span_, op->leading_comments_);
      plan.kind = is_acc ? MatmulKind::kAccumulate : MatmulKind::kFresh;
      plan.lhs_src = AsVarLike(rebuilt->args_[lhs_index]);
      plan.rhs_src = AsVarLike(rebuilt->args_[lhs_index + 1]);
      plan.acc_init = is_acc ? AsVarLike(rebuilt->args_[0]) : nullptr;
      plan.init_cond = is_acc && rebuilt->args_.size() == 4 ? rebuilt->args_[3] : nullptr;
      plan.M = plan.m = window_.physical_m;
      plan.N = plan.n = window_.physical_n;
      plan.K = k_;
      plan.k = *selected_k_;
      plan.valid_m = MakeIndex(window_.valid_m, op->span_);
      plan.valid_n = MakeIndex(window_.valid_n, op->span_);
      INTERNAL_CHECK_SPAN(plan.lhs_src && plan.rhs_src && (!is_acc || plan.acc_init), op->span_)
          << "Internal error: canonical reduction lost its operand or accumulator binding";
      auto rewrite = BuildKLoopRewrite(plan);
      var_remap_[op->var_.get()] = rewrite.return_var;
      return SeqStmts::Flatten(std::move(rewrite.stmts), op->span_);
    }
    var_remap_[op->var_.get()] = var;
    return std::make_shared<AssignStmt>(var, rebuilt, op->span_, op->leading_comments_);
  }

 private:
  CallPtr RebuildLoad(const CallPtr& call, bool lhs) {
    std::vector<ExprPtr> args;
    args.reserve(call->args_.size());
    for (const auto& arg : call->args_) args.push_back(VisitExpr(arg));
    auto offsets = As<MakeTuple>(args[1]);
    INTERNAL_CHECK_SPAN(offsets && offsets->elements_.size() == 2, call->span_)
        << "Internal error: canonical split-K tile.load lost its 2D offsets";
    std::vector<ExprPtr> new_offsets = offsets->elements_;
    if (lhs) {
      new_offsets[0] = OffsetPlus(new_offsets[0], mi_, call->span_);
      args[2] = MakeIndexTuple({window_.physical_m, k_}, call->span_);
      args[3] = MakeIndexTuple({window_.valid_m, k_}, call->span_);
    } else {
      new_offsets[1] = OffsetPlus(new_offsets[1], ni_, call->span_);
      args[2] = MakeIndexTuple({k_, window_.physical_n}, call->span_);
      args[3] = MakeIndexTuple({k_, window_.valid_n}, call->span_);
    }
    args[1] = std::make_shared<MakeTuple>(std::move(new_offsets), call->span_);
    return transform_utils::RebuildRegisteredCall(call, args,
                                                  [this](const ExprPtr& expr) { return VisitExpr(expr); });
  }

  const Var* lhs_load_ = nullptr;
  const Var* rhs_load_ = nullptr;
  const Var* matmul_ = nullptr;
  const Var* matmul_acc_ = nullptr;
  int64_t mi_ = 0;
  int64_t ni_ = 0;
  CanonicalOutputWindow window_;
  int64_t k_ = 0;
  // No selected K means the PTOAS compatibility route: narrow M/N only and
  // retain the established ordinary K-tiling path for backend-owned placement.
  std::optional<int64_t> selected_k_;
  std::string suffix_;
};

struct CanonicalSplitKFold {
  std::vector<StmtPtr> stmts;
  VarPtr final_output;
  VarPtr old_store_result;
};

std::optional<CanonicalSplitKFold> TryFoldCanonicalSplitKAcc(const CanonicalSplitKAccMatch& match,
                                                             std::vector<Diagnostic>& hints) {
  const auto store_info = AnalyzeDirectGmStore(As<Call>(match.store->value_));
  INTERNAL_CHECK_SPAN(store_info.has_value(), match.store->span_)
      << "Internal error: matched canonical split-K store lost its direct-store geometry";
  const auto output_box_alignment = GetCanonicalOutputBoxAlignment(match);
  if (!output_box_alignment) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
                       "canonical split-K M/N tiling needs boxed Mat operand layouts with a supported static "
                       "fractal alignment; left untouched",
                       match.loop->span_);
    return std::nullopt;
  }
  // The source loop already realizes output-stationary accumulation across its
  // K blocks. Use the conservative output-stationary chooser regime for the
  // output grid. Account for the same Mat boxing that RebuildLoad will apply to
  // every physical output window, so chooser capacity cannot admit a logical
  // tile that becomes oversized after padding. PyPTO-owned placement prices
  // the complete source reduction and emits its selected K directly.
  // PTOAS retains the established M/N-only pre-rewrite followed by ordinary
  // K tiling: selected nested schedules can exceed native placement limits
  // (e.g. M272/N384/K176, R3/stage2 overflows Left at 76 KiB).
  // This is an explicit compatibility route, not a retry after a failed plan.
  const auto* ctx = PassContext::Current();
  const auto planner = ctx ? ctx->GetMemoryPlanner() : MemoryPlanner::PyPTO;
  const bool preserve_selected_k = planner != MemoryPlanner::PtoAS;
  auto tiling =
      AnalyzeMatmul(match.shape_source(), hints, /*force_output_stationary=*/true, output_box_alignment,
                    /*direct_defs=*/nullptr, /*restrict_output_grid=*/false,
                    preserve_selected_k ? match.loop : nullptr, store_info->full_m);
  if (!tiling || (!preserve_selected_k && !tiling->needs_mn_tiling())) return std::nullopt;
  if (preserve_selected_k) {
    INTERNAL_CHECK_SPAN(
        tiling->stationarity == utils::Stationarity::kOutputStationary && !tiling->double_buffer_c,
        match.loop->span_)
        << "Internal error: canonical reduction requires an output-stationary single-C plan";
  }

  DirectGmPlacer placer(*store_info, match.store->span_);
  std::vector<StmtPtr> stmts;
  VarPtr chain = placer.Init(stmts);
  const auto& shape_src = match.shape_source();
  auto out_ty = As<TileType>(shape_src->var_->GetType());
  INTERNAL_CHECK_SPAN(out_ty, shape_src->span_)
      << "Internal error: canonical split-K reduction result lost its TileType";
  const int64_t num_m = (match.M + tiling->m - 1) / tiling->m;
  const int64_t num_n = (match.N + tiling->n - 1) / tiling->n;
  // Output-sensitive expansion: each source statement is cloned once per
  // required output tile, matching BuildSplitKGrid. Work is O(input IR plus
  // emitted IR); there is no repeated scan of the surrounding program.
  int step = 0;
  for (int64_t nj = 0; nj < num_n; ++nj) {
    const int64_t ni = nj * tiling->n;
    const int64_t n_eff = std::min<int64_t>(tiling->n, match.N - ni);
    for (int64_t mj = 0; mj < num_m; ++mj) {
      const int64_t mi = mj * tiling->m;
      const int64_t m_eff = std::min<int64_t>(tiling->m, match.M - mi);
      const std::string suffix = "_mn" + std::to_string(step);
      const auto window = BuildCanonicalOutputWindow(match, m_eff, n_eff, *output_box_alignment);
      auto init = BuildAccInitWithValidShape(window.physical_m, window.physical_n,
                                             MakeIndex(window.valid_m, match.init->span_),
                                             MakeIndex(window.valid_n, match.init->span_), out_ty->dtype_,
                                             match.init->var_->name_hint_ + suffix, match.init->span_);
      for (auto& init_stmt : init.stmts) stmts.push_back(std::move(init_stmt));

      std::unordered_map<const Var*, ExprPtr> seed = {{match.init->var_.get(), init.value}};
      auto clone = DeepClone(match.loop, seed, /*clone_def_vars=*/true);
      auto cloned_loop = As<ForStmt>(clone.cloned_body);
      INTERNAL_CHECK_SPAN(cloned_loop, match.loop->span_)
          << "Internal error: canonical split-K loop clone is not a ForStmt";
      const auto selected_k = preserve_selected_k ? std::optional<int64_t>(tiling->k) : std::nullopt;
      CanonicalSplitKRetiler retiler(match, clone.var_map, init.value, mi, ni, window, selected_k, suffix);
      auto narrowed = As<ForStmt>(retiler.VisitStmt(cloned_loop));
      INTERNAL_CHECK_SPAN(narrowed, match.loop->span_)
          << "Internal error: canonical split-K retiling did not return a ForStmt";
      stmts.push_back(narrowed);
      chain = placer.PlaceAt(stmts, narrowed->return_vars_[0], MakeIndex(mi, match.store->span_),
                             MakeIndex(ni, match.store->span_), chain, step);
      ++step;
    }
  }
  return CanonicalSplitKFold{std::move(stmts), chain, match.store->var_};
}

/// Rewrite every canonical create/loop/store triplet in one SeqStmts scan.
/// ``use_counts`` is built once over the original function, so matching all
/// nested sequences remains O(N) rather than recursively rescanning each
/// subtree. A running remap preserves an earlier store's output chain in later
/// siblings.
StmtPtr RewriteCanonicalSplitKSeq(const SeqStmtsPtr& seq,
                                  const std::unordered_map<const Var*, size_t>& use_counts,
                                  std::vector<Diagnostic>& hints) {
  if (!seq || seq->stmts_.size() < 3) return seq;
  std::vector<StmtPtr> out;
  out.reserve(seq->stmts_.size());
  std::unordered_map<const Var*, VarPtr> remap;
  bool changed = false;

  for (size_t i = 0; i < seq->stmts_.size();) {
    auto current = remap.empty() ? seq->stmts_[i] : transform_utils::Substitute(seq->stmts_[i], remap);
    if (i + 2 < seq->stmts_.size()) {
      auto next = remap.empty() ? seq->stmts_[i + 1] : transform_utils::Substitute(seq->stmts_[i + 1], remap);
      auto third =
          remap.empty() ? seq->stmts_[i + 2] : transform_utils::Substitute(seq->stmts_[i + 2], remap);
      auto match = MatchCanonicalSplitKAcc(As<AssignStmt>(current), As<ForStmt>(next), As<AssignStmt>(third),
                                           use_counts, &hints);
      if (match) {
        if (auto fold = TryFoldCanonicalSplitKAcc(*match, hints)) {
          for (auto& stmt : fold->stmts) out.push_back(std::move(stmt));
          remap[fold->old_store_result.get()] = fold->final_output;
          i += 3;
          changed = true;
          continue;
        }
      }
    }
    if (current.get() != seq->stmts_[i].get()) changed = true;
    out.push_back(std::move(current));
    ++i;
  }
  if (!changed) return seq;
  return SeqStmts::Flatten(std::move(out), seq->span_);
}

/// First phase of AutoTileMatmulL0: move canonical split-K reductions outside
/// their M/N output grid before dbC planning or ordinary call-level tiling.
class CanonicalSplitKPreMutator : public IRMutator {
 public:
  CanonicalSplitKPreMutator(const std::unordered_map<const Var*, size_t>& use_counts,
                            std::vector<Diagnostic>& hints)
      : use_counts_(use_counts), hints_(hints) {}

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Match before recursively mutating children so definition identities still
    // refer to the original function-wide read index. The qualified base visit
    // then reaches canonical triplets in nested source regions; newly generated
    // loops do not match because their fresh Vars are absent from that index.
    // Under PyPTO-owned placement their K schedule is already emitted;
    // ordinary AutoTile sees Left/Right operands and cannot choose again.
    // The PTOAS compatibility route still leaves K to ordinary AutoTile.
    auto rewritten = As<SeqStmts>(RewriteCanonicalSplitKSeq(op, use_counts_, hints_));
    INTERNAL_CHECK_SPAN(rewritten, op->span_)
        << "Internal error: canonical split-K pre-phase did not preserve SeqStmts";
    return IRMutator::VisitStmt_(rewritten);
  }

 private:
  const std::unordered_map<const Var*, size_t>& use_counts_;
  std::vector<Diagnostic>& hints_;
};

FunctionPtr RewriteCanonicalSplitKAcc(const FunctionPtr& func, std::vector<Diagnostic>& hints) {
  CanonicalReadCounter counter;
  counter.VisitStmt(func->body_);
  CanonicalSplitKPreMutator mutator(counter.counts, hints);
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body == func->body_) return func;
  auto rewritten = MutableCopy(func);
  rewritten->body_ = new_body;
  return rewritten;
}

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
  auto bias_operand = BuildBiasOperand(stmts, t.bias, n_eff, t.N, MakeIndex(ni, sp), t.bias_load,
                                       base + "_tbias" + std::to_string(step), sp);
  auto c_call = bias_operand ? reg.Create("tile.matmul_bias", {sa->var_, sb->var_, bias_operand}, sp)
                             : reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
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
  INTERNAL_CHECK_SPAN(t.m <= t.grid_m && t.n <= t.grid_n, sp)
      << "Internal error: full-K tile must not exceed the work grid; got M=" << t.grid_m << " m=" << t.m
      << " N=" << t.grid_n << " n=" << t.n;

  // Loop order + buffering. When the chooser picked an operand-stationary point
  // (k == K), the stationary operand is the OUTER (held) panel and is SINGLE-
  // buffered — the chooser budgeted it the full L0 buffer (no /2). A-stationary
  // -> A held -> rows outer; B-stationary -> B held -> cols outer. Output-
  // stationary double-buffers both and hoists one operand to the outer loop; the
  // chooser already made that bandwidth-weighted choice while scoring the wall
  // (LoadCycles' min-hoist) and recorded it in `os_holds_a`, so we obey it here
  // rather than re-derive from raw bytes — the two objectives disagree under the
  // ~130:85 L0A:L0B bandwidth ratio, and diverging would emit a different loop
  // order than the wall was scored under.
  const bool a_stationary = t.stationarity == utils::Stationarity::kAStationary;
  const bool b_stationary = t.stationarity == utils::Stationarity::kBStationary;
  const bool stationary_single_buffered = a_stationary || b_stationary;
  const bool row_outer = stationary_single_buffered ? a_stationary : t.os_holds_a;

  // Interior = the region tiled by FULL m x n blocks; the L-shaped partial
  // boundary beyond it is peeled into straight-line tiles below.
  const int64_t full_m = (t.grid_m / t.m) * t.m;
  const int64_t full_n = (t.grid_n / t.n) * t.n;

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
    std::vector<StmtPtr> bias_stmts;
    auto bias_operand = BuildBiasOperand(bias_stmts, t.bias, t.n, t.N, ni, t.bias_load, base + "_bias", sp);
    auto c_call = bias_operand ? reg.Create("tile.matmul_bias", {sa->var_, sb->var_, bias_operand}, sp)
                               : reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
    auto c_var = std::make_shared<Var>(base + "_c", c_call->GetType(), sp);
    std::vector<StmtPtr> inner_body{inner_extract};
    // Bias varies with N. In a row-outer schedule N is the moving inner axis;
    // otherwise N is stationary and the load-backed bias window can be hoisted
    // beside the outer Right-panel extract rather than reloaded for every M tile.
    if (row_outer) {
      for (auto& stmt : bias_stmts) inner_body.push_back(std::move(stmt));
    }
    inner_body.push_back(std::make_shared<AssignStmt>(c_var, c_call, sp));
    VarPtr inner_chain = placer.PlaceAt(inner_body, c_var, mi, ni, out_inner, /*step=*/0);
    inner_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{inner_chain}, sp));
    // overlap_stores stays false: the one-accumulator schedule
    // (matmul_i, store_i, matmul_{i+1}, store_{i+1}) drains each L0C result before
    // the next matmul overwrites it.  dbC=2 (double_buffer_c) instead sets the
    // stronger double_buffer_c attr, which floats *both* stores below *both*
    // matmuls (matmul c, matmul c₁, store c, store c₁) so the two [m, n] Acc tiles
    // stay co-live; the chooser budgeted them at L0C/2 so both fit. Under PTOAS,
    // ptoas places the distinct live ranges. Under PYPTO, flat depth-2 pipeline
    // membership keeps MemoryReuse from coalescing the pair; DSA_RP initially
    // exports it as a strict separation. In all cases tile i's FIXPIPE drain
    // overlaps tile i+1's MAD. The moving-operand extract is double-buffered
    // (Load tier, hoisted) in both schedules.
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
    std::vector<StmtPtr> outer_body{outer_extract};
    if (!row_outer) {
      for (auto& stmt : bias_stmts) outer_body.push_back(std::move(stmt));
    }
    outer_body.push_back(inner_for);
    outer_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{inner_rv}, sp));
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
  // Bottom strip [full_m, grid_m) x [0, grid_n) (covers the corner), then
  // right strip [0, full_m) x [full_n, grid_n). Whole storage-padding blocks
  // are absent from this rectangle, not emitted as zero-valid tail tiles.
  // step 0 is used by the interior placement; the tail continues from step 1.
  int tail_step = 1;
  for (int64_t ni = 0; full_m < t.grid_m && ni < t.grid_n; ni += t.n) {
    chain = EmitFullKTile(stmts, t, placer, chain, full_m, ni, t.grid_m - full_m,
                          std::min<int64_t>(t.n, t.grid_n - ni), base, tail_step++);
  }
  for (int64_t mi = 0; full_n < t.grid_n && mi < full_m; mi += t.m) {
    chain = EmitFullKTile(stmts, t, placer, chain, mi, full_n, t.m, t.grid_n - full_n, base, tail_step++);
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
  const int64_t num_m = (t.grid_m + t.m - 1) / t.m;
  const int64_t num_n = (t.grid_n + t.n - 1) / t.n;

  std::vector<StmtPtr> stmts;
  VarPtr chain = placer.Init(stmts);
  int step = 0;
  for (int64_t nj = 0; nj < num_n; ++nj) {
    const int64_t ni = nj * t.n;
    const int64_t n_eff = std::min<int64_t>(t.n, t.grid_n - ni);
    for (int64_t mj = 0; mj < num_m; ++mj) {
      const int64_t mi = mj * t.m;
      const int64_t m_eff = std::min<int64_t>(t.m, t.grid_m - mi);
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
/// tile within L0c. This helper handles the direct-store consumer:
///
///   * **Direct-store** — the sole consumer is a 2D ``tile.store(c, base, out)``:
///     each sub-tile stores straight to ``out[mi:, ni:]`` (the DDR-output case
///     our solver kernels need).  The store is folded in and emitted at the
///     store site.
///
/// The Mat-scratch alternative is handled earlier by ``TryFoldMatScratch``.
/// ``result_uses`` / ``store_stmt`` come from the precomputed SiblingIndex.
/// Returns nullopt (with a PerfHint) when neither placement applies — an
/// arbitrary ``matmul_acc`` with a caller-supplied [M, N] accumulator, a Vec
/// left operand, and mixed/non-matmul on-chip consumers are deferred. The
/// canonical frontend split-K create/pipeline/store form is handled earlier at
/// the enclosing-loop level.
std::optional<MNFold> TryFoldMNTiling(const MatmulTiling& t, int result_uses, const AssignStmt* store_stmt,
                                      std::vector<Diagnostic>& hints) {
  const Span sp = t.assign->span_;
  auto skip = [&](const std::string& msg) -> std::optional<MNFold> {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006", msg, sp);
    return std::nullopt;
  };

  if (t.is_acc()) {
    return skip(
        "oversized tile.matmul_acc does not match the canonical create -> split-K pipeline -> store "
        "form handled by loop-level M/N tiling; slicing this caller-owned [M, N] accumulator is "
        "unsupported, so the call is left untouched");
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

  // Direct-store: the sole consumer is a 2D or blocked NZ tile.store. The grid is emitted
  // later at the store site, where the caller re-applies the then-current remap
  // — so a prior fold that redefined this output is rewritten correctly (a
  // stale-output SSA guard); resolving it here would miss folds emitted before
  // this one.
  if (store_stmt && result_uses == 1) {
    auto store_call = As<Call>(store_stmt->value_);
    INTERNAL_CHECK_SPAN(store_call, store_stmt->span_)
        << "Internal error: SiblingIndex store_of mapped a non-Call AssignStmt";
    const auto store_info = AnalyzeDirectGmStore(store_call);
    if (!store_info) {
      return skip(
          "tile.store does not have supported direct-store geometry — M/N fold not applicable; "
          "left untouched");
    }
    INTERNAL_CHECK_SPAN(!store_info->is_nz() || (t.m == t.M && t.grid_m == t.M), sp)
        << "Internal error: NZ direct-store fold requires an N/K-only tiling plan";
    DirectGmPlacer placer(*store_info, sp);
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
  if (mode != kRoundRint) return false;
  // `pto.tinsert` carries no `satmode` either, so FIXPIPE cannot reproduce a cast
  // that asked for a specific destination saturation: folding one would make
  // `saturation_mode="on"` and `"off"` compile to the same instruction even though
  // the API defines different finite-overflow results for them. A float
  // destination has no default saturation, so an *absent* effective mode is the
  // "did not ask" case and stays foldable; anything else keeps the Vector
  // `pto.tcvt`, which is the only path that honors the request.
  return !GetSaturationMode(cast).has_value();
}

/// Scope-local consumer facts shared by selection and rewriting. Classify
/// before choosing a smaller work rectangle: a returned tile or caller-owned
/// accumulator cannot silently acquire a different physical result type.
struct MatmulOutputConsumer {
  int result_uses;
  int operand_uses;
  DataType scratch_dtype;
  const Var* remap_target;
  const Var* extra_remap = nullptr;
  const AssignStmt* store_stmt = nullptr;
  bool supports_output_grid = false;
  int64_t fixed_output_m = 0;  ///< NZ placement keeps all parent rows in every output tile.
};

std::optional<MatmulOutputConsumer> AnalyzeMatmulOutputConsumer(const AssignStmtPtr& assign,
                                                                const SiblingIndex& index,
                                                                const Stmt* next_effect) {
  const auto call = As<Call>(assign->value_);
  const auto type = As<TileType>(assign->var_->GetType());
  if (!call || !type ||
      (!IsOp(call, "tile.matmul") && !IsOp(call, "tile.matmul_bias") && !IsOp(call, "tile.matmul_acc"))) {
    return std::nullopt;
  }
  const auto count = [](const std::unordered_map<const Var*, int>& uses, const Var* var) {
    const auto found = uses.find(var);
    return found == uses.end() ? 0 : found->second;
  };
  const Var* result = assign->var_.get();
  MatmulOutputConsumer plan{count(index.use_counts, result), count(index.matmul_operand_uses, result),
                            type->dtype_, result};
  if (auto store = index.store_of.find(result); store != index.store_of.end()) {
    plan.store_stmt = store->second;
  }
  // Preserve the existing FIXPIPE cast-fold contract while making its consumer
  // information available before selection. This does not admit new consumers
  // or move the cast; it replaces the same later classification at emission.
  if (auto cast = index.cast_of.find(result); cast != index.cast_of.end()) {
    const Var* cb = cast->second->var_.get();
    const auto cb_type = As<TileType>(cb->GetType());
    const auto cast_call = As<Call>(cast->second->value_);
    const int uses = count(index.use_counts, cb);
    const int operands = count(index.matmul_operand_uses, cb);
    if (cb_type && CastFoldableToFixpipeMat(cast_call, type, cb_type->dtype_) && plan.result_uses == 1 &&
        uses >= 1 && uses == operands) {
      plan.scratch_dtype = cb_type->dtype_;
      plan.remap_target = cb;
      plan.extra_remap = result;
      plan.result_uses = uses;
      plan.operand_uses = operands;
    }
  }
  if (IsOp(call, "tile.matmul_acc")) return plan;

  if (plan.store_stmt && plan.result_uses == 1) {
    const auto store = As<Call>(plan.store_stmt->value_);
    const auto store_info = AnalyzeDirectGmStore(store);
    // Reconstructed bias loads may not move across an effect to the store.
    // A resident Bias operand needs no reload and retains its full-N minimum.
    const auto bias = IsOp(call, "tile.matmul_bias") && call->args_.size() == 3
                          ? As<TileType>(call->args_[2]->GetType())
                          : nullptr;
    const bool bias_snapshot_safe = !IsOp(call, "tile.matmul_bias") || plan.store_stmt == next_effect ||
                                    (bias && bias->GetMemorySpace() == MemorySpace::Bias);
    if (store_info) {
      plan.supports_output_grid = bias_snapshot_safe;
      plan.fixed_output_m = store_info->full_m;
    }
  }
  if (plan.result_uses >= 1 && plan.operand_uses == plan.result_uses &&
      backend::BackendConfig::IsConfigured()) {
    const auto* ctx = PassContext::Current();
    const auto* handler = ctx ? ctx->GetBackendHandler() : backend::GetBackend()->GetHandler();
    const bool low_precision = plan.scratch_dtype == DataType::BF16 || plan.scratch_dtype == DataType::FP16;
    if (handler && (!handler->RequiresLowPrecisionMatScratch() || low_precision)) {
      plan.supports_output_grid = true;
    }
  }
  return plan;
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
/// requires index-typed elements, not constants). Arbitrary ``matmul_acc`` and
/// Vec-left stay deferred; the canonical split-K form is handled before this
/// local call-level fold.
std::optional<std::pair<std::vector<StmtPtr>, VarPtr>> TryFoldMatScratch(const MatmulTiling& t,
                                                                         int result_uses, int operand_uses,
                                                                         DataType scratch_dtype,
                                                                         std::vector<Diagnostic>& hints) {
  const Span sp = t.assign->span_;
  // Arbitrary matmul_acc / Vec-left are deferred (the direct-store path already
  // hinted these). Canonical split-K is rewritten at the enclosing-loop level.
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
  const uint64_t mat_capacity = handler ? handler->GetCapacityBytes(MemorySpace::Mat) : 0;
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
  const auto valid_m = As<ConstInt>(t.output_valid_m);
  const auto valid_n = As<ConstInt>(t.output_valid_n);
  if (valid_m && valid_n && valid_m->value_ == t.M && valid_n->value_ == t.N) {
    return std::make_pair(std::move(stmts), scratch);
  }
  // The allocation keeps its physical box, but the grid wrote only the valid
  // rectangle. A chained matmul must not read unwritten padding as K values.
  auto view = OpRegistry::GetInstance().Create("tile.set_validshape",
                                               {scratch, t.output_valid_m, t.output_valid_n}, sp);
  auto valid_scratch = std::make_shared<Var>(base + "_valid", view->GetType(), sp);
  stmts.push_back(std::make_shared<AssignStmt>(valid_scratch, view, sp));
  return std::make_pair(std::move(stmts), valid_scratch);
}

class AutoTileMutator : public IRMutator {
 public:
  explicit AutoTileMutator(std::unordered_set<const ForStmt*> pipeline_dbc_plan)
      : pipeline_dbc_plan_(std::move(pipeline_dbc_plan)) {}

  std::vector<Diagnostic> hints;

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    const bool should_double_buffer_c = pipeline_dbc_plan_.count(op.get()) != 0;
    // Recurse first: nested pipelines make their own local dbC decision. An
    // enclosing pipeline does not inherit the marker and therefore does not
    // multiply the accumulator buffering depth.
    auto visited = IRMutator::VisitStmt_(op);
    auto loop = As<ForStmt>(visited);
    if (!should_double_buffer_c || !loop) return visited;

    auto result = MutableCopy(loop);
    result->attrs_ =
        StripAttr(StripAttr(loop->attrs_, kPipelineOverlapStoresAttr), kPipelineDoubleBufferCAttr);
    result->attrs_.emplace_back(kPipelineOverlapStoresAttr, false);
    result->attrs_.emplace_back(kPipelineDoubleBufferCAttr, true);
    return result;
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Per-SeqStmts substitution map: when we rewrite ``c = tile.matmul(...)``
    // into a ForStmt with a fresh return_var, subsequent statements in the
    // same SeqStmts that referenced ``c`` need to be redirected to that
    // return_var.  Scoped to this SeqStmts so substitutions don't leak into
    // sibling regions.
    std::unordered_map<const Var*, VarPtr> remap;
    // Build one scope-local sibling index. Besides ordinary result-use
    // classification, this proves whether a bias load is an immutable snapshot
    // we may replace with window loads at the same observation point.
    SiblingIndex sibling_index = BuildSiblingIndex(op->stmts_);
    // For a direct-store fold, the generated computation moves to the original
    // consumer store. Record the first non-load statement after every position
    // so a bias reload cannot cross an intervening effect on that route.
    std::vector<size_t> next_non_load(op->stmts_.size(), op->stmts_.size());
    size_t next_barrier = op->stmts_.size();
    for (size_t i = op->stmts_.size(); i-- > 0;) {
      next_non_load[i] = next_barrier;
      auto assign = As<AssignStmt>(op->stmts_[i]);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!call || !IsOp(call, "tile.load")) next_barrier = i;
    }
    // Only a single-use load followed by read-only sibling loads is
    // reconstructible. Any store, computation, or control-flow statement is a
    // barrier. This preserves the snapshot across intervening effects while
    // allowing the common "load all operands, then matmul" form.
    DirectDefMap direct_defs;
    DirectDefMap load_defs;
    std::unordered_map<const Var*, size_t> load_indices;
    size_t read_only_run_start = 0;
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      auto assign = As<AssignStmt>(op->stmts_[i]);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (assign && call && IsOp(call, "tile.load")) {
        load_defs.emplace(assign->var_.get(), assign);
        load_indices.emplace(assign->var_.get(), i);
        continue;
      }
      if (call && IsOp(call, "tile.matmul_bias") && call->args_.size() == 3) {
        auto bias = AsVarLike(call->args_[2]);
        auto def = bias ? load_defs.find(bias.get()) : load_defs.end();
        auto def_index = bias ? load_indices.find(bias.get()) : load_indices.end();
        auto uses = bias ? sibling_index.use_counts.find(bias.get()) : sibling_index.use_counts.end();
        if (def != load_defs.end() && def_index != load_indices.end() &&
            def_index->second >= read_only_run_start && uses != sibling_index.use_counts.end() &&
            uses->second == 1) {
          direct_defs.emplace(bias.get(), def->second);
        }
      }
      read_only_run_start = i + 1;
    }
    // Defs to drop: a downcast ``cb = tile.cast(c, bf16)`` whose matmul+cast was
    // folded into a low-precision Mat scratch — the scratch already holds the
    // bf16 intermediate, so the now same-dtype cast is a dead no-op to remove.
    std::unordered_set<const Var*> dropped;
    // Bias full loads precede the matmul that proves they can be replaced, so
    // remove them in one final linear filter after the rewrite succeeds.
    std::unordered_set<const Var*> retroactively_dropped;
    // M/N tiling folds the matmul's consumer store into the per-sub-tile
    // rewrite.  We drop the matmul at its own position and emit the sub-tile
    // stmts where the store was (preserving the order of any statements between
    // them), keyed by the store statement's identity.
    std::unordered_map<const Stmt*, MNFold> pending_folds;
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
      // Eligibility was established from the original sibling list, but an
      // earlier folded store may have remapped this load's tensor input. Keep
      // the defining load used for window reconstruction synchronized with
      // the current SSA statement. The assignment Var remains the stable key
      // used to remove the original full load after a successful rewrite.
      if (auto current_assign = As<AssignStmt>(current)) {
        auto def = direct_defs.find(current_assign->var_.get());
        auto current_call = As<Call>(current_assign->value_);
        if (def != direct_defs.end() && current_call && IsOp(current_call, "tile.load")) {
          def->second = current_assign;
        }
      }

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
          const uint64_t l0c_bytes = bh ? bh->GetCapacityBytes(MemorySpace::Acc) : 0;
          const auto acc_bytes = src_acc && m_ci && n_ci
                                     ? utils::StaticPhysicalAllocationBytes(src_ty, MemorySpace::Acc, bh)
                                     : std::optional<uint64_t>{};
          if (src_acc && m_ci && n_ci && l0c_bytes && acc_bytes && *acc_bytes <= l0c_bytes) {
            const Var* cb = cast_as->var_.get();
            auto uc = sibling_index.use_counts.find(cb);
            auto mo = sibling_index.matmul_operand_uses.find(cb);
            const int cb_uses = uc == sibling_index.use_counts.end() ? 0 : uc->second;
            const int cb_mm = mo == sibling_index.matmul_operand_uses.end() ? 0 : mo->second;
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
        const Stmt* next_effect =
            next_non_load[i] < op->stmts_.size() ? op->stmts_[next_non_load[i]].get() : nullptr;
        const auto consumer = AnalyzeMatmulOutputConsumer(assign, sibling_index, next_effect);
        if (auto tiling =
                AnalyzeMatmul(assign, hints, /*force_output_stationary=*/false,
                              /*output_box_alignment=*/std::nullopt, &direct_defs,
                              consumer && consumer->supports_output_grid,
                              /*source_reduction=*/nullptr, consumer ? consumer->fixed_output_m : 0)) {
          if (!tiling->needs_mn_tiling()) {
            // Whole output fits L0c — tile K only.  k < K here (k == K with
            // m == M, n == N needs no matmul tiling and was skipped by
            // AnalyzeMatmul); the chooser may return a non-divisor k that
            // BuildKLoopRewrite peels.
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
          INTERNAL_CHECK_SPAN(consumer, assign->span_)
              << "Internal error: analyzed matmul is missing its output consumer facts";
          const int result_uses = consumer->result_uses;
          const int operand_uses = consumer->operand_uses;
          const auto scratch_dtype = consumer->scratch_dtype;
          const Var* remap_target = consumer->remap_target;
          const Var* extra_remap = consumer->extra_remap;
          // Mat-scratch: result consumed entirely on-chip at matmul-operand
          // positions — assemble the sub-tiles into an L1/Mat scratch and remap the
          // matmul result (or its downcast) to it.  Emitted at the matmul site
          // (like the K-only rewrite), with no store to defer.  Checked before the
          // direct-store fold so its hints stay clean.
          // #1908 guard: under the legacy PyPTO planner, a chained Mat-scratch
          // producer must stay output-stationary. Its opportunistic reuse classes
          // cannot subdivide an expired monolithic A/B-stationary operand panel for
          // the consumer's smaller pipelined buffers, so a later slot is allocated
          // past L0A/L0B capacity. DSA_RP and PTOAS place buffers from their actual
          // lifetimes and can pack this layout, so preserve their chooser-selected
          // stationarity. OS is always a legal fallback for PyPTO and the oversized
          // producer must be tiled, so re-choose OS-only rather than defer.
          const MatmulTiling* fold_tiling = &*tiling;
          std::optional<MatmulTiling> os_tiling;
          const auto* ctx = PassContext::Current();
          const MemoryPlanner planner = ctx ? ctx->GetMemoryPlanner() : MemoryPlanner::PyPTO;
          if (planner == MemoryPlanner::PyPTO && consumer->fixed_output_m == 0 &&
              tiling->stationarity != utils::Stationarity::kOutputStationary) {
            std::vector<Diagnostic> discard;  // the first AnalyzeMatmul already emitted the hints
            os_tiling = AnalyzeMatmul(assign, discard, /*force_output_stationary=*/true,
                                      /*output_box_alignment=*/std::nullopt, &direct_defs,
                                      consumer->supports_output_grid);
            if (os_tiling) fold_tiling = &*os_tiling;
          }
          if (auto ms = TryFoldMatScratch(*fold_tiling, result_uses, operand_uses, scratch_dtype, hints)) {
            // Mat-scratch may re-choose output stationarity for #1908. Base
            // source-load removal on the schedule we actually emit: the first
            // choice may N-tile while the forced OS choice keeps full N (or
            // vice versa).
            if (fold_tiling->bias_load_def && fold_tiling->n != fold_tiling->N) {
              retroactively_dropped.insert(fold_tiling->bias_load_def->var_.get());
            }
            for (auto& s : ms->first) out.push_back(std::move(s));
            remap[remap_target] = ms->second;
            if (extra_remap) {
              remap[extra_remap] = ms->second;  // c -> scratch (cast reads the scratch)
              dropped.insert(remap_target);     // ... and drop the now-dead cast def
            }
            changed = true;
            continue;
          }
          const AssignStmt* store_stmt = consumer->store_stmt;
          const bool reconstructs_bias = tiling->bias_load_def && tiling->n != tiling->N;
          const bool bias_snapshot_reaches_store =
              !reconstructs_bias || !store_stmt ||
              (next_non_load[i] < op->stmts_.size() &&
               op->stmts_[next_non_load[i]].get() == static_cast<const Stmt*>(store_stmt));
          if (reconstructs_bias && store_stmt && !bias_snapshot_reaches_store) {
            hints.emplace_back(
                DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-011",
                "tile.matmul_bias N-window loads would move to its folded consumer store across an "
                "intervening effect; left untouched to preserve the original bias snapshot",
                tiling->assign->span_);
          }
          if (bias_snapshot_reaches_store) {
            if (auto fold = TryFoldMNTiling(*tiling, result_uses, store_stmt, hints)) {
              if (tiling->bias_load_def && tiling->n != tiling->N) {
                retroactively_dropped.insert(tiling->bias_load_def->var_.get());
              }
              remap[fold->store_result_var.get()] = fold->return_var;
              pending_folds.emplace(static_cast<const Stmt*>(fold->store), std::move(*fold));
              changed = true;
              continue;  // drop the matmul; sub-tile stmts emit at the store site
            }
          }
          // M/N tiling not applicable — fall through and leave it untouched.
        }
      }
      auto visited = VisitStmt(current);
      if (visited.get() != child.get()) changed = true;
      out.push_back(visited);
    }
    if (!retroactively_dropped.empty()) {
      std::vector<StmtPtr> filtered;
      filtered.reserve(out.size());
      for (auto& stmt : out) {
        auto assign = As<AssignStmt>(stmt);
        if (assign && retroactively_dropped.count(assign->var_.get())) continue;
        filtered.push_back(std::move(stmt));
      }
      out = std::move(filtered);
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }

 private:
  std::unordered_set<const ForStmt*> pipeline_dbc_plan_;
};

FunctionPtr TransformFunction(const FunctionPtr& func, std::vector<Diagnostic>& hints) {
  if (!func || !func->body_) return func;
  if (!IsInCoreType(func->func_type_)) return func;
  // Canonical loop-carried split-K output tiling changes both the Acc inventory
  // and loop identities. Rewrite it first so the #2131 dbC capacity/placement
  // plan is computed from the exact IR the ordinary AutoTile phase will visit.
  auto canonical = RewriteCanonicalSplitKAcc(func, hints);
  AutoTileMutator mutator(BuildPipelineDbCPlan(canonical));
  auto new_body = mutator.VisitStmt(canonical->body_);
  for (auto& d : mutator.hints) hints.push_back(std::move(d));
  if (new_body == canonical->body_) return canonical;
  auto new_func = MutableCopy(canonical);
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
