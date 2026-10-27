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

#ifndef SRC_IR_TRANSFORMS_AUTO_TILE_MATMUL_L0_INTERNAL_H_
#define SRC_IR_TRANSFORMS_AUTO_TILE_MATMUL_L0_INTERNAL_H_

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace auto_tile_matmul_l0 {

// Private contracts shared by selection, IR emission, and pipeline planning.
inline constexpr const char* kPassName = "AutoTileMatmulL0";

/// Direct sibling definitions visible to one SeqStmts-level matmul rewrite.
/// Bias N-tiling uses this to prove that a Mat bias came from a natural
/// ``tile.load`` and can therefore be reloaded as an independent legal window.
using DirectDefMap = std::unordered_map<const Var*, AssignStmtPtr>;

enum class MatmulKind {
  kFresh,
  kAccumulate,
  kBias,
};

/// Operands + chosen L0 tile shape for a tileable matmul.  Produced by
/// ``AnalyzeMatmul``; the caller dispatches on ``needs_mn_tiling()`` to build
/// either the whole-output K-loop or the unrolled M/N grid of sub-tiles.
struct MatmulTiling {
  AssignStmtPtr assign;
  MatmulKind kind = MatmulKind::kFresh;
  VarPtr lhs;         ///< [M, K] left operand — Mat (or Vec for the PV pattern; see stage_lhs_to_mat)
  VarPtr rhs;         ///< [K, N] right operand — Mat
  VarPtr bias;        ///< optional [1, N] bias for tile.matmul_bias
  CallPtr bias_load;  ///< defining Mat tile.load, required when bias is N-tiled
  AssignStmtPtr bias_load_def;  ///< removable single-use snapshot load for N-window reconstruction
  VarPtr acc_init;              ///< caller-provided accumulator for matmul_acc; null for fresh matmul/bias
  /// Caller's ``init_cond`` predicate (``tile.matmul_acc`` arity 4); null
  /// otherwise.  Copied onto ``KLoopRewrite::init_cond`` by ``MakeKLoop``.
  ExprPtr init_cond = nullptr;
  bool stage_lhs_to_mat = false;
  int64_t M = 0, N = 0, K = 0;
  int64_t m = 0, n = 0, k = 0;
  /// Work bounds shared by selection and M/N emission. M/N retain the original
  /// physical boxes for input contracts and Mat-scratch allocation.
  int64_t grid_m = 0, grid_n = 0;
  /// Logical output extents. Ordinarily identical to M/N; a box-padded
  /// boundary matmul keeps the larger physical M/N in the operands/result.
  ExprPtr output_valid_m = nullptr;
  ExprPtr output_valid_n = nullptr;
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
  /// floats the drains past the next matmul, keeping the two tiles co-live.
  /// Under PTOAS, InitMemRef keeps the overlapping-live-range buffers distinct
  /// and ptoas places them. Under PYPTO, flat depth-2 pipeline membership keeps
  /// MemoryReuse from coalescing them; DSA_RP consumes the same relation as a
  /// strict placement separation before its capacity fallback. Set from
  /// L0TileResult::double_buffer_c; only true for full-K tiles (see the assert
  /// in AnalyzeMatmul).
  bool double_buffer_c = false;
  [[nodiscard]] bool is_acc() const { return kind == MatmulKind::kAccumulate; }
  /// True when the chosen L0 tile is smaller than the [M, N] output on either
  /// axis — the output Acc would overflow L0c, so the output must be tiled.
  [[nodiscard]] bool needs_mn_tiling() const { return m != M || n != N; }
};

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
  std::unordered_map<const Var*, VarPtr> vars;              ///< owning pointers for type inspection

 protected:
  void VisitVarLike_(const VarPtr& op) override;
  // Skip the LHS (a def); count only reads in the RHS value.
  void VisitStmt_(const AssignStmtPtr& op) override;
  // A *direct* Var at a matrix-operand position (``tile.matmul`` /
  // ``tile.matmul_bias`` args {0,1}; ``tile.matmul_acc`` args {1,2} — arg 0 is
  // the Acc accumulator, NOT a matrix operand) is a Mat-safe consumer use: the
  // consumer K-tiles that operand, so an L1/Mat scratch produced upstream is
  // legal there. Classifying by operand index is essential — a scratch fed to
  // ``matmul_acc`` arg 0 would be an illegal Mat-for-Acc substitution and must
  // stay deferred. The bias argument is likewise not a matrix-operand use.
  void VisitExpr_(const CallPtr& op) override;

 private:
  bool in_matmul_operand_ = false;
};

// Shared static-shape checks; these do not mutate IR or query pass configuration.
int64_t AlignStaticExtent(int64_t extent, int64_t alignment, const Span& span);
bool IsStatic2DInSpaces(const TileTypePtr& tile, std::initializer_list<MemorySpace> allowed, int64_t& out_d0,
                        int64_t& out_d1);
bool IsStatic2DTuple(const ExprPtr& expr, int64_t dim0, int64_t dim1);
uint32_t DTypeBytes(const DataType& dt);

/// Select a legal L0 schedule; unsupported cases retain the original IR and may emit hints.
std::optional<MatmulTiling> AnalyzeMatmul(
    const AssignStmtPtr& assign, std::vector<Diagnostic>& hints, bool force_output_stationary = false,
    std::optional<tile_view_semantics::BoxedTileAlignment> output_box_alignment = std::nullopt,
    const DirectDefMap* direct_defs = nullptr, bool restrict_output_grid = false,
    const ForStmtPtr& source_reduction = nullptr, int64_t fixed_output_m = 0);

/// Plan against the post-canonical-rewrite function so loop identities and capacity agree.
/// Automatic annotation remains disabled; explicit caller buffering is preserved.
std::unordered_set<const ForStmt*> BuildPipelineDbCPlan(const FunctionPtr& func);

}  // namespace auto_tile_matmul_l0
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_AUTO_TILE_MATMUL_L0_INTERNAL_H_
