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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/auto_tile_matmul_l0/internal.h"

namespace pypto {
namespace ir {
namespace auto_tile_matmul_l0 {

namespace {

/// Reject a manually materialized cube operand that cannot physically fit in
/// its L0 space.  This check is deliberately narrower than the allocator's
/// whole-space capacity check: it fires only when one static Left/Right tile is
/// impossible by itself, so reuse, pipeline-depth shedding, and planner choice
/// cannot change the outcome.
void CheckExplicitL0OperandCapacity(const VarPtr& operand, const TileTypePtr& tile,
                                    MemorySpace expected_space, const char* operand_role,
                                    const char* hardware_name, const char* op_name,
                                    const backend::BackendHandler* handler) {
  if (!operand || !tile || !handler || tile->GetMemorySpace() != expected_space) return;

  const auto required = utils::StaticPhysicalAllocationBytes(tile, expected_space, handler);
  if (!required) return;  // This targeted diagnostic requires an exact static footprint.

  INTERNAL_CHECK_SPAN(expected_space == MemorySpace::Left || expected_space == MemorySpace::Right,
                      operand->span_)
      << "Internal error: explicit matmul operand capacity check only accepts Left or Right";
  const uint64_t capacity = handler->GetCapacityBytes(expected_space);
  if (capacity == 0 || *required <= capacity) return;

  // This is a user-authored scheduling error, not an internal pass invariant.
  // Construct the ValueError directly so the diagnostic keeps the DSL source
  // location without appending FatalLogger's implementation-facing
  // "Check failed" line.
  std::ostringstream message;
  message << op_name << " " << operand_role << " operand '"
          << auto_name::GetCompatibleBaseName(operand->name_hint_) << "' is already resident in "
          << MemorySpaceToString(expected_space) << " (" << hardware_name << ") with physical shape "
          << FormatShape(tile->shape_) << " and dtype " << tile->dtype_.ToString() << ", requiring "
          << *required << " bytes, but the active backend provides " << capacity
          << " bytes. AutoTileMatmulL0 does not retile operands already placed in Left or Right. Keep this "
             "operand in Mat and pass it directly to "
          << op_name << " so AutoTileMatmulL0 can choose legal L0 tiles, or manually extract a smaller "
          << MemorySpaceToString(expected_space) << " tile.";
  if (operand->span_.is_valid()) message << " [" << operand->span_.to_string() << "]";
  throw pypto::ValueError(message.str());
}

}  // namespace

/// Decide whether `assign` is a Mat-resident matmul we know how to tile, and if
/// so which L0 tile shape to use.  Returns the tiling plan on success;
/// otherwise nullopt and (when useful) appends a PerfHint.  The caller
/// dispatches K-only vs M/N tiling on ``MatmulTiling::needs_mn_tiling()``.
/// A nonzero fixed_output_m is the NZ direct-store constraint: preserve all
/// parent rows and limit each emitted N tile to the NZ FIXPIPE field width.
std::optional<MatmulTiling> AnalyzeMatmul(
    const AssignStmtPtr& assign, std::vector<Diagnostic>& hints, bool force_output_stationary,
    std::optional<tile_view_semantics::BoxedTileAlignment> output_box_alignment,
    const DirectDefMap* direct_defs, bool restrict_output_grid, bool keep_output_shape,
    const ForStmtPtr& source_reduction, int64_t fixed_output_m) {
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_) return std::nullopt;

  // Plain, accumulating, and bias matmuls share the L0 tile chooser. Parse the
  // operation kind once so operand indexing and later legality decisions cannot
  // drift into inconsistent combinations of boolean flags.
  MatmulKind kind;
  if (IsOp(call, "tile.matmul")) {
    kind = MatmulKind::kFresh;
  } else if (IsOp(call, "tile.matmul_acc")) {
    kind = MatmulKind::kAccumulate;
  } else if (IsOp(call, "tile.matmul_bias")) {
    kind = MatmulKind::kBias;
  } else {
    return std::nullopt;
  }
  const bool is_acc = kind == MatmulKind::kAccumulate;
  const bool is_bias = kind == MatmulKind::kBias;
  const std::string& op_name = call->op_->name_;

  // Operand layout: (lhs, rhs) for matmul; (acc, lhs, rhs[, init_cond]) for
  // matmul_acc; (lhs, rhs, bias) for matmul_bias.
  // Use ``AsVarLike`` for the operands so IterArg (Var subclass) is accepted —
  // this is the common case for the accumulator inside a pipelined K-loop.
  const size_t expected_arity = kind == MatmulKind::kFresh ? 2u : 3u;
  // Only the accumulate kind carries init_cond (arity 4); a fresh matmul has no
  // accumulator to predicate and matmul_bias has no init_cond operand.
  const bool has_init_cond = is_acc && call->args_.size() == expected_arity + 1u;
  if (call->args_.size() != expected_arity && !has_init_cond) return std::nullopt;
  const size_t lhs_idx = is_acc ? 1u : 0u;
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
  if (is_acc) {
    acc_var = AsVarLike(call->args_[0]);
    if (!acc_var) return std::nullopt;
    auto acc_tile = As<TileType>(acc_var->GetType());
    if (!acc_tile || acc_tile->shape_.size() != 2) return std::nullopt;
  }

  // The ordinary rewrite below deliberately owns only Mat/Vec -> L0 tiling.
  // Already-Left/Right operands express a manual L0 schedule and are otherwise
  // left untouched. Diagnose the one unambiguously invalid manual case here:
  // an individual operand whose physical tile is larger than the entire L0
  // space. Failing before MemoryReuse avoids a misleading packing fallback
  // warning for a failure no packing strategy can repair.
  const auto* ctx = PassContext::Current();
  const auto* diagnostic_handler =
      pypto::backend::BackendConfig::IsConfigured()
          ? (ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler())
          : nullptr;
  CheckExplicitL0OperandCapacity(lhs, lhs_tile, MemorySpace::Left, "left", "L0A", op_name.c_str(),
                                 diagnostic_handler);
  CheckExplicitL0OperandCapacity(rhs, rhs_tile, MemorySpace::Right, "right", "L0B", op_name.c_str(),
                                 diagnostic_handler);

  VarPtr bias_var;
  TileTypePtr bias_tile;
  CallPtr bias_load;
  AssignStmtPtr bias_load_def;
  int64_t bias_n = 0;
  if (is_bias) {
    bias_var = AsVarLike(call->args_[2]);
    if (!bias_var) return std::nullopt;
    bias_tile = As<TileType>(bias_var->GetType());
    int64_t bias_m = 0;
    if (!IsStatic2DInSpaces(bias_tile, {MemorySpace::Mat, MemorySpace::Bias}, bias_m, bias_n) ||
        bias_m != 1) {
      return std::nullopt;
    }
    if (bias_tile->GetMemorySpace() == MemorySpace::Mat && direct_defs) {
      auto def = direct_defs->find(bias_var.get());
      if (def != direct_defs->end()) {
        auto candidate = As<Call>(def->second->value_);
        if (candidate && IsOp(candidate, "tile.load") && candidate->args_.size() == 4) {
          auto offsets = As<MakeTuple>(candidate->args_[1]);
          if (offsets && offsets->elements_.size() == 2 && IsStatic2DTuple(candidate->args_[2], 1, bias_n) &&
              IsStatic2DTuple(candidate->args_[3], 1, bias_n)) {
            bias_load = candidate;
            bias_load_def = def->second;
          }
        }
      }
    }
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
  // New matmul_bias coverage intentionally starts at the native cube boundary:
  // both matrix operands are already in Mat. Keep the historical Vec-left
  // exception for plain matmul/matmul_acc K-tiling, but do not widen it here.
  if (is_bias && lhs_tile->GetMemorySpace() != MemorySpace::Mat) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-011",
                       "tile.matmul_bias: a Vec-resident left matrix operand is outside the biased-matmul "
                       "AutoTile contract; left untouched (stage both matrix operands in Mat)",
                       assign->span_);
    return std::nullopt;
  }
  // K mismatch is an ill-typed matmul — the op verifier should have caught it
  // upstream.  Treat as an internal invariant.
  INTERNAL_CHECK(K_lhs == K_rhs) << op_name << ": K dimensions don't match (lhs K=" << K_lhs
                                 << ", rhs K=" << K_rhs << ")";
  INTERNAL_CHECK_SPAN(!is_bias || bias_n == N, assign->span_)
      << "Internal error: tile.matmul_bias bias N does not match rhs N";
  const int64_t K = K_lhs;

  uint32_t bytes_a = DTypeBytes(lhs_tile->dtype_);
  uint32_t bytes_b = DTypeBytes(rhs_tile->dtype_);
  // Output dtype is set by the matmul op's deduction (FP32 / INT32 today, but
  // future cube paths may add half-precision accumulation).  Read from the
  // call's result type rather than hardcoding so the chooser sees the actual
  // accumulator footprint.
  auto out_tile = As<TileType>(call->GetType());
  INTERNAL_CHECK(out_tile) << "Internal error: tile.matmul result is not a TileType";
  const auto output_valid_shape = tile_view_semantics::GetEffectiveTileView(*out_tile).valid_shape;
  INTERNAL_CHECK_SPAN(output_valid_shape.size() == 2, assign->span_)
      << "Internal error: tile.matmul result valid_shape must be 2D";
  uint32_t bytes_c = DTypeBytes(out_tile->dtype_);
  if (bytes_a == 0 || bytes_b == 0 || bytes_c == 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-003",
                       op_name + ": unsupported sub-byte dtype on operand or accumulator — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  // A Mat/Vec-resident matmul reaches the chooser and therefore requires an
  // active backend. Preserve the historical fail-loud behavior here; the
  // explicit-L0 diagnostic above is optional when capacity is unavailable.
  const auto* handler = ctx ? ctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  INTERNAL_CHECK(handler) << "Internal error: BackendHandler is null";
  if (is_bias && bias_tile->GetMemorySpace() == MemorySpace::Mat &&
      !handler->SupportsMatToBiasMove(bias_tile->dtype_, out_tile->dtype_)) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-011",
                       "tile.matmul_bias: this backend cannot move a " + bias_tile->dtype_.ToString() +
                           " Mat bias into a " + out_tile->dtype_.ToString() +
                           " Bias tile; left untouched (use a backend-supported dtype pair or provide a "
                           "Bias-resident source)",
                       assign->span_);
    return std::nullopt;
  }

  utils::L0TileConfig cfg;
  cfg.M = static_cast<int>(M);
  cfg.N = static_cast<int>(N);
  cfg.K = static_cast<int>(K);
  const uint64_t l0a_bytes = handler->GetCapacityBytes(MemorySpace::Left);
  const uint64_t l0b_bytes = handler->GetCapacityBytes(MemorySpace::Right);
  const uint64_t l0c_bytes = handler->GetCapacityBytes(MemorySpace::Acc);
  // The general capacity API is 64-bit; the L0 chooser still uses 32-bit limits.
  // Keep that local representation from silently truncating future SoC data.
  INTERNAL_CHECK_SPAN(std::max({l0a_bytes, l0b_bytes, l0c_bytes}) <= std::numeric_limits<uint32_t>::max(),
                      assign->span_)
      << "Internal error: SoC capacity exceeds the L0 tile chooser's uint32_t limits";
  cfg.l0a_bytes = static_cast<uint32_t>(l0a_bytes);
  cfg.l0b_bytes = static_cast<uint32_t>(l0b_bytes);
  cfg.l0c_bytes = static_cast<uint32_t>(l0c_bytes);
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
  cfg.l0c_align_m = handler->GetL0cMAlignment(out_tile->dtype_);
  // Extracts/moves materialize the implicit Left/Right layouts, even when
  // the input lives in Mat or Vec. Their packed K axes are dtype-dependent:
  // INT8 requires 32 elements, not the backend's generic 16-element grid.
  const auto left_alignment = tile_view_semantics::GetBoxedTileAlignment(
      tile_view_semantics::GetImplicitTileView(lhs_tile->shape_, MemorySpace::Left), lhs_tile->dtype_);
  const auto right_alignment = tile_view_semantics::GetBoxedTileAlignment(
      tile_view_semantics::GetImplicitTileView(rhs_tile->shape_, MemorySpace::Right), rhs_tile->dtype_);
  INTERNAL_CHECK_SPAN(left_alignment && right_alignment, assign->span_)
      << "Internal error: L0 matrix operand boxing is not statically known";
  cfg.align_k =
      static_cast<int>(std::max<int64_t>({cfg.align_k, left_alignment->cols, right_alignment->rows}));
  if (is_bias) {
    const auto lhs_alignment = tile_view_semantics::GetBoxedTileAlignment(*lhs_tile);
    const auto rhs_alignment = tile_view_semantics::GetBoxedTileAlignment(*rhs_tile);
    if (!lhs_alignment || !rhs_alignment) {
      hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-011",
                         "tile.matmul_bias: matrix operand boxed alignment is not statically known; left "
                         "untouched",
                         assign->span_);
      return std::nullopt;
    }
    cfg.align_m = static_cast<int>(std::max<int64_t>(cfg.align_m, lhs_alignment->rows));
    cfg.align_n = static_cast<int>(std::max<int64_t>(cfg.align_n, rhs_alignment->cols));
    cfg.align_k =
        static_cast<int>(std::max<int64_t>({cfg.align_k, lhs_alignment->cols, rhs_alignment->rows}));
    cfg.box_align_m = static_cast<int>(lhs_alignment->rows);
    cfg.box_align_n = static_cast<int>(rhs_alignment->cols);
  }
  int single_bias_max_n = 0;
  if (is_bias) {
    const uint64_t bias_bytes = handler->GetCapacityBytes(MemorySpace::Bias);
    INTERNAL_CHECK_SPAN(bias_bytes > 0, assign->span_)
        << "Internal error: backend reports a zero-byte bias table";
    const uint64_t max_bias_n = bias_bytes / bytes_c;
    if (max_bias_n < static_cast<uint64_t>(handler->GetMinL0TileDim())) {
      hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-011",
                         "tile.matmul_bias: the backend bias table cannot hold one minimum-width "
                         "accumulator-typed bias tile; left untouched",
                         assign->span_);
      return std::nullopt;
    }
    single_bias_max_n = static_cast<int>(
        std::min<uint64_t>(max_bias_n, static_cast<uint64_t>(std::numeric_limits<int>::max())));
    cfg.max_n = single_bias_max_n;
    // A Mat-backed bias is rebuilt inside the emitted output-grid pipelines,
    // so its Bias allocation is replicated by one or two pipeline levels.
    // An already-Bias-resident operand is defined outside those loops and is
    // reused directly; it consumes exactly one architectural Bias slot.
    if (bias_tile->GetMemorySpace() == MemorySpace::Mat) {
      cfg.max_n_pipelined = single_bias_max_n / 2;
      cfg.max_n_nested_pipelined = single_bias_max_n / 4;
    }
  }
  if (output_box_alignment) {
    INTERNAL_CHECK_SPAN(output_box_alignment->rows > 0 && output_box_alignment->cols > 0 &&
                            output_box_alignment->rows <= std::numeric_limits<int>::max() &&
                            output_box_alignment->cols <= std::numeric_limits<int>::max(),
                        assign->span_)
        << "Internal error: canonical split-K operand boxing has an invalid alignment ["
        << output_box_alignment->rows << ", " << output_box_alignment->cols << "]";
    cfg.box_align_m = static_cast<int>(output_box_alignment->rows);
    cfg.box_align_n = static_cast<int>(output_box_alignment->cols);
  }
  cfg.min_m = handler->GetMinL0TileDim();
  cfg.min_n = handler->GetMinL0TileDim();
  cfg.min_k = std::max(handler->GetMinL0TileDim(), cfg.align_k);
  if (is_bias) {
    cfg.min_m = std::max(cfg.min_m, cfg.align_m);
    cfg.min_n = std::max(cfg.min_n, cfg.align_n);
    // The architectural Bias memory has no Bias-to-Bias sub-window operation.
    // Restrict an already-resident source to full N during selection so an
    // unsupported narrower-N optimum cannot hide a legal M/K-tiled candidate.
    if (bias_tile->GetMemorySpace() == MemorySpace::Bias) {
      cfg.min_n = static_cast<int>(N);
    }
  }
  if (is_bias && (M % cfg.align_m != 0 || N % cfg.align_n != 0 || K % cfg.align_k != 0)) {
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-011",
        "tile.matmul_bias: M, N, and K must be multiples of the operands' boxed layout alignment so every "
        "Mat-to-L0/Bias window has a legal physical box; left untouched",
        assign->span_);
    return std::nullopt;
  }
  // Operand-stationary (the chooser's allow_a/b_stationary; dbA/dbB derived):
  // pin the chosen operand in a SINGLE-buffered (full) L0 buffer across the moving
  // grid (requires k == K), streaming the other operand double-buffered.
  // BuildFullKPipelined realizes it by making the stationary (outer) loop
  // ForKind::Sequential — the held operand then uses the full L0 buffer the chooser
  // budgeted (no /2) and is loaded once per outer step (no re-stream across the
  // moving axis). The chooser adopts A/B-stationary only on a strictly lower wall,
  // so it stays output-stationary for compute-bound shapes.
  // Chained Mat-scratch producers under the legacy PyPTO planner pass
  // force_output_stationary=true to turn these off (see the #1908 guard at the
  // fold site): that allocator cannot pack a single-buffered A/B-stationary
  // producer against the consumer matmul's pipelined operands. DSA_RP and
  // PTOAS retain the operand-stationary candidates.
  cfg.allow_a_stationary = !force_output_stationary;
  cfg.allow_b_stationary = !force_output_stationary;
  // PyPTO and DSA-RP select their genuine single-C optimum at full capacity,
  // rather than strip buffering from geometry chosen at half capacity. Preserve
  // PTOAS's existing dbC enumeration: forcing its chained Mat-scratch kernels
  // to single-C exposes an unhandled native operand-allocation limitation.
  // This is planner compatibility, not a claim that PTOAS dbC is faster. The
  // legacy PyPTO opt-in does not alter either planner policy.
  cfg.allow_double_buffer_c = ctx && ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS;
  // tile.matmul_acc threads the caller's accumulator into the K-loop's
  // iter-arg, so each invocation reads C from L1 at start and writes back at
  // end (gamma_c = 2 in the chooser's traffic model).  Plain tile.matmul
  // starts from a fresh Acc placeholder so C is write-only (gamma_c = 1).
  cfg.c_read = is_acc;
  if (source_reduction) {
    const auto iterations = transform_utils::EvalConstTripCount(source_reduction);
    cfg.reduction_iterations = iterations && *iterations > 0 && *iterations <= std::numeric_limits<int>::max()
                                   ? static_cast<int>(*iterations)
                                   : 0;
    cfg.reduction_pipeline_stages = std::max(1, source_reduction->GetAttr<int>(kPipelineStagesAttr, 1));
    cfg.independent_output_tiles = true;
    // A canonical reduction overwrites its placeholder on its first source
    // iteration and drains once after all iterations. It has no held-panel
    // output sweep or cross-output accumulator ping-pong emitter.
    cfg.c_read = false;
    cfg.allow_a_stationary = cfg.allow_b_stationary = cfg.allow_double_buffer_c = false;
    if (cfg.reduction_iterations == 0) {
      hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
                         "canonical split-K trip count is unknown or outside the cost model range; "
                         "using one source block as a single-C cost lower bound",
                         source_reduction->span_);
    }
  }
  cfg.allow_padding = false;
  // Permit a non-divisor final K block: the chooser may return a k that does not
  // divide K, and BuildKLoopRewrite peels the partial last K iteration.  The peel
  // is only valid when K obeys the emitted operand K alignment — then the tail
  // K - floor(K/k)*k obeys that same alignment. A misaligned K has no valid
  // K-tiling (any tail or whole-K block violates operand boxing), so skip it
  // here with a perf hint rather than emit invalid extracts (the pre-roofline path
  // also bailed on unsupported K).
  cfg.allow_k_boundary = true;
  if (K % cfg.align_k != 0) {
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-007",
        op_name + ": K=" + std::to_string(K) + " is not a multiple of the required L0 operand K alignment " +
            std::to_string(cfg.align_k) + " — misaligned K blocks and tails are unsupported; left untouched.",
        assign->span_);
    return std::nullopt;
  }

  // A supported output placement may omit complete padding-only boxes, but
  // selection and emission must use the same work rectangle. Otherwise a
  // physical N=416 / valid N=17 source produces stores at offset 256 with
  // extent zero. Unknown or empty valid extents retain the existing policy;
  // a partial last fractal remains real work with its original valid shape.
  // Do not apply this to caller-owned accumulators or unsupported consumers:
  // their K-only result must retain the original physical type. Vec-left M/N
  // placement remains outside this pass's current local-grid contract.
  // A Mat bias without a reconstructible load snapshot has the same full-N
  // requirement for this optimization: narrowing it would replace a legal
  // K-only rewrite with an unsupported bias-window fold. Keep its old grid.
  const bool can_bound_bias = !is_bias || bias_load || bias_tile->GetMemorySpace() == MemorySpace::Bias;
  if (restrict_output_grid && !keep_output_shape && lhs_tile->GetMemorySpace() == MemorySpace::Mat &&
      can_bound_bias) {
    auto work_extent = [&](int64_t physical, const ExprPtr& valid, int alignment) {
      const auto extent = As<ConstInt>(valid);
      if (!extent || extent->value_ <= 0) return physical;
      return std::min(physical, AlignStaticExtent(extent->value_, alignment, assign->span_));
    };
    cfg.M = static_cast<int>(work_extent(M, output_valid_shape[0], cfg.align_m));
    // A resident Bias tile cannot be windowed. Respect its full-N minimum,
    // without expanding a physical source that is below the chooser minimum.
    cfg.N = static_cast<int>(std::min<int64_t>(
        N, std::max<int64_t>(cfg.min_n, work_extent(N, output_valid_shape[1], cfg.align_n))));
  }

  if (keep_output_shape) {
    // The consumer cannot accept an M/N grid. Search only reductions that keep
    // the original output intact, rather than leaving oversized L0 operands
    // for an allocator to pack. A K-only reduction has one accumulator and no
    // output-grid pipeline, so dbC and operand-stationary regimes cannot apply.
    cfg.min_m = cfg.M;
    cfg.min_n = cfg.N;
    cfg.allow_a_stationary = false;
    cfg.allow_b_stationary = false;
    cfg.allow_double_buffer_c = false;
  }

  if (fixed_output_m != 0) {
    // An NZ Acc store must cover the entire parent row axis. Restrict the
    // solver before selection, rather than rejecting its unconstrained M/N
    // optimum and overlooking a legal N/K-only plan. All regimes share these
    // bounds, including operand-stationary and double-buffered-C candidates.
    INTERNAL_CHECK_SPAN(fixed_output_m == M && cfg.M == M, assign->span_)
        << "Internal error: NZ output tiling must preserve the complete physical M extent";
    cfg.min_m = static_cast<int>(M);
    // This argument denotes the NZ full-row store contract, whose FIXPIPE
    // nSize field is 12 bits. The 16-column grid makes 4080 the largest tile.
    const int max_nz_store_columns = handler->GetAccToNzGmStoreMaxCols();
    CHECK_SPAN(max_nz_store_columns > 0, assign->span_)
        << "NZ tile.store is supported only on A2/A3. Store to pl.ND on this backend.";
    cfg.max_n = cfg.max_n == 0 ? max_nz_store_columns : std::min(cfg.max_n, max_nz_store_columns);
    if (cfg.max_n_pipelined != 0) cfg.max_n_pipelined = std::min(cfg.max_n_pipelined, cfg.max_n);
    if (cfg.max_n_nested_pipelined != 0) {
      cfg.max_n_nested_pipelined = std::min(cfg.max_n_nested_pipelined, cfg.max_n);
    }
  }

  utils::L0TileResult res;
  try {
    res = utils::ChooseL0Tile(cfg);
  } catch (const pypto::ValueError& e) {
    CHECK_SPAN(fixed_output_m == 0, assign->span_)
        << "NZ tile.store has no legal N/K tiling plan that preserves all " << M
        << " destination rows. The full row extent must fit the L0A operand and a minimum-width L0C "
        << "tile. Split independent matrices along a batch axis, or store to pl.ND. " << e.what();
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-005",
                       op_name + ": ChooseL0Tile rejected configuration — left untouched. " + e.what(),
                       assign->span_);
    return std::nullopt;
  }
  INTERNAL_CHECK_SPAN(fixed_output_m == 0 || res.m == M, assign->span_)
      << "Internal error: NZ output tiling selected a partial M tile";

  // Already L0-sized — nothing to do.
  if (res.m == M && res.n == N && res.k == K) return std::nullopt;

  // A Bias-resident vector can be reused for K-tiling or M-only tiling, but the
  // architectural bias table cannot form an N sub-window. Normal pre-inference
  // inputs are Mat-resident and take the supported load-backed Mat window ->
  // Bias move path.
  if (is_bias && res.n != N && bias_tile->GetMemorySpace() == MemorySpace::Bias) {
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
        "tile.matmul_bias needs N tiling, but its source is already resident in the architectural Bias "
        "buffer; forming an N sub-window would require an unsupported Bias-to-Bias transfer, so the call "
        "is left untouched (keep the pre-inference bias source in Mat)",
        assign->span_);
    return std::nullopt;
  }
  if (is_bias && res.n != N && bias_tile->GetMemorySpace() == MemorySpace::Mat && !bias_load) {
    // An unsupported bias window rules out this N grid, not a K-only plan.
    // Retry once with the full output to reuse the original bias snapshot.
    // Preserve source-reduction scheduling and NZ store limits in the retry.
    if (!keep_output_shape) {
      std::vector<Diagnostic> fallback_hints;
      if (auto k_only = AnalyzeMatmul(assign, fallback_hints, true, output_box_alignment, direct_defs,
                                      restrict_output_grid, true, source_reduction, fixed_output_m)) {
        for (const auto& hint : fallback_hints) hints.push_back(hint);
        return k_only;
      }
    }
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-011",
        "tile.matmul_bias needs N tiling, but its Mat bias is not a single-use full-rectangular [1, N] "
        "tile.load separated from the call only by read-only sibling loads; a legal snapshot-preserving "
        "bias window cannot be reconstructed, so the call is left untouched",
        assign->span_);
    return std::nullopt;
  }

  if (!res.perf_hint.empty()) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-008",
                       op_name + ": ChooseL0Tile fallback. " + res.perf_hint, assign->span_);
  }

  MatmulTiling t;
  t.assign = assign;
  t.kind = kind;
  t.lhs = lhs;
  t.rhs = rhs;
  t.bias = bias_var;
  t.bias_load = bias_load;
  t.bias_load_def = bias_load_def;
  // A Vec-resident left operand is staged into Mat before the K-loop (see
  // BuildMove(..., Mat)); Mat-resident left operands extract directly.  The right
  // operand is always Mat (checked above), so it never needs staging.
  t.stage_lhs_to_mat = lhs_tile->GetMemorySpace() == MemorySpace::Vec;
  t.acc_init = acc_var;  // null for fresh matmul/bias, set for tile.matmul_acc
  // The caller's split-K predicate, composed with the generated ``ko == 0``.
  // Captured as-is rather than cloned: the emitted ForStmt replaces the
  // original AssignStmt in the same SeqStmts and uses the predicate exactly
  // once, so its definition still dominates the new use.
  t.init_cond = has_init_cond ? call->args_[3] : nullptr;
  t.M = M;
  t.N = N;
  t.K = K;
  t.m = res.m;
  t.n = res.n;
  t.k = res.k;
  t.grid_m = cfg.M;
  t.grid_n = cfg.N;
  t.output_valid_m = output_valid_shape[0];
  t.output_valid_n = output_valid_shape[1];
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

}  // namespace auto_tile_matmul_l0
}  // namespace ir
}  // namespace pypto
