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

#ifndef PYPTO_BACKEND_COMMON_BACKEND_HANDLER_H_
#define PYPTO_BACKEND_COMMON_BACKEND_HANDLER_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

/**
 * @brief The (src, dst) dtype pairs an architecture converts in one `pto.tcvt`.
 *
 * Transcribed from the ISA Supported Conversions table in the pto-isa tcvt
 * documentation. Order is irrelevant; duplicates and self-edges are ignored by
 * consumers.
 */
struct TcvtAdjacency {
  std::vector<std::pair<DataType, DataType>> edges;
};

/**
 * @brief Closed-form GEMM cost-model parameters consumed by ChooseL0Tile.
 *
 * Bandwidths are in BYTES PER CORE CYCLE, so the chooser can weight L1->L0
 * traffic and the L0C drain directly in cycles. Defaults are Ascend a2a3 (910B),
 * on-device sweep-calibrated: L1->L0A ~130, L1->L0B ~85 B/cyc (~1.52:1). The
 * FIXPIPE L0C drain is PER-M-ROW, per output tile (scaled by the tile count
 * `ceil(M/m)*ceil(N/n)`):
 *   per_tile = drain_fixed_cycles                                              // fixed issue overhead
 *            + m * ( max(drain_row_cycles, bytes_c*n/bw_drain)                 // per-row: floor OR
 * throughput
 *                    + drain_penalty_cycles*(odd(N1)-1) )                      // misalignment serialization
 * with `N1 = ceil(n / N0)`, `N0 = drain_c0_bytes / bytes_c` (= 8 for the fp32
 * L0C accumulator, output-dtype independent). Direct fit of an on-device FIXPIPE
 * sweep. Per the writeback model (pto-isa fixpipe-model.md + nz-fractal-layout.md),
 * FIXPIPE addresses one M-row of the `N1 M1 M0 N0` FRACTAL_NZ accumulator at a time
 * (=> cost ∝ m), each row a grouped nburst/loop over the `N1 = ceil(n/N0)`
 * N-fractals. The per-M-row cost is `max(floor, throughput)`: a fixed burst-issue
 * FLOOR `drain_row_cycles` (row addressing/setup, N-independent) that dominates
 * narrow N, OR the fractal THROUGHPUT `bytes_c*n/bw_drain` that dominates wide N
 * (crossover ~n=131 fp32) -- the sweep confirmed both regimes (a flat base below
 * the crossover, byte-throughput above). A non-pow2 fractal count adds the odd
 * residual `odd(N1)-1` serial burst passes at `drain_penalty_cycles` per M-row
 * (the non-pow2 N-fractal-count cliff -- n=80 -> odd(10)=5, and n=96 -> odd(12)=3 despite
 * 96%32==0; it is NOT literally N%32). Splitting the OUTPUT (M/N) adds drains;
 * splitting K does not (accumulate in one L0C). Device-validated (drain 0.93-1.09x,
 * loads R^2 0.993).
 *
 * CALIBRATION SURFACE (a2a3, so the next backend / lowering path can tell when it
 * has left the fitted envelope): forced-tile FIXPIPE + MTE1 sweep, direct-store
 * (Acc->GM) and Mat-scratch (Acc->Mat) drains, bf16 and fp32; m in {16..256},
 * n aligned {16..512}; N-fractal N1 sampled at power-of-2 and small non-pow2 counts
 * (n up to 128 densely, plus n=320); FIXPIPE isolated via the dbc0-dbc1 difference.
 * The odd(N1) FORM is op-sim-validated across odd-parts {1,3,5,7,9,11,13,15}
 * (16-aligned n in {128..240}, camodel FIXP lane): R^2 0.968, and two engineered
 * contradictions (n=192 oddPart 3 vs a bounded-remainder's 8; n=144 oddPart 9 vs 2)
 * both track oddPart, decisively refuting a bounded 1-2-pass remainder form. So the
 * penalty scales with the ODD PART of the fractal count, not a bounded remainder. The
 * MAGNITUDE (drain_penalty=2.6) stays device-anchored (n=80 sweep) -- op-sim over-
 * states absolute FIXP ~4x, so it is used for form discrimination only, not refit.
 * Non-16-aligned n (e.g. 136/264, odd 17/33) can't be measured (ptoas rejects), so
 * those exact points remain model-only, but the form holds out to oddPart 15.
 *
 * The MAD term mirrors the cube's per-TMATMUL cost
 * `mad_head_cycles + cpr * ceil(m/16) * ceil(k/kt) * ceil(n/16)`, where
 * `kt = mad_k_fractal_bytes / bytes_a` and `cpr` (1 for 2-byte inputs, 2 for
 * 4-byte) is derived from the operand byte width in the chooser.
 */
struct L0CostModel {
  double bw_l0a = 129.7;  ///< L1->L0A bytes/cycle (a2a3 on-device MTE1 sweep, R^2 0.993; was op-sim 200).
  double bw_l0b =
      85.4;  ///< L1->L0B bytes/cycle (a2a3 on-device MTE1 sweep; ~1.52:1 vs L0A, same ratio as before).
  double bw_drain =
      118.0;  ///< FIXPIPE per-row byte throughput, L0C bytes/cycle (a2a3 sweep; dominates wide N).
  double drain_fixed_cycles = 164.0;  ///< Per-FIXPIPE-drain fixed issue overhead, m-independent (a2a3 sweep).
  double drain_row_cycles = 4.45;  ///< FIXPIPE per-M-row FLOOR (burst-issue setup, N-independent); dominates
                                   ///< narrow N below the bytes/bw crossover (~n=131 fp32). Per-row cost =
                                   ///< max(floor, bytes_c*n/bw_drain) (a2a3 sweep).
  double drain_penalty_cycles =
      2.6;                  ///< Misalignment cost: FIXPIPE cycles per L0C M-row per extra
                            ///< serial burst pass, charged (odd(N1)-1) times (a2a3 on-device sweep).
  int drain_c0_bytes = 32;  ///< NZ fractal C0 constant in bytes (N0 = C0 / bytes_c; 8 for fp32, 16 for bf16).
  int mad_head_cycles = 21;      ///< Fixed per-TMATMUL issue overhead, per K-block (a2a3 op-sim fit,
                                 ///< device-confirmed within ~2%; was 6, +15/tile clean on device).
                                 ///< Enters C_mad as `k_blocks*mad_head + cpr*ceil(m/16)*k_fractals*
                                 ///< ceil(n/16)` (see MadCycles) -- NOT `mad_head + 16*K`.
                                 ///< Backend-specific: a new backend should shallow-K device-spot-check
                                 ///< it (it competes with the drain floor on shallow-K/small tiles).
  int mad_k_fractal_bytes = 32;  ///< Cube K-fractal width in bytes (kt = this / bytes_a).
  int mad_fp32_passes = 2;       ///< Cube passes per K-fractal for FULL fp32 MMAD (the fp32
                                 ///< mantissa is decomposed into partial products). ARCH-SPECIFIC:
                                 ///< a2a3=2; a5=8 (a5-sim: fp32 MMAD is ~4x a2a3 per fractal -- a
                                 ///< fuller decomposition; the fractal WIDTH kt is C0-derived and
                                 ///< arch-invariant, so this pass count is the only fp32 cube knob).
                                 ///< bf16 and other <=2-byte inputs are 1 pass (unaffected).
};

/**
 * @brief Backend-specific behavior dispatch interface
 *
 * BackendHandler centralises every behavioural difference between backends
 * (e.g. Ascend910B vs Ascend950). Passes and codegen never branch on
 * BackendType directly; instead they invoke virtual methods on a handler
 * obtained from PassContext or from a Backend instance.
 *
 * Adding a new backend requires only:
 *   1. Implement a Backend subclass (see Backend910B / Backend950).
 *   2. Implement a BackendHandler subclass.
 *   3. Override Backend::GetHandler() to return the new handler singleton.
 *
 * No existing pass / codegen needs to change.
 */
class BackendHandler {
 public:
  virtual ~BackendHandler() = default;

  // ---------------------------------------------------------------------------
  // Codegen hooks
  // ---------------------------------------------------------------------------

  /**
   * @brief PTO MLIR target arch attribute string (e.g. "a2a3", "a5").
   *
   * Used by PTOCodegen when emitting `module attributes {pto.target_arch = ...}`.
   */
  [[nodiscard]] virtual std::string GetPtoTargetArch() const = 0;

  /**
   * @brief Method name used on `launch_spec` to set the per-task core count.
   *
   * Different runtimes expose different APIs for the same concept
   * (Ascend910B: "set_block_num"; Ascend950: "set_core_num").
   */
  [[nodiscard]] virtual std::string GetLaunchSpecCoreCountMethod() const = 0;

  /**
   * @brief Default simulator platform name (e.g. "a2a3sim", "a5sim").
   *
   * Used by Python-side runner / compiled program defaults.
   */
  [[nodiscard]] virtual std::string GetDefaultSimPlatform() const = 0;

  /**
   * @brief Extra flags appended to the ptoas compiler invocation.
   *
   * Some PTOAS releases require an explicit ISA selector even when the MLIR
   * module already carries a backend-specific target_arch attribute (e.g.
   * Ascend910B needs ["--pto-arch", "a3"], Ascend950 needs
   * ["--pto-arch", "a5"]).
   */
  [[nodiscard]] virtual std::vector<std::string> GetExtraPtoasFlags() const = 0;

  /**
   * @brief Whether a dtype has an end-to-end in-core storage and PTO ABI on
   *        this backend.
   *
   * Semantic 4-bit dtypes are packed in PyPTO's memory accounting, but that
   * alone does not make them executable. Backends opt in only after their
   * load/store and instruction ABI is available. Byte-addressable dtypes keep
   * their existing support path.
   */
  [[nodiscard]] virtual bool SupportsIncoreDataType(const DataType& dtype) const {
    return dtype.GetBit() != 4;
  }

  // ---------------------------------------------------------------------------
  // Pass behavioural hooks
  // ---------------------------------------------------------------------------

  /**
   * @brief Whether this backend needs the `__gm_pipe_buffer` injection in
   *        ExpandMixedKernelPass.
   *
   * Ascend910B routes cross-core pipe data through a GM-backed slot buffer;
   * Ascend950 uses on-chip cross-core hardware and does not need it.
   */
  [[nodiscard]] virtual bool RequiresGMPipeBuffer() const = 0;

  /**
   * @brief Whether this backend needs the MemoryReuse load + tpop_from_aic
   *        in-place hazard guard for split-AIV functions.
   */
  [[nodiscard]] virtual bool RequiresSplitLoadTpopWorkaround() const = 0;

  /**
   * @brief Whether PyPTO/DSA-RP must materialize PTOAS level-3 explicit tmp
   *        scratch and emit static-valid codegen views.
   *
   * True on Ascend910B (a2a3): InitMemRef appends compiler-owned scratch for
   * ops such as tile.ci / narrowing tile.cast / tile.sort32, and codegen
   * bridges explicit-tmp forms with static valid_shape. False on Ascend950:
   * PTOAS level-2 PlanMemory owns implicit tmp instead.
   */
  [[nodiscard]] virtual bool RequiresLevel3TmpScratch() const = 0;

  /**
   * @brief Whether AIV-side V-to-C tpush must materialise a fractal-layout
   *        adapter `tile.move` before the actual tpush.
   *
   * Ascend950 hardware cross-core pipe expects fractal layout at the boundary
   * (Left / Right / Mat -> NZ), so the AIV producer must convert.
   * Ascend910B routes via UB -> GM -> Mat which accepts ND directly, so no
   * adapter is needed.
   */
  [[nodiscard]] virtual bool RequiresVtoCFractalAdapt() const = 0;

  /**
   * @brief Whether A2A3 split AIV wrappers must source the subblock id from
   *        the runtime context.
   *
   * Only relevant on Ascend910B. Other backends always return false.
   */
  [[nodiscard]] virtual bool RequiresRuntimeSubblockBridge() const = 0;

  /**
   * @brief Whether mixed kernels with no split mode (or `SplitMode::None`)
   *        must still be dispatched on both AIV lanes for cross-core sync.
   *
   * On Ascend910B the AIC side performs cross-core pipe handshakes against
   * both AIVs, so a `no_split` mixed kernel cannot dispatch a single AIV
   * lane without deadlocking. ExpandMixedKernel marks such functions with the
   * `dual_aiv_dispatch` attribute so that downstream passes (notably
   * SplitVectorKernel) and the orchestration codegen know to keep both lanes
   * active and replay sync-only payload on the secondary lane.
   *
   * Ascend950 hardware cross-core pipe does not require this workaround.
   */
  [[nodiscard]] virtual bool RequiresNoSplitDualAivDispatch() const = 0;

  /**
   * @brief Whether a tiled (offset) Acc->Mat FIXPIPE writeback must downcast to
   *        a low-precision (bf16/f16) destination.
   *
   * The only offset Acc->Mat path on A2/A3 is `pto.tinsert`, whose verifier
   * requires `src=f32, dst=f16/bf16` — it cannot keep f32 (PTOAS
   * `TInsertOp::verify`). So AutoTileMatmulL0's oversized chained-matmul result,
   * when M/N-tiled into an L1/Mat scratch, must be bf16/f16 on Ascend910B; the
   * pass folds a `tile.cast(result, bf16)` into the per-sub-tile assemble.
   *
   * Ascend950 (a5) `tinsert` accepts `dst=f32`, so the Mat scratch may stay f32
   * there and this returns false (no cast required, the producer may keep f32).
   */
  [[nodiscard]] virtual bool RequiresLowPrecisionMatScratch() const = 0;

  /**
   * @brief Whether this backend's store pipe honours a bf16 atomic-add into GM.
   *
   * A bf16 atomic-add lowers to `set_atomic_bf16`, honoured on the A2/A3 store
   * path (Ascend910B) and not on the A5 one (Ascend950). Both arches'
   * `SetAtomicAdd<T>` helpers *do* list `bfloat16_t`, so that dispatch
   * static_assert is not the evidence; the pinned ST suite is. a2a3 covers
   * atomic-add into a bf16 destination (`tstore_acc2gm` case 60,
   * `<1, float, bfloat16_t, bfloat16_t, ...>`) while a5's port of that same
   * case runs it non-atomically (`<0, ...>`), and the `set_atomic_bf16`
   * costmodel mock exists only under `costmodel/a2a3/`.
   *
   * The `AtomicAddDtypeValid` property verifier gates every atomic-add site on
   * this, so A5 users get a clean PyPTO error at pipeline input instead of a
   * downstream C++ compile failure. The atomic dispatch keys on the GM
   * *destination* dtype, so this also covers the cube path (fp32 Acc -> bf16 GM
   * via fix-pipe).
   *
   * Ascend910B: true. Ascend950: false.
   */
  [[nodiscard]] virtual bool SupportsBf16AtomicAdd() const = 0;

  /**
   * @brief Whether the cube fix-pipe may store an Acc-resident tile straight
   *        into a GM tensor of @p dtype.
   *
   * The fix-pipe narrows an accumulator on its way to global memory, but only
   * into a fixed destination set. pto-isa encodes this as the non-quant
   * `CheckAcc2gm` whitelist and ptoas as a `pto.tstore` verifier rule:
   *
   *   Ascend910B (a2a3): INT32 / FP32 / FP16 / BF16
   *   Ascend950  (a5)  : INT32 / FP32 / FP16 / BF16
   *
   * The two arches accept the same set today. The hook stays per-backend
   * because the sets are independent facts about each pinned target, not one
   * shared constant -- they have differed before and may again.
   *
   * Anything else -- notably INT8/INT16 -- must reach GM either through a Vec
   * tile (an explicit `pl.cast` narrows in the vector unit, then stores) or via
   * the quantized fix-pipe path, never a plain Acc->GM store. Legality is
   * therefore a property of the *tile's memory space*, not of the user-visible
   * dtypes: the identical DSL program is legal when its matmul result routes
   * through Vec and illegal when it stays in Acc.
   *
   * The two sets above mirror ptoas exactly; keep them in step when the pinned
   * assembler moves (`toolchain/versions.env`, `runtime/pto_isa.pin`). As of
   * ptoas v0.57 / pto-isa 83d01313 both arches are `i32/f32/f16/bf16`
   * (`PTO.cpp` "acc tstore dst element type", a2a3/a5 `CheckStaticAcc`).
   */
  [[nodiscard]] virtual bool SupportsAccToGmDtype(const DataType& dtype) const = 0;

  /**
   * @brief Compute the destination tile view for a cross-core transfer.
   *
   * Encapsulates the per-backend rule for how to lay out the bridge tile
   * crossing the AIC/AIV boundary.
   *
   * Ascend910B (a2a3): cross-core transfer goes through GM. Left/Right/Mat
   *   destinations all use NZ (col_major blayout, row_major slayout) because
   *   GM -> Mat transfer requires fractal layout. Vec destinations preserve
   *   the original view: the GM-backed C2V pop materialises through an ND
   *   GlobalTensor on the consumer side, and PTO-ISA only supports Vec loads
   *   for matching ND/DN/NZ layouts.
   *
   * Ascend950 (a5): hardware cross-core pipe carries data in fractal layout
   *   directly. Left / Right / Mat all use NZ at the transfer boundary
   *   because A5 V2C inserts Vec tiles into the Mat FIFO with
   *   `TINSERT_IMPL<TInsertMode::NZ>`; Vec preserves the caller-requested
   *   final view:
   *   Left -> NZ (col_major blayout, row_major slayout)
   *   Right -> NZ (col_major blayout, row_major slayout)
   *   Mat -> NZ (col_major blayout, row_major slayout)
   *   Vec -> preserve original view
   *
   * @param dest_ms Destination memory space (must be Vec / Mat / Left / Right).
   * @param original_view Caller-supplied view of the source tile.
   * @return TileView to use at the cross-core transfer boundary.
   */
  [[nodiscard]] virtual ir::TileView BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                                const ir::TileView& original_view) const = 0;

  // ---------------------------------------------------------------------------
  // Performance-hint thresholds (issue #1180)
  // ---------------------------------------------------------------------------

  /**
   * @brief GM access granularity in bytes.
   *
   * The hardware fetches at this granularity, so a tile innermost dimension
   * smaller than this value forces the bus to discard part of every fetch.
   * Ascend910B: 512 bytes. Ascend950: 128 bytes.
   */
  [[nodiscard]] virtual uint32_t GetGmAccessGranularityBytes() const = 0;

  /**
   * @brief L2 cache line size in bytes.
   *
   * A tile innermost dimension below this size leaves part of every cache
   * line unused. Both Ascend910B and Ascend950 use 512-byte L2 cache lines.
   */
  [[nodiscard]] virtual uint32_t GetL2CacheLineBytes() const = 0;

  /**
   * @brief Recommended minimum innermost-dim size, in bytes, for tile ops
   *        whose data round-trips through GM (`tile.load` / `tile.store`).
   *
   * Below this threshold the TileInnermostDimGranularity perf-hint check
   * (PH001) emits an advisory diagnostic.
   *
   * On Ascend910B this equals the GM granularity (512 B); on Ascend950 it is
   * the GM granularity (128 B), with 512 B preferable to fully utilise the
   * L2 cache line but 128 B taken as the hard threshold for the hint.
   */
  [[nodiscard]] virtual uint32_t GetRecommendedInnermostDimBytes() const = 0;

  // ---------------------------------------------------------------------------
  // L0-tiling parameters (consumed by AutoTileMatmulL0 / ChooseL0Tile)
  // ---------------------------------------------------------------------------

  /**
   * @brief L0a (Left) on-chip SRAM capacity, in bytes.
   *
   * Used by ChooseL0Tile to bound `m * k * bytes_a` (per buffer when
   * double-buffered). Must match the AIC-core `MemorySpace::Left` size in the
   * SoC config; encoded here so passes do not depend on the SoC walker.
   */
  [[nodiscard]] virtual uint32_t GetL0aCapacityBytes() const = 0;

  /**
   * @brief L0b (Right) on-chip SRAM capacity, in bytes.
   */
  [[nodiscard]] virtual uint32_t GetL0bCapacityBytes() const = 0;

  /**
   * @brief L0c (Acc) on-chip SRAM capacity, in bytes.
   */
  [[nodiscard]] virtual uint32_t GetL0cCapacityBytes() const = 0;

  /**
   * @brief Bias-table on-chip SRAM capacity, in bytes.
   *
   * Used by AutoTileMatmulL0 to cap the N extent of ``tile.matmul_bias``
   * output tiles. Must match the AIC-core ``MemorySpace::Bias`` size in the
   * SoC config and the PTO ISA Mat-to-Bias transfer limit.
   */
  [[nodiscard]] virtual uint32_t GetBiasCapacityBytes() const = 0;

  /**
   * @brief Whether the PTO ISA can transfer this dtype pair from Mat to Bias.
   *
   * AutoTileMatmulL0 uses this before materialising a Mat-resident
   * `tile.matmul_bias` operand in the architectural Bias table. The memory
   * graph only describes reachability; this query captures the narrower raw
   * PTO-ISA dtype contract for that edge. Callers must separately ensure their
   * IR operation can express any requested dtype conversion.
   */
  [[nodiscard]] virtual bool SupportsMatToBiasMove(const DataType& source_dtype,
                                                   const DataType& bias_dtype) const = 0;

  /**
   * @brief Mat (L1) on-chip SRAM capacity, in bytes.
   *
   * Used by passes that need a conservative per-core capacity gate without
   * walking the global Backend SoC object. This keeps pass-level backend
   * decisions on the BackendHandler / PassContext path.
   */
  [[nodiscard]] virtual uint64_t GetMatCapacityBytes() const = 0;

  /**
   * @brief Cube fractal alignment in *elements* for L0 tile dimensions.
   *
   * Distinct from memory access alignment (which is a byte-level concept on
   * the SoC `Mem` record). This is the m/n/k alignment imposed by the cube
   * hardware's fractal tile shape — typically 16 across Ascend AI Core
   * generations.
   */
  [[nodiscard]] virtual int GetL0FractalAlignment() const { return 16; }

  /**
   * @brief Physical M-row alignment of an L0C accumulator allocation.
   *
   * This may be stricter than the legal cube work-shape alignment. The L0
   * chooser, dbC planner, and InitMemRef allocator share it for physical SRAM
   * footprint accounting; it does not reject a smaller logical or valid M
   * extent.
   */
  [[nodiscard]] virtual int GetL0cMAlignment(const DataType& accumulator_dtype) const {
    (void)accumulator_dtype;
    return GetL0FractalAlignment();
  }

  /**
   * @brief Minimum legal value for L0 tile dimensions m, n, k.
   *
   * The cube unit cannot operate below this dimension; ChooseL0Tile rejects
   * candidates smaller than this value (and emits a perf-hint when the
   * outer matmul shape is itself smaller than this threshold).
   */
  [[nodiscard]] virtual int GetMinL0TileDim() const { return 16; }

  /**
   * @brief Closed-form GEMM cost-model parameters (L1<->L0 / drain bandwidths
   * and MAD constants) consumed by ChooseL0Tile.
   *
   * Default is Ascend a2a3 (910B), validated against the pto-isa perf-sim. The
   * a5 (950) numbers are not yet measured; the a2a3 default stands in as a
   * placeholder until characterised — override here once measured.
   */
  [[nodiscard]] virtual L0CostModel GetL0CostModel() const { return L0CostModel{}; }

  /**
   * @brief Native single-instruction `pto.tcvt` conversions for this architecture.
   *
   * Each entry is one (src, dst) dtype pair the ISA can emit as a single
   * `tcvt`; the set differs per architecture (a5 has FP8/HF8/FP4 sources and
   * no INT32 -> FP16 deq, a2a3 is the reverse). LegalizeTileCast treats this as
   * a directed graph and expands any cast whose pair is absent into the
   * shortest chain of native hops, so the table is the single source of truth
   * for what a backend can convert directly.
   *
   * Returned by reference from a handler-owned static — callers may hold the
   * reference for the duration of a pass.
   */
  [[nodiscard]] virtual const TcvtAdjacency& GetTcvtAdjacency() const = 0;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_BACKEND_HANDLER_H_
