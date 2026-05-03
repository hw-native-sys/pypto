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
#include <vector>

#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

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
   * @brief Whether this backend needs the LegalizePtoBufferReuse split-load /
   *        tpop hazard workaround for AIV functions.
   */
  [[nodiscard]] virtual bool RequiresSplitLoadTpopWorkaround() const = 0;

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
   * @brief Cube fractal alignment in *elements* for L0 tile dimensions.
   *
   * Distinct from memory access alignment (which is a byte-level concept on
   * the SoC `Mem` record). This is the m/n/k alignment imposed by the cube
   * hardware's fractal tile shape — typically 16 across Ascend AI Core
   * generations.
   */
  [[nodiscard]] virtual int GetL0FractalAlignment() const { return 16; }

  /**
   * @brief Minimum legal value for L0 tile dimensions m, n, k.
   *
   * The cube unit cannot operate below this dimension; ChooseL0Tile rejects
   * candidates smaller than this value (and emits a perf-hint when the
   * outer matmul shape is itself smaller than this threshold).
   */
  [[nodiscard]] virtual int GetMinL0TileDim() const { return 16; }
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_BACKEND_HANDLER_H_
