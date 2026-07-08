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

#ifndef PYPTO_BACKEND_950_BACKEND_950_HANDLER_H_
#define PYPTO_BACKEND_950_BACKEND_950_HANDLER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

/**
 * @brief BackendHandler implementation for Ascend950 (a5).
 *
 * Hardware cross-core pipe carries fractal-layout data directly, so the AIV
 * producer must materialise an adapter `tile.move` before tpush, but no GM
 * slot buffer is needed and the split-load tpop hazard does not apply.
 * Ptoas needs an extra `--pto-arch a5` selector.
 */
class Ascend950Handler : public BackendHandler {
 public:
  static const Ascend950Handler& Instance();

  [[nodiscard]] std::string GetPtoTargetArch() const override { return "a5"; }
  [[nodiscard]] std::string GetLaunchSpecCoreCountMethod() const override { return "set_core_num"; }
  [[nodiscard]] std::string GetDefaultSimPlatform() const override { return "a5sim"; }
  [[nodiscard]] std::vector<std::string> GetExtraPtoasFlags() const override { return {"--pto-arch", "a5"}; }

  [[nodiscard]] bool RequiresGMPipeBuffer() const override { return false; }
  [[nodiscard]] bool RequiresSplitLoadTpopWorkaround() const override { return false; }
  [[nodiscard]] bool RequiresVtoCFractalAdapt() const override { return true; }
  [[nodiscard]] bool RequiresRuntimeSubblockBridge() const override { return false; }
  [[nodiscard]] bool RequiresNoSplitDualAivDispatch() const override { return false; }
  // A5 acc->mat tinsert accepts dst=f32, so the Mat scratch may stay f32.
  [[nodiscard]] bool RequiresLowPrecisionMatScratch() const override { return false; }

  // A5 store pipe does NOT support bf16 atomic-add (pto-isa SetAtomicAdd<T>
  // rejects bfloat16_t on the a5 path); require an fp32 accumulator + cast.
  [[nodiscard]] bool SupportsBf16AtomicAdd() const override { return false; }

  [[nodiscard]] ir::TileView BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                        const ir::TileView& original_view) const override;

  [[nodiscard]] uint32_t GetGmAccessGranularityBytes() const override { return 128; }
  [[nodiscard]] uint32_t GetL2CacheLineBytes() const override { return 512; }
  [[nodiscard]] uint32_t GetRecommendedInnermostDimBytes() const override { return 128; }

  // L0 capacity (matches Create950SoC AIC core memory layout; grounded in the pto-isa
  // hardware reference include/pto/common/buffer_limits.hpp under PTO_NPU_ARCH_A5:
  // L0A/L0B 64 KB, L0C 256 KB (2x a2a3), Mat/CB 512 KB). These are CORRECT for a5 and
  // are what already make a5 tile differently from a2a3 (bigger accumulator -> bigger
  // tiles / fewer M/N splits). Only the ROOFLINE CONSTANTS below are still a2a3-derived.
  [[nodiscard]] uint32_t GetL0aCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0bCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0cCapacityBytes() const override { return 256ULL * 1024; }
  [[nodiscard]] uint64_t GetMatCapacityBytes() const override { return 512ULL * 1024; }

  // a5 roofline cost-model constants -- FULLY a5-sim-CALIBRATED (all 7; raw work-cycle fit;
  // no a5 device, so a5-sim is the ground truth). Recipe (a5_cost_model_device_task.md): fit
  // each from an a5-sim forced-tile sweep via WORK-CYCLE extraction by instruction source
  // (excluding WAIT_FLAG/BAR stalls) on the cube / MTE1 (LOAD_2Dv2) / FIXP lanes -- that
  // method reproduces the shipped a2a3 *device* model within ~15% on the load/cube lanes
  // (so fit RAW, no transfer correction), though NOT on the drain lane (see bw_drain caveat
  // below). The form (per-M-row max(floor,throughput) + oddPart) is arch-general; only the
  // magnitudes + the fp32 cube pass count vary. Single-tile roofline ranking validated on
  // a5-sim (Spearman ~0.9); multi-tile wall validation is blocked by broken a5-sim span
  // tooling (a follow-up).
  [[nodiscard]] L0CostModel GetL0CostModel() const override {
    L0CostModel m;
    // FULL a5-sim calibration (raw work-cycle fit; no a5 device, so a5-sim is ground truth).
    // Load bandwidths (analytic bytes / LOAD_2Dv2 work-cycles): a5 L0 ports are faster than
    // a2a3 (bw_l0a 129.7->206, bw_l0b 85.4->224 -- note a5 L0B is not the slower port).
    m.bw_l0a = 206.3;  // a5-sim MTE1 (LOAD_2Dv2), analytic-bytes fit (a2a3: 129.7)
    m.bw_l0b = 223.8;  // a5-sim MTE1 (a2a3: 85.4)
    // Drain (FIX_L0C_TO_DST work-cycles): fits the a5-sim FIXP sweep to <1%. a5 drain is
    // ~4x costlier than a2a3 (118) -> the FIXP lane dominates a5 GEMM (Exp-C trace: FIXP
    // 44-52% of the wall, cube util below pto-isa's >50% target -- drain-bound, not an emit
    // defect).
    // ⚠ bw_drain=30 carries a known ~4x SIM-vs-DEVICE gap (a2a3-sim drain is likewise ~4x
    // a2a3-device's 118), so it OVER-predicts absolute a5 walls ~34%. But this does NOT
    // mis-pick: the a5-sim VALIDATION (a5_validate_picks.py + reconstructed multi-tile walls)
    // showed the chooser's tile is the sim's #2-#3, within ~1-2% of the measured-lane best on
    // all tested shapes -- the over-prediction is uniform enough to preserve the ranking.
    // Making dbC=2 the a5 DEFAULT does NOT fix it and was REFUTED (Exp B): dbC=2's full-K
    // constraint forces tiny fp32 tiles -> more drains -> a ~40% LOSS for fp32 (bf16 only
    // ~1.25x). The real absolute-accuracy lever is a5-DEVICE drain data or tail-accurate
    // drain-hidden pricing, not a schedule flip or a bw_drain hand-edit.
    m.bw_drain = 30.0;              // a5-sim FIXP throughput (a2a3: 118; see warning above)
    m.drain_fixed_cycles = 343.4;   // a5-sim per-drain fixed (a2a3: 164)
    m.drain_row_cycles = 4.59;      // a5-sim narrow-N floor (a2a3: 4.45; ~arch-invariant)
    m.drain_penalty_cycles = 0.26;  // a5-sim misalignment (a2a3: 2.6) -- a5 is much MILDER
    m.drain_c0_bytes = 32;          // ISA NZ-fractal C0; a5-invariant
    m.mad_k_fractal_bytes = 32;     // ISA cube K-fractal; a5-invariant
    // Cube: a5-sim MEASURED (the primary, highest-confidence calibration).
    m.mad_head_cycles = 25;  // a5-sim: intercept of fp32 AND bf16 k-sweeps (a2a3: 21)
    m.mad_fp32_passes = 8;   // a5-sim CONFIRMED: full fp32 MMAD ~4x a2a3/fractal (a2a3=2;
                             // mmad = 25 + 8*fractals, R^2=1). bf16 stays 1 pass -- the 8x
                             // is fp32-only, as intended.
    return m;
  }

 private:
  Ascend950Handler() = default;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_950_BACKEND_950_HANDLER_H_
