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

  // a5 roofline cost-model constants (EXPLICIT, so each is an a5-sim-calibratable slot).
  // Cube is a5-sim-CALIBRATED (mad_head=25, mad_fp32_passes=8, bf16 cpr=1); bw + drain are
  // still a2a3-inherited (a5-sim sweep pending -- see below). Calibration recipe (full
  // spec: a5_cost_model_device_task.md): fit each constant from an a5-sim forced-tile
  // sweep using WORK-CYCLE extraction by instruction source (excluding WAIT_FLAG/BAR
  // stalls) on the cube / MTE1 (LOAD_2Dv2) / FIXP lanes. That method reproduces the shipped
  // a2a3 *device* model within ~15%, so fit RAW a5-sim work-cycle numbers -- NO transfer
  // correction (the earlier BW/~1.54 idea was for raw pipe sums, superseded). No a5 device
  // exists, so a5-sim IS the ground truth; validate picks with an a5-sim OLD-vs-NEW wall
  // check. The form (per-M-row max(floor,throughput) + oddPart) is arch-general; only these
  // magnitudes vary (plus the already-measured fp32 cube pass count).
  [[nodiscard]] L0CostModel GetL0CostModel() const override {
    L0CostModel m;
    // BW + drain: still a2a3-inherited. TODO(a5): a5-sim couldn't fit these yet (a5-sim
    // ~8x slower + LOAD source is LOAD_2Dv2/MTE1 not LOAD_L1_TO_DST); left at a2a3 = the
    // documented inheritance (no worse than status quo). Preliminary a5 signal: drain
    // misalignment looks MILDER on a5 (n=128->144 +1.20x vs a2a3 +3.96x) -> drain_penalty
    // likely < 2.6. Refit via analytic bytes (see a5_cost_model_device_task.md).
    m.bw_l0a = 129.7;              // TODO(a5): refit (a2a3: 129.7)
    m.bw_l0b = 85.4;               // TODO(a5): refit (a2a3: 85.4)
    m.bw_drain = 118.0;            // TODO(a5): refit (a2a3: 118.0)
    m.drain_fixed_cycles = 164.0;  // TODO(a5): refit (a2a3: 164.0)
    m.drain_row_cycles = 4.45;     // TODO(a5): refit (a2a3: 4.45)
    m.drain_penalty_cycles = 2.6;  // TODO(a5): refit -- likely lower on a5 (a2a3: 2.6)
    m.drain_c0_bytes = 32;         // ISA NZ-fractal C0; a5-invariant
    m.mad_k_fractal_bytes = 32;    // ISA cube K-fractal; a5-invariant
    // Cube: MEASURED on a5-sim (the primary calibration, high confidence).
    m.mad_head_cycles = 25;  // a5-sim: intercept of fp32 AND bf16 k-sweeps (a2a3: 21)
    m.mad_fp32_passes = 8;   // a5-sim CONFIRMED: full fp32 MMAD ~4x a2a3/fractal
                             // (a2a3=2). mmad = 25 + 512*k_fractal at m=128 (4121) and
                             // m=256 (8217). bf16 stays 1 pass (bf16 mmad=281=25+256,
                             // unaffected by the 4x) -- so the 8x is fp32-only, as intended.
    return m;
  }

 private:
  Ascend950Handler() = default;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_950_BACKEND_950_HANDLER_H_
