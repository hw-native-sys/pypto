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

  // TODO(a5-calibration): a5 roofline cost-model constants. These are EXPLICIT (rather
  // than the inherited BackendHandler default) so each can be refit from an a5-sim
  // sweep, but every value below currently EQUALS the a2a3 default, so this override is
  // a behavioural NO-OP until the numbers are replaced -- a5 tile picks are unchanged
  // by this stub. Calibration recipe (mirrors the a2a3 work; see the a5 calibration
  // harness / device task): fit each constant from an a5-sim forced-tile sweep with
  // per-pipe isolation (cube / MTE1 / MTE2 / FIXP lanes), then transfer the a2a3
  // op-sim->device correction (BW /~1.54, mad_head as-is, FIXPIPE magnitude
  // device-anchored) since raw op-sim over-states port BW ~1.5x and FIXPIPE ~4x. The
  // form (per-M-row max(floor,throughput) + oddPart misalignment) is arch-general and
  // does NOT need refitting -- only these magnitudes. Ship as sim-provisional (a5
  // *device* validation pending) and audit the resulting pick changes vs the a2a3
  // baseline before trusting it broadly.
  [[nodiscard]] L0CostModel GetL0CostModel() const override {
    L0CostModel m;
    m.bw_l0a = 129.7;              // TODO(a5): refit from a5-sim MTE1 lane (a2a3: dev 129.7)
    m.bw_l0b = 85.4;               // TODO(a5): refit from a5-sim MTE2 lane (a2a3: dev 85.4)
    m.bw_drain = 118.0;            // TODO(a5): refit from a5-sim FIXP lane (a2a3: dev 118.0)
    m.drain_fixed_cycles = 164.0;  // TODO(a5): refit (a2a3: 164.0)
    m.drain_row_cycles = 4.45;     // TODO(a5): refit a5 per-row floor (a2a3: 4.45)
    m.drain_penalty_cycles = 2.6;  // TODO(a5): refit a5 misalignment penalty (a2a3: 2.6)
    m.drain_c0_bytes = 32;         // ISA NZ-fractal C0; a5-invariant (keep 32)
    m.mad_head_cycles = 21;        // TODO(a5): confirm on a5 cube (a2a3 transfers ~2%; 21)
    m.mad_k_fractal_bytes = 32;    // ISA cube K-fractal; a5-invariant (keep 32)
    return m;
  }

 private:
  Ascend950Handler() = default;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_950_BACKEND_950_HANDLER_H_
