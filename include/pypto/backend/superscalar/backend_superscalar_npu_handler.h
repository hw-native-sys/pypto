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

#ifndef PYPTO_BACKEND_SUPERSCALAR_BACKEND_SUPERSCALAR_NPU_HANDLER_H_
#define PYPTO_BACKEND_SUPERSCALAR_BACKEND_SUPERSCALAR_NPU_HANDLER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

/**
 * @brief BackendHandler implementation for SuperscalarNPU.
 *
 * SuperscalarNPU has a DDR + TREG memory model and no cube/vector cores, so the
 * Ascend-specific cross-core / mixed-kernel / L0-tiling behaviours do not apply.
 * Code generation is not implemented yet; the codegen-facing accessors return
 * neutral placeholders and the cross-core hook is unreachable on this backend.
 */
class SuperscalarNPUHandler : public BackendHandler {
 public:
  static const SuperscalarNPUHandler& Instance();

  // Codegen hooks — codegen is not implemented for SuperscalarNPU yet; these
  // return neutral placeholders so the interface is satisfied.
  [[nodiscard]] std::string GetPtoTargetArch() const override { return "superscalar"; }
  [[nodiscard]] std::string GetLaunchSpecCoreCountMethod() const override { return "set_core_num"; }
  [[nodiscard]] std::string GetDefaultSimPlatform() const override { return "superscalarsim"; }
  [[nodiscard]] std::vector<std::string> GetExtraPtoasFlags() const override { return {}; }

  // Pass behavioural hooks — none of the Ascend cross-core / mixed-kernel
  // workarounds apply to SuperscalarNPU.
  [[nodiscard]] bool RequiresGMPipeBuffer() const override { return false; }
  [[nodiscard]] bool RequiresSplitLoadTpopWorkaround() const override { return false; }
  [[nodiscard]] bool RequiresVtoCFractalAdapt() const override { return false; }
  [[nodiscard]] bool RequiresRuntimeSubblockBridge() const override { return false; }
  [[nodiscard]] bool RequiresNoSplitDualAivDispatch() const override { return false; }

  // No cross-core transfers exist on this backend.
  [[nodiscard]] ir::TileView BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                        const ir::TileView& original_view) const override;

  // Performance-hint thresholds — TREG is on-chip, so GM-oriented thresholds are
  // not meaningful; return 0 to disable the GM innermost-dim perf hints.
  [[nodiscard]] uint32_t GetGmAccessGranularityBytes() const override { return 0; }
  [[nodiscard]] uint32_t GetL2CacheLineBytes() const override { return 0; }
  [[nodiscard]] uint32_t GetRecommendedInnermostDimBytes() const override { return 0; }

  // L0-tiling parameters — SuperscalarNPU has no cube L0 buffers.
  [[nodiscard]] uint32_t GetL0aCapacityBytes() const override { return 0; }
  [[nodiscard]] uint32_t GetL0bCapacityBytes() const override { return 0; }
  [[nodiscard]] uint32_t GetL0cCapacityBytes() const override { return 0; }

  // On-chip storage is the TREG register file.
  [[nodiscard]] ir::MemorySpace GetDefaultOnChipMemorySpace() const override { return ir::MemorySpace::TREG; }

 private:
  SuperscalarNPUHandler() = default;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_SUPERSCALAR_BACKEND_SUPERSCALAR_NPU_HANDLER_H_
