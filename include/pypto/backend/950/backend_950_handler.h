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

  // L0 capacity (matches Create950SoC AIC core memory layout).
  [[nodiscard]] uint32_t GetL0aCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0bCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0cCapacityBytes() const override { return 256ULL * 1024; }
  [[nodiscard]] uint64_t GetMatCapacityBytes() const override { return 512ULL * 1024; }

  // TODO(a5-calibration): explicit placeholder. The roofline constants inherited
  // from BackendHandler are a2a3-calibrated until a5 op-sim/device data exists.
  [[nodiscard]] L0CostModel GetL0CostModel() const override { return L0CostModel{}; }

 private:
  Ascend950Handler() = default;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_950_BACKEND_950_HANDLER_H_
