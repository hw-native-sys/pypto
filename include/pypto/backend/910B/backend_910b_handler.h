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

#ifndef PYPTO_BACKEND_910B_BACKEND_910B_HANDLER_H_
#define PYPTO_BACKEND_910B_BACKEND_910B_HANDLER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

/**
 * @brief BackendHandler implementation for Ascend910B (a2a3).
 *
 * Cross-core data is routed via GM, so the AIV side does not need a fractal
 * adapter; the GM-backed slot buffer must be injected by ExpandMixedKernel;
 * the AIV split-load tpop hazard workaround applies on this backend only.
 */
class Ascend910BHandler : public BackendHandler {
 public:
  static const Ascend910BHandler& Instance();

  [[nodiscard]] std::string GetPtoTargetArch() const override { return "a2a3"; }
  [[nodiscard]] std::string GetLaunchSpecCoreCountMethod() const override { return "set_block_num"; }
  [[nodiscard]] std::string GetDefaultSimPlatform() const override { return "a2a3sim"; }
  [[nodiscard]] std::vector<std::string> GetExtraPtoasFlags() const override { return {"--pto-arch", "a3"}; }

  [[nodiscard]] bool RequiresGMPipeBuffer() const override { return true; }
  [[nodiscard]] bool RequiresSplitLoadTpopWorkaround() const override { return true; }
  [[nodiscard]] bool RequiresVtoCFractalAdapt() const override { return false; }
  [[nodiscard]] bool RequiresRuntimeSubblockBridge() const override { return true; }
  [[nodiscard]] bool RequiresNoSplitDualAivDispatch() const override { return true; }

  [[nodiscard]] ir::TileView BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                        const ir::TileView& original_view) const override;

  [[nodiscard]] uint32_t GetGmAccessGranularityBytes() const override { return 512; }
  [[nodiscard]] uint32_t GetL2CacheLineBytes() const override { return 512; }
  [[nodiscard]] uint32_t GetRecommendedInnermostDimBytes() const override { return 512; }

  // L0 capacity (matches Create910BSoC AIC core memory layout).
  [[nodiscard]] uint32_t GetL0aCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0bCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0cCapacityBytes() const override { return 128ULL * 1024; }

 private:
  Ascend910BHandler() = default;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_910B_BACKEND_910B_HANDLER_H_
