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

#include "pypto/backend/backend_910b.h"

#include <memory>

namespace pypto {
namespace backend {

Backend910B::Backend910B() : Backend() {
  Core aic_core(ir::CoreType::CUBE, {
                                        Mem(ir::MemorySpace::L1, 512ULL * 1024, 128),  // 512KB L1
                                        Mem(ir::MemorySpace::L0A, 64ULL * 1024, 64),   // 64KB L0A
                                        Mem(ir::MemorySpace::L0B, 64ULL * 1024, 64),   // 64KB L0B
                                        Mem(ir::MemorySpace::L0C, 128ULL * 1024, 128)  // 128KB L0C
                                    });

  Core aiv_core(ir::CoreType::VECTOR, {
                                          Mem(ir::MemorySpace::UB, 192ULL * 1024, 128),  // 192KB UB
                                      });

  Cluster aic_cluster(aic_core, 1);  // 1 core per cluster
  Cluster aiv_cluster(aiv_core, 1);  // 1 core per cluster

  Die die({{aic_cluster, 24}, {aiv_cluster, 48}});  // 24 AIC cores and 48 AIV cores per die
  soc_ = std::make_shared<SoC>(die, 1);             // 1 die

  mem_graph_[ir::MemorySpace::DDR] = {ir::MemorySpace::UB, ir::MemorySpace::L1};
  mem_graph_[ir::MemorySpace::UB] = {ir::MemorySpace::DDR};
  mem_graph_[ir::MemorySpace::L1] = {ir::MemorySpace::L0A, ir::MemorySpace::L0B};
  mem_graph_[ir::MemorySpace::L0C] = {ir::MemorySpace::L1, ir::MemorySpace::DDR};
}

}  // namespace backend
}  // namespace pypto
