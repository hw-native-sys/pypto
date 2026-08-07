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

#ifndef SRC_IR_TRANSFORMS_AUTO_TILE_VECTOR_REPORT_H_
#define SRC_IR_TRANSFORMS_AUTO_TILE_VECTOR_REPORT_H_

#include <optional>
#include <string>

#include "src/ir/transforms/auto_tile/vector_plan.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {

/**
 * Persist a deterministic JSON descriptor and pseudocode explanation for one
 * selected vector AutoTile schedule when the active PassContext carries a
 * ReportInstrument.
 *
 * @return The pseudocode report path, or std::nullopt for a bare pass run.
 */
[[nodiscard]] std::optional<std::string> WriteVectorScheduleReport(const VectorGraph& graph,
                                                                   const VectorSchedulePlan& plan);

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_AUTO_TILE_VECTOR_REPORT_H_
