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

#ifndef PYPTO_CODEGEN_PTO_TILE_OUTPUT_BINDINGS_H_
#define PYPTO_CODEGEN_PTO_TILE_OUTPUT_BINDINGS_H_

#include <map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto::backend {
class Backend;
}

namespace pypto::codegen {

struct TileOutputBinding {
  ir::VarPtr var;
  ir::TileTypePtr type;  ///< Includes the MemRef assigned by storage planning.
};

// One entry per native multi-output Tile assignment. A Call expression may be
// shared by assignments; their SSA result variables identify distinct writes.
using TileOutputBindings = std::map<const ir::Var*, std::vector<TileOutputBinding>>;

TileOutputBindings PrepareTileOutputBindings(const ir::StmtPtr& body, const backend::Backend& backend);

}  // namespace pypto::codegen

#endif  // PYPTO_CODEGEN_PTO_TILE_OUTPUT_BINDINGS_H_
