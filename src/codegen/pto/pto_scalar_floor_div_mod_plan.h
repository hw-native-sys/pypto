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

#ifndef SRC_CODEGEN_PTO_PTO_SCALAR_FLOOR_DIV_MOD_PLAN_H_
#define SRC_CODEGEN_PTO_PTO_SCALAR_FLOOR_DIV_MOD_PLAN_H_

#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"

namespace pypto::codegen {

// Signed FloorDiv and FloorMod nodes whose truncating form has the same result
// at every use. The pointers refer to the unchanged IR passed to emission.
std::unordered_set<const ir::Expr*> PrepareScalarFloorDivMod(const ir::StmtPtr& body,
                                                             const std::vector<ir::VarPtr>& params = {});

}  // namespace pypto::codegen

#endif  // SRC_CODEGEN_PTO_PTO_SCALAR_FLOOR_DIV_MOD_PLAN_H_
