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

#ifndef PYPTO_CODEGEN_PTO_TPOP_CHAIN_REORDER_H_
#define PYPTO_CODEGEN_PTO_TPOP_CHAIN_REORDER_H_

#include <map>
#include <string>
#include <vector>

#include "pypto/ir/stmt.h"

namespace pypto {

namespace ir {
class Var;
}

namespace codegen {

struct TpopResultInfo {
  int split = 0;
  std::string op_name;
};

/// Reorder top-level statements so each tpop chain follows pop-use-free order.
/// Hardware requires: tpop(tile) -> use(tile) -> tfree(tile) before the next tpop.
/// Groups tpop assignment, its direct users, and its tfree into sequential chains.
/// Recurses into nested control flow (for/if/while) bodies.
std::vector<ir::StmtPtr> ReorderTpopChains(const std::vector<ir::StmtPtr>& stmts,
                                           const std::map<const ir::Var*, TpopResultInfo>& tpop_result_vars);

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_PTO_TPOP_CHAIN_REORDER_H_
