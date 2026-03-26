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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_TPOP_CHAIN_NORMALIZER_H_
#define PYPTO_IR_TRANSFORMS_UTILS_TPOP_CHAIN_NORMALIZER_H_

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/core_affinity.h"

namespace pypto {
namespace ir {
namespace tpop_chain {

bool IsTpopAssignStmt(const StmtPtr& stmt, VarPtr* result_var = nullptr);
bool IsExpectedTpopOp(const std::string& op_name, FunctionType func_type);
bool IsExpectedTpopAssignStmt(const StmtPtr& stmt, FunctionType func_type, VarPtr* result_var = nullptr);
bool IsTfreeStmt(const StmtPtr& stmt, VarPtr* tile_var = nullptr, std::string* op_name = nullptr);

std::string GetTfreeOpName(core_affinity::CoreSide side);
CallPtr CreateTfree(core_affinity::CoreSide side, const ExprPtr& tile, const Span& span);

std::unordered_set<const Var*> CollectStmtVarRefs(const StmtPtr& stmt);
std::unordered_set<const Var*> CollectStmtDefinedVars(const StmtPtr& stmt);
std::unordered_set<const Var*> CollectCallArgVarRefs(const StmtPtr& stmt);
std::vector<const Var*> GetSortedVarRefs(const std::unordered_set<const Var*>& refs);

bool StmtReferencesVar(const StmtPtr& stmt, const Var* var);

struct TpopChain {
  size_t tpop_idx;
  std::vector<size_t> user_idxs;
  size_t tfree_idx = std::numeric_limits<size_t>::max();
  VarPtr tpop_var;
  size_t last_use_idx;
};

const Var* CanonicalizeTpopRef(const Var* var, const std::unordered_map<const Var*, VarPtr>& tpop_var_remap);

std::vector<StmtPtr> NormalizeTpopChains(const std::vector<StmtPtr>& stmts, core_affinity::CoreSide side,
                                         const std::unordered_map<const Var*, VarPtr>& tpop_var_remap);

}  // namespace tpop_chain
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_TPOP_CHAIN_NORMALIZER_H_
