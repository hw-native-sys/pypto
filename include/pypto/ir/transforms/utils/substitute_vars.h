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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SUBSTITUTE_VARS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SUBSTITUTE_VARS_H_

#include <unordered_map>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

/// Substitute variable references by pointer identity.
///
/// Walks the IR subtree and replaces each Var whose raw pointer appears
/// in @p var_map with the mapped VarPtr.  IterArg nodes are handled by
/// the base IRMutator (preserving their type for ForStmt/WhileStmt slots).
///
/// @param body    Statement subtree to transform.
/// @param var_map Pointer-based substitution map (original Var* -> replacement VarPtr).
/// @return Transformed statement subtree.
StmtPtr SubstituteVars(const StmtPtr& body, const std::unordered_map<const Var*, VarPtr>& var_map);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SUBSTITUTE_VARS_H_
