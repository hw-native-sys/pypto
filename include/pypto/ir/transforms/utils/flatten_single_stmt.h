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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_FLATTEN_SINGLE_STMT_H_
#define PYPTO_IR_TRANSFORMS_UTILS_FLATTEN_SINGLE_STMT_H_

#include <memory>

#include "pypto/ir/function.h"

namespace pypto::ir {

/**
 * @brief Recursively flatten single-statement SeqStmts and OpStmts
 *
 * This utility simplifies IR by removing unnecessary nesting:
 * - SeqStmts with only one statement is replaced by that statement
 * - OpStmts with only one statement is replaced by that statement
 * - Process is applied recursively
 *
 * Example transformations:
 *   SeqStmts([OpStmts([AssignStmt(x, 1)])])
 *   => AssignStmt(x, 1)
 *
 *   SeqStmts([OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])])
 *   => OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])
 *
 *   Function body = SeqStmts([OpStmts([AssignStmt(x, 1)])])
 *   => Function body = AssignStmt(x, 1)
 *
 * Note: This pass does NOT enforce that Function/IfStmt/ForStmt body must be SeqStmts.
 * It will flatten them if they contain only a single statement.
 *
 * @param func Input function
 * @return Transformed function with flattened single-statement blocks
 */
FunctionPtr FlattenSingleStmt(const FunctionPtr& func);

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_FLATTEN_SINGLE_STMT_H_
