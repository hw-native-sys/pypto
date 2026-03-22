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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_NORMALIZE_STMT_STRUCTURE_H_
#define PYPTO_IR_TRANSFORMS_UTILS_NORMALIZE_STMT_STRUCTURE_H_

#include "pypto/ir/function.h"

namespace pypto::ir {

/**
 * @brief Normalize statement structure in IR
 *
 * This utility ensures IR is in a normalized form:
 * 1. Single-child SeqStmts are unwrapped (no redundant nesting)
 * 2. No nested SeqStmts (SeqStmts as child of SeqStmts)
 *
 * Example transformations:
 *   SeqStmts([ReturnStmt(x)])
 *   => ReturnStmt(x)
 *
 * @param func Input function
 * @return Transformed function with normalized statement structure
 */
FunctionPtr NormalizeStmtStructure(const FunctionPtr& func);

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_NORMALIZE_STMT_STRUCTURE_H_
