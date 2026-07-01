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

#ifndef PYPTO_CODEGEN_CODEGEN_PRECONDITIONS_H_
#define PYPTO_CODEGEN_CODEGEN_PRECONDITIONS_H_

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace codegen {

/// Verify required IR properties before orchestration codegen.
void VerifyOrchestrationCodegenPreconditions(const ir::ProgramPtr& program, const ir::FunctionPtr& func);

/// Verify required IR properties before distributed codegen.
/// This check is conditional to avoid rejecting valid non-comm-domain programs.
void VerifyDistributedCodegenPreconditions(const ir::ProgramPtr& program);

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_CODEGEN_PRECONDITIONS_H_
