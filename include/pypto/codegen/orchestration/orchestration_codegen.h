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

#ifndef PYPTO_CODEGEN_ORCHESTRATION_ORCHESTRATION_CODEGEN_H_
#define PYPTO_CODEGEN_ORCHESTRATION_ORCHESTRATION_CODEGEN_H_

#include <map>
#include <string>

#include "pypto/ir/function.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace codegen {

/**
 * @brief Result of orchestration code generation
 *
 * Contains generated C++ code and metadata about kernel functions.
 */
struct OrchestrationResult {
  std::string code;                                            ///< Generated C++ orchestration code
  std::map<std::string, int> func_name_to_id;                  ///< Kernel function name -> ID mapping
  std::map<std::string, ir::CoreType> func_name_to_core_type;  ///< Kernel function name -> core type
};

/**
 * @brief Generate C++ orchestration code for a function
 *
 * Generates C++ code using PTO2 runtime API:
 * - aicpu_orchestration_config(TaskArg* orch_args) returns PTO2OrchestrationConfig
 * - aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args)
 * - from_task_arg() for ND external tensors, make_tensor for internal tensors
 * - PTOParam + rt_submit_*_task for task submission (rt_submit_aic_task /
 *   rt_submit_aiv_task for single-core kernels; rt_submit_task for mixed kernels)
 * - No manual dependency management (runtime handles automatically)
 *
 * @param program The IR Program (used to resolve callee functions and validate references)
 * @param func The orchestration function to generate code for
 * @return OrchestrationResult containing generated code and function metadata
 * @throws ValueError if referenced functions are missing from the program
 */
OrchestrationResult GenerateOrchestration(const ir::ProgramPtr& program, const ir::FunctionPtr& func);

/**
 * @brief Infer the core type of a function from operand MemorySpace
 *
 * Determines CoreType based on MemorySpace semantics:
 * - Left/Right/Acc/Mat (CUBE buffers) -> CUBE
 * - Vec (Vector buffer) -> VECTOR
 *
 * @param func The function to infer the core type for
 * @return The core type of the function
 * @throws ValueError if function contains both CUBE and VECTOR ops
 */
ir::CoreType InferFunctionCoreType(const ir::FunctionPtr& func);

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_ORCHESTRATION_ORCHESTRATION_CODEGEN_H_
