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

#include <string>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace codegen {

/// One runtime registration. Several signatures can share the same generated code.
struct KernelRegistration {
  int func_id;
  std::string code_name;
  ir::CoreType core_type;
  std::vector<std::string> signature;  ///< Tensor directions in payload order; scalars excluded.
};

/// Generated orchestration and its runtime registrations.
struct OrchestrationResult {
  std::string code;
  std::vector<KernelRegistration> kernel_registrations;
  /// Orchestration entry's per-tensor runtime ArgDirection names ("IN"/"OUT"/
  /// "INOUT"), in orch_args tensor order (declaration order, scalars skipped).
  /// This matches the orch tensor index the runtime uses in
  /// bind_callable_to_runtime_impl (orch_args.tensor(i)), letting it set the
  /// ChipCallable signature so read-only IN tensors skip the D2H copy-back and
  /// pure-OUT tensors take the on-device memset fast path.
  std::vector<std::string> orchestration_signature;
};

/**
 * @brief Generate C++ orchestration code for a function
 *
 * Generates C++ code using simpler runtime API:
 * - aicpu_orchestration_config(const ChipTaskArgs& orch_args) returns OrchestrationConfig
 * - aicpu_orchestration_entry(const ChipTaskArgs& orch_args)
 * - orch_args.tensor(i).ref() for ND external tensors, make_tensor for internal tensors
 * - PTOParam + rt_submit_*_task for task submission (rt_submit_aic_task /
 *   rt_submit_aiv_task for single-core kernels; rt_submit_task for mixed kernels
 *   and for a single AIV kernel stamped `dual_aiv_dispatch`, which must occupy
 *   both vector lanes of one cluster)
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
