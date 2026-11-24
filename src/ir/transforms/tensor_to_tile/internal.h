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

#ifndef SRC_IR_TRANSFORMS_TENSOR_TO_TILE_INTERNAL_H_
#define SRC_IR_TRANSFORMS_TENSOR_TO_TILE_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/call_shape_binding.h"

namespace pypto {
namespace ir {
namespace tensor_to_tile {

// Private stage boundaries. Each conversion owns its analysis and mutator
// state; only transformed functions and explicit output bindings cross stages.

// A return position and an output parameter are different identities: a
// wrapper may need private output parameters which it does not itself return.
struct AddedOutputBinding {
  size_t param_index;
};
using AddedOutputBindings = std::vector<AddedOutputBinding>;
using AddedOutputMap = std::unordered_map<std::string, AddedOutputBindings>;

struct IncoreTransformResult {
  FunctionPtr func;
  AddedOutputBindings added_outputs;
};

struct WrapperTransformResult {
  FunctionPtr func;
  AddedOutputBindings added_outputs;
};

using CallShapePlans = std::map<std::string, CallShapeBindingPlan>;

// Preserve the existing generated output naming in kernels and callers.
std::string MakeOutParamName(size_t index);

// Scalar staging and access analysis share the same GM alias rules.
StmtPtr CanonicalizeScalarUpdates(const FunctionPtr& func);
/// Propagate SIMT GM reference updates into dispatchable kernel signatures.
ProgramPtr PropagateSimtKernelDirections(const ProgramPtr& program);
void UpgradeWrittenTensorParamDirections(const std::vector<StmtPtr>& stmts, const std::vector<VarPtr>& params,
                                         std::vector<ParamDirection>& param_directions);

// Used by scalar-fill matching and returned assemble-loop rewriting.
bool ExprUsesVar(const ExprPtr& expr, const Var* target);
bool StmtUsesVar(const StmtPtr& stmt, const Var* target);

// Capacity checks raise this signal only for a concrete on-chip allocation
// request. Other conversion, verification and code-generation errors propagate.
class TensorAllocationFailure : public pypto::ValueError {
 public:
  struct Request {
    FunctionPtr function;
    MemorySpace space;
    uint64_t required;
    uint64_t capacity;
    std::string detail;
  };

  TensorAllocationFailure(FunctionPtr function, MemorySpace space, uint64_t required, uint64_t capacity,
                          const std::string& detail)
      : TensorAllocationFailure(
            std::vector<Request>{{std::move(function), space, required, capacity, detail}}) {}

  explicit TensorAllocationFailure(std::vector<Request> requests)
      : ValueError(FirstMessage(requests)), requests(std::move(requests)) {}

  // Function passes collect every failed kernel before asking for one replay.
  // The original first error remains the user-facing error if recovery fails.
  std::vector<Request> requests;

 private:
  static std::string FirstMessage(const std::vector<Request>& requests) {
    INTERNAL_CHECK(!requests.empty()) << "Capacity feedback requires a failed allocation";
    const auto& first = requests.front();
    return MemorySpaceToString(first.space) + " buffer usage (" + std::to_string(first.required) +
           " bytes) exceeds platform limit (" + std::to_string(first.capacity) + " bytes)" + first.detail;
  }
};

// One pipeline invocation owns its immutable pre-outline checkpoint and retry
// plans. Nothing is stored globally or survives into another compilation.
class TensorAllocationRetry {
 public:
  explicit TensorAllocationRetry(ProgramPtr original) : original_(std::move(original)) {}

  // Reject an indivisible rank-2 Mat operand that cannot fit even by itself.
  // Check the declared whole-operand load before Tensor conversion and concrete
  // loads before L0 expansion; aggregate packing stays with the allocator.
  void CheckMatLoadCapacity(const ProgramPtr& program) const;

  // Return a new pre-outline program, or nullptr when recovery is unsupported
  // or no smaller legal plan remains. The caller then rethrows the allocator error.
  ProgramPtr Retry(const TensorAllocationFailure& failure, const ProgramPtr& current);

  // Publish costs only after allocation and the rest of the pipeline succeed.
  void EmitSelectedDiagnostics() const;

 private:
  struct Plan {
    uint64_t divisor;
    StmtPtr replacement;
    std::vector<Diagnostic> diagnostics;
  };
  ProgramPtr original_;
  std::unordered_map<const InCoreScopeStmt*, Plan> plans_;
};

IncoreTransformResult TransformIncoreFunction(const FunctionPtr& func);

WrapperTransformResult PropagateOutputsThroughWrapper(
    const FunctionPtr& func, const AddedOutputMap& added_outputs,
    const std::unordered_map<std::string, FunctionPtr>& functions, const CallShapePlans& shape_plans);

FunctionPtr UpdateCallSites(const FunctionPtr& func, const AddedOutputMap& added_outputs,
                            const std::unordered_map<std::string, FunctionPtr>& functions,
                            const CallShapePlans& shape_plans);

}  // namespace tensor_to_tile
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_TENSOR_TO_TILE_INTERNAL_H_
