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
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
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
void UpgradeWrittenTensorParamDirections(const std::vector<StmtPtr>& stmts, const std::vector<VarPtr>& params,
                                         std::vector<ParamDirection>& param_directions);

// Used by scalar-fill matching and returned assemble-loop rewriting.
bool ExprUsesVar(const ExprPtr& expr, const Var* target);
bool StmtUsesVar(const StmtPtr& stmt, const Var* target);

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
