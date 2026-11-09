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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SIGNATURE_SYMBOLS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SIGNATURE_SYMBOLS_H_

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir {

/// A valid-shape symbol with runtime-readable (parameter, dimension) slots.
struct ValidShapeSymbol {
  VarPtr symbol;
  std::vector<std::pair<size_t, size_t>> slots;
};

struct UnboundValidShapeSymbol {
  size_t parameter;
  ExprPtr expression;
  VarPtr symbol;
};

/// Ordered by first appearance, independent of pointer values and names.
struct SignatureSymbols {
  std::vector<VarPtr> bound;
  std::vector<ValidShapeSymbol> valid_shape;
  std::vector<UnboundValidShapeSymbol> unbound;
};

/// Classify physical-shape/scalar bindings and bare valid-shape sources.
/// Visits each shape-expression DAG node once per phase, including shared
/// parameter types. Does not infer bindings from returns, strides or attrs.
SignatureSymbols ClassifySignatureSymbols(const FunctionPtr& function);

/// Fold immutable tensor.dim bindings onto the signature extents they read.
/// Also valid before SSA: parameters and multiply assigned locals are excluded.
/// Before inline expansion, provide the callee table to count scalar writebacks.
/// Two linear body walks count definitions and apply transitive substitutions.
StmtPtr FoldParamDimReads(const std::vector<VarPtr>& params, const StmtPtr& body,
                          const std::unordered_map<std::string, FunctionPtr>* functions = nullptr);

/// Cache the caller's bare tensor dimension symbols once before inline expansion.
std::unordered_set<const Var*> CollectParamDimSymbols(const std::vector<VarPtr>& params);

/// Fold a cloned body using cached caller symbols and protected writeback Vars.
/// Only the body is traversed; large caller tables are queried, never copied.
/// Constant folding requires the caller to re-infer dependent operation types.
StmtPtr FoldSignatureDimReads(const StmtPtr& body, const std::unordered_set<const Var*>& symbols,
                              const std::unordered_set<const Var*>& preserved, bool fold_constants = false,
                              const std::unordered_map<std::string, FunctionPtr>* functions = nullptr);

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SIGNATURE_SYMBOLS_H_
