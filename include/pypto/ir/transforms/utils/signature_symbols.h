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
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"

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

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SIGNATURE_SYMBOLS_H_
