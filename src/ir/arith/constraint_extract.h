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

/*
 * The arithmetic simplification module takes reference from:
 * - Apache TVM (https://github.com/apache/tvm), Apache License 2.0
 * - MLC-Python (https://github.com/mlc-ai/mlc-python), Apache License 2.0
 */

#ifndef SRC_IR_ARITH_CONSTRAINT_EXTRACT_H_
#define SRC_IR_ARITH_CONSTRAINT_EXTRACT_H_

#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace ir {
namespace arith {

/// Decompose an AND-chain into individual constraints using an iterative worklist.
/// For `a && b`, returns `[a && b, a, b]` (pre-order traversal).
inline std::vector<ExprPtr> ExtractConstraints(const ExprPtr& expr) {
  std::vector<ExprPtr> result;
  std::vector<ExprPtr> worklist({expr});
  while (!worklist.empty()) {
    ExprPtr current = worklist.back();
    worklist.pop_back();
    result.push_back(current);
    if (auto and_node = As<And>(current)) {
      // Push right then left to process left first, maintaining pre-order traversal.
      worklist.push_back(and_node->right_);
      worklist.push_back(and_node->left_);
    }
  }
  return result;
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_ARITH_CONSTRAINT_EXTRACT_H_
