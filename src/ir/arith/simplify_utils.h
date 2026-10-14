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

#ifndef SRC_IR_ARITH_SIMPLIFY_UTILS_H_
#define SRC_IR_ARITH_SIMPLIFY_UTILS_H_

#include "pypto/core/dtype.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace arith {

/// Retain integer conversions when a rewrite selects an operand of another
/// dtype, e.g. an INDEX substituted into explicitly INT32 arithmetic.
inline ExprPtr PreserveIntegerResultType(const ExprPtr& original, const ExprPtr& result) {
  auto scalar = As<ScalarType>(original->GetType());
  if (!scalar || !(scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX)) return result;
  auto result_dtype = GetScalarDtype(result);
  if (!(result_dtype.IsInt() || result_dtype == DataType::INDEX) || result_dtype == scalar->dtype_) {
    return result;
  }
  if (auto folded = TryConstFoldCast(result, scalar->dtype_)) return folded;
  return MakeCast(result, scalar->dtype_, original->span_);
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_ARITH_SIMPLIFY_UTILS_H_
