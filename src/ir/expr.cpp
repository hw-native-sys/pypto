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
#include "pypto/ir/expr.h"

#include <memory>
#include <utility>
#include <vector>

#include "pypto/ir/kind_traits.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

MakeTuple::MakeTuple(std::vector<ExprPtr> elements, Span span)
    : Expr(std::move(span)), elements_(std::move(elements)) {
  // Collect types from all element expressions
  std::vector<TypePtr> element_types;
  element_types.reserve(elements_.size());
  for (const auto& elem : elements_) {
    element_types.push_back(elem->GetType());
  }

  // Set result type to TupleType
  type_ = std::make_shared<TupleType>(std::move(element_types));
}

TupleGetItemExpr::TupleGetItemExpr(ExprPtr tuple, int index, Span span)
    : Expr(std::move(span)), tuple_(std::move(tuple)), index_(index) {
  // Type checking: tuple must have TupleType
  auto tuple_type = As<TupleType>(tuple_->GetType());
  CHECK(tuple_type) << "TupleGetItemExpr requires tuple to have TupleType, got "
                    << tuple_->GetType()->TypeName();

  // Bounds checking
  CHECK(index >= 0 && index < static_cast<int>(tuple_type->types_.size()))
      << "TupleGetItemExpr index " << index << " out of bounds for tuple with " << tuple_type->types_.size()
      << " elements";

  // Set result type to the accessed element's type
  type_ = tuple_type->types_[index];
}

}  // namespace ir
}  // namespace pypto
