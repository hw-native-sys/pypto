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

#include "pypto/ir/function.h"

#include <memory>
#include <utility>
#include <vector>

namespace pypto {
namespace ir {

TypePtr MakeCallResultType(const std::vector<TypePtr>& return_types, bool is_submit) {
  // Use the same packing rule for parser-built calls and pass-synthesized calls.
  // TaskId belongs to the launch, so it must not alter the callee's signature.
  if (is_submit) {
    auto elements = return_types;
    elements.push_back(std::make_shared<ScalarType>(DataType::TASK_ID));
    return std::make_shared<TupleType>(std::move(elements));
  }
  if (return_types.empty()) return GetUnknownType();
  if (return_types.size() == 1) return return_types.front();
  return std::make_shared<TupleType>(return_types);
}

}  // namespace ir
}  // namespace pypto
