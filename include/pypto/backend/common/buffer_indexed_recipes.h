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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_INDEXED_RECIPES_H_
#define PYPTO_BACKEND_COMMON_BUFFER_INDEXED_RECIPES_H_

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto::backend {

/// Indexed operations use flattened physical element offsets. Gather's third
/// input is writable scratch; scatter clears its entire destination first.
struct BufferIndexedRecipe {
  const char* logical_op;
  const char* buffer_op;
  const char* native_op;
  bool scatter;
  size_t input_count;
};

const std::vector<BufferIndexedRecipe>& GetBufferIndexedRecipes();
const BufferIndexedRecipe* FindLogicalBufferIndexedRecipe(const std::string& name);
const BufferIndexedRecipe* FindBufferIndexedRecipe(const std::string& name);
bool SupportsLogicalBufferIndexed(const BufferIndexedRecipe& recipe, const ir::CallPtr& call);
void ValidateBufferIndexedOperands(const BufferIndexedRecipe& recipe, const std::vector<ir::ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs);
std::vector<std::pair<size_t, size_t>> BufferIndexedDisjointPairs(const BufferIndexedRecipe& recipe);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_INDEXED_RECIPES_H_
