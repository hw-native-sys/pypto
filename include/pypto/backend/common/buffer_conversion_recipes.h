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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_CONVERSION_RECIPES_H_
#define PYPTO_BACKEND_COMMON_BUFFER_CONVERSION_RECIPES_H_

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto::backend {

enum class BufferConversionKind { Cast, Compare, CompareScalar, Select, SelectScalar };

/// Conversion recipes carry heterogeneous buffers and packed predicates. Each
/// instruction writes its last operand; scratch is a separate writable input.
struct BufferConversionRecipe {
  const char* logical_op;
  const char* buffer_op;
  const char* native_op;
  BufferConversionKind kind;
  size_t input_count;
  int scalar_index = -1;
  int scratch_index = -1;
};

struct BufferConversionAliasPair {
  size_t first;
  size_t second;
  bool exact_allowed = false;
};

const std::vector<BufferConversionRecipe>& GetBufferConversionRecipes();
const BufferConversionRecipe* FindLogicalBufferConversionRecipe(const std::string& name);
const BufferConversionRecipe* FindBufferConversionRecipe(const std::string& name);
bool SupportsLogicalBufferConversion(const BufferConversionRecipe& recipe, const ir::CallPtr& call);
void ValidateBufferConversionOperands(const BufferConversionRecipe& recipe,
                                      const std::vector<ir::ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs);
std::vector<BufferConversionAliasPair> BufferConversionAliasPairs(const BufferConversionRecipe& recipe);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_CONVERSION_RECIPES_H_
