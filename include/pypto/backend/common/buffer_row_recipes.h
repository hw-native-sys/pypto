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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_ROW_RECIPES_H_
#define PYPTO_BACKEND_COMMON_BUFFER_ROW_RECIPES_H_

#include <string>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto::backend {

/// Both forms have two inputs and one explicit destination. Reduction input 1
/// is written scratch; broadcast input 1 is a read-only column vector. Keeping
/// that distinction here gives selection, validation and emission one contract.
struct BufferRowRecipe {
  const char* logical_op;
  const char* buffer_op;
  const char* native_op;
  bool reduction;
};

const std::vector<BufferRowRecipe>& GetBufferRowRecipes();
const BufferRowRecipe* FindLogicalBufferRowRecipe(const std::string& name);
const BufferRowRecipe* FindBufferRowRecipe(const std::string& name);
bool SupportsLogicalBufferRow(const BufferRowRecipe& recipe, const ir::CallPtr& call);
void ValidateBufferRowOperands(const BufferRowRecipe& recipe, const std::vector<ir::ExprPtr>& args);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_ROW_RECIPES_H_
