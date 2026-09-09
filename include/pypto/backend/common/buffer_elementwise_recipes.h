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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_ELEMENTWISE_RECIPES_H_
#define PYPTO_BACKEND_COMMON_BUFFER_ELEMENTWISE_RECIPES_H_

#include <cstddef>
#include <string>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto {
namespace backend {

/// Native precision attributes are selected by the recipe, never by lowering text.
enum class BufferPrecisionKind { None, Div, Log, Recip };
enum class BufferElementwiseTypePolicy { MatchingVec, StaticDenseFP32 };
enum class BufferDestinationAliasPolicy { ExactOrDisjoint, Disjoint };

/// One source of truth for conversion, explicit operands, and native emission.
/// All inputs read data/metadata; operand input_count writes data and reads
/// metadata. Writes cover the active region, not necessarily the allocation.
struct BufferElementwiseRecipe {
  const char* logical_op;
  const char* buffer_op;
  const char* native_op;
  size_t input_count;
  BufferPrecisionKind precision = BufferPrecisionKind::None;
  BufferElementwiseTypePolicy types = BufferElementwiseTypePolicy::StaticDenseFP32;
  BufferDestinationAliasPolicy destination_alias = BufferDestinationAliasPolicy::ExactOrDisjoint;
};

const std::vector<BufferElementwiseRecipe>& GetBufferElementwiseRecipes();
const BufferElementwiseRecipe* FindLogicalBufferElementwiseRecipe(const std::string& name);
const BufferElementwiseRecipe* FindBufferElementwiseRecipe(const std::string& name);

/// Sorted snapshot of actual logical names; physical forms remain recipe-limited.
std::vector<std::string> GetBufferElementwiseRecipeNames();

/// Validate the real explicit operands; does not create or mutate logical Tile IR.
void ValidateBufferElementwiseOperands(const BufferElementwiseRecipe& recipe,
                                       const std::vector<ir::ExprPtr>& args);
const char* BufferPrecisionAttributeName(BufferPrecisionKind precision);

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_BUFFER_ELEMENTWISE_RECIPES_H_
