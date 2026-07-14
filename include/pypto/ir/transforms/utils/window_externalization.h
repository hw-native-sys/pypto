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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_WINDOW_EXTERNALIZATION_H_
#define PYPTO_IR_TRANSFORMS_UTILS_WINDOW_EXTERNALIZATION_H_

#include <cstddef>
#include <string>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {
namespace window_externalization {

/// Return the GlobalVar name carried by a call, or an empty string for builtin calls.
std::string GetCallFuncName(const CallPtr& call);

/// Info about an InCore function's Out params and their return mappings.
struct OutParamReturnMapping {
  size_t param_index;   ///< Position in param list
  size_t return_index;  ///< Which return value stores to this Out param
  VarPtr param_var;     ///< The Out param variable
};

/// Map eligible Out/InOut params to the function return values that store into them.
std::vector<OutParamReturnMapping> BuildOutParamReturnMappings(const FunctionPtr& func,
                                                               bool include_inout = false);

bool HasWindowizeEnabledFunction(const ProgramPtr& program);
ProgramPtr ApplyWindowExternalization(const ProgramPtr& program);

}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_WINDOW_EXTERNALIZATION_H_
