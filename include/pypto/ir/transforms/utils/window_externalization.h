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

#include "pypto/ir/program.h"

namespace pypto {
namespace ir {
namespace window_externalization {

bool HasWindowizeEnabledFunction(const ProgramPtr& program);
ProgramPtr ApplyWindowExternalization(const ProgramPtr& program);

}  // namespace window_externalization
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_WINDOW_EXTERNALIZATION_H_
