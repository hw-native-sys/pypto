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

#ifndef PYPTO_BACKEND_COMMON_PTO_NATIVE_OPS_H_
#define PYPTO_BACKEND_COMMON_PTO_NATIVE_OPS_H_

#include <string_view>

namespace pypto::backend {

/// Native spelling shared by Tile callbacks and Buffer compute recipes.
/// This is emission metadata, not a capability table: each IR path retains its
/// own operand, storage and target validation before reaching native emission.
struct PtoNativeOpSpec {
  const char* native_op;
  const char* precision_attribute = nullptr;
};

/// Only explicitly migrated compute families are listed. Missing names are an
/// internal registration error, never a reason to enable an unsupported recipe.
const PtoNativeOpSpec& GetPtoNativeOpSpec(std::string_view logical_op);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_PTO_NATIVE_OPS_H_
