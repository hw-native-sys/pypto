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

#ifndef PYPTO_BACKEND_COMMON_TARGET_H_
#define PYPTO_BACKEND_COMMON_TARGET_H_

#include <optional>
#include <string>

namespace pypto {
namespace backend {

enum class TargetType {
  Ascend910B,
  Ascend910C,
  Ascend950,
};

/// Convert TargetType to human-readable string (e.g., "Ascend910B")
std::string TargetTypeToString(TargetType target);

/// Parse string to TargetType. Accepts "910B", "Ascend910B", etc.
/// Throws pypto::ValueError on unrecognized string.
TargetType TargetTypeFromString(const std::string& s);

/// Read PYPTO_TARGET env var and parse to TargetType.
/// Returns nullopt if env var is not set.
/// Throws pypto::ValueError if env var is set but unrecognized.
std::optional<TargetType> GetTargetFromEnv();

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_TARGET_H_
