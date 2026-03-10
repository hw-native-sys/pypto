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

#include "pypto/backend/common/target.h"

#include <cstdlib>
#include <optional>
#include <string>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"

namespace pypto {
namespace backend {

std::string TargetTypeToString(TargetType target) {
  switch (target) {
    case TargetType::Ascend910B:
      return "Ascend910B";
    case TargetType::Ascend910C:
      return "Ascend910C";
    case TargetType::Ascend950:
      return "Ascend950";
  }
  INTERNAL_CHECK(false) << "Unknown TargetType";
  return "";  // unreachable
}

TargetType TargetTypeFromString(const std::string& s) {
  if (s == "910B" || s == "Ascend910B") return TargetType::Ascend910B;
  if (s == "910C" || s == "Ascend910C") return TargetType::Ascend910C;
  if (s == "950" || s == "Ascend950") return TargetType::Ascend950;
  throw pypto::ValueError("Unknown target type: '" + s + "'. Valid values: 910B, 910C, 950");
}

std::optional<TargetType> GetTargetFromEnv() {
  const char* env = std::getenv("PYPTO_TARGET");
  if (env == nullptr || std::string(env).empty()) return std::nullopt;
  return TargetTypeFromString(env);
}

}  // namespace backend
}  // namespace pypto
