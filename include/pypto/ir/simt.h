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

#ifndef PYPTO_IR_SIMT_H_
#define PYPTO_IR_SIMT_H_

#include <cstdint>
#include <string>

#include "pypto/core/error.h"

namespace pypto::ir {

/// A compile-time query axis, deliberately distinct from an integer operand.
enum class SimtAxis : uint8_t { X, Y, Z };

inline std::string SimtAxisToString(SimtAxis axis) {
  switch (axis) {
    case SimtAxis::X:
      return "X";
    case SimtAxis::Y:
      return "Y";
    case SimtAxis::Z:
      return "Z";
  }
  throw pypto::TypeError("Unknown SimtAxis");
}

inline SimtAxis StringToSimtAxis(const std::string& axis) {
  if (axis == "X") return SimtAxis::X;
  if (axis == "Y") return SimtAxis::Y;
  if (axis == "Z") return SimtAxis::Z;
  throw pypto::TypeError("Unknown SimtAxis: " + axis);
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_SIMT_H_
