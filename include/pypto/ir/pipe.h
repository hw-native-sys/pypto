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

#ifndef PYPTO_IR_PIPE_H_
#define PYPTO_IR_PIPE_H_

#include <string>

#include "pypto/core/error.h"

namespace pypto {
namespace ir {

/**
 * @brief Pipeline type enumeration for hardware execution units
 */
enum PipeType : int {
  MTE1,  ///< Memory Transfer Engine 1 (L1 -> L0A/L0B)
  MTE2,  ///< Memory Transfer Engine 2 (DDR -> UB)
  MTE3,  ///< Memory Transfer Engine 3 (UB -> DDR)
  M,     ///< Matrix Unit
  V,     ///< Vector Unit
  S,     ///< Scalar Unit
  FIX,   ///< Fix Pipe (L0C -> UB)
  ALL    ///< All Pipes
};

/**
 * @brief Convert PipeType to its string name
 * @param pipe The pipeline type
 * @return String representation (e.g., "MTE1", "MTE2", "V")
 */
inline std::string PipeTypeToString(PipeType pipe) {
  switch (pipe) {
    case PipeType::MTE1:
      return "MTE1";
    case PipeType::MTE2:
      return "MTE2";
    case PipeType::MTE3:
      return "MTE3";
    case PipeType::M:
      return "M";
    case PipeType::V:
      return "V";
    case PipeType::S:
      return "S";
    case PipeType::FIX:
      return "FIX";
    case PipeType::ALL:
      return "ALL";
    default:
      throw pypto::TypeError("Unknown PipeType: " + std::to_string(static_cast<int>(pipe)));
  }
}

/**
 * @brief Core type enumeration (numeric values must match runtime add_task expectation)
 */
enum CoreType : int {
  CUBE = 0,   ///< Cube Core (Alias for AIC)
  VECTOR = 1  ///< Vector Core (Alias for AIV)
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_PIPE_H_
