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
 * @brief Core type enumeration
 */
enum CoreType : int {
  VECTOR,  ///< Vector Core (Alias for AIV)
  CUBE     ///< Cube Core (Alias for AIC)
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_PIPE_H_
