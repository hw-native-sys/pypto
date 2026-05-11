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

#ifndef PYPTO_IR_MEMORY_SPACE_H_
#define PYPTO_IR_MEMORY_SPACE_H_

#include <string>

namespace pypto {
namespace ir {

/**
 * @brief Memory space enumeration
 *
 * Defines the available memory spaces in the hardware hierarchy:
 * - DDR: Double Data Rate memory (off-chip)
 * - Vec: Vector/unified buffer (on-chip shared memory)
 * - Mat: Matrix/L1 buffer
 * - Left: Left matrix operand buffer
 * - Right: Right matrix operand buffer
 * - Acc: Accumulator buffer
 * - Bias: Bias buffer
 * - ScalarLocal: On-core scalar register file / C stack (for ArrayType)
 */
enum class MemorySpace {
  DDR,          ///< DDR memory (off-chip)
  Vec,          ///< Vector/unified buffer (on-chip)
  Mat,          ///< Matrix/L1 buffer
  Left,         ///< Left matrix operand buffer
  Right,        ///< Right matrix operand buffer
  Acc,          ///< Accumulator buffer
  Bias,         ///< Bias buffer
  ScalarLocal,  ///< On-core scalar register file / C stack (for ArrayType)
};

/**
 * @brief Convert MemorySpace enum to string
 *
 * @param space Memory space enum value
 * @return String representation
 */
std::string MemorySpaceToString(MemorySpace space);

/**
 * @brief Convert string to MemorySpace enum
 *
 * @param str String representation (e.g., "DDR", "Vec", "Mat")
 * @return MemorySpace enum value
 */
MemorySpace StringToMemorySpace(const std::string& str);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMORY_SPACE_H_
