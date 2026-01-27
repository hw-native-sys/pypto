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

#ifndef PYPTO_IR_MEMREF_H_
#define PYPTO_IR_MEMREF_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declarations
class Expr;
using ExprPtr = std::shared_ptr<const Expr>;

class MemRef;
using MemRefPtr = std::shared_ptr<const MemRef>;

/**
 * @brief Memory space enumeration
 *
 * Defines the available memory spaces in the hardware hierarchy:
 * - DDR: Double Data Rate memory (off-chip)
 * - UB: Unified Buffer (on-chip shared memory)
 * - L1: L1 cache
 * - L0A: L0A buffer (matrix A operand)
 * - L0B: L0B buffer (matrix B operand)
 * - L0C: L0C buffer (matrix C/result)
 */
enum class MemorySpace {
  DDR,  ///< DDR memory (off-chip)
  UB,   ///< Unified Buffer (on-chip)
  L1,   ///< L1 cache
  L0A,  ///< L0A buffer
  L0B,  ///< L0B buffer
  L0C   ///< L0C buffer
};

/**
 * @brief Convert MemorySpace enum to string
 *
 * @param space Memory space enum value
 * @return String representation
 */
std::string MemorySpaceToString(MemorySpace space);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMREF_H_
