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
 * - LeftScale: L0A-side MX block-scale buffer (A5)
 * - RightScale: L0B-side MX block-scale buffer (A5)
 * - ScalarLocal: On-core scalar register file / C stack (for ArrayType)
 */
enum class MemorySpace {
  DDR = 0,          ///< DDR memory (off-chip)
  Vec = 1,          ///< Vector/unified buffer (on-chip)
  Mat = 2,          ///< Matrix/L1 buffer
  Left = 3,         ///< Left matrix operand buffer
  Right = 4,        ///< Right matrix operand buffer
  Acc = 5,          ///< Accumulator buffer
  Bias = 6,         ///< Bias buffer
  ScalarLocal = 7,  ///< On-core scalar register file / C stack (for ArrayType)
  LeftScale = 8,    ///< L0A-side MX block-scale buffer (A5)
  RightScale = 9,   ///< L0B-side MX block-scale buffer (A5)
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

/**
 * @brief Whether *some* target implements a single-instruction tile move from
 *        @p src to @p dst.
 *
 * The union over every backend of PTOAS's `TMovOp::verify` address-space table.
 * A pair outside this union is unimplementable everywhere, so IR-level code can
 * reject it without knowing the target — which matters because type deduction
 * runs while parsing, before any backend is selected.
 *
 * Use this only for that "impossible anywhere" question. Once a backend is
 * configured, `Backend::GetSoC().GetMemoryGraph()` gives the exact per-target
 * adjacency (A5 implements Vec -> Mat, A2/A3 does not) -- the same data
 * `Backend::FindMemPath` walks; passes and codegen want that one.
 *
 * The row worth knowing: **no target moves anything into `Acc`.** Only the MAD
 * unit writes L0C, so an accumulator has to be created in `Acc` — no copy can
 * put it there afterwards.
 *
 * @param src Source memory space
 * @param dst Destination memory space
 * @return True when at least one target implements the move
 */
[[nodiscard]] bool IsTileMoveEverSupported(MemorySpace src, MemorySpace dst);

/**
 * @brief Whether @p dst has any inbound edge at all in the move graph.
 *
 * The `std::any_of` over every possible source of
 * @ref IsTileMoveEverSupported. False means the space is unreachable by copy on
 * every target, so a value that must live there has to be *created* there --
 * no pass can insert a bridge. Today `Acc` is the only such space: nothing
 * writes L0C except the MAD unit.
 *
 * @param dst Destination memory space
 * @return True when some target implements some move into @p dst
 */
[[nodiscard]] bool IsTileMoveEverPossibleInto(MemorySpace dst);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_MEMORY_SPACE_H_
