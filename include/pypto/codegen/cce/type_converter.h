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

#ifndef PYPTO_CODEGEN_CCE_TYPE_CONVERTER_H_
#define PYPTO_CODEGEN_CCE_TYPE_CONVERTER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

/**
 * @brief Utility for converting IR types to pto-isa C++ types
 *
 * TypeConverter handles the translation from PyPTO IR type representations
 * to corresponding pto-isa C++ type strings used in generated code.
 */
class TypeConverter {
 public:
  TypeConverter() = default;

  /**
   * @brief Convert TileType to pto-isa TileType string
   *
   * @param tile_type The ir::TileTypePtr
   * @param rows The number of rows
   * @param cols The number of columns
   * @return pto-isa Tile declaration string (e.g., "Tile<TileType::Left, float, 1, 1, BLayout::RowMajor, -1,
   * -1>;")
   */
  [[nodiscard]] std::string ConvertTileType(const ir::TileTypePtr& tile_type, int64_t rows,
                                            int64_t cols) const;

  /**
   * @brief Convert MemorySpace to pto-isa TileType string
   *
   * Maps PyPTO MemorySpace to pto-isa TileType enum values:
   * - Left → "TileType::Left"
   * - Right → "TileType::Right"
   * - Acc → "TileType::Acc"
   * - Mat → "TileType::Mat"
   * - Vec → "TileType::Vec"
   *
   * @param space The memory space
   * @return TileType string (e.g., "TileType::Left", "TileType::Vec")
   */
  [[nodiscard]] std::string ConvertMemorySpaceToTileType(ir::MemorySpace space) const;

  /**
   * @brief Convert PipeType to pto-isa pipe type string
   *
   * Maps PyPTO PipeType to pto-isa pipe type strings with "PIPE_" prefix:
   * - MTE1 → "PIPE_MTE1"
   * - MTE2 → "PIPE_MTE2"
   * - MTE3 → "PIPE_MTE3"
   * - M → "PIPE_M"
   * - V → "PIPE_V"
   * - S → "PIPE_S"
   * - FIX → "PIPE_FIX"
   * - ALL → "PIPE_ALL"
   *
   * @param pipe The pipe type
   * @return C++ pipe type string with "PIPE_" prefix
   */
  [[nodiscard]] std::string ConvertPipeType(ir::PipeType pipe) const;

  /**
   * @brief Convert event ID to pto-isa event ID string
   *
   * Maps event ID (0-7) to pto-isa event ID strings with "EVENT_ID" prefix:
   * - 0 → "EVENT_ID0"
   * - 1 → "EVENT_ID1"
   * - ...
   * - 7 → "EVENT_ID7"
   *
   * @param event_id The event ID (must be in range [0, 7])
   * @return C++ event ID string with "EVENT_ID" prefix
   */
  [[nodiscard]] std::string ConvertEventId(int event_id) const;

  /**
   * @brief Convert cast round mode (int) to pto-isa RoundMode string
   *
   * Maps cast mode integer to pto-isa RoundMode enum values with "RoundMode::CAST_" prefix:
   * - 0 (None)  → "RoundMode::CAST_NONE"
   * - 1 (RINT)  → "RoundMode::CAST_RINT"
   * - 2 (ROUND) → "RoundMode::CAST_ROUND"
   * - 3 (FLOOR) → "RoundMode::CAST_FLOOR"
   * - 4 (CEIL)  → "RoundMode::CAST_CEIL"
   * - 5 (TRUNC) → "RoundMode::CAST_TRUNC"
   * - 6 (ODD)   → "RoundMode::CAST_ODD"
   *
   * @param mode The cast round mode (must be in range [0, 6])
   * @return RoundMode string with "RoundMode::CAST_" prefix
   */
  [[nodiscard]] std::string ConvertCastRoundMode(int mode) const;

  /**
   * @brief Generate Shape type instantiation
   *
   * Converts a shape vector to pto-isa Shape template instantiation.
   * Pads to 5 dimensions with leading 1s.
   *
   * Example: [128, 64] → "Shape<1, 1, 1, 128, 64>"
   *
   * @param dims The shape dimensions (must be constant values)
   * @return Shape type string
   */
  [[nodiscard]] std::string GenerateShapeType(const std::vector<int64_t>& dims) const;

  /**
   * @brief Generate Stride type instantiation for row-major layout
   *
   * Converts a shape vector to pto-isa Stride template instantiation.
   * Calculates row-major strides and pads to 5 dimensions.
   *
   * Example: [128, 64] → "Stride<1, 1, 1, 64, 1>"
   *
   * @param shape The shape dimensions (used to calculate strides)
   * @return Stride type string
   */
  [[nodiscard]] std::string GenerateStrideType(const std::vector<int64_t>& shape) const;
};

}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_CCE_TYPE_CONVERTER_H_
