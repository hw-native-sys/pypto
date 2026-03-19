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

/**
 * @file span.h
 * @brief Source location tracking for IR nodes
 *
 * This header provides the Span class for tracking source code locations
 * throughout the IR. Span objects are immutable and can be shared across
 * multiple IR nodes.
 */

#ifndef PYPTO_IR_SPAN_H_
#define PYPTO_IR_SPAN_H_

#include <string>

namespace pypto {
namespace ir {

/**
 * @brief Represents a source code location span
 *
 * Immutable structure that captures the location of an IR node in the original
 * source code. Used for error reporting and debugging.
 *
 * The span includes:
 * - Source filename
 * - Beginning line and column (1-indexed)
 * - Ending line and column (1-indexed, -1 means unknown)
 */
class Span {
 public:
  const std::string filename_;  ///< Source filename
  const int begin_line_;        ///< Beginning line number (1-indexed)
  const int begin_column_;      ///< Beginning column number (1-indexed)
  const int end_line_;          ///< Ending line number (1-indexed), -1 means unknown
  const int end_column_;        ///< Ending column number (1-indexed), -1 means unknown

  /**
   * @brief Construct a source span
   *
   * @param file Source filename
   * @param begin_line Begin line (1-indexed)
   * @param begin_column Begin column (1-indexed)
   * @param end_line End line (1-indexed), -1 means unknown
   * @param end_column End column (1-indexed), -1 means unknown
   */
  Span(std::string file, int begin_line, int begin_column, int end_line = -1, int end_column = -1);

  /**
   * @brief Convert span to string representation
   *
   * @return String in format "filename:begin_line:begin_column"
   */
  [[nodiscard]] std::string to_string() const;

  /**
   * @brief Check if the span is valid (has valid line/column numbers)
   *
   * @return true if all line/column numbers are positive
   */
  [[nodiscard]] bool is_valid() const;

  /**
   * @brief Create an unknown/invalid span
   *
   * @return Span with empty filename and invalid coordinates
   */
  static Span unknown();
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_SPAN_H_
