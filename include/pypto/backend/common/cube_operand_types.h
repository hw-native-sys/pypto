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

#ifndef PYPTO_BACKEND_COMMON_CUBE_OPERAND_TYPES_H_
#define PYPTO_BACKEND_COMMON_CUBE_OPERAND_TYPES_H_

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

#include "pypto/core/dtype.h"

namespace pypto::backend {

/// Architectures a Cube operand pair is available on, as a bit mask.
inline constexpr uint8_t kCubeArchA2A3 = 1U << 0;
inline constexpr uint8_t kCubeArchA5 = 1U << 1;
inline constexpr uint8_t kCubeArchAll = kCubeArchA2A3 | kCubeArchA5;

/// The Cube instruction families that consume an (lhs, rhs) operand pair.
enum class CubeOpFamily : uint8_t {
  kMatmul = 1U << 0,  ///< pto.tmatmul / .acc / .bias
  kGemv = 1U << 1,    ///< pto.tgemv / .acc / .bias
  kMx = 1U << 2,      ///< pto.tmatmul.mx / .acc / .bias
};

/// One (lhs, rhs) element-type pair the Cube multiplies, the L0C accumulator
/// type it produces, and where it is available.
struct CubeOperandPair {
  DataType lhs;
  DataType rhs;
  DataType accumulator;
  uint8_t archs;     ///< kCubeArch* bits
  uint8_t families;  ///< CubeOpFamily bits

  [[nodiscard]] constexpr bool InFamily(CubeOpFamily family) const {
    return (families & static_cast<uint8_t>(family)) != 0;
  }

  /// @param pto_target_arch BackendHandler::GetPtoTargetArch() ("a2a3" / "a5")
  [[nodiscard]] bool SupportsArch(std::string_view pto_target_arch) const {
    if (pto_target_arch == "a2a3") return (archs & kCubeArchA2A3) != 0;
    if (pto_target_arch == "a5") return (archs & kCubeArchA5) != 0;
    return false;
  }
};

namespace cube_operand_detail {
inline constexpr uint8_t kMatmulGemv =
    static_cast<uint8_t>(CubeOpFamily::kMatmul) | static_cast<uint8_t>(CubeOpFamily::kGemv);
inline constexpr uint8_t kMatmulGemvMx = kMatmulGemv | static_cast<uint8_t>(CubeOpFamily::kMx);
}  // namespace cube_operand_detail

/**
 * @brief Cube operand pairs PTOAS accepts, per architecture and instruction family.
 *
 * Mirrors PTOAS v0.72 ``verifyMatmulTypeTriple`` (lib/PTO/IR/PTOTile/PTOMatrixQuantAndAdd.cpp)
 * and pto-isa ``npu/a5/TMatmul.hpp``: every target multiplies i8 x i8 -> i32 and
 * f16 / bf16 / f32 same-type pairs -> f32; A5 additionally multiplies any FP8
 * (E4M3FN / E5M2) pair, including the mixed orders, and HiF8 x HiF8 -> f32.
 *
 * Family bits record what PyPTO currently exposes for each pair. The matmul and
 * gemv families share one PTOAS rule (``pto.tgemv*`` verifies through the same
 * ``verifyMatmulTypeTriple``, and the a5 TileLib templates for ``pto.tgemv`` /
 * ``.acc`` / ``.bias`` carry the same ``MATMUL_*_DTYPES`` signature lists as
 * ``pto.tmatmul``), so every row belongs to both; MX data is E4M3FN x E4M3FN only.
 *
 * The ``kGemv`` bits are not decoration: ``CheckMatmulOperandArch`` skips the
 * architecture check entirely for a pair it cannot find here, so clearing one
 * would silently stop rejecting that pair on a2a3. GEMV's own dtype allow-list is
 * written out longhand in ``ValidateGemvInputDtypes``, and a ``static_assert``
 * there holds the two in agreement.
 */
inline constexpr CubeOperandPair kCubeOperandPairs[] = {
    {DataType::INT8, DataType::INT8, DataType::INT32, kCubeArchAll, cube_operand_detail::kMatmulGemv},
    {DataType::FP16, DataType::FP16, DataType::FP32, kCubeArchAll, cube_operand_detail::kMatmulGemv},
    {DataType::BF16, DataType::BF16, DataType::FP32, kCubeArchAll, cube_operand_detail::kMatmulGemv},
    {DataType::FP32, DataType::FP32, DataType::FP32, kCubeArchAll, cube_operand_detail::kMatmulGemv},
    {DataType::FP8E4M3FN, DataType::FP8E4M3FN, DataType::FP32, kCubeArchA5,
     cube_operand_detail::kMatmulGemvMx},
    {DataType::FP8E4M3FN, DataType::FP8E5M2, DataType::FP32, kCubeArchA5, cube_operand_detail::kMatmulGemv},
    {DataType::FP8E5M2, DataType::FP8E4M3FN, DataType::FP32, kCubeArchA5, cube_operand_detail::kMatmulGemv},
    {DataType::FP8E5M2, DataType::FP8E5M2, DataType::FP32, kCubeArchA5, cube_operand_detail::kMatmulGemv},
    {DataType::HF8, DataType::HF8, DataType::FP32, kCubeArchA5, cube_operand_detail::kMatmulGemv},
};

/// The table entry for (lhs, rhs) in @p family on any architecture, or nullptr.
inline const CubeOperandPair* FindCubeOperandPair(CubeOpFamily family, DataType lhs, DataType rhs) {
  for (const auto& pair : kCubeOperandPairs) {
    if (pair.InFamily(family) && pair.lhs == lhs && pair.rhs == rhs) return &pair;
  }
  return nullptr;
}

/// Is @p dtype one of the 1-byte float Cube operand formats A5 adds?
///
/// These are the element types whose matmul accumulates in FP32 but which no
/// unscaled FIXPIPE writeback can reproduce, so a result tensor must carry the
/// accumulator dtype rather than the operand dtype.
inline bool IsCubeFloat8Operand(DataType dtype) {
  return dtype == DataType::FP8E4M3FN || dtype == DataType::FP8E5M2 || dtype == DataType::HF8;
}

/// Human-readable list of the @p family pairs, e.g. for diagnostics. An empty
/// @p pto_target_arch lists every architecture's pairs.
inline std::string DescribeCubeOperandPairs(CubeOpFamily family, std::string_view pto_target_arch = {}) {
  std::ostringstream out;
  bool first = true;
  for (const auto& pair : kCubeOperandPairs) {
    if (!pair.InFamily(family)) continue;
    if (!pto_target_arch.empty() && !pair.SupportsArch(pto_target_arch)) continue;
    if (!first) out << ", ";
    first = false;
    out << pair.lhs.ToString() << " x " << pair.rhs.ToString();
    if (pto_target_arch.empty() && pair.archs == kCubeArchA5) out << " (A5)";
  }
  return out.str();
}

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_CUBE_OPERAND_TYPES_H_
