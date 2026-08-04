/*
 * Copyright (c) PyPTO Contributors. This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please
 * refer to the License for details. You may not use this file except in compliance with the License. THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "src/ir/transforms/auto_tile/vector_cost_910b.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {
namespace {

constexpr double kCountModeFloorCycles = 16.0;
constexpr double kRowReducePassCycles = 45.0;
constexpr double kRowReduceFinalCycles = 51.0;
constexpr double kColReduceSlopeCycles = 16.0;
constexpr double kColReduceLevelCycles = 30.0;

struct AxisReductionFormula {
  int64_t cols;
  double slope;
  double bias;
};

// PTO-ISA A2/A3 formula backend, formula_params.csv @ d5eaf8e4. These
// constants are copied unchanged from the silicon-grounded 910B scheduler.
constexpr std::array<AxisReductionFormula, 12> kRowMaxFp32{{
    {8, 0.875, 46},
    {32, 0.2188, 47},
    {64, 0.1094, 32},
    {96, 0.0937, 61},
    {128, 0.0625, 48},
    {144, 0.0692, 77},
    {176, 0.0558, 79},
    {208, 0.0517, 100},
    {240, 0.0448, 100},
    {272, 0.0485, 98},
    {304, 0.0434, 110},
    {336, 0.0418, 131},
}};

constexpr std::array<AxisReductionFormula, 12> kRowSumFp32{{
    {8, 0.875, 58},
    {32, 0.2187, 59},
    {64, 0.1094, 44},
    {96, 0.0938, 75},
    {128, 0.0625, 61.5},
    {144, 0.0698, 92},
    {176, 0.0575, 93},
    {208, 0.0535, 99},
    {240, 0.0469, 99},
    {272, 0.0484, 104},
    {304, 0.0431, 104},
    {336, 0.0419, 121},
}};

constexpr std::array<AxisReductionFormula, 20> kRowMaxFp16{{
    {32, 0.2187, 48},   {64, 0.1094, 43},   {96, 0.0729, 43},   {128, 0.0547, 33},  {160, 0.0562, 62},
    {208, 0.0433, 63},  {240, 0.0388, 62},  {272, 0.0369, 74},  {304, 0.0323, 80},  {336, 0.0302, 97},
    {368, 0.0276, 93},  {400, 0.0279, 92},  {432, 0.025, 101},  {464, 0.0232, 119}, {496, 0.0222, 119},
    {528, 0.0254, 111}, {560, 0.0241, 110}, {592, 0.0232, 127}, {624, 0.022, 127},  {656, 0.0224, 125},
}};

constexpr std::array<AxisReductionFormula, 20> kRowSumFp16{{
    {32, 0.2188, 62},   {64, 0.1094, 57},   {96, 0.0729, 57},   {128, 0.0547, 47},  {160, 0.0563, 78},
    {208, 0.0436, 79},  {240, 0.0391, 79},  {272, 0.0366, 94},  {304, 0.0323, 102}, {336, 0.0298, 115},
    {368, 0.0272, 111}, {400, 0.0275, 103}, {432, 0.0255, 103}, {464, 0.0238, 121}, {496, 0.0229, 121},
    {528, 0.0249, 111}, {560, 0.0234, 106}, {592, 0.0219, 140}, {624, 0.0208, 140}, {656, 0.0213, 126},
}};

constexpr std::array<AxisReductionFormula, 5> kColMaxFp32{{
    {8, 2.125, 24},
    {32, 0.5391, 16},
    {64, 0.2734, 17},
    {96, 0.3646, 2},
    {128, 0.1571, 10},
}};

constexpr std::array<AxisReductionFormula, 6> kColMaxFp16{{
    {32, 0.5352, 16},
    {64, 0.2695, 16},
    {96, 0.1836, 19},
    {128, 0.1367, 17},
    {160, 0.2187, 2},
    {192, 0.1823, 2.3325},
}};

constexpr std::array<AxisReductionFormula, 13> kColSumFp32{{
    {8, 2.375, 29},
    {16, 1.1958, 18.271},
    {32, 0.6016, 15},
    {64, 0.3047, 15},
    {96, 0.2187, 14},
    {128, 0.1728, 8},
    {144, 0.1578, 27},
    {176, 0.1314, 30},
    {208, 0.1166, 36},
    {240, 0.1031, 36},
    {272, 0.096, 35},
    {304, 0.0876, 35},
    {336, 0.0833, 34},
}};

constexpr std::array<AxisReductionFormula, 21> kColSumFp16{{
    {16, 1.1916, 20.664}, {32, 0.5977, 15},  {64, 0.3008, 15},  {96, 0.2044, 15},  {128, 0.1523, 15},
    {160, 0.1312, 14},    {208, 0.1029, 25}, {240, 0.09, 28},   {272, 0.0838, 27}, {304, 0.0755, 30},
    {336, 0.069, 30},     {368, 0.069, 30},  {400, 0.0616, 29}, {432, 0.057, 36},  {464, 0.0536, 36},
    {496, 0.0507, 36},    {528, 0.0498, 35}, {560, 0.0474, 35}, {592, 0.0453, 35}, {624, 0.0434, 35},
    {656, 0.0431, 34},
}};

struct PrimitiveCost {
  double slope;
  double fixed;
  bool count_mode;
};

PrimitiveCost PrimitiveGrounding(VectorPrimitive primitive) {
  switch (primitive) {
    case VectorPrimitive::Generic:
      return {2.0, 32.0, false};
    case VectorPrimitive::Add:
      return {2.0, 24.0, true};
    case VectorPrimitive::Mul:
      return {2.0, 25.0, true};
    case VectorPrimitive::Div:
      return {4.0, 30.0, true};
    case VectorPrimitive::Exp:
      return {2.0, 31.0, false};
    case VectorPrimitive::Log:
      return {2.0, 33.0, false};
    case VectorPrimitive::Abs:
      return {1.0, 29.0, false};
    case VectorPrimitive::Sqrt:
      return {2.0, 39.0, false};
    case VectorPrimitive::Rsqrt:
      return {1.0, 24.0, false};
    case VectorPrimitive::ScalarAdd:
      return {1.0, 31.0, false};
    case VectorPrimitive::ScalarMul:
      return {1.0, 26.0, false};
    case VectorPrimitive::ScalarMax:
      return {1.0, 23.0, false};
    case VectorPrimitive::ScalarMin:
      return {1.0, 30.0, false};
    // Preserve the existing conservative local grounding for native paths
    // absent from the original PTO-ISA primitive table. No coefficient is fit.
    case VectorPrimitive::Cast:
      return {1.0, 24.0, false};
    case VectorPrimitive::Recip:
      return {2.0, 30.0, false};
    case VectorPrimitive::RowSum:
    case VectorPrimitive::RowExtrema:
    case VectorPrimitive::ColSum:
    case VectorPrimitive::ColExtrema:
      return {0.0, 0.0, false};
  }
  return {2.0, 32.0, false};
}

template <size_t N>
double InterpolateReductionCycles(const std::array<AxisReductionFormula, N>& table, int64_t valid_rows,
                                  int64_t valid_cols) {
  if (valid_rows <= 0 || valid_cols <= 0) return -1.0;
  auto at = [&](const AxisReductionFormula& entry) {
    return entry.slope * static_cast<double>(valid_rows) * static_cast<double>(entry.cols) + entry.bias;
  };
  const auto upper =
      std::lower_bound(table.begin(), table.end(), valid_cols,
                       [](const AxisReductionFormula& entry, int64_t cols) { return entry.cols < cols; });
  if (upper == table.begin()) return std::round(at(*upper));
  if (upper == table.end()) {
    const auto& last = table.back();
    return std::round(at(last) * static_cast<double>(valid_cols) / static_cast<double>(last.cols));
  }
  if (upper->cols == valid_cols) return std::round(at(*upper));
  const auto& lower = *(upper - 1);
  const double alpha =
      static_cast<double>(valid_cols - lower.cols) / static_cast<double>(upper->cols - lower.cols);
  return std::round(at(lower) + alpha * (at(*upper) - at(lower)));
}

double GroundedReduction(VectorOpKind kind, const DataType& dtype, int64_t rows, int64_t cols) {
  if (kind == VectorOpKind::RowSum) {
    if (dtype == DataType::FP32) return InterpolateReductionCycles(kRowSumFp32, rows, cols);
    if (dtype == DataType::FP16) return InterpolateReductionCycles(kRowSumFp16, rows, cols);
  } else if (kind == VectorOpKind::RowMax) {
    if (dtype == DataType::FP32) return InterpolateReductionCycles(kRowMaxFp32, rows, cols);
    if (dtype == DataType::FP16) return InterpolateReductionCycles(kRowMaxFp16, rows, cols);
  } else if (kind == VectorOpKind::ColSum) {
    if (dtype == DataType::FP32) return InterpolateReductionCycles(kColSumFp32, rows, cols);
    if (dtype == DataType::FP16) return InterpolateReductionCycles(kColSumFp16, rows, cols);
  } else if (kind == VectorOpKind::ColMax) {
    if (dtype == DataType::FP32) return InterpolateReductionCycles(kColMaxFp32, rows, cols);
    if (dtype == DataType::FP16) return InterpolateReductionCycles(kColMaxFp16, rows, cols);
  }
  return -1.0;
}

VectorReductionCost RowReductionCost(VectorOpKind kind, const DataType& dtype, int64_t rows, int64_t cols,
                                     int64_t vector_register_bytes) {
  const double grounded = GroundedReduction(kind, dtype, rows, cols);
  if (grounded >= 0.0) return {grounded, false};
  const int64_t epr = std::max<int64_t>(1, vector_register_bytes / DTypeBytes(dtype));
  const int64_t passes = std::max<int64_t>(1, (cols + epr - 1) / epr);
  return {kRowReducePassCycles * static_cast<double>(passes - 1) + kRowReduceFinalCycles, true};
}

}  // namespace

double PointwiseCycles910B(VectorPrimitive primitive, VectorGeometry geometry, const DataType& output_dtype,
                           int64_t rows, int64_t cols, bool stream_start, bool row_expand_composite,
                           int64_t vector_register_bytes) {
  const int64_t element_bytes = DTypeBytes(output_dtype);
  const int64_t epr = std::max<int64_t>(1, vector_register_bytes / element_bytes);
  const bool expanded = geometry == VectorGeometry::RowExpand || geometry == VectorGeometry::ColExpand;
  const int64_t repeats = expanded ? rows * ((cols + epr - 1) / epr) : (rows * cols + epr - 1) / epr;
  PrimitiveCost cost = PrimitiveGrounding(primitive);
  const bool composite = geometry == VectorGeometry::RowExpand && row_expand_composite;
  if (composite) cost.fixed += 19.0;  // vbrcb + PIPE_V barrier
  double cycles = cost.slope * static_cast<double>(repeats);
  if (stream_start || composite) cycles += cost.fixed;
  if (cost.count_mode) {
    bool count_mode = cols % epr != 0;
    if (geometry == VectorGeometry::RowExpand) {
      const int64_t block_elems = std::max<int64_t>(1, 32 / element_bytes);
      count_mode = cols / epr > rows || (cols + block_elems - 1) / block_elems > 255;
    }
    if (count_mode) cycles += kCountModeFloorCycles;
  }
  return cycles;
}

VectorReductionCost ReductionCycles910B(VectorOpKind kind, const DataType& dtype, int64_t rows, int64_t cols,
                                        int64_t vector_register_bytes) {
  const double grounded = GroundedReduction(kind, dtype, rows, cols);
  if (grounded >= 0.0) return {grounded, false};
  if (kind == VectorOpKind::ColSum || kind == VectorOpKind::ColMax) {
    return {kColReduceSlopeCycles * static_cast<double>(std::max<int64_t>(0, rows - 1)) +
                kColReduceLevelCycles * (rows > 1 ? std::log2(static_cast<double>(rows)) : 0.0),
            true};
  }
  return RowReductionCost(kind, dtype, rows, cols, vector_register_bytes);
}

double GeneratedReductionMergeCycles910B(int reduced_axis, int64_t free_tile, int64_t iterations,
                                         const DataType& dtype, int64_t work_units,
                                         int64_t vector_register_bytes) {
  if (iterations <= 0) return 0.0;
  const int64_t epr = std::max<int64_t>(1, vector_register_bytes / DTypeBytes(dtype));
  const int64_t repeats = (std::max<int64_t>(1, free_tile) + epr - 1) / epr;
  const PrimitiveCost add = PrimitiveGrounding(VectorPrimitive::Add);
  const bool count_mode = reduced_axis == 1 || free_tile % epr != 0;
  const double per_task =
      add.slope * static_cast<double>(repeats) + add.fixed + (count_mode ? kCountModeFloorCycles : 0.0);
  return per_task * static_cast<double>(iterations) * static_cast<double>(std::max<int64_t>(1, work_units));
}

double GeneratedSoftmaxCycles910B(bool update, int64_t free_tile, int64_t chunk_extent, int64_t iterations,
                                  const DataType& dtype, int64_t work_units, int64_t vector_register_bytes,
                                  bool* used_reduction_fallback) {
  if (chunk_extent <= 0 || iterations <= 0) return 0.0;
  const int64_t element_bytes = DTypeBytes(dtype);
  const int64_t epr = std::max<int64_t>(1, vector_register_bytes / element_bytes);
  const int64_t free = std::max<int64_t>(1, free_tile);
  const int64_t wide_repeats = (free * chunk_extent + epr - 1) / epr;
  const int64_t row_expand_repeats = free * ((chunk_extent + epr - 1) / epr);
  const int64_t thin_repeats = (free + epr - 1) / epr;
  const int64_t block_elems = std::max<int64_t>(1, 32 / element_bytes);
  const bool row_expand_count =
      chunk_extent / epr > free || (chunk_extent + block_elems - 1) / block_elems > 255;
  const bool thin_count = 1 % epr != 0;

  const VectorReductionCost row_max =
      RowReductionCost(VectorOpKind::RowMax, dtype, free, chunk_extent, vector_register_bytes);
  const VectorReductionCost row_sum =
      RowReductionCost(VectorOpKind::RowSum, dtype, free, chunk_extent, vector_register_bytes);
  if (used_reduction_fallback != nullptr)
    *used_reduction_fallback |= row_max.used_fallback || row_sum.used_fallback;

  const PrimitiveCost add = PrimitiveGrounding(VectorPrimitive::Add);
  const PrimitiveCost exp = PrimitiveGrounding(VectorPrimitive::Exp);
  const PrimitiveCost mul = PrimitiveGrounding(VectorPrimitive::Mul);
  double per_task = row_max.cycles + row_sum.cycles;

  // One wide row-expand subtraction starts an independent stream and includes
  // the emitted vbrcb + barrier composite.
  per_task += add.slope * static_cast<double>(row_expand_repeats) + add.fixed + 19.0;
  if (row_expand_count) per_task += kCountModeFloorCycles;
  per_task += exp.slope * static_cast<double>(wide_repeats);

  if (update) {
    // Online merge: three thin add-family operations in two streams, one thin
    // exp and one thin multiply. This is the exact work descriptor replayed by
    // EmitSoftmaxChunk and consumed by the silicon-grounded scheduler.
    per_task += add.slope * static_cast<double>(3 * thin_repeats) + 2.0 * add.fixed;
    if (thin_count) per_task += 3.0 * kCountModeFloorCycles;
    per_task += exp.slope * static_cast<double>(thin_repeats);
    per_task += mul.slope * static_cast<double>(thin_repeats);
  }
  return per_task * static_cast<double>(iterations) * static_cast<double>(std::max<int64_t>(1, work_units));
}

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto
