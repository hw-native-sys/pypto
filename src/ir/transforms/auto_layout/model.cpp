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

#include "pypto/ir/transforms/auto_layout/model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace auto_layout {

namespace {

void HashCombine(size_t& seed, size_t v) noexcept {
  seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

bool IsColMajor32BAligned(const std::vector<int64_t>& shape, DataType dtype) {
  const int bpe = DTypeByteWidthSafe(dtype);
  return bpe > 0 && shape.size() == 2 && shape[0] > 1 && (shape[0] * bpe) % 32 == 0;
}

bool SupportsFlatVecTransposeDtype(DataType dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
         dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
         dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::UINT32;
}

bool ContainsLayout(const std::vector<LayoutSpec>& layouts, const LayoutSpec& target) {
  return std::find(layouts.begin(), layouts.end(), target) != layouts.end();
}

double EstimateVecGmTransferCost(const LayoutCostModel& model, const ValueNode& tile_node,
                                 const ValueNode& gm_node, const LayoutSpec& gm_layout, double contig_bw,
                                 double strided_bw) {
  const double total_bytes = static_cast<double>(tile_node.ByteSize());
  if (tile_node.shape.size() != 2 || gm_node.shape.size() < 2) {
    return total_bytes / contig_bw;
  }
  const int64_t rows = tile_node.shape[tile_node.shape.size() - 2];
  const int64_t cols = tile_node.shape.back();
  const int64_t parent_rows = gm_node.shape[gm_node.shape.size() - 2];
  const int64_t parent_cols = gm_node.shape.back();
  const int bpe = DTypeByteWidth(tile_node.dtype);
  const double gran_b = static_cast<double>(std::max<int64_t>(1, model.GmAccessGranularityBytes()));

  bool strided = false;
  double burst_bytes = 0.0;
  int64_t num_bursts = 0;
  if (gm_layout == LayoutSpec::DN()) {
    strided = rows < parent_rows;
    burst_bytes = static_cast<double>(rows * bpe);
    num_bursts = cols;
  } else if (gm_layout == LayoutSpec::NZ()) {
    const int c0 = C0Elements(tile_node.dtype);
    strided = rows < parent_rows;
    burst_bytes = static_cast<double>(rows * 32);
    num_bursts = std::max<int64_t>(1, cols / c0);
  } else {
    strided = cols < parent_cols;
    burst_bytes = static_cast<double>(cols * bpe);
    num_bursts = rows;
  }
  if (!strided) {
    return total_bytes / contig_bw;
  }
  const double eff_bw = std::min(contig_bw, strided_bw);
  const double penalty = static_cast<double>(num_bursts) * std::max(0.0, (gran_b - burst_bytes) / gran_b) *
                         model.vec_strided_burst_penalty_cycles;
  return (total_bytes / eff_bw) + penalty;
}

}  // namespace

std::string LayoutSpec::ShortName() const {
  if (kind == Kind::kTensor) {
    return TensorLayoutToString(tensor_layout);
  }
  if (tag == "row_bcast") {
    return "flat_row_bcast";
  }
  if (slayout == TileLayout::none_box) {
    return "flat_" + TileLayoutToString(blayout);
  }
  if (fractal == 1024 && blayout == TileLayout::col_major) {
    return "acc_nz_1024";
  }
  if (fractal == 512 && blayout == TileLayout::col_major && slayout == TileLayout::row_major) {
    return "fractal_nz";
  }
  if (fractal == 512 && blayout == TileLayout::row_major && slayout == TileLayout::col_major) {
    return "fractal_zn";
  }
  std::string base =
      TileLayoutToString(blayout) + "/" + TileLayoutToString(slayout) + "/" + std::to_string(fractal);
  if (!tag.empty()) {
    base += "/" + tag;
  }
  return base;
}

bool LayoutSpec::operator==(const LayoutSpec& other) const {
  if (kind != other.kind) {
    return false;
  }
  if (kind == Kind::kTensor) {
    return tensor_layout == other.tensor_layout;
  }
  return blayout == other.blayout && slayout == other.slayout && fractal == other.fractal && tag == other.tag;
}

bool LayoutSpec::operator<(const LayoutSpec& other) const {
  if (kind != other.kind) {
    return static_cast<int>(kind) < static_cast<int>(other.kind);
  }
  if (kind == Kind::kTensor) {
    return static_cast<int>(tensor_layout) < static_cast<int>(other.tensor_layout);
  }
  return std::tie(blayout, slayout, fractal, tag) <
         std::tie(other.blayout, other.slayout, other.fractal, other.tag);
}

size_t LayoutSpecHash::operator()(const LayoutSpec& spec) const noexcept {
  size_t seed = std::hash<uint8_t>{}(static_cast<uint8_t>(spec.kind));
  if (spec.kind == LayoutSpec::Kind::kTensor) {
    HashCombine(seed, std::hash<int>{}(static_cast<int>(spec.tensor_layout)));
  } else {
    HashCombine(seed, std::hash<int>{}(static_cast<int>(spec.blayout)));
    HashCombine(seed, std::hash<int>{}(static_cast<int>(spec.slayout)));
    HashCombine(seed, std::hash<int32_t>{}(spec.fractal));
    HashCombine(seed, std::hash<std::string>{}(spec.tag));
  }
  return seed;
}

size_t LayoutPairHash::operator()(const std::pair<LayoutSpec, LayoutSpec>& p) const noexcept {
  const LayoutSpecHash hash;
  size_t seed = hash(p.first);
  HashCombine(seed, hash(p.second));
  return seed;
}

std::optional<int> TryC0Elements(DataType dtype) {
  const int bits = static_cast<int>(dtype.GetBit());
  if (bits < 8 || (256 % bits) != 0) {
    return std::nullopt;
  }
  return 256 / bits;
}

int C0Elements(DataType dtype) {
  auto c0 = TryC0Elements(dtype);
  if (!c0.has_value()) {
    throw pypto::ValueError("Unsupported or sub-byte dtype for NZ layout: " + dtype.ToString());
  }
  return *c0;
}

int DTypeByteWidthSafe(DataType dtype) {
  const uint64_t bits = storage_size::GetStorageBitWidth(dtype);
  if (bits < 8 || (bits % 8) != 0) {
    return 0;
  }
  return static_cast<int>(bits / 8);
}

int DTypeByteWidth(DataType dtype) {
  const int width = DTypeByteWidthSafe(dtype);
  if (width <= 0) {
    throw pypto::ValueError("Unsupported or sub-byte dtype for byte width: " + dtype.ToString());
  }
  return width;
}

bool IsFlatVecTransposePair(const LayoutSpec& source, const LayoutSpec& destination) {
  return (source == LayoutSpec::RowMajorFlat() && destination == LayoutSpec::ColMajorFlat()) ||
         (source == LayoutSpec::ColMajorFlat() && destination == LayoutSpec::RowMajorFlat());
}

bool IsNzCompatibleShape(const std::vector<int64_t>& shape, DataType dtype) {
  if (shape.size() != 2 && shape.size() != 3) {
    return false;
  }
  const int64_t rows = shape[shape.size() - 2];
  const int64_t cols = shape.back();
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const auto c0 = TryC0Elements(dtype);
  if (!c0.has_value()) {
    return false;
  }
  return (rows % 16) == 0 && (cols % *c0) == 0;
}

int64_t ValueNode::NumElements() const {
  // A zero dimension makes the whole footprint zero, regardless of dimension order.
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
  int64_t product = 1;
  for (int64_t extent : shape) {
    if (extent < 0) {
      throw pypto::ValueError("Value '" + name + "' has negative shape extent " + std::to_string(extent));
    }
    if (product > std::numeric_limits<int64_t>::max() / extent) {
      throw pypto::ValueError("Element count for value '" + name + "' exceeds int64_t");
    }
    product *= extent;
  }
  return product;
}

int64_t ValueNode::ByteSize() const {
  const int64_t elements = NumElements();
  if (elements == 0) return 0;
  const uint64_t bits = storage_size::GetStorageBitWidth(dtype);
  // Compute in bytes so a representable size does not overflow in bits.
  if (bits == 4) return elements / 2 + elements % 2;
  const auto bytes_per_element = static_cast<int64_t>(bits / 8);
  if (bytes_per_element == 0 || elements > std::numeric_limits<int64_t>::max() / bytes_per_element) {
    throw pypto::ValueError("Storage size for value '" + name + "' cannot be represented in bytes");
  }
  return elements * bytes_per_element;
}

bool OperatorConfig::operator==(const OperatorConfig& other) const {
  return op_id == other.op_id && op_name == other.op_name && input_layouts == other.input_layouts &&
         output_layout == other.output_layout && op_cost_cycles == other.op_cost_cycles &&
         notes == other.notes && local_vec_conversions == other.local_vec_conversions &&
         local_vec_load_layout == other.local_vec_load_layout;
}

LayoutSpec OperatorConfig::ExecutionInputLayout(size_t index) const {
  const auto conversion = std::find_if(local_vec_conversions.begin(), local_vec_conversions.end(),
                                       [index](const auto& entry) { return entry.first == index; });
  return conversion != local_vec_conversions.end() ? conversion->second : input_layouts.at(index);
}

void LayoutCostModel::ValidateUseHorizon() const {
  if (expected_uses < 1) throw pypto::ValueError("AutoLayout expected_uses must be positive");
  if (steady_state && expected_uses != 1) {
    throw pypto::ValueError("AutoLayout steady_state requires expected_uses=1");
  }
}

std::string LayoutCostModel::TargetArch() const {
  if (backend_type.has_value()) {
    return (*backend_type == backend::BackendType::Ascend950) ? "a5" : "a2a3";
  }
  return resolved_arch_hint.empty() ? "a2a3" : resolved_arch_hint;
}

int64_t LayoutCostModel::GmAccessGranularityBytes() const {
  return TargetArch() == "a5" ? 128 : l2_cache_line_bytes;
}

bool LayoutCostModel::SupportsWideNzMatLoad() const {
  // The pinned A2/A3 TLOAD implementation splits wide NZ Mat gaps into
  // individual bursts. Other targets keep the 16-bit gap restriction.
  return backend_type == backend::BackendType::Ascend910;
}

bool LayoutCostModel::SupportsAccNzStore(const ValueNode& src_node, const ValueNode& dst_node) const {
  if (TargetArch() == "a5") {
    return false;
  }
  if (dst_node.shape.size() != 2 && dst_node.shape.size() != 3) {
    return false;
  }
  const int64_t rows = dst_node.shape[dst_node.shape.size() - 2];
  const int64_t cols = dst_node.shape.back();
  return src_node.memory_space == MemorySpace::Acc && src_node.shape.size() >= 2 &&
         src_node.shape[src_node.shape.size() - 2] == rows && src_node.dtype == DataType::FP32 &&
         dst_node.dtype == DataType::FP32 && (rows % 16) == 0 && (cols % 16) == 0 &&
         IsNzCompatibleShape(dst_node.shape, dst_node.dtype);
}

double LayoutCostModel::EstimateVbrcbCost(int64_t rows) const {
  if (TargetArch() == "a2a3") {
    return vec_vbrcb_fixed_cycles + static_cast<double>(rows) * vec_vbrcb_cycles_per_row;
  }
  return 0.0;
}

bool LayoutCostModel::IsValidLoadConfig(const ValueNode& src_node, const ValueNode& dst_node,
                                        const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const {
  if (src_node.dtype.IsPackedFp4() || dst_node.dtype.IsPackedFp4()) {
    return src_layout == src_node.current_layout && dst_layout == dst_node.current_layout;
  }
  if (dst_node.memory_space == MemorySpace::Mat) {
    if (src_layout == src_node.current_layout && src_node.admissible_layouts.size() == 1) {
      return dst_layout == dst_node.current_layout;
    }
    if (src_layout == LayoutSpec::NZ()) {
      if (!IsNzCompatibleShape(dst_node.shape, dst_node.dtype)) {
        return false;
      }
      const int c0 = C0Elements(dst_node.dtype);
      if (c0 > 0 && dst_node.shape.size() >= 2 && src_node.shape.size() >= 2) {
        const int64_t col_blocks = dst_node.shape.back() / c0;
        const int64_t gap =
            src_node.shape[src_node.shape.size() - 2] - dst_node.shape[dst_node.shape.size() - 2];
        if (!SupportsWideNzMatLoad() && col_blocks > 1 && gap > 65535) {
          return false;
        }
      }
    }
    return (src_layout == LayoutSpec::ND() || src_layout == LayoutSpec::NZ()) &&
           (dst_layout == dst_node.current_layout);
  }
  if (dst_node.memory_space == MemorySpace::Vec) {
    if (src_layout == LayoutSpec::ND()) {
      return dst_layout == dst_node.current_layout;
    }
    if (src_layout == LayoutSpec::DN()) {
      return dst_layout == LayoutSpec::ColMajorFlat() && src_node.shape.size() == 2 &&
             IsColMajor32BAligned(dst_node.shape, dst_node.dtype);
    }
    if (src_layout == LayoutSpec::NZ()) {
      return dst_layout == LayoutSpec::FractalNZ() &&
             ContainsLayout(dst_node.admissible_layouts, LayoutSpec::FractalNZ()) &&
             IsNzCompatibleShape(dst_node.shape, dst_node.dtype);
    }
    return false;
  }
  return src_layout == LayoutSpec::ND() && dst_layout == dst_node.current_layout;
}

bool LayoutCostModel::IsValidVecStoreConfig(const ValueNode& src_node, const ValueNode& dst_node,
                                            const LayoutSpec& src_layout,
                                            const LayoutSpec& dst_layout) const {
  if (src_node.dtype.IsPackedFp4() || dst_node.dtype.IsPackedFp4()) {
    return src_layout == src_node.current_layout && dst_layout == dst_node.current_layout;
  }
  if (src_layout == LayoutSpec::RowBcastFlat()) {
    return false;
  }
  if (src_layout == LayoutSpec::FractalNZ()) {
    return dst_layout == LayoutSpec::NZ() &&
           ContainsLayout(src_node.admissible_layouts, LayoutSpec::FractalNZ()) &&
           IsNzCompatibleShape(src_node.shape, src_node.dtype);
  }
  const bool is_col_vec = src_node.shape.size() == 2 && src_node.shape[1] == 1 && src_node.shape[0] > 1;
  if (src_layout == LayoutSpec::ColMajorFlat()) {
    if (dst_layout == LayoutSpec::DN()) {
      return dst_node.shape.size() == 2 && IsColMajor32BAligned(src_node.shape, src_node.dtype);
    }
    if (dst_layout == LayoutSpec::ND()) {
      return is_col_vec;
    }
    return false;
  }
  if (src_layout == LayoutSpec::RowMajorFlat()) {
    return dst_layout == LayoutSpec::ND() && !is_col_vec;
  }
  return false;
}

bool LayoutCostModel::IsValidMoveConfig(const ValueNode& src_node, const ValueNode& dst_node,
                                        const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const {
  if (!src_layout.IsTile() || !dst_layout.IsTile()) {
    return false;
  }
  if (src_node.dtype.IsPackedFp4() || dst_node.dtype.IsPackedFp4()) {
    // Packed moves can change the carrier shape as well as the layout. The
    // original descriptor pair was validated when the IR call was built.
    if (src_layout != src_node.current_layout || dst_layout != dst_node.current_layout) return false;
    if (src_node.memory_space == MemorySpace::Vec && dst_node.memory_space == MemorySpace::Vec) {
      return src_layout == LayoutSpec::RowMajorFlat() && dst_layout == LayoutSpec::FractalNZ();
    }
    return true;
  }
  if (src_node.memory_space == MemorySpace::Vec && dst_node.memory_space == MemorySpace::Vec &&
      IsFlatVecTransposePair(src_layout, dst_layout)) {
    // Preserve an explicit relayout. New repacking is handled as a separate
    // materialization so the solver can account for its workspace.
    return src_layout == src_node.current_layout && dst_layout == dst_node.current_layout &&
           src_node.dtype == dst_node.dtype && src_node.shape == dst_node.shape &&
           dst_node.has_dense_vec_span && SupportsVecTileConversion(src_node, src_layout, dst_layout);
  }
  if (src_node.memory_space == MemorySpace::Mat && dst_node.memory_space == MemorySpace::Left) {
    return src_layout == src_node.current_layout && dst_layout == dst_node.current_layout;
  }
  if (src_node.memory_space == MemorySpace::Mat && dst_node.memory_space == MemorySpace::Right) {
    return src_layout == src_node.current_layout && dst_layout == dst_node.current_layout;
  }
  return true;
}

double LayoutCostModel::EffectiveMte2LoadMultiplier(double source_bytes, const ValueNode& dst_node,
                                                    double per_run_inv) const {
  if (per_run_inv <= 1.0 || mte2_l2_reread_discount >= 1.0) {
    return 1.0;
  }
  const double tile_b = static_cast<double>(dst_node.ByteSize());
  const double load_b = per_run_inv * tile_b;
  if (load_b <= source_bytes || load_b <= 0.0) {
    return 1.0;
  }
  return (source_bytes + mte2_l2_reread_discount * (load_b - source_bytes)) / load_b;
}

double LayoutCostModel::EstimateLoadCost(const ValueNode& src_node, const ValueNode& dst_node,
                                         const LayoutSpec& src_layout,
                                         const LayoutSpec& /*dst_layout*/) const {
  const double total_bytes = static_cast<double>(dst_node.ByteSize());
  if (dst_node.memory_space == MemorySpace::Mat) {
    if (src_layout == LayoutSpec::NZ()) {
      return total_bytes / mte2_nz2nz_bytes_per_cycle;
    }
    const int64_t rows = dst_node.shape.size() >= 2 ? dst_node.shape[dst_node.shape.size() - 2] : 1;
    const int64_t cols = !dst_node.shape.empty() ? dst_node.shape.back() : 1;
    const int64_t parent_cols = !src_node.shape.empty() ? src_node.shape.back() : cols;
    const int bpe = DTypeByteWidth(dst_node.dtype);
    const int64_t row_bytes = cols * bpe;
    const double gran_b = static_cast<double>(std::max<int64_t>(1, GmAccessGranularityBytes()));
    const bool strided = cols < parent_cols;
    const double nd_bw = strided
                             ? mte2_nd2nz_bytes_per_cycle
                             : std::max(mte2_nd2nz_bytes_per_cycle, mte2_nd2nz_contiguous_bytes_per_cycle);
    const int c0 = C0Elements(dst_node.dtype);
    const int64_t col_blocks = std::max<int64_t>(1, (cols + c0 - 1) / c0);
    const double sub512_penalty =
        strided
            ? static_cast<double>(rows) * std::max(0.0, (gran_b - static_cast<double>(row_bytes)) / gran_b) *
                  mte2_nd2nz_sub512b_penalty_cycles
            : 0.0;
    return total_bytes / nd_bw + static_cast<double>(rows * col_blocks) * mte2_nd2nz_row_penalty_cycles +
           sub512_penalty;
  }
  return EstimateVecGmTransferCost(*this, dst_node, src_node, src_layout, mte2_vec_bytes_per_cycle,
                                   mte2_vec_strided_bytes_per_cycle);
}

double LayoutCostModel::EstimateMoveCost(const ValueNode& src_node, const ValueNode& dst_node,
                                         const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const {
  const double total_bytes = static_cast<double>(dst_node.ByteSize());
  if (src_node.memory_space == MemorySpace::Vec && dst_node.memory_space == MemorySpace::Vec) {
    const bool transpose = IsFlatVecTransposePair(src_layout, dst_layout);
    return (transpose ? vec_transpose_fixed_cycles : 0.0) +
           (transpose ? 2.0 : 1.0) * total_bytes / onchip_relayout_bytes_per_cycle;
  }
  double bw = mte1_l1_to_l0_bytes_per_cycle;
  if (dst_node.memory_space == MemorySpace::Left) {
    bw = mte1_l0a_bytes_per_cycle;
  } else if (dst_node.memory_space == MemorySpace::Right) {
    bw = mte1_l0b_bytes_per_cycle;
  }
  const double base = total_bytes / bw;
  if (src_layout != dst_layout) {
    return base + mte1_transpose_penalty_cycles;
  }
  return base;
}

double LayoutCostModel::EstimateMatmulCost(const ValueNode& lhs_node, const ValueNode& rhs_node,
                                           const ValueNode& /*out_node*/) const {
  const int64_t m_dim = lhs_node.shape[lhs_node.shape.size() - 2];
  const int64_t k_dim = lhs_node.shape.back();
  const int64_t n_dim = rhs_node.shape.back();
  const int bpe = DTypeByteWidth(lhs_node.dtype);
  const int fp32_passes = mad_fp32_passes.has_value() ? *mad_fp32_passes : (TargetArch() == "a5" ? 8 : 2);
  const int cpr = (bpe == 4) ? std::max(1, fp32_passes) : std::max(1, bpe / 2);
  const int k_fractal = std::max(1, 32 / bpe);
  const int64_t fractals = ((m_dim + 15) / 16) * ((n_dim + 15) / 16) * ((k_dim + k_fractal - 1) / k_fractal);
  return cube_issue_overhead_cycles +
         static_cast<double>(cpr) * static_cast<double>(fractals) * (4096.0 / cube_macs_per_cycle);
}

double LayoutCostModel::EstimateStoreCost(const ValueNode& src_node, const ValueNode& dst_node,
                                          const LayoutSpec& dst_layout,
                                          const std::optional<LayoutSpec>& /*src_layout*/) const {
  const double total_bytes = static_cast<double>(src_node.ByteSize());
  if (src_node.memory_space == MemorySpace::Vec) {
    return EstimateVecGmTransferCost(*this, src_node, dst_node, dst_layout, mte3_vec_bytes_per_cycle,
                                     mte3_vec_strided_bytes_per_cycle);
  }
  const double bw = (dst_layout == LayoutSpec::NZ()) ? fixpipe_acc_to_nz_bytes_per_cycle
                                                     : fixpipe_acc_to_nd_bytes_per_cycle;
  const int64_t m_rows = src_node.shape.size() >= 2 ? src_node.shape[src_node.shape.size() - 2] : 1;
  const int64_t n_cols = !src_node.shape.empty() ? src_node.shape.back() : 1;
  const int bpe_c = std::max(1, DTypeByteWidth(src_node.dtype));
  const int64_t n0 = std::max<int64_t>(1, 32 / bpe_c);
  const int64_t n1 = std::max<int64_t>(1, (n_cols + n0 - 1) / n0);
  int64_t odd_n1 = n1;
  while ((odd_n1 & 1) == 0) {
    odd_n1 >>= 1;
  }
  const double odd_penalty =
      static_cast<double>(m_rows) * fixpipe_odd_n1_penalty_cycles * static_cast<double>(odd_n1 - 1);
  return fixpipe_drain_fixed_cycles + (total_bytes / bw) + odd_penalty;
}

std::optional<VecConversionStorage> LayoutCostModel::GetVecConversionStorage(
    const ValueNode& val_node, const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const {
  if (val_node.memory_space != MemorySpace::Vec || src_layout == dst_layout || val_node.dtype.IsPackedFp4()) {
    return std::nullopt;
  }
  if (!src_layout.IsTile() || !dst_layout.IsTile()) {
    return std::nullopt;
  }
  // A2/A3 non-Mat TMOV requires identical physical shapes. A strided subview
  // retains its backing pitch in PTOAS, while the writer allocates a compact
  // conversion destination.
  if (val_node.is_strided_subview && TargetArch() == "a2a3") {
    return std::nullopt;
  }
  if (src_layout == LayoutSpec::FractalNZ() || dst_layout == LayoutSpec::FractalNZ()) {
    return std::nullopt;
  }
  if (IsFlatVecTransposePair(src_layout, dst_layout)) {
    // Both contiguous axes must satisfy 32-byte alignment. The conversion
    // emits a TTRANS destination plus separate source-shaped scratch.
    const int bytes = DTypeByteWidthSafe(val_node.dtype);
    if (!SupportsFlatVecTransposeDtype(val_node.dtype) || !val_node.has_dense_vec_span ||
        val_node.is_strided_subview || (src_layout == LayoutSpec::ColMajorFlat() && val_node.is_vec_slice) ||
        !IsColMajor32BAligned(val_node.shape, val_node.dtype) || val_node.shape[1] <= 1 ||
        val_node.shape[1] % (32 / bytes) != 0) {
      return std::nullopt;
    }
    const auto tile_bytes = static_cast<uint64_t>(val_node.ByteSize());
    return VecConversionStorage{tile_bytes, tile_bytes};
  }
  if (dst_layout == LayoutSpec::RowBcastFlat() && src_layout == LayoutSpec::ColMajorFlat() &&
      val_node.shape.size() == 2 && val_node.shape[1] == 1 && val_node.shape[0] > 1 &&
      ContainsLayout(val_node.admissible_layouts, LayoutSpec::RowBcastFlat())) {
    return VecConversionStorage{static_cast<uint64_t>(val_node.shape[0]) * 32, 0};
  }
  return std::nullopt;
}

bool LayoutCostModel::SupportsVecTileConversion(const ValueNode& val_node, const LayoutSpec& src_layout,
                                                const LayoutSpec& dst_layout) const {
  const auto storage = GetVecConversionStorage(val_node, src_layout, dst_layout);
  if (!storage ||
      std::find(val_node.blocked_vec_conversions.begin(), val_node.blocked_vec_conversions.end(),
                std::make_pair(src_layout, dst_layout)) != val_node.blocked_vec_conversions.end()) {
    return false;
  }
  if (!val_node.vec_conversion_headroom) return true;
  const auto headroom = *val_node.vec_conversion_headroom;
  return storage->destination_bytes <= headroom &&
         storage->scratch_bytes <= headroom - storage->destination_bytes;
}

double LayoutCostModel::EstimateConversionCost(const ValueNode& val_node, const LayoutSpec& src_layout,
                                               const LayoutSpec& dst_layout) const {
  if (src_layout == dst_layout) {
    return 0.0;
  }
  const double total_bytes = static_cast<double>(val_node.ByteSize());
  if (val_node.memory_space == MemorySpace::DDR) {
    return gm_pack_fixed_cycles + (total_bytes / gm_pack_bytes_per_cycle);
  }
  if (val_node.memory_space == MemorySpace::Vec) {
    if (!SupportsVecTileConversion(val_node, src_layout, dst_layout)) {
      return vec_unsupported_conversion_cycles;
    }
    if (dst_layout == LayoutSpec::RowBcastFlat()) {
      const int64_t rows = val_node.shape[0];
      const int c0 = C0Elements(val_node.dtype);
      return vec_row_expand_issue_cycles + static_cast<double>(rows * c0) / vec_elements_per_cycle +
             EstimateVbrcbCost(rows);
    }
    // TTRANS reads the original tile and writes its transposed copy. The
    // companion transpose view simply reinterprets the same allocation.
    return vec_transpose_fixed_cycles + (2.0 * total_bytes / onchip_relayout_bytes_per_cycle);
  }
  return vec_unsupported_conversion_cycles;
}

size_t VecMemoryRegion::EffectiveLastUse(size_t definition, size_t last_use) const {
  auto it = std::upper_bound(loops.begin(), loops.end(), last_use,
                             [](size_t use, const LoopScope& loop) { return use < loop.begin; });
  if (it == loops.begin()) return last_use;
  const LoopScope* loop = &*std::prev(it);
  while (definition < loop->begin) {
    last_use = std::max(last_use, loop->end);
    if (!loop->parent) break;
    loop = &loops[*loop->parent];
  }
  return last_use;
}

void LayoutProblemInstance::AddValue(ValueNode node) {
  auto it = values.find(node.name);
  if (it == values.end()) {
    value_order.push_back(node.name);
    values.emplace(node.name, std::move(node));
  } else {
    it->second = std::move(node);
  }
}

const ValueNode* LayoutProblemInstance::FindValue(const std::string& name) const {
  auto it = values.find(name);
  return it == values.end() ? nullptr : &it->second;
}

double LayoutProblemInstance::EstimateConversionCost(const ValueNode& val_node, const LayoutSpec& src_layout,
                                                     const LayoutSpec& dst_layout) const {
  if (src_layout == dst_layout) {
    return 0.0;
  }
  if (val_node.dtype.IsPackedFp4()) return cost_model.vec_unsupported_conversion_cycles;
  const bool loader_pack = val_node.is_named_nz_input && val_node.memory_space == MemorySpace::DDR &&
                           !val_node.producer_op_id && !val_node.is_external_output &&
                           src_layout == val_node.current_layout && src_layout == LayoutSpec::ND() &&
                           dst_layout == LayoutSpec::NZ() &&
                           ContainsLayout(val_node.admissible_layouts, dst_layout);
  if (loader_pack && cost_model.steady_state && val_node.is_reusable_weight) {
    return 0.0;
  }
  auto val_it = custom_conversion_costs.find(val_node.name);
  if (val_it != custom_conversion_costs.end()) {
    auto pair_it = val_it->second.find(std::make_pair(src_layout, dst_layout));
    if (pair_it != val_it->second.end()) {
      // Measured costs override prices only for materializable conversions.
      if (val_node.memory_space == MemorySpace::Vec &&
          (forbid_unmaterializable_gm_conversions ||
           (val_node.is_strided_subview && cost_model.TargetArch() == "a2a3") ||
           val_node.vec_conversion_headroom.has_value()) &&
          !cost_model.SupportsVecTileConversion(val_node, src_layout, dst_layout)) {
        return cost_model.vec_unsupported_conversion_cycles;
      }
      return pair_it->second;
    }
  }
  if (forbid_unmaterializable_gm_conversions && val_node.memory_space == MemorySpace::DDR &&
      val_node.producer_op_id.has_value() &&
      !(val_node.is_external_output && dst_layout == val_node.current_layout)) {
    return cost_model.vec_unsupported_conversion_cycles;
  }
  if (loader_pack && val_node.shape.size() >= 2) {
    // With one C0 column block, NZ blocking only splits the row axis of each
    // dense matrix. The loader can view the same bytes, including in a bank.
    const auto rows = val_node.shape[val_node.shape.size() - 2];
    const auto c0 = TryC0Elements(val_node.dtype);
    if (c0 && val_node.shape.back() == *c0 && rows > 0 && rows % 16 == 0) return 0.0;
  }
  return cost_model.EstimateConversionCost(val_node, src_layout, dst_layout);
}

double OperatorCostMultiplier(const LayoutProblemInstance& instance, const OperationNode& op) {
  double load_scale = 1.0;
  if (op.op_name == "tile.load" && !op.inputs.empty()) {
    const auto* src_val = instance.FindValue(op.inputs[0]);
    const auto* dst_val = instance.FindValue(op.output);
    if (src_val != nullptr && dst_val != nullptr && dst_val->memory_space == MemorySpace::Mat) {
      const double per_run_inv = static_cast<double>(op.invocations) /
                                 static_cast<double>(std::max<int64_t>(1, instance.cost_model.expected_uses));
      const double source_bytes =
          op.projected_mte2_source_bytes.value_or(static_cast<double>(src_val->ByteSize()));
      load_scale = instance.cost_model.EffectiveMte2LoadMultiplier(source_bytes, *dst_val, per_run_inv);
    }
  }
  return static_cast<double>(op.invocations) * load_scale;
}

bool CanShareOnChipConversion(int64_t source_invocations, int64_t consumer_invocations) {
  if (consumer_invocations == 0) return true;
  return source_invocations > 0 && consumer_invocations >= source_invocations &&
         (consumer_invocations == std::numeric_limits<int64_t>::max() ||
          consumer_invocations % source_invocations == 0);
}

std::optional<int64_t> OnChipConversionInvocations(
    const ValueNode& value, const std::unordered_map<std::string, const OperationNode*>& ops_by_id) {
  if (value.memory_space == MemorySpace::DDR) return std::nullopt;
  if (value.conversion_invocations) return value.conversion_invocations;
  if (!value.producer_op_id) return std::nullopt;
  const auto prod_it = ops_by_id.find(*value.producer_op_id);
  if (prod_it == ops_by_id.end()) {
    throw pypto::ValueError("Unknown producer_op_id '" + *value.producer_op_id + "' for value '" +
                            value.name + "'");
  }
  return prod_it->second->invocations;
}

double ConversionMultiplier(const LayoutProblemInstance& instance, const ValueNode& value,
                            const std::unordered_map<std::string, const OperationNode*>& ops_by_id,
                            const std::vector<int64_t>& conversion_invocations) {
  if (value.memory_space == MemorySpace::DDR && value.is_reusable_weight) return 1.0;
  const auto source = OnChipConversionInvocations(value, ops_by_id);
  if (!source) return static_cast<double>(instance.cost_model.expected_uses);
  for (int64_t count : conversion_invocations) {
    if (!CanShareOnChipConversion(*source, count)) {
      throw pypto::ValueError("On-chip conversion for '" + value.name +
                              "' has differing producer/consumer execution counts; producer=" +
                              std::to_string(*source) + ", consumer=" + std::to_string(count));
    }
  }
  return static_cast<double>(*source);
}

static CostBreakdown EvaluateAssignmentImpl(const LayoutProblemInstance& instance,
                                            const LayoutAssignment& assignment,
                                            bool allow_incompatible_conversions) {
  const auto& model = instance.cost_model;
  model.ValidateUseHorizon();
  std::unordered_map<std::string, double> per_op_costs;
  double op_cost_total = 0.0;

  std::unordered_map<std::string, LayoutSpec> produced_layout;
  for (const auto& [name, node] : instance.values) {
    if (!node.producer_op_id.has_value()) {
      produced_layout.emplace(name, node.current_layout);
    }
  }

  std::unordered_map<std::string, std::map<LayoutSpec, std::set<int64_t>>> consumer_invocations;
  std::unordered_map<std::string, OperatorConfig> resolved_configs;
  std::unordered_map<std::string, const OperationNode*> ops_by_id;
  ops_by_id.reserve(instance.operations.size());
  for (const auto& op : instance.operations) ops_by_id[op.op_id] = &op;

  for (const auto& op : instance.operations) {
    if (op.configs.empty()) {
      throw pypto::ValueError("Operation '" + op.op_id + "' has no legal configurations");
    }
    const OperatorConfig* chosen_cfg = nullptr;
    auto cfg_it = assignment.op_configs.find(op.op_id);
    if (cfg_it != assignment.op_configs.end()) {
      auto match_it = std::find(op.configs.begin(), op.configs.end(), cfg_it->second);
      if (match_it == op.configs.end()) {
        throw pypto::ValueError("Configuration for '" + op.op_id + "' is not in its legal catalog");
      }
      chosen_cfg = &(*match_it);
    } else {
      const auto* out_node = instance.FindValue(op.output);
      const LayoutSpec out_target = [&]() {
        auto it = assignment.value_layouts.find(op.output);
        if (it != assignment.value_layouts.end()) return it->second;
        return out_node ? out_node->current_layout : LayoutSpec::ND();
      }();
      std::vector<LayoutSpec> in_targets;
      in_targets.reserve(op.inputs.size());
      for (const auto& inp : op.inputs) {
        auto it = assignment.value_layouts.find(inp);
        if (it != assignment.value_layouts.end()) {
          in_targets.push_back(it->second);
        } else {
          const auto* inp_node = instance.FindValue(inp);
          in_targets.push_back(inp_node ? inp_node->current_layout : LayoutSpec::ND());
        }
      }
      std::vector<const OperatorConfig*> matching;
      for (const auto& c : op.configs) {
        if (c.input_layouts == in_targets && c.output_layout == out_target) {
          matching.push_back(&c);
        }
      }
      if (matching.empty()) {
        for (const auto& c : op.configs) {
          if (c.output_layout == out_target) {
            matching.push_back(&c);
          }
        }
      }
      if (matching.empty()) {
        for (const auto& c : op.configs) {
          matching.push_back(&c);
        }
      }
      chosen_cfg = *std::min_element(matching.begin(), matching.end(),
                                     [](const OperatorConfig* a, const OperatorConfig* b) {
                                       return a->op_cost_cycles < b->op_cost_cycles;
                                     });
    }

    resolved_configs[op.op_id] = *chosen_cfg;
    const double cost = OperatorCostMultiplier(instance, op) * chosen_cfg->op_cost_cycles;
    per_op_costs[op.op_id] = cost;
    op_cost_total += cost;
    produced_layout[op.output] = chosen_cfg->output_layout;

    for (size_t i = 0; i < op.inputs.size(); ++i) {
      const std::string& inp_name = op.inputs[i];
      const LayoutSpec& req_layout = chosen_cfg->input_layouts[i];
      consumer_invocations[inp_name][req_layout].insert(op.invocations);
    }
  }

  if (!model.allow_external_output_relayout) {
    for (const auto& [val_name, val_node] : instance.values) {
      if (val_node.is_external_output) {
        consumer_invocations[val_name][val_node.current_layout].insert(model.expected_uses);
      }
    }
  }

  std::unordered_map<std::string, double> per_value_conv;
  double conv_cost_total = 0.0;
  for (const auto& [val_name, demands] : consumer_invocations) {
    const auto* val_node = instance.FindValue(val_name);
    if (val_node == nullptr) continue;
    if (val_node->producer_op_id && ops_by_id.count(*val_node->producer_op_id) == 0) {
      throw pypto::ValueError("Unknown producer_op_id '" + *val_node->producer_op_id + "' for value '" +
                              val_name + "'");
    }
    const auto prod_it = produced_layout.find(val_name);
    const LayoutSpec& src_lay = prod_it != produced_layout.end() ? prod_it->second : val_node->current_layout;
    const auto source_invocations = OnChipConversionInvocations(*val_node, ops_by_id);
    double val_conv = 0.0;
    for (const auto& [dst_lay, invocations] : demands) {
      if (dst_lay == src_lay) continue;
      if (allow_incompatible_conversions && source_invocations &&
          std::any_of(invocations.begin(), invocations.end(),
                      [&](int64_t count) { return !CanShareOnChipConversion(*source_invocations, count); })) {
        val_conv += model.vec_unsupported_conversion_cycles;
        continue;
      }
      const double multiplier =
          ConversionMultiplier(instance, *val_node, ops_by_id, {invocations.begin(), invocations.end()});
      if (multiplier == 0 && val_node->memory_space == MemorySpace::Vec &&
          !model.SupportsVecTileConversion(*val_node, src_lay, dst_lay)) {
        val_conv += model.vec_unsupported_conversion_cycles;
      } else {
        val_conv += multiplier * instance.EstimateConversionCost(*val_node, src_lay, dst_lay);
      }
    }
    if (instance.forbid_unmaterializable_gm_conversions && !val_node->producer_op_id.has_value() &&
        val_node->memory_space == MemorySpace::DDR && demands.size() > 1) {
      val_conv += model.vec_unsupported_conversion_cycles;
    }
    per_value_conv[val_name] = val_conv;
    conv_cost_total += val_conv;
  }

  std::unordered_map<std::string, LayoutSpec> resolved_value_layouts = assignment.value_layouts;
  for (const auto& [name, layout] : produced_layout) {
    const auto* node = instance.FindValue(name);
    if (node != nullptr && node->producer_op_id.has_value()) {
      auto sup_it = assignment.value_layouts.find(name);
      if (sup_it != assignment.value_layouts.end() && sup_it->second != layout) {
        throw pypto::ValueError("Value layout for '" + name + "' contradicts its producer configuration");
      }
      resolved_value_layouts[name] = layout;
    } else {
      resolved_value_layouts.emplace(name, layout);
    }
  }

  LayoutAssignment full_assignment{std::move(resolved_value_layouts), std::move(resolved_configs)};
  return CostBreakdown{
      std::move(full_assignment), op_cost_total + conv_cost_total, op_cost_total, conv_cost_total,
      std::move(per_op_costs),    std::move(per_value_conv),
  };
}

CostBreakdown EvaluateAssignment(const LayoutProblemInstance& instance, const LayoutAssignment& assignment) {
  return EvaluateAssignmentImpl(instance, assignment, false);
}

CostBreakdown detail::EvaluateAssignmentForSearch(const LayoutProblemInstance& instance,
                                                  const LayoutAssignment& assignment) {
  return EvaluateAssignmentImpl(instance, assignment, true);
}

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto
