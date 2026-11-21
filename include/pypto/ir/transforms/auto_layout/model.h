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

#ifndef PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_MODEL_H_
#define PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/dtype.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace auto_layout {

/// Unified value layout descriptor covering both GM TensorLayout and on-chip TileLayoutDesc.
struct LayoutSpec {
  enum class Kind : uint8_t {
    kTensor = 0,
    kTile = 1,
  };

  Kind kind = Kind::kTensor;
  TensorLayout tensor_layout = TensorLayout::ND;
  TileLayout blayout = TileLayout::row_major;
  TileLayout slayout = TileLayout::none_box;
  int32_t fractal = 512;
  std::string tag;

  static LayoutSpec Tensor(TensorLayout layout) {
    LayoutSpec spec;
    spec.kind = Kind::kTensor;
    spec.tensor_layout = layout;
    return spec;
  }

  static LayoutSpec ND() { return Tensor(TensorLayout::ND); }
  static LayoutSpec DN() { return Tensor(TensorLayout::DN); }
  static LayoutSpec NZ() { return Tensor(TensorLayout::NZ); }

  static LayoutSpec Tile(TileLayout blayout, TileLayout slayout, int32_t fractal = 512,
                         std::string tag = "") {
    LayoutSpec spec;
    spec.kind = Kind::kTile;
    spec.blayout = blayout;
    spec.slayout = slayout;
    spec.fractal = fractal;
    spec.tag = std::move(tag);
    return spec;
  }

  static LayoutSpec RowMajorFlat() { return Tile(TileLayout::row_major, TileLayout::none_box, 512); }
  static LayoutSpec ColMajorFlat() { return Tile(TileLayout::col_major, TileLayout::none_box, 512); }
  static LayoutSpec RowBcastFlat() {
    return Tile(TileLayout::row_major, TileLayout::none_box, 512, "row_bcast");
  }
  static LayoutSpec FractalNZ() { return Tile(TileLayout::col_major, TileLayout::row_major, 512); }
  static LayoutSpec FractalZN() { return Tile(TileLayout::row_major, TileLayout::col_major, 512); }
  static LayoutSpec FractalAccNZ() { return Tile(TileLayout::col_major, TileLayout::row_major, 1024); }

  [[nodiscard]] bool IsTensor() const { return kind == Kind::kTensor; }
  [[nodiscard]] bool IsTile() const { return kind == Kind::kTile; }
  [[nodiscard]] std::string ShortName() const;

  bool operator==(const LayoutSpec& other) const;
  bool operator!=(const LayoutSpec& other) const { return !(*this == other); }
  bool operator<(const LayoutSpec& other) const;
};

struct LayoutSpecHash {
  size_t operator()(const LayoutSpec& spec) const noexcept;
};

struct LayoutPairHash {
  size_t operator()(const std::pair<LayoutSpec, LayoutSpec>& p) const noexcept;
};

/// Return the number of elements in one 32-byte fractal C0 line (`256 / dtype.GetBit()`).
/// Rejects sub-byte dtypes (such as BOOL or INT4) and non-divisible widths with `pypto::ValueError`.
int C0Elements(DataType dtype);

/// Non-throwing variant of `C0Elements`; returns `std::nullopt` for unsupported dtypes.
std::optional<int> TryC0Elements(DataType dtype);

/// Whole-byte element width (`1` for BOOL/byte dtypes, `0` for packed sub-byte nibbles).
int DTypeByteWidthSafe(DataType dtype);

/// Whole-byte element width; throws `pypto::ValueError` if `dtype` is sub-byte (< 8 bits storage).
int DTypeByteWidth(DataType dtype);

/// Whether two layouts differ only in the contiguous axis of a flat Vec tile.
bool IsFlatVecTransposePair(const LayoutSpec& source, const LayoutSpec& destination);

/// Check whether a logical 2D `[R, C]` or 3D `[B, R, C]` shape can be blocked as `TensorLayout::NZ`.
bool IsNzCompatibleShape(const std::vector<int64_t>& shape, DataType dtype);

/// A GM tensor (`DDR`) or on-chip tile (`Mat`/`Vec`/`Left`/`Right`/`Acc`) value in a layout problem.
struct ValueNode {
  std::string name;
  std::vector<int64_t> shape;
  DataType dtype = DataType::FP16;
  MemorySpace memory_space = MemorySpace::DDR;
  LayoutSpec current_layout = LayoutSpec::ND();
  std::vector<LayoutSpec> admissible_layouts;
  std::optional<std::string> producer_op_id;
  bool is_reusable_weight = false;
  bool is_external_output = false;
  /// A Vec subview whose source pitch requires a larger physical PTO tile shape.
  bool is_strided_subview = false;
  // Free Vec bytes at the producer, where shared conversions are materialized.
  std::optional<uint64_t> vec_conversion_headroom;
  // TTRANS reads a full dense valid rectangle. A CM-to-RM conversion first
  // takes a transpose view, which the backend only supports on a whole tile.
  bool has_dense_vec_span = true;
  bool is_vec_slice = false;
  /// Activation count at the writer's insertion site when it differs from the defining op's count.
  std::optional<int64_t> conversion_invocations;
  // Conversion pairs penalized in a capacity-repair solve.
  std::vector<std::pair<LayoutSpec, LayoutSpec>> blocked_vec_conversions;
  // The loader explicitly offers to supply this read-only entry input in NZ.
  bool is_named_nz_input = false;
  // A write through a tile or any of its aliases prevents cached copies.
  bool has_mutable_vec_backing = false;

  [[nodiscard]] int64_t NumElements() const;
  [[nodiscard]] int64_t ByteSize() const;
};

struct VecConversionStorage {
  uint64_t destination_bytes;
  uint64_t scratch_bytes;
};

/// One legal input/output layout configuration for an IR operation.
struct OperatorConfig {
  std::string op_id;
  std::string op_name;
  std::vector<LayoutSpec> input_layouts;
  LayoutSpec output_layout = LayoutSpec::ND();
  double op_cost_cycles = 0.0;
  std::string notes;
  // A read of mutable storage can convert privately at the operation. The
  // indexed input layout still describes the original backing, and this pair
  // describes the operand index and the layout read by the instruction.
  std::vector<std::pair<size_t, LayoutSpec>> local_vec_conversions = {};
  // A load may need its GM layout's native Vec representation before copying
  // into the layout used by the mutable result and its aliases.
  std::optional<LayoutSpec> local_vec_load_layout = std::nullopt;

  [[nodiscard]] LayoutSpec ExecutionInputLayout(size_t index) const;

  bool operator==(const OperatorConfig& other) const;
  bool operator!=(const OperatorConfig& other) const { return !(*this == other); }
};

// The loop transfers themselves cost no cycles. Their IR uses demand the
// carried layout, and each result has its own conversion insertion site.
inline constexpr char kVecLoopInitOp[] = "control.vec_loop_init";
inline constexpr char kVecLoopYieldOp[] = "control.vec_loop_yield";

/// An operation node in a layout problem instance.
struct OperationNode {
  std::string op_id;
  std::string op_name;
  std::vector<std::string> inputs;
  std::string output;
  int64_t invocations = 1;
  std::vector<OperatorConfig> configs;
  // Per forward, across the projected matrices and devices that can read them.
  std::optional<double> projected_mte2_source_bytes = std::nullopt;
};

/// Calibratable analytical cost model for GM, L1 (`Mat`), UB (`Vec`), and L0 (`Left`/`Right`/`Acc`).
struct LayoutCostModel {
  int64_t expected_uses = 1;
  // Compile-time topology for named HOST inputs. It also proves that a
  // whole matrix forwarded by HOST can use one resident device allocation.
  std::optional<int64_t> host_world_size;
  std::optional<backend::BackendType> backend_type;
  std::string resolved_arch_hint = "a2a3";

  double mte2_nz2nz_bytes_per_cycle = 123.0;
  double mte2_nd2nz_bytes_per_cycle = 77.0;
  double mte2_nd2nz_contiguous_bytes_per_cycle = 102.0;
  double mte2_nd2nz_row_penalty_cycles = 0.0;
  double mte2_nd2nz_sub512b_penalty_cycles = 2.5;
  double mte2_l2_reread_discount = 0.30;
  int64_t l2_cache_line_bytes = 512;
  double mte2_vec_bytes_per_cycle = 192.0;
  double mte2_vec_strided_bytes_per_cycle = 96.0;
  double mte3_vec_bytes_per_cycle = 192.0;
  double mte3_vec_strided_bytes_per_cycle = 96.0;
  double vec_strided_burst_penalty_cycles = 3.0;
  double mte1_l1_to_l0_bytes_per_cycle = 129.7;
  double mte1_l0a_bytes_per_cycle = 129.7;
  double mte1_l0b_bytes_per_cycle = 85.4;
  double mte1_transpose_penalty_cycles = 8.0;
  double cube_macs_per_cycle = 4096.0;
  double cube_issue_overhead_cycles = 21.0;
  std::optional<int> mad_fp32_passes;
  double vec_elements_per_cycle = 128.0;
  double vec_reshape_metadata_cycles = 1.0;
  double vec_transpose_fixed_cycles = 0.0;
  double vec_vbrcb_fixed_cycles = 4.0;
  double vec_vbrcb_cycles_per_row = 0.5;
  double vec_row_expand_issue_cycles = 4.0;
  double vec_unsupported_conversion_cycles = 1e12;
  double fixpipe_acc_to_nd_bytes_per_cycle = 118.0;
  double fixpipe_acc_to_nz_bytes_per_cycle = 122.0;
  double fixpipe_drain_fixed_cycles = 0.0;
  double fixpipe_odd_n1_penalty_cycles = 2.6;
  double gm_pack_bytes_per_cycle = 64.0;
  double gm_pack_fixed_cycles = 120.0;
  double onchip_relayout_bytes_per_cycle = 256.0;
  bool allow_external_input_relayout = false;
  bool allow_external_output_relayout = false;
  bool reselect_internal_layouts = false;
  // Score one forward after the loader prepares named reusable NZ inputs.
  bool steady_state = false;

  void ValidateUseHorizon() const;
  [[nodiscard]] std::string TargetArch() const;
  [[nodiscard]] int64_t GmAccessGranularityBytes() const;
  [[nodiscard]] bool SupportsWideNzMatLoad() const;

  [[nodiscard]] bool SupportsAccNzStore(const ValueNode& src_node, const ValueNode& dst_node) const;
  [[nodiscard]] double EstimateVbrcbCost(int64_t rows) const;

  [[nodiscard]] bool IsValidLoadConfig(const ValueNode& src_node, const ValueNode& dst_node,
                                       const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const;
  [[nodiscard]] bool IsValidVecStoreConfig(const ValueNode& src_node, const ValueNode& dst_node,
                                           const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const;
  [[nodiscard]] bool IsValidMoveConfig(const ValueNode& src_node, const ValueNode& dst_node,
                                       const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const;

  [[nodiscard]] double EffectiveMte2LoadMultiplier(double source_bytes, const ValueNode& dst_node,
                                                   double per_run_inv) const;
  [[nodiscard]] double EstimateLoadCost(const ValueNode& src_node, const ValueNode& dst_node,
                                        const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const;
  [[nodiscard]] double EstimateMoveCost(const ValueNode& src_node, const ValueNode& dst_node,
                                        const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const;
  [[nodiscard]] double EstimateMatmulCost(const ValueNode& lhs_node, const ValueNode& rhs_node,
                                          const ValueNode& out_node) const;
  [[nodiscard]] double EstimateStoreCost(const ValueNode& src_node, const ValueNode& dst_node,
                                         const LayoutSpec& dst_layout,
                                         const std::optional<LayoutSpec>& src_layout = std::nullopt) const;
  [[nodiscard]] bool SupportsVecTileConversion(const ValueNode& val_node, const LayoutSpec& src_layout,
                                               const LayoutSpec& dst_layout) const;
  /// Storage sizes for a materializable layout pair, before capacity constraints.
  [[nodiscard]] std::optional<VecConversionStorage> GetVecConversionStorage(
      const ValueNode& val_node, const LayoutSpec& src_layout, const LayoutSpec& dst_layout) const;
  [[nodiscard]] double EstimateConversionCost(const ValueNode& val_node, const LayoutSpec& src_layout,
                                              const LayoutSpec& dst_layout) const;
};

using CustomConversionCostMap =
    std::unordered_map<std::string,
                       std::unordered_map<std::pair<LayoutSpec, LayoutSpec>, double, LayoutPairHash>>;

/// Vec pressure events for one kernel call site, indexed by operation and value.
struct VecMemoryRegion {
  struct LoopScope {
    size_t begin;
    size_t end;
    std::optional<size_t> parent;
  };

  size_t first_operation;
  uint64_t capacity;
  std::vector<size_t> operation_positions;
  std::vector<int64_t> original_bytes;
  std::unordered_map<std::string, size_t> copy_positions;
  std::unordered_map<std::string, size_t> last_uses;
  std::unordered_map<std::string, std::vector<std::string>> backings;
  std::vector<LoopScope> loops;  // In increasing order of body entry.

  /// A value defined before a loop and read inside it is live through the backedge.
  [[nodiscard]] size_t EffectiveLastUse(size_t definition, size_t last_use) const;
};

/// Complete layout optimization problem instance.
struct LayoutProblemInstance {
  std::unordered_map<std::string, ValueNode> values;
  std::vector<std::string> value_order;
  std::vector<OperationNode> operations;
  LayoutCostModel cost_model;
  /// Optional per-value `(src, dst) -> cost` overrides, in modeled cycles.
  CustomConversionCostMap custom_conversion_costs;
  /// When true (set by `ExtractLayoutInstance`), penalize unmaterializable GM internal conversions and
  /// multi-representation GM inputs so the solver avoids unrewritable assignments.
  bool forbid_unmaterializable_gm_conversions = false;
  std::vector<VecMemoryRegion> vec_memory_regions;

  void AddValue(ValueNode node);
  [[nodiscard]] const ValueNode* FindValue(const std::string& name) const;
  [[nodiscard]] double EstimateConversionCost(const ValueNode& val_node, const LayoutSpec& src_layout,
                                              const LayoutSpec& dst_layout) const;
};

/// Candidate assignment of layouts to values and configurations to operations.
struct LayoutAssignment {
  std::unordered_map<std::string, LayoutSpec> value_layouts;
  std::unordered_map<std::string, OperatorConfig> op_configs;
};

/// Detailed cost evaluation of one `LayoutAssignment` on a `LayoutProblemInstance`.
struct CostBreakdown {
  LayoutAssignment assignment;
  double total_cost_cycles = 0.0;
  double operator_cost_cycles = 0.0;
  double conversion_cost_cycles = 0.0;
  std::unordered_map<std::string, double> per_op_costs;
  std::unordered_map<std::string, double> per_value_conversion_costs;
};

/// Compute operator cost multiplier (invocations * MTE2 L2 reread discount).
double OperatorCostMultiplier(const LayoutProblemInstance& instance, const OperationNode& op);

/// Compute conversion cost multiplier for `value` given distinct consumer `conversion_invocations`.
double ConversionMultiplier(const LayoutProblemInstance& instance, const ValueNode& value,
                            const std::unordered_map<std::string, const OperationNode*>& ops_by_id,
                            const std::vector<int64_t>& conversion_invocations);

/// Whether a conversion can be placed at the source and shared with this consumer.
bool CanShareOnChipConversion(int64_t source_invocations, int64_t consumer_invocations);

/// Evaluate `assignment` under the shared-conversion hyperedge objective.
CostBreakdown EvaluateAssignment(const LayoutProblemInstance& instance, const LayoutAssignment& assignment);

/// Enumerate all dataflow-consistent, backend-legal layout assignments, sorted by total cost.
std::vector<CostBreakdown> EnumerateAssignments(const LayoutProblemInstance& instance,
                                                bool deduplicate = true);

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_AUTO_LAYOUT_MODEL_H_
