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

#include "pypto/ir/transforms/utils/split_axis_utils.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/utils/split_axis/internal.h"

namespace pypto {
namespace ir {
namespace split_axis {

namespace detail {

// Whether a split-axis extent is statically ODD, i.e. the two AIV lanes hold
// DIFFERENT extents (lane 0 = ceil, lane 1 = floor). A dynamic extent has no
// compile-time parity and is treated as even (the pre-existing floordiv
// behaviour); ShardSplitCode keeps such a boundary on the even split code.
bool IsOddSplitExtent(const ExprPtr& dim_size) {
  auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size);
  return ci != nullptr && (ci->value_ % 2) != 0;
}

// The axis @p after was halved along relative to @p before, or nullopt when the two
// are the same width. Exactly one axis may differ: this reads back a halving that
// already happened, it does not decide one.
std::optional<TileInfo> SplitInfoFromHalvedType(const TypePtr& before_type, const TypePtr& after_type) {
  auto before = std::dynamic_pointer_cast<const TileType>(before_type);
  auto after = std::dynamic_pointer_cast<const TileType>(after_type);
  if (!before || !after || before->shape_.size() != after->shape_.size()) return std::nullopt;
  for (size_t d = 0; d < before->shape_.size(); ++d) {
    if (structural_equal(before->shape_[d], after->shape_[d])) continue;
    return TileInfo{ComputeHalfDimSize(before->shape_[d]), static_cast<int>(d)};
  }
  return std::nullopt;
}

// Replace one axis of a TileType's valid_shape, preserving everything else about
// the type (layout, memref, memory space) — the localization is metadata-only.
TypePtr WithValidDim(const std::shared_ptr<const TileType>& tt, int dim, const ExprPtr& valid_dim) {
  TileView tv = tile_view_semantics::GetEffectiveTileView(*tt);
  INTERNAL_CHECK(dim >= 0 && dim < static_cast<int>(tv.valid_shape.size()))
      << "Internal error: split dim " << dim << " out of range for a rank-" << tv.valid_shape.size()
      << " valid_shape";
  tv.valid_shape[dim] = valid_dim;
  return std::make_shared<TileType>(tt->shape_, tt->dtype_, tt->memref_, std::move(tv), tt->memory_space_);
}

}  // namespace detail

using detail::IsOddSplitExtent;
using detail::SplitInfoFromHalvedType;
using detail::WithValidDim;

int SplitDimension(SplitMode mode) {
  INTERNAL_CHECK(mode == SplitMode::UpDown || mode == SplitMode::LeftRight)
      << "Internal error: SplitDimension expects UpDown or LeftRight, got SplitMode("
      << static_cast<int>(mode) << ")";
  return (mode == SplitMode::UpDown) ? 0 : 1;
}

bool IsReduceOnSplitAxis(const CallPtr& call, int split_dim) {
  // Submits carry a GlobalVar callee and no op_; reduce ops are always plain
  // Calls with a non-null op_, so this guard correctly skips Submits (no
  // SubmitPtr handling needed — see pass-submit-awareness.md).
  if (!call->op_) return false;

  // Row reductions collapse the last axis. Splitting on that axis
  // (SplitMode::LeftRight for a 2D tile) would leave each lane with a partial
  // reduction. Reduce ops always take the tile first, so a missing / non-Tile
  // arg0 can only be malformed IR — assume the 2D last axis.
  if (IsOp(call, "tile.row_sum") || IsOp(call, "tile.row_max") || IsOp(call, "tile.row_min") ||
      IsOp(call, "tile.row_prod") || IsOp(call, "tile.row_argmax") || IsOp(call, "tile.row_argmin")) {
    std::shared_ptr<const TileType> input_tile;
    if (!call->args_.empty() && call->args_[0]) {
      input_tile = std::dynamic_pointer_cast<const TileType>(call->args_[0]->GetType());
    }
    const int last_axis = input_tile ? static_cast<int>(input_tile->shape_.size()) - 1 : 1;
    return split_dim == last_axis;
  }
  // Column reductions collapse the first axis (axis 0). Splitting on that axis
  // (SplitMode::UpDown) would leave each lane with a partial reduction.
  if (IsOp(call, "tile.col_sum") || IsOp(call, "tile.col_max") || IsOp(call, "tile.col_min") ||
      IsOp(call, "tile.col_prod") || IsOp(call, "tile.col_argmax") || IsOp(call, "tile.col_argmin")) {
    return split_dim == 0;
  }
  return false;
}

TypePtr WithFullSplitAxisValid(const TypePtr& type, int split_dim) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || split_dim < 0 || split_dim >= static_cast<int>(tt->shape_.size())) return type;
  TileView tv = tile_view_semantics::GetEffectiveTileView(*tt);
  if (split_dim >= static_cast<int>(tv.valid_shape.size())) return type;
  if (AreExprsEqual(tv.valid_shape[split_dim], tt->shape_[split_dim])) return type;
  tv.valid_shape[split_dim] = tt->shape_[split_dim];
  return std::make_shared<TileType>(tt->shape_, tt->dtype_, tt->memref_, std::move(tv), tt->memory_space_);
}

namespace {

bool IsSingletonDim(const ExprPtr& dim_size) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size)) {
    return ci->value_ == 1;
  }
  return false;
}

}  // namespace

bool IsSingletonSplitAxis(const TileType& type, int split_dim) {
  return split_dim >= 0 && split_dim < static_cast<int>(type.shape_.size()) &&
         IsSingletonDim(type.shape_[split_dim]);
}

bool IsBroadcastOnSplitAxis(const TileType& operand, int result_split_dim, int result_rank) {
  const int axis = static_cast<int>(operand.shape_.size()) - result_rank + result_split_dim;
  return axis < 0 || IsSingletonSplitAxis(operand, axis);
}

namespace {

bool IsUnsupportedAutoSplitGenerator(const CallPtr& call) {
  return IsOp(call, "tile.ci") || IsOp(call, "tile.random");
}

// Ops allowed to keep a full-width tile operand under a halved result. Each
// rewrites its own operands in the generic halving block below, and each has a
// well-defined full-width-source meaning:
//   * tile.full / tile.create take a shape tuple, not a tile;
//   * tile.slice reads a per-lane window out of a larger source, so a full-width
//     src is the normal spelling (AdjustOffsets gives each lane its own half);
//   * tile.reshape / tile.reinterpret_view over a full-width source are diverted
//     into a full view + per-lane slice above, and the dynamic half extent that
//     cannot express that slice is rejected there rather than reaching here.
bool OperandsMayStayFullWidth(const CallPtr& call) {
  return IsOp(call, "tile.full") || IsOp(call, "tile.create") || IsOp(call, "tile.slice") ||
         IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view");
}

// A tile operand the generic halving path would carry into the halved call at
// full width, together with the operand's OWN index for the result's split axis.
struct FullWidthOperand {
  ExprPtr arg;
  size_t arg_index = 0;
  int split_dim = 0;
};

// Whether the operator declares that argument ``arg_index`` carries no
// lane-indexed data (hardware scratch), so leaving it full width is correct.
//
// The claim has to come from the REGISTRY, but only for the operands the
// operator's own type deduction cannot speak about -- see
// ``OpLaneInvariantArgSpec`` and ``DeducerPinsOperandExtent`` below.
std::optional<LaneInvariantArg> DeclaredLaneInvariantKind(const CallPtr& call, size_t arg_index) {
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return std::nullopt;
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op->name_)) return std::nullopt;
  return registry.GetEntry(op->name_).GetLaneInvariantArgKind(arg_index);
}

// An absolutely-indexed operand -- tile.gather's source table or tile.scatter's
// destination -- whose PRODUCER was partitioned, or an empty ``arg`` when there
// is none.
//
// Being declared lane-invariant only says the operand MAY stay full width; it is
// not a rewrite. Nothing stops the operand's producer from being halved on its
// own -- and a tile.load inside the region is halved, with the lane offset baked
// into it. The index values are absolute into the WHOLE operand, so lane 1 then
// addresses the wrong half and an index past that half is out of bounds.
//
// This is the ordinary path rather than a corner: tensor.gather lowers to
// tile.load + tile.gather (see op_conversion_registry.cpp), so the operand is
// routinely produced inside the region. Neither other gate sees it -- the
// operand is tracked, so the full-width check skips it, and the result type is
// deduced from the index operand (gather) or mirrors the destination (scatter),
// so the type-consistency check is satisfied either way. Reject it until the
// pass can keep such a producer at full width, or rebase the indices.
struct AbsoluteOperand {
  ExprPtr arg;
  int split_dim = 0;
  LaneInvariantArg kind = LaneInvariantArg::IndexAddressedSource;
};

AbsoluteOperand FindPartitionedAbsoluteOperand(
    const CallPtr& call, const std::unordered_map<const Var*, TileInfo>& tile_vars,
    const std::unordered_map<const Var*, VarPtr>& var_replacements) {
  for (size_t i = 0; i < call->args_.size(); ++i) {
    auto kind = DeclaredLaneInvariantKind(call, i);
    if (kind != LaneInvariantArg::IndexAddressedSource &&
        kind != LaneInvariantArg::AbsoluteIndexedDestination) {
      continue;
    }
    // Through OperandSplitInfo, so an INLINE tuple projection counts as partitioned.
    // `tile.gather(pair[0], indices, tmp)` is the case: the table halves with the
    // tuple while the indices stay absolute, and neither other gate sees it -- the
    // result shape comes from `indices`, so type consistency is satisfied either way.
    auto info = OperandSplitInfo(call->args_[i], tile_vars, var_replacements);
    if (!info) continue;
    return {call->args_[i], info->split_dim, *kind};
  }
  return {};
}

// Everything needed to ask the operator a follow-up question about the node the
// pass is about to emit: the argument list the halved call will carry (tracked
// operands already swapped for their halved replacements) and the halved result
// type those arguments must reproduce.
struct HalvedCallProbe {
  const std::vector<ExprPtr>* args = nullptr;
  std::shared_ptr<const TileType> expected;
};

// Whether the operator's own f_deduce_type PINS the extent of argument
// ``arg_index`` on ``arg_split_dim``.
//
// The caller has already established that the halved call type-checks with this
// operand at full width. That is only *evidence* the operand is legal there if
// the deducer would have objected to a wrong width -- so ask it: re-deduce with
// the operand DOUBLED on that axis.
//
//   throws, or the deduced result changes -> the deducer reads this operand's
//       extent, so its silence at full width is a positive answer and the
//       type-consistency gate has already vouched for the operand;
//   unchanged                             -> the deducer never looks, its
//       silence proves nothing, and only a registry declaration can say whether
//       full width is in contract.
//
// Doubling rather than halving is deliberate. A bounded-below scratch contract
// ("at least as large as the input", tile.sels' mask must COVER src) accepts
// both widths, and doubling correctly reports that as blind; halving would make
// such a deducer throw and be misread as a pin.
//
// A dynamic extent cannot be doubled into a comparable probe, so it is reported
// as unpinned -- the conservative direction, since the caller then falls back to
// requiring a declaration.
bool DeducerPinsOperandExtent(const CallPtr& call, const HalvedCallProbe& probe, size_t arg_index,
                              int arg_split_dim) {
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op || !probe.args || !probe.expected) return false;
  if (arg_index >= probe.args->size()) return false;
  if (!OpRegistry::GetInstance().IsRegistered(op->name_)) return false;

  const auto& arg = (*probe.args)[arg_index];
  auto arg_tt = std::dynamic_pointer_cast<const TileType>(arg->GetType());
  if (!arg_tt || arg_split_dim < 0 || arg_split_dim >= static_cast<int>(arg_tt->shape_.size())) return false;
  auto extent = std::dynamic_pointer_cast<const ConstInt>(arg_tt->shape_[arg_split_dim]);
  if (!extent) return false;

  std::vector<ExprPtr> widened_shape = arg_tt->shape_;
  widened_shape[arg_split_dim] =
      std::make_shared<ConstInt>(extent->value_ * 2, extent->dtype(), extent->span_);
  auto widened_type = std::make_shared<TileType>(std::move(widened_shape), arg_tt->dtype_, arg_tt->memref_,
                                                 std::nullopt, arg_tt->memory_space_);
  std::vector<ExprPtr> widened_args = *probe.args;
  widened_args[arg_index] = std::make_shared<Var>("__split_probe", widened_type, arg->span_);

  std::shared_ptr<const TileType> deduced;
  try {
    auto probe_call = OpRegistry::GetInstance().Create(op->name_, widened_args, call->kwargs_, call->span_);
    deduced = std::dynamic_pointer_cast<const TileType>(probe_call->GetType());
  } catch (const pypto::Error&) {
    return true;  // The deducer refuses the wrong width: it is reading the operand.
  }
  if (!deduced || deduced->shape_.size() != probe.expected->shape_.size()) return true;
  for (size_t d = 0; d < probe.expected->shape_.size(); ++d) {
    if (!structural_equal(deduced->shape_[d], probe.expected->shape_[d])) return true;
  }
  return false;  // Same answer for either width: the deducer is blind to this operand.
}

// The first tile operand the halved call would carry at full width along the
// split axis, or an empty ``arg`` when there is none.
//
// Halving only the result type is correct exactly when each tile operand was
// itself partitioned by its producer (so it is in ``tile_vars`` and the later
// Substitute swaps in the halved var), is broadcast along the split axis, or is
// lane-invariant by contract.
//
// Operands align to the result from the RIGHT -- the alignment BroadcastShapes
// gives an op like tile.add([M, N], [N]) -> [M, N]. Indexing an operand with the
// result's absolute split dim is therefore wrong as soon as the ranks differ:
// under UP_DOWN it reads the [N] bias's only axis as the split axis and rejects
// a legal shared bias, and under LEFT_RIGHT it finds that index out of range and
// skips the operand that genuinely does span the split axis. Map the axis onto
// the operand instead, and treat a missing leading axis as broadcast.
//
// ``probe`` decides what this function IS. With a probe it is a GATE, and an
// operand whose extent the deducer pins is skipped -- the type-consistency check
// already answered for it, so this heuristic never gets to overrule the
// operator itself. Without a probe it is only a DIAGNOSTIC refinement, naming
// the likely operand behind a type-consistency failure; a misjudged axis is
// then a wrong number in a message rather than a rejected program.
FullWidthOperand FindFullWidthOperand(const CallPtr& call, int result_split_dim, int result_rank,
                                      const std::unordered_map<const Var*, TileInfo>& tile_vars,
                                      const std::unordered_map<const Var*, VarPtr>& var_replacements,
                                      const HalvedCallProbe* probe = nullptr) {
  if (OperandsMayStayFullWidth(call)) return {};
  for (size_t i = 0; i < call->args_.size(); ++i) {
    const auto& arg = call->args_[i];
    auto arg_tt = std::dynamic_pointer_cast<const TileType>(arg->GetType());
    if (!arg_tt) continue;
    // Declared as carrying no lane-indexed data (hardware scratch, or a table
    // addressed by a separate index operand): a full-width operand is in
    // contract. FindPartitionedAbsoluteOperand covers the converse case.
    if (DeclaredLaneInvariantKind(call, i).has_value()) continue;
    const int arg_rank = static_cast<int>(arg_tt->shape_.size());
    const int arg_split_dim = arg_rank - (result_rank - result_split_dim);
    // Share the broadcast rule with explicit-region admission.
    if (IsBroadcastOnSplitAxis(*arg_tt, result_split_dim, result_rank)) continue;
    if (arg_split_dim >= arg_rank) continue;
    // Through OperandSplitInfo: this gate is about operands that stayed FULL WIDTH, and
    // an inline projection of a halved tuple did not. Matching only Var flagged one as
    // full width -- a false rejection, safe but wrong, and the same Var-only assumption
    // that produced the silent defects elsewhere.
    if (OperandSplitInfo(arg, tile_vars, var_replacements)) continue;
    if (probe != nullptr && DeducerPinsOperandExtent(call, *probe, i, arg_split_dim)) continue;
    return {arg, i, arg_split_dim};
  }
  return {};
}

// Map a tile.slice RESULT axis back to the axis its shape / offset / valid_shape
// tuples are indexed by. Those tuples carry the pre-drop rank; drop_dims erases
// axes from the result afterwards and then pads back up to 2D by PREPENDING unit
// axes, so the two ranks disagree whenever drop_dims is non-empty.
//
// Reuses the deducer's own ParseSliceDropDims so the mapping cannot drift from
// the rule it is inverting. A result axis that is one of the synthetic padded
// unit axes has no pre-drop counterpart; it is singleton, so the halving path
// never asks about it, and returning the identity keeps callers total.
// The tile.slice source axes that survive drop_dims, ascending; nullopt when the
// call carries no usable drop_dims and the two axis spaces coincide.
//
// ParseSliceDropDims returns the axes ASCENDING, in range and each at most once,
// so the survivors fall out of a single merge against them -- no per-axis lookup
// into drop_dims. Walking the two in step also makes the ordering assumption
// explicit at the one place that depends on it.
std::optional<std::vector<int>> SliceKeptAxes(const CallPtr& call) {
  if (call->args_.size() < 5) return std::nullopt;
  auto shape_tuple = std::dynamic_pointer_cast<const MakeTuple>(call->args_[1]);
  if (!shape_tuple) return std::nullopt;
  std::vector<int64_t> drop_dims;
  try {
    drop_dims = ParseSliceDropDims(call->args_[4], shape_tuple->elements_, "tile.slice");
  } catch (const pypto::Error&) {
    return std::nullopt;
  }
  if (drop_dims.empty()) return std::nullopt;

  std::vector<int> kept;
  kept.reserve(shape_tuple->elements_.size());
  size_t next_drop = 0;
  for (int d = 0; d < static_cast<int>(shape_tuple->elements_.size()); ++d) {
    if (next_drop < drop_dims.size() && drop_dims[next_drop] == static_cast<int64_t>(d)) {
      ++next_drop;
      continue;
    }
    kept.push_back(d);
  }
  return kept;
}

// Sub-2D results are clamped back to 2D with leading unit axes, which shifts
// every surviving axis right by that padding.
int SliceResultPadding(size_t kept_count) { return kept_count < 2 ? static_cast<int>(2 - kept_count) : 0; }

int MapResultAxisToSliceArgAxis(const CallPtr& call, int result_axis) {
  auto kept = SliceKeptAxes(call);
  if (!kept.has_value()) return result_axis;
  const int kept_index = result_axis - SliceResultPadding(kept->size());
  if (kept_index < 0 || kept_index >= static_cast<int>(kept->size())) return result_axis;
  return (*kept)[kept_index];
}

// The inverse: which RESULT axis a pre-drop source axis ends up on, or -1 when
// drop_dims erases it.
//
// The two directions are inverses over the SURVIVING axes only, which is why
// both are expressed over the same SliceKeptAxes: the split axis is carried on
// the OPERAND, while the halving path indexes the RESULT type with it, and
// conflating the two put a 16-row window over an 8-row source (gh#2614). A
// synthetic padded result axis has no pre-drop counterpart, so the result->arg
// direction returns the identity for it while the arg->result direction never
// produces it; neither round-trips through the other there.
int MapSliceArgAxisToResultAxis(const CallPtr& call, int arg_axis) {
  auto kept = SliceKeptAxes(call);
  if (!kept.has_value()) return arg_axis;
  auto it = std::lower_bound(kept->begin(), kept->end(), arg_axis);
  if (it == kept->end() || *it != arg_axis) return -1;
  return SliceResultPadding(kept->size()) + static_cast<int>(std::distance(kept->begin(), it));
}

std::string DescribeSplitExtent(const ExprPtr& dim_size) {
  auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size);
  return ci ? std::to_string(ci->value_) : std::string("a runtime extent");
}

// The rejection text both operand gates share -- what is wrong, and every
// spelling that is in contract. One builder so the two gates cannot drift into
// giving different advice about the same mistake.
std::string FullWidthOperandDiagnostic(const std::string& op_name, const FullWidthOperand& full) {
  auto full_var = AsVarLike(full.arg);
  auto full_tt = std::dynamic_pointer_cast<const TileType>(full.arg->GetType());
  std::ostringstream os;
  os << "LowerAutoVectorSplit: '" << op_name
     << "' inside the automatically split vector region carries a full-width operand"
     << (full_var ? " '" + full_var->name_hint_ + "'" : " at argument " + std::to_string(full.arg_index))
     << " (extent "
     << (full_tt ? DescribeSplitExtent(full_tt->shape_[full.split_dim]) : std::string("unknown"))
     << " on its dim " << full.split_dim
     << ", the region's split axis), but this pass halves the op's RESULT to the per-lane extent. "
        "Nothing partitions that operand, so the two lanes would both read the whole tile under a "
        "per-lane result type. Derive the per-lane half first -- load or slice the value inside the "
        "split region, or write an explicit pl.tile.aiv_shard(...) -- and feed that to this op. If the "
        "value is meant to be shared by both lanes, keep it outside the split region or give its split "
        "axis a singleton extent so it broadcasts.";
  return os.str();
}

// The argument list the halved call will actually carry: ``new_args`` (shape
// tuples already halved for the ops that describe their own shape) with each
// tracked operand swapped for the halved replacement the trailing Substitute
// installs.
std::vector<ExprPtr> BuildHalvedCallArgs(const std::vector<ExprPtr>& new_args,
                                         const std::unordered_map<const Var*, VarPtr>& var_replacements) {
  std::vector<ExprPtr> probe_args;
  probe_args.reserve(new_args.size());
  for (const auto& arg : new_args) {
    auto replaced = ReplacedOperand(arg, var_replacements);
    probe_args.push_back(replaced ? replaced : arg);
  }
  return probe_args;
}

// Whether the halved call still agrees with type inference over the arguments it
// will actually carry. This is the pass's PRIMARY correctness gate for halving:
// it asks the operator itself, needs no per-operator metadata, and therefore
// cannot go stale as operators are added or their contracts change.
//
// Compare the deduced PHYSICAL shape only: HalveTileShape also localizes
// valid_shape per lane, which deduction has no way to reproduce and which is not
// what this guards. A deduction that throws is a mismatch too -- the operator is
// refusing the very arguments the halved node would carry.
//
// What it covers, measured against the operators registered today: for every
// operand whose extent the deducer reads, this check alone rejects the
// full-width form -- tile.add's operands (the gh#2203 report), a rank-1 bias on
// the split axis, tile.row_argmax's shape-matched tmp, and an axis-permuting
// tile.transpose_view over a source no lane owns a half of. The declared
// exemptions exist only for the operands it is BLIND to; see
// FindFullWidthOperand's ``probe`` and OpLaneInvariantArgSpec.
bool HalvedCallStaysTypeConsistent(const CallPtr& call, const HalvedCallProbe& probe) {
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op || !probe.args || !probe.expected) return false;
  if (!OpRegistry::GetInstance().IsRegistered(op->name_)) return true;

  std::shared_ptr<const TileType> deduced;
  try {
    auto probe_call = OpRegistry::GetInstance().Create(op->name_, *probe.args, call->kwargs_, call->span_);
    deduced = std::dynamic_pointer_cast<const TileType>(probe_call->GetType());
  } catch (const pypto::Error&) {
    return false;
  }
  if (!deduced || deduced->shape_.size() != probe.expected->shape_.size()) return false;
  for (size_t d = 0; d < probe.expected->shape_.size(); ++d) {
    if (!structural_equal(deduced->shape_[d], probe.expected->shape_[d])) return false;
  }
  return true;
}

// How far apart the two lanes' data sits for a tile whose physical half is
// @p box_half: the body's partition stride when ResolveLaneStride found one,
// else the box half itself (the default box partition).
ExprPtr LaneStep(const ExprPtr& lane_stride, const ExprPtr& box_half) {
  return lane_stride ? lane_stride : box_half;
}

ExprPtr MakeConstLike(const ExprPtr& ref, int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, GetScalarDtype(ref), span);
}

// Lane L's valid extent on the split axis: ``clamp(V - L * S, 0, S)``, where
// @p lane_extent is the partition stride S (the tile's physical half under the
// default box partition, the balanced ``ceil(V / 2)`` under a rebalanced body).
ExprPtr LocalizeValidDimForSplit(const ExprPtr& valid_dim, const ExprPtr& original_dim,
                                 const ExprPtr& lane_extent, const ExprPtr& subblock_idx) {
  if (!valid_dim) return valid_dim;
  if (!subblock_idx) {
    return lane_extent;
  }
  // Shortcut: a fully-valid axis whose stride is EXACTLY half of it splits into
  // two identical lanes, so the stride is already the per-lane extent. Anything
  // else falls through to the clamp — an odd axis (the ceil stride over-states
  // lane 1 by one cell, which is what pto-isa reads the runtime extents for),
  // and a rebalanced stride (which is narrower than the full axis by design).
  // A dynamic axis keeps the shortcut, as it had before parity mattered.
  if (AreExprsEqual(valid_dim, original_dim)) {
    auto original_const = std::dynamic_pointer_cast<const ConstInt>(original_dim);
    auto extent_const = std::dynamic_pointer_cast<const ConstInt>(lane_extent);
    const bool exact_half = original_const == nullptr || extent_const == nullptr ||
                            extent_const->value_ * 2 == original_const->value_;
    if (exact_half) return lane_extent;
  }

  auto span = valid_dim->span_;
  auto subblock_offset = MakeMul(subblock_idx, lane_extent, span);
  // Saturating subtract, NOT `max(valid - offset, 0)`: a valid extent can carry
  // an UNSIGNED dtype (tile.set_validshape accepts UINT64), where `valid -
  // offset` wraps to a huge value for the lane the extent does not reach and the
  // clamp would then hand that lane a FULL half instead of nothing. Clamping the
  // minuend first is correct for both signednesses.
  auto reached = MakeMax(valid_dim, subblock_offset, span);
  auto remaining = MakeSub(reached, subblock_offset, span);
  return MakeMin(remaining, lane_extent, span);
}

// Whether a tile.set_validshape split-axis operand must be localized to the
// current subblock. Localization (subtracting half on lane 1) is only correct
// when the valid extent genuinely spans both lanes -- i.e. it equals the full
// pre-split extent, or it provably overflows the halved physical box (in which
// case leaving it unlocalized would also trip the PTOAS "operand <= shape dim"
// verifier). A smaller operand is a *replicated* valid extent both AIV lanes
// share (e.g. a fused-attention head count, valid_row=5 on a [16]->[8] split):
// localizing it would collapse lane 1 to 0 and silently corrupt that lane.
bool ValidOperandNeedsLocalize(const ExprPtr& valid_dim, const ExprPtr& original_dim,
                               const ExprPtr& half_dim_size) {
  if (!valid_dim) return false;
  if (AreExprsEqual(valid_dim, original_dim)) return true;
  auto valid_const = std::dynamic_pointer_cast<const ConstInt>(valid_dim);
  auto half_const = std::dynamic_pointer_cast<const ConstInt>(half_dim_size);
  return valid_const != nullptr && half_const != nullptr && valid_const->value_ > half_const->value_;
}

CallPtr RebuildCallWithSplit(const CallPtr& call, int split_code) {
  std::vector<std::pair<std::string, std::any>> new_kwargs;
  bool has_split = false;
  for (const auto& [key, val] : call->kwargs_) {
    if (key == "split") {
      new_kwargs.emplace_back("split", std::any(split_code));
      has_split = true;
    } else {
      new_kwargs.emplace_back(key, val);
    }
  }
  if (!has_split) {
    new_kwargs.emplace_back("split", std::any(split_code));
  }
  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), call->GetType(), call->span_);
}

// The halved tile's TileView.
//
// An explicit pre-split view is carried through with its split-axis extent
// localized to the current lane. A tile with NO view is fully valid, which
// needs no view of its own after an EVEN halving — both lanes fill their box.
// An ODD halving is the exception: the ceil box over-states lane 1 by one cell,
// so the lane's true extent has to be materialized as a view even though the
// pre-split tile carried none. Downstream that extent is what reaches the
// store's row count and pto-isa's per-lane TPOP valid operands.
std::optional<TileView> HalvedTileView(const std::shared_ptr<const TileType>& tt, int dim,
                                       const std::vector<ExprPtr>& new_shape, const ExprPtr& subblock_idx,
                                       const ExprPtr& lane_stride) {
  const ExprPtr step = LaneStep(lane_stride, new_shape[dim]);
  if (const auto& tile_view = tt->tile_view_; tile_view.has_value()) {
    TileView tv = tile_view.value();
    if (dim < static_cast<int>(tv.valid_shape.size())) {
      tv.valid_shape[dim] =
          LocalizeValidDimForSplit(tv.valid_shape[dim], tt->shape_[dim], step, subblock_idx);
    }
    return tv;
  }
  // A tile with no view is fully valid; on the box partition that stays true
  // per lane unless the axis is odd. A rebalanced body always needs the view:
  // its stride is narrower than the box, so neither lane fills its box.
  if (!subblock_idx || (!lane_stride && !IsOddSplitExtent(tt->shape_[dim]))) return std::nullopt;
  TileView tv = tile_view_semantics::GetEffectiveTileView(*tt);
  tv.valid_shape = new_shape;
  tv.valid_shape[dim] = LocalizeValidDimForSplit(tt->shape_[dim], tt->shape_[dim], step, subblock_idx);
  return tv;
}

TypePtr HalveTileShape(const TypePtr& type, int dim, const ExprPtr& subblock_idx,
                       const ExprPtr& lane_stride) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = ComputeHalfDimSize(tt->shape_[dim]);

  auto new_tile_view = HalvedTileView(tt, dim, new_shape, subblock_idx, lane_stride);
  return std::make_shared<TileType>(new_shape, tt->dtype_, tt->memref_, new_tile_view, tt->memory_space_);
}

// The per-element split mapping for a call whose result is a TupleType -- one axis
// index per tuple element -- or
// nullopt when the operator's answer is not a halving this pass can emit.
//
// A tuple-returning op has no single "result split dim" to follow: tile.gather_compare
// answers a row split with a `dst` halved on dim 0 and a `cdst` ([1, rows]) halved on
// dim 1. Rather than declare that mapping per operator -- the metadata that kept going
// stale (gh#2612) -- ASK the operator: re-deduce from the arguments the halved call
// will carry and read which axis of each element moved. The mapping is then whatever
// the operator itself says it is, and a new tuple-returning op needs no registration.
//
// Every element must halve on exactly one axis. An element that comes back UNCHANGED is
// the dangerous case, not the harmless one: the operator produced a full-width output
// from an operand each lane owns only half of, so both lanes hold a value derived from
// half the data and neither is the whole answer. The shape is right and the contents are
// wrong, which is precisely what type consistency cannot see -- so it is refused. That
// is also what correctly rejects a COLUMN split of tile.gather_compare, whose two
// outputs are both sized from the source's rows and so do not move at all.
std::optional<std::vector<int>> DiscoverTupleElementSplits(const std::shared_ptr<const TupleType>& original,
                                                           const TypePtr& deduced_type) {
  auto deduced = std::dynamic_pointer_cast<const TupleType>(deduced_type);
  if (!deduced || deduced->types_.size() != original->types_.size()) return std::nullopt;

  std::vector<int> split_dims;
  split_dims.reserve(original->types_.size());
  for (size_t i = 0; i < original->types_.size(); ++i) {
    auto before = std::dynamic_pointer_cast<const TileType>(original->types_[i]);
    auto after = std::dynamic_pointer_cast<const TileType>(deduced->types_[i]);
    if (!before || !after || before->shape_.size() != after->shape_.size()) return std::nullopt;

    int split_dim = -1;
    for (size_t d = 0; d < before->shape_.size(); ++d) {
      if (structural_equal(before->shape_[d], after->shape_[d])) continue;
      if (split_dim >= 0) return std::nullopt;  // Two axes moved: not a halving.
      if (!structural_equal(after->shape_[d], ComputeHalfDimSize(before->shape_[d]))) return std::nullopt;
      split_dim = static_cast<int>(d);
    }
    if (split_dim < 0) return std::nullopt;  // Element unchanged -- see above.
    split_dims.push_back(split_dim);
  }
  return split_dims;
}

ExprPtr HalveTupleElement(const ExprPtr& tuple_expr, int dim) {
  auto tuple = std::dynamic_pointer_cast<const MakeTuple>(tuple_expr);
  if (!tuple || dim < 0 || dim >= static_cast<int>(tuple->elements_.size())) return tuple_expr;
  std::vector<ExprPtr> new_elements = tuple->elements_;
  new_elements[dim] = ComputeHalfDimSize(new_elements[dim]);
  return std::make_shared<MakeTuple>(std::move(new_elements), tuple_expr->span_);
}

ExprPtr LocalizeTupleElementForSplit(const ExprPtr& tuple_expr, int dim, const ExprPtr& original_dim,
                                     const ExprPtr& half_dim_size, const ExprPtr& subblock_idx) {
  auto tuple = std::dynamic_pointer_cast<const MakeTuple>(tuple_expr);
  if (!tuple || dim < 0 || dim >= static_cast<int>(tuple->elements_.size())) return tuple_expr;
  std::vector<ExprPtr> new_elements = tuple->elements_;
  new_elements[dim] =
      LocalizeValidDimForSplit(tuple->elements_[dim], original_dim, half_dim_size, subblock_idx);
  return std::make_shared<MakeTuple>(std::move(new_elements), tuple_expr->span_);
}

CallPtr RebuildTpopWithHalvedShape(const CallPtr& call, int split_code, int split_dim,
                                   const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  // NOT widened for a runtime extent, unlike the boundary ops in
  // LocalizeExplicitBoundaryValid. A hand-written tpop declares its own
  // valid_shape and its consumers inherit that declaration, so widening the pop
  // alone leaves each consumer writing a partial destination out of a full
  // source — measured to produce wrong data on a2a3 for every extent except the
  // two where the widening is a no-op. See
  // tests/st/runtime/cross_core/test_cross_core_split_parity.py, whose
  // xfailing params record what a runtime split-axis extent still gets wrong on
  // this path.
  auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx, lane_stride);

  std::vector<std::pair<std::string, std::any>> new_kwargs;
  bool has_split = false;
  for (const auto& [key, val] : call->kwargs_) {
    if (key == "split") {
      new_kwargs.emplace_back("split", std::any(split_code));
      has_split = true;
    } else {
      new_kwargs.emplace_back(key, val);
    }
  }
  if (!has_split) {
    new_kwargs.emplace_back("split", std::any(split_code));
  }

  return std::make_shared<Call>(call->op_, call->args_, std::move(new_kwargs), new_result_type, call->span_);
}

// The two AIV lanes' split-axis extents for a boundary tile, when the box, the
// valid extent and the partition stride are all compile-time constants.
//
// ``eL = clamp(V - L * S, 0, S)`` — the same clamp LocalizeValidDimForSplit
// materializes into the popped tile's valid_shape, and therefore exactly what
// PTOAS hands pto-isa as the tile's runtime valid extent.
struct LaneExtents {
  int64_t lane0 = 0;
  int64_t lane1 = 0;
  int64_t stride = 0;    ///< the partition stride S
  int64_t box_half = 0;  ///< ceil(box / 2), the per-lane physical box
  int64_t valid = 0;     ///< V, the pre-split valid extent on the split axis
};

std::optional<LaneExtents> ComputeLaneExtents(const std::shared_ptr<const TileType>& tt, int split_dim,
                                              const ExprPtr& lane_stride) {
  if (!tt || split_dim < 0 || split_dim >= static_cast<int>(tt->shape_.size())) return std::nullopt;
  auto box = std::dynamic_pointer_cast<const ConstInt>(tt->shape_[split_dim]);
  if (!box) return std::nullopt;
  const auto valid_shape = tile_view_semantics::GetEffectiveTileView(*tt).valid_shape;
  if (split_dim >= static_cast<int>(valid_shape.size())) return std::nullopt;
  auto valid = std::dynamic_pointer_cast<const ConstInt>(valid_shape[split_dim]);
  if (!valid) return std::nullopt;

  LaneExtents extents;
  extents.box_half = (box->value_ + 1) / 2;
  extents.valid = valid->value_;
  extents.stride = extents.box_half;
  if (lane_stride) {
    auto stride_const = std::dynamic_pointer_cast<const ConstInt>(lane_stride);
    if (!stride_const) return std::nullopt;
    extents.stride = stride_const->value_;
  }
  extents.lane0 = std::min(valid->value_, extents.stride);
  extents.lane1 = std::min(std::max(valid->value_ - extents.stride, static_cast<int64_t>(0)), extents.stride);
  return extents;
}

ExprPtr AdjustOffsets(const ExprPtr& offsets_expr, int split_dim, const ExprPtr& half_size,
                      const ExprPtr& subblock_idx) {
  auto offsets = std::dynamic_pointer_cast<const MakeTuple>(offsets_expr);
  if (!offsets || split_dim < 0 || split_dim >= static_cast<int>(offsets->elements_.size())) {
    return offsets_expr;
  }

  std::vector<ExprPtr> new_elements = offsets->elements_;
  auto original_offset = offsets->elements_[split_dim];

  ExprPtr adjusted;
  if (auto subblock_const = std::dynamic_pointer_cast<const ConstInt>(subblock_idx)) {
    if (subblock_const->value_ == 0) {
      adjusted = original_offset;
    } else if (subblock_const->value_ == 1) {
      if (auto original_const = std::dynamic_pointer_cast<const ConstInt>(original_offset);
          original_const && original_const->value_ == 0) {
        adjusted = half_size;
      } else {
        adjusted = MakeAdd(original_offset, half_size, original_offset->span_);
      }
    }
  }

  if (!adjusted) {
    // offset = original + get_subblock_idx() * half_size
    auto adjustment = MakeMul(subblock_idx, half_size, original_offset->span_);
    adjusted = MakeAdd(original_offset, adjustment, original_offset->span_);
  }
  new_elements[split_dim] = adjusted;

  return std::make_shared<MakeTuple>(std::move(new_elements), offsets->span_);
}

// A `tile.store` of a tracked (halved) tile, rebuilt with its destination offset
// moved to this lane's half, or nullptr when @p call is not such a store.
//
// The tile itself is swapped for its halved replacement by the trailing Substitute;
// this is the other half of that rewrite. Without it the two AIV lanes write the
// SAME rows from different data and lane 1's half is silently lost.
//
// The stored operand is a Var in the common case, but it is NOT always one. A tuple
// projection consumed inline -- `pl.tile.store(pair[0], [0, 0], out)`, which the DSL
// produces verbatim because neither the parser nor FlattenCallExpr hoists a projection
// into its own binding -- arrives as a TupleGetItemExpr. Substitute still rewrites the
// tuple underneath it to the halved one, so matching only Var here left exactly the
// silent overlapping write this function exists to prevent. Read the element's axis
// back off the halved tuple type instead.
CallPtr LocalizeStoreOffset(const CallPtr& call, const std::unordered_map<const Var*, TileInfo>& tile_vars,
                            const std::unordered_map<const Var*, VarPtr>& var_replacements,
                            const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  if (!IsOp(call, "tile.store") || call->args_.size() < 3) return nullptr;

  auto info = OperandSplitInfo(call->args_[0], tile_vars, var_replacements);
  if (!info) return nullptr;

  std::vector<ExprPtr> new_args = call->args_;
  new_args[1] = AdjustOffsets(call->args_[1], info->split_dim, LaneStep(lane_stride, info->half_dim_size),
                              subblock_idx);
  return std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, call->GetType(), call->span_);
}

TypePtr ApplyTrackedTileShape(const TypePtr& type, int dim, const ExprPtr& half_dim_size,
                              const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt || dim < 0 || dim >= static_cast<int>(tt->shape_.size())) return type;

  std::vector<ExprPtr> new_shape = tt->shape_;
  new_shape[dim] = half_dim_size;

  auto new_tile_view = HalvedTileView(tt, dim, new_shape, subblock_idx, lane_stride);
  return std::make_shared<TileType>(new_shape, tt->dtype_, tt->memref_, new_tile_view, tt->memory_space_);
}

// Product of static tile dims in [lo, hi). Returns -1 if any dim is non-const
// (real products are >= 1, so -1 is an unambiguous "not static" sentinel).
int64_t StaticDimProduct(const std::vector<ExprPtr>& shape, int lo, int hi) {
  int64_t p = 1;
  for (int d = lo; d < hi; ++d) {
    auto ci = std::dynamic_pointer_cast<const ConstInt>(shape[d]);
    if (!ci) return -1;
    p *= ci->value_;
  }
  return p;
}

bool HasAutoEquivalentReinterpretShape(const CallPtr& call) {
  INTERNAL_CHECK(call) << "Internal error: auto-shape comparison requires a non-null Call";
  INTERNAL_CHECK_SPAN(IsOp(call, "tile.reinterpret_view"), call->span_)
      << "Internal error: auto-shape comparison expects tile.reinterpret_view";
  INTERNAL_CHECK_SPAN(call->args_.size() == 2, call->span_)
      << "Internal error: explicit tile.reinterpret_view must carry data and shape arguments";

  auto explicit_type = std::dynamic_pointer_cast<const TileType>(call->GetType());
  INTERNAL_CHECK_SPAN(explicit_type, call->span_)
      << "Internal error: tile.reinterpret_view must produce TileType";
  auto auto_call =
      OpRegistry::GetInstance().Create("tile.reinterpret_view", {call->args_[0]}, call->kwargs_, call->span_);
  auto auto_type = std::dynamic_pointer_cast<const TileType>(auto_call->GetType());
  INTERNAL_CHECK_SPAN(auto_type, call->span_)
      << "Internal error: auto-shaped tile.reinterpret_view must produce TileType";
  if (explicit_type->shape_.size() != auto_type->shape_.size()) return false;
  for (size_t i = 0; i < explicit_type->shape_.size(); ++i) {
    if (!AreExprsEqual(explicit_type->shape_[i], auto_type->shape_[i])) return false;
  }
  return true;
}

// Handle a tile.reshape whose input is an already-split tile. Reshape preserves
// row-major element order, so the split partition (first half vs second half of
// the input's split dim) lands on a specific result dimension; this finds it and
// halves that dimension, re-tracking the (possibly migrated) split axis.
//
// Returns: the rewritten statement when the split axis migrates to a *different*
// result dim (e.g. the rms_norm [N,1]->[1,N] column reshape); nullptr when it
// stays on the same dim index OR the row-major partition cannot be flat-tracked
// (caller falls through to generic halving, which also covers dynamic dims and
// the non-contiguous LEFT_RIGHT prefix); throws (reject) only when flat-tracking
// applies but no result dim carries the halved split cleanly -- rather than
// silently miscompile.
StmtPtr TryMigrateReshapeSplit(const CallPtr& call, const std::shared_ptr<const AssignStmt>& assign,
                               const std::shared_ptr<const TileType>& in_tt, int in_split_dim,
                               const ExprPtr& subblock_idx,
                               std::unordered_map<const Var*, TileInfo>& tile_vars,
                               std::unordered_map<const Var*, VarPtr>& var_replacements,
                               const ExprPtr& lane_stride) {
  auto res_tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
  if (!res_tt || call->args_.size() < 2) return nullptr;
  INTERNAL_CHECK_SPAN(in_split_dim >= 0 && in_split_dim < static_cast<int>(in_tt->shape_.size()), call->span_)
      << "Internal error: input split dim " << in_split_dim << " out of bounds for rank "
      << in_tt->shape_.size();

  // Flat-offset tracking only applies when the split partition is a contiguous
  // prefix (every input dim before the split dim is 1 -- always true for the
  // dim-0 UP_DOWN split, not for a LEFT_RIGHT col split of a multi-row tile),
  // all extents are static, and the input split axis is EVEN -- an odd axis has
  // no exact element count to match a result dim's first half against. Anything
  // else defers to generic halving (same-axis path), which ceil-halves.
  const int64_t prefix_in = StaticDimProduct(in_tt->shape_, 0, in_split_dim);
  auto orig_c = std::dynamic_pointer_cast<const ConstInt>(in_tt->shape_[in_split_dim]);
  const int64_t inner_in =
      StaticDimProduct(in_tt->shape_, in_split_dim + 1, static_cast<int>(in_tt->shape_.size()));

  // An ODD input split axis has no exact element count to match a result dim's
  // first half against, so migration cannot be expressed. Falling through is
  // only safe when the generic (same-axis) path can still carry the split: if
  // the result's own split dim is gone or singleton -- `[15, 1] -> [1, 15]` --
  // that path leaves the reshape FULL width while each lane holds a half, so
  // both lanes would read the same rows. Reject instead of miscompiling.
  if (orig_c && (orig_c->value_ % 2) != 0) {
    const bool same_axis_carries_split = in_split_dim < static_cast<int>(res_tt->shape_.size()) &&
                                         !IsSingletonDim(res_tt->shape_[in_split_dim]);
    CHECK_SPAN(same_axis_carries_split, call->span_)
        << "SplitVectorKernel: tile.reshape moves an ODD split axis (dim " << in_split_dim << ", extent "
        << orig_c->value_ << ") onto a result whose dim " << in_split_dim
        << " cannot carry it, and an odd axis cannot be tracked through the reshape (its two lanes hold "
           "different extents). Pad that axis to an even extent and narrow it back with "
           "pl.tile.set_validshape(...), or keep the reshape out of the split scope.";
    return nullptr;
  }
  if (prefix_in != 1 || !orig_c || inner_in < 0) return nullptr;

  // Number of elements in the first half (row-major) of the split partition.
  const int64_t split_flat = (orig_c->value_ / 2) * inner_in;

  // Find the result dim whose first half matches that flat prefix exactly.
  int d_out = -1;
  int64_t prefix_out = 1;
  for (int d = 0; d < static_cast<int>(res_tt->shape_.size()); ++d) {
    auto out_c = std::dynamic_pointer_cast<const ConstInt>(res_tt->shape_[d]);
    if (!out_c) return nullptr;  // dynamic result dim -> defer to generic
    const int64_t inner_out =
        StaticDimProduct(res_tt->shape_, d + 1, static_cast<int>(res_tt->shape_.size()));
    if (inner_out < 0) return nullptr;
    if (prefix_out == 1 && (out_c->value_ % 2) == 0 && (out_c->value_ / 2) * inner_out == split_flat) {
      d_out = d;
      break;
    }
    prefix_out *= out_c->value_;
  }
  // Flat-tracking applies but no clean per-dim halving exists -> reject.
  if (d_out < 0) {
    throw pypto::ValueError(
        "SplitVectorKernel: tile.reshape moves the split axis (dim " + std::to_string(in_split_dim) +
        ") across a layout this pass cannot track under split; keep the reduction/reshape out of the "
        "split scope.");
  }

  // Split axis stays on the same dim index -> generic halving handles it; only
  // migrate when it actually moves.
  if (d_out == in_split_dim) return nullptr;

  // A migrated axis carries its own half, which is derived from the RESULT dim's
  // box — it cannot express a partition balanced on another axis's valid extent.
  CHECK_SPAN(!lane_stride, call->span_)
      << "SplitVectorKernel: tile.reshape moves the split axis from dim " << in_split_dim << " to dim "
      << d_out
      << " inside a split region whose ragged boundary was balanced across the two AIV lanes. The two "
         "partitions cannot be reconciled: pad the boundary tile's valid extent to its physical box "
         "with pl.tile.set_validshape(...), or move the reshape out of the split scope.";

  // Halve the migrated dim on both the reshape target arg and the result type.
  ExprPtr half_dim_size = ComputeHalfDimSize(res_tt->shape_[d_out]);
  auto new_result_type = HalveTileShape(call->GetType(), d_out, subblock_idx, /*lane_stride=*/nullptr);
  std::vector<ExprPtr> new_args = call->args_;
  new_args[1] = HalveTupleElement(call->args_[1], d_out);

  auto new_call =
      std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type, call->span_);
  auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
  TileInfo info{half_dim_size, d_out};
  tile_vars[assign->var_.get()] = info;
  tile_vars[new_var.get()] = info;
  var_replacements[assign->var_.get()] = new_var;
  return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
}

StmtPtr ProcessStmt(const StmtPtr& stmt, SplitMode mode, int split_dim,
                    std::unordered_map<const Var*, TileInfo>& tile_vars, bool is_aiv,
                    const ExprPtr& subblock_idx, std::unordered_map<const Var*, VarPtr>& var_replacements,
                    const ExprPtr& lane_stride) {
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    if (auto projected = RetypeTupleProjection(assign, tile_vars, var_replacements)) return projected;

    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (!call || !call->op_) return stmt;

    const auto& op_name = call->op_->name_;

    if (IsOp(call, "tile.tpush_to_aiv") || IsOp(call, "tile.tpush_to_aic")) {
      INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
          << "Internal error: " << op_name << " must carry the pushed tile";
      const int push_code =
          IsOp(call, "tile.tpush_to_aic")
              ? GatherSplitCode(mode, call->args_[0]->GetType(), split_dim, op_name, call->span_)
              : ShardSplitCode(mode, call->args_[0]->GetType(), split_dim, lane_stride, op_name, call->span_);
      auto new_call = RebuildCallWithSplit(call, push_code);
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }

    // tpop_from_aic: AIV consumes from cube — halve the popped tile to match split vector lanes.
    // tpop_from_aiv: AIC consumes from vector — keep full tile shape; only sync split attribute
    // (vector-side split affects AIV compute, not the matmul operand tile delivered to cube).
    if (IsOp(call, "tile.tpop_from_aiv")) {
      auto new_call =
          RebuildCallWithSplit(call, GatherSplitCode(mode, call->GetType(), split_dim, op_name, call->span_));
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }
    if (IsOp(call, "tile.tpop_from_aic")) {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      auto new_call = RebuildTpopWithHalvedShape(
          call, ShardSplitCode(mode, call->GetType(), split_dim, lane_stride, op_name, call->span_),
          split_dim, subblock_idx, lane_stride);
      auto new_var =
          std::make_shared<Var>(assign->var_->name_hint_, new_call->GetType(), assign->var_->span_);
      if (tt && split_dim < static_cast<int>(tt->shape_.size())) {
        TileInfo info{ComputeHalfDimSize(tt->shape_[split_dim]), split_dim};
        tile_vars[assign->var_.get()] = info;
        tile_vars[new_var.get()] = info;
      }
      var_replacements[assign->var_.get()] = new_var;
      return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
    }

    // AIV only: tile.load — halve result shape, halve shape/valid_shape args, adjust offset.
    // Singleton split-dim tiles (e.g. broadcast [1, 128] under UP_DOWN) are preserved as-is.
    if (is_aiv && IsOp(call, "tile.load") && call->args_.size() >= 4) {
      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
      bool is_singleton = tt && IsSingletonSplitAxis(*tt, split_dim);

      if (is_singleton) {
        return stmt;
      }

      // Rank-1 (and rank-0) loads carry no 2D split axis: which physical axis is
      // "the split axis" only becomes defined once the tile is reshaped to 2D.
      // Halving them here is unsafe -- under UP_DOWN it would split a rank-1
      // column vector along the wrong axis (e.g. a [128] scale later reshaped to
      // [1, 128] would be halved to [64] and then fail to reshape). Bypass them
      // under every split mode and let the consuming reshape introduce and slice
      // the split axis (see the tile.reshape handling below). LEFT_RIGHT already
      // bypassed rank-1 loads via split_dim >= rank; this also covers UP_DOWN.
      if (!tt || static_cast<int>(tt->shape_.size()) < 2 ||
          split_dim >= static_cast<int>(tt->shape_.size())) {
        return stmt;
      }
      ExprPtr half_dim_size = ComputeHalfDimSize(tt->shape_[split_dim]);

      auto new_result_type = HalveTileShape(call->GetType(), split_dim, subblock_idx, lane_stride);
      const ExprPtr load_step = LaneStep(lane_stride, half_dim_size);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[1] = AdjustOffsets(call->args_[1], split_dim, load_step, subblock_idx);
      new_args[2] = HalveTupleElement(call->args_[2], split_dim);
      new_args[3] = LocalizeTupleElementForSplit(call->args_[3], split_dim, tt->shape_[split_dim], load_step,
                                                 subblock_idx);

      auto new_call =
          std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type, call->span_);
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
      TileInfo info{half_dim_size, split_dim};
      tile_vars[assign->var_.get()] = info;
      tile_vars[new_var.get()] = info;
      var_replacements[assign->var_.get()] = new_var;
      return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
    }

    // AIV only: tile.store — adjust offset using tracked tile info
    if (is_aiv) {
      if (auto localized =
              LocalizeStoreOffset(call, tile_vars, var_replacements, subblock_idx, lane_stride)) {
        return std::make_shared<AssignStmt>(assign->var_, localized, assign->span_);
      }
    }

    // AIV only: any other op producing TileType — halve result shape (and static shape args when present).
    // Reject reduce ops that reduce on the split axis (partial reduction is semantically incorrect).
    // Skip halving when the output split-dim is singleton (broadcast / degenerate tiles).
    if (is_aiv) {
      // Find the primary tracked (already-split) tile input. Its split dim can
      // differ from the global split_dim once a reshape has migrated the split
      // axis across dimensions (the rms_norm [N,1]<->[1,N] column reshape), so the
      // reduce guard and elementwise halving must follow the input's dim.
      int in_split_dim = -1;
      std::shared_ptr<const TileType> in_tt;
      for (const auto& a : call->args_) {
        if (auto info = OperandSplitInfo(a, tile_vars, var_replacements)) {
          in_split_dim = info->split_dim;
          in_tt = std::dynamic_pointer_cast<const TileType>(a->GetType());
          break;
        }
      }

      // Reduce on the (possibly migrated) split axis is a partial reduction —
      // reject it on the input's tracked dim, not just the global split_dim.
      const int reduce_split_dim = (in_split_dim >= 0) ? in_split_dim : split_dim;
      if (IsReduceOnSplitAxis(call, reduce_split_dim)) {
        throw pypto::ValueError("SplitVectorKernel: reduce op '" + op_name +
                                "' reduces on the split axis (dim " + std::to_string(reduce_split_dim) +
                                "); partial reduction in a split kernel is not supported");
      }

      // An absolutely-indexed operand is unsafe the moment its producer is
      // partitioned, whatever happens to the RESULT. Both of the checks below
      // therefore run here rather than beside the halving: the trailing
      // Substitute rewrites the argument even on the paths that halve nothing,
      // so gating them on a halveable TileType result would let exactly those
      // paths through.
      //   * a singleton result returns early below -- a gather whose `indices`
      //     are [1, N] yields a [1, N] result, so nothing is halved while the
      //     table underneath it still is;
      //   * a TupleType result skips the whole block -- see the check after this
      //     one.
      if (auto absolute = FindPartitionedAbsoluteOperand(call, tile_vars, var_replacements); absolute.arg) {
        auto absolute_var = AsVarLike(absolute.arg);
        const bool is_dst = absolute.kind == LaneInvariantArg::AbsoluteIndexedDestination;
        CHECK_SPAN(false, call->span_)
            << "LowerAutoVectorSplit: '" << op_name << "' addresses its "
            << (is_dst ? "destination" : "source") << " operand"
            << (absolute_var ? " '" + absolute_var->name_hint_ + "'" : "")
            << " at ABSOLUTE indices, so both AIV lanes must see it whole -- but that operand was "
               "partitioned along dim "
            << absolute.split_dim
            << " by its producer inside the split region. The index values are not rebased, so each lane "
            << (is_dst ? "writes to the wrong half (and past the end of it once an index exceeds the "
                         "halved extent)."
                       : "reads the wrong half (and out of bounds once an index exceeds the halved "
                         "extent).")
            << " Produce it OUTSIDE the automatically split region and pass it in, so it stays full "
               "width while the index operand is halved. Note that pl.gather / pl.scatter over a tensor "
               "lower to a tile.load feeding the tile op, so loading it inside the region hits this too.";
      }

      // A TupleType result carries one split axis PER ELEMENT, so the generic path
      // below -- which follows a single result_split_dim -- cannot express it.
      // Doing nothing is not an option either: the trailing Substitute still
      // rewrites this call's arguments, so a halved operand would end up under a
      // result type nobody re-derived (tile.gather_compare over a halved [128, 128]
      // src keeps declaring Tuple[[256, out_cols], [1, 256]] -- illegal IR that
      // misallocates). Halve each element on the axis the OPERATOR moves it along.
      if (auto tuple_type = std::dynamic_pointer_cast<const TupleType>(call->GetType());
          tuple_type && in_split_dim >= 0) {
        const auto probe_args = BuildHalvedCallArgs(call->args_, var_replacements);
        TypePtr deduced;
        std::string refusal;
        try {
          deduced =
              OpRegistry::GetInstance().Create(op_name, probe_args, call->kwargs_, call->span_)->GetType();
        } catch (const pypto::Error& err) {
          refusal = err.what();  // The operator refuses the halved arguments outright.
        }

        // The two ways this fails are different diagnoses, so do not merge them.
        //
        // REFUSED: the operator threw. Either a constraint does not survive halving
        // (tile.tquant_mx requires M % 16 == 0, which a per-lane M can break), or a
        // workspace was sized from the FULL source -- after LowerCompositeOps decomposes
        // it, tile.tquant_mx_raw requires its [1, groups] scratch to match a count
        // derived from src, and that singleton dim 0 keeps the generic path from halving
        // it. Repartitioning such a workspace is not implemented: its extent lives in the
        // deducer, and its producing tile.create was already emitted at full width.
        // Quote the operator rather than guessing which of the two it was.
        CHECK_SPAN(deduced != nullptr, call->span_)
            << "LowerAutoVectorSplit: '" << op_name
            << "' returns a tuple of tiles and consumes an operand the split already partitioned, but it "
               "REFUSES the halved arguments: "
            << refusal
            << "\nA workspace operand sized from the full source is the usual cause -- the split cannot "
               "resize one, because its extent is derived inside the operator and its allocation was "
               "already emitted at full width. Allocate the workspace at the per-lane size yourself, or "
               "move the operation outside the automatically split region.";

        // ACCEPTED but an element did not move: the operator produced a full-width output
        // from an operand each lane owns half of, so its shape is right and its contents
        // are wrong -- which type consistency alone cannot see.
        auto splits = DiscoverTupleElementSplits(tuple_type, deduced);
        CHECK_SPAN(splits.has_value(), call->span_)
            << "LowerAutoVectorSplit: '" << op_name
            << "' returns a tuple of tiles and consumes an operand the split already partitioned, but its "
               "own type inference does not answer the halved arguments with one halved axis per tuple "
               "element. An element that keeps its full extent was computed from an operand each AIV lane "
               "owns only half of, so neither lane holds the whole answer -- its shape would still look "
               "right. Move the operation outside the automatically split region, or feed it only values "
               "that are shared by both lanes.";

        // Halve through the same machinery the single-TileType path uses, so an odd
        // extent is localized per lane identically. Re-deduction only discovered WHICH
        // axis; DiscoverTupleElementSplits already checked the two agree on the extent.
        std::vector<TypePtr> new_elements;
        new_elements.reserve(splits->size());
        for (size_t i = 0; i < splits->size(); ++i) {
          new_elements.push_back(
              HalveTileShape(tuple_type->types_[i], (*splits)[i], subblock_idx, lane_stride));
        }
        auto new_result_type = std::make_shared<TupleType>(std::move(new_elements));
        // Emit the original args and let the trailing Substitute swap the halved
        // operands in, exactly as the single-TileType path below does.
        auto new_call =
            std::make_shared<Call>(call->op_, call->args_, call->kwargs_, new_result_type, call->span_);
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
        var_replacements[assign->var_.get()] = new_var;
        return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
      }

      auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());

      // An explicit reinterpret shape may redistribute bytes across dimensions.
      // The AIV split machinery deliberately tracks a physical axis, not a flat
      // element interval, so only the canonical auto shape is safe to halve. An
      // explicit spelling of that same shape is accepted and rewritten below;
      // arbitrary byte-equivalent shapes must stay outside the split scope.
      if (tt && IsOp(call, "tile.reinterpret_view") && call->args_.size() == 2) {
        CHECK_SPAN(HasAutoEquivalentReinterpretShape(call), call->span_)
            << "SplitVectorKernel: tile.reinterpret_view with an explicit shape inside a split kernel "
               "must match its auto-inferred shape. Omit shape= (recommended), or move an arbitrary "
               "byte-equivalent reinterpret_view outside the split scope.";
      }

      // tile.reshape that moves the split axis to a different result dim.
      // Do not route reinterpret_view through flat element-count migration: its
      // accepted auto shape keeps the physical split axis at the same index.
      if (tt && IsOp(call, "tile.reshape") && in_split_dim >= 0 && in_tt) {
        if (auto migrated = TryMigrateReshapeSplit(call, assign, in_tt, in_split_dim, subblock_idx, tile_vars,
                                                   var_replacements, lane_stride)) {
          return migrated;
        }
        // nullptr -> split extent stays in place; fall through to generic halving.
      }

      // Result split dim: follow the tracked input's (possibly migrated) dim; root
      // ops with no tracked input use the global split dim.
      //
      // in_split_dim is an axis of the OPERAND, and indexing the RESULT type with
      // it is only valid while the two share an axis correspondence. A
      // rank-changing view breaks that: tile.slice with drop_dims erases axes and
      // then clamps back to 2D by PREPENDING unit axes, so a source axis 0 can
      // land on result axis 1. Reading the result at the source's axis then found
      // the synthetic unit axis, took the singleton early-return below, and passed
      // the slice through untouched while its source was halved underneath it --
      // a 16-row window over an 8-row source (gh#2614). Map through the op's own
      // rule instead of assuming identity.
      int result_split_dim = (in_split_dim >= 0) ? in_split_dim : split_dim;
      if (in_split_dim >= 0 && IsOp(call, "tile.slice")) {
        const int mapped = MapSliceArgAxisToResultAxis(call, in_split_dim);
        if (mapped >= 0) {
          result_split_dim = mapped;
        } else {
          // drop_dims erased the very axis the split partitions. Its unit-extent
          // requirement is on the slice WINDOW, not on the source, so a tracked
          // [16, 4] source windowed [1, 4] reaches here with a live split axis.
          // The result then carries no axis distinguishing the lanes, while each
          // lane's window sits in its own half -- lane 1 selects source row 8
          // where the program asked for row 0, and the untracked result gets no
          // store offset, so both lanes write the same place with different data.
          // Only a singleton source axis is safe: both lanes then hold the same
          // single row, so dropping it loses nothing.
          const bool axis_is_shared = in_tt && in_split_dim < static_cast<int>(in_tt->shape_.size()) &&
                                      IsSingletonDim(in_tt->shape_[in_split_dim]);
          CHECK_SPAN(axis_is_shared, call->span_)
              << "LowerAutoVectorSplit: '" << op_name << "' drops dim " << in_split_dim
              << ", which is the axis the automatic split partitions its source along. Each AIV lane "
                 "would select that window from its own half -- lane 1 reads a different row of the "
                 "original than the program asked for -- and the result carries no axis left to tell the "
                 "lanes apart, so a later store cannot give them separate destinations. Slice the axis "
                 "before the split region, keep the dropped axis out of the split axis, or move the "
                 "slice outside the automatically split region.";
        }
      }
      if (tt && result_split_dim < static_cast<int>(tt->shape_.size())) {
        if (IsSingletonDim(tt->shape_[result_split_dim])) {
          return stmt;
        }
        CHECK_SPAN(!IsUnsupportedAutoSplitGenerator(call), call->span_)
            << "SplitVectorKernel: automatic split-axis halving of '" << op_name
            << "' is not supported because its generated values depend on position. Move the operation "
               "outside the automatically-halved split region.";
        auto half_dim_size = ComputeHalfDimSize(tt->shape_[result_split_dim]);

        // tile.reshape or tile.reinterpret_view lifts a full (un-split) source
        // tile -- typically a rank-1 load for reshape, or an untracked tile
        // parameter for reinterpret_view -- onto a shape whose split axis spans
        // the full width. These are offsetless views, so halving only the result
        // type leaves BOTH AIV lanes reading the first
        // half of the full buffer; lane 1 then silently reuses lane 0's data
        // (observed as lane 1 applying the wrong half of the per-channel dequant
        // scale in dsv4 proj_b's INT8 GEMM epilogue). Emit the view at full
        // width and follow it with a per-subblock column slice so each lane reads
        // its own half. Views whose input is already split fall through to the
        // plain result-halving below (their producer already partitioned the data).
        if (IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view")) {
          // Through OperandSplitInfo: an INLINE projection is already partitioned, and
          // treating it as full width emits a full-width view plus a per-lane slice on
          // top of an operand Substitute then halves -- lane 1 slicing from 128 into a
          // 128-wide input.
          const bool input_is_split =
              OperandSplitInfo(call->args_[0], tile_vars, var_replacements).has_value();
          auto half_const = std::dynamic_pointer_cast<const ConstInt>(half_dim_size);
          // The per-lane slice is what partitions a full-width source, and it can
          // only be materialized from a STATIC half extent. Without it both views
          // fall through to plain result-halving, which is precisely the silent
          // "lane 1 reuses lane 0's data" miscompile above -- so the dynamic case
          // is rejected for reshape exactly as it is for reinterpret_view, rather
          // than left to a later consumer.
          if (!input_is_split) {
            CHECK_SPAN(half_const != nullptr, call->span_)
                << "SplitVectorKernel: " << op_name
                << " over a full-width source requires a static split extent so the pass can "
                   "materialize a per-lane slice. Split/load the source before "
                << op_name << ", or move " << op_name << " outside the split scope.";
          }
          if (!input_is_split && half_const != nullptr) {
            auto full_var =
                std::make_shared<Var>(assign->var_->name_hint_, call->GetType(), assign->var_->span_);
            auto full_view = std::make_shared<AssignStmt>(full_var, call, assign->span_);

            std::vector<ExprPtr> shape_elems;
            std::vector<ExprPtr> offset_elems;
            shape_elems.reserve(tt->shape_.size());
            offset_elems.reserve(tt->shape_.size());
            for (int d = 0; d < static_cast<int>(tt->shape_.size()); ++d) {
              if (d == result_split_dim) {
                shape_elems.push_back(MakeIndexConst(half_const->value_, assign->span_));
                offset_elems.push_back(
                    MakeMul(subblock_idx, MakeIndexConst(half_const->value_, assign->span_), assign->span_));
              } else {
                auto dim_const = std::dynamic_pointer_cast<const ConstInt>(tt->shape_[d]);
                if (IsOp(call, "tile.reinterpret_view")) {
                  CHECK_SPAN(dim_const != nullptr, call->span_)
                      << "SplitVectorKernel: tile.reinterpret_view over a full-width source requires a "
                         "static target shape so the pass can materialize a per-lane slice. Split/load "
                         "the source before reinterpret_view, or move reinterpret_view outside the split "
                         "scope.";
                }
                INTERNAL_CHECK_SPAN(dim_const != nullptr, assign->span_)
                    << "Internal error: tile.reshape non-split result dim " << d
                    << " must be static to slice the split axis";
                shape_elems.push_back(MakeIndexConst(dim_const->value_, assign->span_));
                offset_elems.push_back(MakeIndexConst(0, assign->span_));
              }
            }
            auto shape_tuple = std::make_shared<MakeTuple>(std::move(shape_elems), assign->span_);
            auto offset_tuple = std::make_shared<MakeTuple>(std::move(offset_elems), assign->span_);
            auto slice_call = OpRegistry::GetInstance().Create(
                "tile.slice", {full_var, shape_tuple, offset_tuple}, {}, assign->span_);
            auto slice_var =
                std::make_shared<Var>(assign->var_->name_hint_, slice_call->GetType(), assign->var_->span_);
            auto slice_assign = std::make_shared<AssignStmt>(slice_var, slice_call, assign->span_);

            TileInfo info{half_dim_size, result_split_dim};
            // Track both the original var and the slice replacement, matching the
            // other tile-producing branches: a later tile.store / loop init that
            // references the original var (before the final Substitute) must still
            // find the tile info to adjust its split-dim offset.
            tile_vars[assign->var_.get()] = info;
            tile_vars[slice_var.get()] = info;
            var_replacements[assign->var_.get()] = slice_var;
            return std::make_shared<SeqStmts>(std::vector<StmtPtr>{full_view, slice_assign}, assign->span_);
          }
        }

        auto new_result_type = HalveTileShape(call->GetType(), result_split_dim, subblock_idx, lane_stride);

        const ExprPtr lane_step = LaneStep(lane_stride, half_dim_size);
        std::vector<ExprPtr> new_args = call->args_;
        if ((IsOp(call, "tile.full") || IsOp(call, "tile.create")) && call->args_.size() >= 1) {
          new_args[0] = HalveTupleElement(call->args_[0], result_split_dim);
        } else if ((IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view")) &&
                   call->args_.size() >= 2) {
          new_args[1] = HalveTupleElement(call->args_[1], result_split_dim);
        } else if (IsOp(call, "tile.slice") && call->args_.size() >= 3) {
          // tile.slice = (src, shape, offset[, valid_shape[, drop_dims]]). The
          // generic result-type halving above shrinks the split dim of the
          // result TileType, but the static shape tuple (arg[1]) is left at full
          // width unless it is rewritten here -- codegen then emits a
          // pto.subview whose sizes (full) disagree with the partition the
          // tstore expects (half), the qk_pv strided sub-slice miscompile that
          // motivated the explicit-AIV-split RFC. Halve the shape tuple so it
          // tracks the halved result type.
          //
          // shape / offset / valid_shape are all indexed in the PRE-DROP rank,
          // while result_split_dim indexes the result. drop_dims erases axes
          // after those tuples are built and then pads back up to 2D by
          // PREPENDING unit axes, so the two disagree as soon as drop_dims is
          // non-empty: slice([1, 128, 128], drop_dims=[0]) -> [128, 128] puts the
          // result's dim 0 on the source's dim 1. Map the axis back before
          // touching any of them, or the wrong source axis is halved and the
          // type-consistency check then rejects a legal split.
          const int slice_split_dim = MapResultAxisToSliceArgAxis(call, result_split_dim);
          new_args[1] = HalveTupleElement(call->args_[1], slice_split_dim);

          // Offset (arg[2]) localization mirrors the reshape->slice path above:
          // only add the per-subblock base when the SOURCE tile is NOT already
          // split. A split-tracked source has already been partitioned by its
          // producer, so its offset is in lane-local coordinates and must be
          // left untouched; an unsplit (full-width) source needs
          // +subblock_idx*half so each lane reads its own half.
          // Through OperandSplitInfo: an INLINE projection is already in lane-local
          // coordinates, and adding the per-subblock base on top of it sends lane 1
          // reading from the END of a source that only holds this lane's half.
          const bool slice_src_is_split =
              OperandSplitInfo(call->args_[0], tile_vars, var_replacements).has_value();
          if (!slice_src_is_split) {
            new_args[2] = AdjustOffsets(call->args_[2], slice_split_dim, lane_step, subblock_idx);
          }

          // Optional explicit valid_shape (arg[3]) must stay consistent with the
          // result type's valid_shape, which HalveTileShape already localized to
          // this subblock regardless of src split state. An empty MakeTuple
          // sentinel (the "no valid_shape" form paired with drop_dims) and the
          // optional drop_dims (arg[4]) are passed through unchanged --
          // LocalizeTupleElementForSplit is a no-op on a tuple whose split_dim
          // is out of range.
          if (call->args_.size() >= 4) {
            new_args[3] = LocalizeTupleElementForSplit(call->args_[3], slice_split_dim,
                                                       tt->shape_[result_split_dim], lane_step, subblock_idx);
          }
        } else if (IsOp(call, "tile.set_validshape") && call->args_.size() == 3) {
          // args = (tile, valid_row, valid_col). Halving the result type alone
          // leaves the split-dim valid operand at its full pre-split extent, so
          // a full/overflowing operand exceeds the halved physical box (PTOAS
          // rejects it with "row/col operand <= shape dim"). Localize the
          // split-dim operand the same way HalveTileShape localizes the type's
          // valid_shape -- but ONLY when it genuinely spans both lanes. A smaller
          // operand is a replicated extent both AIV lanes share; localizing it
          // would collapse lane 1 to 0 and silently corrupt that lane.
          // set_validshape carries only (tile, valid_row, valid_col), so the
          // operand index is valid only for a 2D split dim; guard against a
          // migrated/higher result_split_dim that would index past the args.
          const int operand_idx = 1 + result_split_dim;  // dim 0 -> row, 1 -> col
          if (operand_idx < static_cast<int>(call->args_.size()) &&
              ValidOperandNeedsLocalize(call->args_[operand_idx], tt->shape_[result_split_dim], lane_step)) {
            new_args[operand_idx] = LocalizeValidDimForSplit(
                call->args_[operand_idx], tt->shape_[result_split_dim], lane_step, subblock_idx);
          }
        }

        // Halving only the RESULT is sound under two conditions, asked in this
        // order because the first subsumes the second wherever it can speak.
        //
        // (1) TYPE CONSISTENCY -- the operator's own answer, no metadata.
        // Re-deduce from the arguments the halved node will actually carry and
        // refuse to emit one whose declared result contradicts them. That is the
        // gh#2203 defect verbatim -- `tile.add([256,128],[256,128]) -> [128,128]`
        // fails print->parse and misleads every later consumer that re-derives
        // types from operands -- and it equally catches a rank-1 bias on the
        // split axis, a shape-matched scratch left at full width, and an
        // axis-permuting tile.transpose_view over a source no lane owns a half
        // of. (tile.transpose is caught earlier by the transpose hazard check;
        // transpose_view is a pure view and is not.)
        //
        // When it fails, name the full-width operand if there is one: that is
        // the cause in nearly every real program, and the generic wording below
        // is only for the axis-mapping case. FindFullWidthOperand runs WITHOUT a
        // probe here -- purely to refine the message, so a misjudged axis costs a
        // wrong number in the text rather than a rejected program.
        const int result_rank = static_cast<int>(tt->shape_.size());
        const auto probe_args = BuildHalvedCallArgs(new_args, var_replacements);
        const HalvedCallProbe probe{&probe_args, std::dynamic_pointer_cast<const TileType>(new_result_type)};
        if (!HalvedCallStaysTypeConsistent(call, probe)) {
          auto full = FindFullWidthOperand(call, result_split_dim, result_rank, tile_vars, var_replacements);
          CHECK_SPAN(false, call->span_)
              << (full.arg ? FullWidthOperandDiagnostic(op_name, full)
                           : "LowerAutoVectorSplit: halving '" + op_name +
                                 "' to the per-lane extent along dim " + std::to_string(result_split_dim) +
                                 " would emit a node whose declared result no longer matches type "
                                 "inference over its own arguments, so it cannot be lowered for two AIV "
                                 "lanes. This happens when the op maps its axes in a way the split cannot "
                                 "follow -- an axis-permuting view such as tile.transpose_view over a "
                                 "source no lane owns a half of is the usual case. Derive the per-lane "
                                 "half before the op (load or slice the value inside the split region, or "
                                 "write an explicit pl.tile.aiv_shard), or move the op outside the "
                                 "automatically split region.");
        }

        // (2) BLIND OPERANDS -- the backstop for what (1) cannot see. A deducer
        // that never reads an operand's extent says nothing by accepting it, so
        // a full-width operand it ignores still has to be either lane-local or
        // declared lane-invariant. tile.sel deduces its result from lhs/rhs
        // alone, yet its mask is per-element data that must be sharded with
        // them; tile.sels validates only that the mask COVERS src, so an
        // over-wide mask passes deduction while feeding lane 1 lane 0's rows.
        //
        // Passing the probe is what keeps this heuristic in its lane: an operand
        // whose extent the deducer pins is skipped, because (1) already decided
        // it. Without that, every operator with a legal full-width operand would
        // need a registry declaration purely to suppress a check that had no
        // business running -- which is how tile.rsqrt came to declare a scratch
        // its own deducer requires to match the input exactly.
        if (auto blind = FindFullWidthOperand(call, result_split_dim, result_rank, tile_vars,
                                              var_replacements, &probe);
            blind.arg) {
          CHECK_SPAN(false, call->span_) << FullWidthOperandDiagnostic(op_name, blind);
        }

        auto new_call = std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_result_type,
                                               call->span_);
        auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_result_type, assign->var_->span_);
        TileInfo info{half_dim_size, result_split_dim};
        tile_vars[assign->var_.get()] = info;
        tile_vars[new_var.get()] = info;
        var_replacements[assign->var_.get()] = new_var;
        return std::make_shared<AssignStmt>(new_var, new_call, assign->span_);
      }
    }

    return stmt;
  }

  if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    if (!call || !call->op_) return stmt;

    if (IsOp(call, "tile.tpush_to_aiv") || IsOp(call, "tile.tpush_to_aic")) {
      INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
          << "Internal error: " << call->op_->name_ << " must carry the pushed tile";
      const int push_code =
          IsOp(call, "tile.tpush_to_aic")
              ? GatherSplitCode(mode, call->args_[0]->GetType(), split_dim, call->op_->name_, call->span_)
              : ShardSplitCode(mode, call->args_[0]->GetType(), split_dim, lane_stride, call->op_->name_,
                               call->span_);
      auto new_call = RebuildCallWithSplit(call, push_code);
      return std::make_shared<EvalStmt>(new_call, eval->span_);
    }

    if (is_aiv) {
      if (auto localized =
              LocalizeStoreOffset(call, tile_vars, var_replacements, subblock_idx, lane_stride)) {
        return std::make_shared<EvalStmt>(localized, eval->span_);
      }
    }

    return stmt;
  }

  // A store reached as a RETURN EXPRESSION takes neither of the arms above, and the
  // trailing Substitute swaps in the halved tile regardless -- so without this the
  // ordinary `return pl.tile.store(v, [0, 0], out)` spelling leaves both AIV lanes
  // writing from row 0 while each holds a different half. Binding the store first
  // was never meant to be load-bearing.
  if (auto ret = std::dynamic_pointer_cast<const ReturnStmt>(stmt); ret && is_aiv) {
    auto localized = LocalizeReturnStores(ret, tile_vars, var_replacements, subblock_idx, lane_stride);
    return localized ? localized : stmt;
  }

  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    // Eagerly substitute initValues while rebuilding iter_args. If this is
    // deferred to the final Substitute pass, it can create a second IterArg
    // instance whose pointer diverges from the one referenced by the rebuilt
    // loop body, breaking structural equality.
    std::vector<IterArgPtr> new_iter_args =
        RepairIterArgs(for_stmt->iter_args_, tile_vars, var_replacements, subblock_idx, lane_stride);
    std::vector<VarPtr> new_return_vars = for_stmt->return_vars_;
    auto flat = std::vector<StmtPtr>();
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(for_stmt->body_)) {
      flat = seq->stmts_;
    } else {
      flat.push_back(for_stmt->body_);
    }
    auto new_body_stmts =
        ProcessStmts(flat, mode, split_dim, tile_vars, is_aiv, subblock_idx, var_replacements, lane_stride);
    StmtPtr new_body = (new_body_stmts.size() == 1)
                           ? new_body_stmts[0]
                           : std::make_shared<SeqStmts>(new_body_stmts, for_stmt->span_);

    // Propagate tile_vars tracking from iter_args to return_vars.
    // ForStmt return_vars are the loop-exit versions of the corresponding
    // iter_args.  If an iter_arg carries a halved tile, the return_var must
    // inherit the tile info so that downstream tile.store gets the correct
    // subblock offset adjustment.
    INTERNAL_CHECK_SPAN(for_stmt->iter_args_.size() == for_stmt->return_vars_.size(), for_stmt->span_)
        << "Internal error: ForStmt iter_args and return_vars sizes must match, got "
        << for_stmt->iter_args_.size() << " vs " << for_stmt->return_vars_.size();
    // The backedge is the third edge of the carry, and the only one the two
    // Repair* helpers do not touch.
    ValidateCarryBackedge(new_body, new_iter_args, tile_vars, var_replacements, for_stmt->span_);
    new_return_vars = RepairReturnVars(for_stmt->return_vars_, new_iter_args, tile_vars, var_replacements,
                                       subblock_idx, lane_stride);
    return loop_repair::RebuildForStmt(for_stmt, new_iter_args, new_body, new_return_vars);
  }

  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto then_flat = std::vector<StmtPtr>();
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(if_stmt->then_body_)) {
      then_flat = seq->stmts_;
    } else {
      then_flat.push_back(if_stmt->then_body_);
    }
    auto new_then = ProcessStmts(then_flat, mode, split_dim, tile_vars, is_aiv, subblock_idx,
                                 var_replacements, lane_stride);
    StmtPtr new_then_body =
        (new_then.size() == 1) ? new_then[0] : std::make_shared<SeqStmts>(new_then, if_stmt->span_);

    std::optional<StmtPtr> new_else;
    if (const auto& else_body_opt = if_stmt->else_body_; else_body_opt.has_value()) {
      const StmtPtr& else_body = else_body_opt.value();
      auto else_flat = std::vector<StmtPtr>();
      if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(else_body)) {
        else_flat = seq->stmts_;
      } else {
        else_flat.push_back(else_body);
      }
      auto new_else_stmts = ProcessStmts(else_flat, mode, split_dim, tile_vars, is_aiv, subblock_idx,
                                         var_replacements, lane_stride);
      new_else = (new_else_stmts.size() == 1) ? new_else_stmts[0]
                                              : std::make_shared<SeqStmts>(new_else_stmts, if_stmt->span_);
    }
    // Repair the merge the same way the ForStmt arm repairs its carry: a merge
    // variable whose branches both yield a lane-local value is lane-local too.
    // Left alone it contradicts both Yield values AND stays untracked, so a
    // following tile.store gets no lane offset and both AIV lanes write from
    // output row 0.
    auto new_return_vars = RepairIfReturnVars(if_stmt->return_vars_, new_then_body, new_else, tile_vars,
                                              var_replacements, subblock_idx, lane_stride, if_stmt->span_);
    auto new_if = MutableCopy(if_stmt);
    new_if->then_body_ = new_then_body;
    new_if->else_body_ = new_else;
    new_if->return_vars_ = new_return_vars;
    return new_if;
  }

  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    auto new_stmts = ProcessStmts(seq->stmts_, mode, split_dim, tile_vars, is_aiv, subblock_idx,
                                  var_replacements, lane_stride);
    return std::make_shared<SeqStmts>(new_stmts, seq->span_);
  }

  return stmt;
}

std::string ReserveFreshName(std::unordered_set<std::string>& used_names, const std::string& base_name) {
  std::string name = base_name;
  if (used_names.count(name) != 0) {
    name = auto_name::GenerateFreshNameLike(base_name, used_names);
  }
  used_names.insert(name);
  return name;
}

}  // namespace

// Shared loop/branch carry repair. Extracted so the AUTO affinity-gated path in
// lower_auto_vector_split_pass.cpp gets the same treatment as ProcessStmts: it
// recurses into compound bodies itself (only VECTOR leaves may be halved), so it
// cannot reach ProcessStmt's ForStmt branch and previously left every iter_arg
// full width and untracked. A legal tile accumulator -- halved init, tile.add on
// the carry -- was then reported as carrying a full-width operand.
std::vector<IterArgPtr> RepairIterArgs(const std::vector<IterArgPtr>& iter_args,
                                       std::unordered_map<const Var*, TileInfo>& tile_vars,
                                       std::unordered_map<const Var*, VarPtr>& var_replacements,
                                       const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  std::vector<IterArgPtr> new_iter_args;
  new_iter_args.reserve(iter_args.size());
  // Propagate tile_vars from init values to iter_args BEFORE processing the body:
  // an iter_arg carries its init into the loop, so a tracked halved init makes the
  // carry lane-local too, and operations on it inside the body must see that.
  for (const auto& ia : iter_args) {
    auto new_init_value = ia->initValue_;
    if (new_init_value && !var_replacements.empty()) {
      new_init_value = transform_utils::Substitute(new_init_value, var_replacements);
    }
    TypePtr new_type = ia->GetType();
    bool has_tracked_tile = false;
    TileInfo tracked_info;
    if (ia->initValue_) {
      // Through OperandSplitInfo, so an init that is an INLINE tuple projection --
      // `pl.range(2, init_values=(pair[1],))` -- is tracked like a bound one. Matching
      // only Var left the carry, the loop exit, and every consumer after the loop at
      // full width while Substitute halved the init underneath them.
      if (auto info = OperandSplitInfo(ia->initValue_, tile_vars, var_replacements)) {
        has_tracked_tile = true;
        tracked_info = *info;
        tile_vars[ia.get()] = *info;
        new_type = ApplyTrackedTileShape(ia->GetType(), info->split_dim, info->half_dim_size, subblock_idx,
                                         lane_stride);
      } else if (auto init_var = AsVarLike(ia->initValue_)) {
        // A whole-TUPLE carry. Its init was halved, and the Substitute above already
        // swapped it in, so leaving the carry's declared type alone would put a
        // per-lane init under a full-width carry. Tuple vars are never in
        // ``tile_vars`` -- they are not tiles -- so adopt the init's halved type,
        // whose elements already carry the per-element split.
        if (auto replaced = var_replacements.find(init_var.get()); replaced != var_replacements.end()) {
          if (auto halved = std::dynamic_pointer_cast<const TupleType>(replaced->second->GetType())) {
            new_type = halved;
          }
        }
      }
    }
    if (new_type != ia->GetType() || new_init_value != ia->initValue_) {
      auto new_iter_arg = std::make_shared<IterArg>(ia->name_hint_, new_type, new_init_value, ia->span_);
      new_iter_args.push_back(new_iter_arg);
      var_replacements[ia.get()] = new_iter_arg;
      if (has_tracked_tile) tile_vars[new_iter_arg.get()] = tracked_info;
    } else {
      new_iter_args.push_back(ia);
    }
  }
  return new_iter_args;
}

// Loop-exit versions of the carries: a return_var inherits its iter_arg's tile
// info so a downstream tile.store gets the per-lane offset.
std::vector<VarPtr> RepairReturnVars(const std::vector<VarPtr>& return_vars,
                                     const std::vector<IterArgPtr>& new_iter_args,
                                     std::unordered_map<const Var*, TileInfo>& tile_vars,
                                     std::unordered_map<const Var*, VarPtr>& var_replacements,
                                     const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  std::vector<VarPtr> new_return_vars = return_vars;
  for (size_t i = 0; i < new_iter_args.size() && i < new_return_vars.size(); ++i) {
    auto it = tile_vars.find(new_iter_args[i].get());
    if (it == tile_vars.end()) {
      // The loop-exit version of a TUPLE carry: not in tile_vars, but the carry now
      // declares the halved tuple, and the return var must agree or the projections
      // after the loop read full-width element types off a per-lane value.
      auto carried = std::dynamic_pointer_cast<const TupleType>(new_iter_args[i]->GetType());
      if (!carried || structural_equal(carried, new_return_vars[i]->GetType())) continue;
      auto new_return_var =
          std::make_shared<Var>(new_return_vars[i]->name_hint_, carried, new_return_vars[i]->span_);
      var_replacements[return_vars[i].get()] = new_return_var;
      new_return_vars[i] = new_return_var;
      continue;
    }
    tile_vars[new_return_vars[i].get()] = it->second;
    auto new_type = ApplyTrackedTileShape(new_return_vars[i]->GetType(), it->second.split_dim,
                                          it->second.half_dim_size, subblock_idx, lane_stride);
    if (new_type != new_return_vars[i]->GetType()) {
      auto new_return_var =
          std::make_shared<Var>(new_return_vars[i]->name_hint_, new_type, new_return_vars[i]->span_);
      new_return_vars[i] = new_return_var;
      tile_vars[new_return_var.get()] = it->second;
      var_replacements[return_vars[i].get()] = new_return_var;
    }
  }
  return new_return_vars;
}

namespace {

// The tile info a branch's Yield carries for merge slot ``index``, or nullopt
// when that slot is not lane-local. A Yield value is left referencing the
// pre-halving var (the trailing Substitute rewrites it later), and the halving
// path registers BOTH the old and the new var in ``tile_vars`` -- so looking the
// yielded value up there is what tells the two apart.
std::optional<TileInfo> YieldedTileInfo(const YieldStmtPtr& yield, size_t index,
                                        const std::unordered_map<const Var*, TileInfo>& tile_vars,
                                        const std::unordered_map<const Var*, VarPtr>& var_replacements) {
  if (!yield || index >= yield->value_.size()) return std::nullopt;
  // Through OperandSplitInfo, so an INLINE tuple projection on the backedge is seen
  // for what it is. Matching only Var got this wrong in BOTH directions: a full-width
  // carry fed `pl.yield_(pair[1])` looked like "neither side is lane-local" and was
  // waved through, putting a half-width value under a full-width declared type with no
  // lane offset on the loop exit; and a legitimately halved carry fed the same
  // projection was refused as a width disagreement that did not exist.
  //
  // An inline CALL (`pl.yield_(pl.tile.add(acc, acc))`) still answers nullopt, which is
  // right: this pass halves statements, so a call inline in a Yield is never halved and
  // the carry mismatch it produces must stay rejected.
  return OperandSplitInfo(yield->value_[index], tile_vars, var_replacements);
}

// The halved TupleType a branch yields at @p index, or nullptr when that position is
// not a tuple the split rewrote.
//
// Tuple vars never enter ``tile_vars`` -- they are not tiles -- so the TileInfo lookup
// above cannot see them, and a merge whose branches both produce a tuple would keep its
// original full-width declared type while both Yields carry halved ones.
std::shared_ptr<const TupleType> YieldedHalvedTupleType(
    const YieldStmtPtr& yield, size_t index, const std::unordered_map<const Var*, VarPtr>& var_replacements) {
  if (!yield || index >= yield->value_.size()) return nullptr;
  auto value_var = AsVarLike(yield->value_[index]);
  auto it = value_var ? var_replacements.find(value_var.get()) : var_replacements.end();
  if (it == var_replacements.end()) return nullptr;
  return std::dynamic_pointer_cast<const TupleType>(it->second->GetType());
}

bool SameTileInfo(const TileInfo& a, const TileInfo& b) {
  return a.split_dim == b.split_dim && structural_equal(a.half_dim_size, b.half_dim_size);
}

}  // namespace

void ValidateCarryBackedge(const StmtPtr& new_body, const std::vector<IterArgPtr>& new_iter_args,
                           const std::unordered_map<const Var*, TileInfo>& tile_vars,
                           const std::unordered_map<const Var*, VarPtr>& var_replacements, const Span& span) {
  if (new_iter_args.empty()) return;
  auto yield = transform_utils::GetLastYieldStmt(new_body);
  // A loop that carries state ends in a Yield feeding every slot; anything else
  // is malformed IR the SSA verifier rejects, so there is nothing to compare.
  if (!yield || yield->value_.size() != new_iter_args.size()) return;

  for (size_t i = 0; i < new_iter_args.size(); ++i) {
    // A TUPLE carry is validated on its declared type, not on tile_vars -- tuple vars
    // never enter it, so without this the backedge of a halved tuple carry went
    // completely unchecked and a full-width yield sailed through.
    if (auto carried_tuple = std::dynamic_pointer_cast<const TupleType>(new_iter_args[i]->GetType())) {
      auto yielded_tuple = YieldedHalvedTupleType(yield, i, var_replacements);
      const bool agrees = yielded_tuple ? structural_equal(yielded_tuple, carried_tuple)
                                        : structural_equal(yield->value_[i]->GetType(), carried_tuple);
      CHECK_SPAN(agrees, span)
          << "LowerAutoVectorSplit: the tuple loop carry '" << new_iter_args[i]->name_hint_
          << "' inside the automatically split vector region declares a per-lane type that the value "
             "yielded back into it does not match. The carry's declared tuple would contradict the "
             "value flowing into it on every iteration after the first. Derive both the same way -- "
             "produce the tuple inside the split region on both the init and the backedge, or keep "
             "both shared by the two lanes -- or move the loop outside the automatically split region.";
      continue;
    }

    auto carry_it = tile_vars.find(new_iter_args[i].get());
    const bool carry_is_lane_local = carry_it != tile_vars.end();
    auto yielded = YieldedTileInfo(yield, i, tile_vars, var_replacements);

    if (!carry_is_lane_local && !yielded.has_value()) continue;
    if (carry_is_lane_local && yielded.has_value() && SameTileInfo(carry_it->second, *yielded)) continue;

    // An UNBOUND yielded expression is its own diagnosis, and by far the more common
    // one: this pass halves STATEMENTS, and the parser does not hoist an expression
    // passed to pl.yield_, so `pl.yield_(pl.tile.add(acc, acc))` leaves the call inline
    // in the Yield where nothing ever halves it. Saying the carry and the backedge were
    // "derived differently" sends the author looking at the two ends of a carry that is
    // fine; the fix is to bind the value first.
    if (!AsVarLike(yield->value_[i]) &&
        !std::dynamic_pointer_cast<const TupleGetItemExpr>(yield->value_[i])) {
      CHECK_SPAN(false, span)
          << "LowerAutoVectorSplit: the loop carry '" << new_iter_args[i]->name_hint_
          << "' inside the automatically split vector region is fed an expression computed inline in "
             "the Yield. This pass halves statements, so a value that is never bound to a name is "
             "never halved, and it cannot match a per-lane carry. Bind it first -- `tmp = <expr>` on "
             "its own line, then `pl.yield_(tmp)` -- or move the loop outside the automatically split "
             "region.";
    }

    auto yielded_var = AsVarLike(yield->value_[i]);
    const std::string yielded_name =
        yielded_var ? " '" + yielded_var->name_hint_ + "'" : std::string(" the yielded value");
    CHECK_SPAN(false, span)
        << "LowerAutoVectorSplit: the loop carry '" << new_iter_args[i]->name_hint_
        << "' inside the automatically split vector region "
        << (carry_is_lane_local
                ? "is lane-local (the split halved its init value), but the body yields" + yielded_name +
                      " back into it, which no lane owns a half of. The carry's declared per-lane type "
                      "would contradict the value flowing into it on every iteration after the first."
                : "is full width, but the body yields" + yielded_name +
                      " back into it, which the split made lane-local. The carry cannot hold a per-lane "
                      "value under a full-width declared type.")
        << " Derive both the same way -- load or slice the value inside the split region on both the "
           "init and the backedge, or keep both shared by the two lanes -- or move the loop outside the "
           "automatically split region.";
  }
}

std::vector<VarPtr> RepairIfReturnVars(const std::vector<VarPtr>& return_vars, const StmtPtr& new_then_body,
                                       const std::optional<StmtPtr>& new_else_body,
                                       std::unordered_map<const Var*, TileInfo>& tile_vars,
                                       std::unordered_map<const Var*, VarPtr>& var_replacements,
                                       const ExprPtr& subblock_idx, const ExprPtr& lane_stride,
                                       const Span& span) {
  std::vector<VarPtr> new_return_vars = return_vars;
  if (return_vars.empty()) return new_return_vars;

  // An IfStmt that defines return_vars must have an else branch -- SSAVerifier
  // rejects the else-less shape outright, and ConvertToSSA synthesizes the else
  // Yield for a source-level no-else phi. So both definitions exist by the time
  // this pass runs, and a missing one is a compiler bug rather than a shape to
  // support.
  INTERNAL_CHECK_SPAN(new_else_body.has_value(), span)
      << "Internal error: LowerAutoVectorSplit found an IfStmt defining " << new_return_vars.size()
      << " return var(s) with no else branch; SSA requires both branches to define the merge.";

  auto then_yield = transform_utils::GetLastYieldStmt(new_then_body);
  auto else_yield = transform_utils::GetLastYieldStmt(*new_else_body);

  for (size_t i = 0; i < new_return_vars.size(); ++i) {
    // A TUPLE merge. Its element types carry the per-element split, so adopting the
    // yielded type is the whole repair -- there is no single axis to record, and the
    // projections downstream read each element's axis back off this type.
    auto then_tuple = YieldedHalvedTupleType(then_yield, i, var_replacements);
    auto else_tuple = YieldedHalvedTupleType(else_yield, i, var_replacements);
    if (then_tuple || else_tuple) {
      CHECK_SPAN(then_tuple && else_tuple && structural_equal(then_tuple, else_tuple), span)
          << "LowerAutoVectorSplit: the branches of an if inside the automatically split vector region "
             "disagree about tuple merge variable '"
          << new_return_vars[i]->name_hint_
          << "': the split halved the tuple one branch yields differently from the other's (or halved "
             "only one of them). There is no single per-lane type for the merged tuple, and picking "
             "either one silently gives one AIV lane the wrong extent on some element. Make both "
             "branches produce the tuple the same way, or move the if outside the automatically split "
             "region.";
      auto new_return_var =
          std::make_shared<Var>(new_return_vars[i]->name_hint_, then_tuple, new_return_vars[i]->span_);
      var_replacements[return_vars[i].get()] = new_return_var;
      new_return_vars[i] = new_return_var;
      continue;
    }

    auto then_info = YieldedTileInfo(then_yield, i, tile_vars, var_replacements);
    auto else_info = YieldedTileInfo(else_yield, i, tile_vars, var_replacements);

    if (!then_info.has_value() && !else_info.has_value()) continue;
    CHECK_SPAN(then_info.has_value() && else_info.has_value() && SameTileInfo(*then_info, *else_info), span)
        << "LowerAutoVectorSplit: the branches of an if inside the automatically split vector region "
           "disagree about merge variable '"
        << new_return_vars[i]->name_hint_
        << "': one yields a value the split made lane-local while the other does not (or they are split "
           "along different axes). There is no single per-lane type for the merged value, and picking "
           "either one silently gives one AIV lane the wrong extent. Make both branches yield values "
           "derived the same way -- load or slice inside the region on both sides, or keep the value "
           "shared by both lanes on both sides -- or move the if outside the automatically split region.";

    tile_vars[new_return_vars[i].get()] = *then_info;
    auto new_type = ApplyTrackedTileShape(new_return_vars[i]->GetType(), then_info->split_dim,
                                          then_info->half_dim_size, subblock_idx, lane_stride);
    if (new_type != new_return_vars[i]->GetType()) {
      auto new_return_var =
          std::make_shared<Var>(new_return_vars[i]->name_hint_, new_type, new_return_vars[i]->span_);
      new_return_vars[i] = new_return_var;
      tile_vars[new_return_var.get()] = *then_info;
      var_replacements[return_vars[i].get()] = new_return_var;
    }
  }
  return new_return_vars;
}

std::vector<StmtPtr> ProcessStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_dim,
                                  std::unordered_map<const Var*, TileInfo>& tile_vars, bool is_aiv,
                                  const ExprPtr& subblock_idx,
                                  std::unordered_map<const Var*, VarPtr>& var_replacements,
                                  const ExprPtr& lane_stride) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    result.push_back(
        ProcessStmt(stmt, mode, split_dim, tile_vars, is_aiv, subblock_idx, var_replacements, lane_stride));
  }
  return result;
}

SubblockInjectionResult InjectSubblockIdx(const FunctionPtr& func, bool is_aiv) {
  std::vector<StmtPtr> body_stmts;
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(func->body_)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(func->body_);
  }

  std::unordered_set<std::string> used_names;
  for (const auto& p : func->params_) {
    used_names.insert(p->name_hint_);
  }
  std::vector<VarPtr> def_vars;
  transform_utils::CollectDefVars(func->body_, def_vars);
  for (const auto& v : def_vars) {
    used_names.insert(v->name_hint_);
  }

  if (!is_aiv) {
    return {nullptr, std::move(body_stmts), std::move(used_names)};
  }

  auto idx_type = std::make_shared<ScalarType>(DataType::INDEX);
  std::string subblock_var_name = ReserveFreshName(used_names, "subblock_idx");

  auto& op_reg = OpRegistry::GetInstance();
  auto subblock_op = op_reg.GetOp("tile.get_subblock_idx");
  auto subblock_call =
      std::make_shared<Call>(subblock_op, std::vector<ExprPtr>{},
                             std::vector<std::pair<std::string, std::any>>{}, idx_type, func->span_);
  auto subblock_idx_var = std::make_shared<Var>(subblock_var_name, idx_type, func->span_);
  auto assign_stmt = std::make_shared<AssignStmt>(subblock_idx_var, subblock_call, func->span_);
  body_stmts.insert(body_stmts.begin(), assign_stmt);
  return {subblock_idx_var, std::move(body_stmts), std::move(used_names)};
}

SubblockInjectionResult InjectSubblockIdxIntoStmts(const std::vector<StmtPtr>& region_stmts,
                                                   const std::unordered_set<std::string>& used_names) {
  // An empty region (DCE-emptied, or a ``pass``-only body whose sole binding was
  // dropped) carries no compute to localize, so there is nothing to inject a
  // per-lane index for. Return a no-op result (null index, empty body) the caller
  // splices as nothing — i.e. it erases the region rather than crashing.
  if (region_stmts.empty()) {
    return {nullptr, {}, used_names};
  }
  std::vector<StmtPtr> body_stmts = region_stmts;

  // Seed the name set with the caller-supplied names plus the region's own def
  // vars so the injected subblock index never clashes with an existing binding.
  std::unordered_set<std::string> names = used_names;
  for (const auto& s : region_stmts) {
    std::vector<VarPtr> def_vars;
    transform_utils::CollectDefVars(s, def_vars);
    for (const auto& v : def_vars) {
      names.insert(v->name_hint_);
    }
  }

  const Span& span = region_stmts.front()->span_;
  auto idx_type = std::make_shared<ScalarType>(DataType::INDEX);
  std::string subblock_var_name = ReserveFreshName(names, "subblock_idx");

  auto& op_reg = OpRegistry::GetInstance();
  auto subblock_op = op_reg.GetOp("tile.get_subblock_idx");
  auto subblock_call = std::make_shared<Call>(
      subblock_op, std::vector<ExprPtr>{}, std::vector<std::pair<std::string, std::any>>{}, idx_type, span);
  auto subblock_idx_var = std::make_shared<Var>(subblock_var_name, idx_type, span);
  auto assign_stmt = std::make_shared<AssignStmt>(subblock_idx_var, subblock_call, span);
  body_stmts.insert(body_stmts.begin(), assign_stmt);
  return {subblock_idx_var, std::move(body_stmts), std::move(names)};
}

namespace {

// Scan for the balanced partition stride (see ResolveLaneStride in the header).
//
// Walks the PRE-lowering body in order, tracking which vars carry
// boundary-derived data. It answers one question — "does every split-axis tile
// in this body come from one ragged Cube -> Vector boundary?" — and any doubt
// (a second boundary shape, a gather, an independently split tile, a dynamic
// extent) makes it decline, leaving the universal box partition in place.
class LaneStrideScanner : public IRVisitor {
 public:
  explicit LaneStrideScanner(int split_dim) : split_dim_(split_dim) {}

  /// The boundary's ``(box, valid)`` on the split axis, or nullopt to decline.
  [[nodiscard]] std::optional<std::pair<int64_t, int64_t>> GetBoundary() const {
    if (blocked_ || !boundary_.has_value()) return std::nullopt;
    return boundary_;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    Consider(As<Call>(op->value_), op->var_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    Consider(As<Call>(op->expr_), nullptr);
    IRVisitor::VisitStmt_(op);
  }

 private:
  void Consider(const CallPtr& call, const VarPtr& result) {
    if (blocked_ || !call || !call->op_) return;

    // The Vector -> Cube direction re-joins the lanes positionally at their own
    // extents, which only abut when the two are equal.
    if (IsOp(call, "tile.aic_gather") || IsOp(call, "tile.tpush_to_aic") ||
        IsOp(call, "tile.tpop_from_aiv") ||
        core_affinity::ClassifyMoveDirection(call) == core_affinity::CVDirection::VECTOR_TO_CUBE) {
      blocked_ = true;
      return;
    }

    // Cube -> Vector boundary: the tile it partitions is the cube-side operand
    // (a move / shard) or the popped result (an already-lowered tpop).
    const bool is_move_boundary =
        core_affinity::ClassifyMoveDirection(call) == core_affinity::CVDirection::CUBE_TO_VECTOR;
    if (is_move_boundary || IsOp(call, "tile.aiv_shard") || IsOp(call, "tile.tpop_from_aic")) {
      auto full_type = IsOp(call, "tile.tpop_from_aic")
                           ? call->GetType()
                           : (call->args_.empty() ? nullptr : call->args_[0]->GetType());
      RecordBoundary(full_type, result);
      return;
    }

    // Everything else that produces a tile split on this axis must derive from
    // the boundary; a fresh one (a load, a generator) spans the full box.
    auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
    if (!tt || split_dim_ >= static_cast<int>(tt->shape_.size())) return;
    if (IsSingletonDim(tt->shape_[split_dim_])) return;
    // Cube-affine ops keep their full width (the affinity gate never halves
    // them), so they neither derive from nor conflict with the partition.
    if (core_affinity::ClassifyCallAffinity(call) == core_affinity::CoreAffinity::CUBE) return;

    // EVERY operand the split axis partitions must come from the boundary, not
    // just one of them: `tile.add(shard, vec_param)` mixes a boundary-derived
    // half with a full-width parameter, and a balanced stride would give the
    // two operands different rows. A singleton operand is replicated across the
    // lanes rather than partitioned, so it neither derives nor conflicts.
    bool partitioned_operand = false;
    for (const auto& arg : call->args_) {
      auto arg_tt = std::dynamic_pointer_cast<const TileType>(arg->GetType());
      if (!arg_tt || split_dim_ >= static_cast<int>(arg_tt->shape_.size())) continue;
      if (IsSingletonDim(arg_tt->shape_[split_dim_])) continue;
      partitioned_operand = true;
      auto var = AsVarLike(arg);
      if (!var || derived_.count(var.get()) == 0) {
        blocked_ = true;
        return;
      }
    }
    // No partitioned operand at all: this op mints its own split-axis tile.
    if (!partitioned_operand) {
      blocked_ = true;
      return;
    }
    if (result) derived_.insert(result.get());
  }

  void RecordBoundary(const TypePtr& full_type, const VarPtr& result) {
    auto tt = std::dynamic_pointer_cast<const TileType>(full_type);
    auto extents = ComputeLaneExtents(tt, split_dim_, /*lane_stride=*/nullptr);
    if (!extents.has_value()) {
      blocked_ = true;
      return;
    }
    auto box = std::dynamic_pointer_cast<const ConstInt>(tt->shape_[split_dim_]);
    const std::pair<int64_t, int64_t> shape{box->value_, extents->valid};
    if (boundary_.has_value() && *boundary_ != shape) {
      blocked_ = true;  // two boundaries of different shape: no single partition
      return;
    }
    boundary_ = shape;
    if (result) derived_.insert(result.get());
  }

  int split_dim_;
  bool blocked_ = false;
  std::optional<std::pair<int64_t, int64_t>> boundary_;
  std::unordered_set<const Var*> derived_;
};

}  // namespace

ExprPtr ResolveLaneStride(const std::vector<StmtPtr>& stmts, int split_dim) {
  if (split_dim < 0) return nullptr;
  LaneStrideScanner scanner(split_dim);
  for (const auto& stmt : stmts) {
    scanner.VisitStmt(stmt);
  }
  auto boundary = scanner.GetBoundary();
  if (!boundary.has_value()) return nullptr;
  const auto [box, valid] = *boundary;
  const int64_t balanced = (valid + 1) / 2;
  // Nothing to rebalance when the balanced stride IS the box half — a fully
  // valid boundary, or one short by a single cell. Leaving the stride null
  // there keeps the IR of every such kernel byte-identical to before. An EMPTY
  // boundary (valid == 0) has no partition to balance either, and a zero stride
  // is not a legal one: keep the box partition, where both lanes are empty.
  if (balanced <= 0 || valid >= box || balanced == (box + 1) / 2) return nullptr;
  return std::make_shared<ConstInt>(balanced, DataType::INDEX, Span::unknown());
}

// The CEIL half, so an odd extent 2k+1 gives both lanes the same (k+1)-cell BOX
// and lets the per-lane valid extent carry the raggedness (lane 0 fills k+1,
// lane 1 fills k). That mirrors pto-isa's TILE_UP_DOWN_ODD /
// TILE_LEFT_RIGHT_ODD ("AIV0 = rows/2 + 1, AIV1 = rows/2"), which derives lane
// 1's slot offset from the lanes' RUNTIME valid extents. An even extent is
// unaffected: ceil(2k / 2) == k.
//
// A dynamic extent keeps floordiv(dim, 2) — its parity is unknown at compile
// time, so no odd split code can be stamped for it either (see ShardSplitCode).
ExprPtr ComputeHalfDimSize(const ExprPtr& dim_size) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size)) {
    return std::make_shared<ConstInt>((ci->value_ + 1) / 2, ci->dtype(), ci->span_);
  }
  auto two = std::make_shared<ConstInt>(2, GetScalarDtype(dim_size), dim_size->span_);
  return MakeFloorDiv(dim_size, two, dim_size->span_);
}

StmtPtr LocalizeReturnStores(const std::shared_ptr<const ReturnStmt>& ret,
                             const std::unordered_map<const Var*, TileInfo>& tile_vars,
                             const std::unordered_map<const Var*, VarPtr>& var_replacements,
                             const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  std::vector<ExprPtr> new_values = ret->value_;
  bool changed = false;
  for (auto& value : new_values) {
    auto call = std::dynamic_pointer_cast<const Call>(value);
    if (!call || !call->op_) continue;
    if (auto localized = LocalizeStoreOffset(call, tile_vars, var_replacements, subblock_idx, lane_stride)) {
      value = localized;
      changed = true;
    }
  }
  if (!changed) return nullptr;
  return std::make_shared<ReturnStmt>(std::move(new_values), ret->span_, ret->leading_comments_);
}

// The split the pass has already applied to ONE OPERAND, whatever expression carries
// it, or nullopt when the operand is not lane-local.
//
// This is the single answer to "did the split partition this operand, and along which
// axis" and every consumer must go through it. A bound operand is looked up in
// ``tile_vars``; an INLINE tuple projection -- `pl.tile.store(pair[0], ...)`,
// `pl.tile.add(pair[1], pair[1])`, which the DSL emits verbatim because nothing hoists
// a projection into its own binding -- is not a Var and never appears there, so its
// axis is read back off the halved tuple type instead.
//
// Matching only Var here is a SILENT wrong answer, not a missed optimization: the
// operand still gets substituted for its halved replacement afterwards, so the
// consuming node keeps a full-width declared type over per-lane data.
std::optional<TileInfo> OperandSplitInfo(const ExprPtr& arg,
                                         const std::unordered_map<const Var*, TileInfo>& tile_vars,
                                         const std::unordered_map<const Var*, VarPtr>& var_replacements) {
  if (!arg) return std::nullopt;
  if (auto arg_var = AsVarLike(arg)) {
    auto it = tile_vars.find(arg_var.get());
    return it != tile_vars.end() ? std::optional<TileInfo>{it->second} : std::nullopt;
  }
  if (auto get_item = std::dynamic_pointer_cast<const TupleGetItemExpr>(arg)) {
    auto tuple_var = AsVarLike(get_item->tuple_);
    auto it = tuple_var ? var_replacements.find(tuple_var.get()) : var_replacements.end();
    if (it == var_replacements.end()) return std::nullopt;
    auto halved = std::dynamic_pointer_cast<const TupleType>(it->second->GetType());
    const auto index = static_cast<size_t>(get_item->index_);
    if (!halved || get_item->index_ < 0 || index >= halved->types_.size()) return std::nullopt;
    return SplitInfoFromHalvedType(get_item->GetType(), halved->types_[index]);
  }
  return std::nullopt;
}

// The replacement for one operand, or nullptr when nothing replaces it.
//
// A bound operand is replaced directly. An INLINE tuple projection is not in the
// map at all -- the tuple underneath it is -- so rebuild the projection over the
// replacement, which re-derives the element type from the halved tuple.
ExprPtr ReplacedOperand(const ExprPtr& arg, const std::unordered_map<const Var*, VarPtr>& var_replacements) {
  if (auto arg_var = AsVarLike(arg)) {
    auto it = var_replacements.find(arg_var.get());
    return it != var_replacements.end() ? it->second : nullptr;
  }
  if (auto get_item = std::dynamic_pointer_cast<const TupleGetItemExpr>(arg)) {
    auto tuple_var = AsVarLike(get_item->tuple_);
    auto it = tuple_var ? var_replacements.find(tuple_var.get()) : var_replacements.end();
    if (it == var_replacements.end()) return nullptr;
    return std::make_shared<TupleGetItemExpr>(it->second, get_item->index_, get_item->span_);
  }
  return nullptr;
}

StmtPtr RetypeTupleProjection(const std::shared_ptr<const AssignStmt>& assign,
                              std::unordered_map<const Var*, TileInfo>& tile_vars,
                              std::unordered_map<const Var*, VarPtr>& var_replacements) {
  auto get_item = std::dynamic_pointer_cast<const TupleGetItemExpr>(assign->value_);
  if (!get_item) return nullptr;
  auto tuple_var = AsVarLike(get_item->tuple_);
  auto replaced = tuple_var ? var_replacements.find(tuple_var.get()) : var_replacements.end();
  if (replaced == var_replacements.end()) return nullptr;

  // The halved tuple type already carries the per-element mapping its producing call
  // discovered, so rebuilding the access against it is what gives this var its
  // per-lane element type -- TupleGetItemExpr derives its own type from the tuple.
  auto new_get = std::make_shared<TupleGetItemExpr>(replaced->second, get_item->index_, get_item->span_);
  auto new_var = std::make_shared<Var>(assign->var_->name_hint_, new_get->GetType(), assign->var_->span_);
  auto before = std::dynamic_pointer_cast<const TileType>(assign->var_->GetType());
  auto after = std::dynamic_pointer_cast<const TileType>(new_get->GetType());
  INTERNAL_CHECK_SPAN(before && after && before->shape_.size() == after->shape_.size(), assign->span_)
      << "Internal error: halving a tuple result changed element " << get_item->index_ << " of '"
      << (tuple_var ? tuple_var->name_hint_ : "") << "' from " << assign->var_->GetType()->TypeName()
      << " to " << new_get->GetType()->TypeName();

  // Track the axis that moved. It is what a later tile.store needs in order to give
  // each lane its own destination, and the elements of one tuple need not agree on it:
  // tile.gather_compare answers a row split with `dst` halved on dim 0 and `cdst`
  // ([1, rows]) halved on dim 1.
  if (auto info = SplitInfoFromHalvedType(assign->var_->GetType(), new_get->GetType())) {
    tile_vars[assign->var_.get()] = *info;
    tile_vars[new_var.get()] = *info;
  }
  var_replacements[assign->var_.get()] = new_var;
  return std::make_shared<AssignStmt>(new_var, new_get, assign->span_);
}

std::optional<std::pair<int64_t, int64_t>> StaticLaneExtents(const TypePtr& full_type, int split_dim,
                                                             const ExprPtr& lane_stride) {
  auto extents =
      ComputeLaneExtents(std::dynamic_pointer_cast<const TileType>(full_type), split_dim, lane_stride);
  if (!extents.has_value()) return std::nullopt;
  return std::make_pair(extents->lane0, extents->lane1);
}

bool HasStaticLaneExtents(const TypePtr& full_type, int split_dim, const ExprPtr& lane_stride) {
  return StaticLaneExtents(full_type, split_dim, lane_stride).has_value();
}

int ShardSplitCode(SplitMode mode, const TypePtr& full_type, int split_dim, const ExprPtr& lane_stride,
                   const std::string& op_name, const Span& span) {
  if (mode == SplitMode::None) return kSplitNone;
  auto tt = std::dynamic_pointer_cast<const TileType>(full_type);
  // What the FIFO can carry at all. The boundary ops are checked when their type
  // is deduced; a HAND-WRITTEN tile.tpush_to_aiv / tile.tpop_from_aic pair never
  // goes through that deduction, and reaches the transport here instead. Both
  // must obey the same contract, because it is pto-isa's geometry rather than a
  // pass's convention: the pop derives its GM row stride and its lane offset
  // from the popped tile's own valid extents, so a narrowed COLUMN extent
  // mis-strides the read against the producer's full-box pitch (measured on
  // a2a3: a [16, 16] boundary read back with valid_col 12 or 8 misses the
  // golden, while 16 matches).
  if (tt) {
    CheckSplitBoundaryCarriesValid(op_name, tt->shape_,
                                   tile_view_semantics::GetEffectiveTileView(*tt).valid_shape, split_dim,
                                   /*halve=*/true, span);
  }
  auto extents = ComputeLaneExtents(tt, split_dim, lane_stride);
  // No compile-time lane extents (a runtime box, valid extent or stride): the
  // even code, which is exact only when the boundary tile carries the FULL
  // split-axis box rather than the lane's extent. The producer lays each row out
  // at its box position (PTO codegen widens every split tpush's columns), so
  // lane 1's band sits at the box half and the even code points pto-isa there;
  // which lane holds how much data is carried by the tile's CONSUMERS, where no
  // band offset depends on it. LocalizeExplicitBoundaryValid establishes that
  // pairing for a pl.split_aiv region's boundary op, and for a pl.split body
  // that holds one (WithFullSplitAxisValid).
  //
  // The pairing is what makes this safe. The even code beside a PER-LANE extent
  // silently mis-places lane 1: a 16-row box valid to 12 leaves the lanes 8 and
  // 4, and pto-isa reads lane 1's band at row 4 while the producer wrote it at
  // row 8. A hand-written tile.tpop_from_aic still pairs this way and is still
  // wrong for such an extent -- widening it alone regresses the device (see
  // RebuildTpopWithHalvedShape). It went unnoticed for as long as it did
  // because tests/st/runtime/cross_core/test_cross_core_split_parity.py fed
  // both operands uniform constants, which makes every row and column of the
  // product identical and any band offset indistinguishable.
  if (!extents.has_value()) return SplitCodeFor(mode, /*odd_extent=*/false);

  if (extents->lane0 == extents->lane1) return SplitCodeFor(mode, /*odd_extent=*/false);
  if (extents->lane0 == extents->lane1 + 1) return SplitCodeFor(mode, /*odd_extent=*/true);
  // Lane 1 pops nothing, so pto-isa never dereferences its band — the even code
  // is exact whatever lane 0 holds. This is the empty-tail shard (a valid extent
  // that does not reach the second lane at all).
  if (extents->lane1 == 0) return SplitCodeFor(mode, /*odd_extent=*/false);

  // Only the BOX partition can land here: the balanced stride is ceil(V / 2) by
  // construction, so its lanes never differ by more than one. ResolveLaneStride
  // declined to rebalance this body, and the reason is what the fix must target.
  CHECK_SPAN(false, span)
      << op_name << ": the Cube -> Vector boundary tile's valid split-axis extent (" << extents->valid
      << " of a " << PythonPrint(tt->shape_[split_dim]) << "-wide dim " << split_dim
      << ") leaves the two AIV lanes " << extents->lane0 << " and " << extents->lane1
      << " cells. pto-isa places lane 1's band at its own extent (TILE_UP_DOWN / TILE_LEFT_RIGHT) or one "
         "past it (the _ODD modes), so the lanes must be equal, differ by exactly one, or leave lane 1 "
         "empty. The compiler balances a ragged boundary across the lanes automatically, but only when "
         "the whole split body derives from it — this one also holds an independently split value (a "
         "tile.load, a generator, a full-width operand) or a Vector -> Cube gather, which spans the full "
         "box. Either move that value out of the split scope, or widen the valid extent to "
      << (2 * extents->box_half) << " / " << (2 * extents->box_half - 1)
      << " with pl.tile.set_validshape(...), or narrow it to at most " << extents->box_half
      << " so the whole value stays on lane 0.";
  return kSplitNone;  // unreachable: CHECK_SPAN(false, ...) always throws
}

int GatherSplitCode(SplitMode mode, const TypePtr& full_type, int split_dim, const std::string& op_name,
                    const Span& span) {
  if (mode == SplitMode::None) return kSplitNone;
  auto tt = std::dynamic_pointer_cast<const TileType>(full_type);
  // A gather is never rebalanced (ResolveLaneStride declines a body that holds
  // one), so its lanes always sit on the box partition.
  auto extents = ComputeLaneExtents(tt, split_dim, /*lane_stride=*/nullptr);
  if (extents.has_value() && extents->lane0 != extents->lane1 && extents->lane1 != 0) {
    CHECK_SPAN(false, span)
        << op_name << ": the Vector -> Cube boundary tile's split axis (dim " << split_dim
        << ") leaves the two AIV lanes " << extents->lane0 << " and " << extents->lane1
        << " cells. Each lane hands the cube a band placed at its OWN extent, so the two only abut when "
           "the lanes are equal — the gather has no odd form. Pad that axis to "
        << (2 * extents->box_half) << " and narrow the cube-side result with pl.tile.set_validshape(...).";
  }
  return SplitCodeFor(mode, /*odd_extent=*/false);
}

TypePtr LocalizeShardValidForLane(const TypePtr& shard_type, const TypePtr& operand_type, int split_dim,
                                  const ExprPtr& subblock_idx, const ExprPtr& lane_stride) {
  auto shard = std::dynamic_pointer_cast<const TileType>(shard_type);
  auto operand = std::dynamic_pointer_cast<const TileType>(operand_type);
  if (!shard || !operand || !subblock_idx || split_dim < 0 ||
      split_dim >= static_cast<int>(shard->shape_.size()) ||
      split_dim >= static_cast<int>(operand->shape_.size())) {
    return shard_type;
  }
  const auto operand_valid = tile_view_semantics::GetEffectiveTileView(*operand).valid_shape;
  // A fully-valid EVEN split axis on the box partition needs no repair: both
  // lanes hold `half`, which is exactly what ReshapeSplitAxis already produced.
  // A fully-valid ODD axis still does (its lanes hold ceil and floor), and so
  // does any rebalanced body (its stride is narrower than the box half) — only
  // the lane index, in scope here, can tell the lanes apart.
  if (static_cast<int>(operand_valid.size()) <= split_dim ||
      (!lane_stride && AreExprsEqual(operand_valid[split_dim], operand->shape_[split_dim]) &&
       !IsOddSplitExtent(operand->shape_[split_dim]))) {
    return shard_type;
  }
  auto lane_extent = LocalizeValidDimForSplit(operand_valid[split_dim], operand->shape_[split_dim],
                                              LaneStep(lane_stride, shard->shape_[split_dim]), subblock_idx);
  return WithValidDim(shard, split_dim, lane_extent);
}

}  // namespace split_axis
}  // namespace ir
}  // namespace pypto
