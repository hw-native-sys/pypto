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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_CORE_AFFINITY_H_
#define PYPTO_IR_TRANSFORMS_UTILS_CORE_AFFINITY_H_

#include <optional>

#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace core_affinity {

enum class PipeDirection { C2V = 1, V2C = 2 };

enum class CoreSide { AIC, AIV };

enum class CVDirection { NONE, CUBE_TO_VECTOR, VECTOR_TO_CUBE };

constexpr int kDirMaskC2V = static_cast<int>(PipeDirection::C2V);
constexpr int kDirMaskV2C = static_cast<int>(PipeDirection::V2C);

bool IsCubeMemorySpace(MemorySpace ms);

std::optional<MemorySpace> GetFirstTileArgMemory(const CallPtr& call);

CVDirection ClassifyMoveDirection(const CallPtr& call);

/// True when the call's execution lane is STATED rather than inferred. Two ways
/// to state one:
///
///   * the registry declares it (`set_core_affinity`) — `pld.tile.put` / `get`,
///     the cross-core transfer ops, the SPMD index ops, `tile.create`;
///   * the author writes it as a `core_type` kwarg on a barrier or cross-core
///     event, which `ClassifyCallAffinity` dispatches on (rules 2a / 2b).
///
/// A stated lane outranks an inferred one, so both the `pl.split_aiv` region
/// placement override below and the AivSplitValid verifier's check (e) leave
/// such calls alone. A GlobalVar callee (cross-function Call / Submit) is not an
/// operator and states nothing.
bool HasStatedLane(const CallPtr& call);

/// True when this call was spliced out of a `pl.split_aiv` region by
/// LowerAutoVectorSplit, i.e. the author placed it on the vector lane. See
/// `kCorePlacementAttr` (transforms/utils/attrs.h) for the carrier and its
/// pass 20 -> pass 21 lifetime.
bool IsAivRegionPlaced(const CallPtr& call);

/// The lane this call runs on, BEFORE the `pl.split_aiv` region placement
/// override — i.e. what the op itself, its kwargs and its operand/result memory
/// spaces imply. `ClassifyCallAffinity` is this plus the override, and is what
/// passes should use; this form exists for the override to consult, and for
/// diagnostics that need to explain what changed.
CoreAffinity ClassifyIntrinsicCallAffinity(const CallPtr& call);

/// True when this call's operator declares `set_no_duplicate()`, i.e. running
/// it on a second core would change what the program means. False for a
/// GlobalVar callee (not an operator) and for unregistered names.
bool IsNoDuplicateCall(const CallPtr& call);

CoreAffinity ClassifyCallAffinity(const CallPtr& call);

struct CVBoundaryMove {
  CVDirection direction;
  VarPtr dest_var;
  ExprPtr source_tile;
  TypePtr result_type;
  // True when this boundary originates from an explicit split-reshape op
  // (tile.aiv_shard / tile.aic_gather) rather than a cross-C/V tile.move. The
  // op already encodes the half/full shape in result_type and the cross-core
  // fractal/post-move behaviour differs (see ExpandMixedKernel's boundary arm).
  bool op_driven = false;
  // Split MODE carried by the originating op (1 = UP_DOWN/axis0,
  // 2 = LEFT_RIGHT/axis1); 0 for tile.move boundaries (split assigned later by
  // SplitVectorKernel). The pto-isa split CODE stamped onto the generated
  // tpush/tpop is derived from this mode plus the boundary tile's extents —
  // see split_axis::ShardSplitCode / GatherSplitCode.
  int split = 0;
  // Partition stride the halving used, from the originating op's optional
  // `lane_stride` attr (see split_axis::ResolveLaneStride). 0 = the default box
  // partition, where the stride is the tile's own physical half.
  int lane_stride = 0;
};

}  // namespace core_affinity
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_CORE_AFFINITY_H_
