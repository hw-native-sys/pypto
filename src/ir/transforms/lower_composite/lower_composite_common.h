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

#ifndef SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_COMMON_H_
#define SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_COMMON_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace lower_composite {

// ============================================================================
// Self-clearing credit barrier — the shared, stateless barrier-signal protocol
// ============================================================================
//
// Every ``pld.tensor.*`` collective synchronises through one protocol:
//
//     Body:      barrier(1); barrier(2); ...; barrier(N)   # g counted within
//                                                           # this call only
//       barrier(g):
//         for peer != my_rank: notify(signal, peer, <my cell>, 1, op=AtomicAdd)
//         for src  != my_rank: wait  (signal, <src cell>, g,   cmp=Ge)
//
//     Epilogue:  for src != my_rank:
//                    notify(signal, my_rank, <src cell>, -N, op=AtomicAdd)
//
// ``AtomicAdd`` turns each cell into a credit counter: every notify is a
// producer's ``+1``, and the epilogue is the sole consumer's ``-N``. Because
// adds and subtracts are atomic and commutative, the signal is provably
// all-zero again once every rank has finished its epilogue for a call — the
// signal carries no state that outlives one call, so every call's ``g`` restarts
// at 1 and no cross-call bookkeeping is needed. A slow rank can inflate a fast
// rank's own next-call credit by at most 1 while it finishes the current call
// (bounded skew), so the counter never overflows and a fast rank can never
// observe a spurious pass.
//
// ``kGe`` (not ``kEq``) is load-bearing: a fast peer can advance a cell past the
// value the waiting rank is looking for before that rank ever polls it, so an
// equality wait would never unblock. For the same reason ``kSet`` must never be
// mixed with ``kAtomicAdd`` on the same cells — a set could clobber an already
// advanced counter.
//
// Because the protocol is call-local, the mesh (``[NR, 1]``, one cell per rank)
// and ring (``[2*(NR-1), NR]``, one row per round) signal shapes are the only
// remaining incompatibility between collectives sharing one signal buffer — see
// ``ValidateMeshSignalShape``. There is no compile-time generation ledger to
// poison, so collectives are legal inside ``for``/``while``/``if``, and a mesh
// allreduce's per-chunk credit total may be a runtime-computed scalar rather
// than a compile-time constant.
// ============================================================================

std::vector<ExprPtr> CollapseShapeTo2D(const std::vector<ExprPtr>& shape, const Span& span);

std::vector<ExprPtr> CollapseShapeToLinear2D(const std::vector<ExprPtr>& shape, const Span& span);

// The current memory planner assigns distinct storage to the loop-carried
// accumulator and the branch/yield SSA aliases, so lowering can account for
// up to nine physical tile buffers before reuse. At the maximum chunk width,
// nine 16-KiB tiles stay below the smallest supported 184-KiB VEC UB budget
// with room for scalar metadata; statically smaller inputs shrink this width.
constexpr int64_t kAllReduceChunkBytes = 16LL * 1024;

constexpr int64_t kPTOTileAlignmentBytes = 32;

void CheckAllReduceTargetIsPackedNd(const DistributedTensorTypePtr& target_type, const Span& span);

bool IsRowMajorLinearPrefix(const std::vector<ExprPtr>& valid, const std::vector<ExprPtr>& physical);

const std::vector<ExprPtr>* GetPartialValidShape(const DistributedTensorTypePtr& target_type,
                                                 const Span& span);

CallPtr CreateAllReduceTargetView(const ExprPtr& target, const std::vector<ExprPtr>& flat_shape,
                                  const std::vector<ExprPtr>& flat_valid_shape,
                                  const std::vector<ExprPtr>* partial_valid_shape, const Span& span);

/// Validates that ``signal_type`` matches the mesh barrier convention (one
/// cell per rank: ``[NR, 1]``). Ring allreduce's signal is ``[2*(NR-1), NR]``
/// instead — one row per round, addressed ``[row, rank]``. Sharing one buffer
/// between the two conventions no longer trips a generation-table state error
/// (the self-clearing protocol is call-local, so there is no cross-call state
/// to mismatch); this shape check is the sole remaining guard against a mesh
/// op silently targeting the wrong cell of a ring-shaped signal. Skip the
/// second-dimension check when it is symbolic, matching the ring rule's own
/// existing shape checks.
void ValidateMeshSignalShape(const DistributedTensorTypePtr& signal_type, const std::string& op_name,
                             const Span& span);

/// Per-dtype UB chunking constants shared by every InCore collective lowering.
/// ``storage_bits`` is the physical width of one logical element.
/// ``chunk_elements`` is the widest logical-element count that fits one
/// ``kAllReduceChunkBytes`` tile; ``alignment_elements`` is the logical-element
/// count spanning one ``kPTOTileAlignmentBytes`` PTO tile row.
struct CollectiveChunkGeometry {
  int64_t storage_bits = 0;
  int64_t chunk_elements = 0;
  int64_t alignment_elements = 0;
  ExprPtr alignment_elements_idx;
  ExprPtr alignment_minus_one_idx;
  ExprPtr max_chunk_cols;
};

/// Derives the chunk geometry for ``dtype``. ``op_name`` is woven into the
/// diagnostics so a caller-specific message survives the extraction.
CollectiveChunkGeometry MakeChunkGeometry(const DataType& dtype, const Span& span, const char* op_name);

/// Picks the physical chunk width for a chunked collective. A statically known
/// ``logical_extent`` smaller than one full chunk is rounded up to the nearest
/// 32-byte-aligned element count, so a short collective does not reserve a full
/// ``kAllReduceChunkBytes`` tile and the caller keeps its remaining VEC UB
/// budget. Anything else — including a symbolic extent — uses the full chunk.
ExprPtr SelectStaticChunkCols(const CollectiveChunkGeometry& geo, const ExprPtr& logical_extent,
                              const Span& span);

/// Sizes the 2D VEC staging tile that a ``pld.tile.put`` / ``pld.tile.get``
/// transfer slides through.
///
/// pto-isa TPUT/TGET read the full extent from the partition views and 2D-slide
/// the transfer through the stage, so the stage only has to *fit within* the
/// flattened transfer rather than equal it (see comm_op::ValidateStageFitsTransfer).
/// Sizing the stage from the whole transfer therefore buys nothing and costs
/// everything: a [1, 65537] FP32 transfer would reserve 256 KiB of VEC and fail
/// in AllocateMemoryAddr. Cap it to one ``kAllReduceChunkBytes`` tile instead.
///
/// ``transfer_shape`` is flattened to [rows = prod(leading dims), cols = innermost].
/// A dynamic dim contributes the chunk bound directly, matching MakeTputStageShape's
/// contract for pld.tensor.put / pld.tensor.get: the stage is a static UB
/// allocation, and pto-isa reads the runtime extent from the partition views.
/// ``ValidateStageFitsTransfer`` therefore compares only statically known dims
/// (``!cols_static || stage_cols <= transfer_cols``). A symbolic SIZE of runtime
/// 17 still gets a 4096-element FP32 stage — that is the same bound a user-level
/// ``chunk_cols`` would supply, not a stage-fits-transfer violation. The result
/// never exceeds a *statically known* transfer dim.
ExprPtr MakeCollectiveStageShape(const std::vector<ExprPtr>& transfer_shape,
                                 const CollectiveChunkGeometry& geo, const Span& span, const char* op_name);

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_LOWER_COMPOSITE_LOWER_COMPOSITE_COMMON_H_
