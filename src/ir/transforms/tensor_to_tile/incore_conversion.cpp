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

#include <algorithm>
#include <any>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/op_conversion_registry.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/tensor_to_tile/internal.h"
#include "src/ir/transforms/tensor_to_tile/type_propagation.h"

namespace pypto {
namespace ir {
namespace tensor_to_tile {

using tile_conversion_utils::MakeShapeTuple;
using tile_conversion_utils::MakeZeroOffsets;
using transform_utils::FlattenToStmts;

namespace {

std::string MakeTileValueName(const std::string& source_name) {
  return auto_name::BuildName(auto_name::GetBaseName(source_name), "", "tile");
}

std::string MakeStoreResultName(size_t index) {
  return auto_name::BuildName("ret" + std::to_string(index), "", "store");
}

CallPtr MarkCompilerMatBridge(const CallPtr& call, MemorySpace space) {
  if (!call || space != MemorySpace::Mat) return call;
  auto marked = MutableCopy(call);
  marked->attrs_.emplace_back(kCompilerTensorToTileMatBridgeAttr, true);
  return marked;
}

/// Physical extent a cube tile must allocate on the matmul's M axis.
///
/// PTO-ISA keeps two independent notions of size for a cube operand, and only
/// one of them is constrained.  The *logical* extent is essentially free:
/// ``pto.mad`` derives ``%m`` from the operand's valid extent, bounds it at
/// ``[1, 4095]``, and documents ``%m == 1`` as a first-class case.  The
/// *physical* extent must be a whole number of NZ fractal boxes — ptoas
/// enforces exactly that (``'pto.alloc_tile' op expects result boxed tile rows
/// to be a multiple of innerRows (16)``), and pto-isa's ``TExtract`` repeats it
/// as a static assertion on the Mat source it reads.  So the allocation is
/// rounded up to the box and the tensor's true extent rides in ``valid_shape``,
/// which the hardware already addresses through compact mode.
///
/// The padding is free on both axes of the cost model: a load moves only the
/// valid extent, so no extra DMA; and the MAD cost is ``ceil(M/16)`` passes,
/// which rounding M up to a multiple of 16 leaves unchanged.
///
/// @p axis is the axis of @p shape that carries M (1 for a transposed operand,
/// whose natural load puts K on rows), and @p align the extent every cube tile
/// of the call shares (see ``ResolveCubeMAlignment``).  Returns @p shape
/// unchanged — leaving the emitted tile byte-identical to its historical form —
/// when no padding applies, when the extent is dynamic (nothing to round at
/// compile time), or when the tile is not rank 2 (a rank >= 3 operand lowers to
/// ``tile.batch_matmul``, whose rows ``FlattenTileNdTo2D`` row-packs into one
/// ``[B*M, N]`` tile; the box rule binds that packed extent, not this one).
std::vector<ExprPtr> BoxCubeMAxis(const std::vector<ExprPtr>& shape, int64_t align, size_t axis,
                                  const Span& span) {
  if (align <= 1 || shape.size() != 2 || axis >= shape.size()) return shape;
  auto extent = As<ConstInt>(shape[axis]);
  if (!extent || extent->value_ <= 0) return shape;

  const int64_t remainder = extent->value_ % align;
  if (remainder == 0) return shape;
  auto boxed = shape;
  boxed[axis] = std::make_shared<ConstInt>(extent->value_ + align - remainder, DataType::INDEX, span);
  return boxed;
}

/// Whether @p call reads the operand @p req describes transposed, which moves
/// that operand's M from its row axis to its column axis.
bool ReadsOperandTransposed(const CallPtr& call, const InputSpaceReq& req) {
  return req.trans_kwarg && call->GetKwarg<bool>(*req.trans_kwarg, false);
}

/// Axis of the operand @p req describes that carries the matmul's M.
size_t CubeMAxis(const CallPtr& call, const InputSpaceReq& req) {
  return ReadsOperandTransposed(call, req) ? 1 : 0;
}

/// The physical M alignment every cube tile of @p call must share, or 0 when M
/// cannot be boxed at compile time.
///
/// The granularity differs per tile even though the extent must not.  An Acc
/// box is 16 rows for every dtype; a Mat operand's row box is also 16, but its
/// *column* box is ``32 / sizeof(dtype)`` — 8 for FP32, 32 for INT8 — and a
/// transposed left operand has its M on exactly that column axis.  So the
/// alignment is the lcm of the deciding operand's own box and the accumulator's
/// 16 rows; every granularity involved is a power of two, so the lcm is the max.
///
/// @p decider_idx names the operand whose layout decides (the left operand, see
/// ``InputSpaceReq::m_align_from_arg``).  Returns 0 when that operand is not a
/// statically boxed rank-2 tile, which is what keeps rank >= 3 and unresolved
/// layouts on their historical unboxed path.
int64_t ResolveCubeMAlignment(const CallPtr& call,
                              const std::unordered_map<size_t, InputSpaceReq>& input_reqs,
                              size_t decider_idx) {
  auto req_it = input_reqs.find(decider_idx);
  if (req_it == input_reqs.end() || decider_idx >= call->args_.size()) return 0;
  const auto& decider = req_it->second;

  const auto& arg_type = call->args_[decider_idx]->GetType();
  std::vector<ExprPtr> shape;
  DataType dtype = DataType::FP32;
  // ``AsTensorTypeLike`` so a ``DistributedTensorType`` operand is boxed like the plain GM
  // tensor it is. Falling through to the ``return 0`` below would leave a non-fractal window
  // operand unboxed, and it would reach the cube with a physical geometry ptoas rejects.
  if (auto tensor_type = AsTensorTypeLike(arg_type)) {
    shape = tensor_type->shape_;
    dtype = tensor_type->dtype_;
  } else if (auto tile_type = As<TileType>(arg_type)) {
    shape = tile_type->shape_;
    dtype = tile_type->dtype_;
  } else {
    return 0;
  }
  if (shape.size() != 2) return 0;

  const auto view = tile_view_semantics::GetImplicitTileView(shape, decider.demanded_space.Get());
  const auto box = tile_view_semantics::GetBoxedTileAlignment(view, dtype);
  if (!box) return 0;
  const int64_t own = ReadsOperandTransposed(call, decider) ? box->cols : box->rows;
  if (own <= 1) return 0;
  return std::max<int64_t>(own, kAccFractalRows);
}

/// Axis of the operand @p req describes that carries the matmul's N.
///
/// The mirror of ``CubeMAxis``: an untransposed right operand has N on its
/// columns, and a transposed one is loaded naturally with K on its columns, so
/// its N is the row axis.
size_t CubeNAxis(const CallPtr& call, const InputSpaceReq& req) {
  return ReadsOperandTransposed(call, req) ? 0 : 1;
}

/// The physical N alignment the right operand must be allocated to, or 0 when N
/// cannot be boxed at compile time.
///
/// Unlike M, N needs no cross-operand decider: it is carried by the right
/// operand and the accumulator alone, and both are reached from this one shape.
/// It does need reconciling across *memory spaces*, which M does not:
///
///   * the operand is loaded into ``Mat`` and then promoted on to ``Right`` by
///     InferTileMemorySpace, and ``Right`` boxes the same fractal under the
///     opposite ``slayout`` -- which swaps the row and column granularities
///     (fp32: ``Mat`` is 16x8, ``Right`` is 8x16). A column count legal only
///     where it was loaded is rejected at the very next op, so the two have to
///     be reconciled, and the lcm of the box's own two extents is a multiple of
///     either by construction.
///   * the product lands in an ``Acc`` whose N box is 16 for every dtype.
///
/// Every granularity involved is a power of two, so both lcms are maxima.
int64_t ResolveCubeNAlignment(const CallPtr& call, const InputSpaceReq& req, size_t idx) {
  if (idx >= call->args_.size()) return 0;
  const auto& arg_type = call->args_[idx]->GetType();
  std::vector<ExprPtr> shape;
  DataType dtype = DataType::FP32;
  // Window operands are boxed like plain GM tensors here too -- see ResolveCubeMAlignment.
  if (auto tensor_type = AsTensorTypeLike(arg_type)) {
    shape = tensor_type->shape_;
    dtype = tensor_type->dtype_;
  } else if (auto tile_type = As<TileType>(arg_type)) {
    shape = tile_type->shape_;
    dtype = tile_type->dtype_;
  } else {
    return 0;
  }
  if (shape.size() != 2) return 0;

  const auto view = tile_view_semantics::GetImplicitTileView(shape, req.demanded_space.Get());
  const auto box = tile_view_semantics::GetBoxedTileAlignment(view, dtype);
  if (!box || box->rows <= 0 || box->cols <= 0) return 0;
  return std::max<int64_t>({box->rows, box->cols, kAccFractalRows});
}

/// The boxed axis of operand @p idx and the extent it must be allocated to.
///
/// One resolver for both load-emitting sites, so a bridged operand and a
/// consumer-driven one cannot disagree. An operand declares at most one boxed
/// axis, since the axis it does not carry is the contraction axis.
std::pair<int64_t, size_t> ResolveBoxedAxis(const CallPtr& call,
                                            const std::unordered_map<size_t, InputSpaceReq>& input_reqs,
                                            size_t idx, const InputSpaceReq& req) {
  if (req.cube_m_axis) {
    return {ResolveCubeMAlignment(call, input_reqs, req.m_align_from_arg.value_or(idx)),
            CubeMAxis(call, req)};
  }
  if (req.cube_n_axis) {
    return {ResolveCubeNAlignment(call, req, idx), CubeNAxis(call, req)};
  }
  return {0, 0};
}

bool IsPassthroughTensorOp(const CallPtr& call) {
  return IsOp(call, "tensor.dim") || IsOp(call, "tensor.view");
}

/// Declared GM cache policy per source tensor, keyed by the param Var the
/// declaration resolved to. ``CachePolicy`` stored as ``int``, the type the
/// ``tile.load`` ``cache`` kwarg is registered with.
using CachePolicyByParam = std::unordered_map<const Var*, int>;

/**
 * @brief Resolve an InCore function's ``cache_policy`` attr to its param Vars.
 *
 * ``OutlineIncoreScopes`` (pass 9) records ``pl.set_cache_policy`` declarations
 * as (param index, policy) pairs, because at that point the param Vars are
 * freshly minted. Here the indices are turned back into Var identities — the
 * form every load site below matches its source arg against — and the attr is
 * erased on the way out (see ``EraseCachePolicyAttr``): param indices are only
 * valid across passes 9..11, since later passes both append to and prepend onto
 * param lists.
 */
CachePolicyByParam BuildCachePolicyByParam(const FunctionPtr& func) {
  CachePolicyByParam policies;
  auto indices = func->GetAttr<std::vector<std::pair<int32_t, int>>>(kAttrCachePolicyParams);
  policies.reserve(indices.size());
  for (const auto& [idx, policy] : indices) {
    INTERNAL_CHECK_SPAN(idx >= 0 && static_cast<size_t>(idx) < func->params_.size(), func->span_)
        << "Internal error: cache_policy param index " << idx << " out of range for function '" << func->name_
        << "' with " << func->params_.size() << " param(s)";
    // insert_or_assign, not emplace: a tensor declared twice takes its last
    // declaration, deterministically (the attr is sorted by index).
    policies.insert_or_assign(func->params_[static_cast<size_t>(idx)].get(), policy);
  }
  return policies;
}

/// Drop the consumed ``cache_policy`` attr. Nothing downstream may see it —
/// its param indices go stale the moment a later pass grows the param list.
std::vector<std::pair<std::string, std::any>> EraseCachePolicyAttr(
    const std::vector<std::pair<std::string, std::any>>& attrs) {
  std::vector<std::pair<std::string, std::any>> kept;
  kept.reserve(attrs.size());
  for (const auto& kv : attrs) {
    if (kv.first == kAttrCachePolicyParams) continue;
    kept.push_back(kv);
  }
  return kept;
}

/// Longest ``IterArg`` init chain followed when resolving a load source. Loop
/// nesting is the real bound (single digits); this only stops malformed IR from
/// spinning.
constexpr int kMaxIterArgInitChain = 64;

/**
 * @brief Resolve a load source to the function parameter it ultimately reads.
 *
 * A loop-carried tensor reaches its load as an ``IterArg``, not as the param
 * ``Var`` the declaration named -- ``IterArg`` is its own ``ObjectKind``, so
 * ``AsVarLike`` hands it back as itself and a param-keyed lookup misses. Follow
 * ``initValue_`` back to the root first, mirroring how
 * ``PTOCodegen::TryGetTensorView`` resolves the same shape. Returns the
 * innermost resolvable Var-like, which for a non-carried source is the source
 * itself.
 */
VarPtr ResolveToRootParam(const ExprPtr& src) {
  auto var = AsVarLike(src);
  for (int depth = 0; var && depth < kMaxIterArgInitChain; ++depth) {
    auto iter_arg = As<IterArg>(var);
    if (!iter_arg) return var;
    auto init = AsVarLike(iter_arg->initValue_);
    if (!init) return var;
    var = init;
  }
  return var;
}

/**
 * @brief Add the declared ``cache`` kwarg for a synthesised ``tile.load``.
 *
 * No-op unless ``src`` is a param carrying a declaration. An explicit
 * ``cache=`` already in ``kwargs`` always wins (precedence: per-access kwarg,
 * then the scope declaration, then ``CachePolicy::kDefault``).
 */
void AppendCachePolicyKwarg(const ExprPtr& src, const CachePolicyByParam& policies,
                            std::vector<std::pair<std::string, std::any>>* kwargs) {
  if (policies.empty()) return;
  auto var = ResolveToRootParam(src);
  if (!var) return;
  auto it = policies.find(var.get());
  if (it == policies.end()) return;
  const bool stated_explicitly =
      std::any_of(kwargs->begin(), kwargs->end(), [](const auto& kv) { return kv.first == "cache"; });
  if (stated_explicitly) return;
  kwargs->emplace_back("cache", it->second);
}

/**
 * @brief Stamp the declared policy onto a ``tile.load`` already in the body.
 *
 * A hand-written (or earlier-pass) load of a declared tensor must honour the
 * declaration exactly as a synthesised one does, unless it states its own
 * ``cache=``. Returns nullptr when nothing changes, so callers keep their
 * copy-on-write short-circuit. Only ``Call`` is considered: a ``Submit``
 * launches a task through a ``GlobalVar`` callee and can never carry a tile op.
 */
CallPtr StampCachePolicyOnLoad(const CallPtr& call, const CachePolicyByParam& policies) {
  if (policies.empty() || !IsOp(call, "tile.load") || call->args_.empty()) return nullptr;
  auto kwargs = call->kwargs_;
  AppendCachePolicyKwarg(call->args_[0], policies, &kwargs);
  // Unchanged when the source carries no declaration, or when the load already
  // states its own ``cache=``.
  if (kwargs.size() == call->kwargs_.size()) return nullptr;
  auto stamped = MutableCopy(call);
  stamped->kwargs_ = std::move(kwargs);
  return stamped;
}

void CheckReinterpretViewIncoreLayout(const CallPtr& call) {
  if (!IsOp(call, "tensor.reinterpret_view")) return;

  INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
      << "Internal error: tensor.reinterpret_view reached conversion without a data argument";
  auto source_type = As<TensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(source_type, call->span_)
      << "Internal error: tensor.reinterpret_view source must be TensorType before tile conversion";
  const TensorLayout source_layout =
      source_type->tensor_view_.has_value() ? source_type->tensor_view_->layout : TensorLayout::ND;
  CHECK_SPAN(source_layout == TensorLayout::ND, call->span_)
      << "tensor.reinterpret_view in an InCore function currently supports only packed ND tensors; "
         "DN layout changes which logical axis is physically contiguous, and that information cannot "
         "yet be preserved by tensor-to-tile lowering";
}

/// Converters without explicit input_reqs consume ordinary tensor operands as
/// tiles. Memory operations and distributed transfers manage GM operands
/// themselves; loading their operands would erase the destination identity.
bool UsesDefaultTileInputs(const CallPtr& call) {
  if (!call || call->op_->name_.rfind("tensor.", 0) != 0) return false;
  return !(IsOp(call, "tensor.slice") || IsOp(call, "tensor.assemble") || IsOp(call, "tensor.read") ||
           IsOp(call, "tensor.write") || IsOp(call, "tensor.expand_clone") || IsOp(call, "tensor.gather") ||
           IsOp(call, "tensor.paged_gather") || IsOp(call, "tensor.create_l1") ||
           IsOp(call, "tensor.gather_row"));
}

/// A dense BF16 DN -> packed FP4 cast needs a column-major Vec tile. Ordinary
/// BF16 GM loads intentionally produce the default Vec layout, so this one
/// consumer instead loads the identical ND storage and transposes its metadata.
bool NeedsPackedFp4DnCastLoad(const CallPtr& call, size_t index) {
  if (index != 0 || !IsOp(call, "tensor.cast") || call->args_.empty()) return false;
  auto source = AsTensorTypeLike(call->args_[0]->GetType());
  auto result = AsTensorTypeLike(call->GetType());
  return source && result && source->dtype_ == DataType::BF16 && result->dtype_.IsPackedFp4() &&
         source->tensor_view_.has_value() && source->tensor_view_->layout == TensorLayout::DN;
}

/**
 * @brief Visitor that collects tensor-typed variables used directly by converted ops.
 *
 * Traverses the IR tree via IRVisitor and records the name of every Var/IterArg argument
 * whose type is TensorType and that appears in a call to an op registered in
 * OpConversionRegistry (i.e. an op that will be converted from tensor.* to tile.*).
 *
 * Used by TransformIncoreFunction to decide which tensor parameters require a synthesised
 * default Vec-space tile.load in Phase 1.  Parameters that are only referenced by
 * non-converted ops (e.g. tile.load, tile.move) already manage their own tile
 * representation and must NOT get an extra load inserted.
 *
 * Also excludes parameters used by tensor.slice and tensor.matmul since those conversions
 * create their own block.load with proper offsets/memory spaces.
 */
class TensorArgsInConvertedOpsCollector : public IRVisitor {
 public:
  explicit TensorArgsInConvertedOpsCollector(const OpConversionRegistry& conv_registry)
      : conv_registry_(conv_registry) {}

  [[nodiscard]] const std::unordered_set<const Var*>& GetUsed() const { return used_; }

  /**
   * @brief Trace from collected IterArgs to their ForStmt/WhileStmt initValue_ expressions.
   *
   * When an IterArg is in used_ (consumed by a converted op), its initValue_ may be a
   * function parameter eligible for a shared entry load. Follow each newly
   * discovered seed once, including chains of nested IterArgs, in O(N). Only
   * tile-valued carries consume such a preload; GM carries load their current
   * value at the computation, which may differ from their initializer.
   */
  template <typename IsTileValue>
  void TraceIterArgInitValues(IsTileValue is_tile_value) {
    std::vector<const Var*> worklist(used_.begin(), used_.end());
    for (size_t i = 0; i < worklist.size(); ++i) {
      if (!is_tile_value(worklist[i])) continue;
      auto it = iter_arg_to_init_.find(worklist[i]);
      if (it == iter_arg_to_init_.end()) continue;
      if (auto var = AsVarLike(it->second);
          var && AsTensorTypeLike(var->GetType()) && used_.insert(var.get()).second) {
        worklist.push_back(var.get());
      }
    }
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    auto call = As<Call>(op->value_);
    const auto* conv_entry = (call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_))
                                 ? conv_registry_.Lookup(call->op_->name_)
                                 : nullptr;
    if (conv_entry) {
      // Skip ops whose inputs are handled by their own converter (self-loading):
      // they create loads with specific offsets/spaces, so Phase-1 default Vec loads
      // would be redundant or wrong.
      if (!UsesDefaultTileInputs(call)) {
        IRVisitor::VisitStmt_(op);
        return;
      }
      // Per-arg exclusion: args covered by input_reqs are handled by framework auto-bridging.
      // Other args (e.g. matmul_acc's acc, which has no input_req) still need Phase-1 loads
      // so they reach the converter as TileType.
      for (size_t i = 0; i < call->args_.size(); ++i) {
        if (conv_entry->input_reqs.count(i) || NeedsPackedFp4DnCastLoad(call, i)) continue;
        const auto& arg = call->args_[i];
        // ``AsTensorTypeLike`` also collects ``DistributedTensorType`` params: inside
        // an InCore scope a window is this rank's local GM, so an op with no
        // input_req (e.g. tensor.row_max) needs the same Phase-1 entry load a plain
        // tensor param gets. The exact-kind ``As<TensorType>`` skipped it, and the
        // converter then saw an unbridged tensor operand.
        if (auto iter_arg = As<IterArg>(arg)) {
          if (AsTensorTypeLike(iter_arg->GetType())) used_.insert(iter_arg.get());
        } else if (auto var = As<Var>(arg)) {
          if (AsTensorTypeLike(var->GetType())) used_.insert(var.get());
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    for (const auto& iter_arg : op->iter_args_) {
      iter_arg_to_init_[iter_arg.get()] = iter_arg->initValue_;
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    for (const auto& iter_arg : op->iter_args_) {
      iter_arg_to_init_[iter_arg.get()] = iter_arg->initValue_;
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  const OpConversionRegistry& conv_registry_;
  std::unordered_set<const Var*> used_;
  std::unordered_map<const Var*, ExprPtr> iter_arg_to_init_;
};

// Consumer-driven memory space collection.
//
// Pre-scans the function body to build a map from variables to the memory
// space their downstream consumers require (as declared via InputSpaceReq
// in OpConversionRegistry).  This lets load-like ops (tensor.slice on
// TensorType) produce the right space directly, avoiding a redundant
// load(Vec) + move(Mat) sequence.
// ============================================================================

/**
 * @brief Resolved consumer memory space requirement for a variable.
 */
struct ConsumerSpaceReq {
  MemorySpace space;  ///< Required memory space. The consumer-driven load is always
                      ///< natural; a transposed (b_trans/a_trans) operand is realised
                      ///< by a zero-copy tile.transpose_view in BridgeInputSpaces.
  /// Resolved form of ``InputSpaceReq::cube_m_axis``, so a load-like producer
  /// that answers the demand directly boxes exactly what ``BridgeInputSpaces``
  /// would have: ``cube_m_align`` is the extent every cube tile of the consuming
  /// call shares (0 when M cannot be boxed), and ``cube_m_axis`` names the axis
  /// of *this* operand that carries M — 1 when the consumer reads it
  /// transposed, whose natural load puts K on rows.
  int64_t cube_m_align = 0;
  size_t cube_m_axis = 0;
};

/// Fold @p incoming into @p existing, the requirement already recorded for a
/// shared producer.
///
/// Two rules, and both paths that record a demand -- a direct operand use and
/// the backward sweep over inherit-input / alias / loop-carry edges -- have to
/// apply them identically, or which demand a producer sees depends on the order
/// its consumers happen to be visited in.
///
///   * A specialized space (Mat/Left/Right/Acc/Bias) beats the default Vec, so
///     a load-like producer can emit that space directly.
///   * The M boxing survives only where every consumer agrees on it. A differing
///     axis or alignment means one of them reads the tile at its declared
///     physical shape, and one buffer cannot carry two physical extents; the
///     padding is dropped so the mismatch is reported against the shape the
///     author actually wrote, the same way whichever consumer is visited first.
void MergeConsumerReq(ConsumerSpaceReq* existing, const ConsumerSpaceReq& incoming) {
  if (existing->space == MemorySpace::Vec && incoming.space != MemorySpace::Vec) {
    *existing = incoming;
    return;
  }
  if (existing->space != incoming.space) return;
  if (existing->cube_m_axis != incoming.cube_m_axis || existing->cube_m_align != incoming.cube_m_align) {
    existing->cube_m_align = 0;
  }
}

/**
 * @brief Visitor that collects consumer memory space requirements for variables.
 *
 * For each op with declared InputSpaceReq, records which variables need which
 * memory space.  Replaces the special-purpose MatmulSlicePatternCollector with
 * a general mechanism driven entirely by registered converter metadata.
 */
class ConsumerSpaceCollector : public IRVisitor {
 public:
  explicit ConsumerSpaceCollector(const OpConversionRegistry& registry) : registry_(registry) {}

  [[nodiscard]] std::optional<ConsumerSpaceReq> GetConsumerReq(const Var* var) const {
    auto it = consumer_reqs_.find(var);
    return it != consumer_reqs_.end() ? std::optional{it->second} : std::nullopt;
  }

  /// Second phase: propagate collected requirements backward through
  ///   (a) ops registered with `set_output_memory_inherit_input()` — output
  ///       memory equals the first tile/tensor-typed input's, so a demand on
  ///       the output is equivalently a demand on that input, and
  ///   (b) plain SSA aliases `y = x` where both sides are shaped Vars (the
  ///       parser elides no-op `tensor.fillpad(pad=zero)` into this form when
  ///       the input's valid_shape already zeroes the pad region).
  ///
  /// Edges are recorded in program order during the forward visit. Since the
  /// inherit-input and alias relations are acyclic and flow strictly backward
  /// (output/dst defined after input/src), a single reverse-order sweep
  /// reaches the fixed point in O(N). Total pass cost stays O(N log N).
  void PropagateThroughInheritInputOps() {
    for (auto it = propagation_edges_.rbegin(); it != propagation_edges_.rend(); ++it) {
      const auto& [dst, src] = *it;
      auto out_it = consumer_reqs_.find(dst);
      if (out_it == consumer_reqs_.end()) continue;
      const auto& req = out_it->second;
      auto [ins_it, inserted] = consumer_reqs_.try_emplace(src, req);
      // Same reconciliation as a direct use: a source reached both directly and
      // through an alias must not keep whichever demand this sweep happened to
      // reach first (see MergeConsumerReq).
      if (!inserted) MergeConsumerReq(&ins_it->second, req);
    }
  }

 protected:
  /// A loop carry's demand is equally a demand on the value that seeds it: the
  /// two share one buffer for the whole loop, so an accumulator allocated
  /// before the loop must be built the way the body's `tile.matmul_acc` reads
  /// it. The edge flows strictly backward (the seed is defined before the
  /// loop), so the reverse sweep in PropagateThroughInheritInputOps resolves it
  /// in the same single pass as the others.
  void VisitExpr_(const IterArgPtr& op) override {
    if (op && op->initValue_) {
      if (auto seed = AsVarLike(op->initValue_)) {
        propagation_edges_.emplace_back(op.get(), seed.get());
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    auto is_shaped = [](const TypePtr& t) { return As<TensorType>(t) || As<TileType>(t); };

    // Record a propagation edge `dst -> src` in program order when the RHS is
    // either a plain SSA alias (both sides shaped) or an inherit-input Call
    // (first shaped input carries the memory-space relation). The reverse walk
    // in phase 2 then resolves all back-propagation in a single pass.
    if (op->var_ && is_shaped(op->var_->GetType())) {
      if (auto src_var = As<Var>(op->value_); src_var && is_shaped(src_var->GetType())) {
        propagation_edges_.emplace_back(op->var_.get(), src_var.get());
      } else if (auto call = As<Call>(op->value_);
                 call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
        // Deliberately the raw `OutputMemoryInheritsInput()` flag, NOT
        // `op_predicates::OutputInheritsSourceBuffer`: the edge recorded here carries
        // the memory *space* relation, which is what the flag declares. Buffer
        // identity requires an explicit storage alias and is answered
        // by `ResultAliasedArgIndex` elsewhere in this pass.
        auto& op_reg = OpRegistry::GetInstance();
        if (op_reg.IsRegistered(call->op_->name_) &&
            op_reg.GetEntry(call->op_->name_).OutputMemoryInheritsInput()) {
          for (const auto& arg : call->args_) {
            if (auto arg_var = As<Var>(arg); arg_var && is_shaped(arg_var->GetType())) {
              propagation_edges_.emplace_back(op->var_.get(), arg_var.get());
              break;
            }
          }
        }
      }
    }

    auto call = As<Call>(op->value_);
    if (!call || std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    const auto* entry = registry_.Lookup(call->op_->name_);
    if (!entry || entry->input_reqs.empty()) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    // Both 2D tile.matmul and ND tile.batch_matmul realise a transposed operand
    // as a zero-copy tile.transpose_view added in BridgeInputSpaces (issues #1776
    // / ND extension): the consumer-driven load is ALWAYS natural, and the view
    // supplies the transpose. (No more transpose-at-load baking.)
    for (const auto& [idx, req] : entry->input_reqs) {
      if (idx >= call->args_.size()) continue;
      // `AsVarLike`, not `As<Var>`: a loop-carried operand is an `IterArg`,
      // which carries its own ObjectKind and would otherwise record no demand
      // at all (see `ir-kind-traits.md`). A split-K accumulator reaches its
      // `tile.matmul_acc` exactly that way.
      if (auto var = AsVarLike(call->args_[idx])) {
        // A transposed use moves this operand's M to its column axis, and the
        // alignment is the one the whole call shares (see ConsumerSpaceReq).
        const auto [align, boxed_axis] = ResolveBoxedAxis(call, entry->input_reqs, idx, req);
        const ConsumerSpaceReq resolved{req.demanded_space.Get(), align, boxed_axis};
        // Several consumers can share one producer (a sliced KV feeding two
        // matmuls, an accumulator seed read by two accumulations); MergeConsumerReq
        // is the single rule for reconciling them.
        auto [it, inserted] = consumer_reqs_.try_emplace(var.get(), resolved);
        if (!inserted) MergeConsumerReq(&it->second, resolved);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  const OpConversionRegistry& registry_;
  std::unordered_map<const Var*, ConsumerSpaceReq> consumer_reqs_;
  // `dst -> src` edges captured in program order — covers both Call-valued
  // inherit-input ops and plain SSA aliases. A single reverse-order walk in
  // PropagateThroughInheritInputOps reaches the fixed point.
  std::vector<std::pair<const Var*, const Var*>> propagation_edges_;
};

/// Predict which SSA values become tiles, including phi values. A GM handle
/// carried through a write remains GM; a carry updated by a computation needs
/// a tile seed. Propagate along value-flow edges, not every operand use. Each
/// edge and marked value is processed once, including cyclic loop carries.
/// A separate backward dependency walk conservatively excludes parameters
/// whose GM storage or loaded tile values may be modified from load sharing.
class TensorConversionAnalysis : public IRVisitor {
 public:
  explicit TensorConversionAnalysis(const OpConversionRegistry& registry) : registry_(registry) {}

  void Propagate() {
    // Yield edges can precede the tuple definitions in their branch/body.
    // Resolve projections only after the complete definition index exists.
    for (const auto& [source, target] : projection_flows_) {
      if (auto var = AsVarLike(ResolveFlowSource(source))) AddFlow(var, target);
    }
    while (!worklist_.empty()) {
      const auto* value = worklist_.back();
      worklist_.pop_back();
      auto it = users_.find(value);
      if (it == users_.end()) continue;
      for (const auto* user : it->second) MarkTile(user);
    }
    while (!write_worklist_.empty()) {
      const auto* value = write_worklist_.back();
      write_worklist_.pop_back();
      auto it = sources_.find(value);
      if (it == sources_.end()) continue;
      for (const auto* source : it->second) MarkWritten(source);
    }
  }

  [[nodiscard]] bool IsTile(const Var* var) const { return tiles_.count(var) != 0; }
  [[nodiscard]] bool IsTile(const VarPtr& var) const { return IsTile(var.get()); }
  [[nodiscard]] bool HasOpaqueCalls() const { return has_opaque_calls_; }
  [[nodiscard]] bool MayWriteSource(const VarPtr& var) const {
    return written_sources_.count(var.get()) != 0;
  }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (std::dynamic_pointer_cast<const GlobalVar>(op->op_)) has_opaque_calls_ = true;
    if (const auto* entry = LookupOpEntry(op->op_)) {
      for (size_t i = 0; i < op->args_.size(); ++i) {
        if (!ArgEffectWrites(entry->GetArgEffect(i, op->kwargs_))) continue;
        var_collectors::VarDefUseCollector refs;
        refs.VisitExpr(op->args_[i]);
        for (const auto* var : refs.var_uses_ordered) MarkWritten(var);
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    has_opaque_calls_ = true;
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // Include tuple packing/projection and other expression-valued aliases,
    // not just direct Call operands. Each assignment's expression is visited
    // once and the collector stops at Var/IterArg references.
    var_collectors::VarDefUseCollector refs;
    refs.VisitExpr(op->value_);
    sources_[op->var_.get()] = std::move(refs.var_uses_ordered);
    if (As<TupleType>(op->var_->GetType())) tuple_definitions_[op->var_.get()] = op->value_;
    if (As<TileType>(op->var_->GetType())) {
      MarkTile(op->var_.get());
    } else if (AsTensorTypeLike(op->var_->GetType())) {
      if (AsVarLike(op->value_) || As<TupleGetItemExpr>(op->value_)) {
        AddFlow(op->value_, op->var_);
      } else if (auto call = As<Call>(op->value_); call && registry_.Lookup(call->op_->name_)) {
        if (IsOp(call, "tensor.write") || IsOp(call, "tensor.assemble")) {
          AddFlow(call->args_[0], op->var_);
        } else if (IsOp(call, "tensor.expand_clone")) {
          AddFlow(call->args_[1], op->var_);
        } else {
          MarkTile(op->var_.get());
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    AddLoopFlows(op->body_, op->iter_args_, op->return_vars_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    AddLoopFlows(op->body_, op->iter_args_, op->return_vars_);
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    AddYieldFlows(op->then_body_, op->return_vars_);
    if (op->else_body_) AddYieldFlows(*op->else_body_, op->return_vars_);
    IRVisitor::VisitStmt_(op);
  }

 private:
  void MarkTile(const Var* var) {
    if (tiles_.insert(var).second) worklist_.push_back(var);
  }

  void MarkWritten(const Var* var) {
    if (written_sources_.insert(var).second) write_worklist_.push_back(var);
  }

  /// Reuse immutable MakeTuple nodes as shared descriptors. Aliases select
  /// the same descriptor, and projections select only their indexed element.
  /// Memoization visits each alias/projection once without copying wide tuples.
  /// Tensor leaves remain Var/IterArg nodes so their phi cycles use the worklist.
  ExprPtr ResolveFlowSource(const ExprPtr& source) {
    auto cached = resolved_flow_sources_.find(source.get());
    if (cached != resolved_flow_sources_.end()) return cached->second;

    ExprPtr resolved;
    if (auto var = AsVarLike(source)) {
      if (!As<TupleType>(var->GetType())) {
        resolved = var;
      } else if (auto definition = tuple_definitions_.find(var.get());
                 definition != tuple_definitions_.end()) {
        resolved = ResolveFlowSource(definition->second);
      }
    } else if (As<MakeTuple>(source)) {
      resolved = source;
    } else if (auto projection = As<TupleGetItemExpr>(source)) {
      if (auto tuple = As<MakeTuple>(ResolveFlowSource(projection->tuple_))) {
        resolved = ResolveFlowSource(tuple->elements_[projection->index_]);
      }
    }
    resolved_flow_sources_.emplace(source.get(), resolved);
    return resolved;
  }

  void AddFlow(const ExprPtr& source, const VarPtr& target) {
    if (As<TupleGetItemExpr>(source)) {
      projection_flows_.emplace_back(source, target);
      return;
    }
    auto var = AsVarLike(source);
    if (!var) return;
    users_[var.get()].push_back(target.get());
    sources_[target.get()].push_back(var.get());
    if (As<TileType>(var->GetType())) MarkTile(var.get());
  }

  void AddYieldFlows(const StmtPtr& body, const std::vector<VarPtr>& targets) {
    auto yield = transform_utils::GetLastYieldStmt(body);
    if (!yield) return;
    for (size_t i = 0; i < targets.size() && i < yield->value_.size(); ++i) {
      AddFlow(yield->value_[i], targets[i]);
    }
  }

  void AddLoopFlows(const StmtPtr& body, const std::vector<IterArgPtr>& iter_args,
                    const std::vector<VarPtr>& returns) {
    std::vector<VarPtr> targets;
    targets.reserve(iter_args.size());
    for (size_t i = 0; i < iter_args.size(); ++i) {
      AddFlow(iter_args[i]->initValue_, iter_args[i]);
      if (i < returns.size()) AddFlow(iter_args[i], returns[i]);
      targets.push_back(iter_args[i]);
    }
    AddYieldFlows(body, targets);
  }

  const OpConversionRegistry& registry_;
  std::unordered_map<const Var*, ExprPtr> tuple_definitions_;
  std::unordered_map<const Expr*, ExprPtr> resolved_flow_sources_;
  std::vector<std::pair<ExprPtr, VarPtr>> projection_flows_;
  std::unordered_map<const Var*, std::vector<const Var*>> users_;
  std::unordered_set<const Var*> tiles_;
  std::vector<const Var*> worklist_;
  std::unordered_map<const Var*, std::vector<const Var*>> sources_;
  std::unordered_set<const Var*> written_sources_;
  std::vector<const Var*> write_worklist_;
  bool has_opaque_calls_ = false;
};

// ============================================================================
// TensorToTileMutator: converts tensor ops to tile ops in InCore function
// bodies. Materializes compute operands and tile-valued control-flow seeds,
// then runs converters from OpConversionRegistry. The base handles SSA
// substitution and propagation of the resulting types.
// ============================================================================

class TensorToTileMutator : public TypePropagatingMutator {
 public:
  TensorToTileMutator(const OpConversionRegistry& conv_registry, const OpRegistry& op_registry,
                      const ConsumerSpaceCollector& consumer_collector,
                      const TensorConversionAnalysis& tile_values, CachePolicyByParam cache_policies)
      : conv_registry_(conv_registry),
        op_registry_(op_registry),
        consumer_collector_(consumer_collector),
        tile_values_(tile_values),
        cache_policies_(std::move(cache_policies)) {}

  /// A cached load is a compute operand, never an SSA replacement for its GM
  /// source. Only parameters proven unwritten throughout the function enter
  /// this map; all other GM operands load at the consuming statement.
  StmtPtr PreloadReadOnlyParam(const VarPtr& var) {
    std::vector<StmtPtr> stmts;
    auto tile = LoadTensorOperand(var, var->span_, stmts);
    preloaded_tiles_[var.get()] = tile;
    INTERNAL_CHECK_SPAN(stmts.size() == 1, var->span_) << "Internal error: parameter already preloaded";
    return stmts.front();
  }

 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return ConvertLoop(op); }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return ConvertLoop(op); }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto saved_targets = std::move(yield_tile_targets_);
    SetYieldTileTargets(op->return_vars_);
    auto result = TypePropagatingMutator::VisitStmt_(op);
    yield_tile_targets_ = std::move(saved_targets);
    return result;
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    auto result = As<YieldStmt>(IRMutator::VisitStmt_(op));
    auto values = result->value_;
    std::vector<StmtPtr> stmts;
    for (size_t i = 0; i < values.size() && i < yield_tile_targets_.size(); ++i) {
      if (yield_tile_targets_[i] && AsTensorTypeLike(values[i]->GetType())) {
        values[i] = LoadTensorOperand(values[i], op->span_, stmts);
      }
    }
    if (values != result->value_) {
      auto copy = MutableCopy(result);
      copy->value_ = std::move(values);
      result = std::move(copy);
    }
    stmts.push_back(result);
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  /// Honour a ``pl.set_cache_policy`` declaration on a ``tile.load`` that was
  /// already in the body (user-written, or produced by an earlier pass) rather
  /// than synthesised here. Hooked on the generic Call visit so the loads a
  /// converter emits in its own prologue are covered too.
  ExprPtr VisitExpr_(const CallPtr& op) override {
    // Qualified with IRMutator: TypePropagatingMutator declares only the
    // IterArg overload, which hides the base Call one from name lookup.
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    if (!call) return visited;
    auto stamped = StampCachePolicyOnLoad(call, cache_policies_);
    return stamped ? ExprPtr(stamped) : visited;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Pin this Var's address for the pass so a freed-then-reused address cannot
    // alias a stale var_remap_ entry (see TypePropagatingMutator::RetainVar).
    RetainVar(op->var_);
    CheckReinterpretViewIncoreLayout(As<Call>(op->value_));
    auto new_value = VisitExpr(op->value_);
    auto call = As<Call>(new_value);

    // Non-call values: propagate type change
    if (!call) {
      // Sharing a read-only operand through a plain SSA alias must preserve
      // the alias's GM type while making its already-loaded value reusable.
      auto cached = preloaded_tiles_.find(new_value.get());
      if (cached != preloaded_tiles_.end()) preloaded_tiles_[op->var_.get()] = cached->second;
      return HandlePassThroughAssign(op, new_value);
    }

    // Function calls (GlobalVar) pass through — only process op calls
    if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
      LOG_DEBUG << "[TensorToTileMutator] Skipping GlobalVar call: " << call->op_->name_;
      return HandlePassThroughAssign(op, new_value);
    }

    const auto* entry = conv_registry_.Lookup(call->op_->name_);
    if (!entry) {
      if (IsOp(call, "tensor.view")) {
        CHECK_SPAN(!call->args_.empty() && AsTensorTypeLike(call->args_[0]->GetType()), call->span_)
            << "tensor.view in an InCore function requires a GM Tensor input that remains tensor-like "
               "through ConvertTensorToTileOps; viewing the result of an op lowered to Tile is not supported";
      }
      // Verify unregistered TensorOps are expected passthroughs
      if (op_registry_.IsRegistered(call->op_->name_)) {
        const auto& op_entry = op_registry_.GetEntry(call->op_->name_);
        INTERNAL_CHECK_SPAN(op_entry.GetOpCategory() != "TensorOp" || IsPassthroughTensorOp(call),
                            call->span_)
            << "TensorOp \"" << call->op_->name_ << "\" has no registered tile conversion. "
            << "Add a conversion in src/ir/transforms/op_conversion_registry.cpp.";
      }
      return HandlePassThroughAssign(op, new_value);
    }

    // Consumer-driven space override for load-like ops (e.g. tensor.slice
    // feeding into tensor.matmul → load to Mat instead of default Vec).
    if (IsOp(call, "tensor.slice")) {
      auto consumer_req = consumer_collector_.GetConsumerReq(op->var_.get());
      if (consumer_req) {
        auto override_load = HandleConsumerDrivenLoad(op, call, *consumer_req);
        if (override_load) return override_load;
      }
    }

    // Consumer-driven row boxing for an allocation that feeds a cube
    // accumulator. Unlike an operand, an accumulator is never loaded from GM, so
    // its create site is the only place its physical row count can still be
    // rounded up to the box.
    if (IsOp(call, "tensor.create")) {
      auto consumer_req = consumer_collector_.GetConsumerReq(op->var_.get());
      if (consumer_req && consumer_req->cube_m_align > 0) {
        auto boxed_create = HandleBoxedAccCreate(op, call, *consumer_req);
        if (boxed_create) return boxed_create;
      }
    }

    // Auto-bridge: load TensorType args to the memory space required by input_reqs
    auto [bridged_args, bridge_stmts] =
        BridgeInputSpaces(call, entry->input_reqs, NeedsPackedFp4DnCastLoad(As<Call>(op->value_), 0));

    // Run the converter with bridged args
    auto conv_result = entry->func(bridged_args, call->kwargs_, call->span_, conversion_context_);

    // Collect all statements: bridge prologue + converter prologue + final assignment
    std::vector<StmtPtr> stmts;
    stmts.reserve(bridge_stmts.size() + conv_result.prologue.size() + 1);

    // Bridge statements are fully resolved — no recursive visit needed
    for (auto& s : bridge_stmts) stmts.push_back(std::move(s));

    // Converter prologue may contain nested tensor ops — recurse
    for (auto& prologue_stmt : conv_result.prologue) {
      stmts.push_back(VisitStmt(prologue_stmt));
    }

    // Revisit result after mutating prologue — prologue conversions may have
    // remapped vars that the result expression references.
    auto new_result = VisitExpr(conv_result.result);

    auto tile_name = MakeTileValueName(op->var_->name_hint_);
    auto tile_var = std::make_shared<Var>(tile_name, new_result->GetType(), op->var_->span_);
    RecordPackedTile(tile_var, new_result);
    stmts.push_back(std::make_shared<AssignStmt>(tile_var, new_result, op->span_));
    var_remap_[op->var_.get()] = tile_var;

    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    CheckReinterpretViewIncoreLayout(As<Call>(op->expr_));
    auto new_expr = VisitExpr(op->expr_);
    // Helper: return updated EvalStmt only when the expression actually changed.
    auto maybe_update = [&]() -> StmtPtr {
      if (new_expr.get() == op->expr_.get()) return StmtPtr(op);
      auto result = MutableCopy(op);
      result->expr_ = new_expr;
      return result;
    };

    auto call = As<Call>(new_expr);
    if (!call || std::dynamic_pointer_cast<const GlobalVar>(call->op_)) return maybe_update();

    const auto* entry = conv_registry_.Lookup(call->op_->name_);
    if (!entry) return maybe_update();

    auto [bridged_args, bridge_stmts] =
        BridgeInputSpaces(call, entry->input_reqs, NeedsPackedFp4DnCastLoad(As<Call>(op->expr_), 0));
    auto conv_result = entry->func(bridged_args, call->kwargs_, call->span_, conversion_context_);

    std::vector<StmtPtr> stmts;
    stmts.reserve(bridge_stmts.size() + conv_result.prologue.size() + 1);
    for (auto& s : bridge_stmts) stmts.push_back(std::move(s));
    for (auto& prologue_stmt : conv_result.prologue) {
      stmts.push_back(VisitStmt(prologue_stmt));
    }
    auto new_result = VisitExpr(conv_result.result);
    stmts.push_back(std::make_shared<EvalStmt>(new_result, op->span_));
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

 private:
  /// Track packed storage in O(N); views can inherit pitch absent from TileView.
  void RecordPackedTile(const VarPtr& var, const ExprPtr& value) {
    auto type = As<TileType>(var->GetType());
    if (!type) return;
    const auto view = tile_view_semantics::GetEffectiveTileView(*type);
    if (view.blayout != TileLayout::row_major || view.slayout != TileLayout::none_box ||
        !view.stride.empty()) {
      return;
    }
    bool packed = conversion_context_.packed_tiles.count(value) != 0;
    if (auto call = As<Call>(value)) {
      const auto* entry = LookupOpEntry(call->op_);
      packed = entry && entry->GetOpCategory() == "TileOp" &&
               entry->GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::Functional &&
               !op_predicates::OutputInheritsSourceBuffer(call->op_->name_);
      if (entry && IsOp(call, "tile.set_validshape")) {
        packed = conversion_context_.packed_tiles.count(call->args_[0]) != 0;
      }
    }
    if (packed) conversion_context_.packed_tiles.insert(var);
  }

  StmtPtr HandlePassThroughAssign(const AssignStmtPtr& op, const ExprPtr& new_value) {
    auto result = TypePropagatingMutator::HandlePassThroughAssign(op, new_value);
    auto assign = As<AssignStmt>(result);
    INTERNAL_CHECK_SPAN(assign, op->span_)
        << "Internal error: pass-through assignment must stay an assignment";
    RecordPackedTile(assign->var_, assign->value_);
    return result;
  }

  void SetYieldTileTargets(const std::vector<VarPtr>& vars) {
    yield_tile_targets_.clear();
    for (const auto& var : vars) yield_tile_targets_.push_back(tile_values_.IsTile(var));
  }

  template <typename LoopPtr>
  StmtPtr ConvertLoop(const LoopPtr& op) {
    std::vector<StmtPtr> stmts;
    auto loop = MutableCopy(op);
    auto saved_targets = std::move(yield_tile_targets_);
    SetYieldTileTargets(op->return_vars_);
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      const auto& iter = op->iter_args_[i];
      auto init = VisitExpr(iter->initValue_);
      if (tile_values_.IsTile(iter) && AsTensorTypeLike(init->GetType())) {
        init = LoadTensorOperand(init, iter->span_, stmts);
      }
      if (init != iter->initValue_) {
        auto new_iter = std::make_shared<IterArg>(iter->name_hint_, init->GetType(), init, iter->span_);
        RetainVar(new_iter);
        var_remap_[iter.get()] = new_iter;
        loop->iter_args_[i] = std::move(new_iter);
      }
    }
    stmts.push_back(TypePropagatingMutator::VisitStmt_(LoopPtr(loop)));
    for (const auto& iter : op->iter_args_) var_remap_.erase(iter.get());
    yield_tile_targets_ = std::move(saved_targets);
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  ExprPtr LoadTensorOperand(const ExprPtr& source, const Span& span, std::vector<StmtPtr>& stmts) {
    auto cached = preloaded_tiles_.find(source.get());
    if (cached != preloaded_tiles_.end()) return cached->second;

    auto tensor_type = AsTensorTypeLike(source->GetType());
    INTERNAL_CHECK_SPAN(tensor_type, span) << "Internal error: expected a GM tensor operand";
    auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), span);
    auto valid = MakeShapeTuple(tensor_type->shape_, span);
    auto var = AsVarLike(source);
    auto req = var ? consumer_collector_.GetConsumerReq(var.get()) : std::nullopt;
    // There is no GM -> Acc load. Keep the natural load so memory inference
    // reports the unsupported accumulator input at its consumer.
    if (req && req->space == MemorySpace::Acc) req.reset();
    std::vector<std::pair<std::string, std::any>> kwargs;
    if (req) kwargs.emplace_back("target_memory", req->space);
    AppendCachePolicyKwarg(source, cache_policies_, &kwargs);
    ExprPtr shapes = valid;
    if (req && req->cube_m_align > 0) {
      auto boxed = BoxCubeMAxis(tensor_type->shape_, req->cube_m_align, req->cube_m_axis, span);
      if (!AreExprVectorsEqual(boxed, tensor_type->shape_)) shapes = MakeShapeTuple(boxed, span);
    }
    auto load = MarkCompilerMatBridge(
        op_registry_.Create("tile.load", {source, offsets, shapes, valid}, kwargs, span),
        req ? req->space : MemorySpace::Vec);
    auto tile =
        std::make_shared<Var>(MakeTileValueName(var ? var->name_hint_ : "operand"), load->GetType(), span);
    stmts.push_back(std::make_shared<AssignStmt>(tile, load, span));
    RecordPackedTile(tile, load);
    return tile;
  }

  /// Handle a `tensor.create` that seeds a cube accumulator: allocate whole NZ
  /// fractal boxes on the row axis and declare the requested rectangle as
  /// `valid_shape`.
  ///
  /// `tile.matmul_acc` requires the accumulator and the matrix product to agree
  /// on *physical* M, so the accumulator has to be boxed by exactly the rule
  /// `BoxCubeMAxis` applies to the left operand (see its comment for why the box
  /// binds the physical extent and not the logical one). The operand reaches
  /// that rule through its bridge load; an accumulator is never loaded — there
  /// is no data path into Acc — so its allocation is the only site left.
  ///
  /// The narrowing rides on a separate `tile.set_validshape` because
  /// `tile.create` takes no valid extent. It is metadata-only, so the pair costs
  /// nothing. Returns nullptr when no padding applies, leaving the historical
  /// single-`tile.create` lowering byte-identical.
  StmtPtr HandleBoxedAccCreate(const AssignStmtPtr& op, const CallPtr& call, const ConsumerSpaceReq& req) {
    if (call->args_.size() != 1) return nullptr;
    auto shape_tuple = As<MakeTuple>(call->args_[0]);
    if (!shape_tuple || shape_tuple->elements_.size() != 2) return nullptr;
    const auto& shape = shape_tuple->elements_;

    auto boxed = BoxCubeMAxis(shape, req.cube_m_align, req.cube_m_axis, call->span_);
    if (AreExprVectorsEqual(boxed, shape)) return nullptr;

    // Route the boxed allocation through the registered converter so it keeps
    // the capacity check and the kwarg filtering `tensor.create` performs.
    const auto* entry = conv_registry_.Lookup("tensor.create");
    INTERNAL_CHECK_SPAN(entry, call->span_)
        << "Internal error: tensor.create has no registered tile conversion";
    auto converted = As<Call>(
        entry->func({MakeShapeTuple(boxed, call->span_)}, call->kwargs_, call->span_, conversion_context_)
            .result);
    INTERNAL_CHECK_SPAN(converted, call->span_)
        << "Internal error: the tensor.create conversion must produce a Call";

    // ... then stamp the space. The converter deliberately leaves `tensor.create`
    // unresolved because it has no consumer context to derive a space from; here
    // there is one, and it is the same demand that asked for the boxing. Stating
    // it matters beyond saving InferTileMemorySpace the work: an Acc tile's
    // implicit view is boxed NZ, so a seed left unresolved would carry the raw
    // row-major view and disagree with the `tile.matmul_acc` result it is
    // carried against across a loop.
    auto create_kwargs = converted->kwargs_;
    create_kwargs.emplace_back("target_memory", req.space);
    // Compact, because the padding is what makes the two readings of an L0C
    // stride disagree. `mad` lays the product out at a pitch of
    // ceil(validRow/16)*16 -- 112 for a 100-row product -- while a non-compact
    // reader derives its stride from the physical row count, which the box
    // rounded to 112 or, at a 32-row alignment (a transposed INT8 operand), to
    // 128. Compact makes every reader recompute the pitch `mad` actually used.
    // Reaching here means padding applies, so the two never coincide by
    // accident; `AccCompactValid` rejects the unstamped form outright.
    if (req.space == MemorySpace::Acc) create_kwargs.emplace_back("compact", true);
    auto storage = op_registry_.Create("tile.create", converted->args_, create_kwargs, call->span_);

    auto storage_var = std::make_shared<Var>(MakeTileValueName(op->var_->name_hint_) + "_storage",
                                             storage->GetType(), op->var_->span_);
    auto narrowed =
        op_registry_.Create("tile.set_validshape", {storage_var, shape[0], shape[1]}, call->span_);
    auto tile_var =
        std::make_shared<Var>(MakeTileValueName(op->var_->name_hint_), narrowed->GetType(), op->var_->span_);
    var_remap_[op->var_.get()] = tile_var;

    std::vector<StmtPtr> stmts = {
        std::make_shared<AssignStmt>(storage_var, storage, op->span_),
        std::make_shared<AssignStmt>(tile_var, narrowed, op->span_),
    };
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  /// Handle tensor.slice whose consumer needs a specific memory space — produce tile.load with that space.
  StmtPtr HandleConsumerDrivenLoad(const AssignStmtPtr& op, const CallPtr& call,
                                   const ConsumerSpaceReq& req) {
    // Acc is not a load target (see BridgeInputSpaces): nothing but the matrix
    // unit writes L0C.
    if (req.space == MemorySpace::Acc) return nullptr;
    const auto& input = call->args_[0];
    auto tensor_type = AsTensorTypeLike(input->GetType());
    if (!tensor_type) return nullptr;

    const auto& shape_arg = call->args_[1];
    const auto& offset_arg = call->args_[2];
    auto shape_type = As<TupleType>(shape_arg->GetType());
    auto offset_type = As<TupleType>(offset_arg->GetType());
    INTERNAL_CHECK_SPAN(shape_type && offset_type, call->span_)
        << "Internal error: tensor.slice shape and offset must be tuples during conversion";
    auto full_shape = ExtractTupleElements(shape_arg, shape_type->types_.size());
    auto offsets = ExtractTupleElements(offset_arg, offset_type->types_.size());
    std::vector<ExprPtr> requested_valid;
    if (call->args_.size() >= 4) {
      auto valid_shape_tuple = As<MakeTuple>(call->args_[3]);
      INTERNAL_CHECK_SPAN(valid_shape_tuple, call->span_)
          << "Internal error: tensor.slice valid_shape must be a MakeTuple during conversion";
      requested_valid = valid_shape_tuple->elements_;
    }
    auto valid_shape = MakeShapeTuple(
        InferTensorSliceFullValidShape(*tensor_type, full_shape, offsets, requested_valid,
                                       GetKwargOr<bool>(call->kwargs_, "clamp", false), call->span_),
        call->span_);
    auto drop_dims = ParseSliceDropDims(call->args_.size() == 5 ? call->args_[4] : nullptr, full_shape,
                                        "tensor.slice conversion");
    CHECK_SPAN(drop_dims.size() < full_shape.size(), call->span_)
        << "tensor.slice conversion does not support rank-0 tiles; keep one unit axis or use tensor.read "
           "for a scalar result";

    // A cube operand answered here rather than in BridgeInputSpaces must get the
    // same row boxing, or a sliced left operand keeps its unaligned physical row
    // count all the way to ptoas. `valid_shape` already names the slice window,
    // so only the allocation grows. Skipped when the slice drops a dimension:
    // the result is then not the rank-2 tile the M-axis rule is about.
    ExprPtr physical_shape_arg = shape_arg;
    if (req.cube_m_align > 0 && drop_dims.empty()) {
      auto boxed = BoxCubeMAxis(full_shape, req.cube_m_align, req.cube_m_axis, call->span_);
      if (!AreExprVectorsEqual(boxed, full_shape)) {
        physical_shape_arg = MakeShapeTuple(boxed, call->span_);
      }
    }

    // The consumer-driven load is always natural; a transposed (b_trans/a_trans)
    // operand gets a zero-copy tile.transpose_view at the matmul site instead.
    std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", req.space}};
    AppendCachePolicyKwarg(input, cache_policies_, &load_kwargs);
    auto load_call = MarkCompilerMatBridge(
        op_registry_.Create("tile.load", {input, offset_arg, physical_shape_arg, valid_shape}, load_kwargs,
                            call->span_),
        req.space);

    auto tile_name = MakeTileValueName(op->var_->name_hint_);
    if (drop_dims.empty()) {
      auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), op->var_->span_);
      var_remap_[op->var_.get()] = tile_var;
      auto result = MutableCopy(op);
      result->var_ = tile_var;
      result->value_ = load_call;
      return result;
    }

    auto loaded_var = std::make_shared<Var>(tile_name + "_full_rank", load_call->GetType(), op->var_->span_);
    auto reduced_shape = MakeShapeTuple(ApplyDropDims(full_shape, drop_dims), call->span_);
    auto reshape_call = op_registry_.Create("tile.reshape", {loaded_var, reduced_shape}, {}, call->span_);
    auto tile_var = std::make_shared<Var>(tile_name, reshape_call->GetType(), op->var_->span_);
    var_remap_[op->var_.get()] = tile_var;
    std::vector<StmtPtr> stmts = {
        std::make_shared<AssignStmt>(loaded_var, load_call, op->span_),
        std::make_shared<AssignStmt>(tile_var, reshape_call, op->span_),
    };
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  /// Restore a DN cast's value layout after an earlier producer materialized it
  /// as an ordinary row-major BF16 tile. A slice is already a value snapshot:
  /// reloading its GM source here could observe intervening writes, and changing
  /// the shared tile would alter other consumers. Transpose into fresh storage,
  /// then reinterpret that transposed matrix as the original column-major one.
  ExprPtr PreparePackedFp4DnCastTile(const ExprPtr& source, const Span& span, std::vector<StmtPtr>& stmts) {
    auto type = As<TileType>(source->GetType());
    INTERNAL_CHECK_SPAN(type && type->dtype_ == DataType::BF16, span)
        << "Internal error: packed FP4 DN cast preparation requires a BF16 tile";
    const auto view = tile_view_semantics::GetEffectiveTileView(*type);
    if (view.blayout == TileLayout::col_major) return source;

    const bool static_2d = type->shape_.size() == 2 &&
                           std::all_of(type->shape_.begin(), type->shape_.end(), [](const ExprPtr& extent) {
                             auto value = As<ConstInt>(extent);
                             return value && value->value_ > 0;
                           });
    // tile.transpose allocates a full-valid result, so check before it can erase
    // partial/padded validity. The native DN cast only accepts dense rectangles.
    CHECK_SPAN(static_2d && view.slayout == TileLayout::none_box &&
                   tile_view_semantics::ShapeExprListsEquivalent(view.valid_shape, type->shape_),
               span)
        << "packed FP4 DN cast requires a full static 2D valid shape in unboxed storage; partial or "
           "padded DN casts are unsupported";
    const std::vector<ExprPtr> expected_stride = {type->shape_[1],
                                                  std::make_shared<ConstInt>(1, DataType::INDEX, span)};
    CHECK_SPAN(
        view.stride.empty() || tile_view_semantics::ShapeExprListsEquivalent(view.stride, expected_stride),
        span)
        << "packed FP4 DN cast preparation requires a contiguous row-major source tile";

    auto transpose = op_registry_.Create("tile.transpose",
                                         {source, std::make_shared<ConstInt>(0, DataType::INDEX, span),
                                          std::make_shared<ConstInt>(1, DataType::INDEX, span)},
                                         {}, span);
    auto transposed = std::make_shared<Var>("fp4_dn_transposed", transpose->GetType(), span);
    stmts.push_back(std::make_shared<AssignStmt>(transposed, transpose, span));
    auto column_view = op_registry_.Create("tile.transpose_view", {transposed}, {}, span);
    auto column_tile = std::make_shared<Var>("fp4_dn_values", column_view->GetType(), span);
    stmts.push_back(std::make_shared<AssignStmt>(column_tile, column_view, span));
    return column_tile;
  }

  /// Auto-bridge TensorType args to the memory space required by input_reqs.
  /// Returns the (possibly modified) args and any load statements to prepend.
  std::pair<std::vector<ExprPtr>, std::vector<StmtPtr>> BridgeInputSpaces(
      const CallPtr& call, const std::unordered_map<size_t, InputSpaceReq>& input_reqs,
      bool preserve_packed_fp4_dn) {
    auto args = call->args_;
    std::vector<StmtPtr> stmts;
    // Preserve the tensor call's layout intent from before VisitExpr remapped
    // its producer. The rewritten operand may already be a row-major TileType,
    // which the tensor-only GM bridge below cannot recognize as a DN cast.
    if (preserve_packed_fp4_dn && As<TileType>(args[0]->GetType())) {
      args[0] = PreparePackedFp4DnCastTile(args[0], call->span_, stmts);
    }

    // The default requirements are the same ones used to select read-only
    // entry loads. Apply them only to this call's operands. GM memory ops keep
    // their source/destination handles even if another consumer loaded them.
    if (UsesDefaultTileInputs(call)) {
      std::unordered_map<const Expr*, ExprPtr> loaded_args;
      for (size_t i = 0; i < args.size(); ++i) {
        if (input_reqs.count(i) || !AsTensorTypeLike(args[i]->GetType())) continue;
        if (NeedsPackedFp4DnCastLoad(call, i)) {
          auto source_type = AsTensorTypeLike(args[i]->GetType());
          auto shape = source_type->shape_;
          INTERNAL_CHECK_SPAN(shape.size() == 2, call->span_)
              << "Internal error: dense packed FP4 DN cast must be two-dimensional";
          std::swap(shape[0], shape[1]);
          auto storage_view =
              op_registry_.Create("tensor.view", {args[i], MakeShapeTuple(shape, call->span_)},
                                  {{"layout", TensorLayout::ND}}, call->span_);
          auto storage = std::make_shared<Var>("fp4_dn_storage", storage_view->GetType(), call->span_);
          stmts.push_back(std::make_shared<AssignStmt>(storage, storage_view, call->span_));
          auto extent = MakeShapeTuple(shape, call->span_);
          std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec}};
          AppendCachePolicyKwarg(args[i], cache_policies_, &load_kwargs);
          auto load =
              op_registry_.Create("tile.load", {storage, MakeZeroOffsets(2, call->span_), extent, extent},
                                  load_kwargs, call->span_);
          auto loaded = std::make_shared<Var>("fp4_dn_loaded", load->GetType(), call->span_);
          stmts.push_back(std::make_shared<AssignStmt>(loaded, load, call->span_));
          RecordPackedTile(loaded, load);
          auto transpose = op_registry_.Create("tile.transpose_view", {loaded}, {}, call->span_);
          auto column_tile = std::make_shared<Var>("fp4_dn_values", transpose->GetType(), call->span_);
          stmts.push_back(std::make_shared<AssignStmt>(column_tile, transpose, call->span_));
          args[i] = std::move(column_tile);
          continue;
        }
        auto [it, inserted] = loaded_args.try_emplace(args[i].get());
        if (inserted) it->second = LoadTensorOperand(args[i], call->span_, stmts);
        args[i] = it->second;
      }
    }
    if (input_reqs.empty()) return {std::move(args), std::move(stmts)};

    // An operand of rank > 2 means this matmul lowers to tile.batch_matmul, not
    // tile.matmul (see the rank dispatch in op_conversion_registry.cpp).
    //
    // Both 2D tile.matmul AND ND tile.batch_matmul realise a transposed operand
    // as a zero-copy tile.transpose_view over ONE natural load/move (issues #1776
    // / ND extension). FlattenTileNdTo2D slices the whole transposed view per
    // batch; the tile-level (batch_)matmul carries no transpose semantic.

    // Emit a `tile.load` of `arg` (TensorType) into `space`, append its AssignStmt,
    // and return the bound load Var. The load is always natural; a transposed
    // operand is realised by a zero-copy tile.transpose_view on the result.
    //
    // `m_align` / `m_axis` mark a cube operand (see InputSpaceReq::cube_m_axis):
    // the load then allocates a whole number of NZ fractal boxes on the axis that
    // carries M and declares the tensor's true extent as valid_shape.
    // `BoxCubeMAxis` returns the physical shape; it equals `tensor_type->shape_`
    // whenever no padding applies, so a fractal-sized operand keeps its
    // historical byte-identical load.
    auto emit_load = [&](const ExprPtr& arg, const TensorTypePtr& tensor_type, MemorySpace space, size_t idx,
                         int64_t m_align, size_t m_axis) -> VarPtr {
      auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), call->span_);
      auto valid = MakeShapeTuple(tensor_type->shape_, call->span_);
      auto boxed = BoxCubeMAxis(tensor_type->shape_, m_align, m_axis, call->span_);
      auto shapes =
          AreExprVectorsEqual(boxed, tensor_type->shape_) ? valid : MakeShapeTuple(boxed, call->span_);
      std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", space}};
      AppendCachePolicyKwarg(arg, cache_policies_, &load_kw);
      auto load = MarkCompilerMatBridge(
          op_registry_.Create("tile.load", {arg, offsets, shapes, valid}, load_kw, call->span_), space);
      std::string var_name;
      if (auto var = As<Var>(arg)) {
        auto space_str = MemorySpaceToString(space);
        std::transform(space_str.begin(), space_str.end(), space_str.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        var_name = var->name_hint_ + "_" + space_str;
      } else {
        var_name = "bridged_" + std::to_string(idx);
      }
      auto load_var = std::make_shared<Var>(var_name, load->GetType(), call->span_);
      stmts.push_back(std::make_shared<AssignStmt>(load_var, load, call->span_));
      return load_var;
    };

    // Iterate in sorted index order to produce deterministic statement ordering.
    std::vector<size_t> sorted_indices;
    sorted_indices.reserve(input_reqs.size());
    for (const auto& [idx, _] : input_reqs) sorted_indices.push_back(idx);
    std::sort(sorted_indices.begin(), sorted_indices.end());

    // Zero-copy reinterpret a Mat-resident tile as its transpose (NZ<->ZN),
    // aliasing the SAME L1 buffer (issue #1776). Valid only for Mat tiles.
    auto emit_view = [&](const VarPtr& v) -> VarPtr {
      auto view = op_registry_.Create("tile.transpose_view", {v}, {}, call->span_);
      auto view_var = std::make_shared<Var>(v->name_hint_ + "_t", view->GetType(), call->span_);
      stmts.push_back(std::make_shared<AssignStmt>(view_var, view, call->span_));
      return view_var;
    };

    // Bridge a non-Mat (e.g. Vec) tile to Mat in its NATURAL orientation via a
    // V2C move. transpose_view needs a Mat-resident tile (a col_major Vec tile
    // cannot be pushed V2C — a2a3 TPUSH transfers only ND/NZ tiles), so a Vec
    // compute result feeding a b_trans matmul (mixed kernel) is moved to Mat with
    // its original shape first, then reinterpreted as its transpose on the Mat side.
    auto emit_move_to_mat = [&](const VarPtr& v) -> VarPtr {
      std::vector<std::pair<std::string, std::any>> move_kw = {{"target_memory", MemorySpace::Mat}};
      auto mv = op_registry_.Create("tile.move", {v}, move_kw, call->span_);
      auto mv_var = std::make_shared<Var>(v->name_hint_ + "_mat", mv->GetType(), call->span_);
      stmts.push_back(std::make_shared<AssignStmt>(mv_var, mv, call->span_));
      return mv_var;
    };

    for (size_t idx : sorted_indices) {
      const auto& req = input_reqs.at(idx);
      if (idx >= args.size()) continue;
      // Acc is never a bridge target: only the matrix unit writes L0C, so no
      // load can put a GM tensor there. An Acc req exists to carry the cube
      // row-box demand back to the operand's producer (HandleBoxedAccCreate);
      // leaving the operand alone here keeps InferTileMemorySpace's "no data
      // path into Acc memory" diagnostic, which names the real limitation.
      if (req.demanded_space.Get() == MemorySpace::Acc) continue;
      const bool use_view = req.trans_kwarg ? call->GetKwarg<bool>(*req.trans_kwarg, false) : false;
      // A window operand bridges exactly like a plain GM tensor (issue #1694):
      // ``AsTensorTypeLike`` matches both kinds. With the exact-kind
      // ``As<TensorType>`` a ``pld.DistributedTensor`` reaching a matmul directly
      // fell through to the tile branch, passed through unbridged, and tripped the
      // converter's unreachable guard.
      auto tensor_type = AsTensorTypeLike(args[idx]->GetType());

      if (tensor_type) {
        // GM operand: load NATURAL (2D and ND alike), then reinterpret as its
        // transpose with a zero-copy view when b_trans/a_trans.
        // A transposed operand is boxed on its *column* axis: the
        // tile.transpose_view below reinterprets the row axis as the matmul's K,
        // so M is the column extent the natural load allocates.
        const auto [box_align, boxed_axis] = ResolveBoxedAxis(call, input_reqs, idx, req);
        auto loaded = emit_load(args[idx], tensor_type, req.demanded_space.Get(), idx, box_align, boxed_axis);
        args[idx] = use_view ? emit_view(loaded) : loaded;
        continue;
      }

      // Tile operand. Only a transposed operand needs rewriting; a non-transposed
      // tile passes through.
      if (!use_view) continue;
      auto var = As<Var>(args[idx]);
      if (!var) continue;  // non-Var tile expr: leave as-is
      auto tile_ty = As<TileType>(var->GetType());
      const bool mat_resident =
          tile_ty && tile_ty->memory_space_.value_or(MemorySpace::Vec) == MemorySpace::Mat;
      // Mat-resident operand (e.g. a consumer-driven Mat load): zero-copy view.
      // Non-Mat operand (Vec compute result, mixed kernel): move to Mat in its
      // natural shape first, then view the Mat tile.
      args[idx] = mat_resident ? emit_view(var) : emit_view(emit_move_to_mat(var));
    }

    return {std::move(args), std::move(stmts)};
  }

  const OpConversionRegistry& conv_registry_;
  const OpRegistry& op_registry_;
  const ConsumerSpaceCollector& consumer_collector_;
  const TensorConversionAnalysis& tile_values_;
  std::unordered_map<const Expr*, ExprPtr> preloaded_tiles_;
  ConversionContext conversion_context_;
  std::vector<bool> yield_tile_targets_;
  /// Declared GM cache policies of this function's params (empty when the
  /// function carries no ``pl.set_cache_policy`` declaration).
  const CachePolicyByParam cache_policies_;
};

struct ReturnedAssembleLoopRewrite {
  size_t stmt_index;
  std::optional<size_t> dead_init_stmt_index;
  ForStmtPtr new_for_stmt;
  VarPtr new_return_var;
};

std::optional<ReturnedAssembleLoopRewrite> RewriteReturnedAssembleLoopToStore(
    const std::vector<StmtPtr>& stmts, const ExprPtr& ret_expr, const VarPtr& out_param,
    const TensorTypePtr& out_tensor_type, const OpRegistry& op_registry) {
  auto ret_var = As<Var>(ret_expr);
  if (!ret_var) return std::nullopt;

  for (size_t stmt_index = 0; stmt_index < stmts.size(); ++stmt_index) {
    auto for_stmt = As<ForStmt>(stmts[stmt_index]);
    if (!for_stmt || for_stmt->iter_args_.size() != 1 || for_stmt->return_vars_.size() != 1 ||
        for_stmt->return_vars_[0].get() != ret_var.get()) {
      continue;
    }

    const auto& old_iter_arg = for_stmt->iter_args_[0];
    auto body_stmts = FlattenToStmts(for_stmt->body_);

    AssignStmtPtr assemble_assign;
    YieldStmtPtr yield_stmt;
    for (const auto& body_stmt : body_stmts) {
      auto assign = As<AssignStmt>(body_stmt);
      if (assign) {
        auto call = As<Call>(assign->value_);
        bool is_target_assemble = false;
        if (call && IsOp(call, "tile.assemble") && call->args_.size() == 3) {
          if (auto iter = As<IterArg>(call->args_[0])) {
            is_target_assemble = iter.get() == old_iter_arg.get();
          } else if (auto var = As<Var>(call->args_[0])) {
            is_target_assemble = var.get() == old_iter_arg.get();
          }
        }
        if (is_target_assemble) {
          if (assemble_assign) return std::nullopt;
          auto assemble_call = As<Call>(assign->value_);
          INTERNAL_CHECK_SPAN(assemble_call, assign->span_)
              << "Internal error: expected tile.assemble call in assemble loop rewrite";
          if (ExprUsesVar(assemble_call->args_[1], old_iter_arg.get()) ||
              ExprUsesVar(assemble_call->args_[2], old_iter_arg.get())) {
            return std::nullopt;
          }
          assemble_assign = assign;
          continue;
        }
      }

      if (auto yield = As<YieldStmt>(body_stmt)) {
        if (yield->value_.size() != 1 || yield_stmt) return std::nullopt;
        yield_stmt = yield;
        continue;
      }

      if (StmtUsesVar(body_stmt, old_iter_arg.get())) {
        return std::nullopt;
      }
    }

    if (!assemble_assign || !yield_stmt) return std::nullopt;

    auto yielded_var = As<Var>(yield_stmt->value_[0]);
    if (!yielded_var || yielded_var.get() != assemble_assign->var_.get()) {
      return std::nullopt;
    }

    auto assemble_call = As<Call>(assemble_assign->value_);
    INTERNAL_CHECK_SPAN(assemble_call, assemble_assign->span_)
        << "Internal error: expected tile.assemble call in assemble loop rewrite";

    auto new_iter_arg =
        std::make_shared<IterArg>(old_iter_arg->name_hint_, out_tensor_type, out_param, old_iter_arg->span_);
    auto store_call = op_registry.Create(
        "tile.store", {assemble_call->args_[1], assemble_call->args_[2], new_iter_arg}, assemble_call->span_);
    auto store_var = std::make_shared<Var>(assemble_assign->var_->name_hint_, store_call->GetType(),
                                           assemble_assign->var_->span_);

    std::vector<StmtPtr> new_body_stmts;
    new_body_stmts.reserve(body_stmts.size());
    for (const auto& body_stmt : body_stmts) {
      if (body_stmt == assemble_assign) {
        auto new_assign = MutableCopy(assemble_assign);
        new_assign->var_ = store_var;
        new_assign->value_ = store_call;
        new_body_stmts.push_back(std::move(new_assign));
        continue;
      }
      if (body_stmt == yield_stmt) {
        auto new_yield = MutableCopy(yield_stmt);
        new_yield->value_ = std::vector<ExprPtr>{store_var};
        new_body_stmts.push_back(std::move(new_yield));
        continue;
      }
      new_body_stmts.push_back(body_stmt);
    }

    auto new_return_var = std::make_shared<Var>(for_stmt->return_vars_[0]->name_hint_, out_tensor_type,
                                                for_stmt->return_vars_[0]->span_);
    auto new_for_stmt = MutableCopy(for_stmt);
    new_for_stmt->iter_args_ = std::vector<IterArgPtr>{new_iter_arg};
    new_for_stmt->body_ = SeqStmts::Flatten(std::move(new_body_stmts), for_stmt->body_->span_);
    new_for_stmt->return_vars_ = std::vector<VarPtr>{new_return_var};

    std::optional<size_t> dead_init_stmt_index;
    if (auto init_var = As<Var>(old_iter_arg->initValue_)) {
      bool has_other_uses = false;
      for (size_t other_index = 0; other_index < stmts.size(); ++other_index) {
        const StmtPtr& stmt_to_check = other_index == stmt_index ? new_for_stmt : stmts[other_index];
        if (StmtUsesVar(stmt_to_check, init_var.get())) {
          has_other_uses = true;
          break;
        }
      }
      if (!has_other_uses) {
        for (size_t other_index = 0; other_index < stmts.size(); ++other_index) {
          auto init_assign = As<AssignStmt>(stmts[other_index]);
          if (init_assign && init_assign->var_.get() == init_var.get()) {
            dead_init_stmt_index = other_index;
            break;
          }
        }
      }
    }

    return ReturnedAssembleLoopRewrite{
        stmt_index,
        dead_init_stmt_index,
        new_for_stmt,
        new_return_var,
    };
  }

  return std::nullopt;
}

struct ReturnOutputBinding {
  size_t return_index;
  TypePtr original_type;
  ExprPtr value;
  std::optional<size_t> storage_index;
};

// A caller allocates new outputs before entering this function, so their types
// cannot depend on body-local SSA values. Do not speculate about data-dependent
// extents: callers must supply an explicit Out or a signature-visible extent.
class OutputTypeScopeValidator : public IRVisitor {
 public:
  OutputTypeScopeValidator(const FunctionPtr& func, const StmtPtr& body, Span span)
      : function_name_(func->name_), span_(std::move(span)) {
    var_collectors::VarDefUseCollector collector;
    collector.VisitStmt(body);
    body_locals_ = std::move(collector.var_defs);
  }

  void Validate(const TypePtr& type) {
    if (seen_types_.insert(type.get()).second) var_collectors::VisitTypeExprFields(*this, type);
  }

  void VisitExpr(const ExprPtr& expr) override {
    // Output types can share shape-expression DAGs. Check each node once across
    // all outputs rather than expanding the same dimensions per return slot.
    if (expr && seen_exprs_.insert(expr.get()).second) IRVisitor::VisitExpr(expr);
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    CHECK_SPAN(body_locals_.count(op.get()) == 0, span_)
        << "Cannot materialize output of '" << function_name_
        << "': output type depends on body-local dimension '" << op->name_hint_
        << "'. Pass the extent as a scalar parameter or provide an explicit Out tensor.";
  }

 private:
  std::string function_name_;
  Span span_;
  std::unordered_set<const Var*> body_locals_;
  std::unordered_set<const Type*> seen_types_;
  std::unordered_set<const Expr*> seen_exprs_;
};

// A returned GM allocation becomes caller-owned storage. Removing its local
// allocation preserves the one-buffer identity instead of copying the result
// into a second allocation or confusing it with a kernel input.
class ReturnedAllocationRewriter : public IRMutator {
 public:
  explicit ReturnedAllocationRewriter(const std::unordered_map<const Var*, VarPtr>& outputs)
      : outputs_(outputs) {
    for (const auto& [source, output] : outputs) var_remap_.emplace(source, output);
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (outputs_.count(op->var_.get()) != 0 && IsOp(As<Call>(op->value_), "tensor.create")) {
      return SeqStmts::Flatten({}, op->span_);
    }
    return IRMutator::VisitStmt_(op);
  }

 private:
  const std::unordered_map<const Var*, VarPtr>& outputs_;
};

}  // namespace

// Lower a kernel body and bind every newly materialized output to an explicit
// parameter, retaining the kernel's InCore/AIC/AIV function type.
IncoreTransformResult TransformIncoreFunction(const FunctionPtr& func) {
  auto& conv_registry = OpConversionRegistry::GetInstance();
  auto& op_registry = OpRegistry::GetInstance();
  // Structural nodes (body SeqStmts, the rebuilt Function) belong to the whole
  // function, so they carry its span.  Synthesized *ops* must not: an entry load
  // is attributed to the parameter that motivated it and an exit store to the
  // `return` that motivated it, so post-pass diagnostics point at real source
  // lines instead of the `def` line.
  const auto& span = func->span_;

  auto canonical_body = CanonicalizeScalarUpdates(func);
  // Save every logical return before lowering can change its storage identity.
  // In particular, tensor.create is otherwise promoted to tile.create, which
  // would turn an existing GM store destination into a tile and require a
  // second output buffer at the return.
  ReturnStmtPtr return_stmt;
  for (const auto& stmt : FlattenToStmts(canonical_body)) {
    if (auto ret = As<ReturnStmt>(stmt)) return_stmt = ret;
  }
  std::vector<ReturnOutputBinding> outputs;
  if (return_stmt) {
    outputs.reserve(return_stmt->value_.size());
    for (size_t i = 0; i < return_stmt->value_.size(); ++i) {
      const auto& value = return_stmt->value_[i];
      // Unannotated functions can carry a return value without declared
      // return_types. Preserve its original type before conversion as well.
      const auto& type = i < func->return_types_.size() ? func->return_types_[i] : value->GetType();
      outputs.push_back({i, type, value, std::nullopt});
    }
  }

  std::vector<VarPtr> new_params = func->params_;
  std::vector<ParamDirection> new_param_directions = func->param_directions_;
  std::vector<TypePtr> new_return_types;
  AddedOutputBindings added_outputs;
  const size_t output_start = func->params_.size() - transform_utils::TrailingCommCtxParamCount(func);
  OutputTypeScopeValidator output_types(func, canonical_body, return_stmt ? return_stmt->span_ : span);
  auto add_output = [&](const TypePtr& type) {
    output_types.Validate(type);
    const size_t param_index = output_start + added_outputs.size();
    auto param = std::make_shared<Var>(MakeOutParamName(added_outputs.size()), type, return_stmt->span_);
    // Append now and rotate the unchanged CommCtx suffix once at the end.
    new_params.push_back(param);
    new_param_directions.push_back(ParamDirection::Out);
    added_outputs.push_back({param_index});
    return param;
  };

  class AllocationCollector : public IRVisitor {
   public:
    std::vector<VarPtr> vars;
    std::vector<StmtPtr> storage_definitions;
    std::vector<ExprPtr> gm_uses;

   protected:
    void VisitExpr_(const CallPtr& op) override {
      // Pre-lift only storage that an explicit memory operation requires in
      // GM. Tensor-level local updates retain tile staging and one exit store.
      if (IsOp(op, "tile.store") && op->args_.size() > 2) {
        gm_uses.push_back(op->args_[2]);
      } else if (IsOp(op, "tile.load") && !op->args_.empty()) {
        gm_uses.push_back(op->args_[0]);
      } else if (IsOp(op, "tensor.expand_clone") && op->args_.size() == 2) {
        // This converter supplies its own GM loads/stores and requires both
        // source and destination to remain TensorType.
        gm_uses.insert(gm_uses.end(), op->args_.begin(), op->args_.end());
      }
      IRVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (IsOp(call, "tensor.create") && As<TensorType>(op->var_->GetType())) {
        vars.push_back(op->var_);
      }
      // Prove GM identity using only aliases and explicit storage writes.
      // Indexing a flattened definition list deliberately excludes loop carry
      // init edges: a functional accumulator may return a new tile even when
      // the loop was seeded by this allocation. Such values must retain tile
      // promotion and receive their output store after lowering.
      const bool gm_write = IsOp(call, "tile.store") || IsOp(call, "tensor.write") ||
                            IsOp(call, "tensor.assemble") || IsOp(call, "tensor.expand_clone");
      if (AsVarLike(op->value_) || As<TupleGetItemExpr>(op->value_) || As<MakeTuple>(op->value_) ||
          gm_write) {
        storage_definitions.push_back(op);
      }
      IRVisitor::VisitStmt_(op);
    }
  } allocations;
  allocations.VisitStmt(canonical_body);
  if (!allocations.vars.empty() && !outputs.empty()) {
    std::vector<VarPtr> roots = func->params_;
    roots.insert(roots.end(), allocations.vars.begin(), allocations.vars.end());
    std::vector<ExprPtr> returned_values;
    returned_values.reserve(outputs.size());
    for (const auto& output : outputs) returned_values.push_back(output.value);
    const auto storage_body = SeqStmts::Flatten(allocations.storage_definitions, span);
    const auto indices =
        return_lineage::TraceExprsToParamIndices(returned_values, storage_body, roots, nullptr);
    // Storage identity alone does not require a GM parameter. Ordinary local
    // allocations, including Tensor-level updates and control-flow aliases,
    // keep their original promotion and receive one output store after lowering.
    const auto gm_indices =
        return_lineage::TraceExprsToParamIndices(allocations.gm_uses, storage_body, roots, nullptr);
    std::unordered_set<const Var*> gm_roots;
    for (const auto& index : gm_indices) {
      if (index && *index >= func->params_.size()) gm_roots.insert(roots[*index].get());
    }
    std::unordered_map<const Var*, VarPtr> allocation_outputs;
    for (const auto& index : indices) {
      if (!index || *index < func->params_.size()) continue;
      const auto& allocation = roots[*index];
      if (gm_roots.count(allocation.get()) == 0) continue;
      auto [it, inserted] = allocation_outputs.emplace(allocation.get(), nullptr);
      if (inserted) it->second = add_output(allocation->GetType());
    }
    // Only complete, type/view-preserving aliases resolve to an allocation
    // root. All returns of that allocation share this one caller-owned Out.
    if (!allocation_outputs.empty()) {
      canonical_body = ReturnedAllocationRewriter(allocation_outputs).VisitStmt(canonical_body);
    }
  }

  // Pre-scan: collect consumer memory space requirements (e.g. tensor.slice → tensor.matmul
  // needs Mat-space loads).  Driven by InputSpaceReq metadata in OpConversionRegistry.
  // Then propagate demands backward through pass-through ops (tensor.fillpad etc.) so a
  // chain like `slice → fillpad → matmul` routes the slice's load directly into Mat.
  ConsumerSpaceCollector consumer_collector(conv_registry);
  consumer_collector.VisitStmt(canonical_body);
  consumer_collector.PropagateThroughInheritInputOps();

  // Resolve the scope-declared GM cache policies onto this function's params.
  // Every load this pass synthesises below, and every load already in the body,
  // carries the declaration onward as a ``cache`` kwarg; the function attr is
  // erased when the transformed function is rebuilt.
  auto cache_policies = BuildCachePolicyByParam(func);

  TensorConversionAnalysis tile_values(conv_registry);
  tile_values.VisitStmt(canonical_body);
  tile_values.Propagate();

  TensorToTileMutator mutator(conv_registry, op_registry, consumer_collector, tile_values, cache_policies);

  // New body statements (prefix tile.loads + mutated body)
  std::vector<StmtPtr> new_stmts;

  // Phase 1: Share entry loads for read-only default compute operands. These
  // tiles are kept in a separate cache, never installed in the SSA var map.
  TensorArgsInConvertedOpsCollector collector(conv_registry);
  collector.VisitStmt(canonical_body);
  collector.TraceIterArgInitValues([&tile_values](const Var* var) { return tile_values.IsTile(var); });
  const auto& params_used_by_converted_ops = collector.GetUsed();
  for (const auto& var : func->params_) {
    if (!AsTensorTypeLike(var->GetType()) || tile_values.MayWriteSource(var) ||
        tile_values.HasOpaqueCalls()) {
      continue;
    }
    if (params_used_by_converted_ops.count(var.get())) {
      new_stmts.push_back(mutator.PreloadReadOnlyParam(var));
    }
  }

  // Phase 2: Transform body via mutator (handles control flow recursion + op conversion)
  auto body_stmts = FlattenToStmts(canonical_body);

  // Separate return statement from body (will be replaced in Phase 3)
  std::vector<StmtPtr> non_return_stmts;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      return_stmt = ret;
    } else {
      non_return_stmts.push_back(stmt);
    }
  }

  auto body_to_transform = SeqStmts::Flatten(std::move(non_return_stmts), span);
  auto mutated = mutator.VisitStmt(body_to_transform);
  auto transformed = FlattenToStmts(mutated);
  new_stmts.insert(new_stmts.end(), transformed.begin(), transformed.end());

  // Refresh the saved return bindings with the rewritten GM outputs and
  // converted computations. Data dependencies are not storage aliases.
  if (return_stmt) {
    for (auto& output : outputs) {
      output.value = mutator.VisitExpr(return_stmt->value_[output.return_index]);
    }
  }

  class TileRootCollector : public IRVisitor {
   public:
    std::vector<VarPtr> vars;

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      if (As<TileType>(op->var_->GetType()) && !AsVarLike(op->value_)) {
        vars.push_back(op->var_);
      }
      IRVisitor::VisitStmt_(op);
    }
  } tile_roots;
  auto indexed_body = SeqStmts::Flatten(new_stmts, span);
  tile_roots.VisitStmt(indexed_body);
  std::vector<VarPtr> storage_roots = new_params;
  storage_roots.insert(storage_roots.end(), tile_roots.vars.begin(), tile_roots.vars.end());
  std::vector<ExprPtr> returned_values;
  returned_values.reserve(outputs.size());
  for (const auto& output : outputs) returned_values.push_back(output.value);
  auto storage_indices =
      return_lineage::TraceExprsToParamIndices(returned_values, indexed_body, storage_roots, nullptr);
  for (size_t i = 0; i < outputs.size(); ++i) outputs[i].storage_index = storage_indices[i];

  std::unordered_map<const Expr*, ExprPtr> stored_values;

  if (return_stmt) {
    std::vector<ExprPtr> new_return_exprs;
    const auto& ret_span = return_stmt->span_;
    for (const auto& binding : outputs) {
      auto ret_expr = binding.value;
      if (auto tile_type = As<TileType>(ret_expr->GetType())) {
        const Expr* identity =
            binding.storage_index ? storage_roots[*binding.storage_index].get() : ret_expr.get();
        if (auto stored = stored_values.find(identity); stored != stored_values.end()) {
          new_return_types.push_back(stored->second->GetType());
          new_return_exprs.push_back(stored->second);
          continue;
        }
        auto orig_tensor_type = As<TensorType>(binding.original_type);
        INTERNAL_CHECK_SPAN(orig_tensor_type, func->span_)
            << "Internal error: return type " << binding.return_index << " should be TensorType but got "
            << binding.original_type->TypeName();
        auto out_param = add_output(orig_tensor_type);
        ExprPtr stored;
        if (auto rewrite = RewriteReturnedAssembleLoopToStore(new_stmts, ret_expr, out_param,
                                                              orig_tensor_type, op_registry)) {
          new_stmts[rewrite->stmt_index] = rewrite->new_for_stmt;
          if (rewrite->dead_init_stmt_index.has_value()) {
            new_stmts.erase(new_stmts.begin() + static_cast<std::ptrdiff_t>(*rewrite->dead_init_stmt_index));
          }
          stored = rewrite->new_return_var;
        } else {
          auto offsets = MakeZeroOffsets(tile_type->shape_.size(), ret_span);
          auto store = op_registry.Create("tile.store", {ret_expr, offsets, out_param}, ret_span);
          auto result = std::make_shared<Var>(MakeStoreResultName(added_outputs.size() - 1), store->GetType(),
                                              ret_span);
          new_stmts.push_back(std::make_shared<AssignStmt>(result, store, ret_span));
          stored = result;
        }
        stored_values.emplace(identity, stored);
        new_return_types.push_back(stored->GetType());
        new_return_exprs.push_back(stored);
        continue;
      }

      new_return_types.push_back(ret_expr->GetType());
      new_return_exprs.push_back(ret_expr);
    }
    auto new_return = MutableCopy(return_stmt);
    new_return->value_ = std::move(new_return_exprs);
    new_stmts.push_back(std::move(new_return));
  } else {
    INTERNAL_CHECK_SPAN(func->return_types_.empty(), func->span_)
        << "Internal error: function '" << func->name_ << "' has no ReturnStmt but declares "
        << func->return_types_.size() << " return type(s) — possible malformed IR";
    new_stmts.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, span));
  }
  std::rotate(new_params.begin() + static_cast<std::ptrdiff_t>(output_start),
              new_params.begin() + static_cast<std::ptrdiff_t>(func->params_.size()), new_params.end());
  std::rotate(new_param_directions.begin() + static_cast<std::ptrdiff_t>(output_start),
              new_param_directions.begin() + static_cast<std::ptrdiff_t>(func->params_.size()),
              new_param_directions.end());
  UpgradeWrittenTensorParamDirections(new_stmts, new_params, new_param_directions);

  auto new_body = SeqStmts::Flatten(std::move(new_stmts), span);
  auto new_func = std::make_shared<Function>(func->name_, new_params, new_param_directions, new_return_types,
                                             new_body, span, func->func_type_, func->level_, func->role_,
                                             EraseCachePolicyAttr(func->attrs_),
                                             func->requires_runtime_binding_, func->ir_stage_);

  return {new_func, std::move(added_outputs)};
}

}  // namespace tensor_to_tile
}  // namespace ir
}  // namespace pypto
