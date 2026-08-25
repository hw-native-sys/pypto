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
 * @file memory.cpp
 * @brief Memory tile operations (get_block_idx, load, store)
 *
 * This file implements memory operations for tile-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// Helper to get kwargs value with default (uses vector to preserve order)
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::optional<T>& default_value = std::nullopt) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  if (default_value) {
    return *default_value;
  }
  throw ValueError("Missing kwarg: " + key);
}

TypePtr DeduceTileGetBlockIdxType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_idx returns INDEX scalar (maps to index type in PTO codegen,
  // consistent with offset arithmetic used in tile.load/tile.store)
  return std::make_shared<ScalarType>(DataType::INDEX);
}

TypePtr DeduceTileGetBlockNumType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_num returns INDEX scalar (same type as get_block_idx)
  return std::make_shared<ScalarType>(DataType::INDEX);
}

TypePtr DeduceTileGetSubblockIdxType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_subblock_idx returns INDEX scalar (maps to index type in PTO codegen)
  return std::make_shared<ScalarType>(DataType::INDEX);
}

TypePtr DeduceTileLoadType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  // load signature: (tensor, offsets_tuple, shapes_tuple, valid_shape_tuple)
  CHECK(args.size() == 4) << "The operator " << op_name
                          << " requires 4 arguments (tensor, offsets, shapes, valid_shape), but got "
                          << args.size();

  // First argument must be a tensor-shaped source. AsTensorTypeLike accepts
  // both plain TensorType and DistributedTensorType — the latter lets a kernel
  // locally load its own window slice (e.g. read back a signal cell after a
  // pld.system.wait barrier), mirroring tile.store's DistributedTensor dst.
  auto tensor_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name
                     << " requires first argument to be a TensorType or DistributedTensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (offsets)
  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "The operator " << op_name
                       << " requires second argument to be a tuple (offsets), but got "
                       << args[1]->GetType()->TypeName();

  // Third argument must be TupleType (shapes)
  auto shapes_tuple = As<MakeTuple>(args[2]);
  CHECK(shapes_tuple) << "The operator " << op_name
                      << " requires third argument to be a tuple (shapes), but got "
                      << args[2]->GetType()->TypeName();

  // Fourth argument must be TupleType (valid_shape)
  auto valid_shape_tuple = As<MakeTuple>(args[3]);
  CHECK(valid_shape_tuple) << "The operator " << op_name
                           << " requires fourth argument to be a tuple (valid shape), but got "
                           << args[3]->GetType()->TypeName();

  // Verify offsets, shapes and valid_shape have same number of dimensions
  CHECK(offsets_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires offsets and shapes to have same number of dimensions, but got "
      << offsets_tuple->elements_.size() << " offsets and " << shapes_tuple->elements_.size() << " shapes";
  CHECK(valid_shape_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires valid_shape and shapes to have the same number of dimensions, but got "
      << valid_shape_tuple->elements_.size() << " valid_shape and " << shapes_tuple->elements_.size()
      << " shapes";
  CHECK(shapes_tuple->elements_.size() > 0)
      << "The operator " << op_name << " requires at least one dimension, but got empty shapes tuple";

  // target_memory is optional: when absent, memory_space stays unresolved and
  // InferTileMemorySpace will pick it from consumer demand. Layout is deferred in
  // that case — the pass recomputes TileView via GetImplicitTileView once the
  // space is known.
  std::optional<MemorySpace> target_memory_opt;
  for (const auto& [k, v] : kwargs) {
    if (k == "target_memory") {
      target_memory_opt = AnyCast<MemorySpace>(v, "target_memory");
      break;
    }
  }
  const bool is_mx_load =
      tensor_type->tensor_view_.has_value() && IsMxTensorLayout(tensor_type->tensor_view_->layout);
  if (is_mx_load) {
    CHECK(tensor_type->dtype_ == DataType::FP8E8M0 || tensor_type->dtype_ == DataType::UINT8)
        << "The operator " << op_name << " of an MX-layout tensor requires FP8E8M0 or UINT8 dtype, but got "
        << tensor_type->dtype_.ToString();
    CHECK(tensor_type->shape_.size() == 2)
        << "The operator " << op_name << " of an MX-layout tensor requires a 2D tensor, got rank "
        << tensor_type->shape_.size();
    CHECK(shapes_tuple->elements_.size() == 2)
        << "The operator " << op_name << " of an MX-layout tensor requires a 2D load window, got rank "
        << shapes_tuple->elements_.size();
    CHECK(valid_shape_tuple->elements_.size() == 2)
        << "The operator " << op_name << " of an MX-layout tensor requires 2D valid_shape, got rank "
        << valid_shape_tuple->elements_.size();
    const TensorView& source_view = *tensor_type->tensor_view_;
    const auto packed_stride =
        tensor_view_semantics::BuildLogicalStridesFromLayout(tensor_type->shape_, source_view.layout);
    CHECK(source_view.stride.empty() ||
          tile_view_semantics::ShapeExprListsEquivalent(source_view.stride, packed_stride))
        << "The operator " << op_name
        << " of an MX-layout tensor only supports packed 2D sources; strided sources are not supported";
    // MX cube scale loads are Mat-only (TLoadMxCube*) and require the caller to
    // spell the target explicitly. The public load interface keeps its ordinary
    // Vec default, so an omitted target fails instead of being silently changed.
    CHECK(target_memory_opt.has_value() && *target_memory_opt == MemorySpace::Mat)
        << "The operator " << op_name << " of an MX-layout tensor requires target_memory=MemorySpace.Mat";
  }
  // Nz/Zn layout: only chosen when target_memory is known. If it is absent,
  // the default-constructed view is kept and InferTileMemorySpace rebuilds it
  // once the memory space is resolved.
  //
  // Source-DN equivalence (RFC #1300 §3.3 + P6): a DN-tagged source tensor
  // describes the same physical bytes as the canonical-pair ND view, so
  // ``tile.load`` of a DN source produces the transposed (ZN) Mat layout.
  // A transposed matmul operand is realised by a zero-copy ``tile.transpose_view``
  // at the matmul site, not by the load.
  bool source_is_dn = tensor_type->tensor_view_.value_or(TensorView{}).layout == TensorLayout::DN;
  TileView tile_view;
  if (is_mx_load) {
    // A5 TLoadMxCubeCheck: MX_A_* → row-major ZZ (SFractal=32); MX_B_* → col-major NN.
    const bool is_mx_b = tensor_type->tensor_view_->layout == TensorLayout::MX_B_NN;
    if (is_mx_b) {
      tile_view.blayout = TileLayout::col_major;
      tile_view.slayout = TileLayout::col_major;
    } else {
      tile_view.blayout = TileLayout::row_major;
      tile_view.slayout = TileLayout::row_major;
    }
    tile_view.fractal = tile_view_semantics::kMXScaleFractal;
  } else if (target_memory_opt.has_value() && *target_memory_opt == MemorySpace::Mat) {
    tile_view.blayout = TileLayout::col_major;
    tile_view.slayout = TileLayout::row_major;
    if (source_is_dn) {
      std::swap(tile_view.blayout, tile_view.slayout);
    }
    // A single-row 2-D Mat operand (cube GEMV lhs / bias) is an ND row
    // vector, not the NZ fractal used by multi-row matmul operands. PTO-ISA
    // declares it as Tile<Mat, 1, K, BLayout::RowMajor, ...,
    // SLayout::NoneBox>; that pair routes the Mat->Left move through the
    // rows==1 vector path instead of the regular extraction path, whose row
    // alignment excludes M=1. In a rank-3+ Mat load, shape[0] is a batch
    // dimension, so keep the canonical NZ view.
    const auto& shape = shapes_tuple->elements_;
    if (shape.size() == 2) {
      const ExprPtr& row_dim = source_is_dn ? shape[1] : shape[0];
      if (auto rows = As<ConstInt>(row_dim); rows && rows->value_ == 1) {
        tile_view.blayout = TileLayout::row_major;
        tile_view.slayout = TileLayout::none_box;
      }
    }
    // Column vector: independent of the destination, and in particular still
    // true when `target_memory` is absent. This arm used to sit inside the
    // `has_value()` branch, so an unset load of an [N, 1] tile kept the default
    // row_major -- an explicit claim contradicting InferImplicitTileLayoutFromShape,
    // which makes it col_major. Because the two disagreed the view could not
    // canonicalize away, and a downstream row_expand_add read the wrong layout.
  } else if (auto last_dim = As<ConstInt>(shapes_tuple->elements_.back());
             last_dim && last_dim->value_ == 1) {
    tile_view.blayout = TileLayout::col_major;
  }

  // Build tile shape from shapes tuple (always in source-tensor coordinates).
  std::vector<ExprPtr> tile_shape(shapes_tuple->elements_.begin(), shapes_tuple->elements_.end());

  // A load copies the source into a fresh tile, so only the valid extent is read:
  // the destination tile may deliberately overhang the source (that is what makes
  // a ragged tail expressible), but the bytes actually read must exist and must
  // be real data. Intersecting with the source valid region enforces both, and
  // rejects a valid_shape request that provably reads past the source. clamp=True
  // narrows such a request to the source edge instead of rejecting it.
  //
  // As with tensor.slice, the rule needs the window to be a rectangle in source
  // coordinates. A lower-rank window (e.g. a 2D tile out of a 3D tensor) is a
  // reinterpreting read whose dim correspondence is not this rectangle, so it
  // keeps the valid_shape it was given.
  if (tile_shape.size() == tensor_type->shape_.size()) {
    tile_view.valid_shape = InferWindowReadValidShape({
        /*source_physical=*/tensor_type->shape_,
        /*source_valid=*/GetEffectiveTensorValidShape(*tensor_type),
        /*offsets=*/offsets_tuple->elements_,
        /*window=*/tile_shape,
        /*requested_valid=*/valid_shape_tuple->elements_,
        /*kind=*/WindowReadKind::kClampedWindow,
        /*clamp=*/GetKwargOr<bool>(kwargs, "clamp", false),
        /*op_name=*/op_name,
        /*bounds_remedy=*/
        "Pass clamp=True -- pl.load(x, offsets, shapes, clamp=True) -- to narrow the read to the "
        "source edge instead",
        /*span=*/args[0]->span_,
    });
  } else {
    tile_view.valid_shape = valid_shape_tuple->elements_;
  }

  // Optional GM cache-access policy. Absent = the caller stated none; an
  // explicit kDefault is distinct from absence and out-ranks a scope-level
  // declaration downstream. Range-checked here, at the op boundary, for the
  // same reason `atomic` is: the DSL types it as CachePolicy, but the text
  // parser and hand-built or deserialized IR can hand over any int, and an
  // unknown one would otherwise surface at codegen with no context.
  const int cache = GetKwarg<int>(kwargs, "cache", static_cast<int>(CachePolicy::kDefault));
  CHECK(cache == static_cast<int>(CachePolicy::kDefault) || cache == static_cast<int>(CachePolicy::kBypass))
      << "The operator " << op_name
      << " cache kwarg must be CachePolicy.DEFAULT or CachePolicy.BYPASS, but got int " << cache;

  // Return TileType with same dtype as tensor and TileView containing valid_shape.
  // When target_memory is specified, write it into memory_space_ so the constructed
  // type is internally coherent (tile_view layout and memory_space agree). This
  // lets CanonicalizeTileViewInPlace collapse the explicit Mat-style view to
  // nullopt against the matching implicit, giving a unique canonical encoding
  // that round-trips through print/parse.
  return std::make_shared<TileType>(tile_shape, tensor_type->dtype_, std::nullopt, tile_view,
                                    target_memory_opt);
}

TypePtr DeduceTileStoreType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // store signature: (tile, offsets_tuple, output_tensor[, shapes_tuple])
  // shapes_tuple is an optional 4th argument injected by FlattenTileNdTo2D
  // for ND tensors to carry the ND partition shape for codegen.
  // When present, shapes_tuple has the same rank as offsets_tuple (both ND).
  CHECK(args.size() == 3 || args.size() == 4)
      << "The operator " << op_name
      << " requires 3 or 4 arguments (tile, offsets, output_tensor[, shapes]), but got " << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (offsets)
  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "The operator " << op_name
                       << " requires second argument to be a tuple (offsets), but got "
                       << args[1]->GetType()->TypeName();

  // Third argument must be the output tensor. AsTensorTypeLike accepts both
  // plain TensorType and DistributedTensorType — the latter is needed for the
  // local operation on DistributedTensor.
  auto output_tensor_type = AsTensorTypeLike(args[2]->GetType());
  CHECK(output_tensor_type)
      << "The operator " << op_name
      << " requires third argument to be a TensorType or DistributedTensorType, but got "
      << args[2]->GetType()->TypeName();

  // Optional fourth argument (when 4 args total) must be a shapes tuple
  MakeTuplePtr shapes_tuple;
  if (args.size() == 4) {
    shapes_tuple = As<MakeTuple>(args[3]);
    CHECK(shapes_tuple) << "The operator " << op_name
                        << " requires optional 4th argument to be a shapes tuple (MakeTuple)";
    CHECK(!shapes_tuple->elements_.empty())
        << "The operator " << op_name << " requires non-empty shapes tuple when provided";
    CHECK(shapes_tuple->elements_.size() == offsets_tuple->elements_.size())
        << "The operator " << op_name
        << " requires shapes and offsets to have the same number of dimensions, but got "
        << shapes_tuple->elements_.size() << " shapes and " << offsets_tuple->elements_.size() << " offsets";
  }

  // Optional atomic-add combine mode (split-K accumulation into GM). Absent =
  // AtomicType::kNone (plain overwrite store).
  int atomic = GetKwarg<int>(kwargs, "atomic", 0);
  CHECK(atomic == static_cast<int>(AtomicType::kNone) || atomic == static_cast<int>(AtomicType::kAdd))
      << "The operator " << op_name
      << " atomic kwarg must be AtomicType.None_ or AtomicType.Add, but got int " << atomic;
  if (atomic == static_cast<int>(AtomicType::kAdd)) {
    const DataType& dt = tile_type->dtype_;
    // Hardware atomic-add dtypes. bf16 is honoured on the A2/A3 (Ascend910B) and
    // kirinX90 profiles (pto-isa SetAtomicAdd<bfloat16_t> -> set_atomic_bf16);
    // it is NOT supported on the A5/kirin9030 store path, where a bf16 atomic
    // store is rejected downstream by the pto-isa static_assert.
    CHECK(dt == DataType::FP32 || dt == DataType::BF16 || dt == DataType::FP16 || dt == DataType::INT32 ||
          dt == DataType::INT16 || dt == DataType::INT8)
        << "The operator " << op_name
        << " with atomic=AtomicType.Add requires an fp32/bf16/fp16/int32/int16/int8 tile (hardware "
           "atomic-add "
           "dtypes), but got "
        << dt.ToString();
  }

  // ---- Valid-region union -------------------------------------------------
  // A store writes into the destination tensor, so the tensor it returns holds
  // what that tensor already held plus the region just written.
  const std::vector<ExprPtr>& dest_shape = output_tensor_type->shape_;
  const size_t dest_rank = dest_shape.size();

  // The optional ``shapes`` operand carries FlattenTileNdTo2D's ND partition,
  // which is a *collapsed-dims* descriptor rather than a rectangle in
  // destination coordinates: it is built as leading 1s followed by the
  // pre-flatten tile shape, whose leading extent may be the product of several
  // destination axes. A [2, 3, 8] gather, for one, stores its collapsed [6, 8]
  // tile as partition [1, 6, 8], where 6 spans two axes of a destination whose
  // own axis 1 is only 3. Codegen consumes that through pto.partition_view,
  // which understands the collapse; reading it as an origin-anchored rectangle
  // here would both mis-bound the write and place the union on the wrong axes.
  // So the ND form keeps the destination type it had — recovering the written
  // region on ND axes is the ND-to-2D mapping problem, not this rule's.
  if (shapes_tuple) {
    return output_tensor_type;
  }
  std::vector<ExprPtr> transfer_physical = tile_type->shape_;
  std::vector<ExprPtr> transfer_valid = GetValidShape(tile_type);

  // As for assemble, the union is derivable only when the transfer, the offsets,
  // and the destination share one rank. A store whose tile addresses the
  // destination through a reinterpreting view — a lower-rank window, or a DN
  // layout that permutes the axes — is not a rectangle on these axes, so it keeps
  // the destination type it returned before this rule existed.
  if (transfer_physical.size() != dest_rank || offsets_tuple->elements_.size() != dest_rank) {
    return output_tensor_type;
  }

  // Only the tile's valid extent is moved by the DMA, so a tile whose physical
  // allocation is larger than its real contents is bounded by what it actually
  // transfers rather than by its allocation.
  std::vector<ExprPtr> dest_valid = GetValidShape(output_tensor_type);
  std::vector<ExprPtr> result_valid = InferWriteValidShapeUnion({
      /*target_physical=*/dest_shape,
      /*target_valid=*/dest_valid,
      /*source_physical=*/std::move(transfer_physical),
      /*source_valid=*/std::move(transfer_valid),
      /*offsets=*/offsets_tuple->elements_,
      /*kind=*/WriteBoundsKind::kValidRegionTransfer,
      /*op_name=*/op_name,
      /*bounds_remedy=*/
      "A store performs no layout conversion (RFC #1300 P7): the offsets and the tile extent must "
      "already be in the destination's coordinate system. If the tile was read through a view whose "
      "layout differs from the destination's -- a DN view transposes it -- take the matching view of "
      "the destination, pl.view(out, layout=...), and store into that",
      /*span=*/args[2]->span_,
  });

  // A store that leaves the destination's valid region exactly as it found it —
  // which every store into a fully valid destination does — returns the
  // destination type itself, so the common case keeps the very type, and the
  // memref and comm-group binding on it, that it carried before this rule existed.
  if (AreExprVectorsEqual(result_valid, dest_valid)) {
    return output_tensor_type;
  }

  TensorView result_view = output_tensor_type->tensor_view_.value_or(TensorView{});
  result_view.valid_shape = std::move(result_valid);
  if (auto dt = As<DistributedTensorType>(args[2]->GetType())) {
    return std::make_shared<DistributedTensorType>(dest_shape, output_tensor_type->dtype_,
                                                   output_tensor_type->memref_,
                                                   std::make_optional(result_view), dt->window_buffer_);
  }
  return std::make_shared<TensorType>(dest_shape, output_tensor_type->dtype_, output_tensor_type->memref_,
                                      std::make_optional(result_view));
}

TypePtr DeduceTileMoveType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  // Validate args: expect exactly 1 argument (tile)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // Validate first argument is TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Extract MemorySpace
  MemorySpace space = GetKwarg<MemorySpace>(kwargs, "target_memory");

  const auto& input_shape = tile_type->shape_;

  const TileView source_view = tile_view_semantics::GetEffectiveTileView(*tile_type);

  // A move retargets the tile, so where the destination has a layout of its own
  // (Mat/Acc/L0/scale are boxed) that layout -- not the source's -- is what the
  // result carries.  Seeding blayout/slayout from the source instead recovered
  // the destination layout only as a side effect of canonicalizing against
  // nullopt here and re-canonicalizing against the real space in
  // OpRegistry::Create, which only fires when the view collapses (valid_shape ==
  // shape and pad null) -- so the result layout silently depended on the tile's
  // valid extent.
  //
  // Where the destination's layout coincides with the space-agnostic one (Vec
  // and the other flat spaces) that double canonicalization was a no-op, so
  // those keep the source's blayout/slayout: a Mat->Vec move deliberately
  // carries the source's layout today.  fractal is never inherited -- it is the
  // destination buffer's boxing granularity.  See
  // docs/en/dev/ir/05-operators.md "Result view of tile.move".
  const auto dst_layout = tile_view_semantics::GetImplicitTileLayout(input_shape, space);
  const bool destination_dictates_layout =
      dst_layout != tile_view_semantics::GetImplicitTileLayout(input_shape, std::nullopt);

  TileView tile_view;
  tile_view_semantics::SetTileLayout(tile_view, dst_layout);
  if (!destination_dictates_layout) {
    tile_view.blayout = source_view.blayout;
    tile_view.slayout = source_view.slayout;
  }

  // Right is the one destination whose ISA layout is not its implicit one: L0B
  // requires a RowMajor block layout for TMATMUL even on an [N,1] shape, whose
  // implicit blayout is col_major.  Left and the scale spaces need no override --
  // GetImplicitTileLayout already returns their ISA layouts.
  if (space == MemorySpace::Right) {
    tile_view.blayout = TileLayout::row_major;
  }

  // Ordinary destinations permit explicit layouts. Scale destinations have
  // hardware-fixed layouts because PTOAS uses them to distinguish ScaleLeft
  // from ScaleRight even though both lower to loc=scaling.
  const TileLayout requested_blayout = GetKwarg<TileLayout>(kwargs, "blayout", tile_view.blayout);
  const TileLayout requested_slayout = GetKwarg<TileLayout>(kwargs, "slayout", tile_view.slayout);
  if (space == MemorySpace::LeftScale || space == MemorySpace::RightScale) {
    CHECK(requested_blayout == tile_view.blayout && requested_slayout == tile_view.slayout)
        << "The operator " << op_name
        << " does not allow blayout/slayout to override the hardware-fixed layout for "
        << MemorySpaceToString(space);
  }
  tile_view.blayout = requested_blayout;
  tile_view.slayout = requested_slayout;

  // Keep original shape
  std::vector<ExprPtr> output_shape = input_shape;

  // Preserve input valid_shape (may be narrower than shape_)
  tile_view.valid_shape = source_view.valid_shape.empty() ? input_shape : source_view.valid_shape;

  // Preserve pad value from input tile
  if (source_view.pad != PadValue::null) {
    tile_view.pad = source_view.pad;
  }

  // MX LeftScale/RightScale must be !pto.f8E8M0 so EmitC maps loc=scaling → ScaleLeft
  // (ui8+scaling wrongly becomes Fixpipe TileType::Scaling). Host-prequant may load UINT8.
  DataType out_dtype = tile_type->dtype_;
  if (space == MemorySpace::LeftScale || space == MemorySpace::RightScale) {
    CHECK(input_shape.size() == 2) << "The operator " << op_name
                                   << " into LeftScale/RightScale requires a 2D tile, got rank "
                                   << input_shape.size();
    CHECK(tile_type->memory_space_ == MemorySpace::Mat)
        << "The operator " << op_name
        << " into LeftScale/RightScale requires the input tile to be in Mat memory";
    CHECK(out_dtype == DataType::UINT8 || out_dtype == DataType::FP8E8M0)
        << "The operator " << op_name
        << " into LeftScale/RightScale requires UINT8 or FP8E8M0 dtype, but got " << out_dtype.ToString();
    const TileLayout required_layout =
        space == MemorySpace::LeftScale ? TileLayout::row_major : TileLayout::col_major;
    CHECK(source_view.blayout == required_layout && source_view.slayout == required_layout &&
          source_view.fractal == tile_view_semantics::kMXScaleFractal)
        << "The operator " << op_name << " into " << MemorySpaceToString(space)
        << " requires the source Mat tile to use the matching "
        << (space == MemorySpace::LeftScale ? "row/row/32" : "col/col/32") << " layout";
    if (out_dtype == DataType::UINT8) {
      out_dtype = DataType::FP8E8M0;
    }
  }

  // Stamp memory_space_ for every destination so the view is canonicalized once,
  // against the space it is actually a view of (same contract as tile.load).
  return std::make_shared<TileType>(output_shape, out_dtype, std::nullopt, tile_view, space);
}

TypePtr DeduceTileAllocType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // alloc signature: (memory_space, size) — returns PtrType (allocation identity)
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  return GetPtrType();
}

TypePtr DeduceTileCreateTileType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  // create_tile signature: (shape)
  // TileType requires static compile-time constant shapes
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  // Extract dtype attribute
  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");

  // First argument must be MakeTuple with static ConstInt elements
  auto make_tuple = As<MakeTuple>(args[0]);
  CHECK(make_tuple)
      << "The operator " << op_name
      << " requires first argument to be a MakeTuple expression with static shape values, but got "
      << args[0]->TypeName();

  // Validate all elements are ConstInt (static compile-time constants)
  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(make_tuple->elements_.size());

  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(const_int) << "The operator " << op_name << " shape element " << i
                     << " must be a compile-time constant (ConstInt), but got "
                     << make_tuple->elements_[i]->TypeName();
    CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                 << " must be positive, got " << const_int->value_;
    tile_shape.push_back(make_tuple->elements_[i]);
  }

  CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";

  // When target_memory is Acc, deduce the Nz TileView so the result type
  // matches what tile.matmul / tile.matmul_acc produce.  This keeps Acc-typed
  // iter-arg / yield chains structurally consistent in passes such as
  // AutoTileMatmulL0.
  std::optional<MemorySpace> target_memory_opt;
  for (const auto& [k, v] : kwargs) {
    if (k == "target_memory") {
      target_memory_opt = AnyCast<MemorySpace>(v, "target_memory");
      break;
    }
  }
  CHECK(!target_memory_opt.has_value() ||
        (*target_memory_opt != MemorySpace::LeftScale && *target_memory_opt != MemorySpace::RightScale))
      << "The operator " << op_name
      << " does not support target_memory=LeftScale/RightScale; create the scale tile with tile.load "
         "to Mat followed by tile.move";

  TileView tile_view;
  // `transpose=true` requests the transposed Mat (ZN) fractal layout
  // (blayout=row_major, slayout=col_major) — the layout a matmul B-operand
  // carries when loaded with b_trans, and the only Mat layout a DN-source
  // gather_row (DN2ZN tload) can fill. Default false keeps the canonical NZ.
  bool transpose_layout = false;
  // `flat_layout=true` requests a flat (non-fractal, slayout=none_box) L1/cbuf
  // tile: a contiguous byte-staging buffer rather than the boxed NZ layout Mat
  // tiles normally carry.
  bool flat_layout = false;
  // `compact=true` DECLARES that the fresh L0C buffer holds a valid-region-packed
  // product: `mad` lays its result out with an N-fractal stride of
  // `ceil(validRow/16)*16` taken from the L0A operand's valid rows (pto-isa
  // `TMatmul.hpp`), so an accumulator seeded here for a row-narrowed matmul is
  // written at that pitch and every reader must recompute it the same way.
  // Declaring it at creation rather than stamping it later is what makes the mode
  // survive: a pass-applied type refinement is discarded the moment any pass
  // re-deduces the call (InferTileMemorySpace does), whereas a kwarg is re-read.
  // `tile.set_validshape` then inherits the mode onto the narrowed seed without
  // re-interpreting bytes it did not write (the inherit-only contract of #2474).
  bool compact_layout = false;
  for (const auto& [k, v] : kwargs) {
    if (k == "transpose") transpose_layout = AnyCast<bool>(v, "transpose");
    if (k == "flat_layout") flat_layout = AnyCast<bool>(v, "flat_layout");
    if (k == "compact") compact_layout = AnyCast<bool>(v, "compact");
  }
  // The transposed Mat (ZN) layout is a 2D L1 matmul-`b_trans` operand layout; it
  // is meaningless for a non-Mat space or a non-2D shape. Fail fast rather than
  // emit an invalid tile (mirrors tile.load's Mat-only transpose guard).
  CHECK(!transpose_layout || (tile_shape.size() == 2 && target_memory_opt == MemorySpace::Mat))
      << "The operator " << op_name
      << " supports transpose=true only for a 2D tile with target_memory=Mat (L1)";
  // flat_layout is a Mat (L1/cbuf) staging layout and mutually exclusive with the
  // transposed NZ layout.
  CHECK(!flat_layout || (target_memory_opt == MemorySpace::Mat && !transpose_layout))
      << "The operator " << op_name
      << " supports flat_layout=true only for target_memory=Mat (L1) without transpose";
  // Compact is a fractal-pitch property of an accumulator. Left/Right get theirs
  // from the partial `tile.extract` that fills them, so `tile.create` only ever
  // needs to declare it for L0C.
  CHECK(!compact_layout || target_memory_opt == MemorySpace::Acc)
      << "The operator " << op_name
      << " supports compact=true only for target_memory=Acc (L0C), which is the only space whose "
         "fractal pitch a matmul derives from the valid row count";

  // A flat L1 tile keeps the canonical flat view (blayout=row_major,
  // slayout=none_box, fractal default) — it is deliberately NOT boxed. We also
  // stamp memory_space_=Mat at creation so InferTileMemorySpace sees the space
  // is already resolved and preserves the none_box view instead of overwriting
  // it with Mat's implicit boxed layout (see ComputeRewrittenType).
  std::optional<MemorySpace> creation_space = std::nullopt;
  if (flat_layout) {
    creation_space = MemorySpace::Mat;
  } else if (target_memory_opt.has_value() && *target_memory_opt == MemorySpace::Acc) {
    // Acc's boxed NZ layout, stamped so the view is canonicalized against the
    // space it is a view of rather than against nullopt (see 02-types.md).
    tile_view_semantics::SetTileLayout(
        tile_view, tile_view_semantics::GetImplicitTileLayout(tile_shape, MemorySpace::Acc));
    if (compact_layout) {
      tile_view.compact = CompactMode::normal;
    }
    creation_space = MemorySpace::Acc;
  } else if (transpose_layout) {
    tile_view.blayout = TileLayout::row_major;
    tile_view.slayout = TileLayout::col_major;
  } else {
    tile_view.blayout = tile_view_semantics::InferImplicitTileLayoutFromShape(tile_shape);
  }
  tile_view.valid_shape = tile_shape;
  return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view, creation_space);
}

TypePtr DeduceTileFullType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  // tile.full signature: (shape, value)
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Extract dtype attribute
  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");

  // First argument must be MakeTuple with static ConstInt elements
  auto make_tuple = As<MakeTuple>(args[0]);
  CHECK(make_tuple)
      << "The operator " << op_name
      << " requires first argument to be a MakeTuple expression with static shape values, but got "
      << args[0]->TypeName();

  // Validate all elements are ConstInt (static compile-time constants)
  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(make_tuple->elements_.size());

  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(const_int) << "The operator " << op_name << " shape element " << i
                     << " must be a compile-time constant (ConstInt), but got "
                     << make_tuple->elements_[i]->TypeName();
    CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                 << " must be positive, got " << const_int->value_;
    tile_shape.push_back(make_tuple->elements_[i]);
  }

  CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";

  // Second argument must be ConstInt or ConstFloat
  CHECK(As<ConstInt>(args[1]) || As<ConstFloat>(args[1]))
      << "The operator " << op_name
      << " requires second argument to be a constant value (ConstInt or ConstFloat), but got "
      << args[1]->TypeName();

  // Return TileType with the static shape and dtype
  TileView tile_view;
  tile_view.valid_shape = tile_shape;
  return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileCiType(const std::vector<ExprPtr>& args,
                         const std::vector<std::pair<std::string, std::any>>& kwargs,
                         const std::string& op_name) {
  // tile.ci signature: (start, shape[, tmp]) with attrs {dtype, descending}.
  // A2/A3 requires the optional scratch operand when PTOAS PlanMemory is
  // skipped; InitMemRef materializes the canonical workspace when absent.
  CHECK(args.size() == 2 || args.size() == 3)
      << "The operator " << op_name << " requires 2 or 3 arguments (start, shape[, tmp]), but got "
      << args.size();

  // Extract dtype and validate it is one of the supported integer types.
  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");
  CHECK(dtype == DataType::INT16 || dtype == DataType::INT32 || dtype == DataType::UINT16 ||
        dtype == DataType::UINT32)
      << "The operator " << op_name << " requires dtype to be one of {INT16, INT32, UINT16, UINT32}, but got "
      << dtype.ToString();

  // First argument is the scalar start value; its dtype must match the destination dtype.
  auto start_scalar_type = As<ScalarType>(args[0]->GetType());
  CHECK(start_scalar_type) << "The operator " << op_name
                           << " requires first argument 'start' to be a scalar, but got "
                           << args[0]->GetType()->TypeName();
  CHECK(start_scalar_type->dtype_ == dtype)
      << "The operator " << op_name << " requires 'start' dtype (" << start_scalar_type->dtype_.ToString()
      << ") to match destination dtype (" << dtype.ToString() << ")";

  // Second argument must be a MakeTuple of static ConstInt elements.
  auto make_tuple = As<MakeTuple>(args[1]);
  CHECK(make_tuple)
      << "The operator " << op_name
      << " requires second argument 'shape' to be a MakeTuple of compile-time constants, but got "
      << args[1]->TypeName();

  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(make_tuple->elements_.size());
  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(const_int) << "The operator " << op_name << " shape element " << i
                     << " must be a compile-time constant (ConstInt), but got "
                     << make_tuple->elements_[i]->TypeName();
    CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                 << " must be positive, got " << const_int->value_;
    tile_shape.push_back(make_tuple->elements_[i]);
  }
  CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";

  // ISA constraint: destination Cols != 1 (column vectors not supported by pto.tci).
  auto last_dim = As<ConstInt>(tile_shape.back());
  CHECK(last_dim && last_dim->value_ != 1)
      << "The operator " << op_name << " requires the innermost dimension (Cols) to be != 1, got "
      << (last_dim ? last_dim->value_ : -1);

  // ISA constraint: pto.tci only populates the first row and ignores valid rows, so every
  // leading dimension must be 1. Reject multi-row shapes here to keep type metadata truthful.
  for (size_t i = 0; i + 1 < tile_shape.size(); ++i) {
    auto leading_dim = As<ConstInt>(tile_shape[i]);
    CHECK(leading_dim && leading_dim->value_ == 1)
        << "The operator " << op_name << " only populates the first row because pto.tci ignores valid rows; "
        << "leading dimensions must be 1, but got " << (leading_dim ? leading_dim->value_ : -1)
        << " at index " << i;
  }

  // descending kwarg is optional and defaults to false.
  (void)GetKwarg<bool>(kwargs, "descending", false);

  if (args.size() == 3) {
    auto tmp_type = As<TileType>(args[2]->GetType());
    CHECK(tmp_type) << "The operator " << op_name
                    << " requires optional third argument 'tmp' to be a TileType, but got "
                    << args[2]->GetType()->TypeName();
    CHECK(tmp_type->dtype_ == DataType::FP32 || tmp_type->dtype_ == DataType::INT32 ||
          tmp_type->dtype_ == DataType::UINT32)
        << "The operator " << op_name << " requires tmp dtype to be FP32, INT32, or UINT32, but got "
        << tmp_type->dtype_.ToString();
  }

  TileView tile_view;
  tile_view.valid_shape = tile_shape;
  return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileTriType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                          const std::string& op_name) {
  CHECK(args.size() == 2 || args.size() == 3)
      << "The operator " << op_name << " requires 2 or 3 arguments (diagonal, shape, [valid_shape]), but got "
      << args.size();

  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");
  CHECK(dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
        dtype == DataType::INT32 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
        dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32)
      << "The operator " << op_name
      << " requires dtype to be one of {INT8, UINT8, INT16, INT32, UINT16, UINT32, FP16, BF16, FP32}, "
         "but got "
      << dtype.ToString();

  auto diagonal_type = As<ScalarType>(args[0]->GetType());
  CHECK(diagonal_type) << "The operator " << op_name << " requires diagonal to be a scalar, but got "
                       << args[0]->GetType()->TypeName();
  CHECK(diagonal_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires diagonal to be an INT32 scalar, but got "
      << diagonal_type->dtype_.ToString();

  auto shape_tuple = As<MakeTuple>(args[1]);
  CHECK(shape_tuple) << "The operator " << op_name
                     << " requires shape to be a MakeTuple of compile-time constants, but got "
                     << args[1]->TypeName();
  CHECK(shape_tuple->elements_.size() == 2)
      << "The operator " << op_name << " requires a 2D shape, but got rank " << shape_tuple->elements_.size();

  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(2);
  for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
    auto dim = As<ConstInt>(shape_tuple->elements_[i]);
    CHECK(dim) << "The operator " << op_name << " shape element " << i << " must be a compile-time constant";
    CHECK(dim->value_ > 0) << "The operator " << op_name << " shape element " << i
                           << " must be positive, got " << dim->value_;
    tile_shape.push_back(shape_tuple->elements_[i]);
  }

  std::vector<ExprPtr> valid_shape = tile_shape;
  if (args.size() == 3) {
    auto valid_tuple = As<MakeTuple>(args[2]);
    CHECK(valid_tuple) << "The operator " << op_name
                       << " requires valid_shape to be a MakeTuple of compile-time constants";
    CHECK(valid_tuple->elements_.size() == tile_shape.size())
        << "The operator " << op_name << " requires valid_shape rank " << tile_shape.size() << ", but got "
        << valid_tuple->elements_.size();
    valid_shape.clear();
    for (size_t i = 0; i < valid_tuple->elements_.size(); ++i) {
      auto valid_dim = As<ConstInt>(valid_tuple->elements_[i]);
      auto physical_dim = As<ConstInt>(tile_shape[i]);
      CHECK(valid_dim) << "The operator " << op_name << " valid_shape element " << i
                       << " must be a compile-time constant";
      CHECK(valid_dim->value_ > 0 && valid_dim->value_ <= physical_dim->value_)
          << "The operator " << op_name << " requires 0 < valid_shape[" << i << "] <= shape[" << i
          << "], but got " << valid_dim->value_ << " and " << physical_dim->value_;
      valid_shape.push_back(valid_tuple->elements_[i]);
    }
  }

  (void)GetKwarg<bool>(kwargs, "upper", false);
  TileView tile_view;
  tile_view.valid_shape = valid_shape;
  return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileRandomType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  // tile.random signature: (key0, key1, counter0, counter1, counter2, counter3, shape,
  // [valid_shape]) with attrs {dtype, rounds}. Generates a tile of counter-based
  // (Philox/ChaCha) pseudo-random values; the 6 scalars seed the generator (key +
  // 128-bit counter) and the shape tuple gives the destination extent. There is no
  // source tile. The optional trailing valid_shape tuple narrows the written region:
  // pto.trandom only fills the dst valid rows/cols, leaving the rest untouched.
  CHECK(args.size() == 7 || args.size() == 8)
      << "The operator " << op_name
      << " requires 7 or 8 arguments (key0, key1, counter0, counter1, counter2, counter3, "
         "shape, [valid_shape]), but got "
      << args.size();

  // Destination dtype: pto.trandom emits 32-bit lanes only (INT32 or UINT32).
  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");
  CHECK(dtype == DataType::INT32 || dtype == DataType::UINT32)
      << "The operator " << op_name << " requires dtype to be one of {INT32, UINT32}, but got "
      << dtype.ToString();

  // rounds attr controls the cipher round count; the hardware only accepts 7 or 10.
  int rounds = GetKwarg<int>(kwargs, "rounds", 10);
  CHECK(rounds == 7 || rounds == 10) << "The operator " << op_name
                                     << " requires rounds to be 7 or 10, but got " << rounds;

  // The 6 seed arguments are 32-bit integer scalars (key[0..1], counter[0..3]).
  for (size_t i = 0; i < 6; ++i) {
    auto scalar_type = As<ScalarType>(args[i]->GetType());
    CHECK(scalar_type) << "The operator " << op_name << " requires argument " << i
                       << " (seed scalar) to be a scalar, but got " << args[i]->GetType()->TypeName();
    CHECK(scalar_type->dtype_ == DataType::INT32)
        << "The operator " << op_name << " requires seed argument " << i << " to have INT32 dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  // Shape must be a literal tuple of positive compile-time constants.
  auto make_tuple = As<MakeTuple>(args[6]);
  CHECK(make_tuple) << "The operator " << op_name
                    << " requires the shape argument to be a MakeTuple of compile-time constants, but got "
                    << args[6]->TypeName();

  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(make_tuple->elements_.size());
  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(const_int) << "The operator " << op_name << " shape element " << i
                     << " must be a compile-time constant (ConstInt), but got "
                     << make_tuple->elements_[i]->TypeName();
    CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                 << " must be positive, got " << const_int->value_;
    tile_shape.push_back(make_tuple->elements_[i]);
  }
  CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";
  // pto.trandom is a 2D row/col generator and FlattenTileNd does not lower it, so
  // reject N-D shapes here rather than emit a tile the codegen cannot handle.
  CHECK(tile_shape.size() == 2) << "The operator " << op_name
                                << " requires a 2D shape (rows, cols), but got rank " << tile_shape.size();

  // Default: the entire destination is populated (valid == full shape). An optional
  // valid_shape tuple narrows the written region (must match rank and 0 < v <= shape).
  std::vector<ExprPtr> valid_shape = tile_shape;
  if (args.size() == 8) {
    auto valid_tuple = As<MakeTuple>(args[7]);
    CHECK(valid_tuple) << "The operator " << op_name
                       << " requires valid_shape to be a MakeTuple of compile-time constants, but got "
                       << args[7]->TypeName();
    CHECK(valid_tuple->elements_.size() == tile_shape.size())
        << "The operator " << op_name << " valid_shape rank (" << valid_tuple->elements_.size()
        << ") must match shape rank (" << tile_shape.size() << ")";
    valid_shape.clear();
    valid_shape.reserve(valid_tuple->elements_.size());
    for (size_t i = 0; i < valid_tuple->elements_.size(); ++i) {
      auto v = As<ConstInt>(valid_tuple->elements_[i]);
      CHECK(v) << "The operator " << op_name << " valid_shape element " << i
               << " must be a compile-time constant (ConstInt)";
      auto dim = As<ConstInt>(tile_shape[i]);
      CHECK(v->value_ > 0 && (!dim || v->value_ <= dim->value_))
          << "The operator " << op_name << " valid_shape element " << i << " (" << v->value_
          << ") must be in (0, shape dim " << (dim ? dim->value_ : -1) << "]";
      valid_shape.push_back(valid_tuple->elements_[i]);
    }
  }

  TileView tile_view;
  tile_view.valid_shape = valid_shape;
  return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileReadType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  // tile.read: Read a scalar value from a tile at given indices
  // Args: (tile, indices_tuple)
  // Returns: ScalarType with tile's element dtype
  CHECK(args.size() == 2) << "tile.read requires exactly 2 arguments (tile, indices), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.read requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (indices)
  auto indices_type = As<TupleType>(args[1]->GetType());
  CHECK(indices_type) << "tile.read requires indices to be TupleType, but got "
                      << args[1]->GetType()->TypeName();

  // Validate indices count matches tile rank
  CHECK(indices_type->types_.size() == tile_type->shape_.size())
      << "tile.read indices count (" << indices_type->types_.size() << ") must match tile rank ("
      << tile_type->shape_.size() << ")";

  // Validate all index elements are ScalarType with integer dtype
  for (size_t i = 0; i < indices_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(indices_type->types_[i]);
    CHECK(scalar_type) << "tile.read index element " << i << " must be ScalarType, but got "
                       << indices_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tile.read index element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  return std::make_shared<ScalarType>(tile_type->dtype_);
}

TypePtr DeduceTileWriteType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // tile.write: Write a scalar value into a tile at given indices
  // Args: (tile, indices_tuple, value)
  // Returns: TileType (the destination tile, for chaining)
  CHECK(args.size() == 3) << "tile.write requires exactly 3 arguments (tile, indices, value), but got "
                          << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.write requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  auto indices_type = As<TupleType>(args[1]->GetType());
  CHECK(indices_type) << "tile.write requires indices to be TupleType, but got "
                      << args[1]->GetType()->TypeName();

  CHECK(indices_type->types_.size() == tile_type->shape_.size())
      << "tile.write indices count (" << indices_type->types_.size() << ") must match tile rank ("
      << tile_type->shape_.size() << ")";

  for (size_t i = 0; i < indices_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(indices_type->types_[i]);
    CHECK(scalar_type) << "tile.write index element " << i << " must be ScalarType, but got "
                       << indices_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tile.write index element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  auto value_type = As<ScalarType>(args[2]->GetType());
  CHECK(value_type) << "tile.write requires third argument (value) to be a ScalarType, but got "
                    << args[2]->GetType()->TypeName();

  CHECK(value_type->dtype_ == tile_type->dtype_)
      << "tile.write requires value dtype to match tile dtype, but got value dtype "
      << value_type->dtype_.ToString() << " and tile dtype " << tile_type->dtype_.ToString();

  return args[0]->GetType();
}

REGISTER_OP("tile.write")
    .set_op_category("TileOp")
    .set_description("Write a scalar value into a tile at given indices")
    .add_argument("tile", "Destination tile (TileType)")
    .add_argument("indices", "Index dimensions (TupleType of ScalarType)")
    .add_argument("value", "Scalar value to write (ScalarType)")
    // Rewrites one element and passes every other element of the tile through
    // to the result, so the prior content is read. No write channel: this is a
    // tile-local write, not one of the GM store paths the mixed-store
    // diagnostic orders against each other.
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileWriteType(args, kwargs, "tile.write");
    });

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("tile.get_block_idx")
    .set_op_category("TileOp")
    .no_execution_memory_access()
    .set_description("Get the current block index")
    .set_core_affinity(core_affinity::CoreAffinity::SHARED)
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGetBlockIdxType(args, kwargs, "tile.get_block_idx");
    });

REGISTER_OP("tile.get_subblock_idx")
    .set_op_category("TileOp")
    .no_execution_memory_access()
    .set_description("Get the current sub-block (vector core) index")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGetSubblockIdxType(args, kwargs, "tile.get_subblock_idx");
    });

REGISTER_OP("tile.get_block_num")
    .set_op_category("TileOp")
    .no_execution_memory_access()
    .set_description("Get the total number of blocks in the current SPMD task")
    .set_core_affinity(core_affinity::CoreAffinity::SHARED)
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGetBlockNumType(args, kwargs, "tile.get_block_num");
    });

REGISTER_OP("tile.read")
    .set_op_category("TileOp")
    .set_description("Read a scalar value from a tile at given indices")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("indices", "Index dimensions (TupleType of ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReadType(args, kwargs, "tile.read");
    });

REGISTER_OP("tile.create")
    .set_op_category("TileOp")
    .set_description("Create a tile")
    .set_core_affinity(core_affinity::CoreAffinity::SHARED)
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .set_attr<DataType>("dtype")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<bool>("transpose")
    .set_attr<bool>("flat_layout")
    .set_attr<bool>("compact")
    .no_execution_memory_access()
    // No fallback: when target_memory is absent, memory_space stays unresolved and
    // InferTileMemorySpace picks the space from consumer demand.
    .set_output_memory_from_kwarg("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCreateTileType(args, kwargs, "tile.create");
    });

REGISTER_OP("tile.load")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Copy data from tensor to unified buffer (tile)")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("offsets",
                  "Offsets in each dimension, in source tensor coordinates (TupleType of ScalarType)")
    .add_argument(
        "shapes",
        "Shape of region to load in each dimension, in source tensor coordinates (TupleType of ScalarType)")
    .add_argument(
        "valid_shape",
        "Valid shape of tile in each dimension, in source tensor coordinates (TupleType of ScalarType). ")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<bool>("clamp")
    // Declared GM cache-access policy, carried as an int (``ir::CachePolicy``)
    // so serialization / structural comparison need no new enum arm.
    .set_attr<int>("cache")
    // No fallback: when target_memory is absent, memory_space stays unresolved and
    // InferTileMemorySpace picks the space from consumer demand.
    .set_output_memory_from_kwarg("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileLoadType(args, kwargs, "tile.load");
    });

REGISTER_OP("tile.store")
    .set_op_category("TileOp")
    .set_description("Copy data from unified buffer (tile) to tensor")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .add_argument("shapes",
                  "Optional ND partition shape (TupleType). "
                  "Injected by FlattenTileNdTo2D for ND tensors.")
    .set_attr<int>("atomic")
    .set_input_memory(0, {MemorySpace::Vec, MemorySpace::Acc})
    .set_output_reuses_input(2)
    // A plain store overwrites the region it lands on: the untouched remainder
    // is neither loaded nor re-stored, so nothing moves *into* the kernel and
    // the destination is a pure write. An atomic store is not an overwrite at
    // all — `out += x` reads the accumulator it adds to.
    .set_arg_effect(2,
                    [](const std::vector<std::pair<std::string, std::any>>& kwargs) {
                      return GetIntKwarg(kwargs, "atomic", static_cast<int>(AtomicType::kNone)) ==
                                     static_cast<int>(AtomicType::kNone)
                                 ? ArgEffect::Write
                                 : ArgEffect::ReadWrite;
                    })
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileStoreType(args, kwargs, "tile.store");
    });

// ============================================================================
// tile.mscatter: scatter-store tile elements to tensor via per-element indices
// Maps to pto.mscatter: mem[idx[i, j]] = src[i, j]
// ============================================================================

TypePtr DeduceTileMscatterType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name
                          << " requires 3 arguments (src, idx, output_tensor), but got " << args.size();

  // First arg: src tile (FP16/FP32/INT16/INT32)
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32 ||
        src_type->dtype_ == DataType::INT16 || src_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires src dtype to be FP16, FP32, INT16, or INT32, but got "
      << src_type->dtype_.ToString();

  // Second arg: idx tile (INT32, same rank as src)
  auto idx_type = As<TileType>(args[1]->GetType());
  CHECK(idx_type) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                  << args[1]->GetType()->TypeName();
  CHECK(idx_type->dtype_ == DataType::INT32)
      << "The operator " << op_name << " requires idx dtype to be INT32, but got "
      << idx_type->dtype_.ToString();
  CHECK(idx_type->shape_.size() == src_type->shape_.size())
      << "The operator " << op_name << " requires idx rank to match src rank (" << src_type->shape_.size()
      << "), but got " << idx_type->shape_.size();
  for (size_t i = 0; i < src_type->shape_.size(); ++i) {
    auto src_dim = As<ConstInt>(src_type->shape_[i]);
    auto idx_dim = As<ConstInt>(idx_type->shape_[i]);
    if (src_dim && idx_dim) {
      CHECK(src_dim->value_ == idx_dim->value_)
          << "The operator " << op_name << " requires idx shape to match src shape at dimension " << i
          << ", but got " << idx_dim->value_ << " vs " << src_dim->value_;
    }
  }

  // Third arg: output tensor (same dtype as src, must not be scalar).
  // AsTensorTypeLike accepts both TensorType and DistributedTensorType — the
  // latter is needed when scattering into a per-rank window-buffer slice.
  auto tensor_type = AsTensorTypeLike(args[2]->GetType());
  CHECK(tensor_type) << "The operator " << op_name
                     << " requires third argument to be a TensorType or DistributedTensorType, but got "
                     << args[2]->GetType()->TypeName();
  CHECK(!tensor_type->shape_.empty())
      << "The operator " << op_name
      << " requires output_tensor to have at least 1 dimension (scalar not supported)";
  CHECK(tensor_type->dtype_ == src_type->dtype_)
      << "The operator " << op_name << " requires output_tensor dtype (" << tensor_type->dtype_.ToString()
      << ") to match src dtype (" << src_type->dtype_.ToString() << ")";

  // mscatter returns the output tensor's type unchanged. Returning the original
  // GetType() (rather than the AsTensorTypeLike upcast) keeps the ObjectKind
  // and DistributedTensorType::window_buffer_ intact for downstream passes.
  return args[2]->GetType();
}

REGISTER_OP("tile.mscatter")
    .set_op_category("TileOp")
    .set_description(
        "Scatter-store elements from src tile to tensor at per-element indices "
        "(maps to pto.mscatter)")
    .add_argument("src", "Source tile (FP16, FP32, INT16, or INT32)")
    .add_argument("idx", "Index tile (INT32, same rank as src)")
    .add_argument("output_tensor", "Output tensor (TensorType, same dtype as src)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_reuses_input(2)
    // Scatters `src` into the indexed cells of `output_tensor` without reading
    // any of it — the same pure-write destination contract as tile.store.
    .set_arg_effect(2, ArgEffect::Write)
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMscatterType(args, kwargs, "tile.mscatter");
    });

namespace {
constexpr int kMgatherGatherOobUndefined = 0;
constexpr int kMgatherGatherOobZero = 3;

bool IsMgatherElementDtype(const DataType& dtype) {
  return dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
         dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::UINT32 ||
         dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
         dtype == DataType::FP8E4M3FN || dtype == DataType::FP8E5M2 || dtype == DataType::HF8;
}

bool IsStaticContiguousTensor(const TensorType& type) {
  if (!type.tensor_view_.has_value()) return true;
  const TensorView& view = *type.tensor_view_;
  if (view.layout != TensorLayout::ND) return false;
  if (view.stride.empty()) return true;
  if (view.stride.size() != type.shape_.size()) return false;

  int64_t expected_stride = 1;
  for (size_t offset = 0; offset < type.shape_.size(); ++offset) {
    const size_t dim_index = type.shape_.size() - 1 - offset;
    auto stride = As<ConstInt>(view.stride[dim_index]);
    auto dim = As<ConstInt>(type.shape_[dim_index]);
    if (!stride || !dim) return false;
    if (dim->value_ != 1 && stride->value_ != expected_stride) return false;
    expected_stride *= dim->value_;
  }
  return true;
}

bool IsNdTensor(const TensorType& type) {
  return !type.tensor_view_.has_value() || type.tensor_view_->layout == TensorLayout::ND;
}
}  // namespace

TypePtr DeduceTileMgatherType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::string& op_name) {
  CHECK(args.size() >= 2 && args.size() <= 4)
      << "The operator " << op_name
      << " requires (mem, idx), optionally followed by scratch and/or valid_shape, but got " << args.size()
      << " arguments";

  auto mem_type = AsTensorTypeLike(args[0]->GetType());
  CHECK(mem_type) << "The operator " << op_name
                  << " requires mem to be a TensorType or DistributedTensorType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(IsMgatherElementDtype(mem_type->dtype_))
      << "The operator " << op_name
      << " requires mem dtype in {I8, U8, I16, U16, I32, U32, FP16, BF16, FP32, "
         "FP8E4M3FN, FP8E5M2, HF8}, but got "
      << mem_type->dtype_.ToString();
  CHECK(!mem_type->shape_.empty()) << "The operator " << op_name
                                   << " requires mem to have at least one dimension";

  int coalesce = GetKwarg<int>(kwargs, "coalesce", static_cast<int>(MgatherCoalesceMode::kRow));
  CHECK(coalesce == static_cast<int>(MgatherCoalesceMode::kRow) ||
        coalesce == static_cast<int>(MgatherCoalesceMode::kElem))
      << "The operator " << op_name << " requires coalesce in {0 (row), 1 (elem)}, but got " << coalesce;
  int gather_oob = GetKwarg<int>(kwargs, "gather_oob", kMgatherGatherOobUndefined);
  CHECK(gather_oob >= kMgatherGatherOobUndefined && gather_oob <= kMgatherGatherOobZero)
      << "The operator " << op_name
      << " requires gather_oob in {0 (undefined), 1 (clamp), 2 (wrap), 3 (zero)}, but got " << gather_oob;
  MemorySpace target_memory = GetKwarg<MemorySpace>(kwargs, "target_memory", MemorySpace::Vec);
  CHECK(target_memory == MemorySpace::Vec || target_memory == MemorySpace::Mat)
      << "The operator " << op_name << " requires target_memory to be Vec or Mat";

  std::vector<ExprPtr> output_shape;
  std::vector<ExprPtr> output_valid_shape;
  TileView tile_view;
  if (target_memory == MemorySpace::Vec) {
    CHECK(args.size() == 2) << "The operator " << op_name << " permits scratch only for Mat elem mode";
    auto idx_type = As<TileType>(args[1]->GetType());
    CHECK(idx_type) << "The operator " << op_name
                    << " with Vec output requires idx to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
    CHECK(idx_type->dtype_ == DataType::INT32)
        << "The operator " << op_name << " requires idx dtype to be INT32, but got "
        << idx_type->dtype_.ToString();
    CHECK(idx_type->shape_.size() == 2)
        << "The operator " << op_name << " requires a 2D idx tile, but got rank " << idx_type->shape_.size();
    const TileView idx_view = tile_view_semantics::GetEffectiveTileView(*idx_type);
    if (coalesce == static_cast<int>(MgatherCoalesceMode::kElem)) {
      output_shape = idx_type->shape_;
      output_valid_shape = idx_view.valid_shape;
    } else {
      CHECK(mem_type->shape_.size() >= 2)
          << "The operator " << op_name
          << " row mode requires mem rank >= 2 so rows have an element dimension, but got rank "
          << mem_type->shape_.size();
      auto first_dim = As<ConstInt>(idx_type->shape_[0]);
      auto second_dim = As<ConstInt>(idx_type->shape_[1]);
      CHECK(first_dim && second_dim) << "The operator " << op_name
                                     << " row mode requires a static [1, R] or [R, 1] idx shape";
      const bool row_vector = first_dim->value_ == 1;
      const bool column_vector = second_dim->value_ == 1;
      CHECK(row_vector || column_vector)
          << "The operator " << op_name << " row mode requires a [1, R] or [R, 1] idx shape, but got ["
          << first_dim->value_ << ", " << second_dim->value_ << "]";
      const size_t row_dim = row_vector ? 1 : 0;
      output_shape = {idx_type->shape_[row_dim], mem_type->shape_.back()};
      CHECK(idx_view.valid_shape.size() == 2)
          << "The operator " << op_name << " requires a 2D idx valid shape";
      output_valid_shape = {idx_view.valid_shape[row_dim], mem_type->shape_.back()};
    }
    tile_view.blayout = TileLayout::row_major;
  } else {
    CHECK(IsNdTensor(*mem_type)) << "The operator " << op_name
                                 << " with Mat output requires mem to use ND tensor layout";
    auto idx_type = AsTensorTypeLike(args[1]->GetType());
    CHECK(idx_type) << "The operator " << op_name
                    << " with Mat output requires idx to be a GM TensorType, but got "
                    << args[1]->GetType()->TypeName();
    CHECK(IsNdTensor(*idx_type)) << "The operator " << op_name
                                 << " with Mat output requires idx to use ND tensor layout";
    CHECK(idx_type->dtype_ == DataType::INT32)
        << "The operator " << op_name << " requires Mat idx dtype to be INT32, but got "
        << idx_type->dtype_.ToString();
    CHECK(idx_type->shape_.size() == 2)
        << "The operator " << op_name << " requires a 2D Mat idx tensor, but got rank "
        << idx_type->shape_.size();
    if (coalesce == static_cast<int>(MgatherCoalesceMode::kElem)) {
      CHECK(args.size() == 3 || args.size() == 4)
          << "The operator " << op_name
          << " Mat elem mode requires GM scratch and optionally accepts valid_shape";
      CHECK(args[2].get() != args[0].get() && args[2].get() != args[1].get())
          << "The operator " << op_name
          << " Mat elem scratch must not alias mem or idx; use a distinct GM tensor";
      auto scratch_type = AsTensorTypeLike(args[2]->GetType());
      CHECK(scratch_type) << "The operator " << op_name << " Mat elem scratch must be a GM tensor";
      CHECK(scratch_type->dtype_ == mem_type->dtype_)
          << "The operator " << op_name << " Mat elem scratch dtype must match mem dtype";
      CHECK(IsStaticContiguousTensor(*scratch_type))
          << "The operator " << op_name << " Mat elem scratch must be a contiguous ND tensor";
      output_shape = idx_type->shape_;
      int64_t output_elements = 1;
      for (const auto& dim : output_shape) {
        auto constant = As<ConstInt>(dim);
        CHECK(constant) << "The operator " << op_name << " Mat elem output shape must be static";
        output_elements *= constant->value_;
      }
      int64_t scratch_elements = 1;
      for (const auto& dim : scratch_type->shape_) {
        auto constant = As<ConstInt>(dim);
        CHECK(constant) << "The operator " << op_name << " Mat elem scratch shape must be static";
        scratch_elements *= constant->value_;
      }
      CHECK(scratch_elements >= output_elements)
          << "The operator " << op_name << " Mat elem scratch requires at least " << output_elements
          << " elements, but got " << scratch_elements;
    } else {
      CHECK(args.size() == 2 || args.size() == 3)
          << "The operator " << op_name << " Mat row mode accepts only an optional valid_shape";
      CHECK(mem_type->shape_.size() >= 2)
          << "The operator " << op_name << " Mat row mode requires mem rank >= 2";
      auto first_dim = As<ConstInt>(idx_type->shape_[0]);
      CHECK(first_dim && first_dim->value_ == 1)
          << "The operator " << op_name << " Mat row mode requires a [1, R] GM idx tensor";
      output_shape = {idx_type->shape_[1], mem_type->shape_.back()};
    }
    auto output_rows = As<ConstInt>(output_shape[0]);
    auto output_cols = As<ConstInt>(output_shape[1]);
    CHECK(output_rows && output_cols) << "The operator " << op_name << " Mat output shape must be static";
    const int64_t element_bytes = static_cast<int64_t>(mem_type->dtype_.GetByte());
    CHECK(element_bytes > 0) << "The operator " << op_name << " requires a byte-addressable mem dtype";
    const int64_t c0 = 32 / element_bytes;
    CHECK(output_rows->value_ % 16 == 0)
        << "The operator " << op_name << " Mat output rows must be a multiple of 16";
    CHECK(output_cols->value_ % c0 == 0)
        << "The operator " << op_name << " Mat output cols must be a multiple of " << c0;
    output_valid_shape = output_shape;
    const size_t valid_shape_index = coalesce == static_cast<int>(MgatherCoalesceMode::kElem) ? 3 : 2;
    if (args.size() > valid_shape_index) {
      auto valid_shape = As<MakeTuple>(args[valid_shape_index]);
      CHECK(valid_shape) << "The operator " << op_name
                         << " Mat valid_shape must be a MakeTuple of compile-time constants";
      CHECK(valid_shape->elements_.size() == output_shape.size())
          << "The operator " << op_name << " Mat valid_shape must have rank " << output_shape.size()
          << ", but got " << valid_shape->elements_.size();
      output_valid_shape.clear();
      for (size_t i = 0; i < valid_shape->elements_.size(); ++i) {
        auto valid_dim = As<ConstInt>(valid_shape->elements_[i]);
        auto physical_dim = As<ConstInt>(output_shape[i]);
        CHECK(valid_dim) << "The operator " << op_name << " Mat valid_shape element " << i
                         << " must be a compile-time constant";
        CHECK(valid_dim->value_ > 0 && valid_dim->value_ <= physical_dim->value_)
            << "The operator " << op_name << " requires 0 < Mat valid_shape[" << i << "] <= output_shape["
            << i << "], but got " << valid_dim->value_ << " and " << physical_dim->value_;
        output_valid_shape.push_back(valid_shape->elements_[i]);
      }
    }
    tile_view.blayout = TileLayout::col_major;
    tile_view.slayout = TileLayout::row_major;
  }

  tile_view.valid_shape = output_valid_shape;
  return std::make_shared<TileType>(output_shape, mem_type->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.mgather")
    .set_op_category("TileOp")
    .set_description(
        "Gather-load rows or elements from a GM tensor into a fresh Vec or Mat tile "
        "(maps to pto.mgather)")
    .add_argument("mem", "GM source table (TensorType or DistributedTensorType)")
    .add_argument("idx", "INT32 2D index tile for Vec, or GM tensor for Mat")
    .add_argument("scratch", "Optional GM scratch tensor for Mat elem mode")
    .add_argument("valid_shape", "Optional 2D written region for Mat output")
    .set_attr<int>("coalesce")
    .set_attr<int>("gather_oob")
    .set_attr<MemorySpace>("target_memory")
    .set_output_memory_from_kwarg("target_memory", MemorySpace::Vec)
    .not_inplace_safe()
    // Argument 2 is the GM `scratch` tensor only in Mat *elem* mode, where the
    // gathered elements are staged through it. In Mat row mode that position
    // holds `valid_shape`, and in Vec mode it is absent — declaring an
    // unconditional write there would claim a tuple operand is a written
    // buffer and could promote a read-only parameter to an output.
    .set_arg_effect(2,
                    [](const std::vector<std::pair<std::string, std::any>>& kwargs) {
                      const bool mat_output =
                          GetMemorySpaceKwarg(kwargs, "target_memory", MemorySpace::Vec) == MemorySpace::Mat;
                      const bool elem_mode =
                          GetIntKwarg(kwargs, "coalesce", static_cast<int>(MgatherCoalesceMode::kRow)) ==
                          static_cast<int>(MgatherCoalesceMode::kElem);
                      return mat_output && elem_mode ? ArgEffect::Write : ArgEffect::Read;
                    })
    .set_write_channel(WriteChannel::Dma)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMgatherType(args, kwargs, "tile.mgather");
    });

REGISTER_OP("tile.move")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Move tile between memory levels (Vec/Mat/Left/Right/LeftScale/RightScale)")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<TileLayout>("blayout")
    .set_attr<TileLayout>("slayout")
    // PTO TMOV requires distinct source and destination addresses. Keep memory
    // planners from placing the result on any input buffer; baked-address PTO
    // codegen also validates this invariant for explicit or hand-built aliases.
    .not_inplace_safe()
    .set_output_memory_from_kwarg("target_memory", MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMoveType(args, kwargs, "tile.move");
    });

// tile.alloc is emitted by InitMemRef, which runs after ExpandMixedKernel —
// by then the program is already split into AIC/AIV functions and the
// classification is only consulted by the expanded-kernel verifier. VECTOR
// preserves the pre-refactor behavior (tile.* fallback → VECTOR); a future
// refinement could classify by the memory_space arg if a use case arises.
REGISTER_OP("tile.alloc")
    .set_op_category("TileOp")
    .no_execution_memory_access()
    .set_description("Declare on-chip memory allocation, returning a Ptr")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("memory_space", "Memory space (int enum value)")
    .add_argument("size", "Size in bytes (scalar)")
    // `pinned` marks an author-declared allocation (one-argument `pl.MemRef`).
    // PyPTO memory planners leave such a buffer's membership exactly as the
    // author wrote it: they neither pack other tiles into it nor move its
    // tiles elsewhere.
    .set_attr<bool>("pinned")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileAllocType(args, kwargs, "tile.alloc");
    });

REGISTER_OP("tile.full")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Create a tile of specified shape and filling value in UB")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("value", "Filling value (ConstInt or ConstFloat)")
    .set_attr<DataType>("dtype")
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileFullType(args, kwargs, "tile.full");
    });

REGISTER_OP("tile.ci")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Generate a contiguous integer sequence into a destination tile (pto.tci)")
    .add_argument("start", "Starting integer scalar (must match dst dtype)")
    .add_argument("shape", "Destination shape (TupleType of ConstInt)")
    .add_argument("tmp", "Optional A2/A3 scratch tile (FP32 Vec)")
    .set_attr<DataType>("dtype")
    .set_attr<bool>("descending")
    .set_input_memory(2, MemorySpace::Vec)
    // The A2/A3 PTOAS level3 TCI form takes tmp as an explicit scratch input
    // and may still read it while producing dst, so MemoryReuse cannot recycle
    // tmp's allocation for the output. A5 normally uses the tmp-free form, and
    // InitMemRef never synthesizes this operand for A5.
    .forbid_output_alias(2)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCiType(args, kwargs, "tile.ci");
    });

REGISTER_OP("tile.tri")
    .set_op_category("TileOp")
    .set_description("Generate a lower/upper triangular mask tile (pto.ttri)")
    .add_argument("diagonal", "Diagonal offset scalar (INT32)")
    .add_argument("shape", "Destination shape (2D TupleType of ConstInt)")
    .add_argument("valid_shape", "Optional written region (2D TupleType of ConstInt, <= shape)")
    .set_attr<DataType>("dtype")
    .set_attr<bool>("upper")
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTriType(args, kwargs, "tile.tri");
    });

REGISTER_OP("tile.random")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Generate counter-based pseudo-random values into a destination tile (pto.trandom)")
    .add_argument("key0", "First key word (INT32 scalar)")
    .add_argument("key1", "Second key word (INT32 scalar)")
    .add_argument("counter0", "Counter word 0 (INT32 scalar)")
    .add_argument("counter1", "Counter word 1 (INT32 scalar)")
    .add_argument("counter2", "Counter word 2 (INT32 scalar)")
    .add_argument("counter3", "Counter word 3 (INT32 scalar)")
    .add_argument("shape", "Destination shape (TupleType of ConstInt)")
    .add_argument("valid_shape", "Optional written region (TupleType of ConstInt, <= shape)")
    .set_attr<DataType>("dtype")
    .set_attr<int>("rounds")
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRandomType(args, kwargs, "tile.random");
    });

}  // namespace ir
}  // namespace pypto
