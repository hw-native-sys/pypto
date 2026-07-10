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
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// tile.load's per-dim intersect with the source's own valid region is the
// shared IntersectWindowValidDim in type_inference.h: the load window of
// ``shapes[i]`` elements at ``offsets[i]`` into a tensor whose valid extent is
// ``valid(tensor)[i]`` sees clamp(valid(tensor)[i] - offsets[i], 0, shapes[i]),
// clipped by the explicit ``valid_shapes=`` argument. Never widens past the
// tensor's real region.

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
  // load signature: (tensor, offsets_tuple, shapes_tuple, valid_shapes_tuple)
  CHECK(args.size() == 4) << "The operator " << op_name
                          << " requires 4 arguments (tensor, offsets, shapes, valid_shapes), but got "
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

  // Fourth argument must be TupleType (valid_shapes)
  auto valid_shapes_tuple = As<MakeTuple>(args[3]);
  CHECK(valid_shapes_tuple) << "The operator " << op_name
                            << " requires fourth argument to be a tuple (valid shapes), but got "
                            << args[3]->GetType()->TypeName();

  // Verify offsets, shapes and valid_shapes have same number of dimensions
  CHECK(offsets_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires offsets and shapes to have same number of dimensions, but got "
      << offsets_tuple->elements_.size() << " offsets and " << shapes_tuple->elements_.size() << " shapes";
  CHECK(valid_shapes_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires valid_shapes and shapes to have same number of dimensions, but got "
      << valid_shapes_tuple->elements_.size() << " valid_shapes and " << shapes_tuple->elements_.size()
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
  // Nz/Zn layout: only chosen when target_memory is known. If it is absent,
  // the default-constructed view is kept and InferTileMemorySpace rebuilds it
  // once the memory space is resolved.
  //
  // Source-DN equivalence (RFC #1300 §3.3 + P6): a DN-tagged source tensor
  // describes the same physical bytes as the canonical-pair ND view, so
  // ``tile.load`` of a DN source produces the transposed (ZN) Mat layout.
  // A transposed matmul operand is realised by a zero-copy ``tile.transpose_view``
  // at the matmul site, not by the load.
  bool source_is_dn =
      tensor_type->tensor_view_.has_value() && tensor_type->tensor_view_->layout == TensorLayout::DN;
  TileView tile_view;
  if (target_memory_opt.has_value()) {
    if (*target_memory_opt == MemorySpace::Mat) {
      tile_view.blayout = TileLayout::col_major;
      tile_view.slayout = TileLayout::row_major;
      if (source_is_dn) {
        std::swap(tile_view.blayout, tile_view.slayout);
      }
    } else if (auto last_dim = As<ConstInt>(shapes_tuple->elements_.back());
               last_dim && last_dim->value_ == 1) {
      tile_view.blayout = TileLayout::col_major;
    }
  }

  // Build tile shape from shapes tuple (always in source-tensor coordinates).
  std::vector<ExprPtr> tile_shape(shapes_tuple->elements_.begin(), shapes_tuple->elements_.end());

  // Q1 = INTERSECT: when the source tensor carries an EXPLICIT valid region, the
  // tile's valid region is the load window clipped to that region and then
  // intersected with the explicit ``valid_shapes=`` argument (never widening past
  // the tensor's real valid extent). When the source has no explicit valid region
  // (the overwhelmingly common case: valid == full shape), keep ``valid_shapes``
  // verbatim — today's exact behavior and IR — so existing loads are untouched.
  const auto& valid_shapes = valid_shapes_tuple->elements_;
  const bool tensor_has_valid_region =
      tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->valid_shape.empty();
  if (tensor_has_valid_region) {
    const std::vector<ExprPtr> tensor_valid = GetValidShape(tensor_type);
    const auto& offsets = offsets_tuple->elements_;
    // tensor_valid is ranked by the source tensor; offsets/shapes/valid_shapes are
    // the per-tensor-dim load window. They coincide for an in-coordinate load; if a
    // rank-reducing form ever diverges, fall back to verbatim rather than misalign.
    if (tensor_valid.size() == valid_shapes.size() && offsets.size() == valid_shapes.size()) {
      std::vector<ExprPtr> intersected;
      intersected.reserve(valid_shapes.size());
      for (size_t i = 0; i < valid_shapes.size(); ++i) {
        intersected.push_back(IntersectWindowValidDim(tensor_valid[i], offsets[i], tile_shape[i],
                                                      valid_shapes[i], args[0]->span_));
      }
      tile_view.valid_shape = std::move(intersected);
    } else {
      tile_view.valid_shape = valid_shapes;
    }
    // Inherit the source tensor view's pad: accesses inside the tile but outside
    // the narrowed valid region take the tensor's pad value (previously dropped).
    if (tensor_type->tensor_view_->pad != PadValue::null) {
      tile_view.pad = tensor_type->tensor_view_->pad;
    }
  } else {
    tile_view.valid_shape = valid_shapes;
  }

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
  if (args.size() == 4) {
    auto shapes_tuple = As<MakeTuple>(args[3]);
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

  // store returns the output tensor (same type)
  return output_tensor_type;
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

  TileView tile_view;
  tile_view.blayout = source_view.blayout;
  tile_view.slayout = source_view.slayout;

  // Hardcoded layout for Left/Right (hardware requirements)
  if (space == MemorySpace::Left) {
    tile_view.blayout = TileLayout::col_major;  // L0A requires ColMajor block layout for TMATMUL
    tile_view.slayout = TileLayout::row_major;
  } else if (space == MemorySpace::Right) {
    tile_view.blayout = TileLayout::row_major;
    tile_view.slayout = TileLayout::col_major;
  }

  // Explicit kwargs override everything
  tile_view.blayout = GetKwarg<TileLayout>(kwargs, "blayout", tile_view.blayout);
  tile_view.slayout = GetKwarg<TileLayout>(kwargs, "slayout", tile_view.slayout);

  // Keep original shape
  std::vector<ExprPtr> output_shape = input_shape;

  // Preserve input valid_shape (may be narrower than shape_)
  tile_view.valid_shape = source_view.valid_shape.empty() ? input_shape : source_view.valid_shape;

  // Preserve pad value from input tile
  if (source_view.pad != PadValue::null) {
    tile_view.pad = source_view.pad;
  }

  // Return TileType with computed shape and same dtype (no explicit MemRef)
  return std::make_shared<TileType>(output_shape, tile_type->dtype_, std::nullopt, tile_view);
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
  // create_tile signature: (shape[, valid_shape])
  // TileType requires static compile-time constant shapes. The optional 2nd arg
  // is a MakeTuple valid_shape narrowing the initially-valid sub-region — needed
  // for the accumulator pattern, e.g. create_tile(valid_shape=[0, 0]) ("nothing
  // valid yet") so a later assemble union can grow the box (see "The empty
  // accumulator" in docs/en/dev/ir/08-valid_shape.md).
  CHECK(args.size() == 1 || args.size() == 2)
      << "The operator " << op_name << " requires 1 or 2 arguments (shape[, valid_shape]), but got "
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

  // Optional valid_shape (2nd arg): the initially-valid sub-region. Each element
  // is a ConstInt in [0, shape[i]] (0 is allowed — the empty-accumulator case) or
  // a symbolic expr (bounds unchecked, per the step-3 verifier convention). When
  // absent, the tile is fully valid (valid_shape == shape).
  std::vector<ExprPtr> valid_shape = tile_shape;
  if (args.size() == 2) {
    auto valid_tuple = As<MakeTuple>(args[1]);
    CHECK(valid_tuple) << "The operator " << op_name
                       << " requires the valid_shape argument to be a MakeTuple, but got "
                       << args[1]->TypeName();
    CHECK(valid_tuple->elements_.size() == tile_shape.size())
        << "The operator " << op_name << " valid_shape rank (" << valid_tuple->elements_.size()
        << ") must match shape rank (" << tile_shape.size() << ")";
    valid_shape.clear();
    valid_shape.reserve(valid_tuple->elements_.size());
    for (size_t i = 0; i < valid_tuple->elements_.size(); ++i) {
      auto v = As<ConstInt>(valid_tuple->elements_[i]);
      if (v) {
        auto dim = As<ConstInt>(tile_shape[i]);  // always ConstInt here
        CHECK(v->value_ >= 0 && (!dim || v->value_ <= dim->value_))
            << "The operator " << op_name << " valid_shape element " << i << " (" << v->value_
            << ") must be in [0, shape dim " << (dim ? dim->value_ : -1) << "]";
      }
      valid_shape.push_back(valid_tuple->elements_[i]);
    }
  }

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

  TileView tile_view;
  // `transpose=true` requests the transposed Mat (ZN) fractal layout
  // (blayout=row_major, slayout=col_major) — the layout a matmul B-operand
  // carries when loaded with b_trans, and the only Mat layout a DN-source
  // gather_row (DN2ZN tload) can fill. Default false keeps the canonical NZ.
  bool transpose_layout = false;
  // `flat_layout=true` requests a flat (non-fractal, slayout=none_box) L1/cbuf
  // tile: a contiguous byte-staging buffer rather than the boxed NZ layout Mat
  // tiles normally carry. Used for the mix/aic_only soft `system.syncall` L1
  // scratch (pto-isa `Tile<TileType::Mat, ..., SLayout::NoneBox>`), whose 8
  // int32 counter slots must be contiguous — a fractal layout mis-places them.
  bool flat_layout = false;
  for (const auto& [k, v] : kwargs) {
    if (k == "transpose") transpose_layout = AnyCast<bool>(v, "transpose");
    if (k == "flat_layout") flat_layout = AnyCast<bool>(v, "flat_layout");
  }
  // The transposed Mat (ZN) layout is a 2D L1 matmul-`b_trans` operand layout; it
  // is meaningless for a non-Mat space or a non-2D shape. Fail fast rather than
  // emit an invalid tile (mirrors tile.load's Mat-only transpose guard).
  CHECK(!transpose_layout ||
        (tile_shape.size() == 2 && target_memory_opt.has_value() && *target_memory_opt == MemorySpace::Mat))
      << "The operator " << op_name
      << " supports transpose=true only for a 2D tile with target_memory=Mat (L1)";
  // flat_layout is a Mat (L1/cbuf) staging layout and mutually exclusive with the
  // transposed NZ layout.
  CHECK(!flat_layout ||
        (target_memory_opt.has_value() && *target_memory_opt == MemorySpace::Mat && !transpose_layout))
      << "The operator " << op_name
      << " supports flat_layout=true only for target_memory=Mat (L1) without transpose";

  // A flat L1 tile keeps the canonical flat view (blayout=row_major,
  // slayout=none_box, fractal default) — it is deliberately NOT boxed. We also
  // stamp memory_space_=Mat at creation so InferTileMemorySpace sees the space
  // is already resolved and preserves the none_box view instead of overwriting
  // it with Mat's implicit boxed layout (see ComputeRewrittenType).
  std::optional<MemorySpace> creation_space = std::nullopt;
  if (flat_layout) {
    creation_space = MemorySpace::Mat;
  } else if (target_memory_opt.has_value() && *target_memory_opt == MemorySpace::Acc) {
    tile_view.blayout = TileLayout::col_major;
    tile_view.slayout = TileLayout::row_major;
    tile_view.fractal = 1024;
  } else if (transpose_layout) {
    tile_view.blayout = TileLayout::row_major;
    tile_view.slayout = TileLayout::col_major;
  } else if (tile_shape.size() == 2) {
    auto rows_const = As<ConstInt>(tile_shape[0]);
    auto cols_const = As<ConstInt>(tile_shape[1]);
    if (rows_const && cols_const && rows_const->value_ > 1 && cols_const->value_ == 1) {
      tile_view.blayout = TileLayout::col_major;
    }
  }
  tile_view.valid_shape = std::move(valid_shape);
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
  // tile.ci signature: (start, shape) with attrs {dtype, descending}
  CHECK(args.size() == 2) << "The operator " << op_name
                          << " requires exactly 2 arguments (start, shape), but got " << args.size();

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

  TileView tile_view;
  tile_view.valid_shape = tile_shape;
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
    .add_argument("valid_shape",
                  "Optional initially-valid sub-region (TupleType of ConstInt, each in [0, shape])")
    .set_attr<DataType>("dtype")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<bool>("transpose")
    .set_attr<bool>("flat_layout")
    // No fallback: when target_memory is absent, memory_space stays unresolved and
    // InferTileMemorySpace picks the space from consumer demand.
    .set_output_memory_from_kwarg("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCreateTileType(args, kwargs, "tile.create");
    });

REGISTER_OP("tile.load")
    .set_op_category("TileOp")
    .set_description("Copy data from tensor to unified buffer (tile)")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("offsets",
                  "Offsets in each dimension, in source tensor coordinates (TupleType of ScalarType)")
    .add_argument(
        "shapes",
        "Shape of region to load in each dimension, in source tensor coordinates (TupleType of ScalarType)")
    .add_argument(
        "valid_shapes",
        "Valid shape of tile in each dimension, in source tensor coordinates (TupleType of ScalarType). ")
    .set_attr<MemorySpace>("target_memory")
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
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMscatterType(args, kwargs, "tile.mscatter");
    });

REGISTER_OP("tile.move")
    .set_op_category("TileOp")
    .set_description("Move tile between memory levels (Vec/Mat/Left/Right)")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<TileLayout>("blayout")
    .set_attr<TileLayout>("slayout")
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
    .set_description("Declare on-chip memory allocation, returning a Ptr")
    .set_core_affinity(core_affinity::CoreAffinity::VECTOR)
    .add_argument("memory_space", "Memory space (int enum value)")
    .add_argument("size", "Size in bytes (scalar)")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileAllocType(args, kwargs, "tile.alloc");
    });

REGISTER_OP("tile.full")
    .set_op_category("TileOp")
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
    .set_description("Generate a contiguous integer sequence into a destination tile (pto.tci)")
    .add_argument("start", "Starting integer scalar (must match dst dtype)")
    .add_argument("shape", "Destination shape (TupleType of ConstInt)")
    .set_attr<DataType>("dtype")
    .set_attr<bool>("descending")
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCiType(args, kwargs, "tile.ci");
    });

REGISTER_OP("tile.random")
    .set_op_category("TileOp")
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
