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
 * @file pto_ops_memory.cpp
 * @brief PTO codegen registration for memory / tensor / array / SPMD ops.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/pto_ops_common.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/distributed/comm_layout.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
#include "src/backend/common/pto_ops_internal.h"

namespace pypto {
namespace backend {

using ir::As;
using ir::AsTensorTypeLike;
using ir::AsVarLike;
using ir::CallPtr;
using ir::ExprPtr;
using ir::ScalarType;
using ir::TensorType;
using ir::Var;

using pto_ops_detail::AsPto;
using pto_ops_detail::CheckArity;
using pto_ops_detail::EmitFlatOffsetSSAFromValues;
using pto_ops_detail::EmitIndexOperand;
using pto_ops_detail::EmitPartitionViewPTO;
using pto_ops_detail::GetDimStrings;
using pto_ops_detail::GetIndexOffsetCodes;
using pto_ops_detail::GetSizeCodes;
using pto_ops_detail::MakePartitionTensorViewType;

// Helper function for StoreFP
static std::string MakeStoreFPCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CheckArity(op, pto_op_name, 3);
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string fp = codegen.GetExprAsCode(op->args_[1]);
  std::string mem = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(pto_op_name + " ins(" + src + ", " + fp + ") outs(" + mem + ")");
  return "";
}

// tile.load: emit pto.subview + pto.tload
static std::string MakeTileLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  auto tensor = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(tensor, op->span_) << "tile.load first argument must be a Var or IterArg";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "tile.load second argument must be a tuple (offsets)";

  INTERNAL_CHECK_SPAN(op->args_.size() >= 3, op->span_)
      << "tile.load expects at least 3 arguments (tensor, offsets, shapes), but got " << op->args_.size();

  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(shapes_tuple, op->span_) << "tile.load third argument must be a tuple (shapes)";

  // valid_shapes is optional: when omitted (callers built before the 4-arg
  // signature was introduced, or hand-written IR), fall back to shapes so the
  // partition_view covers the entire physical region — equivalent to the DSL
  // behavior `pl.load(..., valid_shapes=None)`.
  auto valid_shapes_tuple = shapes_tuple;
  if (op->args_.size() >= 4) {
    valid_shapes_tuple = As<ir::MakeTuple>(op->args_[3]);
    INTERNAL_CHECK_SPAN(valid_shapes_tuple, op->span_)
        << "tile.load fourth argument must be a tuple (valid_shapes)";
  }

  auto tensor_type = AsTensorTypeLike(tensor->GetType());
  INTERNAL_CHECK_SPAN(tensor_type, op->span_) << "tile.load tensor argument must have TensorType";

  const size_t ndim = shapes_tuple->elements_.size();
  INTERNAL_CHECK_SPAN(ndim >= 1, op->span_) << "tile.load shapes tuple must have at least one element";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK_SPAN(!tile_buf.empty(), op->span_) << "tile.load requires assignment target (tile_buf)";

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();

  // RFC #1300 P7: the IR's offsets / shapes / valid_shapes are already in
  // canonical coordinates (matching the source TensorType's shape). There is
  // no implicit dn_swap here — earlier passes ensure all coordinate systems
  // match before codegen.
  std::vector<std::string> partition_dims = GetDimStrings(valid_shapes_tuple->elements_);
  std::vector<std::string> offset_codes = GetIndexOffsetCodes(offsets_tuple->elements_, codegen);
  std::vector<std::string> size_codes = GetSizeCodes(valid_shapes_tuple->elements_, codegen);

  // ND2NZ constraint: a natural Mat load fills an NZ tile (the implicit Mat view,
  // blayout=col_major / slayout=row_major), and the hardware ND2NZ path requires a
  // 2-dim GlobalTensor. When such an NZ load has a rank>2 source window, collapse
  // the contiguous leading dims to 2D — emit a fresh 2D tensor_view ([prod(leading
  // window dims), last] with strides [last, 1]) over the same base pointer and a
  // matching 2D partition (offset folded mixed-radix over the source tensor dims).
  // Transposed (DN) Mat loads keep their ND window for DN addressing; Vec/other
  // loads are plain ND and are not NZ.
  auto result_tile_type = ir::As<ir::TileType>(op->GetType());
  bool is_nz_mat_load = false;
  if (result_tile_type && result_tile_type->memory_space_ == ir::MemorySpace::Mat) {
    const auto rv = ir::tile_view_semantics::GetEffectiveTileView(*result_tile_type);
    is_nz_mat_load = rv.blayout == ir::TileLayout::col_major && rv.slayout == ir::TileLayout::row_major;
  }
  if (is_nz_mat_load && ndim > 2) {
    const ir::Span& span = op->span_;
    // The leading-dim collapse folds the row dims [0, ndim-1) into one axis with a
    // contiguous row stride, so it is sound only when the valid sub-box of those
    // dims is contiguous in row-major order (see IsRowMajorCollapseContiguous —
    // the shared rule that FlattenTileNdTo2D uses to route non-contiguous operands
    // to a per-batch load, so this guard should never fire on a batch_matmul
    // operand). A partial middle dim under a non-singleton outer dim (e.g. a
    // multi-batch slice that also cuts the matrix-row dim) is non-contiguous.
    INTERNAL_CHECK_SPAN(ir::tile_conversion_utils::IsRowMajorCollapseContiguous(valid_shapes_tuple->elements_,
                                                                                tensor_type->shape_),
                        span)
        << "tile.load NZ 2D source-window collapse: the valid sub-box of the leading dims is not "
           "contiguous in row-major order (a partial middle dim under a non-singleton outer dim), so "
           "the collapse cannot legalize this to a 2D ND2NZ load";
    // ConstInt-folding index arithmetic: a static window (the common matmul case)
    // folds to clean constants, while a dynamic dim/offset/valid stays symbolic and
    // is materialized by GetSizeCodes / GetIndexOffsetCodes (arith.muli/addi) below
    // — the same constant-or-symbol handling the function-parameter make_tensor_view
    // uses, so this path is not limited to static-shape tensors.
    auto idx = [&](int64_t v) -> ir::ExprPtr {
      return std::make_shared<ir::ConstInt>(v, DataType::INDEX, span);
    };
    auto fold_mul = [&](const ir::ExprPtr& a, const ir::ExprPtr& b) -> ir::ExprPtr {
      auto ca = ir::As<ir::ConstInt>(a);
      auto cb = ir::As<ir::ConstInt>(b);
      if (ca && cb) return idx(ca->value_ * cb->value_);
      if (ca && ca->value_ == 1) return b;
      if (cb && cb->value_ == 1) return a;
      return ir::MakeMul(a, b, span);
    };
    auto fold_add = [&](const ir::ExprPtr& a, const ir::ExprPtr& b) -> ir::ExprPtr {
      auto ca = ir::As<ir::ConstInt>(a);
      auto cb = ir::As<ir::ConstInt>(b);
      if (ca && cb) return idx(ca->value_ + cb->value_);
      if (ca && ca->value_ == 0) return b;
      if (cb && cb->value_ == 0) return a;
      return ir::MakeAdd(a, b, span);
    };

    // make_tensor_view describes the collapsed SOURCE TENSOR: [prod(tensor leading
    // dims), tensor_last] with a contiguous row stride [tensor_last, 1] — the
    // tensor's real row stride, so a window narrower than the last dim (e.g. a
    // K-slice) still strides by the full tensor width. The partition then slices the
    // window's VALID region; its offset folds mixed-radix over the tensor dims.
    ir::ExprPtr tensor_rows = tensor_type->shape_[0];
    ir::ExprPtr valid_rows = valid_shapes_tuple->elements_[0];
    ir::ExprPtr row_offset = offsets_tuple->elements_[0];
    for (size_t i = 1; i + 1 < ndim; ++i) {
      tensor_rows = fold_mul(tensor_rows, tensor_type->shape_[i]);
      valid_rows = fold_mul(valid_rows, valid_shapes_tuple->elements_[i]);
      row_offset = fold_add(fold_mul(row_offset, tensor_type->shape_[i]), offsets_tuple->elements_[i]);
    }
    const ir::ExprPtr tensor_cols = tensor_type->shape_.back();

    const std::vector<std::string> view_shape = GetSizeCodes({tensor_rows, tensor_cols}, codegen);
    const std::string one = codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
    const std::string view2d = codegen.NewNamedTemp(tensor->name_hint_ + "_view2d");
    std::ostringstream mtv;
    mtv << view2d << " = pto.make_tensor_view " << codegen.GetVarName(tensor) << ", shape = ["
        << view_shape[0] << ", " << view_shape[1] << "], strides = [" << view_shape[1] << ", " << one
        << "] {layout = #pto.layout<nd>}: !pto.tensor_view<?x?x" << dtype_str << ">";
    codegen.Emit(mtv.str());
    tensor_view = view2d;
    tensor_view_type = "!pto.tensor_view<?x?x" + dtype_str + ">";
    partition_dims = {"?", "?"};
    offset_codes = GetIndexOffsetCodes({row_offset, offsets_tuple->elements_.back()}, codegen);
    size_codes = GetSizeCodes({valid_rows, valid_shapes_tuple->elements_.back()}, codegen);
  }

  std::string partition_type = MakePartitionTensorViewType(partition_dims, dtype_str);
  std::string partition_view = EmitPartitionViewPTO(tensor->name_hint_, tensor_view, tensor_view_type,
                                                    partition_type, offset_codes, size_codes, codegen);

  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << partition_view << " : " << partition_type << ") outs(";
  tload_line << tile_buf << " : " << tile_buf_type << ")";
  codegen.Emit(tload_line.str());

  // No follow-up `pto.set_validshape` is emitted: every `pto.alloc_tile`
  // already carries the desired `valid_row` / `valid_col` operands, and the
  // partition_view above already reflects the same valid region.

  return "";
}

// tile.store: emit pto.partition_view + pto.tstore
static std::string MakeTileStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  auto tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(tile, op->span_) << "tile.store first argument must be a Var or IterArg";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "tile.store second argument must be a tuple (offsets)";

  auto tile_type = As<ir::TileType>(tile->GetType());
  INTERNAL_CHECK_SPAN(tile_type, op->span_) << "tile.store first argument must have TileType";
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*tile_type);
  const auto& valid_shape = tile_view.valid_shape;
  INTERNAL_CHECK_SPAN(valid_shape.size() == 2, op->span_) << "tile.store tile valid_shape must be 2D";

  auto height_code = codegen.GetExprAsCode(valid_shape[0]);
  auto width_code = codegen.GetExprAsCode(valid_shape[1]);

  auto output_tensor = AsVarLike(op->args_[2]);
  INTERNAL_CHECK_SPAN(output_tensor, op->span_) << "tile.store output_tensor must be a Var or IterArg";

  auto tensor_type = AsTensorTypeLike(output_tensor->GetType());
  INTERNAL_CHECK_SPAN(tensor_type, op->span_) << "tile.store output_tensor must have TensorType";

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::string partition_view;
  std::string partition_type;
  const size_t tensor_rank = tensor_type->shape_.size();

  // RFC #1300 P7: the IR's offsets / shapes are already in canonical
  // coordinates (matching the source TensorType's shape). No implicit
  // dn_swap here — the IR-level lowering passes (P6 + canonical TensorView)
  // are responsible for ensuring all coordinate systems match before codegen.

  // Check if FlattenTileNdTo2D injected an explicit shapes tuple as args[3].
  ir::MakeTuplePtr shapes_tuple;
  if (tensor_rank > 2 && op->args_.size() > 3) {
    shapes_tuple = As<ir::MakeTuple>(op->args_[3]);
  }

  if (shapes_tuple) {
    // N-rank partition path: use the explicit shapes tuple from FlattenTileNdTo2D.
    const auto& shape_elems = shapes_tuple->elements_;
    const auto& offset_elems = offsets_tuple->elements_;
    partition_type = MakePartitionTensorViewType(GetDimStrings(shape_elems), dtype_str);
    partition_view = EmitPartitionViewPTO(output_tensor->name_hint_, tensor_view, tensor_view_type,
                                          partition_type, GetIndexOffsetCodes(offset_elems, codegen),
                                          GetSizeCodes(shape_elems, codegen), codegen);
  } else {
    // Standard 1D/2D path
    std::string height_dim = "?", width_dim = "?";
    if (auto h = As<ir::ConstInt>(valid_shape[0])) height_dim = std::to_string(h->value_);
    if (auto w = As<ir::ConstInt>(valid_shape[1])) width_dim = std::to_string(w->value_);
    partition_type = MakePartitionTensorViewType({height_dim, width_dim}, dtype_str);
    partition_view = EmitPartitionViewPTO(
        output_tensor->name_hint_, tensor_view, tensor_view_type, partition_type,
        GetIndexOffsetCodes(offsets_tuple->elements_, codegen), {height_code, width_code}, codegen);
  }

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  if (!tile_buf_type.empty()) {
    tstore_line << " : " << tile_buf_type;
  }
  tstore_line << ") outs(" << partition_view << " : " << partition_type << ")";

  // Optional atomic-add combine mode (split-K accumulation into GM). The attr
  // is emitted only for atomic_add — a plain store omits it so non-atomic
  // codegen stays byte-identical (pto.tstore's atomicType defaults to none).
  const int atomic_int = op->GetKwarg<int>("atomic", 0);
  INTERNAL_CHECK_SPAN(atomic_int == static_cast<int>(ir::AtomicType::kNone) ||
                          atomic_int == static_cast<int>(ir::AtomicType::kAdd),
                      op->span_)
      << "tile.store atomic kwarg must encode AtomicType::kNone or kAdd, got " << atomic_int;
  if (atomic_int == static_cast<int>(ir::AtomicType::kAdd)) {
    // bf16 atomic-add into GM is only honoured on the A2/A3 store path
    // (pto-isa set_atomic_bf16); the A5 store path rejects it. Fail here with a
    // clean, backend-aware user error instead of deferring to a downstream
    // pto-isa static_assert. The hardware atomic dispatch keys on the GM
    // *destination* dtype, so this also guards the cube path (fp32 Acc -> bf16
    // GM via fix-pipe), where the source tile is fp32 but the target is bf16.
    if (tensor_type->dtype_ == DataType::BF16) {
      const auto* handler = codegen.GetBackendHandler();
      CHECK_SPAN(handler->SupportsBf16AtomicAdd(), op->span_)
          << "tile.store with atomic=AtomicType.Add into a bf16 global tensor is not supported on the '"
          << handler->GetPtoTargetArch()
          << "' backend; bf16 atomic-add requires the Ascend910B (A2/A3) profile. Accumulate into an fp32 "
             "tensor and cast to bf16 after the reduction instead.";
    }
    tstore_line << " {atomicType = #pto<atomic_type atomic_add>}";
  }
  codegen.Emit(tstore_line.str());

  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
    codegen.RegisterTensorView(result_var, tensor_view);
    codegen.RegisterVarToMlir(result_var, tensor_view);
    codegen.RegisterBasePtr(result_var, codegen.GetTensorBasePtr(output_tensor));
    // SSA-capture form ``data = pl.store(local, [0, 0], data)`` rebinds the
    // DistributedTensor LHS to a fresh Var; mirror the base-ptr alias so
    // ``pld.tile.remote_load`` etc. on the rebound name resolve the same
    // CommContext as the original source.
    codegen.RegisterCommCtxFor(result_var, codegen.GetCommCtxSSAFor(output_tensor.get()));
  }

  return "";
}

// tile.mscatter(src, idx, output_tensor) -> pto.mscatter
// Generates:
//   %pview = pto.partition_view %tensor_view, offsets=[0,...], sizes=[d0,...] : ... -> ...
//   pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
//                outs(%pview : !pto.partition_tensor_view<...>)
static std::string MakeTileMscatterCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  INTERNAL_CHECK(op->args_.size() == 3)
      << "tile.mscatter requires 3 arguments (src, idx, output_tensor), got " << op->args_.size();

  auto src = AsVarLike(op->args_[0]);
  INTERNAL_CHECK(src) << "tile.mscatter src must be a Var or IterArg";
  auto idx = AsVarLike(op->args_[1]);
  INTERNAL_CHECK(idx) << "tile.mscatter idx must be a Var or IterArg";
  auto output_tensor = AsVarLike(op->args_[2]);
  INTERNAL_CHECK(output_tensor) << "tile.mscatter output_tensor must be a Var or IterArg";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "tile.mscatter output_tensor must have TensorType";

  std::string src_name = codegen.GetVarName(src);
  std::string idx_name = codegen.GetVarName(idx);
  std::string src_type_annot = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string idx_type_annot = codegen.GetExprTypeAnnotation(op->args_[1]);

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());

  // Build pto.partition_view covering the entire tensor (mscatter uses per-element
  // indices, so the partition is the whole tensor — offsets all zero, sizes = shape).
  std::string partition_view = codegen.NewNamedTemp(output_tensor->name_hint_ + "_pview");
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [";
  for (size_t i = 0; i < tensor_type->shape_.size(); ++i) {
    if (i > 0) partition_line << ", ";
    partition_line << codegen.GetOrEmitConstant(static_cast<int64_t>(0), DataType::INDEX);
  }
  partition_line << "], sizes = [";
  std::string partition_type = "!pto.partition_tensor_view<";
  for (size_t i = 0; i < tensor_type->shape_.size(); ++i) {
    if (i > 0) {
      partition_line << ", ";
      partition_type += "x";
    }
    if (auto c = As<ir::ConstInt>(tensor_type->shape_[i])) {
      partition_line << codegen.GetOrEmitConstant(c->value_, DataType::INDEX);
      partition_type += std::to_string(c->value_);
    } else {
      partition_line << codegen.GetExprAsCode(tensor_type->shape_[i]);
      partition_type += "?";
    }
  }
  partition_line << "]";
  partition_type += "x" + dtype_str + ">";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  // Emit pto.mscatter with partition_view in outs()
  std::ostringstream mscatter_line;
  mscatter_line << "pto.mscatter ins(" << src_name << ", " << idx_name;
  if (!src_type_annot.empty() && !idx_type_annot.empty()) {
    mscatter_line << " : " << src_type_annot << ", " << idx_type_annot;
  }
  mscatter_line << ") outs(" << partition_view << " : " << partition_type << ")";
  codegen.Emit(mscatter_line.str());

  // Propagate tensor_view, base-ptr, and CommContext aliases to the result var
  // so downstream ops on an SSA-rebound DistributedTensor LHS still resolve.
  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
    codegen.RegisterTensorView(result_var, tensor_view);
    codegen.RegisterVarToMlir(result_var, tensor_view);
    codegen.RegisterBasePtr(result_var, codegen.GetTensorBasePtr(output_tensor));
    codegen.RegisterCommCtxFor(result_var, codegen.GetCommCtxSSAFor(output_tensor.get()));
  }

  return "";
}

// Helper function for tile.alloc (no-op: allocation handled elsewhere)
static std::string MakeTileAllocCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No MLIR emission - pto.alloc_tile generated from MemRefs in TileTypes
}

// Get or emit a flat offset SSA value for a MakeTuple of indices and shape.
static std::string GetFlatOffsetSSA(const ir::MakeTuplePtr& indices_tuple,
                                    const std::vector<ir::ExprPtr>& shape, codegen::PTOCodegen& codegen) {
  const auto& indices = indices_tuple->elements_;

  int64_t flat_offset = 0;
  bool all_constant = true;
  for (size_t i = 0; i < indices.size() && all_constant; ++i) {
    auto idx_val = As<ir::ConstInt>(indices[i]);
    if (!idx_val) {
      all_constant = false;
      break;
    }

    int64_t stride = 1;
    for (size_t j = i + 1; j < shape.size(); ++j) {
      auto dim_val = As<ir::ConstInt>(shape[j]);
      if (!dim_val) {
        all_constant = false;
        break;
      }
      stride *= dim_val->value_;
    }
    if (!all_constant) break;
    flat_offset += idx_val->value_ * stride;
  }

  if (all_constant) {
    return codegen.GetOrEmitConstant(flat_offset, DataType::INDEX);
  }

  std::vector<std::string> index_ssa;
  index_ssa.reserve(indices.size());
  for (const auto& index : indices) {
    if (auto c = As<ir::ConstInt>(index)) {
      index_ssa.push_back(codegen.GetOrEmitConstant(c->value_, DataType::INDEX));
      continue;
    }
    index_ssa.push_back(codegen.EmitCastToIndex(index, codegen.GetExprAsCode(index)));
  }
  return EmitFlatOffsetSSAFromValues(index_ssa, shape, codegen, "flat_offset");
}

// Helper function for tile.read (indices -> flat offset -> pto.tgetval)
static std::string MakeTileReadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 2) << "tile.read requires 2 arguments, but got " << op->args_.size();

  auto tile_type = As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tile_type, op->span_) << "tile.read first argument must be TileType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(indices_tuple, op->span_) << "tile.read second argument must be MakeTuple (indices)";

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string result = codegen.GetCurrentResultTarget();
  std::string scalar_type = codegen.GetTypeString(tile_type->dtype_);

  std::string off = GetFlatOffsetSSA(indices_tuple, tile_type->shape_, codegen);

  std::ostringstream oss;
  oss << result << " = pto.tgetval ins(" << src << ", " << off;
  if (!src_type.empty()) {
    oss << " : " << src_type << ", index";
  } else {
    oss << " : , index";
  }
  oss << ") outs : " << scalar_type;
  codegen.Emit(oss.str());
  return "";
}

// Helper function for tile.write (indices -> flat offset -> pto.tsetval)
static std::string MakeTileWriteCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 3) << "tile.write requires 3 arguments, but got " << op->args_.size();

  auto tile_type = As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tile_type, op->span_) << "tile.write first argument must be TileType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(indices_tuple, op->span_) << "tile.write second argument must be MakeTuple (indices)";

  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tile_type_str = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string value = codegen.GetExprAsCode(op->args_[2]);
  std::string value_type = codegen.GetExprTypeAnnotation(op->args_[2]);

  std::string off = GetFlatOffsetSSA(indices_tuple, tile_type->shape_, codegen);

  std::ostringstream oss;
  oss << "pto.tsetval ins(" << off << ", " << value;
  oss << " : index";
  if (!value_type.empty()) oss << ", " << value_type;
  oss << ") outs(" << tile;
  if (!tile_type_str.empty()) oss << " : " << tile_type_str;
  oss << ")";
  codegen.Emit(oss.str());

  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
    codegen.RegisterVarToMlir(result_var, tile);
  }
  return "";
}

static std::string MakeTensorReadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.read requires 2 arguments, but got " << op->args_.size();

  auto tensor_type_ptr = AsTensorTypeLike(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tensor_type_ptr, op->span_) << "tensor.read first argument must be TensorType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(indices_tuple, op->span_) << "tensor.read second argument must be MakeTuple (indices)";

  auto scalar_type_ptr = As<ir::ScalarType>(op->GetType());
  INTERNAL_CHECK_SPAN(scalar_type_ptr, op->span_) << "tensor.read result must be ScalarType";
  std::string scalar_type = codegen.GetTypeString(scalar_type_ptr->dtype_);

  // store_scalar/load_scalar need the base !pto.ptr; resolve via the tensor var
  // even after a slice-assign rebound it to a tensor_view (issue #1493).
  std::string src = codegen.GetTensorBasePtr(AsVarLike(op->args_[0]));
  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string result = codegen.GetCurrentResultTarget();

  if (src_type.empty()) {
    src_type = "!pto.ptr<" + codegen.GetTypeString(tensor_type_ptr->dtype_) + ">";
  }

  std::string off = GetFlatOffsetSSA(indices_tuple, tensor_type_ptr->shape_, codegen);

  std::ostringstream oss;
  oss << result << " = pto.load_scalar " << src << "[" << off << "]";
  if (!src_type.empty()) {
    oss << " : " << src_type;
  }
  oss << " -> " << scalar_type;
  codegen.Emit(oss.str());
  return "";
}

static std::string MakeTensorWriteCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 3) << "tensor.write requires 3 arguments, but got " << op->args_.size();

  auto tensor_type_ptr = AsTensorTypeLike(op->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(tensor_type_ptr, op->span_) << "tensor.write first argument must be TensorType";

  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(indices_tuple, op->span_) << "tensor.write second argument must be MakeTuple (indices)";

  // store_scalar needs the base !pto.ptr; resolve via the tensor var even after
  // a prior slice-assign rebound it to a tensor_view (issue #1493).
  std::string tensor = codegen.GetTensorBasePtr(AsVarLike(op->args_[0]));
  std::string tensor_type_str = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string value = codegen.GetExprAsCode(op->args_[2]);
  std::string value_type = codegen.GetExprTypeAnnotation(op->args_[2]);

  if (tensor_type_str.empty()) {
    tensor_type_str = "!pto.ptr<" + codegen.GetTypeString(tensor_type_ptr->dtype_) + ">";
  }

  std::string off = GetFlatOffsetSSA(indices_tuple, tensor_type_ptr->shape_, codegen);

  std::ostringstream oss;
  oss << "pto.store_scalar " << value << ", " << tensor << "[" << off << "]";
  if (!tensor_type_str.empty() || !value_type.empty()) {
    oss << " : ";
    if (!tensor_type_str.empty()) oss << tensor_type_str;
    if (!tensor_type_str.empty() && !value_type.empty()) oss << ", ";
    if (!value_type.empty()) oss << value_type;
  }
  codegen.Emit(oss.str());

  auto result_var = codegen.GetCurrentResultVar();
  if (result_var != nullptr) {
    codegen.RegisterTensorView(result_var, tensor);
    codegen.RegisterVarToMlir(result_var, tensor);
    codegen.RegisterBasePtr(result_var, tensor);
    codegen.RegisterCommCtxFor(result_var, codegen.GetCommCtxSSAFor(AsVarLike(op->args_[0]).get()));
  }
  return "";
}

static std::string MakeTensorDimCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.dim requires 2 arguments, but got " << op->args_.size();
  auto input_tensor = ir::As<ir::TensorType>(op->args_[0]->GetType());
  CHECK(input_tensor) << "tensor.dim need TensorType for first arg, but got "
                      << op->args_[0]->GetType()->TypeName();
  auto axis = codegen.GetConstIntValue(op->args_[1]);
  CHECK(axis >= 0 && static_cast<size_t>(axis) < input_tensor->shape_.size())
      << "tensor.dim axis " << axis << " out of range for tensor with rank " << input_tensor->shape_.size();
  auto shape = input_tensor->shape_[axis];
  std::string shape_name;
  if (auto dyn_shape = ir::As<ir::Var>(shape)) {
    shape_name = codegen.GetVarName(dyn_shape);
  } else if (auto static_shape = ir::As<ir::ConstInt>(shape)) {
    shape_name = codegen.GetOrEmitConstant(static_shape->value_, DataType::INDEX);
  } else {
    INTERNAL_CHECK_SPAN(false, op->span_) << "Internal error: tensor.dim shape is neither Var nor ConstInt";
  }
  auto target_var = codegen.GetCurrentResultVar();
  if (target_var != nullptr && !shape_name.empty()) {
    codegen.RegisterVarToMlir(target_var, shape_name);
  }

  return "";
}

// Emit a value SSA whose MLIR type matches the array's element dtype. The IR's
// array.update_element verifier permits an `index`-typed value into an integer
// array (and vice-versa), so the C++ orchestration path relies on implicit
// conversion. PTO/MLIR is strictly typed, so any dtype mismatch is bridged with
// an explicit arith cast here (index_cast for index<->int, trunci/extsi/extui
// for int width changes).
static std::string EmitLocalArrayValue(codegen::PTOCodegen& codegen, const ir::ExprPtr& value,
                                       DataType target) {
  std::string ssa = codegen.GetExprAsCode(value);
  auto value_type = ir::As<ScalarType>(value->GetType());
  if (!value_type || value_type->dtype_ == target) {
    return ssa;
  }
  DataType src = value_type->dtype_;
  std::string mlir_op;
  if (src == DataType::INDEX || target == DataType::INDEX) {
    mlir_op = "arith.index_cast";
  } else if (src.GetBit() > target.GetBit()) {
    mlir_op = "arith.trunci";
  } else if (src.GetBit() < target.GetBit()) {
    mlir_op = src.IsUnsignedInt() ? "arith.extui" : "arith.extsi";
  } else {
    // Same bit width but distinct dtype (signed vs unsigned, e.g. i32 vs ui32):
    // no arith width/index cast applies, yet the operand type must still match
    // the element dtype. Bridge with the MLIR escape-hatch cast. Unreachable for
    // verifier-valid IR (array.update_element requires equal dtypes for
    // non-index values), but keeps the operand well-typed rather than silently
    // emitting a mistyped value.
    mlir_op = "builtin.unrealized_conversion_cast";
  }
  std::string out = codegen.NewTemp();
  codegen.Emit(out + " = " + mlir_op + " " + ssa + " : " + codegen.GetTypeString(src) + " to " +
               codegen.GetTypeString(target));
  return out;
}

void RegisterMemoryOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  // Register ops with custom codegen logic
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  // On-core arrays (ArrayType) -> PTOAS stack-local arrays. The IR's
  // SSA-functional update_element semantics are realized in place: PTOCodegen's
  // AssignStmt dispatch aliases an array.update_element result Var to the input
  // array's SSA name BEFORE invoking the codegen below, so the emitted
  // pto.local_array_set mutates the same `pto.declare_local_array` storage.
  reg("array.create", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 1) << "array.create requires 1 argument (extent)";
    auto array_type = ir::As<ir::ArrayType>(op->GetType());
    CHECK(array_type) << "array.create must return ArrayType";
    std::string result = codegen.GetCurrentResultTarget();
    INTERNAL_CHECK_SPAN(!result.empty(), op->span_) << "array.create requires an assignment target";
    codegen.Emit(result + " = pto.declare_local_array -> " +
                 codegen::FormatLocalArrayTypeString(*array_type));
    return std::string("");
  });

  reg("array.get_element", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 2) << "array.get_element requires 2 arguments (array, index)";
    auto array_type = ir::As<ir::ArrayType>(op->args_[0]->GetType());
    CHECK(array_type) << "array.get_element first argument must be an ArrayType";
    std::string result = codegen.GetCurrentResultTarget();
    INTERNAL_CHECK_SPAN(!result.empty(), op->span_) << "array.get_element requires an assignment target";
    std::string arr = codegen.GetExprAsCode(op->args_[0]);
    std::string idx = EmitIndexOperand(codegen, op->args_[1], "array.get_element index");
    codegen.Emit(result + " = pto.local_array_get " + arr + "[" + idx +
                 "] : " + codegen::FormatLocalArrayTypeString(*array_type) + " -> " +
                 codegen::DataTypeToMLIR(array_type->dtype_));
    return std::string("");
  });

  reg("array.update_element", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 3) << "array.update_element requires 3 arguments (array, index, value)";
    auto array_type = ir::As<ir::ArrayType>(op->args_[0]->GetType());
    CHECK(array_type) << "array.update_element first argument must be an ArrayType";
    // arr resolves to the input array's SSA; the AssignStmt dispatch has already
    // aliased the result Var to this name, so the write is in place.
    std::string arr = codegen.GetExprAsCode(op->args_[0]);
    std::string idx = EmitIndexOperand(codegen, op->args_[1], "array.update_element index");
    std::string value = EmitLocalArrayValue(codegen, op->args_[2], array_type->dtype_);
    codegen.Emit("pto.local_array_set " + arr + "[" + idx + "], " + value + " : " +
                 codegen::FormatLocalArrayTypeString(*array_type) + ", " +
                 codegen::DataTypeToMLIR(array_type->dtype_));
    return std::string("");
  });

  // SPMD identity ops read from synthetic i32 params that PTOCodegen appends to
  // the func.func signature whenever the function body contains
  // tile.get_block_idx / tile.get_block_num / tile.get_subblock_idx. The kernel
  // wrapper resolves the runtime values from intrinsic.h::get_block_idx(args) /
  // get_block_num(args) / get_sub_block_id(args) and forwards them as the
  // trailing call args (canonical order: block_idx, block_num, subblock_idx).
  // subblock_idx deliberately reads the runtime lane id rather than the ccec
  // get_subblockid() register, which returns a stale value under the
  // tensormap_and_ringbuffer dispatch (see intrinsic.h).
  auto reg_spmd_identity_op = [&](const char* tile_op, std::string (codegen::PTOCodegen::*getter)() const) {
    reg(tile_op, [tile_op, getter](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = AsPto(codegen_base);
      CHECK(op->args_.empty()) << tile_op << " takes no arguments, got " << op->args_.size();
      std::string result = codegen.GetCurrentResultTarget();
      INTERNAL_CHECK_SPAN(!result.empty(), op->span_) << tile_op << " requires assignment target";
      std::string arg_ssa = (codegen.*getter)();
      INTERNAL_CHECK_SPAN(!arg_ssa.empty(), op->span_)
          << tile_op << " requires PTOCodegen SPMD signature params to be initialised";
      codegen.Emit(result + " = arith.index_cast " + arg_ssa + " : i32 to index");
      return std::string("");
    });
  };
  reg_spmd_identity_op("tile.get_block_idx", &codegen::PTOCodegen::GetSpmdBlockIdxArgSSA);
  reg_spmd_identity_op("tile.get_block_num", &codegen::PTOCodegen::GetSpmdBlockNumArgSSA);
  reg_spmd_identity_op("tile.get_subblock_idx", &codegen::PTOCodegen::GetSpmdSubblockIdxArgSSA);

  reg("tile.read", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileReadCodegenPTO(op, codegen);
  });
  reg("tile.write", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileWriteCodegenPTO(op, codegen);
  });
  reg("tensor.read", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTensorReadCodegenPTO(op, codegen);
  });
  reg("tensor.write", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTensorWriteCodegenPTO(op, codegen);
  });

  // ``tensor.as_layout`` (RFC #1300 §3.3): pure metadata reinterpret over the
  // same physical buffer. A compiler pass may prepend ``b_dn =
  // tensor.as_layout(b, DN)`` at the top of an InCore body to bridge an
  // ND ↔ DN view (``b_dn`` carries TensorType ``[..., b, a] DN`` with explicit
  // canonical strides) and rewrite the body to use ``b_dn`` in place of ``b``.
  //
  // Codegen lowers this to a fresh ``pto.make_tensor_view`` bound to the
  // input's underlying buffer (the function parameter SSA), using the LHS's
  // own ``(shape, stride, layout)`` from its TensorType. Downstream
  // ``tile.load`` lookups via ``GetOrCreateTensorView`` find the LHS through
  // the ``RegisterTensorView`` call below.
  reg("tensor.as_layout", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    auto& codegen = AsPto(codegen_base);
    CHECK(op->args_.size() == 1) << "tensor.as_layout requires 1 arg (input)";
    auto input_var = AsVarLike(op->args_[0]);
    CHECK(input_var) << "tensor.as_layout input must be a Var/IterArg";

    auto lhs_var = codegen.GetCurrentResultVar();
    INTERNAL_CHECK_SPAN(static_cast<bool>(lhs_var), op->span_)
        << "Internal error: tensor.as_layout result var must be set by VisitStmt_(AssignStmt)";
    auto lhs_type = As<ir::TensorType>(lhs_var->GetType());
    CHECK(lhs_type) << "tensor.as_layout output must be TensorType, got " << lhs_var->GetType()->TypeName();
    INTERNAL_CHECK_SPAN(lhs_type->tensor_view_.has_value(), op->span_)
        << "Internal error: tensor.as_layout output must have an explicit TensorView "
           "(set by DeduceTensorAsLayoutType + CanonicalizeView)";

    const size_t rank = lhs_type->shape_.size();
    const auto& view = lhs_type->tensor_view_.value();
    INTERNAL_CHECK_SPAN(view.stride.size() == rank, op->span_)
        << "Internal error: tensor.as_layout output stride rank " << view.stride.size()
        << " does not match shape rank " << rank;

    // The result SSA name (auto-allocated by VisitStmt_(AssignStmt) for the
    // backend-dispatched RHS Call) doubles as the tensor_view SSA name —
    // register it in tensor_to_view so downstream tile.load lookups resolve.
    std::string result_buf = codegen.GetCurrentResultTarget();
    INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_) << "Internal error: result buf must be set";
    codegen.RegisterTensorView(lhs_var, result_buf);

    // Materialize shape and stride SSA names.
    auto emit_dim = [&](const ir::ExprPtr& dim) -> std::string {
      if (auto c = As<ir::ConstInt>(dim)) {
        return codegen.GetOrEmitConstant(c->value_, DataType::INDEX);
      }
      return codegen.EmitCastToIndex(dim, codegen.GetExprAsCode(dim));
    };
    std::vector<std::string> shape_dim_names(rank);
    for (size_t j = 0; j < rank; ++j) shape_dim_names[j] = emit_dim(lhs_type->shape_[j]);
    std::vector<std::string> stride_names(rank);
    for (size_t j = 0; j < rank; ++j) stride_names[j] = emit_dim(view.stride[j]);

    std::string layout_str = "nd";
    switch (view.layout) {
      case ir::TensorLayout::DN:
        layout_str = "dn";
        break;
      case ir::TensorLayout::NZ:
        layout_str = "nz";
        break;
      case ir::TensorLayout::ND:
        break;
    }

    std::ostringstream oss;
    oss << result_buf << " = pto.make_tensor_view " << codegen.GetVarName(input_var) << ", shape = [";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) oss << ", ";
      oss << shape_dim_names[j];
    }
    oss << "], strides = [";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) oss << ", ";
      oss << stride_names[j];
    }
    oss << "] {layout = #pto.layout<" << layout_str << ">}";
    oss << ": !pto.tensor_view<";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) oss << "x";
      oss << "?";
    }
    oss << "x" << codegen.GetTypeString(lhs_type->dtype_) << ">";
    return oss.str();
  });

  reg("tile.load", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileLoadCodegenPTO(op, codegen);
  });
  reg("tile.store", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileStoreCodegenPTO(op, codegen);
  });

  // tile.mscatter: src and idx must be row_major (MTE3 DMA reads UB linearly)
  if (exclude_ops.count("tile.mscatter") == 0) {
    backend.RegisterOp("tile.mscatter")
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeTileMscatterCodegenPTO(op, codegen);
        })
        .set_input_layout(0, ir::TileLayout::row_major)
        .set_input_layout(1, ir::TileLayout::row_major);
  }

  reg("tile.alloc", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTileAllocCodegenPTO(op, codegen);
  });

  reg("tile.create", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
    (void)op;
    (void)codegen_base;
    return std::string("");  // No MLIR emission - tile allocation handled by pto.alloc_tile
  });

  reg("tile.store_fp", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeStoreFPCodegenPTO("pto.tstore.fp", op, codegen);
  });

  reg("tensor.dim", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeTensorDimCodegenPTO(op, codegen);
  });
}
}  // namespace backend
}  // namespace pypto
