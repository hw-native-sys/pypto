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
 * @file pto_ops_distributed.cpp
 * @brief PTO codegen registration for distributed (pld.*) ops.
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
using pto_ops_detail::EmitPartitionViewPTO;
using pto_ops_detail::GetDimStrings;
using pto_ops_detail::GetIndexOffsetCodes;
using pto_ops_detail::GetSizeCodes;
using pto_ops_detail::MakePartitionTensorViewType;

// ============================================================================
// Distributed N6: pld.tile.remote_load / pld.system.notify / pld.system.wait
// ============================================================================

namespace {

// Resolve a DistributedTensor argument to its parameter Var + matching
// CommContext SSA. The argument is expected to be a Var directly bound to a
// function parameter (no aliasing); the verifier on remote_load / notify /
// wait already requires DistributedTensorType, but additionally checks that
// the var has a CommContext ptr threaded through the func.func signature.
struct DistTensorBinding {
  ir::VarPtr var;
  std::shared_ptr<const ir::DistributedTensorType> type;
  std::string local_ptr_ssa;
  std::string ctx_ssa;
};

DistTensorBinding ResolveDistTensorBinding(const ExprPtr& arg, codegen::PTOCodegen& codegen,
                                           const char* op_name) {
  auto var = AsVarLike(arg);
  CHECK(var) << op_name << " expects DistributedTensor argument to be a Var-like expression, got "
             << arg->TypeName();
  auto dist_type = As<ir::DistributedTensorType>(var->GetType());
  CHECK(dist_type) << op_name << " expects DistributedTensorType, got " << var->GetType()->TypeName();
  // Resolve via the base-ptr alias mechanism rather than the raw ``var_to_mlir``
  // binding. For a function parameter both return ``%argN`` (the ``!pto.ptr``);
  // for an SSA-rebound Var (``data = pl.store(local, [0, 0], data)``) the
  // store codegen rewrites ``var_to_mlir`` to the tensor-view alias but
  // propagates the underlying base pointer via ``RegisterBasePtr`` —
  // ``GetTensorBasePtr`` follows that alias and returns the original
  // ``!pto.ptr`` SSA, which is what ``pto.addptr`` expects below.
  std::string local_ptr = codegen.GetTensorBasePtr(var);
  std::string ctx_ssa = codegen.GetCommCtxSSAFor(var.get());
  CHECK(!ctx_ssa.empty()) << op_name << " requires a CommContext pointer arg threaded for DistributedTensor '"
                          << var->name_hint_ << "', but none was found in the function signature";
  return {var, dist_type, std::move(local_ptr), std::move(ctx_ssa)};
}

// Emit:
//   (1) a single ``func.call`` to the per-dtype module-level
//       ``@CommRemoteOffset_<dtype>`` helper (see
//       ``PTOCodegen::EmitCommRemoteOffsetHelpers``) — returns the
//       peer-vs-local **element offset** (``index``);
//   (2) a ``pto.addptr`` against the local DistributedTensor pointer, and
//   (3) a ``pto.make_tensor_view`` rooted at the resulting peer pointer.
//
// Steps (2) and (3) live at the call site (i.e. inside the user kernel's
// ``func.func``) for two intertwined PTOAS constraints:
//
// * ``pto.addptr`` must feed ``pto.make_tensor_view`` /
//   ``initialize_l2g2l_pipe(gm_addr)`` / ``load|store_scalar`` *within
//   the same func.func*. A helper that ended with ``addptr → return``
//   would only feed ``func.return``, which PTOAS rejects.
// * ``pto.make_tensor_view`` always lowers to ``memref<…, strided<[?,
//   ?], offset: ?>>`` when strides are passed as operands, but
//   ``!pto.tensor_view<…>`` source syntax cannot carry a strided layout
//   suffix — so the view cannot be returned across a func boundary
//   either.
//
// Both forbidden ops therefore have to live in the user kernel. The
// helper still pulls its weight: it bundles the CommContext field reads
// and the byte→element division (which depends on dtype), so multiple
// remote ops share that work via ``func.call`` without duplicating the
// scalar arithmetic at each call site.
//
// Generated MLIR (2-D example, ``DistributedTensor[[1, 64], FP32]``):
//
//   %peer_idx = arith.index_cast %peer : i32 to index
//   %delems = func.call @CommRemoteOffset_f32(%ctx, %peer_idx)
//           : (!pto.ptr<i64>, index) -> index
//   %peer_ptr = pto.addptr %local_ptr, %delems
//             : !pto.ptr<f32> -> !pto.ptr<f32>
//   %peer_view = pto.make_tensor_view %peer_ptr,
//                   shape = [%c1, %c64], strides = [%c64, %c1]
//                   {layout = #pto.layout<nd>}
//                   : !pto.tensor_view<?x?xf32>
struct PeerViewInfo {
  std::string ssa;
  std::string view_type_str;
};

PeerViewInfo EmitCommRemoteView(const DistTensorBinding& target, const ExprPtr& peer_expr,
                                codegen::PTOCodegen& codegen) {
  const auto& shape = target.type->shape_;
  const size_t rank = shape.size();
  CHECK(rank >= 1) << "DistributedTensor must have rank >= 1 for peer view emission";
  const std::string dtype_str = codegen.GetTypeString(target.type->dtype_);
  const std::string ptr_type = "!pto.ptr<" + dtype_str + ">";

  // Peer rank may be any scalar int; the helper takes it as ``index``, so
  // normalise here. Constants and i32/i64 values flow through
  // EmitCastToIndex (no-op when already index-typed).
  std::string peer_ssa = codegen.EmitCastToIndex(peer_expr, codegen.GetExprAsCode(peer_expr));

  // (1) Call the per-dtype offset helper. Registering here causes the helper
  //     definition to be emitted at module-flush time — any new op that calls
  //     EmitCommRemoteView is wired up automatically, no codegen-side opt-in.
  const std::string func_name = codegen.RegisterCommRemoteOffsetHelper(target.type->dtype_);
  std::string delems = codegen.NewTemp();
  codegen.Emit(delems + " = func.call @" + func_name + "(" + target.ctx_ssa + ", " + peer_ssa +
               ") : (!pto.ptr<i64>, index) -> index");

  // (2) addptr from the local pointer by the returned element offset.
  std::string peer_ptr = codegen.NewTemp();
  codegen.Emit(peer_ptr + " = pto.addptr " + target.local_ptr_ssa + ", " + delems + " : " + ptr_type +
               " -> " + ptr_type);

  // (3) make_tensor_view at the call site. Same shape/stride emission as
  // ``EmitMakeTensorViews``: row-major strides, ``{layout = #pto.layout<nd>}``
  // attribute, dynamic-shape result type (``?x?x…xT``). ``addptr``'s
  // direct consumer is this ``make_tensor_view`` in the same func →
  // PTOAS's per-func lowering rule is satisfied.
  std::vector<std::string> shape_ssa(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (auto ci = As<ir::ConstInt>(shape[i])) {
      shape_ssa[i] = codegen.GetOrEmitConstant(ci->value_, DataType::INDEX);
    } else {
      shape_ssa[i] = codegen.EmitCastToIndex(shape[i], codegen.GetExprAsCode(shape[i]));
    }
  }
  std::vector<std::string> stride_ssa(rank);
  stride_ssa[rank - 1] = codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
  for (size_t j = rank - 1; j > 0; --j) {
    std::string mul = codegen.NewTemp();
    codegen.Emit(mul + " = arith.muli " + stride_ssa[j] + ", " + shape_ssa[j] + " : index");
    stride_ssa[j - 1] = mul;
  }

  std::string peer_view = codegen.NewTemp();
  std::ostringstream view_type;
  view_type << "!pto.tensor_view<";
  for (size_t i = 0; i < rank; ++i) {
    if (i > 0) view_type << "x";
    view_type << "?";
  }
  view_type << "x" << dtype_str << ">";

  std::ostringstream mv;
  mv << peer_view << " = pto.make_tensor_view " << peer_ptr << ", shape = [";
  for (size_t i = 0; i < rank; ++i) {
    if (i > 0) mv << ", ";
    mv << shape_ssa[i];
  }
  mv << "], strides = [";
  for (size_t i = 0; i < rank; ++i) {
    if (i > 0) mv << ", ";
    mv << stride_ssa[i];
  }
  mv << "] {layout = #pto.layout<nd>} : " << view_type.str();
  codegen.Emit(mv.str());

  return {peer_view, view_type.str()};
}

}  // namespace

// pld.tile.remote_load(target, peer, offsets, shape) — load a peer's slice of
// a window-bound DistributedTensor into a local tile. Lowers to:
//   delems = func.call @CommRemoteOffset_<dtype>(ctx, peer) : ... -> index
//   peer_ptr = pto.addptr local_ptr, delems
//   peer_view = pto.make_tensor_view peer_ptr, shape=..., strides=...
//   pto.partition_view peer_view, offsets=..., sizes=<shape>
//   pto.tload ins(<pview>) outs(<tile>)
static std::string MakeRemoteLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 4) << "pld.tile.remote_load requires 4 arguments (target, peer, offsets, "
                                  "shape), got "
                               << op->args_.size();

  auto binding = ResolveDistTensorBinding(op->args_[0], codegen, "pld.tile.remote_load");
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "pld.tile.remote_load offsets must be MakeTuple";
  auto shapes_tuple = As<ir::MakeTuple>(op->args_[3]);
  INTERNAL_CHECK_SPAN(shapes_tuple, op->span_) << "pld.tile.remote_load shape must be MakeTuple";

  auto peer_view = EmitCommRemoteView(binding, op->args_[1], codegen);

  const std::string dtype_str = codegen.GetTypeString(binding.type->dtype_);
  const auto& shape_elems = shapes_tuple->elements_;
  std::string partition_type = MakePartitionTensorViewType(GetDimStrings(shape_elems), dtype_str);
  std::string partition_view = EmitPartitionViewPTO(
      binding.var->name_hint_ + "_peer", peer_view.ssa, peer_view.view_type_str, partition_type,
      GetIndexOffsetCodes(offsets_tuple->elements_, codegen), GetSizeCodes(shape_elems, codegen), codegen);

  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK_SPAN(!tile_buf.empty(), op->span_)
      << "pld.tile.remote_load requires assignment target (tile_buf)";
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();

  std::ostringstream tload;
  tload << "pto.tload ins(" << partition_view << " : " << partition_type << ") outs(" << tile_buf << " : "
        << tile_buf_type << ")";
  codegen.Emit(tload.str());
  return "";
}

// pld.tile.remote_store(src_tile, target, peer, offsets) — write a local tile
// into a peer's slice of a window-bound DistributedTensor. Lowers to:
//   delems    = func.call @CommRemoteOffset_<dtype>(ctx, peer) : ... -> index
//   peer_ptr  = pto.addptr local_ptr, delems
//   peer_view = pto.make_tensor_view peer_ptr, shape=..., strides=...
//   pto.partition_view peer_view, offsets=..., sizes=<tile.valid_shape padded
//                                                     with leading 1s>
//   pto.tstore ins(<tile>) outs(<pview>)
//
// The tile's valid_shape is 2-D (height, width); when target_rank > 2 the
// leading (target_rank - 2) partition dims are size-1 — matching the
// notify codegen's one_dims(rank, "1") pattern — so a 2-D tile push lands
// on the inner two dims of an N-D peer slice without forcing the caller to
// reshape.
static std::string MakeRemoteStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 4)
      << "pld.tile.remote_store requires 4 arguments (src_tile, target, peer, offsets), got "
      << op->args_.size();

  auto src_tile = AsVarLike(op->args_[0]);
  INTERNAL_CHECK_SPAN(src_tile, op->span_) << "pld.tile.remote_store src_tile must be a Var or IterArg";
  auto tile_type = As<ir::TileType>(src_tile->GetType());
  INTERNAL_CHECK_SPAN(tile_type, op->span_) << "pld.tile.remote_store src_tile must have TileType";

  auto binding = ResolveDistTensorBinding(op->args_[1], codegen, "pld.tile.remote_store");
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[3]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "pld.tile.remote_store offsets must be MakeTuple";

  auto peer_view = EmitCommRemoteView(binding, op->args_[2], codegen);
  const std::string dtype_str = codegen.GetTypeString(binding.type->dtype_);

  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*tile_type);
  const auto& valid_shape = tile_view.valid_shape;
  INTERNAL_CHECK_SPAN(valid_shape.size() == 2, op->span_)
      << "pld.tile.remote_store tile valid_shape must be 2D";
  const size_t target_rank = binding.type->shape_.size();
  INTERNAL_CHECK_SPAN(target_rank >= 2, op->span_)
      << "pld.tile.remote_store target rank must be >= 2 to hold a 2-D tile";

  std::vector<std::string> dim_strs(target_rank - 2, "1");
  std::vector<std::string> size_codes(target_rank - 2,
                                      codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX));
  auto append_dim = [&](const ir::ExprPtr& expr) {
    if (auto c = As<ir::ConstInt>(expr)) {
      dim_strs.push_back(std::to_string(c->value_));
    } else {
      dim_strs.emplace_back("?");
    }
    size_codes.push_back(codegen.GetExprAsCode(expr));
  };
  append_dim(valid_shape[0]);
  append_dim(valid_shape[1]);
  const std::string partition_type = MakePartitionTensorViewType(dim_strs, dtype_str);

  std::string partition_view = EmitPartitionViewPTO(
      binding.var->name_hint_ + "_peer", peer_view.ssa, peer_view.view_type_str, partition_type,
      GetIndexOffsetCodes(offsets_tuple->elements_, codegen), size_codes, codegen);

  std::string tile_buf = codegen.GetVarName(src_tile);
  std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  if (!tile_buf_type.empty()) {
    tstore_line << " : " << tile_buf_type;
  }
  tstore_line << ") outs(" << partition_view << " : " << partition_type << ")";
  codegen.Emit(tstore_line.str());
  return "";
}

// pld.system.notify(target, peer, offsets, value, *, op) — atomically signal a
// peer rank's slot in a DistributedTensor signal matrix.
//   delems = func.call @CommRemoteOffset_<dtype>(ctx, peer) : ... -> index
//   peer_ptr = pto.addptr local_ptr, delems
//   peer_view = pto.make_tensor_view peer_ptr, shape=..., strides=...
//   pto.partition_view peer_view, sizes=[1, ..., 1]
//   pto.comm.tnotify(<pview>, <value>) {notifyOp = #pto<notify_op (set|atomic_add)>}
static std::string MakeNotifyCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 4) << "pld.system.notify requires 4 arguments (target, peer, offsets, "
                                  "value), got "
                               << op->args_.size();

  auto binding = ResolveDistTensorBinding(op->args_[0], codegen, "pld.system.notify");
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "pld.system.notify offsets must be MakeTuple";

  const int notify_op_int = op->GetKwarg<int>("op", 0);
  CHECK(notify_op_int == static_cast<int>(ir::NotifyOp::kAtomicAdd) ||
        notify_op_int == static_cast<int>(ir::NotifyOp::kSet))
      << "pld.system.notify op kwarg must encode NotifyOp::kAtomicAdd or kSet, got " << notify_op_int;
  const std::string notify_attr =
      notify_op_int == static_cast<int>(ir::NotifyOp::kAtomicAdd) ? "atomic_add" : "set";

  auto peer_view = EmitCommRemoteView(binding, op->args_[1], codegen);

  // Notify slot is a single signal cell — partition_view sizes are 1 per dim.
  const std::string dtype_str = codegen.GetTypeString(binding.type->dtype_);
  const size_t rank = binding.type->shape_.size();
  std::vector<std::string> one_dims(rank, "1");
  std::vector<std::string> one_size_ssa(rank,
                                        codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX));
  std::string partition_type = MakePartitionTensorViewType(one_dims, dtype_str);
  std::string partition_view = EmitPartitionViewPTO(
      binding.var->name_hint_ + "_peer", peer_view.ssa, peer_view.view_type_str, partition_type,
      GetIndexOffsetCodes(offsets_tuple->elements_, codegen), one_size_ssa, codegen);

  // PTOAS contract: tnotify value's MLIR type must match the signal element
  // type. Emit using the value's own ScalarType — mismatched IR-level dtypes
  // surface here as a PTOAS verifier diagnostic rather than as silently
  // garbled DMA.
  std::string value_ssa = codegen.GetExprAsCode(op->args_[3]);
  auto value_scalar = As<ir::ScalarType>(op->args_[3]->GetType());
  CHECK(value_scalar) << "pld.system.notify value must have ScalarType, got "
                      << op->args_[3]->GetType()->TypeName();
  std::string value_type = codegen.GetTypeString(value_scalar->dtype_);
  std::ostringstream tnotify;
  tnotify << "pto.comm.tnotify(" << partition_view << ", " << value_ssa << " : " << partition_type << ", "
          << value_type << ") {notifyOp = #pto<notify_op " << notify_attr << ">}";
  codegen.Emit(tnotify.str());
  return "";
}

// pld.system.wait(signal, offsets, expected, *, cmp) — block until local signal
// slot satisfies cmp against expected. wait is local (no peer arithmetic):
//   pto.partition_view <local_view>, offsets=..., sizes=[1,..,1]
//   pto.comm.twait(<pview>, <expected>) {cmp = #pto<wait_cmp (eq|ge)>}
static std::string MakeWaitCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  CHECK(op->args_.size() == 3) << "pld.system.wait requires 3 arguments (signal, offsets, expected), got "
                               << op->args_.size();

  auto signal_var = AsVarLike(op->args_[0]);
  CHECK(signal_var) << "pld.system.wait signal must be a Var-like expression";
  auto dist_type = As<ir::DistributedTensorType>(signal_var->GetType());
  CHECK(dist_type) << "pld.system.wait signal must be DistributedTensorType, got "
                   << signal_var->GetType()->TypeName();

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK_SPAN(offsets_tuple, op->span_) << "pld.system.wait offsets must be MakeTuple";

  const int cmp_int = op->GetKwarg<int>("cmp", 0);
  CHECK(cmp_int == static_cast<int>(ir::WaitCmp::kEq) || cmp_int == static_cast<int>(ir::WaitCmp::kGe))
      << "pld.system.wait cmp kwarg must encode WaitCmp::kEq or kGe, got " << cmp_int;
  const std::string cmp_attr = cmp_int == static_cast<int>(ir::WaitCmp::kEq) ? "eq" : "ge";

  // Reuse the local tensor_view created by EmitMakeTensorViews — wait only
  // touches the local signal slot. The view's MLIR type must match the
  // emit-time form (all dims printed as ``?``), not the IR-level concrete
  // shape, otherwise the SSA value picks up two incompatible types when other
  // uses (tile.load, etc.) reference the same view. Mirrors tile.load at
  // line 1192 above.
  std::string local_view = codegen.GetOrCreateTensorView(signal_var);
  std::string local_view_type = codegen.GetTensorViewTypeString(dist_type.get());
  const std::string dtype_str = codegen.GetTypeString(dist_type->dtype_);
  const size_t rank = dist_type->shape_.size();

  std::vector<std::string> one_dims(rank, "1");
  std::vector<std::string> one_size_ssa(rank,
                                        codegen.GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX));
  std::string partition_type = MakePartitionTensorViewType(one_dims, dtype_str);
  std::string partition_view =
      EmitPartitionViewPTO(signal_var->name_hint_ + "_local", local_view, local_view_type, partition_type,
                           GetIndexOffsetCodes(offsets_tuple->elements_, codegen), one_size_ssa, codegen);

  // PTOAS contract: twait expected value's MLIR type must match the signal
  // element type. Emit using the expected value's own ScalarType — see notify
  // codegen above for the rationale.
  std::string expected_ssa = codegen.GetExprAsCode(op->args_[2]);
  auto expected_scalar = As<ir::ScalarType>(op->args_[2]->GetType());
  CHECK(expected_scalar) << "pld.system.wait expected must have ScalarType, got "
                         << op->args_[2]->GetType()->TypeName();
  std::string expected_type = codegen.GetTypeString(expected_scalar->dtype_);
  std::ostringstream twait;
  twait << "pto.comm.twait(" << partition_view << ", " << expected_ssa << " : " << partition_type << ", "
        << expected_type << ") {cmp = #pto<wait_cmp " << cmp_attr << ">}";
  codegen.Emit(twait.str());
  return "";
}

// pld.tile.put(dst, peer, src, stage[, dst_offsets, src_offsets, shape],
//              *, atomic) - synchronous cross-rank bulk write of the local
// slice `src` into the peer rank's slice of `dst`. `stage` is a VEC scratch
// TileType pre-allocated by an IR-level `tile.create` (so the memory allocator
// gives it a UB address before codegen at --pto-level=level3).
// Lowers to:
//   delems   = func.call @CommRemoteOffset_<dtype>(ctx, peer) : ... -> index
//   dst_ptr  = pto.addptr <dst_local_ptr>, delems
//   dst_view = pto.make_tensor_view dst_ptr, shape=..., strides=...
//   dst_pv   = pto.partition_view dst_view, offsets=<dst_offsets>, sizes=<transfer_shape>
//   src_pv   = pto.partition_view <src_local_view>, offsets=<src_offsets>, sizes=<transfer_shape>
//   pto.comm.tput(dst_pv, src_pv, buf(%stage)
//       : <ptype>, <ptype>, <stage_type>) {atomicType = #pto<atomic_type (atomic_none|atomic_add)>}
//
// Full-slice tile.put (4 args) uses zero offsets and the full dst/src shape.
// Subregion tile.put (7 args) uses the explicit offsets and transfer shape that
// ConvertTensorToTileOps forwarded from user-facing pld.tensor.put. The stage
// tile carries the full transfer extent OR (when chunk_rows/chunk_cols were
// supplied) a sub-tile of it; pto-isa TPUT reads the full extent from the
// partition views and 2-D-slides the transfer through the stage tile, so the
// stage only has to fit within the flattened transfer (verified by
// DeducePutTileType), not equal it.
static std::string MakePutCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const size_t n = op->args_.size();
  CHECK(n == 4 || n == 5 || n == 7 || n == 8)
      << "pld.tile.put requires 4/5 (single/double stage, full-slice) or 7/8 (single/double stage, "
         "subregion) arguments (dst, peer, src, stage[, stage2][, dst_offsets, src_offsets, shape]), got "
      << n;
  // Optional second staging tile (ping-pong double-buffering). The region
  // tuples, when present, always follow the stage operand(s).
  const bool has_stage2 = (n == 5 || n == 8);
  const bool has_region = (n == 7 || n == 8);
  const size_t region_base = 4 + (has_stage2 ? 1 : 0);

  // dst: remote (peer-addressed) DistributedTensor destination.
  auto dst_binding = ResolveDistTensorBinding(op->args_[0], codegen, "pld.tile.put");

  // src: local source — DistributedTensor (window-bound) or plain Tensor.
  // Both reuse the local tensor_view created by EmitMakeTensorViews (no peer
  // arithmetic, like wait's signal). TPUT only requires src to be a readable
  // local GM region; window membership is not needed on the source side.
  auto src_var = AsVarLike(op->args_[2]);
  CHECK(src_var) << "pld.tile.put src must be a Var-like expression";
  auto src_tt = AsTensorTypeLike(src_var->GetType());
  CHECK(src_tt) << "pld.tile.put src must be a Tensor or DistributedTensor, got "
                << src_var->GetType()->TypeName();

  const int atomic_int = op->GetKwarg<int>("atomic", 0);
  CHECK(atomic_int == static_cast<int>(ir::AtomicType::kNone) ||
        atomic_int == static_cast<int>(ir::AtomicType::kAdd))
      << "pld.tile.put atomic kwarg must encode AtomicType::kNone or kAdd, got " << atomic_int;
  const std::string atomic_attr =
      atomic_int == static_cast<int>(ir::AtomicType::kAdd) ? "atomic_add" : "atomic_none";

  const auto& shape = dst_binding.type->shape_;
  const size_t rank = shape.size();
  INTERNAL_CHECK_SPAN(rank >= 1, op->span_) << "pld.tile.put requires rank >= 1";
  const std::string dtype_str = codegen.GetTypeString(dst_binding.type->dtype_);

  std::vector<std::string> dst_offsets;
  std::vector<std::string> src_offsets;
  std::vector<std::string> size_ssa;
  std::vector<ExprPtr> transfer_shape;

  if (!has_region) {
    // Full-slice partition views: offsets all-zero, sizes = full shape. dst and
    // src share the same partition_tensor_view type (same dtype + static shape).
    std::string c0 = codegen.GetOrEmitConstant(static_cast<int64_t>(0), DataType::INDEX);
    dst_offsets.assign(rank, c0);
    src_offsets.assign(rank, c0);
    transfer_shape = shape;
    size_ssa = GetSizeCodes(transfer_shape, codegen);
  } else {
    // Subregion partition views: user-facing tensor.put supplied the two
    // offset tuples plus a shared static transfer shape. The explicit stage
    // tile was sized to this transfer shape by ConvertTensorToTileOps.
    auto dst_offsets_tuple = As<ir::MakeTuple>(op->args_[region_base]);
    auto src_offsets_tuple = As<ir::MakeTuple>(op->args_[region_base + 1]);
    auto shape_tuple = As<ir::MakeTuple>(op->args_[region_base + 2]);
    INTERNAL_CHECK_SPAN(dst_offsets_tuple, op->span_) << "pld.tile.put dst_offsets must be MakeTuple";
    INTERNAL_CHECK_SPAN(src_offsets_tuple, op->span_) << "pld.tile.put src_offsets must be MakeTuple";
    INTERNAL_CHECK_SPAN(shape_tuple, op->span_) << "pld.tile.put shape must be MakeTuple";
    INTERNAL_CHECK_SPAN(dst_offsets_tuple->elements_.size() == rank, op->span_)
        << "pld.tile.put dst_offsets rank must match tensor rank";
    INTERNAL_CHECK_SPAN(src_offsets_tuple->elements_.size() == rank, op->span_)
        << "pld.tile.put src_offsets rank must match tensor rank";
    INTERNAL_CHECK_SPAN(shape_tuple->elements_.size() == rank, op->span_)
        << "pld.tile.put shape rank must match tensor rank";
    dst_offsets = GetIndexOffsetCodes(dst_offsets_tuple->elements_, codegen);
    src_offsets = GetIndexOffsetCodes(src_offsets_tuple->elements_, codegen);
    transfer_shape = shape_tuple->elements_;
    size_ssa = GetSizeCodes(transfer_shape, codegen);
  }

  std::string partition_type = MakePartitionTensorViewType(GetDimStrings(transfer_shape), dtype_str);

  // dst: CommRemoteOffset + addptr + make_tensor_view at the call site, then
  // a full-slice or subregion partition_view.
  auto dst_peer_view = EmitCommRemoteView(dst_binding, op->args_[1], codegen);
  std::string dst_pview =
      EmitPartitionViewPTO(dst_binding.var->name_hint_ + "_peer", dst_peer_view.ssa,
                           dst_peer_view.view_type_str, partition_type, dst_offsets, size_ssa, codegen);

  // src: local tensor_view + full-slice or subregion partition_view (no peer arithmetic).
  // Use the shared helper for the source view type so it matches the dynamic-dim
  // tensor_view SSA that GetOrCreateTensorView emits (mirroring dst's peer view
  // and every other tensor-view op in this file); a hand-rolled static-shape
  // string would mismatch that SSA's type.
  std::string src_local_view = codegen.GetOrCreateTensorView(src_var);
  std::string src_view_type = codegen.GetTensorViewTypeString(src_tt.get());
  std::string src_pview = EmitPartitionViewPTO(src_var->name_hint_ + "_local", src_local_view, src_view_type,
                                               partition_type, src_offsets, size_ssa, codegen);

  std::string stage = codegen.GetExprAsCode(op->args_[3]);
  std::string stage_type = codegen.GetExprTypeAnnotation(op->args_[3]);
  INTERNAL_CHECK_SPAN(!stage_type.empty(), op->span_)
      << "Internal error: pld.tile.put stage tile " << stage << " has no tile_buf type annotation";

  // Optional second staging tile selects pto-isa's ping-pong TPUT overload. Both
  // tiles ride in a single buf(...) operand group, each contributing a trailing
  // tile_buf type in the operand type list.
  std::string stage2, stage2_type;
  if (has_stage2) {
    stage2 = codegen.GetExprAsCode(op->args_[4]);
    stage2_type = codegen.GetExprTypeAnnotation(op->args_[4]);
    INTERNAL_CHECK_SPAN(!stage2_type.empty(), op->span_)
        << "Internal error: pld.tile.put stage2 tile " << stage2 << " has no tile_buf type annotation";
  }

  // The VEC staging tile(s) are not synthesized here: pld.tensor.put has already
  // been lowered to tile.create + pld.tile.put so the allocator can assign the
  // stage tile(s) real UB addresses before this PTO emission step.

  // TPUT reads the local source GM through MTE2. If the caller populated that
  // source via an immediately preceding TSTORE, order the store before TPUT's
  // source read; otherwise one rank can observe stale zeros while another wins
  // the timing race.
  codegen.Emit("pto.barrier <PIPE_ALL>");

  std::ostringstream tput;
  tput << "pto.comm.tput(" << dst_pview << ", " << src_pview << ", buf(" << stage;
  if (has_stage2) tput << ", " << stage2;
  tput << ") : " << partition_type << ", " << partition_type << ", " << stage_type;
  if (has_stage2) tput << ", " << stage2_type;
  tput << ") {atomicType = #pto<atomic_type " << atomic_attr << ">}";
  codegen.Emit(tput.str());

  // Drain TPUT's writes before returning, so a following `pld.system.notify`
  // (cross-rank signal that the data has landed) does not race ahead of them.
  // Emitted unconditionally: the chunked sliding path strictly requires it (its
  // last chunk's MTE3 store is otherwise still in-flight, a deterministic stale
  // read), and the single-shot path — though self-draining for that store — has
  // the same cross-rank data-before-signal obligation, so the extra barrier is
  // harmless there.
  //
  // WORKAROUND for PTOAS#872: the proper fix drains prior stores inside
  // TNOTIFY_IMPL (`pipe_barrier(PIPE_ALL); dsb(DSB_DDR)` before the signal),
  // which also adds the DDR-observability fence a pipe barrier alone can't give.
  // Remove this once that lands.
  codegen.Emit("pto.barrier <PIPE_ALL>");
  return "";
}

static std::string MakeGetCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const size_t n = op->args_.size();
  CHECK(n == 4 || n == 5 || n == 7 || n == 8)
      << "pld.tile.get requires 4/5 (single/double stage, full-slice) or 7/8 (single/double stage, "
         "subregion) arguments (dst, peer, src, stage[, stage2][, dst_offsets, src_offsets, shape]), got "
      << n;
  const bool has_stage2 = (n == 5 || n == 8);
  const bool has_region = (n == 7 || n == 8);
  const size_t region_base = 4 + (has_stage2 ? 1 : 0);

  // dst: local destination — DistributedTensor (window-bound) or plain Tensor.
  // Both reuse the local tensor_view created by EmitMakeTensorViews (no peer
  // arithmetic, like wait's signal). TGET only requires dst to be a writable
  // local GM region; window membership is not needed on the destination side.
  auto dst_var = AsVarLike(op->args_[0]);
  CHECK(dst_var) << "pld.tile.get dst must be a Var-like expression";
  auto dst_tt = AsTensorTypeLike(dst_var->GetType());
  CHECK(dst_tt) << "pld.tile.get dst must be a Tensor or DistributedTensor, got "
                << dst_var->GetType()->TypeName();

  // src: remote (peer-addressed) DistributedTensor source.
  auto src_binding = ResolveDistTensorBinding(op->args_[2], codegen, "pld.tile.get");
  auto peer_scalar = As<ir::ScalarType>(op->args_[1]->GetType());
  CHECK(peer_scalar) << "pld.tile.get peer must be ScalarType at codegen, got "
                     << op->args_[1]->GetType()->TypeName();

  const auto& dst_shape = dst_tt->shape_;
  const size_t rank = dst_shape.size();
  const std::string dtype_str = codegen.GetTypeString(dst_tt->dtype_);

  std::vector<std::string> dst_offsets;
  std::vector<std::string> src_offsets;
  std::vector<std::string> size_ssa;
  std::vector<ExprPtr> transfer_shape;

  if (!has_region) {
    std::string c0 = codegen.GetOrEmitConstant(static_cast<int64_t>(0), DataType::INDEX);
    dst_offsets.assign(rank, c0);
    src_offsets.assign(rank, c0);
    transfer_shape = dst_shape;
    size_ssa = GetSizeCodes(transfer_shape, codegen);
  } else {
    auto dst_offsets_tuple = As<ir::MakeTuple>(op->args_[region_base]);
    auto src_offsets_tuple = As<ir::MakeTuple>(op->args_[region_base + 1]);
    auto shape_tuple = As<ir::MakeTuple>(op->args_[region_base + 2]);
    INTERNAL_CHECK_SPAN(dst_offsets_tuple, op->span_) << op->op_->name_ << " dst_offsets must be MakeTuple";
    INTERNAL_CHECK_SPAN(src_offsets_tuple, op->span_) << op->op_->name_ << " src_offsets must be MakeTuple";
    INTERNAL_CHECK_SPAN(shape_tuple, op->span_) << op->op_->name_ << " shape must be MakeTuple";
    INTERNAL_CHECK_SPAN(dst_offsets_tuple->elements_.size() == rank, op->span_)
        << op->op_->name_ << " dst_offsets rank must match tensor rank";
    INTERNAL_CHECK_SPAN(src_offsets_tuple->elements_.size() == rank, op->span_)
        << op->op_->name_ << " src_offsets rank must match tensor rank";
    INTERNAL_CHECK_SPAN(shape_tuple->elements_.size() == rank, op->span_)
        << op->op_->name_ << " shape rank must match tensor rank";
    dst_offsets = GetIndexOffsetCodes(dst_offsets_tuple->elements_, codegen);
    src_offsets = GetIndexOffsetCodes(src_offsets_tuple->elements_, codegen);
    transfer_shape = shape_tuple->elements_;
    size_ssa = GetSizeCodes(transfer_shape, codegen);
  }

  std::string partition_type = MakePartitionTensorViewType(GetDimStrings(transfer_shape), dtype_str);

  // dst: local tensor_view + full-slice partition_view.
  std::string dst_local_view = codegen.GetOrCreateTensorView(dst_var);
  std::string dst_view_type = codegen.GetTensorViewTypeString(dst_tt.get());
  std::string dst_pview = EmitPartitionViewPTO(dst_var->name_hint_ + "_local", dst_local_view, dst_view_type,
                                               partition_type, dst_offsets, size_ssa, codegen);

  // src: CommRemoteOffset + addptr + make_tensor_view at the call site, then
  // a full-slice partition_view.
  auto src_peer_view = EmitCommRemoteView(src_binding, op->args_[1], codegen);
  std::string src_pview =
      EmitPartitionViewPTO(src_binding.var->name_hint_ + "_peer", src_peer_view.ssa,
                           src_peer_view.view_type_str, partition_type, src_offsets, size_ssa, codegen);

  std::string stage = codegen.GetExprAsCode(op->args_[3]);
  std::string stage_type = codegen.GetExprTypeAnnotation(op->args_[3]);
  INTERNAL_CHECK_SPAN(!stage_type.empty(), op->span_)
      << "Internal error: pld.tile.get stage tile " << stage << " has no tile_buf type annotation";

  // Optional second staging tile selects pto-isa's ping-pong TGET overload (both
  // tiles in one buf(...) group, each with a trailing tile_buf type).
  std::string stage2, stage2_type;
  if (has_stage2) {
    stage2 = codegen.GetExprAsCode(op->args_[4]);
    stage2_type = codegen.GetExprTypeAnnotation(op->args_[4]);
    INTERNAL_CHECK_SPAN(!stage2_type.empty(), op->span_)
        << "Internal error: pld.tile.get stage2 tile " << stage2 << " has no tile_buf type annotation";
  }

  // Mirror TPUT's ordering guard. TGET may read a peer source that was just
  // populated from a local TSTORE before the cross-rank handshake; keep the
  // local store visible before the peer-side TGET source read.
  codegen.Emit("pto.barrier <PIPE_ALL>");

  std::ostringstream tget;
  tget << "pto.comm.tget(" << dst_pview << ", " << src_pview << ", buf(" << stage;
  if (has_stage2) tget << ", " << stage2;
  tget << ") : " << partition_type << ", " << partition_type << ", " << stage_type;
  if (has_stage2) tget << ", " << stage2_type;
  tget << ")";
  codegen.Emit(tget.str());

  // Drain TGET's writes into the local dst before returning. As with TPUT, when
  // the staging tile is smaller than the transfer pto-isa TGET 2-D-slides the
  // transfer through multiple chunks; a following local read of `dst` must not
  // race ahead of the last chunk's MTE3 store.
  //
  // WORKAROUND for PTOAS#872 (TGET counterpart): remove once PTOAS drains a
  // chunked tget itself.
  codegen.Emit("pto.barrier <PIPE_ALL>");
  return "";
}

// Emit ``%rk_pair = pto.load_scalar %ctx[%slot] : !pto.ptr<i64> -> i64`` for
// the (rankId, rankNum) u64 slot. Shared by ``pld.system.rank`` (low 32 bits)
// and ``pld.system.nranks`` (high 32 bits) — see comm_layout.h for the static
// asserts that anchor rankNum at rankId + 4 in the same i64 slot.
static std::string EmitLoadRankPair(codegen::PTOCodegen& cg, const std::string& ctx_ssa) {
  namespace cl = codegen::distributed::comm_layout;
  constexpr int64_t kRankSlotIdx = static_cast<int64_t>(cl::kRankIdOffset / cl::kWindowSlotStride);
  std::string slot_c = cg.GetOrEmitConstant(kRankSlotIdx, DataType::INDEX);
  std::string rk_pair = cg.NewTemp();
  cg.Emit(rk_pair + " = pto.load_scalar " + ctx_ssa + "[" + slot_c + "] : !pto.ptr<i64> -> i64");
  return rk_pair;
}

void RegisterDistributedOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  // Register ops with custom codegen logic
  auto reg = [&](const char* op_name, BackendCodegenFunc fn) {
    if (exclude_ops.count(op_name) > 0) return;
    backend.RegisterOp(op_name).f_codegen(std::move(fn));
  };

  // Distributed N6 ops — cross-rank tile load + per-rank signal notify/wait +
  // synchronous bulk get/put. See MakeRemoteLoadCodegenPTO /
  // MakeNotifyCodegenPTO / MakeWaitCodegenPTO / MakeGetCodegenPTO /
  // MakePutCodegenPTO for the emitted MLIR shape.
  // Cross-rank ops lower to a single
  // ``func.call @CommRemoteOffset_<dtype>`` against a module-level helper
  // emitted by PTOCodegen::EmitCommRemoteOffsetHelpers; the helper returns
  // the peer-vs-local element offset (``index``) and the call site emits
  // ``pto.addptr`` + ``pto.make_tensor_view`` locally so PTOAS's per-func
  // "addptr must feed make_tensor_view" check is satisfied. The helper's
  // byte-offset literals are pinned to ``comm_layout::k*`` constants
  // (PyPTO compile-time static_asserts catch any CommContext ABI drift).
  reg("pld.tile.remote_load", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeRemoteLoadCodegenPTO(op, codegen);
  });
  reg("pld.tile.remote_store", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
    return MakeRemoteStoreCodegenPTO(op, codegen);
  });
  reg("pld.system.notify",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeNotifyCodegenPTO(op, codegen); });
  reg("pld.system.wait",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeWaitCodegenPTO(op, codegen); });

  // Distributed N7 ops — CommContext accessor lowering.
  //
  // ``pld.system.get_comm_ctx(dist_t) -> CommCtxType``: pure SSA alias. No
  // MLIR is emitted; the ctx-ptr arg slot that PTOCodegen's
  // ``GenerateFunction`` appended for ``dist_t`` (see
  // ``fs_.dist_tensor_to_ctx`` / ``GetCommCtxSSAFor``) is published as the
  // current expression value, which the surrounding ``VisitStmt_(AssignStmt)``
  // then binds to the LHS Var. Downstream ``pld.system.rank(ctx)`` /
  // ``pld.system.nranks(ctx)`` codegen resolves ``ctx`` via the standard
  // ``GetExprAsCode(call->args_[0])`` path.
  reg("pld.system.get_comm_ctx",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
        auto& cg = AsPto(codegen_base);
        CHECK(op->args_.size() == 1) << "pld.system.get_comm_ctx expects exactly 1 argument, got "
                                     << op->args_.size();
        auto var = ir::AsVarLike(op->args_[0]);
        CHECK(var) << "pld.system.get_comm_ctx expects a Var (DistributedTensor param), got "
                   << op->args_[0]->TypeName();
        std::string ctx_ssa = cg.GetCommCtxSSAFor(var.get());
        CHECK(!ctx_ssa.empty())
            << "No CommContext ptr arg threaded for DistributedTensor '" << var->name_hint_
            << "' — ensure the func.func ctx segment was emitted (PTOCodegen::GenerateFunction)";
        if (auto lhs = cg.GetCurrentResultVar()) {
          cg.RegisterVarToMlir(lhs, ctx_ssa);
        }
        cg.SetCurrentExprValue(ctx_ssa);
        return "";
      });

  // ``pld.system.rank(ctx)``: IR ``ScalarType(INT32)``; MLIR type is ``i32``.
  // ``i32`` (PTOAS rejects ``arith.trunci`` to ``ui32``). Low 32 bits of the
  // (rankId, rankNum) u64 slot via ``pto.load_scalar`` + ``arith.trunci``.
  reg("pld.system.rank", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
    auto& cg = AsPto(codegen_base);
    CHECK(op->args_.size() == 1) << "pld.system.rank expects exactly 1 argument, got " << op->args_.size();
    std::string ctx_ssa = cg.GetExprAsCode(op->args_[0]);
    std::string rk_pair = EmitLoadRankPair(cg, ctx_ssa);
    std::string rk = cg.GetCurrentResultTarget();
    cg.Emit(rk + " = arith.trunci " + rk_pair + " : i64 to i32");
    cg.SetCurrentExprValue(rk);
    return "";
  });

  // ``pld.system.nranks(ctx)``: same INT32 IR / i32 MLIR convention.
  // High 32 bits of the same slot —
  // ``kRankNumOffset == kRankIdOffset + 4`` lets us shift the already-loaded
  // i64 right by 32 instead of issuing a second pto.load_scalar.
  reg("pld.system.nranks", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
    auto& cg = AsPto(codegen_base);
    CHECK(op->args_.size() == 1) << "pld.system.nranks expects exactly 1 argument, got " << op->args_.size();
    namespace cl = codegen::distributed::comm_layout;
    static_assert(cl::kRankNumOffset == cl::kRankIdOffset + 4,
                  "pld.system.nranks codegen assumes rankNum sits in the high 32 bits of rankId's i64 slot");
    std::string ctx_ssa = cg.GetExprAsCode(op->args_[0]);
    std::string rk_pair = EmitLoadRankPair(cg, ctx_ssa);
    std::string c32 = cg.GetOrEmitConstant(static_cast<int64_t>(32), DataType::INT64);
    std::string rn_i64 = cg.NewTemp();
    cg.Emit(rn_i64 + " = arith.shrui " + rk_pair + ", " + c32 + " : i64");
    std::string rn = cg.GetCurrentResultTarget();
    cg.Emit(rn + " = arith.trunci " + rn_i64 + " : i64 to i32");
    cg.SetCurrentExprValue(rn);
    return "";
  });
  reg("pld.tile.get",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeGetCodegenPTO(op, codegen); });
  reg("pld.tile.put",
      [](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakePutCodegenPTO(op, codegen); });
}
}  // namespace backend
}  // namespace pypto
